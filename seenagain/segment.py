"""Stage 3: per walk, find objects and place them in 3D.

For every registered frame
  - monocular depth (Depth Anything V2 Small) is fitted to that frame's own SfM points
    (1/z = a*disparity + b, robust least squares), so depth is in the shared frame's units;
For every k-th frame (keyframes)
  - SAM 2.1 proposes masks; background-sized and border-hugging masks are dropped, parts inside
    a bigger kept mask are dropped;
  - each mask is lifted to 3D with the fitted depth, and gets a DINOv2 fingerprint of its crop.
Then masks from different frames are fused into objects when their 3D voxels overlap and
their fingerprints agree. Only objects seen in >= min_views frames are kept ("several views agree").

Outputs: depth/<frame>.npy (half res), objects_<walk>.json, thumbs/<walk>/OBJ-xxx.jpg
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from .common import scene_dir, load_cameras, save_json, unproject, world_to_cam, camera_center

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


class Models:
    def __init__(self):
        self.dproc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.dnet = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf").to(DEV).eval()
        self.eproc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.enet = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml",
                         "checkpoints/sam2.1_hiera_base_plus.pt", device=DEV)
        self.sam = SAM2AutomaticMaskGenerator(sam, points_per_side=24, pred_iou_thresh=0.8,
                                              stability_score_thresh=0.9, min_mask_region_area=400)

    @torch.no_grad()
    def disparity(self, rgb):
        x = self.dproc(images=Image.fromarray(rgb), return_tensors="pt").to(DEV)
        d = self.dnet(**x).predicted_depth[0]
        d = torch.nn.functional.interpolate(d[None, None], size=rgb.shape[:2], mode="bilinear")[0, 0]
        return d.float().cpu().numpy()

    @torch.no_grad()
    def embed(self, crops):
        x = self.eproc(images=[Image.fromarray(c) for c in crops], return_tensors="pt").to(DEV)
        f = self.enet(**x).pooler_output.float().cpu().numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)


def fit_depth(disp, uv, z):
    """Fit 1/z = a*disp + b on the frame's SfM points; returns metric-ish depth map or None."""
    h, w = disp.shape
    u = np.clip(uv[:, 0].round().astype(int), 0, w - 1)
    v = np.clip(uv[:, 1].round().astype(int), 0, h - 1)
    d, iz = disp[v, u], 1.0 / z
    keep = np.ones(len(d), bool)
    for _ in range(3):
        if keep.sum() < 20:
            return None, None
        A = np.stack([d[keep], np.ones(keep.sum())], 1)
        (a, b), *_ = np.linalg.lstsq(A, iz[keep], rcond=None)
        r = np.abs(a * d + b - iz)
        keep = r < 2.5 * np.median(r[keep]) + 1e-9
    inv = a * disp + b
    depth = np.where(inv > 1e-6, 1.0 / np.maximum(inv, 1e-6), 0.0)
    rel = np.median(np.abs(1.0 / np.maximum(a * d[keep] + b, 1e-6) - z[keep]) / z[keep])
    return depth.astype(np.float32), float(rel)


def filter_masks(masks, h, w, min_frac, max_frac):
    keep = []
    for m in sorted(masks, key=lambda m: -m["area"]):
        frac = m["area"] / (h * w)
        if not (min_frac <= frac <= max_frac):
            continue
        x, y, bw, bh = m["bbox"]
        borders = (x <= 2) + (y <= 2) + (x + bw >= w - 3) + (y + bh >= h - 3)
        if borders >= 2:
            continue
        seg = m["segmentation"]
        if any((seg & k["segmentation"]).sum() > 0.6 * m["area"] for k in keep):
            continue  # a part of an object we already have
        keep.append(m)
    return keep


def lift(cam, seg, depth, n=600, rng=np.random.default_rng(0)):
    seg = cv2.erode(seg.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    vs, us = np.nonzero(seg & (depth > 0))
    if len(us) < 50:
        return None
    i = rng.choice(len(us), min(n, len(us)), replace=False)
    z = depth[vs[i], us[i]]
    med = np.median(z)
    ok = np.abs(z - med) < 0.2 * med  # objects are compact; drop depth bleed at the edges
    if ok.sum() < 30:
        return None
    uv = np.stack([us[i][ok], vs[i][ok]], 1).astype(np.float32)
    return unproject(cam, uv, z[ok])


def crop(rgb, seg, bbox, pad=0.15):
    x, y, w, h = [int(v) for v in bbox]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(rgb.shape[1], x + w + px), min(rgb.shape[0], y + h + py)
    c = rgb[y0:y1, x0:x1].copy()
    m = seg[y0:y1, x0:x1]
    c[~m] = (c[~m] * 0.25 + 128 * 0.75).astype(np.uint8)  # fade the background, keep context
    return c


def voxels(P, size):
    return set(map(tuple, np.floor(P / size).astype(np.int64)))


def fuse(obs, vox, sim_min, overlap_min, min_views):
    n = len(obs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    V = [voxels(o["points"], vox) for o in obs]
    E = np.stack([o["emb"] for o in obs]) if n else np.zeros((0, 384))
    S = E @ E.T
    for i in range(n):
        for j in range(i + 1, n):
            if obs[i]["frame"] == obs[j]["frame"] or S[i, j] < sim_min:
                continue
            inter = len(V[i] & V[j])
            if inter and inter / min(len(V[i]), len(V[j])) >= overlap_min:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    objs = []
    for g in groups.values():
        frames = {obs[i]["frame"] for i in g}
        if len(frames) < min_views:
            continue
        P = np.concatenate([obs[i]["points"] for i in g])
        e = E[g].mean(0)
        objs.append(dict(members=g, frames=sorted(frames), points=P,
                         centroid=np.median(P, 0), emb=e / np.linalg.norm(e)))
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--walk", required=True)
    ap.add_argument("--key-stride", type=int, default=3)
    ap.add_argument("--min-frac", type=float, default=0.004)
    ap.add_argument("--max-frac", type=float, default=0.25)
    ap.add_argument("--sim-min", type=float, default=0.55)
    ap.add_argument("--overlap-min", type=float, default=0.25)
    ap.add_argument("--min-views", type=int, default=3)
    a = ap.parse_args()

    sd = scene_dir(a.scene)
    cams = load_cameras(sd)
    sp = np.load(sd / "sparse_obs.npz")
    names = sorted(n for n, c in cams.items() if c["walk"] == a.walk)
    allP = np.load(sd / "points3D.npz")["xyz"]
    scale = float(np.median([np.median(np.linalg.norm(allP - camera_center(cams[n]), axis=1))
                             for n in names]))
    vox = 0.02 * scale
    print(f"walk {a.walk}: {len(names)} registered frames, scene scale {scale:.3f} units, voxel {vox:.4f}")

    M = Models()
    (sd / "depth").mkdir(exist_ok=True)
    (sd / "thumbs" / a.walk).mkdir(parents=True, exist_ok=True)
    obs, depth_err = [], {}
    for k, n in enumerate(names):
        c = cams[n]
        rgb = cv2.cvtColor(cv2.imread(str(sd / "undist" / "images" / n)), cv2.COLOR_BGR2RGB)
        uv, xyz = sp[n + ":uv"], sp[n + ":xyz"]
        z = world_to_cam(c, xyz)[:, 2]
        good = z > 0
        depth, rel = fit_depth(M.disparity(rgb), uv[good], z[good])
        if depth is None:
            continue
        depth_err[n] = rel
        np.save(sd / "depth" / f"{n}.npy", cv2.resize(depth, (c["w"] // 2, c["h"] // 2),
                                                      interpolation=cv2.INTER_NEAREST).astype(np.float16))
        if k % a.key_stride:
            continue
        h, w = rgb.shape[:2]
        masks = filter_masks(M.sam.generate(rgb), h, w, a.min_frac, a.max_frac)
        items = []
        for m in masks:
            P = lift(c, m["segmentation"], depth)
            if P is None:
                continue
            ext = np.percentile(P, 95, 0) - np.percentile(P, 5, 0)
            if ext.max() > 0.6 * scale:
                continue  # wall / desk / floor sized
            items.append((m, P))
        if not items:
            continue
        crops = [crop(rgb, m["segmentation"], m["bbox"]) for m, _ in items]
        embs = M.embed(crops)
        for (m, P), e, cr in zip(items, embs, crops):
            obs.append(dict(frame=n, points=P, emb=e, area=m["area"], crop=cr))
        print(f"  {n}: depth fit err {rel:.1%}, {len(items)} objects")

    objs = fuse(obs, vox, a.sim_min, a.overlap_min, a.min_views)
    # stable IDs: sort along the walk (first frame seen), then name
    objs.sort(key=lambda o: (o["frames"][0], float(o["centroid"][0])))
    out = []
    for i, o in enumerate(objs, 1):
        oid = f"{a.walk}-{i:03d}"
        best = max(o["members"], key=lambda j: obs[j]["area"])
        cv2.imwrite(str(sd / "thumbs" / a.walk / f"{oid}.jpg"), cv2.cvtColor(obs[best]["crop"], cv2.COLOR_RGB2BGR))
        P = o["points"]
        sub = P[np.random.default_rng(0).choice(len(P), min(1500, len(P)), replace=False)]
        out.append(dict(id=oid, walk=a.walk, centroid=o["centroid"].tolist(),
                        extent=(np.percentile(P, 95, 0) - np.percentile(P, 5, 0)).tolist(),
                        frames=o["frames"], n_views=len(o["frames"]), emb=o["emb"].tolist(),
                        points=sub.tolist(), thumb=f"thumbs/{a.walk}/{oid}.jpg"))
    save_json(sd / f"objects_{a.walk}.json",
              dict(walk=a.walk, scale=scale, voxel=vox, n_masks=len(obs), objects=out,
                   depth_fit_err_median=float(np.median(list(depth_err.values()))) if depth_err else None))
    print(f"walk {a.walk}: {len(obs)} masks -> {len(out)} objects (seen in >= {a.min_views} views)")


if __name__ == "__main__":
    main()
