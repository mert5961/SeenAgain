"""Stage 2: put every walk into ONE metric-up-to-scale frame with COLMAP (pycolmap).

All walks are reconstructed jointly, so walk B's cameras land in walk A's coordinates and
"where was this object last time" becomes a projection, not a guess. (Yodo localises new
rounds against the existing twin; a joint reconstruction is the small-scale version of that.)

Pairs: neighbours inside a walk + a strided set of cross-walk pairs.
Outputs (pinhole, undistorted): images_undist/, cameras.json, sparse_obs.npz, points3D.npz.
"""
import argparse
import itertools
import shutil
from collections import defaultdict

import numpy as np
import pycolmap

from .common import scene_dir, walk_of, save_json


def make_pairs(names, overlap, cross_stride):
    by_walk = defaultdict(list)
    for n in sorted(names):
        by_walk[walk_of(n)].append(n)
    pairs = set()
    for w in by_walk.values():
        for i in range(len(w)):
            for k in range(1, overlap + 1):
                if i + k < len(w):
                    pairs.add((w[i], w[i + k]))
    for w1, w2 in itertools.combinations(sorted(by_walk), 2):
        for i, a in enumerate(by_walk[w1]):
            for b in by_walk[w2][i % cross_stride::cross_stride]:
                pairs.add((a, b))
    return sorted(pairs), by_walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--cross-stride", type=int, default=4)
    ap.add_argument("--mapper", choices=["global", "incremental"], default="global",
                    help="global (GLOMAP) placed 150/150 frames on the desk scene; incremental split it into 15 pieces")
    ap.add_argument("--reuse-matches", action="store_true", help="keep db.db, only redo mapping")
    a = ap.parse_args()
    sd = scene_dir(a.scene)
    img_dir = sd / "images"
    names = sorted(p.name for p in img_dir.glob("*.jpg"))

    for p in (["sparse", "undist"] if a.reuse_matches else ["db.db", "sparse", "undist"]):
        q = sd / p
        if q.is_dir():
            shutil.rmtree(q)
        elif q.exists():
            q.unlink()

    pairs, by_walk = make_pairs(names, a.overlap, a.cross_stride)
    if not a.reuse_matches:
        pycolmap.extract_features(sd / "db.db", img_dir, camera_mode=pycolmap.CameraMode.SINGLE)
        (sd / "pairs.txt").write_text("".join(f"{x} {y}\n" for x, y in pairs))
        print(f"{len(names)} images, {len(pairs)} pairs")
        po = pycolmap.ImportedPairingOptions()
        po.match_list_path = sd / "pairs.txt"
        pycolmap.match_image_pairs(sd / "db.db", pairing_options=po)

    (sd / "sparse").mkdir()
    mapping = pycolmap.global_mapping if a.mapper == "global" else pycolmap.incremental_mapping
    recs = mapping(sd / "db.db", img_dir, sd / "sparse")
    best_i = max(recs, key=lambda i: recs[i].num_reg_images())
    rec = recs[best_i]
    print(f"{len(recs)} model(s); using #{best_i}: {rec.summary()}")

    pycolmap.undistort_images(sd / "undist", sd / "sparse" / str(best_i), img_dir)
    rec = pycolmap.Reconstruction(sd / "undist" / "sparse")

    cams, obs = {}, {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cam = rec.cameras[img.camera_id]
        T = img.cam_from_world()
        M = T.matrix()
        fx, fy, cx, cy = cam.params[:4]
        cams[img.name] = dict(walk=walk_of(img.name), w=cam.width, h=cam.height,
                              fx=fx, fy=fy, cx=cx, cy=cy,
                              R=M[:, :3].tolist(), t=M[:, 3].tolist())
        uv, xyz = [], []
        for p2 in img.points2D:
            if p2.has_point3D():
                uv.append(p2.xy)
                xyz.append(rec.points3D[p2.point3D_id].xyz)
        obs[img.name + ":uv"] = np.array(uv, np.float32).reshape(-1, 2)
        obs[img.name + ":xyz"] = np.array(xyz, np.float32).reshape(-1, 3)
    save_json(sd / "cameras.json", cams)
    np.savez_compressed(sd / "sparse_obs.npz", **obs)
    P = np.array([p.xyz for p in rec.points3D.values()], np.float32)
    C = np.array([p.color for p in rec.points3D.values()], np.uint8)
    np.savez_compressed(sd / "points3D.npz", xyz=P, rgb=C)

    report = {}
    for w, ns in sorted(by_walk.items()):
        reg = sum(n in cams for n in ns)
        report[w] = dict(frames=len(ns), registered=reg)
        print(f"walk {w}: {reg}/{len(ns)} frames registered")
    save_json(sd / "sfm_report.json", report)


if __name__ == "__main__":
    main()
