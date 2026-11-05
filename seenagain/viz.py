"""Figures for the README: top-down change map, and round frames with the baseline register projected in."""
import argparse
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import scene_dir, load_json, load_cameras, project

COL = {"unchanged": "#3a9e5f", "moved": "#e0662f", "removed": "#e8352f", "in_place": "#c03fb0",
       "added": "#2f7fe0", "review": "#e3b21b", "not_seen": "#8a8a8a", "added_unverified": "#8a8a8a"}
NAME = {"unchanged": "confirmed unchanged", "moved": "moved", "removed": "removed", "in_place": "changed in place",
        "added": "added", "review": "to a person", "not_seen": "not seen clearly", "added_unverified": "new, unverified"}


def frame_basis(cams):
    up = -np.mean([c["R"].T @ np.array([0, 1, 0]) for c in cams.values()], 0)
    up /= np.linalg.norm(up)
    fwd = np.mean([c["R"].T @ np.array([0, 0, 1]) for c in cams.values()], 0)
    fwd -= up * (fwd @ up)
    fwd /= np.linalg.norm(fwd)
    return np.cross(fwd, up), fwd, up


def topdown(sd, rec, A, B, cams, mpu, out):
    ex, ey, _ = frame_basis(cams)
    P = np.load(sd / "points3D.npz")["xyz"]
    to2 = lambda X: np.stack([X @ ex, X @ ey], -1) * mpu
    oa = {o["id"]: o for o in A["objects"]}
    ob = {o["id"]: o for o in B["objects"]}
    fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
    q = to2(P)
    lo, hi = np.percentile(q, 2, 0), np.percentile(q, 98, 0)
    ax.scatter(q[:, 0], q[:, 1], s=0.3, c="#cfcfcf", lw=0)
    seen = set()
    for r in rec["rows"]:
        st = r["state"]
        src = oa.get(r.get("asset")) or ob.get(r.get("new"))
        c = to2(np.array(src["centroid"]))
        lab = NAME[st] if st not in seen else None
        seen.add(st)
        ax.scatter(*c, s=70, c=COL[st], edgecolor="white", lw=0.8, zorder=3, label=lab)
        if r.get("matched") and st in ("moved", "review", "in_place"):
            d = to2(np.array(ob[r["matched"]]["centroid"]))
            ax.annotate("", d, c, arrowprops=dict(arrowstyle="->", color=COL[st], lw=1.6), zorder=4)
        if st != "unchanged":
            ax.text(c[0] + 0.01, c[1] + 0.01, r.get("asset") or r.get("new"), fontsize=7, color="#333")
    oc = to2(np.array([o["centroid"] for o in A["objects"] + B["objects"]]))
    lo, hi = np.minimum(lo, oc.min(0)) - 0.05, np.maximum(hi, oc.max(0)) + 0.05
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("metres (approx. scale)"), ax.set_ylabel("metres (approx. scale)")
    ax.set_title(f"Round {rec['round']} vs baseline {rec['base']}: top-down")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def overlay_frames(sd, rec, A, B, cams, walk, outdir, step):
    """Baseline assets projected into the round's frames, boxed and coloured by receipt state."""
    st = {r["asset"]: r["state"] for r in rec["rows"] if r.get("asset")}
    dest = {r["asset"]: r["matched"] for r in rec["rows"] if r.get("asset") and r["state"] == "moved"}
    ob = {o["id"]: o for o in B["objects"]}
    names = sorted(n for n, c in cams.items() if c["walk"] == walk)[::step]
    outdir.mkdir(parents=True, exist_ok=True)
    for k, n in enumerate(names):
        c = cams[n]
        im = cv2.imread(str(sd / "undist" / "images" / n))
        D = np.load(sd / "depth" / f"{n}.npy").astype(np.float32) if (sd / "depth" / f"{n}.npy").exists() else None
        for o in A["objects"]:
            s = st.get(o["id"])
            if s is None:
                continue
            uv, z = project(c, np.array(o["points"]))
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < c["w"]) & (uv[:, 1] < c["h"])
            if ok.mean() < 0.6:
                continue
            if D is not None and s not in ("removed",):
                u = np.clip((uv[ok, 0] / 2).astype(int), 0, D.shape[1] - 1)
                v = np.clip((uv[ok, 1] / 2).astype(int), 0, D.shape[0] - 1)
                if (D[v, u] < z[ok] * 0.85).mean() > 0.6:
                    continue  # hidden behind something in this frame
            x0, y0 = np.percentile(uv[ok], 5, 0).astype(int)
            x1, y1 = np.percentile(uv[ok], 95, 0).astype(int)
            col = tuple(int(COL[s][i:i + 2], 16) for i in (5, 3, 1))
            th = 1 if s == "unchanged" else 3
            cv2.rectangle(im, (x0, y0), (x1, y1), col, th)
            if o["id"] in dest:
                uv2, z2 = project(c, np.array(ob[dest[o["id"]]]["points"]))
                if (z2 > 0).all():
                    a0 = ((x0 + x1) // 2, (y0 + y1) // 2)
                    a1 = tuple(np.median(uv2, 0).astype(int))
                    cv2.arrowedLine(im, a0, a1, col, 3, cv2.LINE_AA, tipLength=0.25)
                    cv2.circle(im, a1, 6, col, -1, cv2.LINE_AA)
            if s == "unchanged":
                continue  # thin green box only; labels are for things that need attention
            cv2.putText(im, f"{o['id']} {NAME[s]}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(im, f"{o['id']} {NAME[s]}", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 1, cv2.LINE_AA)
        cv2.putText(im, f"round {walk}: baseline assets projected into this frame (green box = confirmed unchanged)", (12, c["h"] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(outdir / f"{k:04d}.jpg"), im)
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--receipt", required=True, help="e.g. receipt_A_B_v2.json")
    ap.add_argument("--metres-per-unit", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    sd = scene_dir(a.scene)
    rec = load_json(sd / a.receipt)
    cams = load_cameras(sd)
    A = load_json(sd / f"objects_{rec['base']}.json")
    B = load_json(sd / f"objects_{rec['round']}.json")
    import pathlib
    out = pathlib.Path(a.out)
    out.mkdir(exist_ok=True)
    tag = a.receipt.replace("receipt_", "").replace(".json", "")
    topdown(sd, rec, A, B, cams, a.metres_per_unit, out / f"topdown_{tag}.png")
    fr = overlay_frames(sd, rec, A, B, cams, rec["round"], sd / f"overlay_{tag}", a.step)
    gif = out / f"overlay_{tag}.gif"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "4", "-i", str(fr / "%04d.jpg"),
                    "-vf", "scale=480:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                    str(gif)], check=True)
    print(out / f"topdown_{tag}.png", gif)


if __name__ == "__main__":
    main()
