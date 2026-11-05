"""Stage 1: phone video -> sharp frames named <walk>_<i>.jpg.

ffmpeg samples densely, then we keep the sharpest frame (variance of Laplacian) in each window,
so motion blur from walking does not reach SfM.
"""
import argparse
import shutil
import subprocess
import tempfile
import pathlib

import cv2

from .common import scene_dir


def sharpness(p):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--walk", required=True, help="walk name, e.g. A, A2, B")
    ap.add_argument("--fps", type=float, default=2.0, help="kept frames per second")
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--max-side", type=int, default=1600)
    a = ap.parse_args()

    out = scene_dir(a.scene) / "images"
    out.mkdir(exist_ok=True)
    for old in out.glob(f"{a.walk}_*.jpg"):
        old.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        s = a.max_side
        vf = f"fps={a.fps * a.oversample},scale='if(gt(iw,ih),min({s},iw),-2)':'if(gt(iw,ih),-2,min({s},ih))'"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", a.video, "-vf", vf, "-q:v", "2",
                        f"{tmp}/%05d.jpg"], check=True)
        raw = sorted(pathlib.Path(tmp).glob("*.jpg"))
        kept = 0
        for i in range(0, len(raw), a.oversample):
            best = max(raw[i:i + a.oversample], key=sharpness)
            shutil.copy(best, out / f"{a.walk}_{kept:04d}.jpg")
            kept += 1
    print(f"{a.walk}: {len(raw)} sampled -> {kept} sharp frames in {out}")


if __name__ == "__main__":
    main()
