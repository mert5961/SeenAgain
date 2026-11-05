"""Contact sheet of a walk's objects (thumbnail + id + views), for checking and for labelling."""
import argparse

import cv2
import numpy as np

from .common import scene_dir, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--walk", required=True)
    ap.add_argument("--cols", type=int, default=7)
    a = ap.parse_args()
    sd = scene_dir(a.scene)
    objs = load_json(sd / f"objects_{a.walk}.json")["objects"]
    tiles = []
    for o in objs:
        im = cv2.imread(str(sd / o["thumb"]))
        h, w = im.shape[:2]
        s = 160 / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        t = np.full((190, 170, 3), 255, np.uint8)
        t[5:5 + im.shape[0], 5:5 + im.shape[1]] = im
        cv2.putText(t, f"{o['id']} v{o['n_views']}", (5, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(t)
    while len(tiles) % a.cols:
        tiles.append(np.full((190, 170, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    out = sd / f"sheet_{a.walk}.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(out)


if __name__ == "__main__":
    main()
