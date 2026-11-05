"""Synthetic scene with known answers: checks the receipt logic without any video.

A desk (back wall at z=3) with blob 'objects'. Cameras on an arc look at it; depth maps are
rendered by z-buffering the blobs over the wall. Round B: one blob moved, one removed, one
added, one hidden behind a board, two identical blobs where only one moves.
"""
import json
import subprocess
import sys

import numpy as np

from seenagain.common import project

RNG = np.random.default_rng(1)
W, H, F = 640, 480, 500.0


def look_at(center, target):
    f = target - center
    f /= np.linalg.norm(f)
    r = np.cross([0, -1, 0], f)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return dict(w=W, h=H, fx=F, fy=F, cx=W / 2, cy=H / 2, R=R, t=-R @ center)


def blob(c, s=0.08, n=400):
    return np.asarray(c) + RNG.normal(0, s / 2, (n, 3))


def render(cam, clouds, wall_z=3.0):
    D = np.full((H // 2, W // 2), 0.0, np.float32)
    # wall: depth along the ray to the plane z=wall_z (cameras at z~0, looking at +z)
    vs, us = np.mgrid[0:H // 2, 0:W // 2]
    rays = np.stack([(us * 2 - cam["cx"]) / F, (vs * 2 - cam["cy"]) / F, np.ones_like(us, float)], -1)
    dirs = rays @ cam["R"]  # world directions (unnormalised, z_cam = 1)
    C = -cam["R"].T @ cam["t"]
    D[:] = (wall_z - C[2]) / dirs[..., 2]
    for P in clouds:
        uv, z = project(cam, P)
        for (u, v), zz in zip((uv / 2).astype(int), z):
            for du in (-2, -1, 0, 1, 2):
                for dv in (-2, -1, 0, 1, 2):
                    x, y = u + du, v + dv
                    if 0 <= x < W // 2 and 0 <= y < H // 2 and zz > 0 and zz < D[y, x]:
                        D[y, x] = zz
    return D


def emb(seed):
    e = np.random.default_rng(seed).normal(size=384)
    return e / np.linalg.norm(e)


def write_walk(sd, walk, objs, extra_occluders, cams):
    allc = [o[2] for o in objs] + extra_occluders
    for n, c in cams.items():
        if c["walk"] == walk:
            np.save(sd / "depth" / f"{n}.npy", render(c, allc).astype(np.float16))
    out = []
    for oid, e, P in objs:
        noisy = e + RNG.normal(0, 0.02, e.shape)
        out.append(dict(id=oid, walk=walk, centroid=np.median(P, 0).tolist(), extent=[0.1] * 3,
                        frames=[], n_views=5, emb=(noisy / np.linalg.norm(noisy)).tolist(),
                        points=P[::4].tolist(), thumb=""))
    (sd / f"objects_{walk}.json").write_text(json.dumps(dict(walk=walk, scale=2.5, objects=out)))


def test_receipt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sd = tmp_path / "work" / "syn"
    (sd / "depth").mkdir(parents=True)
    target = np.array([0, 0, 2.5])
    cams = {}
    for walk in ["A", "B"]:
        for k, ang in enumerate(np.linspace(-0.5, 0.5, 9)):
            c = look_at(np.array([2.0 * np.sin(ang), -0.3, 2.5 - 2.0 * np.cos(ang)]), target)
            c["walk"] = walk
            cams[f"{walk}_{k:04d}"] = c
    (sd / "cameras.json").write_text(json.dumps(
        {n: dict(c, R=c["R"].tolist(), t=c["t"].tolist()) for n, c in cams.items()}))

    pos = dict(lamp=[-0.6, 0, 2.6], book=[-0.2, 0.1, 2.5], plant=[0.3, 0, 2.6],
               cup1=[0.6, 0.1, 2.4], cup2=[0.8, 0.1, 2.4], box=[0.0, -0.3, 2.7])
    E = dict(lamp=emb(1), book=emb(2), plant=emb(3), cup1=emb(9), cup2=emb(9), box=emb(5))
    A = [(f"A-{k}", E[k], blob(p)) for k, p in pos.items()]
    B = [("B-lamp", E["lamp"], blob(pos["lamp"])),                           # unchanged
         ("B-plant", E["plant"], blob([0.35, -0.25, 2.3])),                  # moved
         ("B-cup1", E["cup1"], blob(pos["cup1"])),                           # identical cup, stays
         ("B-cup2", E["cup2"], blob([0.75, -0.3, 2.3])),                     # identical cup, moved
         ("B-bag", emb(7), blob([-0.4, -0.3, 2.3]))]                         # added
    # book: removed (spot visible and empty). box: hidden behind a board in round B.
    board = np.concatenate([np.stack([np.full(400, x), RNG.uniform(-0.5, -0.1, 400),
                                      RNG.uniform(2.0, 2.2, 400)], 1) for x in np.linspace(-0.25, 0.25, 15)])
    write_walk(sd, "A", A, [], cams)
    write_walk(sd, "B", B, [board], cams)

    r = subprocess.run([sys.executable, "-m", "seenagain.match", "--scene", "syn", "--round", "B"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rec = json.loads((sd / "receipt_A_B.json").read_text())
    got = {(x.get("asset") or x.get("new")): x["state"] for x in rec["rows"]}
    print(r.stdout)
    assert got["A-lamp"] == "unchanged"
    assert got["A-cup1"] == "unchanged"
    assert got["A-plant"] == "moved"
    assert got["A-cup2"] == "moved"
    assert got["A-book"] == "removed"
    assert got["A-box"] == "not_seen"
    assert got["B-bag"] == "added"
