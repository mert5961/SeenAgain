"""Shared paths and small helpers (no torch / pycolmap imports here: the two clash in one process)."""
import json
import pathlib

import numpy as np


def scene_dir(scene):
    d = pathlib.Path("work") / scene
    d.mkdir(parents=True, exist_ok=True)
    return d


def walk_of(name):
    """Image names are '<walk>_<index>.jpg'."""
    return name.rsplit("_", 1)[0]


def load_json(p):
    return json.loads(pathlib.Path(p).read_text())


def save_json(p, obj):
    pathlib.Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_cameras(sd):
    """cameras.json written by sfm.py: one entry per registered (undistorted, pinhole) image."""
    cams = load_json(sd / "cameras.json")
    for c in cams.values():
        c["R"] = np.array(c["R"])
        c["t"] = np.array(c["t"])
    return cams


def world_to_cam(c, X):
    return X @ c["R"].T + c["t"]


def project(c, X):
    """World points (N,3) -> pixel uv (N,2) and depth z (N,)."""
    Xc = world_to_cam(c, X)
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    u = c["fx"] * Xc[:, 0] / zs + c["cx"]
    v = c["fy"] * Xc[:, 1] / zs + c["cy"]
    return np.stack([u, v], 1), z


def unproject(c, uv, z):
    """Pixels (N,2) + depth (N,) -> world points (N,3)."""
    x = (uv[:, 0] - c["cx"]) / c["fx"] * z
    y = (uv[:, 1] - c["cy"]) / c["fy"] * z
    Xc = np.stack([x, y, z], 1)
    return (Xc - c["t"]) @ c["R"]  # R^T (Xc - t)


def camera_center(c):
    return -c["R"].T @ c["t"]
