"""Stage 4: compare a new round against the baseline register and write a receipt.

Two stages, after Yodo's "Asset identity across seasons" write-up:
  1. unchanged: baseline asset and round object close in 3D AND similar fingerprint (Hungarian).
  2. leftovers: same fingerprint far away -> moved (two near-equal candidates -> ambiguous, to a person).
Anything still unexplained is checked in 3D with the other round's cameras and depth:
  - the spot was in view, unoccluded, and is now EMPTY       -> removed  (baseline asset)
                                          / was EMPTY before  -> added    (new object, proposed row)
  - the spot was in view and something is still THERE        -> review   (unmatched but occupied)
  - the spot was never seen clearly (out of frame / occluded) -> not seen clearly
"Not seen" is never reported as "unchanged" or "removed".
"""
import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import scene_dir, load_json, save_json, load_cameras, project

DEFAULTS = dict(
    r_static=0.06,     # x scene scale: max 3D distance to count as "not moved"
    sim_static=0.55,   # min fingerprint cosine for stage 1
    sim_moved=0.72,    # min fingerprint cosine to explain a leftover as "moved"
    ambiguity=0.03,    # if 2nd-best candidate is within this cosine, send to review
    in_frame=0.7,      # share of the object's points that must project inside the image
    unoccluded=0.6,    # share of in-frame points not hidden behind something closer
    depth_tol=0.08,    # relative depth band for "surface still there"
    verdict=0.6,       # share of points that must agree on empty / occupied
    min_clear=3,       # frames with a clear view needed for any verdict
    # v2 (written after round B, see README "What round B taught us")
    size_frac=0.0,     # static radius grows with object size: max(r_static*scale, size_frac*extent)
    small_band=1.0,    # moves below small_band * radius go to a person, not "moved"
    in_place=False,    # pair a "removed" and a new object in the same 3D spot -> "changed in place"
)

V2 = dict(size_frac=0.25, small_band=1.5, in_place=True)


def load_objs(sd, walk):
    d = load_json(sd / f"objects_{walk}.json")
    for o in d["objects"]:
        o["c"] = np.array(o["centroid"])
        o["e"] = np.array(o["emb"])
        o["P"] = np.array(o["points"])
    return d


def look(P, frames, cams, depth, prm, scale):
    """Did the other round's cameras see this 3D spot clearly, and was it empty or occupied?"""
    clear, votes = [], []
    for n in frames:
        c = cams[n]
        uv, z = project(c, P)
        inside = (z > 0.05 * scale) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < c["w"]) & (uv[:, 1] < c["h"])
        if inside.mean() < prm["in_frame"]:
            continue
        D = depth(n)
        if D is None:
            continue
        u = np.clip((uv[inside, 0] / 2).astype(int), 0, D.shape[1] - 1)
        v = np.clip((uv[inside, 1] / 2).astype(int), 0, D.shape[0] - 1)
        d, zz = D[v, u].astype(np.float32), z[inside]
        valid = d > 0
        occluded = valid & (d < zz * (1 - prm["depth_tol"]))
        empty = valid & (d > zz * (1 + prm["depth_tol"]))
        occupied = valid & ~occluded & ~empty
        seen = empty | occupied
        if seen.sum() < prm["unoccluded"] * inside.sum():
            continue
        clear.append(n)
        fe = empty.sum() / max(seen.sum(), 1)
        votes.append("empty" if fe >= prm["verdict"] else "occupied" if 1 - fe >= prm["verdict"] else "unclear")
    if len(clear) < prm["min_clear"]:
        return "not_seen", clear, votes
    e, o = votes.count("empty"), votes.count("occupied")
    if e >= prm["verdict"] * len(votes):
        return "empty", clear, votes
    if o >= prm["verdict"] * len(votes):
        return "occupied", clear, votes
    return "unclear", clear, votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--base", default="A")
    ap.add_argument("--round", required=True)
    ap.add_argument("--params", help="json with threshold overrides (fixed before the test round)")
    ap.add_argument("--v2", action="store_true", help="turn on the three rules added after round B")
    ap.add_argument("--metres-per-unit", type=float, help="from one tape measurement (see scale.py)")
    a = ap.parse_args()
    prm = dict(DEFAULTS, **(load_json(a.params) if a.params else {}))
    prm.pop("_note", None)
    if a.v2:
        prm.update(V2)

    sd = scene_dir(a.scene)
    cams = load_cameras(sd)
    A, B = load_objs(sd, a.base), load_objs(sd, a.round)
    scale = A["scale"]
    mpu = a.metres_per_unit
    unit = (lambda x: round(x * mpu, 3)) if mpu else (lambda x: round(x, 4))
    cache = {}

    def depth(n):
        if n not in cache:
            p = sd / "depth" / f"{n}.npy"
            cache[n] = np.load(p) if p.exists() else None
        return cache[n]

    fa = sorted(n for n, c in cams.items() if c["walk"] == a.base)
    fb = sorted(n for n, c in cams.items() if c["walk"] == a.round)
    oa, ob = A["objects"], B["objects"]
    rows, used_a, used_b = [], set(), set()

    # stage 1: unchanged
    if oa and ob:
        dist = np.linalg.norm(np.stack([o["c"] for o in oa])[:, None] - np.stack([o["c"] for o in ob])[None], axis=2)
        sim = np.stack([o["e"] for o in oa]) @ np.stack([o["e"] for o in ob]).T
        size = np.array([max(o["extent"]) for o in oa])
        r = np.maximum(prm["r_static"] * scale, prm["size_frac"] * size)[:, None]
        gate = (dist < r) & (sim > prm["sim_static"])
        cost = np.where(gate, dist / r + (1 - sim) / (1 - prm["sim_static"]), 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if gate[i, j]:
                used_a.add(i), used_b.add(j)
                rows.append(dict(asset=oa[i]["id"], state="unchanged", matched=ob[j]["id"],
                                 moved_by=unit(dist[i, j]), similarity=round(float(sim[i, j]), 3)))

        # stage 2: moved / ambiguous
        la = [i for i in range(len(oa)) if i not in used_a]
        lb = [j for j in range(len(ob)) if j not in used_b]
        if la and lb:
            s2 = sim[np.ix_(la, lb)]
            c2 = np.where(s2 > prm["sim_moved"], 1 - s2, 1e6)
            for ii, jj in zip(*linear_sum_assignment(c2)):
                if c2[ii, jj] >= 1e6:
                    continue
                i, j = la[ii], lb[jj]
                rivals = sorted(s2[ii], reverse=True) + sorted(s2[:, jj], reverse=True)
                close = [s for s in rivals if s > prm["sim_moved"] and abs(s - s2[ii, jj]) < prm["ambiguity"]]
                ambiguous = len(close) > 2  # itself appears twice in rivals
                small = dist[i, j] < prm["small_band"] * r[i, 0]
                state = "review" if ambiguous or small else "moved"
                used_a.add(i), used_b.add(j)
                rows.append(dict(asset=oa[i]["id"], state=state, matched=ob[j]["id"],
                                 moved_by=unit(dist[i, j]), similarity=round(float(s2[ii, jj]), 3),
                                 why="ambiguous: another candidate looks the same" if ambiguous else
                                 "possible small move, inside the noise band" if small else
                                 "same fingerprint, new position"))

    # leftovers: check the spot in 3D with the other round's views
    for i, o in enumerate(oa):
        if i in used_a:
            continue
        v, clear, votes = look(o["P"], fb, cams, depth, prm, scale)
        state = {"empty": "removed", "occupied": "review", "unclear": "review", "not_seen": "not_seen"}[v]
        rows.append(dict(asset=o["id"], state=state, clear_views=len(clear), votes=votes,
                         why={"removed": "spot seen clearly and empty",
                              "review": f"no match, but the spot looks {v}",
                              "not_seen": f"only {len(clear)} clear views of the spot"}[state]))
    for j, o in enumerate(ob):
        if j in used_b:
            continue
        v, clear, votes = look(o["P"], fa, cams, depth, prm, scale)
        state = {"empty": "added", "occupied": "review", "unclear": "review", "not_seen": "added_unverified"}[v]
        rows.append(dict(asset=None, new=o["id"], state=state, clear_views=len(clear), votes=votes,
                         why={"added": "spot was seen empty in the baseline",
                              "review": f"new object, but the baseline spot looks {v}",
                              "added_unverified": "baseline never saw this spot clearly"}[state]))

    if prm["in_place"]:
        # a "removed" asset and a new object occupying the same 3D spot: the thing turned or changed shape
        news = [x for x in rows if x.get("new") and x["state"] in ("added", "review", "added_unverified")]
        nb = {o["id"]: o for o in ob}
        for x in [x for x in rows if x["state"] == "removed"]:
            o = oa[[k["id"] for k in oa].index(x["asset"])]
            best, bd = None, None
            for y in news:
                q = nb[y["new"]]
                d = np.linalg.norm(o["c"] - q["c"])
                if d < 0.5 * (max(o["extent"]) + max(q["extent"])) and (bd is None or d < bd):
                    best, bd = y, d
            if best:
                news.remove(best)
                rows.remove(best)
                x.update(state="in_place", matched=best["new"], moved_by=unit(bd),
                         why="spot of the old view is empty, a new shape overlaps it: turned / reshaped")

    order = ["removed", "moved", "in_place", "added", "review", "not_seen", "added_unverified", "unchanged"]
    rows.sort(key=lambda r: (order.index(r["state"]), r.get("asset") or r.get("new")))
    counts = {s: sum(r["state"] == s for r in rows) for s in order}
    rec = dict(base=a.base, round=a.round, units="m" if mpu else "scene units", params=prm,
               counts=counts, rows=rows)
    tag = "_v2" if a.v2 else ""
    save_json(sd / f"receipt_{a.base}_{a.round}{tag}.json", rec)

    label = {"in_place": "CHANGED: in place (turned / reshaped)", "unchanged": "confirmed unchanged", "moved": "CHANGED: moved", "removed": "CHANGED: removed",
             "added": "CHANGED: added (proposed row)", "review": "to a person", "not_seen": "not seen clearly",
             "added_unverified": "new, baseline spot not seen"}
    md = [f"# Receipt: round {a.round} vs baseline {a.base}", "",
          " · ".join(f"{label[k]}: {v}" for k, v in counts.items() if v), "",
          "| Asset | State | Match | Moved by | Evidence |", "|---|---|---|---|---|"]
    for r in rows:
        ev = r.get("why", "") or f"similarity {r.get('similarity')}"
        if "similarity" in r and "why" not in r:
            ev = f"similarity {r['similarity']}"
        md.append(f"| {r.get('asset') or '—'} | {label[r['state']]} | {r.get('matched') or r.get('new') or ''} "
                  f"| {r.get('moved_by', '')} | {ev} |")
    (sd / f"receipt_{a.base}_{a.round}{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
