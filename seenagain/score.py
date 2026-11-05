"""Score receipts against ground truth: changes caught, right type, false changes, load on people."""
import argparse

from .common import scene_dir, load_json

CHANGE = {"moved", "removed", "added", "in_place"}
TYPE_OK = {"rotated in place": {"in_place"}, "moved": {"moved"}}


def score(rec, gt):
    rows = {r["asset"]: r for r in rec["rows"] if r.get("asset")}
    out = dict(changes=len(gt), caught=0, typed=0, missed=[], false_changes=[], review=0, not_seen=0, detail=[])
    for aid, g in gt.items():
        r = rows.get(aid)
        state = r["state"] if r else "not in register"
        caught = state in CHANGE or state == "review"
        kind = "moved" if g["truth"].startswith("moved") else g["truth"]
        typed = state in TYPE_OK.get(kind, set())
        out["caught"] += caught and state in CHANGE
        out["typed"] += typed
        if not caught:
            out["missed"].append(aid)
        out["detail"].append((g["what"], g["truth"], state + (f" {r['moved_by']} m" if r and r.get("moved_by") else "")))
    for aid, r in rows.items():
        if aid not in gt and r["state"] in CHANGE:
            out["false_changes"].append(f"{aid} {r['state']}")
    out["review"] = sum(r["state"] == "review" for r in rec["rows"])
    out["not_seen"] = sum(r["state"] in ("not_seen", "added_unverified") for r in rec["rows"])
    out["unchanged"] = rec["counts"].get("unchanged", 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    sd = scene_dir(a.scene)
    gt = load_json(a.gt)
    lines = ["| Round | Rules | Real changes | Caught as a change | Right type | False changes | To a person | Not seen | Confirmed unchanged |",
             "|---|---|---|---|---|---|---|---|---|"]
    details = []
    for rnd in ["A2", "B"]:
        for tag, name in [("", "v1 (frozen on control)"), ("_v2", "v2 (post-hoc, after B)")]:
            p = sd / f"receipt_A_{rnd}{tag}.json"
            if not p.exists():
                continue
            s = score(load_json(p), gt[rnd]["changes"])
            label = "A2 (control, nothing changed)" if rnd == "A2" else "B (5 real changes)"
            lines.append(f"| {label} | {name} | {s['changes']} | {s['caught']} | {s['typed']} | "
                         f"{len(s['false_changes'])}{' (' + ', '.join(s['false_changes']) + ')' if s['false_changes'] else ''} | "
                         f"{s['review']} | {s['not_seen']} | {s['unchanged']} |")
            if s["detail"]:
                details.append(f"\n**Round {rnd}, {name}**\n\n| Object | Truth | System |\n|---|---|---|")
                details += [f"| {w} | {t} | {st} |" for w, t, st in s["detail"]]
    text = "\n".join(lines) + "\n" + "\n".join(details) + "\n"
    (sd / "scores.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
