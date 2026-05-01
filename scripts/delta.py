"""Diff two snapshots to surface policy-relevant changes.

Outputs <new>/deltas.json with:
  added           — repos present in new but not old
  removed         — repos in old but not new (likely takedowns / privatizations)
  download_surges — top movers by 30-day download delta
  license_flips   — repos whose license string changed
  gated_changes   — repos that flipped gated/private status
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(snapshot_dir: Path) -> dict[str, dict]:
    parquet = snapshot_dir / "models.parquet"
    csv = snapshot_dir / "models.csv"
    if parquet.exists():
        import pandas as pd
        df = pd.read_parquet(parquet)
        return {r["repo_id"]: r for r in df.to_dict("records") if r.get("repo_id")}
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv)
        return {r["repo_id"]: r for r in df.to_dict("records") if r.get("repo_id")}
    raise FileNotFoundError(f"no models.parquet or models.csv in {snapshot_dir}")


def _safe(v):
    try:
        if v is None or (isinstance(v, float) and v != v):  # NaN
            return None
    except Exception:
        pass
    return v


def diff(old: dict[str, dict], new: dict[str, dict], surge_top_n: int = 50) -> dict:
    old_ids, new_ids = set(old), set(new)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)

    surges = []
    license_flips = []
    gated_changes = []
    for rid in old_ids & new_ids:
        o, n = old[rid], new[rid]
        o_dl, n_dl = _safe(o.get("downloads_30d")), _safe(n.get("downloads_30d"))
        if o_dl is not None and n_dl is not None:
            delta = (n_dl or 0) - (o_dl or 0)
            if delta > 0:
                surges.append((rid, o_dl, n_dl, delta))
        o_lic, n_lic = _safe(o.get("license")), _safe(n.get("license"))
        if o_lic != n_lic:
            license_flips.append({"repo_id": rid, "old": o_lic, "new": n_lic})
        o_g, n_g = _safe(o.get("gated")), _safe(n.get("gated"))
        o_p, n_p = _safe(o.get("private")), _safe(n.get("private"))
        if o_g != n_g or o_p != n_p:
            gated_changes.append({
                "repo_id": rid,
                "old_gated": o_g, "new_gated": n_g,
                "old_private": o_p, "new_private": n_p,
            })

    surges.sort(key=lambda x: x[3], reverse=True)
    surges_out = [{"repo_id": r, "old_30d": a, "new_30d": b, "delta": d}
                  for r, a, b, d in surges[:surge_top_n]]

    # Enrich added/removed with light context where available.
    def _ctx(rid, src):
        r = src.get(rid, {})
        return {
            "repo_id": rid,
            "family": _safe(r.get("family")),
            "originating_lab": _safe(r.get("originating_lab")),
            "uploader_country": _safe(r.get("uploader_country")),
            "quant_format": _safe(r.get("quant_format")),
            "downloads_30d": _safe(r.get("downloads_30d")),
        }

    return {
        "n_added": len(added),
        "n_removed": len(removed),
        "added": [_ctx(r, new) for r in added],
        "removed": [_ctx(r, old) for r in removed],
        "download_surges": surges_out,
        "license_flips": license_flips,
        "gated_changes": gated_changes,
    }


def main():
    p = argparse.ArgumentParser(description="Diff two dated snapshots.")
    p.add_argument("old", help="Path to older snapshot directory (e.g. snapshots/2026-04-24)")
    p.add_argument("new", help="Path to newer snapshot directory")
    p.add_argument("--out", default=None, help="Output path (default: <new>/deltas.json)")
    args = p.parse_args()

    old_dir, new_dir = Path(args.old), Path(args.new)
    old_rows = _load_rows(old_dir)
    new_rows = _load_rows(new_dir)
    result = diff(old_rows, new_rows)
    result["old_snapshot"] = old_dir.name
    result["new_snapshot"] = new_dir.name

    out = Path(args.out) if args.out else (new_dir / "deltas.json")
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[delta] {old_dir.name} -> {new_dir.name}: "
          f"+{result['n_added']} -{result['n_removed']}, "
          f"{len(result['download_surges'])} surges, "
          f"{len(result['license_flips'])} license flips, "
          f"{len(result['gated_changes'])} gated changes -> {out}")


if __name__ == "__main__":
    main()
