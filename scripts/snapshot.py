"""Run a full snapshot of safety-keyword-matching HF models for a given date.

Writes:
  snapshots/<DATE>/raw/models.jsonl       — one raw model_info dict per line
  snapshots/<DATE>/models.parquet         — flat enriched table
  snapshots/<DATE>/summary.json           — aggregates for the static site
  snapshots/<DATE>/manifest.json          — search terms, counts, runtime metadata
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Allow `python scripts/snapshot.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.enrich import enrich  # noqa: E402


def _coerce(obj):
    """Make a model_info object JSON-serializable as a dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        out = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v, default=str)
                out[k] = v
            except TypeError:
                out[k] = json.loads(json.dumps(v, default=str))
        return out
    return {"_repr": repr(obj)}


def parse_terms_file(path: str) -> list[tuple[str, int | None]]:
    """Parse a terms file. Each non-comment line is `term` or `term:cap`."""
    out: list[tuple[str, int | None]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                term, cap_s = line.rsplit(":", 1)
                term = term.strip()
                try:
                    cap: int | None = int(cap_s.strip())
                except ValueError:
                    # Treat 'no-saf' / 'de-align' style terms as uncapped — the colon
                    # was actually part of the term, not a cap. Re-attach it.
                    term, cap = line, None
            else:
                term, cap = line, None
            if term:
                out.append((term, cap))
    return out


def discover_repo_ids(
    api: HfApi,
    terms_with_caps: list[tuple[str, int | None]],
    rate_limit: float,
    default_per_term_cap: int | None = None,
    max_repos: int | None = None,
):
    """Return {repo_id: set(search_terms)} for repos hit by any term.

    terms_with_caps: list of (term, cap) — cap of None means "use default_per_term_cap"
                     (which itself can be None for fully uncapped).
    max_repos:       early-exit once the unique-repo count crosses this threshold.
    """
    hits: dict[str, set[str]] = defaultdict(set)
    n = len(terms_with_caps)
    for ti, (term, term_cap) in enumerate(terms_with_caps, 1):
        cap = term_cap if term_cap is not None else default_per_term_cap
        before = len(hits)
        n_term = 0
        try:
            for m in api.list_models(search=term):
                rid = getattr(m, "modelId", None) or getattr(m, "id", None)
                if rid:
                    hits[rid].add(term)
                n_term += 1
                if cap and n_term >= cap:
                    print(f"[discover] {term!r}: capped at {cap}", flush=True)
                    break
                if n_term % 1000 == 0:
                    print(f"[discover] {term!r}: {n_term} so far (unique total: {len(hits)})", flush=True)
        except Exception as e:
            print(f"[discover] error on term '{term}': {e}", file=sys.stderr, flush=True)
        added = len(hits) - before
        cap_note = f" (cap={cap})" if cap else " (uncapped)"
        print(f"[discover] {ti}/{n} {term!r}: {n_term} matches, +{added} new"
              f"{cap_note} (unique total: {len(hits)})", flush=True)
        if max_repos and len(hits) >= max_repos:
            print(f"[discover] reached max_repos={max_repos}, stopping discovery", flush=True)
            break
        time.sleep(rate_limit)
    return hits


def fetch_with_retry(api: HfApi, repo_id: str, max_retries: int = 5):
    attempt = 0
    while True:
        try:
            return api.model_info(
                repo_id,
                expand=[
                    "downloads", "downloadsAllTime", "likes", "tags",
                    "cardData", "siblings", "library_name", "pipeline_tag",
                    "gated", "private", "license", "createdAt", "lastModified",
                    "author",
                ],
            )
        except HfHubHTTPError as e:
            attempt += 1
            if attempt > max_retries:
                return {"id": repo_id, "_error": f"HTTP {getattr(e.response, 'status_code', '?')}"}
            time.sleep(0.5 * (2 ** (attempt - 1)))
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                return {"id": repo_id, "_error": str(e)[:200]}
            time.sleep(0.5 * (2 ** (attempt - 1)))


def build_summary(rows: list[dict], date: str, search_terms: list[str]) -> dict:
    n = len(rows)
    by_country = Counter(r.get("uploader_country") or "unknown" for r in rows)
    by_lab = Counter(r.get("originating_lab") or "unknown" for r in rows)
    by_family = Counter(r.get("family") or "unknown" for r in rows)
    by_quant = Counter(r.get("quant_format") or "unquantized" for r in rows)
    by_bucket = Counter(r.get("compute_class") or "unknown" for r in rows)
    by_license = Counter((r.get("license") or "unspecified") for r in rows)
    by_lang = Counter()
    for r in rows:
        for lang in (r.get("languages") or [])[:3]:
            by_lang[lang] += 1

    quantized_n = sum(1 for r in rows if r.get("is_quantized"))
    gguf_n = sum(1 for r in rows if r.get("quant_format") == "gguf")

    # Downloads aggregates — only for rows where data is available.
    dl_30d = sum((r.get("downloads_30d") or 0) for r in rows)
    dl_total = sum((r.get("downloads_all_time") or 0) for r in rows)

    # Top movers (most-downloaded this month).
    top_30d = sorted(
        [r for r in rows if r.get("downloads_30d") is not None],
        key=lambda r: r["downloads_30d"], reverse=True,
    )[:25]

    return {
        "date": date,
        "search_terms": search_terms,
        "total_repos": n,
        "quantized_repos": quantized_n,
        "gguf_repos": gguf_n,
        "downloads_30d_sum": dl_30d,
        "downloads_all_time_sum": dl_total,
        "by_uploader_country": dict(by_country.most_common()),
        "by_originating_lab": dict(by_lab.most_common()),
        "by_family": dict(by_family.most_common()),
        "by_quant_format": dict(by_quant.most_common()),
        "by_compute_class": dict(by_bucket.most_common()),
        "by_license": dict(by_license.most_common()),
        "by_language": dict(by_lang.most_common(20)),
        "top_30d": [
            {"repo_id": r["repo_id"], "downloads_30d": r["downloads_30d"],
             "family": r.get("family"), "originating_lab": r.get("originating_lab"),
             "quant_format": r.get("quant_format")}
            for r in top_30d
        ],
    }


def write_outputs(out_dir: Path, raw_records: list[dict], rows: list[dict], summary: dict, manifest: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    with (raw_dir / "models.jsonl").open("w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec, default=str) + "\n")

    # Parquet (preferred) with CSV fallback.
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "models.parquet", index=False)
    except Exception as e:
        print(f"[snapshot] parquet write failed ({e}), writing CSV", file=sys.stderr)
        import csv
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            with (out_dir / "models.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def update_index(snapshots_root: Path):
    """Maintain snapshots/index.json — a manifest the static site reads."""
    entries = []
    for d in sorted(snapshots_root.iterdir()):
        if not d.is_dir():
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries.append({
            "date": s.get("date") or d.name,
            "total_repos": s.get("total_repos"),
            "quantized_repos": s.get("quantized_repos"),
            "gguf_repos": s.get("gguf_repos"),
            "downloads_30d_sum": s.get("downloads_30d_sum"),
            "summary_path": f"snapshots/{d.name}/summary.json",
        })
    (snapshots_root / "index.json").write_text(
        json.dumps({"snapshots": entries}, indent=2), encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser(description="Take a dated snapshot of HF safety-keyword models.")
    p.add_argument("--terms", default="snapshot_terms.txt",
                   help="Curated terms file. Lines may be 'term' or 'term:cap'.")
    p.add_argument("--out", default="snapshots", help="Snapshots root directory")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (defaults to UTC today)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None, help="Cap on repos enriched (for dry runs)")
    p.add_argument("--per-term-cap", type=int, default=None,
                   help="Default cap for terms without an inline cap (None = uncapped)")
    p.add_argument("--max-repos", type=int, default=50000,
                   help="Global ceiling — stop discovery once unique repos crosses this")
    args = p.parse_args()

    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(args.out) / date

    terms_with_caps = parse_terms_file(args.terms)
    search_terms = [t for t, _ in terms_with_caps]

    api = HfApi(token=args.hf_token) if args.hf_token else HfApi()

    capped = [(t, c) for t, c in terms_with_caps if c is not None]
    print(f"[snapshot] {date}: discovering for {len(terms_with_caps)} terms "
          f"({len(capped)} with inline caps, default_cap={args.per_term_cap}, "
          f"max_repos={args.max_repos})", flush=True)
    t0 = time.time()
    hits = discover_repo_ids(
        api, terms_with_caps, rate_limit=args.rate,
        default_per_term_cap=args.per_term_cap, max_repos=args.max_repos,
    )
    print(f"[snapshot] discovered {len(hits)} unique repos in {time.time() - t0:.1f}s", flush=True)

    repo_ids = list(hits.keys())
    if args.limit:
        repo_ids = repo_ids[: args.limit]

    raw_records, rows = [], []
    for i, rid in enumerate(repo_ids, 1):
        info = fetch_with_retry(api, rid)
        rec = info if isinstance(info, dict) else _coerce(info)
        rec.setdefault("modelId", rid)
        terms_hit = sorted(hits.get(rid, []))
        rec["_search_terms"] = terms_hit
        raw_records.append(rec)

        try:
            row = enrich(rec, search_term=";".join(terms_hit))
        except Exception as e:
            row = {"repo_id": rid, "_enrich_error": str(e)[:200]}
        rows.append(row)

        if i % 25 == 0 or i == len(repo_ids):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (len(repo_ids) - i) / rate if rate else 0
            print(f"[snapshot] enriched {i}/{len(repo_ids)} "
                  f"({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)
        time.sleep(args.rate)

    summary = build_summary(rows, date, search_terms)
    manifest = {
        "date": date,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_search_terms": len(search_terms),
        "n_unique_repos": len(repo_ids),
        "limit_applied": args.limit,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    write_outputs(out_dir, raw_records, rows, summary, manifest)
    update_index(Path(args.out))
    print(f"[snapshot] done -> {out_dir}")


if __name__ == "__main__":
    main()
