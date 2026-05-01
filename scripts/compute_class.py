"""Estimate parameter count and map (params, bits) → consumer-deployability bucket."""
import re

_PARAM_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", re.I)
_MOE_RE = re.compile(r"(\d+)\s*[x*]\s*(\d+(?:\.\d+)?)\s*b", re.I)


def estimate_params_b(repo_id: str, tags=None, card_data=None) -> float | None:
    """Return parameter count in billions (best-effort from name / tags / cardData)."""
    name = (repo_id or "").lower()

    # MoE pattern wins (e.g. 8x7B → 56B total, but active is 12-14B; we report total).
    m = _MOE_RE.search(name)
    if m:
        n_experts = int(m.group(1))
        per_expert = float(m.group(2))
        return round(n_experts * per_expert, 2)

    # Plain "7B" / "70b" / "1.5B" / "350M".
    candidates = []
    for match in _PARAM_RE.finditer(name):
        val = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "b":
            candidates.append(val)
        elif unit == "m":
            candidates.append(val / 1000.0)
    if candidates:
        # When multiple sizes appear (e.g. "llama-3-8b-instruct-7b-merged"), pick the largest.
        return round(max(candidates), 2)

    # Tags like "params:7B".
    for t in (tags or []):
        if isinstance(t, str) and t.lower().startswith("params:"):
            tail = t.split(":", 1)[1]
            for match in _PARAM_RE.finditer(tail):
                val = float(match.group(1))
                unit = match.group(2).lower()
                return round(val if unit == "b" else val / 1000.0, 2)
    return None


def memory_gb(params_b: float | None, bits: int | None) -> float | None:
    """Approximate weights memory footprint in GB for a given bit-width."""
    if params_b is None:
        return None
    b = bits if bits is not None else 16  # default fp16 if unspecified
    return round(params_b * b / 8.0, 2)


def compute_class(params_b: float | None, bits: int | None) -> str:
    """Bucket by what hardware can run it.

    Buckets reflect framing used in policy docs ("consumer GPU", "laptop") rather
    than precise VRAM math — intentional, since serving overhead, KV cache, and
    context length all shift the real ceiling. 4-bit assumed when bits unknown
    and a 'quantized' tag is present (caller passes effective bits).
    """
    if params_b is None:
        return "unknown"
    mem = memory_gb(params_b, bits)
    if mem is None:
        return "unknown"
    if mem <= 2:
        return "phone"          # ≤2 GB — runs on phones / Raspberry Pi
    if mem <= 6:
        return "laptop-cpu"     # ≤6 GB — laptop CPU / 8GB RAM
    if mem <= 10:
        return "consumer-gpu-12gb"  # RTX 3060 12GB / 4070
    if mem <= 22:
        return "consumer-gpu-24gb"  # RTX 3090 / 4090
    if mem <= 48:
        return "workstation"        # dual-GPU or A6000
    if mem <= 160:
        return "single-node-server"
    return "datacenter"
