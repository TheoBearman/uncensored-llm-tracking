"""Detect quantization format and bit-width from a HF model record."""
import re

FORMATS = ["gguf", "gptq", "awq", "exl2", "exl3", "mlx", "bitsandbytes", "fp8", "marlin", "hqq", "aqlm"]

_BITS_RE = re.compile(r"(?:^|[-_/.])([2-8])[-_]?bit(?:[-_]|$)", re.I)
_GGUF_QUANT_RE = re.compile(r"\b(Q[2-8]_[KS01]_?[A-Z]?|IQ[1-4]_[A-Z]+|F16|F32|BF16)\b", re.I)


def _norm(s):
    return (s or "").lower()


def detect_format(repo_id: str, tags=None, library_name=None, filenames=None) -> str | None:
    tags = [_norm(t) for t in (tags or [])]
    name = _norm(repo_id)
    lib = _norm(library_name)
    files = [_norm(f) for f in (filenames or [])]

    if "gguf" in tags or "gguf" in name or any(f.endswith(".gguf") for f in files):
        return "gguf"
    if lib == "mlx" or "mlx" in tags or "mlx" in name:
        return "mlx"
    for fmt in ["gptq", "awq", "exl3", "exl2", "marlin", "aqlm", "hqq", "fp8"]:
        if fmt in tags or fmt in name:
            return fmt
    if "bitsandbytes" in tags or any(s in name for s in ["bnb-4bit", "bnb-8bit", "nf4", "int4", "int8"]):
        return "bitsandbytes"
    return None


def detect_bits(repo_id: str, tags=None, filenames=None) -> int | None:
    """Return effective bit-width if detectable, else None."""
    name = _norm(repo_id)
    tags = [_norm(t) for t in (tags or [])]
    files = [_norm(f) for f in (filenames or [])]
    blob = " ".join([name] + tags + files)

    if "nf4" in blob or "int4" in blob or "4bit" in blob or "4-bit" in blob:
        return 4
    if "int8" in blob or "8bit" in blob or "8-bit" in blob:
        return 8
    if "fp8" in blob:
        return 8
    if "fp16" in blob or "f16" in blob or "bf16" in blob:
        return 16

    m = _BITS_RE.search(blob)
    if m:
        return int(m.group(1))

    # GGUF Q-quants — pick smallest hint from filenames
    bits_seen = []
    for f in files:
        m2 = _GGUF_QUANT_RE.search(f)
        if not m2:
            continue
        q = m2.group(1).upper()
        if q.startswith("Q"):
            try:
                bits_seen.append(int(q[1]))
            except ValueError:
                pass
        elif q in ("F16", "BF16"):
            bits_seen.append(16)
        elif q == "F32":
            bits_seen.append(32)
        elif q.startswith("IQ"):
            try:
                bits_seen.append(int(q[2]))
            except ValueError:
                pass
    if bits_seen:
        return min(bits_seen)
    return None


def is_quantized(fmt: str | None, bits: int | None) -> bool:
    if fmt in ("gguf", "gptq", "awq", "exl2", "exl3", "bitsandbytes", "aqlm", "hqq", "marlin"):
        return True
    if bits is not None and bits <= 8:
        return True
    return False
