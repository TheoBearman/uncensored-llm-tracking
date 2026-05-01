"""Combine raw HF model_info into a single enriched record."""
from __future__ import annotations

from .quantization import detect_format, detect_bits, is_quantized
from .lineage import parse_base_model, attribute_family, attribute_lab, base_model_relation
from .geo import attribute_country, extract_languages
from .compute_class import estimate_params_b, memory_gb, compute_class


def _safe_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def enrich(record: dict, search_term: str | None = None) -> dict:
    """Map a raw model_info dict (or huggingface_hub ModelInfo) to a flat enriched row."""
    repo_id = _safe_get(record, "modelId") or _safe_get(record, "id") or _safe_get(record, "model_id")
    tags = _safe_get(record, "tags") or []
    card = _safe_get(record, "cardData") or _safe_get(record, "card_data") or {}
    siblings = _safe_get(record, "siblings") or []
    filenames = []
    for s in siblings:
        if isinstance(s, str):
            rfn = s
        else:
            rfn = _safe_get(s, "rfilename") or _safe_get(s, "filename")
        if rfn:
            filenames.append(rfn)
    library = _safe_get(record, "library_name") or _safe_get(card, "library_name")
    pipeline_tag = _safe_get(record, "pipeline_tag")
    license_ = _safe_get(card, "license") or _safe_get(record, "license")
    gated = _safe_get(record, "gated")
    private = _safe_get(record, "private")
    downloads = _safe_get(record, "downloads")
    downloads_all_time = _safe_get(record, "downloads_all_time") or _safe_get(record, "downloadsAllTime")
    likes = _safe_get(record, "likes")
    created_at = _safe_get(record, "createdAt") or _safe_get(record, "created_at")
    last_modified = _safe_get(record, "lastModified") or _safe_get(record, "last_modified")
    author = _safe_get(record, "author") or (repo_id.split("/")[0] if repo_id and "/" in repo_id else None)

    fmt = detect_format(repo_id, tags=tags, library_name=library, filenames=filenames)
    bits = detect_bits(repo_id, tags=tags, filenames=filenames)
    quantized = is_quantized(fmt, bits)

    base_models = parse_base_model(card, tags)
    family = attribute_family(repo_id, base_models)
    lab = attribute_lab(repo_id, base_models, family)
    relation = base_model_relation(card)

    country = attribute_country(repo_id, base_models)
    languages = extract_languages(card, tags)

    params_b = estimate_params_b(repo_id, tags=tags, card_data=card)
    eff_bits = bits if bits is not None else (4 if quantized else 16)
    mem_gb = memory_gb(params_b, eff_bits)
    bucket = compute_class(params_b, eff_bits)

    return {
        "repo_id": repo_id,
        "author": author,
        "search_term": search_term,
        "created_at": str(created_at) if created_at else None,
        "last_modified": str(last_modified) if last_modified else None,
        "downloads_30d": downloads,
        "downloads_all_time": downloads_all_time,
        "likes": likes,
        "license": license_,
        "gated": gated,
        "private": private,
        "library_name": library,
        "pipeline_tag": pipeline_tag,
        "tags": list(tags) if tags else [],
        "languages": languages,
        # Quantization
        "quant_format": fmt,
        "quant_bits": bits,
        "is_quantized": quantized,
        # Lineage
        "base_models": base_models,
        "base_model_relation": relation,
        "family": family,
        "originating_lab": lab,
        # Geo
        "uploader_country": country,
        # Compute class
        "params_b": params_b,
        "memory_gb_estimated": mem_gb,
        "compute_class": bucket,
    }
