"""Geo / language enrichment.

Country attribution is best-effort: HF does not expose uploader nationality,
so we rely on a curated namespace → country map for known orgs and labs.
Unknown namespaces are tagged 'unknown' (NOT inferred from name heuristics)
so downstream analysis can transparently report coverage.
"""

# Curated org-namespace → ISO country code (where the org is headquartered).
COUNTRY_BY_NAMESPACE = {
    # US
    "meta-llama": "US", "facebook": "US", "google": "US", "google-deepmind": "US",
    "microsoft": "US", "openai": "US", "openai-community": "US", "nvidia": "US",
    "ibm-granite": "US", "ibm": "US", "allenai": "US", "stabilityai": "US",
    "togethercomputer": "US", "databricks": "US", "snowflake": "US",
    "xai-org": "US", "anthropic": "US", "cohereforai": "CA", "cohere": "CA",
    "huggingfaceh4": "US", "huggingfacem4": "US", "writer": "US",
    # China
    "qwen": "CN", "alibaba-nlp": "CN", "deepseek-ai": "CN", "01-ai": "CN",
    "moonshotai": "CN", "zhipuai": "CN", "thudm": "CN", "internlm": "CN",
    "baichuan-inc": "CN", "01ai": "CN",
    # France
    "mistralai": "FR",
    # UAE
    "tiiuae": "AE",
    # Israel
    "ai21labs": "IL",
    # Korea
    "lgai-exaone": "KR", "kakaocorp": "KR", "naver": "KR", "upstage": "KR",
    # Japan
    "rinna": "JP", "elyza": "JP", "cyberagent": "JP",
    # UK
    "deepmind": "GB",
    # Misc
    "rekaai": "US",
    "bigscience": "MULTI",  # consortium
    "bigcode": "MULTI",
    "eleutherai": "MULTI",
}


def attribute_country(repo_id: str, base_models=None) -> str | None:
    ns = (repo_id or "").split("/")[0].lower()
    if ns in COUNTRY_BY_NAMESPACE:
        return COUNTRY_BY_NAMESPACE[ns]
    for bm in (base_models or []):
        bns = (bm or "").split("/")[0].lower()
        if bns in COUNTRY_BY_NAMESPACE:
            # community fork inherits attribution as 'derivative' — caller may want
            # to keep these separate; we return None for unknown community uploaders.
            return None
    return None


def extract_languages(card_data, tags) -> list[str]:
    out = []
    if isinstance(card_data, dict):
        lang = card_data.get("language")
        if isinstance(lang, str):
            out.append(lang)
        elif isinstance(lang, list):
            out.extend([x for x in lang if isinstance(x, str)])
    for t in (tags or []):
        if isinstance(t, str) and t.startswith("language:"):
            out.append(t.split(":", 1)[1])
    seen, uniq = set(), []
    for x in out:
        x = (x or "").lower().strip()
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq
