"""Resolve base-model lineage and originating-lab attribution."""
import re

# Originating lab for well-known base-model namespaces / name fragments.
# Used when cardData.base_model is missing or points to a derivative.
LAB_BY_NAMESPACE = {
    "meta-llama": "Meta",
    "facebook": "Meta",
    "mistralai": "Mistral AI",
    "google": "Google",
    "google-deepmind": "Google",
    "deepmind": "Google",
    "microsoft": "Microsoft",
    "openai": "OpenAI",
    "openai-community": "OpenAI",
    "qwen": "Alibaba",
    "alibaba-nlp": "Alibaba",
    "deepseek-ai": "DeepSeek",
    "01-ai": "01.AI",
    "tiiuae": "TII",
    "stabilityai": "Stability AI",
    "bigcode": "BigCode",
    "bigscience": "BigScience",
    "eleutherai": "EleutherAI",
    "nvidia": "NVIDIA",
    "ibm-granite": "IBM",
    "ibm": "IBM",
    "cohereforai": "Cohere",
    "cohere": "Cohere",
    "allenai": "Allen AI",
    "huggingfaceh4": "Hugging Face",
    "huggingfacem4": "Hugging Face",
    "togethercomputer": "Together",
    "databricks": "Databricks",
    "snowflake": "Snowflake",
    "xai-org": "xAI",
    "moonshotai": "Moonshot",
    "zhipuai": "Zhipu",
    "thudm": "Zhipu",
    "internlm": "Shanghai AI Lab",
    "baichuan-inc": "Baichuan",
    "01ai": "01.AI",
    "rekaai": "Reka",
    "ai21labs": "AI21",
    "writer": "Writer",
    "upstage": "Upstage",
    "lgai-exaone": "LG",
    "kakaocorp": "Kakao",
    "naver": "Naver",
    "rinna": "Rinna",
    "elyza": "ELYZA",
    "cyberagent": "CyberAgent",
}

# Family-by-substring fallback when namespace is opaque (community uploader).
FAMILY_BY_SUBSTRING = [
    ("llama-3", "Llama 3"),
    ("llama3", "Llama 3"),
    ("llama-2", "Llama 2"),
    ("llama2", "Llama 2"),
    ("llama-1", "Llama 1"),
    ("llama-7", "Llama 1"),
    ("llama-13", "Llama 1"),
    ("llama-30", "Llama 1"),
    ("llama-65", "Llama 1"),
    ("mixtral", "Mixtral"),
    ("mistral", "Mistral"),
    ("qwen3", "Qwen 3"),
    ("qwen2.5", "Qwen 2.5"),
    ("qwen2", "Qwen 2"),
    ("qwen-", "Qwen"),
    ("qwen_", "Qwen"),
    ("gemma-3", "Gemma 3"),
    ("gemma-2", "Gemma 2"),
    ("gemma", "Gemma"),
    ("phi-3", "Phi 3"),
    ("phi-4", "Phi 4"),
    ("phi3", "Phi 3"),
    ("phi-2", "Phi 2"),
    ("deepseek", "DeepSeek"),
    ("yi-1.5", "Yi 1.5"),
    ("yi-", "Yi"),
    ("falcon", "Falcon"),
    ("mpt-", "MPT"),
    ("pythia", "Pythia"),
    ("bloom", "BLOOM"),
    ("starcoder", "StarCoder"),
    ("codellama", "Code Llama"),
    ("opt-", "OPT"),
    ("granite", "Granite"),
    ("command-r", "Command R"),
    ("dbrx", "DBRX"),
    ("olmo", "OLMo"),
    ("nemotron", "Nemotron"),
    ("solar", "Solar"),
    ("internlm", "InternLM"),
    ("baichuan", "Baichuan"),
]

LAB_BY_FAMILY = {
    "Llama 1": "Meta", "Llama 2": "Meta", "Llama 3": "Meta",
    "Code Llama": "Meta", "OPT": "Meta",
    "Mistral": "Mistral AI", "Mixtral": "Mistral AI",
    "Qwen": "Alibaba", "Qwen 2": "Alibaba", "Qwen 2.5": "Alibaba", "Qwen 3": "Alibaba",
    "Gemma": "Google", "Gemma 2": "Google", "Gemma 3": "Google",
    "Phi 2": "Microsoft", "Phi 3": "Microsoft", "Phi 4": "Microsoft",
    "DeepSeek": "DeepSeek",
    "Yi": "01.AI", "Yi 1.5": "01.AI",
    "Falcon": "TII", "MPT": "MosaicML",
    "Pythia": "EleutherAI", "BLOOM": "BigScience",
    "StarCoder": "BigCode", "Granite": "IBM",
    "Command R": "Cohere", "DBRX": "Databricks", "OLMo": "Allen AI",
    "Nemotron": "NVIDIA", "Solar": "Upstage",
    "InternLM": "Shanghai AI Lab", "Baichuan": "Baichuan",
}

_BASE_MODEL_TAG_RE = re.compile(r"^base_model:(?:finetune:|merge:|adapter:|quantized:)?(.+)$", re.I)


def parse_base_model(card_data, tags) -> list[str]:
    """Return list of base_model repo_ids from cardData and base_model:* tags."""
    out = []
    if isinstance(card_data, dict):
        bm = card_data.get("base_model")
        if isinstance(bm, str):
            out.append(bm)
        elif isinstance(bm, list):
            out.extend([x for x in bm if isinstance(x, str)])
    for t in (tags or []):
        m = _BASE_MODEL_TAG_RE.match(t or "")
        if m:
            out.append(m.group(1).strip())
    # dedupe preserving order
    seen, uniq = set(), []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def attribute_family(repo_id: str, base_models: list[str]) -> str | None:
    """Best-effort mapping of a repo to a model family (e.g. 'Llama 3')."""
    candidates = [repo_id] + list(base_models or [])
    for cand in candidates:
        s = (cand or "").lower()
        for needle, family in FAMILY_BY_SUBSTRING:
            if needle in s:
                return family
    return None


def attribute_lab(repo_id: str, base_models: list[str], family: str | None) -> str | None:
    """Best-effort mapping to the originating lab (upstream developer)."""
    for cand in (base_models or []):
        ns = (cand or "").split("/")[0].lower()
        if ns in LAB_BY_NAMESPACE:
            return LAB_BY_NAMESPACE[ns]
    if family and family in LAB_BY_FAMILY:
        return LAB_BY_FAMILY[family]
    ns = (repo_id or "").split("/")[0].lower()
    if ns in LAB_BY_NAMESPACE:
        return LAB_BY_NAMESPACE[ns]
    return None


def base_model_relation(card_data) -> str | None:
    if isinstance(card_data, dict):
        rel = card_data.get("base_model_relation")
        if isinstance(rel, str):
            return rel.lower()
    return None
