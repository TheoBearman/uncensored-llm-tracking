# Uncensored AI in the Wild: Tracking Publicly Available and Locally Deployable LLMs

This repository contains the code and data for the research paper "Uncensored AI in the Wild: Tracking Publicly Available and Locally Deployable LLMs" which presents the first large-scale empirical analysis of safety-modified open-weight language models.

## Overview

This study analyzes model retrieved using search terms from Hugging Face to identify models explicitly adapted to bypass alignment safeguards. The research demonstrates systematic patterns in how these "uncensored" models are created, distributed, and optimized for local deployment.

## Repository Structure

### Data Collection Scripts
- `scrape_model_names.py` - Main scraper for (incrementally) retrieving model names that hit safety/uncensorship keywords from Hugging Face
- `scrape_metadata.py` - Script to retrieve JSON metadata for model repositories identified using the search script.
- `get_repo_download_counts.py` - Script for obtaining total downloads for all time up to cutoff date for a list of model repos (model IDs)

### Analysis Scripts and Notebooks  
- `process_scrape_results.ipynb` - Filter and process raw scrape data and generates normalized datasets with family attribution
- `hf_model_benchmarker_.py` - Evaluates selected models using Hugging Face API
- `hf_model_benchmarker_gguf.py` - Evaluates GGUF-format models using llama.cpp
- `generate_figures.ipynb` - Generates the paper figures and tables from the output of `process_scrape_results.ipynb` and benchmarking

### Generated Data Files

#### Model Scraping
- `safety_terms.txt` - Search terms used to retrieve models from Hugging Face
- `model_list.txt` - List of model repository names retreived from Hugging Face
- `repo_catalog.tsv` - Complete catalog of scraped repositories with metadata after processing with `model_trends_analysis.py`
- `evaluated_models_metadata.csv` - Metadata for the subset of models evaluated for safety

#### Model Evaluation
- `model_evaluation_results.csv` - Safety evaluation results (annotations for each prompt) for tested models.
- `model_evaluation_results_full.csv.zip` - Safety evaluation results using all LLM-based evaluator models, including the responses and annotations by each of the LLM evaluators that were benchmarked. (WARNING: DATA MAY CONTAIN MATERIAL THAT IS OFFENSIVE AS A RESULT OF DEPICTIONS OF DISCRIMINATION, VIOLENCE, ENCOURAGEMENT OF SELF-HARM, DISINFORMATION, AND OTHER POTENTIAL TYPES OF HARM.)
- `model_evaluator_validation.csv` - Validation dataset 957 (in CSV format) for reliability testing of model response evaluators, containing 300 model–prompt- 958 response data points annotated by human and automated LLM-based evaluators. (WARNING: DATA MAY CONTAIN MATERIAL THAT IS OFFENSIVE AS A RESULT OF DEPICTIONS OF DISCRIMINATION, VIOLENCE, ENCOURAGEMENT OF SELF-HARM, DISINFORMATION, AND OTHER POTENTIAL TYPES OF HARM.)
- `evaluated_models_metadata_revised.csv` - Metadata for tested models
- `prompts.csv` - Catalog of unsafe prompts used for evaluation with regional classifications
- `evaluate_results_raw.json` - Raw results (including full responses) from evaluation experiments (WARNING: DATA MAY CONTAIN MATERIAL THAT IS OFFENSIVE AS A RESULT OF DEPICTIONS OF DISCRIMINATION, VIOLENCE, ENCOURAGEMENT OF SELF-HARM, DISINFORMATION, AND OTHER POTENTIAL TYPES OF HARM.)
- `evaluation_prompt_response_labeler.html` - HTML file for lightweight web app to manually label responses based on whether they are safety-based rejections or complying with prompts

## Continuous tracking pipeline

In addition to the one-shot scripts above (used to produce the paper), the repo
runs a weekly snapshot pipeline that produces policy-relevant time-series data.

### Per-snapshot outputs

Each Monday a GitHub Action writes `snapshots/YYYY-MM-DD/` containing:

| File | Contents |
| --- | --- |
| `models.parquet` | One enriched row per repo: quant format & bits, base-model lineage, originating lab, uploader country, languages, parameter count, estimated memory, compute-class bucket, downloads, likes, license, gated/private flags. |
| `summary.json` | Aggregates consumed by the static site (counts by lab/country/family/quant/compute-class, top 30-day downloads). |
| `manifest.json` | Run metadata: search-term count, runtime, total repos discovered. |
| `raw/models.jsonl` | Raw `model_info` payloads for reproducibility. |
| `deltas.json` | Diff vs. previous snapshot — added / removed (takedowns) / download surges / license flips / gated-status flips. |

### Running locally

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...

# Take a snapshot for today (UTC). --limit caps repos for dry runs.
python scripts/snapshot.py --terms safety_terms.txt --limit 200

# Diff two snapshots
python scripts/delta.py snapshots/2026-04-24 snapshots/2026-05-01
```

### Enrichment modules (`scripts/`)

- `quantization.py` — detects GGUF / GPTQ / AWQ / EXL2 / MLX / bitsandbytes / FP8 and effective bit-width from tags, repo names, and GGUF filenames (e.g. `Q4_K_M`).
- `lineage.py` — parses `cardData.base_model` and `base_model:*` tags, then attributes each repo to a base-model family (Llama 3, Qwen 2.5, …) and the originating lab (Meta, Alibaba, Mistral AI, …). Powers upstream-developer attribution for EU AI Act / US EO derivative debates.
- `geo.py` — best-effort uploader country via a curated namespace map; unknowns are reported transparently rather than guessed.
- `compute_class.py` — estimates parameter count from name (`7B`, `8x7B`, `1.5B`) and maps `(params, bits) → memory_gb → bucket` (`phone` / `laptop-cpu` / `consumer-gpu-12gb` / `consumer-gpu-24gb` / `workstation` / `single-node-server` / `datacenter`).
- `enrich.py` — composes the above into a single flat row schema.

### Automation

- `.github/workflows/weekly-snapshot.yml` — runs `snapshot.py` + `delta.py` every Monday 06:00 UTC, commits artifacts. Requires `HF_TOKEN` repository secret.
- `.github/workflows/publish-site.yml` — builds and deploys `site/` to GitHub Pages whenever a snapshot lands. The site reads `snapshots/index.json` and per-day `summary.json` / `deltas.json` (parquet files are intentionally excluded from the deployed payload).

### Static site (`site/`)

A no-build vanilla-JS dashboard for browsing the time series: total / GGUF growth, distribution by lab / country / compute-class / quant format, top 30-day movers, takedowns, and license flips. Hosted on GitHub Pages.

## Ethics and Safety

This research examines publicly available models to understand AI safety challenges. WARNING: The uncensored model evaluation results include sensitive responses that may relate to harmful issuee. All responses were generated by open-weight (or open-source), publicly available LLMs without human intervention. Furthermore, the raw results of prompt testing may be offensive due to their coverage of topics such as discrimination, violence, and other types of harm.

## Citation

```bibtex
[PLACEHOLDER - Citation to be added upon publication]
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the research or data access requests, please contact the author.
