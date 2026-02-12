<div align="center">

# SlotGCG: Exploiting the Positional Vulnerability in LLMs for Jailbreak Attacks

[**Paper (OpenReview)**](https://openreview.net/forum?id=Fn2rSOnpNf)

</div>

<p align="center">
  <img src="assets/overview.png" alt="SlotGCG Overview" width="800"/>
</p>

## Overview

**SlotGCG** is an optimization-based red-teaming method that extends **Greedy Coordinate Gradient (GCG)** by explicitly searching for *vulnerable insertion slots* within a prompt (instead of restricting adversarial tokens to a fixed suffix position).

At a high level, SlotGCG:
1. **Probes positional vulnerability** using a *Vulnerable Slot Score (VSS)* over candidate insertion positions (*slots*).
2. **Selects high-VSS slots** and runs token-level discrete optimization (GCG-style) **at those positions**.
3. Improves attack success rate and often converges faster compared to suffix-only optimization.

This repository is built on top of the **HarmBench** evaluation pipeline and provides several SlotGCG variants, including position-aware initialization and attention-guided heuristics.

> **Responsible use:** This code is intended for *research and safety evaluation* of LLMs. Only run attacks on models and systems you own or are authorized to test, and follow applicable laws/policies.

## What’s in this repo

Implemented method families (names may vary by config):

- **GCG_Posinit_Attention**: position-initialized GCG with attention-guided optimization
- **AttGCG**: attention-guided GCG
- **I_GCG**: iterative GCG with multiple initialization strategies
- **GCG_HIJ**: hierarchical injection strategies
- **Ensemble variants**: multi-model / distributed runs via **Ray**

## Installation

### Using Conda (recommended)

```bash
# Clone the repository
git clone https://github.com/SJSoJSooJ/SlotGCG.git
cd SlotGCG

# Create a conda environment (Python 3.11)
conda create -n SlotGCG python=3.11 -y
conda activate SlotGCG

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (if required by the pipeline)
python -m spacy download en_core_web_sm
```

> **Windows note:** If you see line-ending warnings like `LF will be replaced by CRLF`, consider forcing LF for `.sh` files (e.g., via `.gitattributes`) to avoid execution issues on Linux/WSL.

## Running Experiments

Always activate your environment first:

```bash
conda activate SlotGCG
```

### Run Complete Pipeline

This runs the full HarmBench-style pipeline end-to-end (generate test cases → generate completions → evaluate):

```bash
python scripts/run_pipeline.py \
  --methods gcg_posinit_attention \
  --models llama2_7b \
  --step all \
  --mode local
```

**Arguments (high level):**
- `--methods`: which SlotGCG/GCG variant to run
- `--models`: target model key(s)
- `--step all`: run the full pipeline
- `--mode`: execution mode (`local`, `local_parallel` with Ray, or `slurm`)

## Configuration

### Add / edit models

Add new model entries in:

- `configs/model_configs/models.yaml`

Most variants should work with new models via configuration (no code changes).

### Add new methods

Attack methods live under:

- `baselines/`

To add a new variant, create a new module and implement the `RedTeamingMethod` interface defined in:

- `baselines/baseline.py`

## Acknowledgements

This repository builds upon and integrates ideas/tools from:

- **HarmBench** – standardized evaluation pipeline for automated red teaming
- **llm-attacks** – original GCG implementation
- **FastChat** – LLM serving utilities
- **Ray** – distributed computation
- **vLLM** – fast inference engine
- **Transformers** – Hugging Face transformers

We thank the authors and maintainers of these projects.

## Citation

If you use SlotGCG in your research, please cite the paper:

```bibtex
@inproceedings{jeong2026slotgcg,
  title     = {SlotGCG: Exploiting the Positional Vulnerability in LLMs for Jailbreak Attacks},
  author    = {Jeong, Seungwon and Jeong, Jiwoo and Kim, Hyeonjin and Lee, Yunseok and Lee, Woojin},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Fn2rSOnpNf}
}
```
