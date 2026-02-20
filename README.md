<div align="center">

# SlotGCG: Exploiting the Positional Vulnerability in LLMs for Jailbreak Attacks

[**Paper (OpenReview)**](https://openreview.net/forum?id=Fn2rSOnpNf)

</div>

<p align="center">
  <img src="assets/overview.png" alt="SlotGCG Overview" width="800"/>
</p>

## :compass: Overview

**SlotGCG** is an optimization-based red-teaming method that extends **Greedy Coordinate Gradient (GCG)** by explicitly **searching for *vulnerable insertion slots* within a prompt** (instead of restricting adversarial tokens to a fixed suffix position).

Overview of SlotGCG:
1. **Probes positional vulnerability** using a *Vulnerable Slot Score (VSS)* over candidate insertion positions (*slots*).
2. **Selects high-VSS slots** and runs token-level optimization **at those positions**.
3. Improves attack success rate and often converges faster compared to suffix-only optimization.

This repository is built on top of the **[HarmBench](https://github.com/centerforaisafety/HarmBench)** evaluation pipeline and provides several SlotGCG variants.

## :test_tube: Supported SlotGCG variants

| Method               | Paper | Description                                                                                                                                                                                                          |
| -------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GCG**              | [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)     | Token-swap coordinate descent attack.                                                                                                                                                                                |
| **GCG-Transfer**     |      | **GCG** executed in a **multi-run transfer setup**|
| **AttGCG**           | [AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation](https://arxiv.org/abs/2410.09040)     | GCG with attention-guided updates.                                                                                                                                                                                   |
| **AttGCG-Transfer**  |      | **AttGCG** in a **multi-run transfer setup**          |
| **I-GCG**            | [Improved Techniques for Optimization-Based Jailbreaking on Large Language Models](https://arxiv.org/abs/2405.21018)     | GCG with iterative / multi-start initialization.                                                                                                                                                                     |
| **I-GCG-Transfer**   |      | **I-GCG** in a **multi-run transfer setup**                                   |
| **GCG-Hij**          | [Universal Jailbreak Suffixes Are Strong Attention Hijackers](https://arxiv.org/abs/2506.12880)     | GCG with hijacking-style steering.                                                                                                                                                                                   |
| **GCG-Hij-Transfer** |      | **GCG-Hij** in a **multi-run transfer setup**                              |
| **GBDA**             | [Gradient-based Adversarial Attacks against Text Transformers](https://arxiv.org/abs/2104.13733)     | Continuous relaxation + discretization attack.                                                                                                                                                                       |




## :gear: Installation

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

# Download spaCy model
python -m spacy download en_core_web_sm
```

## :rocket: Running Experiments

### Run Complete Pipeline

This runs the full HarmBench-style pipeline (generate test cases → generate completions → evaluate):

```bash
python scripts/run_pipeline.py \
  --methods GCG_posinit_attention \
  --models llama2_7b \
  --step all \
  --mode local
```

```bash
python scripts/run_pipeline.py \
  --methods GBDA_posinit_attention, GBDA \
  --models vicuna_7b_v1_5, llama2_7b \
  --step all \
  --mode local
```

**Arguments (high level):**
- `--methods`: which method(s) to run
- `--models`: target model key(s)
- `--step all`: run the full pipeline
- `--mode`: execution mode (`local`, `local_parallel` with Ray, or `slurm`)

## :wrench: Configuration

### Add / edit models

Add new model entries in:

- `configs/model_configs/models.yaml`

Most variants should work with new models via configuration (no code changes).

### Add new methods

Attack methods live under:

- `baselines/`

To add a new variant, create a new module and implement the `RedTeamingMethod` interface defined in:

- `baselines/baseline.py`

## :pray: Acknowledgements

We thank the following open-source repositories.

- **[HarmBench](https://github.com/centerforaisafety/HarmBench)** – standardized evaluation pipeline for automated red teaming
- **[llm-attacks](https://github.com/llm-attacks/llm-attacks)** – original GCG implementation
- **[AttnGCG-attack](https://github.com/UCSC-VLAA/AttnGCG-attack)** –  AttnGCG attacks implementation
- **[interp-jailbreak](https://github.com/matanbt/interp-jailbreak)** – GCG-Hij attacks implementation
- **[I-GCG](https://github.com/jiaxiaojunQAQ/I-GCG)** – I-GCG attack implementation
- **[text-adversarial-attack](https://github.com/facebookresearch/text-adversarial-attack)** – GBDA attacks implementation
- **[SafeDecoding](https://github.com/uw-nsl/SafeDecoding)** – SafeDecoding implementation
- **[rpo](https://github.com/lapisrocks/rpo)** – RPO implementation
- **[certified-llm-safety](https://github.com/aounon/certified-llm-safety)** – Erase_And_Check implementation
- **[smooth-llm](https://github.com/arobey1/smooth-llm)** – SmoothLLM implementation


We thank the authors and maintainers of these projects.

## :memo: Citation

If you use SlotGCG in your research, please cite the paper:

```bibtex
@inproceedings{
    jeong2026slotgcg,
    title={Slot{GCG}: Exploiting the Positional Vulnerability in {LLM}s for Jailbreak Attacks},
    author={Seungwon Jeong and Jiwoo Jeong and Hyeonjin Kim and Yunseok Lee and Woojin Lee},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=Fn2rSOnpNf}
}
```
