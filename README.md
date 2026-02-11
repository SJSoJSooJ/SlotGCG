<div align="center">

# SlotGCG

### A Slot-Based Approach to Greedy Coordinate Gradient Attacks on Large Language Models

</div>

<p align="center">
  <img src="assets/overview.png" alt="SlotGCG Overview" width="800"/>
</p>

## Overview

SlotGCG is a novel adversarial attack method for large language models (LLMs) that enhances the Greedy Coordinate Gradient (GCG) approach through slot-based optimization. Built on the HarmBench evaluation framework, SlotGCG introduces several variants of position-aware and attention-guided optimization strategies to improve the effectiveness and efficiency of adversarial prompt generation.

This repository includes implementations of multiple SlotGCG variants:
- **GCG_Posinit_Attention**: Position-initialized GCG with attention-based optimization
- **AttGCG**: Attention-guided GCG
- **I_GCG**: Iterative GCG with various initialization strategies
- **GCG_HIJ**: GCG with hierarchical injection strategies
- **Ensemble variants**: Multi-model ensemble attack methods using Ray for distributed computation

## Quick Start

### Installation

#### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/SlotGCG.git
cd SlotGCG

# Create a new conda environment with Python 3.11
conda create -n SlotGCG python=3.11 -y

# Activate the environment
conda activate SlotGCG

# Install required packages
pip install -r requirements.txt

# Download spacy language model
python -m spacy download en_core_web_sm
```

#### Using pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/yourusername/SlotGCG.git
cd SlotGCG

# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Download spacy language model
python -m spacy download en_core_web_sm
```

### Verify Installation

After installation, verify that the key packages are properly installed:

```bash
# Activate your environment first
conda activate SlotGCG  # or: source venv/bin/activate

# Quick check
python -c "import torch; import transformers; print('✓ Installation successful!')"

# Detailed verification (recommended)
python verify_installation.py
```

The verification script will check all core dependencies and provide detailed information about your environment.

### Running Experiments

The evaluation pipeline consists of three main steps: (1) generating test cases, (2) generating completions, and (3) evaluating completions.

**Important:** Always activate your conda environment before running experiments:
```bash
conda activate SlotGCG
```

#### Step 1: Generate Test Cases

```bash
# Generate test cases for GCG with position-initialized attention
python scripts/run_pipeline.py \
  --methods gcg_posinit_attention \
  --models llama2_7b \
  --step 1 \
  --mode local

# Generate test cases for attention-guided GCG
python scripts/run_pipeline.py \
  --methods attngcg_posinit_attention \
  --models llama2_7b \
  --step 1 \
  --mode local

# Generate test cases for iterative GCG
python scripts/run_pipeline.py \
  --methods i_gcg \
  --models llama2_7b \
  --step 1 \
  --mode local
```

#### Step 2: Generate Completions

```bash
# Generate completions using the test cases from Step 1
python scripts/run_pipeline.py \
  --methods gcg_posinit_attention \
  --models llama2_7b \
  --step 2 \
  --mode local
```

#### Step 3: Evaluate Completions

```bash
# Evaluate the generated completions
python scripts/run_pipeline.py \
  --methods gcg_posinit_attention \
  --models llama2_7b \
  --step 3 \
  --mode local
```

#### Run Complete Pipeline

```bash
# Run all steps at once (Steps 1, 2, and 3)
python scripts/run_pipeline.py \
  --methods gcg_posinit_attention \
  --models llama2_7b \
  --step all \
  --mode local

# Or run Steps 2 and 3 together
python scripts/run_pipeline.py \
  --methods gcg_posinit_attention \
  --models llama2_7b \
  --step 2_and_3 \
  --mode local
```

#### Execution Modes

- `--mode local`: Sequential execution on current machine
- `--mode local_parallel`: Parallel execution using Ray across multiple GPUs
- `--mode slurm`: Distributed execution on SLURM cluster

#### Available Methods

```bash
# Standard GCG variants
--methods gcg                          # Basic GCG
--methods gcg_posinit_attention        # GCG with position-init attention
--methods gcg_posinit_random           # GCG with random position-init

# Attention-based variants
--methods attngcg                      # Attention-guided GCG
--methods attngcg_posinit_attention    # AttGCG with position-init

# Iterative variants
--methods i_gcg                        # Iterative GCG
--methods i_gcg_posinit_attention      # I-GCG with position-init

# Hierarchical injection variants
--methods gcg_hij                      # GCG with hierarchical injection
--methods gcg_hij_posinit_attention    # GCG-HIJ with position-init

# Ensemble variants (requires Ray)
--methods gcg_posinit_attention_ensemble
--methods i_gcg_ensemble
```

#### Example: Multi-GPU Execution

```bash
# Use Ray for parallel execution across multiple GPUs
python scripts/run_pipeline.py \
  --methods gcg_posinit_attention_ensemble \
  --models llama2_7b \
  --step all \
  --mode local_parallel
```

### Using Your Own Models

You can add new models in [configs/model_configs/models.yaml](configs/model_configs/models.yaml). Most SlotGCG variants support dynamic configuration for new models without manual adjustments.

### Adding Custom Methods

All attack methods are implemented in the [baselines](baselines) directory. To add a new variant, create a new subfolder and implement the `RedTeamingMethod` interface defined in [baselines/baseline.py](baselines/baseline.py).

## Acknowledgements

This work builds upon several excellent open-source repositories:

- [HarmBench](https://github.com/centerforaisafety/HarmBench) - Standardized evaluation framework for automated red teaming
- [llm-attacks](https://github.com/llm-attacks/llm-attacks) - Original GCG implementation
- [FastChat](https://github.com/lm-sys/FastChat) - LLM serving framework
- [Ray](https://github.com/ray-project/ray) - Distributed computing framework
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference engine
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face transformers library

We thank the authors of these repositories for their contributions to the open-source community.

## Citation

If you find SlotGCG useful in your research, please consider citing:

```bibtex
@article{slotgcg2026,
  title={SlotGCG: A Slot-Based Approach to Greedy Coordinate Gradient Attacks},
  author={Your Name and Collaborators},
  year={2026},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}
```

We also acknowledge the HarmBench framework:

```bibtex
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and Zifan Wang and Norman Mu and Elham Sakhaee and Nathaniel Li and Steven Basart and Bo Li and David Forsyth and Dan Hendrycks},
  year={2024},
  journal={arXiv preprint arXiv:2402.04249}
}
```
