# Primus

**Primus** is a flexible and high-performance training framework designed for large-scale foundation model training and inference. It is designed to support **pretraining**, **posttraining**, and **reinforcement learning** workflows, and is compatible with multiple backends including [Megatron](https://github.com/NVIDIA/Megatron-LM) and ROCm-optimized components.

---

## Table of Contents

- [What's New](#-whats-new)
- [Primus Product Matrix](#-primus-product-matrix)
- [Quick Start Guide](#-quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Dataset Preparation (FineWeb)](#dataset-preparation-fineweb)
  - [Model Training on 8 x MI300X](#model-training-on-8-x-mi300x)
- [Advanced Topics](#advanced-topics)
- [TODOs](#-todos)

---

## 🆕 What's New
- **[2025/06/18]** Added TorchTitan backend support.
- **[2025/05/16]** Added benchmark suite for performance evaluation across models and hardware.
- **[2025/04/18]** Added [Preflight](./tools/preflight/README.md) cluster sanity checker to verify environment readiness.
- **[2025/04/14]** Integrated HipblasLT autotuning for optimized GPU kernel performance.
- **[2025/04/09]** Extended support for LLaMA2, LLaMA3, DeepSeek-V2/V3 models in [Megatron model configs](https://github.com/AMD-AIG-AIMA/Primus/tree/main/primus/configs/models/megatron).
- **[2025/03/04]** Released Megatron trainer module for flexible and efficient large model training.

---

## 🧩 Primus Product Matrix

|    Module    | Role | Key Features | Dependencies / Integration |
|--------------|------|--------------|-----------------------------|
| [**Primus-LM**](https://github.com/AMD-AGI/Primus)         | End-to-end training framework | - Supports multiple training backends (Megatron, TorchTitan, etc.)<br>- Provides high-performance, scalable distributed training<br>- Deeply integrates with Turbo and Safe | - Can invoke Primus-Turbo kernels and modules<br>- Runs on top of Primus-Safe for stable scheduling |
| [**Primus-Turbo**](https://github.com/AMD-AGI/Primus-Turbo)         | High-performance operators & modules | - Provides common LLM training operators (FlashAttention, GEMM, Collectives, GroupedGemm, etc.)<br>- Modular design, directly pluggable into Primus-LM<br>- Optimized for different architectures and precisions | - Built on [**AITER**](https://github.com/ROCm/aiter), [**CK**](https://github.com/ROCm/composable_kernel), [**hipBLASLt**](https://github.com/ROCm/hipBLASLt), [**Triton**](https://github.com/ROCm/triton)  and other operator libraries<br>- Can be enabled via configuration inside Primus-LM |
| **Primus-SaFE** (Coming soon)         | Stability & platform layer | - Cluster sanity check and benchmarking<br>- Kubernets scheduling with topology awareness<br>- Fault tolerance<br>- Stability enhancements | - Building a training platform based on the K8s and Slurm ecosystem |

---

## 🚀 Quick Start Guide

This guide will walk you through the complete setup process for training a Llama model on **8 x AMD MI300X GPUs** using the **FineWeb dataset**.

### Prerequisites

- **Hardware**: 8 x AMD MI300X GPUs (or similar AMD Instinct GPUs)
- **Software**:
  - AMD ROCm drivers (version ≥ 6.0 recommended)
  - Docker (version ≥ 24.0) with ROCm support
  - Proper permissions for Docker and GPU device access
- **Storage**: Sufficient disk space for datasets (~500GB for FineWeb-10B, ~2TB for FineWeb-Edu)
- **Network**: Internet access for downloading datasets and Docker images
- **Access Tokens**: HuggingFace token (for model tokenizers and gated models)

### Environment Setup

#### Step 1: Pull the AMD ROCm Docker Image

```bash
# Pull the official Primus Docker image optimized for MI300X (gfx942)
docker pull docker.io/rocm/primus:v25.9_gfx942
```

#### Step 2: Clone the Primus Repository

```bash
# Clone the repository with all submodules
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus

# If already cloned, initialize submodules
git submodule update --init --recursive
```

#### Step 3: Start the Docker Container

```bash
# Launch the container with GPU access and mounted volumes
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=16g \
  -v $(pwd):/workspace/Primus \
  -v /data:/data \
  --name primus_train \
  docker.io/rocm/primus:v25.9_gfx942 \
  /bin/bash
```

Or use the provided script:

```bash
bash tools/docker/start_container.sh
docker exec -it dev_primus bash
```

#### Step 4: Install Python Dependencies

```bash
cd /workspace/Primus
pip install -r requirements.txt

# Set up pre-commit hooks (optional, for development)
pre-commit install
```

#### Step 5: Set Environment Variables

```bash
# Set your HuggingFace token (required for downloading tokenizers and models)
export HF_TOKEN="your_huggingface_token_here"

# Set data directory
export DATA_PATH="/data/primus_data"
mkdir -p $DATA_PATH
```

---

### Dataset Preparation (FineWeb)

FineWeb is a high-quality pretraining dataset from HuggingFace. We'll use the **FineWeb-Edu** subset (educational content) or **FineWeb-10B** (10 billion tokens) for this guide.

#### Step 1: Download FineWeb Dataset

```bash
# Install required packages
pip install datasets huggingface_hub

# Create a Python script to download FineWeb
cat > /workspace/Primus/download_fineweb.py << 'EOF'
#!/usr/bin/env python3
"""Download FineWeb dataset from HuggingFace."""

import argparse
from pathlib import Path
from datasets import load_dataset

def download_fineweb(dataset_name: str, split: str, output_dir: Path, num_samples: int = None):
    """
    Download FineWeb dataset and save as JSONL.

    Args:
        dataset_name: e.g., 'HuggingFaceFW/fineweb' or 'HuggingFaceFW/fineweb-edu'
        split: 'train' or specific split like 'train[:1%]'
        output_dir: Directory to save the dataset
        num_samples: Optional limit on number of samples
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Downloading {dataset_name} (split: {split})...")
    dataset = load_dataset(
        dataset_name,
        name="default",
        split=split,
        trust_remote_code=True,
        streaming=False  # Set to True for very large datasets
    )

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        print(f"[Info] Limited to {len(dataset)} samples")

    output_file = output_dir / "fineweb_megatron.json"
    print(f"[Info] Saving dataset to {output_file}...")
    dataset.to_json(output_file, lines=True)

    print(f"[Info] Dataset saved successfully. Total samples: {len(dataset)}")
    print(f"[Info] File size: {output_file.stat().st_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        choices=["HuggingFaceFW/fineweb", "HuggingFaceFW/fineweb-edu"],
        help="FineWeb dataset variant"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train[:1%]",
        help="Dataset split (e.g., 'train', 'train[:10%]', 'train[:100000]')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/primus_data/fineweb",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download"
    )

    args = parser.parse_args()
    download_fineweb(
        args.dataset,
        args.split,
        Path(args.output_dir),
        args.num_samples
    )
EOF

chmod +x /workspace/Primus/download_fineweb.py
```

#### Step 2: Download the Dataset

Choose one of the following options:

**Option A: FineWeb-Edu (1% for testing - ~5GB, ~1M samples)**
```bash
python3 /workspace/Primus/download_fineweb.py \
  --dataset "HuggingFaceFW/fineweb-edu" \
  --split "train[:1%]" \
  --output-dir "$DATA_PATH/fineweb"
```

**Option B: FineWeb-Edu (10% - ~50GB, ~10M samples)**
```bash
python3 /workspace/Primus/download_fineweb.py \
  --dataset "HuggingFaceFW/fineweb-edu" \
  --split "train[:10%]" \
  --output-dir "$DATA_PATH/fineweb"
```

**Option C: Full FineWeb (sample 10B tokens - ~100GB)**
```bash
python3 /workspace/Primus/download_fineweb.py \
  --dataset "HuggingFaceFW/fineweb" \
  --split "train[:1000000]" \
  --output-dir "$DATA_PATH/fineweb"
```

#### Step 3: Preprocess Dataset for Megatron

Download the tokenizer model (Llama tokenizer):

```bash
# Download Llama 3.1 tokenizer
mkdir -p $DATA_PATH/tokenizers/llama3.1
huggingface-cli download meta-llama/Meta-Llama-3.1-8B \
  --include "tokenizer.model" "tokenizer.json" "tokenizer_config.json" \
  --local-dir $DATA_PATH/tokenizers/llama3.1
```

Preprocess the dataset into Megatron binary format:

```bash
cd /workspace/Primus

# Set paths
INPUT_JSON="$DATA_PATH/fineweb/fineweb_megatron.json"
OUTPUT_PREFIX="$DATA_PATH/fineweb/llama3.1/fineweb"
TOKENIZER_MODEL="$DATA_PATH/tokenizers/llama3.1/tokenizer.model"

# Run preprocessing
python3 examples/megatron/preprocess_data.py \
  --input "$INPUT_JSON" \
  --output-prefix "$OUTPUT_PREFIX" \
  --tokenizer-type SentencePieceTokenizer \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --append-eod \
  --workers $(nproc) \
  --partitions 8

# This will create:
# - $OUTPUT_PREFIX_text_document.bin
# - $OUTPUT_PREFIX_text_document.idx
```

**Expected output files:**
```
$DATA_PATH/fineweb/llama3.1/fineweb_text_document.bin
$DATA_PATH/fineweb/llama3.1/fineweb_text_document.idx
```

---

### Model Training on 8 x MI300X

#### Step 1: Create Training Configuration

Create a custom config file for Llama training with FineWeb:

```bash
mkdir -p examples/megatron/configs/MI300X

cat > examples/megatron/configs/MI300X/llama3.1_8B-fineweb-pretrain.yaml << 'EOF'
work_group: ${PRIMUS_TEAM:amd}
user_name: ${PRIMUS_USER:root}
exp_name: ${PRIMUS_EXP_NAME:llama3.1_8B-fineweb-pretrain}
workspace: ./output

modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml

    # model to run
    model: llama3.1_8B.yaml
    overrides:
      # Logging
      wandb_project: "Primus_Llama_Pretrain_FineWeb"
      disable_wandb: false
      disable_tensorboard: false
      stderr_sink_level: INFO
      log_avg_skip_iterations: 10
      log_avg_reset_interval: 100

      # Training iterations
      train_iters: 10000
      micro_batch_size: 4
      global_batch_size: 512

      # Sequence length
      seq_length: 8192
      max_position_embeddings: 8192

      # Learning rate
      lr: 3.0e-4
      min_lr: 3.0e-5
      lr_warmup_iters: 2000
      lr_decay_iters: 10000
      lr_decay_style: cosine
      weight_decay: 0.1
      adam_beta1: 0.9
      adam_beta2: 0.95
      clip_grad: 1.0

      # Data settings
      eod_mask_loss: true
      init_method_std: 0.01
      norm_epsilon: 1.0e-5

      # Parallelism for 8 GPUs
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 2
      expert_model_parallel_size: 1

      # Overlap and optimization
      overlap_grad_reduce: true
      overlap_param_gather: true
      gradient_accumulation_fusion: true
      use_distributed_optimizer: true

      # Dataset paths (will be set via environment or override)
      mock_data: false
      train_data_path: /data/primus_data/fineweb/llama3.1/fineweb_text_document
      valid_data_path: null
      test_data_path: null
      data_cache_path: /data/primus_data/cache

      # Checkpoint settings
      finetune: false
      auto_continue_train: true
      load: null
      no_load_optim: false
      no_load_rng: false
      save: ./output/checkpoints/llama3.1_8B_fineweb
      save_interval: 1000
      no_save_optim: false
      no_save_rng: false
      disable_last_saving: false
      ckpt_format: torch

      # Primus Turbo optimizations
      enable_primus_turbo: true
      use_turbo_attention: true
      use_turbo_grouped_mlp: true

      # Cross entropy optimization
      cross_entropy_fusion_impl: "te"
      cross_entropy_loss_fusion: true

      # Mixed precision
      bf16: true
      fp16: false

      # Evaluation
      eval_interval: 500
      eval_iters: 10
EOF
```

#### Step 2: Run Training on Single Node (8 x MI300X)

```bash
cd /workspace/Primus

# Set environment variables
export HF_TOKEN="your_huggingface_token_here"
export DATA_PATH="/data/primus_data"
export PRIMUS_EXP_NAME="llama3.1_8B_fineweb_pretrain"

# Run training
EXP=examples/megatron/configs/MI300X/llama3.1_8B-fineweb-pretrain.yaml \
  bash ./examples/run_local_pretrain.sh
```

**Alternative: Interactive mode for debugging**

```bash
# Enter the container
docker exec -it primus_train bash
cd /workspace/Primus

# Run training with direct command
EXP=examples/megatron/configs/MI300X/llama3.1_8B-fineweb-pretrain.yaml \
  bash ./examples/run_pretrain.sh
```

#### Step 3: Monitor Training

**View real-time logs:**
```bash
tail -f output/llama3.1_8B-fineweb-pretrain/logs/rank_0.log
```

**Monitor GPU utilization:**
```bash
watch -n 1 rocm-smi
```

**Check training metrics:**
- TensorBoard (if enabled): `tensorboard --logdir output/llama3.1_8B-fineweb-pretrain/tensorboard`
- Weights & Biases (if enabled): Check your W&B dashboard

**Expected training output:**
```
iteration      100/  10000 | elapsed time per iteration (ms): 1234.5 | learning rate: 1.500E-05 | global batch size:   512 |
lm loss: 3.456E+00 | loss scale: 1.0 | grad norm: 2.345 | num zeros: 0.0 | params norm: 1.234E+02 |
throughput: 123K tokens/s | tokens: 51.2M
```

#### Step 4: Multi-Node Training (Optional)

For training across multiple nodes with SLURM:

```bash
# Set the number of nodes (e.g., 4 nodes = 32 GPUs)
export DOCKER_IMAGE="docker.io/rocm/primus:v25.9_gfx942"
export NNODES=4

# Update parallelism in config for 32 GPUs
# tensor_model_parallel_size: 4
# pipeline_model_parallel_size: 4

# Run with SLURM
EXP=examples/megatron/configs/MI300X/llama3.1_8B-fineweb-pretrain.yaml \
  bash ./examples/run_slurm_pretrain.sh
```

---

## Advanced Topics

### HipBLASLt Auto-Tuning for Maximum Performance

HipBLASLt tuning can significantly improve training throughput (10-20% speedup):

**Stage 1: Dump GEMM shapes**
```bash
export PRIMUS_HIPBLASLT_TUNING_STAGE=1
EXP=examples/megatron/configs/MI300X/llama3.1_8B-fineweb-pretrain.yaml \
  bash ./examples/run_local_pretrain.sh
```

**Stage 2: Tune kernels**
```bash
export PRIMUS_HIPBLASLT_TUNING_STAGE=2
EXP=examples/megatron/configs/MI300X/llama3.1_8B-fineweb-pretrain.yaml \
  bash ./examples/run_local_pretrain.sh
```

**Stage 3: Train with tuned kernels**
```bash
export PRIMUS_HIPBLASLT_TUNING_STAGE=3
EXP=examples/megatron/configs/MI300X/llama3.1_8B-fineweb-pretrain.yaml \
  bash ./examples/run_local_pretrain.sh
```

For more details, see [examples/README.md](./examples/README.md).

### Converting to HuggingFace Format

After training, convert checkpoints to HuggingFace format:

```bash
# TODO: Add conversion script reference
# See Megatron-LM documentation for checkpoint conversion
```

### Supported Models

The following models are pre-configured and tested on MI300X:

| Model | Parameters | Config File |
|-------|-----------|-------------|
| Llama 2 7B | 7B | `llama2_7B-pretrain.yaml` |
| Llama 2 70B | 70B | `llama2_70B-pretrain.yaml` |
| Llama 3 8B | 8B | `llama3_8B-pretrain.yaml` |
| Llama 3 70B | 70B | `llama3_70B-pretrain.yaml` |
| Llama 3.1 8B | 8B | `llama3.1_8B-pretrain.yaml` |
| Llama 3.1 70B | 70B | `llama3.1_70B-pretrain.yaml` |
| Llama 3.1 405B | 405B | `llama3.1_405B-pretrain.yaml` |
| DeepSeek-V2 Lite | 16B | `deepseek_v2_lite-pretrain.yaml` |
| DeepSeek-V2 | 236B | `deepseek_v2-pretrain.yaml` |
| DeepSeek-V3 | 671B | `deepseek_v3-pretrain.yaml` |
| Mixtral 8x7B | 47B | `mixtral_8x7B_v0.1-pretrain.yaml` |
| Mixtral 8x22B | 141B | `mixtral_8x22B_v0.1-pretrain.yaml` |

All configs are in [`examples/megatron/configs/MI300X/`](./examples/megatron/configs/MI300X/).

### Troubleshooting

**Out of Memory (OOM) errors:**
- Reduce `micro_batch_size` in the config
- Increase `pipeline_model_parallel_size` or `tensor_model_parallel_size`
- Reduce `seq_length`
- Enable activation checkpointing: `recompute_granularity: selective`

**Slow training:**
- Run HipBLASLt auto-tuning (see above)
- Enable all Primus Turbo optimizations: `enable_primus_turbo: true`
- Check GPU utilization with `rocm-smi`

**Dataset not loading:**
- Verify `.bin` and `.idx` files exist
- Check `train_data_path` in config points to prefix (without `.bin` or `.idx`)
- Ensure preprocessing completed successfully

**NCCL/RCCL communication errors:**
- Check network connectivity between nodes
- Set proper network interface: `export NCCL_SOCKET_IFNAME=eth0`
- Verify IB/RoCE configuration if using InfiniBand

For more help, see:
- [examples/README.md](./examples/README.md) - Detailed training guide
- [GitHub Issues](https://github.com/AMD-AIG-AIMA/Primus/issues)

---

## 📝 TODOs

- [ ] Support for Primus-RL (training/inference modules for RLHF, OnlineDPO, GRPO, etc.)
- [ ] Add support for more model architectures and backends
- [ ] Integrated checkpoint conversion utilities
- [ ] Enhanced multi-modal pretraining support

---

## Citation

If you use Primus in your research, please cite:

```bibtex
@software{primus2025,
  title = {Primus: High-Performance Training Framework for Foundation Models},
  author = {AMD AIG Team},
  year = {2025},
  url = {https://github.com/AMD-AIG-AIMA/Primus}
}
```

## License

See [LICENSE](./LICENSE) for license information.

## Contributing

Contributions are welcome! Please see our contributing guidelines and submit pull requests to the main repository.
