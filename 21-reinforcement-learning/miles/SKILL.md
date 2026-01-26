---
name: miles-rl-training
description: Provides guidance for enterprise-grade RL training using miles, a production-ready fork of slime. Use when training large MoE models with FP8/INT4, needing train-inference alignment, or requiring speculative RL for maximum throughput.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Reinforcement Learning, MoE, FP8, INT4, Enterprise, SGLang, Megatron-LM]
dependencies: [sglang-router>=0.2.3, ray, torch>=2.0.0, transformers>=4.40.0]
---

# miles: Enterprise-Grade RL for Large-Scale Model Training

miles is a high-performance, enterprise-ready RL framework optimized for large-scale model post-training. Built as a production fork of slime, it addresses critical challenges in MoE training stability, low-precision training, and train-inference alignment.

## When to Use miles

**Choose miles when you need:**
- Training 1TB+ MoE models (DeepSeek V3, Qwen3-MoE)
- FP8 or INT4 quantization-aware training
- Bit-wise identical train-inference alignment
- Speculative RL for maximum throughput
- Production stability with enterprise support

**Consider alternatives when:**
- You want the research-grade original → use **slime**
- You need flexible backend swapping → use **verl**
- You want PyTorch-native abstractions → use **torchforge**

## Key Features

### Low-Precision Training
- **Unified FP8**: End-to-end FP8 for both inference and training
- **INT4 QAT**: 1TB models on single-machine VRAM (H200)
- **Rollout Routing Replay (R3)**: Bit-wise expert alignment for MoE

### Performance Optimizations
- **Speculative RL**: 25%+ rollout speedup with online SFT draft models
- **Zero-Copy Weight Sync**: CUDA IPC zero-copy mapping
- **Partial Rollout**: Recycle half-finished trajectories

### Train-Inference Alignment
- **TIS/MIS**: Truncated/Masked Importance Sampling for off-policy correction
- **Kernel-level optimization**: FlashAttention-3, DeepGEMM integration

## Installation

```bash
# Recommended: Docker
docker pull radixark/miles:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -it radixark/miles:latest /bin/bash

# From source
git clone https://github.com/radixark/miles.git
cd miles
pip install -r requirements.txt
pip install -e .
```

## Quick Start: FP8 GRPO Training

```bash
python train.py \
    --advantage-estimator grpo \
    --model-name qwen3-30b-a3b \
    --hf-checkpoint /path/to/qwen3-30b-a3b-hf \
    --rollout-batch-size 512 \
    --n-samples-per-prompt 8 \
    --fp8-format e4m3 \
    --fp8-recipe blockwise
```

---

## Workflow 1: FP8 MoE Training

Use this workflow for training large MoE models with unified FP8 precision.

### Prerequisites Checklist
- [ ] H100/H200 GPUs with FP8 support
- [ ] MoE model (DeepSeek V3, Qwen3-MoE)
- [ ] Docker environment with miles

### Step 1: Enable FP8 Environment

```bash
# Required environment variables
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Step 2: Configure FP8 Training

```bash
python train.py \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --hf-checkpoint /path/to/deepseek-v3 \
    --advantage-estimator grpo \
    --fp8-format e4m3 \
    --fp8-recipe blockwise \
    --use-r3 \
    --prompt-data /path/to/data.jsonl \
    --num-rollout 3000
```

### Step 3: Enable R3 (Rollout Routing Replay)

R3 records expert routing during inference and replays during training:

```bash
--use-r3                    # Enable routing replay
--r3-buffer-size 1000       # Number of routing decisions to cache
```

### Verification Checklist
- [ ] FP8 tensors visible in profiler
- [ ] Routing decisions match between train/inference
- [ ] No NaN/Inf in loss values

---

## Workflow 2: INT4 Quantization-Aware Training

Use this workflow for fitting 1TB+ models on single-machine VRAM.

### Prerequisites
- [ ] H200 or equivalent (80GB+ VRAM per GPU)
- [ ] Model with INT4 support

### Step 1: Enable INT4 QAT

```bash
# Environment variables for INT4
export OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1
export OPEN_TRAINING_INT4_GROUP_SIZE=128
```

### Step 2: Configure INT4 Training

```bash
python train.py \
    --actor-num-gpus-per-node 8 \
    --hf-checkpoint /path/to/large-model \
    --use-int4-qat \
    --int4-group-size 128 \
    --advantage-estimator grpo \
    --prompt-data /path/to/data.jsonl
```

### Memory Savings

| Model Size | BF16 VRAM | INT4 VRAM | Reduction |
|------------|-----------|-----------|-----------|
| 70B | 140GB | 45GB | 3.1x |
| 235B | 470GB | 150GB | 3.1x |
| 671B | 1.3TB | 420GB | 3.1x |

---

## Workflow 3: Speculative RL Training

Use this workflow for maximum rollout throughput with draft models.

### How Speculative RL Works

1. Small draft model generates candidate tokens
2. Target model verifies in parallel
3. Draft model updated via online SFT to track policy

### Step 1: Prepare Draft Model

```bash
# Convert draft model
python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /path/to/small-model \
    --save /path/to/draft-checkpoint
```

### Step 2: Configure Speculative Training

```bash
python train.py \
    --actor-num-gpus-per-node 8 \
    --hf-checkpoint /path/to/target-model \
    --use-speculative-rl \
    --draft-model-path /path/to/draft-model \
    --spec-lookahead 5 \
    --online-sft-interval 10 \
    --advantage-estimator grpo \
    --prompt-data /path/to/data.jsonl
```

### Expected Speedup

- **Standard rollout**: Baseline
- **Speculative RL**: 25-40% faster rollout
- **With partial rollout**: Additional 10-15% throughput

---

## Configuration Reference

### Cluster Resources

```bash
# Training nodes
--actor-num-nodes 1
--actor-num-gpus-per-node 8

# Inference nodes
--rollout-num-gpus 8
--rollout-num-gpus-per-engine 2  # SGLang tensor parallelism

# Co-location (share GPUs)
--colocate
```

### Megatron Parallelism

```bash
# Model parallelism
--tensor-model-parallel-size 8
--pipeline-model-parallel-size 2
--context-parallel-size 2
--expert-model-parallel-size 4  # MoE expert parallelism

# Data packing
--use-dynamic-batch-size
--max-tokens-per-gpu 4608
```

### Low-Precision Options

```bash
# FP8
--fp8-format e4m3           # or: e5m2
--fp8-recipe blockwise      # or: tensorwise

# INT4
--use-int4-qat
--int4-group-size 128
```

### Train-Inference Alignment

```bash
# R3 (Routing Replay)
--use-r3
--r3-buffer-size 1000

# Importance Sampling Corrections
--use-tis                   # Truncated Importance Sampling
--tis-threshold 0.9
--use-mis                   # Masked Importance Sampling
--mis-threshold 0.1
```

---

## Sample Data Structure

miles uses a structured `Sample` type:

```python
@dataclass
class Sample:
    prompt: str | list[dict]      # Input prompt
    tokens: list[int]             # Token IDs
    response: str                 # Generated response
    reward: float | dict          # RL reward signal
    loss_mask: list[int]          # 1=compute loss, 0=mask
    status: Status                # PENDING|COMPLETED|TRUNCATED|ABORTED
    metadata: dict                # Custom data
    rollout_log_probs: list[float]  # SGLang log probabilities
    rollout_routed_experts: list    # Expert routing (MoE)
```

---

## Common Issues and Solutions

### Issue: FP8 Training Collapse

**Symptoms**: Loss explodes, NaN values

**Solutions**:
```bash
# Use block scaling
--fp8-recipe blockwise
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1

# Enable R3 for MoE
--use-r3

# Reduce learning rate
--lr 5e-7
```

### Issue: Train-Inference Mismatch

**Symptoms**: Policy divergence, reward collapse

**Solutions**:
```bash
# Enable routing replay
--use-r3

# Use importance sampling correction
--use-tis --tis-threshold 0.9

# Verify log probs match
--verify-logprobs
```

### Issue: INT4 Accuracy Degradation

**Symptoms**: Worse performance than BF16

**Solutions**:
```bash
# Increase group size
--int4-group-size 256

# Use mixed precision for sensitive layers
--int4-skip-layers "lm_head,embed"

# Warm start from BF16
--warmup-steps 100
```

### Issue: Speculative Draft Drift

**Symptoms**: Low acceptance rate over time

**Solutions**:
```bash
# More frequent draft updates
--online-sft-interval 5

# Increase draft model learning rate
--draft-lr 1e-5

# Reduce lookahead
--spec-lookahead 3
```

---

## Supported Models

| Family | Models | Notes |
|--------|--------|-------|
| DeepSeek | R1, V3, V3.2 | Full MoE support |
| Qwen | 2, 2.5, 3 (including MoE) | Official support |
| Llama | 3, 3.1, 3.3, 4 | Dense models |
| Gemma | 2, 3, 3N | Dense models |
| GLM | 4.5, 4.6, 4.7 | From slime heritage |
| MiniMax | M2, M2.1 | MoE support |

---

## Advanced Topics

### Multi-Agent Co-Evolution

miles supports MrlX for multi-agent RL:

```bash
python train.py \
    --use-multi-agent \
    --agent-config agents.yaml \
    --co-evolution-interval 10
```

See [MrlX documentation](https://github.com/AQ-MedAI/MrlX) for details.

### Zero-Copy Weight Sync

Optimize weight transfer with CUDA IPC:

```bash
--use-zero-copy-sync
--sync-bucket-size 1073741824  # 1GB buckets
```

### Custom Hooks

```bash
# Custom initialization
--custom-megatron-init-path init_hooks.py

# Custom training step
--custom-megatron-before-train-step-hook-path step_hooks.py

# Custom generation
--custom-generate-function-path generate.py
```

---

## Resources

- **GitHub**: https://github.com/radixark/miles
- **Unified FP8 Blog**: https://lmsys.org/blog/2025-11-25-fp8-rl/
- **R3 Paper**: https://arxiv.org/pdf/2510.11370
- **Slime (upstream)**: https://github.com/THUDM/slime
- **SGLang**: https://github.com/sgl-project/sglang
