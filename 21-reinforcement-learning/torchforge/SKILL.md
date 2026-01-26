---
name: torchforge-rl-training
description: Provides guidance for PyTorch-native agentic RL using torchforge, Meta's library separating infra from algorithms. Use when you want clean RL abstractions, easy algorithm experimentation, or scalable training with Monarch and TorchTitan.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Reinforcement Learning, PyTorch, GRPO, SFT, Monarch, TorchTitan, Meta]
dependencies: [torch>=2.9.0, torchtitan>=0.2.0, vllm, monarch]
---

# torchforge: PyTorch-Native Agentic RL Library

torchforge is Meta's PyTorch-native RL library that separates infrastructure concerns from algorithm concerns. It enables rapid RL research by letting you focus on algorithms while handling distributed training, inference, and weight sync automatically.

## When to Use torchforge

**Choose torchforge when you need:**
- Clean separation between RL algorithms and infrastructure
- PyTorch-native abstractions (no Ray dependency)
- Easy algorithm experimentation (GRPO, DAPO, SAPO in ~100 lines)
- Scalable training with Monarch actor system
- Integration with TorchTitan for model parallelism

**Consider alternatives when:**
- You need production-ready stability → use **miles** or **verl**
- You want Megatron-native training → use **slime**
- torchforge is experimental and APIs may change

## Key Features

- **Algorithm isolation**: Implement RL algorithms without touching infrastructure
- **Scalability**: From single GPU to thousands via Monarch
- **Modern stack**: TorchTitan (training), vLLM (inference), TorchStore (sync)
- **Loss functions**: GRPO, DAPO, CISPO, GSPO, SAPO built-in

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ Application Layer (Your Code)                           │
│ - Define reward models, loss functions, sampling        │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│ Forge API Layer                                         │
│ - TextTrainBatch, ForwardBackwardResult                │
│ - Service interfaces (async/await)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│ Distributed Services (Monarch)                          │
│ ├── Trainer (TorchTitan FSDP)                          │
│ ├── Generator (vLLM inference)                          │
│ ├── Reference Model (frozen KL baseline)               │
│ └── Reward Actors (compute rewards)                    │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Create environment
conda create -n forge python=3.12
conda activate forge

# Install (handles PyTorch nightly + dependencies)
./scripts/install.sh

# Verify
python -c "import torch, forge, vllm; print('OK')"
```

### ROCm Installation

```bash
./scripts/install_rocm.sh
```

## Quick Start

### SFT Training (2+ GPUs)

```bash
python -m apps.sft.main --config apps/sft/llama3_8b.yaml
```

### GRPO Training (3+ GPUs)

```bash
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

---

## Workflow 1: GRPO Training for Math Reasoning

Use this workflow for training reasoning models with group-relative advantages.

### Prerequisites Checklist
- [ ] 3+ GPUs (GPU0: trainer, GPU1: ref_model, GPU2: generator)
- [ ] Model from HuggingFace Hub
- [ ] Training dataset (GSM8K, MATH, etc.)

### Step 1: Create Configuration

```yaml
# config/grpo_math.yaml
model: "Qwen/Qwen2.5-7B-Instruct"

dataset:
  path: "openai/gsm8k"
  split: "train"
  streaming: true

training:
  batch_size: 4
  learning_rate: 1e-6
  seq_len: 4096
  dtype: bfloat16
  gradient_accumulation_steps: 4

grpo:
  n_samples: 8           # Responses per prompt
  clip_low: 0.2
  clip_high: 0.28
  beta: 0.1              # KL penalty coefficient
  temperature: 0.7

services:
  generator:
    procs: 1
    num_replicas: 1
    with_gpus: true
  trainer:
    procs: 1
    num_replicas: 1
    with_gpus: true
  ref_model:
    procs: 1
    num_replicas: 1
    with_gpus: true
```

### Step 2: Define Reward Function

```python
# grading.py
from forge.rl.grading import RewardActor
import re

class MathReward(RewardActor):
    def grade(self, prompt: str, completion: str, metadata: dict) -> float:
        # Extract answer from completion
        match = re.search(r'\\boxed{([^}]+)}', completion)
        if not match:
            return 0.0

        answer = match.group(1).strip()
        ground_truth = metadata.get("answer", "")

        return 1.0 if answer == ground_truth else 0.0
```

### Step 3: Launch Training

```bash
python -m apps.grpo.main --config config/grpo_math.yaml
```

### Step 4: Monitor Progress
- [ ] Check W&B dashboard for loss curves
- [ ] Verify entropy is decreasing (policy becoming more deterministic)
- [ ] Monitor KL divergence (should stay bounded)

---

## Workflow 2: Custom Loss Function

Use this workflow to implement new RL algorithms.

### Step 1: Create Loss Class

```python
# src/forge/rl/loss/custom_loss.py
from forge.rl.loss.ops import (
    compute_logprobs,
    compute_ratio,
    compute_kl,
    aggregate,
)
import torch

class CustomLoss:
    def __init__(self, clip_range: float = 0.2, beta: float = 0.1):
        self.clip_range = clip_range
        self.beta = beta

    def __call__(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        advantages: torch.Tensor,
        generator_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # Compute current log probabilities
        logprobs = compute_logprobs(logits, target_ids)

        # Compute importance ratio
        ratio = compute_ratio(logprobs, generator_logprobs, loss_mask)

        # Clipped policy gradient
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_range,
            1 + self.clip_range
        )
        pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        # KL penalty
        kl = compute_kl(logprobs, ref_logprobs, loss_mask)

        # Total loss
        loss = aggregate(pg_loss + self.beta * kl, loss_mask)

        metrics = {
            "pg_loss": aggregate(pg_loss, loss_mask).item(),
            "kl": kl.mean().item(),
            "ratio_mean": ratio.mean().item(),
        }

        return loss, metrics
```

### Step 2: Integrate into Application

```python
# apps/custom/main.py
from forge.rl.loss.custom_loss import CustomLoss

loss_fn = CustomLoss(clip_range=0.2, beta=0.1)

# In training loop
result = loss_fn(
    logits=trainer_output.logits,
    target_ids=batch.target_ids,
    advantages=advantages,
    generator_logprobs=completions.logprobs,
    loss_mask=batch.target_mask,
    ref_logprobs=ref_logprobs,
)
```

---

## Workflow 3: Multi-GPU Distributed Training

Use this workflow for scaling to multiple GPUs or nodes.

### Configuration for Distributed

```yaml
# config/distributed.yaml
model: "meta-llama/Meta-Llama-3.1-8B-Instruct"

parallelism:
  tensor_parallel_degree: 2    # Split model across GPUs
  pipeline_parallel_degree: 1
  data_parallel_shard_degree: 2

services:
  generator:
    procs: 2                   # 2 processes for TP=2
    num_replicas: 1
    with_gpus: true
  trainer:
    procs: 2
    num_replicas: 1
    with_gpus: true
```

### Launch with SLURM

```bash
# Submit job
sbatch --nodes=2 --gpus-per-node=8 run_grpo.sh
```

### Launch Locally (Multi-GPU)

```bash
# 8 GPU setup
python -m apps.grpo.main \
    --config config/distributed.yaml \
    --trainer.procs 4 \
    --generator.procs 4
```

---

## Core API Reference

### TextTrainBatch

Input structure for training:

```python
@dataclass
class TextTrainBatch:
    input_ids: Tensor      # [batch, seq_len] - Input tokens
    target_ids: Tensor     # [batch, seq_len] - Target tokens
    target_mask: Tensor    # [batch, seq_len] - Which tokens to train on
    target_weights: Tensor # [batch, seq_len] - Per-token weights (advantages)
```

### ForwardBackwardResult

Output from training step:

```python
@dataclass
class ForwardBackwardResult:
    loss: float
    metrics: dict  # e.g., {"perplexity": 2.3, "kl": 0.01}
```

### Completion

Generated output from vLLM:

```python
@dataclass
class Completion:
    text: str              # Generated text
    token_ids: list[int]   # Token IDs
    logprobs: list[float]  # Log probabilities
    metadata: dict         # Custom metadata
```

---

## Built-in Loss Functions

### GRPO (Group Relative Policy Optimization)

```python
from forge.rl.loss import GRPOLoss

loss_fn = GRPOLoss(
    clip_low=0.2,      # Lower clip bound
    clip_high=0.28,    # Upper clip bound
    beta=0.1,          # KL penalty
    agg_type="fixed_horizon"
)
```

### DAPO

```python
from forge.rl.loss import DAPOLoss

loss_fn = DAPOLoss(
    clip_range=0.2,
    adaptive_kl=True,
    kl_target=0.01
)
```

### Available Losses

- `GRPOLoss` - Group Relative Policy Optimization
- `DAPOLoss` - Doubly-Adaptive PPO
- `CISPOLoss` - Constrained Importance Sampling PO
- `GSPOLoss` - Grouped Sample PO
- `SAPOLoss` - Sample-Averaged PO

---

## Common Issues and Solutions

### Issue: Not Enough GPUs

**Symptoms**: "Insufficient GPU resources" error

**Solutions**:
```yaml
# Reduce service requirements
services:
  generator:
    procs: 1
    with_gpus: true
  trainer:
    procs: 1
    with_gpus: true
  # Remove ref_model (uses generator weights)
```

Or use CPU for reference model:
```yaml
ref_model:
  with_gpus: false
```

### Issue: OOM During Generation

**Symptoms**: CUDA OOM in vLLM

**Solutions**:
```yaml
# Reduce batch size
grpo:
  n_samples: 4  # Reduce from 8

# Or reduce sequence length
training:
  seq_len: 2048
```

### Issue: Slow Weight Sync

**Symptoms**: Long pauses between training and generation

**Solutions**:
```bash
# Enable RDMA (if available)
export TORCHSTORE_USE_RDMA=1

# Or reduce sync frequency
training:
  sync_interval: 10  # Sync every 10 steps
```

### Issue: Policy Collapse

**Symptoms**: Entropy drops to zero, reward stops improving

**Solutions**:
```yaml
# Increase KL penalty
grpo:
  beta: 0.2  # Increase from 0.1

# Or add entropy bonus
training:
  entropy_coef: 0.01
```

---

## Resources

- **Documentation**: https://meta-pytorch.org/torchforge
- **GitHub**: https://github.com/meta-pytorch/torchforge
- **Discord**: https://discord.gg/YsTYBh6PD9
- **TorchTitan**: https://github.com/pytorch/torchtitan
- **Monarch**: https://github.com/meta-pytorch/monarch
