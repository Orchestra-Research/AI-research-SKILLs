---
name: pytorch-fsdp2
description: Adds PyTorch FSDP2 (fully_shard) to training scripts with correct init, sharding, mixed precision/offload config, and distributed checkpointing. Use when models exceed single-GPU memory or when you need DTensor-based sharding with DeviceMesh.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [PyTorch, FSDP2, Fully Sharded Data Parallel, Distributed Training, DTensor, Device Mesh, Sharded Checkpointing, Mixed Precision, Offload, Torch Distributed]
dependencies: [torch]
---

# Skill: Use PyTorch FSDP2 (`fully_shard`) correctly in a training script

This skill teaches a coding agent how to **add PyTorch FSDP2** to a training loop with correct initialization, sharding, mixed precision/offload configuration, and checkpointing.

> FSDP2 in PyTorch is exposed primarily via `torch.distributed.fsdp.fully_shard` and the `FSDPModule` methods it adds in-place to modules. See: `references/pytorch_fully_shard_api.md`, `references/pytorch_fsdp2_tutorial.md`.

---

## When to use this skill

Use FSDP2 when:
- Your model **doesn’t fit** on one GPU (parameters + gradients + optimizer state).
- You want an eager-mode sharding approach that is **DTensor-based per-parameter sharding** (more inspectable, simpler sharded state dicts) than FSDP1.  
- You may later compose DP with **Tensor Parallel** using **DeviceMesh**.

Avoid (or be careful) if:
- You need strict backwards-compatible checkpoints across PyTorch versions (DCP warns against this).
- You’re forced onto older PyTorch versions without the FSDP2 stack.

## Alternatives (when FSDP2 is not the best fit)

- **DistributedDataParallel (DDP)**: Use the standard data-parallel wrapper when you want classic distributed data parallel training.
- **FullyShardedDataParallel (FSDP1)**: Use the original FSDP wrapper for parameter sharding across data-parallel workers.

Reference: `references/pytorch_ddp_notes.md`, `references/pytorch_fsdp1_api.md`.

---

## Contract the agent must follow

1. **Launch with `torchrun`** and set the CUDA device per process (usually via `LOCAL_RANK`).  
2. **Apply `fully_shard()` bottom-up**, i.e., shard submodules (e.g., Transformer blocks) before the root module.  
3. **Call `model(input)`**, not `model.forward(input)`, so the FSDP2 hooks run (unless you explicitly `unshard()` or register the forward method).  
4. **Create the optimizer after sharding** and make sure it is built on the **DTensor parameters** (post-`fully_shard`).  
5. **Checkpoint using Distributed Checkpoint (DCP)** or the distributed-state-dict helpers, not naïve `torch.save(model.state_dict())` unless you deliberately gather to full tensors.

(Each of these rules is directly described in the official API docs/tutorial; see references.)

---

## Step-by-step procedure

### 0) Version & environment sanity
- Prefer a recent stable PyTorch where the docs show FSDP2 and DCP updated recently.
- Use `torchrun --nproc_per_node <gpus_per_node> ...` and ensure `RANK`, `WORLD_SIZE`, `LOCAL_RANK` are visible.

Reference: `references/pytorch_fsdp2_tutorial.md` (launch commands and setup), `references/pytorch_fully_shard_api.md` (user contract).

---

### 1) Initialize distributed and set device
Minimal, correct pattern:
- `dist.init_process_group(backend="nccl")`
- `torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))`
- Optionally create a `DeviceMesh` to describe the data-parallel group(s)

Reference: `references/pytorch_device_mesh_tutorial.md` (why DeviceMesh exists & how it manages process groups).

---

### 2) Build model on meta device (recommended for very large models)
For big models, initialize on `meta`, apply sharding, then materialize weights on GPU:
- `with torch.device("meta"): model = ...`
- apply `fully_shard(...)` on submodules, then `fully_shard(model)`
- `model.to_empty(device="cuda")`
- `model.reset_parameters()` (or your init routine)

Reference: `references/pytorch_fsdp2_tutorial.md` (migration guide shows this flow explicitly).

---

### 3) Apply `fully_shard()` bottom-up (wrapping policy = “apply where needed”)
**Do not** only call `fully_shard` on the topmost module.

Recommended sharding pattern for transformer-like models:
- iterate modules, `if isinstance(m, TransformerBlock): fully_shard(m, ...)`
- then `fully_shard(model, ...)`

Why:
- `fully_shard` forms “parameter groups” for collective efficiency and excludes params already grouped by earlier calls. Bottom-up gives better overlap and lower peak memory.

Reference: `references/pytorch_fully_shard_api.md` (bottom-up requirement and why).

---

### 4) Configure `reshard_after_forward` for memory/perf trade-offs
Default behavior:
- `None` means `True` for non-root modules and `False` for root modules (good default).

Heuristics:
- If you’re memory-bound: keep defaults or force `True` on many blocks.
- If you’re throughput-bound and can afford memory: consider keeping unsharded params longer (root often `False`).
- Advanced: use an `int` to reshard to a smaller mesh after forward (e.g., intra-node) if it’s a meaningful divisor.

Reference: `references/pytorch_fully_shard_api.md` (full semantics).

---

### 5) Mixed precision & offload (optional but common)
FSDP2 uses:
- `mp_policy=MixedPrecisionPolicy(param_dtype=..., reduce_dtype=..., output_dtype=..., cast_forward_inputs=...)`
- `offload_policy=CPUOffloadPolicy()` if you want CPU offload

Rules of thumb:
- Start with BF16 parameters/reductions on H100/A100-class GPUs (if numerically stable for your model).
- Keep `reduce_dtype` aligned with your gradient reduction expectations.
- If you use CPU offload, budget for PCIe/NVLink traffic and runtime overhead.

Reference: `references/pytorch_fully_shard_api.md` (MixedPrecisionPolicy / OffloadPolicy classes).

---

### 6) Optimizer, gradient clipping, accumulation
- Create the optimizer **after** sharding so it holds DTensor params.
- If you need gradient accumulation / no_sync:
  - use the FSDP2 mechanism (`set_requires_gradient_sync`) instead of FSDP1’s `no_sync()`.

Gradient clipping:
- Use the approach shown in the FSDP2 tutorial (“Gradient Clipping and Optimizer with DTensor”), because parameters/gradients are DTensors.

Reference: `references/pytorch_fsdp2_tutorial.md`.

---

### 7) Checkpointing: prefer DCP or distributed state dict helpers
Two recommended approaches:

**A) Distributed Checkpoint (DCP) — best default**
- DCP saves/loads from multiple ranks in parallel and supports load-time resharding.
- DCP produces **multiple files** (often at least one per rank) and operates “in place”.

**B) Distributed state dict helpers**
- `get_model_state_dict` / `set_model_state_dict` with `StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True, ...)`
- For optimizer: `get_optimizer_state_dict` / `set_optimizer_state_dict`

Avoid:
- Saving DTensor state dicts with plain `torch.save` unless you intentionally convert with `DTensor.full_tensor()` and manage memory carefully.

References:
- `references/pytorch_dcp_overview.md` (DCP behavior and caveats)
- `references/pytorch_dcp_recipe.md` and `references/pytorch_dcp_async_recipe.md` (end-to-end usage)
- `references/pytorch_fsdp2_tutorial.md` (DTensor vs DCP state-dict flows)
- `references/pytorch_examples_fsdp2.md` (working checkpoint scripts)

---

## Workflow checklists (copy-paste friendly)

### Workflow A: Retrofit FSDP2 into an existing training script
- [ ] Launch with `torchrun` and initialize the process group.
- [ ] Set the CUDA device from `LOCAL_RANK`; create a `DeviceMesh` if you need multi-dim parallelism.
- [ ] Build the model (use `meta` if needed), apply `fully_shard` bottom-up, then `fully_shard(model)`.
- [ ] Create the optimizer after sharding so it captures DTensor parameters.
- [ ] Use `model(inputs)` so hooks run; use `set_requires_gradient_sync` for accumulation.
- [ ] Add DCP save/load via `torch.distributed.checkpoint` helpers.

Reference: `references/pytorch_fsdp2_tutorial.md`, `references/pytorch_fully_shard_api.md`, `references/pytorch_device_mesh_tutorial.md`, `references/pytorch_dcp_recipe.md`.

### Workflow B: Add DCP save/load (minimal pattern)
- [ ] Wrap state in `Stateful` or assemble state via `get_state_dict`.
- [ ] Call `dcp.save(...)` from all ranks to a shared path.
- [ ] Call `dcp.load(...)` and restore with `set_state_dict`.
- [ ] Validate any resharding assumptions when loading into a different mesh.

Reference: `references/pytorch_dcp_recipe.md`.

## Debug checklist (what the agent should check first)

1. **All ranks on distinct GPUs?**  
   If not, verify `torch.cuda.set_device(LOCAL_RANK)` and your `torchrun` flags.
2. **Did you accidentally call `forward()` directly?**  
   Use `model(input)` or explicitly `unshard()` / register forward.
3. **Is `fully_shard()` applied bottom-up?**  
   If only root is sharded, expect worse memory/perf and possible confusion.
4. **Optimizer created at the right time?**  
   Must be built on DTensor parameters *after* sharding.
5. **Checkpointing path consistent?**  
   - If using DCP, don’t mix with ad-hoc `torch.save` unless you understand conversions.
   - Be mindful of PyTorch-version compatibility warnings for DCP.

---

## Common issues and fixes

- **Forward hooks not running** → Call `model(inputs)` (or `unshard()` explicitly) instead of `model.forward(...)`.
- **Optimizer sees non-DTensor params** → Create optimizer after all `fully_shard` calls.
- **Only root module sharded** → Apply `fully_shard` bottom-up on submodules before the root.
- **Memory spikes after forward** → Set `reshard_after_forward=True` for more modules.
- **Gradient accumulation desync** → Use `set_requires_gradient_sync` instead of FSDP1’s `no_sync()`.

Reference: `references/pytorch_fully_shard_api.md`, `references/pytorch_fsdp2_tutorial.md`.

---

## Minimal reference implementation outline (agent-friendly)

The coding agent should implement a script with these labeled blocks:

- `init_distributed()`: init process group, set device
- `build_model_meta()`: model on meta, apply `fully_shard`, materialize weights
- `build_optimizer()`: optimizer created after sharding
- `train_step()`: forward/backward/step with `model(inputs)` and DTensor-aware patterns
- `checkpoint_save/load()`: DCP or distributed state dict helpers

Concrete examples live in `references/pytorch_examples_fsdp2.md` and the official tutorial reference.

---

## References
- `references/pytorch_fsdp2_tutorial.md`
- `references/pytorch_fully_shard_api.md`
- `references/pytorch_ddp_notes.md`
- `references/pytorch_fsdp1_api.md`
- `references/pytorch_device_mesh_tutorial.md`
- `references/pytorch_tp_tutorial.md`
- `references/pytorch_dcp_overview.md`
- `references/pytorch_dcp_recipe.md`
- `references/pytorch_dcp_async_recipe.md`
- `references/pytorch_examples_fsdp2.md`
- `references/torchtitan_fsdp_notes.md` (optional, production notes)
- `references/ray_train_fsdp2_example.md` (optional, integration example)
