# Reference: Getting Started with Fully Sharded Data Parallel (FSDP2) tutorial

**Source (official):** PyTorch Tutorials — “Getting Started with Fully Sharded Data Parallel (FSDP2)”  
https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html  
Created: Mar 17, 2022 • Last updated: Sep 02, 2025 • Last verified: Nov 05, 2024

## What the tutorial emphasizes

### How FSDP2 differs from DDP and FSDP1
- FSDP shards **parameters, gradients, and optimizer state**; parameters are all-gathered for compute and reduce-scattered for grads.
- Compared to FSDP1, FSDP2:
  - uses **DTensor per-parameter sharding** (more direct manipulation; sharded state dicts)
  - improves memory management for more deterministic memory behavior
  - supports extensibility points for custom all-gather (e.g., float8/NF4 use cases)

### Model initialization flow (meta-device pattern)
The tutorial’s migration section shows a typical pattern:
- initialize model on `meta`
- apply `fully_shard` to the intended layers (policy expressed by explicit calls)
- apply `fully_shard` to the root module
- materialize weights via `to_empty(device="cuda")`, then run `reset_parameters()`

### State dict workflows
The tutorial describes two main ways:

**A) DTensor APIs (manual)**
- Loading: use `distribute_tensor(full_tensor, meta_param.device_mesh, meta_param.placements)` then `model.load_state_dict(..., assign=True)`
- Saving: call `DTensor.full_tensor()` to all-gather; optionally CPU-offload on rank0 to avoid peak GPU memory

**B) DCP distributed state-dict helpers (recommended when no custom handling needed)**
- Loading: `set_model_state_dict(..., StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))`
- Saving: `get_model_state_dict(..., StateDictOptions(full_state_dict=True, cpu_offload=True))`
- Points to `pytorch/examples` for optimizer state dict save/load with `set_optimizer_state_dict` / `get_optimizer_state_dict`

### Migration guide mapping
The tutorial explicitly maps FSDP1 concepts to FSDP2:
- `sharding_strategy` ↔ `reshard_after_forward` (+ 2D mesh for HYBRID)
- `cpu_offload` ↔ `offload_policy` (`CPUOffloadPolicy`)
- `no_sync()` ↔ `set_requires_gradient_sync`
- `sync_module_states` moves to DCP broadcast-from-rank0 flows

## Practical takeaways for agents
- Express wrapping policy by **explicitly applying `fully_shard`** to chosen submodules.
- Use DCP APIs for flexible checkpointing and resharding unless you must interop with third-party formats.
