# Reference: `torch.distributed.fsdp.fully_shard` API (FSDP2)

**Source (official):** PyTorch docs — `torch.distributed.fsdp.fully_shard`  
https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html  
Created: Dec 04, 2024 • Last updated: Oct 13, 2025

## Key facts (paraphrased from the API docs)

### User contract highlights
- `fully_shard(model)` converts `model.parameters()` to **DTensor** at init, then hooks **all-gather** before forward/backward and **free/reshard** after.  
- The optimizer **must be initialized with DTensor parameters** and step must happen on DTensors.
- Call `model(input)` (not `model.forward(input)`) so hooks run; otherwise explicitly `unshard()` or register the forward method for hooking.
- Apply `fully_shard` **bottom-up**: shard submodules first, then the root module, to form efficient communication groups and enable overlap.
- `fully_shard` “unions” the module type in-place with `FSDPModule`, enabling methods like `unshard()` / `reshard()`.

> Short excerpt (<= 25 words): “Users generally should not call fully_shard() only on the topmost root module.”

### Signature & core args
`fully_shard(module, *, mesh=None, reshard_after_forward=None, shard_placement_fn=None, mp_policy=MixedPrecisionPolicy(...), offload_policy=OffloadPolicy(), ignored_params=None)`

- **mesh** (`DeviceMesh`):  
  - 1D mesh ⇒ “classic” FSDP sharding, placement `(Shard(0),)`  
  - 2D mesh ⇒ Hybrid sharding (HSDP): sharded across one dim, replicated across the other, placement `(Replicate(), Shard(0))`
- **reshard_after_forward**:
  - `True`: free unsharded params after forward (re-all-gather during backward)
  - `False`: keep unsharded params after forward (avoid backward all-gather)
  - `None`: defaults to `True` for non-root, `False` for root
  - `int`: reshard to a smaller world-size after forward (must divide shard-dim size)
- **shard_placement_fn**: override per-parameter sharding dim (requires even sharding if not dim-0)
- **ignored_params**: parameters not sharded / not moved / not reduced

## Mixed precision & offload policy classes (same doc page)

### `MixedPrecisionPolicy`
Controls:
- `param_dtype`: dtype used for unsharded parameters during forward/backward
- `reduce_dtype`: dtype used for gradient reduction
- `output_dtype`: dtype used for forward output
- `cast_forward_inputs`: whether to cast forward inputs to `param_dtype`

### `OffloadPolicy` and `CPUOffloadPolicy`
OffloadPolicy controls:
- `param_device` / `reduce_device` / `output_device` (and for CPU offload policy, also `optimizer_state_device`)

## Practical implications for agents
- **Bottom-up sharding** is not optional: it affects grouping and memory/perf.
- **Don’t bypass hooks**: using `model.forward` directly breaks all-gather scheduling.
- **Optimizer construction order matters**: construct optimizer after `fully_shard`.
