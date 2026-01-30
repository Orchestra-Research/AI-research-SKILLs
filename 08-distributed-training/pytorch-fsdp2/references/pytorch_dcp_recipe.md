# Reference: Getting Started with Distributed Checkpoint (DCP) recipe

**Source (official):** PyTorch Tutorials recipe — “Getting Started with Distributed Checkpoint (DCP)”  
https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html  
Created: Oct 02, 2023 • Last updated: Jul 10, 2025 • Last verified: Nov 05, 2024

## Key ideas shown in the recipe
- DCP saves/loads in parallel, and supports resharding across topologies at load time.
- It provides helpers under `torch.distributed.checkpoint.state_dict` to manage distributed `state_dict` generation/loading.

## Example structure (high level)
- Wrap application state in a `Stateful` object, so DCP automatically calls `state_dict()` / `load_state_dict()`
- Use `dcp.save(...)` / `dcp.load(...)`
- Use `get_state_dict` / `set_state_dict` helpers to correctly obtain and apply model/optimizer state dicts in distributed settings

## Practical agent guidance
If adding checkpointing to an FSDP2 training script, this recipe’s patterns are the safest default.
