# Reference: Asynchronous Saving with Distributed Checkpoint (DCP) recipe

**Source (official):** PyTorch Tutorials recipe — “Asynchronous Saving with Distributed Checkpoint (DCP)”  
https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html  
Created: Jul 22, 2024 • Last updated: Sep 29, 2025 • Last verified: Nov 05, 2024

## What async checkpointing changes
- Moves checkpointing off the critical training path via `torch.distributed.checkpoint.async_save`
- Introduces extra memory overhead because async save first copies model state into internal CPU buffers

## Practical agent guidance
- Use async save when checkpoint stalls are significant and you have headroom for CPU memory.
- Consider pinned memory strategies described in the recipe if performance matters.
