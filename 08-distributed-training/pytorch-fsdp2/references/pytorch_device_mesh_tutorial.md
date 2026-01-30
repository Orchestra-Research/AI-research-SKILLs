# Reference: Getting Started with DeviceMesh (PyTorch tutorial)

**Source (official):** PyTorch Recipes — “Getting Started with DeviceMesh”  
https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html  
Created: Jan 24, 2024 • Last updated: Jul 18, 2025 • Last verified: Nov 05, 2024

## What DeviceMesh is (as defined by the tutorial)
DeviceMesh is a higher-level abstraction that **manages ProcessGroups**, making it easier to set up the right communication groups for multi-dimensional parallelism.

The tutorial motivation:
- Without DeviceMesh, users must manually compute rank groupings (replicate/shard groups) and create multiple process groups.
- With DeviceMesh, you describe topology with a shape (e.g., 2D mesh), and slice submeshes by dimension name.

## Why this matters for FSDP2
FSDP2 `fully_shard(..., mesh=...)` takes a `DeviceMesh`:
- 1D mesh: standard full sharding across DP workers.
- 2D mesh: hybrid sharding (HSDP), combining replication + sharding across mesh dimensions.

So the agent should:
- Prefer to create a DeviceMesh early (after init_process_group and setting CUDA device).
- Pass the correct (sub)mesh into `fully_shard` if composing with TP or other dimensions.
