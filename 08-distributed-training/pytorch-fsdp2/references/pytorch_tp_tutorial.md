# Reference: Tensor Parallel (TP) tutorial (and how it composes with FSDP)

**Source (official):** PyTorch Tutorials — “Large Scale Transformer model training with Tensor Parallel (TP)”  
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html  
Created: Apr 19, 2024 • Last updated: Jul 18, 2025 • Last verified: Nov 05, 2024

## Key composition pattern: TP intra-host + FSDP inter-host
The tutorial recommends:
- Run TP on a fast intra-host fabric (e.g., NVLink).
- Run FSDP across hosts (inter-host).

It shows a **2D DeviceMesh** pattern and slicing:
- `mesh_2d = init_device_mesh("cuda", (dp, tp))`
- `tp_mesh = mesh_2d["tp"]` and `dp_mesh = mesh_2d["dp"]`
- Apply TP with `parallelize_module(..., tp_mesh, ...)`
- Apply FSDP2 with `fully_shard(..., mesh=dp_mesh, ...)`

## Practical agent guidance
If the user is already doing TP:
- Ensure FSDP2 `mesh` only includes the DP dimension (often inter-host).
- Leave the TP dimension to `torch.distributed.tensor.parallel`.
