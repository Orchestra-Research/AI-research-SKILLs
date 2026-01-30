# Reference: Fully Sharded Data Parallel (FSDP1) API

**Source (official):** PyTorch docs — “Fully Sharded Data Parallel”  
https://docs.pytorch.org/docs/stable/fsdp.html  
Last accessed: Jan 30, 2026

## Key points (paraphrased from the API docs)
- `torch.distributed.fsdp.FullyShardedDataParallel` is the original FSDP wrapper for sharding module parameters across data-parallel workers.
