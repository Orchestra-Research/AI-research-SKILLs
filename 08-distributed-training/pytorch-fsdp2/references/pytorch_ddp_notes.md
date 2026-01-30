# Reference: Distributed Data Parallel (DDP) notes

**Source (official):** PyTorch docs — “Distributed Data Parallel”  
https://docs.pytorch.org/docs/stable/notes/ddp.html  
Last accessed: Jan 30, 2026

## Key points (paraphrased from the notes)
- DDP is the standard PyTorch wrapper for distributed data parallel training.
- Typical usage includes initializing the process group, wrapping the model with `DistributedDataParallel`, and training normally.
