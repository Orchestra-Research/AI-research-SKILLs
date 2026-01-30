# Reference: Distributed Checkpoint (DCP) overview (torch.distributed.checkpoint)

**Source (official):** PyTorch docs — `torch.distributed.checkpoint`  
https://docs.pytorch.org/docs/stable/distributed.checkpoint.html  
Created: Nov 16, 2022 • Last updated: Oct 08, 2025

## What DCP does
- Supports saving/loading from **multiple ranks in parallel**
- Handles **load-time resharding**, enabling saving with one cluster topology and loading into another
- Produces **multiple files per checkpoint** (often at least one per rank)
- Operates “in place”: the model allocates storage first; DCP loads into that storage

## Important caveats
- The docs warn: **no guarantees of backwards compatibility** across PyTorch versions for saved `state_dict`s.
- Process-group usage: if you pass a process group, only those ranks should call save/load, and all tensors must belong to that group.

## Where to learn usage
The doc links to official “Getting Started with DCP” and “Asynchronous Saving with DCP” recipes.
