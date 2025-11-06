# GitHub Issues

Recent issues from the repository (39 total).

## Open Issues (27)

### #3833: notebook_launcher Kaggle RuntimeError: CUDA error: initialization error
**Labels:** No labels | **Created:** 2025-11-06
[View on GitHub](https://github.com/huggingface/accelerate/issues/3833)

### #3792: tests/test_accelerator.py: Random weights test flakes (possibly aarch64 specific)
**Labels:** No labels | **Created:** 2025-09-24
[View on GitHub](https://github.com/huggingface/accelerate/issues/3792)

### #3806: Installing accelerate slows down Pylance
**Labels:** No labels | **Created:** 2025-10-07
[View on GitHub](https://github.com/huggingface/accelerate/issues/3806)

### #3265: accelerator.prepare
**Labels:** No labels | **Created:** 2024-11-26
[View on GitHub](https://github.com/huggingface/accelerate/issues/3265)

### #3832: FSDP MoE PEFT hangs during forward pass
**Labels:** No labels | **Created:** 2025-11-04
[View on GitHub](https://github.com/huggingface/accelerate/issues/3832)

### #3813: Supporting new hardware
**Labels:** No labels | **Created:** 2025-10-13
[View on GitHub](https://github.com/huggingface/accelerate/issues/3813)

### #3805: Beginner question about usage details
**Labels:** No labels | **Created:** 2025-10-05
[View on GitHub](https://github.com/huggingface/accelerate/issues/3805)

### #3830: Torchao fp8 fails if using accelerate config file with Trainer
**Labels:** No labels | **Created:** 2025-11-03
[View on GitHub](https://github.com/huggingface/accelerate/issues/3830)

### #3789: script: accelerate.py function: clip_grad_norm_  if parameters == [p for p in model.parameters()]: RuntimeError: Boolean value of Tensor with more than one value is ambiguous
**Labels:** No labels | **Created:** 2025-09-20
[View on GitHub](https://github.com/huggingface/accelerate/issues/3789)

### #3829: Does Accelerate automatically set the DataLoader’s sampler to a DistributedSampler?
**Labels:** No labels | **Created:** 2025-11-03
[View on GitHub](https://github.com/huggingface/accelerate/issues/3829)

### #3824: load_checkpoint_and_dispatch not work
**Labels:** No labels | **Created:** 2025-10-31
[View on GitHub](https://github.com/huggingface/accelerate/issues/3824)

### #3826: Accelerate save_state() error using FSDP2/TP
**Labels:** No labels | **Created:** 2025-11-01
[View on GitHub](https://github.com/huggingface/accelerate/issues/3826)

### #3827: Accelerate mixed torch.Tensor and DTensor error when using TE FP8 and FSDP/TP
**Labels:** No labels | **Created:** 2025-11-01
[View on GitHub](https://github.com/huggingface/accelerate/issues/3827)

### #3804: File "/opt/conda/lib/python3.11/site-packages/torch/distributed/device_mesh.py", line 132, in create_sub_mesh [rank1]:     device_mesh._dim_group_infos[mesh_dim_indices[0]] [rank1]:     [rank1]: IndexError: list index out of range
**Labels:** No labels | **Created:** 2025-10-01
[View on GitHub](https://github.com/huggingface/accelerate/issues/3804)

### #3825: Error generating example: 'weight' must be 2-D in model.generate()
**Labels:** No labels | **Created:** 2025-11-01
[View on GitHub](https://github.com/huggingface/accelerate/issues/3825)

### #3803: should_reduce_batch_size function not returning expected output for out of memory exception on H100 NVL MIG
**Labels:** No labels | **Created:** 2025-10-01
[View on GitHub](https://github.com/huggingface/accelerate/issues/3803)

### #3802: i want to request a demo code for StatefulDataLoader , i want to use data checkpoint to recover the train stage`s data state, not only model state , how to use ,StatefulDataLoader  or some code to reach it ?
**Labels:** No labels | **Created:** 2025-09-30
[View on GitHub](https://github.com/huggingface/accelerate/issues/3802)

### #3801: os.env/torch.cuda.empty_cache are not thread safe
**Labels:** No labels | **Created:** 2025-09-30
[View on GitHub](https://github.com/huggingface/accelerate/issues/3801)

### #3822: `TypeError: NoneType object is not callable` occurs when using `fsdp_auto_wrap_policy=NO_WRAP` with `fsdp_activation_checkpointing=true`
**Labels:** No labels | **Created:** 2025-10-30
[View on GitHub](https://github.com/huggingface/accelerate/issues/3822)

### #3821: KeyError: "A parameter in the optimizer couldn't be switched to its sharded version" occurs under specific Accelerate configuration
**Labels:** No labels | **Created:** 2025-10-30
[View on GitHub](https://github.com/huggingface/accelerate/issues/3821)


## Recently Closed Issues (12)

### #3758: TypeError: device() received an invalid combination of arguments
**Labels:** No labels | **Closed:** 2025-11-05
[View on GitHub](https://github.com/huggingface/accelerate/issues/3758)

### #3793: ND-Parallel with FSDP v2 occasionally hang when parallelism_config_dp_replicate_size > 1.
**Labels:** No labels | **Closed:** 2025-11-03
[View on GitHub](https://github.com/huggingface/accelerate/issues/3793)

### #3794: Error when evaluating with multi-gpu
**Labels:** No labels | **Closed:** 2025-11-03
[View on GitHub](https://github.com/huggingface/accelerate/issues/3794)

### #3795: torch.distributed.DistNetworkError
**Labels:** No labels | **Closed:** 2025-11-03
[View on GitHub](https://github.com/huggingface/accelerate/issues/3795)

### #3797: Question: ReduceLROnPlateau wrapped by AcceleratedScheduler in DDP may multiply LR by num_processes?
**Labels:** No labels | **Closed:** 2025-11-03
[View on GitHub](https://github.com/huggingface/accelerate/issues/3797)

### #3788: Issue with IterableDataset and variable batch sequence length
**Labels:** No labels | **Closed:** 2025-10-28
[View on GitHub](https://github.com/huggingface/accelerate/issues/3788)

### #3784: AttributeError: 'Accelerator' object has no attribute 'deepspeed_config'. Did you mean: 'deepspeed_plugin'?
**Labels:** No labels | **Closed:** 2025-10-27
[View on GitHub](https://github.com/huggingface/accelerate/issues/3784)

### #3780: "Gradient accumulation on training samples of variable size" example broken when using DeepSpeed
**Labels:** No labels | **Closed:** 2025-10-24
[View on GitHub](https://github.com/huggingface/accelerate/issues/3780)

### #3770: Need for proper mention in the documentation  that accelerate.cpu_offload   can not be used for training
**Labels:** No labels | **Closed:** 2025-10-21
[View on GitHub](https://github.com/huggingface/accelerate/issues/3770)

### #3778: Unable to specify mp_policy for FSDP2 from config file
**Labels:** No labels | **Closed:** 2025-10-21
[View on GitHub](https://github.com/huggingface/accelerate/issues/3778)

