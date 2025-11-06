# Releases

Version history for this repository (69 releases).

## v1.11.0: v1.11.0: TE MXFP8, FP16/BF16 with MPS, Python 3.10 
**Published:** 2025-10-20

## TE MXFP8 support

We've added support for MXFP8 in our TransformerEngine integration. To use that, you need to set  `use_mxfp8_block_scaling` in `fp8_config`. See nvidia docs [here]. (https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html#MXFP8-and-block-scaling) 

* Add support for TE MXFP8 recipe in accelerate by @pstjohn in https://github.com/huggingface/accelerate/pull/3688

## FP16/BF16 Training for MPS devices

BF16 and FP16 support for MPS devices is finally here. You can now pass `mixed_precision = "fp16" or "bf16"` when training on a mac (`fp16` requires torch 2.8 and `bf16` requires torch 2.6)
* Add bf16/fp16 support for amp with mps device by @SunMarc in https://github.com/huggingface/accelerate/pull/3373

## FSDP updates

The following PRs  add respectively support to `ignored_params` and `no_sync()` for FSDPv2:
* feat: add ignored_params support for fsdp2 by @kmehant in https://github.com/huggingface/accelerate/pull/3731
* fix: model.set_requires_gradient_sync(False) should be called to turn off gradient synchronization in FSDP2 by @EquationWalker in https://github.com/huggingface/accelerate/pull/3762

Mixed precision can now be passed as a dtype string from accelerate cli flag or `fsdp_config` in accelerate config file:
* feat: allow mixed precision policy as dtype by @kmehant in https://github.com/huggingface/accelerate/pull/3751

## Nd-parallel updates

Some minor updates concerning nd-parallelism. 

* Context Parallelism docs typos fixed by @sergiopaniego in https://github.com/huggingface/accelerate/pull/3761
* Feat: add to_json by @S1ro1 in https://github.com/huggingface/accelerate/pull/3743
* make torch_native_parallelism examples device agnostic by @yao-matrix in https://github.com/huggingface/accelerate/pull/3759
* [ND Parallel] Update examples, cleanup by @S1ro1 in https://github.com/huggingface/accelerate/pull/3737

## Bump to Python 3.10 

We've dropped support for python 3.9 as it reached EOL in October. 
* Bump to python3.10 + update linter by @SunMarc in https://github.com/huggingface/accelerate/pull/3809

### Lots of minor fixes: 
* fix: CPU RAM efficient loading for nd or HSDP parallelisms by @kmehant in https://github.com/huggingface/accelerate/pull/3740
* xpu INT64 all_gather issue fixed in 2.9 by @yao-matrix in https://github.com/huggingface/accelerate/pull/3756
* Specify device_ids in torch.distributed.barrier for PartialState by @qgallouedec in https://github.com/huggingface/accelerate/pull/3744
* fix: specify device for process_tensor in example usage by @qgallouedec in https://github.com/huggingface/accelerate/pull/3755
* Lower complexity of get_balanced_memory by adding a set by @SamuelBarryCS in https://github.com/huggingface/accelerate/pull/3776
* Fix (skip) cuda cache flush when origin device is `cpu` and offloaded to `meta` by @Qubitium in https://github.com/huggingface/accelerate/pull/3796
* Fix convert LayerNorm without bias to fp8 by @mjun0812 in https://github.com/huggingface/accelerate/pull/3725
* Add optional typing by @cyyever in https://github.com/huggingface/accelerate/pull/3769
* refactor: Use  `with`  in Accelerator.autocast()instead of ` __enter__()` and` __exit__()` for more elegant style. by @EquationWalker in https://github.com/huggingface/accelerate/pull/3767
* switch XPU ccl backend to torch-builtin xccl in test_zero3_integration by @yao-matrix in https://github.com/huggingface/accelerate/pull/3773
* fix FSDP2 test case failure on XPU by @yao-matrix in https://github.com/huggingface/accelerate/pull/3771
* Fix tests by @SunMarc in https://github.com/huggingface/accelerate/pull/3722
* Protect import for device_mesh  by @SunMarc in https://github.com/huggingface/accelerate/pull/3742
* Fix `SWANLAB_MODE` by @SunMarc in https://github.com/huggingface/accelerate/pull/3808
* Fix tracking swanlab by @SunMarc in https://github.com/huggingface/accelerate/pull/3810
* refactor: nit change for get_parameters_from_modules (code debt) by @kmehant in https://github.com/huggingface/accelerate/pull/3815
* Remove deprecated FindTiedParametersResult by @cyyever in https://github.com/huggingface/accelerate/pull/3786
* Add optional typing by @cyyever in https://github.com/huggingface/accelerate/pull/3769
* remove mlflow from testing  by @SunMarc in https://github.com/huggingface/accelerate/pull/3783
* enable 2 model hook ut cases on XPU by @yao-matrix in https://github.com/huggingface/accelerate/pull/3774
* Added Tip for better rendering by @sergiopaniego in https://github.com/huggingface/accelerate/pull/3781
* Fix typos by @cyyever in https://github.com/huggingface/accelerate/pull/3753
* fix: torch_npu import error in some envs by @yanyongyu in https://github.com/huggingface/accelerate/pull/3764
* Fix: typo makes tests fail by @S1ro1 in https://github.com/huggingface/accelerate/pull/3765
* fix Muti node CUDA error: invalid device ordinal #3775 by @RicardoDominguez in https://github.com/huggingface/accelerate/pull/3779
* use reset_peak_memory_stats on xpu by @yao-matrix in https://github.com/huggingface/accelerate/pull/3772


## New Contributors
* @mjun0812 made their first contribution in https://github.com/huggingface/accelerate/pull/3725
* @sergiopaniego made their first contribution in https://github.com/huggingface/accelerate/pull/3761
* @EquationWalker made their first contribution in https://github.com/huggingface/accelerate/pull/3762
* @yanyongyu made their first contribution in https://github.com/huggingface/accelerate/pull/3764
* @RicardoDominguez made their first contribution in https://github.com/huggingface/accelerate/pull/3779
* @SamuelBarryCS made their first contribution in https://github.com/huggingface/accelerate/pull/3776
* @Qubitium made their first contribution in https://github.com/huggingface/accelerate/pull/3796

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.10.1...v1.11.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.11.0)

---

## v1.10.1: v1.10.1: Patchfix
**Published:** 2025-08-25

- Feat: add to_json by @S1ro1 in https://github.com/huggingface/accelerate/pull/3743
- Protect import for device_mesh by @SunMarc in https://github.com/huggingface/accelerate/pull/3742. 

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.10.0...v1.10.1

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.10.1)

---

## v1.10.0: v1.10.0: N-D Parallelism
**Published:** 2025-08-07

# N-D Parallelism

Training large models across multiple GPUs can be complex, especially when combining [different parallelism strategies](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (e.g TP, CP, DP). To simplify this process, we've collaborated with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl/) to introduce an easy-to-use integration that allows you to apply any combination of parallelism strategies directly in your training script. Just pass a `ParallelismConfig` specifying the size of each parallelism type—it's that simple.
Learn more about how it works in our latest [blogpost](https://github.com/huggingface/blog/pull/3006).

```python
parallelism_config = ParallelismConfig(
    dp_shard_size=2,
    dp_replicate_size=2,
    cp_size=2,
    tp_size=2,
)
accelerator = Accelerator(
    parallelism_config=parallelism_config,
   ...
)
model = AutoModelForCausalLM.from_pretrained("your-model-name", device_mesh=accelerator.torch_device_mesh)
model = accelerator.prepare(model)
```

* Parallelism config + TP + HSDP +  BYODM (Bring Your Own Device Mesh) by @SalmanMohammadi in https://github.com/huggingface/accelerate/pull/3682
* Feat: context parallel v2.0 by @S1ro1 in https://github.com/huggingface/accelerate/pull/3700
* set default submesh_tp_size to prevent unset local variable error by @winglian in https://github.com/huggingface/accelerate/pull/3687
* Add Parallelism getter property to Accelerator class by @WoosungMyung in https://github.com/huggingface/accelerate/pull/3703
* Fix: prepare works even if nothing except tp specified (rare) by @S1ro1 in https://github.com/huggingface/accelerate/pull/3707
* Set parallelism_config in constructor due to Trainer reset of State by @winglian in https://github.com/huggingface/accelerate/pull/3713
* Fix: tp size wouldn't read from env by @S1ro1 in https://github.com/huggingface/accelerate/pull/3716
* Remove `ParallelismConfig` from `PartialState` by @SunMarc in https://github.com/huggingface/accelerate/pull/3720


# FSDP improvements

We've fixed ignored modules attribute. With this, it is now possible to train PEFT model that moe layers that contrains `q_proj` and `v_proj` parameters. This is especially important for fine-tuning `gpt-oss` model. 

* ENH: Allow FSDP ignored modules to be regex by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/3698
* TST Add test for FSDP ignored_modules as str by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/3719

# Minor improvements
* feature: CpuOffload pre_forward don't attempt to move if already on device by @JoeGaffney in https://github.com/huggingface/accelerate/pull/3695
* Fix: Ensure environment variable values are case-insensitive in Accelerate by @jp1924 in https://github.com/huggingface/accelerate/pull/3712
* remove use_ipex by @SunMarc in https://github.com/huggingface/accelerate/pull/3721

# New Contributors
* @SalmanMohammadi made their first contribution in https://github.com/huggingface/accelerate/pull/3682
* @WoosungMyung made their first contribution in https://github.com/huggingface/accelerate/pull/3703
* @jp1924 made their first contribution in https://github.com/huggingface/accelerate/pull/3712
* @JoeGaffney made their first contribution in https://github.com/huggingface/accelerate/pull/3695

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.9.0...v1.10.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.10.0)

---

## v1.9.0: v1.9.0: Trackio support, Model loading speedup, Minor distributed improvements
**Published:** 2025-07-16

# Trackio tracker support
We've added support for a trackio, lightweight, 💯 free experiment tracking Python library built on top of 🤗 Datasets and Spaces.

![Screen Recording 2025-06-11 at 5 39 32 PM](https://github.com/user-attachments/assets/5cf12286-54e7-4119-8a20-88c2cbd37ab6)

Main features are: 
- *Local-first* design: dashboard runs locally by default. You can also host it on Spaces by specifying a `space_id`.
- Persists logs locally (or in a private Hugging Face Dataset)
- Visualize experiments with a Gradio dashboard locally (or on Hugging Face Spaces)
- Everything here, including hosting on Hugging Faces, is **free**!

To use it with accelerate, you need to set `log_with` and initialize the trackers 
```python 
accelerator = Accelerator(log_with="trackio")
config={"learning_rate": 0.001, "batch_size": 32}
# init_kwargs in order to host the dashboard on spaces
init_kwargs = {"trackio": {"space_id": "hf_username/space_name"}
accelerator.init_trackers("example_project", config=config, init_kwargs=init_kwargs})
```
Thanks @pcuenca for the integration ! 
* trackio by @pcuenca in https://github.com/huggingface/accelerate/pull/3669

## Model loading speedup when relying `set_module_tensor_to_device `
Setting tensor while clearing cache is very slow, so we added `clear_device` option to disable it. 
Another small optimization is using `non_blocking` everywhere and syncing just before returning control to the user. This makes the loading slightly faster.
* Speedup model loading by 4-5x in Diffusers ⚡ by @a-r-r-o-w in https://github.com/huggingface/accelerate/pull/3674

## FDSP, Deepspeed, FP8 minor improvements

* Add support for e5e2 and default to hybrid when launcher is used by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3640
* Fix FP8 tests, enable FP8 to be used without direct `Accelerator()` configuring by @pstjohn in https://github.com/huggingface/accelerate/pull/3677
* Bunch of FSDP improvements by @S1ro1 in https://github.com/huggingface/accelerate/pull/3671
* Fix: properly error when DDP + Dtensor model by @S1ro1 in https://github.com/huggingface/accelerate/pull/3629
* Fix fsdp2 example typo by @shimizust in https://github.com/huggingface/accelerate/pull/3657
* Added a check in no_sync() to avoid errors when using deepspeed zero2/3 by @xliu0105 in https://github.com/huggingface/accelerate/pull/3656

## 🚨🚨🚨 Breaking changes 🚨🚨🚨
`find_executable_batch_size()` will no longer halves the batch after every OOM. Instead, we will multiply the batch size by 0.9. This should help user not waste gpu capacity. 

* “Stop Halving My Batch!” · Default back-off 0.5 → 0.9 by @SunMarc in https://github.com/huggingface/accelerate/pull/3684

## What's Changed

* [typo] shards instead of shard  by @SunMarc in https://github.com/huggingface/accelerate/pull/3645
* Docs: Fix typos in gradient accumulation guide by @kilavvy in https://github.com/huggingface/accelerate/pull/3649
* xpu enablement on left cases  by @yao-matrix in https://github.com/huggingface/accelerate/pull/3654
* unpin datasets in examples requirements by @SunMarc in https://github.com/huggingface/accelerate/pull/3681
* fix: wandb config not saved in offline mode by @ved1beta in https://github.com/huggingface/accelerate/pull/3648
* accelerate/data_loader.py: do not yield if the base_dataloader is empty by @0xnightwind in https://github.com/huggingface/accelerate/pull/3659
* warn for invalid keys by @ved1beta in https://github.com/huggingface/accelerate/pull/3613
* Update Gaudi runner image to latest SynapseAI and enable previously disabled tests by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3653

## New Contributors
* @kilavvy made their first contribution in https://github.com/huggingface/accelerate/pull/3649
* @shimizust made their first contribution in https://github.com/huggingface/accelerate/pull/3657
* @xliu0105 made their first contribution in https://github.com/huggingface/accelerate/pull/3656
* @0xnightwind made their first contribution in https://github.com/huggingface/accelerate/pull/3659

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.8.1...v1.9.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.9.0)

---

## v1.8.1: v1.8.1: Patchfix
**Published:** 2025-06-20

- Add support for e5e2 and default to hybrid when launcher is used by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3640
- shards by @SunMarc in https://github.com/huggingface/accelerate/pull/3645

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.8.0...v1.8.1

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.8.1)

---

## v1.8.0: v1.8.0: FSDPv2 + FP8, Regional Compilation for DeepSpeed, Faster Distributed Training on Intel CPUs, ipex.optimize deprecation 
**Published:** 2025-06-19

# FSDPv2 refactor + FP8 support

We've simplified how to prepare FSDPv2 models, as there were too many ways to compose FSDP2 with other features (e.g., FP8, torch.compile, activation checkpointing, etc.). Although the setup is now more restrictive, it leads to fewer errors and a more performant user experience. We’ve also added support for FP8. You can read about the results [here](https://github.com/huggingface/accelerate/tree/main/examples/fsdp2). Thanks to @S1ro1 for this contribution!

* [FSDP2] Refactor + FP8 by @S1ro1 in https://github.com/huggingface/accelerate/pull/3585

# Faster Distributed Training on Intel CPUs

We updated the `CCL_WORKER_COUNT` variable and added `KMP` parameters for Intel CPU users. This significantly improves distributed training performance (e.g., Tensor Parallelism), with up to a 40% speed-up on Intel 4th Gen Xeon when training transformer TP models.

* Set ccl and KMP param in simple launch by @jiqing-feng in https://github.com/huggingface/accelerate/pull/3575

# Regional Compilation for DeepSpeed

We added support for regional compilation with the DeepSpeed engine. DeepSpeed’s .compile() modifies models in-place using torch.nn.Module.compile(...), rather than the out-of-place torch.compile(...), so we had to account for that. Thanks @IlyasMoutawwakil for this feature!

* Fix deepspeed regional compilation by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3609

# ipex.optimize deprecation
`ipex.optimize` is being deprecated. Most optimizations have been upstreamed to PyTorch, and future improvements will land there directly. For users without PyTorch 2.8, we’ll continue to rely on IPEX for now.

* remove ipex.optimize in accelerate by @yao-matrix in https://github.com/huggingface/accelerate/pull/3608

# Better XPU Support
We've greatly expanded and stabilized support for Intel XPUs:

* enable fsdp2 benchmark on XPU by @yao-matrix in https://github.com/huggingface/accelerate/pull/3590
* enable big_model_inference on xpu by @yao-matrix in https://github.com/huggingface/accelerate/pull/3595
* enable test_load_checkpoint_and_dispatch_with_broadcast cases on XPU by @yao-matrix in 
* enable test_cli & test_example cases on XPU by @yao-matrix in https://github.com/huggingface/accelerate/pull/3578
* enable torchao and pippy test cases on XPU by @yao-matrix in https://github.com/huggingface/accelerate/pull/3599 
* enable regional_compilation benchmark on xpu by @yao-matrix in https://github.com/huggingface/accelerate/pull/3592
* fix xpu 8bit value loading by @jiqing-feng in https://github.com/huggingface/accelerate/pull/3623
* add device-agnostic GradScaler by @yao-matrix in https://github.com/huggingface/accelerate/pull/3588
* add xpu support in TorchTensorParallelPlugin by @yao-matrix in https://github.com/huggingface/accelerate/pull/3627

# Trackers

We've added support for [SwanLab](https://github.com/SwanHubX/SwanLab) as an experiment tracking backend. Huge thanks to @ShaohonChen for this contribution ! We also deferred all tracker initializations to prevent premature setup of distributed environments.

* Integrate SwanLab for offline/online experiment tracking for Accelerate by @ShaohonChen in https://github.com/huggingface/accelerate/pull/3605
* Fix: Defer Tracker Initialization to Prevent Premature Distributed Setup by @yuanjua in https://github.com/huggingface/accelerate/pull/3581


## What's Changed
* Fix bf16 training with TP  by @SunMarc in https://github.com/huggingface/accelerate/pull/3610
* better handle FP8 with and without deepspeed by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3611
* Update Gaudi Runners by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3593
* goodbye torch_ccl by @yao-matrix in https://github.com/huggingface/accelerate/pull/3580
* Add support for standalone mode when default port is occupied on single node by @laitifranz in https://github.com/huggingface/accelerate/pull/3576
* Resolve logger warnings by @emmanuel-ferdman in https://github.com/huggingface/accelerate/pull/3582
* Add kwargs to optimizer, scheduler and dataloader using function `accelerator().load_state()` by @luiz0992 in https://github.com/huggingface/accelerate/pull/3540
* [docs] no hard-coded cuda in the ddp documentation by @faaany in https://github.com/huggingface/accelerate/pull/3589
* change to use torch.device by @yao-matrix in https://github.com/huggingface/accelerate/pull/3594
* Fix: list object has no attribute keys by @S1ro1 in https://github.com/huggingface/accelerate/pull/3603
* Update Gaudi Runners by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3593
* Fix bf16 training with TP  by @SunMarc in https://github.com/huggingface/accelerate/pull/3610
* better handle FP8 with and without deepspeed by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3611
* Remove device_count for TPU launcher to avoid initializing runtime by @sorgfresser in https://github.com/huggingface/accelerate/pull/3587
* Fix missing te.LayerNorm in intel_transformer_engine by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3619
* Add fp8_e5m2 support in `dtype_byte_size` by @SunMarc in https://github.com/huggingface/accelerate/pull/3625
* [Deepspeed] deepspeed auto grad accum by @kashif in https://github.com/huggingface/accelerate/pull/3630
* Remove hardcoded cuda from fsdpv2 by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3631
* Integrate SwanLab for offline/online experiment tracking for Accelerate by @ShaohonChen in https://github.com/huggingface/accelerate/pull/3605
* Fix Typos in Documentation and Comments by @leopardracer in https://github.com/huggingface/accelerate/pull/3621
* feat: use datasets.IterableDataset shard if possible by @SunMarc in https://github.com/huggingface/accelerate/pull/3635
* [DeepSpeed] sync gradient accum steps from deepspeed plugin by @kashif in https://github.com/huggingface/accelerate/pull/3632
* Feat: add cpu offload by @S1ro1 in https://github.com/huggingface/accelerate/pull/3636
* Fix: correct labels for fsdp2 examples by @S1ro1 in https://github.com/huggingface/accelerate/pull/3637
* fix grad acc deepspeed by @SunMarc in https://github.com/huggingface/accelerate/pull/3638

## New Contributors
* @laitifranz made their first contribution in https://github.com/huggingface/accelerate/pull/3576
* @emmanuel-ferdman made their first contribution in https://github.com/huggingface/accelerate/pull/3582
* @yuanjua made their first contribution in https://github.com/huggingface/accelerate/pull/3581
* @sorgfresser made their first contribution in https://github.com/huggingface/accelerate/pull/3587
* @ShaohonChen made their first contribution in https://github.com/huggingface/accelerate/pull/3605
* @leopardracer made their first contribution in https://github.com/huggingface/accelerate/pull/3621

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.7.0...v1.8.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.8.0)

---

## v1.7.0: v1.7.0 : Regional compilation, Layerwise casting hook, FSDPv2 + QLoRA
**Published:** 2025-05-15

# Regional compilation

Instead of compiling the entire model at once, regional compilation targets repeated blocks (such as decoder layers) first. This allows the compiler to cache and reuse optimized code for subsequent blocks, significantly reducing the cold start compilation time typically seen during the first inference. Thanks @IlyasMoutawwakil  for the feature ! You can view the full benchmark [here](https://github.com/huggingface/accelerate/tree/main/benchmarks/torch.compile), and check out our updated [compilation guide](https://huggingface.co/docs/accelerate/en/usage_guides/compilation) for more details!

![compilation_time-1](https://github.com/user-attachments/assets/38795d12-6ee7-4a10-84c6-d29a0877e36c)

To enable this feature, set `use_regional_compilation=True` in the `TorchDynamoPlugin` configuration.

```python
# Configure the compilation backend
dynamo_plugin = TorchDynamoPlugin(
    use_regional_compilation=True,
    ... # other parameters
)
# Initialize accelerator with the plugin
accelerator = Accelerator(dynamo_plugin=dynamo_plugin)
# This will apply compile_regions to your model
model = accelerator.prepare(model)
```

# Layerwise casting hook

We've introduced a new hook that enables per-layer upcasting and downcasting (e.g., for Linear layers) during inference. This allows users to run models with separate storage and compute dtypes, resulting in memory savings. The concept was first implemented in [diffusers](https://huggingface.co/docs/diffusers/main/en/optimization/memory#layerwise-casting), where downcasting models to FP8 proved effective without major quality degradation. Contributed by @sayakpaul in https://github.com/huggingface/accelerate/pull/3427

```python
model = ....
storage_dtype = torch.float8_e4m3fn
compute_dtype = torch.bfloat16
attach_layerwise_casting_hooks(
            model,
            storage_dtype=storage_dtype,
            compute_dtype=compute_dtype,
        )
```


# Better FSDP2 support 

This release includes numerous new features and bug fixes. Notably, we’ve added support for `FULL_STATE_DICT`, a widely used option in FSDP, now enabling `.save_pretrained()` in transformers to work with FSDP2 wrapped models. QLoRA training is now supported as well but more testing is needed. We have also resolved a backend issue related to parameter offloading to CPU. Additionally, a significant memory spike that occurred when `cpu_ram_efficient_loading=True` was enabled has been fixed. Several other minor improvements and fixes are also included—see the **What’s Changed** section for full details.

- `FULL_STATE_DICT` have been enabled by @S1ro1 in https://github.com/huggingface/accelerate/pull/3527
-  QLoRA support by @winglian in https://github.com/huggingface/accelerate/pull/3546
-  set backend correctly for CUDA+FSDP2+cpu-offload in https://github.com/huggingface/accelerate/pull/3574
-  memory spike fixed when using `cpu_ram_efficient_loading=True` by @S1ro1 in https://github.com/huggingface/accelerate/pull/3482

# Better HPU support:

We have added a [documentation](https://huggingface.co/docs/accelerate/en/usage_guides/gaudi) for Intel Gaudi hardware ! 
The support is already available since v1.5.0 through this [PR](https://github.com/huggingface/accelerate/pull/3378). 

- Add the HPU into accelerate config by @yuanwu2017 in https://github.com/huggingface/accelerate/pull/3495
- Add Gaudi doc by @regisss in https://github.com/huggingface/accelerate/pull/3537

# Torch.compile breaking change for `dynamic` argument

We've updated the logic for setting `self.dynamic` to explicitly preserve None rather than defaulting to `False` when the `USE_DYNAMIC` environment variable is unset. This change aligns the behavior with the PyTorch documentation for [torch.compile](https://docs.pytorch.org/stable/generated/torch.compile.html). Thanks to @yafshar for contributing this improvement in [#3567](https://github.com/huggingface/accelerate/pull/3567).

## What's Changed
* use device agnostic torch.OutOfMemoryError from pytorch 2.5.0 by @yao-matrix in https://github.com/huggingface/accelerate/pull/3475
* Adds style bot by @zach-huggingface in https://github.com/huggingface/accelerate/pull/3478
* Fix a tiny typo in `low_precision_training` guide by @sadra-barikbin in https://github.com/huggingface/accelerate/pull/3488
* Fix check_tied_parameters_in_config for multimodal models by @SunMarc in https://github.com/huggingface/accelerate/pull/3479
* Don't create new param for TorchAO sequential offloading due to weak BC guarantees by @a-r-r-o-w in https://github.com/huggingface/accelerate/pull/3444
* add support for custom function for reducing the batch size by @winglian in https://github.com/huggingface/accelerate/pull/3071
* Fix fp8 deepspeed config by @SunMarc in https://github.com/huggingface/accelerate/pull/3492
* fix warning error by @faaany in https://github.com/huggingface/accelerate/pull/3491
* [bug] unsafe_serialization option in "merge-weights" doesn't work by @cyr0930 in https://github.com/huggingface/accelerate/pull/3496
* Add the HPU into accelerate config by @yuanwu2017 in https://github.com/huggingface/accelerate/pull/3495
* Use `torch.distributed.checkpoint.state_dict.set_model_state_dict` in `load_checkpoint_in_model` by @ringohoffman in https://github.com/huggingface/accelerate/pull/3432
* nit: needed sanity checks for fsdp2 by @kmehant in https://github.com/huggingface/accelerate/pull/3499
* (Part 1) fix: make TP training compatible with new transformers by @kmehant in https://github.com/huggingface/accelerate/pull/3457
* Fix deepspeed tests by @S1ro1 in https://github.com/huggingface/accelerate/pull/3503
* Add FP8 runners + tweak building FP8 image by @zach-huggingface in https://github.com/huggingface/accelerate/pull/3493
* fix: apply torchfix to set `weights_only=True` by @bzhong-solink in https://github.com/huggingface/accelerate/pull/3497
* Fix: require transformers version for tp tests by @S1ro1 in https://github.com/huggingface/accelerate/pull/3504
* Remove deprecated PyTorch/XLA APIs by @zpcore in https://github.com/huggingface/accelerate/pull/3484
* Fix cache issue by upgrading github actions version  by @SunMarc in https://github.com/huggingface/accelerate/pull/3513
* [Feat] Layerwise casting hook by @sayakpaul in https://github.com/huggingface/accelerate/pull/3427
* Add torchao to FP8 error message by @jphme in https://github.com/huggingface/accelerate/pull/3514
* Fix unwanted cuda init due to torchao by @SunMarc in https://github.com/huggingface/accelerate/pull/3530
* Solve link error in internal_mechanism documentation (#3506) by @alvaro-mazcu in https://github.com/huggingface/accelerate/pull/3507
* [FSDP2] Enable FULL_STATE_DICT by @S1ro1 in https://github.com/huggingface/accelerate/pull/3527
* [FSDP2] Fix memory spike with `cpu_ram_efficient_loading=True` by @S1ro1 in https://github.com/huggingface/accelerate/pull/3482
* [FSDP2] Issues in Wrap Policy and Mixed Precision by @jhliu17 in https://github.com/huggingface/accelerate/pull/3528
* Fix logic in `accelerator.prepare` + IPEX for 2+ `nn.Models` and/or `optim.Optimizers` by @mariusarvinte in https://github.com/huggingface/accelerate/pull/3517
* Update Docker builds to align with CI requirements by @matthewdouglas in https://github.com/huggingface/accelerate/pull/3532
* Fix CI due to missing package  by @SunMarc in https://github.com/huggingface/accelerate/pull/3535
* Update big_modeling.md for layerwise casting by @sayakpaul in https://github.com/huggingface/accelerate/pull/3548
* [FSDP2] Fix: "..." is not a buffer or a paremeter by @S1ro1 in
* fix notebook_launcher for Colab TPU compatibility. by @BogdanDidenko in https://github.com/huggingface/accelerate/pull/3541
* Fix typos by @omahs in https://github.com/huggingface/accelerate/pull/3549
* Dynamo regional compilation by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3529
* add support for port 0 auto-selection in multi-GPU environments by @hellobiondi in https://github.com/huggingface/accelerate/pull/3501
* Fix the issue where `set_epoch` does not take effect. by @hongjx175 in https://github.com/huggingface/accelerate/pull/3556
* [FSDP2] Fix casting in `_cast_and_contiguous` by @dlvp in https://github.com/huggingface/accelerate/pull/3559
* [FSDP] Make env var and dataclass flag consistent for `cpu_ram_efficient_loading`  by @SumanthRH in https://github.com/huggingface/accelerate/pull/3307
* canonicalize optimized names before fixing optimizer in fdsp2 by @pstjohn in https://github.com/huggingface/accelerate/pull/3560
* [docs] update deepspeed config path by @faaany in https://github.com/huggingface/accelerate/pull/3561
* preserve parameter keys when removing  prefix by @mjkvaak-amd in https://github.com/huggingface/accelerate/pull/3564
* Add Gaudi doc by @regisss in https://github.com/huggingface/accelerate/pull/3537
* Update dynamic env handling to preserve None when USE_DYNAMIC is unset by @yafshar in https://github.com/huggingface/accelerate/pull/3567
* add a `synchronize` call for xpu in `_gpu_gather` by @faaany in https://github.com/huggingface/accelerate/pull/3563
* simplify model.to logic by @yao-matrix in https://github.com/huggingface/accelerate/pull/3562
* tune env command output by @yao-matrix in https://github.com/huggingface/accelerate/pull/3570
* Add regional compilation to cli tools and env vars by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3572
* reenable FSDP2+qlora support by @winglian in https://github.com/huggingface/accelerate/pull/3546
* Fix prevent duplicate GPU usage in distributed processing by @ved1beta in https://github.com/huggingface/accelerate/pull/3526
* set backend correctly for CUDA+FSDP2+cpu-offload by @SunMarc in https://github.com/huggingface/accelerate/pull/3574
* enable test_dispatch_model_tied_weights_memory_with_nested_offload_cpu on xpu by @yao-matrix in https://github.com/huggingface/accelerate/pull/3569


## New Contributors
* @zach-huggingface made their first contribution in https://github.com/huggingface/accelerate/pull/3478
* @sadra-barikbin made their first contribution in https://github.com/huggingface/accelerate/pull/3488
* @ringohoffman made their first contribution in https://github.com/huggingface/accelerate/pull/3432
* @bzhong-solink made their first contribution in https://github.com/huggingface/accelerate/pull/3497
* @zpcore made their first contribution in https://github.com/huggingface/accelerate/pull/3484
* @jphme made their first contribution in https://github.com/huggingface/accelerate/pull/3514
* @alvaro-mazcu made their first contribution in https://github.com/huggingface/accelerate/pull/3507
* @jhliu17 made their first contribution in https://github.com/huggingface/accelerate/pull/3528
* @BogdanDidenko made their first contribution in https://github.com/huggingface/accelerate/pull/3541
* @hellobiondi made their first contribution in https://github.com/huggingface/accelerate/pull/3501
* @hongjx175 made their first contribution in https://github.com/huggingface/accelerate/pull/3556
* @dlvp made their first contribution in https://github.com/huggingface/accelerate/pull/3559
* @pstjohn made their first contribution in https://github.com/huggingface/accelerate/pull/3560
* @mjkvaak-amd made their first contribution in https://github.com/huggingface/accelerate/pull/3564
* @yafshar made their first contribution in https://github.com/huggingface/accelerate/pull/3567
* @ved1beta made their first contribution in https://github.com/huggingface/accelerate/pull/3526

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.6.0...v1.7.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.7.0)

---

## v1.6.0: v1.6.0: FSDPv2, DeepSpeed TP and XCCL backend support
**Published:** 2025-04-01

# FSDPv2 support
This release introduces the support for [FSDPv2](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) thanks to @S1ro1.

If you are using python code, you need to set `fsdp_version=2` in `FullyShardedDataParallelPlugin`:
```python 
from accelerate import FullyShardedDataParallelPlugin, Accelerator

fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2
    # other options...
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

If want to convert a YAML config that contains the FSDPv1 config to FSDPv2 one , use our conversion tool: 
```
accelerate to-fsdp2 --config_file config.yaml --output_file new_config.yaml`
```

To learn more about the difference between FSDPv1 and FSDPv2, read the following [documentation](https://huggingface.co/docs/accelerate/main/en/concept_guides/fsdp1_vs_fsdp2). 

# DeepSpeed TP support

We have added initial support for DeepSpeed + TP. Not many changes were required as the DeepSpeed APIs was already compatible. We only needed to make sure that the dataloader was compatible with TP and that we were able to save the TP weights. Thanks @inkcherry for the work ! https://github.com/huggingface/accelerate/pull/3390.

To use TP with deepspeed, you need to update the setting in the deepspeed config file by including `tensor_parallel` key:
```
    ....
    "tensor_parallel":{
      "autotp_size": ${autotp_size}
    },
   ...
```
More details in this deepspeed [PR](https://github.com/deepspeedai/DeepSpeed/pull/6922). 


# Support for XCCL distributed backend

We've added support for XCCL which is an Intel distributed backend which can be used with XPU devices. More details in this torch [PR](https://github.com/pytorch/pytorch/issues/141741). Thanks @dvrogozh for the [integration](https://github.com/huggingface/accelerate/pull/3401) ! 

## What's Changed
* Add `log_artifact`, `log_artifacts` and `log_figure` capabilities to the MLflowTracker. by @luiz0992 in https://github.com/huggingface/accelerate/pull/3419
* tensor parallel dataloder for deepspeed accelerator by @inkcherry in https://github.com/huggingface/accelerate/pull/3390
* Fix prod issues by @muellerzr in https://github.com/huggingface/accelerate/pull/3441
* Fix attribute issue with deepspeed tp by @SunMarc in https://github.com/huggingface/accelerate/pull/3443
* Fixed typo in the multi node FSDP slurm example script by @JacobB33 in https://github.com/huggingface/accelerate/pull/3447
* feat: Add no_ssh and slurm multinode launcher options for deepspeed by @hsmallbone in https://github.com/huggingface/accelerate/pull/3329
* Fixup ao module filter func by @muellerzr in https://github.com/huggingface/accelerate/pull/3450
* remove device index workaround on xpu since xpu supports integer device index as cuda now by @yao-matrix in https://github.com/huggingface/accelerate/pull/3448
* enable 2 UT cases on XPU by @yao-matrix in https://github.com/huggingface/accelerate/pull/3445
* Fix AMD GPU support with should_reduce_batch_size() by @cameronshinn in https://github.com/huggingface/accelerate/pull/3405
* Fix device KeyError in tied_params_map by @dvrogozh in https://github.com/huggingface/accelerate/pull/3403
* Initial FSDP2 support by @S1ro1 in https://github.com/huggingface/accelerate/pull/3394
* Fix: clip grad norm in fsdp2 by @S1ro1 in https://github.com/huggingface/accelerate/pull/3465
* Update @ by @muellerzr in https://github.com/huggingface/accelerate/pull/3466
* Fix seeding of new generator for multi GPU by @albertcthomas in https://github.com/huggingface/accelerate/pull/3459
* Fix get_balanced_memory for MPS by @booxter in https://github.com/huggingface/accelerate/pull/3464
* Update CometMLTracker to allow re-using experiment by @Lothiraldan in https://github.com/huggingface/accelerate/pull/3328
* Apply ruff py39 fixes by @cyyever in https://github.com/huggingface/accelerate/pull/3461
* xpu: enable xccl distributed backend by @dvrogozh in https://github.com/huggingface/accelerate/pull/3401
* Update ruff target-version to py39 and apply more fixes by @cyyever in https://github.com/huggingface/accelerate/pull/3470
* [MLU] fix deepspeed dependency  by @huismiling in https://github.com/huggingface/accelerate/pull/3472
* remove use_xpu to fix ut issues, we don't need this since XPU is OOB … by @yao-matrix in https://github.com/huggingface/accelerate/pull/3460
* Bump ruff to 0.11.2 by @cyyever in https://github.com/huggingface/accelerate/pull/3471

## New Contributors
* @luiz0992 made their first contribution in https://github.com/huggingface/accelerate/pull/3419
* @inkcherry made their first contribution in https://github.com/huggingface/accelerate/pull/3390
* @JacobB33 made their first contribution in https://github.com/huggingface/accelerate/pull/3447
* @hsmallbone made their first contribution in https://github.com/huggingface/accelerate/pull/3329
* @yao-matrix made their first contribution in https://github.com/huggingface/accelerate/pull/3448
* @cameronshinn made their first contribution in https://github.com/huggingface/accelerate/pull/3405
* @S1ro1 made their first contribution in https://github.com/huggingface/accelerate/pull/3394
* @albertcthomas made their first contribution in https://github.com/huggingface/accelerate/pull/3459
* @booxter made their first contribution in https://github.com/huggingface/accelerate/pull/3464
* @Lothiraldan made their first contribution in https://github.com/huggingface/accelerate/pull/3328
* @cyyever made their first contribution in https://github.com/huggingface/accelerate/pull/3461

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.5.2...v1.6.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.6.0)

---

## v1.5.2: Patch: v1.5.2
**Published:** 2025-03-14

**Bug Fixes**:
* Fixed an issue with `torch.get_default_device()` requiring a higher version than what we support
* Fixed a broken `pytest` import in prod

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.5.0...v1.5.2

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.5.2)

---

## v1.5.0: v1.5.0: HPU support
**Published:** 2025-03-12

## HPU Support
* Adds in HPU accelerator support for 🤗 Accelerate


## What's Changed
* [bug] fix device index bug for model training loaded with bitsandbytes by @faaany in https://github.com/huggingface/accelerate/pull/3408
* [docs] add the missing `import torch` by @faaany in https://github.com/huggingface/accelerate/pull/3396
* minor doc fixes by @nbroad1881 in https://github.com/huggingface/accelerate/pull/3365
* fix: ensure CLI args take precedence over config file. by @cyr0930 in https://github.com/huggingface/accelerate/pull/3409
* fix: Add `device=torch.get_default_device()` in `torch.Generator`s by @saforem2 in https://github.com/huggingface/accelerate/pull/3420
* Add Tecorigin SDAA accelerator support by @siqi654321 in https://github.com/huggingface/accelerate/pull/3330
* fix typo : thier -> their by @hackty in https://github.com/huggingface/accelerate/pull/3423
* Fix quality by @muellerzr in https://github.com/huggingface/accelerate/pull/3424
* Distributed inference example for llava_next by @VladOS95-cyber in https://github.com/huggingface/accelerate/pull/3417
* HPU support by @IlyasMoutawwakil in https://github.com/huggingface/accelerate/pull/3378

## New Contributors
* @cyr0930 made their first contribution in https://github.com/huggingface/accelerate/pull/3409
* @saforem2 made their first contribution in https://github.com/huggingface/accelerate/pull/3420
* @siqi654321 made their first contribution in https://github.com/huggingface/accelerate/pull/3330
* @hackty made their first contribution in https://github.com/huggingface/accelerate/pull/3423
* @VladOS95-cyber made their first contribution in https://github.com/huggingface/accelerate/pull/3417
* @IlyasMoutawwakil made their first contribution in https://github.com/huggingface/accelerate/pull/3378

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.4.0...v1.5.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.5.0)

---

## v1.4.0: v1.4.0: `torchao` FP8, TP & dataLoader support, fix memory leak
**Published:** 2025-02-17

## `torchao` FP8, initial Tensor Parallel support, and memory leak fixes

### `torchao` FP8

This release introduces a new FP8 API and brings in a new backend: [`torchao`](https://github.com/pytorch/ao/tree/main/torchao/float8). To use, pass in `AORecipeKwargs` to the `Accelerator` while setting `mixed_precision="fp8"`. This is initial support, as it matures we will incorporate more into it (such as `accelerate config`/yaml) in future releases. See our benchmark examples [here](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8/torchao)

## TensorParallel

We have intial support for an in-house solution to TP when working with accelerate dataloaders. check out the PR [here](https://github.com/huggingface/accelerate/pull/3173)

## Bug fixes
* fix triton version check by @faaany in https://github.com/huggingface/accelerate/pull/3345
* fix torch_dtype in estimate memory by @SunMarc in https://github.com/huggingface/accelerate/pull/3383
* works for fp8 with deepspeed by @XiaobingSuper in https://github.com/huggingface/accelerate/pull/3361
* [`memory leak`] Replace GradientState -> DataLoader reference with weakrefs by @tomaarsen in https://github.com/huggingface/accelerate/pull/3391

## What's Changed
* fix triton version check by @faaany in https://github.com/huggingface/accelerate/pull/3345
* [tests] enable BNB test cases in `tests/test_quantization.py` on XPU by @faaany in https://github.com/huggingface/accelerate/pull/3349
* [Dev] Update release directions by @muellerzr in https://github.com/huggingface/accelerate/pull/3352
* [tests] make cuda-only test work on other hardware accelerators by @faaany in https://github.com/huggingface/accelerate/pull/3302
* [tests] remove `require_non_xpu` test markers by @faaany in https://github.com/huggingface/accelerate/pull/3301
* Support more functionalities for MUSA backend by @fmo-mt in https://github.com/huggingface/accelerate/pull/3359
* [tests] enable more bnb tests on XPU   by @faaany in https://github.com/huggingface/accelerate/pull/3350
* feat: support tensor parallel & Data loader by @kmehant in https://github.com/huggingface/accelerate/pull/3173
* DeepSpeed github repo move sync by @stas00 in https://github.com/huggingface/accelerate/pull/3376
* [tests] Fix bnb cpu error by @faaany in https://github.com/huggingface/accelerate/pull/3351
* fix torch_dtype in estimate memory by @SunMarc in https://github.com/huggingface/accelerate/pull/3383
* works for fp8 with deepspeed by @XiaobingSuper in https://github.com/huggingface/accelerate/pull/3361
* fix: typos in documentation files by @maximevtush in https://github.com/huggingface/accelerate/pull/3388
* [examples] upgrade code for seed setting   by @faaany in https://github.com/huggingface/accelerate/pull/3387
* [`memory leak`] Replace GradientState -> DataLoader reference with weakrefs by @tomaarsen in https://github.com/huggingface/accelerate/pull/3391
* add xpu check in `get_quantized_model_device_map` by @faaany in https://github.com/huggingface/accelerate/pull/3397
* Torchao float8 training by @muellerzr in https://github.com/huggingface/accelerate/pull/3348

## New Contributors
* @kmehant made their first contribution in https://github.com/huggingface/accelerate/pull/3173
* @XiaobingSuper made their first contribution in https://github.com/huggingface/accelerate/pull/3361
* @maximevtush made their first contribution in https://github.com/huggingface/accelerate/pull/3388

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.3.0...v1.4.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.4.0)

---

## v1.3.0: v1.3.0 Bug fixes + Require torch 2.0
**Published:** 2025-01-17

## Torch 2.0
As it's been ~2 years since torch 2.0 was first released, we are now requiring this as the **minimum version for Accelerate**, which similarly was done in `transformers` as of its last release.

## Core
* [docs] no hard-coding cuda by @faaany in https://github.com/huggingface/accelerate/pull/3270
* fix load_state_dict for npu by @ji-huazhong in https://github.com/huggingface/accelerate/pull/3211
* Add `keep_torch_compile` param to `unwrap_model` and `extract_model_from_parallel` for distributed compiled model. by @ggoggam in https://github.com/huggingface/accelerate/pull/3282
* [tests] make cuda-only test case device-agnostic by @faaany in https://github.com/huggingface/accelerate/pull/3340
* latest bnb no longer has optim_args attribute on optimizer by @winglian in https://github.com/huggingface/accelerate/pull/3311
* add torchdata version check to avoid "in_order" error by @faaany in https://github.com/huggingface/accelerate/pull/3344
* [docs] fix typo, change "backoff_filter" to "backoff_factor" by @suchot in https://github.com/huggingface/accelerate/pull/3296
* dataloader: check that in_order is in kwargs before trying to drop it by @dvrogozh in https://github.com/huggingface/accelerate/pull/3346
* feat(tpu): remove nprocs from xla.spawn by @tengomucho in https://github.com/huggingface/accelerate/pull/3324

## Big Modeling
* Fix test_nested_hook by @SunMarc in https://github.com/huggingface/accelerate/pull/3289
* correct the return statement of _init_infer_auto_device_map by @Nech-C in https://github.com/huggingface/accelerate/pull/3279
* Use torch.xpu.mem_get_info for XPU by @dvrogozh in https://github.com/huggingface/accelerate/pull/3275
* Ensure that tied parameter is children of module by @pablomlago in https://github.com/huggingface/accelerate/pull/3327
* Fix for offloading when using TorchAO >= 0.7.0 by @a-r-r-o-w in https://github.com/huggingface/accelerate/pull/3332
* Fix offload generate tests by @SunMarc in https://github.com/huggingface/accelerate/pull/3334

## Examples
* Give example on how to handle gradient accumulation with cross-entropy by @ylacombe in https://github.com/huggingface/accelerate/pull/3193

## Full Changelog
## What's Changed
* [docs] no hard-coding cuda by @faaany in https://github.com/huggingface/accelerate/pull/3270
* fix load_state_dict for npu by @ji-huazhong in https://github.com/huggingface/accelerate/pull/3211
* Fix test_nested_hook by @SunMarc in https://github.com/huggingface/accelerate/pull/3289
* correct the return statement of _init_infer_auto_device_map by @Nech-C in https://github.com/huggingface/accelerate/pull/3279
* Give example on how to handle gradient accumulation with cross-entropy by @ylacombe in https://github.com/huggingface/accelerate/pull/3193
* Use torch.xpu.mem_get_info for XPU by @dvrogozh in https://github.com/huggingface/accelerate/pull/3275
* Add `keep_torch_compile` param to `unwrap_model` and `extract_model_from_parallel` for distributed compiled model. by @ggoggam in https://github.com/huggingface/accelerate/pull/3282
* Ensure that tied parameter is children of module by @pablomlago in https://github.com/huggingface/accelerate/pull/3327
* Bye bye torch <2 by @muellerzr in https://github.com/huggingface/accelerate/pull/3331
* Fixup docker build err by @muellerzr in https://github.com/huggingface/accelerate/pull/3333
* feat(tpu): remove nprocs from xla.spawn by @tengomucho in https://github.com/huggingface/accelerate/pull/3324
* Fix offload generate tests by @SunMarc in https://github.com/huggingface/accelerate/pull/3334
* [tests] make cuda-only test case device-agnostic by @faaany in https://github.com/huggingface/accelerate/pull/3340
* latest bnb no longer has optim_args attribute on optimizer by @winglian in https://github.com/huggingface/accelerate/pull/3311
* Fix for offloading when using TorchAO >= 0.7.0 by @a-r-r-o-w in https://github.com/huggingface/accelerate/pull/3332
* add torchdata version check to avoid "in_order" error by @faaany in https://github.com/huggingface/accelerate/pull/3344
* [docs] fix typo, change "backoff_filter" to "backoff_factor" by @suchot in https://github.com/huggingface/accelerate/pull/3296
* dataloader: check that in_order is in kwargs before trying to drop it by @dvrogozh in https://github.com/huggingface/accelerate/pull/3346

## New Contributors
* @ylacombe made their first contribution in https://github.com/huggingface/accelerate/pull/3193
* @ggoggam made their first contribution in https://github.com/huggingface/accelerate/pull/3282
* @pablomlago made their first contribution in https://github.com/huggingface/accelerate/pull/3327
* @tengomucho made their first contribution in https://github.com/huggingface/accelerate/pull/3324
* @suchot made their first contribution in https://github.com/huggingface/accelerate/pull/3296

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.2.1...v1.3.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.3.0)

---

## v1.2.1: v1.2.1: Patchfix
**Published:** 2024-12-13

* fix: add max_memory to _init_infer_auto_device_map's return statement in https://github.com/huggingface/accelerate/pull/3279 by @Nech-C 
* fix load_state_dict for npu in https://github.com/huggingface/accelerate/pull/3211 by @statelesshz 

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.2.0...v1.2.1

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.2.1)

---

## v1.2.0: v1.2.0: Bug Squashing & Fixes across the board
**Published:** 2024-12-13

## Core
* enable `find_executable_batch_size` on XPU by @faaany in https://github.com/huggingface/accelerate/pull/3236
* Use `numpy._core` instead of `numpy.core` by @qgallouedec in https://github.com/huggingface/accelerate/pull/3247
* Add warnings and fallback for unassigned devices in infer_auto_device_map by @Nech-C in https://github.com/huggingface/accelerate/pull/3066
* Allow for full dynamo config passed to Accelerator by @muellerzr in https://github.com/huggingface/accelerate/pull/3251
* [WIP] FEAT Decorator to purge accelerate env vars by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/3252
* [`data_loader`] Optionally also propagate set_epoch to batch sampler by @tomaarsen in https://github.com/huggingface/accelerate/pull/3246
* use XPU instead of GPU in the `accelerate config` prompt text by @faaany in https://github.com/huggingface/accelerate/pull/3268

## Big Modeling
* Fix `align_module_device`, ensure only cpu tensors for `get_state_dict_offloaded_model` by @kylesayrs in https://github.com/huggingface/accelerate/pull/3217
* Remove hook for bnb 4-bit  by @SunMarc in https://github.com/huggingface/accelerate/pull/3223
* [docs] add instruction to install bnb on non-cuda devices by @faaany in https://github.com/huggingface/accelerate/pull/3227
* Take care of case when "_tied_weights_keys" is not an attribute by @fabianlim in https://github.com/huggingface/accelerate/pull/3226
* Update deferring_execution.md by @max-yue in https://github.com/huggingface/accelerate/pull/3262
* Revert default behavior of `get_state_dict_from_offload` by @kylesayrs in https://github.com/huggingface/accelerate/pull/3253
* Fix: Resolve #3060, `preload_module_classes` is lost for nested modules by @wejoncy in https://github.com/huggingface/accelerate/pull/3248

## DeepSpeed
* Select the DeepSpeedCPUOptimizer based on the original optimizer class. by @eljandoubi in https://github.com/huggingface/accelerate/pull/3255
* support for wrapped schedulefree optimizer when using deepspeed by @winglian in https://github.com/huggingface/accelerate/pull/3266

## Documentation

* Update code in tracking documentation  by @faaany in https://github.com/huggingface/accelerate/pull/3235
* Replaced set/check breakpoint with set/check trigger in the troubleshooting documentation by @relh in https://github.com/huggingface/accelerate/pull/3259

* Update set-seed by @faaany in https://github.com/huggingface/accelerate/pull/3228
* Fix typo by @faaany in https://github.com/huggingface/accelerate/pull/3221
* Use real path for `checkpoint` by @faaany in https://github.com/huggingface/accelerate/pull/3220
* Fixed multiple typos for Tutorials and Guides docs by @henryhmko in https://github.com/huggingface/accelerate/pull/3274

## New Contributors
* @winglian made their first contribution in https://github.com/huggingface/accelerate/pull/3266
* @max-yue made their first contribution in https://github.com/huggingface/accelerate/pull/3262
* @as12138 made their first contribution in https://github.com/huggingface/accelerate/pull/3261
* @relh made their first contribution in https://github.com/huggingface/accelerate/pull/3259
* @wejoncy made their first contribution in https://github.com/huggingface/accelerate/pull/3248
* @henryhmko made their first contribution in https://github.com/huggingface/accelerate/pull/3274


## Full Changelog
* Fix `align_module_device`, ensure only cpu tensors for `get_state_dict_offloaded_model` by @kylesayrs in https://github.com/huggingface/accelerate/pull/3217
* remove hook for bnb 4-bit  by @SunMarc in https://github.com/huggingface/accelerate/pull/3223
* enable `find_executable_batch_size` on XPU by @faaany in https://github.com/huggingface/accelerate/pull/3236
* take care of case when "_tied_weights_keys" is not an attribute by @fabianlim in https://github.com/huggingface/accelerate/pull/3226
* [docs] update code in tracking documentation  by @faaany in https://github.com/huggingface/accelerate/pull/3235
* Add warnings and fallback for unassigned devices in infer_auto_device_map by @Nech-C in https://github.com/huggingface/accelerate/pull/3066
* [`data_loader`] Optionally also propagate set_epoch to batch sampler by @tomaarsen in https://github.com/huggingface/accelerate/pull/3246
* [docs] add instruction to install bnb on non-cuda devices by @faaany in https://github.com/huggingface/accelerate/pull/3227
* Use `numpy._core` instead of `numpy.core` by @qgallouedec in https://github.com/huggingface/accelerate/pull/3247
* Allow for full dynamo config passed to Accelerator by @muellerzr in https://github.com/huggingface/accelerate/pull/3251
* [WIP] FEAT Decorator to purge accelerate env vars by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/3252
* use XPU instead of GPU in the `accelerate config` prompt text by @faaany in https://github.com/huggingface/accelerate/pull/3268
* support for wrapped schedulefree optimizer when using deepspeed by @winglian in https://github.com/huggingface/accelerate/pull/3266
* Update deferring_execution.md by @max-yue in https://github.com/huggingface/accelerate/pull/3262
* Fix: Resolve #3257 by @as12138 in https://github.com/huggingface/accelerate/pull/3261
* Replaced set/check breakpoint with set/check trigger in the troubleshooting documentation by @relh in https://github.com/huggingface/accelerate/pull/3259
* Select the DeepSpeedCPUOptimizer based on the original optimizer class. by @eljandoubi in https://github.com/huggingface/accelerate/pull/3255
* Revert default behavior of `get_state_dict_from_offload` by @kylesayrs in https://github.com/huggingface/accelerate/pull/3253
* Fix: Resolve #3060, `preload_module_classes` is lost for nested modules by @wejoncy in https://github.com/huggingface/accelerate/pull/3248
* [docs] update set-seed by @faaany in https://github.com/huggingface/accelerate/pull/3228
* [docs] fix typo by @faaany in https://github.com/huggingface/accelerate/pull/3221
* [docs] use real path for `checkpoint` by @faaany in https://github.com/huggingface/accelerate/pull/3220
* Fixed multiple typos for Tutorials and Guides docs by @henryhmko in https://github.com/huggingface/accelerate/pull/3274

## Code Diff
Release diff: https://github.com/huggingface/accelerate/compare/v1.1.1...v1.2.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.2.0)

---

## v1.1.0: v1.1.0: Python 3.9 minimum, torch dynamo deepspeed support, and bug fixes
**Published:** 2024-11-01

## Internals:
* Allow for a `data_seed` argument in https://github.com/huggingface/accelerate/pull/3150
* Trigger `weights_only=True` by default for all compatible objects when checkpointing and saving with `torch.save` in https://github.com/huggingface/accelerate/pull/3036
* Handle negative values for `dim` input in `pad_across_processes` in https://github.com/huggingface/accelerate/pull/3114
* Enable cpu bnb distributed lora finetune in https://github.com/huggingface/accelerate/pull/3159

## DeepSpeed
* Support torch dynamo for deepspeed>=0.14.4 in https://github.com/huggingface/accelerate/pull/3069

## Megatron
* update Megatron-LM plugin code to version 0.8.0 or higher in https://github.com/huggingface/accelerate/pull/3174

## Big Model Inference
* New `has_offloaded_params` utility added in https://github.com/huggingface/accelerate/pull/3188

## Examples
* Florence2 distributed inference example in https://github.com/huggingface/accelerate/pull/3123

## Full Changelog
* Handle negative values for `dim` input in `pad_across_processes` by @mariusarvinte in https://github.com/huggingface/accelerate/pull/3114
* Fixup DS issue with weakref by @muellerzr in https://github.com/huggingface/accelerate/pull/3143
* Refactor scaler to util by @muellerzr in https://github.com/huggingface/accelerate/pull/3142
* DS fix, continued by @muellerzr in https://github.com/huggingface/accelerate/pull/3145
* Florence2 distributed inference example by @hlky in https://github.com/huggingface/accelerate/pull/3123
* POC: Allow for a `data_seed` by @muellerzr in https://github.com/huggingface/accelerate/pull/3150
* Adding multi gpu speech generation by @dame-cell in https://github.com/huggingface/accelerate/pull/3149
* support torch dynamo for deepspeed>=0.14.4 by @oraluben in https://github.com/huggingface/accelerate/pull/3069
* Fixup Zero3 + `save_model` by @muellerzr in https://github.com/huggingface/accelerate/pull/3146
* Trigger `weights_only=True` by default for all compatible objects by @muellerzr in https://github.com/huggingface/accelerate/pull/3036
* Remove broken dynamo test by @oraluben in https://github.com/huggingface/accelerate/pull/3155
* fix version check bug in `get_xpu_available_memory` by @faaany in https://github.com/huggingface/accelerate/pull/3165
* enable cpu bnb distributed lora finetune by @jiqing-feng in https://github.com/huggingface/accelerate/pull/3159
* [Utils] `has_offloaded_params` by @kylesayrs in https://github.com/huggingface/accelerate/pull/3188
* fix bnb by @eljandoubi in https://github.com/huggingface/accelerate/pull/3186
* [docs] update neptune API by @faaany in https://github.com/huggingface/accelerate/pull/3181
* docs: fix a wrong word in comment in src/accelerate/accelerate.py:1255 by @Rebornix-zero in https://github.com/huggingface/accelerate/pull/3183
* [docs] use nn.module instead of tensor as model  by @faaany in https://github.com/huggingface/accelerate/pull/3157
* Fix typo by @kylesayrs in https://github.com/huggingface/accelerate/pull/3191
* MLU devices : Checks if mlu is available via an cndev-based check which won't trigger the drivers and leave mlu by @huismiling in https://github.com/huggingface/accelerate/pull/3187
* update Megatron-LM plugin code to version 0.8.0 or higher. by @eljandoubi in https://github.com/huggingface/accelerate/pull/3174
* 🚨 🚨 🚨 Goodbye Python 3.8! 🚨 🚨 🚨  by @muellerzr in https://github.com/huggingface/accelerate/pull/3194
* Update transformers.deepspeed references from transformers 4.46.0 release by @loadams in https://github.com/huggingface/accelerate/pull/3196
* eliminate dead code by @statelesshz in https://github.com/huggingface/accelerate/pull/3198
* take `torch.nn.Module` model into account when moving to device   by @faaany in https://github.com/huggingface/accelerate/pull/3167
* [docs] add xpu part and fix bug in `torchrun`  by @faaany in https://github.com/huggingface/accelerate/pull/3166
* Models With Tied Weights Need Re-Tieing After FSDP Param Init by @fabianlim in https://github.com/huggingface/accelerate/pull/3154
* add the missing xpu for local sgd by @faaany in https://github.com/huggingface/accelerate/pull/3163
* typo fix in big_modeling.py by @a-r-r-o-w in https://github.com/huggingface/accelerate/pull/3207
* [Utils] `align_module_device` by @kylesayrs in https://github.com/huggingface/accelerate/pull/3204

## New Contributors
* @mariusarvinte made their first contribution in https://github.com/huggingface/accelerate/pull/3114
* @hlky made their first contribution in https://github.com/huggingface/accelerate/pull/3123
* @dame-cell made their first contribution in https://github.com/huggingface/accelerate/pull/3149
* @kylesayrs made their first contribution in https://github.com/huggingface/accelerate/pull/3188
* @eljandoubi made their first contribution in https://github.com/huggingface/accelerate/pull/3186
* @Rebornix-zero made their first contribution in https://github.com/huggingface/accelerate/pull/3183
* @loadams made their first contribution in https://github.com/huggingface/accelerate/pull/3196

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.0.1...v1.1.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.1.0)

---

## v1.0.1: v1.0.1: Bugfix
**Published:** 2024-10-12

## Bugfixes

* Fixes an issue where the `auto` values were no longer being parsed when using [deepspeed](https://github.com/huggingface/accelerate/pull/3143)
* Fixes a broken test in the deepspeed tests related to the [auto values](https://github.com/huggingface/accelerate/pull/3145)

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v1.0.0...v1.0.1

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.0.1)

---

## v1.0.0: Accelerate 1.0.0 is here!
**Published:** 2024-10-07

## 🚀 Accelerate 1.0 🚀 

With `accelerate` 1.0, we are officially stating that the core parts of the API are now "stable" and ready for the future of what the world of distributed training and PyTorch has to handle. With these release notes, we will focus first on the major breaking changes to get your code fixed, followed by what is new specifically between 0.34.0 and 1.0.

To read more, check out our official blog [here](https://huggingface.co/blog/accelerate-v1)

## Migration assistance

* Passing in `dispatch_batches`, `split_batches`, `even_batches`, and `use_seedable_sampler` to the `Accelerator()` should now be handled by creating an `accelerate.utils.DataLoaderConfiguration()` and passing this to the `Accelerator()` instead (`Accelerator(dataloader_config=DataLoaderConfiguration(...))`)
* `Accelerator().use_fp16` and `AcceleratorState().use_fp16` have been removed; this should be replaced by checking `accelerator.mixed_precision == "fp16"`
* `Accelerator().autocast()` no longer accepts a `cache_enabled` argument. Instead, an `AutocastKwargs()` instance should be used which handles this flag (among others) passing it to the `Accelerator` (`Accelerator(kwargs_handlers=[AutocastKwargs(cache_enabled=True)])`)
* `accelerate.utils.is_tpu_available` should be replaced with `accelerate.utils.is_torch_xla_available`
* `accelerate.utils.modeling.shard_checkpoint` should be replaced with `split_torch_state_dict_into_shards` from the `huggingface_hub` library
* `accelerate.tqdm.tqdm()` no longer accepts `True`/`False` as the first argument, and instead, `main_process_only` should be passed in as a named argument

## Multiple Model DeepSpeed Support

After long request, we finally have multiple model DeepSpeed support in Accelerate! (though it is quite early still). Read the full tutorial [here](https://huggingface.co/docs/accelerate/v1.0.0/en/usage_guides/deepspeed_multiple_model#using-multiple-models-with-deepspeed), however essentially:

When using multiple models, a DeepSpeed plugin should be created for each model (and as a result, a separate config). a few examples are below:

### Knowledge distillation

(Where we train only one model, zero3, and another is used for inference, zero2)

```python
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

zero2_plugin = DeepSpeedPlugin(hf_ds_config="zero2_config.json")
zero3_plugin = DeepSpeedPlugin(hf_ds_config="zero3_config.json")

deepspeed_plugins = {"student": zero2_plugin, "teacher": zero3_plugin}


accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)
```

To then select which plugin to be used at a certain time (aka when calling `prepare`), we call `accelerator.state.select_deepspeed_plugin("name"), where the first plugin is active by default:

```python
accelerator.state.select_deepspeed_plugin("student")
student_model, optimizer, scheduler = ...
student_model, optimizer, scheduler, train_dataloader = accelerator.prepare(student_model, optimizer, scheduler, train_dataloader)

accelerator.state.select_deepspeed_plugin("teacher") # This will automatically enable zero init
teacher_model = AutoModel.from_pretrained(...)
teacher_model = accelerator.prepare(teacher_model)
```

### Multiple disjoint models

For disjoint models, separate accelerators should be used for each model, and their own `.backward()` should be called later:

```python
for batch in dl:
    outputs1 = first_model(**batch)
    first_accelerator.backward(outputs1.loss)
    first_optimizer.step()
    first_scheduler.step()
    first_optimizer.zero_grad()
    
    outputs2 = model2(**batch)
    second_accelerator.backward(outputs2.loss)
    second_optimizer.step()
    second_scheduler.step()
    second_optimizer.zero_grad()
```

## FP8

We've enabled MS-AMP support up to FSDP. At this time we are not going forward with implementing FSDP support with MS-AMP, due to design issues between both libraries that don't make them inter-op easily. 

## FSDP
* Fixed FSDP auto_wrap using characters instead of full str for layers
* Re-enable setting state dict type manually

## Big Modeling
* Removed cpu restriction for bnb training

## What's Changed
* Fix FSDP auto_wrap using characters instead of full str for layers by @muellerzr in https://github.com/huggingface/accelerate/pull/3075
* Allow DataLoaderAdapter subclasses to be pickled by implementing `__reduce__` by @byi8220 in https://github.com/huggingface/accelerate/pull/3074
* Fix three typos in src/accelerate/data_loader.py by @xiabingquan in https://github.com/huggingface/accelerate/pull/3082
* Re-enable setting state dict type by @muellerzr in https://github.com/huggingface/accelerate/pull/3084
* Support sequential cpu offloading with torchao quantized tensors by @a-r-r-o-w in https://github.com/huggingface/accelerate/pull/3085
* fix bug in `_get_named_modules` by @faaany in https://github.com/huggingface/accelerate/pull/3052
* use the correct available memory API for XPU by @faaany in https://github.com/huggingface/accelerate/pull/3076
* fix `skip_keys` usage in forward hooks by @152334H in https://github.com/huggingface/accelerate/pull/3088
* Update README.md to include distributed image generation gist by @sayakpaul in https://github.com/huggingface/accelerate/pull/3077
* MAINT: Upgrade ruff to v0.6.4 by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/3095
* Revert "Enable Unwrapping for Model State Dicts (FSDP)" by @SunMarc in https://github.com/huggingface/accelerate/pull/3096
* MS-AMP support (w/o FSDP) by @muellerzr in https://github.com/huggingface/accelerate/pull/3093
* [docs] DataLoaderConfiguration docstring by @stevhliu in https://github.com/huggingface/accelerate/pull/3103
* MAINT: Permission for GH token in stale.yml by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/3102
* [docs] Doc sprint by @stevhliu in https://github.com/huggingface/accelerate/pull/3099
* Update image ref for docs by @muellerzr in https://github.com/huggingface/accelerate/pull/3105
* No more t5 by @muellerzr in https://github.com/huggingface/accelerate/pull/3107
* [docs] More docstrings by @stevhliu in https://github.com/huggingface/accelerate/pull/3108
* 🚨🚨🚨 The Great Deprecation 🚨🚨🚨 by @muellerzr in https://github.com/huggingface/accelerate/pull/3098
* POC: multiple model/configuration DeepSpeed support by @muellerzr in https://github.com/huggingface/accelerate/pull/3097
* Fixup test_sync w/ deprecated stuff by @muellerzr in https://github.com/huggingface/accelerate/pull/3109
* Switch to XLA instead of TPU by @SunMarc in https://github.com/huggingface/accelerate/pull/3118
* [tests] skip pippy tests for XPU  by @faaany in https://github.com/huggingface/accelerate/pull/3119
* Fixup multiple model DS tests by @muellerzr in https://github.com/huggingface/accelerate/pull/3131
* remove cpu restriction for bnb training by @jiqing-feng in https://github.com/huggingface/accelerate/pull/3062
* fix deprecated `torch.cuda.amp.GradScaler` FutureWarning for pytorch 2.4+ by @Mon-ius in https://github.com/huggingface/accelerate/pull/3132
* 🐛 [HotFix] Handle Profiler Activities Based on PyTorch Version by @yhna940 in https://github.com/huggingface/accelerate/pull/3136
* only move model to device when model is in cpu and target device is xpu by @faaany in https://github.com/huggingface/accelerate/pull/3133
* fix tip brackets typo by @davanstrien in https://github.com/huggingface/accelerate/pull/3129
* typo of "scalar" instead of "scaler" by @tonyzhaozh in https://github.com/huggingface/accelerate/pull/3116
* MNT Permission for PRs for GH token in stale.yml by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/3112

## New Contributors
* @xiabingquan made their first contribution in https://github.com/huggingface/accelerate/pull/3082
* @a-r-r-o-w made their first contribution in https://github.com/huggingface/accelerate/pull/3085
* @152334H made their first contribution in https://github.com/huggingface/accelerate/pull/3088
* @sayakpaul made their first contribution in https://github.com/huggingface/accelerate/pull/3077
* @Mon-ius made their first contribution in https://github.com/huggingface/accelerate/pull/3132
* @davanstrien made their first contribution in https://github.com/huggingface/accelerate/pull/3129
* @tonyzhaozh made their first contribution in https://github.com/huggingface/accelerate/pull/3116

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.34.2...v1.0.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v1.0.0)

---

## v0.34.1: v0.34.1 Patchfix
**Published:** 2024-09-05

## Bug fixes
* Fixes an issue where processed `DataLoaders` could no longer be pickled in #3074 thanks to @byi8220 
* Fixes an issue when using FSDP where `default_transformers_cls_names_to_wrap` would separate `_no_split_modules` by characters instead of keeping it as a list of layer names in #3075 

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.34.0...v0.34.1

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.34.1)

---

## v0.34.0: v0.34.0: StatefulDataLoader Support, FP8 Improvements, and PyTorch Updates!
**Published:** 2024-09-03

## Dependency Changes
- **Updated Safetensors Requirement:** The library now requires `safetensors` version 0.4.3.
- **Added support for Numpy 2.0:** The library now fully supports `numpy` 2.0.0

## Core

### New Script Behavior Changes
- **Process Group Management:** PyTorch now requires users to destroy process groups after training. The `accelerate` library will handle this automatically with `accelerator.end_training()`, or you can do it manually using `PartialState().destroy_process_group()`.
- **MLU Device Support:** Added support for saving and loading RNG states on MLU devices by @huismiling
- **NPU Support:** Corrected backend and distributed settings when using `transfer_to_npu`, ensuring better performance and compatibility.

### DataLoader Enhancements
- **Stateful DataDataLoader:** We are excited to announce that early support has been added for the `StatefulDataLoader` from `torchdata`, allowing better handling of data loading states. Enable by passing `use_stateful_dataloader=True` to the `DataLoaderConfiguration`, and when calling `load_state()` the `DataLoader` will automatically be resumed from its last step, no more having to iterate through passed batches.
- **Decoupled Data Loader Preparation:** The `prepare_data_loader()` function is now **independent** of the `Accelerator`, giving you more flexibility towards which API levels you would like to use.
- **XLA Compatibility:** Added support for skipping initial batches when using XLA.
- **Improved State Management:** Bug fixes and enhancements for saving/loading `DataLoader` states, ensuring smoother training sessions.
- **Epoch Setting:** Introduced the `set_epoch` function for `MpDeviceLoaderWrapper`.

### FP8 Training Improvements
- **Enhanced FP8 Training:** Fully Sharded Data Parallelism (FSDP) and DeepSpeed support now work seamlessly with `TransformerEngine` FP8 training, including better defaults for the quantized FP8 weights.
- **Integration baseline**: We've added a new suite of examples and benchmarks to ensure that our `TransformerEngine` integration works exactly as intended. These scripts run one half using 🤗 Accelerate's integration, the other with raw `TransformersEngine`, providing users with a nice example of what we do under the hood with accelerate, and a good sanity check to make sure nothing breaks down over time. Find them [here](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8)
- **Import Fixes:** Resolved issues with import checks for the Transformers Engine that has downstream issues.
- **FP8 Docker Images:** We've added new docker images for `TransformerEngine` and `accelerate` as well. Use `docker pull huggingface/accelerate@gpu-fp8-transformerengine` to quickly get an environment going. 


## `torchpippy` no more, long live `torch.distributed.pipelining`
- With the latest PyTorch release, `torchpippy` is now fully integrated into torch core, and as a result we are **exclusively supporting the PyTorch implementation from now on**
- There are breaking examples and changes that comes from this shift. Namely:
  - Tracing of inputs is done with a shape *each GPU will see*, rather than the size of the total batch. So for 2 GPUs, one should pass in an input of `[1, n, n]` rather than `[2, n, n]` as before. 
  - **We no longer support Encoder/Decoder models**. PyTorch tracing for `pipelining` no longer supports encoder/decoder models, so the `t5` example has been removed. 
  - **Computer vision model support currently does not work**: There are some tracing issues regarding resnet's we are actively looking into.
- **If either of these changes are too breaking, we recommend pinning your accelerate version**. If the encoder/decoder model support is **actively blocking your inference using pippy**, please open an issue and let us know. We can look towards adding in the old support for `torchpippy` potentially if needed.

## Fully Sharded Data Parallelism (FSDP)
- **Environment Flexibility:** Environment variables are now **fully optional** for FSDP, simplifying configuration. You can now fully create a `FullyShardedDataParallelPlugin` yourself manually *with no need for environment patching*:
```python
from accelerate import FullyShardedDataParallelPlugin
fsdp_plugin = FullyShardedDataParallelPlugin(...)
```
- **FSDP RAM efficient loading:** Added a utility to enable RAM-efficient model loading (by setting the proper environmental variable). **This is generally needed if not using `accelerate launch` and need to ensure the env variables are setup properly for model loading**:
```python
from accelerate.utils import enable_fsdp_ram_efficient_loading, disable_fsdp_ram_efficient_loading
enable_fsdp_ram_efficient_loading()
```
- **Model State Dict Management:** Enhanced support for unwrapping model state dicts in FSDP, making it easier to manage distributed models.

## New Examples
- **Configuration and Models:** Improved configuration handling and introduced a configuration zoo for easier experimentation. You can learn more [here](https://github.com/huggingface/accelerate/tree/main/examples/config_yaml_templates). This was largely inspired by the `axolotl` library, so very big kudos to their wonderful [work](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples)
- **FSDP + SLURM Example:** Added a [minimal configuration example for running jobs with SLURM and using FSDP](https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode_fsdp.sh)

## Bug Fixes
* Fix bug of clip_grad_norm_ for xla fsdp by @hanwen-sun in https://github.com/huggingface/accelerate/pull/2941
* Explicit check for `step` when loading the state by @muellerzr in https://github.com/huggingface/accelerate/pull/2992
* Fix `find_tied_params` for models with shared layers by @qubvel in https://github.com/huggingface/accelerate/pull/2986
* clear memory after offload by @SunMarc in https://github.com/huggingface/accelerate/pull/2994
* fix default value for rank size in cpu threads_per_process assignment logic by @rbrugaro in https://github.com/huggingface/accelerate/pull/3009
* Fix batch_sampler maybe None error by @candlewill in https://github.com/huggingface/accelerate/pull/3025
* Do not import `transformer_engine` on import by @oraluben in https://github.com/huggingface/accelerate/pull/3056
* Fix torchvision to be compatible with torch version in CI by @SunMarc in https://github.com/huggingface/accelerate/pull/2982
* Fix gated test by @muellerzr in https://github.com/huggingface/accelerate/pull/2993
* Fix typo on warning str: "on the meta device device" -> "on the meta device" by @HeAndres in https://github.com/huggingface/accelerate/pull/2997
* Fix deepspeed tests by @muellerzr in https://github.com/huggingface/accelerate/pull/3003
* Fix torch version check by @muellerzr in https://github.com/huggingface/accelerate/pull/3024
* Fix fp8 benchmark on single GPU by @muellerzr in https://github.com/huggingface/accelerate/pull/3032
* Fix typo in comment by @zmoki688 in https://github.com/huggingface/accelerate/pull/3045
* Speed up tests by shaving off subprocess when not needed by @muellerzr in https://github.com/huggingface/accelerate/pull/3042
* Remove `skip_first_batches` support for StatefulDataloader and fix all the tests by @muellerzr in https://github.com/huggingface/accelerate/pull/3068

## New Contributors
* @byi8220 made their first contribution in https://github.com/huggingface/accelerate/pull/2957
* @alex-jw-brooks made their first contribution in https://github.com/huggingface/accelerate/pull/2959
* @XciD made their first contribution in https://github.com/huggingface/accelerate/pull/2981
* @hanwen-sun made their first contribution in https://github.com/huggingface/accelerate/pull/2941
* @HeAndres made their first contribution in https://github.com/huggingface/accelerate/pull/2997
* @yitongh made their first contribution in https://github.com/huggingface/accelerate/pull/2966
* @qubvel made their first contribution in https://github.com/huggingface/accelerate/pull/2986
* @rbrugaro made their first contribution in https://github.com/huggingface/accelerate/pull/3009
* @candlewill made their first contribution in https://github.com/huggingface/accelerate/pull/3025
* @siddk made their first contribution in https://github.com/huggingface/accelerate/pull/3047
* @oraluben made their first contribution in https://github.com/huggingface/accelerate/pull/3056
* @tmm1 made their first contribution in https://github.com/huggingface/accelerate/pull/3055
* @zmoki688 made their first contribution in https://github.com/huggingface/accelerate/pull/3045

## Full Changelog:
* Require safetensors>=0.4.3 by @byi8220 in https://github.com/huggingface/accelerate/pull/2957
* Fix torchvision to be compatible with torch version in CI by @SunMarc in https://github.com/huggingface/accelerate/pull/2982
* Enable Unwrapping for Model State Dicts (FSDP) by @alex-jw-brooks in https://github.com/huggingface/accelerate/pull/2959
* chore: Update runs-on configuration for CI workflows by @XciD in https://github.com/huggingface/accelerate/pull/2981
* add MLU devices for rng state saving and loading. by @huismiling in https://github.com/huggingface/accelerate/pull/2940
* remove .md to allow proper linking by @nbroad1881 in https://github.com/huggingface/accelerate/pull/2977
* Fix bug of clip_grad_norm_ for xla fsdp by @hanwen-sun in https://github.com/huggingface/accelerate/pull/2941
* Fix gated test by @muellerzr in https://github.com/huggingface/accelerate/pull/2993
* Explicit check for `step` when loading the state by @muellerzr in https://github.com/huggingface/accelerate/pull/2992
* Fix typo on warning str: "on the meta device device" -> "on the meta device" by @HeAndres in https://github.com/huggingface/accelerate/pull/2997
* Support skip_first_batches for XLA by @yitongh in https://github.com/huggingface/accelerate/pull/2966
* clear memory after offload by @SunMarc in https://github.com/huggingface/accelerate/pull/2994
* Fix deepspeed tests by @muellerzr in https://github.com/huggingface/accelerate/pull/3003
* Make env variables optional for FSDP by @muellerzr in https://github.com/huggingface/accelerate/pull/2998
* Add small util to enable FSDP offloading quickly by @muellerzr in https://github.com/huggingface/accelerate/pull/3006
* update version to 0.34.dev0 by @SunMarc in https://github.com/huggingface/accelerate/pull/3007
* Fix `find_tied_params` for models with shared layers by @qubvel in https://github.com/huggingface/accelerate/pull/2986
* Enable FSDP & Deepspeed + FP8 by @muellerzr in https://github.com/huggingface/accelerate/pull/2983
* fix default value for rank size in cpu threads_per_process assignment logic by @rbrugaro in https://github.com/huggingface/accelerate/pull/3009
* Wrong import check for TE by @muellerzr in https://github.com/huggingface/accelerate/pull/3016
* destroy process group in `end_training` by @SunMarc in https://github.com/huggingface/accelerate/pull/3012
* Tweak defaults for quantized-typed FP8 TE weights by @muellerzr in https://github.com/huggingface/accelerate/pull/3018
* Set correct NPU backend and distributed_type when using transfer_to_npu by @ArthurinRUC in https://github.com/huggingface/accelerate/pull/3021
* Fix torch version check by @muellerzr in https://github.com/huggingface/accelerate/pull/3024
* Add end_training/destroy_pg to everything and unpin numpy by @muellerzr in https://github.com/huggingface/accelerate/pull/3030
* Improve config handling and add a zoo by @muellerzr in https://github.com/huggingface/accelerate/pull/3029
* Add early support for `torchdata.stateful_dataloader.StatefulDataLoader` within the `Accelerator` by @byi8220 in https://github.com/huggingface/accelerate/pull/2895
* Fix fp8 benchmark on single GPU by @muellerzr in https://github.com/huggingface/accelerate/pull/3032
* Fix batch_sampler maybe None error by @candlewill in https://github.com/huggingface/accelerate/pull/3025
* Fixup dataloader state dict bugs + incorporate load/save_state API by @muellerzr in https://github.com/huggingface/accelerate/pull/3034
* Decouple `prepare_data_loader()` from Accelerator by @siddk in https://github.com/huggingface/accelerate/pull/3047
* Update CONTRIBUTING.md Setup Instructions by @siddk in https://github.com/huggingface/accelerate/pull/3046
* Add a SLURM example with minimal config by @muellerzr in https://github.com/huggingface/accelerate/pull/2950
* Add FP8 docker images by @muellerzr in https://github.com/huggingface/accelerate/pull/3048
* Update torchpippy by @muellerzr in https://github.com/huggingface/accelerate/pull/2938
* Do not import `transformer_engine` on import by @oraluben in https://github.com/huggingface/accelerate/pull/3056
* use duck-typing to ensure underlying optimizer supports schedulefree hooks by @tmm1 in https://github.com/huggingface/accelerate/pull/3055
* Fix typo in comment by @zmoki688 in https://github.com/huggingface/accelerate/pull/3045
* add set_epoch for MpDeviceLoaderWrapper by @hanwen-sun in https://github.com/huggingface/accelerate/pull/3053
* Speed up tests by shaving off subprocess when not needed by @muellerzr in https://github.com/huggingface/accelerate/pull/3042
* Remove `skip_first_batches` support for StatefulDataloader and fix all the tests by @muellerzr in https://github.com/huggingface/accelerate/pull/3068


## Detailed Full Changelog: 
* https://github.com/huggingface/accelerate/compare/v0.33.0...v0.34.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.34.0)

---

## v0.33.0: v0.33.0: MUSA backend support and bugfixes
**Published:** 2024-08-08

## MUSA backend support and bugfixes

Small release this month, with key focuses on some added support for backends and bugs:
* Support MUSA (Moore Threads GPU) backend in accelerate by @fmo-mt in https://github.com/huggingface/accelerate/pull/2917
* Allow multiple process per device by @cifkao in https://github.com/huggingface/accelerate/pull/2916
* Add `torch.float8_e4m3fn` format `dtype_byte_size` by @SunMarc in https://github.com/huggingface/accelerate/pull/2945
* Properly handle Params4bit in set_module_tensor_to_device by @matthewdouglas in https://github.com/huggingface/accelerate/pull/2934

## What's Changed
* [tests] fix bug in torch_device by @faaany in https://github.com/huggingface/accelerate/pull/2909
* Fix slowdown on init with `device_map="auto"` by @muellerzr in https://github.com/huggingface/accelerate/pull/2914
* fix: bug where `multi_gpu` was being set and warning being printed even with `num_processes=1` by @HarikrishnanBalagopal in https://github.com/huggingface/accelerate/pull/2921
* Better error when a bad directory is given for weight merging by @muellerzr in https://github.com/huggingface/accelerate/pull/2852
* add xpu device check before moving tensor directly to xpu device  by @faaany in https://github.com/huggingface/accelerate/pull/2928
* Add huggingface_hub version to setup.py by @nullquant in https://github.com/huggingface/accelerate/pull/2932
* Correct loading of models with shared tensors when using accelerator.load_state() by @jkuntzer in https://github.com/huggingface/accelerate/pull/2875
* Hotfix PyTorch Version Installation in CI Workflow for Minimum Version Matrix by @yhna940 in https://github.com/huggingface/accelerate/pull/2889
* Fix import test by @muellerzr in https://github.com/huggingface/accelerate/pull/2931
* Consider pynvml available when installed through the nvidia-ml-py distribution by @matthewdouglas in https://github.com/huggingface/accelerate/pull/2936
* Improve test reliability for Accelerator.free_memory() by @matthewdouglas in https://github.com/huggingface/accelerate/pull/2935
* delete CCL env var setting by @Liangliang-Ma in https://github.com/huggingface/accelerate/pull/2927
* feat(ci): add `pip` caching in CI by @SauravMaheshkar in https://github.com/huggingface/accelerate/pull/2952

## New Contributors
* @HarikrishnanBalagopal made their first contribution in https://github.com/huggingface/accelerate/pull/2921
* @fmo-mt made their first contribution in https://github.com/huggingface/accelerate/pull/2917
* @nullquant made their first contribution in https://github.com/huggingface/accelerate/pull/2932
* @cifkao made their first contribution in https://github.com/huggingface/accelerate/pull/2916
* @jkuntzer made their first contribution in https://github.com/huggingface/accelerate/pull/2875
* @matthewdouglas made their first contribution in https://github.com/huggingface/accelerate/pull/2936
* @Liangliang-Ma made their first contribution in https://github.com/huggingface/accelerate/pull/2927
* @SauravMaheshkar made their first contribution in https://github.com/huggingface/accelerate/pull/2952

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.32.1...v0.33.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.33.0)

---

## v0.32.0: v0.32.0: Profilers, new hooks, speedups, and more!
**Published:** 2024-07-03

## Core
* Utilize shard saving from the `huggingface_hub` rather than our own implementation (https://github.com/huggingface/accelerate/pull/2795)
* Refactor logging to use logger in `dispatch_model` (https://github.com/huggingface/accelerate/pull/2855)
* The `Accelerator.step` number is now restored when using `save_state` and `load_state` (https://github.com/huggingface/accelerate/pull/2765)
* A new profiler has been added allowing users to collect performance metrics during model training and inference, including detailed analysis of execution time and memory consumption. These can then be generated in Chrome's tracing tool. Read more about it [here](https://huggingface.co/docs/accelerate/usage_guides/profiler) (https://github.com/huggingface/accelerate/pull/2883)
* Reduced import times for doing `import accelerate` and any other major core import by 68%, now should be only slightly longer than doing `import torch` (https://github.com/huggingface/accelerate/pull/2845)
* Fixed a bug in `get_backend` and added a `clear_device_cache` utility  (https://github.com/huggingface/accelerate/pull/2857)

## Distributed Data Parallelism
* Introduce DDP communication hooks to have more flexibility in how gradients are communicated across workers, overriding the standard `allreduce`. (https://github.com/huggingface/accelerate/pull/2841)
*  Make `log_line_prefix_template` optional the `notebook_launcher` (https://github.com/huggingface/accelerate/pull/2888)
## FSDP
* If the output directory doesn't exist when using `accelerate merge-weights`, one will be automatically created (https://github.com/huggingface/accelerate/pull/2854)
* When merging weights, the default is now `.safetensors` (https://github.com/huggingface/accelerate/pull/2853)

## XPU
* Migrate to pytorch's native XPU backend on `torch>=2.4` (https://github.com/huggingface/accelerate/pull/2825)
* Add `@require_triton` test decorator and enable `test_dynamo` work on xpu (https://github.com/huggingface/accelerate/pull/2878)
* Fixed `load_state_dict` not working on `xpu` and refine xpu `safetensors` version check (https://github.com/huggingface/accelerate/pull/2879)
## XLA
* Added support for XLA Dynamo backends for both training and inference (https://github.com/huggingface/accelerate/pull/2892)

## Examples
* Added a new multi-cpu SLURM example using `accelerate launch` (https://github.com/huggingface/accelerate/pull/2902)

## Full Changelog
* Use shard saving from huggingface_hub by @SunMarc in https://github.com/huggingface/accelerate/pull/2795
* doc: fix link by @imba-tjd in https://github.com/huggingface/accelerate/pull/2844
* Revert "Slight rename" by @SunMarc in https://github.com/huggingface/accelerate/pull/2850
* remove warning hook addede during dispatch_model by @SunMarc in https://github.com/huggingface/accelerate/pull/2843
* Remove underlines between badges by @novialriptide in https://github.com/huggingface/accelerate/pull/2851
* Auto create dir when merging FSDP weights by @helloworld1 in https://github.com/huggingface/accelerate/pull/2854
* Add DDP Communication Hooks by @yhna940 in https://github.com/huggingface/accelerate/pull/2841
* Refactor logging to use logger in `dispatch_model` by @panjd123 in https://github.com/huggingface/accelerate/pull/2855
* xpu: support xpu backend from stock pytorch (>=2.4) by @dvrogozh in https://github.com/huggingface/accelerate/pull/2825
* Drop torch re-imports in npu and mlu paths by @dvrogozh in https://github.com/huggingface/accelerate/pull/2856
* Default FSDP weights merge to safetensors by @helloworld1 in https://github.com/huggingface/accelerate/pull/2853
* [tests] fix bug in `test_tracking.ClearMLTest` by @faaany in https://github.com/huggingface/accelerate/pull/2863
* [tests] use `torch_device` instead of `0` for device check  by @faaany in https://github.com/huggingface/accelerate/pull/2861
* [tests] skip bnb-related tests instead of failing on xpu by @faaany in https://github.com/huggingface/accelerate/pull/2860
* Potentially fix tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2862
* [tests] enable XPU backend for `test_zero3_integration` by @faaany in https://github.com/huggingface/accelerate/pull/2864
* Support saving and loading of step while saving and loading state by @bipinKrishnan in https://github.com/huggingface/accelerate/pull/2765
* Add Profiler Support for Performance Analysis by @yhna940 in https://github.com/huggingface/accelerate/pull/2883
* Speed up imports and add a CI by @muellerzr in https://github.com/huggingface/accelerate/pull/2845
* Make `log_line_prefix_template` Optional in Elastic Launcher for Backward Compatibility by @yhna940 in https://github.com/huggingface/accelerate/pull/2888
* Add XLA Dynamo backends for training and inference by @johnsutor in https://github.com/huggingface/accelerate/pull/2892
* Added a MultiCPU SLURM example using Accelerate Launch and MPIRun by @okhleif-IL in https://github.com/huggingface/accelerate/pull/2902
* make more cuda-only tests device-agnostic  by @faaany in https://github.com/huggingface/accelerate/pull/2876
* fix mlu device longTensor bugs by @huismiling in https://github.com/huggingface/accelerate/pull/2887
* add `require_triton` and enable `test_dynamo` work on xpu by @faaany in https://github.com/huggingface/accelerate/pull/2878
* fix `load_state_dict` for xpu and refine xpu safetensor version check  by @faaany in https://github.com/huggingface/accelerate/pull/2879
* Fix get_backend bug and add clear_device_cache function by @NurmaU in https://github.com/huggingface/accelerate/pull/2857

## New Contributors
* @McPatate made their first contribution in https://github.com/huggingface/accelerate/pull/2836
* @imba-tjd made their first contribution in https://github.com/huggingface/accelerate/pull/2844
* @novialriptide made their first contribution in https://github.com/huggingface/accelerate/pull/2851
* @panjd123 made their first contribution in https://github.com/huggingface/accelerate/pull/2855
* @dvrogozh made their first contribution in https://github.com/huggingface/accelerate/pull/2825
* @johnsutor made their first contribution in https://github.com/huggingface/accelerate/pull/2892
* @okhleif-IL made their first contribution in https://github.com/huggingface/accelerate/pull/2902
* @NurmaU made their first contribution in https://github.com/huggingface/accelerate/pull/2857

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.31.0...v0.32.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.32.0)

---

## v0.31.0: v0.31.0: Better support for sharded state dict with FSDP and Bugfixes
**Published:** 2024-06-07

## Core
* Set `timeout` default to PyTorch defaults based on backend by @muellerzr in https://github.com/huggingface/accelerate/pull/2758
* fix duplicate elements in split_between_processes by @hkunzhe in https://github.com/huggingface/accelerate/pull/2781
* Add Elastic Launch Support to `notebook_launcher` by @yhna940 in https://github.com/huggingface/accelerate/pull/2788
* Fix Wrong use of sync_gradients used to implement sync_each_batch  by @fabianlim in https://github.com/huggingface/accelerate/pull/2790

## FSDP
* Introduce shard-merging util for FSDP by @muellerzr in https://github.com/huggingface/accelerate/pull/2772
* Enable sharded state dict + offload to cpu resume by @muellerzr in https://github.com/huggingface/accelerate/pull/2762
* Enable config for fsdp activation checkpointing by @helloworld1 in https://github.com/huggingface/accelerate/pull/2779

## Megatron
* Upgrade huggingface's megatron to nvidia's megatron when use MegatronLMPlugin by @zhangsheng377 in https://github.com/huggingface/accelerate/pull/2501

## What's Changed
* Add feature to allow redirecting std streams into log files when using torchrun as the launcher.  by @lyuwen in https://github.com/huggingface/accelerate/pull/2740
* Update modeling.py by adding try-catch section to skip the unavailable devices by @MeVeryHandsome in https://github.com/huggingface/accelerate/pull/2681
* Fixed the problem of incorrect conditional judgment statement when configuring enable_cpu_affinity by @statelesshz in https://github.com/huggingface/accelerate/pull/2748
* Fix stacklevel in `logging` to log the actual user call site (instead of the call site inside the logger wrapper) of log functions by @luowyang in https://github.com/huggingface/accelerate/pull/2730
* LOMO / FIX: Support multiple optimizers by @younesbelkada in https://github.com/huggingface/accelerate/pull/2745
* Fix max_memory assignment by @SunMarc in https://github.com/huggingface/accelerate/pull/2751
* Fix duplicate environment variable check in multi-cpu condition by @yhna940 in https://github.com/huggingface/accelerate/pull/2752
* Simplify CLI args validation and ensure CLI args take precedence over config file. by @Iain-S in https://github.com/huggingface/accelerate/pull/2757
* Fix sagemaker config by @muellerzr in https://github.com/huggingface/accelerate/pull/2753
* fix cpu omp num threads set by @jiqing-feng in https://github.com/huggingface/accelerate/pull/2755
* Revert "Simplify CLI args validation and ensure CLI args take precedence over config file." by @muellerzr in https://github.com/huggingface/accelerate/pull/2763
* Enable sharded cpu resume by @muellerzr in https://github.com/huggingface/accelerate/pull/2762
* Sets default to PyTorch defaults based on backend by @muellerzr in https://github.com/huggingface/accelerate/pull/2758
* optimize get_module_leaves speed by @BBuf in https://github.com/huggingface/accelerate/pull/2756
* fix minor typo by @TemryL in https://github.com/huggingface/accelerate/pull/2767
* Fix small edge case in get_module_leaves by @SunMarc in https://github.com/huggingface/accelerate/pull/2774
* Skip deepspeed test by @SunMarc in https://github.com/huggingface/accelerate/pull/2776
* Enable config for fsdp activation checkpointing by @helloworld1 in https://github.com/huggingface/accelerate/pull/2779
* Add arg from CLI to fix failing test by @muellerzr in https://github.com/huggingface/accelerate/pull/2783
* Skip tied weights disk offload test by @SunMarc in https://github.com/huggingface/accelerate/pull/2782
* Introduce shard-merging util for FSDP by @muellerzr in https://github.com/huggingface/accelerate/pull/2772
* FIX / FSDP : Guard fsdp utils for earlier PyTorch versions by @younesbelkada in https://github.com/huggingface/accelerate/pull/2794
* Upgrade huggingface's megatron to nvidia's megatron when use MegatronLMPlugin by @zhangsheng377 in https://github.com/huggingface/accelerate/pull/2501
* Fixup CLI test by @muellerzr in https://github.com/huggingface/accelerate/pull/2796
* fix duplicate elements in split_between_processes by @hkunzhe in https://github.com/huggingface/accelerate/pull/2781
* Add Elastic Launch Support to `notebook_launcher` by @yhna940 in https://github.com/huggingface/accelerate/pull/2788
* Fix Wrong use of sync_gradients used to implement sync_each_batch  by @fabianlim in https://github.com/huggingface/accelerate/pull/2790
* Fix type in accelerator.py by @qgallouedec in https://github.com/huggingface/accelerate/pull/2800
* fix comet ml test by @SunMarc in https://github.com/huggingface/accelerate/pull/2804
* New template by @muellerzr in https://github.com/huggingface/accelerate/pull/2808
* Fix access error for torch.mps when using torch==1.13.1 on macOS by @SunMarc in https://github.com/huggingface/accelerate/pull/2806
* 4-bit quantization meta device bias loading bug by @SunMarc in https://github.com/huggingface/accelerate/pull/2805
* State dictionary retrieval from offloaded modules by @blbadger in https://github.com/huggingface/accelerate/pull/2619
* add cuda dep for a test by @SunMarc in https://github.com/huggingface/accelerate/pull/2820
* Remove out-dated xpu device check code in `get_balanced_memory` by @faaany in https://github.com/huggingface/accelerate/pull/2826
* Fix DeepSpeed config validation error by changing `stage3_prefetch_bucket_size` value to an integer  by @adk9 in https://github.com/huggingface/accelerate/pull/2814
* Improve test speeds by up to 30% in multi-gpu settings by @muellerzr in https://github.com/huggingface/accelerate/pull/2830
* monitor-interval, take 2 by @muellerzr in https://github.com/huggingface/accelerate/pull/2833
* Optimize the megatron plugin by @zhangsheng377 in https://github.com/huggingface/accelerate/pull/2822
* fix fstr format by @Jintao-Huang in https://github.com/huggingface/accelerate/pull/2810

## New Contributors
* @lyuwen made their first contribution in https://github.com/huggingface/accelerate/pull/2740
* @MeVeryHandsome made their first contribution in https://github.com/huggingface/accelerate/pull/2681
* @luowyang made their first contribution in https://github.com/huggingface/accelerate/pull/2730
* @Iain-S made their first contribution in https://github.com/huggingface/accelerate/pull/2757
* @BBuf made their first contribution in https://github.com/huggingface/accelerate/pull/2756
* @TemryL made their first contribution in https://github.com/huggingface/accelerate/pull/2767
* @helloworld1 made their first contribution in https://github.com/huggingface/accelerate/pull/2779
* @hkunzhe made their first contribution in https://github.com/huggingface/accelerate/pull/2781
* @adk9 made their first contribution in https://github.com/huggingface/accelerate/pull/2814
* @Jintao-Huang made their first contribution in https://github.com/huggingface/accelerate/pull/2810

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.30.1...v0.31.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.31.0)

---

## v0.30.1: v0.30.1: Bugfixes
**Published:** 2024-05-10

## Patchfix
* Fix duplicate environment variable check in multi-cpu condition thanks to @yhna940 in https://github.com/huggingface/accelerate/pull/2752
* Fix issue with missing values in the SageMaker config leading to not being able to launch in https://github.com/huggingface/accelerate/pull/2753
* Fix CPU OMP num threads setting thanks to @jiqing-feng in https://github.com/huggingface/accelerate/pull/2755
* Fix FSDP checkpoint unable to resume when using offloading and sharded weights due to CUDA OOM when loading the optimizer and model https://github.com/huggingface/accelerate/pull/2762
* Fixed the problem of incorrect conditional judgment statement when configuring enable_cpu_affinity thanks to @statelesshz in https://github.com/huggingface/accelerate/pull/2748
* Fix stacklevel in logging to log the actual user call site (instead of the call site inside the logger wrapper) of log functions thanks to @luowyang in https://github.com/huggingface/accelerate/pull/2730
* Fix support for multiple optimizers when using LOMO thanks to @younesbelkada in https://github.com/huggingface/accelerate/pull/2745

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.30.0...v0.30.1

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.30.1)

---

## v0.30.0: v0.30.0: Advanced optimizer support, MoE DeepSpeed support, add upcasting for FSDP, and more
**Published:** 2024-05-03

## Core
* We've simplified the `tqdm` wrapper to make it fully passthrough, no need to have `tqdm(main_process_only, *args)`, it is now just `tqdm(*args)` and you can pass in `is_main_process` as a kwarg.
* We've added support for advanced optimizer usage:
  * Schedule free optimizer introduced by [Meta](https://github.com/facebookresearch/schedule_free/tree/main) by @muellerzr in https://github.com/huggingface/accelerate/pull/2631
  * LOMO optimizer introduced by [OpenLMLab](https://github.com/OpenLMLab/LOMO) by @younesbelkada in https://github.com/huggingface/accelerate/pull/2695
* Enable BF16 autocast to everything during FP8 and enable FSDP by @muellerzr in https://github.com/huggingface/accelerate/pull/2655
* Support dataloader send_to_device calls to use non-blocking by @drhead in https://github.com/huggingface/accelerate/pull/2685
* allow gather_for_metrics to be more flexible by @SunMarc in https://github.com/huggingface/accelerate/pull/2710
* Add `cann` version info to command accelerate env for NPU by @statelesshz in https://github.com/huggingface/accelerate/pull/2689
* Add MLU rng state setter by @ArthurinRUC in https://github.com/huggingface/accelerate/pull/2664
* device agnostic testing for hooks&utils&big_modeling by @statelesshz in https://github.com/huggingface/accelerate/pull/2602

## Documentation
* Through collaboration between @fabianlim (lead contribuitor), @stas00, @pacman100, and @muellerzr we have a new concept guide out for FSDP and DeepSpeed explicitly detailing how each interop and explaining fully and clearly how each of those work. This was a momumental effort by @fabianlim to ensure that everything can be as accurate as possible to users. I highly recommend visiting this new documentation, available [here](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed)
* New distributed inference examples have been added thanks to @SunMarc in https://github.com/huggingface/accelerate/pull/2672
* Fixed some docs for using internal trackers by @brentyi in https://github.com/huggingface/accelerate/pull/2650

## DeepSpeed
* Accelerate can now handle MoE models when using deepspeed, thanks to @pacman100 in https://github.com/huggingface/accelerate/pull/2662
* Allow "auto" for gradient clipping in YAML by @regisss in https://github.com/huggingface/accelerate/pull/2649
* Introduce a `deepspeed`-specific Docker image by @muellerzr in https://github.com/huggingface/accelerate/pull/2707. To use, pull the `gpu-deepspeed` tag `docker pull huggingface/accelerate:cuda-deepspeed-nightly`

## Megatron
* Megatron plugin can support NPU by @zhangsheng377 in https://github.com/huggingface/accelerate/pull/2667


## Big Modeling
* Add strict arg to load_checkpoint_and_dispatch by @SunMarc in https://github.com/huggingface/accelerate/pull/2641

## Bug Fixes
* Fix up state with xla + performance regression by @muellerzr in https://github.com/huggingface/accelerate/pull/2634
* Parenthesis on xpu_available by @muellerzr in https://github.com/huggingface/accelerate/pull/2639
* Fix `is_train_batch_min` type in DeepSpeedPlugin by @yhna940 in https://github.com/huggingface/accelerate/pull/2646
* Fix backend check by @jiqing-feng in https://github.com/huggingface/accelerate/pull/2652
* Fix the rng states of sampler's generator to be synchronized for correct sharding of dataset across GPUs by @pacman100 in https://github.com/huggingface/accelerate/pull/2694
* Block AMP for MPS device  by @SunMarc in https://github.com/huggingface/accelerate/pull/2699
* Fixed issue when doing multi-gpu training with bnb when the first gpu is not used by @SunMarc in https://github.com/huggingface/accelerate/pull/2714
* Fixup `free_memory` to deal with garbage collection by @muellerzr in https://github.com/huggingface/accelerate/pull/2716
* Fix sampler serialization failing by @SunMarc in https://github.com/huggingface/accelerate/pull/2723
* Fix deepspeed offload device type in the arguments to be more accurate by @yhna940 in https://github.com/huggingface/accelerate/pull/2717

## Full Changelog
* Schedule free optimizer support by @muellerzr in https://github.com/huggingface/accelerate/pull/2631
* Fix up state with xla + performance regression by @muellerzr in https://github.com/huggingface/accelerate/pull/2634
* Parenthesis on xpu_available by @muellerzr in https://github.com/huggingface/accelerate/pull/2639
* add third-party device prefix to `execution_device`   by @faaany in https://github.com/huggingface/accelerate/pull/2612
* add strict arg to load_checkpoint_and_dispatch by @SunMarc in https://github.com/huggingface/accelerate/pull/2641
* device agnostic testing for hooks&utils&big_modeling by @statelesshz in https://github.com/huggingface/accelerate/pull/2602
* Docs fix for using internal trackers by @brentyi in https://github.com/huggingface/accelerate/pull/2650
* Allow "auto" for gradient clipping in YAML by @regisss in https://github.com/huggingface/accelerate/pull/2649
* Fix `is_train_batch_min` type in DeepSpeedPlugin by @yhna940 in https://github.com/huggingface/accelerate/pull/2646
* Don't use deprecated `Repository` anymore by @Wauplin in https://github.com/huggingface/accelerate/pull/2658
* Fix test_from_pretrained_low_cpu_mem_usage_measured failure by @yuanwu2017 in https://github.com/huggingface/accelerate/pull/2644
* Add MLU rng state setter by @ArthurinRUC in https://github.com/huggingface/accelerate/pull/2664
* fix backend check by @jiqing-feng in https://github.com/huggingface/accelerate/pull/2652
* Megatron plugin can support NPU by @zhangsheng377 in https://github.com/huggingface/accelerate/pull/2667
* Revert "fix backend check" by @muellerzr in https://github.com/huggingface/accelerate/pull/2669
* `tqdm`: `*args` should come ahead of `main_process_only` by @rb-synth in https://github.com/huggingface/accelerate/pull/2654
* Handle MoE models with DeepSpeed by @pacman100 in https://github.com/huggingface/accelerate/pull/2662
* Fix deepspeed moe test with version check by @pacman100 in https://github.com/huggingface/accelerate/pull/2677
* Pin DS...again.. by @muellerzr in https://github.com/huggingface/accelerate/pull/2679
* fix backend check by @jiqing-feng in https://github.com/huggingface/accelerate/pull/2670
* Deprecate tqdm args + slight logic tweaks by @muellerzr in https://github.com/huggingface/accelerate/pull/2673
* Enable BF16 autocast to everything during FP8 + some tweaks to enable FSDP by @muellerzr in https://github.com/huggingface/accelerate/pull/2655
* Fix the rng states of sampler's generator to be synchronized for correct sharding of dataset across GPUs by @pacman100 in https://github.com/huggingface/accelerate/pull/2694
* Simplify test logic by @pacman100 in https://github.com/huggingface/accelerate/pull/2697
* Add source code for DataLoader Animation by @muellerzr in https://github.com/huggingface/accelerate/pull/2696
* Block AMP for MPS device  by @SunMarc in https://github.com/huggingface/accelerate/pull/2699
* Do a pip freeze during workflows by @muellerzr in https://github.com/huggingface/accelerate/pull/2704
* add cann version info to command accelerate env by @statelesshz in https://github.com/huggingface/accelerate/pull/2689
* Add version checks for the import of DeepSpeed moe utils by @pacman100 in https://github.com/huggingface/accelerate/pull/2705
* Change dataloader send_to_device calls to non-blocking by @drhead in https://github.com/huggingface/accelerate/pull/2685
* add distributed examples by @SunMarc in https://github.com/huggingface/accelerate/pull/2672
* Add diffusers to req by @muellerzr in https://github.com/huggingface/accelerate/pull/2711
* fix bnb multi gpu training by @SunMarc in https://github.com/huggingface/accelerate/pull/2714
* allow gather_for_metrics to be more flexible by @SunMarc in https://github.com/huggingface/accelerate/pull/2710
* Add Upcasting for FSDP in Mixed Precision. Add Concept Guide for FSPD and DeepSpeed.  by @fabianlim in https://github.com/huggingface/accelerate/pull/2674
* Segment out a deepspeed docker image by @muellerzr in https://github.com/huggingface/accelerate/pull/2707
* Fixup `free_memory` to deal with garbage collection by @muellerzr in https://github.com/huggingface/accelerate/pull/2716
* fix sampler serialization by @SunMarc in https://github.com/huggingface/accelerate/pull/2723
* Fix sampler failing test by @SunMarc in https://github.com/huggingface/accelerate/pull/2728
* Docs: Fix build main documentation by @SunMarc in https://github.com/huggingface/accelerate/pull/2729
* Fix Documentation in FSDP and DeepSpeed Concept Guide by @fabianlim in https://github.com/huggingface/accelerate/pull/2725
* Fix deepspeed offload device type by @yhna940 in https://github.com/huggingface/accelerate/pull/2717
* FEAT: Add LOMO optimizer by @younesbelkada in https://github.com/huggingface/accelerate/pull/2695
* Fix tests on main by @muellerzr in https://github.com/huggingface/accelerate/pull/2739

## New Contributors
* @brentyi made their first contribution in https://github.com/huggingface/accelerate/pull/2650
* @regisss made their first contribution in https://github.com/huggingface/accelerate/pull/2649
* @yhna940 made their first contribution in https://github.com/huggingface/accelerate/pull/2646
* @Wauplin made their first contribution in https://github.com/huggingface/accelerate/pull/2658
* @ArthurinRUC made their first contribution in https://github.com/huggingface/accelerate/pull/2664
* @jiqing-feng made their first contribution in https://github.com/huggingface/accelerate/pull/2652
* @zhangsheng377 made their first contribution in https://github.com/huggingface/accelerate/pull/2667
* @rb-synth made their first contribution in https://github.com/huggingface/accelerate/pull/2654
* @drhead made their first contribution in https://github.com/huggingface/accelerate/pull/2685

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.29.3...v0.30.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.30.0)

---

## v0.29.3: v0.29.3: Patchfix
**Published:** 2024-04-17

* Fixes issue with backend refactor not working on CPU-based distributed environments by @jiqing-feng: https://github.com/huggingface/accelerate/pull/2670
* Fixes issue where `load_checkpoint_and_dispatch` needs a `strict` argument
* by @SunMarc: https://github.com/huggingface/accelerate/pull/2641

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.29.2...v0.29.3

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.29.3)

---

## v0.29.2: v0.29.2: Patchfix
**Published:** 2024-04-09

* Fixes xpu missing parenthesis https://github.com/huggingface/accelerate/pull/2639
* Fixes XLA and performance degradation on init with the state https://github.com/huggingface/accelerate/pull/2634

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.29.2)

---

## v0.29.1: v0.29.1: Patchfix
**Published:** 2024-04-05

Fixed an import which would cause running accelerate CLI to fail if pytest wasn't installed

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.29.1)

---

## v0.29.0: v0.29.0: NUMA affinity control, MLU Support, and DeepSpeed Improvements
**Published:** 2024-04-05

## Core
* Accelerate can now optimize NUMA affinity, which can help increase throughput on NVIDIA multi-GPU systems. To enable it either follow the prompt during `accelerate config`, set the `ACCELERATE_CPU_AFFINITY=1` env variable, or manually using the following:
```python
from accelerate.utils import set_numa_affinity

# For GPU 0
set_numa_affinity(0)
```
Big thanks to @stas00 for the recommendation, request, and feedback during development

* Allow for setting deterministic algorithms in `set_seed` by @muellerzr in https://github.com/huggingface/accelerate/pull/2569
* Fixed the test script for TPU v2/v3 by @vanbasten23 in https://github.com/huggingface/accelerate/pull/2542
* Cambricon MLU device support introduced by @huismiling in https://github.com/huggingface/accelerate/pull/2552
* A big refactor was performed to the PartialState and AcceleratorState to allow for easier future-proofing and simplification of adding new devices by @muellerzr in https://github.com/huggingface/accelerate/pull/2576
* Fixed a reproducibility issue in distributed environments with Dataloader shuffling when using `BatchSamplerShard` by @universuen in https://github.com/huggingface/accelerate/pull/2584
* `notebook_launcher` can use multiple GPUs in Google Colab if using a custom instance that supports multiple GPUs by @StefanTodoran in https://github.com/huggingface/accelerate/pull/2561

## Big Model Inference
* Add log message for RTX 4000 series when performing multi-gpu inference with device_map which can lead to hanging by @SunMarc in https://github.com/huggingface/accelerate/pull/2557
* Fix `load_checkpoint_in_model` behavior when unexpected keys are in the checkpoint by @fxmarty in https://github.com/huggingface/accelerate/pull/2588

## DeepSpeed
* Fix issue with the mapping of `main_process_ip` and `master_addr` when not using standard as deepspeed launcher by @asdfry in https://github.com/huggingface/accelerate/pull/2495
* Improve deepspeed env gen by checking for bad keys, by @muellerzr and @ricklamers in https://github.com/huggingface/accelerate/pull/2565
* We now support custom deepspeed env files. Like normal `deepspeed`, set it with the `DS_ENV_FILE` environmental variable by @muellerzr in https://github.com/huggingface/accelerate/pull/2566
* Resolve ZeRO-3 Initialization Failure in already-started distributed environments by @sword865 in https://github.com/huggingface/accelerate/pull/2578

## What's Changed
* Fix test_script.py on TPU v2/v3 by @vanbasten23 in https://github.com/huggingface/accelerate/pull/2542
* Add mapping `main_process_ip` and `master_addr` when not using standard as deepspeed launcher by @asdfry in https://github.com/huggingface/accelerate/pull/2495
* split_between_processes for Dataset by @geronimi73 in https://github.com/huggingface/accelerate/pull/2433
* Include working driver check by @muellerzr in https://github.com/huggingface/accelerate/pull/2558
* 🚨🚨🚨Move to using tags rather than latest for docker images and consolidate image repos 🚨 🚨🚨 by @muellerzr in https://github.com/huggingface/accelerate/pull/2554
* Add Cambricon MLU accelerator support by @huismiling in https://github.com/huggingface/accelerate/pull/2552
* Add NUMA affinity control for NVIDIA GPUs by @muellerzr in https://github.com/huggingface/accelerate/pull/2535
* Add log message for RTX 4000 series when performing multi-gpu inference with device_map by @SunMarc in https://github.com/huggingface/accelerate/pull/2557
* Improve deepspeed env gen by @muellerzr in https://github.com/huggingface/accelerate/pull/2565
* Allow for setting deterministic algorithms by @muellerzr in https://github.com/huggingface/accelerate/pull/2569
* Unpin deepspeed by @muellerzr in https://github.com/huggingface/accelerate/pull/2570
* Rm uv install by @muellerzr in https://github.com/huggingface/accelerate/pull/2577
* Allow for custom deepspeed env files by @muellerzr in https://github.com/huggingface/accelerate/pull/2566
* [docs] Missing functions from API by @stevhliu in https://github.com/huggingface/accelerate/pull/2580
* Update data_loader.py to Ensure Reproducibility in Multi-Process Environments with Dataloader Shuffle by @universuen in https://github.com/huggingface/accelerate/pull/2584
* Refactor affinity and make it stateful by @muellerzr in https://github.com/huggingface/accelerate/pull/2579
* Refactor and improve model estimator tool by @muellerzr in https://github.com/huggingface/accelerate/pull/2581
* Fix `load_checkpoint_in_model` behavior when unexpected keys are in the checkpoint by @fxmarty in https://github.com/huggingface/accelerate/pull/2588
* Guard stateful objects by @muellerzr in https://github.com/huggingface/accelerate/pull/2572
* Expound PartialState docstring by @muellerzr in https://github.com/huggingface/accelerate/pull/2589
* [docs] Fix kwarg docstring by @stevhliu in https://github.com/huggingface/accelerate/pull/2590
* Allow notebook_launcher to launch to multiple GPUs from Colab by @StefanTodoran in https://github.com/huggingface/accelerate/pull/2561
* Fix warning log for unused checkpoint keys by @fxmarty in https://github.com/huggingface/accelerate/pull/2594
* Resolve ZeRO-3 Initialization Failure in Pre-Set Torch Distributed Environments (huggingface/transformers#28803) by @sword865 in https://github.com/huggingface/accelerate/pull/2578
* Refactor PartialState and AcceleratorState by @muellerzr in https://github.com/huggingface/accelerate/pull/2576
* Allow for force unwrapping by @muellerzr in https://github.com/huggingface/accelerate/pull/2595
* Pin hub for tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2608
* Default false for trust_remote_code by @muellerzr in https://github.com/huggingface/accelerate/pull/2607
* fix llama example for pippy by @SunMarc in https://github.com/huggingface/accelerate/pull/2616
* Fix links in Quick Tour by @muellerzr in https://github.com/huggingface/accelerate/pull/2617
* Link to bash in env reporting by @muellerzr in https://github.com/huggingface/accelerate/pull/2623
* Unpin hub by @muellerzr in https://github.com/huggingface/accelerate/pull/2625

## New Contributors
* @asdfry made their first contribution in https://github.com/huggingface/accelerate/pull/2495
* @geronimi73 made their first contribution in https://github.com/huggingface/accelerate/pull/2433
* @huismiling made their first contribution in https://github.com/huggingface/accelerate/pull/2552
* @universuen made their first contribution in https://github.com/huggingface/accelerate/pull/2584
* @StefanTodoran made their first contribution in https://github.com/huggingface/accelerate/pull/2561
* @sword865 made their first contribution in https://github.com/huggingface/accelerate/pull/2578

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.28.0...v0.29.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.29.0)

---

## v0.28.0: v0.28.0: DataLoaderConfig, XLA improvements, FSDP + QLORA foundations, Gradient Synchronization Tweaks, and Bug Fixes
**Published:** 2024-03-12

## Core
* Introduce a `DataLoaderConfiguration` and begin deprecation of arguments in the `Accelerator`
```diff
+from accelerate import DataLoaderConfiguration
+dl_config = DataLoaderConfiguration(split_batches=True, dispatch_batches=True)
-accelerator = Accelerator(split_batches=True, dispatch_batches=True)
+accelerator = Accelerator(dataloader_config=dl_config)
```
* Allow gradients to be synced each data batch while performing gradient accumulation, useful when training in FSDP by @fabianlim in https://github.com/huggingface/accelerate/pull/2531
```diff
from accelerate import GradientAccumulationPlugin
plugin = GradientAccumulationPlugin(
+    num_steps=2, 
    sync_each_batch=sync_each_batch
)
accelerator = Accelerator(gradient_accumulation_plugin=plugin)
```


## Torch XLA
* Support for XLA on the GPU by @anw90 in https://github.com/huggingface/accelerate/pull/2176
* Enable gradient accumulation on TPU in https://github.com/huggingface/accelerate/pull/2453

## FSDP
* Support downstream FSDP + QLORA support through tweaks by allowing configuration of buffer precision by @pacman100 in https://github.com/huggingface/accelerate/pull/2544

## `launch` changes
* Support `mpirun` for multi-cpu training by @dmsuehir in https://github.com/huggingface/accelerate/pull/2493

## What's Changed
* Fix model metadata issue check by @muellerzr in https://github.com/huggingface/accelerate/pull/2435
* Use py 3.9 by @muellerzr in https://github.com/huggingface/accelerate/pull/2436
* Fix seedable sampler logic and expound docs by @muellerzr in https://github.com/huggingface/accelerate/pull/2434
* Fix tied_pointers_to_remove type by @fxmarty in https://github.com/huggingface/accelerate/pull/2439
* Make test assertions more idiomatic by @akx in https://github.com/huggingface/accelerate/pull/2420
* Prefer `is_torch_tensor` over `hasattr` for torch.compile. by @PhilJd in https://github.com/huggingface/accelerate/pull/2387
* Enable more Ruff lints & fix issues by @akx in https://github.com/huggingface/accelerate/pull/2419
* Fix warning when dispatching model  by @SunMarc in https://github.com/huggingface/accelerate/pull/2442
* Make torch xla available on GPU by @anw90 in https://github.com/huggingface/accelerate/pull/2176
* Include pippy_file_path by @muellerzr in https://github.com/huggingface/accelerate/pull/2444
* [Big deprecation] Introduces a `DataLoaderConfig` by @muellerzr in https://github.com/huggingface/accelerate/pull/2441
* Check for None by @muellerzr in https://github.com/huggingface/accelerate/pull/2452
* Fix the pytest version to be less than 8.0.1 by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/2461
* Fix wrong `is_namedtuple` implementation by @fxmarty in https://github.com/huggingface/accelerate/pull/2475
* Use grad-accum on TPU by @muellerzr in https://github.com/huggingface/accelerate/pull/2453
* Add pre-commit configuration by @akx in https://github.com/huggingface/accelerate/pull/2451
* Replace `os.path.sep.join` path manipulations with a helper by @akx in https://github.com/huggingface/accelerate/pull/2446
* DOC: Fixes to Accelerator docstring by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/2443
* Context manager fixes by @akx in https://github.com/huggingface/accelerate/pull/2450
* Fix TPU with new `XLA` device type by @will-cromar in https://github.com/huggingface/accelerate/pull/2467
* Free mps memory by @SunMarc in https://github.com/huggingface/accelerate/pull/2483
* [FIX] allow `Accelerator` to detect distributed type from the "LOCAL_RANK" env variable for XPU  by @faaany in https://github.com/huggingface/accelerate/pull/2473
* Fix CI tests due to pathlib issues by @muellerzr in https://github.com/huggingface/accelerate/pull/2491
* Remove all cases of torchrun in tests and centralize as `accelerate launch` by @muellerzr in https://github.com/huggingface/accelerate/pull/2498
* Fix link typo by @SunMarc in https://github.com/huggingface/accelerate/pull/2503
* [docs] Accelerator API by @stevhliu in https://github.com/huggingface/accelerate/pull/2465
* Docstring fixup by @muellerzr in https://github.com/huggingface/accelerate/pull/2504
* [docs] Divide training and inference by @stevhliu in https://github.com/huggingface/accelerate/pull/2466
* add custom dtype INT2 by @SunMarc in https://github.com/huggingface/accelerate/pull/2505
* quanto compatibility for cpu/disk offload by @SunMarc in https://github.com/huggingface/accelerate/pull/2481
* [docs] Quicktour by @stevhliu in https://github.com/huggingface/accelerate/pull/2456
* Check if hub down by @muellerzr in https://github.com/huggingface/accelerate/pull/2506
* Remove offline stuff by @muellerzr in https://github.com/huggingface/accelerate/pull/2509
* Fixed 0MiB bug in convert_file_size_to_int by @StoyanStAtanasov in https://github.com/huggingface/accelerate/pull/2507
* Fix edge case in infer_auto_device_map when dealing with buffers by @SunMarc in https://github.com/huggingface/accelerate/pull/2511
* [docs] Fix typos by @omahs in https://github.com/huggingface/accelerate/pull/2490
* fix typo in launch.py (`----main_process_port` to `--main_process_port`) by @DerrickWang005 in https://github.com/huggingface/accelerate/pull/2516
* Add copyright + some ruff lint things by @muellerzr in https://github.com/huggingface/accelerate/pull/2523
* Don't manage `PYTORCH_NVML_BASED_CUDA_CHECK` when calling `accelerate.utils.imports.is_cuda_available()` by @luiscape in https://github.com/huggingface/accelerate/pull/2524
* Quanto compatibility with QBitsTensor by @SunMarc in https://github.com/huggingface/accelerate/pull/2526
* Remove unnecessary `env=os.environ.copy()`s by @akx in https://github.com/huggingface/accelerate/pull/2449
* Launch mpirun from accelerate launch for multi-CPU training by @dmsuehir in https://github.com/huggingface/accelerate/pull/2493
* Enable using dash or underscore for CLI args by @muellerzr in https://github.com/huggingface/accelerate/pull/2527
* Update the default behavior of `zero_grad(set_to_none=None)` to align with PyTorch by @yongchanghao in https://github.com/huggingface/accelerate/pull/2472
* Update link to dynamo/compile doc by @WarmongeringBeaver in https://github.com/huggingface/accelerate/pull/2533
* Check if the buffers fit GPU memory after device map auto inferred by @notsyncing in https://github.com/huggingface/accelerate/pull/2412
* [Refactor] Refactor send_to_device to treat tensor-like first by @vmoens in https://github.com/huggingface/accelerate/pull/2438
* Overdue email change... by @muellerzr in https://github.com/huggingface/accelerate/pull/2534
* [docs] Troubleshoot by @stevhliu in https://github.com/huggingface/accelerate/pull/2538
* Remove extra double-dash in error message by @drscotthawley in https://github.com/huggingface/accelerate/pull/2541
* Allow Gradients to be Synced Each Data Batch While Performing Gradient Accumulation by @fabianlim in https://github.com/huggingface/accelerate/pull/2531
* Update FSDP mixed precision setter to enable fsdp+qlora by @pacman100 in https://github.com/huggingface/accelerate/pull/2544
* Use uv instead of pip install for github CI by @muellerzr in https://github.com/huggingface/accelerate/pull/2546

## New Contributors
* @anw90 made their first contribution in https://github.com/huggingface/accelerate/pull/2176
* @StoyanStAtanasov made their first contribution in https://github.com/huggingface/accelerate/pull/2507
* @omahs made their first contribution in https://github.com/huggingface/accelerate/pull/2490
* @DerrickWang005 made their first contribution in https://github.com/huggingface/accelerate/pull/2516
* @luiscape made their first contribution in https://github.com/huggingface/accelerate/pull/2524
* @dmsuehir made their first contribution in https://github.com/huggingface/accelerate/pull/2493
* @yongchanghao made their first contribution in https://github.com/huggingface/accelerate/pull/2472
* @WarmongeringBeaver made their first contribution in https://github.com/huggingface/accelerate/pull/2533
* @vmoens made their first contribution in https://github.com/huggingface/accelerate/pull/2438
* @drscotthawley made their first contribution in https://github.com/huggingface/accelerate/pull/2541
* @fabianlim made their first contribution in https://github.com/huggingface/accelerate/pull/2531

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.27.2...v0.28.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.28.0)

---

## v0.27.0: v0.27.0: PyTorch 2.2.0 Support, PyTorch-Native Pipeline Parallism, DeepSpeed XPU support, and Bug Fixes
**Published:** 2024-02-09

## PyTorch 2.2.0 Support

With the latest release of PyTorch 2.2.0, we've guaranteed that there are no breaking changes regarding it

## PyTorch-Native Pipeline Parallel Inference

With this release we are excited to announce support for pipeline-parallel inference by integrating PyTorch's [PiPPy](https://github.com/pytorch/PiPPy) framework (so no need to use Megatron or DeepSpeed)! This supports automatic model-weight splitting to each device using a similar API to `device_map="auto"`. This is still under heavy development, however the inference side is stable enough that we are ready for a release. Read more about it in [our docs](https://huggingface.co/docs/accelerate/usage_guides/distributed_inference#memory-efficient-pipeline-parallelism-experimental) and check out the [example zoo](https://github.com/huggingface/accelerate/tree/main/examples/inference).

Requires `pippy` of version 0.2.0 or later (`pip install torchpippy -U`)

Example usage (combined with `accelerate launch` or `torchrun`):

```python
from accelerate import PartialState, prepare_pippy
model = AutoModelForSequenceClassification.from_pretrained("gpt2")
model = prepare_pippy(model, split_points="auto", example_args=(input,))
input = input.to("cuda:0")
with torch.no_grad():
    output = model(input)
# The outputs are only on the final process by default
# You can pass in `gather_outputs=True` to prepare_pippy to
# make them available on all processes
if PartialState().is_last_process:
    output = torch.stack(tuple(output[0]))
    print(output.shape)
```

## DeepSpeed

This release provides support for utilizing DeepSpeed on XPU devices thanks to @faaany 

## What's Changed
* Convert model.hf_device_map back to Dict by @SunMarc in https://github.com/huggingface/accelerate/pull/2326
* Fix model memory issue by @muellerzr in https://github.com/huggingface/accelerate/pull/2327
* Fixed typos in readme files of docs folder. by @rishit5 in https://github.com/huggingface/accelerate/pull/2329
* Disable P2P in *just* the 4000 series by @muellerzr in https://github.com/huggingface/accelerate/pull/2332
* Avoid duplicating memory for tied weights in `dispatch_model`, and in forward with offloading  by @fxmarty in https://github.com/huggingface/accelerate/pull/2330
* Show DeepSpeed option when multi-XPU is selected in `accelerate config` by @faaany in https://github.com/huggingface/accelerate/pull/2346
* FIX: add oneCCL environment variable for non-MPI launcher (accelerate launch) by @faaany in https://github.com/huggingface/accelerate/pull/2339
* device agnostic test_accelerator/test_multigpu by @wangshuai09 in https://github.com/huggingface/accelerate/pull/2343
* Fix mpi4py/failing deepspeed test issues by @muellerzr in https://github.com/huggingface/accelerate/pull/2353
* Fix `block_size` picking in `megatron_lm_gpt_pretraining` example. by @nilq in https://github.com/huggingface/accelerate/pull/2342
* Fix dispatch_model with tied weights test on T4 by @fxmarty in https://github.com/huggingface/accelerate/pull/2354
* bugfix to allow usage of TE or MSAMP in `FP8RecipeKwargs` by @sudhakarsingh27 in https://github.com/huggingface/accelerate/pull/2355
* Pin DeepSpeed until patch by @muellerzr in https://github.com/huggingface/accelerate/pull/2366
* Remove init_hook_kwargs by @fxmarty in https://github.com/huggingface/accelerate/pull/2365
* device agnostic optimizer testing by @statelesshz in https://github.com/huggingface/accelerate/pull/2363
* `add_hook_to_module` and `remove_hook_from_module` compatibility with fx.GraphModule by @fxmarty in https://github.com/huggingface/accelerate/pull/2369
* Adding `requires_grad` to `kwargs` when registering empty parameters. by @BlackSamorez in https://github.com/huggingface/accelerate/pull/2376
* Add `adapter_only` option to `save_fsdp_model` and `load_fsdp_model` to only save/load PEFT weights by @AjayP13 in https://github.com/huggingface/accelerate/pull/2321
* device agnostic cli/data_loader/grad_sync/kwargs_handlers/memory_utils testing by @wangshuai09 in https://github.com/huggingface/accelerate/pull/2356
* Fix batch_size sanity check logic for `split_batches ` by @izhx in https://github.com/huggingface/accelerate/pull/2344
* Pin Torch version to <2.2.0 by @Rocketknight1 in https://github.com/huggingface/accelerate/pull/2394
* Address PIP-632 deprecation of distutils by @AieatAssam in https://github.com/huggingface/accelerate/pull/2388
* [don't merge yet] unpin torch by @ydshieh in https://github.com/huggingface/accelerate/pull/2406
* Revert "[don't merge yet] unpin torch" by @muellerzr in https://github.com/huggingface/accelerate/pull/2407
* Fix CI due to pytest by @muellerzr in https://github.com/huggingface/accelerate/pull/2408
* Added activateEnviroment.sh to readme by @TJ-Solergibert in https://github.com/huggingface/accelerate/pull/2409
* Fix XPU inference by @notsyncing in https://github.com/huggingface/accelerate/pull/2383
* Fix the size of int and bool type when computing module size by @notsyncing in https://github.com/huggingface/accelerate/pull/2411
* Adding Local SGD support for NPU by @statelesshz in https://github.com/huggingface/accelerate/pull/2415
* Unpin torch by @muellerzr in https://github.com/huggingface/accelerate/pull/2418
* Use Ruff for formatting too by @akx in https://github.com/huggingface/accelerate/pull/2400
* torch-native pipeline parallelism for big models by @muellerzr in https://github.com/huggingface/accelerate/pull/2345
* Update FSDP docs by @pacman100 in https://github.com/huggingface/accelerate/pull/2430
* Make output end up on all GPUs at the end by @muellerzr in https://github.com/huggingface/accelerate/pull/2423
* Migrate pippy examples over and run tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2424
* [FIX] fix the wrong `nproc_per_node` in the multi gpu test   by @faaany in https://github.com/huggingface/accelerate/pull/2422
* Fix fp8 things by @muellerzr in https://github.com/huggingface/accelerate/pull/2403
* [FIX] allow `Accelerator` to prepare models in eval mode for XPU&CPU  by @faaany in https://github.com/huggingface/accelerate/pull/2426
* [Fix] make all tests pass on XPU   by @faaany in https://github.com/huggingface/accelerate/pull/2427

## New Contributors
* @rishit5 made their first contribution in https://github.com/huggingface/accelerate/pull/2329
* @faaany made their first contribution in https://github.com/huggingface/accelerate/pull/2346
* @wangshuai09 made their first contribution in https://github.com/huggingface/accelerate/pull/2343
* @nilq made their first contribution in https://github.com/huggingface/accelerate/pull/2342
* @BlackSamorez made their first contribution in https://github.com/huggingface/accelerate/pull/2376
* @AjayP13 made their first contribution in https://github.com/huggingface/accelerate/pull/2321
* @Rocketknight1 made their first contribution in https://github.com/huggingface/accelerate/pull/2394
* @AieatAssam made their first contribution in https://github.com/huggingface/accelerate/pull/2388
* @ydshieh made their first contribution in https://github.com/huggingface/accelerate/pull/2406
* @notsyncing made their first contribution in https://github.com/huggingface/accelerate/pull/2383
* @akx made their first contribution in https://github.com/huggingface/accelerate/pull/2400

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.26.1...v0.27.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.27.0)

---

## v0.26.1: v0.26.1: Patch Release
**Published:** 2024-01-11

## What's Changed
* Raise error when using batches of different sizes with `dispatch_batches=True` by @SunMarc in https://github.com/huggingface/accelerate/pull/2325


**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.26.0...v0.26.1

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.26.1)

---

## v0.26.0: v0.26.0 - MS-AMP Support, Critical Regression Fixes, and More
**Published:** 2024-01-11

## Support for MS-AMP

This release adds support for the [MS-AMP](https://github.com/Azure/MS-AMP) (Microsoft Automatic Mixed Precision Library) into Accelerate as an alternative backend for doing FP8 training on appropriate hardware. It is the default backend of choice. Read more in the docs [here](https://huggingface.co/docs/accelerate/concept_guides/low_precision_training). Introduced in https://github.com/huggingface/accelerate/pull/2232 by @muellerzr

## Core

In the prior release a new sampler for the `DataLoader` was introduced that while across seeds does not show statistical differences in the results, repeating the same seed would result in a different end-accuracy that was scary to some users. We have now disabled this behavior by default as it required some additional setup, and brought back the original implementation. To have the new sampling technique (which can provide more accurate repeated results) pass `use_seedable_sampler=True` to the `Accelerator`. We will be propagating this up to the `Trainer` soon.

## Big Model Inference

* NPU support was added thanks to @statelesshz in https://github.com/huggingface/accelerate/pull/2222
* When generating an automatic `device_map` we've made it possible to not returned grouped key results if desired in https://github.com/huggingface/accelerate/pull/2233
* We now handle corner cases better when users pass `device_map="cuda"` etc thanks to @younesbelkada in https://github.com/huggingface/accelerate/pull/2254

## FSDP and DeepSpeed

* Many improvements to the docs have been made thanks to @stass. Along with this we've made it easier to adjust the config for the sharding strategy and other config values thanks to @pacman100 in https://github.com/huggingface/accelerate/pull/2288

* A regression in Accelerate 0.23.0 occurred that showed learning is much slower on multi-GPU setups compared to a single GPU. https://github.com/huggingface/accelerate/pull/2304 has now fixed this thanks to @pacman100

* The DeepSpeed integration now also handles `auto` values better when making a configuration in https://github.com/huggingface/accelerate/pull/2313

## Bits and Bytes
* `Params4bit` added to bnb classes in set_module_tensor_to_device() by @poedator in https://github.com/huggingface/accelerate/pull/2315

## Device Agnostic Testing

For developers, we've made it much easier to run the *tests* on different devices with no change to the code thanks to @statelesshz in https://github.com/huggingface/accelerate/pull/2123 and https://github.com/huggingface/accelerate/pull/2235

## Bug Fixes
* Check notebook launcher for 3090+ by @muellerzr in https://github.com/huggingface/accelerate/pull/2212
* Fix dtype bug when `offload_state_dict=True` and `dtype` is specified by @fxmarty in https://github.com/huggingface/accelerate/pull/2116
* fix tqdm wrapper to print when process id ==0 by @kashif in https://github.com/huggingface/accelerate/pull/2223
* fix BFloat16 is not supported on MPS (#2226) by @jxysoft in https://github.com/huggingface/accelerate/pull/2227
* Fix MpDeviceLoaderWrapper not having attribute batch_sampler by @vanbasten23 in https://github.com/huggingface/accelerate/pull/2242
* [deepspeed] fix setting `auto` values for comm buffers by @stas00 in https://github.com/huggingface/accelerate/pull/2295
* Fix infer_auto_device_map when tied weights share the same prefix name by @fxmarty in https://github.com/huggingface/accelerate/pull/2324
* Fixes bug in swapping weights when replacing with Transformer-Engine layers by @sudhakarsingh27 in https://github.com/huggingface/accelerate/pull/2305
* Fix breakpoint API in test_script.py on TPU. by @vanbasten23 in https://github.com/huggingface/accelerate/pull/2263
* Bring old seed technique back by @muellerzr in https://github.com/huggingface/accelerate/pull/2319


## Major Contributors

* @statelesshz for their work on device-agnostic testing and NPU support
* @stas00 for many docfixes when it comes to DeepSpeed and FSDP

## General Changelog
* add missing whitespace by @stas00 in https://github.com/huggingface/accelerate/pull/2206
* MNT Delete the delete doc workflows by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/2217
* Update docker images by @muellerzr in https://github.com/huggingface/accelerate/pull/2213
* Add  allgather check for xpu by @abhilash1910 in https://github.com/huggingface/accelerate/pull/2199
* Check notebook launcher for 3090+ by @muellerzr in https://github.com/huggingface/accelerate/pull/2212
* Fix dtype bug when `offload_state_dict=True` and `dtype` is specified by @fxmarty in https://github.com/huggingface/accelerate/pull/2116
* fix tqdm wrapper to print when process id ==0 by @kashif in https://github.com/huggingface/accelerate/pull/2223
* [data_loader] expand the error message by @stas00 in https://github.com/huggingface/accelerate/pull/2221
* Update the 'Frameworks using Accelerate' section to include Amphion by @RMSnow in https://github.com/huggingface/accelerate/pull/2225
* [Docs] Add doc for cpu/disk offload by @SunMarc in https://github.com/huggingface/accelerate/pull/2231
* device agnostic testing by @statelesshz in https://github.com/huggingface/accelerate/pull/2123
* Make cleaning optional for device map by @muellerzr in https://github.com/huggingface/accelerate/pull/2233
* Add npu support to big model inference by @statelesshz in https://github.com/huggingface/accelerate/pull/2222
* fix the DS failing test by @pacman100 in https://github.com/huggingface/accelerate/pull/2237
* Fix nb tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2230
* fix BFloat16 is not supported on MPS (#2226) by @jxysoft in https://github.com/huggingface/accelerate/pull/2227
* Fix MpDeviceLoaderWrapper not having attribute batch_sampler by @vanbasten23 in https://github.com/huggingface/accelerate/pull/2242
* [`Big-Modeling`] Harmonize device check to handle corner cases by @younesbelkada in https://github.com/huggingface/accelerate/pull/2254
* Support `log_images` for aim tracker by @Justin900429 in https://github.com/huggingface/accelerate/pull/2257
* Integrate MS-AMP Support for FP8 as a seperate backend by @muellerzr in https://github.com/huggingface/accelerate/pull/2232
* refactor deepspeed dataloader prepare logic by @pacman100 in https://github.com/huggingface/accelerate/pull/2238
* device agnostic deepspeed&fsdp testing by @statelesshz in https://github.com/huggingface/accelerate/pull/2235
* Solve CUDA issues by @muellerzr in https://github.com/huggingface/accelerate/pull/2272
* Uninstall DVC in the Trainer tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2271
* Rm DVCLive from test reqs as latest version causes failures by @muellerzr in https://github.com/huggingface/accelerate/pull/2279
* typo fix by @stas00 in https://github.com/huggingface/accelerate/pull/2276
* Add condition before using `check_tied_parameters_on_same_device` by @SunMarc in https://github.com/huggingface/accelerate/pull/2218
* [doc] FSDP improvements by @stas00 in https://github.com/huggingface/accelerate/pull/2274
* [deepspeed docs] auto-values aren't being covered by @stas00 in https://github.com/huggingface/accelerate/pull/2286
* Improve FSDP config usability by @pacman100 in https://github.com/huggingface/accelerate/pull/2288
* [doc] language fixes by @stas00 in https://github.com/huggingface/accelerate/pull/2292
* Bump tj-actions/changed-files from 22.2 to 41 in /.github/workflows by @dependabot in https://github.com/huggingface/accelerate/pull/2300
* add back dvclive to tests by @dberenbaum in https://github.com/huggingface/accelerate/pull/2280
* Fixes bug in swapping weights when replacing with Transformer-Engine layers by @sudhakarsingh27 in https://github.com/huggingface/accelerate/pull/2305
* Fix breakpoint API in test_script.py on TPU. by @vanbasten23 in https://github.com/huggingface/accelerate/pull/2263
* make test_state_checkpointing device agnostic by @statelesshz in https://github.com/huggingface/accelerate/pull/2290
* [deepspeed] documentation by @stas00 in https://github.com/huggingface/accelerate/pull/2296
* Add more missing items by @muellerzr in https://github.com/huggingface/accelerate/pull/2309
* Update docs: Add warning for device_map=None for load_checkpoint_and_dispatch by @PhilJd in https://github.com/huggingface/accelerate/pull/2308
* [deepspeed] fix setting `auto` values for comm buffers by @stas00 in https://github.com/huggingface/accelerate/pull/2295
* DeepSpeed refactoring by @pacman100 in https://github.com/huggingface/accelerate/pull/2313
* Fix DeepSpeed related regression by @pacman100 in https://github.com/huggingface/accelerate/pull/2304
* Update test_deepspeed.py by @pacman100 in https://github.com/huggingface/accelerate/pull/2323
* Bring old seed technique back by @muellerzr in https://github.com/huggingface/accelerate/pull/2319
* Fix batch_size sanity check in `prepare_data_loader` by @izhx in https://github.com/huggingface/accelerate/pull/2310
* `Params4bit` added to bnb classes in set_module_tensor_to_device() by @poedator in https://github.com/huggingface/accelerate/pull/2315
* Fix infer_auto_device_map when tied weights share the same prefix name by @fxmarty in https://github.com/huggingface/accelerate/pull/2324

## New Contributors
* @fxmarty made their first contribution in https://github.com/huggingface/accelerate/pull/2116
* @RMSnow made their first contribution in https://github.com/huggingface/accelerate/pull/2225
* @jxysoft made their first contribution in https://github.com/huggingface/accelerate/pull/2227
* @vanbasten23 made their first contribution in https://github.com/huggingface/accelerate/pull/2242
* @Justin900429 made their first contribution in https://github.com/huggingface/accelerate/pull/2257
* @dependabot made their first contribution in https://github.com/huggingface/accelerate/pull/2300
* @sudhakarsingh27 made their first contribution in https://github.com/huggingface/accelerate/pull/2305
* @PhilJd made their first contribution in https://github.com/huggingface/accelerate/pull/2308
* @izhx made their first contribution in https://github.com/huggingface/accelerate/pull/2310
* @poedator made their first contribution in https://github.com/huggingface/accelerate/pull/2315

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.25.0...v0.26.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.26.0)

---

## v0.25.0: v0.25.0: safetensors by default, new trackers, and plenty of bug fixes
**Published:** 2023-12-01

## Safetensors default

As of this release, `safetensors` will be the default format saved when applicable! To read more about safetensors and why it's best to use it for safety (and not pickle/torch.save), check it out [here](https://github.com/huggingface/safetensors)

## New Experiment Trackers

This release has two new experiment trackers, ClearML and DVCLive!

To use them, just pass `clear_ml` or `dvclive` to `log_with` in the `Accelerator` init. h/t to @eugen-ajechiloae-clearml and @dberenbaum 

## DeepSpeed

* Accelerate's DeepSpeed integration now supports NPU devices, h/t to @statelesshz
* DeepSpeed can now be launched via accelerate on single GPU setups

## FSDP

FSDP had a huge refactoring so that the interface when using FSDP is the exact same as every other scenario when using `accelerate`. No more needing to call `accelerator.prepare()` twice!

## Other useful enhancements

* We now raise and try to disable P2P communications on consumer GPUs for the 3090 series and beyond.  Without this users were seeing timeout issues and the like as NVIDIA dropped P2P support. If using `accelerate launch` we will automatically disable, and if we sense that it is still enabled on distributed setups using 3090's +, we will raise an error. 

* When doing `.gather()`, if tensors are on different devices we explicitly will raise an error (for now only valid on CUDA)

## Bug fixes

* Fixed a bug that caused dataloaders to not shuffle despite `shuffle=True` when using multiple GPUs and the new `SeedableRandomSampler`.

## General Changelog
* Add logs offloading by @SunMarc in https://github.com/huggingface/accelerate/pull/2075
* Add ClearML tracker by @eugen-ajechiloae-clearml in https://github.com/huggingface/accelerate/pull/2034
* CRITICAL: fix failing ci by @muellerzr in https://github.com/huggingface/accelerate/pull/2088
* Fix flag typo by @kuza55 in https://github.com/huggingface/accelerate/pull/2090
* Fix batch sampler by @muellerzr in https://github.com/huggingface/accelerate/pull/2097
* fixed ip address typo by @Fluder-Paradyne in https://github.com/huggingface/accelerate/pull/2099
* Fix memory leak in fp8 causing OOM (and potentially 3x vRAM usage) by @muellerzr in https://github.com/huggingface/accelerate/pull/2089
* fix warning when offload by @SunMarc in https://github.com/huggingface/accelerate/pull/2105
* Always use SeedableRandomSampler by @muellerzr in https://github.com/huggingface/accelerate/pull/2110
* Fix issue with tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2111
* Make SeedableRandomSampler the default always by @muellerzr in https://github.com/huggingface/accelerate/pull/2117
* Use "and" instead of comma in Bibtex citation by @qgallouedec in https://github.com/huggingface/accelerate/pull/2119
* Add explicit error if empty batch received by @YuryYakhno in https://github.com/huggingface/accelerate/pull/2115
* Allow for ACCELERATE_SEED env var by @muellerzr in https://github.com/huggingface/accelerate/pull/2126
* add DeepSpeed support for NPU by @statelesshz in https://github.com/huggingface/accelerate/pull/2054
* Sync states for npu fsdp by @jq460494839 in https://github.com/huggingface/accelerate/pull/2113
* Fix import error when torch>=2.0.1 and torch.distributed is disabled by @natsukium in https://github.com/huggingface/accelerate/pull/2121
* Make safetensors the default by @muellerzr in https://github.com/huggingface/accelerate/pull/2120
* Raise error when saving with param on meta device by @SunMarc in https://github.com/huggingface/accelerate/pull/2132
* Leave native `save` as `False` by @muellerzr in https://github.com/huggingface/accelerate/pull/2138
* fix retie_parameters by @SunMarc in https://github.com/huggingface/accelerate/pull/2137
* Deal with shared memory scenarios by @muellerzr in https://github.com/huggingface/accelerate/pull/2136
* specify config file path on README by @kwonmha in https://github.com/huggingface/accelerate/pull/2140
* Fix safetensors contiguous by @SunMarc in https://github.com/huggingface/accelerate/pull/2145
* Fix more tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2146
* [docs] fixed a couple of broken links by @MKhalusova in https://github.com/huggingface/accelerate/pull/2147
* [docs] troubleshooting guide by @MKhalusova in https://github.com/huggingface/accelerate/pull/2133
* [Docs] fix doc typos by @kashif in https://github.com/huggingface/accelerate/pull/2150
* Add note about GradientState being in-sync with the dataloader by default by @muellerzr in https://github.com/huggingface/accelerate/pull/2134
* Deprecated runner stuff by @muellerzr in https://github.com/huggingface/accelerate/pull/2152
* Add examples to tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2131
* Disable pypi for merge workflows + fix trainer tests by @muellerzr in https://github.com/huggingface/accelerate/pull/2153
* Adds dvclive tracker by @dberenbaum in https://github.com/huggingface/accelerate/pull/2139
* check port availability only in main deepspeed/torchrun launcher by @Jingru in https://github.com/huggingface/accelerate/pull/2078
* Do not attempt to pad nested tensors by @frankier in https://github.com/huggingface/accelerate/pull/2041
* Add warning for problematic libraries by @muellerzr in https://github.com/huggingface/accelerate/pull/2151
* Add ZeRO++ to DeepSpeed usage docs by @SumanthRH in https://github.com/huggingface/accelerate/pull/2166
* Fix Megatron-LM Arguments Bug by @yuanenming in https://github.com/huggingface/accelerate/pull/2168
* Fix non persistant buffer dispatch by @SunMarc in https://github.com/huggingface/accelerate/pull/1941
* Updated torchrun instructions by @TJ-Solergibert in https://github.com/huggingface/accelerate/pull/2096
* New CI Runners by @muellerzr in https://github.com/huggingface/accelerate/pull/2087
* Revert "New CI Runners" by @muellerzr in https://github.com/huggingface/accelerate/pull/2172
* [Working again] New CI by @muellerzr in https://github.com/huggingface/accelerate/pull/2173
* fsdp refactoring by @pacman100 in https://github.com/huggingface/accelerate/pull/2177
* Pin DVC by @muellerzr in https://github.com/huggingface/accelerate/pull/2196
* Apply DVC warning to Accelerate by @muellerzr in https://github.com/huggingface/accelerate/pull/2197
* Explicitly disable P2P using `launch`, and pick up in `state` if a user will face issues.  by @muellerzr in https://github.com/huggingface/accelerate/pull/2195
* Better error when device mismatches when calling gather() on CUDA by @muellerzr in https://github.com/huggingface/accelerate/pull/2180
* unpins dvc by @dberenbaum in https://github.com/huggingface/accelerate/pull/2200
* Assemble state dictionary for offloaded models by @blbadger in https://github.com/huggingface/accelerate/pull/2156
* Allow deepspeed without distributed launcher by @pacman100 in https://github.com/huggingface/accelerate/pull/2204

## New Contributors
* @eugen-ajechiloae-clearml made their first contribution in https://github.com/huggingface/accelerate/pull/2034
* @kuza55 made their first contribution in https://github.com/huggingface/accelerate/pull/2090
* @Fluder-Paradyne made their first contribution in https://github.com/huggingface/accelerate/pull/2099
* @YuryYakhno made their first contribution in https://github.com/huggingface/accelerate/pull/2115
* @jq460494839 made their first contribution in https://github.com/huggingface/accelerate/pull/2113
* @kwonmha made their first contribution in https://github.com/huggingface/accelerate/pull/2140
* @dberenbaum made their first contribution in https://github.com/huggingface/accelerate/pull/2139
* @Jingru made their first contribution in https://github.com/huggingface/accelerate/pull/2078
* @frankier made their first contribution in https://github.com/huggingface/accelerate/pull/2041
* @yuanenming made their first contribution in https://github.com/huggingface/accelerate/pull/2168
* @TJ-Solergibert made their first contribution in https://github.com/huggingface/accelerate/pull/2096
* @blbadger made their first contribution in https://github.com/huggingface/accelerate/pull/2156

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.24.1...v0.25.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.25.0)

---

## v0.24.1: v0.24.1: Patch Release for Samplers
**Published:** 2023-10-30

- Fixes https://github.com/huggingface/accelerate/issues/2091 by changing how checking for custom samplers is done

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.24.1)

---

## v0.24.0: v0.24.0: Improved Reproducability, Bug fixes, and other Small Improvements
**Published:** 2023-10-24

## Improved Reproducibility

One critical issue with Accelerate is training runs were different when using an iterable dataset, no matter what seeds were set. v0.24.0 introduces the `dataloader.set_epoch()` function to all `Accelerate` `DataLoaders`, where if the underlying dataset (or sampler) has the ability to set the epoch for reproducability it will do so. This is similar to the implementation already existing in transformers. To use:

```python
dataloader = accelerator.prepare(dataloader)
# Say we want to resume at epoch/iteration 2
dataloader.set_epoch(2)
``` 

For more information see this [PR](https://github.com/huggingface/accelerate/pull/2057), we will update the docs on a subsequent release with more information on this API.

## Documentation

* The quick tour docs have gotten a complete makeover thanks to @MKhalusova. Take a look [here](https://hf.co/docs/accelerate/quicktour)
* We also now have documentation on how to perform multinode training, see the [launch docs](https://hf.co/docs/accelerate/basic_tutorials/launch)

## Internal structure
* Shared file systems are now supported under `save` and `save_state` via the `ProjectConfiguration` dataclass. See #1953 for more info.
* FSDP can now be used for `bfloat16` mixed precision via `torch.autocast`
* `all_gather_into_tensor` is now used as the main gather operation, reducing memory in the cases of big tensors
* Specifying `drop_last=True` will now properly have the desired affect when performing `Accelerator().gather_for_metrics()`


## What's Changed
* Update big_modeling.md by @kli-casia in https://github.com/huggingface/accelerate/pull/1976
* Fix model copy after `dispatch_model` by @austinapatel in https://github.com/huggingface/accelerate/pull/1971
* FIX: Automatic checkpoint path inference issue by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/1989
* Fix skip first batch for deepspeed example by @SumanthRH in https://github.com/huggingface/accelerate/pull/2001
* [docs] Quick tour refactor by @MKhalusova in https://github.com/huggingface/accelerate/pull/2008
* Add basic documentation for multi node training by @SumanthRH in https://github.com/huggingface/accelerate/pull/1988
* update torch_dynamo backends by @SunMarc in https://github.com/huggingface/accelerate/pull/1992
* Sync states for xpu fsdp by @abhilash1910 in https://github.com/huggingface/accelerate/pull/2005
* update fsdp docs by @pacman100 in https://github.com/huggingface/accelerate/pull/2026
* Enable shared file system with `save` and `save_state` via ProjectConfiguration by @muellerzr in https://github.com/huggingface/accelerate/pull/1953
* Fix save on each node by @muellerzr in https://github.com/huggingface/accelerate/pull/2036
* Allow FSDP to use with `torch.autocast` for bfloat16 mixed precision by @brcps12 in https://github.com/huggingface/accelerate/pull/2033
* Fix DeepSpeed version to <0.11 by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/2043
* Unpin deepspeed by @muellerzr in https://github.com/huggingface/accelerate/pull/2044
* Reduce memory by using `all_gather_into_tensor` by @muellerzr in https://github.com/huggingface/accelerate/pull/1968
* Safely end training even if trackers weren't initialized by @Ben-Epstein in https://github.com/huggingface/accelerate/pull/1994
* Fix integration CI by @muellerzr in https://github.com/huggingface/accelerate/pull/2047
* Make fsdp ram efficient loading optional by @pacman100 in https://github.com/huggingface/accelerate/pull/2037
* Let drop_last modify `gather_for_metrics` by @muellerzr in https://github.com/huggingface/accelerate/pull/2048
* fix docstring by @zhangsibo1129 in https://github.com/huggingface/accelerate/pull/2053
* Fix stalebot by @muellerzr in https://github.com/huggingface/accelerate/pull/2052
* Add space to docs by @muellerzr in https://github.com/huggingface/accelerate/pull/2055
* Fix the error when the "train_batch_size" is absent in DeepSpeed config by @LZHgrla in https://github.com/huggingface/accelerate/pull/2060
* remove unused constants by @statelesshz in https://github.com/huggingface/accelerate/pull/2045
* fix: remove useless token by @rtrompier in https://github.com/huggingface/accelerate/pull/2069
* DOC: Fix broken link to designing a device map by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/2073
* Let iterable dataset shard have a length if implemented by @muellerzr in https://github.com/huggingface/accelerate/pull/2066
* Allow for samplers to be seedable and reproducable by @muellerzr in https://github.com/huggingface/accelerate/pull/2057
* Fix docstring typo by @qgallouedec in https://github.com/huggingface/accelerate/pull/2072
* Warn when kernel version is too low on Linux by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/2077

## New Contributors
* @kli-casia made their first contribution in https://github.com/huggingface/accelerate/pull/1976
* @MKhalusova made their first contribution in https://github.com/huggingface/accelerate/pull/2008
* @brcps12 made their first contribution in https://github.com/huggingface/accelerate/pull/2033
* @Ben-Epstein made their first contribution in https://github.com/huggingface/accelerate/pull/1994
* @zhangsibo1129 made their first contribution in https://github.com/huggingface/accelerate/pull/2053
* @LZHgrla made their first contribution in https://github.com/huggingface/accelerate/pull/2060
* @rtrompier made their first contribution in https://github.com/huggingface/accelerate/pull/2069
* @qgallouedec made their first contribution in https://github.com/huggingface/accelerate/pull/2072

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.23.0...v0.24.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.24.0)

---

## v0.23.0: v0.23.0: Model Memory Estimation tool, Breakpoint API, Multi-Node Notebook Launcher Support, and more!
**Published:** 2023-09-14

## Model Memory Estimator

A new model estimation tool to help calculate how much memory is needed for inference has been added. This does not download the pretrained weights, and utilizes `init_empty_weights` to stay memory efficient during the calculation. 

Usage directions:

```bash
accelerate estimate-memory {model_name} --library {library_name} --dtypes fp16 int8
```
Or:
```python
from accelerate.commands.estimate import estimate_command_parser, estimate_command, gather_data

parser = estimate_command_parser()
args = parser.parse_args(["bert-base-cased", "--dtypes", "float32"])
output = gather_data(args)
```

## 🤗 Hub is a first-class citizen

We've made the `huggingface_hub` library a first-class citizen of the framework! While this is mainly for the model estimation tool, this opens the doors for further integrations should they be wanted

## `Accelerator` Enhancements:

- `gather_for_metrics` will now also de-dupe for non-tensor objects. See #1937 
- `mixed_precision="bf16"` support on NPU devices. See #1949 
- New `breakpoint` API to help when dealing with trying to break from a condition on a single process. See #1940 
- 

## Notebook Launcher Enhancements:

- The notebook launcher now supports launching across multiple nodes! See #1913 

## FSDP Enhancements:

- Activation checkpointing is now natively supported in the framework. See https://github.com/huggingface/accelerate/pull/1891
- `torch.compile` support was fixed. See #1919 

## DeepSpeed Enhancements:

- XPU/ccl support (#1827)
- Easier gradient accumulation support, simply set `gradient_accumulation_steps` to `"auto"` in your deepspeed config, and Accelerate will use the one passed to `Accelerator` instead (#1901)
- Support for custom schedulers and deepspeed optimizers (#1909)

## What's Changed
* Update release instructions by @sgugger in https://github.com/huggingface/accelerate/pull/1877
* fix detach_hook by @SunMarc in https://github.com/huggingface/accelerate/pull/1880
* Enable power users to bypass device_map="auto" training block by @muellerzr in https://github.com/huggingface/accelerate/pull/1881
* Introduce model memory estimator by @muellerzr in https://github.com/huggingface/accelerate/pull/1876
* Update with new url for explore by @muellerzr in https://github.com/huggingface/accelerate/pull/1884
* Enable a token to be used by @muellerzr in https://github.com/huggingface/accelerate/pull/1886
* Add doc on model memory usage by @muellerzr in https://github.com/huggingface/accelerate/pull/1887
* Add hub as core dep by @muellerzr in https://github.com/huggingface/accelerate/pull/1885
* update import of deepspeed integration from transformers by @pacman100 in https://github.com/huggingface/accelerate/pull/1894
* Final nits on model util by @muellerzr in https://github.com/huggingface/accelerate/pull/1896
* Fix nb launcher test by @muellerzr in https://github.com/huggingface/accelerate/pull/1899
* Add FSDP activation checkpointing feature by @arde171 in https://github.com/huggingface/accelerate/pull/1891
* Solve at least one failing test by @muellerzr in https://github.com/huggingface/accelerate/pull/1898
* Deepspeed integration for XPU/ccl by @abhilash1910 in https://github.com/huggingface/accelerate/pull/1827
* Add PR template by @muellerzr in https://github.com/huggingface/accelerate/pull/1906
* deepspeed grad_acc_steps fixes by @pacman100 in https://github.com/huggingface/accelerate/pull/1901
* Skip pypi transformers until release by @muellerzr in https://github.com/huggingface/accelerate/pull/1911
* Fix docker images by @muellerzr in https://github.com/huggingface/accelerate/pull/1910
* Use hosted CI runners for building docker images by @muellerzr in https://github.com/huggingface/accelerate/pull/1915
* fix: add debug argument to sagemaker configuration by @maximegmd in https://github.com/huggingface/accelerate/pull/1904
* improve help info when run `accelerate config` on npu by @statelesshz in https://github.com/huggingface/accelerate/pull/1895
* support logging with mlflow in case of mlflow-skinny installed by @ghtaro in https://github.com/huggingface/accelerate/pull/1874
* More CI fun - run all test parts always by @muellerzr in https://github.com/huggingface/accelerate/pull/1916
* Expose auto in dataclass by @muellerzr in https://github.com/huggingface/accelerate/pull/1914
* Add support for deepspeed optimizer and custom scheduler by @pacman100 in https://github.com/huggingface/accelerate/pull/1909
* reduce gradient first for XLA when unscaling the gradients in mixed precision training with AMP. by @statelesshz in https://github.com/huggingface/accelerate/pull/1926
* Check for invalid keys by @muellerzr in https://github.com/huggingface/accelerate/pull/1935
* clean num devices by @SunMarc in https://github.com/huggingface/accelerate/pull/1936
* Bring back pypi to runners by @muellerzr in https://github.com/huggingface/accelerate/pull/1939
* Support multi-node notebook launching by @ggaaooppeenngg in https://github.com/huggingface/accelerate/pull/1913
* fix the fsdp docs by @pacman100 in https://github.com/huggingface/accelerate/pull/1947
* Fix docs by @ggaaooppeenngg in https://github.com/huggingface/accelerate/pull/1951
* Protect tensorflow dependency by @SunMarc in https://github.com/huggingface/accelerate/pull/1959
* fix safetensor saving by @SunMarc in https://github.com/huggingface/accelerate/pull/1954
* FIX: patch_environment restores pre-existing environment variables when finished by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/1960
* Better guards for slow imports by @muellerzr in https://github.com/huggingface/accelerate/pull/1963
* [`Tests`] Finish all todos by @younesbelkada in https://github.com/huggingface/accelerate/pull/1957
* Rm strtobool by @muellerzr in https://github.com/huggingface/accelerate/pull/1964
* Implementing gather_for_metrics with dedup for non tensor objects by @Lorenzobattistela in https://github.com/huggingface/accelerate/pull/1937
* add bf16 mixed precision support for NPU by @statelesshz in https://github.com/huggingface/accelerate/pull/1949
* Introduce breakpoint API by @muellerzr in https://github.com/huggingface/accelerate/pull/1940
* fix torch compile with FSDP by @pacman100 in https://github.com/huggingface/accelerate/pull/1919
* Add `force_hooks` to `dispatch_model` by @austinapatel in https://github.com/huggingface/accelerate/pull/1969
* update FSDP and DeepSpeed docs by @pacman100 in https://github.com/huggingface/accelerate/pull/1973
* Flex fix patch for accelerate by @abhilash1910 in https://github.com/huggingface/accelerate/pull/1972
* Remove checkpoints only on main process by @Kepnu4 in https://github.com/huggingface/accelerate/pull/1974

## New Contributors
* @arde171 made their first contribution in https://github.com/huggingface/accelerate/pull/1891
* @maximegmd made their first contribution in https://github.com/huggingface/accelerate/pull/1904
* @ghtaro made their first contribution in https://github.com/huggingface/accelerate/pull/1874
* @ggaaooppeenngg made their first contribution in https://github.com/huggingface/accelerate/pull/1913
* @Lorenzobattistela made their first contribution in https://github.com/huggingface/accelerate/pull/1937
* @austinapatel made their first contribution in https://github.com/huggingface/accelerate/pull/1969
* @Kepnu4 made their first contribution in https://github.com/huggingface/accelerate/pull/1974

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.22.0...v0.23.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.23.0)

---

## v0.22.0: v0.22.0: Distributed operation framework, Gradient Accumulation enhancements, FSDP enhancements, and more!
**Published:** 2023-08-23

## Experimental distributed operations checking framework

A new framework has been introduced which can help catch `timeout` errors caused by distributed operations failing *before* they occur. As this adds a tiny bit of overhead, it is an opt-in scenario. Simply run your code with `ACCELERATE_DEBUG_MODE="1"` to enable this. Read more in the [docs](https://huggingface.co/docs/accelerate/main/en/usage_guides/debug), introduced via https://github.com/huggingface/accelerate/pull/1756

## `Accelerator.load_state` can now load the most recent checkpoint automatically

If a `ProjectConfiguration` has been made, using `accelerator.load_state()` (without any arguments passed) can now automatically find and load the latest checkpoint used, introduced via https://github.com/huggingface/accelerate/pull/1741

## Multiple enhancements to gradient accumulation

In this release multiple new enhancements to distributed gradient accumulation have been added. 

* `accelerator.accumulate()` now supports passing in multiple models introduced via https://github.com/huggingface/accelerate/pull/1708
* A util has been introduced to perform multiple forwards, then multiple backwards, and finally sync the gradients only on the last `.backward()` via https://github.com/huggingface/accelerate/pull/1726

## FSDP Changes

* FSDP support has been added for NPU and XPU devices via https://github.com/huggingface/accelerate/pull/1803 and https://github.com/huggingface/accelerate/pull/1806
* A new method for supporting RAM-efficient loading of models with FSDP has been added via https://github.com/huggingface/accelerate/pull/1777

## DataLoader Changes

* Custom slice functions are now supported in the `DataLoaderDispatcher` added via https://github.com/huggingface/accelerate/pull/1846


## What's New?
* fix failing test on 8GPU by @statelesshz in https://github.com/huggingface/accelerate/pull/1724
* Better control over DDP's `no_sync` by @NouamaneTazi in https://github.com/huggingface/accelerate/pull/1726
* Get rid of calling `get_scale()` by patching the step method of optimizer. by @yuxinyuan in https://github.com/huggingface/accelerate/pull/1720
* fix the bug in npu by @statelesshz in https://github.com/huggingface/accelerate/pull/1728
* Adding a shape check for `set_module_tensor_to_device`. by @Narsil in https://github.com/huggingface/accelerate/pull/1731
* Fix errors when optimizer is not a Pytorch optimizer. by @yuxinyuan in https://github.com/huggingface/accelerate/pull/1733
* Make balanced memory able to work with non contiguous GPUs ids by @thomwolf in https://github.com/huggingface/accelerate/pull/1734
* Fixed typo in `__repr__` of AlignDevicesHook by @KacperWyrwal in https://github.com/huggingface/accelerate/pull/1735
* Update docs by @muellerzr in https://github.com/huggingface/accelerate/pull/1736
* Fixed the bug that split dict incorrectly by @yuangpeng in https://github.com/huggingface/accelerate/pull/1742
* Let load_state automatically grab the latest save by @muellerzr in https://github.com/huggingface/accelerate/pull/1741
* fix `KwargsHandler.to_kwargs` not working with `os.environ` initialization in `__post_init__` by @CyCle1024 in https://github.com/huggingface/accelerate/pull/1738
* fix typo by @cauyxy in https://github.com/huggingface/accelerate/pull/1747
* Check for misconfiguration of single node & single GPU by @muellerzr in https://github.com/huggingface/accelerate/pull/1746
* Remove unused constant by @muellerzr in https://github.com/huggingface/accelerate/pull/1749
* Rework new constant for operations by @muellerzr in https://github.com/huggingface/accelerate/pull/1748
* Expose `autocast` kwargs and simplify `autocast` wrapper by @muellerzr in https://github.com/huggingface/accelerate/pull/1740
* Fix FSDP related issues by @pacman100 in https://github.com/huggingface/accelerate/pull/1745
* FSDP enhancements and fixes by @pacman100 in https://github.com/huggingface/accelerate/pull/1753
* Fix check failure in `Accelerator.save_state` using multi-gpu by @CyCle1024 in https://github.com/huggingface/accelerate/pull/1760
* Fix error when `max_memory` argument is in unexpected order by @ranchlai in https://github.com/huggingface/accelerate/pull/1759
* Fix offload on disk when executing on CPU by @sgugger in https://github.com/huggingface/accelerate/pull/1762
* Change `is_aim_available()` function to not match aim >= 4.0.0 by @alberttorosyan in https://github.com/huggingface/accelerate/pull/1769
* Introduce an experimental distributed operations framework by @muellerzr in https://github.com/huggingface/accelerate/pull/1756
* Support wrapping multiple models in Accelerator.accumulate() by @yuxinyuan in https://github.com/huggingface/accelerate/pull/1708
* Contigous on gather by @muellerzr in https://github.com/huggingface/accelerate/pull/1771
* [FSDP] Fix `load_fsdp_optimizer` by @awgu in https://github.com/huggingface/accelerate/pull/1755
* simplify and correct the deepspeed example by @pacman100 in https://github.com/huggingface/accelerate/pull/1775
* Set ipex default in state by @muellerzr in https://github.com/huggingface/accelerate/pull/1776
* Fix import error when torch>=2.0.1 and `torch.distributed` is disabled by @natsukium in https://github.com/huggingface/accelerate/pull/1800
* reserve 10% GPU in `get_balanced_memory` to avoid OOM by @ranchlai in https://github.com/huggingface/accelerate/pull/1798
* add support of float memory size in `convert_file_size_to_int` by @ranchlai in https://github.com/huggingface/accelerate/pull/1799
* Allow users to resume from previous wandb runs with `allow_val_change`  by @SumanthRH in https://github.com/huggingface/accelerate/pull/1796
* Add FSDP for XPU by @abhilash1910 in https://github.com/huggingface/accelerate/pull/1803
* Add FSDP for NPU by @statelesshz in https://github.com/huggingface/accelerate/pull/1806
* Fix pytest import by @muellerzr in https://github.com/huggingface/accelerate/pull/1808
* More specific logging in `gather_for_metrics` by @dleve123 in https://github.com/huggingface/accelerate/pull/1784
* Detect device map auto and raise a helpful error when trying to not use model parallelism by @muellerzr in https://github.com/huggingface/accelerate/pull/1810
* Typo fix by @muellerzr in https://github.com/huggingface/accelerate/pull/1812
* Expand device-map warning by @muellerzr in https://github.com/huggingface/accelerate/pull/1819
* Update bibtex to reflect team growth by @muellerzr in https://github.com/huggingface/accelerate/pull/1820
* Improve docs on grad accumulation by @vwxyzjn in https://github.com/huggingface/accelerate/pull/1817
* add warning when using to and cuda by @SunMarc in https://github.com/huggingface/accelerate/pull/1790
* Fix bnb import by @muellerzr in https://github.com/huggingface/accelerate/pull/1813
* Update docs and docstrings to match `load_and_quantize_model` arg by @JonathanRayner in https://github.com/huggingface/accelerate/pull/1822
* Expose a bit of args/docstring fixup by @muellerzr in https://github.com/huggingface/accelerate/pull/1824
* Better test by @muellerzr in https://github.com/huggingface/accelerate/pull/1825
* Minor idiomatic change for fp8 check. by @float-trip in https://github.com/huggingface/accelerate/pull/1829
* Use device as context manager for `init_on_device` by @shingjan in https://github.com/huggingface/accelerate/pull/1826
* Ipex bug fix for device properties in modelling by @abhilash1910 in https://github.com/huggingface/accelerate/pull/1834
* FIX: Bug with `unwrap_model` and `keep_fp32_wrapper=False` by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/1838
* Fix `verify_device_map` by @Rexhaif in https://github.com/huggingface/accelerate/pull/1842
* Change CUDA check by @muellerzr in https://github.com/huggingface/accelerate/pull/1833
* Fix the noneffective parameter: `gpu_ids` (Rel. Issue #1848) by @devymex in https://github.com/huggingface/accelerate/pull/1850
* support for ram efficient loading of model with FSDP by @pacman100 in https://github.com/huggingface/accelerate/pull/1777
* Loading logic safetensors by @SunMarc in https://github.com/huggingface/accelerate/pull/1853
* fix dispatch for quantized model by @SunMarc in https://github.com/huggingface/accelerate/pull/1855
* Update `fsdp_with_peak_mem_tracking`.py by @pacman100 in https://github.com/huggingface/accelerate/pull/1856
* Add env variable for `init_on_device` by @shingjan in https://github.com/huggingface/accelerate/pull/1852
* remove casting to FP32 when saving state dict by @pacman100 in https://github.com/huggingface/accelerate/pull/1868
* support custom slice function in `DataLoaderDispatcher` by @thevasudevgupta in https://github.com/huggingface/accelerate/pull/1846
* Include a note to the forums in the bug report by @muellerzr in https://github.com/huggingface/accelerate/pull/1871

## Significant community contributions

The following contributors have made significant changes to the library over the last release:

* @yuxinyuan 
  * Support wrapping multiple models in `Accelerator.accumulate()` (#1708)
  * Fix errors when optimizer is not a Pytorch optimizer. (#1733)
  * Get rid of calling get_scale() by patching the step method of optimizer. (#1720)
* @NouamaneTazi 
  * Better control over DDP's `no_sync` (#1726)
* @abhilash1910 
  * Add FSDP for XPU (#1803)
  * Ipex bug fix for device properties in modelling (#1834)
* @statelesshz 
  * Add FSDP for NPU (#1806)
  * fix failing test on 8GPU (#1724)
  * fix the bug in npu (#1728)
* @thevasudevgupta 
  * support custom slice function in `DataLoaderDispatcher` (#1846)

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.21.0...v0.22.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.22.0)

---

## v0.21.0: v0.21.0: Model quantization and NPUs
**Published:** 2023-07-13

## Model quantization with bitsandbytes

You can now quantize any model (no just Transformer models) using Accelerate. This is mainly for models having a lot of linear layers. See the [documentation](https://huggingface.co/docs/accelerate/usage_guides/quantization) for more information!

* Bnb quantization  by @SunMarc in #1626

## Support for Ascend NPUs

Accelerate now supports Ascend NPUs.

* Add Ascend NPU accelerator support  by @statelesshz in #1676

## What's new?

Accelerate now requires Python 3.8+ and PyTorch 1.10+ :

* 🚨🚨🚨 Spring cleaning: Python 3.8 🚨🚨🚨  by @muellerzr in #1661
* 🚨🚨🚨 Spring cleaning: PyTorch 1.10 🚨🚨🚨  by @muellerzr in #1662


* [doc build] Use secrets  by @mishig25 in #1551
* Update launch.mdx  by @LiamSwayne in #1553
* Avoid double wrapping of all accelerate.prepare objects  by @muellerzr in #1555
* Update README.md  by @LiamSwayne in #1556
* Fix load_state_dict when there is one device and disk  by @sgugger in #1557
* Fix tests not being ran on multi-GPU nightly  by @muellerzr in #1558
* fix the typo when setting the "_accelerator_prepared" attribute  by @Yura52 in #1560
* [`core`] Fix possibility to pass`NoneType` objects in `prepare`  by @younesbelkada in #1561
* Reset dataloader end_of_datalaoder at each iter  by @sgugger in #1562
* Update big_modeling.mdx  by @LiamSwayne in #1564
* [`bnb`] Fix failing int8 tests  by @younesbelkada in #1567
* Update gradient sync docs to reflect importance of `optimizer.step()`  by @dleve123 in #1565
* Update mixed precision integrations in README  by @sgugger in #1569
* Raise error instead of warn  by @muellerzr in #1568
* Introduce listify, fix tensorboard silently failing  by @muellerzr in #1570
* Check for bak and expand docs on directory structure  by @muellerzr in #1571
* Perminant solution  by @muellerzr in #1577
* fix the bug in xpu  by @mingxiaoh in #1508
* Make sure that we only set is_accelerator_prepared on items accelerate actually prepares  by @muellerzr in #1578
* Expand `prepare()` doc  by @muellerzr in #1580
* Get Torch version using  importlib instead of pkg_resources  by @catwell in #1585
* improve oob performance when use mpirun to start DDP finetune without `accelerate launch`  by @sywangyi in #1575
* Update training_tpu.mdx  by @LiamSwayne in #1582
* Return false if CUDA available  by @muellerzr in #1581
* fix logger level  by @caopulan in #1579
* Fix test  by @muellerzr in #1586
* Update checkpoint.mdx  by @LiamSwayne in #1587
* FSDP updates  by @pacman100 in #1576
* Update modeling.py  by @ain-soph in #1595
* Integration tests  by @muellerzr in #1593
* Add triggers for CI workflow  by @muellerzr in #1597
* Remove asking xpu plugin for non xpu devices  by @abhilash1910 in #1594
* Remove GPU safetensors env variable  by @sgugger in #1603
* reset end_of_dataloader for dataloader_dispatcher  by @megavaz in #1609
* fix for arc gpus  by @abhilash1910 in #1615
* Ignore low_zero option when only device is available  by @sgugger in #1617
* Fix failing multinode tests  by @muellerzr in #1616
* Doc to md  by @sgugger in #1618
* Fix tb issue  by @muellerzr in #1623
* Fix workflow  by @muellerzr in #1625
* Fix transformers sync bug with accumulate  by @muellerzr in #1624
* fixes offload dtype  by @SunMarc in #1631
* fix: Megatron is not installed. please build it from source.  by @yuanwu2017 in #1636
* deepspeed z2/z1 state_dict bloating fix  by @pacman100 in #1638
* Swap disable rich  by @muellerzr in #1640
* fix autocasting bug  by @pacman100 in #1637
* fix modeling low zero  by @abhilash1910 in #1634
* Add skorch to runners  by @muellerzr in #1646
* add save model  by @SunMarc in #1641
* Change dispatch_model when we have only one device  by @SunMarc in #1648
* Doc save model  by @SunMarc in #1650
* Fix device_map  by @SunMarc in #1651
* Check for port usage before launch  by @muellerzr in #1656
* [`BigModeling`] Add missing check for quantized models  by @younesbelkada in #1652
* Bump integration  by @muellerzr in #1658
* TIL  by @muellerzr in #1657
* docker cpu py version  by @muellerzr in #1659
* [`BigModeling`] Final fix for dispatch int8 and fp4 models  by @younesbelkada in #1660
* remove safetensor dep on shard_checkpoint  by @SunMarc in #1664
* change the import place to avoid import error  by @pacman100 in #1653
* Update broken Runhouse link in examples/README.md  by @dongreenberg in #1668
* Bnb quantization  by @SunMarc in #1626
* replace save funct in doc  by @SunMarc in #1672
* Doc big model inference  by @SunMarc in #1670
* Add docs for saving Transformers models  by @deppen8 in #1671
* fix bnb tests  by @SunMarc in #1679
* Fix workflow CI  by @muellerzr in #1690
* remove duplicate class  by @SunMarc in #1691
* update readme in examples  by @statelesshz in #1678
* Fix nightly tests  by @muellerzr in #1696
* Fixup docs  by @muellerzr in #1697
* Improve quality errors  by @muellerzr in #1698
* Move mixed precision wrapping ahead of DDP/FSDP wrapping  by @ChenWu98 in #1682
* Add offload for 8-bit model  by @SunMarc in #1699
* Deepcopy on Accelerator to return self  by @muellerzr in #1694
* Update tracking.md  by @stevhliu in #1702
* Skip tests when bnb isn't available  by @muellerzr in #1706
* Fix launcher validation   by @abhilash1910 in #1705
* Fixes for issue #1683: failed to run accelerate config in colab  by @Erickrus in #1692
* Fix the bug where DataLoaderDispatcher gets stuck in an infinite wait when the dataset is an IterDataPipe during multi-process training.  by @yuxinyuan in #1709
* add multi_gpu decorator  by @SunMarc in #1712
* Modify loading checkpoint behavior  by @SunMarc in #1715
* fix version  by @SunMarc in #1701
* Keep old behavior  by @muellerzr in #1716
* Optimize `get_scale` to reduce async calls  by @muellerzr in #1718
* Remove duplicate code  by @muellerzr in #1717
* New tactic  by @muellerzr in #1719
* add Comfy-UI  by @pacman100 in #1723
* add compatibility with peft  by @SunMarc in #1725

## Significant community contributions

The following contributors have made significant changes to the library over the last release:

* @LiamSwayne
    * Update launch.mdx (#1553)
    * Update README.md (#1556)
    * Update big_modeling.mdx (#1564)
    * Update training_tpu.mdx (#1582)
    * Update checkpoint.mdx (#1587)
* @mingxiaoh
    * fix the bug in xpu (#1508)
* @statelesshz
    * update readme in examples (#1678)
    * Add Ascend NPU accelerator support (#1676)
* @ChenWu98
    * Move mixed precision wrapping ahead of DDP/FSDP wrapping (#1682)


[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.21.0)

---

## v0.20.3: v0.20.3: Patch release
**Published:** 2023-06-08

- Reset dataloader end_of_datalaoder at each iter in #1562 by @sgugger 

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.20.3)

---

## v0.20.2: v0.20.2: Patch release
**Published:** 2023-06-08

- fix the typo when setting the "_accelerator_prepared" attribute in #1560 by @Yura52 
- [core] Fix possibility to pass] `NoneType` objects in `prepare` in #1561 by @younesbelkada

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.20.2)

---

## v0.20.1: v0.20.1: Patch release
**Published:** 2023-06-07

- Avoid double wrapping of all accelerate.prepare objects by @muellerzr  in #1555
- Fix load_state_dict when there is one device and disk by @sgugger in #1557

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.20.1)

---

## v0.20.0: v0.20.0: MPS and fp4 support on Big Model Inference, 4-bit QLoRA, Intel GPU, Distributed Inference, and much more!
**Published:** 2023-06-07

## Big model inference

Support has been added to run `device_map="auto"` on the MPS device. Big model inference also work with models loaded in 4 bits in Transformers.

* Add mps support to big inference modeling  by @SunMarc in #1545
* Adds fp4 support for model dispatching by @younesbelkada in #1505

## 4-bit QLoRA Support

* 4-bit QLoRA via bitsandbytes (4-bit base model + LoRA)  by @TimDettmers in #1458

## Distributed Inference Utilities

This version introduces a new `Accelerator.split_between_processes` utility to help with performing distributed infernece with non-tensorized or non-dataloader workflows. Read more [here](https://huggingface.co/docs/accelerate/usage_guides/distributed_inference)

## Introduce XPU support for Intel GPU

* Intel GPU support initialization  by @abhilash1910 in #1118

## Add support for the new PyTorch XLA TPU runtime

* Accelerate now supports the latest TPU runtimes #1393, #1385 

## A new optimizer method: `LocalSGD`

* This is a new wrapper around SGD which enables efficient multi-GPU training in the case when no fast interconnect is possible by @searchivarius in #1378 

## Papers with 🤗 Accelerate

* We now have an entire section of the docs dedicated to official paper implementations and citations using the framework #1399, see it live [here](https://hf.co/docs/accelerate/usage_guides/training_zoo#in-science)

## Breaking changes

`logging_dir` has been fully deprecated, please use `project_dir` or a `Project_configuration`

## What's new?

* use existing mlflow experiment if exists  by @Rusteam in #1403
* changes required for DS integration  by @pacman100 in #1406
* fix deepspeed failing tests  by @pacman100 in #1411
* Make mlflow logging dir optional  by @mattplo-decath in #1413
* Fix bug on ipex for diffusers  by @abhilash1910 in #1426
* Improve Slack Updater  by @muellerzr in #1433
* Let quality yell at the user if it's a version difference  by @muellerzr in #1438
* Ensure that it gets installed  by @muellerzr in #1439
* [`core`] Introducing `CustomDtype` enum for custom dtypes  by @younesbelkada in #1434
* Fix XPU  by @muellerzr in #1440
* Make sure torch compiled model can also be unwrapped  by @patrickvonplaten in #1437
* fixed: ZeroDivisionError: division by zero  by @sreio in #1436
* fix potential OOM when resuming with multi-GPU training  by @exhyy in #1444
* Fixes in infer_auto_device_map  by @sgugger in #1441
* Raise error when logging improperly  by @muellerzr in #1446
* Fix ci  by @muellerzr in #1447
* Distributed prompting/inference utility  by @muellerzr in #1410
* Add to  by @muellerzr in #1448
* split_between_processes  by @stevhliu in #1449
* [docs] Replace `state.rank` -> `process_index`  by @pcuenca in #1450
* Auto multigpu logic  by @muellerzr in #1452
* Update with cli instructions  by @muellerzr in #1453
* Adds `in_order` argument that defaults to False, to log in order.  by @JulesGM in #1262
* fix error for CPU DDP using trainer api.  by @sywangyi in #1455
* Refactor and simplify xpu device in state  by @abhilash1910 in #1456
* Document how to use commands with python module instead of argparse  by @muellerzr in #1457
* 4-bit QLoRA via bitsandbytes (4-bit base model + LoRA)  by @TimDettmers in #1458
* Fix skip first batch being perminant  by @muellerzr in #1466
* update conversion of layers to retain original data type.  by @avisinghal6 in #1467
* Check for xpu specifically  by @muellerzr in #1472
* update `register_empty_buffer` to match torch args  by @NouamaneTazi in #1465
* Update gradient accumulation docs, and remove redundant example  by @iantbutler01 in #1461
* Imrpove sagemaker  by @muellerzr in #1470
* Split tensors as part of `split_between_processes`  by @muellerzr in #1477
* Move to device  by @muellerzr in #1478
* Fix gradient state bugs in multiple dataloader  by @Ethan-yt in #1483
* Add rdzv-backend  by @muellerzr in #1490
* Only use IPEX if available  by @muellerzr in #1495
* Update README.md  by @lyhue1991 in #1493
* Let gather_for_metrics always run  by @muellerzr in #1496
* Use empty like when we only need to create buffers  by @thomasw21 in #1497
* Allow key skipping in big model inference  by @sgugger in #1491
* fix crash when ipex is installed and torch has no xpu  by @sywangyi in #1502
* [`bnb`] Add fp4 support for dispatch  by @younesbelkada in #1505
* Fix 4bit model on multiple devices  by @SunMarc in #1506
* adjust overriding of model's forward function  by @prathikr in #1492
* Add assertion when call prepare with deepspeed config.  by @tensimiku in #1468
* NVME path support for deepspeed  by @abhilash1910 in #1484
* should set correct dtype to ipex optimize and use amp logic in native…  by @sywangyi in #1511
* Swap env vars for XPU and IPEX + CLI  by @muellerzr in #1513
* Fix a bug when parameters tied belong to the same module  by @sgugger in #1514
* Fixup deepspeed/cli tests  by @muellerzr in #1526
* Refactor mp into its own wrapper  by @muellerzr in #1527
* Check tied parameters  by @SunMarc in #1529
* Raise ValueError on iterable dataset if we've hit the end and attempting to go beyond it  by @muellerzr in #1531
* Officially support naive PP for quantized models + PEFT  by @younesbelkada in #1523
* remove ipexplugin, let ACCELERATE_USE_IPEX/ACCELERATE_USE_XPU control the ipex and xpu  by @sywangyi in #1503
* Prevent using extra VRAM for static device_map  by @LSerranoPEReN in #1536
* Update deepspeed.mdx  by @LiamSwayne in #1541
* Update performance.mdx  by @LiamSwayne in #1543
* Update deferring_execution.mdx  by @LiamSwayne in #1544
* Apply deprecations  by @muellerzr in #1537
* Add mps support to big inference modeling  by @SunMarc in #1545
* [documentation] grammar fixes in gradient_synchronization.mdx  by @LiamSwayne in #1547
* Eval mode  by @muellerzr in #1540
* Update migration.mdx  by @LiamSwayne in #1549

## Significant community contributions

The following contributors have made significant changes to the library over the last release:

* @will-cromar
    * Support TPU v4 with new PyTorch/XLA TPU runtime (#1393)
    * Support TPU v2 and v3 on new PyTorch/XLA TPU runtime (#1385)
* @searchivarius
    * Adding support for local SGD. (#1378)
* @abhilash1910
    * Intel GPU support initialization (#1118)
    * Fix bug on ipex for diffusers (#1426)
    * Refactor and simplify xpu device in state (#1456)
    * NVME path support for deepspeed (#1484)
* @sywangyi
    * fix error for CPU DDP using trainer api. (#1455)
    * fix crash when ipex is installed and torch has no xpu (#1502)
    * should set correct dtype to ipex optimize and use amp logic in native… (#1511)
    * remove ipexplugin, let ACCELERATE_USE_IPEX/ACCELERATE_USE_XPU control the ipex and xpu (#1503)
* @Ethan-yt
    * Fix gradient state bugs in multiple dataloader (#1483)


[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.20.0)

---

## v0.19.0: v0.19.0: IPEX Support, Foundations for Transformers Integration, FP8 for Ada Lovelace GPUs, and Squashed Bugs
**Published:** 2023-05-08

## What's New

- Support for Intel IPEX support has been added, check out the how-to guide [now!](https://huggingface.co/docs/accelerate/main/en/usage_guides/ipex)
- Various modifications have been added to begin work on having 🤗 Accelerate be the foundation for the `Trainer`, keep an eye on the repos to see how our progress is coming along!
- FP8 training is now supported on Ada Lovelance GPUs
- The `wandb` integration now supports logging of images and tables through `tracker.log_images` and `tracker.log_tables` respectively
- Many, many squashed bugs! (see the full detailed report for just what they were)
- **17** new contributors to the framework, congratulations to all who took their first step!  🚀 

# What's Changed

* Fix pypi image by @muellerzr in https://github.com/huggingface/accelerate/pull/1249
* raise error when dataloader with None as batch_size when using DS by @pacman100 in https://github.com/huggingface/accelerate/pull/1250
* Handle missing deepspeed config by @HeyangQin in https://github.com/huggingface/accelerate/pull/1251
* [`core`] Add Quantization support for `dispatch_model` by @younesbelkada in https://github.com/huggingface/accelerate/pull/1237
* Check attribute 'overflow' exists in optimizer. by @tensimiku in https://github.com/huggingface/accelerate/pull/1259
* ipex intel extension for pytorch integration by @sywangyi in https://github.com/huggingface/accelerate/pull/1255
* fix issue template by @stas00 in https://github.com/huggingface/accelerate/pull/1264
* Change error raised to ValueError by @sgugger in https://github.com/huggingface/accelerate/pull/1267
* Fix reduce operation by @xyfJASON in https://github.com/huggingface/accelerate/pull/1268
* Raise import error if fp8 not available in `has_transfomer_engine_layers` by @muellerzr in https://github.com/huggingface/accelerate/pull/1283
* Add missing FP8 options to CLI by @muellerzr in https://github.com/huggingface/accelerate/pull/1284
* Update quicktour.mdx by @standardAI in https://github.com/huggingface/accelerate/pull/1273
* Minor fix whitespace colon by @guspan-tanadi in https://github.com/huggingface/accelerate/pull/1272
* fix attribute error in DataloaderShared by @ZhiyuanChen in https://github.com/huggingface/accelerate/pull/1278
* Fix TypeError bug in honor_type by @muellerzr in https://github.com/huggingface/accelerate/pull/1285
* Raise more explicit error when transformer_engine isn't installed by @muellerzr in https://github.com/huggingface/accelerate/pull/1287
* Expound error on `recursively_apply` by @muellerzr in https://github.com/huggingface/accelerate/pull/1286
* Only check for dtype if it has it in get_state_dict by @muellerzr in https://github.com/huggingface/accelerate/pull/1288
* [`bnb`] fix bnb slow test by @younesbelkada in https://github.com/huggingface/accelerate/pull/1292
* Raise better error on `notebook_launcher` by @muellerzr in https://github.com/huggingface/accelerate/pull/1293
* Make note about grad accum and prec in performance documentation by @muellerzr in https://github.com/huggingface/accelerate/pull/1296
* fix for load_checkpoint_and_dispatch(device_map=None) by @anentropic in https://github.com/huggingface/accelerate/pull/1297
* Set the state device dependant to Accelerator on multigpu by @muellerzr in https://github.com/huggingface/accelerate/pull/1220
* add usage guide for ipex plugin by @sywangyi in https://github.com/huggingface/accelerate/pull/1270
* Simplify MPS implementation by @sgugger in https://github.com/huggingface/accelerate/pull/1308
* Bug fix in setattr by @aashiqmuhamed in https://github.com/huggingface/accelerate/pull/1312
* Allow xpu backend by @muellerzr in https://github.com/huggingface/accelerate/pull/1313
* Default to nccl by @muellerzr in https://github.com/huggingface/accelerate/pull/1314
* offload the previous module hook before the current module is moved to… by @williamberman in https://github.com/huggingface/accelerate/pull/1315
* Ensure that dynamo is compatible with mixed precision by @muellerzr in https://github.com/huggingface/accelerate/pull/1318
* Upgrade torch version on main tests by @muellerzr in https://github.com/huggingface/accelerate/pull/1323
* Add test flag and import check for dynamo by @muellerzr in https://github.com/huggingface/accelerate/pull/1322
* ensure module prefixes only match that module by @xloem in https://github.com/huggingface/accelerate/pull/1319
* remove repetitive entries from device lists by @xloem in https://github.com/huggingface/accelerate/pull/1321
* Fix failing test on main by @muellerzr in https://github.com/huggingface/accelerate/pull/1332
* Verbosity, Progress Bar for Loading by @xloem in https://github.com/huggingface/accelerate/pull/1329
* Skip failing torch 2.0+ test by @muellerzr in https://github.com/huggingface/accelerate/pull/1339
* Remove unused amp import util by @muellerzr in https://github.com/huggingface/accelerate/pull/1340
* Fix nested context manager for main_process_first() by @flukeskywalker in https://github.com/huggingface/accelerate/pull/1304
* Small progress bar fix by @xloem in https://github.com/huggingface/accelerate/pull/1341
* Pop more backend options by @muellerzr in https://github.com/huggingface/accelerate/pull/1342
* Support FP8 mixed precision training for Ada Lovelace GPUs by @Dango233 in https://github.com/huggingface/accelerate/pull/1348
* using deepspeed.comm for distrbiuted init by @pacman100 in https://github.com/huggingface/accelerate/pull/1352
* [`bnb`] Fix bnb slow test by @younesbelkada in https://github.com/huggingface/accelerate/pull/1355
* Better check for packages availability by @apbard in https://github.com/huggingface/accelerate/pull/1356
* fix: typing issues, and replace deprecated python typing (Optional, Union) to `|` by @kiyoon in https://github.com/huggingface/accelerate/pull/1363
* Fix default FSDP_MIN_NUM_PARAMS so it's an int by @sam-hieken in https://github.com/huggingface/accelerate/pull/1367
* Special transformers case from args by @muellerzr in https://github.com/huggingface/accelerate/pull/1364
* Improve `accelerate env` reporting by @muellerzr in https://github.com/huggingface/accelerate/pull/1376
* Seperate out contextmanager generation by @muellerzr in https://github.com/huggingface/accelerate/pull/1379
* delete textfile after tests are done by @muellerzr in https://github.com/huggingface/accelerate/pull/1381
* Fix flakey thread issue by @muellerzr in https://github.com/huggingface/accelerate/pull/1387
* fix config bug for 'mixed_precision' from 'yaml.safe_load()' by @ys-eric-choi in https://github.com/huggingface/accelerate/pull/1386
* Log Images and other types to wandb by @tcapelle in https://github.com/huggingface/accelerate/pull/962
* Bump torch version by @muellerzr in https://github.com/huggingface/accelerate/pull/1392
* Fix gather_obj by @muellerzr in https://github.com/huggingface/accelerate/pull/1391
* Update training_zoo.mdx by @yuvalkirstain in https://github.com/huggingface/accelerate/pull/1397

## New Contributors
* @HeyangQin made their first contribution in https://github.com/huggingface/accelerate/pull/1251
* @tensimiku made their first contribution in https://github.com/huggingface/accelerate/pull/1259
* @xyfJASON made their first contribution in https://github.com/huggingface/accelerate/pull/1268
* @standardAI made their first contribution in https://github.com/huggingface/accelerate/pull/1273
* @guspan-tanadi made their first contribution in https://github.com/huggingface/accelerate/pull/1272
* @anentropic made their first contribution in https://github.com/huggingface/accelerate/pull/1297
* @aashiqmuhamed made their first contribution in https://github.com/huggingface/accelerate/pull/1312
* @williamberman made their first contribution in https://github.com/huggingface/accelerate/pull/1315
* @xloem made their first contribution in https://github.com/huggingface/accelerate/pull/1319
* @flukeskywalker made their first contribution in https://github.com/huggingface/accelerate/pull/1304
* @Dango233 made their first contribution in https://github.com/huggingface/accelerate/pull/1348
* @apbard made their first contribution in https://github.com/huggingface/accelerate/pull/1356
* @kiyoon made their first contribution in https://github.com/huggingface/accelerate/pull/1363
* @sam-hieken made their first contribution in https://github.com/huggingface/accelerate/pull/1367
* @ys-eric-choi made their first contribution in https://github.com/huggingface/accelerate/pull/1386
* @tcapelle made their first contribution in https://github.com/huggingface/accelerate/pull/962
* @yuvalkirstain made their first contribution in https://github.com/huggingface/accelerate/pull/1397

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.18.0...v0.19.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.19.0)

---

## v0.18.0: v0.18.0: GradientState enhancements and Big Model Inference Fixes
**Published:** 2023-03-24

## What's Changed

- A new `GradientAccumulationPlugin` has been added to handle more configurations with the `GradientState`. Specifically you can optionally disable having `Accelerate` automatically adjust the length of the scheduler relative to gradient accumulation steps through it. Otherwise Accelerate will now automatically handle ensuring that the schedulers built for non-gradient accumulation will work during gradient accumulation
- Some fixes related to the launch configuration and TPU launches were adjusted, and the `dynamo_backend` warning has been silenced.
- Big model inference saw a number of fixes related to linear layers, `drop_last` on linear layers, tied weight loading, and handling of multiple tied parameters
- A new integration example with RunhouseML has been added, read more here: https://github.com/huggingface/accelerate/tree/main/examples#simple-multi-gpu-hardware-launcher

## Breaking Changes

- `find_tied_parameters` now deals with groups of tied parameters (instead of only pairs of them). As a result it now returns a list of list of strings instead of a dictionary.

## What's New?
* Add documentation around FSDP state dict save behavior by @VikParuchuri in https://github.com/huggingface/accelerate/pull/1181
* add `use_orig_params` to FullyShardedDataParallelPlugin by @pacman100 in https://github.com/huggingface/accelerate/pull/1184
* Only convert linear layers with weights multiple of 16 by @sgugger in https://github.com/huggingface/accelerate/pull/1188
* Set drop last to ensure modulo16 restriction for fp8 by @ksivaman in https://github.com/huggingface/accelerate/pull/1189
* Accelerator should not call `to` on modules that wraps `accelerate` loaded models by @younesbelkada in https://github.com/huggingface/accelerate/pull/1172
* Fixup passing overlapping args to the script by @muellerzr in https://github.com/huggingface/accelerate/pull/1198
* Make the Scheduler adjust the steps taken relative to the gradient accumulation steps by @muellerzr in https://github.com/huggingface/accelerate/pull/1187
* Fix tied weights load by @sgugger in https://github.com/huggingface/accelerate/pull/1204
* Better error message when using multi-GPU and Accelerate on torch <1.9.1 by @muellerzr in https://github.com/huggingface/accelerate/pull/1203
* Fix typo in TPU config by @muellerzr in https://github.com/huggingface/accelerate/pull/1202
* Fix example in accumulate method documentation by @VikParuchuri in https://github.com/huggingface/accelerate/pull/1211
* ds offload optim fix to use CPUAdam by @pacman100 in https://github.com/huggingface/accelerate/pull/1208
* Move when the GradientState test is performed so that it is not None by @muellerzr in https://github.com/huggingface/accelerate/pull/1219
* Fix bug in loading launch config by @neumyor in https://github.com/huggingface/accelerate/pull/1218
* Fix get_logger kwarg documentation issue by @bcol23 in https://github.com/huggingface/accelerate/pull/1222
* docs: add finetuner to ppl who use accelerate by @bwanglzu in https://github.com/huggingface/accelerate/pull/1224
* Silence dynamo_backend by @muellerzr in https://github.com/huggingface/accelerate/pull/1226
* Add additional check when patching env by @Chris-hughes10 in https://github.com/huggingface/accelerate/pull/1229
* Make grad accum steps mutable on the Accelerator object by @muellerzr in https://github.com/huggingface/accelerate/pull/1233
* devcontainer: "extensions" has been removed and replaced by customizations by @dbpprt in https://github.com/huggingface/accelerate/pull/1075
* remove empty dicts while saving accelerate config by @pacman100 in https://github.com/huggingface/accelerate/pull/1236
* backfill ds plugin attributes when using ds_config by @pacman100 in https://github.com/huggingface/accelerate/pull/1235
* Change multinode to multigpu in notebook tutorial by @muellerzr in https://github.com/huggingface/accelerate/pull/1247
* Hardware Auto-Setup Example/Tutorial for Distributed Launch by @carolineechen in https://github.com/huggingface/accelerate/pull/1227
* Handle multiple tied parameters by @sgugger in https://github.com/huggingface/accelerate/pull/1241

## New Contributors
* @hackpert made their first contribution in https://github.com/huggingface/accelerate/pull/1180
* @VikParuchuri made their first contribution in https://github.com/huggingface/accelerate/pull/1181
* @ksivaman made their first contribution in https://github.com/huggingface/accelerate/pull/1189
* @neumyor made their first contribution in https://github.com/huggingface/accelerate/pull/1218
* @bcol23 made their first contribution in https://github.com/huggingface/accelerate/pull/1222
* @bwanglzu made their first contribution in https://github.com/huggingface/accelerate/pull/1224
* @carolineechen made their first contribution in https://github.com/huggingface/accelerate/pull/1227

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.17.1...v0.18.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.18.0)

---

## v0.17.1: v0.17.1: Patch release
**Published:** 2023-03-13

* Fix CPU error always being raised by @muellerzr in #1175
* fixed typo in launch.py tpu_pod_launcher by @hackpert in #1180
* Support special mapping of dtypes when preparing device map by @sgugger in #1179

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.17.1)

---

## v0.17.0: v0.17.0: PyTorch 2.0 support, Process Control Enhancements, TPU pod support and FP8 mixed precision training
**Published:** 2023-03-09

## PyTorch 2.0 support

This release fully supports the upcoming PyTorch 2.0 release. You can choose to use `torch.compile` or not and then customize the options in `accelerate.config` or via a `TorchDynamoPlugin`.

* update support for torch dynamo compile  by @pacman100 in #1150

## Process Control Enhancements

This release adds a new `PartialState`, which contains most of the capabilities of the `AcceleratorState` however it is designed to be used by the user to assist in any process control mechanisms around it. With this, users also now do not need to have `if accelerator.state.is_main_process` when utilizing classes such as the `Tracking` API, as these now will automatically use only the main process for their work by default. 

* Refactor process executors to be in AcceleratorState  by @muellerzr in #1039

## TPU Pod Support (Experimental)

Launching from TPU pods is now supported, please see [this issue](https://github.com/huggingface/accelerate/issues/501#issuecomment-1424614540) for more information

* Introduce TPU Pod launching to `accelerate launch`  by @muellerzr in #1049

## FP8 mixed precision training (Experimental)

This release adds experimental support for FP8 mixed precision training, which requires the [transformer-engine](https://github.com/NVIDIA/TransformerEngine) library as well as a Hopper GPU (or higher).

* Fp8 integration  by @sgugger in #1086

## What's new?

* v0.17.0.dev0 by @sgugger (direct commit on main)
* Deepspeed param check  by @dhar174 in #1015
* enabling `mps` device by default and removing related config  by @pacman100 in #1030
* fix: links to gradient synchronization  by @prassanna-ravishankar in #1035
* do not scale gradient in bf16 mode  by @kashif in #1036
* Pass keywords arguments of backward function deeper to DeepSpeed  by @DistinctVision in #1037
* Add daily slack notifier for nightlies  by @muellerzr in #1042
* Make sure direct parameters are properly set on device  by @sgugger in #1043
* Add `cpu_offload_with_hook`  by @sgugger in #1045
* Update quality tools to 2023  by @sgugger in #1046
* Load tensors directly on device  by @sgugger in #1028
* Fix cpu_offload_with_hook code snippet  by @pcuenca in #1047
* Use create_task  by @muellerzr in #1052
* Fix args by adding in the defaults  by @muellerzr in #1053
* deepspeed `hidden_size` auto value default fixes  by @pacman100 in #1060
* Introduce PartialState  by @muellerzr in #1055
* Flag for deprecation  by @muellerzr in #1061
* Try with this  by @muellerzr in #1062
* Update integrations  by @muellerzr in #1063
* Swap utils over to use PartialState  by @muellerzr in #1065
* update fsdp docs and removing deepspeed version pinning  by @pacman100 in #1059
* Fix/implement process-execution decorators on the Accelerator  by @muellerzr in #1070
* Refactor state and make `PartialState` first class citizen   by @muellerzr in #1071
* Add error if passed --config_file does not exist  by @muellerzr in #1074
* SageMaker image_uri is now optional  by @<NOT FOUND> in #1077
* Allow custom SageMaker Estimator arguments  by @<NOT FOUND> in #1080
* Fix tpu_cluster arg  by @muellerzr in #1081
* Update complete_cv_example.py  by @fcossio in #1082
* Added SageMaker local mode config section  by @<NOT FOUND> in #1084
* Fix config  by @muellerzr in #1090
* adds missing "lfs" in pull  by @CSchoel in #1091
* add multi_cpu support to reduce  by @alex-hh in #1094
* Update README.md  by @BM-K in #1100
* Tracker rewrite and lazy process checker  by @muellerzr in #1079
* Update performance.mdx  by @fcossio in #1107
* Attempt to unwrap tracker.  by @pcuenca in #1109
* TensorBoardTracker: wrong arg def  by @stas00 in #1111
* Actually raise if exception  by @muellerzr in #1124
* Add test for ops and fix reduce  by @muellerzr in #1122
* Deep merge SageMaker `additional_args`, allowing more flexible configuration and `env` variable support  by @dbpprt in #1113
* Move dynamo.optimize to the end of model preparation  by @ymwangg in #1128
* Refactor `launch` for greater extensibility  by @Yard1 in #1123
* [Big model loading] Correct GPU only loading  by @patrickvonplaten in #1121
* Add tee and role to launch  by @muellerzr in #1132
* Expand warning and grab all GPUs available by default  by @muellerzr in #1134
* Fix multinode with GPU ids when each node has 1  by @muellerzr in #1127
* deepspeed dataloader prepare fix  by @pacman100 in #1126
* fix ds dist init kwargs issue  by @pacman100 in #1138
* fix lr scheduler issue  by @pacman100 in #1140
* fsdp bf16 enable autocast  by @pacman100 in #1125
* Fix notebook_launcher  by @muellerzr in #1141
* fix partial state  by @pacman100 in #1144
* FSDP enhancements and fixes  by @pacman100 in #1145
* Fixed typos in notebook  by @SamuelLarkin in #1146
* Include a note in the gradient synchronization docs on "what can go wrong" and show the timings  by @muellerzr in #1153
* [Safetensors] Relax missing metadata constraint  by @patrickvonplaten in #1151
* Solve arrow keys being environment dependant for accelerate config by @p1atdev (direct commit on main)
* Load custom state to cpu  by @Guangxuan-Xiao in #1156
* :memo: add a couple more trackers to the docs  by @nateraw in #1158
* Let GradientState know active dataloaders and reset the remainder  by @muellerzr in #1162
* Attempt to fix import error when PyTorch is build without `torch.distributed` module  by @mfuntowicz in #1108
* [`Accelerator`] Fix issue with 8bit models  by @younesbelkada in #1155
* Document skip_first_batches in the checkpoint usage guides  by @muellerzr in #1164
* Fix what files get deleted through `total_limit`  by @muellerzr in #1165
* Remove outdated command directions and use in tests  by @muellerzr in #1166

## Significant community contributions

The following contributors have made significant changes to the library over the last release:

* @Yard1
    * Refactor `launch` for greater extensibility (#1123)


[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.17.0)

---

## v0.16.0: v0.16.0: Improved and Interactive Documentation, DataLoader Improvements
**Published:** 2023-01-31

## New code exploration doc tool

A new interactive tool has been introduced to the documentation to help users quickly learn how to utilize features of the framework before providing more details on them as shown below: 
![image](https://user-images.githubusercontent.com/7831895/215847833-3434ec6f-1f8d-41b8-9368-48f76fb7783d.png)

Not only does it provide a code diff, but it also includes an explanation and links to more resources the user should check out to learn more:

![image](https://user-images.githubusercontent.com/7831895/215847922-5879f5d4-fa11-44de-9550-1e986da57bcb.png)

Try it out today in the [docs](https://huggingface.co/docs/accelerate/en/usage_guides/explore)

* Add in code exploration tool to docs  by @muellerzr in #1014
* Light vs dark theme based on pick  by @muellerzr in #1023

## Skip batches in dataloaders

When resuming training, you can more efficiently skip batches in your dataloader with the new `skip_first_batches` function (also available as a method on your `Accelerator`).

* Efficiently skip batches in a dataloader  by @sgugger in #1002

## DeepSpeed integration enhancements:

A new ZeRO-3 init context manager is added to provide granular control to users in situations involving nested/multiple models. Refactoring of DeepSpeed Config file support to remove ambiguity between it and Accelerate config.  

Adding support for `auto` entries in the DeeSpeed config file to be filled via the `accelerate launch` command. Try it out today by referring to the section [Things to note when using DeepSpeed Config File](https://huggingface.co/docs/accelerate/main/en/usage_guides/deepspeed#deepspeed-config-file)

* ds zero-3 init context manager  by @pacman100 in #932
* raise error for duplicate accelerate config  values when using `deepspeed_config_file`  by @pacman100 in #941


## What's new?

* Flag to silence subprocess.CalledProcessError in launch  by @Cyberes in #902
* Add usage examples  by @muellerzr in #904
* Expand sanity checks  by @muellerzr in #905
* Fix conditional  by @muellerzr in #907
* fix issue that amp bf16 does not work for cpu in env with cuda.  by @sywangyi in #906
* fsdp enhancements  by @pacman100 in #911
* Fix typos accelerate -> accelerator  by @pcuenca in #915
* 🚨🚨🚨 Act on deprecations 🚨🚨🚨  by @muellerzr in #917
* fix accelerate test failure with cpu config  by @sywangyi in #909
* Introduce `project_dir` and limit the number of saved checkpoints  by @muellerzr in #916
* Specify inference  by @muellerzr in #921
* Support `init_on_device`  by @thomasw21 in #926
* ds-z3-init and prepending ds env variables with `ACCELERATE_`  by @pacman100 in #928
* Honor model dtype in `load_checkpoint`  by @sgugger in #920
* ds zero-3 init context manager  by @pacman100 in #932
* Fix silly typo  by @tornikeo in #939
* add `mixed_precision_type` property to `AcceleratorState`  by @pacman100 in #935
* fix batch size in prepare_dataloader for iterable datasets  by @sanderland in #937
* fix mp related test fails  by @pacman100 in #943
* Fix tracker  by @muellerzr in #942
* Fix offload when weights are on the GPU  by @sgugger in #945
* raise error for duplicate accelerate config  values when using `deepspeed_config_file`  by @pacman100 in #941
* Add is_initialized method and refactor  by @muellerzr in #949
* Fix DeepSpeed tests  by @muellerzr in #950
* Don't automatically offload buffers when loading checkpoints  by @sgugger in #951
* Typo fix in src/accelerate/utils/modeling.py  by @ryderwishart in #955
* support master port when using ds multi-node launcher  by @pacman100 in #959
* Allowing encoded configuration for DeepSpeed  by @cli99 in #895
* Update README.md  by @Don9wanKim in #968
* Raise minimum version for distrib launch  by @muellerzr in #978
* Fix tied parameters test in big model inference  by @sgugger in #979
* Fix type error on line 36  by @dhar174 in #981
* Ensure that last batch doesn't get dropped if perfectly even in gather_for_metrics  by @muellerzr in #982
* Skip wandb test for now  by @muellerzr in #984
* Fix test for converting tensor to proper dtype  by @sgugger in #983
* in sync with trfs, removing style_doc utils and using doc-builder instead  by @pacman100 in #988
* Add new release_memory util  by @muellerzr in #990
* adding support for kwargs in `load_state`  by @pacman100 in #989
* Fix scheduler incorrect steps when gradient accumulation enabled  by @markovalexander in #999
* Fix parameters tying in dispatch_model  by @sgugger in #1000
* improve deepspeed notes  by @stas00 in #1003
* Update toctree  by @muellerzr in #1008
* Add styleguide  by @muellerzr in #1007
* Maintain accumulation steps  by @muellerzr in #1011
* Saving and loading state hooks  by @patrickvonplaten in #991
* Fix test introduced in PR and introduce AcceleratorTestCase  by @muellerzr in #1016
* Allow the torch device to be set with an env var  by @Yard1 in #1009
* Fix import of LrScheduler  by @sgugger in #1017
* Don't force mixed precision as no in examples  by @sgugger in #1018
* Include steppage in performance docs  by @muellerzr in #1013
* Fix env var  by @muellerzr in #1024
* Change default for keep_fp32_wrapper  by @muellerzr in #1025
* Fix slow test by keeping tied weights on the same GPU  by @sgugger in #1026
* Start of adding examples  by @muellerzr in #1001
* More improvements to docstrings + examples  by @muellerzr in #1010
* With example  by @muellerzr in #1027
* sagemaker launcher fixes  by @pacman100 in #1031

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.16.0)

---

## v0.15.0: v0.15.0: Pytorch 2.0 stack support
**Published:** 2022-12-02


## PyTorch 2.0 stack support

We are very excited by the newly announced PyTorch 2.0 stack and you can try it using Accelerate on any model by using the `dynamo_backend` argument of the `Accelerator`, or when filling your config with `accelerate config`.

Note that to get the best performance, we recommend:
- using an Ampere GPU (or more recent)
- sticking to fixed shaped for now

* Add support for torch dynamo  by @sgugger in #829

## New CLI commands

* Added two new commands, `accelerate config update` and `accelerate config default`. The first will update a config file to have the latest keys added from latter releases of Accelerate, and the second will create a default configuration file automatically mimicking `write_default_config()` introduced in #851 and #853 by @muellerzr 
* Also introduced a filterable help for `accelerate launch` which will show options relevant to the choices shown, such as `accelerate launch --multi_gpu` will show launch parameters relevant to multi-gpu training. 

## What's new?

* fix 🐛  by @pacman100 in #836
* Deepspeed example should use gather_for_metrics  by @HammadB in #821
* Highlight selection with pretty colors  by @muellerzr in #839
* Add `join_uneven_inputs` context manager to Accelerator  by @Chris-hughes10 in #820
* Introduce `default-config` command  by @muellerzr in #840
* Fix log error and add log level to get_logger  by @muellerzr in #842
* Fix if/else  by @muellerzr in #849
* Fix complete_cv example  by @muellerzr in #848
* Refactor Accelerate config and introduce a multi-argument CLI interface  by @muellerzr in #851
* Clean up, add update command  by @muellerzr in #853
* Revert "Update pr docs actions  by @mishig25 in #827)" 
* Switch default log to warn  by @muellerzr in #859
* Remove mixed precision hook as part of the unwrap_model  by @muellerzr in #860
* update deepspeed error message wrt `batch_size`  by @pacman100 in #861
* fix failing deepspeed test  by @pacman100 in #868
* Even more log level refined, leave alone if not explicitly set  by @muellerzr in #871
* Solve pickling issues  by @muellerzr in #872
* Spring cleaning  by @muellerzr in #865
* fixing lr_scheduler prepare issue when using pytorch nightly  by @pacman100 in #878
* fix fsdp state_dict_config because of PyTorch changes  by @pacman100 in #877
* Update deprecated logging warn  by @SHi-ON in #881
* fix a bug  by @xiaohu2015 in #887
* Allow safetensors offload  by @sgugger in #873
* fixing lr scheduler for pytorch nightly  by @pacman100 in #884
* Prefix all accelerate env vars with ACCELERATE  by @muellerzr in #890
* fix prefix issues in tests  by @pacman100 in #891
* Fix windows cli selector  by @muellerzr in #893
* Better description for improper kwargs  by @muellerzr in #894
* Support bfloat16 in load_offloaded_weight  by @sgugger in #892

## Significant community contributions

The following contributors have made significant changes to the library over the last release:

* @Chris-hughes10
    * Add `join_uneven_inputs` context manager to Accelerator (#820)

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.15.0)

---

## v0.14.0: v0.14.0: Megatron-LM integration and support for PyTorch 1.13
**Published:** 2022-11-08

## Megatron LM integration

Accelerate now supports Megatron-LM for the three model classes (BERT, GPT-2 and T5). You can learn more in the [documentation](https://huggingface.co/docs/accelerate/usage_guides/megatron_lm).

* Megatron-LM integration  by @pacman100 in #667
* ensure megatron is 2.2.0+  by @jeffra in #755
* updating docs to use fork of megatron-lm and minor example/docs fix  by @pacman100 in #766
* adding support to return logits and generate for Megatron-LM GPT models  by @pacman100 in #819

## PyTorch 1.13 support

Fixes a bug that returned SIGKILL errors on Windows.

* Isolate distrib_run  by @muellerzr in #828

## Kaggle support with the `notebook_launcher`

With Kaggle now giving instances with two T4 GPUs, Accelerate can leverage this to do multi-gpu training from the notebook

* Work in kaggle!  by @muellerzr in #783

## What's new?

* Add `non_blocking` kwarg to `send_to_device()`  by @NouamaneTazi in #607
* [ds launcher] un-hijack PYTHONPATH  by @stas00 in #741
* Fix num_processes is not defined  by @muellerzr in #746
* [Device map] nn.Parameter don't have children  by @patrickvonplaten in #747
* Use HTML relative paths for tiles  by @lewtun in #749
* Add gpu_ids to SageMakerConfig though it should never be set  by @muellerzr in #751
* Change num_cpu_threads_per_process default  by @muellerzr in #753
* Return unclipped gradient from grad_clip_norm_  by @samuelstevens in #756
* refactor  by @pacman100 in #758
* update docs  by @pacman100 in #759
* Only wrap modules in DDP if they require grad  by @samuelstevens in #761
* Move io_same_device hook to before attach_align_device hook on cpu_offload and disk_offload.  by @piEsposito in #768
* Regression cli tests  by @muellerzr in #772
* Fix number of devices in get_balanced_memory  by @sgugger in #774
* Fix all github actions issues + depreciations  by @muellerzr in #773
* Fix flakey wandb test  by @muellerzr in #775
* Add defaults for launchers  by @muellerzr in #778
* Allow BatchSamplerShard to not even out batches  by @sgugger in #776
* Make rich toggleable and seperate out a new environment utility file  by @muellerzr in #779
* Add same_network + docs  by @muellerzr in #780
* fix transformers tests  by @ArthurZucker in #777
* Add Dev Container configuration  by @Chris-hughes10 in #782
* separate dataloader generator from sampler generator  by @pacman100 in #789
* Consider top-level buffers when computing `infer_auto_device_map`   by @younesbelkada in #792
* Add `even_batches` keyword to Accelerator  by @Chris-hughes10 in #781
* Fix device_map="auto" on CPU-only envs  by @sgugger in #797
* Fix extraction of state dict in offload  by @sgugger in #795
* fix: add pdsh as default launcher  by @zanussbaum in #800
* Deal with optimizer.differentiable in PyTorch 1.13.0  by @comaniac in #803
* Introduce a pod-config command  by @muellerzr in #802
* Refactor CLI to improve readability  by @muellerzr in #810
* adding support to pickle and unpickle `AcceleratedOptimizer`  by @pacman100 in #811
* add `recurse` argument in `remove_hook_from_module`  by @younesbelkada in #812
* Act on deprecations  by @muellerzr in #813
* Mlflow-tracker-v2 🔥  by @nbroad1881 in #794
* Update CLI docs and use mps rather than mps_device  by @muellerzr in #814
* Rename pod-config to tpu-config + docs  by @muellerzr in #818
* Update docs  by @muellerzr in #823
* rename sklearn to proper dep  by @muellerzr in #825
* Rename  by @muellerzr in #824
* Update pr docs actions  by @mishig25 in #827

## Significant community contributions

The following contributors have made significant changes to the library over the last release:

* @Chris-hughes10
    * Add Dev Container configuration (#782)
    * Add `even_batches` keyword to Accelerator (#781)

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.14.0)

---

## v0.13.2: v0.13.2 Patch release
**Published:** 2022-10-17

- [Device map] nn.Parameter don't have children in #747 by @patrickvonplaten 

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.13.2)

---

## v0.13.1: v0.13.1 Patch release
**Published:** 2022-10-07

- Fix num_processes is not defined #746 by @muellerzr 

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.13.1)

---

## v0.13.0: v0.13.0 Launcher update (multinode and GPU selection) and mutliple bug fixes
**Published:** 2022-10-05

## Better multinode support in the launcher

The `accelerate command` launch did not work well for distributed training using several machines. This is fixed in this version.

* Use torchrun for multinode  by @muellerzr in #631
* Fix multi-node issues from launch  by @muellerzr in #672

## Launch training on specific GPUs only

Instead of prefixing your launch command with `CUDA_VISIBLE_DEVICES=xxx` you can now specify the GPUs you want to use in your Accelerate config.

* Allow for GPU-ID specification on CLI  by @muellerzr in #732

## Better tracebacks and rich support

The tracebacks are now cleaned up to avoid printing several times the same error, and rich is integrated as an optional dependency.

* Integrate Rich into Accelerate  by @muellerzr in #613
* Make rich an optional dep  by @muellerzr in #673

## What's new?

* Fix typo in docs/index.mdx  by @mishig25 in #610
* Fix DeepSpeed CI  by @muellerzr in #612
* Added GANs example to examples  by @EyalMichaeli in #619
* Fix example  by @muellerzr in #620
* Update README.md  by @ezhang7423 in #622
* Fully remove `subprocess` from the multi-gpu launcher  by @muellerzr in #623
* M1 mps fixes  by @pacman100 in #625
* Fix multi-node issues and simplify param logic  by @muellerzr in #627
* update MPS support docs  by @pacman100 in #629
* minor tracker fixes for complete* examples  by @pacman100 in #630
* Put back in place the guard  by @muellerzr in #634
* make init_trackers to launch on main process  by @Gladiator07 in #642
* remove check for main process for trackers initialization  by @Gladiator07 in #643
* fix link  by @philschmid in #645
* Add static_graph arg to DistributedDataParallelKwargs.  by @rom1504 in #637
* Small nits to grad accum docs  by @muellerzr in #656
* Saving hyperparams in yaml file for Tensorboard for #521  by @Shreyz-max in #657
* Use debug for loggers  by @muellerzr in #655
* Improve docstrings more  by @muellerzr in #666
* accelerate bibtex  by @pacman100 in #660
* Cache torch_tpu check  by @muellerzr in #670
* Manim animation of big model inference  by @muellerzr in #671
* Add aim tracker for accelerate  by @muellerzr in #649
* Specify local network on multinode  by @muellerzr in #674
* Test for min torch version + fix all issues  by @muellerzr in #638
* deepspeed enhancements and fixes  by @pacman100 in #676
* DeepSpeed launcher related changes  by @pacman100 in #626
* adding torchrun elastic params  by @pacman100 in #680
* :bug: fix  by @pacman100 in #683
* Fix skip in dispatch dataloaders  by @sgugger in #682
* Clean up DispatchDataloader a bit more  by @sgugger in #686
* rng state sync for FSDP  by @pacman100 in #688
* Fix DataLoader with samplers that are batch samplers  by @sgugger in #687
* fixing support for Apple Silicon GPU in `notebook_launcher`  by @pacman100 in #695
* fixing rng sync when using custom sampler and batch_sampler  by @pacman100 in #696
* Improve `init_empty_weights` to override tensor constructor  by @thomasw21 in #699
* override DeepSpeed `grad_acc_steps` from `accelerator` obj  by @pacman100 in #698
* [doc] Fix 404'd link in memory usage guides  by @tomaarsen in #702
* Add in report generation for test failures and make fail-fast false  by @muellerzr in #703
* Update runners with report structure, adjust env variable  by @muellerzr in #704
* docs: examples readability improvements  by @ryanrussell in #709
* docs: `utils` readability fixups  by @ryanrussell in #711
* refactor(test_tracking): `key_occurrence` readability fixup  by @ryanrussell in #710
* docs: `hooks` readability improvements  by @ryanrussell in #712
* sagemaker fixes and improvements  by @pacman100 in #708
* refactor(accelerate): readability improvements  by @ryanrussell in #713
* More docstring nits  by @muellerzr in #715
* Allow custom device placements for different objects  by @sgugger in #716
* Specify gradients in model preparation  by @muellerzr in #722
* Fix regression issue  by @muellerzr in #724
* Fix default for num processes  by @sgugger in #726
* Build and Release docker images on a release  by @muellerzr in #725
* Make running tests more efficient  by @muellerzr in #611
* Fix old naming  by @muellerzr in #727
* Fix issue with one-cycle logic  by @muellerzr in #728
* Remove auto-bug label in issue template  by @sgugger in #735
* Add a tutorial on proper benchmarking  by @muellerzr in #734
* Add an example zoo to the documentation  by @muellerzr in #737
* trlx  by @muellerzr in #738
* Fix memory leak  by @muellerzr in #739
* Include examples for CI  by @muellerzr in #740
* Auto grad accum example  by @muellerzr in #742


[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.13.0)

---

## v0.12.0: v0.12.0 New doc, gather_for_metrics, balanced device map and M1 support
**Published:** 2022-08-04


## New documentation

The whole documentation has been revamped, just go look at it [here](https://huggingface.co/docs/accelerate)!

* Complete revamp of the docs  by @muellerzr in #495


## New gather_for_metrics method

When doing distributed evaluation, the dataloader loops back at the beginning of the dataset to make batches that have a round multiple of the number of processes. This causes the predictions to be slightly bigger than the length of the dataset, which used to require some truncating. This is all done behind the scenes now if you replace the `gather` your did in evaluation by `gather_for_metrics`.

* Reenable Gather for Metrics  by @muellerzr in #590
* Fix gather_for_metrics  by @muellerzr in #578
* Add a gather_for_metrics capability  by @muellerzr in #540

## Balanced device maps

When loading big models for inference, `device_map="auto"` used to fill the GPUs sequentially, making it hard to use a batch size > 1. It now balances the weights evenly on the GPUs so if you have more GPU space than the model size, you can do predictions with a bigger batch size!

## M1 GPU support

Accelerate now supports M1 GPUs, to learn more about how to setup your environment, see the [documentation](https://huggingface.co/docs/accelerate/v0.12.0/en/usage_guides/mps#accelerated-pytorch-training-on-mac).

* M1 GPU `mps` device integration  by @pacman100 in #596

## What's new?

* Small fixed for balanced device maps  by @sgugger in #583
* Add balanced option for auto device map creation  by @sgugger in #534
* fixing deepspeed slow tests issue  by @pacman100 in #604
* add more conditions on casting  by @younesbelkada in #606
* Remove redundant `.run` in `WandBTracker`.  by @zh-plus in #605
* Fix some typos + wordings  by @muellerzr in #603
* reorg of test scripts and minor changes to tests  by @pacman100 in #602
* Move warning  by @muellerzr in #598
* Shorthand way to grab a tracker  by @muellerzr in #594
* Pin deepspeed  by @muellerzr in #595
* Improve docstring  by @muellerzr in #591
* TESTS!  by @muellerzr in #589
* Fix DispatchDataloader  by @sgugger in #588
* Use main_process_first in the examples  by @muellerzr in #581
* Skip and raise NotImplementedError for gather_for_metrics for now  by @muellerzr in #580
* minor FSDP launcher fix  by @pacman100 in #579
* Refine test in set_module_tensor_to_device  by @sgugger in #577
* Fix `set_module_tensor_to_device`  by @sgugger in #576
* Add 8 bit support - chapter II  by @younesbelkada in #539
* Fix tests, add wandb to gitignore  by @muellerzr in #573
* Fix step  by @muellerzr in #572
* Speed up main CI  by @muellerzr in #571
* ccl version check and import different module according to version  by @sywangyi in #567
* set default num_cpu_threads_per_process to improve oob performance  by @sywangyi in #562
* Add a tqdm helper  by @muellerzr in #564
* Rename actions to be a bit more accurate  by @muellerzr in #568
* Fix clean  by @muellerzr in #569
* enhancements and fixes for FSDP and DeepSpeed  by @pacman100 in #532
* fix: saving model weights  by @csarron in #556
* add on_main_process decorators  by @ZhiyuanChen in #488
* Update imports.py  by @KimBioInfoStudio in #554
* unpin `datasets`  by @lhoestq in #563
* Create good defaults in `accelerate launch`  by @muellerzr in #553
* Fix a few minor issues with example code in docs  by @BenjaminBossan in #551
* deepspeed version `0.6.7` fix  by @pacman100 in #544
* Rename test extras to testing  by @muellerzr in #545
* Add production testing + fix failing CI  by @muellerzr in #547
* Add a gather_for_metrics capability  by @muellerzr in #540
* Allow for kwargs to be passed to trackers  by @muellerzr in #542
* Add support for downcasting bf16 on TPUs  by @muellerzr in #523
* Add more documentation for device maps computations  by @sgugger in #530
* Restyle prepare one  by @muellerzr in #531
* Pick a better default for offload_state_dict  by @sgugger in #529
* fix some parameter setting does not work for CPU DDP and bf16 fail in…  by @sywangyi in #527
* Fix accelerate tests command  by @sgugger in #528

## Significant community contributions

The following contributors have made significant changes to the library over the last release:

* @sywangyi
    * ccl version check and import different module according to version (#567)
    * set default num_cpu_threads_per_process to improve oob performance (#562)
    * fix some parameter setting does not work for CPU DDP and bf16 fail in… (#527)
* @ZhiyuanChen
    * add on_main_process decorators (#488)

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.12.0)

---

## v0.11.0: v0.11.0 Gradient Accumulation and SageMaker Data Parallelism
**Published:** 2022-07-18

## Gradient Accumulation

Accelerate now handles gradient accumulation if you want, just pass along `gradient_accumulation_steps=xxx` when instantiating the `Accelerator` and put all your training loop step under a `with accelerator.accumulate(model):`. Accelerate will then handle the loss re-scaling and gradient accumulation for you (avoiding slowdowns in distributed training when gradients only need to be synced when you want to step). More details in the [documentation](https://huggingface.co/docs/accelerate/gradient_accumulation#letting-accelerate-handle-gradient-accumulation).

* Add gradient accumulation doc  by @muellerzr in #511
* Make gradient accumulation work with dispatched dataloaders  by @muellerzr in #510
* Introduce automatic gradient accumulation wrapper + fix a few test issues  by @muellerzr in #484

## Support for SageMaker Data parallelism

Accelerate now support SageMaker specific brand of data parallelism.

* SageMaker enhancements to allow custom docker image, input channels referring to s3/remote data locations and metrics logging  by @pacman100 in #504
* SageMaker DP Support  by @pacman100 in #494

## What's new?

* Fix accelerate tests command  by @sgugger in #528
* FSDP integration enhancements and fixes  by @pacman100 in #522
* Warn user if no trackers are installed  by @muellerzr in #524
* Fixup all example CI tests and properly fail  by @muellerzr in #517
* fixing deepspeed multi-node launcher  by @pacman100 in #514
* Add special Parameters modules support  by @younesbelkada in #519
* Don't unwrap in save_state()  by @cccntu in #489
* Fix a bug when reduce a tensor.  by @wwhio in #513
* Add benchmarks  by @sgugger in #506
* Fix DispatchDataLoader length when `split_batches=True`  by @sgugger in #509
* Fix scheduler in gradient accumulation example  by @muellerzr in #500
* update dataloader wrappers to have `total_batch_size` attribute  by @pacman100 in #493
* Introduce automatic gradient accumulation wrapper + fix a few test issues  by @muellerzr in #484
* add use_distributed property  by @ZhiyuanChen in #487
* fixing fsdp autowrap functionality  by @pacman100 in #475
* Use datasets 2.2.0 for now  by @muellerzr in #481
* Rm gradient accumulation on TPU  by @muellerzr in #479
* Revert "Pin datasets for now  by @muellerzr in #477)" 
* Pin datasets for now  by @muellerzr in #477
* Some typos and cosmetic fixes  by @douwekiela in #472
* Fix when TPU device check is ran  by @muellerzr in #469
* Refactor Utility Documentation  by @muellerzr in #467
* Add docbuilder to quality  by @muellerzr in #468
* Expose some is_*_available utils in docs  by @muellerzr in #466
* Cleanup CI Warnings  by @muellerzr in #465
* Link CI slow runners to the commit  by @muellerzr in #464
* Fix subtle bug in BF16  by @muellerzr in #463
* Include bf16 support for TPUs and CPUs, and a better check for if a CUDA device supports BF16  by @muellerzr in #462
* Handle bfloat16 weights in disk offload without adding memory overhead  by @noamwies in #460) 
* Handle bfloat16 weights in disk offload  by @sgugger in #460
* Raise a clear warning if a user tries to modify the AcceleratorState  by @muellerzr in #458
* Right step point  by @muellerzr in #459
* Better checks for if a TPU device exists  by @muellerzr in #456
* Offload and modules with unused submodules  by @sgugger in #442

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.11.0)

---

## v0.10.0: V0.10.0 DeepSpeed integration revamp and TPU speedup
**Published:** 2022-06-15

This release adds two major new features: the DeepSpeed integration has been revamped to match the one in Transformers Trainer, with multiple new options unlocked, and the TPU integration has been sped up.

This version also officially stops supporting Python 3.6 and requires Python 3.7+

## DeepSpeed integration revamp

Users can now specify a DeepSpeed config file when they want to use DeepSpeed, which unlocks many new options. More details in the new [documentation](https://huggingface.co/docs/accelerate/deepspeed).

* Migrate HFDeepSpeedConfig from trfrs to accelerate  by @pacman100 in #432
* DeepSpeed Revamp  by @pacman100 in #405

## TPU speedup

If you're using TPUs we have sped up the dataloaders and models quite a bit, on top of a few bug fixes.

* Revamp TPU internals to be more efficient + enable mixed precision types  by @muellerzr in #441

## What's new?

* Fix docstring  by @muellerzr in #447
* Add psutil as depenedency  by @sgugger in #445
* fix fsdp torch version dependency  by @pacman100 in #437
* Create Gradient Accumulation Example  by @muellerzr in #431
* init  by @muellerzr in #429
* Introduce `no_sync` context wrapper + clean up some more warnings for DDP  by @muellerzr in #428
* updating tests to resolve runner failures wrt deepspeed revamp  by @pacman100 in #427
* Fix secrets in Docker workflow  by @muellerzr in #426
* Introduce a Dependency Checker to trigger new Docker Builds on main  by @muellerzr in #424
* Enable slow tests nightly  by @muellerzr in #421
* Push out python 3.6 + fix all tests related to the upgrade  by @muellerzr in #420
* Speedup main CI  by @muellerzr in #419
* Switch to evaluate for metrics  by @sgugger in #417
* Create an issue template for Accelerate  by @muellerzr in #415
* Introduce post-merge runners  by @muellerzr in #416
* Fix debug_launcher issues  by @muellerzr in #413
* Use main egg  by @muellerzr in #414
* Introduce nightly runners  by @muellerzr in #410
* Update requirements to pin tensorboard and include psutil  by @muellerzr in #408
* Fix CUDA examples tests  by @muellerzr in #407
* Move datasets and transformers to under func  by @muellerzr in #411
* Fix CUDA Dockerfile  by @muellerzr in #409
* Hotfix all failing GPU tests  by @muellerzr in #401
* improve metrics logged in examples  by @pacman100 in #399
* Refactor offload_state_dict and fix in offload_weight  by @sgugger in #398
* Refactor version checking into a utility  by @muellerzr in #395
* Include fastai in frameworks  by @muellerzr in #396
* Add packaging to requirements  by @muellerzr in #394
* Better dispatch for submodules  by @sgugger in #392
* Build Docker Images nightly  by @muellerzr in #391
* Small bugfix for the stalebot workflow  by @muellerzr in #390
* Introduce stalebot   by @muellerzr in #387
* Create Dockerfiles for Accelerate  by @muellerzr in #377
* Mix precision -> Mixed precision  by @muellerzr in #388
* Fix OneCycle step length when in multiprocess  by @muellerzr in #385

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.10.0)

---

## v0.9.0: v0.9.0: Refactor utils to use in Transformers
**Published:** 2022-05-20

# v0.9.0: Refactor utils to use in Transformers

This release offers no significant new API, it is just needed to have access to some utils in Transformers. 

* Handle deprication errors in launch  by @muellerzr in #360
* Update launchers.py  by @tmabraham in #363
* fix tracking  by @pacman100 in #361
* Remove tensor call  by @muellerzr in #365
* Add a utility for writing a barebones config file  by @muellerzr in #371
* fix deepspeed model saving  by @pacman100 in #370
* deepspeed save model temp fix  by @pacman100 in #374
* Refactor tests to use accelerate launch  by @muellerzr in #373
* fix zero stage-1  by @pacman100 in #378
* fix shuffling for ShufflerIterDataPipe instances  by @loubnabnl in #376
* Better check for deepspeed availability  by @sgugger in #379
* Refactor some parts in utils  by @sgugger in #380

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.9.0)

---

## v0.8.0: v0.8.0: Big model inference
**Published:** 2022-05-12

# v0.8.0: Big model inference

## Big model inference

To handle very large models, new functionality has been added in Accelerate:
- a context manager to initalize empty models
- a function to load a sharded checkpoint directly on the right devices
- a set of custom hooks that allow execution of a model split on different devices, as well as CPU or disk offload
- a magic method that auto-determines a device map for a given model, maximizing the GPU spaces, available RAM before using disk offload as a last resort.
- a function that wraps the last three blocks in one simple call (`load_checkpoint_and_dispatch`)

See more in the [documentation](https://huggingface.co/docs/accelerate/main/en/big_modeling)

* Big model inference  by @sgugger in #345

## What's new

* Create peak_memory_uasge_tracker.py  by @pacman100 in #336
* Fixed a typo to enable running accelerate correctly  by @Idodox in #339
* Introduce multiprocess logger  by @muellerzr in #337
* Refactor utils into its own module  by @muellerzr in #340
* Improve num_processes question in CLI  by @muellerzr in #343
* Handle Manual Wrapping in FSDP. Minor fix of fsdp example.  by @pacman100 in #342
* Better prompt for number of training devices  by @muellerzr in #344
* Fix prompt for num_processes  by @pacman100 in #347
* Fix sample calculation in examples  by @muellerzr in #352
* Fixing metric eval in distributed setup  by @pacman100 in #355
* DeepSpeed and FSDP plugin support through script  by @pacman100 in #356


[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.8.0)

---

## v0.7.1: v0.7.1 Patch release
**Published:** 2022-04-29

# v0.7.1 Patch release

- Fix fdsp config in cluster [331](https://github.com/huggingface/accelerate/pull/331)
- Add guards for batch size finder [334](https://github.com/huggingface/accelerate/pull/334)
- Patchfix infinite loop [335](https://github.com/huggingface/accelerate/pull/335)

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.7.1)

---

## v0.7.0: v0.7.0: Logging API, FSDP, batch size finder and examples revamp
**Published:** 2022-04-28

v0.7.0: Logging API, FSDP, batch size finder and examples revamp

## Logging API

Use any of your favorite logging libraries (TensorBoard, Wandb, CometML...) with just a few lines of code inside your training scripts with Accelerate. All details are in the [documentation](https://huggingface.co/docs/accelerate/tracking).

* Add logging capabilities by @muellerzr in https://github.com/huggingface/accelerate/pull/293

## Support for FSDP (fully sharded DataParallel)

PyTorch recently released a new model wrapper for sharded DDP training called [FSDP](https://pytorch.org/docs/stable/fsdp.html). This release adds support for it (note that it doesn't work with mixed precision yet). See all caveats in the [documentation](https://huggingface.co/docs/accelerate/fsdp).

* PyTorch FSDP Feature Incorporation by @pacman100 in https://github.com/huggingface/accelerate/pull/321

## Batch size finder

Say goodbye to the CUDA OOM errors with the new `find_executable_batch_size` decorator. Just decorate your training function and pick a starting batch size, then let Accelerate do the rest.

* Add a memory-aware decorator for CUDA OOM avoidance by @muellerzr in https://github.com/huggingface/accelerate/pull/324

## Examples revamp

The [Accelerate examples](https://github.com/huggingface/accelerate/tree/main/examples) are now split in two: you can find in the base folder a very simple nlp and computer vision examples, as well as complete versions incorporating all features. But you can also browse the examples in the `by_feature` subfolder, which will show you exactly what code to add for each given feature (checkpointing, tracking, cross-validation etc.)

* Refactor Examples by Feature by @muellerzr in https://github.com/huggingface/accelerate/pull/312

## What's Changed
* Document save/load state by @muellerzr in https://github.com/huggingface/accelerate/pull/290
* Refactor precisions to its own enum by @muellerzr in https://github.com/huggingface/accelerate/pull/292
* Load model and optimizet states on CPU to void OOMs by @sgugger in https://github.com/huggingface/accelerate/pull/299
* Fix example for datasets v2 by @sgugger in https://github.com/huggingface/accelerate/pull/298
* Leave default as None in `mixed_precision` for launch command by @sgugger in https://github.com/huggingface/accelerate/pull/300
* Pass `lr_scheduler` to `Accelerator.prepare` by @sgugger in https://github.com/huggingface/accelerate/pull/301
* Create new TestCase classes and clean up W&B tests by @muellerzr in https://github.com/huggingface/accelerate/pull/304
* Have custom trackers work with the API by @muellerzr in https://github.com/huggingface/accelerate/pull/305
* Write tests for comet_ml by @muellerzr in https://github.com/huggingface/accelerate/pull/306
* Fix training in DeepSpeed by @sgugger in https://github.com/huggingface/accelerate/pull/308
* Update example scripts by @muellerzr in https://github.com/huggingface/accelerate/pull/307
* Use --no_local_rank for DeepSpeed launch by @sgugger in https://github.com/huggingface/accelerate/pull/309
* Fix Accelerate CLI CPU option + small fix for W&B tests by @muellerzr in https://github.com/huggingface/accelerate/pull/311
* Fix DataLoader sharding for deepspeed in accelerate by @m3rlin45 in https://github.com/huggingface/accelerate/pull/315
* Create a testing framework for example scripts and fix current ones by @muellerzr in https://github.com/huggingface/accelerate/pull/313
* Refactor Tracker logic and write guards for logging_dir by @muellerzr in https://github.com/huggingface/accelerate/pull/316
* Create Cross-Validation example by @muellerzr in https://github.com/huggingface/accelerate/pull/317
* Create alias for Accelerator.free_memory by @muellerzr in https://github.com/huggingface/accelerate/pull/318
* fix typo in docs of accelerate tracking by @loubnabnl in https://github.com/huggingface/accelerate/pull/320
* Update examples to show how to deal with extra validation copies by @muellerzr in https://github.com/huggingface/accelerate/pull/319
* Fixup all checkpointing examples by @muellerzr in https://github.com/huggingface/accelerate/pull/323
* Introduce reduce operator by @muellerzr in https://github.com/huggingface/accelerate/pull/326

## New Contributors
* @m3rlin45 made their first contribution in https://github.com/huggingface/accelerate/pull/315
* @loubnabnl made their first contribution in https://github.com/huggingface/accelerate/pull/320
* @pacman100 made their first contribution in https://github.com/huggingface/accelerate/pull/321

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.6.0...v0.7.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.7.0)

---

## v0.6.2: v0.6.2: Fix launcher with mixed precision
**Published:** 2022-03-31

The launcher was ignoring the mixed precision attribute of the config since v0.6.0. This patch fixes that.

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.6.2)

---

## v0.6.1: v0.6.1: Hot fix
**Published:** 2022-03-18

Patches an issue with mixed precision (see #286)

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.6.1)

---

## v0.6.0: v0.6.0: Checkpointing and bfloat16 support
**Published:** 2022-03-18

This release adds support for bloat16 mixed precision training (requires PyTorch >= 1.10) and a brand-new checkpoint utility to help with resuming interrupted trainings. We also get a completely revamped [documentation frontend](https://huggingface.co/docs/accelerate/index).

## Checkpoints

Save the current state of all your objects (models, optimizers, RNG states) with `accelerator.save_state(path_to_checkpoint)` and reload everything by calling `accelerator.load_state(path_to_checkpoint)`

* Add in checkpointing capability by @muellerzr in https://github.com/huggingface/accelerate/pull/255
* Implementation of saving and loading custom states by @muellerzr in https://github.com/huggingface/accelerate/pull/270

## BFloat16 support

Accelerate now supports bfloat16 mixed precision training. As a result the old `--fp16` argument has been deprecated to be replaced by the more generic `--mixed-precision`.

* Add bfloat16 support #243 by @ikergarcia1996 in https://github.com/huggingface/accelerate/pull/247

## New env subcommand

You can now type `accelerate env` to have a copy-pastable summary of your environment and default configuration. Very convenient when opening a new issue!

* add env command by @johnnv1 in https://github.com/huggingface/accelerate/pull/280

## New doc frontend

The documentation has been switched to the new Hugging Face frontend, like Transformers and Datasets.

* Convert documentation to the new front by @sgugger in https://github.com/huggingface/accelerate/pull/271

## What's Changed

* Fix send_to_device with non-tensor data by @sgugger in https://github.com/huggingface/accelerate/pull/177
* Handle UserDict in all utils by @sgugger in https://github.com/huggingface/accelerate/pull/179
* Use collections.abc.Mapping to handle both the dict and the UserDict types by @mariosasko in https://github.com/huggingface/accelerate/pull/180
* fix: use `store_true` on argparse in nlp example by @monologg in https://github.com/huggingface/accelerate/pull/183
* Update README.md by @TevenLeScao in https://github.com/huggingface/accelerate/pull/187
* Add signature check for `set_to_none` in Optimizer.zero_grad by @sgugger in https://github.com/huggingface/accelerate/pull/189
* fix typo in code snippet by @MrZilinXiao in https://github.com/huggingface/accelerate/pull/199
* Add high-level API reference to README by @Chris-hughes10 in https://github.com/huggingface/accelerate/pull/204
* fix rng_types in accelerator by @s-kumano in https://github.com/huggingface/accelerate/pull/206
* Pass along drop_last in DispatchDataLoader by @sgugger in https://github.com/huggingface/accelerate/pull/212
* Rename state to avoid name conflicts with pytorch's Optimizer class. by @yuxinyuan in https://github.com/huggingface/accelerate/pull/224
* Fix lr scheduler num samples by @sgugger in https://github.com/huggingface/accelerate/pull/227
* Add customization point for init_process_group kwargs by @sgugger in https://github.com/huggingface/accelerate/pull/228
* Fix typo in installation docs by @jaketae in https://github.com/huggingface/accelerate/pull/234
* make deepspeed optimizer match parameters of passed optimizer by @jmhessel in https://github.com/huggingface/accelerate/pull/246
* Upgrade black to version ~=22.0 by @LysandreJik in https://github.com/huggingface/accelerate/pull/250
* add support of gather_object by @ZhiyuanChen in https://github.com/huggingface/accelerate/pull/238
* Add launch flags --module and --no_python (#256) by @parameter-concern in https://github.com/huggingface/accelerate/pull/258
* Accelerate + Animus/Catalyst = 🚀  by @Scitator in https://github.com/huggingface/accelerate/pull/249
* Add  `debug_launcher` by @sgugger in https://github.com/huggingface/accelerate/pull/259
* enhance compatibility of honor type by @ZhiyuanChen in https://github.com/huggingface/accelerate/pull/241
* Add a flag to use CPU only in the config by @sgugger in https://github.com/huggingface/accelerate/pull/263
* Basic fixes for DeepSpeed by @sgugger in https://github.com/huggingface/accelerate/pull/264
* Ability to set the seed with randomness from inside Accelerate by @muellerzr in https://github.com/huggingface/accelerate/pull/266
* Don't use dispatch_batches when torch is < 1.8.0 by @sgugger in https://github.com/huggingface/accelerate/pull/269
* Make accelerated model with AMP possible to pickle by @BenjaminBossan in https://github.com/huggingface/accelerate/pull/274
* Contributing guide by @LysandreJik in https://github.com/huggingface/accelerate/pull/254
* replace texts and link (master -> main) by @johnnv1 in https://github.com/huggingface/accelerate/pull/282
* Use workflow from doc-builder by @sgugger in https://github.com/huggingface/accelerate/pull/275
* Pass along execution info to the exit of autocast by @sgugger in https://github.com/huggingface/accelerate/pull/284

## New Contributors
* @mariosasko made their first contribution in https://github.com/huggingface/accelerate/pull/180
* @monologg made their first contribution in https://github.com/huggingface/accelerate/pull/183
* @TevenLeScao made their first contribution in https://github.com/huggingface/accelerate/pull/187
* @MrZilinXiao made their first contribution in https://github.com/huggingface/accelerate/pull/199
* @Chris-hughes10 made their first contribution in https://github.com/huggingface/accelerate/pull/204
* @s-kumano made their first contribution in https://github.com/huggingface/accelerate/pull/206
* @yuxinyuan made their first contribution in https://github.com/huggingface/accelerate/pull/224
* @jaketae made their first contribution in https://github.com/huggingface/accelerate/pull/234
* @jmhessel made their first contribution in https://github.com/huggingface/accelerate/pull/246
* @ikergarcia1996 made their first contribution in https://github.com/huggingface/accelerate/pull/247
* @ZhiyuanChen made their first contribution in https://github.com/huggingface/accelerate/pull/238
* @parameter-concern made their first contribution in https://github.com/huggingface/accelerate/pull/258
* @Scitator made their first contribution in https://github.com/huggingface/accelerate/pull/249
* @muellerzr made their first contribution in https://github.com/huggingface/accelerate/pull/255
* @BenjaminBossan made their first contribution in https://github.com/huggingface/accelerate/pull/274
* @johnnv1 made their first contribution in https://github.com/huggingface/accelerate/pull/280

**Full Changelog**: https://github.com/huggingface/accelerate/compare/v0.5.1...v0.6.0

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.6.0)

---

## v0.5.1: v0.5.1: Patch release
**Published:** 2021-09-27

# v0.5.1: Patch release

Fix the two following bugs:
- `convert_to_fp32` returned booleans instead of tensors #173
- wrong dataloader lenght when `dispatch_batches=True` #175 

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.5.1)

---

## v0.5.0: v0.5.0 Dispatch batches from main DataLoader
**Published:** 2021-09-23

# v0.5.0 Dispatch batches from main DataLoader

This release introduces support for iterating through a `DataLoader` only on the main process, that then dispatches the batches to all processes.

## Dispatch batches from main DataLoader

The motivation behind this come from dataset streaming which introduces two difficulties:
- there might be some timeouts for some elements of the dataset, which might then be different in each process launched, thus it's impossible to make sure the data is iterated though the same way on each process
- when using IterableDataset, each process goes through the dataset, thus applies the preprocessing on all elements. This can yield to the training being slowed down by this preprocessing.

This new feature is activated by default for all `IterableDataset`.

- Central dataloader #164 (@sgugger)
- Dynamic default for `dispatch_batches` #168 (@sgugger)

## Various fixes

- fix fp16 covert back to fp32 for issue: unsupported operand type(s) for /: 'dict' and 'int' #149 (@Doragd)
- [Docs] Machine config is yaml not json #151 (@patrickvonplaten)
- Fix gather for 0d tensor #152 (@sgugger)
- [DeepSpeed] allow untested optimizers deepspeed #150 (@patrickvonplaten)
- Raise errors instead of warnings with better tests #170 (@sgugger)

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.5.0)

---

## v0.4.0: v0.4.0 Experimental DeepSpeed and multi-node CPU support
**Published:** 2021-08-10

# v0.4.0 Experimental DeepSpeed support

This release adds support for [DeepSpeed](https://github.com/microsoft/DeepSpeed). While the basics are there to support ZeRO-2, ZeRo-3, as well a CPU and NVME offload, the API might evolve a little bit as we polish it in the near future.

It also adds support for multi-node CPU. In both cases, just filling the questionnaire outputted by `accelerate config` and then launching your script with `accelerate launch` is enough, there are no changes in the main API.

## DeepSpeed support

- Add DeepSpeed support #82 (@vasudevgupta7)
- DeepSpeed documentation #140 (@sgugger)

## Multinode CPU support

- Add distributed multi-node cpu only support (MULTI_CPU) #63 (@ddkalamk)

## Various fixes

- Fix batch_sampler error for IterableDataset #62 (@ddkalamk)
- Honor namedtuples in inputs/outputs #67 (@sgugger)
- Fix examples README #70 (@cccntu)
- TPU not available in kaggle #73 (@yuangan)
- Pass args in notebook_launcher for multi-GPU #78 (@sgugger)
- Fix `accelerate test` with no config file #79 (@cccntu)
- Use `optimizer` for consistency #81 (@kumapo)
- Update README.md #87 (@Separius)
- Add `unscale_gradients` method. #88 (@sgugger)
- Add Accelerator.free_memory #89 (@sgugger)
- [Feature] Add context manager to allow main process first. #98 (@Guillem96)
- Pass along kwargs to backward #104 (@sgugger)
- Add course banner #107 (@sgugger)
- added closure argument to optimizer.step() #105 (@pmelchior)
- Fix import error for torch 1.4.0 #108 (@sgugger)
- Unwrap optimizer before unscaling #115 (@sgugger)
- Fix DataLoader length when split_batches=True #121 (@sgugger)
- Fix `OptimWrapper` init #127 (@sgugger)
- Fix fp16 by converting outputs back to FP32 #134 (@sgugger)
- Add caveat on weight-tying on TPUs #138 (@sgugger)
- Add optimizer not stepped property #139 (@sgugger)

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.4.0)

---

## v0.3.0: v0.3.0 Notebook launcher and multi-node training
**Published:** 2021-04-29

# v0.3.0 Notebook launcher and multi-node training

## Notebook launcher

After doing all the data preprocessing in your notebook, you can launch your training loop using the new `notebook_launcher` functionality. This is especially useful for Colab or Kaggle with TPUs! [Here](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/accelerate/simple_nlp_example.ipynb) is an example on Colab (don't forget to select a TPU runtime).

This launcher also works if you have multiple GPUs on your machine. You just have to pass along `num_processes=your_number_of_gpus` in the call to `notebook_launcher`.

- Notebook launcher #44 (@sgugger)
- Add notebook/colab example #52 (@sgugger)
- Support for multi-GPU in notebook_launcher #56 (@sgugger)

## Multi-node training

Our multi-node training test setup was flawed and the previous releases of 🤗 Accelerate were not working for multi-node distributed training. This is all fixed now and we have ensured to have more robust tests!

- fix cluster.py indent error #35 (@JTT94)
- Set all defaults from config in launcher #38 (@sgugger)
- Fix port in config creation #50 (@sgugger)

## Various bug fixes

- Fix typos in examples README #28 (@arjunchandra)
- Fix load from config #31 (@sgugger)
- docs: minor spelling tweaks #33 (@brettkoonce)
- Add `set_to_none` to AcceleratedOptimizer.zero_grad #43 (@sgugger)
- fix #53 #54 (@Guitaricet)
- update launch.py #58 (@Jesse1eung)

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.3.0)

---

## v0.2.1: v0.2.1: Patch release
**Published:** 2021-04-19

Fix a bug preventing the load of a config with `accelerate launch`

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.2.1)

---

## v0.2.0: v0.2.0 SageMaker launcher
**Published:** 2021-04-15

# v0.2.0 SageMaker launcher

## SageMaker launcher

It's now possible to launch your training script on AWS instances using SageMaker via `accelerate launch`.

- Launch script on SageMaker #26 (@philschmid )
- Add defaults for compute_environmnent #23 (@sgugger )
- Add Configuration setup for SageMaker #17 (@philschmid )

## Kwargs handlers

To customize how the different objects used for mixed precision or distributed training are instantiated, a new API called `KwargsHandler` is added. This allows the user to pass along the kwargs that will be passed to those objects if used (and it is ignored if those are not used in the current setup, so the script can still run on any kind of setup).

- Add KwargsHandlers #15 (@sgugger )

## Pad across processes

Trying to gather tensors that are not of the same size across processes resulted in a process hang, a new method `Accelerator.pad_across_processes` has been added to help with that.

- Add utility to pad tensor across processes to max length #19 (@sgugger )

## Various bug fixes

- added thumbnail #25 (@philschmid )
- Cleaner diffs in README and index #22 (@sgugger )
- Use proper size #21 (@sgugger )
- Alternate diff #20 (@sgugger )
- Add YAML config support #16 (@sgugger )
- Don't error on non-Tensors objects in move to device #13 (@sgugger )
- Add CV example #10 (@sgugger )
- Readme clean-up #9 (@thomwolf )
- More flexible RNG synchronization #8 (@sgugger )
- Fix typos and tighten grammar in README #7 (@lewtun )
- Update README.md #6 (@voidful )
- Fix TPU training in example #4 (@thomwolf )
- Fix example name in README #3 (@LysandreJik )

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.2.0)

---

## v0.1.0: v0.1.0 Initial release
**Published:** 2021-03-05

Initial release of 🤗 Accelerate. Checkout the main [README](https://github.com/huggingface/accelerate) or the [docs](https://huggingface.co/docs/accelerate/) to learn more about it!

[View on GitHub](https://github.com/huggingface/accelerate/releases/tag/v0.1.0)

---

