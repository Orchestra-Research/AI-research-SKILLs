# Releases

Version history for this repository (28 releases).

## v0.17.1: 0.17.1
**Published:** 2025-08-21

This patch release contains a few fixes (via #2710) for the newly introduced [`target_parameters`](https://huggingface.co/docs/peft/main/en/developer_guides/lora#targeting-nnparameter-directly) feature, which allows LoRA to target `nn.Parameter`s directly (useful for mixture of expert layers). Most notably:

- PEFT no longer removes possibly [existing parametrizations](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html) from the parameter.
- Adding multiple adapters (via `model.add_adapter` or `model.load_adapter`) did not work correctly. Since a solution is not trivial, PEFT now raises an error to prevent this situation.

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.17.1)

---

## v0.17.0: 0.17.0: SHiRA, MiSS, LoRA for MoE, and more
**Published:** 2025-08-01

# Highlights

<img width="1248" height="560" alt="peft-v0 17 0" src="https://github.com/user-attachments/assets/a206c099-10ee-4c13-80c1-0de7ed1df5cf" />

## New Methods

### SHiRA

@kkb-code contributed [Sparse High Rank Adapters](https://huggingface.co/docs/peft/main/en/package_reference/shira) (SHiRA, [paper](https://huggingface.co/papers/2406.13175)) which promise to offer a potential gain in performance over LoRAs - especially the concept loss when using multiple adapters is improved. Since the adapters only train on 1-2% of the weights and are inherently sparse, switching between adapters may be cheaper than with LoRAs. (#2584)

### MiSS

 @JL-er added a new PEFT method, MiSS ([Matrix Shard Sharing](https://arxiv.org/abs/2409.15371)) in #2604. This method is an evolution of [Bone](https://huggingface.co/docs/peft/package_reference/bone), which, according to our [PEFT method comparison benchmark](https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison), gives excellent results when it comes to performance and memory efficiency. If you haven't tried it, you should do so now.

At the same time, Bone will be deprecated in favor of MiSS and will be removed in PEFT v0.19.0. If you already have a Bone checkpoint, you can use [`scripts/convert-bone-to-miss.py`](https://github.com/huggingface/peft/tree/main/scripts/convert-bone-to-miss.py) to convert it into a MiSS checkpoint and proceed with training using MiSS.

## Enhancements

### LoRA for `nn.Parameter`

LoRA is now able to target `nn.Parameter` directly (#2638, #2665)! Ever had this complicated `nn.Module` with promising parameters inside but it was too custom to be supported by your favorite fine-tuning library? No worries, now you can target `nn.Parameters` directly using the [`target_parameters`](https://huggingface.co/docs/peft/main/en/developer_guides/lora#targeting-nnparameter-directly) config attribute which works similarly to `target_modules`.

This option can be especially useful for models with **Mixture of Expert** (MoE) layers, as those often use `nn.Parameter`s directly and cannot be targeted with `target_modules`. For example, for the [Llama4 family of models](https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164), use the following config to target the MoE weights:

```python
config = LoraConfig(
    ...,
    target_modules=[],  # <= prevent targeting any modules
    target_parameters=["feed_forward.experts.down_proj", "feed_forward.experts.gate_up_proj"],
)
```

Note that this feature is still experimental as it comes with a few caveats and therefore might change in the future. Also, MoE weights with many experts can be quite huge, so expect a higher memory usage than compared to targeting normal `nn.Linear` layers.

### Injecting adapters based on a `state_dict`

Sometimes, it is possible that there is a PEFT adapter checkpoint but the corresponding PEFT config is not known for whatever reason. To inject the PEFT layers for this checkpoint, you would usually have to reverse-engineer the corresponding PEFT config, most notably the `target_modules` argument, based on the `state_dict` from the checkpoint. This can be cumbersome and error prone. To avoid this, it is also possible to call `inject_adapter_in_model` and pass the loaded `state_dict` as an argument:

```python
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model

model = ...
state_dict = load_file(<path-to-safetensors-file>)
lora_config = LoraConfig()  # <= no need to specify further
model = inject_adapter_in_model(lora_config, model, state_dict=state_dict)
```

Find more on [`state_dict` based injection in the docs](https://huggingface.co/docs/peft/main/en/developer_guides/low_level_api#injection-based-on-a-statedict).

## Changes

### Compatibility

A bug in prompt learning methods caused `modules_to_save` to be ignored. Especially classification tasks are affected since they usually add the classification/score layer to `modules_to_save`. In consequence, these layers were neither trained nor stored after training. This has been corrected now. (#2646)

### All Changes

* Bump version to 0.16.1.dev0 after release by @BenjaminBossan in https://github.com/huggingface/peft/pull/2632
* FEAT: Add GH action to deploy method comparison app by @BenjaminBossan in https://github.com/huggingface/peft/pull/2625
* enable FSDP example for model `hugging-quants/Meta-Llama-3.1-8B-Instr… by @kaixuanliu in https://github.com/huggingface/peft/pull/2626
* FIX: Create mask function signature change in transformers 4.53.1 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2633
* FIX: Correctly skip AWQ test based on torch version by @BenjaminBossan in https://github.com/huggingface/peft/pull/2631
* FIX: Faulty OFT parameter device test by @BenjaminBossan in https://github.com/huggingface/peft/pull/2630
* Fix #2634: Allow peft_type to be a string by @githubnemo in https://github.com/huggingface/peft/pull/2635
* SHiRA Adapters by @kkb-code in https://github.com/huggingface/peft/pull/2584
* FIX: Prompt learning methods modules_to_save issue by @BenjaminBossan in https://github.com/huggingface/peft/pull/2646
* FIX: Error in workflow file to deploy method comparison app by @BenjaminBossan in https://github.com/huggingface/peft/pull/2645
* FEAT Allow LoRA  to target nn.Parameter by @BenjaminBossan in https://github.com/huggingface/peft/pull/2638
* Update BibTeX entry by @cx-alberto-simoes in https://github.com/huggingface/peft/pull/2659
* FIX Prefix tuning after transformers PR 38635 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2662
* make method comparison device agnostic, so it can expand to more accelerators like XPU by @yao-matrix in https://github.com/huggingface/peft/pull/2610
* Update tokenizer parameter in sfttrainer across multiple examples by @gapsong in https://github.com/huggingface/peft/pull/2664
* Update lora.md by @qgallouedec in https://github.com/huggingface/peft/pull/2666
* GPT2 compatible version of LLama-Adapters by @efraimdahl in https://github.com/huggingface/peft/pull/2643
* Method Comparison: Improve formatting/layout of table by @githubnemo in https://github.com/huggingface/peft/pull/2670
* ENH: Targeting multiple parameters on the same module by @BenjaminBossan in https://github.com/huggingface/peft/pull/2665
* Update extending vocab docs by @githubnemo in https://github.com/huggingface/peft/pull/2669
* FIX Failing target_parameters param usage count by @BenjaminBossan in https://github.com/huggingface/peft/pull/2676
* Fix trainable tokens with fsdp by @BenjaminBossan in https://github.com/huggingface/peft/pull/2681
* FIX: Small fixes to target_parameters by @BenjaminBossan in https://github.com/huggingface/peft/pull/2677
* TST: Add more HF Hub model caching by @BenjaminBossan in https://github.com/huggingface/peft/pull/2682
* FIX: Missing device map for facebook/opt-125m by @BenjaminBossan in https://github.com/huggingface/peft/pull/2675
* Fix not detecting regex-targeted embedding layer by @githubnemo in https://github.com/huggingface/peft/pull/2649
* Add MiSS as a replacement for Bone. by @JL-er in https://github.com/huggingface/peft/pull/2604
* [WIP] ENH: Adapter injection based on state_dict by @BenjaminBossan in https://github.com/huggingface/peft/pull/2637
* Release 0.17.0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2691

## New Contributors
* @kaixuanliu made their first contribution in https://github.com/huggingface/peft/pull/2626
* @kkb-code made their first contribution in https://github.com/huggingface/peft/pull/2584
* @cx-alberto-simoes made their first contribution in https://github.com/huggingface/peft/pull/2659
* @efraimdahl made their first contribution in https://github.com/huggingface/peft/pull/2643

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.16.0...v0.17.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.17.0)

---

## v0.16.0: 0.16.0: LoRA-FA, RandLoRA, C³A, and much more
**Published:** 2025-07-03

# Highlights

![peft-v0 16 0](https://github.com/user-attachments/assets/fcced016-7237-410f-b569-53e0f932208d)

## New Methods

### LoRA-FA

In #2468, @AaronZLT added the [LoRA-FA optimizer](https://huggingface.co/docs/peft/main/en/developer_guides/lora#lora-fa-optimizer) to PEFT. This optimizer is based on `AdamW` and it increases memory efficiency of LoRA training. This means that you can train LoRA with less memory, or, with the same memory budget, use higher LoRA ranks, potentially getting better results.

### RandLoRA

Thanks to @PaulAlbert31, a new PEFT method called [`RandLoRA`](https://huggingface.co/docs/peft/main/en/package_reference/randlora) was added to PEFT (#2464). Similarly to VeRA, it uses non-learnable random low rank matrices that are combined through learnable matrices. This way, RandLoRA can approximate full rank updates of the weights. Training models quantized with bitsandbytes is supported.

### C³A

@Phoveran added [Circular Convolution Adaptation](https://huggingface.co/docs/peft/main/en/package_reference/c3a), C3A, in #2577. This new PEFT method can overcome the limit of low rank adaptations as seen e.g. in LoRA while still promising to be fast and memory efficient.

## Enhancements

Thanks to @gslama12 and @SP1029, LoRA now supports `Conv2d` layers with `groups != 1`. This requires the rank `r` being divisible by `groups`. See #2403 and #2567 for context.

@dsocek added support for Intel Neural Compressor (INC) quantization to LoRA in #2499.

DoRA now supports `Conv1d` layers thanks to @EskildAndersen (#2531).

Passing `init_lora_weights="orthogonal"` now enables orthogonal weight initialization for LoRA (#2498).

@gapsong brought us Quantization-Aware LoRA training in #2571. This can make QLoRA training more efficient, please check the [included example](https://github.com/huggingface/peft/tree/main/examples/qalora_finetuning). Right now, only GPTQ is supported.

There has been a big refactor of Orthogonal Finetuning, [OFT](https://huggingface.co/docs/peft/package_reference/oft), thanks to @zqiu24 (#2575). This makes the PEFT method run more quickly and require less memory. It is, however, incompatible with old OFT checkpoints. If you have old OFT checkpoints, either pin the PEFT version to `<0.16.0` or retrain it with the new PEFT version.

Thanks to @keepdying, LoRA hotswapping with compiled models no longer leads to CUDA graph re-records (#2611).

# Changes

## Compatibility

- #2481: The value of `required_grads_` of `modules_to_save` is now set to `True` when used directly with `inject_adapter`. This is relevant for PEFT integrations, e.g. Transformers or Diffusers.
- Due to a [big refactor of vision language models]( https://github.com/huggingface/transformers/pull/37033) (VLMs) in Transformers, the model architecture has been slightly adjusted. One consequence of this is that if you use a PEFT prompt learning method that is applied to `vlm.language_model`, it will no longer work, please apply it to `vlm` directly (see #2554 for context). Morever, the refactor results in different checkpoints. We managed to ensure _backwards compatability_ in PEFT, i.e. old checkpoints can be loaded successfully. There is, however, no _forward compatibility_, i.e. loading checkpoints trained after the refactor is not possible with package versions from before the refactor. In this case, you need to upgrade PEFT and transformers. More context in #2574.
- #2579: There have been bigger refactors in Transformers concerning attention masks. This required some changes on the PEFT side which can affect prompt learning methods. For prefix tuning specifically, this can result in numerical differences but overall performance should be the same. For other prompt learning methods, numerical values should be the same, except if the base model uses 4d attention masks, like Gemma. If you load old prompt learning checkpoints, please double-check that they still perform as expected, especially if they're trained on Gemma or similar models. If not, please re-train them or pin PEFT and transformers to previous versions (`<0.16.0` and `<4.52.0`, respectively).

## All Changes

* Bump version and minor instruction fix by @githubnemo in https://github.com/huggingface/peft/pull/2439
* FIX for ConvNd layers using the groups argument. by @gslama12 in https://github.com/huggingface/peft/pull/2403
* DOC: Tip on how to merge with DeepSpeed by @BenjaminBossan in https://github.com/huggingface/peft/pull/2446
* Fix incorrect link in docs by @kenning in https://github.com/huggingface/peft/pull/2444
* Fix typos by @omahs in https://github.com/huggingface/peft/pull/2447
* Refactor to better support LoRA variants by @BenjaminBossan in https://github.com/huggingface/peft/pull/2443
* enable 5 test cases on XPU by @yao-matrix in https://github.com/huggingface/peft/pull/2442
* FIX: Faulty test that results in nan weights by @BenjaminBossan in https://github.com/huggingface/peft/pull/2448
* Fix sft example script trl and env var by @BenjaminBossan in https://github.com/huggingface/peft/pull/2454
* LoRA variant init now also receives kwargs by @BenjaminBossan in https://github.com/huggingface/peft/pull/2455
* Fix #2450: Revamp adapter_state_dict_* methods by @githubnemo in https://github.com/huggingface/peft/pull/2456
* Method comparison evaluation suite by @githubnemo in https://github.com/huggingface/peft/pull/2395
* Bump version to reflect patch release by @githubnemo in https://github.com/huggingface/peft/pull/2461
* The paper on the Bone structure has been updated by @JL-er in https://github.com/huggingface/peft/pull/2312
* CI: More caching in tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2472
* fix gpu tests by @jiqing-feng in https://github.com/huggingface/peft/pull/2471
* Fix compare results by @jiqing-feng in https://github.com/huggingface/peft/pull/2473
* fix error_factor for xpu by @jiqing-feng in https://github.com/huggingface/peft/pull/2475
* Fix: Multiple PEFT methods have issues with models loaded in float16 or bfloat16 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2433
* TST Refactor tests to make them simpler by @BenjaminBossan in https://github.com/huggingface/peft/pull/2462
* Use Python 3.9 as RUFF target version and apply fixes by @cyyever in https://github.com/huggingface/peft/pull/2483
* FIX Deleting adapters on auxiliary modules by @BenjaminBossan in https://github.com/huggingface/peft/pull/2466
* fix args by @real-zhangzhe in https://github.com/huggingface/peft/pull/2474
* ENH Add default target_modules for Llama4 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2480
* [Feature Request] Add LoRA-FA to PEFT by @AaronZLT in https://github.com/huggingface/peft/pull/2468
* TST Refactor (continued) of encoder tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2478
* FIX: Error when merging LoRA bias with scale != 1 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2489
* FIX: X-LoRA error when targeting different modules by @BenjaminBossan in https://github.com/huggingface/peft/pull/2488
* Fix: the evaluation_strategy is deprecated by @yuanwu2017 in https://github.com/huggingface/peft/pull/2487
* Testing common uses situational HF_HUB_OFFLINE by @githubnemo in https://github.com/huggingface/peft/pull/2490
* MNT: Update HF Hub download kwargs by @BenjaminBossan in https://github.com/huggingface/peft/pull/2492
* FIX Multi GPU tests: explicit device map by @BenjaminBossan in https://github.com/huggingface/peft/pull/2484
* Fix #2477: Regression accessing `modules_to_save` by @githubnemo in https://github.com/huggingface/peft/pull/2481
* make test_lora_use_dora_linear pass on XPU by @yao-matrix in https://github.com/huggingface/peft/pull/2493
* TST: AQLM test no longer x-fails by @BenjaminBossan in https://github.com/huggingface/peft/pull/2506
* TST make 3 flaky test cases always pass on XPU by @yao-matrix in https://github.com/huggingface/peft/pull/2503
* FIX: CPT should not be tested with sequence classification by @BenjaminBossan in https://github.com/huggingface/peft/pull/2507
* Update Docker image builds for torch 2.7+cu126 by @matthewdouglas in https://github.com/huggingface/peft/pull/2514
* Feature: RandLora integration into peft by @PaulAlbert31 in https://github.com/huggingface/peft/pull/2464
* LORA/MODEL: Use max rank of pattern for `add_weighted_adapter` by @Beinsezii in https://github.com/huggingface/peft/pull/2512
* fix typo for skipping test by @jiqing-feng in https://github.com/huggingface/peft/pull/2519
* docs typo: fix links by @imba-tjd in https://github.com/huggingface/peft/pull/2517
* Add INC dispatcher by @dsocek in https://github.com/huggingface/peft/pull/2499
* ENH: Add default Qwen3 target modules by @BenjaminBossan in https://github.com/huggingface/peft/pull/2522
* MNT: Pin GitHub action hashes for security by @BenjaminBossan in https://github.com/huggingface/peft/pull/2521
* TST: Refactor remaining common tests to use pytest by @BenjaminBossan in https://github.com/huggingface/peft/pull/2491
* ENH: Add tests, docs, types for scaling methods by @BenjaminBossan in https://github.com/huggingface/peft/pull/2526
* TST Mark AutoAWQ as xfail for now by @BenjaminBossan in https://github.com/huggingface/peft/pull/2529
* FIX Prompt learning issue with 4d attention mask by @BenjaminBossan in https://github.com/huggingface/peft/pull/2458
* FIX: Use correct argument name in MultiheadAttention forward by @BenjaminBossan in https://github.com/huggingface/peft/pull/2510
* Method comparison: Support more options for the optimizer by @BenjaminBossan in https://github.com/huggingface/peft/pull/2479
* Randlora documentation and some example usage by @PaulAlbert31 in https://github.com/huggingface/peft/pull/2524
* added support for Conv1d for DoRA by @EskildAndersen in https://github.com/huggingface/peft/pull/2531
* Fix #2535: Prevent adapters targeting themselves by @githubnemo in https://github.com/huggingface/peft/pull/2539
* Fix typos by @omahs in https://github.com/huggingface/peft/pull/2544
* Use HF Papers by @qgallouedec in https://github.com/huggingface/peft/pull/2542
* Address changes in transformers VLM architecture by @githubnemo in https://github.com/huggingface/peft/pull/2554
* CI: Handle errors with MacOS and transformers by @BenjaminBossan in https://github.com/huggingface/peft/pull/2561
* Fix zizmor warnings about unpinned docker images by @githubnemo in https://github.com/huggingface/peft/pull/2565
* align xpu behavior w/ cuda by @yao-matrix in https://github.com/huggingface/peft/pull/2551
* LORA/MODEL: Discard `rank_pattern`, `rank_alpha` for `add_weighted_adapter` by @Beinsezii in https://github.com/huggingface/peft/pull/2550
* fix inconsistent variable naming in load_adapter by @pranav-gade in https://github.com/huggingface/peft/pull/2553
* Prevent applying LoRA to disallowed modules in Mamba-based architectures by @dhiaEddineRhaiem in https://github.com/huggingface/peft/pull/2562
* TST: Refactor unittest to pytest style custom tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2573
* Simple variant application test by @githubnemo in https://github.com/huggingface/peft/pull/2572
* `prepare_model_for_gradient_checkpointing` protected to public by @qgallouedec in https://github.com/huggingface/peft/pull/2569
* Optimize isinstance Check in LoraParallelLinear by @JavaZeroo in https://github.com/huggingface/peft/pull/2576
* FIX: Generation nightly CI failing due to gemma by @BenjaminBossan in https://github.com/huggingface/peft/pull/2580
* FIX: Correctly determine no_split_modules by @BenjaminBossan in https://github.com/huggingface/peft/pull/2570
* ENH: Orthogonal LoRA layer initialization (2) by @BenjaminBossan in https://github.com/huggingface/peft/pull/2498
* ENH: Method comparison improve logging by @BenjaminBossan in https://github.com/huggingface/peft/pull/2591
* DOC Update README, contributing.md, GH templates by @BenjaminBossan in https://github.com/huggingface/peft/pull/2588
* Input sanitizer for benchmark result renderer by @githubnemo in https://github.com/huggingface/peft/pull/2594
* Add Makefile + results for MetaMathQA task by @githubnemo in https://github.com/huggingface/peft/pull/2593
* Track number of (trainable) parameters for MetaMathQA by @githubnemo in https://github.com/huggingface/peft/pull/2598
* ENH: Method comparison allow full finetuning by @BenjaminBossan in https://github.com/huggingface/peft/pull/2597
* enable some left out cases on XPU, all enabled cases pass  by @yao-matrix in https://github.com/huggingface/peft/pull/2596
* FIX: Transformers VLM architecture changes by @BenjaminBossan in https://github.com/huggingface/peft/pull/2574
* Enable XPU regression tests with deterministic by @jiqing-feng in https://github.com/huggingface/peft/pull/2600
* Results with number of parameters + full fine tuning by @githubnemo in https://github.com/huggingface/peft/pull/2602
* Add support for Quantization-Aware Low-Rank Adaptation (QALoRA) by @gapsong in https://github.com/huggingface/peft/pull/2571
* OFT: several improvements to make OFT faster and more memory efficient by @zqiu24 in https://github.com/huggingface/peft/pull/2575
* FIX: Trainable tokens error with DeepSpeed ZeRO3 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2605
* ENH Method comparison: temporary and cancelled result files include timestamp by @BenjaminBossan in https://github.com/huggingface/peft/pull/2617
* FIX: Avoid CUDA Graph re-record when hotswapping LoRAs. by @keepdying in https://github.com/huggingface/peft/pull/2611
* FIX Account for attention mask being a dict, fix generate issues with gemma by @BenjaminBossan in https://github.com/huggingface/peft/pull/2579
* TST Skip (more) failing MacOS tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2620
* FIX Update signature for resolve_lora_variant by @BenjaminBossan in https://github.com/huggingface/peft/pull/2618
* [FEAT] Add C3A Support by @Phoveran in https://github.com/huggingface/peft/pull/2577
* FIX for #2549 - modify lora_B definition for conv layers with groups by @SP1029 in https://github.com/huggingface/peft/pull/2567
* FIX: Type annotation error in PEFT method comparison script by @BenjaminBossan in https://github.com/huggingface/peft/pull/2628
* FIX CI Multi-GPU tests require device_map by @BenjaminBossan in https://github.com/huggingface/peft/pull/2612
* TST Update diffusers hotswap tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2619
* Auto-tagging of PEFT models by @githubnemo in https://github.com/huggingface/peft/pull/2599

## New Contributors
* @kenning made their first contribution in https://github.com/huggingface/peft/pull/2444
* @omahs made their first contribution in https://github.com/huggingface/peft/pull/2447
* @yao-matrix made their first contribution in https://github.com/huggingface/peft/pull/2442
* @cyyever made their first contribution in https://github.com/huggingface/peft/pull/2483
* @real-zhangzhe made their first contribution in https://github.com/huggingface/peft/pull/2474
* @AaronZLT made their first contribution in https://github.com/huggingface/peft/pull/2468
* @yuanwu2017 made their first contribution in https://github.com/huggingface/peft/pull/2487
* @PaulAlbert31 made their first contribution in https://github.com/huggingface/peft/pull/2464
* @Beinsezii made their first contribution in https://github.com/huggingface/peft/pull/2512
* @imba-tjd made their first contribution in https://github.com/huggingface/peft/pull/2517
* @dsocek made their first contribution in https://github.com/huggingface/peft/pull/2499
* @EskildAndersen made their first contribution in https://github.com/huggingface/peft/pull/2531
* @pranav-gade made their first contribution in https://github.com/huggingface/peft/pull/2553
* @dhiaEddineRhaiem made their first contribution in https://github.com/huggingface/peft/pull/2562
* @JavaZeroo made their first contribution in https://github.com/huggingface/peft/pull/2576
* @gapsong made their first contribution in https://github.com/huggingface/peft/pull/2571
* @keepdying made their first contribution in https://github.com/huggingface/peft/pull/2611
* @SP1029 made their first contribution in https://github.com/huggingface/peft/pull/2567

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.15.2...v0.16.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.16.0)

---

## v0.15.2: v0.15.2
**Published:** 2025-04-15

This patch fixes a bug that resulted in prompt learning methods like P-tuning not to work (#2477).

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.15.2)

---

## v0.15.1: v0.15.1
**Published:** 2025-03-27

This patch includes a fix for #2450. In this bug `modules_to_save` was not handled correctly when used in conjunction with DeepSpeed ZeRO stage 3 which resulted in those modules being placeholder values in the saved checkpoints.

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.15.0...v0.15.1

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.15.1)

---

## v0.15.0: v0.15.0
**Published:** 2025-03-19

# Highlights

![peft-v0 15 0](https://github.com/user-attachments/assets/4095edca-7269-403f-be2e-2ef95d6ed474)

## New Methods

### CorDA: Context-Oriented Decomposition Adaptation

@iboing and @5eqn contributed [CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning](https://arxiv.org/abs/2406.05223) . This task-driven initialization method has [two modes](https://huggingface.co/docs/peft/main/en/developer_guides/lora#corda), knowledge-preservation and instruction-preservation, both using external data to select ranks intelligently. The former can be used to select those ranks that correspond to weights not affiliated with knowledge from, say, a QA dataset. The latter can be used to select those ranks that correspond most to the task at hand (e.g., a classification task). (#2231)

### Trainable Tokens: Selective token update
The new [Trainable Tokens](https://huggingface.co/docs/peft/main/en/package_reference/trainable_tokens) tuner allows for selective training of tokens without re-training the full embedding matrix, e.g. when adding support for reasoning / thinking tokens. This is a lot more memory efficient and the saved checkpoint is much smaller. It can be used standalone or [in conjunction with LoRA adapters](https://huggingface.co/docs/peft/main/en/developer_guides/lora#efficiently-train-tokens-alongside-lora) by passing `trainable_token_indices` to `LoraConfig`. (#2376)

## Enhancements

LoRA now supports targeting multihead attention modules (but for now only those with `_qkv_same_embed_dim=True`). These modules were tricky as they may expose linear submodules but won't use their forward methods, therefore needing explicit support. (#1324)

[Hotswapping](https://huggingface.co/docs/peft/main/en/package_reference/hotswap) now allows different alpha scalings and ranks without recompilation of the model when the model is prepared using a call to `prepare_model_for_compiled_hotswap()` before compiling the model. (#2177)

[GPTQModel](https://github.com/ModelCloud/GPTQModel) support was added in #2247 as a replacement for AutoGPTQ which is not maintained anymore.

## Changes
- It's now possible to use `all-linear` as `target_modules` for custom (non-transformers) models (#2267). With this change comes a bugfix where it was possible that non-linear layers were selected when they shared the same name with a linear layer (e.g., `bar.foo` and `baz.foo`).
- The internal tuner API was refactored to make method registration easier. With this change the number of changes to numerous files is reduced to a single `register_peft_method()` call. (#2282)
- `PEFT_TYPE_TO_MODEL_MAPPING` is now deprecated and should not be relied upon. Use `PEFT_TYPE_TO_TUNER_MAPPING` instead. (#2282)
- Mixed adapter batches can now be used in conjunction with beam search. (#2287)
- It was possible that `modules_to_save` keys wrongly matched parts of the state dict if the key was a substring of another key (e.g., `classifier` and `classifier2`). (#2334)
- Auto-casting of the input dtype to the LoRA adapter dtype can now be disabled via `disable_input_dtype_casting=True`. (#2353)
- The config parameters `rank_pattern` and `alpha_pattern` used by many adapters now supports matching full paths as well by specifying the pattern with a caret in front, for example: `^foo` to target `model.foo` but not `model.bar.foo`. (#2419)
- AutoPeftModels do not reduce the embedding size anymore if the tokenizer size differs from the embedding size. Only if there are more tokens in the tokenizer than in the embedding matrix, the matrix will be resized. This is to prevent resizing of embedding matrices in models that have 'spare' tokens built-in. (#2427)

# What's Changed
* FIX: Ensure Device Compatibility for BOFT Forward/Merging by @d-kleine in https://github.com/huggingface/peft/pull/2242
* MNT: Bump version to 0.14.1.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2263
* ENH: fix library interface by @bluenote10 in https://github.com/huggingface/peft/pull/2265
* FIX: Add warning for `adapter_name` conflict with tuner  by @pzdkn in https://github.com/huggingface/peft/pull/2254
* ENH: FIX: Allow `"all-linear"` to target custom models by @BenjaminBossan in https://github.com/huggingface/peft/pull/2267
* MNT: apply sorting of exported symbols in `__all__` by @bluenote10 in https://github.com/huggingface/peft/pull/2280
* MNT: apply sorting of imports by @bluenote10 in https://github.com/huggingface/peft/pull/2279
* FIX: Adoption prompt: New way to obtain position embeddings by @BenjaminBossan in https://github.com/huggingface/peft/pull/2276
* FIX: Int8 check for torchao v0.7.0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2284
* FEAT: Adding CorDA as an optional initialization method of LoRA by @iboing in https://github.com/huggingface/peft/pull/2231
* FIX: typo in lora `config.py` by @innerlee in https://github.com/huggingface/peft/pull/2297
* DOC: Added information regarding freezing the base model in `prepare_model_for_kbit_training` docstring by @NilBiescas in https://github.com/huggingface/peft/pull/2305
* DOC: add `resize_token_embeddings` to docs by @bingwork in https://github.com/huggingface/peft/pull/2290
* FIX: Make CorDA example work by @5eqn in https://github.com/huggingface/peft/pull/2300
* FIX: #2295: Warn when user reloads modified model by @githubnemo in https://github.com/huggingface/peft/pull/2306
* ENH: Extend usage for OLoRA finetune script by @jiqing-feng in https://github.com/huggingface/peft/pull/2308
* CI: Add zizmor for CI (security) linting by @githubnemo in https://github.com/huggingface/peft/pull/2288
* FEAT: Add LoRA multihead attention module by @BenjaminBossan in https://github.com/huggingface/peft/pull/1324
* DOC: Updated documentation for `get_peft_model()` for in-place base model modification by @d-kleine in https://github.com/huggingface/peft/pull/2313
* FIX: Prefix tuning test w/ rotary embedding on multi GPU by @BenjaminBossan in https://github.com/huggingface/peft/pull/2311
* FIX: Adaption prompt errors after changes from transformers #35235 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2314
* FIX: Package checks for torchao, EETQ by @BenjaminBossan in https://github.com/huggingface/peft/pull/2320
* Refactor: PEFT method registration function by @BenjaminBossan in https://github.com/huggingface/peft/pull/2282
* FIX: `low_cpu_mem_usage=True` with 8bit bitsandbytes by @BenjaminBossan in https://github.com/huggingface/peft/pull/2325
* FIX: Reinstate `PEFT_TYPE_TO_MODEL_MAPPING` variable with deprecation by @BenjaminBossan in https://github.com/huggingface/peft/pull/2328
* FIX: reduce CorDA memory consumption + docs by @5eqn in https://github.com/huggingface/peft/pull/2324
* MNT: React on new zizmor version findings by @githubnemo in https://github.com/huggingface/peft/pull/2331
* TST: make cuda-only tests device-agnostic by @faaany in https://github.com/huggingface/peft/pull/2323
* FIX: Generating with mixed adapter batches and with beam search enabled by @BenjaminBossan in https://github.com/huggingface/peft/pull/2287
* FIX: Bug with `modules_to_save` loading if substring by @BenjaminBossan in https://github.com/huggingface/peft/pull/2334
* FIX: Add missing attributes to MultiheadAttention by @BenjaminBossan in https://github.com/huggingface/peft/pull/2335
* FIX: for zizmor permission warnings by @githubnemo in https://github.com/huggingface/peft/pull/2338
* CI: Attempt at adding a cache for models by @githubnemo in https://github.com/huggingface/peft/pull/2327
* FIX: Avoid needless copy from `modules_to_save` by @BenjaminBossan in https://github.com/huggingface/peft/pull/2220
* DOC: Add entry to solve unknown config argument by @BenjaminBossan in https://github.com/huggingface/peft/pull/2340
* FEAT: add gptqmodel support by @jiqing-feng in https://github.com/huggingface/peft/pull/2247
* MNT: Update ruff to v0.9.2 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2343
* TST: Update `torch.compile` tests and docs by @BenjaminBossan in https://github.com/huggingface/peft/pull/2332
* FIX: Documentation & error checking for AdaLoRA timing by @githubnemo in https://github.com/huggingface/peft/pull/2341
* DOC: Better document init_lora_weights=False option by @BenjaminBossan in https://github.com/huggingface/peft/pull/2347
* ENH: Adding Lora implementation for `nn.Conv1d` by @CCLDArjun in https://github.com/huggingface/peft/pull/2333
* FIX: Failing AdaLoRA GPU test by @BenjaminBossan in https://github.com/huggingface/peft/pull/2349
* ENH: Improve invalid peft config error message by @thedebugger in https://github.com/huggingface/peft/pull/2346
* TST: Use different diffusion model for testing by @BenjaminBossan in https://github.com/huggingface/peft/pull/2345
* CI: Use locked install for zizmor by @githubnemo in https://github.com/huggingface/peft/pull/2350
* DOC: fix links to PEFT guides by @makelinux in https://github.com/huggingface/peft/pull/2357
* DOC: rename link to PEFT Quicktour by @makelinux in https://github.com/huggingface/peft/pull/2358
* ENH: Allow disabling input dtype casting for LoRA by @BenjaminBossan in https://github.com/huggingface/peft/pull/2353
* ENH: Hotswap allow different alpha scalings and ranks by @BenjaminBossan in https://github.com/huggingface/peft/pull/2177
* DOC: Fix links to boft by @makelinux in https://github.com/huggingface/peft/pull/2365
* DOC: Explain uninitialized weights warning by @BenjaminBossan in https://github.com/huggingface/peft/pull/2369
* ENH: Optimization for ConvNd if dropout=0. by @gslama12 in https://github.com/huggingface/peft/pull/2371
* FIX: Small fixes to hotswapping by @BenjaminBossan in https://github.com/huggingface/peft/pull/2366
* ENH: `prepare_model_for_compiled_hotswap` raises when no adapter was found by @BenjaminBossan in https://github.com/huggingface/peft/pull/2375
* FIX: Ensure `hf_hub_download` arguments are used when loading locally by @henryzhengr in https://github.com/huggingface/peft/pull/2373
* FIX: Avoid caching in X-LoRA generate by @BenjaminBossan in https://github.com/huggingface/peft/pull/2384
* CI: Skip audio test on single GPU CI by @BenjaminBossan in https://github.com/huggingface/peft/pull/2380
* SEC: Bump transformers version used in examples by @BenjaminBossan in https://github.com/huggingface/peft/pull/2374
* FIX: Failing single GPU tests related to hotswapping by @BenjaminBossan in https://github.com/huggingface/peft/pull/2385
* ENH: Make hotswap error on compile optional by @BenjaminBossan in https://github.com/huggingface/peft/pull/2393
* FEAT: Standalone Custom Tokens Tuner and integrated into LoRA by @githubnemo in https://github.com/huggingface/peft/pull/2376
* FIX: GPTQModel LoRA Compat by @Qubitium in https://github.com/huggingface/peft/pull/2404
* FIX: Model with nested `all-linear` target modules by @BenjaminBossan in https://github.com/huggingface/peft/pull/2391
* FIX: Bug with `PeftConfig.from_pretrained` by @BenjaminBossan in https://github.com/huggingface/peft/pull/2397
* ENH: Add simple script to estimate train memory by @BenjaminBossan in https://github.com/huggingface/peft/pull/2378
* CI: Use new slack secret token name by @githubnemo in https://github.com/huggingface/peft/pull/2409
* ENH: Trainable Tokens: Support for Weight Tying by @githubnemo in https://github.com/huggingface/peft/pull/2399
* TST: enable BNB tests on XPU by @faaany in https://github.com/huggingface/peft/pull/2396
* FIX: Reset the FP32 matmul precision in tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2411
* TST: add the missing `.eval()` for inference by @faaany in https://github.com/huggingface/peft/pull/2408
* FIX: Revert optimization for LoRA scaling == 1 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2416
* ENH: Extend the regex for rank/alpha pattern by @BenjaminBossan in https://github.com/huggingface/peft/pull/2419
* FIX: AutoPeftModels never reduce embedding size by @BenjaminBossan in https://github.com/huggingface/peft/pull/2427
* FIX: Minimal target module optimization bug with IA³ by @BenjaminBossan in https://github.com/huggingface/peft/pull/2432
* FIX: #2422: Modules to save with multiple adapters by @githubnemo in https://github.com/huggingface/peft/pull/2430

## New Contributors
* @bluenote10 made their first contribution in https://github.com/huggingface/peft/pull/2265
* @pzdkn made their first contribution in https://github.com/huggingface/peft/pull/2254
* @iboing made their first contribution in https://github.com/huggingface/peft/pull/2231
* @innerlee made their first contribution in https://github.com/huggingface/peft/pull/2297
* @NilBiescas made their first contribution in https://github.com/huggingface/peft/pull/2305
* @bingwork made their first contribution in https://github.com/huggingface/peft/pull/2290
* @5eqn made their first contribution in https://github.com/huggingface/peft/pull/2300
* @CCLDArjun made their first contribution in https://github.com/huggingface/peft/pull/2333
* @thedebugger made their first contribution in https://github.com/huggingface/peft/pull/2346
* @makelinux made their first contribution in https://github.com/huggingface/peft/pull/2357
* @gslama12 made their first contribution in https://github.com/huggingface/peft/pull/2371
* @henryzhengr made their first contribution in https://github.com/huggingface/peft/pull/2373
* @Qubitium made their first contribution in https://github.com/huggingface/peft/pull/2404

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.14.0...v0.15.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.15.0)

---

## v0.14.0: Version 0.14.0: EVA, Context-aware Prompt Tuning, Bone, and more
**Published:** 2024-12-06

# Highlights

![peft-v0 14 0](https://github.com/user-attachments/assets/9994bc6d-f047-419f-9ab5-a60c6033d5b6)

## New Methods

### Context-aware Prompt Tuning
@tsachiblau added a new soft prompt method called [Context-aware Prompt Tuning (CPT)](https://huggingface.co/docs/peft/main/en/conceptual_guides/prompting#context-aware-prompt-tuning-cpt) which is a combination of In-Context Learning and Prompt Tuning in the sense that, for each training sample, it builds a learnable context from training examples in addition to the single training sample. Allows for sample- and parameter-efficient few-shot classification and addresses recency-bias.

### Explained Variance Adaptation
@sirluk contributed a new LoRA initialization method called [Explained Variance Adaptation (EVA)](https://huggingface.co/docs/peft/main/en/developer_guides/lora#eva). Instead of randomly initializing LoRA weights, this method uses SVD on minibatches of finetuning data to initialize the LoRA weights and is also able to re-allocate the ranks of the adapter based on the explained variance ratio (derived from SVD). Thus, this initialization method can yield better initial values and better rank distribution.

### Bone
@JL-er added an implementation for [Block Affine (Bone) Adaptation](https://huggingface.co/docs/peft/main/en/conceptual_guides/adapter#bone) which utilizes presumed sparsity in the base layer weights to divide them into multiple sub-spaces that share a single low-rank matrix for updates. Compared to LoRA, Bone has the potential to significantly reduce memory usage and achieve faster computation.


## Enhancements
PEFT now supports LoRAs for `int8` torchao quantized models (check [this](https://github.com/huggingface/peft/blob/main/examples/sequence_classification/LoRA-torchao-8bit.ipynb) and [this](https://github.com/huggingface/peft/blob/main/examples/sequence_classification/LoRA-torchao-8bit-dynamic-activation.ipynb) notebook) . In addition, VeRA can now be used with 4 and 8 bit bitsandbytes quantization thanks to @ZiadHelal.

[Hot-swapping of LoRA adapters](https://huggingface.co/docs/peft/main/en/package_reference/hotswap) is now possible using the `hotswap_adapter` function. Now you are able to load one LoRA and replace its weights in-place with the LoRA weights of another adapter which, in general, should be faster than deleting one adapter and loading the other adapter in its place. The feature is built so that no re-compilation of the model is necessary if `torch.compile` was called on the model (right now, this requires ranks and alphas to be the same for the adapters).

LoRA and IA³ now support `Conv3d` layers thanks to @jsilter, and @JINO-ROHIT added a [notebook](https://github.com/huggingface/peft/blob/main/examples/evaluation/lora-lm-eval.ipynb) showcasing PEFT model evaluation using lm-eval-harness toolkit.

With the `target_modules` argument, you can specify which layers to target with the adapter (e.g. LoRA). Now you can also specify which modules *not* to target by using the `exclude_modules` parameter (thanks @JINO-ROHIT).

# Changes

- There have been made several fixes to the OFT implementation, among other things, to fix merging, which makes adapter weights trained with PEFT versions prior to this release incompatible (see #1996 for details).
- Adapter configs are now forward-compatible by accepting unknown keys.
- Prefix tuning was fitted to the `DynamicCache` caching infrastructure of transformers (see #2096). If you are using this PEFT version and a recent version of transformers with an old prefix tuning checkpoint, you should double check that it still works correctly and retrain it if it doesn't.
- Added `lora_bias` parameter to LoRA layers to enable bias on LoRA B matrix. This is useful when extracting LoRA weights from fully fine-tuned parameters with bias vectors so that these can be taken into account.
- #2180 provided a couple of bug fixes to LoKr (thanks @yaswanth19). If you're using LoKr, your old checkpoints should still work but it's recommended to retrain your adapter.
- `from_pretrained` now warns the user if PEFT keys are missing.
- Attribute access to modules in `modules_to_save` is now properly and transparently handled.
- PEFT supports the changes to bitsandbytes 8bit quantization from the [recent v0.45.0 release](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.45.0). To benefit from these improvements, we thus recommend to upgrade bitsandbytes if you're using QLoRA. Expect slight numerical differences in model outputs if you're using QLoRA with 8bit bitsandbytes quantization.

## What's Changed
* Bump version to 0.13.1.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2094
* Support Conv3d layer in LoRA and IA3 by @jsilter in https://github.com/huggingface/peft/pull/2082
* Fix Inconsistent Missing Keys Warning for Adapter Weights in PEFT by @yaswanth19 in https://github.com/huggingface/peft/pull/2084
* FIX: Change check if past_key_values is empty by @BenjaminBossan in https://github.com/huggingface/peft/pull/2106
* Update install.md by @Salehbigdeli in https://github.com/huggingface/peft/pull/2110
* Update OFT to fix merge bugs by @Zeju1997 in https://github.com/huggingface/peft/pull/1996
* ENH: Improved attribute access for modules_to_save by @BenjaminBossan in https://github.com/huggingface/peft/pull/2117
* FIX low_cpu_mem_usage consolidates devices by @BenjaminBossan in https://github.com/huggingface/peft/pull/2113
* TST Mark flaky X-LoRA test as xfail by @BenjaminBossan in https://github.com/huggingface/peft/pull/2114
* ENH: Warn when from_pretrained misses PEFT keys by @BenjaminBossan in https://github.com/huggingface/peft/pull/2118
* FEAT: Adding exclude modules param(#2044) by @JINO-ROHIT in https://github.com/huggingface/peft/pull/2102
* fix merging bug / update boft conv2d scaling variable by @Zeju1997 in https://github.com/huggingface/peft/pull/2127
* FEAT: Support quantization for VeRA using bitsandbytes (#2070) by @ZiadHelal in https://github.com/huggingface/peft/pull/2076
* Bump version to 0.13.2.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2137
* FEAT: Support torchao by @BenjaminBossan in https://github.com/huggingface/peft/pull/2062
* FIX: Transpose weight matrix based on fan_in_fan_out condition in PiSSA initialization (#2103) by @suyang160 in https://github.com/huggingface/peft/pull/2104
* FIX Type annoations in vera/bnb.py by @BenjaminBossan in https://github.com/huggingface/peft/pull/2139
* ENH Make PEFT configs forward compatible by @BenjaminBossan in https://github.com/huggingface/peft/pull/2038
* FIX Raise an error when performing mixed adapter inference and passing non-existing adapter names by @BenjaminBossan in https://github.com/huggingface/peft/pull/2090
* FIX Prompt learning with latest transformers error by @BenjaminBossan in https://github.com/huggingface/peft/pull/2140
* adding peft lora example notebook for ner by @JINO-ROHIT in https://github.com/huggingface/peft/pull/2126
* FIX TST: NaN issue with HQQ GPU test by @BenjaminBossan in https://github.com/huggingface/peft/pull/2143
* FIX: Bug in target module optimization if child module name is suffix of parent module name by @BenjaminBossan in https://github.com/huggingface/peft/pull/2144
* Bump version to 0.13.2.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2145
* FIX Don't assume past_key_valus for encoder models by @BenjaminBossan in https://github.com/huggingface/peft/pull/2149
* Use `SFTConfig` instead of `SFTTrainer` keyword args by @qgallouedec in https://github.com/huggingface/peft/pull/2150
* FIX: Sft train script FSDP QLoRA embedding mean resizing error by @BenjaminBossan in https://github.com/huggingface/peft/pull/2151
* Optimize DoRA in `eval` and `no dropout` by @ariG23498 in https://github.com/huggingface/peft/pull/2122
* FIX Missing low_cpu_mem_usage argument by @BenjaminBossan in https://github.com/huggingface/peft/pull/2156
* MNT: Remove version pin of diffusers by @BenjaminBossan in https://github.com/huggingface/peft/pull/2162
* DOC: Improve docs for layers_pattern argument by @BenjaminBossan in https://github.com/huggingface/peft/pull/2157
* Update HRA by @DaShenZi721 in https://github.com/huggingface/peft/pull/2160
* fix fsdp_auto_wrap_policy by @eljandoubi in https://github.com/huggingface/peft/pull/2167
* MNT Remove Python 3.8 since it's end of life by @BenjaminBossan in https://github.com/huggingface/peft/pull/2135
* Improving error message when users pass layers_to_transform and layers_pattern by @JINO-ROHIT in https://github.com/huggingface/peft/pull/2169
* FEAT Add hotswapping functionality by @BenjaminBossan in https://github.com/huggingface/peft/pull/2120
* Fix to prefix tuning to fit transformers by @BenjaminBossan in https://github.com/huggingface/peft/pull/2096
* MNT: Enable Python 3.12 on CI by @BenjaminBossan in https://github.com/huggingface/peft/pull/2173
* MNT: Update docker nvidia base image to 12.4.1 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2176
* DOC: Extend modules_to_save doc with pooler example by @BenjaminBossan in https://github.com/huggingface/peft/pull/2175
* FIX VeRA failure on multiple GPUs by @BenjaminBossan in https://github.com/huggingface/peft/pull/2163
* FIX: Import location of HF hub errors by @BenjaminBossan in https://github.com/huggingface/peft/pull/2178
* DOC: fix broken link in the README of loftq by @dennis2030 in https://github.com/huggingface/peft/pull/2183
* added checks for layers to transforms and layer pattern in lora by @JINO-ROHIT in https://github.com/huggingface/peft/pull/2159
* ENH: Warn when loading PiSSA/OLoRA together with other adapters by @BenjaminBossan in https://github.com/huggingface/peft/pull/2186
* TST: Skip AQLM test that is incompatible with torch 2.5 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2187
* FIX: Prefix tuning with model on multiple devices by @BenjaminBossan in https://github.com/huggingface/peft/pull/2189
* FIX: Check for prefix tuning + gradient checkpointing fails by @BenjaminBossan in https://github.com/huggingface/peft/pull/2191
* Dora_datacollector_updated by @shirinyamani in https://github.com/huggingface/peft/pull/2197
* [BUG] Issue with using `rank_pattern` and `alpha_pattern` together in `LoraConfig` by @sirluk in https://github.com/huggingface/peft/pull/2195
* evaluation of peft model using lm-eval-harness toolkit by @JINO-ROHIT in https://github.com/huggingface/peft/pull/2190
* Support Bone by @JL-er in https://github.com/huggingface/peft/pull/2172
* BUG🐛: Fixed scale related bugs in LoKr | Added rank_dropout_scale parameter by @yaswanth19 in https://github.com/huggingface/peft/pull/2180
* update load_dataset for examples/feature_extraction by @sinchir0 in https://github.com/huggingface/peft/pull/2207
* [FEAT] New LoRA Initialization Method: Explained Variance Adaptation by @sirluk in https://github.com/huggingface/peft/pull/2142
* [FIX] EVA `meta` device check bug + add multi-gpu functionality by @sirluk in https://github.com/huggingface/peft/pull/2218
* CPT Tuner by @tsachiblau in https://github.com/huggingface/peft/pull/2168
* [FIX] Invalid `None` check for `loftq_config` attribute in `LoraConfig` by @sirluk in https://github.com/huggingface/peft/pull/2215
* TST: Move slow compile tests to nightly CI by @BenjaminBossan in https://github.com/huggingface/peft/pull/2223
* CI Update AutoAWQ version to fix CI by @BenjaminBossan in https://github.com/huggingface/peft/pull/2222
* FIX Correctly set device of input data in bnb test by @BenjaminBossan in https://github.com/huggingface/peft/pull/2227
* CI: Skip EETQ tests while broken by @BenjaminBossan in https://github.com/huggingface/peft/pull/2226
* Add Validation for Invalid `task_type` in PEFT Configurations by @d-kleine in https://github.com/huggingface/peft/pull/2210
* [FEAT] EVA: ensure deterministic behavior of SVD on multi gpu setups by @sirluk in https://github.com/huggingface/peft/pull/2225
* TST: Eva: Speed up consistency tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2224
* CI: Fix failing torchao test by @BenjaminBossan in https://github.com/huggingface/peft/pull/2232
* TST: Update Llava model id in test by @BenjaminBossan in https://github.com/huggingface/peft/pull/2236
* TST: Skip test on multi-GPU as DataParallel fails by @BenjaminBossan in https://github.com/huggingface/peft/pull/2234
* Bump version of MacOS runners from 12 to 13 by @githubnemo in https://github.com/huggingface/peft/pull/2235
* new version Bone by @JL-er in https://github.com/huggingface/peft/pull/2233
* ENH Argument to enable bias for LoRA B by @BenjaminBossan in https://github.com/huggingface/peft/pull/2237
* FIX: Small regression in BNB LoRA output by @BenjaminBossan in https://github.com/huggingface/peft/pull/2238
* Update CPT documentation by @tsachiblau in https://github.com/huggingface/peft/pull/2229
* FIX: Correctly pass low_cpu_mem_usage argument when initializing a PEFT model with task_type by @BenjaminBossan in https://github.com/huggingface/peft/pull/2253
* FIX Correctly determine word embeddings on Deberta by @BenjaminBossan in https://github.com/huggingface/peft/pull/2257
* FIX: Prevent CUDA context initialization due to AWQ by @BenjaminBossan in https://github.com/huggingface/peft/pull/2230
* ENH: Updates for upcoming BNB Int8 release by @matthewdouglas in https://github.com/huggingface/peft/pull/2245
* Prepare for PEFT release of v0.14.0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/2258

## New Contributors
* @jsilter made their first contribution in https://github.com/huggingface/peft/pull/2082
* @yaswanth19 made their first contribution in https://github.com/huggingface/peft/pull/2084
* @Salehbigdeli made their first contribution in https://github.com/huggingface/peft/pull/2110
* @JINO-ROHIT made their first contribution in https://github.com/huggingface/peft/pull/2102
* @ZiadHelal made their first contribution in https://github.com/huggingface/peft/pull/2076
* @suyang160 made their first contribution in https://github.com/huggingface/peft/pull/2104
* @qgallouedec made their first contribution in https://github.com/huggingface/peft/pull/2150
* @eljandoubi made their first contribution in https://github.com/huggingface/peft/pull/2167
* @dennis2030 made their first contribution in https://github.com/huggingface/peft/pull/2183
* @sirluk made their first contribution in https://github.com/huggingface/peft/pull/2195
* @JL-er made their first contribution in https://github.com/huggingface/peft/pull/2172
* @sinchir0 made their first contribution in https://github.com/huggingface/peft/pull/2207
* @tsachiblau made their first contribution in https://github.com/huggingface/peft/pull/2168
* @d-kleine made their first contribution in https://github.com/huggingface/peft/pull/2210
* @githubnemo made their first contribution in https://github.com/huggingface/peft/pull/2235
* @matthewdouglas made their first contribution in https://github.com/huggingface/peft/pull/2245

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.13.2...v0.14.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.14.0)

---

## v0.13.2: v0.13.2: Small patch release
**Published:** 2024-10-11

This patch release contains a small bug fix for an issue that prevented some LoRA checkpoints to be loaded correctly (mostly concerning stable diffusion checkpoints not trained with PEFT when loaded in diffusers, #2144).

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.13.1...v0.13.2

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.13.2)

---

## v0.13.1: v0.13.1: Small patch release
**Published:** 2024-10-08

This patch release contains a small bug fix for the `low_cpu_mem_usage=True` option (#2113).

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.13.0...v0.13.1

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.13.1)

---

## v0.13.0: v0.13.0: LoRA+, VB-LoRA, and more
**Published:** 2024-09-25

![peft-v0 13 0](https://github.com/user-attachments/assets/0423db36-73ca-4eb4-af12-c21610a1b35c)

# Highlights

## New methods

### LoRA+

@kallewoof added [LoRA\+](https://arxiv.org/abs/2402.12354) to PEFT (#1915). This is a function that allows to [initialize an optimizer](https://huggingface.co/docs/peft/main/en/developer_guides/lora#lora-optimized-lora) with settings that are better suited for training a LoRA adapter.

### VB-LoRA

@leo-yangli added a new method to PEFT called [VB-LoRA](https://arxiv.org/abs/2405.15179) (#2039). The idea is to have LoRA layers be composed from a single vector bank (hence "VB") that is shared among all layers. This makes VB-LoRA extremely parameter efficient and the checkpoints especially small (comparable to the VeRA method), while still promising good fine-tuning performance. Check the [VB-LoRA docs](https://huggingface.co/docs/peft/main/en/package_reference/vblora) and [example](https://github.com/huggingface/peft/blob/main/examples/sequence_classification/VBLoRA.ipynb).

## Enhancements

New Hugging Face team member @ariG23498 added the helper function [`rescale_adapter_scale`](https://huggingface.co/docs/peft/main/en/package_reference/helpers#peft.helpers.rescale_adapter_scale) to PEFT (#1951). Use this context manager to temporarily increase or decrease the scaling of the LoRA adapter of a model. It also works for PEFT adapters loaded directly into a transformers or diffusers model.

@ariG23498 also added [DoRA](https://arxiv.org/abs/2402.09353) support for embedding layers (#2006). So if you're using the `use_dora=True` option in the `LoraConfig`, you can now also target embedding layers.

For some time now, we support [inference with batches that are using different adapters](https://huggingface.co/docs/peft/v0.12.0/en/developer_guides/lora#inference-with-different-lora-adapters-in-the-same-batch) for different samples, so e.g. sample 1-5 use "adapter1" and samples 6-10 use "adapter2". However, this only worked for LoRA layers so far. @saeid93 extended this to also work with layers targeted by `modules_to_save` (#1990).

When loading a PEFT adapter, you now have the option to pass `low_cpu_mem_usage=True` (#1961). This will initialize the adapter with empty weights ("meta" device) before loading the weights instead of initializing on CPU or GPU. This can speed up loading PEFT adapters. So use this option especially if you have a lot of adapters to load at the same time or if these adapters are very big. Please let us know if you encounter issues with this option, as we may make this the default in the future.

# Changes

## Safe loading of PyTorch weights

Unless indicated otherwise, PEFT adapters are saved and loaded using the secure `safetensors` format. However, we also support the [PyTorch format](https://pytorch.org/docs/stable/generated/torch.load.html) for checkpoints, which relies on the inherently insecure pickle protocol from Python. In the future, PyTorch will be more strict when loading these files to improve security by making the option `weights_only=True` the default. This is generally recommended and should not cause any trouble with PEFT checkpoints, which is why with this release, PEFT will enable this by default. Please open an issue if this causes trouble.

## What's Changed
* Bump version to 0.12.1.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1950
* CI Fix Windows permission error on merge test by @BenjaminBossan in https://github.com/huggingface/peft/pull/1952
* Check if past_key_values is provided when using prefix_tuning in peft_model by @Nidhogg-lyz in https://github.com/huggingface/peft/pull/1942
* Add lora+ implementation by @kallewoof in https://github.com/huggingface/peft/pull/1915
* FIX: New bloom changes breaking prompt learning by @BenjaminBossan in https://github.com/huggingface/peft/pull/1969
* ENH Update VeRA preconfigured models by @BenjaminBossan in https://github.com/huggingface/peft/pull/1941
* fix: lora+: include lr in optimizer kwargs by @kallewoof in https://github.com/huggingface/peft/pull/1973
* FIX active_adapters for transformers models by @BenjaminBossan in https://github.com/huggingface/peft/pull/1975
* FIX Loading adapter honors offline mode by @BenjaminBossan in https://github.com/huggingface/peft/pull/1976
* chore: Update CI configuration for workflows by @XciD in https://github.com/huggingface/peft/pull/1985
* Cast to fp32 if using bf16 weights on cpu during `merge_and_unload` by @snarayan21 in https://github.com/huggingface/peft/pull/1978
* AdaLora: Trigger warning when user uses 'r' inplace of 'init_r' by @bhargavyagnik in https://github.com/huggingface/peft/pull/1981
* [Add] scaling LoRA adapter weights with a context manager by @ariG23498 in https://github.com/huggingface/peft/pull/1951
* DOC Small fixes for HQQ and section title by @BenjaminBossan in https://github.com/huggingface/peft/pull/1986
* Add docs and examples for X-LoRA by @EricLBuehler in https://github.com/huggingface/peft/pull/1970
* fix: fix docker build gpus by @XciD in https://github.com/huggingface/peft/pull/1987
* FIX: Adjust transformers version check for bloom by @BenjaminBossan in https://github.com/huggingface/peft/pull/1992
* [Hotfix] Fix BOFT mixed precision by @Edenzzzz in https://github.com/huggingface/peft/pull/1925
* [Suggestions] Updates suggested for `helper.rescale_adapter_scale` by @ariG23498 in https://github.com/huggingface/peft/pull/1989
* MAINT: Default to loading weights only for torch.load by @BenjaminBossan in https://github.com/huggingface/peft/pull/1993
* BOFT bug fix when saving by @Zeju1997 in https://github.com/huggingface/peft/pull/1994
* FIX Import error in BOFT half precision test by @BenjaminBossan in https://github.com/huggingface/peft/pull/1995
* Update lora.md (typos) by @nir-sh-automat-it in https://github.com/huggingface/peft/pull/2003
* TST Add LNTuningConfig and LoKrConfig to tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2005
* ENH: Warn when a user provided model name in the config renamed by @BenjaminBossan in https://github.com/huggingface/peft/pull/2004
* FIX CI Correctly report outcome of bnb import test by @BenjaminBossan in https://github.com/huggingface/peft/pull/2007
* Update docs for X-LoRA and some bugfixes by @EricLBuehler in https://github.com/huggingface/peft/pull/2002
* TST: Potentially Skip 8bit bnb regression test if compute capability is too low by @BenjaminBossan in https://github.com/huggingface/peft/pull/1998
* CI Activate single core multi backend bnb tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2008
* Fix usage of deprecated parameters/functions in X-LoRA by @EricLBuehler in https://github.com/huggingface/peft/pull/2010
* [tests] enable `test_vera_dtypes` on XPU  by @faaany in https://github.com/huggingface/peft/pull/2017
* CI Remove regression tests from BNB CI by @BenjaminBossan in https://github.com/huggingface/peft/pull/2024
* [tests] enable regression tests on XPU by @faaany in https://github.com/huggingface/peft/pull/2019
* ENH: Better error msg for replace_lora_weights_loftq when using a local model. by @BenjaminBossan in https://github.com/huggingface/peft/pull/2022
* [tests] make cuda-only cases in `TestModelAndLayerStatus` device-agnostic by @faaany in https://github.com/huggingface/peft/pull/2026
* [tests] enable `test_mixed_adapter_batches_lora_opt_timing` on XPU by @faaany in https://github.com/huggingface/peft/pull/2021
* MAINT: Update ruff version to ~0.6.1 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1965
* ENH Raise error when applying modules_to_save on tuner layer by @BenjaminBossan in https://github.com/huggingface/peft/pull/2028
* FIX: Don't target the classification head when using target_modules="all-linear" by @BenjaminBossan in https://github.com/huggingface/peft/pull/2033
* [tests] enable cuda-only tests in `test_common_gpu.py` to work on XPU by @faaany in https://github.com/huggingface/peft/pull/2031
* [Add] DoRA Embedding by @ariG23498 in https://github.com/huggingface/peft/pull/2006
* [tests] enable `test_gpu_examples.py` on XPU  by @faaany in https://github.com/huggingface/peft/pull/2036
* Bug: set correct pre-commit-hooks version by @ltoniazzi in https://github.com/huggingface/peft/pull/2034
* Warn if using tied target module with `tie_word_embeddings` by @ltoniazzi in https://github.com/huggingface/peft/pull/2025
* ENH: Faster adapter loading if there are a lot of target modules by @BenjaminBossan in https://github.com/huggingface/peft/pull/2045
* FIX: Error with OLoRA init when using bnb by @BenjaminBossan in https://github.com/huggingface/peft/pull/2011
* FIX: Small numerical discrepancy for p-tuning after loading the model by @BenjaminBossan in https://github.com/huggingface/peft/pull/2047
* Add VB-LoRA by @leo-yangli in https://github.com/huggingface/peft/pull/2039
* Fixing scalings logging test by @EricLBuehler in https://github.com/huggingface/peft/pull/2042
* TST: Fewer inference steps for stable diffusion tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2051
* TST Speed up vision model tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/2058
* TST: Make X-LoRA tests faster by @BenjaminBossan in https://github.com/huggingface/peft/pull/2059
* Update permissions for githubtoken stale.yml by @glegendre01 in https://github.com/huggingface/peft/pull/2061
* MAINT: Give stale bot permissions for PRs too by @BenjaminBossan in https://github.com/huggingface/peft/pull/2064
* avoid saving boft_P in adapter model by @sywangyi in https://github.com/huggingface/peft/pull/2050
* fix arguments for PiSSA preprocess by @keakon in https://github.com/huggingface/peft/pull/2053
* Apply deprecated `evaluation_strategy` by @muellerzr in https://github.com/huggingface/peft/pull/1664
* fixing multiple LoRA in the same batch or vit by @saeid93 in https://github.com/huggingface/peft/pull/1990
* FIX: Bug that prevents BOFT from loading multiple adapters by @BenjaminBossan in https://github.com/huggingface/peft/pull/2068
* [tests] skip some tests for XPU devices by @faaany in https://github.com/huggingface/peft/pull/2074
* ENH: PiSSA/OLoRA: Preserve original config on save by @BenjaminBossan in https://github.com/huggingface/peft/pull/2077
* Expose bias to to ModulesToSaveWrapper by @dengdifan in https://github.com/huggingface/peft/pull/2081
* Update setup.py to update contact info by @sayakpaul in https://github.com/huggingface/peft/pull/2086
* ENH: Allow empty initialization of adapter weight by @BenjaminBossan in https://github.com/huggingface/peft/pull/1961
* ENH: Add default target layers for gemma2 architecture by @BenjaminBossan in https://github.com/huggingface/peft/pull/2078
* FIX: Bug in find_minimal_target_modules by @BenjaminBossan in https://github.com/huggingface/peft/pull/2083
* Fix func docstring by @kwonmha in https://github.com/huggingface/peft/pull/2087
* ENH: Better DoRA check in mixed adapter batch inference by @BenjaminBossan in https://github.com/huggingface/peft/pull/2089

## New Contributors
* @Nidhogg-lyz made their first contribution in https://github.com/huggingface/peft/pull/1942
* @XciD made their first contribution in https://github.com/huggingface/peft/pull/1985
* @bhargavyagnik made their first contribution in https://github.com/huggingface/peft/pull/1981
* @ariG23498 made their first contribution in https://github.com/huggingface/peft/pull/1951
* @Edenzzzz made their first contribution in https://github.com/huggingface/peft/pull/1925
* @Zeju1997 made their first contribution in https://github.com/huggingface/peft/pull/1994
* @nir-sh-automat-it made their first contribution in https://github.com/huggingface/peft/pull/2003
* @faaany made their first contribution in https://github.com/huggingface/peft/pull/2017
* @ltoniazzi made their first contribution in https://github.com/huggingface/peft/pull/2034
* @leo-yangli made their first contribution in https://github.com/huggingface/peft/pull/2039
* @glegendre01 made their first contribution in https://github.com/huggingface/peft/pull/2061
* @keakon made their first contribution in https://github.com/huggingface/peft/pull/2053
* @muellerzr made their first contribution in https://github.com/huggingface/peft/pull/1664
* @saeid93 made their first contribution in https://github.com/huggingface/peft/pull/1990
* @dengdifan made their first contribution in https://github.com/huggingface/peft/pull/2081
* @kwonmha made their first contribution in https://github.com/huggingface/peft/pull/2087

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.12.0...v0.13.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.13.0)

---

## v0.12.0: v0.12.0: New methods OLoRA, X-LoRA, FourierFT, HRA, and much more
**Published:** 2024-07-24

# Highlights

![peft-v0 12 0](https://github.com/user-attachments/assets/fc0bbcf9-67f5-42fe-ad5f-aa94f5abcef3)

## New methods

### OLoRA

@tokenizer-decode added support for a new LoRA initialization strategy called [OLoRA](https://arxiv.org/abs/2406.01775) (#1828). With this initialization option, the LoRA weights are initialized to be orthonormal, which promises to improve training convergence. Similar to PiSSA, this can also be applied to models quantized with bitsandbytes. Check out the accompanying [OLoRA examples](https://github.com/huggingface/peft/tree/main/examples/olora_finetuning).

### X-LoRA

@EricLBuehler added the [X-LoRA](https://arxiv.org/abs/2402.07148) method to PEFT (#1491). This is a mixture of experts approach that combines the strength of multiple pre-trained LoRA adapters. Documentation has yet to be added but check out the [X-LoRA tests](https://github.com/huggingface/peft/blob/main/tests/test_xlora.py) for how to use it.

### FourierFT

@Phoveran, @zqgao22, @Chaos96, and @DSAILatHKUST added [discrete Fourier transform fine-tuning](https://arxiv.org/abs/2405.03003) to PEFT (#1838). This method promises to match LoRA in terms of performance while reducing the number of parameters even further. Check out the included [FourierFT notebook](https://github.com/huggingface/peft/blob/main/examples/sequence_classification/FourierFT.ipynb).

### HRA

@DaShenZi721 added support for [Householder Reflection Adaptation](https://arxiv.org/abs/2405.17484) (#1864). This method bridges the gap between low rank adapters like LoRA on the one hand and orthogonal fine-tuning techniques such as OFT and BOFT on the other. As such, it is interesting for both LLMs and image generation models. Check out the [HRA example](https://github.com/huggingface/peft/tree/main/examples/hra_dreambooth) on how to perform DreamBooth fine-tuning.


## Enhancements

* IA³ now supports merging of multiple adapters via the `add_weighted_adapter` method thanks to @alexrs (#1701).
* Call `peft_model.get_layer_status()` and `peft_model.get_model_status()` to get an overview of the layer/model status of the PEFT model. This can be especially helpful when dealing with multiple adapters or for debugging purposes. More information can be found in the [docs](https://huggingface.co/docs/peft/main/en/developer_guides/troubleshooting#check-layer-and-model-status) (#1743).
* DoRA now supports FSDP training, including with bitsandbytes quantization, aka QDoRA ()#1806).
* VeRA has been extended by @dkopi to support targeting layers with different weight shapes (#1817).
* @kallewoof added the possibility for ephemeral GPU offloading. For now, this is only implemented for loading DoRA models, which can be sped up considerably for big models at the cost of a bit of extra VRAM (#1857).
* _Experimental_: It is now possible to tell PEFT to use your [custom LoRA layers through dynamic dispatching](https://huggingface.co/docs/peft/main/en/developer_guides/custom_models#experimental-support-for-dynamic-dispatch-of-custom-modules-in-lora). Use this, for instance, to add LoRA layers for thus far unsupported layer types without the need to first create a PR on PEFT (but contributions are still welcome!) (#1875).

## Examples

* @shirinyamani added a [script and a notebook](https://github.com/huggingface/peft/tree/main/examples/dora_finetuning) to demonstrate DoRA fine-tuning.
* @rahulbshrestha contributed a [notebook](https://github.com/huggingface/peft/blob/main/examples/dna_language_models/dna_lm.ipynb) that shows how to fine-tune a DNA language model with LoRA.

# Changes

## Casting of the adapter dtype

**Important**: If the base model is loaded in float16 (fp16) or bfloat16 (bf16), PEFT now autocasts adapter weights to float32 (fp32) instead of using the dtype of the base model (#1706). This requires more memory than previously but stabilizes training, so it's the more sensible default. To prevent this, pass `autocast_adapter_dtype=False` when calling `get_peft_model`, `PeftModel.from_pretrained`, or `PeftModel.load_adapter`.

## Adapter device placement

The logic of device placement when loading multiple adapters on the same model has been changed (#1742). Previously, PEFT would move all adapters to the device of the base model. Now, only the newly loaded/created adapter is moved to the base model's device. This allows users to have more fine-grained control over the adapter devices, e.g. allowing them to offload unused adapters to CPU more easily.

## [PiSSA](https://huggingface.co/docs/peft/developer_guides/lora#pissa)

* Calling `save_pretrained` with the `convert_pissa_to_lora` argument is deprecated, the argument was renamed to `path_initial_model_for_weight_conversion` (#1828). Also, calling this no longer deletes the original adapter (#1933).
* Using weight conversion (`path_initial_model_for_weight_conversion`) while also using `use_rslora=True` and `rank_pattern` or `alpha_pattern` now raises an error (#1930). This used not to raise but inference would return incorrect outputs. We also warn about this setting during initialization.

# Call for contributions

We are now making sure to tag appropriate issues with the `contributions welcome` label. If you are looking for a way to contribute to PEFT, check out [these issues](https://github.com/huggingface/peft/issues?q=is%3Aissue+is%3Aopen+label%3Acontributions-welcome).

## What's Changed

* Bump version to 0.11.1.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1736
* save and load base model with revision by @mnoukhov in https://github.com/huggingface/peft/pull/1658
* Autocast adapter weights if fp16/bf16 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1706
* FIX BOFT setting env vars breaks C++ compilation by @BenjaminBossan in https://github.com/huggingface/peft/pull/1739
* Bump version to 0.11.2.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1741
* TST: torch compile tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1725
* Add add_weighted_adapter to IA3 adapters by @alexrs in https://github.com/huggingface/peft/pull/1701
* ENH Layer/model status shows devices now by @BenjaminBossan in https://github.com/huggingface/peft/pull/1743
* Fix warning messages about `config.json` when the base `model_id` is local. by @elementary-particle in https://github.com/huggingface/peft/pull/1668
* DOC TST Document and test reproducibility with models using batch norm by @BenjaminBossan in https://github.com/huggingface/peft/pull/1734
* FIX Use correct attribute name for HQQ in merge by @BenjaminBossan in https://github.com/huggingface/peft/pull/1791
* fix docs by @pacman100 in https://github.com/huggingface/peft/pull/1793
* FIX Allow same layer adapters on different devices by @BenjaminBossan in https://github.com/huggingface/peft/pull/1742
* TST Install bitsandbytes for compile tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1796
* FIX BOFT device error after PR 1742 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1799
* TST Add regression test for DoRA, VeRA, BOFT, LN Tuning by @BenjaminBossan in https://github.com/huggingface/peft/pull/1792
* Docs / LoRA: Add more information on `merge_and_unload` docs by @younesbelkada in https://github.com/huggingface/peft/pull/1805
* TST: Add simple BNB regression tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1602
* CI Make torch compile tests run on GPU by @BenjaminBossan in https://github.com/huggingface/peft/pull/1808
* MNT Remove deprecated use of load_in_8bit by @BenjaminBossan in https://github.com/huggingface/peft/pull/1811
* Refactor to make DoRA and QDoRA work with FSDP by @BenjaminBossan in https://github.com/huggingface/peft/pull/1806
* FIX CI: Remove potentially problematic git command by @BenjaminBossan in https://github.com/huggingface/peft/pull/1820
* ENH / Workflow: Notify on slack about peft + transformers main test results by @younesbelkada in https://github.com/huggingface/peft/pull/1821
* FIX CI: Install pytest-reportlog package by @BenjaminBossan in https://github.com/huggingface/peft/pull/1822
* ENH / Workflow: Use repository variable by @younesbelkada in https://github.com/huggingface/peft/pull/1823
* Patch for Cambricon MLUs test  by @huismiling in https://github.com/huggingface/peft/pull/1747
* Fix a documentation typo by @sparsh2 in https://github.com/huggingface/peft/pull/1833
* FIX Failing Llama tests due to new kv cache by @BenjaminBossan in https://github.com/huggingface/peft/pull/1832
* Workflow / Bnb: Add a mechanism to inform us if the import fails by @younesbelkada in https://github.com/huggingface/peft/pull/1830
* Workflow: Fix broken messages by @younesbelkada in https://github.com/huggingface/peft/pull/1842
* feat(ci): add trufflehog secrets detection by @McPatate in https://github.com/huggingface/peft/pull/1841
* DOC Describe torch_device argument in from_pretrained docstring by @BenjaminBossan in https://github.com/huggingface/peft/pull/1843
* Support for different layer shapes for VeRA by @dkopi in https://github.com/huggingface/peft/pull/1817
* CI Activate env to prevent bnb import error by @BenjaminBossan in https://github.com/huggingface/peft/pull/1845
* Fixed PeftMixedModel docstring example #1824 by @namanvats in https://github.com/huggingface/peft/pull/1850
* MNT Upgrade ruff version to ~0.4.8 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1851
* Adding support for an optional initialization strategy OLoRA by @tokenizer-decode in https://github.com/huggingface/peft/pull/1828
* FIX: Adalora ranknum loaded on wrong device by @BenjaminBossan in https://github.com/huggingface/peft/pull/1852
* Workflow / FIX: Fix red status on our CI by @younesbelkada in https://github.com/huggingface/peft/pull/1854
* DOC FIX Comment about init of LoRA Embedding by @BenjaminBossan in https://github.com/huggingface/peft/pull/1855
* DOC Move helpers section to dev developer guide by @BenjaminBossan in https://github.com/huggingface/peft/pull/1856
* CI Testing: Remove import check by @BenjaminBossan in https://github.com/huggingface/peft/pull/1859
* Update lora_based_methods.md by @jtatman in https://github.com/huggingface/peft/pull/1861
* FIX multitask prompt tuning paper link by @cep-ter in https://github.com/huggingface/peft/pull/1862
* Workflow: Attempt to fix the current failures by @younesbelkada in https://github.com/huggingface/peft/pull/1868
* CI testing BNB: remove single GPU tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1866
* CI Downgrade numpy to <2.0 for Mac and Windows by @BenjaminBossan in https://github.com/huggingface/peft/pull/1871
* FIX Error when using VeRA with float16 or bfloat16 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1874
* Workflow: Update bug report template by @younesbelkada in https://github.com/huggingface/peft/pull/1882
* ENH: LoRA support for dynamically dispatching to custom layers by @BenjaminBossan in https://github.com/huggingface/peft/pull/1875
* FIX Init AdaLoRA to be identity transform by @BenjaminBossan in https://github.com/huggingface/peft/pull/1884
* FIX Make special LoRA inits DeepSpeed compatible by @BenjaminBossan in https://github.com/huggingface/peft/pull/1887
* bypass print_trainable_parameter() if model is not peft model by @delock in https://github.com/huggingface/peft/pull/1888
* Fix early import of torch extension in BOFT by @PhyscalX in https://github.com/huggingface/peft/pull/1879
* Dora Fine-tuning added to examples by @shirinyamani in https://github.com/huggingface/peft/pull/1885
* CI: Don't fail fast in test matrix by @BenjaminBossan in https://github.com/huggingface/peft/pull/1896
* FIX TEST: Higher tolerance for AdaLoRA in test by @BenjaminBossan in https://github.com/huggingface/peft/pull/1897
* test: bump absolute tolerance level in test by @kallewoof in https://github.com/huggingface/peft/pull/1891
* ephemeral GPU offload support by @kallewoof in https://github.com/huggingface/peft/pull/1857
* FIX TEST Even higher tolerance for AdaLoRA in test by @BenjaminBossan in https://github.com/huggingface/peft/pull/1898
* FIX Recursion while accessing attribute before initialization by @ret-1 in https://github.com/huggingface/peft/pull/1892
* chore: markdown formatting by @stillmatic in https://github.com/huggingface/peft/pull/1899
* Tutorial Notebook: Using the PEFT library with a DNA Language Model. by @rahulbshrestha in https://github.com/huggingface/peft/pull/1873
* Integrate X-LoRA by @EricLBuehler in https://github.com/huggingface/peft/pull/1491
* FIX: Flaky multitask prompt tuning test fixed by setting the seed by @BenjaminBossan in https://github.com/huggingface/peft/pull/1908
* FourierFT Support by @Phoveran in https://github.com/huggingface/peft/pull/1838
* fix参数encoder_reparameterization_type by @sujeek in https://github.com/huggingface/peft/pull/1926
* Fix attribute check for print_trainable_parameters method by @anch0vy in https://github.com/huggingface/peft/pull/1928
* Synchronize lora's merge, unmerge, etc. modifications to lora's tp_layer. by @zhangsheng377 in https://github.com/huggingface/peft/pull/1919
* support HRA by @DaShenZi721 in https://github.com/huggingface/peft/pull/1864
* FIX PiSSA & OLoRA with rank/alpha pattern, rslora by @BenjaminBossan in https://github.com/huggingface/peft/pull/1930
* support Grouped-Query Attention by @ttw1018 in https://github.com/huggingface/peft/pull/1901
* FIX: More VeRA tests, fix tests, more checks by @BenjaminBossan in https://github.com/huggingface/peft/pull/1900
* [WIP] ENH Add support for Qwen2 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1906
* Decrease memory usage of `merge_and_unload` by @snarayan21 in https://github.com/huggingface/peft/pull/1944
* PiSSA, OLoRA: Delete initial adapter after conversion instead of the active adapter by @BenjaminBossan in https://github.com/huggingface/peft/pull/1933
* Release v0.12.0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1946

## New Contributors

* @mnoukhov made their first contribution in https://github.com/huggingface/peft/pull/1658
* @elementary-particle made their first contribution in https://github.com/huggingface/peft/pull/1668
* @sparsh2 made their first contribution in https://github.com/huggingface/peft/pull/1833
* @McPatate made their first contribution in https://github.com/huggingface/peft/pull/1841
* @dkopi made their first contribution in https://github.com/huggingface/peft/pull/1817
* @namanvats made their first contribution in https://github.com/huggingface/peft/pull/1850
* @tokenizer-decode made their first contribution in https://github.com/huggingface/peft/pull/1828
* @jtatman made their first contribution in https://github.com/huggingface/peft/pull/1861
* @cep-ter made their first contribution in https://github.com/huggingface/peft/pull/1862
* @delock made their first contribution in https://github.com/huggingface/peft/pull/1888
* @PhyscalX made their first contribution in https://github.com/huggingface/peft/pull/1879
* @shirinyamani made their first contribution in https://github.com/huggingface/peft/pull/1885
* @kallewoof made their first contribution in https://github.com/huggingface/peft/pull/1891
* @ret-1 made their first contribution in https://github.com/huggingface/peft/pull/1892
* @stillmatic made their first contribution in https://github.com/huggingface/peft/pull/1899
* @rahulbshrestha made their first contribution in https://github.com/huggingface/peft/pull/1873
* @Phoveran made their first contribution in https://github.com/huggingface/peft/pull/1838
* @sujeek made their first contribution in https://github.com/huggingface/peft/pull/1926
* @anch0vy made their first contribution in https://github.com/huggingface/peft/pull/1928
* @DaShenZi721 made their first contribution in https://github.com/huggingface/peft/pull/1864
* @ttw1018 made their first contribution in https://github.com/huggingface/peft/pull/1901
* @snarayan21 made their first contribution in https://github.com/huggingface/peft/pull/1944

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.11.1...v0.12.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.12.0)

---

## v0.11.1: v0.11.1
**Published:** 2024-05-17

# Patch release v0.11.1

Fix a bug that could lead to C++ compilation errors after importing PEFT (#1738 #1739).

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.11.0...v0.11.1

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.11.1)

---

## v0.11.0: v0.11.0: New PEFT methods BOFT, VeRA, PiSSA, quantization with HQQ and EETQ, and more
**Published:** 2024-05-16

# Highlights

![peft-v0 11 0](https://github.com/huggingface/peft/assets/6229650/ca652d10-c389-4163-ab62-1e0c821c9c5a)

## New methods

### BOFT

Thanks to @yfeng95, @Zeju1997, and @YuliangXiu, PEFT was extended with BOFT: Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization (#1326, [BOFT paper link](https://huggingface.co/papers/2311.06243)). In PEFT v0.7.0, we already added [OFT](https://huggingface.co/papers/2306.07280), but BOFT is even more parameter efficient. Check out the included [BOFT controlnet](https://github.com/huggingface/peft/tree/main/examples/boft_controlnet) and [BOFT dreambooth](https://github.com/huggingface/peft/tree/main/examples/boft_dreambooth) examples.


### VeRA

If the parameter reduction of LoRA is not enough for your use case, you should take a close look at VeRA: Vector-based Random Matrix Adaptation (#1564, [VeRA paper link](https://huggingface.co/papers/2310.11454)). This method resembles LoRA but adds two learnable scaling vectors to the two LoRA weight matrices. However, the LoRA weights themselves are shared across all layers, considerably reducing the number of trainable parameters.

The bulk of this PR was implemented by contributor @vvvm23 with the help of @dkopi.

### PiSSA

PiSSA, Principal Singular values and Singular vectors Adaptation, is a new initialization method for LoRA, which was added by @fxmeng (#1626, [PiSSA paper link](https://huggingface.co/papers/2404.02948)). The improved initialization promises to speed up convergence and improve the final performance of LoRA models. When using models quantized with bitsandbytes, PiSSA initialization should reduce the quantization error, similar to LoftQ.

## Quantization

### HQQ

Thanks to @fahadh4ilyas, PEFT LoRA linear layers now support Half-Quadratic Quantization, HQQ (#1618, [HQQ repo](https://github.com/mobiusml/hqq/)). HQQ is fast and efficient (down to 2 bits), while not requiring calibration data.

### EETQ

Another new quantization method supported in PEFT is Easy & Efficient Quantization for Transformers, EETQ (#1675, [EETQ repo](https://github.com/NetEase-FuXi/EETQ)). This 8 bit quantization method works for LoRA linear layers and should be faster than bitsandbytes.

## Show adapter layer and model status

We added a feature to show adapter layer and model status of PEFT models in #1663. With the newly added methods, you can easily check what adapters exist on your model, whether gradients are active, whether they are enabled, which ones are active or merged. You will also be informed if irregularities have been detected.

To use this new feature, call `model.get_layer_status()` for layer-level information, and `model.get_model_status()` for model-level information. For more details, check out our [docs on layer and model status](https://huggingface.co/docs/peft/main/en/developer_guides/troubleshooting#check-layer-and-model-status).

# Changes

## Edge case of how we deal with `modules_to_save`

We had the issue that when we were using classes such as PeftModelForSequenceClassification, we implicitly added the classifier layers to `model.modules_to_save`. However, this would only add a new `ModulesToSaveWrapper` instance for the first adapter being initialized. When initializing a 2nd adapter via `model.add_adapter`, this information was ignored. Now, `peft_config.modules_to_save` is updated explicitly to add the classifier layers (#1615). This is a departure from how this worked previously, but it reflects the intended behavior better.

Furthermore, when merging together multiple LoRA adapters using `model.add_weighted_adapter`, if these adapters had `modules_to_save`, the original parameters of these modules would be used. This is unexpected and will most likely result in bad outputs. As there is no clear way to merge these modules, we decided to raise an error in this case (#1615).

## What's Changed
* Bump version to 0.10.1.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1578
* FIX Minor issues in docs, re-raising exception by @BenjaminBossan in https://github.com/huggingface/peft/pull/1581
* FIX / Docs: Fix doc link for layer replication by @younesbelkada in https://github.com/huggingface/peft/pull/1582
* DOC: Short section on using transformers pipeline by @BenjaminBossan in https://github.com/huggingface/peft/pull/1587
* Extend PeftModel.from_pretrained() to models with disk-offloaded modules by @blbadger in https://github.com/huggingface/peft/pull/1431
* [feat] Add `lru_cache` to `import_utils` calls that did not previously have it by @tisles in https://github.com/huggingface/peft/pull/1584
* fix deepspeed zero3+prompt tuning bug. word_embeddings.weight shape i… by @sywangyi in https://github.com/huggingface/peft/pull/1591
* MNT: Update GH bug report template by @BenjaminBossan in https://github.com/huggingface/peft/pull/1600
* fix the torch_dtype and quant_storage_dtype by @pacman100 in https://github.com/huggingface/peft/pull/1614
* FIX In the image classification example, Change the model to the LoRA… by @changhwa in https://github.com/huggingface/peft/pull/1624
* Remove duplicated import by @nzw0301 in https://github.com/huggingface/peft/pull/1622
* FIX: bnb config wrong argument names by @BenjaminBossan in https://github.com/huggingface/peft/pull/1603
* FIX Make DoRA work with Conv1D layers by @BenjaminBossan in https://github.com/huggingface/peft/pull/1588
* FIX: Send results to correct channel by @younesbelkada in https://github.com/huggingface/peft/pull/1628
* FEAT: Allow ignoring mismatched sizes when loading by @BenjaminBossan in https://github.com/huggingface/peft/pull/1620
* itemsize is torch>=2.1, use element_size() by @winglian in https://github.com/huggingface/peft/pull/1630
* FIX Multiple adapters and modules_to_save by @BenjaminBossan in https://github.com/huggingface/peft/pull/1615
* FIX Correctly call element_size by @BenjaminBossan in https://github.com/huggingface/peft/pull/1635
* fix: allow load_adapter to use different device by @yhZhai in https://github.com/huggingface/peft/pull/1631
* Adalora deepspeed by @sywangyi in https://github.com/huggingface/peft/pull/1625
* Adding BOFT: Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization by @yfeng95 in https://github.com/huggingface/peft/pull/1326
* Don't use deprecated `Repository` anymore by @Wauplin in https://github.com/huggingface/peft/pull/1641
* FIX Errors in the transformers integration docs by @BenjaminBossan in https://github.com/huggingface/peft/pull/1629
* update figure assets of BOFT by @YuliangXiu in https://github.com/huggingface/peft/pull/1642
* print_trainable_parameters - format `%` to be sensible by @stas00 in https://github.com/huggingface/peft/pull/1648
* FIX: Bug with handling of active adapters by @BenjaminBossan in https://github.com/huggingface/peft/pull/1659
* Remove `dreambooth` Git link by @charliermarsh in https://github.com/huggingface/peft/pull/1660
* add safetensor load in multitask_prompt_tuning by @sywangyi in https://github.com/huggingface/peft/pull/1662
* Adds Vera (Vector Based Random Matrix Adaption) #2 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1564
* Update deepspeed.md by @sanghyuk-choi in https://github.com/huggingface/peft/pull/1679
* ENH: Add multi-backend tests for bnb by @younesbelkada in https://github.com/huggingface/peft/pull/1667
* FIX / Workflow: Fix Mac-OS CI issues by @younesbelkada in https://github.com/huggingface/peft/pull/1680
* FIX Use trl version of tiny random llama by @BenjaminBossan in https://github.com/huggingface/peft/pull/1681
* FIX: Don't eagerly import bnb for LoftQ by @BenjaminBossan in https://github.com/huggingface/peft/pull/1683
* FEAT: Add EETQ support in PEFT by @younesbelkada in https://github.com/huggingface/peft/pull/1675
* FIX / Workflow: Always notify on slack for docker image workflows by @younesbelkada in https://github.com/huggingface/peft/pull/1682
* FIX: upgrade autoawq to latest version by @younesbelkada in https://github.com/huggingface/peft/pull/1684
* FIX: Initialize DoRA weights in float32 if float16 is being used by @BenjaminBossan in https://github.com/huggingface/peft/pull/1653
* fix bf16 model type issue for ia3 by @sywangyi in https://github.com/huggingface/peft/pull/1634
* FIX Issues with AdaLora initialization by @BenjaminBossan in https://github.com/huggingface/peft/pull/1652
* FEAT Show adapter layer and model status by @BenjaminBossan in https://github.com/huggingface/peft/pull/1663
* Fixing the example by providing correct tokenized seq length by @jpodivin in https://github.com/huggingface/peft/pull/1686
* TST: Skiping AWQ tests for now .. by @younesbelkada in https://github.com/huggingface/peft/pull/1690
* Add LayerNorm tuning model by @DTennant in https://github.com/huggingface/peft/pull/1301
* FIX Use different doc builder docker image by @BenjaminBossan in https://github.com/huggingface/peft/pull/1697
* Set experimental dynamo config for compile tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1698
* fix the fsdp peft autowrap policy by @pacman100 in https://github.com/huggingface/peft/pull/1694
* Add LoRA support to HQQ Quantization by @fahadh4ilyas in https://github.com/huggingface/peft/pull/1618
* FEAT Helper to check if a model is a PEFT model by @BenjaminBossan in https://github.com/huggingface/peft/pull/1713
* support Cambricon MLUs device by @huismiling in https://github.com/huggingface/peft/pull/1687
* Some small cleanups in docstrings, copyright note by @BenjaminBossan in https://github.com/huggingface/peft/pull/1714
* Fix docs typo by @NielsRogge in https://github.com/huggingface/peft/pull/1719
* revise run_peft_multigpu.sh by @abzb1 in https://github.com/huggingface/peft/pull/1722
* Workflow: Add slack messages workflow by @younesbelkada in https://github.com/huggingface/peft/pull/1723
* DOC Document the PEFT checkpoint format by @BenjaminBossan in https://github.com/huggingface/peft/pull/1717
* FIX Allow DoRA init on CPU when using BNB by @BenjaminBossan in https://github.com/huggingface/peft/pull/1724
* Adding PiSSA as an optional initialization method of LoRA by @fxmeng in https://github.com/huggingface/peft/pull/1626

## New Contributors
* @tisles made their first contribution in https://github.com/huggingface/peft/pull/1584
* @changhwa made their first contribution in https://github.com/huggingface/peft/pull/1624
* @yhZhai made their first contribution in https://github.com/huggingface/peft/pull/1631
* @yfeng95 made their first contribution in https://github.com/huggingface/peft/pull/1326
* @YuliangXiu made their first contribution in https://github.com/huggingface/peft/pull/1642
* @charliermarsh made their first contribution in https://github.com/huggingface/peft/pull/1660
* @sanghyuk-choi made their first contribution in https://github.com/huggingface/peft/pull/1679
* @jpodivin made their first contribution in https://github.com/huggingface/peft/pull/1686
* @DTennant made their first contribution in https://github.com/huggingface/peft/pull/1301
* @fahadh4ilyas made their first contribution in https://github.com/huggingface/peft/pull/1618
* @huismiling made their first contribution in https://github.com/huggingface/peft/pull/1687
* @NielsRogge made their first contribution in https://github.com/huggingface/peft/pull/1719
* @abzb1 made their first contribution in https://github.com/huggingface/peft/pull/1722
* @fxmeng made their first contribution in https://github.com/huggingface/peft/pull/1626

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.10.0...v0.11.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.11.0)

---

## v0.10.0: v0.10.0: Fine-tune larger QLoRA models with DeepSpeed and FSDP, layer replication, enhance DoRA
**Published:** 2024-03-21

## Highlights

![image](https://github.com/huggingface/peft/assets/49240599/8274f36f-246f-4509-a6e4-804aba574566)

### Support for QLoRA with DeepSpeed ZeRO3 and FSDP

We added a couple of changes to allow QLoRA to work with DeepSpeed ZeRO3 and Fully Sharded Data Parallel (FSDP). For instance, this allows you to fine-tune a 70B Llama model on two GPUs with 24GB memory each. Besides the latest version of PEFT, this requires `bitsandbytes>=0.43.0`, `accelerate>=0.28.0`, `transformers>4.38.2`, `trl>0.7.11`. Check out our docs on [DeepSpeed](https://huggingface.co/docs/peft/v0.10.0/en/accelerate/deepspeed) and [FSDP](https://huggingface.co/docs/peft/v0.10.0/en/accelerate/fsdp) with PEFT, as well as this [blogpost](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) from answer.ai, for more details.

### Layer replication

First time contributor @siddartha-RE added support for layer replication with LoRA. This allows you to duplicate layers of a model and apply LoRA adapters to them. Since the base weights are shared, this costs only very little extra memory, but can lead to a nice improvement of model performance. Find out more in [our docs](https://huggingface.co/docs/peft/v0.10.0/en/developer_guides/lora#memory-efficient-layer-replication-with-lora).

### Improving DoRA

Last release, we added the option to enable [DoRA](https://arxiv.org/abs/2402.09353) in PEFT by simply adding `use_dora=True` to your `LoraConfig`. However, this only worked for non-quantized linear layers. With this PEFT release, we now also support `Conv2d` layers, as well as linear layers quantized with bitsandbytes.

### Mixed LoRA adapter batches

If you have a PEFT model with multiple LoRA adapters attached to it, it's now possible to apply different adapters (or, in fact, no adapter) on different samples in the same batch. To do this, pass a list of adapter names as an additional argument. For example, if you have a batch of three samples:

```python
output = model(**inputs, adapter_names=["adapter1", "adapter2", "__base__"])`
```

Here, `"adapter1"` and `"adapter2"` should be the same name as your corresponding LoRA adapters and `"__base__"` is a special name that refers to the base model without any adapter. Find more details in [our docs](https://huggingface.co/docs/peft/v0.10.0/en/developer_guides/lora#inference-with-different-lora-adapters-in-the-same-batch).

Without this feature, if you wanted to run inference with different LoRA adapters, you'd have to use single samples or try to group batches with the same adapter, then switch between adapters using `set_adapter` -- this is inefficient and inconvenient. Therefore, it is recommended to use this new, faster method from now on when encountering this scenario.

### New LoftQ initialization function

We added an alternative way to initialize LoRA weights for a quantized model using the LoftQ method, which can be more convenient than the existing method. Right now, using LoftQ requires you to go through multiple steps as shown [here](https://github.com/huggingface/peft/blob/8e979fc73248ccb4c5b5a99c415f3e14a37daae6/examples/loftq_finetuning/README.md). Furthermore, it's necessary to keep a separate copy of the quantized weights, as those are not identical to the quantized weights from the default model.

Using the new `replace_lora_weights_loftq` function, it's now possible to apply LoftQ initialization in a single step and without the need for extra copies of the weights. Check out [the docs](https://huggingface.co/docs/peft/v0.10.0/en/developer_guides/lora#a-more-convienient-way) and this [example notebook](https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/LoftQ_weight_replacement.ipynb) to see how it works. Right now, this method only supports 4bit quantization with bitsandbytes, and the model has to be stored in the safetensors format.

## Deprecations

The function `prepare_model_for_int8_training` was deprecated for quite some time and is now removed completely. Use `prepare_model_for_kbit_training` instead.

## What's Changed

Besides these highlights, we added many small improvements and fixed a couple of bugs. All these changes are listed below. As always, we thank all the awesome contributors who helped us improve PEFT.

* Bump version to 0.9.1.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1517
* Fix for "leaf Variable that requires grad" Error in In-Place Operation by @DopeorNope-Lee in https://github.com/huggingface/peft/pull/1372
* FIX [`CI` / `Docker`] Follow up from #1481 by @younesbelkada in https://github.com/huggingface/peft/pull/1487
* CI: temporary disable workflow by @younesbelkada in https://github.com/huggingface/peft/pull/1534
* FIX [`Docs`/ `bnb` / `DeepSpeed`] Add clarification on bnb + PEFT + DS compatibilities by @younesbelkada in https://github.com/huggingface/peft/pull/1529
* Expose bias attribute on tuner layers by @BenjaminBossan in https://github.com/huggingface/peft/pull/1530
* docs: highlight difference between `num_parameters()` and `get_nb_trainable_parameters()` in PEFT by @kmehant in https://github.com/huggingface/peft/pull/1531
* fix: fail when required args not passed when `prompt_tuning_init==TEXT` by @kmehant in https://github.com/huggingface/peft/pull/1519
* Fixed minor grammatical and code bugs by @gremlin97 in https://github.com/huggingface/peft/pull/1542
* Optimize `levenshtein_distance` algorithm in `peft_lora_seq2seq_accelera…` by @SUNGOD3 in https://github.com/huggingface/peft/pull/1527
* Update `prompt_based_methods.md` by @insist93 in https://github.com/huggingface/peft/pull/1548
* FIX Allow AdaLoRA rank to be 0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1540
* FIX: Make adaptation prompt CI happy for transformers 4.39.0 by @younesbelkada in https://github.com/huggingface/peft/pull/1551
* MNT: Use `BitsAndBytesConfig` as `load_in_*` is deprecated by @BenjaminBossan in https://github.com/huggingface/peft/pull/1552
* Add Support for Mistral Model in Llama-Adapter Method by @PrakharSaxena24 in https://github.com/huggingface/peft/pull/1433
* Add support for layer replication in LoRA by @siddartha-RE in https://github.com/huggingface/peft/pull/1368
* QDoRA: Support DoRA with BnB quantization by @BenjaminBossan in https://github.com/huggingface/peft/pull/1518
* Feat: add support for Conv2D DoRA by @sayakpaul in https://github.com/huggingface/peft/pull/1516
* TST Report slowest tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1556
* Changes to support fsdp+qlora and dsz3+qlora by @pacman100 in https://github.com/huggingface/peft/pull/1550
* Update style with ruff 0.2.2 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1565
* FEAT Mixing different LoRA adapters in same batch by @BenjaminBossan in https://github.com/huggingface/peft/pull/1558
* FIX [`CI`] Fix test docker CI by @younesbelkada in https://github.com/huggingface/peft/pull/1535
* Fix LoftQ docs and tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1532
* More convenient way to initialize LoftQ by @BenjaminBossan in https://github.com/huggingface/peft/pull/1543

## New Contributors
* @DopeorNope-Lee made their first contribution in https://github.com/huggingface/peft/pull/1372
* @kmehant made their first contribution in https://github.com/huggingface/peft/pull/1531
* @gremlin97 made their first contribution in https://github.com/huggingface/peft/pull/1542
* @SUNGOD3 made their first contribution in https://github.com/huggingface/peft/pull/1527
* @insist93 made their first contribution in https://github.com/huggingface/peft/pull/1548
* @PrakharSaxena24 made their first contribution in https://github.com/huggingface/peft/pull/1433
* @siddartha-RE made their first contribution in https://github.com/huggingface/peft/pull/1368

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.9.0...v0.10.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.10.0)

---

## v0.9.0: v0.9.0: Merging LoRA weights, new quantization options, DoRA support, and more
**Published:** 2024-02-28

## Highlights

### New methods for merging LoRA weights together
![cat_teapot](https://github.com/huggingface/peft/assets/13534540/5329d4f8-fe17-448e-94dc-b97a8e621659)


With PR #1364, we added new methods for merging LoRA weights together. This is _not_ about merging LoRA weights into the base model. Instead, this is about merging the weights from _different LoRA adapters_ into a single adapter by calling `add_weighted_adapter`. This allows you to combine the strength from multiple LoRA adapters into a single adapter, while being faster than activating each of these adapters individually.

Although this feature has already existed in PEFT for some time, we have added new merging methods that promise much better results. The first is based on [TIES](https://arxiv.org/abs/2306.01708), the second on [DARE](https://arxiv.org/abs/2311.03099) and a new one inspired by both called **Magnitude Prune**. If you haven't tried these new methods, or haven't touched the LoRA weight merging feature at all, you can find more information here:

- [Blog post](https://huggingface.co/blog/peft_merging)
- [PEFT docs](https://huggingface.co/docs/peft/main/en/developer_guides/lora#merge-adapters)
- [Example notebook using diffusers](https://github.com/huggingface/peft/blob/main/examples/multi_adapter_examples/multi_adapter_weighted_inference_diffusers.ipynb)
- [Example notebook using an LLM](https://github.com/huggingface/peft/blob/main/examples/multi_adapter_examples/Lora_Merging.ipynb)

### AWQ and AQLM support for LoRA

Via #1394, we now support [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) in PEFT. This is a new method for 4bit quantization of model weights. 

<img width="1197" alt="Screenshot 2024-02-28 at 09 41 40" src="https://github.com/huggingface/peft/assets/49240599/431d485b-c2b9-4e49-b407-89977875e6ef">

Similarly, we now support [AQLM](https://github.com/Vahe1994/AQLM) via #1476. This method allows to quantize weights to as low as 2 bits. Both methods support quantizing `nn.Linear` layers. To find out more about all the quantization options that work with PEFT, check out our docs [here](https://huggingface.co/docs/peft/developer_guides/quantization).

<img width="1197" alt="Screenshot 2024-02-28 at 09 42 22" src="https://github.com/huggingface/peft/assets/49240599/6f1e250b-8981-4e2a-9fa2-028d76150912">

Note these integrations do not support `merge_and_unload()` yet, meaning for inference you need to always attach the adapter weights into the base model 

## DoRA support

We now support Weight-Decomposed Low-Rank Adaptation aka [DoRA](https://arxiv.org/abs/2402.09353) via #1474. This new method is builds on top of LoRA and has shown very promising results. Especially at lower ranks (e.g. `r=8`), it should perform much better than LoRA. Right now, only non-quantized `nn.Linear` layers are supported. If you'd like to give it a try, just pass `use_dora=True` to your `LoraConfig` and you're good to go.

### Documentation

Thanks to @stevhliu and many other contributors, there have been big improvements to the documentation. You should find it more organized and more up-to-date. Our [DeepSpeed](https://huggingface.co/docs/peft/accelerate/deepspeed) and [FSDP](https://huggingface.co/docs/peft/accelerate/fsdp) guides have also been much improved.

[Check out our improved docs](https://huggingface.co/docs/peft/index) if you haven't already!

### Development

If you're implementing custom adapter layers, for instance a custom `LoraLayer`, note that all subclasses should now implement `update_layer` -- unless they want to use the default method by the parent class. In particular, this means you should no longer use different method names for the subclass, like `update_layer_embedding`. Also, we generally don't permit ranks (`r`) of 0 anymore. For more, see [this PR](https://github.com/huggingface/peft/pull/1268).

Developers should have an easier time now since we fully [embrace ruff](https://github.com/huggingface/peft/pull/1421). If you're the type of person who forgets to call `make style` before pushing to a PR, consider adding a [pre-commit hook](https://huggingface.co/docs/peft/developer_guides/contributing#tests-and-code-quality-checks). Tests are now a bit less verbose by using [plain asserts](https://github.com/huggingface/peft/pull/1448) and generally embracing pytest features more fully. All of this comes thanks to @akx.

## What's Changed

On top of these changes, we have added a lot of small changes since the last release, check out the full changes below. As always, we had a lot of support by many contributors, you're awesome!

* Release patch version 0.8.2 by @pacman100 in https://github.com/huggingface/peft/pull/1428
* [docs] Polytropon API by @stevhliu in https://github.com/huggingface/peft/pull/1422
* Fix `MatMul8bitLtBackward` view issue by @younesbelkada in https://github.com/huggingface/peft/pull/1425
* Fix typos by @szepeviktor in https://github.com/huggingface/peft/pull/1435
* Fixed saving for models that don't have _name_or_path in config by @kovalexal in https://github.com/huggingface/peft/pull/1440
* [docs] README update by @stevhliu in https://github.com/huggingface/peft/pull/1411
* [docs] Doc maintenance by @stevhliu in https://github.com/huggingface/peft/pull/1394
* [`core`/`TPLinear`] Fix breaking change by @younesbelkada in https://github.com/huggingface/peft/pull/1439
* Renovate quality tools by @akx in https://github.com/huggingface/peft/pull/1421
* [Docs] call `set_adapters()` after add_weighted_adapter by @sayakpaul in https://github.com/huggingface/peft/pull/1444
* MNT: Check only selected directories with ruff by @BenjaminBossan in https://github.com/huggingface/peft/pull/1446
* TST: Improve test coverage by skipping fewer tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1445
* Update Dockerfile to reflect how to compile bnb from source by @younesbelkada in https://github.com/huggingface/peft/pull/1437
* [docs] Lora-like guides by @stevhliu in https://github.com/huggingface/peft/pull/1371
* [docs] IA3 by @stevhliu in https://github.com/huggingface/peft/pull/1373
* Add docstrings for set_adapter and keep frozen by @EricLBuehler in https://github.com/huggingface/peft/pull/1447
* Add new merging methods by @pacman100 in https://github.com/huggingface/peft/pull/1364
* FIX Loading with AutoPeftModel.from_pretrained by @BenjaminBossan in https://github.com/huggingface/peft/pull/1449
* Support `modules_to_save` config option when using DeepSpeed ZeRO-3 with ZeRO init enabled. by @pacman100 in https://github.com/huggingface/peft/pull/1450
* FIX Honor HF_HUB_OFFLINE mode if set by user by @BenjaminBossan in https://github.com/huggingface/peft/pull/1454
* [docs] Remove iframe by @stevhliu in https://github.com/huggingface/peft/pull/1456
* [docs] Docstring typo by @stevhliu in https://github.com/huggingface/peft/pull/1455
* [`core` / `get_peft_state_dict`] Ignore all exceptions to avoid unexpected errors by @younesbelkada in https://github.com/huggingface/peft/pull/1458
* [ `Adaptation Prompt`] Fix llama rotary embedding issue with transformers main by @younesbelkada in https://github.com/huggingface/peft/pull/1459
* [`CI`] Add CI tests on transformers main to catch early bugs by @younesbelkada in https://github.com/huggingface/peft/pull/1461
* Use plain asserts in tests by @akx in https://github.com/huggingface/peft/pull/1448
* Add default IA3 target modules for Mixtral by @arnavgarg1 in https://github.com/huggingface/peft/pull/1376
* add `magnitude_prune` merging method by @pacman100 in https://github.com/huggingface/peft/pull/1466
* [docs] Model merging by @stevhliu in https://github.com/huggingface/peft/pull/1423
* Adds an example notebook for showing multi-adapter weighted inference by @sayakpaul in https://github.com/huggingface/peft/pull/1471
* Make tests succeed more on MPS by @akx in https://github.com/huggingface/peft/pull/1463
* [`CI`] Fix adaptation prompt CI on transformers main by @younesbelkada in https://github.com/huggingface/peft/pull/1465
* Update docstring at peft_types.py by @eduardozamudio in https://github.com/huggingface/peft/pull/1475
* FEAT: add awq suppot in PEFT by @younesbelkada in https://github.com/huggingface/peft/pull/1399
* Add pre-commit configuration by @akx in https://github.com/huggingface/peft/pull/1467
* ENH [`CI`] Run tests only when relevant files are modified by @younesbelkada in https://github.com/huggingface/peft/pull/1482
* FIX [`CI` / `bnb`] Fix failing bnb workflow by @younesbelkada in https://github.com/huggingface/peft/pull/1480
* FIX [`PromptTuning`] Simple fix for transformers >= 4.38 by @younesbelkada in https://github.com/huggingface/peft/pull/1484
* FIX: Multitask prompt tuning with other tuning init by @BenjaminBossan in https://github.com/huggingface/peft/pull/1144
* previous_dtype is now inferred from F.linear's result output type. by @MFajcik in https://github.com/huggingface/peft/pull/1010
* ENH: [`CI` / `Docker`]: Create a workflow to temporarly build docker images in case dockerfiles are modified by @younesbelkada in https://github.com/huggingface/peft/pull/1481
* Fix issue with unloading double wrapped modules by @BenjaminBossan in https://github.com/huggingface/peft/pull/1490
* FIX: [`CI` / `Adaptation Prompt`] Fix CI on transformers main by @younesbelkada in https://github.com/huggingface/peft/pull/1493
* Update peft_bnb_whisper_large_v2_training.ipynb: Fix a typo by @martin0258 in https://github.com/huggingface/peft/pull/1494
* covert SVDLinear dtype by @PHOSPHENES8 in https://github.com/huggingface/peft/pull/1495
* Raise error on wrong type for to modules_to_save by @BenjaminBossan in https://github.com/huggingface/peft/pull/1496
* AQLM support for LoRA by @BlackSamorez in https://github.com/huggingface/peft/pull/1476
* Allow trust_remote_code for tokenizers when loading AutoPeftModels by @OfficialDelta in https://github.com/huggingface/peft/pull/1477
* Add default LoRA and IA3 target modules for Gemma by @arnavgarg1 in https://github.com/huggingface/peft/pull/1499
* FIX Bug in prompt learning after disabling adapter by @BenjaminBossan in https://github.com/huggingface/peft/pull/1502
* add example and update deepspeed/FSDP docs by @pacman100 in https://github.com/huggingface/peft/pull/1489
* FIX Safe merging with LoHa and LoKr by @BenjaminBossan in https://github.com/huggingface/peft/pull/1505
* ENH: [`Docker`] Notify us when docker build pass or fail by @younesbelkada in https://github.com/huggingface/peft/pull/1503
* Implement DoRA by @BenjaminBossan in https://github.com/huggingface/peft/pull/1474

## New Contributors
* @szepeviktor made their first contribution in https://github.com/huggingface/peft/pull/1435
* @akx made their first contribution in https://github.com/huggingface/peft/pull/1421
* @EricLBuehler made their first contribution in https://github.com/huggingface/peft/pull/1447
* @eduardozamudio made their first contribution in https://github.com/huggingface/peft/pull/1475
* @MFajcik made their first contribution in https://github.com/huggingface/peft/pull/1010
* @martin0258 made their first contribution in https://github.com/huggingface/peft/pull/1494
* @PHOSPHENES8 made their first contribution in https://github.com/huggingface/peft/pull/1495
* @BlackSamorez made their first contribution in https://github.com/huggingface/peft/pull/1476
* @OfficialDelta made their first contribution in https://github.com/huggingface/peft/pull/1477

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.8.2...v0.9.0


[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.9.0)

---

## v0.8.2: Release v0.8.2
**Published:** 2024-02-01

## What's Changed
* Release v0.8.2.dev0 by @pacman100 in https://github.com/huggingface/peft/pull/1416
* Add IA3 Modules for Phi by @arnavgarg1 in https://github.com/huggingface/peft/pull/1407
* Update custom_models.md by @boyufan in https://github.com/huggingface/peft/pull/1409
* Add positional args to PeftModelForCausalLM.generate by @SumanthRH in https://github.com/huggingface/peft/pull/1393
* [Hub] fix: subfolder existence check by @sayakpaul in https://github.com/huggingface/peft/pull/1417
* FIX: Make merging of adapter weights idempotent by @BenjaminBossan in https://github.com/huggingface/peft/pull/1355
* [`core`] fix critical bug in diffusers by @younesbelkada in https://github.com/huggingface/peft/pull/1427

## New Contributors
* @boyufan made their first contribution in https://github.com/huggingface/peft/pull/1409

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.8.1...v0.8.2

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.8.2)

---

## v0.8.1: Patch Release v0.8.1
**Published:** 2024-01-30

This is a small patch release of PEFT that should:
* Fix breaking change  related to support for saving resized embedding layers and Diffusers models. Contributed by @younesbelkada in https://github.com/huggingface/peft/pull/1414

## What's Changed
* Release 0.8.1.dev0 by @pacman100 in https://github.com/huggingface/peft/pull/1412
* Fix breaking change  by @younesbelkada in https://github.com/huggingface/peft/pull/1414
* Patch Release v0.8.1 by @pacman100 in https://github.com/huggingface/peft/pull/1415


**Full Changelog**: https://github.com/huggingface/peft/compare/v0.8.0...v0.8.1

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.8.1)

---

## v0.8.0: v0.8.0: Poly PEFT method, LoRA improvements, Documentation improvements and more
**Published:** 2024-01-30

# Highlights

## Poly PEFT method
Parameter-efficient fine-tuning (PEFT) for cross-task generalization consists of pre-training adapters on a multi-task training set before few-shot adaptation to test tasks. Polytropon [Ponti et al., 2023] (𝙿𝚘𝚕𝚢) jointly learns an inventory of adapters and a routing function that selects a (variable-size) subset of adapters for each task during both pre-training and few-shot adaptation. To put simply, you can think of it as Mixture of Expert Adapters. 
𝙼𝙷𝚁 (Multi-Head Routing) combines subsets of adapter parameters and outperforms 𝙿𝚘𝚕𝚢 under a comparable parameter budget; by only fine-tuning the routing function and not the adapters (𝙼𝙷𝚁-z) they achieve competitive performance with extreme parameter efficiency.

* Add Poly by @TaoSunVoyage in https://github.com/huggingface/peft/pull/1129

## LoRA improvements

Now, you can specify `all-linear` to `target_modules` param of `LoraConfig` to target all the linear layers which has shown to perform better in QLoRA paper than only targeting query and valuer attention layers
* Add an option 'ALL' to include all linear layers as target modules by @SumanthRH in https://github.com/huggingface/peft/pull/1295

Embedding layers of base models are now automatically saved when the embedding layers are resized when fine-tuning with PEFT approaches like LoRA. This enables extending the vocabulary of tokenizer to include special tokens. This is a common use-case when doing the following:
1. Instruction finetuning with new tokens being added such as <|user|>, <|assistant|>, <|system|>, <|im_end|>, <|im_start|>, <\/s>, \<s> to properly format the conversations
2. Finetuning on a specific language wherein language specific tokens are added, e.g., Korean tokens being added to vocabulary for finetuning LLM on Korean datasets.
3. Instruction finetuning to return outputs in a certain format to enable agent behaviour of new tokens such as <|FUNCTIONS|>, <|BROWSE|>, <|TEXT2IMAGE|>, <|ASR|>, <|TTS|>, <|GENERATECODE|>, <|RAG|>.
A good blogpost to learn more about this https://www.philschmid.de/fine-tune-llms-in-2024-with-trl.
* save the embeddings even when they aren't targetted but resized by @pacman100 in https://github.com/huggingface/peft/pull/1383

New option `use_rslora` in LoraConfig. Use it for ranks greater than 32 and see the increase in fine-tuning performance (same or better performance for ranks lower than 32 as well).
* Added the option to use the corrected scaling factor for LoRA, based on new research. by @Damjan-Kalajdzievski in https://github.com/huggingface/peft/pull/1244

## Documentation improvements
* Refactoring and updating of the concept guides. [docs] Concept guides by @stevhliu in https://github.com/huggingface/peft/pull/1269
* Improving task guides to focus more on how to use different PEFT methods and related nuances instead of focusing more on different type of tasks. It condenses the individual guides into a single one to highlight the commonalities and differences, and to refer to existing docs to avoid duplication. [docs] Task guides by @stevhliu in https://github.com/huggingface/peft/pull/1332
* DOC: Update docstring for the config classes by @BenjaminBossan in https://github.com/huggingface/peft/pull/1343
* LoftQ: edit README.md and example files by @yxli2123 in https://github.com/huggingface/peft/pull/1276
* [Docs] make add_weighted_adapter example clear in the docs. by @sayakpaul in https://github.com/huggingface/peft/pull/1353
* DOC Add PeftMixedModel to API docs by @BenjaminBossan in https://github.com/huggingface/peft/pull/1354
* [docs] Docstring link by @stevhliu in https://github.com/huggingface/peft/pull/1356
* QOL improvements and doc updates by @pacman100 in https://github.com/huggingface/peft/pull/1318
* Doc about AdaLoraModel.update_and_allocate by @kuronekosaiko in https://github.com/huggingface/peft/pull/1341
* DOC: Improve target modules description by @BenjaminBossan in https://github.com/huggingface/peft/pull/1290
* DOC Troubleshooting for unscaling error with fp16 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1336
* DOC Extending the vocab and storing embeddings by @BenjaminBossan in https://github.com/huggingface/peft/pull/1335
* Improve documentation for the `all-linear` flag by @SumanthRH in https://github.com/huggingface/peft/pull/1357
* Fix various typos in LoftQ docs. by @arnavgarg1 in https://github.com/huggingface/peft/pull/1408

## What's Changed
* Bump version to 0.7.2.dev0 post release by @BenjaminBossan in https://github.com/huggingface/peft/pull/1258
* FIX Error in log_reports.py by @BenjaminBossan in https://github.com/huggingface/peft/pull/1261
* Fix ModulesToSaveWrapper __getattr__ by @zhangsheng377 in https://github.com/huggingface/peft/pull/1238
* TST: Revert device_map for AdaLora 4bit GPU test by @BenjaminBossan in https://github.com/huggingface/peft/pull/1266
* remove a duplicated description in peft BaseTuner by @butyuhao in https://github.com/huggingface/peft/pull/1271
* Added the option to use the corrected scaling factor for LoRA, based on new research. by @Damjan-Kalajdzievski in https://github.com/huggingface/peft/pull/1244
* feat: add apple silicon GPU acceleration by @NripeshN in https://github.com/huggingface/peft/pull/1217
* LoftQ: Allow quantizing models loaded on the CPU for LoftQ initialization by @hiyouga in https://github.com/huggingface/peft/pull/1256
* LoftQ: edit README.md and example files by @yxli2123 in https://github.com/huggingface/peft/pull/1276
* TST: Extend LoftQ tests to check CPU initialization by @BenjaminBossan in https://github.com/huggingface/peft/pull/1274
* Refactor and a couple of fixes for adapter layer updates by @BenjaminBossan in https://github.com/huggingface/peft/pull/1268
* [`Tests`] Add bitsandbytes installed from source on new docker images by @younesbelkada in https://github.com/huggingface/peft/pull/1275
* TST: Enable LoftQ 8bit tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1279
* [`bnb`] Add bnb nightly workflow by @younesbelkada in https://github.com/huggingface/peft/pull/1282
* Fixed several errors in StableDiffusion adapter conversion script by @kovalexal in https://github.com/huggingface/peft/pull/1281
* [docs] Concept guides by @stevhliu in https://github.com/huggingface/peft/pull/1269
* DOC: Improve target modules description by @BenjaminBossan in https://github.com/huggingface/peft/pull/1290
* [`bnb-nightly`] Address final comments by @younesbelkada in https://github.com/huggingface/peft/pull/1287
* [BNB] Fix bnb dockerfile for latest version  by @SunMarc in https://github.com/huggingface/peft/pull/1291
* fix fsdp auto wrap policy by @pacman100 in https://github.com/huggingface/peft/pull/1302
* [BNB] fix dockerfile for single gpu by @SunMarc in https://github.com/huggingface/peft/pull/1305
* Fix bnb lora layers not setting active adapter by @tdrussell in https://github.com/huggingface/peft/pull/1294
* Mistral IA3 config defaults by @pacman100 in https://github.com/huggingface/peft/pull/1316
* fix the embedding saving for adaption prompt by @pacman100 in https://github.com/huggingface/peft/pull/1314
* fix diffusers tests by @pacman100 in https://github.com/huggingface/peft/pull/1317
* FIX Use torch.long instead of torch.int in LoftQ for PyTorch versions <2.x by @BenjaminBossan in https://github.com/huggingface/peft/pull/1320
* Extend merge_and_unload to offloaded models by @blbadger in https://github.com/huggingface/peft/pull/1190
* Add an option 'ALL' to include all linear layers as target modules by @SumanthRH in https://github.com/huggingface/peft/pull/1295
* Refactor dispatching logic of LoRA layers by @BenjaminBossan in https://github.com/huggingface/peft/pull/1319
* Fix bug when load the prompt tuning in inference. by @yileld in https://github.com/huggingface/peft/pull/1333
* DOC Troubleshooting for unscaling error with fp16 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1336
* ENH: Add attribute to show targeted module names by @BenjaminBossan in https://github.com/huggingface/peft/pull/1330
* fix some args desc by @zspo in https://github.com/huggingface/peft/pull/1338
* Fix logic in target module finding by @s-k-yx in https://github.com/huggingface/peft/pull/1263
* Doc about AdaLoraModel.update_and_allocate by @kuronekosaiko in https://github.com/huggingface/peft/pull/1341
* DOC: Update docstring for the config classes by @BenjaminBossan in https://github.com/huggingface/peft/pull/1343
* fix `prepare_inputs_for_generation` logic for Prompt Learning methods by @pacman100 in https://github.com/huggingface/peft/pull/1352
* QOL improvements and doc updates by @pacman100 in https://github.com/huggingface/peft/pull/1318
* New transformers caching ETA now v4.38 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1348
* FIX Setting active adapter for quantized layers by @BenjaminBossan in https://github.com/huggingface/peft/pull/1347
* DOC Extending the vocab and storing embeddings by @BenjaminBossan in https://github.com/huggingface/peft/pull/1335
* [Docs] make add_weighted_adapter example clear in the docs. by @sayakpaul in https://github.com/huggingface/peft/pull/1353
* DOC Add PeftMixedModel to API docs by @BenjaminBossan in https://github.com/huggingface/peft/pull/1354
* Add Poly by @TaoSunVoyage in https://github.com/huggingface/peft/pull/1129
* [docs] Docstring link by @stevhliu in https://github.com/huggingface/peft/pull/1356
* Added missing getattr dunder methods for mixed model by @kovalexal in https://github.com/huggingface/peft/pull/1365
* Handle resizing of embedding layers for AutoPeftModel by @pacman100 in https://github.com/huggingface/peft/pull/1367
* account for the new merged/unmerged weight to perform the quantization again by @pacman100 in https://github.com/huggingface/peft/pull/1370
* add mixtral in LoRA mapping by @younesbelkada in https://github.com/huggingface/peft/pull/1380
* save the embeddings even when they aren't targetted but resized by @pacman100 in https://github.com/huggingface/peft/pull/1383
* Improve documentation for the `all-linear` flag by @SumanthRH in https://github.com/huggingface/peft/pull/1357
* Fix LoRA module mapping for Phi models by @arnavgarg1 in https://github.com/huggingface/peft/pull/1375
* [docs] Task guides by @stevhliu in https://github.com/huggingface/peft/pull/1332
* Add generic PeftConfig constructor from kwargs by @sfriedowitz in https://github.com/huggingface/peft/pull/1398
* Fix various typos in LoftQ docs. by @arnavgarg1 in https://github.com/huggingface/peft/pull/1408
* Release: v0.8.0 by @pacman100 in https://github.com/huggingface/peft/pull/1406

## New Contributors
* @butyuhao made their first contribution in https://github.com/huggingface/peft/pull/1271
* @Damjan-Kalajdzievski made their first contribution in https://github.com/huggingface/peft/pull/1244
* @NripeshN made their first contribution in https://github.com/huggingface/peft/pull/1217
* @hiyouga made their first contribution in https://github.com/huggingface/peft/pull/1256
* @tdrussell made their first contribution in https://github.com/huggingface/peft/pull/1294
* @blbadger made their first contribution in https://github.com/huggingface/peft/pull/1190
* @yileld made their first contribution in https://github.com/huggingface/peft/pull/1333
* @s-k-yx made their first contribution in https://github.com/huggingface/peft/pull/1263
* @kuronekosaiko made their first contribution in https://github.com/huggingface/peft/pull/1341
* @TaoSunVoyage made their first contribution in https://github.com/huggingface/peft/pull/1129
* @arnavgarg1 made their first contribution in https://github.com/huggingface/peft/pull/1375
* @sfriedowitz made their first contribution in https://github.com/huggingface/peft/pull/1398

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.7.1...v0.8.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.8.0)

---

## v0.7.1: v0.7.1 patch release
**Published:** 2023-12-12

This is a small patch release of PEFT that should handle:

- Issues with loading multiple adapters when using quantized models (#1243)
- Issues with transformers v4.36 and some prompt learning methods (#1252)

## What's Changed
* [docs] OFT by @stevhliu in https://github.com/huggingface/peft/pull/1221
* Bump version to 0.7.1.dev0 post release by @BenjaminBossan in https://github.com/huggingface/peft/pull/1227
* Don't set config attribute on custom models by @BenjaminBossan in https://github.com/huggingface/peft/pull/1200
* TST: Run regression test in nightly test runner by @BenjaminBossan in https://github.com/huggingface/peft/pull/1233
* Lazy import of bitsandbytes by @BenjaminBossan in https://github.com/huggingface/peft/pull/1230
* FIX: Pin bitsandbytes to <0.41.3 temporarily by @BenjaminBossan in https://github.com/huggingface/peft/pull/1234
* [docs] PeftConfig and PeftModel by @stevhliu in https://github.com/huggingface/peft/pull/1211
* TST: Add tolerance for regression tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1241
* Bnb integration test tweaks by @Titus-von-Koeller in https://github.com/huggingface/peft/pull/1242
* [docs] PEFT integrations by @stevhliu in https://github.com/huggingface/peft/pull/1224
* Revert "FIX Pin bitsandbytes to <0.41.3 temporarily (#1234)" by @Titus-von-Koeller in https://github.com/huggingface/peft/pull/1250
* Fix model argument issue (#1198) by @ngocbh in https://github.com/huggingface/peft/pull/1205
* TST: Add tests for 4bit LoftQ by @BenjaminBossan in https://github.com/huggingface/peft/pull/1208
* [docs] Quantization by @stevhliu in https://github.com/huggingface/peft/pull/1236
* FIX: Truncate slack message to not exceed 3000 chars by @BenjaminBossan in https://github.com/huggingface/peft/pull/1251
* Issue with transformers 4.36 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1252
* Fix: Multiple adapters with bnb layers by @BenjaminBossan in https://github.com/huggingface/peft/pull/1243
* Release: 0.7.1 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1257

## New Contributors
* @Titus-von-Koeller made their first contribution in https://github.com/huggingface/peft/pull/1242
* @ngocbh made their first contribution in https://github.com/huggingface/peft/pull/1205

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.7.0...v0.7.1

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.7.1)

---

## v0.7.0: v0.7.0: Orthogonal Fine-Tuning, Megatron support, better initialization, safetensors, and more
**Published:** 2023-12-06

# Highlights

* Orthogonal Fine-Tuning (OFT): A new adapter that is similar to LoRA and shows a lot of promise for Stable Diffusion, especially with regard to controllability and compositionality. [Give it a try](https://github.com/huggingface/peft/blob/main/examples/oft_dreambooth/train_dreambooth.py)! By @okotaku in https://github.com/huggingface/peft/pull/1160
* Support for parallel linear LoRA layers using Megatron. This should lead to a speed up when using LoRA with Megatron. By @zhangsheng377 in https://github.com/huggingface/peft/pull/1092
* LoftQ provides a new method to [initialize LoRA layers](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora#initialization-options) of quantized models. The big advantage is that the LoRA layer weights are chosen in a way to minimize the quantization error, as described here: https://arxiv.org/abs/2310.08659. By @yxli2123 in https://github.com/huggingface/peft/pull/1150.

## Other notable additions

* It is now possible to choose which adapters are merged when calling `merge` (#1132)
* IA³ now supports adapter deletion, by @alexrs (#1153)
* A new initialization method for LoRA has been added, `"gaussian"` (#1189)
* When training PEFT models with new tokens being added to the embedding layers, the embedding layer is now saved by default (#1147)
* It is now possible to mix certain adapters like LoRA and LoKr in the same model, see [the docs](https://huggingface.co/docs/peft/main/en/developer_guides/mixed_models) (#1163)
* We started an initiative to improve the documenation, some of which should already be reflected in the current docs. Still, help by the community is always welcome. Check out [this issue](https://github.com/huggingface/peft/issues/1089) to get going.

## Migration to v0.7.0

- [Safetensors](https://huggingface.co/docs/safetensors/index) are now the [default format](https://github.com/huggingface/peft/pull/1088) for PEFT adapters. In practice, users should not have to change anything in their code, PEFT takes care of everything -- just be aware that instead of creating a file `adapter_model.bin`, calling `save_pretrained` now creates `adapter_model.safetensors`. Safetensors have numerous advantages over pickle files (which is the PyTorch default format) and well supported on Hugging Face Hub.
- When merging multiple LoRA adapter weights together using `add_weighted_adapter` with the option `combination_type="linear"`, the scaling of the adapter weights is now [performed differently](https://github.com/huggingface/peft/pull/1169), leading to improved results.
- There was a big refactor of the inner workings of some PEFT adapters. For the vast majority of users, this should not make any difference (except making some code run faster). However, if your code is relying on PEFT internals, be aware that the inheritance structure of certain adapter layers has changed (e.g. `peft.lora.Linear` is no longer a subclass of `nn.Linear`, so `isinstance` checks may need updating). Also, to retrieve the original weight of an adapted layer, now use `self.get_base_layer().weight`, not `self.weight` (same for `bias`).

## What's Changed

As always, a bunch of small improvements, bug fixes and doc improvements were added. We thank all the external contributors, both new and recurring. Below is the list of all changes since the last release.

* After release: Bump version to 0.7.0.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1074
* FIX: Skip adaption prompt tests with new transformers versions by @BenjaminBossan in https://github.com/huggingface/peft/pull/1077
* FIX: fix adaptation prompt CI and compatibility with latest transformers (4.35.0) by @younesbelkada in https://github.com/huggingface/peft/pull/1084
* Improve documentation for IA³ by @SumanthRH in https://github.com/huggingface/peft/pull/984
* [`Docker`] Update Dockerfile to force-use transformers main by @younesbelkada in https://github.com/huggingface/peft/pull/1085
* Update the release checklist by @BenjaminBossan in https://github.com/huggingface/peft/pull/1075
* fix-gptq-training by @SunMarc in https://github.com/huggingface/peft/pull/1086
* fix the failing CI tests by @pacman100 in https://github.com/huggingface/peft/pull/1094
* Fix f-string in import_utils by @KCFindstr in https://github.com/huggingface/peft/pull/1091
* Fix IA3 config for Falcon models by @SumanthRH in https://github.com/huggingface/peft/pull/1007
* FIX: Failing nightly CI tests due to IA3 config by @BenjaminBossan in https://github.com/huggingface/peft/pull/1100
* [`core`] Fix safetensors serialization for shared tensors by @younesbelkada in https://github.com/huggingface/peft/pull/1101
* Change to 0.6.1.dev0 by @younesbelkada in https://github.com/huggingface/peft/pull/1102
* Release: 0.6.1 by @younesbelkada in https://github.com/huggingface/peft/pull/1103
* set dev version by @younesbelkada in https://github.com/huggingface/peft/pull/1104
* avoid unnecessary import by @winglian in https://github.com/huggingface/peft/pull/1109
* Refactor adapter deletion by @BenjaminBossan in https://github.com/huggingface/peft/pull/1105
* Added num_dataloader_workers arg to fix Windows issue by @lukaskuhn-lku in https://github.com/huggingface/peft/pull/1107
* Fix import issue transformers with `id_tensor_storage` by @younesbelkada in https://github.com/huggingface/peft/pull/1116
* Correctly deal with  `ModulesToSaveWrapper` when using Low-level API by @younesbelkada in https://github.com/huggingface/peft/pull/1112
* fix doc typo by @coding-famer in https://github.com/huggingface/peft/pull/1121
* Release: v0.6.2 by @pacman100 in https://github.com/huggingface/peft/pull/1125
* Release: v0.6.3.dev0 by @pacman100 in https://github.com/huggingface/peft/pull/1128
* FIX: Adding 2 adapters when target_modules is a str fails by @BenjaminBossan in https://github.com/huggingface/peft/pull/1111
* Prompt tuning: Allow to pass additional args to AutoTokenizer.from_pretrained by @BenjaminBossan in https://github.com/huggingface/peft/pull/1053
* Fix: TorchTracemalloc ruins Windows performance by @lukaskuhn-lku in https://github.com/huggingface/peft/pull/1126
* TST: Improve requires grad testing: by @BenjaminBossan in https://github.com/huggingface/peft/pull/1131
* FEAT: Make safe serialization the default one by @younesbelkada in https://github.com/huggingface/peft/pull/1088
* FEAT: Merging only specified `adapter_names` when calling `merge` by @younesbelkada in https://github.com/huggingface/peft/pull/1132
* Refactor base layer pattern by @BenjaminBossan in https://github.com/huggingface/peft/pull/1106
* [`Tests`] Fix daily CI by @younesbelkada in https://github.com/huggingface/peft/pull/1136
* [`core` / `LoRA`] Add `adapter_names` in bnb layers by @younesbelkada in https://github.com/huggingface/peft/pull/1139
* [`Tests`] Do not stop tests if a job failed by @younesbelkada in https://github.com/huggingface/peft/pull/1141
* CI Add Python 3.11 to test matrix by @BenjaminBossan in https://github.com/huggingface/peft/pull/1143
* FIX: A few issues with AdaLora, extending GPU tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/1146
* Use `huggingface_hub.file_exists` instead of custom helper by @Wauplin in https://github.com/huggingface/peft/pull/1145
* Delete IA3 adapter by @alexrs in https://github.com/huggingface/peft/pull/1153
* [Docs fix] Relative path issue by @mishig25 in https://github.com/huggingface/peft/pull/1157
* Dataset was loaded twice in 4-bit finetuning script by @lukaskuhn-lku in https://github.com/huggingface/peft/pull/1164
* fix `add_weighted_adapter` method by @pacman100 in https://github.com/huggingface/peft/pull/1169
* (minor) correct type annotation by @vwxyzjn in https://github.com/huggingface/peft/pull/1166
* Update release checklist about release notes by @BenjaminBossan in https://github.com/huggingface/peft/pull/1170
* [docs] Migrate doc files to Markdown by @stevhliu in https://github.com/huggingface/peft/pull/1171
* Fix dockerfile build by @younesbelkada in https://github.com/huggingface/peft/pull/1177
* FIX: Wrong use of base layer by @BenjaminBossan in https://github.com/huggingface/peft/pull/1183
* [`Tests`] Migrate to AWS runners by @younesbelkada in https://github.com/huggingface/peft/pull/1185
* Fix code example in quicktour.md by @merveenoyan in https://github.com/huggingface/peft/pull/1181
* DOC Update a few places in the README by @BenjaminBossan in https://github.com/huggingface/peft/pull/1152
* Fix issue where you cannot call PeftModel.from_pretrained with a private adapter by @elyxlz in https://github.com/huggingface/peft/pull/1076
* Added lora support for phi by @umarbutler in https://github.com/huggingface/peft/pull/1186
* add options to save or push model by @callanwu in https://github.com/huggingface/peft/pull/1159
* ENH: Different initialization methods for LoRA by @BenjaminBossan in https://github.com/huggingface/peft/pull/1189
* Training PEFT models with new tokens being added to the embedding layers and tokenizer by @pacman100 in https://github.com/huggingface/peft/pull/1147
* LoftQ: Add LoftQ method integrated into LoRA. Add example code for LoftQ usage. by @yxli2123 in https://github.com/huggingface/peft/pull/1150
* Parallel linear Lora by @zhangsheng377 in https://github.com/huggingface/peft/pull/1092
* [Feature] Support OFT by @okotaku in https://github.com/huggingface/peft/pull/1160
* Mixed adapter models by @BenjaminBossan in https://github.com/huggingface/peft/pull/1163
* [DOCS] README.md by @Akash190104 in https://github.com/huggingface/peft/pull/1054
* Fix parallel linear lora by @zhangsheng377 in https://github.com/huggingface/peft/pull/1202
* ENH: Enable OFT adapter for mixed adapter models by @BenjaminBossan in https://github.com/huggingface/peft/pull/1204
* DOC: Update & improve docstrings and type annotations for common methods and classes by @BenjaminBossan in https://github.com/huggingface/peft/pull/1201
* remove HF tokens by @yxli2123 in https://github.com/huggingface/peft/pull/1207
* [docs] Update index and quicktour by @stevhliu in https://github.com/huggingface/peft/pull/1191
* [docs] API docs by @stevhliu in https://github.com/huggingface/peft/pull/1196
* MNT: Delete the delete doc workflows by @BenjaminBossan in https://github.com/huggingface/peft/pull/1213
* DOC: Initialization options for LoRA by @BenjaminBossan in https://github.com/huggingface/peft/pull/1218
* Fix an issue with layer merging for LoHa and OFT by @lukaskuhn-lku in https://github.com/huggingface/peft/pull/1210
* DOC: How to configure new transformers models by @BenjaminBossan in https://github.com/huggingface/peft/pull/1195
* Raise error when `modules_to_save` is specified and multiple adapters are being unloaded by @pacman100 in https://github.com/huggingface/peft/pull/1137
* TST: Add regression tests 2 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1115
* Release: 0.7.0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1214

## New Contributors
* @KCFindstr made their first contribution in https://github.com/huggingface/peft/pull/1091
* @winglian made their first contribution in https://github.com/huggingface/peft/pull/1109
* @lukaskuhn-lku made their first contribution in https://github.com/huggingface/peft/pull/1107
* @coding-famer made their first contribution in https://github.com/huggingface/peft/pull/1121
* @Wauplin made their first contribution in https://github.com/huggingface/peft/pull/1145
* @alexrs made their first contribution in https://github.com/huggingface/peft/pull/1153
* @merveenoyan made their first contribution in https://github.com/huggingface/peft/pull/1181
* @elyxlz made their first contribution in https://github.com/huggingface/peft/pull/1076
* @umarbutler made their first contribution in https://github.com/huggingface/peft/pull/1186
* @callanwu made their first contribution in https://github.com/huggingface/peft/pull/1159
* @yxli2123 made their first contribution in https://github.com/huggingface/peft/pull/1150
* @zhangsheng377 made their first contribution in https://github.com/huggingface/peft/pull/1092
* @okotaku made their first contribution in https://github.com/huggingface/peft/pull/1160
* @Akash190104 made their first contribution in https://github.com/huggingface/peft/pull/1054

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.6.2...v0.7.0

# Significant community contributions

The following contributors have made significant changes to the library over the last release:

@alexrs

* Delete IA3 adapter by @alexrs in https://github.com/huggingface/peft/pull/1153

@callanwu

* add options to save or push model by @callanwu in https://github.com/huggingface/peft/pull/1159

@elyxlz

* Fix issue where you cannot call PeftModel.from_pretrained with a private adapter by @elyxlz in https://github.com/huggingface/peft/pull/1076

@lukaskuhn-lku

* Fix: TorchTracemalloc ruins Windows performance by @lukaskuhn-lku in https://github.com/huggingface/peft/pull/1126
* Dataset was loaded twice in 4-bit finetuning script by @lukaskuhn-lku in https://github.com/huggingface/peft/pull/1164

@okotaku

* [Feature] Support OFT by @okotaku in https://github.com/huggingface/peft/pull/1160

@yxli2123

* LoftQ: Add LoftQ method integrated into LoRA. Add example code for LoftQ usage. by @yxli2123 in https://github.com/huggingface/peft/pull/1150

@zhangsheng377

* Parallel linear Lora by @zhangsheng377 in https://github.com/huggingface/peft/pull/1092
* Fix parallel linear lora by @zhangsheng377 in https://github.com/huggingface/peft/pull/1202

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.7.0)

---

## v0.6.2: v0.6.2 Patch Release: Refactor of adapter deletion API and fixes to `ModulesToSaveWrapper` when using Low-level API
**Published:** 2023-11-14

This patch release refactors the adapter deletion API and fixes to `ModulesToSaveWrapper` when using Low-level API.

## Refactor adapter deletion
* Refactor adapter deletion by @BenjaminBossan in https://github.com/huggingface/peft/pull/1105

## Fix `ModulesToSaveWrapper` when using Low-level API
* Correctly deal with  `ModulesToSaveWrapper` when using Low-level API by @younesbelkada in https://github.com/huggingface/peft/pull/1112

## What's Changed

## What's Changed
* Release: 0.6.1 by @younesbelkada in https://github.com/huggingface/peft/pull/1103
* set dev version by @younesbelkada in https://github.com/huggingface/peft/pull/1104
* avoid unnecessary import by @winglian in https://github.com/huggingface/peft/pull/1109
* Refactor adapter deletion by @BenjaminBossan in https://github.com/huggingface/peft/pull/1105
* Added num_dataloader_workers arg to fix Windows issue by @lukaskuhn-lku in https://github.com/huggingface/peft/pull/1107
* Fix import issue transformers with `id_tensor_storage` by @younesbelkada in https://github.com/huggingface/peft/pull/1116
* Correctly deal with  `ModulesToSaveWrapper` when using Low-level API by @younesbelkada in https://github.com/huggingface/peft/pull/1112
* fix doc typo by @coding-famer in https://github.com/huggingface/peft/pull/1121

## New Contributors
* @winglian made their first contribution in https://github.com/huggingface/peft/pull/1109
* @lukaskuhn-lku made their first contribution in https://github.com/huggingface/peft/pull/1107
* @coding-famer made their first contribution in https://github.com/huggingface/peft/pull/1121

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.6.1...v0.6.2

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.6.2)

---

## v0.6.1: 0.6.1 Patch Release: compatibility of Adaptation Prompt with transformers 4.35.0
**Published:** 2023-11-09

This patch release fixes the compatbility issues with Adaptation Prompt that users faced with transformers 4.35.0. Moreover, it fixes an issue with token classification PEFT models when saving them using safetensors

## Adaptation prompt fixes

* FIX: Skip adaption prompt tests with new transformers versions by @BenjaminBossan in https://github.com/huggingface/peft/pull/1077
* FIX: fix adaptation prompt CI and compatibility with latest transformers (4.35.0) by @younesbelkada in https://github.com/huggingface/peft/pull/1084

## Safetensors fixes:

* [`core`] Fix safetensors serialization for shared tensors by @younesbelkada in https://github.com/huggingface/peft/pull/1101

## What's Changed
* After release: Bump version to 0.7.0.dev0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1074
* Improve documentation for IA³ by @SumanthRH in https://github.com/huggingface/peft/pull/984
* [`Docker`] Update Dockerfile to force-use transformers main by @younesbelkada in https://github.com/huggingface/peft/pull/1085
* Update the release checklist by @BenjaminBossan in https://github.com/huggingface/peft/pull/1075
* fix-gptq-training by @SunMarc in https://github.com/huggingface/peft/pull/1086
* fix the failing CI tests by @pacman100 in https://github.com/huggingface/peft/pull/1094
* Fix f-string in import_utils by @KCFindstr in https://github.com/huggingface/peft/pull/1091
* Fix IA3 config for Falcon models by @SumanthRH in https://github.com/huggingface/peft/pull/1007
* FIX: Failing nightly CI tests due to IA3 config by @BenjaminBossan in https://github.com/huggingface/peft/pull/1100
* Change to 0.6.1.dev0 by @younesbelkada in https://github.com/huggingface/peft/pull/1102

## New Contributors
* @KCFindstr made their first contribution in https://github.com/huggingface/peft/pull/1091

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.6.0...v0.6.1

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.6.1)

---

## v0.6.0: 🧨 Diffusers now uses 🤗 PEFT, new tuning methods, better quantization support, higher flexibility and more
**Published:** 2023-11-03

# Highlights

## Integration with diffusers
<img src="https://github.com/huggingface/peft/assets/13534540/57eea400-ce5f-4bf8-a499-711fd83f590b" width="600" height="600">


🧨 Diffusers now leverage PEFT as a backend for LoRA inference for Stable Diffusion models (#873, #993, #961). Relevant PRs on 🧨 Diffusers are https://github.com/huggingface/diffusers/pull/5058, https://github.com/huggingface/diffusers/pull/5147, https://github.com/huggingface/diffusers/pull/5151 and https://github.com/huggingface/diffusers/pull/5359.  This helps in unlocking a vast number of practically demanding use cases around adapter-based inference 🚀. Now you can do the following with easy-to-use APIs and it supports different checkpoint formats (Diffusers format, Kohya format ...):
1. use multiple LoRAs
2. switch between them instantaneously
3. scale and combine them
4. merge/unmerge
5. enable/disable

For details, refer to the documentation at [Inference with PEFT](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference). 

## New tuning methods

- Multitask Prompt Tuning: Thanks @mayank31398 for implementing this method from https://arxiv.org/abs/2303.02861 (#400)
- LoHa (low-rank Hadamard product): @kovalexal did a great job adding LoHa from https://arxiv.org/abs/2108.06098 (#956)
- LoKr (Kronecker Adapter): Not happy with just one new adapter, @kovalexal also added LoKr from https://arxiv.org/abs/2212.10650 to PEFT (#978)

## Other notable additions

- Allow merging of LoRA weights when using 4bit and 8bit quantization (bitsandbytes), thanks to @jiqing-feng (#851, #875)
- IA³ now supports 4bit quantization thanks to @His-Wardship (#864)
- We increased the speed of adapter layer initialization: This should be most notable when creating a PEFT LoRA model on top of a large base model (#887, #915, #994)
- More fine-grained control when configuring LoRA: It is now possible to have different ranks and alpha values for different layers (#873)

## Experimental features

- For some adapters like LoRA, it is now possible to activate multiple adapters at the same time (#873)

## Breaking changes

- It is no longer allowed to create a LoRA adapter with rank 0 (`r=0`). This used to be possible, in which case the adapter was ignored.

# What's Changed

As always, a bunch of small improvements, bug fixes and doc improvements were added. We thank all the external contributors, both new and recurring. Below is the list of all changes since the last release.

* Fixed typos in custom_models.mdx by @Psancs05 in https://github.com/huggingface/peft/pull/847
* Release version 0.6.0.dev0 by @pacman100 in https://github.com/huggingface/peft/pull/849
* DOC: Add a contribution guide by @BenjaminBossan in https://github.com/huggingface/peft/pull/848
* clarify the new model size by @stas00 in https://github.com/huggingface/peft/pull/839
* DOC: Remove backlog section from README.md by @BenjaminBossan in https://github.com/huggingface/peft/pull/853
* MNT: Refactor tuner forward methods for simplicity by @BenjaminBossan in https://github.com/huggingface/peft/pull/833
* 🎉 Add Multitask Prompt Tuning by @mayank31398 in https://github.com/huggingface/peft/pull/400
* Fix typos in ia3.py by @metaprotium in https://github.com/huggingface/peft/pull/844
* Support merge lora module for 4bit and 8bit linear by @jiqing-feng in https://github.com/huggingface/peft/pull/851
* Fix seq2seq prompt tuning (#439) by @glerzing in https://github.com/huggingface/peft/pull/809
* MNT: Move tuners to subpackages by @BenjaminBossan in https://github.com/huggingface/peft/pull/807
* FIX: Error in forward of 4bit linear lora layer by @BenjaminBossan in https://github.com/huggingface/peft/pull/878
* MNT: Run tests that were skipped previously by @BenjaminBossan in https://github.com/huggingface/peft/pull/884
* FIX: PeftModel save_pretrained Doc (#881) by @houx15 in https://github.com/huggingface/peft/pull/888
* Upgrade docker actions to higher versions by @younesbelkada in https://github.com/huggingface/peft/pull/889
* Fix error using deepspeed zero2 + load_in_8bit + lora by @tmm1 in https://github.com/huggingface/peft/pull/874
* Fix doc for semantic_segmentation_lora by @raghavanone in https://github.com/huggingface/peft/pull/891
* fix_gradient_accumulation_steps_in_examples by @zspo in https://github.com/huggingface/peft/pull/898
* FIX: linting issue in example by @BenjaminBossan in https://github.com/huggingface/peft/pull/908
* ENH Remove redundant initialization layer calls by @BenjaminBossan in https://github.com/huggingface/peft/pull/887
* [docs] Remove duplicate section by @stevhliu in https://github.com/huggingface/peft/pull/911
* support prefix tuning for starcoder models by @pacman100 in https://github.com/huggingface/peft/pull/913
* Merge lora module to 8bit model by @jiqing-feng in https://github.com/huggingface/peft/pull/875
* DOC: Section on common issues encountered with PEFT by @BenjaminBossan in https://github.com/huggingface/peft/pull/909
* Enh speed up init emb conv2d by @BenjaminBossan in https://github.com/huggingface/peft/pull/915
* Make base_model.peft_config single source of truth by @BenjaminBossan in https://github.com/huggingface/peft/pull/921
* Update accelerate dependency version by @rohithkrn in https://github.com/huggingface/peft/pull/892
* fix lora layer init by @SunMarc in https://github.com/huggingface/peft/pull/928
* Fixed LoRA conversion for kohya_ss by @kovalexal in https://github.com/huggingface/peft/pull/916
* [`CI`] Pin diffusers  by @younesbelkada in https://github.com/huggingface/peft/pull/936
* [`LoRA`] Add scale_layer / unscale_layer by @younesbelkada in https://github.com/huggingface/peft/pull/935
* TST: Add GH action to run unit tests with torch.compile by @BenjaminBossan in https://github.com/huggingface/peft/pull/943
* FIX: torch compile gh action installs pytest by @BenjaminBossan in https://github.com/huggingface/peft/pull/944
* Fix NotImplementedError for no bias. by @Datta0 in https://github.com/huggingface/peft/pull/946
* TST: Fix some tests that would fail with torch.compile by @BenjaminBossan in https://github.com/huggingface/peft/pull/949
* ENH Allow compile GH action to run on torch nightly by @BenjaminBossan in https://github.com/huggingface/peft/pull/952
* Install correct PyTorch nightly in GH action by @BenjaminBossan in https://github.com/huggingface/peft/pull/954
* support multiple ranks and alphas for LoRA by @pacman100 in https://github.com/huggingface/peft/pull/873
* feat: add type hints by @SauravMaheshkar in https://github.com/huggingface/peft/pull/858
* FIX: setting requires_grad on adapter layers by @BenjaminBossan in https://github.com/huggingface/peft/pull/905
* [`tests`] add transformers & diffusers integration tests by @younesbelkada in https://github.com/huggingface/peft/pull/962
* Fix integrations_tests.yml by @younesbelkada in https://github.com/huggingface/peft/pull/965
* Add 4-bit support to IA3 - Outperforms QLoRA in both speed and memory consumption by @His-Wardship in https://github.com/huggingface/peft/pull/864
* Update integrations_tests.yml by @younesbelkada in https://github.com/huggingface/peft/pull/966
* add the lora target modules for Mistral Models by @pacman100 in https://github.com/huggingface/peft/pull/974
* TST: Fix broken save_pretrained tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/969
* [tests] add multiple active adapters tests by @pacman100 in https://github.com/huggingface/peft/pull/961
* Fix missing tokenizer attribute in test by @BenjaminBossan in https://github.com/huggingface/peft/pull/977
* Add implementation of LyCORIS LoHa (FedPara-like adapter) for SD&SDXL models by @kovalexal in https://github.com/huggingface/peft/pull/956
* update BibTeX by @pacman100 in https://github.com/huggingface/peft/pull/989
* FIX: issues with (un)merging multiple LoRA and IA³ adapters by @BenjaminBossan in https://github.com/huggingface/peft/pull/976
* add lora target modules for stablelm models by @kbulutozler in https://github.com/huggingface/peft/pull/982
* Correct minor errors in example notebooks for causal language modelling by @SumanthRH in https://github.com/huggingface/peft/pull/926
* Fix typo in custom_models.mdx by @Pairshoe in https://github.com/huggingface/peft/pull/964
* Add base model metadata to model card by @BenjaminBossan in https://github.com/huggingface/peft/pull/975
* MNT Make .merged a property by @BenjaminBossan in https://github.com/huggingface/peft/pull/979
* Fix lora creation by @pacman100 in https://github.com/huggingface/peft/pull/993
* TST: Comment out flaky LoHA test by @BenjaminBossan in https://github.com/huggingface/peft/pull/1002
* ENH Support Conv2d layers for IA³ by @BenjaminBossan in https://github.com/huggingface/peft/pull/972
* Fix word_embeddings match for deepspeed wrapped model by @mayank31398 in https://github.com/huggingface/peft/pull/1000
* FEAT: Add `safe_merge` option in `merge` by @younesbelkada in https://github.com/huggingface/peft/pull/1001
* [`core` / `LoRA`] Add `safe_merge` to bnb layers by @younesbelkada in https://github.com/huggingface/peft/pull/1009
* ENH: Refactor LoRA bnb layers for faster initialization by @BenjaminBossan in https://github.com/huggingface/peft/pull/994
* FIX Don't assume model_config contains the key model_type by @BenjaminBossan in https://github.com/huggingface/peft/pull/1012
* FIX stale.py uses timezone-aware datetime by @BenjaminBossan in https://github.com/huggingface/peft/pull/1016
* FEAT: Add fp16 + cpu merge support by @younesbelkada in https://github.com/huggingface/peft/pull/1017
* fix lora scaling and unscaling by @pacman100 in https://github.com/huggingface/peft/pull/1027
* [`LoRA`] Revert original behavior for scale / unscale by @younesbelkada in https://github.com/huggingface/peft/pull/1029
* [`LoRA`] Raise error when adapter name not found in `set_scale` by @younesbelkada in https://github.com/huggingface/peft/pull/1034
* Fix target_modules type in config.from_pretrained by @BenjaminBossan in https://github.com/huggingface/peft/pull/1046
* docs(README): bit misspell current path link StackLLaMa by @guspan-tanadi in https://github.com/huggingface/peft/pull/1047
* Fixed wrong construction of LoHa weights, updated adapters conversion script by @kovalexal in https://github.com/huggingface/peft/pull/1021
* Fix P-tuning for sequence classification docs by @ehcalabres in https://github.com/huggingface/peft/pull/1049
* FIX: Setting active adapter correctly by @BenjaminBossan in https://github.com/huggingface/peft/pull/1051
* Fix Conv1D merge error for IA3 by @SumanthRH in https://github.com/huggingface/peft/pull/1014
* Add implementation of LyCORIS LoKr (KronA-like adapter) for SD&SDXL models by @kovalexal in https://github.com/huggingface/peft/pull/978
* [`core`] Fix `use_reentrant` issues by @younesbelkada in https://github.com/huggingface/peft/pull/1036
* [`tests`] Update Dockerfile to use cuda 12.2 by @younesbelkada in https://github.com/huggingface/peft/pull/1050
* Add testing for regex matching and other custom kwargs by @SumanthRH in https://github.com/huggingface/peft/pull/1031
* Fix Slack bot not displaying error messages by @younesbelkada in https://github.com/huggingface/peft/pull/1068
* Fix slow tests not running by @younesbelkada in https://github.com/huggingface/peft/pull/1071
* Release version 0.6.0 by @BenjaminBossan in https://github.com/huggingface/peft/pull/1072

## New Contributors
* @Psancs05 made their first contribution in https://github.com/huggingface/peft/pull/847
* @metaprotium made their first contribution in https://github.com/huggingface/peft/pull/844
* @jiqing-feng made their first contribution in https://github.com/huggingface/peft/pull/851
* @houx15 made their first contribution in https://github.com/huggingface/peft/pull/888
* @tmm1 made their first contribution in https://github.com/huggingface/peft/pull/874
* @raghavanone made their first contribution in https://github.com/huggingface/peft/pull/891
* @zspo made their first contribution in https://github.com/huggingface/peft/pull/898
* @rohithkrn made their first contribution in https://github.com/huggingface/peft/pull/892
* @Datta0 made their first contribution in https://github.com/huggingface/peft/pull/946
* @kbulutozler made their first contribution in https://github.com/huggingface/peft/pull/982
* @Pairshoe made their first contribution in https://github.com/huggingface/peft/pull/964
* @ehcalabres made their first contribution in https://github.com/huggingface/peft/pull/1049

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.5.0...v0.6.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.6.0)

---

## v0.5.0: GPTQ Quantization, Low-level API
**Published:** 2023-08-22

# GPTQ Integration 
Now, you can finetune GPTQ quantized models using PEFT. Here are some examples of how to use PEFT with a GPTQ model: [colab notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) and [finetuning](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) script.

* GPTQ Integration by @SunMarc in https://github.com/huggingface/peft/pull/771

# Low-level API
Enables users and developers to use PEFT as a utility library, at least for injectable adapters (LoRA, IA3, AdaLoRA). It exposes an API to modify the model in place to inject the new layers into the model.

* [`core`] PEFT refactor + introducing inject_adapter_in_model public method by @younesbelkada https://github.com/huggingface/peft/pull/749
* [`Low-level-API`] Add docs about LLAPI by @younesbelkada in https://github.com/huggingface/peft/pull/836

# Support for XPU and NPU devices

Leverage the support for more devices for loading and fine-tuning PEFT adapters.

* Support XPU adapter loading   by @abhilash1910 in https://github.com/huggingface/peft/pull/737
* Support Ascend NPU adapter loading by @statelesshz in https://github.com/huggingface/peft/pull/772

# Mix-and-match LoRAs

Stable support and new ways of merging multiple LoRAs. There are currently 3 ways of merging loras supported: `linear`, `svd` and `cat`.

* Added additional parameters to mixing multiple LoRAs through SVD, added ability to mix LoRAs through concatenation by @kovalexal in https://github.com/huggingface/peft/pull/817

## What's Changed
* Release version 0.5.0.dev0 by @pacman100 in https://github.com/huggingface/peft/pull/717
* Fix subfolder issue by @younesbelkada in https://github.com/huggingface/peft/pull/721
* Add falcon to officially supported LoRA & IA3 modules by @younesbelkada in https://github.com/huggingface/peft/pull/722
* revert change by @pacman100 in https://github.com/huggingface/peft/pull/731
* fix(pep561): include packaging type information by @aarnphm in https://github.com/huggingface/peft/pull/729
* [`Llama2`] Add disabling TP behavior by @younesbelkada in https://github.com/huggingface/peft/pull/728
* [`Patch`] patch trainable params for 4bit layers by @younesbelkada in https://github.com/huggingface/peft/pull/733
* FIX: Warning when initializing prompt encoder by @BenjaminBossan in https://github.com/huggingface/peft/pull/716
* ENH: Warn when disabling adapters and bias != 'none' by @BenjaminBossan in https://github.com/huggingface/peft/pull/741
* FIX: Disabling adapter works with modules_to_save by @BenjaminBossan in https://github.com/huggingface/peft/pull/736
* Updated Example in Class:LoraModel by @TianyiPeng in https://github.com/huggingface/peft/pull/672
* [`AdaLora`] Fix adalora inference issue by @younesbelkada in https://github.com/huggingface/peft/pull/745
* Add btlm to officially supported LoRA by @Trapper4888 in https://github.com/huggingface/peft/pull/751
* [`ModulesToSave`] add correct hook management for modules to save by @younesbelkada in https://github.com/huggingface/peft/pull/755
* Example notebooks for LoRA with custom models by @BenjaminBossan in https://github.com/huggingface/peft/pull/724
* Add tests for AdaLoRA, fix a few bugs by @BenjaminBossan in https://github.com/huggingface/peft/pull/734
* Add progressbar unload/merge by @BramVanroy in https://github.com/huggingface/peft/pull/753
* Support XPU adapter loading   by @abhilash1910 in https://github.com/huggingface/peft/pull/737
* Support Ascend NPU adapter loading by @statelesshz in https://github.com/huggingface/peft/pull/772
* Allow passing inputs_embeds instead of input_ids by @BenjaminBossan in https://github.com/huggingface/peft/pull/757
* [`core`] PEFT refactor + introducing `inject_adapter_in_model` public method by @younesbelkada in https://github.com/huggingface/peft/pull/749
* Add adapter error handling by @BenjaminBossan in https://github.com/huggingface/peft/pull/800
* add lora default target module for codegen by @sywangyi in https://github.com/huggingface/peft/pull/787
* DOC: Update docstring of PeftModel.from_pretrained by @BenjaminBossan in https://github.com/huggingface/peft/pull/799
* fix crash when using torch.nn.DataParallel for LORA inference by @sywangyi in https://github.com/huggingface/peft/pull/805
* Peft model signature by @kiansierra in https://github.com/huggingface/peft/pull/784
* GPTQ Integration by @SunMarc in https://github.com/huggingface/peft/pull/771
* Only fail  quantized Lora unload when actually merging by @BlackHC in https://github.com/huggingface/peft/pull/822
* Added additional parameters to mixing multiple LoRAs through SVD, added ability to mix LoRAs through concatenation by @kovalexal in https://github.com/huggingface/peft/pull/817
* TST: add test about loading custom models by @BenjaminBossan in https://github.com/huggingface/peft/pull/827
* Fix unbound error in ia3.py by @His-Wardship in https://github.com/huggingface/peft/pull/794
* [`Docker`] Fix gptq dockerfile by @younesbelkada in https://github.com/huggingface/peft/pull/835
* [`Tests`] Add 4bit slow training tests by @younesbelkada in https://github.com/huggingface/peft/pull/834
* [`Low-level-API`] Add docs about LLAPI by @younesbelkada in https://github.com/huggingface/peft/pull/836
* Type annotation fix by @vwxyzjn in https://github.com/huggingface/peft/pull/840

## New Contributors
* @TianyiPeng made their first contribution in https://github.com/huggingface/peft/pull/672
* @Trapper4888 made their first contribution in https://github.com/huggingface/peft/pull/751
* @abhilash1910 made their first contribution in https://github.com/huggingface/peft/pull/737
* @statelesshz made their first contribution in https://github.com/huggingface/peft/pull/772
* @kiansierra made their first contribution in https://github.com/huggingface/peft/pull/784
* @BlackHC made their first contribution in https://github.com/huggingface/peft/pull/822
* @His-Wardship made their first contribution in https://github.com/huggingface/peft/pull/794
* @vwxyzjn made their first contribution in https://github.com/huggingface/peft/pull/840

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.4.0...v0.5.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.5.0)

---

## v0.4.0: QLoRA, IA3 PEFT method, support for QA and Feature Extraction tasks, AutoPeftModelForxxx for simplified UX , LoRA for custom models with new added utils
**Published:** 2023-07-18

# QLoRA Support:
QLoRA uses 4-bit quantization to compress a pretrained language model. The LM parameters are then frozen and a relatively small number of trainable parameters are added to the model in the form of Low-Rank Adapters. During finetuning, QLoRA backpropagates gradients through the frozen 4-bit quantized pretrained language model into the Low-Rank Adapters. The LoRA layers are the only parameters being updated during training. For more details read the blog [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA
](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

* 4-bit QLoRA via bitsandbytes (4-bit base model + LoRA) by @TimDettmers in https://github.com/huggingface/peft/pull/476
* [`core`] Protect 4bit import by @younesbelkada in https://github.com/huggingface/peft/pull/480
* [`core`] Raise warning on using `prepare_model_for_int8_training` by @younesbelkada in https://github.com/huggingface/peft/pull/483

# New PEFT methods: IA3 from T-Few paper
To make fine-tuning more efficient, IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) rescales inner activations with learned vectors. These learned vectors are injected into the attention and feedforward modules in a typical transformer-based architecture. These learned vectors are the only trainable parameters during fine-tuning, and thus the original weights remain frozen. Dealing with learned vectors (as opposed to learned low-rank updates to a weight matrix like LoRA) keeps the number of trainable parameters much smaller. For more details, read the paper [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)

* Add functionality to support IA3 by @SumanthRH in https://github.com/huggingface/peft/pull/578

# Support for new tasks: QA and Feature Extraction
Addition of `PeftModelForQuestionAnswering` and `PeftModelForFeatureExtraction` classes to support QA and Feature Extraction tasks, respectively. This enables exciting new use-cases with PEFT, e.g., [LoRA for semantic similarity tasks](https://huggingface.co/docs/peft/task_guides/semantic-similarity-lora).

* feat: Add PeftModelForQuestionAnswering by @sjrl in https://github.com/huggingface/peft/pull/473
* add support for Feature Extraction using PEFT by @pacman100 in https://github.com/huggingface/peft/pull/647

# AutoPeftModelForxxx for better and Simplified UX
Introduces a new paradigm, AutoPeftModelForxxx intended for users that want to rapidly load and run peft models.

```
from peft import AutoPeftModelForCausalLM

peft_model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
```

* Introducing `AutoPeftModelForxxx` by @younesbelkada in https://github.com/huggingface/peft/pull/694


# LoRA for custom models
Not a transformer model, no problem, we have got you covered. PEFT now enables the usage of LoRA with custom models.

* FEAT: Make LoRA work with custom models by @BenjaminBossan in https://github.com/huggingface/peft/pull/676

# New LoRA utilities
Improvements to `add_weighted_adapter` method to support SVD for combining multiple LoRAs when creating new LoRA. 
New utils such as `unload` and `delete_adapter` providing users much better control about how they deal with the adapters.

* [Core] Enhancements and refactoring of LoRA method by @pacman100 in https://github.com/huggingface/peft/pull/695

# PEFT and Stable Diffusion
PEFT is very extensible and easy to use for performing DreamBooth of Stable Diffusion. Community has added conversion scripts to be able to use PEFT models with Civitai/webui format and vice-versa.

* LoRA for Conv2d layer, script to convert kohya_ss LoRA to PEFT by @kovalexal in https://github.com/huggingface/peft/pull/461
* Added Civitai LoRAs conversion to PEFT, PEFT LoRAs conversion to webui by @kovalexal in https://github.com/huggingface/peft/pull/596
* [Bugfix] Fixed LoRA conv2d merge by @kovalexal in https://github.com/huggingface/peft/pull/637
* Fixed LoraConfig alpha modification on add_weighted_adapter by @kovalexal in https://github.com/huggingface/peft/pull/654

## What's Changed
* Release: v0.4.0.dev0 by @pacman100 in https://github.com/huggingface/peft/pull/391
* do not use self.device. In FSDP cpu offload mode. self.device is "CPU… by @sywangyi in https://github.com/huggingface/peft/pull/352
* add accelerate example for DDP and FSDP in sequence classification fo… by @sywangyi in https://github.com/huggingface/peft/pull/358
* [`CI`] Fix CI - pin urlib by @younesbelkada in https://github.com/huggingface/peft/pull/402
* [docs] Fix index by @stevhliu in https://github.com/huggingface/peft/pull/397
* Fix documentation links on index page by @mikeorzel in https://github.com/huggingface/peft/pull/406
* Zero 3 init ReadME update by @dumpmemory in https://github.com/huggingface/peft/pull/399
* [`Tests`] Add soundfile to docker images by @younesbelkada in https://github.com/huggingface/peft/pull/401
* 4-bit QLoRA via bitsandbytes (4-bit base model + LoRA) by @TimDettmers in https://github.com/huggingface/peft/pull/476
* [`core`] Protect 4bit import by @younesbelkada in https://github.com/huggingface/peft/pull/480
* [`core`] Raise warning on using `prepare_model_for_int8_training` by @younesbelkada in https://github.com/huggingface/peft/pull/483
* Remove merge_weights by @Atry in https://github.com/huggingface/peft/pull/392
* [`core`] Add gradient checkpointing check by @younesbelkada in https://github.com/huggingface/peft/pull/404
* [docs] Fix LoRA image classification docs by @stevhliu in https://github.com/huggingface/peft/pull/524
* [docs] Prettify index by @stevhliu in https://github.com/huggingface/peft/pull/478
* change comment in tuners.lora, lora_alpha float to int by @codingchild2424 in https://github.com/huggingface/peft/pull/448
* [`LoRA`] Allow applying LoRA at different stages by @younesbelkada in https://github.com/huggingface/peft/pull/429
* Enable PeftConfig & PeftModel to load from revision by @lewtun in https://github.com/huggingface/peft/pull/433
* [`Llama-Adapter`] fix half precision inference + add tests by @younesbelkada in https://github.com/huggingface/peft/pull/456
* fix merge_and_unload when LoRA targets embedding layer by @0x000011b in https://github.com/huggingface/peft/pull/438
* return load_result when load_adapter by @dkqkxx in https://github.com/huggingface/peft/pull/481
* Fixed problem with duplicate same code. by @hotchpotch in https://github.com/huggingface/peft/pull/517
* Add starcoder model to target modules dict by @mrm8488 in https://github.com/huggingface/peft/pull/528
* Fix a minor typo where a non-default token_dim would crash prompt tuning by @thomas-schillaci in https://github.com/huggingface/peft/pull/459
* Remove device_map when training 4,8-bit model. by @SunMarc in https://github.com/huggingface/peft/pull/534
* add library name to model card by @pacman100 in https://github.com/huggingface/peft/pull/549
* Add thousands separator in print_trainable_parameters by @BramVanroy in https://github.com/huggingface/peft/pull/443
* [doc build] Use secrets by @mishig25 in https://github.com/huggingface/peft/pull/556
* improve readability of LoRA code by @martin-liu in https://github.com/huggingface/peft/pull/409
* [`core`] Add safetensors integration by @younesbelkada in https://github.com/huggingface/peft/pull/553
* [`core`] Fix config kwargs by @younesbelkada in https://github.com/huggingface/peft/pull/561
* Fix typo and url to `openai/whisper-large-v2` by @alvarobartt in https://github.com/huggingface/peft/pull/563
* feat: add type hint to `get_peft_model` by @samsja in https://github.com/huggingface/peft/pull/566
* Add issues template by @younesbelkada in https://github.com/huggingface/peft/pull/562
* [BugFix] Set alpha and dropout defaults in LoraConfig by @apbard in https://github.com/huggingface/peft/pull/390
* enable lora for mpt by @sywangyi in https://github.com/huggingface/peft/pull/576
* Fix minor typo in bug-report.yml by @younesbelkada in https://github.com/huggingface/peft/pull/582
* [`core`] Correctly passing the kwargs all over the place by @younesbelkada in https://github.com/huggingface/peft/pull/575
* Fix adalora device mismatch issue by @younesbelkada in https://github.com/huggingface/peft/pull/583
* LoRA for Conv2d layer, script to convert kohya_ss LoRA to PEFT by @kovalexal in https://github.com/huggingface/peft/pull/461
* Fix typo at peft_model.py by @Beomi in https://github.com/huggingface/peft/pull/588
* [`test`] Adds more CI tests by @younesbelkada in https://github.com/huggingface/peft/pull/586
* when from_pretrained is called in finetune case of lora with flag "… by @sywangyi in https://github.com/huggingface/peft/pull/591
* feat: Add PeftModelForQuestionAnswering by @sjrl in https://github.com/huggingface/peft/pull/473
* Improve the README when using PEFT by @pacman100 in https://github.com/huggingface/peft/pull/594
* [`tests`] Fix dockerfile by @younesbelkada in https://github.com/huggingface/peft/pull/608
* Fix final failing slow tests by @younesbelkada in https://github.com/huggingface/peft/pull/609
* [`core`] Add `adapter_name` in `get_peft_model` by @younesbelkada in https://github.com/huggingface/peft/pull/610
* [`core`] Stronger import of bnb by @younesbelkada in https://github.com/huggingface/peft/pull/605
* Added Civitai LoRAs conversion to PEFT, PEFT LoRAs conversion to webui by @kovalexal in https://github.com/huggingface/peft/pull/596
* update whisper test by @pacman100 in https://github.com/huggingface/peft/pull/617
* Update README.md, citation by @pminervini in https://github.com/huggingface/peft/pull/616
* Update train_dreambooth.py by @nafiturgut in https://github.com/huggingface/peft/pull/624
* [`Adalora`] Add adalora 4bit by @younesbelkada in https://github.com/huggingface/peft/pull/598
* [`AdaptionPrompt`] Add 8bit + 4bit support for adaption prompt by @younesbelkada in https://github.com/huggingface/peft/pull/604
* Add seq2seq prompt tuning support by @thomas-schillaci in https://github.com/huggingface/peft/pull/519
* [Bugfix] Fixed LoRA conv2d merge by @kovalexal in https://github.com/huggingface/peft/pull/637
* [Bugfix] Inserted adapter_name to get_peft_model_state_dict function by @nafiturgut in https://github.com/huggingface/peft/pull/626
* fix Prefix-tuning error in clm Float16 evaluation by @sywangyi in https://github.com/huggingface/peft/pull/520
* fix ptun and prompt tuning generation issue by @sywangyi in https://github.com/huggingface/peft/pull/543
* feat(model): Allow from_pretrained to accept PeftConfig class by @aarnphm in https://github.com/huggingface/peft/pull/612
* Fix `PeftModel.disable_adapter` by @ain-soph in https://github.com/huggingface/peft/pull/644
* bitsandbytes version check by @glerzing in https://github.com/huggingface/peft/pull/646
* DOC: Remove loralib requirements from examples, a few small fixes by @BenjaminBossan in https://github.com/huggingface/peft/pull/640
* style: tentatively add hints for some public function by @aarnphm in https://github.com/huggingface/peft/pull/614
* Add pytest-cov for reporting test coverage by @BenjaminBossan in https://github.com/huggingface/peft/pull/641
* Require Python version >= 3.8 by @BenjaminBossan in https://github.com/huggingface/peft/pull/649
* Fixed LoraConfig alpha modification on add_weighted_adapter by @kovalexal in https://github.com/huggingface/peft/pull/654
* [docs] API example by @stevhliu in https://github.com/huggingface/peft/pull/650
* FIX: bug resulting in config copies not working by @BenjaminBossan in https://github.com/huggingface/peft/pull/653
* Update clm-prompt-tuning.mdx by @richard087 in https://github.com/huggingface/peft/pull/652
* Adding support for RoBERTa layers_transform in COMMON_LAYERS_PATTERN by @sunyuhan19981208 in https://github.com/huggingface/peft/pull/669
* TST: Remove skipping certain tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/668
* Added wandb support for lora train_dreambooth by @nafiturgut in https://github.com/huggingface/peft/pull/639
* FIX: Embedding LoRA weights are initialized randomly by @BenjaminBossan in https://github.com/huggingface/peft/pull/681
* Fix broken docker images by @younesbelkada in https://github.com/huggingface/peft/pull/684
* Add functionality to support IA3 by @SumanthRH in https://github.com/huggingface/peft/pull/578
* Fix base_model_torch_dtype when using model.half() after init by @rayrayraykk in https://github.com/huggingface/peft/pull/688
* Init IA³ weights randomly when so configured by @BenjaminBossan in https://github.com/huggingface/peft/pull/693
* add support for Feature Extraction using PEFT by @pacman100 in https://github.com/huggingface/peft/pull/647
* Fix a small bug in forward method of IA³ by @BenjaminBossan in https://github.com/huggingface/peft/pull/696
* Update Readme to include IA3 by @SumanthRH in https://github.com/huggingface/peft/pull/702
* Fix code typo in int8-asr.mdx by @zsamboki in https://github.com/huggingface/peft/pull/698
* chore(type): annotate that peft does contains type hints by @aarnphm in https://github.com/huggingface/peft/pull/678
* Introducing `AutoPeftModelForxxx` by @younesbelkada in https://github.com/huggingface/peft/pull/694
* [WIP] FIX for disabling adapter, adding tests by @BenjaminBossan in https://github.com/huggingface/peft/pull/683
* [Core] Enhancements and refactoring of LoRA method by @pacman100 in https://github.com/huggingface/peft/pull/695
* [`Feature`] Save only selected adapters for LoRA by @younesbelkada in https://github.com/huggingface/peft/pull/705
* [`Auto`] Support `AutoPeftModel` for custom HF models by @younesbelkada in https://github.com/huggingface/peft/pull/707
* FEAT: Make LoRA work with custom models by @BenjaminBossan in https://github.com/huggingface/peft/pull/676
* [`core`] Better hub kwargs management by @younesbelkada in https://github.com/huggingface/peft/pull/712
* FIX: Removes warnings about unknown pytest marker by @BenjaminBossan in https://github.com/huggingface/peft/pull/715

## New Contributors
* @sywangyi made their first contribution in https://github.com/huggingface/peft/pull/352
* @mikeorzel made their first contribution in https://github.com/huggingface/peft/pull/406
* @TimDettmers made their first contribution in https://github.com/huggingface/peft/pull/476
* @Atry made their first contribution in https://github.com/huggingface/peft/pull/392
* @codingchild2424 made their first contribution in https://github.com/huggingface/peft/pull/448
* @lewtun made their first contribution in https://github.com/huggingface/peft/pull/433
* @0x000011b made their first contribution in https://github.com/huggingface/peft/pull/438
* @dkqkxx made their first contribution in https://github.com/huggingface/peft/pull/481
* @hotchpotch made their first contribution in https://github.com/huggingface/peft/pull/517
* @thomas-schillaci made their first contribution in https://github.com/huggingface/peft/pull/459
* @SunMarc made their first contribution in https://github.com/huggingface/peft/pull/534
* @BramVanroy made their first contribution in https://github.com/huggingface/peft/pull/443
* @mishig25 made their first contribution in https://github.com/huggingface/peft/pull/556
* @martin-liu made their first contribution in https://github.com/huggingface/peft/pull/409
* @alvarobartt made their first contribution in https://github.com/huggingface/peft/pull/563
* @samsja made their first contribution in https://github.com/huggingface/peft/pull/566
* @apbard made their first contribution in https://github.com/huggingface/peft/pull/390
* @kovalexal made their first contribution in https://github.com/huggingface/peft/pull/461
* @Beomi made their first contribution in https://github.com/huggingface/peft/pull/588
* @sjrl made their first contribution in https://github.com/huggingface/peft/pull/473
* @pminervini made their first contribution in https://github.com/huggingface/peft/pull/616
* @nafiturgut made their first contribution in https://github.com/huggingface/peft/pull/624
* @aarnphm made their first contribution in https://github.com/huggingface/peft/pull/612
* @ain-soph made their first contribution in https://github.com/huggingface/peft/pull/644
* @glerzing made their first contribution in https://github.com/huggingface/peft/pull/646
* @BenjaminBossan made their first contribution in https://github.com/huggingface/peft/pull/640
* @richard087 made their first contribution in https://github.com/huggingface/peft/pull/652
* @sunyuhan19981208 made their first contribution in https://github.com/huggingface/peft/pull/669
* @SumanthRH made their first contribution in https://github.com/huggingface/peft/pull/578
* @rayrayraykk made their first contribution in https://github.com/huggingface/peft/pull/688
* @zsamboki made their first contribution in https://github.com/huggingface/peft/pull/698

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.3.0...v0.4.0

# Significant community contributions
The following contributors have made significant changes to the library over the last release:

@TimDettmers
* 4-bit QLoRA via bitsandbytes (4-bit base model + LoRA) by @TimDettmers in https://github.com/huggingface/peft/pull/476

@SumanthRH 
* Add functionality to support IA3 by @SumanthRH in https://github.com/huggingface/peft/pull/578

@kovalexal 
* LoRA for Conv2d layer, script to convert kohya_ss LoRA to PEFT by @kovalexal in https://github.com/huggingface/peft/pull/461
* Added Civitai LoRAs conversion to PEFT, PEFT LoRAs conversion to webui by @kovalexal in https://github.com/huggingface/peft/pull/596
* [Bugfix] Fixed LoRA conv2d merge by @kovalexal in https://github.com/huggingface/peft/pull/637
* Fixed LoraConfig alpha modification on add_weighted_adapter by @kovalexal in https://github.com/huggingface/peft/pull/654

@sywangyi 
* do not use self.device. In FSDP cpu offload mode. self.device is "CPU… by @sywangyi in https://github.com/huggingface/peft/pull/352
* add accelerate example for DDP and FSDP in sequence classification fo… by @sywangyi in https://github.com/huggingface/peft/pull/358
* enable lora for mpt by @sywangyi in https://github.com/huggingface/peft/pull/576
* fix Prefix-tuning error in clm Float16 evaluation by @sywangyi in https://github.com/huggingface/peft/pull/520
* fix ptun and prompt tuning generation issue by @sywangyi in https://github.com/huggingface/peft/pull/543
* when from_pretrained is called in finetune case of lora with flag "… by @sywangyi in https://github.com/huggingface/peft/pull/591

@aarnphm
* feat(model): Allow from_pretrained to accept PeftConfig class by @aarnphm in https://github.com/huggingface/peft/pull/612
* style: tentatively add hints for some public function by @aarnphm in https://github.com/huggingface/peft/pull/614
* chore(type): annotate that peft does contains type hints by @aarnphm in https://github.com/huggingface/peft/pull/678

@martin-liu
* improve readability of LoRA code by @martin-liu in https://github.com/huggingface/peft/pull/409

@thomas-schillaci
* Add seq2seq prompt tuning support by @thomas-schillaci in https://github.com/huggingface/peft/pull/519

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.4.0)

---

## v0.3.0: Docs, Testing Suite, Multi Adapter Support, New methods and examples
**Published:** 2023-05-03

## Brand new Docs
With task guides, conceptual guides, integration guides, and code references all available at your fingertips, 🤗 PEFT's docs (found at https://huggingface.co/docs/peft) provide an insightful and easy-to-follow resource for anyone looking to how to use 🤗 PEFT. Whether you're a seasoned pro or just starting out, PEFT's documentation will help you to get the most out of it.

* [WIP-docs] Accelerate scripts by @stevhliu in https://github.com/huggingface/peft/pull/355
* [docs] Quicktour update by @stevhliu in https://github.com/huggingface/peft/pull/346
* [docs] Conceptual overview of prompting methods by @stevhliu in https://github.com/huggingface/peft/pull/339
* [docs] LoRA for token classification by @stevhliu in https://github.com/huggingface/peft/pull/302
* [docs] int8 training by @stevhliu in https://github.com/huggingface/peft/pull/332
* [docs] P-tuning for sequence classification by @stevhliu in https://github.com/huggingface/peft/pull/281
* [docs] Prompt tuning for CLM by @stevhliu in https://github.com/huggingface/peft/pull/264
* [docs] Prefix tuning for Seq2Seq by @stevhliu in https://github.com/huggingface/peft/pull/272
* [docs] Add API references by @stevhliu in https://github.com/huggingface/peft/pull/241
* [docs] Build notebooks from Markdown by @stevhliu in https://github.com/huggingface/peft/pull/240
* [docs] Supported models tables by @MKhalusova in https://github.com/huggingface/peft/pull/364
* [docs] Task guide with Dreambooth LoRA example by @MKhalusova in https://github.com/huggingface/peft/pull/330
* [docs] LoRA conceptual guide by @MKhalusova in https://github.com/huggingface/peft/pull/331
* [docs] Task guide for semantic segmentation with LoRA by @MKhalusova in https://github.com/huggingface/peft/pull/307
* Move image classification example to the docs by @MKhalusova in https://github.com/huggingface/peft/pull/239

## Comprehensive Testing Suite
Comprised of both unit and integration tests, it rigorously tests core features, examples, and various models on different setups, including single and multiple GPUs. This commitment to testing helps ensure that PEFT maintains the highest levels of correctness, usability, and performance, while continuously improving in all areas.

* [`CI`] Add ci tests by @younesbelkada in https://github.com/huggingface/peft/pull/203
* Fix CI tests by @younesbelkada in https://github.com/huggingface/peft/pull/210
* [`CI`] Add more ci tests by @younesbelkada in https://github.com/huggingface/peft/pull/223
* [`tests`] Adds more tests + fix failing tests by @younesbelkada in https://github.com/huggingface/peft/pull/238
* [`tests`] Adds GPU tests by @younesbelkada in https://github.com/huggingface/peft/pull/256
* [`tests`] add slow tests to GH workflow by @younesbelkada in https://github.com/huggingface/peft/pull/304
* [`core`] Better log messages by @younesbelkada in https://github.com/huggingface/peft/pull/366

## Multi Adapter Support
PEFT just got even more versatile with its new Multi Adapter Support! Now you can train and infer with multiple adapters, or even combine multiple LoRA adapters in a weighted combination. This is especially handy for RLHF training, where you can save memory by using a single base model with multiple adapters for actor, critic, reward, and reference. And the icing on the cake? Check out the LoRA Dreambooth inference example notebook to see this feature in action.

* Multi Adapter support by @pacman100 in https://github.com/huggingface/peft/pull/263

##  New PEFT methods: AdaLoRA and Adaption Prompt
PEFT just got even better, thanks to the contributions of the community! The AdaLoRA method is one of the exciting new additions. It takes the highly regarded LoRA method and improves it by allocating trainable parameters across the model to maximize performance within a given parameter budget. Another standout is the Adaption Prompt method, which enhances the already popular Prefix Tuning by introducing zero init attention.

* The Implementation of AdaLoRA (ICLR 2023) by @QingruZhang in https://github.com/huggingface/peft/pull/233
* Implement adaption prompt from Llama-Adapter paper by @yeoedward in https://github.com/huggingface/peft/pull/268

## New LoRA utilities
Good news for LoRA users! PEFT now allows you to merge LoRA parameters into the base model's parameters, giving you the freedom to remove the PEFT wrapper and apply downstream optimizations related to inference and deployment. Plus, you can use all the features that are compatible with the base model without any issues.

* [`utils`] add merge_lora utility function by @younesbelkada in https://github.com/huggingface/peft/pull/227
* Add nn.Embedding Support to Lora by @Splo2t in https://github.com/huggingface/peft/pull/337

## What's Changed
* release v0.3.0.dev0 by @pacman100 in https://github.com/huggingface/peft/pull/166
* fixing merged_linear lora issues by @pacman100 in https://github.com/huggingface/peft/pull/172
* Replace base_model's function temporarily by @PanQiWei in https://github.com/huggingface/peft/pull/170
* Support for LLaMA models by @zphang in https://github.com/huggingface/peft/pull/160
* [`core`] Fix peft multi-gpu issue by @younesbelkada in https://github.com/huggingface/peft/pull/145
* Update README.md by @dumpmemory in https://github.com/huggingface/peft/pull/167
* ChatGLM support by @mymusise in https://github.com/huggingface/peft/pull/180
* [`CI`] Add ci tests by @younesbelkada in https://github.com/huggingface/peft/pull/203
* Fix CI tests by @younesbelkada in https://github.com/huggingface/peft/pull/210
* Update train_dreambooth.py by @haofanwang in https://github.com/huggingface/peft/pull/204
* Fix failing test on `main` by @younesbelkada in https://github.com/huggingface/peft/pull/224
* Causal LM generation fix for prefix tuning: GPT2 model by @vineetm in https://github.com/huggingface/peft/pull/222
* [`CI`] Add more ci tests by @younesbelkada in https://github.com/huggingface/peft/pull/223
* Show CONFIG_NAME instead of "config.json" by @aitor-gamarra in https://github.com/huggingface/peft/pull/231
* add docs by @pacman100 in https://github.com/huggingface/peft/pull/214
* [`utils`] add merge_lora utility function by @younesbelkada in https://github.com/huggingface/peft/pull/227
* Have fix typo in README by @guspan-tanadi in https://github.com/huggingface/peft/pull/243
* Move image classification example to the docs by @MKhalusova in https://github.com/huggingface/peft/pull/239
* [docs] Add API references by @stevhliu in https://github.com/huggingface/peft/pull/241
* [docs] Build notebooks from Markdown by @stevhliu in https://github.com/huggingface/peft/pull/240
* [`core`] Fix offload issue by @younesbelkada in https://github.com/huggingface/peft/pull/248
* [`Automation`] Add stale bot by @younesbelkada in https://github.com/huggingface/peft/pull/247
* [resources] replace pdf links with abs links by @stas00 in https://github.com/huggingface/peft/pull/255
* [`Automation`] Update stale.py by @younesbelkada in https://github.com/huggingface/peft/pull/254
* docs: have fix bit typo README by @guspan-tanadi in https://github.com/huggingface/peft/pull/252
* Update other.py by @tpoisonooo in https://github.com/huggingface/peft/pull/250
* Fixing a bug where a wrong parameter name is used for the offload_folder by @toncho11 in https://github.com/huggingface/peft/pull/257
* [`tests`] Adds more tests + fix failing tests by @younesbelkada in https://github.com/huggingface/peft/pull/238
* The Implementation of AdaLoRA (ICLR 2023) by @QingruZhang in https://github.com/huggingface/peft/pull/233
* Add BLIP2 Example by @younesbelkada in https://github.com/huggingface/peft/pull/260
* Multi Adapter support by @pacman100 in https://github.com/huggingface/peft/pull/263
* Fix typo in examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py by @rmill040 in https://github.com/huggingface/peft/pull/277
* [`tests`] Adds GPU tests by @younesbelkada in https://github.com/huggingface/peft/pull/256
* Fix half precision forward by @younesbelkada in https://github.com/huggingface/peft/pull/261
* fix trainable params setting by @pacman100 in https://github.com/huggingface/peft/pull/283
* [docs] Prefix tuning for Seq2Seq by @stevhliu in https://github.com/huggingface/peft/pull/272
* Fix lora_dropout operator type when dropout=0 by @bigeagle in https://github.com/huggingface/peft/pull/288
* [`test`] Add Dockerfile  by @younesbelkada in https://github.com/huggingface/peft/pull/278
* fix and update examples and readme by @pacman100 in https://github.com/huggingface/peft/pull/295
* [docs] Prompt tuning for CLM by @stevhliu in https://github.com/huggingface/peft/pull/264
* Change gather for gather_for_metrics in eval. by @JulesGM in https://github.com/huggingface/peft/pull/296
* Fix: unexpected keyword argument 'has_fp16_weights' by @cyberfox in https://github.com/huggingface/peft/pull/299
* [`tests`] add CI training tests by @younesbelkada in https://github.com/huggingface/peft/pull/311
* [docs] Task guide for semantic segmentation with LoRA by @MKhalusova in https://github.com/huggingface/peft/pull/307
* [docs] P-tuning for sequence classification by @stevhliu in https://github.com/huggingface/peft/pull/281
* Fix `merge_and_unload` when having additional trainable modules by @pacman100 in https://github.com/huggingface/peft/pull/322
* feat(ci): add `pip` caching to CI by @SauravMaheshkar in https://github.com/huggingface/peft/pull/314
* Fix eval for causal language modeling example by @BabyChouSr in https://github.com/huggingface/peft/pull/327
* [docs] LoRA for token classification by @stevhliu in https://github.com/huggingface/peft/pull/302
* [docs] int8 training by @stevhliu in https://github.com/huggingface/peft/pull/332
* fix lora modules_to_save issue by @pacman100 in https://github.com/huggingface/peft/pull/343
* [docs] Task guide with Dreambooth LoRA example by @MKhalusova in https://github.com/huggingface/peft/pull/330
* [docs] LoRA conceptual guide by @MKhalusova in https://github.com/huggingface/peft/pull/331
* [docs] Conceptual overview of prompting methods by @stevhliu in https://github.com/huggingface/peft/pull/339
* Implement adaption prompt from Llama-Adapter paper by @yeoedward in https://github.com/huggingface/peft/pull/268
* [`tests`] add slow tests to GH workflow by @younesbelkada in https://github.com/huggingface/peft/pull/304
* [`core`] Better log messages by @younesbelkada in https://github.com/huggingface/peft/pull/366
* Use `try` and `finally` in `disable_adapter()` to catch exceptions by @mukobi in https://github.com/huggingface/peft/pull/368
* [docs] Supported models tables by @MKhalusova in https://github.com/huggingface/peft/pull/364
* [WIP-docs] Accelerate scripts by @stevhliu in https://github.com/huggingface/peft/pull/355
* [docs] Quicktour update by @stevhliu in https://github.com/huggingface/peft/pull/346
* [`CI`] Fix nightly CI issues by @younesbelkada in https://github.com/huggingface/peft/pull/375
* Fix a link to the example script by @nzw0301 in https://github.com/huggingface/peft/pull/383
* Add nn.Embedding Support to Lora by @Splo2t in https://github.com/huggingface/peft/pull/337
* Fix missing arg for transpose in AdaLora by @yasyf in https://github.com/huggingface/peft/pull/347
* fix INT8 prepare function by @pacman100 in https://github.com/huggingface/peft/pull/389

## New Contributors
* @PanQiWei made their first contribution in https://github.com/huggingface/peft/pull/170
* @mymusise made their first contribution in https://github.com/huggingface/peft/pull/180
* @haofanwang made their first contribution in https://github.com/huggingface/peft/pull/204
* @vineetm made their first contribution in https://github.com/huggingface/peft/pull/222
* @aitor-gamarra made their first contribution in https://github.com/huggingface/peft/pull/231
* @guspan-tanadi made their first contribution in https://github.com/huggingface/peft/pull/243
* @MKhalusova made their first contribution in https://github.com/huggingface/peft/pull/239
* @stevhliu made their first contribution in https://github.com/huggingface/peft/pull/241
* @stas00 made their first contribution in https://github.com/huggingface/peft/pull/255
* @tpoisonooo made their first contribution in https://github.com/huggingface/peft/pull/250
* @toncho11 made their first contribution in https://github.com/huggingface/peft/pull/257
* @QingruZhang made their first contribution in https://github.com/huggingface/peft/pull/233
* @rmill040 made their first contribution in https://github.com/huggingface/peft/pull/277
* @bigeagle made their first contribution in https://github.com/huggingface/peft/pull/288
* @JulesGM made their first contribution in https://github.com/huggingface/peft/pull/296
* @cyberfox made their first contribution in https://github.com/huggingface/peft/pull/299
* @BabyChouSr made their first contribution in https://github.com/huggingface/peft/pull/327
* @yeoedward made their first contribution in https://github.com/huggingface/peft/pull/268
* @mukobi made their first contribution in https://github.com/huggingface/peft/pull/368
* @nzw0301 made their first contribution in https://github.com/huggingface/peft/pull/383
* @Splo2t made their first contribution in https://github.com/huggingface/peft/pull/337
* @yasyf made their first contribution in https://github.com/huggingface/peft/pull/347

## Significant community contributions
The following contributors have made significant changes to the library over the last release:

@QingruZhang
* The Implementation of AdaLoRA (ICLR 2023) in https://github.com/huggingface/peft/pull/233

@yeoedward
* Implement adaption prompt from Llama-Adapter paper in https://github.com/huggingface/peft/pull/268

@Splo2t
* Add nn.Embedding Support to Lora in https://github.com/huggingface/peft/pull/337

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.3.0)

---

## v0.2.0: v0.2.0
**Published:** 2023-03-10

## Whisper large tuning using PEFT LoRA+INT-8 on T4 GPU in Colab notebooks
We tested PEFT on  [@OpenAI](https://twitter.com/OpenAI)'s Whisper Large model and got:
i) 5x larger batch sizes
ii) Less than 8GB GPU VRAM
iii) Best part? Almost no degredation to WER 🤯

Without PEFT:
- OOM on a T4 GPU ❌
- 6GB checkpoint ❌
- 13.64 WER ✅

With PEFT:
- Train on a T4 GPU ✅
- 60MB checkpoint ✅
- 14.01 WER ✅

* adding whisper large peft+int8 training example by @pacman100 in https://github.com/huggingface/peft/pull/95

## `prepare_for_int8_training` utility
This utility  enables preprocessing the base model to be ready for INT8 training. 

* [`core`]  add `prepare_model_for_training` by @younesbelkada in https://github.com/huggingface/peft/pull/85
* [`core`] Some changes with `prepare_model_for_training` & few fixes by @younesbelkada in https://github.com/huggingface/peft/pull/105

## `disable_adapter()` context manager
Enables to disable adapter layers to get the outputs from the frozen base models. 
An exciting application of this feature allows only a single model copy to be used for policy model and reference model generations in RLHF. 

* add disable adapter context manager by @pacman100 in https://github.com/huggingface/peft/pull/106

## What's Changed
* release v0.2.0.dev0 by @pacman100 in https://github.com/huggingface/peft/pull/69
* Update README.md by @sayakpaul in https://github.com/huggingface/peft/pull/72
* Fixed typo in Readme by @Muhtasham in https://github.com/huggingface/peft/pull/73
* Update README.md by @pacman100 in https://github.com/huggingface/peft/pull/77
* convert prompt tuning vocab to fp32 by @mayank31398 in https://github.com/huggingface/peft/pull/68
* [`core`]  add `prepare_model_for_training` by @younesbelkada in https://github.com/huggingface/peft/pull/85
* [`bnb`] add flan-t5 example by @younesbelkada in https://github.com/huggingface/peft/pull/86
* making `prepare_model_for_training` flexible by @pacman100 in https://github.com/huggingface/peft/pull/90
* adding whisper large peft+int8 training example by @pacman100 in https://github.com/huggingface/peft/pull/95
* making `bnb` optional by @pacman100 in https://github.com/huggingface/peft/pull/97
* add support for regex target modules in lora by @pacman100 in https://github.com/huggingface/peft/pull/104
* [`core`] Some changes with `prepare_model_for_training` & few fixes by @younesbelkada in https://github.com/huggingface/peft/pull/105
* Fix typo by @mrm8488 in https://github.com/huggingface/peft/pull/107
* add disable adapter context manager by @pacman100 in https://github.com/huggingface/peft/pull/106
* add `EleutherAI/gpt-neox-20b` to support matrix by @pacman100 in https://github.com/huggingface/peft/pull/109
* fix merging lora weights for inference by @pacman100 in https://github.com/huggingface/peft/pull/117
* [`core`] Fix autocast issue by @younesbelkada in https://github.com/huggingface/peft/pull/121
* fixes `prepare_for_int8_training` by @pacman100 in https://github.com/huggingface/peft/pull/127
* issue#126: torch.load device issue. by @gabinguo in https://github.com/huggingface/peft/pull/134
* fix: count params when zero init'd by @zanussbaum in https://github.com/huggingface/peft/pull/140
* chore: update `pyproject.toml` by @SauravMaheshkar in https://github.com/huggingface/peft/pull/125
* support option for encoder only prompts by @mayank31398 in https://github.com/huggingface/peft/pull/150
* minor fixes to the examples by @pacman100 in https://github.com/huggingface/peft/pull/149
* Add local saving for whisper largev2 example notebook by @alvanli in https://github.com/huggingface/peft/pull/163
* fix count by @dumpmemory in https://github.com/huggingface/peft/pull/162
* Add Prefix Tuning citation by @zphang in https://github.com/huggingface/peft/pull/159
* lora fixes and adding 8bitMegredLinear lora by @pacman100 in https://github.com/huggingface/peft/pull/157
* Update README.md by @pacman100 in https://github.com/huggingface/peft/pull/164
* minor changes by @pacman100 in https://github.com/huggingface/peft/pull/165

## New Contributors
* @Muhtasham made their first contribution in https://github.com/huggingface/peft/pull/73
* @mayank31398 made their first contribution in https://github.com/huggingface/peft/pull/68
* @mrm8488 made their first contribution in https://github.com/huggingface/peft/pull/107
* @gabinguo made their first contribution in https://github.com/huggingface/peft/pull/134
* @zanussbaum made their first contribution in https://github.com/huggingface/peft/pull/140
* @SauravMaheshkar made their first contribution in https://github.com/huggingface/peft/pull/125
* @alvanli made their first contribution in https://github.com/huggingface/peft/pull/163
* @dumpmemory made their first contribution in https://github.com/huggingface/peft/pull/162
* @zphang made their first contribution in https://github.com/huggingface/peft/pull/159

## Significant community contributions

The following contributors have made significant changes to the library over the last release:

* @mayank31398 
    * Prompt Tuning method enhancements and fixes (#68, #150)

**Full Changelog**: https://github.com/huggingface/peft/compare/v0.1.0...v0.2.0

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.2.0)

---

## v0.1.0: v0.1.0 Initial release
**Published:** 2023-02-10

Initial release of 🤗 PEFT. Checkout the main [README](https://github.com/huggingface/peft) to learn more about it!

[View on GitHub](https://github.com/huggingface/peft/releases/tag/v0.1.0)

---

