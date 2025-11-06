# Releases

Version history for this repository (67 releases).

## v0.25.0: v0.25.0
**Published:** 2025-11-06

## Features

* 💤 Switch to sleep level=2 and split wake-ups in GRPO and RLOO trainers by @xxrjun in https://github.com/huggingface/trl/pull/4296
* Added custom `prepare_model_for_kbit_training` to save VRAM by @sergiopaniego in https://github.com/huggingface/trl/pull/4335
* Add `add_generation_prompt` to processor_kwargs in GRPO and RLOO trainer by @qgallouedec in https://github.com/huggingface/trl/pull/4361
* Add support for Trackio completions logging in GRPOTrainer by @taha-yassine in https://github.com/huggingface/trl/pull/4359
* Support chat_template_kwargs by @pramodith in https://github.com/huggingface/trl/pull/4350
* GRPO: ScaleRL -> Support casting LM Head to FP32 by @pramodith in https://github.com/huggingface/trl/pull/4303
* Support casting to fp32 when word embeddings are tied to lm_head by @pramodith in https://github.com/huggingface/trl/pull/4446
* 💬 Add chat to vLLM client and server, update trainer calls by @qgallouedec in https://github.com/huggingface/trl/pull/4450

## Experimental

* 🚚 Move BCO to `trl.experimental` by @qgallouedec in https://github.com/huggingface/trl/pull/4312
* 👑 [experimental] GOLD Trainer by @kashif in https://github.com/huggingface/trl/pull/4349
* Add PAPOTrainer for preference-based optimization by @SolarWindRider in https://github.com/huggingface/trl/pull/4334
* [GFPO] fix the GFPO loss calculation error caused by unmodified old_per_token_logps by @Peter-Chou in https://github.com/huggingface/trl/pull/4454
* 🕹️ Add rollout function for OpenEnv integration by @lewtun in https://github.com/huggingface/trl/pull/4310

## Fixes

* [Activation-checkpointing] add tensor dedup and param offloading by @kashif in https://github.com/huggingface/trl/pull/4247
* Fix attn_implementation name in OnlineDPO for transformers v5 by @albertvillanova in https://github.com/huggingface/trl/pull/4322
* Hotfix: Fall back to config.text_config._name_or_path if missing config._name_or_path by @albertvillanova in https://github.com/huggingface/trl/pull/4324
* Fix GRPO and RLOO trainers for continuous batching by @albertvillanova in https://github.com/huggingface/trl/pull/4348
* Fix: `add_generation_prompt=True` for conversational only by @qgallouedec in https://github.com/huggingface/trl/pull/4362
* Remove ignored max_length parameter from PRMTrainer data collator by @albertvillanova in https://github.com/huggingface/trl/pull/4355
* Fix add_generation_prompt arg for paged transformers in GRPO and RLOO trainers by @albertvillanova in https://github.com/huggingface/trl/pull/4370
* Fix GKD Liger memory spike by @qgallouedec in https://github.com/huggingface/trl/pull/4140
* Fix GRPO with replay buffer by inserting images in the prompt by @albertvillanova in https://github.com/huggingface/trl/pull/4391
* fix: Remove chat template setting from non-SFT trainer scripts by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4437
* 🖼️ Fix reporting images with vLLM by @qgallouedec in https://github.com/huggingface/trl/pull/4476

## Documentation and Examples

* Added SFT LoRA notebook by @sergiopaniego in https://github.com/huggingface/trl/pull/4244
* Update notebooks README with latest additions by @sergiopaniego in https://github.com/huggingface/trl/pull/4316
* Add notebooks to Examples docs and restructure by @sergiopaniego in https://github.com/huggingface/trl/pull/4317
* Highlight OpenEnv in landing docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4327
* Update OpenEnv docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4328
* Add OpenEnv blog to landing by @sergiopaniego in https://github.com/huggingface/trl/pull/4333
* 🗞️ Update "What's New" by @qgallouedec in https://github.com/huggingface/trl/pull/4338
* Update Reducing Memory Consumption guide with more details by @sergiopaniego in https://github.com/huggingface/trl/pull/4332
* Fixed links inside Tips in docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4360
* 🔥 docs: Add RapidFire AI integration guide by @kamran-rapidfireAI in https://github.com/huggingface/trl/pull/4340
* Fix paper link for "Towards Efficient and Exact Optimization of Language Model Alignment" by @qgallouedec in https://github.com/huggingface/trl/pull/4409
* Migrate experimental trl feature docs  by @ethanknights in https://github.com/huggingface/trl/pull/4411
* Update SFT QLoRA notebook with **14B** model on free Colab by @sergiopaniego in https://github.com/huggingface/trl/pull/4336
* Create "Talks" subsection by @sergiopaniego in https://github.com/huggingface/trl/pull/4414
* Openenv wordle example by @burtenshaw in https://github.com/huggingface/trl/pull/4357
* docs: Remove outdated conversational dataset conversion guidance by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4422
* docs: List all trainers that support Liger Kernel by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4432
* Add On-Policy Distillation from thinking labs to paper index. by @pramodith in https://github.com/huggingface/trl/pull/4410
* Upload notebook with T4 selected by @sergiopaniego in https://github.com/huggingface/trl/pull/4449
* Removed outdated warning about batch contamination by @Harras3 in https://github.com/huggingface/trl/pull/4423
* Removed Sentiment Tuning Examples by @Harras3 in https://github.com/huggingface/trl/pull/4424
* docs: Remove outdated notebooks by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4435
* docs: Move Multi-Adapter RL section to PEFT integration by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4436
* Update `max_length` explanation for VLM in online trainers by @sergiopaniego in https://github.com/huggingface/trl/pull/4220
* Updated OpenEnv docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4418
* add llasa-tutorial by @Deep-unlearning in https://github.com/huggingface/trl/pull/4456

## Deprecations

* Replace deprecated AutoModelForVision2Seq with AutoModelForImageTextToText by @albertvillanova in https://github.com/huggingface/trl/pull/4353
* Replace deprecated list with tuple indexing in PPOTrainer by @albertvillanova in https://github.com/huggingface/trl/pull/4356
* Remove liger loss in favor of liger kernel by @sergiopaniego in https://github.com/huggingface/trl/pull/4364
* 🐍 Drop Python 3.9 by @qgallouedec in https://github.com/huggingface/trl/pull/4183

## What's Changed

* ⬆️ Bump dev version by @qgallouedec in https://github.com/huggingface/trl/pull/4293
* Update links to docs in README to latest packaged version by @sergiopaniego in https://github.com/huggingface/trl/pull/4084
* 🧺 [4/N] Refactor `_generate` in GRPO/RLOO: Move `forward_kwargs` outside generation method by @qgallouedec in https://github.com/huggingface/trl/pull/4154
* Fix missing CI slow tests: ImportError: vLLM is not installed by @albertvillanova in https://github.com/huggingface/trl/pull/4304
* Added SFT LoRA notebook by @sergiopaniego in https://github.com/huggingface/trl/pull/4244
* ⚰️ Remove deprecated by @qgallouedec in https://github.com/huggingface/trl/pull/4301
* Silence TRL experimental warnings in CI by @albertvillanova in https://github.com/huggingface/trl/pull/4307
* Filter expected setup_chat_format deprecation warning in CI by @albertvillanova in https://github.com/huggingface/trl/pull/4306
* [Activation-checkpointing] add tensor dedup and param offloading by @kashif in https://github.com/huggingface/trl/pull/4247
* Remove parameterized as test extra dependency by @albertvillanova in https://github.com/huggingface/trl/pull/4315
* Update notebooks README with latest additions by @sergiopaniego in https://github.com/huggingface/trl/pull/4316
* 🚚 Move BCO to `trl.experimental` by @qgallouedec in https://github.com/huggingface/trl/pull/4312
* 🧺 [5/N] Refactor `_generate` in GRPO/RLOO: Insert images in the prompt by @qgallouedec in https://github.com/huggingface/trl/pull/4155
* 💤 Switch to sleep level=2 and split wake-ups in GRPO and RLOO trainers by @xxrjun in https://github.com/huggingface/trl/pull/4296
* Replace unittest skipTest from transformers with pytest.skip by @albertvillanova in https://github.com/huggingface/trl/pull/4297
* Add notebooks to Examples docs and restructure by @sergiopaniego in https://github.com/huggingface/trl/pull/4317
* Fix attn_implementation name in OnlineDPO for transformers v5 by @albertvillanova in https://github.com/huggingface/trl/pull/4322
* 🕹️ Add rollout function for OpenEnv integration by @lewtun in https://github.com/huggingface/trl/pull/4310
* Highlight OpenEnv in landing docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4327
* Update OpenEnv docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4328
* Move BCO tests to tests/experimental by @albertvillanova in https://github.com/huggingface/trl/pull/4326
* Hotfix: Fall back to config.text_config._name_or_path if missing config._name_or_path by @albertvillanova in https://github.com/huggingface/trl/pull/4324
* Add OpenEnv blog to landing by @sergiopaniego in https://github.com/huggingface/trl/pull/4333
* 🗞️ Update "What's New" by @qgallouedec in https://github.com/huggingface/trl/pull/4338
* Update Reducing Memory Consumption guide with more details by @sergiopaniego in https://github.com/huggingface/trl/pull/4332
* Added custom `prepare_model_for_kbit_training` to save VRAM by @sergiopaniego in https://github.com/huggingface/trl/pull/4335
* [vllm] update comment about communication group host ip by @kashif in https://github.com/huggingface/trl/pull/4337
* Fix GRPO and RLOO trainers for continuous batching by @albertvillanova in https://github.com/huggingface/trl/pull/4348
* Fixed links inside Tips in docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4360
* Fix CI issue for vlm_gemma_3n model by @kaixuanliu in https://github.com/huggingface/trl/pull/4278
* Add `add_generation_prompt` to processor_kwargs in GRPO and RLOO trainer by @qgallouedec in https://github.com/huggingface/trl/pull/4361
* Fix: `add_generation_prompt=True` for conversational only by @qgallouedec in https://github.com/huggingface/trl/pull/4362
* Use explicit tiny-Qwen2_5_VL model_id parameter in CI tests by @albertvillanova in https://github.com/huggingface/trl/pull/4325
* Move tests of experimental GRPO with replay buffer to tests/experimental by @albertvillanova in https://github.com/huggingface/trl/pull/4329
* Implement CI test workflow for experimental module by @albertvillanova in https://github.com/huggingface/trl/pull/4330
* Replace deprecated AutoModelForVision2Seq with AutoModelForImageTextToText by @albertvillanova in https://github.com/huggingface/trl/pull/4353
* Move tests of BCO trainer args to tests/experimental by @albertvillanova in https://github.com/huggingface/trl/pull/4354
* Remove ignored max_length parameter from PRMTrainer data collator by @albertvillanova in https://github.com/huggingface/trl/pull/4355
* Replace deprecated list with tuple indexing in PPOTrainer by @albertvillanova in https://github.com/huggingface/trl/pull/4356
* Add support for Trackio completions logging in GRPOTrainer by @taha-yassine in https://github.com/huggingface/trl/pull/4359
* Fix add_generation_prompt arg for paged transformers in GRPO and RLOO trainers by @albertvillanova in https://github.com/huggingface/trl/pull/4370
* Align make test_experimental with make test by @albertvillanova in https://github.com/huggingface/trl/pull/4371
* 🔥 docs: Add RapidFire AI integration guide by @kamran-rapidfireAI in https://github.com/huggingface/trl/pull/4340
* 👑 [experimental] GOLD Trainer by @kashif in https://github.com/huggingface/trl/pull/4349
* Support chat_template_kwargs by @pramodith in https://github.com/huggingface/trl/pull/4350
* [GOLD] Set teacher tokenizer name if using ULD loss by @kashif in https://github.com/huggingface/trl/pull/4389
* Fix typo in GOLD docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4394
* Hotfix CI for Python 3.9 by setting test as xfail until transformers release by @albertvillanova in https://github.com/huggingface/trl/pull/4388
* [tests] Update rope_scaling configuration for tiny qwen-vl models by @kashif in https://github.com/huggingface/trl/pull/4405
* [GOLD] Update code example for GOLD Trainer by @cmpatino in https://github.com/huggingface/trl/pull/4406
* Hotfix CI with dev dependencies: xfail test_prepare_inputs_for_generation by @albertvillanova in https://github.com/huggingface/trl/pull/4372
* Fix paper link for "Towards Efficient and Exact Optimization of Language Model Alignment" by @qgallouedec in https://github.com/huggingface/trl/pull/4409
* Migrate experimental trl feature docs  by @ethanknights in https://github.com/huggingface/trl/pull/4411
* Update SFT QLoRA notebook with **14B** model on free Colab by @sergiopaniego in https://github.com/huggingface/trl/pull/4336
* Add PAPOTrainer for preference-based optimization by @SolarWindRider in https://github.com/huggingface/trl/pull/4334
* Fix GKD Liger memory spike by @qgallouedec in https://github.com/huggingface/trl/pull/4140
* Remove liger loss in favor of liger kernel by @sergiopaniego in https://github.com/huggingface/trl/pull/4364
* Add license to test file and disable docstyle in GOLD script by @qgallouedec in https://github.com/huggingface/trl/pull/4412
* Replace duplicate test with model_id parametrized test by @albertvillanova in https://github.com/huggingface/trl/pull/4415
* Fix raising of deprecation warning for liger_loss by @albertvillanova in https://github.com/huggingface/trl/pull/4417
* Consolidate slow tests into main test files by @ishitab02 in https://github.com/huggingface/trl/pull/4408
* Fix CI experimental tests TypeError for GRPOWithReplayBufferTrainer.update_with_replay_buffer by @albertvillanova in https://github.com/huggingface/trl/pull/4366
* Fix GRPO with replay buffer by inserting images in the prompt by @albertvillanova in https://github.com/huggingface/trl/pull/4391
* GRPO: ScaleRL -> Support casting LM Head to FP32 by @pramodith in https://github.com/huggingface/trl/pull/4303
* Create "Talks" subsection by @sergiopaniego in https://github.com/huggingface/trl/pull/4414
* Openenv wordle example by @burtenshaw in https://github.com/huggingface/trl/pull/4357
* docs: Remove outdated conversational dataset conversion guidance by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4422
* docs: List all trainers that support Liger Kernel by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4432
* fix: Remove chat template setting from non-SFT trainer scripts by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4437
* Add On-Policy Distillation from thinking labs to paper index. by @pramodith in https://github.com/huggingface/trl/pull/4410
* Upload notebook with T4 selected by @sergiopaniego in https://github.com/huggingface/trl/pull/4449
* Support casting to fp32 when word embeddings are tied to lm_head by @pramodith in https://github.com/huggingface/trl/pull/4446
* Update tokenizer apply_chat_template with return_dict=True default by @albertvillanova in https://github.com/huggingface/trl/pull/4448
* Removed outdated warning about batch contamination by @Harras3 in https://github.com/huggingface/trl/pull/4423
* 🐍 Drop Python 3.9 by @qgallouedec in https://github.com/huggingface/trl/pull/4183
* Removed Sentiment Tuning Examples by @Harras3 in https://github.com/huggingface/trl/pull/4424
* docs: Remove outdated notebooks by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4435
* docs: Move Multi-Adapter RL section to PEFT integration by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4436
* Moved masked_mean, masked_var and masked_whiten to ppo_trainer.py by @Harras3 in https://github.com/huggingface/trl/pull/4444
* Update `max_length` explanation for VLM in online trainers by @sergiopaniego in https://github.com/huggingface/trl/pull/4220
* [fix] wordle model_id updates by @burtenshaw in https://github.com/huggingface/trl/pull/4453
* Updated OpenEnv docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4418
* add llasa-tutorial by @Deep-unlearning in https://github.com/huggingface/trl/pull/4456
* 💬 Add chat to vLLM client and server, update trainer calls by @qgallouedec in https://github.com/huggingface/trl/pull/4450
* [GFPO] fix the GFPO loss calculation error caused by unmodified old_per_token_logps by @Peter-Chou in https://github.com/huggingface/trl/pull/4454
* 🖼️ Fix reporting images with vLLM by @qgallouedec in https://github.com/huggingface/trl/pull/4476
* Release: v0.25 by @qgallouedec in https://github.com/huggingface/trl/pull/4478

## New Contributors
* @xxrjun made their first contribution in https://github.com/huggingface/trl/pull/4296
* @taha-yassine made their first contribution in https://github.com/huggingface/trl/pull/4359
* @kamran-rapidfireAI made their first contribution in https://github.com/huggingface/trl/pull/4340
* @ethanknights made their first contribution in https://github.com/huggingface/trl/pull/4411
* @SolarWindRider made their first contribution in https://github.com/huggingface/trl/pull/4334
* @ishitab02 made their first contribution in https://github.com/huggingface/trl/pull/4408
* @Harras3 made their first contribution in https://github.com/huggingface/trl/pull/4423
* @Deep-unlearning made their first contribution in https://github.com/huggingface/trl/pull/4456

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.24.0...v0.25.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.25.0)

---

## v0.24.0: v0.24.0
**Published:** 2025-10-16

## Features

* Add accuracy reward by @pramodith in https://github.com/huggingface/trl/pull/4270
* Add support for `token_type_ids` in `DPOTrainer` by @aweers in https://github.com/huggingface/trl/pull/4285
* 💰 `RichProgressCallback` enhancement by @qgallouedec in https://github.com/huggingface/trl/pull/4245
* Include `chat_template_kwargs` in `apply_chat_template` by @cmpatino in https://github.com/huggingface/trl/pull/4233
* 🏷️ Account for `token_type_ids` in `DataCollatorForVisionLanguageModeling` by @qgallouedec in https://github.com/huggingface/trl/pull/4190
* 🎨 Support mixing image+text and text-only examples by @qgallouedec in https://github.com/huggingface/trl/pull/4203
* 🎁 `RewardTrainer` refactor by @qgallouedec in https://github.com/huggingface/trl/pull/4093
* 🎞️ Support sequence classification models in `clone_chat_template` by @qgallouedec in https://github.com/huggingface/trl/pull/4097
* ✨ Add logging for training completion and model saving in training scripts by @qgallouedec in https://github.com/huggingface/trl/pull/4048
* 🖨️ Print rich table for messages by @qgallouedec in https://github.com/huggingface/trl/pull/4160
* 😴 Add `vllm_enable_sleep_mode` to RLOO Trainer by @sergiopaniego in https://github.com/huggingface/trl/pull/4107
* 📽 Multi image support for GRPO/RLOO by @qgallouedec in https://github.com/huggingface/trl/pull/4113
* 👁️ Add VLM support to RLOO trainer by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4067
* ℹ️ Enable XPU for vLLM client by @jiqing-feng in https://github.com/huggingface/trl/pull/4031
* 🧶 feat: Add WeaveCallback for W&B Weave integration by @parambharat in https://github.com/huggingface/trl/pull/4089

## Fixes

* [Online-DPO] fix the completion_len == max_new_tokens crash by @kashif in https://github.com/huggingface/trl/pull/4193
* Fix entropy and accuracy calculation for prompt_tuning techniques. by @pramodith in https://github.com/huggingface/trl/pull/4196
* Fix prompt-completion labeling with add_generation_prompt and warning by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4201
* 🌡️ Have vLLM return processed (temperature scaled) log probs by @YonatanGideoni in https://github.com/huggingface/trl/pull/4163
* Fix handling of f_divergence_type in DPO by @albertvillanova in https://github.com/huggingface/trl/pull/4171
* ⚡ Fix Flash Attention x Padding-Free loss by @qgallouedec in https://github.com/huggingface/trl/pull/4170
* Pass required token_type_ids by @albertvillanova in https://github.com/huggingface/trl/pull/4148
* 👩‍🦯 Fix usage of VLM using text only by @SamuelBarryCS in https://github.com/huggingface/trl/pull/4080
* ⚓ [vllm] ensure MASTER_ADDR/MASTER_PORT are set safely by @kashif in https://github.com/huggingface/trl/pull/4057
* 📤 Fix a dataset loading bug in scripts by @singing-cat in https://github.com/huggingface/trl/pull/4124
* 🐯 fix: use_liger_kernel with IterableDataset by @jue-jue-zi in https://github.com/huggingface/trl/pull/4087
* [GKD] Fix `batchmean` reduce op in GKDTrainer's loss by @cmpatino in https://github.com/huggingface/trl/pull/4105
* Fix get_peft_model() so that prepare_model_for_kbit_training does not reapply to an instance of PeftModel, thus freezing all the layers by @Hoesu in https://github.com/huggingface/trl/pull/4081
* Aux loss is already included in the loss returned by Transformers by @pramodith in https://github.com/huggingface/trl/pull/4078
* ♨️ [GRPO] Fix potential hang in `get_high_entropy_mask` by @akakakakakaa in https://github.com/huggingface/trl/pull/4041

## Documentation

* Remove logging.md: trainer-specific metrics documentation by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4269
* Remove using_llama_models.md: outdated Llama2-specific documentation by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4268
* Remove how_to_train.md: outdated training FAQ by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4267
* Add Qwen3-VL notebooks (SFT, GRPO) by @sergiopaniego in https://github.com/huggingface/trl/pull/4275
* Remove obsolete research_projects directory by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4243
* Add Efficient Online Training with GRPO and vLLM in TRL to community tutorials by @sergiopaniego in https://github.com/huggingface/trl/pull/4219
* Add trainers taxonomy to docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4195
* Updated vLLM integration guide by @sergiopaniego in https://github.com/huggingface/trl/pull/4162
* [DOCS] Lora without regret by @burtenshaw in https://github.com/huggingface/trl/pull/4181
* Add docstring for OnlineTrainerState by @albertvillanova in https://github.com/huggingface/trl/pull/4166
* ⚖️ Align SFT and DPO for model creation and deprecate `DPOConfig.padding_value` in favour or `pad_token_id` by @qgallouedec in https://github.com/huggingface/trl/pull/4006
* 🏞️ Context Parallelism benchmark guide by @sergiopaniego in https://github.com/huggingface/trl/pull/4075
* ▶️ Add video to community tutorials by @qgallouedec in https://github.com/huggingface/trl/pull/4090
* Reviewed HF jobs updated docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4088

## Deprecations

* Deprecate `BestOfNSampler` by @qgallouedec in https://github.com/huggingface/trl/pull/4291
* Raise deprecation warning for Python 3.9 by @albertvillanova in https://github.com/huggingface/trl/pull/4226
* Deprecate unused dataset_formatting module by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4242
* Warnings pointing to RFC by @qgallouedec in https://github.com/huggingface/trl/pull/4224
* 🅰️ Remove apex by @qgallouedec in https://github.com/huggingface/trl/pull/4139
* 🗑️ Remove deprecated `AlignPropTrainer`, `DDPOTrainer` and `IterativeSFTTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/4068

## Experimental

* 🧪 Add `trl.experimental` Submodule by @August-murr in https://github.com/huggingface/trl/pull/4073
* [GRPO]: Sample from a Replay Buffer To Substitute Groups with 0 std. by @pramodith in https://github.com/huggingface/trl/pull/4060
* 🪙 [Experimental] Support GSPO-token by @hjh0119 in https://github.com/huggingface/trl/pull/3820
* 🌪️ [GFPO]: implement GFPO in GRPOTrainer by @Peter-Chou in https://github.com/huggingface/trl/pull/3989
* 🌾 [Experimental] BEMA for ref model by @qgallouedec in https://github.com/huggingface/trl/pull/3898

## What's Changed

* ⬆️ Bump dev version by @qgallouedec in https://github.com/huggingface/trl/pull/4054
* Remove redundant 'None' from docstrings by @albertvillanova in https://github.com/huggingface/trl/pull/4058
* Hotfix: Add ParallelismConfig fallback for transformers with old accelerate by @albertvillanova in https://github.com/huggingface/trl/pull/4063
* Fix CI failure in slow GRPO test due to missing pillow dependency by @albertvillanova in https://github.com/huggingface/trl/pull/4064
* 💡 Fix type hint to `make_parser` function in multiple scripts by @qgallouedec in https://github.com/huggingface/trl/pull/4050
* Improve docstring of AlignPropTrainer by @albertvillanova in https://github.com/huggingface/trl/pull/4059
* ♨️ [GRPO] Fix potential hang in `get_high_entropy_mask` by @akakakakakaa in https://github.com/huggingface/trl/pull/4041
* Set Ruff src for first-party imports by @albertvillanova in https://github.com/huggingface/trl/pull/4074
* 🧪 Add `trl.experimental` Submodule by @August-murr in https://github.com/huggingface/trl/pull/4073
* 🌾 [Experimental] BEMA for ref model by @qgallouedec in https://github.com/huggingface/trl/pull/3898
* ✂️ [GRPO VLM] Update split sizes to generalize by @zucchini-nlp in https://github.com/huggingface/trl/pull/4032
* 🛠️ Fix CI by @qgallouedec in https://github.com/huggingface/trl/pull/4076
* 🐳 Docker update + Simplify Jobs doc by @qgallouedec in https://github.com/huggingface/trl/pull/3931
* Aux loss is already included in the loss returned by Transformers by @pramodith in https://github.com/huggingface/trl/pull/4078
* Reviewed HF jobs updated docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4088
* 🗑️ Remove deprecated `AlignPropTrainer`, `DDPOTrainer` and `IterativeSFTTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/4068
* ▶️ Add video to community tutorials by @qgallouedec in https://github.com/huggingface/trl/pull/4090
* Align slow tests with regular tests by @albertvillanova in https://github.com/huggingface/trl/pull/4085
* Add support for testing experimental features by @albertvillanova in https://github.com/huggingface/trl/pull/4082
* Community Tutorials design adaptation for videos by @sergiopaniego in https://github.com/huggingface/trl/pull/4095
* 🏞️ Context Parallelism benchmark guide by @sergiopaniego in https://github.com/huggingface/trl/pull/4075
* ⌨️ Pin num2words by @lewtun in https://github.com/huggingface/trl/pull/4094
* Add deprecation warnings to docstrings by @albertvillanova in https://github.com/huggingface/trl/pull/4083
* 📜 Convert `set` to `list` of tags by @qgallouedec in https://github.com/huggingface/trl/pull/4092
* 🧶 feat: Add WeaveCallback for W&B Weave integration by @parambharat in https://github.com/huggingface/trl/pull/4089
* ⚖️ Align SFT and DPO for model creation and deprecate `DPOConfig.padding_value` in favour or `pad_token_id` by @qgallouedec in https://github.com/huggingface/trl/pull/4006
* 🌪️ [GFPO]: implement GFPO in GRPOTrainer by @Peter-Chou in https://github.com/huggingface/trl/pull/3989
* ℹ️ feat: Add NPU and XPU support for activation offloading by @zilongzheng in https://github.com/huggingface/trl/pull/4056
* ℹ️ Enable XPU for vLLM client by @jiqing-feng in https://github.com/huggingface/trl/pull/4031
* Fix get_peft_model() so that prepare_model_for_kbit_training does not reapply to an instance of PeftModel, thus freezing all the layers by @Hoesu in https://github.com/huggingface/trl/pull/4081
* [GKD] Fix `batchmean` reduce op in GKDTrainer's loss by @cmpatino in https://github.com/huggingface/trl/pull/4105
* 👁️ Add VLM support to RLOO trainer by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4067
* Some nits GRPO and RLOO trainer docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4108
* Fix typos by @cyyever in https://github.com/huggingface/trl/pull/4106
* Fix typos by @qgallouedec in https://github.com/huggingface/trl/pull/4109
* Fix VLM configs in generate_tiny_models by @albertvillanova in https://github.com/huggingface/trl/pull/4101
* docs: correct option name to enable vllm sleep mode by @muupan in https://github.com/huggingface/trl/pull/4102
* CI hotfix: xfail test_training_with_transformers_paged for transformers<4.57.0 by @albertvillanova in https://github.com/huggingface/trl/pull/4120
* Fix code style with make precommit by @albertvillanova in https://github.com/huggingface/trl/pull/4119
* 🟩 Drop `image_split_sizes` in favour of `image_grid_thw` by @qgallouedec in https://github.com/huggingface/trl/pull/4111
* 🔭 Align param passing to VLM configs in generate_tiny_models by @albertvillanova in https://github.com/huggingface/trl/pull/4118
* 📽 Multi image support for GRPO/RLOO by @qgallouedec in https://github.com/huggingface/trl/pull/4113
* 😴 Add `vllm_enable_sleep_mode` to RLOO Trainer by @sergiopaniego in https://github.com/huggingface/trl/pull/4107
* 🐯 fix: use_liger_kernel with IterableDataset by @jue-jue-zi in https://github.com/huggingface/trl/pull/4087
* 📤 Fix a dataset loading bug in scripts by @singing-cat in https://github.com/huggingface/trl/pull/4124
* ⚓ [vllm] ensure MASTER_ADDR/MASTER_PORT are set safely by @kashif in https://github.com/huggingface/trl/pull/4057
* 📌 Pin vLLM version by @qgallouedec in https://github.com/huggingface/trl/pull/4122
* 👋 Remove `backend` parameter from `GuidedDecodingParams` by @qgallouedec in https://github.com/huggingface/trl/pull/4123
* 🧹 Remove `max_batch_tokens`, `num_blocks` and `block_size` from generation kwargs by @qgallouedec in https://github.com/huggingface/trl/pull/4065
* Remove Python version < 3.13 constraint from vllm extra dependencies by @albertvillanova in https://github.com/huggingface/trl/pull/4125
* 👩‍🦯 Fix usage of VLM using text only by @SamuelBarryCS in https://github.com/huggingface/trl/pull/4080
* [SFTrainer]: Fix DFT Loss by @pramodith in https://github.com/huggingface/trl/pull/4112
* Improve typing of SFT trainer by @cyyever in https://github.com/huggingface/trl/pull/4007
* 🌺 Fix GPT-OSS test by @qgallouedec in https://github.com/huggingface/trl/pull/4134
* 🪙 [Experimental] Support GSPO-token by @hjh0119 in https://github.com/huggingface/trl/pull/3820
* Fix CI: torch.AcceleratorError: CUDA error: device-side assert triggered by @albertvillanova in https://github.com/huggingface/trl/pull/4138
* 🤸‍♀️ Fix DFT test by @qgallouedec in https://github.com/huggingface/trl/pull/4135
* 🌵 Mark GKD trainer test as expected failure due to OOM issue by @qgallouedec in https://github.com/huggingface/trl/pull/4126
* [GRPO]: Sample from a Replay Buffer To Substitute Groups with 0 std. by @pramodith in https://github.com/huggingface/trl/pull/4060
* Fix import statement and GRPO test case by @qgallouedec in https://github.com/huggingface/trl/pull/4141
* Refactor trainers classes to use BaseTrainer with shared functionality by @albertvillanova in https://github.com/huggingface/trl/pull/4128
* Fixed some <Tip> rendering issues by @sergiopaniego in https://github.com/huggingface/trl/pull/4143
* 😷 Refactor GRPO/RLOO to isolate `_generate` by @qgallouedec in https://github.com/huggingface/trl/pull/4114
* 🟩 Drop `image_split_sizes` in favour of `image_grid_thw` by @qgallouedec in https://github.com/huggingface/trl/pull/4156
* 📽 Multi image support for GRPO replay buffer by @qgallouedec in https://github.com/huggingface/trl/pull/4157
* 😷 Refactor GRPO/RLOO to isolate `_generate` for GRPO with replay buffer by @qgallouedec in https://github.com/huggingface/trl/pull/4158
* Add docstring for OnlineTrainerState by @albertvillanova in https://github.com/huggingface/trl/pull/4166
* Pass required token_type_ids by @albertvillanova in https://github.com/huggingface/trl/pull/4148
* 💡 Replace `<Tip>` with new markdown syntax by @qgallouedec in https://github.com/huggingface/trl/pull/4161
* Remove unnecessary list comprehensions by @albertvillanova in https://github.com/huggingface/trl/pull/4164
* Add missing FDivergenceType docstring by @albertvillanova in https://github.com/huggingface/trl/pull/4165
* Fix docstrings with 'deprecated' Sphinx directive by @albertvillanova in https://github.com/huggingface/trl/pull/4174
* Fix docstring interlink to parent class for NashMDTrainer and XPOTrainer by @albertvillanova in https://github.com/huggingface/trl/pull/4179
* Fix link in docstring of RLOOTrainer by @albertvillanova in https://github.com/huggingface/trl/pull/4180
* 🖨️ Print rich table for messages by @qgallouedec in https://github.com/huggingface/trl/pull/4160
* 🅰️ Remove apex by @qgallouedec in https://github.com/huggingface/trl/pull/4139
* Fix CI ValueError: Unknown loss type: dapo by @albertvillanova in https://github.com/huggingface/trl/pull/4173
* Fix PEFT interlinks in docstrings by @albertvillanova in https://github.com/huggingface/trl/pull/4178
* ✨ Add logging for training completion and model saving in training scripts by @qgallouedec in https://github.com/huggingface/trl/pull/4048
* 👾 Use our own `require_bitsandbytes` by @qgallouedec in https://github.com/huggingface/trl/pull/4137
* 🎞️ Support sequence classification models in `clone_chat_template` by @qgallouedec in https://github.com/huggingface/trl/pull/4097
* ⚡ Fix Flash Attention x Padding-Free loss by @qgallouedec in https://github.com/huggingface/trl/pull/4170
* 🎁 `RewardTrainer` refactor by @qgallouedec in https://github.com/huggingface/trl/pull/4093
* 🧺 [1/N] Refactor `_generate` in GRPO/RLOO: list of ints instead of tensors by @qgallouedec in https://github.com/huggingface/trl/pull/4146
* Fix handling of f_divergence_type in DPO by @albertvillanova in https://github.com/huggingface/trl/pull/4171
* 🔣 Fix test: replace `trainer.tokenizer` by `trainer.processing_class` by @qgallouedec in https://github.com/huggingface/trl/pull/4185
* Fix CI ImportError: FlashAttention2 and decorator order for all parameterized tests by @albertvillanova in https://github.com/huggingface/trl/pull/4176
* Hotfix wrong formatting of docstrings with blockquote tips by @albertvillanova in https://github.com/huggingface/trl/pull/4187
* 🌡️ Have vLLM return processed (temperature scaled) log probs by @YonatanGideoni in https://github.com/huggingface/trl/pull/4163
* Replace remaining trainer.tokenizer with trainer.processing_class in GRPO test by @albertvillanova in https://github.com/huggingface/trl/pull/4192
* [DOCS] Lora without regret by @burtenshaw in https://github.com/huggingface/trl/pull/4181
* [DOCS/FIX] lora without regrets - fix lr by @burtenshaw in https://github.com/huggingface/trl/pull/4207
* Remove custome_container for building the docs by @albertvillanova in https://github.com/huggingface/trl/pull/4198
* Remove tokenizer creation from `sft` example script by @sergiopaniego in https://github.com/huggingface/trl/pull/4197
* Hotfix: Exclude transformers 4.57.0 for Python 3.9 by @albertvillanova in https://github.com/huggingface/trl/pull/4209
* Replace unittest with pytest by @albertvillanova in https://github.com/huggingface/trl/pull/4188
* Updated vLLM integration guide by @sergiopaniego in https://github.com/huggingface/trl/pull/4162
* Remove `Optional` from `processing_class` in `PPOTrainer` by @sergiopaniego in https://github.com/huggingface/trl/pull/4212
* Replace setup with pyproject and fix packaging unintended modules by @albertvillanova in https://github.com/huggingface/trl/pull/4194
* Removed tokenizer/processor creation from example scripts by @sergiopaniego in https://github.com/huggingface/trl/pull/4211
* Apply style and revert change in `sft_video_llm` example by @qgallouedec in https://github.com/huggingface/trl/pull/4214
* Fix `trl-internal-testing/tiny-DbrxForCausalLM` by @qgallouedec in https://github.com/huggingface/trl/pull/4213
* Fix prompt-completion labeling with add_generation_prompt and warning by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4201
* Fix LoRA params in Python in LoRA without regret by @sergiopaniego in https://github.com/huggingface/trl/pull/4215
* [DOCS] fix prose in lora guide by @burtenshaw in https://github.com/huggingface/trl/pull/4217
* Add trainers taxonomy to docs by @sergiopaniego in https://github.com/huggingface/trl/pull/4195
* 🎨 Support mixing image+text and text-only examples by @qgallouedec in https://github.com/huggingface/trl/pull/4203
* 🧺 [2/N] Refactor `_generate` in GRPO/RLOO: Use `prompt_ids` from generation by @qgallouedec in https://github.com/huggingface/trl/pull/4152
* Fix entropy and accuracy calculation for prompt_tuning techniques. by @pramodith in https://github.com/huggingface/trl/pull/4196
* Add Efficient Online Training with GRPO and vLLM in TRL to community tutorials by @sergiopaniego in https://github.com/huggingface/trl/pull/4219
* 🏷️ Account for `token_type_ids` in `DataCollatorForVisionLanguageModeling` by @qgallouedec in https://github.com/huggingface/trl/pull/4190
* Exclude vllm dependencies from dev extra by @albertvillanova in https://github.com/huggingface/trl/pull/4229
* Fix CI unittest asserts by @albertvillanova in https://github.com/huggingface/trl/pull/4234
* Fix callable annotations by @albertvillanova in https://github.com/huggingface/trl/pull/4216
* Remove unused Path import in __init__.py by @albertvillanova in https://github.com/huggingface/trl/pull/4227
* Update CI Docker image to pytorch/pytorch:2.8.0 by @albertvillanova in https://github.com/huggingface/trl/pull/4232
* Replace setup with pyproject in CI tests paths by @albertvillanova in https://github.com/huggingface/trl/pull/4230
* Fix CI IndentationError for Python 3.13.8 by @albertvillanova in https://github.com/huggingface/trl/pull/4240
* Remove unused log_example_reports.py script by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4241
* 🧘 Enhance markdown style by @qgallouedec in https://github.com/huggingface/trl/pull/4235
* Warnings pointing to RFC by @qgallouedec in https://github.com/huggingface/trl/pull/4224
* Fix CI slow test ValueError: Backward pass should have cleared tracker of all tensors by @sywangyi in https://github.com/huggingface/trl/pull/4236
* Fix CI CUDA out of memory errors by improving GPU memory management by @albertvillanova in https://github.com/huggingface/trl/pull/4238
* Install peft from main for CI tests with dev dependencies by @albertvillanova in https://github.com/huggingface/trl/pull/4250
* Fix CI ImportError for 'require_torch_gpu_if_bnb_not_multi_backend_enabled' by @albertvillanova in https://github.com/huggingface/trl/pull/4253
* Fix CI slow test ValueError: Unknown loss type: dapo by @albertvillanova in https://github.com/huggingface/trl/pull/4254
* 🧺 [3/N] Refactor `_generate` in GRPO/RLOO: Rely on generator for prompt truncation by @qgallouedec in https://github.com/huggingface/trl/pull/4153
* Remove obsolete research_projects directory by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4243
* Deprecate unused dataset_formatting module by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4242
* Fix CI slow test AttributeError: 'TestSFTTrainerSlow' object has no attribute 'addCleanup' by @albertvillanova in https://github.com/huggingface/trl/pull/4255
* [Online-DPO] fix the completion_len == max_new_tokens crash by @kashif in https://github.com/huggingface/trl/pull/4193
* Include `chat_template_kwargs` in `apply_chat_template` by @cmpatino in https://github.com/huggingface/trl/pull/4233
* Fix Python version check for skipping tests on Python 3.13.8 by @albertvillanova in https://github.com/huggingface/trl/pull/4246
* Raise deprecation warning for Python 3.9 by @albertvillanova in https://github.com/huggingface/trl/pull/4226
* Fix docstring interlinks by @albertvillanova in https://github.com/huggingface/trl/pull/4221
* Use FutureWarning instead of DeprecationWarning by @albertvillanova in https://github.com/huggingface/trl/pull/4266
* Fix style with make precommit by @albertvillanova in https://github.com/huggingface/trl/pull/4265
* Add Qwen3-VL notebooks (SFT, GRPO) by @sergiopaniego in https://github.com/huggingface/trl/pull/4275
* Fix typo in Colab link by @sergiopaniego in https://github.com/huggingface/trl/pull/4276
* Fix docstrings with Sphinx 'deprecated' directive by @albertvillanova in https://github.com/huggingface/trl/pull/4279
* Fix CI slow test OSError: You are trying to access a gated repo by @albertvillanova in https://github.com/huggingface/trl/pull/4283
* 💰 `RichProgressCallback` enhancement by @qgallouedec in https://github.com/huggingface/trl/pull/4245
* Fix CI dev test TypeError: unexpected keyword argument 'load_in_4bit' by @albertvillanova in https://github.com/huggingface/trl/pull/4262
* Replace unittest skipTest with pytest.skip by @albertvillanova in https://github.com/huggingface/trl/pull/4263
* Fix CI slow tests: ImportError: vLLM is not installed by @albertvillanova in https://github.com/huggingface/trl/pull/4287
* Remove logging.md: trainer-specific metrics documentation by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4269
* Remove using_llama_models.md: outdated Llama2-specific documentation by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4268
* Add support for `token_type_ids` in `DPOTrainer` by @aweers in https://github.com/huggingface/trl/pull/4285
* Remove how_to_train.md: outdated training FAQ by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4267
* Add accuracy reward by @pramodith in https://github.com/huggingface/trl/pull/4270
* Remove unused commands directory by @behroozazarkhalili in https://github.com/huggingface/trl/pull/4258
* Deprecate `BestOfNSampler` by @qgallouedec in https://github.com/huggingface/trl/pull/4291
* Release: v0.24 by @qgallouedec in https://github.com/huggingface/trl/pull/4292

## New Contributors

* @zucchini-nlp made their first contribution in https://github.com/huggingface/trl/pull/4032
* @parambharat made their first contribution in https://github.com/huggingface/trl/pull/4089
* @zilongzheng made their first contribution in https://github.com/huggingface/trl/pull/4056
* @jiqing-feng made their first contribution in https://github.com/huggingface/trl/pull/4031
* @Hoesu made their first contribution in https://github.com/huggingface/trl/pull/4081
* @cmpatino made their first contribution in https://github.com/huggingface/trl/pull/4105
* @singing-cat made their first contribution in https://github.com/huggingface/trl/pull/4124
* @SamuelBarryCS made their first contribution in https://github.com/huggingface/trl/pull/4080
* @YonatanGideoni made their first contribution in https://github.com/huggingface/trl/pull/4163
* @aweers made their first contribution in https://github.com/huggingface/trl/pull/4285

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.23.0...v0.24.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.24.0)

---

## v0.23.1: v0.23.1
**Published:** 2025-10-02

## What's Changed

* ♨️ [GRPO] Fix potential hang in `get_high_entropy_mask` by @akakakakakaa in https://github.com/huggingface/trl/pull/4041
* Aux loss is already included in the loss returned by Transformers by @pramodith in https://github.com/huggingface/trl/pull/4078
* Fix get_peft_model() so that prepare_model_for_kbit_training does not reapply to an instance of PeftModel, thus freezing all the layers by @Hoesu in https://github.com/huggingface/trl/pull/4081
* 🐯 fix: use_liger_kernel with IterableDataset by @jue-jue-zi in https://github.com/huggingface/trl/pull/4087
* [SFTrainer]: Fix DFT Loss by @pramodith in https://github.com/huggingface/trl/pull/4112
* ⚡ Fix Flash Attention x Padding-Free loss by @qgallouedec in https://github.com/huggingface/trl/pull/4170

## New Contributors

* @Hoesu made their first contribution in https://github.com/huggingface/trl/pull/4081

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.23.0...v0.23.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.23.1)

---

## v0.23.0: v0.23.0
**Published:** 2025-09-10

## Major

### 🥓 Context Parallelism

SFT now supports Context Parallelism (CP) for training large language models on very large sequences. You can now train with an arbitrarily long sequence length.

<img width="844" height="336" alt="Screenshot 2025-09-09 at 10 39 30 PM" src="https://github.com/user-attachments/assets/f1dfc349-440a-4e05-aac9-439a3c286f08" />

by @kashif in https://github.com/huggingface/trl/pull/3994

### 🧨 Dynamic Fine-Tuning


Dynamic Fine-Tuning (DFT) is a nnow supported in TRL.

```python
from trl import SFTConfig

training_args = SFTConfig(
    loss_type="dft",
    ...
)
```

<img width="692" height="472" alt="Screenshot 2025-09-09 at 10 37 36 PM" src="https://github.com/user-attachments/assets/4ee2b4ab-7cc6-4578-bfac-c38124891510" />

by @qgallouedec in https://github.com/huggingface/trl/pull/4042

### 🪵 Truncated Importance Sampling (TIS) to address rollout-training mismatch

Different implementations are used for rollout generation (vLLM) and model training. The implementation gap implicitly turns the on-policy RL to be off-policy. Truncated Importance Sampling (TIS) a simple yet effective importance sampling technique for handling such discrepancy. This is now implemented in GRPO.

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...
    use_vllm=True,
    vllm_importance_sampling_correction=True, # default True
    vllm_importance_sampling_cap=2.0, # hyper-parameter C
)
```

by @LeonEricsson in https://github.com/huggingface/trl/pull/3867

### 🥣 [SFTTrainer]: Add Aux Loss for MoE models

Mixture of Experts (MoE) models require an auxiliary loss to ensure that the different experts are used evenly. This auxiliary loss is now supported in SFTTrainer.

```python
training_args = SFTConfig(
    model_init_kwargs={"output_router_logits": True},
    ...
)
```

by @pramodith in https://github.com/huggingface/trl/pull/4012

### 💤 [GRPO/RLOO] Adds an option to sleep vllm when running in colocated mode

When running GRPO (or RLOO) with vLLM in colocated mode, the vLLM server consume VRAM during optimization while not being used. We now have an option to put the vLLM server to sleep during optimization to free up VRAM.

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., vllm_sleep_enabled=True)
```

by @edbeeching in https://github.com/huggingface/trl/pull/3968

### ⚖️ Add vLLM server mode and VLM support to OnlineDPOTrainer

You can now use vLLM server mode with OnlineDPOTrainer. Additionally, VLM models are now supported.

by @vaelev in https://github.com/huggingface/trl/pull/3783


### Comprehensive Paper Index Enhancement with 9 New Algorithm Implementations

The paper index has been significantly enhanced with the addition of 9+ new algorithm implementations, providing a more comprehensive resource for users.

by @behroozazarkhalili in https://github.com/huggingface/trl/pull/3990

### Other Notable Changes

* 👷 Added Kernels on the Hub x TRL guide by @sergiopaniego in https://github.com/huggingface/trl/pull/3969
* 🌵 Refactor entropy_from_logits for memory efficiency by @qgallouedec in https://github.com/huggingface/trl/pull/4013

## What's Changed

* ⬆️ Bump dev version by @qgallouedec in https://github.com/huggingface/trl/pull/3978
* 👮 Fix GRPO CLI by setting parameters for `get_soft_overlong_punishment` by @qgallouedec in https://github.com/huggingface/trl/pull/3972
* 🪃 `args.gradient_checkpointing = False` instead of `args = dataclasses.replace(args, gradient_checkpointing=False)` by @qgallouedec in https://github.com/huggingface/trl/pull/3981
* [GRPO] Adds an option to sleep vllm when running in colocated mode by @edbeeching in https://github.com/huggingface/trl/pull/3968
* 🎯 Add Trackio integration documentation and update TOC by @qgallouedec in https://github.com/huggingface/trl/pull/3971
* ⚖️ Fix scale_rewards issue in GRPO by @Peter-Chou in https://github.com/huggingface/trl/pull/3992
* ⏰ fix: add return to shift_tokens_right by @ginkyenglee in https://github.com/huggingface/trl/pull/3987
* Add pre-commit and hf-doc-builder as dev dependencies by @albertvillanova in https://github.com/huggingface/trl/pull/3993
* [GRPO] Truncated Importance Sampling to address rollout-training mismatch by @LeonEricsson in https://github.com/huggingface/trl/pull/3867
* Fixed tags shown problem in memory usage docs by @sergiopaniego in https://github.com/huggingface/trl/pull/3999
* ✖️ Support pad-to-multiple-of and padding-free by @qgallouedec in https://github.com/huggingface/trl/pull/3996
* 💾 [bugfix] fix PPO save_checkpoint by @hjh0119 in https://github.com/huggingface/trl/pull/3998
* [GRPO]: Fix Multi-GPU training for Entropy based masking of tokens. by @pramodith in https://github.com/huggingface/trl/pull/3964
* 📏 `torch_dype` to `dtype` everywhere by @sergiopaniego in https://github.com/huggingface/trl/pull/4000
* Comprehensive Paper Index Enhancement with 9 New Algorithm Implementations by @behroozazarkhalili in https://github.com/huggingface/trl/pull/3990
* [SFT] fix: collator docstring by @LeonEricsson in https://github.com/huggingface/trl/pull/4011
* 👷 Added Kernels on the Hub x TRL guide by @sergiopaniego in https://github.com/huggingface/trl/pull/3969
* 🌵 Refactor entropy_from_logits for memory efficiency by @qgallouedec in https://github.com/huggingface/trl/pull/4013
* [SFTTrainer]: Add Aux Loss for MoE models. by @pramodith in https://github.com/huggingface/trl/pull/4012
* Add missing doc strings in SFTrainer by @pramodith in https://github.com/huggingface/trl/pull/4003
* ⚖️ Add vLLM server mode and VLM support to OnlineDPOTrainer by @vaelev in https://github.com/huggingface/trl/pull/3783
* Fix typo in GRPO quickstart by @dwisdom0 in https://github.com/huggingface/trl/pull/4020
* Align docstring parameters with function definitions by @albertvillanova in https://github.com/huggingface/trl/pull/4017
* Fix formatting errors in docstrings by @albertvillanova in https://github.com/huggingface/trl/pull/4025
* [doc] Paper index for Truncated Importance Sampling by @LeonEricsson in https://github.com/huggingface/trl/pull/4026
* [doc] Group paper index by trainer by @LeonEricsson in https://github.com/huggingface/trl/pull/4027
* Add missing trainer docstrings by @albertvillanova in https://github.com/huggingface/trl/pull/4030
* Add autodoc for AlignPropTrainer and AlignPropConfig by @albertvillanova in https://github.com/huggingface/trl/pull/4033
* 🥓 [docs] add CP docs by @kashif in https://github.com/huggingface/trl/pull/3994
* ⚖️ Remove `average_tokens_across_devices` default replacement by @qgallouedec in https://github.com/huggingface/trl/pull/4039
* CI hotfix: xfail test_training_with_transformers_paged by @albertvillanova in https://github.com/huggingface/trl/pull/4046
* Update transformers minimum version to 4.56.1 by @albertvillanova in https://github.com/huggingface/trl/pull/4047
* 🧨 DFT by @qgallouedec in https://github.com/huggingface/trl/pull/4042
* Update VLM arch check to `AutoModelForImageTextToText` for DPO and Online DPO by @sergiopaniego in https://github.com/huggingface/trl/pull/4049
* 🏂 Fix label shifting logic in `SFTTrainer` for compatibility with CP by @qgallouedec in https://github.com/huggingface/trl/pull/4038
* Add autodoc for BestOfNSampler and improve docstrings by @albertvillanova in https://github.com/huggingface/trl/pull/4034
* ✨ Improve SFT doc  by @qgallouedec in https://github.com/huggingface/trl/pull/4005
* 💬 Remove setting chat template in sft script by @qgallouedec in https://github.com/huggingface/trl/pull/4037
* 🪪 Update SFTTrainer to handle labels correctly and add configuration example in paper index by @qgallouedec in https://github.com/huggingface/trl/pull/4051
* 🗜 Hotfix: avoid passing `quantization_config=None` by @qgallouedec in https://github.com/huggingface/trl/pull/4019
* Release: 0.23 by @qgallouedec in https://github.com/huggingface/trl/pull/4053

## New Contributors

* @Peter-Chou made their first contribution in https://github.com/huggingface/trl/pull/3992
* @ginkyenglee made their first contribution in https://github.com/huggingface/trl/pull/3987
* @albertvillanova made their first contribution in https://github.com/huggingface/trl/pull/3993
* @hjh0119 made their first contribution in https://github.com/huggingface/trl/pull/3998
* @vaelev made their first contribution in https://github.com/huggingface/trl/pull/3783
* @dwisdom0 made their first contribution in https://github.com/huggingface/trl/pull/4020

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.22.0...v0.23.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.23.0)

---

## v0.22.2: v0.22.2
**Published:** 2025-09-03

## What's Changed

* ⚖️ Fix scale_rewards issue in GRPO by @Peter-Chou in https://github.com/huggingface/trl/pull/3992
* ⏰ fix: add return to shift_tokens_right by @ginkyenglee in https://github.com/huggingface/trl/pull/3987
* ✖️ Support pad-to-multiple-of and padding-free by @qgallouedec in https://github.com/huggingface/trl/pull/3996

## New Contributors
* @Peter-Chou made their first contribution in https://github.com/huggingface/trl/pull/3992

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.22.1...v0.22.2

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.22.2)

---

## v0.22.1: v0.22.1
**Published:** 2025-08-29

## What changed
- Refactor version retrieval to use `importlib.metadata` by @qgallouedec
- Release: 0.22.1 by @qgallouedec

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.22.0...v0.22.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.22.1)

---

## v0.22.0: v0.22.0
**Published:** 2025-08-29

## Major

### 🔮 Native VLM support for `SFTTrainer`

`SFTTrainer` now natively supports Vision-Language Models (VLMs). This includes support for both languauge modeling, prompt-completion data. 
It also supports training on completion-only.

<img width="1136" height="586" alt="Group 291-6" src="https://github.com/user-attachments/assets/2629b8e7-d853-4b7c-91d5-f4c128287e04" />

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=SFTConfig(max_length=None),
    train_dataset=load_dataset("trl-lib/llava-instruct-mix", split="train"),
)
trainer.train()
```

by @qgallouedec in https://github.com/huggingface/trl/pull/3862, https://github.com/huggingface/trl/pull/3907 and https://github.com/huggingface/trl/pull/3908

### 🔥 `RLOOTrainer` refactor

`RLOOTrainer` has been refactored to align with the design principles of other other trainers in the library. You can now use this trainer exactly like GRPO.

```python
from datasets import load_dataset
from trl import RLOOConfig, RLOOTrainer

dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

# Dummy reward function for demonstration purposes
def reward_num_unique_letters(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]

trainer = RLOOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_num_unique_letters,
    train_dataset=dataset,
)
trainer.train()
```

by @shirinyamani in https://github.com/huggingface/trl/pull/3801

### 🧭 HF jobs x TRL guide

You can now levarage Hugging Face Jobs to easily train and deploy your models with TRL.

```bash
hf jobs uv run --flavor a100-large --secrets HF_TOKEN "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" --model_name_or_path Qwen/Qwen2-0.5B --dataset_name trl-lib/Capybara
```

A guide is available in the [docs](https://huggingface.co/docs/trl/main/en/training_jobs).

by @sergiopaniego in https://github.com/huggingface/trl/pull/3890

### 🏌️ DAPO loss type

`GRPOTrainer` now supports DAPO loss type, which aggregates token-level losses by normalizing with the number of active token in the global accumulated batch. This method was introduced to eliminate length bias. Simply use

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    loss_type="dapo",
    ...
)
```

by @qgallouedec in https://github.com/huggingface/trl/pull/3938

### 🪶 [GRPO] PPO Lite: Scale rewards by Std of Batch

The authors of [Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (Lite PPO)](https://huggingface.co/papers/2508.08221) find that the combination of:

1. scaling rewards by the standard deviation computed over the entire batch and
2. aggregating loss over the total number of tokens

can unlock the learning capability of critic-free policies using vanilla PPO loss. Their results demonstrate that this simple combination consistently improves performance, surpassing strategies like GRPO and [DAPO](https://huggingface.co/papers/2503.14476).

TRL supports using these learnings to train a GRPO model by:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    scale_rewards="batch",
    loss_type="dapo",
    ...
)
```

by @pramodith in https://github.com/huggingface/trl/pull/3935

### 🎢 [Callbacks] BEMA

Bias-Corrected Exponential Moving Average (BEMA) improves the stability and efficiency of language model fine-tuning by reducing stochasticity and eliminating bias. To use BEMA with SFT as described in the paper, you can now use the [`BEMACallback`]:

```python
from trl import BEMACallback, SFTTrainer

trainer = SFTTrainer(
    ...
    callbacks=[BEMACallback()],
)
```

by @kashif in https://github.com/huggingface/trl/pull/3855

### Minor

* 🎀 New defaults: `gradient_checkpointing=True` by @qgallouedec in https://github.com/huggingface/trl/pull/3510
* 🎚️ Add dataset mixer by @lewtun in https://github.com/huggingface/trl/pull/3791
* 💇 Add soft overlong punishment reward function and update documentation by @qgallouedec in https://github.com/huggingface/trl/pull/3804
* 🗿 [CPO] Add AlphaPO method via CPOTrainer by @kashif in https://github.com/huggingface/trl/pull/3824
* 🗳️ Extend BCO Trainer dataset format support by @reihig-ut in https://github.com/huggingface/trl/pull/3134
* 🐯 Support assistant-only training and Liger by @qgallouedec in https://github.com/huggingface/trl/pull/3914
* 🎆 Add entropy logging in SFT by @qgallouedec in https://github.com/huggingface/trl/pull/3940
* 📸 Return `position_ids` for `flash_attention_3` by @jue-jue-zi in https://github.com/huggingface/trl/pull/3942

## Deprecations

* 🗑️ Deprecate `setup_chat_format` by @qgallouedec in https://github.com/huggingface/trl/pull/3929
* 🗑 Deprecate `IterativeSFTTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/3905

## What's Changed

* ⬆️ Bump dev version by @qgallouedec in https://github.com/huggingface/trl/pull/3850
* 🔗 Fix collection link in doc by @qgallouedec in https://github.com/huggingface/trl/pull/3852
* Typo fix in new model description by @sergiopaniego in https://github.com/huggingface/trl/pull/3854
* Small style fix in README by @qgallouedec in https://github.com/huggingface/trl/pull/3861
* [GRPO]  👁️ Fix vLLM server mode for VLM GRPO training incompatibility for certain AutoProcessors by @ghubnerr in https://github.com/huggingface/trl/pull/3832
* 👁️ From `AutoModelForVision2Seq` to `AutoModelForImageTextToText` by @qgallouedec in https://github.com/huggingface/trl/pull/3836
* 👋 Remove `--bf16` value in scripts by @sergiopaniego in https://github.com/huggingface/trl/pull/3869
* 🎀 New defaults: `gradient_checkpointing=True` by @qgallouedec in https://github.com/huggingface/trl/pull/3510
* 🦦 Validate `vllm_mode` param in GRPO by @sergiopaniego in https://github.com/huggingface/trl/pull/3866
* 🎚️ Add dataset mixer by @lewtun in https://github.com/huggingface/trl/pull/3791
* ✨ Integrate PEFT model preparation across trainers and utilities by @qgallouedec in https://github.com/huggingface/trl/pull/3882
* ⌨️ Add py.typed by @cyyever in https://github.com/huggingface/trl/pull/3841
* 💇 Add soft overlong punishment reward function and update documentation by @qgallouedec in https://github.com/huggingface/trl/pull/3804
* 🕹️ [GRPO] Fix vllm mode validation in distributed setting by @Kirill-Kravtsov in https://github.com/huggingface/trl/pull/3886
* ⏳ Replaced `unittest.TestCase` with `TrlTestCase` that handles tmp dir by @qgallouedec in https://github.com/huggingface/trl/pull/3863
* 🔮 Native VLM support for `SFTTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/3862
* Minor optimizations in SFT. by @pramodith in https://github.com/huggingface/trl/pull/3884
* 🧩 Fix reward_processing_classes validation in GRPOTrainer by @chi2liu in https://github.com/huggingface/trl/pull/3876
* 🎢 [Callbacks] BEMA by @kashif in https://github.com/huggingface/trl/pull/3855
* 👁️ VLM blog by @qgallouedec in https://github.com/huggingface/trl/pull/3899
* 🪄 Improve quickstart documentation with updated API examples by @behroozazarkhalili in https://github.com/huggingface/trl/pull/3873
* 👔 HF Doc Builder style by @qgallouedec in https://github.com/huggingface/trl/pull/3498
* ✏️ Fix SFTTrainer token accuracy computation with PromptEncoder by @zk-quantum in https://github.com/huggingface/trl/pull/3821
* ☑️  Check eval batch size in grpo by @jp1924 in https://github.com/huggingface/trl/pull/3889
* ⚔️ Optimize truncate_with_protected_tokens to use vectorized operations by @chi2liu in https://github.com/huggingface/trl/pull/3875
* Add tests for get_position_ids_from_packed_seq_lengths by @pramodith in https://github.com/huggingface/trl/pull/3883
* 🌳 Enhance segment tree implementation for non-power-of-2 values by @MengAiDev in https://github.com/huggingface/trl/pull/3888
* ⚡ Optimize completion_ids list conversion in GRPO trainer by @chi2liu in https://github.com/huggingface/trl/pull/3874
* 🗿 [CPO] Add AlphaPO method via CPOTrainer by @kashif in https://github.com/huggingface/trl/pull/3824
* 🗳️ Extend BCO Trainer dataset format support by @reihig-ut in https://github.com/huggingface/trl/pull/3134
* 🐯 Support assistant-only training and Liger by @qgallouedec in https://github.com/huggingface/trl/pull/3914
* 🗑 Deprecate `IterativeSFTTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/3905
* ♻️ `use_cache` should be set in the forward pass by @qgallouedec in https://github.com/huggingface/trl/pull/3891
* 🌓 SFTTrainer for VLM: Support for prompt-completion data by @qgallouedec in https://github.com/huggingface/trl/pull/3907
* ➡️ SFTTrainer for VLM: support completion-only loss by @qgallouedec in https://github.com/huggingface/trl/pull/3908
* 📚 Update BEMACallback documentation to ignore docstyle and fix lag parameter description by @qgallouedec in https://github.com/huggingface/trl/pull/3917
* ✏️ Fix typos by @cyyever in https://github.com/huggingface/trl/pull/3921
* 🧹 Clean SFT tests by @qgallouedec in https://github.com/huggingface/trl/pull/3922
* 🤹‍♂️ Multi-image testing dataset by @qgallouedec in https://github.com/huggingface/trl/pull/3916
* 🧾 Use `logger.warning` instead of `warnings.warn` by @qgallouedec in https://github.com/huggingface/trl/pull/3923
* ♻️ Reuse multimodal message preparation from `SFTTrainer` in `GRPOTrainer` by @MengAiDev in https://github.com/huggingface/trl/pull/3919
* 🗑️ Deprecate `setup_chat_format` by @qgallouedec in https://github.com/huggingface/trl/pull/3929
* 🗞 bugfix 'TrainerState' object is not subscriptable by @ErezYosef in https://github.com/huggingface/trl/pull/3936
* 📦 Wrapping the main execution code to avoid multi-processing issues from vLLM by @kaixuanliu in https://github.com/huggingface/trl/pull/3932
* 🪶 [GRPO] PPO Lite: Scale rewards by Std of Batch by @pramodith in https://github.com/huggingface/trl/pull/3935
* 🏌️ DAPO loss type by @qgallouedec in https://github.com/huggingface/trl/pull/3938
* 🎆 Add entropy logging in SFT by @qgallouedec in https://github.com/huggingface/trl/pull/3940
* 🗂 Update paper_index section by @behroozazarkhalili in https://github.com/huggingface/trl/pull/3937
* [DPO] Adding support for different losses which are now supported by Liger by @Manan17 in https://github.com/huggingface/trl/pull/3815
* [GKD] add liger loss by @kashif in https://github.com/huggingface/trl/pull/3946
* 🤸 [SFT] Drop entropy calculation when using liger by @qgallouedec in https://github.com/huggingface/trl/pull/3947
* ✂️ fix: handle list tensors in split_tensor_dict function by @qgallouedec in https://github.com/huggingface/trl/pull/3951
* 🪶 LitePPO: Fix Docs for paper index by @pramodith in https://github.com/huggingface/trl/pull/3954
* 🦥 Unsloth Docs update by @shimmyshimmer in https://github.com/huggingface/trl/pull/3955
* 📸 Return `position_ids` for `flash_attention_3` by @jue-jue-zi in https://github.com/huggingface/trl/pull/3942
* 🧭 HF jobs x TRL guide by @sergiopaniego in https://github.com/huggingface/trl/pull/3890
* Add HF jobs tag when creating model card via jobs by @sergiopaniego in https://github.com/huggingface/trl/pull/3956
* [CI] Modify tests to handle device allocation for models by @kashif in https://github.com/huggingface/trl/pull/3962
* Fixed some typos and added small details about `trackio` to docs by @sergiopaniego in https://github.com/huggingface/trl/pull/3965
* 🔥 [Refactor] RLOOTrainer by @shirinyamani in https://github.com/huggingface/trl/pull/3801
* ℹ️ Validate examples on xpu by @yao-matrix in https://github.com/huggingface/trl/pull/3897
* 📜 GSPO docs - Sequence importance ratio and differences in relation to GRPO by @almeidava93 in https://github.com/huggingface/trl/pull/3816
* Fix CI by @qgallouedec in https://github.com/huggingface/trl/pull/3975
* 🧱 PyPI publishing workflow by @qgallouedec in https://github.com/huggingface/trl/pull/3976
* Release: v0.22 by @qgallouedec in https://github.com/huggingface/trl/pull/3977

## New Contributors
* @ghubnerr made their first contribution in https://github.com/huggingface/trl/pull/3832
* @Kirill-Kravtsov made their first contribution in https://github.com/huggingface/trl/pull/3886
* @behroozazarkhalili made their first contribution in https://github.com/huggingface/trl/pull/3873
* @zk-quantum made their first contribution in https://github.com/huggingface/trl/pull/3821
* @MengAiDev made their first contribution in https://github.com/huggingface/trl/pull/3888
* @reihig-ut made their first contribution in https://github.com/huggingface/trl/pull/3134
* @ErezYosef made their first contribution in https://github.com/huggingface/trl/pull/3936
* @kaixuanliu made their first contribution in https://github.com/huggingface/trl/pull/3932
* @Manan17 made their first contribution in https://github.com/huggingface/trl/pull/3815
* @shimmyshimmer made their first contribution in https://github.com/huggingface/trl/pull/3955
* @jue-jue-zi made their first contribution in https://github.com/huggingface/trl/pull/3942
* @almeidava93 made their first contribution in https://github.com/huggingface/trl/pull/3816

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.21.0...v0.22.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.22.0)

---

## v0.21.0: v0.21.0
**Published:** 2025-08-05

## Major and breaking

### 🌺 OpenAI GPT OSS & Harmony support

<img width="4544" height="2344" alt="Group 293-2" src="https://github.com/user-attachments/assets/17241da2-1b1d-41bc-a5f8-0983ea46606f" />

Open AI GPT OSS models are here! Check out the [OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) to see an example of how to SFT these models.

by @qgallouedec in https://github.com/huggingface/trl/pull/3848

### Add vLLM transformers backend to online methods

You can now pass `vllm_model_impl` to the TRL vLLM server.
Example, for `transformers` backend:

```
trl vllm-serve ... --vllm_model_impl transformers
```

by @merveenoyan in https://github.com/huggingface/trl/pull/3773

## What's Changed
* ⬆️ Bump dev version by @qgallouedec in https://github.com/huggingface/trl/pull/3793
* Fix broken PEFT+TRL docs link in `using_llama_models.md` by @bwook00 in https://github.com/huggingface/trl/pull/3794
* 🐙 Add MPO VLM example script by @sergiopaniego in https://github.com/huggingface/trl/pull/3799
* Examples list updated in docs by @sergiopaniego in https://github.com/huggingface/trl/pull/3806
* Add vLLM transformers backend to online methods by @merveenoyan in https://github.com/huggingface/trl/pull/3773
* Correction parameter description by @1787648106 in https://github.com/huggingface/trl/pull/3803
* Add GSPO script examples (VLM/LLM) by @sergiopaniego in https://github.com/huggingface/trl/pull/3810
* add xpu support for mergekit by @yao-matrix in https://github.com/huggingface/trl/pull/3800
* GSPO parameters update from v2 by @BounharAbdelaziz in https://github.com/huggingface/trl/pull/3798
* fix CI docs and grpo slow test by @kashif in https://github.com/huggingface/trl/pull/3814
* Performance optimization: Replace list comprehensions with tensor operations in BCO and KTO trainers by @chi2liu in https://github.com/huggingface/trl/pull/3813
* Improve trainer doc by @qgallouedec in https://github.com/huggingface/trl/pull/3818
* Add 'Post training a VLM for reasoning with GRPO using TRL' recipe to Community tutorials by @sergiopaniego in https://github.com/huggingface/trl/pull/3843
* [GRPO]: Fix Entropy Mask Threshold Calculation when using Multi-GPU training by @pramodith in https://github.com/huggingface/trl/pull/3833
* 🪦 Remove deprecated by @qgallouedec in https://github.com/huggingface/trl/pull/3817
* 🌺 OpenAI GPT OSS & Harmony support by @qgallouedec in https://github.com/huggingface/trl/pull/3848
* Release: v0.21 by @qgallouedec in https://github.com/huggingface/trl/pull/3849

## New Contributors
* @bwook00 made their first contribution in https://github.com/huggingface/trl/pull/3794
* @merveenoyan made their first contribution in https://github.com/huggingface/trl/pull/3773
* @1787648106 made their first contribution in https://github.com/huggingface/trl/pull/3803
* @BounharAbdelaziz made their first contribution in https://github.com/huggingface/trl/pull/3798
* @chi2liu made their first contribution in https://github.com/huggingface/trl/pull/3813

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.20.0...v0.21.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.21.0)

---

## v0.20.0: v0.20.0
**Published:** 2025-07-29

## Breaking and major changes

### 🎞️ GSPO

GSPO is a GRPO variant that computes importance sampling weights at the sequence level instead of per-token. 

<img width="930" height="538" alt="Screenshot 2025-07-28 at 10 54 15 PM" src="https://github.com/user-attachments/assets/923835af-dc61-4fd4-8a99-44242d02bb7b" />

📜 Paper: https://huggingface.co/papers/2507.18071

To reproduce the paper's setting, use this configuration:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence",
    loss_type="grpo",
    steps_per_generation=...,
    beta=0.04,  # not explicitly specified in the paper, but they likely used the same value as in the GRPO paper
    epsilon=3e-4,  # https://x.com/ChujieZheng/status/1948933507696525392
)
```

by @qgallouedec in https://github.com/huggingface/trl/pull/3775

### 👁️ [GRPO] Add VLM training capabilities to the GRPO trainer

<img width="1136" height="594" alt="Group 291-4" src="https://github.com/user-attachments/assets/04850e80-9689-472d-acd7-fda331e66dc3" />

The GRPOTrainer can now be used for VLM training. Give a try with this dummy example:

```python
from trl import GRPOTrainer
from datasets import load_dataset

# Dummy vision-language dataset
dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c[0]["content"])) for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    reward_funcs=[reward_num_unique_chars],
    train_dataset=dataset,
)

trainer.train()
```

by @CompN3rd and @kashif in https://github.com/huggingface/trl/pull/3072 in https://github.com/huggingface/trl/pull/3760

### 🐙 MPO

<img width="440" height="438" alt="Screenshot 2025-07-28 at 10 52 15 PM" src="https://github.com/user-attachments/assets/e07a7936-c4c5-480d-9ffd-db5b77a5445e" />

The DPO trainer supports combining multiple loss functions with different weights, enabling more sophisticated optimization strategies. This is particularly useful for implementing algorithms like MPO (Mixed Preference Optimization). MPO is a training approach that combines multiple optimization objectives, as described in the paper [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://huggingface.co/papers/2411.10442).

To combine multiple losses, specify the loss types and corresponding weights as lists:

```python
from trl import DPOConfig

# MPO: Combines DPO (sigmoid) for preference and BCO (bco_pair) for quality
training_args = DPOConfig(
    loss_type=["sigmoid", "bco_pair", "sft"],  # Loss types to combine
    loss_weights=[0.8, 0.2, 1.0]  # Corresponding weights, as used in the MPO paper
)
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2544

### Add support for CB with native transformers

Continuous Batching allows for faster generation using the `transformers` backend. You can now use it with the `GRPOTrainer` by setting `use_transformers_paged=True` in the config.

```python
use_transformers_paged = True
from trl import GRPOConfig
training_args = GRPOConfig(
    # ... other args
    use_transformers_paged=Ture,
)
```

by @ArthurZucker in https://github.com/huggingface/trl/pull/3471

### Add entropy based filtering inside the GRPOTrainer


<img width="788" height="438" alt="Screenshot 2025-07-28 at 10 27 20 PM" src="https://github.com/user-attachments/assets/8073a5db-a98e-4534-aea9-d3dbd2e75f4a" />


In [Beyond the 80/20 Rule: High-Entropy Minority Tokens
Drive Effective Reinforcement Learning for LLM Reasoning](https://huggingface.co/papers/2506.01939), it is shown that utilizing only 20% of the highest entropy tokens leads to similar performance as using all tokens. You can now enable this feature in the `GRPOTrainer` by setting `entropy_filtering=True` in the config.

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    # ... other args
    top_entropy_quantile=0.2,  # Use only the top 20% of tokens based on entropy
)
```

by @pramodith in https://github.com/huggingface/trl/pull/3563

### 👐 FSDP2+GRPO

GRPO now supports FSDP2 training. Just run your script with an FSDP2 config:

```bash
accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml run_grpo.py
```

by @SalmanMohammadi in https://github.com/huggingface/trl/pull/3687

## What's Changed
* ⬆️ Bump dev version by @qgallouedec in https://github.com/huggingface/trl/pull/3626
* fix grpo generation_kwargs by @ahatamiz in https://github.com/huggingface/trl/pull/3634
* fixing num_processes by @shirinyamani in https://github.com/huggingface/trl/pull/3637
* env var for vllm colocate exp added by @shirinyamani in https://github.com/huggingface/trl/pull/3638
* Update dpo_vlm.py by @Clement25 in https://github.com/huggingface/trl/pull/3629
* ☕️ GRPO script reward_funcs error by @tcapelle in https://github.com/huggingface/trl/pull/3639
* 🤝 validate gradient_accumulation_steps vs steps_per_generation for on-policy GRPO by @HarryHsing in https://github.com/huggingface/trl/pull/3493
* Add entropy based filtering inside the GRPOTrainer. by @pramodith in https://github.com/huggingface/trl/pull/3563
* Make sure chat template isn't lost when truncating prompt. by @pramodith in https://github.com/huggingface/trl/pull/3651
* Add paranthesis to correct the check. by @pramodith in https://github.com/huggingface/trl/pull/3658
* Add support for CB with native transformers by @ArthurZucker in https://github.com/huggingface/trl/pull/3471
* feat: Pass trainer state to reward functions by @seungduk-yanolja in https://github.com/huggingface/trl/pull/3669
* Enable completion-only loss in SFTTrainer when using Liger Kernel by @kswhitecross in https://github.com/huggingface/trl/pull/3674
* Add mlflow support for generate_during_eval DPOTrainer by @dhruvmullick in https://github.com/huggingface/trl/pull/3660
* [SFT] drop attention_mask if we have position ids for fa2 by @kashif in https://github.com/huggingface/trl/pull/3673
* Faster `position_ids` computation for FFD packing by @mariosasko in https://github.com/huggingface/trl/pull/3649
* Support datasets 4 by @lhoestq in https://github.com/huggingface/trl/pull/3688
* Update steps_per_generation default description grpo_config.py by @wa008 in https://github.com/huggingface/trl/pull/3685
* Fix non-serializable torch.dtype bug in VLLM weight sync by @CarlosArguilar in https://github.com/huggingface/trl/pull/3690
* fix: support dict access in SFT Trainer by @jannisborn in https://github.com/huggingface/trl/pull/3677
* [fix] type error of quantile by @gitabtion in https://github.com/huggingface/trl/pull/3667
* [CI] Fix slow grpo CI by @kashif in https://github.com/huggingface/trl/pull/3693
* Restore the effect of liger_kernel's monkey_patch on global modules in UT. by @YangKai0616 in https://github.com/huggingface/trl/pull/3680
* Add type hints to `dpo_trainer.py` by @bvantuan in https://github.com/huggingface/trl/pull/3631
* Fix mislabeling: "First-fit decreasing" is actually "Best-fit-decreasing" by @LeonEricsson in https://github.com/huggingface/trl/pull/3696
* ✂️ [BUG when vllm and prompt_truncation are used]: Strip out pad tokens in truncated prompt text by @pramodith in https://github.com/huggingface/trl/pull/3698
* 📣 Use explicit version for checking datasets version by @qgallouedec in https://github.com/huggingface/trl/pull/3702
* 🔭 Fix package discovery configuration in setup.cfg by @qgallouedec in https://github.com/huggingface/trl/pull/3703
* [SFT] Add `seq_lengths` to signature columns by @LeonEricsson in https://github.com/huggingface/trl/pull/3699
* ⚗️ Tiny MoE for test by @qgallouedec in https://github.com/huggingface/trl/pull/3712
* BUG: Disregard pad token entropies for entropy threshold calculation by @pramodith in https://github.com/huggingface/trl/pull/3715
* Fix ORPOTrainer loss scaling with gradient accumulation by @Aratako in https://github.com/huggingface/trl/pull/3716
* [Online DPO] Safeguard logit slice against empty prompt by @LeonEricsson in https://github.com/huggingface/trl/pull/3719
* Remove deprecated `processor.tokenizer` by @Tavish9 in https://github.com/huggingface/trl/pull/3720
* 👋 Remove `--bf16` flag from training scripts by @qgallouedec in https://github.com/huggingface/trl/pull/3724
* ↔️ Fix CB in GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/3722
* 📥 Set environment variables for vLLM distributed training in GRPOTrainer by @qgallouedec in https://github.com/huggingface/trl/pull/3723
* [GRPO] remove common activation offloading substring in all cases by @winglian in https://github.com/huggingface/trl/pull/3738
* 🔧 Fix GRPO sampling logic by @qgallouedec in https://github.com/huggingface/trl/pull/3725
* 🕸 Use `wandb.run.url` instead of `wandb.run.get_url()` (deprecated) by @qgallouedec in https://github.com/huggingface/trl/pull/3726
* Updated `processing_class` docs for trainers by @sergiopaniego in https://github.com/huggingface/trl/pull/3737
* Updated missing `processing_class` docs for rest of trainers by @sergiopaniego in https://github.com/huggingface/trl/pull/3745
* Add comment for `average_tokens_across_devices` by @qgallouedec in https://github.com/huggingface/trl/pull/3746
* uses `steps_per_generation` in vllm max_num_seqs by @akakakakakaa in https://github.com/huggingface/trl/pull/3747
* 🏗️ Refactor top-entropy in GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/3727
* [GRPO] Fix: Processing ref logprobs in batches by @idanshen in https://github.com/huggingface/trl/pull/3740
* Add Object detection grounding recipe to Community tutorials by @sergiopaniego in https://github.com/huggingface/trl/pull/3752
* 🐙 MPO by @qgallouedec in https://github.com/huggingface/trl/pull/2544
* ⚰️ Remove deprecated by @qgallouedec in https://github.com/huggingface/trl/pull/3704
* 👨‍💼 [SFT] Packing with completion_only and assistant_only training by @LeonEricsson in https://github.com/huggingface/trl/pull/3749
* 👁️ [GRPO] Add VLM training capabilities to the trainer by @CompN3rd in https://github.com/huggingface/trl/pull/3072
* Add MPO recipe to Community tutorials by @sergiopaniego in https://github.com/huggingface/trl/pull/3766
* ✋ Prevent NCCL Device Conflicts Between vLLM Server and Trainers by @CarlosArguilar in https://github.com/huggingface/trl/pull/3762
* 🔔 Add deprecation warnings for `AlignPropTrainer` and `DDPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/3755
* 🔠 Support model str in OnlineDPO by @kashif in https://github.com/huggingface/trl/pull/3765
* 🌌 [GRPO] Log generation entropy by @LeonEricsson in https://github.com/huggingface/trl/pull/3700
* 🤓 [GRPO] Documentation for entropy metric by @LeonEricsson in https://github.com/huggingface/trl/pull/3770
* Add uv scripts headers by @lhoestq in https://github.com/huggingface/trl/pull/3767
* Update missing uv dep by @lhoestq in https://github.com/huggingface/trl/pull/3772
* 📐 Fix CI and `GeometricMixtureWrapper` by @qgallouedec in https://github.com/huggingface/trl/pull/3779
* 🩹 [Hotfix] Fix pynccl communicator assertion error with VLLMClient by @CarlosArguilar in https://github.com/huggingface/trl/pull/3774
* 🍿 [SFT] Fix dataset indexing which crashed with a IterableDataset  by @LeonEricsson in https://github.com/huggingface/trl/pull/3771
* 🎞️ GSPO by @qgallouedec in https://github.com/huggingface/trl/pull/3775
* 🤏 [SFT] Improve doc on training on assistant only messages by @lewtun in https://github.com/huggingface/trl/pull/3784
* 📐 Add epsilon hyperparameter recommendation to GSPO by @qgallouedec in https://github.com/huggingface/trl/pull/3790
* 📍 Support training peft model with gradient checkpointing by @qgallouedec in https://github.com/huggingface/trl/pull/3785
* 💬 Fix `clone_chat_template` vocab size and support PEFT instruction tuning by @qgallouedec in https://github.com/huggingface/trl/pull/3763
* 🌋 [GRPO] add support for `pixel_attention_mask` (SmolVLM2) and `image_sizes` (LLaVa-Next) by @kashif in https://github.com/huggingface/trl/pull/3760
* 🔍 Add guidance on choosing `max_length` value and include visualization tool by @qgallouedec in https://github.com/huggingface/trl/pull/3630
* 📘 SFT doc rewrite by @qgallouedec in https://github.com/huggingface/trl/pull/3619
* 👐 FSDP2+GRPO by @SalmanMohammadi in https://github.com/huggingface/trl/pull/3687
* Release: v0.20 by @qgallouedec in https://github.com/huggingface/trl/pull/3792

## New Contributors
* @ahatamiz made their first contribution in https://github.com/huggingface/trl/pull/3634
* @Clement25 made their first contribution in https://github.com/huggingface/trl/pull/3629
* @HarryHsing made their first contribution in https://github.com/huggingface/trl/pull/3493
* @ArthurZucker made their first contribution in https://github.com/huggingface/trl/pull/3471
* @seungduk-yanolja made their first contribution in https://github.com/huggingface/trl/pull/3669
* @kswhitecross made their first contribution in https://github.com/huggingface/trl/pull/3674
* @lhoestq made their first contribution in https://github.com/huggingface/trl/pull/3688
* @CarlosArguilar made their first contribution in https://github.com/huggingface/trl/pull/3690
* @jannisborn made their first contribution in https://github.com/huggingface/trl/pull/3677
* @gitabtion made their first contribution in https://github.com/huggingface/trl/pull/3667
* @YangKai0616 made their first contribution in https://github.com/huggingface/trl/pull/3680
* @Aratako made their first contribution in https://github.com/huggingface/trl/pull/3716
* @CompN3rd made their first contribution in https://github.com/huggingface/trl/pull/3072

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.19.0...v0.20.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.20.0)

---

## v0.19.1: v0.19.1
**Published:** 2025-07-08

## What's Changed

* fix grpo generation_kwargs by @ahatamiz in https://github.com/huggingface/trl/pull/3634
* Make sure chat template isn't lost when truncating prompt. by @pramodith in https://github.com/huggingface/trl/pull/3651
* Add paranthesis to correct the check. by @pramodith in https://github.com/huggingface/trl/pull/3658
* [SFT] drop attention_mask if we have position ids for fa2 by @kashif in https://github.com/huggingface/trl/pull/3673
* Support datasets 4 by @lhoestq in https://github.com/huggingface/trl/pull/3688
* 📣 Use explicit version for checking datasets version by @qgallouedec in https://github.com/huggingface/trl/pull/3702
* Fix non-serializable torch.dtype bug in VLLM weight sync by @CarlosArguilar in https://github.com/huggingface/trl/pull/3690
* ✂️ [BUG when vllm and prompt_truncation are used]: Strip out pad tokens in truncated prompt text by @pramodith in https://github.com/huggingface/trl/pull/3698

## New Contributors

* @ahatamiz made their first contribution in https://github.com/huggingface/trl/pull/3634
* @lhoestq made their first contribution in https://github.com/huggingface/trl/pull/3688
* @CarlosArguilar made their first contribution in https://github.com/huggingface/trl/pull/3690

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.19.0...v0.19.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.19.1)

---

## v0.19.0: v0.19.0
**Published:** 2025-06-21

## Breaking and major changes

### 🧰 [SFT] Tool support

`SFTTrainer` now supports training with tools! You just have to add a column `tools` to your dataset, which contains a list of tool definitions as json schemas. The tools will be automatically registered and can be used in the training process.

```python
from datasets import Dataset
from transformers.utils import get_json_schema
from trl import SFTTrainer

# Fictitious functions to simulate tool calls
def start_timer(duration: int) -> int:
    """
    Starts a timer for the specified duration in seconds.

    Args:
        duration: Duration in seconds to set the timer for.

    Returns:
        The duration set for the timer.
    """
    return duration

def create_reminder(time: str, note: str) -> str:
    """
    Creates a reminder for the specified time and note.

    Args:
        time: The time for the reminder.
        note: The note for the reminder.

    Returns:
        A confirmation message indicating that the reminder has been set.
    """
    return "I'll remind you to call mom at 7 PM."

# Define the JSON schemas for the tools
start_timer = get_json_schema(start_timer)
create_reminder = get_json_schema(create_reminder)

dataset = Dataset.from_dict({
    "messages": [
        [
            {"role": "user", "content": "Set a timer for 10 minutes."},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "start_timer", "arguments": {"duration": 600}}}]},
            {"role": "tool", "name": "start_timer", "content": "600"},
            {"role": "assistant", "content": "Timer set for 10 minutes."},
        ],
        ...,
    ],
    "tools": [
        [start_timer, create_reminder],
        ...,
    ]
})

# Initialize the trainer
trainer = SFTTrainer(model="Qwen3-0.6B", train_dataset=dataset)

# Train the model
trainer.train()
```

by @qgallouedec in <https://github.com/huggingface/trl/pull/3597>

### 📉 FFD packing

We introduce a new packing method: FFD (First Fit Decreasing) packing. This method is designed to optimize the packing of sequences in a way that more efficiently reduces the size of the training dataset by grouping examples more effectively. Previously, we used a _wrapped_ packing method, which often truncated sequences even when they were not longer than the maximum sequence length. The new _FFD_ packing method avoids unnecessary truncation by grouping sequences more intelligently. This new packing strategy is now the default when packing is enabled.

```python
training_args = SFTConfig(..., packing=True)
```

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/packing_2.png)

by @qgallouedec in <https://github.com/huggingface/trl/pull/3521> and accelerated by @mariosasko in <https://github.com/huggingface/trl/pull/3537>

### [Liger] liger DPO support

The `DPOTrainer` now supports the Liger-powered DPO loss, enabling faster training with lower memory usage.

```python
training_args = DPOConfig(..., use_liger_loss=True)
```

by @kashif in <https://github.com/huggingface/trl/pull/2568>

### 💬 Fix `setup_chat_format` and add `clone_chat_template`

We introduce `clone_chat_template`, a more convenient and flexible function for setting up chat templates from any tokenizer that already includes one.  It handles EOS tokens and copies all added tokens from the source tokenizer, preserving their "special" status.
You can either use this function directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import clone_chat_template

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model, tokenizer = clone_chat_template(model, tokenizer, "Qwen/Qwen3-4B")
```

or use the `chat_template_path` parameter in `SFTConfig` to specify a chat template, which will be automatically cloned when the SFTTrainer is initialized.

```python
from trl import SFTConfig

training_args = SFTConfig(chat_template_path="Qwen/Qwen3-4B")
```

by @qgallouedec in <https://github.com/huggingface/trl/pull/3404> and <https://github.com/huggingface/trl/pull/3599>

### 📚 SFTTrainer support chat template kwargs

`SFTTrainer` now supports passing additional keyword arguments to the chat template. This allows for more flexibility in customizing the chat format during training. To enable it, just add a `chat_template_kwargs` column to your your dataset.

```python
example = {'messages': [{'content': 'What is better than ugly?', 'role': 'user'},
                        {'content': 'Beautiful.', 'role': 'assistant'}]
           'chat_template_kwargs': {'my_template_arg': 'my_value'}}
```

by @qgallouedec in <https://github.com/huggingface/trl/pull/3609>

### 🤵‍♂️ SFT on assistant messages only

The `SFTTrainer` now supports training on assistant messages only

```python
example = {'messages': [
    {'role': 'user', 'content': 'What is better than ugly?'},          # masked in the loss
    {'role': 'assistant', 'content': 'Beautiful.'},                    # used in the loss
    {'role': 'user', 'content': 'And what is better than implicit?'},  # masked in the loss
    {'role': 'assistant', 'content': 'Explicit.'},                     # used in the loss
]}
```

by @qgallouedec in <https://github.com/huggingface/trl/pull/3586>

### 🧬 Add `generation_kwargs` as a property of `GRPOConfig` to support additional generation arguments

The `GRPOConfig` now includes a `generation_kwargs` property, allowing users to specify additional generation arguments for the `GRPOTrainer`. This allows for further customization of the generation behavior, such as setting `suppress_tokens`, `num_beams`, etc.
Depending on the generation backend used (transformers or vLLM), this property will be passed either to `transformers.GenerationConfig` (if using transformers) or `vllm.SamplingParams` (if using vLLM).

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., generation_kwargs={"length_penalty": -0.1})
```

by @pramodith in <https://github.com/huggingface/trl/pull/3617>

## New defaults

* 🎀 New default: `beta=0.0` for GRPO by @qgallouedec in <https://github.com/huggingface/trl/pull/3516>
* 🎀 New defaults: preparing the new structure by @qgallouedec in <https://github.com/huggingface/trl/pull/3530>
* 🎀 New defaults: `logging_steps=10` by @qgallouedec in <https://github.com/huggingface/trl/pull/3514>
* 🎀 [SFT][Bugfix] sets average_tokens_across_devices to true in SFTConfig by @edbeeching in <https://github.com/huggingface/trl/pull/3538>
* 🎀 New defaults: `bf16=True` by @qgallouedec in <https://github.com/huggingface/trl/pull/3515>

## Minor changes

* Add support for `IterableDataset` in DPO Trainer by @h-tonywu in <https://github.com/huggingface/trl/pull/3559>
* 🔖 Fix: ensure user-provided `labels` are retained in self._signature_columns by @sxndqc in <https://github.com/huggingface/trl/pull/3589>
* ⭐ Add `vllm_gpu_memory_utilization` recommendation script by @toslali-ibm in <https://github.com/huggingface/trl/pull/3554>

## What's Changed

* ⬆️ Bump dev version by @qgallouedec in <https://github.com/huggingface/trl/pull/3505>
* 📎 Fix clip ratio logging by @qgallouedec in <https://github.com/huggingface/trl/pull/3506>
* 📚 Fix doc building by removing vLLM from dev dependencies in `setup.cfg` by @qgallouedec in <https://github.com/huggingface/trl/pull/3511>
* 🧭 Patch release guide by @qgallouedec in <https://github.com/huggingface/trl/pull/3512>
* 🎀 New default: `beta=0.0` for GRPO by @qgallouedec in <https://github.com/huggingface/trl/pull/3516>
* Add "🐯 Liger GRPO meets TRL" by @qgallouedec in <https://github.com/huggingface/trl/pull/3525>
* 📉 FFD packing by @qgallouedec in <https://github.com/huggingface/trl/pull/3521>
* 🎀 New defaults: preparing the new structure by @qgallouedec in <https://github.com/huggingface/trl/pull/3530>
* 🪦 RIP trl chat  by @shirinyamani in <https://github.com/huggingface/trl/pull/3531>
* 🎀 New defaults: `logging_steps=10` by @qgallouedec in <https://github.com/huggingface/trl/pull/3514>
* 📰 Add blog "No GPU left behind: Unlocking Efficiency with Co-located vLLM in TRL" by @qgallouedec in <https://github.com/huggingface/trl/pull/3527>
* 🎯 Don't use `getattr` to get `gradient_checkpointing` by @qgallouedec in <https://github.com/huggingface/trl/pull/3535>
* 🧭 Remove useless transformers version checks by @qgallouedec in <https://github.com/huggingface/trl/pull/3534>
* 🐳 Add DeepseekV3 model configurations and update tests for new models by @qgallouedec in <https://github.com/huggingface/trl/pull/3536>
* 💭 [Data] Fix DeepSeek-R1 case by @kashif in <https://github.com/huggingface/trl/pull/3522>
* 🎀 [SFT][Bugfix] sets average_tokens_across_devices to true in SFTConfig by @edbeeching in <https://github.com/huggingface/trl/pull/3538>
* ⚡ Faster FFD packing by @mariosasko in <https://github.com/huggingface/trl/pull/3537>
* 📦 Packing with flash attn kwargs to avoid cross-contamination by @thepowerfuldeez in <https://github.com/huggingface/trl/pull/3526>
* 💽 [TRLParser] Fail when unknown args are provided in the config file. by @edbeeching in <https://github.com/huggingface/trl/pull/3543>
* 🛋️ Fix CI and bump accelerate by @qgallouedec in <https://github.com/huggingface/trl/pull/3551>
* 🧮 Rearrange DPOTrainer by @DaizeDong in <https://github.com/huggingface/trl/pull/3501>
* 🆙 Bump transformers to 4.51 and use `_VALID_DICT_FIELDS` by @qgallouedec in <https://github.com/huggingface/trl/pull/3553>
* Update tests_latest.yml by @qgallouedec in <https://github.com/huggingface/trl/pull/3558>
* ℹ️ Unify autocast behavior to `torch.autocast` and make it cover XPU by @yao-matrix in <https://github.com/huggingface/trl/pull/3541>
* Fix dev version by @Tavish9 in <https://github.com/huggingface/trl/pull/3570>
* [Liger] liger DPO support by @kashif in <https://github.com/huggingface/trl/pull/2568>
* Add support for `IterableDataset` in DPO Trainer by @h-tonywu in <https://github.com/huggingface/trl/pull/3559>
* 🏗️ Add test for training with multiple dataloader workers and update worker initialization for compatibility with transformers 4.52.0 by @qgallouedec in <https://github.com/huggingface/trl/pull/3568>
* 🫸 Push model card with checkpoint by @qgallouedec in <https://github.com/huggingface/trl/pull/3550>
* Add Community Tutorial: GRPO text summarization example with Unsloth optimizations by @amanzoni1 in <https://github.com/huggingface/trl/pull/3576>
* 🎀 New defaults: `bf16=True` by @qgallouedec in <https://github.com/huggingface/trl/pull/3515>
* 📨 [SFT] Tokenize directly when applying the chat template by @qgallouedec in <https://github.com/huggingface/trl/pull/3572>
* Adjust max_num_batched_tokens by @toslali-ibm in <https://github.com/huggingface/trl/pull/3565>
* 💡 Fix type hints in trainer/utils.py by @bvantuan in <https://github.com/huggingface/trl/pull/3591>
* 💡 Fix wrong type hint for formatting_func argument in SFTTrainer by @MaiqiVerse in <https://github.com/huggingface/trl/pull/3584>
* 💬 Fix `setup_chat_format` and add `clone_chat_template` by @qgallouedec in <https://github.com/huggingface/trl/pull/3404>
* 🛡️ Adding trust_remote_code to vllm-serve by @maziyarpanahi in <https://github.com/huggingface/trl/pull/3588>
* Fix typos and improve metric descriptions in documentation by @vtjl10 in <https://github.com/huggingface/trl/pull/3585>
* Fix Typo in Documentation and Notebook; Improve Library Installation Comment by @zeevick10 in <https://github.com/huggingface/trl/pull/3593>
* ♻️ Avoids redundant calculation of ref logps in the new policy update loop by @zkpranav in <https://github.com/huggingface/trl/pull/3600>
* 🗳️ Remove `logging_steps` parameter from for simpler setup by @qgallouedec in <https://github.com/huggingface/trl/pull/3612>
* Fix: list-typed tags handling in `Trainer::create_model_card` by @LeonEricsson in <https://github.com/huggingface/trl/pull/3613>
* Fix Typos in Comments and Improve Clarity in Trainer Modules by @maximevtush in <https://github.com/huggingface/trl/pull/3596>
* Change `enforce_eager` default value in vLLM server. by @LeonEricsson in <https://github.com/huggingface/trl/pull/3607>
* [SFT] Clarify default collator docs  by @LeonEricsson in <https://github.com/huggingface/trl/pull/3606>
* 🏁 Refactor reference model initialization in GRPOTrainer by @Tavish9 in <https://github.com/huggingface/trl/pull/3575>
* 🏛️ Fix CI and Iterative SFT by @qgallouedec in <https://github.com/huggingface/trl/pull/3614>
* 👔 Apply doc-builder style by @qgallouedec in <https://github.com/huggingface/trl/pull/3615>
* 🔖 Fix: ensure user-provided `labels` are retained in self._signature_columns by @sxndqc in <https://github.com/huggingface/trl/pull/3589>
* 📚 SFTTrainer support chat template kwargs by @qgallouedec in <https://github.com/huggingface/trl/pull/3609>
* 🦘 Skip no-op ChatML conversion for datasets already in ChatML format by @qgallouedec in <https://github.com/huggingface/trl/pull/3594>
* 🤵‍♂️ SFT on assistant messages only by @qgallouedec in <https://github.com/huggingface/trl/pull/3586>
* 🎁 Put the reward computation in a separate function by @ajtejankar in <https://github.com/huggingface/trl/pull/3620>
* ⭐ Add `vllm_gpu_memory_utilization` recommendation script by @toslali-ibm in <https://github.com/huggingface/trl/pull/3554>
* [GRPO] Fix prompt truncation (`max_prompt_length`) with vLLM.  by @LeonEricsson in <https://github.com/huggingface/trl/pull/3601>
* 🧬 Add `generation_kwargs` as a property of `GRPOConfig` to support additional generation arguments. by @pramodith in <https://github.com/huggingface/trl/pull/3617>
* 📜 Add `chat_template_path` parameter to `SFTConfig` by @qgallouedec in <https://github.com/huggingface/trl/pull/3599>
* ⚔️ Fix bf16 fp16 config conflict issue by @yao-matrix in <https://github.com/huggingface/trl/pull/3598>
* 🔍 Add test to verify chat template consistency by @qgallouedec in <https://github.com/huggingface/trl/pull/3624>
* 🧰 [SFT] Tool support by @qgallouedec in <https://github.com/huggingface/trl/pull/3597>
* Release: v0.19 by @qgallouedec in <https://github.com/huggingface/trl/pull/3625>

## New Contributors

* @thepowerfuldeez made their first contribution in <https://github.com/huggingface/trl/pull/3526>
* @DaizeDong made their first contribution in <https://github.com/huggingface/trl/pull/3501>
* @h-tonywu made their first contribution in <https://github.com/huggingface/trl/pull/3559>
* @amanzoni1 made their first contribution in <https://github.com/huggingface/trl/pull/3576>
* @bvantuan made their first contribution in <https://github.com/huggingface/trl/pull/3591>
* @MaiqiVerse made their first contribution in <https://github.com/huggingface/trl/pull/3584>
* @maziyarpanahi made their first contribution in <https://github.com/huggingface/trl/pull/3588>
* @vtjl10 made their first contribution in <https://github.com/huggingface/trl/pull/3585>
* @zeevick10 made their first contribution in <https://github.com/huggingface/trl/pull/3593>
* @zkpranav made their first contribution in <https://github.com/huggingface/trl/pull/3600>
* @sxndqc made their first contribution in <https://github.com/huggingface/trl/pull/3589>
* @ajtejankar made their first contribution in <https://github.com/huggingface/trl/pull/3620>
* @pramodith made their first contribution in <https://github.com/huggingface/trl/pull/3617>

**Full Changelog**: <https://github.com/huggingface/trl/compare/v0.18.0...v0.19.0>


[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.19.0)

---

## v0.18.2: v0.18.2
**Published:** 2025-06-15

## What's Changed

* 🏗️ Add test for training with multiple dataloader workers and update worker initialization for compatibility with transformers 4.52.0 by @qgallouedec in https://github.com/huggingface/trl/pull/3568

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.18.1...v0.18.2

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.18.2)

---

## v0.18.1: v0.18.1
**Published:** 2025-06-03

## What's Changed

* 📎 Fix clip ratio logging by @qgallouedec in https://github.com/huggingface/trl/pull/3506
* 📚 Fix doc building by removing vLLM from dev dependencies in `setup.cfg` by @qgallouedec in https://github.com/huggingface/trl/pull/3511

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.18.0...v0.18.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.18.1)

---

## v0.18.0: v0.18.0
**Published:** 2025-05-28

## Major or breaking

* PEFT support for Liger GRPO by @SalmanMohammadi in https://github.com/huggingface/trl/pull/3355
* [🐯+GRPO] Support FSDP + Fix bug when using LigerGRPO with DDP by @shivam15s in https://github.com/huggingface/trl/pull/3260
* 🤝 Compatibility of the TRL CLI with accelerate arguments by @qgallouedec in https://github.com/huggingface/trl/pull/3409
* 🧑‍🤝‍🧑 Co-Locating vLLM w/ training to for higher throughput and GPU utilization by @toslali-ibm in https://github.com/huggingface/trl/pull/3394
* ✌️ Add support for FSDP2 by @lewtun in https://github.com/huggingface/trl/pull/3317
* 💔 [GRPO] Decouple gradient accumulation from the number of minibatches generated by @edbeeching in https://github.com/huggingface/trl/pull/3388
* [Models] Activation checkpointing from TorchTune by @kashif in https://github.com/huggingface/trl/pull/2954
* feat: Implement Two-Sided Clipping for GRPO Trainer by @ucalyptus in https://github.com/huggingface/trl/pull/3434
* 🎁 Reward submodule by @qgallouedec in https://github.com/huggingface/trl/pull/3430

## What's Changed
* ⬆️ Bump dev version by @qgallouedec in https://github.com/huggingface/trl/pull/3357
* 🔢 Pad to multiple of by @qgallouedec in https://github.com/huggingface/trl/pull/3362
* 🥸🔢 Adding pad_multiple to SFT trainer by @shirinyamani in https://github.com/huggingface/trl/pull/3365
* 🎭 Fix train and eval mode checking in `GRPOTrainer` and `SFTTrainer` by @I-l-l-I in https://github.com/huggingface/trl/pull/3337
* Better guards for DeepSpeed imports by @lewtun in https://github.com/huggingface/trl/pull/3351
* ⚰️  Remove deprecated by @qgallouedec in https://github.com/huggingface/trl/pull/3364
* 📋 Allow calling trl cli in sft mode with config file by @CloseChoice in https://github.com/huggingface/trl/pull/3380
* PEFT support for Liger GRPO by @SalmanMohammadi in https://github.com/huggingface/trl/pull/3355
* DPO fixes for evaluations by @winglian in https://github.com/huggingface/trl/pull/3377
* Deprecate `TextEnvironment` and tools by @lewtun in https://github.com/huggingface/trl/pull/3389
* [🐯+GRPO] Support FSDP + Fix bug when using LigerGRPO with DDP by @shivam15s in https://github.com/huggingface/trl/pull/3260
* [GRPO] Reference model initialization bug fix by @LeonEricsson in https://github.com/huggingface/trl/pull/3397
* 🌊 Add MLflow metrics in profiling context by @dhruvmullick in https://github.com/huggingface/trl/pull/3400
* 🧑‍🤝‍🧑 Co-Locating vLLM w/ training to for higher throughput and GPU utilization by @toslali-ibm in https://github.com/huggingface/trl/pull/3394
* ✨ [IterativeSFT] Small refresher  by @LeonEricsson in https://github.com/huggingface/trl/pull/3378
* 💔 [SFT] Raise error when `formatting_func` is used with `completion_only_loss` by @LeonEricsson in https://github.com/huggingface/trl/pull/3385
* 🦁 Fix liger initialization by @shivam15s in https://github.com/huggingface/trl/pull/3401
* 👉 [DPO] Model forward pass padding side fix  by @LeonEricsson in https://github.com/huggingface/trl/pull/3307
* 🪪 Remove license classifier by @qgallouedec in https://github.com/huggingface/trl/pull/3402
* 🕺 Migrate setup configuration from `setup.py` to `setup.cfg` and make `rich` an optional dep by @qgallouedec in https://github.com/huggingface/trl/pull/3403
* 🕊️ Un-restrict diffusers by @qgallouedec in https://github.com/huggingface/trl/pull/3407
* ✌️ Add support for FSDP2 by @lewtun in https://github.com/huggingface/trl/pull/3317
* 🤝 Compatibility of the TRL CLI with accelerate arguments by @qgallouedec in https://github.com/huggingface/trl/pull/3409
* 💔 [GRPO] Decouple gradient accumulation from the number of minibatches generated by @edbeeching in https://github.com/huggingface/trl/pull/3388
* 🎲 [GRPO] Shuffle mini batches by @edbeeching in https://github.com/huggingface/trl/pull/3391
* 📝 vLLM-integration documentation by @shirinyamani in https://github.com/huggingface/trl/pull/3376
* 🎁 Reward takes completion ids by @qgallouedec in https://github.com/huggingface/trl/pull/3272
* 🐍 Support Python 3.13 by @qgallouedec in https://github.com/huggingface/trl/pull/2593
* [Models] Activation checkpointing from TorchTune by @kashif in https://github.com/huggingface/trl/pull/2954
* 🧪 Testing support for Qwen3 tiny  by @shirinyamani in https://github.com/huggingface/trl/pull/3415
* Update README.md by @qgallouedec in https://github.com/huggingface/trl/pull/3420
* 🏹 Support kv_cache_dtype to quantize kv-cache in vllm by @winglian in https://github.com/huggingface/trl/pull/3422
* enable `trl env` on xpu by @yao-matrix in https://github.com/huggingface/trl/pull/3438
* use device agnostic empty_cache in ppo & rloo by @yao-matrix in https://github.com/huggingface/trl/pull/3439
* feat: Implement Two-Sided Clipping for GRPO Trainer by @ucalyptus in https://github.com/huggingface/trl/pull/3434
* 🎁 Reward submodule by @qgallouedec in https://github.com/huggingface/trl/pull/3430
* [CI] fix CI failure of transformer dev by @kashif in https://github.com/huggingface/trl/pull/3457
* enable vllm c-s tests on XPU by @yao-matrix in https://github.com/huggingface/trl/pull/3445
* enable activation offloading on XPU by @yao-matrix in https://github.com/huggingface/trl/pull/3444
* 🙅 PPO value_model can't be None, so it shouldn't be Optional by @AMindToThink in https://github.com/huggingface/trl/pull/3300
* [NashMD] fix the edge case where the model is a peft model by @kashif in https://github.com/huggingface/trl/pull/3473
* Update .pre-commit-config.yaml by @kashif in https://github.com/huggingface/trl/pull/3479
* [SFT] update minimal liger version by @kashif in https://github.com/huggingface/trl/pull/3483
* [CI] fix sampler api to make the CI green by @kashif in https://github.com/huggingface/trl/pull/3488
* Fix typo by @nikolai-kummer in https://github.com/huggingface/trl/pull/3489
* [Doc][SFT] Update sft_trainer.md. link prompt-completion dataset example by @HERIUN in https://github.com/huggingface/trl/pull/3486
* Fix mis-aligned prompts and completions in colocate mode by @toslali-ibm in https://github.com/huggingface/trl/pull/3491
* [Docs] sync logging doc to current metrics by @kashif in https://github.com/huggingface/trl/pull/3478
* [GRPO] disabling top_k sampling default by @kashif in https://github.com/huggingface/trl/pull/3494
* [GKD] fix the gkd script by @kashif in https://github.com/huggingface/trl/pull/3497
* 👇 Update grpo.py to fix bugs for cli grpo --reward_funcs my_lib.my_reward by @wa008 in https://github.com/huggingface/trl/pull/3454
* 🛠️ Initialize reward_kwargs to prevent UnboundLocalError in GRPOTrainer by @teilomillet in https://github.com/huggingface/trl/pull/3459
* 🐌 Clean two-sided clipping by @qgallouedec in https://github.com/huggingface/trl/pull/3499
* 🔭 [GRPO] Log advantages and fraction of samples with an std of zero by @edbeeching in https://github.com/huggingface/trl/pull/3502
* 📏 Completion length logging fix + remainder logging fix by @shirinyamani in https://github.com/huggingface/trl/pull/3482
* 🤧 LD-DPO support by @AIR-hl in https://github.com/huggingface/trl/pull/3458
* 🏰 [vllm] Support `base_url` parameter for vLLM client initialization by @re-imagined in https://github.com/huggingface/trl/pull/3324
* ✂️ [DPO] Fix truncation `keep_end` leading to zero'd out samples by @LeonEricsson in https://github.com/huggingface/trl/pull/3398
* Release: v0.18 by @qgallouedec in https://github.com/huggingface/trl/pull/3504

## New Contributors
* @CloseChoice made their first contribution in https://github.com/huggingface/trl/pull/3380
* @SalmanMohammadi made their first contribution in https://github.com/huggingface/trl/pull/3355
* @dhruvmullick made their first contribution in https://github.com/huggingface/trl/pull/3400
* @toslali-ibm made their first contribution in https://github.com/huggingface/trl/pull/3394
* @yao-matrix made their first contribution in https://github.com/huggingface/trl/pull/3438
* @nikolai-kummer made their first contribution in https://github.com/huggingface/trl/pull/3489
* @wa008 made their first contribution in https://github.com/huggingface/trl/pull/3454
* @teilomillet made their first contribution in https://github.com/huggingface/trl/pull/3459
* @re-imagined made their first contribution in https://github.com/huggingface/trl/pull/3324

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.17.0...v0.18.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.18.0)

---

## v0.17.0: v0.17.0
**Published:** 2025-04-25

## Major and breaking

The TRL v0.17 release introduces three major changes that, together, enable significantly faster generation performance in GRPO—up to 10x faster in some configurations.

![autonlp-08](https://github.com/user-attachments/assets/d7fa1362-4db5-43aa-a927-13ccce7fc588)

These three changes are:

- Data parallelism (DP) for the vLLM server
- A new GRPO training strategy that generates once per effective batch
- Support for the V1 engine in vLLM

Below, we provide a summary of these changes and how to use them.

### ⚡ Up to 4x faster: Data Parallel for vLLM server

The TRL vLLM server now supports data parallelism (DP), enabling significantly faster generation speeds—especially for smaller models. This new feature can be used by adding the `--data_parallel_size N` argument when launching the vLLM server.

```bash
trl vllm-serve --model Qwen/Qwen2.5-14B-Instruct --tensor_parallel_size 2 --data_parallel_size 2
```

by @qgallouedec in https://github.com/huggingface/trl/pull/3310

### * ☝️ [GRPO] Generate once per effective batch

Previously, GRPO made one generation request per global batch. The global batch is the total of all local batches, without accounting for gradient accumulation. In other words, if the gradient accumulation step was 8, GRPO would make 8 generation requests per training step.

Now, GRPO groups these global batches into a single "effective batch" and makes only one generation request per effective batch. Since vLLM applies optimizations that are especially effective for large batches, this new approach leads to significantly faster training overall.

No changes are required in the training script, as this is handled internally by the GRPO trainer.

![Untitled-2025-04-08-0623-2](https://github.com/user-attachments/assets/0b466436-5db8-4fac-b348-32daf90ce4c4)

by @qgallouedec in https://github.com/huggingface/trl/pull/3283

### ⏱️ Fix vLLM server to support V1 Engine

vLLM provides two versions of its engine (V0 and V1), and V1 is significantly faster. This version is now supported by TRL and requires vLLM version 0.8.3 or higher.

by @I-l-l-I in https://github.com/huggingface/trl/pull/3276

### 👎 [GRPO] Adds option to disable dropout

Disabling dropout has shown to stabilize training. You can now disable dropout in GRPO by setting the `disable_dropout` argument to `False` in the GRPO config.

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., disable_dropout=True)
```

by @edbeeching in https://github.com/huggingface/trl/pull/3234

### 🩺 Dr. GRPO loss

GRPO now supports the various losses proposed in the recent literature, including the Dr. GRPO loss. The loss type can be set in the GRPO config:

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., loss_type="dr_grpo")
```

![](https://github.com/user-attachments/assets/e52d1dd8-80ae-4c23-8006-b1c606c1fdf0)

by @qgallouedec in https://github.com/huggingface/trl/pull/3256

### 🎲 [GRPO] Make training dataset shuffle optional

The GRPO trainer now has an option to disable shuffling of the training dataset. This is useful for curriculum learning, where the order of the training data is important.

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., shuffle_dataset=False)
```

by @LeonEricsson in https://github.com/huggingface/trl/pull/3334

### ☕ Overlong-filtering for GRPO

Overlong filtering has been shown to significantly stabilize learning and improve performance. You can now use it in TRL!

It simply consists in masking the loss of truncated samples

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., mask_truncated_completions=True)
```

![Untitled-2025-04-08-0623](https://github.com/user-attachments/assets/e1dd664f-bca4-4e62-8802-0dd8772414ea)

by @shirinyamani in https://github.com/huggingface/trl/pull/3248

### 🐯 Integrate Liger GRPO Loss to GRPO Trainer

Liger allows to significantly reduce the memory peak of the loss computation. You can now use it in TRL with the `use_liger_loss` argument in the GRPO config:

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_liger_loss=True)
```

by @shivam15s in https://github.com/huggingface/trl/pull/3184

## Bug fixes

* Fix: Multi gpu hang for ORPO and CPO Trainer by @NanoCode012 in https://github.com/huggingface/trl/pull/3069
* 📊 Fix `clip_ratio` logging and better document logged values by @qgallouedec in https://github.com/huggingface/trl/pull/3145
* ⏯️ Fix: handle None inputs when resuming GRPO Trainer from checkpoint by @PenutChen in https://github.com/huggingface/trl/pull/3148
* 📎 Fix is_clipped to compute the effective clip_ratio by @pandong2011 in https://github.com/huggingface/trl/pull/3175
* 😷 Fix SFT masking EOS when equal to PAD by @qgallouedec in https://github.com/huggingface/trl/pull/3200
* ⏯️ Fix logging when resuming from checkpoint GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/3185
* 💠 Fix multi-gpu padding free by @qgallouedec in https://github.com/huggingface/trl/pull/3245
* 🕷 Fix online DPO crash when model is a DataParallel object by @wilrop in https://github.com/huggingface/trl/pull/3225
* 🏁 Fix adding special tokens in SFT by @qgallouedec in https://github.com/huggingface/trl/pull/3328
* 🍡 Fix using reward model and DeepSpeed ZeRO 3 by @qgallouedec in https://github.com/huggingface/trl/pull/3326

## What's Changed

* Fix: Multi gpu hang for ORPO and CPO Trainer by @NanoCode012 in https://github.com/huggingface/trl/pull/3069
* 📊 Fix `clip_ratio` logging and better document logged values by @qgallouedec in https://github.com/huggingface/trl/pull/3145
* BCOTrainer version upgrade fixes by @claralp in https://github.com/huggingface/trl/pull/2867
* 🐇 [Research] Layer Skip SFT  by @ariG23498 in https://github.com/huggingface/trl/pull/3111
* 🤝 Align GRPO equation doc with the implementation by @qgallouedec in https://github.com/huggingface/trl/pull/3151
* Enable number of printed completions to be set by @lewtun in https://github.com/huggingface/trl/pull/3149
* 🩹 Fix CI by @qgallouedec in https://github.com/huggingface/trl/pull/3155
* ⚰️ Remove deprecated by @qgallouedec in https://github.com/huggingface/trl/pull/3153
* 🔫 Disable triggering CI when PR is draft by @qgallouedec in https://github.com/huggingface/trl/pull/3154
* 👨‍🍳 vLLM serve: destroy process group on exit and pass `worker_cls` as string by @qgallouedec in https://github.com/huggingface/trl/pull/3159
* 💰 Richer rich table - log all the rewards by @qgallouedec in https://github.com/huggingface/trl/pull/3156
* 💎 Gemma 3 VLM SFT example script for single-image and multi-image by @sergiopaniego in https://github.com/huggingface/trl/pull/3131
* [Liger] Liger KTO support by @vaibhavjindal in https://github.com/huggingface/trl/pull/2812
* 🏃 Migrate CI to self-hosted runners by @qgallouedec in https://github.com/huggingface/trl/pull/3174
* ❤️‍🩹 [CI] fix transformers dev CI failure by @kashif in https://github.com/huggingface/trl/pull/3176
* ⏯️ Fix: handle None inputs when resuming GRPO Trainer from checkpoint by @PenutChen in https://github.com/huggingface/trl/pull/3148
* 📎 Fix is_clipped to compute the effective clip_ratio by @pandong2011 in https://github.com/huggingface/trl/pull/3175
* Fix breaking typo for flash_attention reducing_memory_usage.md by @burtenshaw in https://github.com/huggingface/trl/pull/3190
* Show unique prompts in GRPO WandB tables by @lewtun in https://github.com/huggingface/trl/pull/3191
* 🐗 [CI] Fix trufflehog false positives by @lewtun in https://github.com/huggingface/trl/pull/3192
* [GRPO] Improve completion length logging by @edbeeching in https://github.com/huggingface/trl/pull/3188
* 😷 Fix SFT masking EOS when equal to PAD by @qgallouedec in https://github.com/huggingface/trl/pull/3200
* 🗝️ Fix type hint in vLLM client by @qgallouedec in https://github.com/huggingface/trl/pull/3205
* 📚 Accumulate completions for logging by @lewtun in https://github.com/huggingface/trl/pull/3217
* Group completion metrics by common prefix by @lewtun in https://github.com/huggingface/trl/pull/3212
* 🐯 Integrate Liger GRPO Loss to GRPO Trainer by @shivam15s in https://github.com/huggingface/trl/pull/3184
* Update ruff to 11.3 and base Python version to 3.9 by @cyyever in https://github.com/huggingface/trl/pull/3230
* ⏯️ Fix logging when resuming from checkpoint GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/3185
* 📢 Improve GRPO trainer error message for invalid num_generations by @AliBakly in https://github.com/huggingface/trl/pull/3199
* 🎀  Simplify logging text by @qgallouedec in https://github.com/huggingface/trl/pull/3219
* 🌊 Add error for iterable datasets in GRPOTrainer by @qgallouedec in https://github.com/huggingface/trl/pull/3216
* ⏳ PPOTrainer: fix progress bar for num_mini_batches > 1 by @dawidm in https://github.com/huggingface/trl/pull/2531
* ☑ Update PULL_REQUEST_TEMPLATE.md by @qgallouedec in https://github.com/huggingface/trl/pull/3241
* 🔭 Add support for better KL estimator (k3) in PPOTrainer by @AMindToThink in https://github.com/huggingface/trl/pull/3240
* 🏃 Fix and make CI faster by @qgallouedec in https://github.com/huggingface/trl/pull/3160
* 🗑️ Deprecate `ConstantLengthDataset` by @qgallouedec in https://github.com/huggingface/trl/pull/3242
* 📦 [SFT] Deprecate batched `formatting_func` by @YeFD in https://github.com/huggingface/trl/pull/3147
* 💠 Fix multi-gpu padding free by @qgallouedec in https://github.com/huggingface/trl/pull/3245
* ☕ Overlong-filtering for GRPO by @shirinyamani in https://github.com/huggingface/trl/pull/3248
* 📜 Fix license and copyrights by @qgallouedec in https://github.com/huggingface/trl/pull/3264
* ⛏️ Add cli dict parsing for grpo_config by @Tavish9 in https://github.com/huggingface/trl/pull/3082
* 🐯 `is_liger_kernel_available` with min version by @qgallouedec in https://github.com/huggingface/trl/pull/3266
* 🕷 Fix online DPO crash when model is a DataParallel object by @wilrop in https://github.com/huggingface/trl/pull/3225
* 👎 [GRPO] Adds option to disable dropout by @edbeeching in https://github.com/huggingface/trl/pull/3234
* 🚧 Temporarily restrict diffusers to <0.33.0 due to ftfy optional dep issue breaking doc builds by @qgallouedec in https://github.com/huggingface/trl/pull/3273
* ♾️ [CI] Remove `test_raise_error_not_causallm` by @qgallouedec in https://github.com/huggingface/trl/pull/3265
* 🩺 Dr. GRPO loss by @qgallouedec in https://github.com/huggingface/trl/pull/3256
* 🔗 Fix Dr. GRPO paper link by @qgallouedec in https://github.com/huggingface/trl/pull/3275
* Add Fine-tuning a Multimodal Model Using SFT (Single or Multi-Image Dataset) guide to docs by @sergiopaniego in https://github.com/huggingface/trl/pull/3235
* 🕊️ Un-restrict diffusers by @qgallouedec in https://github.com/huggingface/trl/pull/3274
* 🦾 Test vLLM client-server by @qgallouedec in https://github.com/huggingface/trl/pull/3277
* ⏱️ Fix vLLM server to support V1 Engine by @I-l-l-I in https://github.com/huggingface/trl/pull/3276
* Expose EOS token in SFTConfig by @lewtun in https://github.com/huggingface/trl/pull/3299
* 🏷️ Fixed naming error in output_dir for Gemma 3 VLM script by @sergiopaniego in https://github.com/huggingface/trl/pull/3297
* 🧗 Add Ascend NPU support for vLLM server by @ji-huazhong in https://github.com/huggingface/trl/pull/3286
* 🅾️ Fixes typo in SFTTrainer by @taras-sereda in https://github.com/huggingface/trl/pull/3282
* [GRPO] Add metrics for low and high clipped token probabilities by @lewtun in https://github.com/huggingface/trl/pull/3289
* ☝️ [GRPO] Generate once per effective batch by @qgallouedec in https://github.com/huggingface/trl/pull/3283
* 🎲 [GRPO] Make training dataset shuffle optional by @LeonEricsson in https://github.com/huggingface/trl/pull/3334
* 🙋 Add Optional Eager Execution Mode for vLLM Serving by @ucalyptus in https://github.com/huggingface/trl/pull/3335
* Fix typo in text_environments.md by @sunjin-k in https://github.com/huggingface/trl/pull/3305
* ✅ [doc] Update sft_trainer.md in table x->✓ by @HERIUN in https://github.com/huggingface/trl/pull/3313
* 🧸 Fix unset tokenizer pad_token by @LeonEricsson in https://github.com/huggingface/trl/pull/3290
* 💡 Fix type hint in `_generate_and_score_completions` by @syt-nju in https://github.com/huggingface/trl/pull/3336
* 🦄 Add optional uvicorn log level for vLLM serve by @I-l-l-I in https://github.com/huggingface/trl/pull/3338
* [CPO] Check that `max_prompt_length < max_length` by @LeonEricsson in https://github.com/huggingface/trl/pull/3341
* 🏁 Fix adding special tokens in SFT by @qgallouedec in https://github.com/huggingface/trl/pull/3328
* Define default chat template for SFT by @lewtun in https://github.com/huggingface/trl/pull/3309
* 🍡 Fix using reward model and DeepSpeed ZeRO 3 by @qgallouedec in https://github.com/huggingface/trl/pull/3326
* ⚡ Up to 4x faster: Data Parallel for vLLM server by @qgallouedec in https://github.com/huggingface/trl/pull/3310
* Release: v0.17 by @qgallouedec in https://github.com/huggingface/trl/pull/3356

## New Contributors
* @NanoCode012 made their first contribution in https://github.com/huggingface/trl/pull/3069
* @ariG23498 made their first contribution in https://github.com/huggingface/trl/pull/3111
* @PenutChen made their first contribution in https://github.com/huggingface/trl/pull/3148
* @pandong2011 made their first contribution in https://github.com/huggingface/trl/pull/3175
* @shivam15s made their first contribution in https://github.com/huggingface/trl/pull/3184
* @cyyever made their first contribution in https://github.com/huggingface/trl/pull/3230
* @AMindToThink made their first contribution in https://github.com/huggingface/trl/pull/3240
* @YeFD made their first contribution in https://github.com/huggingface/trl/pull/3147
* @Tavish9 made their first contribution in https://github.com/huggingface/trl/pull/3082
* @wilrop made their first contribution in https://github.com/huggingface/trl/pull/3225
* @I-l-l-I made their first contribution in https://github.com/huggingface/trl/pull/3276
* @taras-sereda made their first contribution in https://github.com/huggingface/trl/pull/3282
* @LeonEricsson made their first contribution in https://github.com/huggingface/trl/pull/3334
* @ucalyptus made their first contribution in https://github.com/huggingface/trl/pull/3335
* @sunjin-k made their first contribution in https://github.com/huggingface/trl/pull/3305
* @HERIUN made their first contribution in https://github.com/huggingface/trl/pull/3313
* @syt-nju made their first contribution in https://github.com/huggingface/trl/pull/3336

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.16.0...v0.17.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.17.0)

---

## v0.16.1: v0.16.1
**Published:** 2025-04-04

## What's Changed

* 😷 Fix SFT masking EOS when equal to PAD by @qgallouedec in https://github.com/huggingface/trl/pull/3200
* 📉 Add `learning_rate` argument to `_maybe_log_save_evaluate` by @qgallouedec in https://github.com/huggingface/trl/pull/3206

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.16.0...v0.16.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.16.1)

---

## v0.16.0: v0.16.0
**Published:** 2025-03-22

## Major and breaking

### 🚀 Scaling GRPO to 70B+ Models and Multi-Node Training with vLLM Server & NCCL Communication

Previously, vLLM could only be used by dedicating a single GPU, preventing both the scalability benefits of vLLM and multi-node training. This limitation has now been removed!

GRPO can now scale efficiently with models exceeding **70B parameters**, supporting **multi-node training** with **super-fast performance**.

![autonlp-08-16](https://github.com/user-attachments/assets/6bcadb99-d1cf-42ca-8233-4af83f1ba6e5)

To take advantage of this, simply launch a vLLM server using the following command:  

```bash
trl vllm-serve --model <model_name> --tensor_parallel_size <tp_size>
```  

Then, start GRPO training with `use_vllm=True`.  

Below is a comparison of GRPO throughput with and without vLLM, across different TP values and model sizes. 

@binary-husky and @qgallouedec in https://github.com/huggingface/trl/pull/3094

### 🐦‍🔥 6x faster GRPO with multi-step optimization

This release introduces the multi-step trick, which allows for the reuse of generated data across multiple steps, speeding up the training process.

To support this, we've implemented importance sampling and clipping logic. This enhancement should lead to significant improvements in training speed.

<img width="1097" alt="Screenshot 2025-03-23 at 14 52 28" src="https://github.com/user-attachments/assets/8f1ee339-63c5-43cf-9b0f-5395432513ae" />

To use it, simply set `num_iterations` to a value greater than 1.

```python
training_args = GRPOConfig(..., num_iterations=4)
```

![](https://github.com/user-attachments/assets/1fcbd2fb-580b-42df-a7ca-611ba0962375)

by @qgallouedec in https://github.com/huggingface/trl/pull/2899

### 🌍 Use global normalization in GRPO

As demonstrated in [Dr GRPO](https://github.com/sail-sg/understand-r1-zero), sequence-level normalization can introduce a response level length bias.

![GmsxibSaoAAevq_](https://github.com/user-attachments/assets/ec85edd8-ffaf-460e-a53e-5b417ae9c829)

To address this, we have now switched to normalizing the loss and by the total number of tokens in the batch, ensuring more consistent and unbiased training.

```diff
- loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
+ loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
```


by @edbeeching in https://github.com/huggingface/trl/pull/2881

### ⚖️ Add option not to scale rewards

As demonstrated in [Dr GRPO](https://github.com/sail-sg/understand-r1-zero), scaling rewards can introduce a question-level difficulty bias. To address this, we have now added an option to disable reward scaling in GRPO.

```python
training_args = GRPOConfig(..., scale_rewards=False)
```

```diff
  advantages = rewards - mean_grouped_rewards
- advantages = advantages / std_grouped_rewards
+ if self.args.scale_rewards:
+     advantages = advantages / std_grouped_rewards
```

it's likely that we'll make this (`scale_rewards=False`) the default behavior in the future.

by @qgallouedec in https://github.com/huggingface/trl/pull/3135

### 🤸‍♀️ Domain-specific rewards in GRPO
 
When optimizing across multiple domains, not all reward functions are relevant for every sample. For example, a math verifier's reward does not apply to grammar samples, and a grammar verifier's reward does not apply to math samples.

It is now possible to return `None` for rewards that do not make sense for a given sample. For instance, when the domain is specified in a column like `domain`, you can implement it as follows:

![GmcKjsgaQAAawqj](https://github.com/user-attachments/assets/9c9e9ed0-472e-4083-aea1-573e1366ab74)

```python
def math_reward(completions, domain, **kwargs):
    rewards = []
    for completion, dom in zip(completions, domain):
        if dom == "math":
            rewards.append(verify(completion))
        else:
            rewards.append(None)
    return rewards
```

This allows for more domain-specific reward handling, ensuring that irrelevant rewards are ignored and don’t interfere with optimization.

by @shirinyamani in https://github.com/huggingface/trl/pull/3079

### 🍃 Do not load reference model when `beta == 0.0`

It has been observed that not minimizing the KL divergence between the trained model and the reference model can still yield good results, while significantly reducing memory usage and compute. This is because there is no need to store the reference model in memory or perform a forward pass for it.

When `beta` is set to `0.0`, the reference model is not loaded, and the KL divergence is not computed, leading to savings in both time and memory.

```python
training_args = GRPOConfig(..., beta=0.0)
```

by @ingambe in https://github.com/huggingface/trl/pull/2806

### 🕊️ Padding-free for SFT

Padding-free batching is an alternative approach to packing for reducing memory usage. In this method, a batch is first sampled and then flattened into a single sequence, avoiding padding. Unlike packing, which can result in incomplete sequences by combining parts of different samples, padding-free batching ensures that all sequences remain complete and intact.

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/padding-free.png)

To enable padding-free batching in SFT, simply set `padding_free=True` in the `SFTConfig`, and make sure to use `flash_attention2` as the attention implementation.

```python
training_args = SFTConfig(..., padding_free=True, model_init_kwargs={"attn_implementation": "flash_attention2"})
```

by @qgallouedec in https://github.com/huggingface/trl/pull/3076

### 🎬 Clip Higher for Better Exploration

As outlined in the [DAPO paper](https://huggingface.co/papers/2503.14476), increasing the upper bound epsilon leads to higher entropy during generation, promoting better exploration. To enable this, we’ve added support for adjusting the upper bound epsilon directly in the default GRPO trainer.

![](https://github.com/user-attachments/assets/499168f9-f15a-403e-a114-4603e45425fa)

```python
training_args = GRPOConfig(epsilon_high=0.28)
```

by @shirinyamani in https://github.com/huggingface/trl/pull/3118

## Bug fixes

* 🧶 [GRPO][vLLM + LoRA] Move unmerge of PEFT model after weight loading by @XZ-X in https://github.com/huggingface/trl/pull/2873
* 🪂 Don't gather logits in SFT to avoid hanging by @qgallouedec in https://github.com/huggingface/trl/pull/2890
* ♻️ Fix caching in SFT by @qgallouedec in https://github.com/huggingface/trl/pull/2945
* 🐯 Fix LigerKernel for SFTTrainer by @lewtun @kashif and @qgallouedec in https://github.com/huggingface/trl/pull/2874, https://github.com/huggingface/trl/pull/2940 and https://github.com/huggingface/trl/pull/2949
* 🫔 [GRPO] Pass wrapped model to `unwrap_model_for_generation` for DeepSpeed Stage-3 compatibility by @kiddj in https://github.com/huggingface/trl/pull/2871
* 🛣️ `inference_mode` to `no_grad` when computing `old_per_token_logps` by @qgallouedec in https://github.com/huggingface/trl/pull/2987
* 🏊 [SFT] Compatibility with padding free and iterable dataset by @qgallouedec in https://github.com/huggingface/trl/pull/3053
* Fixing JSD loss computation in GKDTrainer as per definition by @abhigoyal1997 in https://github.com/huggingface/trl/pull/3043

## Minor

* 💬 Add `maybe_convert_to_chatml` map for conversational datasets in SFT by @kashif in https://github.com/huggingface/trl/pull/2862
* 🍟 [SFT] Handles the dataset if it has been preprocessed by @BenasdTW and @DanFosing in https://github.com/huggingface/trl/pull/2863 and https://github.com/huggingface/trl/pull/2939
* ✨ Add vLLM guided decoding support to GRPO Trainer by @kldzj in https://github.com/huggingface/trl/pull/2811
* 🩳 `max_seq_length` to `max_length` by @qgallouedec in https://github.com/huggingface/trl/pull/2895 and https://github.com/huggingface/trl/pull/2947
* Optimize vllm num_generations by @edbeeching in https://github.com/huggingface/trl/pull/2855
* 📍 [GRPO] add gradient_checkpointing by @kashif in https://github.com/huggingface/trl/pull/2848
* 🪪 Adds profiling decorators for GRPOTrainer by @edbeeching in https://github.com/huggingface/trl/pull/2889 and https://github.com/huggingface/trl/pull/2975
* 🐈 Bye bye chat by @qgallouedec in https://github.com/huggingface/trl/pull/2934
* 📇 GRPO: print completions to console and update docs by @nopepper in https://github.com/huggingface/trl/pull/2951
* 👧🏽 Adding DoRA support to model config by @nbasyl in https://github.com/huggingface/trl/pull/2974
* 🧗 Add GRPO Trainer support for third-party accelerators by @ji-huazhong in https://github.com/huggingface/trl/pull/2836
* 🪙 [SFT] Log `num_tokens` and some logging fixes by @qgallouedec in https://github.com/huggingface/trl/pull/3006
* 🌡️ Fix temperature inconsistency in GRPO trainer by @Aladoro in https://github.com/huggingface/trl/pull/3029
* ⛔ Add EOS token to processed input in SFT by @qgallouedec in https://github.com/huggingface/trl/pull/3091
* ⚡ Pack 300 times faster, truncate 100 times faster by @mariosasko in https://github.com/huggingface/trl/pull/3009


## What's Changed
* [SFT] fix check for AutoLigerKernelForCausalLM by @kashif in https://github.com/huggingface/trl/pull/2874
* 🆙 Bump vLLM min version to 0.7.2 by @edbeeching in https://github.com/huggingface/trl/pull/2860
* [GRPO] Fix loss normalization by @edbeeching in https://github.com/huggingface/trl/pull/2881
* 💬 Add `maybe_convert_to_chatml` map for conversational datasets in SFT by @kashif in https://github.com/huggingface/trl/pull/2862
* 🧶 [GRPO][vLLM + LoRA] Move unmerge of PEFT model after weight loading by @XZ-X in https://github.com/huggingface/trl/pull/2873
* 🍟 [SFT] Handles the dataset if it has been preprocessed by @BenasdTW in https://github.com/huggingface/trl/pull/2863
* Optimize vllm num_generations by @edbeeching in https://github.com/huggingface/trl/pull/2855
* 🪂 Don't gather logits in SFT to avoid hanging by @qgallouedec in https://github.com/huggingface/trl/pull/2890
* ✨ Add vLLM guided decoding support to GRPO Trainer by @kldzj in https://github.com/huggingface/trl/pull/2811
* ⚰️ Remove deprecated by @qgallouedec in https://github.com/huggingface/trl/pull/2894
* 🩳 `max_seq_length` to `max_length` by @qgallouedec in https://github.com/huggingface/trl/pull/2895
* 🍃 GRPO - Do not load reference model when beta == 0 by @ingambe in https://github.com/huggingface/trl/pull/2806
* 📍 [GRPO] add gradient_checkpointing by @kashif in https://github.com/huggingface/trl/pull/2848
* 🪪 Adds profiling decorators for GRPOTrainer by @edbeeching in https://github.com/huggingface/trl/pull/2889
* 🐦‍🔥 6x faster GRPO with multi-step optimization by @qgallouedec in https://github.com/huggingface/trl/pull/2899
* 🔹 Fix: Miscalculated mask shape in comments by @linkedlist771 in https://github.com/huggingface/trl/pull/2925
* 🤖 Style bot by @qgallouedec in https://github.com/huggingface/trl/pull/2935
* 🧼 Upgrade ruff by @qgallouedec in https://github.com/huggingface/trl/pull/2938
* 🐈 Bye bye chat by @qgallouedec in https://github.com/huggingface/trl/pull/2934
* ♻️ Fix caching in SFT by @qgallouedec in https://github.com/huggingface/trl/pull/2945
* 📋 Add vLLM version to environment printout by @qgallouedec in https://github.com/huggingface/trl/pull/2946
* ☠️ Update `max_seq_length` to `max_length` in `SFTConfig` by @qgallouedec in https://github.com/huggingface/trl/pull/2947
* 🐯 Fix LigerKernel for SFTTrainer by @lewtun in https://github.com/huggingface/trl/pull/2940
* ✋ Prevent applying the chat template to tokenized datasets by @DanFosing in https://github.com/huggingface/trl/pull/2939
* 📇 GRPO: print completions to console and update docs by @nopepper in https://github.com/huggingface/trl/pull/2951
* ↩️ Fix typo in TextEnvironment init param, should be `max_tool_response` by @shenxiangzhuang in https://github.com/huggingface/trl/pull/2921
* 🗿 Updated DPO default values for alpha and tau by @Ishan-Kumar2 in https://github.com/huggingface/trl/pull/2918
* 📌 Pin liger-kernel and vLLM by @qgallouedec in https://github.com/huggingface/trl/pull/2952
* ⏪ Parameterize `enable_prefix_caching` by @ji-huazhong in https://github.com/huggingface/trl/pull/2900
* 🔢 Fix GRPO doc about `num_iterations` by @qgallouedec in https://github.com/huggingface/trl/pull/2966
* Update grpo_trainer.py by @tpoisonooo in https://github.com/huggingface/trl/pull/2973
* 👧🏽 Adding DoRA support to model config by @nbasyl in https://github.com/huggingface/trl/pull/2974
* 🧗 Add GRPO Trainer support for third-party accelerators by @ji-huazhong in https://github.com/huggingface/trl/pull/2836
* 🕸 Add distributing training guide by @qgallouedec in https://github.com/huggingface/trl/pull/2956
* 👂 Update learning rate doc in `KTOConfig` by @sileod in https://github.com/huggingface/trl/pull/2912
* 🌌 Fix logits computation in trainer prediction step by @logicaltrojan in https://github.com/huggingface/trl/pull/2969
* 🪪 Adds a more fine-grained profiling context by @edbeeching in https://github.com/huggingface/trl/pull/2975
* 🧬 Fix typo in grpo_trainer.py by @congchan in https://github.com/huggingface/trl/pull/2988
* 📜 Update README and doc index by @qgallouedec in https://github.com/huggingface/trl/pull/2986
* 📑 Fix logged metrics for KTO by @vaibhavjindal in https://github.com/huggingface/trl/pull/2982
* ⚰️ Deprecate liger-kernel by @qgallouedec in https://github.com/huggingface/trl/pull/2949
* 🔍 Update GRPO config documentation for beta parameter stability by @nopepper in https://github.com/huggingface/trl/pull/2992
* 🫔 [GRPO] Pass wrapped model to `unwrap_model_for_generation` for DeepSpeed Stage-3 compatibility by @kiddj in https://github.com/huggingface/trl/pull/2871
* 🛣️ `inference_mode` to `no_grad` when computing `old_per_token_logps` by @qgallouedec in https://github.com/huggingface/trl/pull/2987
* 🚀 DeepSpeed integration documentation by @qgallouedec in https://github.com/huggingface/trl/pull/2993
* Update pr_style_bot.yml by @qgallouedec in https://github.com/huggingface/trl/pull/3003
* 🪙 [SFT] Log `num_tokens` and some logging fixes by @qgallouedec in https://github.com/huggingface/trl/pull/3006
* Improve ci by @paulinebm in https://github.com/huggingface/trl/pull/3007
* ✌️Remove double compute of sum in SFTTrainer by @lexasub in https://github.com/huggingface/trl/pull/3001
* 📚 Update customization and distributing training documentation by @qgallouedec in https://github.com/huggingface/trl/pull/2991
* 🌍 Use global normalization for KL logging (to match normalization for loss) by @tchang1997 in https://github.com/huggingface/trl/pull/3004
* 🗜️ Loosened tokenizer type hint on `apply_chat_template` by @jamesbraza in https://github.com/huggingface/trl/pull/3005
* 🎲 Add support for additional generation kwargs in GRPO Trainer by @nopepper in https://github.com/huggingface/trl/pull/2989
* 🚀  Supporting `deepspeed>=0.16.4`'s rename by @jamesbraza in https://github.com/huggingface/trl/pull/2963
* 🌡️ Fix temperature inconsistency in GRPO trainer by @Aladoro in https://github.com/huggingface/trl/pull/3029
* 🏁 Passing custom BOS/EOS token to `GPROTrainer.generation_config` by @jamesbraza in https://github.com/huggingface/trl/pull/3046
* 💠 Fixing `SFTTrainer.compute_loss` crash with `accelerate` by @jamesbraza in https://github.com/huggingface/trl/pull/3048
* 👯 [GRPO] Relax the assumption that prompts are unique within a batch by @qgallouedec in https://github.com/huggingface/trl/pull/3052
* [GRPO] use argument names with processing_class by @kashif in https://github.com/huggingface/trl/pull/3062
* 🦥  Fixed `SFTTrainer.compute_loss` hang from #3048's PR comments by @jamesbraza in https://github.com/huggingface/trl/pull/3056
* 🏊 [SFT] Compatibility with padding free and iterable dataset by @qgallouedec in https://github.com/huggingface/trl/pull/3053
* Fixing JSD loss computation in GKDTrainer as per definition by @abhigoyal1997 in https://github.com/huggingface/trl/pull/3043
* 🎭 Minor spelling fix in documentation (caracteres -> characters) by @esnible in https://github.com/huggingface/trl/pull/3074
* 💎 Gemma 3 SFT example on Codeforces dataset by @qgallouedec in https://github.com/huggingface/trl/pull/3070
* 🫣 [GRPO] add cache_implementation option in GRPO by @kashif in https://github.com/huggingface/trl/pull/3075
* ⛔ Add EOS token to processed input in SFT by @qgallouedec in https://github.com/huggingface/trl/pull/3091
* 🕊️ Padding-free for SFT by @qgallouedec in https://github.com/huggingface/trl/pull/3076
* add "_prepare_fsdp" for DPOTrainer by @faaany in https://github.com/huggingface/trl/pull/2539
* Use main process for dataset.map by @lewtun in https://github.com/huggingface/trl/pull/3106
* Flexible_reward by @shirinyamani in https://github.com/huggingface/trl/pull/3079
* 🎬 Clip higher by @shirinyamani in https://github.com/huggingface/trl/pull/3118
* 🚀 Scaling GRPO to 70B+ Models and Multi-Node Training with vLLM Server & NCCL Communication by @binary-husky in https://github.com/huggingface/trl/pull/3094
* ⚡ Pack 300 times faster, truncate 100 times faster by @mariosasko in https://github.com/huggingface/trl/pull/3009
* ☎️ Documentation for disable gathering of model weights for generation in DeepSpeed ZeRO-3 by @qgallouedec in https://github.com/huggingface/trl/pull/3136
* ⚖️ Add option not to scale rewards (Dr. GRPO) by @qgallouedec in https://github.com/huggingface/trl/pull/3135
* Release: v0.16 by @qgallouedec in https://github.com/huggingface/trl/pull/3137

## New Contributors
* @XZ-X made their first contribution in https://github.com/huggingface/trl/pull/2873
* @BenasdTW made their first contribution in https://github.com/huggingface/trl/pull/2863
* @kldzj made their first contribution in https://github.com/huggingface/trl/pull/2811
* @ingambe made their first contribution in https://github.com/huggingface/trl/pull/2806
* @linkedlist771 made their first contribution in https://github.com/huggingface/trl/pull/2925
* @DanFosing made their first contribution in https://github.com/huggingface/trl/pull/2939
* @nopepper made their first contribution in https://github.com/huggingface/trl/pull/2951
* @shenxiangzhuang made their first contribution in https://github.com/huggingface/trl/pull/2921
* @Ishan-Kumar2 made their first contribution in https://github.com/huggingface/trl/pull/2918
* @tpoisonooo made their first contribution in https://github.com/huggingface/trl/pull/2973
* @nbasyl made their first contribution in https://github.com/huggingface/trl/pull/2974
* @sileod made their first contribution in https://github.com/huggingface/trl/pull/2912
* @logicaltrojan made their first contribution in https://github.com/huggingface/trl/pull/2969
* @congchan made their first contribution in https://github.com/huggingface/trl/pull/2988
* @vaibhavjindal made their first contribution in https://github.com/huggingface/trl/pull/2982
* @kiddj made their first contribution in https://github.com/huggingface/trl/pull/2871
* @paulinebm made their first contribution in https://github.com/huggingface/trl/pull/3007
* @lexasub made their first contribution in https://github.com/huggingface/trl/pull/3001
* @tchang1997 made their first contribution in https://github.com/huggingface/trl/pull/3004
* @Aladoro made their first contribution in https://github.com/huggingface/trl/pull/3029
* @abhigoyal1997 made their first contribution in https://github.com/huggingface/trl/pull/3043
* @esnible made their first contribution in https://github.com/huggingface/trl/pull/3074
* @mariosasko made their first contribution in https://github.com/huggingface/trl/pull/3009

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.15.0...v0.16.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.16.0)

---

## v0.15.2: 
**Published:** 2025-02-25

# What changed

- ♻️ Fix caching in SFT by @qgallouedec in #2945
- 🐯 Fix LigerKernel for SFTTrainer by @lewtun in #2940
- 📌 Pin liger-kernel and vLLM by @qgallouedec in #2952

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.15.1...v0.15.2


[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.15.2)

---

## v0.15.1: v0.15.1
**Published:** 2025-02-18

## What's Changed

- 💬 Add `maybe_convert_to_chatml`map for conversational datasets by @kashif in SFT in #2862
- [SFT] fix check for AutoLigerKernelForCausalLM by @kashif in #2874
- 🍟 [SFT] Handles the dataset if it has been preprocessed by @BenasdTW in #2863
- 🧶 [GRPO][vLLM + LoRA] Move unmerge of PEFT model after weight loading by @XZ-X in #2873
- 🪂 Don't gather logits in SFT to avoid hanging by @qgallouedec in #2890
- Release: v0.15.1 by @qgallouedec

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.15.0...v0.15.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.15.1)

---

## v0.15.0: v0.15.0
**Published:** 2025-02-13

## Major and breaking changes

Coming soon

## What's Changed
* ⬆️ Bump dev version by @qgallouedec in https://github.com/huggingface/trl/pull/2689
* 📦 `trl.templates` in excluded packages by @qgallouedec in https://github.com/huggingface/trl/pull/2690
* 📖  Docs fix spelling issues by @nnsW3 in https://github.com/huggingface/trl/pull/2682
* 📄 Add GRPO batch size note in docs by @sdpkjc in https://github.com/huggingface/trl/pull/2672
* 🙈 Fixed typo in the GRPO documentation by @famouswizard in https://github.com/huggingface/trl/pull/2691
* docs: Fix broken "Good First Issue" link in CONTRIBUTING.md by @famouswizard in https://github.com/huggingface/trl/pull/2693
* 🧠 Fix typo in "understand" in ppo_trainer.md by @famouswizard in https://github.com/huggingface/trl/pull/2695
* ☠️ Remove deprecated by @qgallouedec in https://github.com/huggingface/trl/pull/2692
* 💡 Add "Mini-R1: Reproduce Deepseek R1 „aha moment“ a RL tutorial" by @qgallouedec in https://github.com/huggingface/trl/pull/2697
* 📋 Add eval loss logging during prediction in GRPO by @kashif in https://github.com/huggingface/trl/pull/2694
* fix: Fix typo in filename Update ultrafeedback.py by @brawncode in https://github.com/huggingface/trl/pull/2699
* 📖 Add GRPOTrainer to README.md by @burtenshaw in https://github.com/huggingface/trl/pull/2713
* Improve GRPO example by @lewtun in https://github.com/huggingface/trl/pull/2717
* 📖 Nit Fix in Documentation by @ParagEkbote in https://github.com/huggingface/trl/pull/2722
* 🏰 `num_logits_to_keep` to `logits_to_keep` by @qgallouedec in https://github.com/huggingface/trl/pull/2721
* 💰 Fix incorrect calculation in Olivia's baguette spending logic by @defiberrys in https://github.com/huggingface/trl/pull/2727
* fix: Fix typo in filename in ultrafeedback-prompt.py by @brawncode in https://github.com/huggingface/trl/pull/2716
* docs: Fix typos in alias descriptions by @defiberrys in https://github.com/huggingface/trl/pull/2729
* ⚠️ Fix Attention Masking in GRPO by @andyl98 in https://github.com/huggingface/trl/pull/2708
* 🔂 Use vLLM prefix caching for speedup by @winglian in https://github.com/huggingface/trl/pull/2757
* 💔 Decouple loss computing and generation in GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2762
* 📌  vLLM >= 0.7.1 for device fix by @ctjlewis in https://github.com/huggingface/trl/pull/2766
* 📐 Add vLLM dtype configuration for GRPO trainer by @joey00072 in https://github.com/huggingface/trl/pull/2738
* 📖 Clarification max len in Reward documentation by @ParagEkbote in https://github.com/huggingface/trl/pull/2740
* 🔎 Add missing script argument in PPO documentation by @JohnConnor123 in https://github.com/huggingface/trl/pull/2720
* 🤖 Properly unwrap torch.compile-ed models in GRPO by @winglian in https://github.com/huggingface/trl/pull/2750
* 🔁 🦈 Support iterative GRPO  by @shirinyamani in https://github.com/huggingface/trl/pull/2700
* 🚧 Add Optional ZeRO-3 Weight Gathering for GRPO in Sequence Generation by @SeungyounShin in https://github.com/huggingface/trl/pull/2667
* ↔️  GRPO: Set max_model_len when initializing vLLM instance by @mirceapricop in https://github.com/huggingface/trl/pull/2728
* 💡  GRPO vram-efficiency improvement; only compute relevant logprobs by @tyler-romero in https://github.com/huggingface/trl/pull/2773
* 🙃  Fix reward function in GRPO example by @junuMoon in https://github.com/huggingface/trl/pull/2777
* 💡 Add 'Post training an LLM for reasoning with GRPO in TRL' tutorial by @sergiopaniego in https://github.com/huggingface/trl/pull/2785
* 📉 Optimize GRPO memory usage by redefining `per_device_batch_size` as generations per device by @qgallouedec in https://github.com/huggingface/trl/pull/2776
* 🆚 Distinguish padding and eos when they differ by @binary-husky in https://github.com/huggingface/trl/pull/2793
* 🎯 [SFT] add token accuracy metric by @kashif in https://github.com/huggingface/trl/pull/2597
* 📠 Log completions for GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2772
* 🔬 SFT simplification by @qgallouedec in https://github.com/huggingface/trl/pull/2405
* ➖ Fix GRPO example in README by @qgallouedec in https://github.com/huggingface/trl/pull/2800
* ⛰️ Reduce peak vram consumption with efficient selective log_softmax by @tyler-romero in https://github.com/huggingface/trl/pull/2799
* fix: typos in documentation files by @maximevtush in https://github.com/huggingface/trl/pull/2804
* 📤 GRPO refactor loading the model weights to vllm by @winglian in https://github.com/huggingface/trl/pull/2817
* 🫘 Add `set_seed()` call in GRPO to ensure unique seed for each process by @qgallouedec in https://github.com/huggingface/trl/pull/2824
* ⚖️ Add reward weight in multi-reward settings for GRPO by @hesamsheikh in https://github.com/huggingface/trl/pull/2676
* 🙌 Share vLLM device with training when only 1 available by @qgallouedec in https://github.com/huggingface/trl/pull/2827
* 👴 Update `tokenizer` parameter to `processing_class` in tests by @qgallouedec in https://github.com/huggingface/trl/pull/2828
* 🥾  Allow bootstrap GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2829
* ⚡ Fix GRPO PEFT by @qgallouedec in https://github.com/huggingface/trl/pull/2725
* Fix PeftModel check when moving weights to vlllm by @edbeeching in https://github.com/huggingface/trl/pull/2850
* 🪆 Fix for Incorrect ValueError Handling in reward_weights in grpo_trainer.py by @loveychen in https://github.com/huggingface/trl/pull/2843
* 👨‍👩‍👧 GRPO + PEFT + vLLM by @winglian in https://github.com/huggingface/trl/pull/2818

## New Contributors
* @nnsW3 made their first contribution in https://github.com/huggingface/trl/pull/2682
* @sdpkjc made their first contribution in https://github.com/huggingface/trl/pull/2672
* @famouswizard made their first contribution in https://github.com/huggingface/trl/pull/2691
* @brawncode made their first contribution in https://github.com/huggingface/trl/pull/2699
* @ParagEkbote made their first contribution in https://github.com/huggingface/trl/pull/2722
* @defiberrys made their first contribution in https://github.com/huggingface/trl/pull/2727
* @ctjlewis made their first contribution in https://github.com/huggingface/trl/pull/2766
* @joey00072 made their first contribution in https://github.com/huggingface/trl/pull/2738
* @JohnConnor123 made their first contribution in https://github.com/huggingface/trl/pull/2720
* @shirinyamani made their first contribution in https://github.com/huggingface/trl/pull/2700
* @mirceapricop made their first contribution in https://github.com/huggingface/trl/pull/2728
* @tyler-romero made their first contribution in https://github.com/huggingface/trl/pull/2773
* @junuMoon made their first contribution in https://github.com/huggingface/trl/pull/2777
* @binary-husky made their first contribution in https://github.com/huggingface/trl/pull/2793
* @maximevtush made their first contribution in https://github.com/huggingface/trl/pull/2804
* @hesamsheikh made their first contribution in https://github.com/huggingface/trl/pull/2676
* @loveychen made their first contribution in https://github.com/huggingface/trl/pull/2843

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.9.6...v0.15.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.15.0)

---

## v0.14.0: v0.14.0
**Published:** 2025-01-29

## Major and breaking changes

### 👨‍👨‍👧‍👧 GRPO

by @qgallouedec in https://github.com/huggingface/trl/pull/2565

## What's Changed
* ⚰️ Remove deprecated by @qgallouedec in https://github.com/huggingface/trl/pull/2485
* 🗣️ Improve prose for smol course by @burtenshaw in https://github.com/huggingface/trl/pull/2487
* 🤩 Add SmolVLM tutorials to Community Tutorials page by @sergiopaniego in https://github.com/huggingface/trl/pull/2498
* 🏞️ Proper dataset for documentation images by @qgallouedec in https://github.com/huggingface/trl/pull/2499
* 🗂️ Reorganize documentation by @qgallouedec in https://github.com/huggingface/trl/pull/2483
* [ORPO] fix orpo chosen-nll loss by @kashif in https://github.com/huggingface/trl/pull/2502
* 🏚 Remove unused components by @qgallouedec in https://github.com/huggingface/trl/pull/2480
* Update community_tutorials.md by @qgallouedec in https://github.com/huggingface/trl/pull/2509
* ❎ Remove RLOO example test by @qgallouedec in https://github.com/huggingface/trl/pull/2513
* 👨‍🍳 Clarify DPO data preparation by @qgallouedec in https://github.com/huggingface/trl/pull/2512
* 💧 Generalize `disable_dropout` by @qgallouedec in https://github.com/huggingface/trl/pull/2511
* 👬 Rename collator `PreferenceCollator` to ` DataCollatorForPreference` by @qgallouedec in https://github.com/huggingface/trl/pull/2510
* 📦 Packing documentation by @qgallouedec in https://github.com/huggingface/trl/pull/2503
* ☄️ Update Comet  integration to include LogCompletionsCallback and Trainer.evaluation_loop() by @yaricom in https://github.com/huggingface/trl/pull/2501
* Remove graph breaks for torch.compile() in padding free branch in DataCollatorForCompletionOnlyLM by @Abhishek-TAMU in https://github.com/huggingface/trl/pull/2158
* 🚜 Use field in dataclasses by @qgallouedec in https://github.com/huggingface/trl/pull/2494
* ©️ Update copyrights year by @qgallouedec in https://github.com/huggingface/trl/pull/2547
* 🧑‍🤝‍🧑 Proper metrics gathering across ranks before logging by @zhc7 in https://github.com/huggingface/trl/pull/2474
* ✒️ Fix typo in `formatting_func`'s documentation in `ConstantLengthDataset` by @SamuelLarkin in https://github.com/huggingface/trl/pull/2549
* 🕊️ DPO padding free by @qgallouedec in https://github.com/huggingface/trl/pull/2520
* ℹ️ XPU support for DPO by @faaany in https://github.com/huggingface/trl/pull/2533
* 🔠 Fix SFT truncation documentation by @umbilnm in https://github.com/huggingface/trl/pull/2521
* ↩️ Revert ORPO loss changes by @kashif in https://github.com/huggingface/trl/pull/2527
* 🎴 Add readme for datasets by @August-murr in https://github.com/huggingface/trl/pull/2491
* 💔 Fix dataset type unpair conversion docs by @claralp in https://github.com/huggingface/trl/pull/2550
* [RLOO] Reinforce++ by @kashif in https://github.com/huggingface/trl/pull/2552
* 🏛️ Improve DPO configuration documentation structure by @qgallouedec in https://github.com/huggingface/trl/pull/2561
* ✨ Refine model card method docstring by @qgallouedec in https://github.com/huggingface/trl/pull/2566
* 🪄 Minor comment style modif by @qgallouedec in https://github.com/huggingface/trl/pull/2582
* 🏎️ vllm for Online DPO by @qgallouedec in https://github.com/huggingface/trl/pull/2558
* 🔖 Issues Auto-Labeller by @August-murr in https://github.com/huggingface/trl/pull/2542
* 🐛 Simplify bug report template by @qgallouedec in https://github.com/huggingface/trl/pull/2585
* [RLOO] fix token_level_kl by @kashif in https://github.com/huggingface/trl/pull/2575
* ✂️ Truncate by default by @qgallouedec in https://github.com/huggingface/trl/pull/2587
* 🫢 Add `max_prompt_length` parameter in tests by @qgallouedec in https://github.com/huggingface/trl/pull/2588
* 🎞️ Fix documentation SFT -`max_seq_length` instead of `max_length` by @skandermoalla in https://github.com/huggingface/trl/pull/2590
* 👨‍👨‍👧‍👧 GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2565
* 🫣 Ignore CLI test for Python 3.9 by @qgallouedec in https://github.com/huggingface/trl/pull/2592
* Fix merge error by @qgallouedec in https://github.com/huggingface/trl/pull/2595
* 🧰 Tool fine-tuning support DPO by @August-murr in https://github.com/huggingface/trl/pull/2479
* 💾 Reduce memory peak in GRPO by adding `max_prompt_length` and loop usage in logp computation by @qgallouedec in https://github.com/huggingface/trl/pull/2598
* ⚡ Add uv installation instructions by @stevhliu in https://github.com/huggingface/trl/pull/2601
* 🧩 PPO/RLOO/OnlineDPO sequence generation: make deepsped 3 weight gathering optional by @dawidm in https://github.com/huggingface/trl/pull/2557
* 🫷 Include stop token in policy model's generation_config by @dawidm in https://github.com/huggingface/trl/pull/2528
* ✂️ Reintroduce `truncation_mode` in `DPOTrainer` by @anakin87 in https://github.com/huggingface/trl/pull/2551
* 👋 Drop MDX by @qgallouedec in https://github.com/huggingface/trl/pull/2611
* 💎 Rename an inner var in GRPO to improve clarity by @qgallouedec in https://github.com/huggingface/trl/pull/2616
* 🏆 Custom reward function for GRPO and shiny doc by @qgallouedec in https://github.com/huggingface/trl/pull/2606
* 🥞 Fix DPO gradient accumulation loss scaling by @winglian in https://github.com/huggingface/trl/pull/2615
* 🥞 Fix BCO gradient accumulation loss scaling by @qgallouedec in https://github.com/huggingface/trl/pull/2638
* 🍭 Custom reward function for RLOO by @August-murr in https://github.com/huggingface/trl/pull/2612
* 🌯 Fix context manager runtime error when gather is disabled by @Superskyyy in https://github.com/huggingface/trl/pull/2639
* 🥞 Fix CPO gradient accumulation loss scaling  by @qgallouedec in https://github.com/huggingface/trl/pull/2645
* 🥞 Fix GRPO gradient accumulation loss scaling by @qgallouedec in https://github.com/huggingface/trl/pull/2647
* 🥞 Fix KTO gradient accumulation loss scaling by @qgallouedec in https://github.com/huggingface/trl/pull/2648
* 🚛 Provide all columns of the dataset to the reward function by @qgallouedec in https://github.com/huggingface/trl/pull/2650
* 👐 DeepSpeed integration for GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2652
* 🔎 Finegrained reward logging for GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2651
* 📍 Disable caching when grad checkpointing enable in GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2653
* 📏 Log completion length in GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2659
* 🌀 Fix GRPO default completion length doc by @andyl98 in https://github.com/huggingface/trl/pull/2662
* 🏷️ Add model tags to model trained with GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2663
* 🖊 Fix typos by @omahs in https://github.com/huggingface/trl/pull/2673
* ⚡ vLLM for fast generation in GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2600
* 📉 Use `num_logits_to_keep` to reduce memory usage in GRPO by @qgallouedec in https://github.com/huggingface/trl/pull/2683

## New Contributors
* @Abhishek-TAMU made their first contribution in https://github.com/huggingface/trl/pull/2158
* @zhc7 made their first contribution in https://github.com/huggingface/trl/pull/2474
* @SamuelLarkin made their first contribution in https://github.com/huggingface/trl/pull/2549
* @umbilnm made their first contribution in https://github.com/huggingface/trl/pull/2521
* @stevhliu made their first contribution in https://github.com/huggingface/trl/pull/2601
* @dawidm made their first contribution in https://github.com/huggingface/trl/pull/2557
* @Superskyyy made their first contribution in https://github.com/huggingface/trl/pull/2639
* @andyl98 made their first contribution in https://github.com/huggingface/trl/pull/2662
* @omahs made their first contribution in https://github.com/huggingface/trl/pull/2673

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.13.0...v0.14.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.14.0)

---

## v0.13.0: v0.13.0
**Published:** 2024-12-16

## Major and breaking changes

### 🐾 Process-supervised RM Trainer 

We introduced a new trainer to train Process-supervised Reward Model (PRM) in TRL. A PRM rewards the quality of intermediate steps, promoting structured reasoning over focusing solely on the final outcome.With this trainer, we introduce a new dataset type: Stepwise supervision, which is a variant of the prompt-completion type, but for which completion is divided into several intermediate steps, and each step is associated with a label. Find out more in [the stepwise-supervision section in the TRL documentation](https://huggingface.co/docs/trl/en/dataset_formats#stepwise-supervision).

Here is an example of how to use the `PRMTrainer` to train a PRM on the Math Shepherd dataset:

```python
# train_prm.py
from datasets import load_dataset
from trl import PRMConfig, PRMTrainer
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("Qwen/Qwen2-0.5B", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
train_dataset = load_dataset("trl-lib/math_shepherd", split="train[:10%]")

training_args = PRMConfig(output_dir="Qwen2-0.5B-Reward-Math-Sheperd", logging_steps=10)
trainer = PRMTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

For more information, check out the [PRMTrainer documentation](https://huggingface.co/docs/trl/prm_trainer).

by @qgallouedec and @gaetanlop in https://github.com/huggingface/trl/pull/2127 and https://github.com/huggingface/trl/pull/2148

### 🔀 Add `MergeModelCallBack`

Various works show that model merging can non-trivially improve performance, especially if the models belong to the same architecture. TRL now features a callback that merges the reference model with the current policy and optionally pushes the merged checkpoint to the Hub. This could be done on step/epoch end and/or the end of training. This callback uses Arcee's mergekit lib: https://github.com/arcee-ai/mergekit


```python
from trl import DPOTrainer, MergeModelCallback
from trl.mergekit_utils import MergeConfig

config = MergeConfig()
merge_callback = MergeModelCallback(config)
trainer = DPOTrainer(...,  callbacks=[merge_callback])
```

by @August-murr in https://github.com/huggingface/trl/pull/2282

### 🔨 Support for tools for data utils

TRL preprocessing utils now support tooling. A first step toward agent fine-tuning.

```python
from trl import apply_chat_template

def get_current_temperature(location: str):
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for
    """
    return 22.0

example = apply_chat_template(example, tokenizer, tools=[get_current_temperature])
```

by @August-murr in https://github.com/huggingface/trl/pull/2455

### 🌋 Add support for LLaVA-Next in `DPOTrainer`

VLMs have their own specificities which require special treatment in the trainer. `DPOTrainer` now supports LLaVA-Next models natively.

```python
model = model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
trainer = DPOTrainer(model=model, ...)
```

by @chenweize1998 in https://github.com/huggingface/trl/pull/2413

### 🕹️ CLI and TRLParser refactor 

TRL CLI has been refactored to be more user-friendly and easy to extend. We plan to extend the support to all trainers soon.

(simplified output, for readibility)
```shell
$ trl dpo --help
usage: trl dpo [-h] --dataset_name DATASET_NAME [--dataset_config DATASET_CONFIG] --output_dir OUTPUT_DIR [--loss_type {sigmoid,hinge,ipo}]

options:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME, --dataset-name DATASET_NAME
  --dataset_config DATASET_CONFIG, --dataset-config DATASET_CONFIG
  --output_dir OUTPUT_DIR, --output-dir OUTPUT_DIR
                        The output directory where the model predictions and checkpoints will be written. (default: None)
  --loss_type {sigmoid,hinge,ipo}, --loss-type {sigmoid,hinge,ipo}
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2380 and https://github.com/huggingface/trl/pull/2412

### 🤝 Mixture of judges

TRL features a new judge `AllTrueJudge` that unifies the decision of multiple binary judges. This judge implements the Mixture of Judges as described in the [CGPO paper](https://huggingface.co/papers/2409.20370).

```python
from trl import AllTrueJudge, BaseBinaryJudge

class RandomBinaryJudge(BaseBinaryJudge):
    """
    Random binary judge, for testing purposes.
    """

    def judge(self, prompts, completions, gold_completions=None, shuffle_order=True):
        return [random.choice([0, 1, -1]) for _ in range(len(prompts))]


prompts = ["The capital of France is", "The biggest planet in the solar system is"]
completions = [["Paris", "Marseille"], ["Saturn", "Jupiter"]]
judge = AllTrueJudge(judges=[RandomBinaryJudge(), RandomBinaryJudge()])
judgements = judge.judge(prompts=prompts, completions=completions)
print(judgements)  # [0, 1]
```

by @gaetanlop in https://github.com/huggingface/trl/pull/2159

### ❄️ DPO trainer supports `num_logits_to_keep` to save memory

Save memory by only keeping the top `num_logits_to_keep` logits in the DPO trainer.

```python
training_args = DPOConfig(..., use_num_logits_to_keep=True)
```

by @xyangk in https://github.com/huggingface/trl/pull/2129

### 🗺️ Implementation DiscoPOP Loss

The [DiscoPOP](https://huggingface.co/papers/2406.08414) paper uses LLMs to discover more efficient offline preference optimization losses. In the paper the proposed DiscoPOP loss (which is a log-ratio modulated loss) outperformed other optimization losses on different tasks (IMDb positive text generation, Reddit TLDR summarization, and Alpaca Eval 2.0).

```python
training_args = DPOConfig(..., loss_type="discopop", discopop_tau=0.05)
```

by @fanconic in https://github.com/huggingface/trl/pull/2323

### 🧑‍🍳 Add precompute batch size argument in `DPOTrainer` for reference model

We can now control the batch size for precomputing reference model logits.

```python
training_args = DPOConfig(
...
    precompute_ref_log_probs=True,
    precompute_ref_batch_size=4,
)
```

by @SwayamInSync in https://github.com/huggingface/trl/pull/2426

### 📦 Support for packing tokenized datasets for SFT 

`SFTTrainer` has supported packing datasets for faster training. Now, it support packing tokenized datasets as well.

by @kmehant in https://github.com/huggingface/trl/pull/2011

### 📉 Add PEFT support for `PPOTrainer`

`PPOTrainer` now supports PEFT for efficient training.

```python
PPOTrainer(
    ...,
    peft_config=peft_config,
)
```

by @ccs96307 in https://github.com/huggingface/trl/pull/2344

### 💾 Deprecate `config` in favor of `args` in `PPOTrainer` 

`config` has been deprecated in favor of `args` in `PPOTrainer`.

```diff
  PPOTrainer(
-   config=training_args,
+   args=training_args,
  )
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2384

### 👮 Deprecate `policy` in favor of `model` in `PPOTrainer` 

`policy` has been deprecated in favor of `model` in `PPOTrainer`.

```diff
  PPOTrainer(
-   policy=model,
+   model=model,
  )
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2386


## What's Changed
* ⏫ Bump dev version to `0.13.0.dev0` by @qgallouedec in https://github.com/huggingface/trl/pull/2305
* 📰 Update blog posts in documentation by @qgallouedec in https://github.com/huggingface/trl/pull/2319
* ⚰️ Remove deprecated args, script arguments, and PPOv2 by @qgallouedec in https://github.com/huggingface/trl/pull/2306
* 🧽 Fix judge doc by @qgallouedec in https://github.com/huggingface/trl/pull/2320
* 🪧 Fix slack notification titles by @qgallouedec in https://github.com/huggingface/trl/pull/2322
* 🪪 Check with `token_id` instead of `token` in `DPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2324
* Fix wrong truncating index of tensor in DPOTrainer's concatenated_forward() by @yanghh2000 in https://github.com/huggingface/trl/pull/2332
* Fix gradient_checkpointing_kwargs assignment in examples by @Galaxy-Husky in https://github.com/huggingface/trl/pull/2331
* Bump liger-kernel to 0.4.0 by @ByronHsu in https://github.com/huggingface/trl/pull/2333
* DPO trainer supports num_logits_to_keep to save memory by @xyangk in https://github.com/huggingface/trl/pull/2129
* 🧞 Add `output_layer` to the list of `lm_head_namings` in `AutoModelForCausalLMWithValueHead` by @qgallouedec in https://github.com/huggingface/trl/pull/2328
* 🫴 Better guide users in error reporting by @qgallouedec in https://github.com/huggingface/trl/pull/2327
* 🪡 Various RLOO fixes by @qgallouedec in https://github.com/huggingface/trl/pull/2325
* 💣  Remove transformers version check by @xyangk in https://github.com/huggingface/trl/pull/2343
* 👈 Add `tokenizer` arg back and add deprecation guidelines by @qgallouedec in https://github.com/huggingface/trl/pull/2348
* 🖨️ Fix error text in BCO and KTO tokenizing function by @PhilipMay in https://github.com/huggingface/trl/pull/2286
* Adding video llm fine-tuning example by @mfarre in https://github.com/huggingface/trl/pull/2336
* 👋 Remove deprecated `tokenizer` argument in BCO, GKD, Iterative SFT, Nash MD and XPO by @qgallouedec in https://github.com/huggingface/trl/pull/2349
* ⚖️ Add `use_soft_judge` option to `WinRateCallback` by @kashif in https://github.com/huggingface/trl/pull/2347
* 🪜 Stepwise supervision dataset type by @qgallouedec in https://github.com/huggingface/trl/pull/2148
* 🔮 Inference mode in `GeometricMixtureWrapper.forward` by @kashif in https://github.com/huggingface/trl/pull/2345
* 🗃️ Use specified `data_collator` in `RLOOTrainer` and `PPOTrainer` by @bartoszzuk in https://github.com/huggingface/trl/pull/2360
* 📉 Add PEFT support for `PPOTrainer` by @ccs96307 in https://github.com/huggingface/trl/pull/2344
* 📃 Fix description for parameter "generate_during_eval" in dpo_config  by @dakru012 in https://github.com/huggingface/trl/pull/2364
* 🗺️ Implementation DiscoPOP Loss by @fanconic in https://github.com/huggingface/trl/pull/2323
* 🤝 Mixture of judges by @gaetanlop in https://github.com/huggingface/trl/pull/2159
* 🎲 Move random judges in testing utilities by @qgallouedec in https://github.com/huggingface/trl/pull/2365
* Fix dev install by @lewtun in https://github.com/huggingface/trl/pull/2369
* [winrate callback] remove redundant call to eval and train by @kashif in https://github.com/huggingface/trl/pull/2372
* 🧲 Use our own `require_bitsandbytes` by @qgallouedec in https://github.com/huggingface/trl/pull/2370
* ⏰ Add `start_time` to `_maybe_log_save_evaluate` by @qgallouedec in https://github.com/huggingface/trl/pull/2373
* 🔀 Add `MergeModelCallBack` by @August-murr in https://github.com/huggingface/trl/pull/2282
* 📝 Fix typo in dataset generation script by @jiseshen in https://github.com/huggingface/trl/pull/2379
* ⌛ Update log method to include `start_time` parameter by @qgallouedec in https://github.com/huggingface/trl/pull/2381
* 🙈 Suppress warning for estimating tokens in trainers by @qgallouedec in https://github.com/huggingface/trl/pull/2389
* 📦 Support for packing tokenized datasets for SFT by @kmehant in https://github.com/huggingface/trl/pull/2011
* 💾 Deprecate `config` in favor of `args` in `PPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2384
* 🤏 New models for tests by @qgallouedec in https://github.com/huggingface/trl/pull/2287
* 👮 Deprecate `policy` in favor of `model` in `PPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2386
* 🤐 Fix deprecation warnings by @qgallouedec in https://github.com/huggingface/trl/pull/2392
* 🤐 Fix deprecation warnings by @qgallouedec in https://github.com/huggingface/trl/pull/2395
* 🖋️ Fix warning message formatting in `KTOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2394
* 🧳 Move zen generation script and fix tests by @qgallouedec in https://github.com/huggingface/trl/pull/2393
* 🐢 Fix slow tests by @kashif in https://github.com/huggingface/trl/pull/2397
* 🗝️ Update type hints by @qgallouedec in https://github.com/huggingface/trl/pull/2399
* 🖨 Add Script Utilities section to the documentation by @qgallouedec in https://github.com/huggingface/trl/pull/2407
* 👁️ Added SFT support for `SmolVLM` models via standalone script `sft_vlm_smol_vlm.py` by @sergiopaniego in https://github.com/huggingface/trl/pull/2409
* Add note about special tokens in chat templates for LoRA SFT by @lewtun in https://github.com/huggingface/trl/pull/2414
* 🌐 Community Tutorials by @burtenshaw in https://github.com/huggingface/trl/pull/2411
* 🔓 Remove lm_head check in `AutoModelForCausalLMWithValueHead` by @qgallouedec in https://github.com/huggingface/trl/pull/2398
* 🌋 Add support for LLaVA-Next in `DPOTrainer` by @chenweize1998 in https://github.com/huggingface/trl/pull/2413
* ⚠️ Add warning guidelines and update codebase to follow best practices by @qgallouedec in https://github.com/huggingface/trl/pull/2350
* Super tiny typo fix by @fzyzcjy in https://github.com/huggingface/trl/pull/2419
* 🧑‍🍳 Add precompute batch size argument in `DPOTrainer` for reference model by @SwayamInSync in https://github.com/huggingface/trl/pull/2426
* 📑 Refactor `TrlParser` by @qgallouedec in https://github.com/huggingface/trl/pull/2412
* 🔮 Fix unused precomputed ref log probs in DPO by @dakru012 in https://github.com/huggingface/trl/pull/2431
* 🧮 Fix `max_steps` calculation in `RLOOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2433
* 🗂️ Harmonize run and example batch sizes in RLOO docs by @asparius in https://github.com/huggingface/trl/pull/2439
* 🔗 Add "Open in Colab" badges in community tutorials page by @qgallouedec in https://github.com/huggingface/trl/pull/2441
* ©️ Copyrights update by @qgallouedec in https://github.com/huggingface/trl/pull/2454
* 💬 Fix chat for windows by @qgallouedec in https://github.com/huggingface/trl/pull/2443
* 🆔 Add `datast_config` to `ScriptArguments` by @qgallouedec in https://github.com/huggingface/trl/pull/2440
* 🏎 Fix deepspeed preparation of `ref_model` in `OnlineDPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2417
* 👯 Standardize `model_args` by @qgallouedec in https://github.com/huggingface/trl/pull/2442
* [bugfix] Fix DataCollatorForChatML unexpected generation prompt by @NIL-zhuang in https://github.com/huggingface/trl/pull/2450
* ⚖️ Add `tests_latest.yml` workflow file by @qgallouedec in https://github.com/huggingface/trl/pull/2457
* 🛠️ Update tests and fix PPO by @kashif in https://github.com/huggingface/trl/pull/2463
* 🎞️ Add "Fine-tuning open AI models using Hugging Face TRL" YouTube video to community tutorials by @qgallouedec in https://github.com/huggingface/trl/pull/2467
* 🔨 Support for tools for data utils by @August-murr in https://github.com/huggingface/trl/pull/2455
* 🐾 Process-supervised RM Trainer by @gaetanlop in https://github.com/huggingface/trl/pull/2127
* 🕹️ CLI refactor by @qgallouedec in https://github.com/huggingface/trl/pull/2380
* 👀 Add "PaliGemma 🤝 Direct Preference Optimization" in community tutorials by @qgallouedec in https://github.com/huggingface/trl/pull/2475
* ☄️ Add support for Comet  experiment management SDK integration by @yaricom in https://github.com/huggingface/trl/pull/2462
* 📥 Fix missing `BitsAndBytesConfig` import in doc by @August-murr in https://github.com/huggingface/trl/pull/2478
* 👨‍🏫 smol course links and badges by @qgallouedec in https://github.com/huggingface/trl/pull/2484

## New Contributors
* @yanghh2000 made their first contribution in https://github.com/huggingface/trl/pull/2332
* @Galaxy-Husky made their first contribution in https://github.com/huggingface/trl/pull/2331
* @ByronHsu made their first contribution in https://github.com/huggingface/trl/pull/2333
* @xyangk made their first contribution in https://github.com/huggingface/trl/pull/2129
* @mfarre made their first contribution in https://github.com/huggingface/trl/pull/2336
* @dakru012 made their first contribution in https://github.com/huggingface/trl/pull/2364
* @fanconic made their first contribution in https://github.com/huggingface/trl/pull/2323
* @jiseshen made their first contribution in https://github.com/huggingface/trl/pull/2379
* @kmehant made their first contribution in https://github.com/huggingface/trl/pull/2011
* @burtenshaw made their first contribution in https://github.com/huggingface/trl/pull/2411
* @chenweize1998 made their first contribution in https://github.com/huggingface/trl/pull/2413
* @fzyzcjy made their first contribution in https://github.com/huggingface/trl/pull/2419
* @SwayamInSync made their first contribution in https://github.com/huggingface/trl/pull/2426
* @asparius made their first contribution in https://github.com/huggingface/trl/pull/2439
* @NIL-zhuang made their first contribution in https://github.com/huggingface/trl/pull/2450
* @yaricom made their first contribution in https://github.com/huggingface/trl/pull/2462

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.12.0...v0.13.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.13.0)

---

## v0.12.2: v0.12.2
**Published:** 2024-12-06

## What's Changed
* Pin transformers version <4.47 by @kashif in https://github.com/huggingface/trl/pull/2447


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.12.1...v0.12.2

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.12.2)

---

## v0.12.1: v0.12.1
**Published:** 2024-11-15

## What's Changed

* 👈 Add `tokenizer` arg back and add deprecation guidelines by @qgallouedec in https://github.com/huggingface/trl/pull/2348

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.12.0...v0.12.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.12.1)

---

## v0.12.0: v0.12.0
**Published:** 2024-11-04

## Major and breaking changes

### General reward model support for Online DPO

Online DPO intially only supported a reward model that had the same tokenizer and chat template as the trained model. Now, you can use any reward model.

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import OnlineDPOConfig, OnlineDPOTrainer

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, padding_side="left")

reward_model = AutoModelForSequenceClassification.from_pretrained(training_args.reward_model_path, num_labels=1)
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, truncation=True, truncation_side="left")

dataset = load_dataset(script_args.dataset_name)

training_args = OnlineDPOConfig(output_dir="...")
trainer = OnlineDPOTrainer(
    model=model,
    reward_model=reward_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_processing_class=reward_tokenizer,
)
trainer.train()
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2276


### Migration `PPOv2` -> `PPO`

The `PPOv2` trainer has been renamed to `PPO`. The old `PPO` trainer has been removed. `PPOv2` is now deprecated and will be removed in the next release.

```diff
- trainer = PPOv2Trainer(...)
+ trainer = PPOTrainer(...)
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2174

### Refactor `ScriptArguments`

We had `ScriptArguments`, `SFTScriptArguments`, `DPOScriptArguments` and `RewardScriptArguments`. Since they all share mostly the same fields, we've merged them into a single `ScriptArguments` class.
`SFTScriptArguments`, `DPOScriptArguments` and `RewardScriptArguments` still exist but are deprecated and will be removed in the next release.

```diff
- script_args = DPOScriptArguments(...)
+ script_args = ScriptArguments(...)
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2145

### Soft judges for PairRM

The `PairRMJudge` now when called via the `judge` method has a flag `return_scores` that returns the probability scores of the first completion of the pair (instead of the rank of the preferred completion). The logits for the probability score can be scaled by an optional `temperature` parameter. 

```python
from trl import PairRMJudge
pairrm_judge = PairRMJudge()
prompts = ["Translate 'hello' to French", "What's the capital of Japan?"]
completions = [["Bonjour", "Salut"], ["Kyoto", "Tokyo"]]
results = pairrm_judge.judge(prompts, completions, return_scores=True)
print(results)  # [0.7492601275444031, 0.0005497377132996917]
```

by @kashif in https://github.com/huggingface/trl/pull/2221

### Use pairwise judges for online methods

The `OnlineDPOTrainer` and any trainers that inherit from it (`NashMDTrainer` and `XPOTrainer`) can now accept an initialized `PairwiseJudge` instead of a reward model.

```python
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
judge = PairRMJudge()
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = OnlineDPOConfig(output_dir="Qwen2-0.5B-OnlineDPO", logging_steps=10)
trainer = OnlineDPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
```

by @kashif in https://github.com/huggingface/trl/pull/2243

### Rename trainer arg `tokenizer` to `processing_class`

The `tokenizer` argument in the trainers has been renamed to `processing_class` to better reflect the fact that it can be not only a tokenizer but also a processor.

```diff
- trainer = DPOTrainer(model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
+ trainer = DPOTrainer(model, args=training_args, train_dataset=dataset, processing_class=tokenizer)
```

`tokenizer` is still supported for `SFTTrainer` and `DPOTrainer` but deprecated and will be removed in the next release.

by @qgallouedec in https://github.com/huggingface/trl/pull/2162

### Adding weighted preference optimization (WPO) to DPO

The [WPO](https://huggingface.co/papers/2406.11827) paper adapts off-policy data to resemble on-policy data more closely by reweighting preference pairs according to their probability under the current policy. To use this method, set the `use_weighting` flag to `True` in the [`DPOConfig`].

```python
DPOConfig(..., use_weighting=True)
```

<img width="1112" alt="Screenshot 2024-11-04 at 10 59 38" src="https://github.com/user-attachments/assets/544ddc02-bd09-4f21-b8a4-b81c21561a9b">
<img width="539" alt="Screenshot 2024-11-04 at 10 59 22" src="https://github.com/user-attachments/assets/8d5afe9e-89bd-4d00-8483-dd7ba98997e7">


by @gaetanlop in https://github.com/huggingface/trl/pull/2141

### 🃏 Model card for TRL

Using `trainer.push_to_hub()` now automatically creates a model card that includes:

- A link to the base model used
- A link to the dataset used for training
- A link to the TRL repository
- Sample demo code
- A link to the associated Weights & Biases run
- A link to the paper detailing the training procedure
- Versions of dependencies
- BibTeX citations for both the training procedure and TRL

All links are properly formatted to allow cross-referencing, enabling traceability back to sources (e.g., the model appears linked on the paper’s page).


https://github.com/user-attachments/assets/b903964e-9087-45cc-8fb0-2418fdd87b72



by @qgallouedec in https://github.com/huggingface/trl/pull/2123

## Minor

### Conversational dataset support

You can now use conversational datasets directly, without needing to apply a chat template beforehand, for the following trainers:

- `BCOTrainer` (by @qgallouedec in PR #2107)
- `CPOTrainer` (by @qgallouedec in PR #2144)
- `DPOTrainer` (by @qgallouedec in PR #2131)
- `KTOTrainer` (by @qgallouedec in PR #2248)
- `ORPOTrainer` (by @qgallouedec in PR #2184)

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = load_dataset(dataset_name, split="train")

# Not needed anymore:
#
# def process(row):
#     prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
#     prompt_chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
#     chosen = prompt_chosen[len(prompt) :]
#     prompt_rejected = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)
#     rejected = prompt_rejected[len(prompt) :]
#     return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
#
# dataset = dataset.map(process)

training_args = DPOConfig(output_dir="...")
trainer = DPOTrainer(model, args=training_args, train_dataset=dataset, processing_class=tokenizer)
trainer.train()
```

### Refactor DPO data processing

For more information, see PR #2209.

###  `trl env` for printing system info

You can now use `trl env` to print system information, including the platform, Python version, PyTorch version, CUDA device(s), and versions of various libraries.

```
$ trl env

Copy-paste the following information when reporting an issue:

- Platform: Linux-5.15.0-1048-aws-x86_64-with-glibc2.31
- Python version: 3.11.9
- PyTorch version: 2.4.0
- CUDA device(s): NVIDIA H100 80GB HBM3
- Transformers version: 4.47.0.dev0
- Accelerate version: 0.19.0
- Accelerate config: not found
- Datasets version: 3.0.2
- HF Hub version: 0.26.1
- TRL version: 0.12.0+14ef1ab
- bitsandbytes version: 0.44.1
- DeepSpeed version: 0.15.3
- Diffusers version: 0.30.3
- Liger-Kernel version: 0.3.0
- LLM-Blender version: 0.0.2
- OpenAI version: 1.46.0
- PEFT version: 0.13.2
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2104

### Sequence-Level KD

From [GKD paper](https://huggingface.co/papers/2306.13649):

> Sequence-Level KD (Kim & Rush, 2016). SeqKD maximizes the likelihood of high probability sequences generated by the teacher, and can be viewed as supervised FT on teacher-generated outputs.

SeqKD is taken as a baseline in the paper. It is now possible to use Sequence-Level KD in the `GKDTrainer` by setting `seq_kd=True` in the `GKDConfig`.

```python
training_args = GKDConfig(..., seq_kd=True)
```

by @mst272 in https://github.com/huggingface/trl/pull/2220

### Default `dataset_text_field` to `"text"`

Since many users use `"text"` as the column name for textual data in datasets, we've made it the default (previously a required argument) in `SFTConfig`. Now, specifying `dataset_text_field="text"` is no longer necessary.

```diff
  SFTConfig(
      ...,
-     dataset_text_field="text",
  )
```

by @qgallouedec in https://github.com/huggingface/trl/pull/2078

## What's Changed

* [SFT] fix neftune_noise_alpha in SFTTrainer by @kashif in https://github.com/huggingface/trl/pull/1841
* Standardize `training_args` by @qgallouedec in https://github.com/huggingface/trl/pull/2082
* Fix typo in ORPO example. by @skandermoalla in https://github.com/huggingface/trl/pull/2092
* Fix Inconsistency with IsShardedQLoRA Setting by @fabianlim in https://github.com/huggingface/trl/pull/2089
* Fixes #2087 -  _process_tokens for empty prompts in KTOTrainer by @gabikadlecova in https://github.com/huggingface/trl/pull/2093
* KTO: fix logits metric, add logits metric to BCOTrainer by @claralp in https://github.com/huggingface/trl/pull/2094
* Clean up README and remove openrlbenchmark dependency by @lewtun in https://github.com/huggingface/trl/pull/2085
* Fix PPO/RLOO examples by @lewtun in https://github.com/huggingface/trl/pull/2100
* [CLI] `trl env` for printing system info by @qgallouedec in https://github.com/huggingface/trl/pull/2104
* [RewardTrainer] Tokenize inputs within trainer by @lewtun in https://github.com/huggingface/trl/pull/2102
* Fix documentation links by @qgallouedec in https://github.com/huggingface/trl/pull/2105
* fix formatting by @kashif in https://github.com/huggingface/trl/pull/2109
* [online-dpo] allow parse-args as list of floats by @kashif in https://github.com/huggingface/trl/pull/2108
* Fix pack test by @qgallouedec in https://github.com/huggingface/trl/pull/2111
* `BCOTrainer` conversational dataset support by @qgallouedec in https://github.com/huggingface/trl/pull/2107
* Generalizes VSFT script to support REDACTED by @edbeeching in https://github.com/huggingface/trl/pull/2120
* Update example_overview.md by @kashif in https://github.com/huggingface/trl/pull/2125
* Remove `max_length` from `RewardDataCollatorWithPadding` by @qgallouedec in https://github.com/huggingface/trl/pull/2119
* Standardize pushing to Hub in examples by @qgallouedec in https://github.com/huggingface/trl/pull/2126
* Eos token encouragement Clarification by @August-murr in https://github.com/huggingface/trl/pull/2128
* Tokenize row during in `training_step`  by @qgallouedec in https://github.com/huggingface/trl/pull/2117
* ♻️ Standardize `script_args` by @qgallouedec in https://github.com/huggingface/trl/pull/2130
* Add table for WinRateCallback by @lewtun in https://github.com/huggingface/trl/pull/2116
* 🧹 Style by @qgallouedec in https://github.com/huggingface/trl/pull/2132
* arXiv to HF Papers by @qgallouedec in https://github.com/huggingface/trl/pull/2133
* Add correct label for `WinRateCallback` table by @lewtun in https://github.com/huggingface/trl/pull/2134
* 🃏 Model card for TRL by @qgallouedec in https://github.com/huggingface/trl/pull/2123
* Rename `dpo_visual.py` example to `dpo_vlm.py` by @qgallouedec in https://github.com/huggingface/trl/pull/2139
* [GKD] Set custom EOS tokens in generation config by @lewtun in https://github.com/huggingface/trl/pull/2142
* Fix attention mask warning chat cli by @qgallouedec in https://github.com/huggingface/trl/pull/2147
* [CI] Don't use `eval_strategy="steps"` when no eval dataset by @qgallouedec in https://github.com/huggingface/trl/pull/2152
* Conversational dataset support for `DPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2131
* 🩹 [Hotfix] Add setter for tokenizer by @qgallouedec in https://github.com/huggingface/trl/pull/2163
* ↩️ Revert tokenizer hotfix #2163 by @qgallouedec in https://github.com/huggingface/trl/pull/2165
* chore: update test_cli.py by @eltociear in https://github.com/huggingface/trl/pull/2168
* 🏷️ Model badges in trainer documentation by @qgallouedec in https://github.com/huggingface/trl/pull/2160
* Default `dataset_text_field` to `"text"` by @qgallouedec in https://github.com/huggingface/trl/pull/2078
* Update trl version in CITATION.cff by @qgallouedec in https://github.com/huggingface/trl/pull/2171
* 🗑️ Set deprecation version for DPO and SFT arguments to version 0.13 by @qgallouedec in https://github.com/huggingface/trl/pull/2170
* Conversational dataset support for `CPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2144
* Capybara replaced with ultrafeedback_binarized by @August-murr in https://github.com/huggingface/trl/pull/2183
* minor KTO setting changes + KL batch size by @kawine in https://github.com/huggingface/trl/pull/2153
* 🏷️ Model badges: select only TRL models by @qgallouedec in https://github.com/huggingface/trl/pull/2178
* Rename trainer arg `tokenizer` to `processing_class` by @qgallouedec in https://github.com/huggingface/trl/pull/2162
* Update documentation CLI Chat by @qgallouedec in https://github.com/huggingface/trl/pull/2191
* 🃏 Model card: `"unsloth"` tag by @qgallouedec in https://github.com/huggingface/trl/pull/2173
* [CI] fix dpo gpu ci tests by @kashif in https://github.com/huggingface/trl/pull/2189
* Update CONTRIBUTING.md by @kushal34712 in https://github.com/huggingface/trl/pull/2181
* Fix RLOO checkpointing by @bartoszzuk in https://github.com/huggingface/trl/pull/2114
* Update README.md by @PRIYANKjakharia in https://github.com/huggingface/trl/pull/2186
* `skip_prompt=True` in `TextIteratorStreamer` by @qgallouedec in https://github.com/huggingface/trl/pull/2193
* [CI] Use transformers from source in "tests_no_optional_dep" by @qgallouedec in https://github.com/huggingface/trl/pull/2198
* Fix the bug of DPOTrainer where the coefficient of aux_loss is always 0 during training by @muupan in https://github.com/huggingface/trl/pull/2200
* Fix the bug of aux_loss coefficient being 0 in BCOTrainer, CPOTrainer, KTOTrainer, and ORPOTrainer by @muupan in https://github.com/huggingface/trl/pull/2201
* [DPO] Adding weighted preference optimization (WPO) by @gaetanlop in https://github.com/huggingface/trl/pull/2141
* [GKD] interpolate in prob. space by @kashif in https://github.com/huggingface/trl/pull/2204
* Drop `decoder_input_ids` in `DPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2208
* Update incorrect data processing in DataCollatorForChatML by @ruijunfeng in https://github.com/huggingface/trl/pull/2172
* Update log_example_reports.py by @DhruvKadam-git in https://github.com/huggingface/trl/pull/2182
* Report to `"none"` in GKD test by @qgallouedec in https://github.com/huggingface/trl/pull/2214
* [Judges] Soft judges for PairRM by @kashif in https://github.com/huggingface/trl/pull/2221
* Update README.md by @kushal34712 in https://github.com/huggingface/trl/pull/2180
* Updated README.md with CLI examples and additional usage instructions by @Singhal1808 in https://github.com/huggingface/trl/pull/2199
* `trl env` report all cuda devices by @qgallouedec in https://github.com/huggingface/trl/pull/2216
* Conversational dataset support for `ORPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2184
* 🕊️ Migration `PPOv2` -> `PPO` by @qgallouedec in https://github.com/huggingface/trl/pull/2174
* Add Sequence-Level KD by @mst272 in https://github.com/huggingface/trl/pull/2220
* Update dataset_formats.mdx by @August-murr in https://github.com/huggingface/trl/pull/2222
* 📒 Fix type/format confusions by @qgallouedec in https://github.com/huggingface/trl/pull/2223
* Update commands for code linting in contributing guidelines by @Ben-Schneider-code in https://github.com/huggingface/trl/pull/2225
* Refactor `ScriptArguments` by @qgallouedec in https://github.com/huggingface/trl/pull/2145
* Updated `ScriptArguments` warning messages by @sergiopaniego in https://github.com/huggingface/trl/pull/2230
* DPO support `remove_unused_columns` by @qgallouedec in https://github.com/huggingface/trl/pull/2233
* Setting capture output to False by @August-murr in https://github.com/huggingface/trl/pull/2239
* Update SFT examples by @lewtun in https://github.com/huggingface/trl/pull/2244
* Enhancements to Log Report Script: Improved Error Handling and Logging by @DhruvKadam-git in https://github.com/huggingface/trl/pull/2232
* 🔀 Rename `get_batch_sample` and add `num_items_in_batch` to `compute_loss` by @qgallouedec in https://github.com/huggingface/trl/pull/2246
* Refactor DPO data processing by @qgallouedec in https://github.com/huggingface/trl/pull/2209
* Update dataset_formats.mdx by @cameronphchen in https://github.com/huggingface/trl/pull/2259
* Use `processing_class` instead of `tokenizer` in `LogCompletionsCallback` by @qgallouedec in https://github.com/huggingface/trl/pull/2261
* Adjust padding in batch generation by @gaetanlop in https://github.com/huggingface/trl/pull/2251
* setup_chat_format: throw error if there is already a template in base model by @ngxson in https://github.com/huggingface/trl/pull/2252
* Bump the minimum transformers version to v4.46 by @qgallouedec in https://github.com/huggingface/trl/pull/2245
* Conversational dataset support for `KTOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2248
* [Judges] use the pair-judges in online-preference trainers by @kashif in https://github.com/huggingface/trl/pull/2243
* Update reward_modeling.py by @cameronphchen in https://github.com/huggingface/trl/pull/2266
* ♾️ Fix test generation `max_new_tokens` by @qgallouedec in https://github.com/huggingface/trl/pull/2272
* Refactor `log_reports.py` for Improved Logging, File Processing, and Slack Payload Handling by @Mefisto04 in https://github.com/huggingface/trl/pull/2249
* Replace log(sigmoid(log_odds) with logsigmoid(log_odds) for ORPO by @zhanwenchen in https://github.com/huggingface/trl/pull/2274
* [KTO/BCO Trainer] add bos_token_id only if it exists by @seanexp in https://github.com/huggingface/trl/pull/2279
* Fix the computation of KL divergence loss in Nash MD by @d-tiapkin in https://github.com/huggingface/trl/pull/2277
* Don't pass `eval_dataset` in to trainers when no eval strategy by @qgallouedec in https://github.com/huggingface/trl/pull/2270
* Update callbacks.py for fix small python type error by @anch0vy in https://github.com/huggingface/trl/pull/2285
* Use any reward model for online methods by @qgallouedec in https://github.com/huggingface/trl/pull/2276
* Clean dependencies by @qgallouedec in https://github.com/huggingface/trl/pull/2298
* Fix `_save_checkpoint` for online methods by @qgallouedec in https://github.com/huggingface/trl/pull/2288
* Refactor unit tests to use standard unittest assertion methods by @ccs96307 in https://github.com/huggingface/trl/pull/2283
* Remove stale bot by @qgallouedec in https://github.com/huggingface/trl/pull/2300
* Fix no optional dependencies by @qgallouedec in https://github.com/huggingface/trl/pull/2301
* Add `optimizer_cls_and_kwargs` attribute to PPO and RLOO by @qgallouedec in https://github.com/huggingface/trl/pull/2302
* Specify min versions  by @qgallouedec in https://github.com/huggingface/trl/pull/2303

## New Contributors
* @skandermoalla made their first contribution in https://github.com/huggingface/trl/pull/2092
* @fabianlim made their first contribution in https://github.com/huggingface/trl/pull/2089
* @gabikadlecova made their first contribution in https://github.com/huggingface/trl/pull/2093
* @August-murr made their first contribution in https://github.com/huggingface/trl/pull/2128
* @kushal34712 made their first contribution in https://github.com/huggingface/trl/pull/2181
* @PRIYANKjakharia made their first contribution in https://github.com/huggingface/trl/pull/2186
* @ruijunfeng made their first contribution in https://github.com/huggingface/trl/pull/2172
* @DhruvKadam-git made their first contribution in https://github.com/huggingface/trl/pull/2182
* @Singhal1808 made their first contribution in https://github.com/huggingface/trl/pull/2199
* @mst272 made their first contribution in https://github.com/huggingface/trl/pull/2220
* @Ben-Schneider-code made their first contribution in https://github.com/huggingface/trl/pull/2225
* @sergiopaniego made their first contribution in https://github.com/huggingface/trl/pull/2230
* @cameronphchen made their first contribution in https://github.com/huggingface/trl/pull/2259
* @ngxson made their first contribution in https://github.com/huggingface/trl/pull/2252
* @Mefisto04 made their first contribution in https://github.com/huggingface/trl/pull/2249
* @zhanwenchen made their first contribution in https://github.com/huggingface/trl/pull/2274
* @d-tiapkin made their first contribution in https://github.com/huggingface/trl/pull/2277
* @anch0vy made their first contribution in https://github.com/huggingface/trl/pull/2285
* @ccs96307 made their first contribution in https://github.com/huggingface/trl/pull/2283

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.11.0...v0.12.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.12.0)

---

## v0.11.4: v0.11.4
**Published:** 2024-10-15

## What's Changed

* Fix Inconsistency with IsShardedQLoRA Setting by @fabianlim in https://github.com/huggingface/trl/pull/2089

## New Contributors

* @fabianlim made their first contribution in https://github.com/huggingface/trl/pull/2089

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.11.3...v0.11.4

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.11.4)

---

## v0.11.3: v0.11.3
**Published:** 2024-10-10

## What's Changed

* [GKD] interpolate in prob. space by @kashif in https://github.com/huggingface/trl/pull/2204
* Drop `decoder_input_ids` in `DPOTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2208
* Update incorrect data processing in DataCollatorForChatML by @ruijunfeng in https://github.com/huggingface/trl/pull/2172

## New Contributors

* @ruijunfeng made their first contribution in https://github.com/huggingface/trl/pull/2172

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.11.2...v0.11.3

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.11.3)

---

## v0.11.2: v0.11.2
**Published:** 2024-10-07

## What's Changed

* Fix RLOO checkpointing by @bartoszzuk in https://github.com/huggingface/trl/pull/2114

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.11.1...v0.11.2

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.11.2)

---

## v0.11.1: v0.11.1
**Published:** 2024-09-24

## Bug fix

* allow parse-args as list of floats for Online DPO, XPO and Nash-MD configs by @kashif in #2108

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.11.0...v0.11.1


[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.11.1)

---

## v0.11.0: v0.11.0
**Published:** 2024-09-19

We are excited to introduce the new v0.11.0 release, with many new features and post-training algorithms. The highlights are as follows:

## New post-training methods

### Generalized Knowledge Distillation

<img width="992" alt="Screenshot 2024-09-19 at 10 01 02" src="https://github.com/user-attachments/assets/97afd65d-1a2c-484b-b6dd-b02a2cbe6430">

[Generalized Knowledge Distillation](https://huggingface.co/papers/2306.13649) (GKD) is a post-training method from Google DeepMind that extends standard knowledge distillation by allowing the student to generate outputs during training and receive _online feedback_ from the teacher. It consistently outperforms SFT and in some cases enables the student model to match the performance of the teacher, but with far fewer parameters. 

To train models with this method, check out the [`GKDTrainer`](https://huggingface.co/docs/trl/v0.11.0/en/gkd_trainer).

### Exploratory Preference Optimization

<img width="1224" alt="Screenshot 2024-09-19 at 10 13 27" src="https://github.com/user-attachments/assets/36decb24-ef01-41f1-84e8-53b491eb6c86">

[Exploratory Preference Optimization](https://huggingface.co/papers/2405.21046) is an online post-training method from researchers at Microsoft, MIT, and Wisconsin that extends DPO to incorporate online feedback from reward models or LLM judges. It is similar to [online DPO](https://huggingface.co/docs/trl/v0.11.0/en/online_dpo_trainer), but has a slightly different theoretical basis concerning sample efficiency.

To train models with this method, check out the [`XPOTrainer`](https://huggingface.co/docs/trl/v0.11.0/en/xpo_trainer#xpo-trainer).

### Nash Learning with Human Feedback

<img width="476" alt="Screenshot 2024-09-19 at 10 32 04" src="https://github.com/user-attachments/assets/8e68263f-bf5a-4f68-b451-110c78e27bb6">


[Nash Learning with Human Feedback](https://huggingface.co/papers/2312.00886) is a novel post-training method from Google DeepMind that uses _pairwise preference models_ which are conditioned on two inputs, instead of the single one used in reward models. These preference models are then used to train a policy that consistently produces responses that are preferred over those from competing policies, thus approximating a Nash equilibrium (i.e. a two player game where actions are responses and payoffs are given by the preference model). 

To train models with this method, check out the [`NashMDTrainer`](https://huggingface.co/docs/trl/v0.11.0/en/nash_md_trainer#nash-md-trainer).

## New trainer features

* Online DPO now supports training LoRA adapters with PEFT, which means you can dramatically reduce the amount of VRAM needed to train models with this method. By @qgallouedec in https://github.com/huggingface/trl/pull/2041
* The `OrpoTrainer` has better integration with PyTorchXLA for faster step time on TPUs ⚡ . By @wenxindongwork in https://github.com/huggingface/trl/pull/2001

## Deprecations 🚨

* The `PPOTrainer` is marked for deprecated in favour of `PPOv2Trainer` to provide a consistent API across TRL's trainers. It will be removed in `v0.12.0`. By @qgallouedec in https://github.com/huggingface/trl/pull/2016
* The `RichProgressCallback` has been removed from the example scripts as it caused a variety of problems with logging in distributed environments. You can still use it by adding it manually to the trainer callbacks. By @lewtun in https://github.com/huggingface/trl/pull/2053

## Bugfixes and improvements
* Adds experimental Liger support to SFT script by @edbeeching in https://github.com/huggingface/trl/pull/1992
* move slow-tests CI to new cluster by @glegendre01 in https://github.com/huggingface/trl/pull/1996
* [Online-DPO] fixes to the training scripts and setup.py by @kashif in https://github.com/huggingface/trl/pull/1997
* [pre-commit] update pre-commit yaml by @kashif in https://github.com/huggingface/trl/pull/2002
* [Docs] Add Liger-Kernel usage to SFTTrainer page by @ryankert01 in https://github.com/huggingface/trl/pull/2007
* [ci] pin numpy to < 2 on windows by @kashif in https://github.com/huggingface/trl/pull/2009
* Remove `prompts` arg from `WinrateCallback` by @qgallouedec in https://github.com/huggingface/trl/pull/2010
* Allow `WinRateCallback` to be used without reference model by @qgallouedec in https://github.com/huggingface/trl/pull/2013
* Feat: Add support for APO-zero in KTOTrainer by @KarelDO in https://github.com/huggingface/trl/pull/1952
* Clean configs documentation by @qgallouedec in https://github.com/huggingface/trl/pull/1944
* Refactor reward modelling script to work with chat models by @lewtun in https://github.com/huggingface/trl/pull/2026
* correct formatting of star sign in kto_trainer.mdx by @mattany in https://github.com/huggingface/trl/pull/2031
* Remove unused functions in `core.py` by @northern-64bit in https://github.com/huggingface/trl/pull/2017
* Improves formatting of docstring + newlines by @northern-64bit in https://github.com/huggingface/trl/pull/2006
* Fix `packing` doc in `SFTConfig` and fix error when neither `dataset_text_field` nor `formatting_func` is provided. by @qgallouedec in https://github.com/huggingface/trl/pull/2035
* fix: unpackaging error in Custom Mixture of Experts model when `aux_loss_enabled` is set to True. by @Jonathanjordan21 in https://github.com/huggingface/trl/pull/2039
* Drop canonical namespaces by @qgallouedec in https://github.com/huggingface/trl/pull/2048
* Change `non_eos_penalty` to be consistent across `OnPolicy` trainers by @RylanSchaeffer in https://github.com/huggingface/trl/pull/2033
* Temporary pin the transformers hash in the CI by @qgallouedec in https://github.com/huggingface/trl/pull/2049
* [XPO] xpo trainer by @kashif in https://github.com/huggingface/trl/pull/1943
* Fix logits compuation in KTO trainer prediction step by @issamemari in https://github.com/huggingface/trl/pull/2050
* [Draft, don't merge] Fix failing windows by @LysandreJik in https://github.com/huggingface/trl/pull/2051
* Clean up DPO example by @lewtun in https://github.com/huggingface/trl/pull/2043
* Remove `debug` and `sanity_check` args by @qgallouedec in https://github.com/huggingface/trl/pull/2055
* Gkd trainer by @kashif in https://github.com/huggingface/trl/pull/1814
* Documentation dataset format by @qgallouedec in https://github.com/huggingface/trl/pull/2020
* Add missing autodocs by @qgallouedec in https://github.com/huggingface/trl/pull/2056
* Mask loss in gkd when generating from the student by @gaetanlop in https://github.com/huggingface/trl/pull/2058
* ©️ Copyrights by @qgallouedec in https://github.com/huggingface/trl/pull/2063
* Support for `SFTTrainer.evaluate()` and `SFTTrainer.predict()` with null train_dataset by @Sohaib9920 in https://github.com/huggingface/trl/pull/2004
* make cuda-only tests device-agnostic  by @faaany in https://github.com/huggingface/trl/pull/2044
* Make `ConstantLengthDataset` (or `packing=True`) shuffle examples before they are packed by @muupan in https://github.com/huggingface/trl/pull/2037
* Standardise API for `WinRateCallback` and `LogCompletionsCallback` by @lewtun in https://github.com/huggingface/trl/pull/2061
* Fix dataset in GKD script by @lewtun in https://github.com/huggingface/trl/pull/2067
* [online models] remove min_new_tokens=args.max_new_tokens by @kashif in https://github.com/huggingface/trl/pull/2069
* Standardising datasets for testing by @qgallouedec in https://github.com/huggingface/trl/pull/2065
* [KTO] learning rate recomentations for kto by @kashif in https://github.com/huggingface/trl/pull/2070
* Nash md by @kashif in https://github.com/huggingface/trl/pull/1853
* Use `transformers` utilities when possible by @qgallouedec in https://github.com/huggingface/trl/pull/2064
* Minor doc fixes and comments by @qgallouedec in https://github.com/huggingface/trl/pull/2073
* Added error check to RLOO, PPOv2, OnlineDPO that `ref_policy` and `policy` have different identities by @RylanSchaeffer in https://github.com/huggingface/trl/pull/2057
* `processor(prompt, images=image)` to `processor(images=image, text=prompt)` by @qgallouedec in https://github.com/huggingface/trl/pull/2076
* Use wrapped model for reference completions in `WinRateCallback` and set default `freq` to `eval_steps` in LogCompletionsCallback` by @lewtun in https://github.com/huggingface/trl/pull/2074
* Conversational dataset support for Online DPO by @qgallouedec in https://github.com/huggingface/trl/pull/2075
* [WIP] Fix `logits/chosen` and `logits/rejected` metrics in `kto_trainer`. by @PhilipMay in https://github.com/huggingface/trl/pull/2077
* Standardize dataset naming by @qgallouedec in https://github.com/huggingface/trl/pull/2081
* Fix deepspeed for `PPOv2Trainer` by @qgallouedec in https://github.com/huggingface/trl/pull/2080

## New Contributors
* @AdnaneKhan made their first contribution in https://github.com/huggingface/trl/pull/1822
* @mkopecki made their first contribution in https://github.com/huggingface/trl/pull/1825
* @DZ9 made their first contribution in https://github.com/huggingface/trl/pull/1836
* @MAOJIASONG made their first contribution in https://github.com/huggingface/trl/pull/1840
* @davanstrien made their first contribution in https://github.com/huggingface/trl/pull/1845
* @eliebak made their first contribution in https://github.com/huggingface/trl/pull/1863
* @Rishav-hub made their first contribution in https://github.com/huggingface/trl/pull/1862
* @cemiu made their first contribution in https://github.com/huggingface/trl/pull/1738
* @SunMarc made their first contribution in https://github.com/huggingface/trl/pull/1919
* @karel-contextual made their first contribution in https://github.com/huggingface/trl/pull/1928
* @RylanSchaeffer made their first contribution in https://github.com/huggingface/trl/pull/1932
* @mina-parham made their first contribution in https://github.com/huggingface/trl/pull/1961
* @RhuiDih made their first contribution in https://github.com/huggingface/trl/pull/1887
* @SeungyounShin made their first contribution in https://github.com/huggingface/trl/pull/1969
* @kit1980 made their first contribution in https://github.com/huggingface/trl/pull/1933
* @akakakakakaa made their first contribution in https://github.com/huggingface/trl/pull/1987
* @hvaara made their first contribution in https://github.com/huggingface/trl/pull/1990
* @glegendre01 made their first contribution in https://github.com/huggingface/trl/pull/1996
* @ryankert01 made their first contribution in https://github.com/huggingface/trl/pull/2007
* @KarelDO made their first contribution in https://github.com/huggingface/trl/pull/1952
* @mattany made their first contribution in https://github.com/huggingface/trl/pull/2031
* @northern-64bit made their first contribution in https://github.com/huggingface/trl/pull/2017
* @Jonathanjordan21 made their first contribution in https://github.com/huggingface/trl/pull/2039
* @issamemari made their first contribution in https://github.com/huggingface/trl/pull/2050
* @wenxindongwork made their first contribution in https://github.com/huggingface/trl/pull/2001
* @Sohaib9920 made their first contribution in https://github.com/huggingface/trl/pull/2004
* @faaany made their first contribution in https://github.com/huggingface/trl/pull/2044
* @muupan made their first contribution in https://github.com/huggingface/trl/pull/2037
* @PhilipMay made their first contribution in https://github.com/huggingface/trl/pull/2077

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.9.6...v0.11.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.11.0)

---

## v0.10.1: v0.10.1
**Published:** 2024-08-29

We are excited to introduce the new v0.10.1 release, with many new exciting features and post-training algorithms. The highlights are as follows:

### Online DPO

<img width="1210" alt="Screenshot 2024-08-29 at 15 53 29" src="https://github.com/user-attachments/assets/c11863ca-434c-47d7-8436-dc096683075a">

[Online DPO](https://arxiv.org/abs/2402.04792) is a new alignment method from DeepMind to boost the performance of LLMs. With Online DPO, data is generated on the fly by the trained model (instead of pre-collected). For each prompt, two completions are generated, with a reward model selecting the preferred one. This approach:
* Eliminates the need for a pre-collected preference dataset (it's generated online)
* Enables continuous model improvement
* Yields better results than traditional DPO

To train models with this method, use the `OnlineDPOTrainer`

### Liger Triton kernels for supercharged SFT

![image (18)](https://github.com/user-attachments/assets/3e6cca57-1ddd-44fc-9822-e1b38f8d056c)

* We've integrated [LinkedIn's Liger Triton kernels](https://github.com/linkedin/Liger-Kernel) to the `SFTTrainer` for faster throughput and lower memory usage. To use them, set `use_liger_kernel` in `SFTConfig`

### DPO for VLMs

* We've added support to align vision-language models with DPO, now covering architectures LLaVa-1.5, PaliGemma, and Idefics2. To train VLMs with DPO, use the `dpo_visual.py` script as follows

```bash
accelerate launch examples/scripts/dpo_visual.py \
    --dataset_name HuggingFaceH4/rlaif-v_formatted \
    --model_name_or_path google/paligemma-3b-pt-224 \
    --trust_remote_code \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --output_dir dpo_paligemma_rlaif-v \
    --bf16 \
    --torch_dtype bfloat16
```

### WinRate callback for LLM as a judge

* We've added support to compute win rates over the reference model for methods like DPO. To do so, configure the callback to point to the LLM as judge API (OpenAI or Hugging Face Inference API) and then add:

```python
trainer = DPOTrainer(...)
win_rate_callback = WinRateCallback(..., trainer=trainer)
trainer.add_callback(win_rate_callback)
```

### Anchored Preference Optimisation (APO) for fine-grained human/AI feedback

![](https://contextual.ai/wp-content/uploads/2024/08/blogpost-2-1.png)

* Added the [APO](https://huggingface.co/papers/2408.06266) method, which is an "anchored" version of the alignment objective. There are two variants: `apo_zero` and `apo_down`. The `apo_zero` loss increases the likelihood of winning outputs while decreasing the likelihood of losing outputs, making it suitable when the model is less performant than the winning outputs. On the other hand, `apo_down` decreases the likelihood of both winning and losing outputs, but with a stronger emphasis on reducing the likelihood of losing outputs. This variant is more effective when the model is better than the winning outputs. To use these losses, set `loss_type="apo_zero"` or `loss_type="apo_down"` in the `DPOConfig`


## What's Changed

* Set dev version by @vwxyzjn in https://github.com/huggingface/trl/pull/1817
* Upgrade GitHub actions by @qgallouedec in https://github.com/huggingface/trl/pull/1818
* DPO Llava 1.5 and PaliGemma support by @qgallouedec in https://github.com/huggingface/trl/pull/1797
* Delete unused benchmark.yml workflow by @AdnaneKhan in https://github.com/huggingface/trl/pull/1822
* Consistent use of trust_remote_code by @qgallouedec in https://github.com/huggingface/trl/pull/1806
* Fix: authentication token kwarg not passed when loading PEFT adapters by @mkopecki in https://github.com/huggingface/trl/pull/1825
* refactor trainer callbacks by @kashif in https://github.com/huggingface/trl/pull/1826
* Uniform `model_ref` naming by @qgallouedec in https://github.com/huggingface/trl/pull/1835
* fix ppov2_trainer tensorboard logging bug by @DZ9 in https://github.com/huggingface/trl/pull/1836
* Fix issues of KTOTrainer by @MAOJIASONG in https://github.com/huggingface/trl/pull/1840
* add link to DPO datasets collection by @davanstrien in https://github.com/huggingface/trl/pull/1845
* fix arg parsing in chat.py by @lvwerra in https://github.com/huggingface/trl/pull/1846
* DPO for VLM blog post in doc by @qgallouedec in https://github.com/huggingface/trl/pull/1844
* Add WinRateCallback and Judges by @lewtun in https://github.com/huggingface/trl/pull/1598
* Remove `CI_HUB_USER_TOKEN` by @qgallouedec in https://github.com/huggingface/trl/pull/1852
* Online DPO and Online trainer refactor by @vwxyzjn in https://github.com/huggingface/trl/pull/1809
* [online-DPO] online dpo cleanups by @kashif in https://github.com/huggingface/trl/pull/1864
* arXiv to HF Papers by @qgallouedec in https://github.com/huggingface/trl/pull/1870
* fix fsdp & qlora support by @eliebak in https://github.com/huggingface/trl/pull/1863
* Import missing ```setup_chat_format``` by @Rishav-hub in https://github.com/huggingface/trl/pull/1862
* Bug Fix while training using SFTTrainer with DataCollatorForCompletionOnlyLM by @Rishav-hub in https://github.com/huggingface/trl/pull/1861
* Small fixes to online dpo example by @edbeeching in https://github.com/huggingface/trl/pull/1879
* Skip BigBird save and load test until next transformers version by @qgallouedec in https://github.com/huggingface/trl/pull/1874
* Llama in modelling value head tests by @qgallouedec in https://github.com/huggingface/trl/pull/1878
* Improve judges by @qgallouedec in https://github.com/huggingface/trl/pull/1856
* [Do not merge] Re-add BigBird Pegasus save/load test by @qgallouedec in https://github.com/huggingface/trl/pull/1876
* Re-add BigBird Pegasus save/load test by @qgallouedec in https://github.com/huggingface/trl/pull/1882
* Move BCO to separate BCOTrainer with fixes by @claralp in https://github.com/huggingface/trl/pull/1869
* Update example overview documentation section by @qgallouedec in https://github.com/huggingface/trl/pull/1883
* fix dpo_trainer bug for LLMs without bos_token in config by @DZ9 in https://github.com/huggingface/trl/pull/1885
* Fix SFT for VLM example by @qgallouedec in https://github.com/huggingface/trl/pull/1865
* `evaluation_strategy` -> `eval_strategy` by @qgallouedec in https://github.com/huggingface/trl/pull/1894
* fix serialization of RunningMoments on multiple GPUs by @claralp in https://github.com/huggingface/trl/pull/1892
* [WIP] Fix CI by @qgallouedec in https://github.com/huggingface/trl/pull/1897
* Drop `setUpClass` in reward tester by @qgallouedec in https://github.com/huggingface/trl/pull/1895
* Support `IterableDataset` for `SFTTrainer` by @qgallouedec in https://github.com/huggingface/trl/pull/1899
* Fix data processing in ORPO example script by @qgallouedec in https://github.com/huggingface/trl/pull/1903
* [RPO] use loss from v3 of paper by @kashif in https://github.com/huggingface/trl/pull/1904
* Support Rank Stabilized LoRA in the ModelConfig/LoraConfig by @JohnGiorgi in https://github.com/huggingface/trl/pull/1877
* [Online-DPO] num_generation_per_prompt is fixed by @kashif in https://github.com/huggingface/trl/pull/1898
* Fix GPT2 sentiment notebook reward by @cemiu in https://github.com/huggingface/trl/pull/1738
* Fix `AlignPropTrainer` import by @qgallouedec in https://github.com/huggingface/trl/pull/1908
* Various args and test fix by @qgallouedec in https://github.com/huggingface/trl/pull/1909
* `lr_scheduler.step()` after `optimizer.step()` by @qgallouedec in https://github.com/huggingface/trl/pull/1918
* `torch.cuda.amp.autocast()` -> `torch.amp.autocast("cuda")` by @qgallouedec in https://github.com/huggingface/trl/pull/1921
* Fix orpo trainer loss device by @SunMarc in https://github.com/huggingface/trl/pull/1919
* Add transformers library name for TRL repos by @lewtun in https://github.com/huggingface/trl/pull/1922
* Standardize `dataset_num_proc` usage by @qgallouedec in https://github.com/huggingface/trl/pull/1925
* `PartialState().local_main_process_first()` when map in examples by @qgallouedec in https://github.com/huggingface/trl/pull/1926
* minor BCO fixes by @claralp in https://github.com/huggingface/trl/pull/1923
* Improve DPO/loss doc by @qgallouedec in https://github.com/huggingface/trl/pull/1929
* feat: anchored pref optimization by @karel-contextual in https://github.com/huggingface/trl/pull/1928
* Add tests for DPO for VLM by @qgallouedec in https://github.com/huggingface/trl/pull/1935
* fix model to save in ppov2 by @mnoukhov in https://github.com/huggingface/trl/pull/1776
* Optional Additional Loss to Center Reward Models' Outputs by @RylanSchaeffer in https://github.com/huggingface/trl/pull/1932
* Properly label all models when pushed to the hub by @qgallouedec in https://github.com/huggingface/trl/pull/1940
* Skip token in `push_to_hub` by @qgallouedec in https://github.com/huggingface/trl/pull/1945
* Fix model wrapping for online DPO by @lewtun in https://github.com/huggingface/trl/pull/1946
* Don't mark issues as stale if nobody answered by @qgallouedec in https://github.com/huggingface/trl/pull/1949
* Add a simple-to-understand example for online DPO by @vwxyzjn in https://github.com/huggingface/trl/pull/1947
* Log WandB tables on main process by @lewtun in https://github.com/huggingface/trl/pull/1951
* [ODPO] Fix global step for consistent checkpointing with global updates by @lewtun in https://github.com/huggingface/trl/pull/1950
* "help wanted" in label to exempt from stale by @qgallouedec in https://github.com/huggingface/trl/pull/1956
* Fix response truncation in examples/notebooks/gpt2-sentiment.ipynb by @qgallouedec in https://github.com/huggingface/trl/pull/1957
* [ODPO] Refactor training script to use messages API by @lewtun in https://github.com/huggingface/trl/pull/1958
* Support LLaVA-NeXT in Vision SFT by @qgallouedec in https://github.com/huggingface/trl/pull/1959
* Add issue/PR templates, code of conduct & better contributing guide by @lewtun in https://github.com/huggingface/trl/pull/1963
* Fix issue with precompute_ref_log_probs not working when rpo_alpha is None by @mina-parham in https://github.com/huggingface/trl/pull/1961
* add arg `padding_free` to DataCollatorForCompletionOnlyLM by @RhuiDih in https://github.com/huggingface/trl/pull/1887
* Optimize DPO log probability calculation by retaining necessary cache, saving up to 30GB of memory (#1968) by @SeungyounShin in https://github.com/huggingface/trl/pull/1969
* New mismatch pair creation strategy by @qgallouedec in https://github.com/huggingface/trl/pull/1970
* Fix issue templates location by @qgallouedec in https://github.com/huggingface/trl/pull/1973
* Use weights_only for load by @kit1980 in https://github.com/huggingface/trl/pull/1933
* Fix flaky Hub tests by @lewtun in https://github.com/huggingface/trl/pull/1981
* fix a few minor bugs in ppo.py by @kykim0 in https://github.com/huggingface/trl/pull/1966
* Test for #1970 by @qgallouedec in https://github.com/huggingface/trl/pull/1974
* Restore reruns for flaky tests by @lewtun in https://github.com/huggingface/trl/pull/1982
* Promote `PairRMJudge` to top-level import by @qgallouedec in https://github.com/huggingface/trl/pull/1985
* [DPO] TR-DPO gather the target model params as well when syncing by @kashif in https://github.com/huggingface/trl/pull/1978
* `torch.load` with `weights_only=True` by @qgallouedec in https://github.com/huggingface/trl/pull/1988
* Skip the failing Online DPO test by @qgallouedec in https://github.com/huggingface/trl/pull/1989
* Refactor Online DPO by @vwxyzjn in https://github.com/huggingface/trl/pull/1839
* [DPO] tokenize and process DPO data via batches by @kashif in https://github.com/huggingface/trl/pull/1914
* [RPO] Add ignore_index in DPOTrainer's nn.CrossEntropyLoss by @akakakakakaa in https://github.com/huggingface/trl/pull/1987
* Relax numpy upper bound and bump deepspeed version by @hvaara in https://github.com/huggingface/trl/pull/1990
* Adds experimental Liger support to SFT script by @edbeeching in https://github.com/huggingface/trl/pull/1992

## New Contributors
* @AdnaneKhan made their first contribution in https://github.com/huggingface/trl/pull/1822
* @mkopecki made their first contribution in https://github.com/huggingface/trl/pull/1825
* @DZ9 made their first contribution in https://github.com/huggingface/trl/pull/1836
* @MAOJIASONG made their first contribution in https://github.com/huggingface/trl/pull/1840
* @davanstrien made their first contribution in https://github.com/huggingface/trl/pull/1845
* @eliebak made their first contribution in https://github.com/huggingface/trl/pull/1863
* @Rishav-hub made their first contribution in https://github.com/huggingface/trl/pull/1862
* @cemiu made their first contribution in https://github.com/huggingface/trl/pull/1738
* @SunMarc made their first contribution in https://github.com/huggingface/trl/pull/1919
* @karel-contextual made their first contribution in https://github.com/huggingface/trl/pull/1928
* @RylanSchaeffer made their first contribution in https://github.com/huggingface/trl/pull/1932
* @mina-parham made their first contribution in https://github.com/huggingface/trl/pull/1961
* @RhuiDih made their first contribution in https://github.com/huggingface/trl/pull/1887
* @SeungyounShin made their first contribution in https://github.com/huggingface/trl/pull/1969
* @kit1980 made their first contribution in https://github.com/huggingface/trl/pull/1933
* @akakakakakaa made their first contribution in https://github.com/huggingface/trl/pull/1987
* @hvaara made their first contribution in https://github.com/huggingface/trl/pull/1990

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.9.6...v0.10

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.10.1)

---

## v0.9.6: v0.9.6 release
**Published:** 2024-07-08

We are excited to introduce the new v0.9.6 release. Many new exciting features and algorithms. The highlights are as follows:

* Support for [SimPO](https://arxiv.org/abs/2405.14734) by @fe1ixxu, a reference-free method that also regularizes output length. To use this loss, the users can input `loss_type="simpo"` and `cpo_alpha=0` in the `CPOConfig` and use it with the `CPOTrainer`.
<img width="880" alt="image" src="https://github.com/huggingface/trl/assets/5555347/87551147-3f58-4c6a-9a78-70b513dea76e">

* Added [AlignProp](https://align-prop.github.io/) by @mihirp1998, a method for finetuning Stable Diffusion model using reward gradients.
* Added [Efficient Exact Optimization (EXO)](https://arxiv.org/abs/2402.00856) by @haozheji

We also included many important fixes and improvements such as fixing prints in the CLI with GCP containers by @alvarobartt. Enjoy the release!


## What's Changed
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1710
* Add a variant of CPO, SimPO by @fe1ixxu in https://github.com/huggingface/trl/pull/1703
* [RPO] fix nll loss by @kashif in https://github.com/huggingface/trl/pull/1705
* fix yaml parser for derived config classes by @mnoukhov in https://github.com/huggingface/trl/pull/1713
* Fix default padding_value in dpo_config.py by @mnoukhov in https://github.com/huggingface/trl/pull/1692
* feat(ci): add trufflehog secrets detection by @McPatate in https://github.com/huggingface/trl/pull/1721
* ktotrainer: Refuse datasets which contain only one class of labels by @jetlime in https://github.com/huggingface/trl/pull/1724
* adds AOT by @imelnyk in https://github.com/huggingface/trl/pull/1701
* Workflow: Notify tests results on slack channel by @younesbelkada in https://github.com/huggingface/trl/pull/1744
* better trl parser with yaml config by @mnoukhov in https://github.com/huggingface/trl/pull/1739
* CI / core: Pin `numpy` to `!=2.0.0` for CI and to users by @younesbelkada in https://github.com/huggingface/trl/pull/1747
* `TrlParser`: Add ignore extra args option by @younesbelkada in https://github.com/huggingface/trl/pull/1748
* small KTO fixes by @kawine in https://github.com/huggingface/trl/pull/1734
* CPO / DPO: Fix red CI by @younesbelkada in https://github.com/huggingface/trl/pull/1749
* prepare deepspeed accomodate fp16 and bf16 by @mnoukhov in https://github.com/huggingface/trl/pull/1728
* CI / `KTOTrainer`: Remove old tests by @younesbelkada in https://github.com/huggingface/trl/pull/1750
* change the `process` function in the example of DPO by @AIR-hl in https://github.com/huggingface/trl/pull/1753
* Integrate f-divergence to DPO (Follow up) by @1485840691 in https://github.com/huggingface/trl/pull/1610
* Support for returning past_key_values from the model by @idanshen in https://github.com/huggingface/trl/pull/1742
* Fix masking of response tokens by @mertsayar8 in https://github.com/huggingface/trl/pull/1718
* Support num_train_epochs by @vwxyzjn in https://github.com/huggingface/trl/pull/1743
* Fix: Add dataset_text_field in examples/scripts/sft.py by @scottsuk0306 in https://github.com/huggingface/trl/pull/1758
* New sentiment and descriptiveness dataset by @vwxyzjn in https://github.com/huggingface/trl/pull/1757
* Add CPO-SimPO method by @fe1ixxu in https://github.com/huggingface/trl/pull/1760
* Added Reward Backpropogation Support  by @mihirp1998 in https://github.com/huggingface/trl/pull/1585
* MoE Models: option to add load balancing loss by @claralp in https://github.com/huggingface/trl/pull/1765
* `evaluation_strategy` to `eval_strategy` by @qgallouedec in https://github.com/huggingface/trl/pull/1771
* add Efficient Exact Optimization (EXO) by @haozheji in https://github.com/huggingface/trl/pull/1735
* Remove the leading space in the tldr preference dataset by @vwxyzjn in https://github.com/huggingface/trl/pull/1773
* Fix Documentation Overflow Issues for Long URLs in SFTConfig by @Mubin17 in https://github.com/huggingface/trl/pull/1774
* Visual DPO by @qgallouedec in https://github.com/huggingface/trl/pull/1647
* [DOCS] fix docs and cli example script by @kashif in https://github.com/huggingface/trl/pull/1780
* Fixed typo in SFT trainer docs by @detsutut in https://github.com/huggingface/trl/pull/1788
* [SFT] add model_init_kwargs to training_args by @kashif in https://github.com/huggingface/trl/pull/1787
* Bugfix: Preserve token fields when converting TrainingArguments to SFTConfig by @noahlt in https://github.com/huggingface/trl/pull/1794
* Clean examples by @qgallouedec in https://github.com/huggingface/trl/pull/1791
* Remove extra print in reward_trainer.py by @mnoukhov in https://github.com/huggingface/trl/pull/1799
* Fix `torch_dtype` handling in `{DPO,SFT}Trainer` when provided via CLI by @alvarobartt in https://github.com/huggingface/trl/pull/1807
* Fix `TRL_USE_RICH` environment variable handling by @alvarobartt in https://github.com/huggingface/trl/pull/1808
* 0.9.6 release by @vwxyzjn in https://github.com/huggingface/trl/pull/1816

## New Contributors
* @McPatate made their first contribution in https://github.com/huggingface/trl/pull/1721
* @jetlime made their first contribution in https://github.com/huggingface/trl/pull/1724
* @imelnyk made their first contribution in https://github.com/huggingface/trl/pull/1701
* @AIR-hl made their first contribution in https://github.com/huggingface/trl/pull/1753
* @1485840691 made their first contribution in https://github.com/huggingface/trl/pull/1610
* @idanshen made their first contribution in https://github.com/huggingface/trl/pull/1742
* @mertsayar8 made their first contribution in https://github.com/huggingface/trl/pull/1718
* @scottsuk0306 made their first contribution in https://github.com/huggingface/trl/pull/1758
* @mihirp1998 made their first contribution in https://github.com/huggingface/trl/pull/1585
* @haozheji made their first contribution in https://github.com/huggingface/trl/pull/1735
* @Mubin17 made their first contribution in https://github.com/huggingface/trl/pull/1774
* @detsutut made their first contribution in https://github.com/huggingface/trl/pull/1788
* @noahlt made their first contribution in https://github.com/huggingface/trl/pull/1794

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.9.4...v0.9.6

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.9.6)

---

## v0.9.4: v0.9.4
**Published:** 2024-06-06

Mainly backward compatibility fixes with SFTTrainer. 


## What's Changed
* Fixed doc string and related docs for the SFTConfig update by @GuilhermeFreire in https://github.com/huggingface/trl/pull/1706
* SFTTrainer: Fix backward Compatibility issue with `TrainingArguments` by @younesbelkada in https://github.com/huggingface/trl/pull/1707
* 0.9.4 release by @vwxyzjn in https://github.com/huggingface/trl/pull/1708

## New Contributors
* @GuilhermeFreire made their first contribution in https://github.com/huggingface/trl/pull/1706

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.9.3...v0.9.4

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.9.4)

---

## v0.9.3: v0.9.3 RLOO / PPOv2 Trainer, RM Visualization
**Published:** 2024-06-05

We are excited to introduce the new v0.9.3 release. Many new exciting features and algorithms. The highlights are as follows:


1. **RLOO Trainer**: RLOO (Reinforce Leave-one-out) is a new online RL algorithm for RLHF, proposed by [Ahmadian et al from Cohere](https://cohere.com/research/papers/back-to-basics-revisiting-reinforce-style-optimization-for-learning-from-human-feedback-in-llms-2024-02-23). Check out our docs [here](https://huggingface.co/docs/trl/rloo_trainer) to get started
2. **PPOv2 Trainer**: We are introducing a new experimental PPOv2 trainer which is more aligned with OpenAI's PPO implementation based on https://arxiv.org/abs/2403.17031. Check out our docs [here](https://huggingface.co/docs/trl/ppov2_trainer) to get started
3. **Reward model visualization**: the reward model training now includes visualization on the eval dataset, as shown below.

https://github.com/huggingface/trl/assets/5555347/6575a879-cb2f-4e2e-bb84-a76707f9de84

4. **New losses in the DPO Trainer**: DPOTrainer now includes losses / support for Self-play Preference Optimization, Robust DPO, TR-DPO, Iterative Reasoning Preference Optimization, and Pairwise Noise Contrastive Alignment
5. **New losses in the KTO Trainer**: KTOTrainer now includes the loss for Binary Classifier Optimization (BCO)







## What's Changed
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1568
* fix add_special_tokens issue for data with template by @edixiong in https://github.com/huggingface/trl/pull/1509
* [DPO] add 'bco_pair' loss_type by @seanexp in https://github.com/huggingface/trl/pull/1524
* [DPO] DPOConfig class by @kashif in https://github.com/huggingface/trl/pull/1554
* [SFT] add SFT Trainer Config dataclass by @kashif in https://github.com/huggingface/trl/pull/1530
* FIX: Fix CI on transformers main  by @younesbelkada in https://github.com/huggingface/trl/pull/1576
* [`SFTTrainer`] Add warning in SFTTrainer when dataset already processed  by @younesbelkada in https://github.com/huggingface/trl/pull/1577
* Fix typo detoxifying doc by @qgallouedec in https://github.com/huggingface/trl/pull/1594
* Core: removed unexisting `SftArgumentParser` by @younesbelkada in https://github.com/huggingface/trl/pull/1602
* [`KTOTrainer`] add BCO (reward shift and underlying distribution matching) by @seanexp in https://github.com/huggingface/trl/pull/1599
* [CLI] Use auto device map for model load by @lewtun in https://github.com/huggingface/trl/pull/1596
* Removing `tests/` from package data by @jamesbraza in https://github.com/huggingface/trl/pull/1607
* Docs: Fix build main documentation by @younesbelkada in https://github.com/huggingface/trl/pull/1604
* support loss function for Self-play Preference Optimization by @winglian in https://github.com/huggingface/trl/pull/1612
* Update HH dataset on helpful only subset by @vwxyzjn in https://github.com/huggingface/trl/pull/1613
* corrects loss function for Self-play Preference Optimization hard label version by @angelahzyuan in https://github.com/huggingface/trl/pull/1615
* Fix ZeRO-3 generation context manager by @lewtun in https://github.com/huggingface/trl/pull/1617
* fixed adding bos and eos token unconditionally by @jasonyux in https://github.com/huggingface/trl/pull/1591
* visualize rm prediction by @vwxyzjn in https://github.com/huggingface/trl/pull/1636
* [ORPO] Correct label mask for pad tokens by @IlyaGusev in https://github.com/huggingface/trl/pull/1625
* Update sft_llama2.py to work with the latest API by @xianbaoqian in https://github.com/huggingface/trl/pull/1637
* Fixed wrong logs prefixes in KTOTrainer by @bartoszzuk in https://github.com/huggingface/trl/pull/1641
* Pairwise Noise Contrastive Alignment by @winglian in https://github.com/huggingface/trl/pull/1632
* don't cast the trainable lora layers to half precision by @pacman100 in https://github.com/huggingface/trl/pull/1644
* PPO / Reinforce Trainers by @vwxyzjn in https://github.com/huggingface/trl/pull/1540
* Apply deprecated `evaluation_strategy` by @muellerzr in https://github.com/huggingface/trl/pull/1559
* FEAT: Add support for training collator in PPOTrainer by @younesbelkada in https://github.com/huggingface/trl/pull/1658
* Correct Documentation for cDPO Usage by @AliBakly in https://github.com/huggingface/trl/pull/1655
* Fix inheritance order in PPOv2Config by @Nicolinho in https://github.com/huggingface/trl/pull/1659
* [DPO] Add 'robust' loss_type by @Abilityguy in https://github.com/huggingface/trl/pull/1653
* 🤫 TR-DPO implementation by @syrn1k in https://github.com/huggingface/trl/pull/1593
* Do not upcast adapters when using FSDP+QLoRA by @pacman100 in https://github.com/huggingface/trl/pull/1654
* [Tests] update eval_strategy API by @kashif in https://github.com/huggingface/trl/pull/1662
* Fix ppov2 test case by @vwxyzjn in https://github.com/huggingface/trl/pull/1661
* FIX / PPO: Fix `enable_input_require_grads` issues with PPO models by @younesbelkada in https://github.com/huggingface/trl/pull/1664
* fix dataset load error by @sywangyi in https://github.com/huggingface/trl/pull/1670
* FIX / SFTTrainer: Fix SFTTrainer with `args=None` by @younesbelkada in https://github.com/huggingface/trl/pull/1678
* Fix max_completion_length for encoder_decoder models in KTO Trainer by @samuki in https://github.com/huggingface/trl/pull/1588
* intial RPO loss by @kashif in https://github.com/huggingface/trl/pull/1686
* Fix overriding optimize_device_cache with optimize_cuda_cache in PPOConfig by @alexisrozhkov in https://github.com/huggingface/trl/pull/1690
* Skip packing validation by @alex-jw-brooks in https://github.com/huggingface/trl/pull/1673
* Fix typo in DPOTrainer's warnings by @qgallouedec in https://github.com/huggingface/trl/pull/1688
* Quick fix on GPT4-eval by @vwxyzjn in https://github.com/huggingface/trl/pull/1696
* Release 0.9.2 by @vwxyzjn in https://github.com/huggingface/trl/pull/1697

## New Contributors
* @edixiong made their first contribution in https://github.com/huggingface/trl/pull/1509
* @seanexp made their first contribution in https://github.com/huggingface/trl/pull/1524
* @jamesbraza made their first contribution in https://github.com/huggingface/trl/pull/1607
* @winglian made their first contribution in https://github.com/huggingface/trl/pull/1612
* @angelahzyuan made their first contribution in https://github.com/huggingface/trl/pull/1615
* @jasonyux made their first contribution in https://github.com/huggingface/trl/pull/1591
* @IlyaGusev made their first contribution in https://github.com/huggingface/trl/pull/1625
* @xianbaoqian made their first contribution in https://github.com/huggingface/trl/pull/1637
* @bartoszzuk made their first contribution in https://github.com/huggingface/trl/pull/1641
* @muellerzr made their first contribution in https://github.com/huggingface/trl/pull/1559
* @AliBakly made their first contribution in https://github.com/huggingface/trl/pull/1655
* @Nicolinho made their first contribution in https://github.com/huggingface/trl/pull/1659
* @Abilityguy made their first contribution in https://github.com/huggingface/trl/pull/1653
* @syrn1k made their first contribution in https://github.com/huggingface/trl/pull/1593
* @alexisrozhkov made their first contribution in https://github.com/huggingface/trl/pull/1690
* @alex-jw-brooks made their first contribution in https://github.com/huggingface/trl/pull/1673

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.8.6...v0.9.2

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.9.3)

---

## v0.8.6: v0.8.6: Fixes for CLI
**Published:** 2024-04-22

## What's Changed

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1556
* [CLI] Update __init__.py imports by @kashif in https://github.com/huggingface/trl/pull/1557
* CLI: Add warning when ignored params are passed + parse config file if config if passed by @younesbelkada in https://github.com/huggingface/trl/pull/1565
* Release: v0.8.6 by @younesbelkada in https://github.com/huggingface/trl/pull/1567


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.8.5...v0.8.6

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.8.6)

---

## v0.8.5: v0.8.5: Important fixes for CLIs
**Published:** 2024-04-18

## What's Changed

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1548
* FIX: make the train / test fields modulable by @younesbelkada in https://github.com/huggingface/trl/pull/1551
* enable multiple eos tokens by @lvwerra in https://github.com/huggingface/trl/pull/1553
* Release: v0.8.5 by @younesbelkada in https://github.com/huggingface/trl/pull/1555

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.8.4...v0.8.5

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.8.5)

---

## v0.8.4: v0.8.4: CLI / CPO / KTO important fixes 
**Published:** 2024-04-17

This patch release includes important fixes for the CLI and KTO & CPO trainers

## What's Changed

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1529
* [CPO] fix memory leak due to retained value by @kashif in https://github.com/huggingface/trl/pull/1531
* VSFT hotfix - adds gen prompt to template and processor to hub by @edbeeching in https://github.com/huggingface/trl/pull/1532
* save_model -> save_pretrained in ppo_trainer.mdx by @ejmejm in https://github.com/huggingface/trl/pull/1537
* [KTO] support to load the adapter twice by @claralp in https://github.com/huggingface/trl/pull/1542
* CLI: Set `dataset_text_field` to `None` to allow ChatML automatic template by @younesbelkada in https://github.com/huggingface/trl/pull/1545
* FIX: Fix slow test by @younesbelkada in https://github.com/huggingface/trl/pull/1546
* Fixed ref model not used in PPO generation by @ejmejm in https://github.com/huggingface/trl/pull/1534
* Release: v0.8.4 by @younesbelkada in https://github.com/huggingface/trl/pull/1547

## New Contributors

* @ejmejm made their first contribution in https://github.com/huggingface/trl/pull/1537

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.8.3...v0.8.4

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.8.4)

---

## v0.8.3: v0.8.3: Patch release for CLI
**Published:** 2024-04-12

## What's Changed

This is a patch release that includes an import fix for CLIs

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1523
* [CLI] fix imports by @kashif in https://github.com/huggingface/trl/pull/1527
* Release: v0.8.3 by @younesbelkada in https://github.com/huggingface/trl/pull/1528


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.8.2...v0.8.3

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.8.3)

---

## v0.8.2: v0.8.2: ORPO & CPO Trainer / Vision LLMs support for `SFTTrainer`, KTO fixes
**Published:** 2024-04-11

# ORPO Trainer & Vision LLMs support for SFTTrainer, KTO fixes

This release includes two new trainers: [ORPO](https://huggingface.co/papers/2403.07691) from KAIST and [CPO](https://huggingface.co/papers/2401.08417)  
The release also includes Vision LLM such as Llava support for `SFTTrainer`, please see: https://github.com/huggingface/trl/blob/main/examples/scripts/vsft_llava.py for more details

## ORPO Trainer

* ORPO trainer by @kashif in https://github.com/huggingface/trl/pull/1435
* [ORPO] use log1p for loss by @kashif in https://github.com/huggingface/trl/pull/1491

## CPO Trainer

* Add CPOTrainer by @fe1ixxu in https://github.com/huggingface/trl/pull/1382
* Add `use_cache=False` in `{ORPO,CPO}Trainer.concatenated_forward` by @alvarobartt in https://github.com/huggingface/trl/pull/1478
* [ORPO] Update NLL loss to use `input_ids` instead by @alvarobartt in https://github.com/huggingface/trl/pull/1516

## VLLMs support for SFTTrainer

You can now use `SFTTrainer` to fine-tune VLLMs such as Llava ! 
See: https://github.com/huggingface/trl/blob/main/examples/scripts/vsft_llava.py for more details

* Adds VLM Training support to SFTTrainer + VSFT script by @edbeeching in https://github.com/huggingface/trl/pull/1518

## KTO Fixes

Many fixes were introduced for the KTOTrainer: 

* Update KTO example to use better model and ChatML support by @lewtun in https://github.com/huggingface/trl/pull/1485
* [KTO] Use batching to speed up data processing by @lewtun in https://github.com/huggingface/trl/pull/1470
* Update KTO example with good dataset & chat format by @lewtun in https://github.com/huggingface/trl/pull/1481
* [KTO] fix interleaving, reporting, and hanging bugs by @kawine and @claralp in https://github.com/huggingface/trl/pull/1499
* [KTO] fix metric logging by @claralp in https://github.com/huggingface/trl/pull/1514

## 10x PPO !

* Speed up PPO with ZeRO-3 by 10x 🔥 by @lewtun in https://github.com/huggingface/trl/pull/1483

## Other fixes

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1463
* Use the standard dataset for DPO CLI by @vwxyzjn in https://github.com/huggingface/trl/pull/1456
* [peft] Update test_reward_trainer.py to fix tests by @kashif in https://github.com/huggingface/trl/pull/1471
* Fix hyperparameters in KTO example by @lewtun in https://github.com/huggingface/trl/pull/1474
* docs: add missing Trainer classes and sort alphabetically by @anakin87 in https://github.com/huggingface/trl/pull/1479
* hackey update to ModelConfig to allow lora_target_modules="all-linear" by @galtay in https://github.com/huggingface/trl/pull/1488
* Ignore chat files by @lewtun in https://github.com/huggingface/trl/pull/1486
* Add DPO link in README by @qgallouedec in https://github.com/huggingface/trl/pull/1502
* Fix typo in how_to_train.md by @ftorres16 in https://github.com/huggingface/trl/pull/1503
* Fix DPO Unsloth example in Docs by @arnavgarg1 in https://github.com/huggingface/trl/pull/1494
* Correct ppo_epochs usage by @muhammed-shihebi in https://github.com/huggingface/trl/pull/1480
* Fix `RichProgressCallback` by @eggry in https://github.com/huggingface/trl/pull/1496
* Change the device index to device:index by @yuanwu2017 in https://github.com/huggingface/trl/pull/1490
* FIX: use kwargs for RMTrainer by @younesbelkada in https://github.com/huggingface/trl/pull/1515
* Allow streaming (datasets.IterableDataset) by @BramVanroy in https://github.com/huggingface/trl/pull/1468
* Allow pre-tokenized datasets in SFTTrainer by @BramVanroy in https://github.com/huggingface/trl/pull/1520
* [DOC] Add data description for sfttrainer doc by @BramVanroy in https://github.com/huggingface/trl/pull/1521
* Release: v0.8.2 by @younesbelkada in https://github.com/huggingface/trl/pull/1522

## New Contributors
* @fe1ixxu made their first contribution in https://github.com/huggingface/trl/pull/1382
* @anakin87 made their first contribution in https://github.com/huggingface/trl/pull/1479
* @galtay made their first contribution in https://github.com/huggingface/trl/pull/1488
* @qgallouedec made their first contribution in https://github.com/huggingface/trl/pull/1502
* @ftorres16 made their first contribution in https://github.com/huggingface/trl/pull/1503
* @arnavgarg1 made their first contribution in https://github.com/huggingface/trl/pull/1494
* @muhammed-shihebi made their first contribution in https://github.com/huggingface/trl/pull/1480
* @eggry made their first contribution in https://github.com/huggingface/trl/pull/1496
* @claralp made their first contribution in https://github.com/huggingface/trl/pull/1514

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.8.1...v0.8.2

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.8.2)

---

## v0.8.1: v0.8.1: Patch release for CLIs
**Published:** 2024-03-20

This patch release includes some important fixes for CLIs

## What's Changed

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1454
* Fix chat CLI for model revisions by @lewtun in https://github.com/huggingface/trl/pull/1458
* [chat] add eos token to generate by @lvwerra in https://github.com/huggingface/trl/pull/1459
* Release: v0.8.1 by @younesbelkada in https://github.com/huggingface/trl/pull/1462


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.8.0...v0.8.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.8.1)

---

## v0.8.0: v0.8.0: KTOTrainer, TRL CLIs, QLoRA + FSDP !
**Published:** 2024-03-19

## New Trainer: KTOTrainer: 

We recently introduced the KTOTrainer in order to run KTO algorithms on LLMs !

* fix bugs in KTO implementation by @kawine in https://github.com/huggingface/trl/pull/1380
* [KTO] merge eval dataset only if it exists by @kashif in https://github.com/huggingface/trl/pull/1383
* [KTO] prevent nans from appearing in metrics by @kawine in https://github.com/huggingface/trl/pull/1386
* Kto trainer by @kashif in https://github.com/huggingface/trl/pull/1181
* [KTO] fix tokenization bugs by @kawine in https://github.com/huggingface/trl/pull/1418
* [KTO] model init when args are given by @kashif in https://github.com/huggingface/trl/pull/1413
* [KTO] fix various bugs by @kawine in https://github.com/huggingface/trl/pull/1402

## TRL Command Line Interfaces (CLIs):

Run SFT, DPO and chat with your aligned model directly from the terminal:

**SFT:**

```bash
trl sft --model_name_or_path facebook/opt-125m --dataset_name imdb --output_dir opt-sft-imdb
```

**DPO:**

```bash
trl dpo --model_name_or_path facebook/opt-125m --dataset_name trl-internal-testing/Anthropic-hh-rlhf-processed --output_dir opt-sft-hh-rlhf 
```

**Chat:**

```bash
trl chat --model_name_or_path Qwen/Qwen1.5-0.5B-Chat
```

Read more about CLI in the [relevant documentation section](https://huggingface.co/docs/trl/main/en/clis) or use `--help` for more details.

* FEAT: Add CLIs in TRL !  by @younesbelkada in https://github.com/huggingface/trl/pull/1419
* CI / CLI: Properly raise error when CLI tests failed by @younesbelkada in https://github.com/huggingface/trl/pull/1446
* chat cli by @lvwerra in https://github.com/huggingface/trl/pull/1431
* Fix yaml parsing issue by @younesbelkada in https://github.com/huggingface/trl/pull/1450
* `model` --> `model_name_or_path` by @lvwerra in https://github.com/huggingface/trl/pull/1452
* FEAT: Update README to add DPO + CLIs by @younesbelkada in https://github.com/huggingface/trl/pull/1448

## FSDP + QLoRA:

SFTTrainer now supports FSDP + QLoRA

* Add support for FSDP+QLoRA and DeepSpeed ZeRO3+QLoRA by @pacman100 in https://github.com/huggingface/trl/pull/1416

## Other fixes

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1332
* Update stack llama 2 example to reflect #aa35fec by @nautsimon in https://github.com/huggingface/trl/pull/1333
* FIX: More user friendly error when users don't have PEFT by @younesbelkada in https://github.com/huggingface/trl/pull/1350
* fix 8-bit multi-gpu training bug  by @fancyerii in https://github.com/huggingface/trl/pull/1353
* set seed in sft/dpo/reward_modeling to make result reproducable by @sywangyi in https://github.com/huggingface/trl/pull/1357
* Fix transformers version checking for Python < 3.8 by @samuki in https://github.com/huggingface/trl/pull/1363
* Add some arguments for support XPU by @yuanwu2017 in https://github.com/huggingface/trl/pull/1366
* ENH: Send Docker and transformers main CI results on slack after merging on main by @younesbelkada in https://github.com/huggingface/trl/pull/1370
* FEAT: [`SFTTrainer`] Add  `eval_packing` by @younesbelkada in https://github.com/huggingface/trl/pull/1369
* FEAT: `force_use_ref_model` for power users by @younesbelkada in https://github.com/huggingface/trl/pull/1367
* FIX: fix after #1370 by @younesbelkada in https://github.com/huggingface/trl/pull/1372
* FIX: Change ci to fail-fast=False by @younesbelkada in https://github.com/huggingface/trl/pull/1373
* FIX: Fix the CI again ..  by @younesbelkada in https://github.com/huggingface/trl/pull/1374
* Log ddpo reward as float to fix numpy conversion during bf16 training by @skavulya in https://github.com/huggingface/trl/pull/1391
* Fix the pad_token_id error by @yuanwu2017 in https://github.com/huggingface/trl/pull/1394
* FIX [`RewardModeling`] Fix RM script for PEFT by @younesbelkada in https://github.com/huggingface/trl/pull/1393
* Fix import error from  deprecation in transformers by @lewtun in https://github.com/huggingface/trl/pull/1415
* CI: Fix CI on main by @younesbelkada in https://github.com/huggingface/trl/pull/1422
* [Kto] torch_dtype kwargs fix by @kashif in https://github.com/huggingface/trl/pull/1429
* Create standard dataset for TRL by @vwxyzjn in https://github.com/huggingface/trl/pull/1424
* FIX: fix doc build on main by @younesbelkada in https://github.com/huggingface/trl/pull/1437
* Fix PPOTrainer README example by @nikihowe in https://github.com/huggingface/trl/pull/1441
* Before update the tr_loss, make sure tr_loss_step is in the same device.  by @pengwei715 in https://github.com/huggingface/trl/pull/1439
* Release: v0.8.0 by @younesbelkada in https://github.com/huggingface/trl/pull/1453

## New Contributors
* @nautsimon made their first contribution in https://github.com/huggingface/trl/pull/1333
* @fancyerii made their first contribution in https://github.com/huggingface/trl/pull/1353
* @samuki made their first contribution in https://github.com/huggingface/trl/pull/1363
* @yuanwu2017 made their first contribution in https://github.com/huggingface/trl/pull/1366
* @kawine made their first contribution in https://github.com/huggingface/trl/pull/1380
* @skavulya made their first contribution in https://github.com/huggingface/trl/pull/1391
* @pengwei715 made their first contribution in https://github.com/huggingface/trl/pull/1439

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.11...v0.8.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.8.0)

---

## v0.7.11: v0.7.11: IPO & DPO fixes, faster data processing for multi-GPU, Automatic tagging for all models
**Published:** 2024-02-16

## DPO important fixes

We fixed issues with respect to IPO loss, leading to consistent results according to newest experiements:

* [DPO] average_log_prob when loss is IPO by @kashif in https://github.com/huggingface/trl/pull/1265

We also fixed important bugs with respect to DPO / PEFT and Flash Attention

* [`DPOTrainer`] Fix DPO trainer + mistral + FA2 by @younesbelkada in https://github.com/huggingface/trl/pull/1290

Data processing is now faster for multi-GPU envs

* [`DPOTrainer`] Load data only on main process + fix dpo example test by @younesbelkada in https://github.com/huggingface/trl/pull/1291
* Add multiprocessing in the DPO trainer. by @imraviagrawal in https://github.com/huggingface/trl/pull/1286

Other DPO bugfixes:

* [`PEFT` + `DPO`] Raise value error if one passes a ref_model and a peft_config by @younesbelkada in https://github.com/huggingface/trl/pull/1289
* Fix wrong variable name in DPOTrainer documentation example by @ouhenio in https://github.com/huggingface/trl/pull/1280
* fix padding in dpo trainer by @pacman100 in https://github.com/huggingface/trl/pull/1284 
* Fix AttributeError in dpo_trainer for reference_free case in dpo_loss function by @maliozer in https://github.com/huggingface/trl/pull/1313
* [DPOTrainer] Add multiprocessing for the eval_dataset map by @esceptico in https://github.com/huggingface/trl/pull/1307

## Faster data processing and other enhancements:

* Only load data on main process by @JohnGiorgi in https://github.com/huggingface/trl/pull/1255
* Remove tyro by @vwxyzjn in https://github.com/huggingface/trl/pull/1176

## Automatic tagging for all models

Models now gets tagged correctly even if users do not call `trainer.push_to_hub()`

* [`core` /  `xxxTrainer`] Automatic tagging by @younesbelkada in https://github.com/huggingface/trl/pull/1329

## What's Changed

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1254
* Update Model Generation config to reflect new special tokens by @philschmid in https://github.com/huggingface/trl/pull/1256
* Fix a typo in variable name by @otlaitil in https://github.com/huggingface/trl/pull/1269
* FIx SFTTrainer bugs on TRL main by @younesbelkada in https://github.com/huggingface/trl/pull/1276
* Fix SFT tuner in CI by @vwxyzjn in https://github.com/huggingface/trl/pull/1278
* Fix sft ci by @vwxyzjn in https://github.com/huggingface/trl/pull/1279
* Fix DPO slow tests by @younesbelkada in https://github.com/huggingface/trl/pull/1292
* Fix sft trainer when args is None by @younesbelkada in https://github.com/huggingface/trl/pull/1295
* Fix `DPOTrainer` docstrings by @alvarobartt in https://github.com/huggingface/trl/pull/1298
* Types: Fix PEP 484 implicit-optional compliance by @akx in https://github.com/huggingface/trl/pull/1297
* Update sft_trainer.mdx to add note on launching DDP training by @johnowhitaker in https://github.com/huggingface/trl/pull/1308
* Codemod Unittest assertions to bare asserts by @akx in https://github.com/huggingface/trl/pull/1301
* ENH: Run CI only if relevant files are modified by @younesbelkada in https://github.com/huggingface/trl/pull/1309
* Fix typos in docs for Multi Adapter RL (MARL). by @elhusseiniali in https://github.com/huggingface/trl/pull/1312
* Fix doc snippet PPOTrainer argument train_dataset -> dataset by @j-cb in https://github.com/huggingface/trl/pull/1321
* Best practice recommendation update for dpo_trainer.mdx by @R-seny in https://github.com/huggingface/trl/pull/1325
* pre-commit: replace linters + formatters with Ruff; fix some issues by @akx in https://github.com/huggingface/trl/pull/1300
* Update README.md to clarify model requirement by @markstur in https://github.com/huggingface/trl/pull/1315
* [`core` / `DDPO`] Fix diffusers import issue by @younesbelkada in https://github.com/huggingface/trl/pull/1314
* [`CI`] Add tests on transformers peft main on push main by @younesbelkada in https://github.com/huggingface/trl/pull/1328
* Release: v0.7.11 by @younesbelkada in https://github.com/huggingface/trl/pull/1331

## New Contributors

* @otlaitil made their first contribution in https://github.com/huggingface/trl/pull/1269
* @JohnGiorgi made their first contribution in https://github.com/huggingface/trl/pull/1255
* @ouhenio made their first contribution in https://github.com/huggingface/trl/pull/1280
* @imraviagrawal made their first contribution in https://github.com/huggingface/trl/pull/1286
* @akx made their first contribution in https://github.com/huggingface/trl/pull/1297
* @esceptico made their first contribution in https://github.com/huggingface/trl/pull/1307
* @johnowhitaker made their first contribution in https://github.com/huggingface/trl/pull/1308
* @elhusseiniali made their first contribution in https://github.com/huggingface/trl/pull/1312
* @maliozer made their first contribution in https://github.com/huggingface/trl/pull/1313
* @j-cb made their first contribution in https://github.com/huggingface/trl/pull/1321
* @R-seny made their first contribution in https://github.com/huggingface/trl/pull/1325
* @markstur made their first contribution in https://github.com/huggingface/trl/pull/1315

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.10...v0.7.11

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.11)

---

## v0.7.10: v0.7.10: Automatic templating, `setup_chat_format` API, stronger tests
**Published:** 2024-01-19

# v0.7.10: Minor fixes, Automatic templating, `setup_chat_format` API, stronger tests

This Patch release adds a new feature in TRL for dealing with chat datasets - you can load a directly formatted dataset without the need of formatting it beforehand.

Read more about it here: https://huggingface.co/docs/trl/sft_trainer#dataset-format-support

The release also introduces a new API `setup_chat_format` to correctly resize the model embeddings with the target size when adding new tokens to comply with the chat format. Currently we only support `chatml` format and we can add more formats in the future

Read more about it here: https://huggingface.co/docs/trl/sft_trainer#add-special-tokens-for-chat-format

We also extensively test SFTTrainer and DPOTrainer and the example scripts, `dpo.py` and `sft.py` should be well -battletested. If you see any issue with the script, please let us know on GitHub.

## What's Changed

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1207
* Check tokenize params on DPOTrainer by @pablovicente in https://github.com/huggingface/trl/pull/1197
* Fix shape descriptions in calculate_loss method by @yuta0x89 in https://github.com/huggingface/trl/pull/1204
* Fix FSDP error by @mgerstgrasser in https://github.com/huggingface/trl/pull/1196
* Update Unsloth SFT, DPO docs by @danielhanchen in https://github.com/huggingface/trl/pull/1213
* Fix args type by @zspo in https://github.com/huggingface/trl/pull/1214
* [`core` / `Docker`] Add  workflow to build TRL docker images by @younesbelkada in https://github.com/huggingface/trl/pull/1215
* Refactor RewardConfig to own module by @lewtun in https://github.com/huggingface/trl/pull/1221
* Add support for ChatML dataset format in by @philschmid in https://github.com/huggingface/trl/pull/1208
* Add slow test workflow file by @younesbelkada in https://github.com/huggingface/trl/pull/1223
* Remove a repeating line in how_to_train.md by @kykim0 in https://github.com/huggingface/trl/pull/1226
* Logs metrics on all distributed processes when using DPO & FSDP by @AjayP13 in https://github.com/huggingface/trl/pull/1160
* fix: improve error message when `pad_token_id` is not configured by @yumemio in https://github.com/huggingface/trl/pull/1152
* [`core` / tests ] v1 slow tests by @younesbelkada in https://github.com/huggingface/trl/pull/1218
* [`core` / SFTTrainer] Fix breaking change by @younesbelkada in https://github.com/huggingface/trl/pull/1229
* Fixes slow tests by @younesbelkada in https://github.com/huggingface/trl/pull/1241
* Fix weird doc bug by @younesbelkada in https://github.com/huggingface/trl/pull/1244
* Add `setup_chat_format` for adding new special tokens to model for training chat models by @philschmid in https://github.com/huggingface/trl/pull/1242
* Fix chatml template by @philschmid in https://github.com/huggingface/trl/pull/1248
* fix: fix loss_type and some args desc by @zspo in https://github.com/huggingface/trl/pull/1247
* Release: v0.7.10 by @younesbelkada in https://github.com/huggingface/trl/pull/1253

## New Contributors
* @yuta0x89 made their first contribution in https://github.com/huggingface/trl/pull/1204
* @danielhanchen made their first contribution in https://github.com/huggingface/trl/pull/1213
* @zspo made their first contribution in https://github.com/huggingface/trl/pull/1214
* @philschmid made their first contribution in https://github.com/huggingface/trl/pull/1208
* @kykim0 made their first contribution in https://github.com/huggingface/trl/pull/1226
* @AjayP13 made their first contribution in https://github.com/huggingface/trl/pull/1160
* @yumemio made their first contribution in https://github.com/huggingface/trl/pull/1152

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.9...v0.7.10

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.10)

---

## v0.7.9: v0.7.9: Patch release for DPO & SFTTrainer
**Published:** 2024-01-09

# v0.7.9: Patch release for DPO & SFTTrainer

This is a patch release that fixes critical issues with SFTTrainer & DPOTrainer, together with minor fixes for PPOTrainer and DataCollatorForCompletionOnlyLM

## What's Changed

* Release: v0.7.8 by @younesbelkada in https://github.com/huggingface/trl/pull/1200
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1201
* Fix instruction token masking by @mgerstgrasser in https://github.com/huggingface/trl/pull/1185
* Fix reported KL in PPO trainer by @mgerstgrasser in https://github.com/huggingface/trl/pull/1180
* [`DPOTrainer`] Fix peft + DPO + bf16 if one uses `generate_during_eval` or pre-computed logits by @younesbelkada in https://github.com/huggingface/trl/pull/1203
* Revert "Address issue #1122" by @younesbelkada in https://github.com/huggingface/trl/pull/1205
* Release: v0.7.9 by @younesbelkada in https://github.com/huggingface/trl/pull/1206


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.8...v0.7.9

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.9)

---

## v0.7.8: v0.7.8: Unsloth tag, DPO fixes, PEFT support for DDPO 
**Published:** 2024-01-09

# v0.7.8: Unsloth tag, DPO fixes, PEFT support for DDPO 

## Unsloth tag for `xxxTrainer`

If users use Unsloth library, the `unsloth` tag gets automatically pushed on the Hub.

* [`xxxTrainer`] Add unsloth tag by @younesbelkada in https://github.com/huggingface/trl/pull/1130

## DPO fixes

Some important fixes for DPO has been introduced to address: https://twitter.com/jon_durbin/status/1743575483365699809 and to make DPO faster

* Allow separate devices for target/ref models. by @jondurbin in https://github.com/huggingface/trl/pull/1190
* Allow swapping PEFT adapters for target/ref model. by @jondurbin in https://github.com/huggingface/trl/pull/1193
* Change device access order for speedup of calculating metrics in DPOTrainer by @brcps12 in https://github.com/huggingface/trl/pull/1154

## DDPO + PEFT

Now DDPO supports PEFT

* add: support for `peft` in ddpo. by @sayakpaul in https://github.com/huggingface/trl/pull/1165

## Other fixes

* add peft_module_casting_to_bf16 in DPOTrainer by @sywangyi in https://github.com/huggingface/trl/pull/1143
* SFT Tokenizer Fix by @ChrisCates in https://github.com/huggingface/trl/pull/1142
* Minor fixes to some comments in some examples. by @mattholl in https://github.com/huggingface/trl/pull/1156
* Correct shapes in docstring of PPOTrainer's train_minibatch method by @nikihowe in https://github.com/huggingface/trl/pull/1170
* Update sft_trainer.py by @Hemanthkumar2112 in https://github.com/huggingface/trl/pull/1162
* Fix batch all gather by @vwxyzjn in https://github.com/huggingface/trl/pull/1177
* Address issue #1122 by @maneandrea in https://github.com/huggingface/trl/pull/1174
* Fix misleading variable "epoch" from the training loop from PPOTrainer Doc. by @Jfhseh in https://github.com/huggingface/trl/pull/1171
* SFTTrainer: follow args.remove_unused_columns by @mgerstgrasser in https://github.com/huggingface/trl/pull/1188
* Handle last token from generation prompt by @pablovicente in https://github.com/huggingface/trl/pull/1153

## New Contributors
* @ChrisCates made their first contribution in https://github.com/huggingface/trl/pull/1142
* @brcps12 made their first contribution in https://github.com/huggingface/trl/pull/1154
* @mattholl made their first contribution in https://github.com/huggingface/trl/pull/1156
* @sayakpaul made their first contribution in https://github.com/huggingface/trl/pull/1165
* @nikihowe made their first contribution in https://github.com/huggingface/trl/pull/1170
* @Hemanthkumar2112 made their first contribution in https://github.com/huggingface/trl/pull/1162
* @maneandrea made their first contribution in https://github.com/huggingface/trl/pull/1174
* @Jfhseh made their first contribution in https://github.com/huggingface/trl/pull/1171
* @mgerstgrasser made their first contribution in https://github.com/huggingface/trl/pull/1188
* @pablovicente made their first contribution in https://github.com/huggingface/trl/pull/1153
* @jondurbin made their first contribution in https://github.com/huggingface/trl/pull/1190

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.7...v0.7.8

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.8)

---

## v0.7.7: v0.7.7
**Published:** 2023-12-26

# v0.7.7: Patch release PPO & DDPO tags

A fix has been introduce to fix a breaking change with `PPOTrainer.push_to_hub()` and `DDPOTrainer.push_to_hub()`

* [`PPOTrainer` / `DDPOTrainer`] Fix ppo & ddpo push to Hub by @younesbelkada in https://github.com/huggingface/trl/pull/1141

## What's Changed

* Release: v0.7.6 by @younesbelkada in https://github.com/huggingface/trl/pull/1134
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1135
* clear up the parameters of supervised_finetuning.py by @sywangyi in https://github.com/huggingface/trl/pull/1126
* Add type hints to core.py by @zachschillaci27 in https://github.com/huggingface/trl/pull/1097
* fix_ddpo_demo by @zhangsibo1129 in https://github.com/huggingface/trl/pull/1129
* Add npu support for ppo example by @zhangsibo1129 in https://github.com/huggingface/trl/pull/1128

## New Contributors
* @zachschillaci27 made their first contribution in https://github.com/huggingface/trl/pull/1097
* @zhangsibo1129 made their first contribution in https://github.com/huggingface/trl/pull/1129

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.6...v0.7.7

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.7)

---

## v0.7.6: v0.7.6: Patch release - Multi-tag instead of single tags for `xxxTrainer`
**Published:** 2023-12-22

# Patch release: Multi-tag instead of single tags for `xxxTrainer`

This is a patch release to push multiple tags (e.g. `trl` & `sft`) instead of one tag

## What's Changed

* Release: v0.7.5 by @younesbelkada in https://github.com/huggingface/trl/pull/1131
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/1132
* [`xxxTrainer`] multi-tags support for tagging by @younesbelkada in https://github.com/huggingface/trl/pull/1133


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.5...v0.7.6

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.6)

---

## v0.7.5: v0.7.5: IPO & KTO & cDPO loss, `DPOTrainer` enhancements, automatic tags for `xxxTrainer`
**Published:** 2023-12-22

# IPO & KTO & cDPO loss, `DPOTrainer` enhancements, automatic tags for `xxxTrainer`

## Important enhancements for `DPOTrainer`

This release introduces many new features in TRL for `DPOTrainer`: 

- IPO-loss for a better generalization of DPO algorithm
- KTO & cDPO loss
- You can also pass pre-computed logits to `DPOTrainer`

* [DPO] Refactor eval logging of dpo trainer by @mnoukhov in https://github.com/huggingface/trl/pull/954
* Fixes reward and text gathering in distributed training by @edbeeching in https://github.com/huggingface/trl/pull/850
* remove spurious optimize_cuda_cache deprecation warning on init by @ChanderG in https://github.com/huggingface/trl/pull/1045
* Revert "[DPO] Refactor eval logging of dpo trainer (#954)" by @lvwerra in https://github.com/huggingface/trl/pull/1047
* Fix DPOTrainer + PEFT 2 by @rdk31 in https://github.com/huggingface/trl/pull/1049
* [DPO] IPO Training loss by @kashif in https://github.com/huggingface/trl/pull/1022
* [DPO] cDPO loss by @kashif in https://github.com/huggingface/trl/pull/1035
* [DPO] use ref model logprobs if it exists in the data by @kashif in https://github.com/huggingface/trl/pull/885
* [DP0] save eval_dataset for subsequent calls by @kashif in https://github.com/huggingface/trl/pull/1125
* [DPO] rename kto loss by @kashif in https://github.com/huggingface/trl/pull/1127
* [DPO] add KTO loss by @kashif in https://github.com/huggingface/trl/pull/1075

## Automatic `xxxTrainer` tagging on the Hub

Now, trainers from TRL pushes automatically tags `trl-sft`, `trl-dpo`, `trl-ddpo` when pushing models on the Hub

* [`xxxTrainer`] Add tags to all trainers in TRL by @younesbelkada in https://github.com/huggingface/trl/pull/1120

## unsloth 🤝 TRL

We encourage users to try out [unsloth library](https://github.com/unslothai/unsloth) for faster LLM fine-tuning using PEFT & TRL's SFTTrainer and DPOTrainer

* [`Docs`] Add unsloth optimizations in TRL's documentation by @younesbelkada in https://github.com/huggingface/trl/pull/1119

## What's Changed

* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/970
* [`Tests`] Add non optional packages tests by @younesbelkada in https://github.com/huggingface/trl/pull/974
* [DOCS] Fix outdated references to `examples/` by @alvarobartt in https://github.com/huggingface/trl/pull/977
* Update README.md by @GeekDream-x in https://github.com/huggingface/trl/pull/994
* [DataCollatorForCompletionOnlyLM] Warn on identical `eos_token_id` and `pad_token_id` by @MustSave in https://github.com/huggingface/trl/pull/988
* [`DataCollatorForCompletionOnlyLM`] Add more clarification / guidance in the case `tokenizer.pad_token_id == tokenizer.eos_token_id` by @younesbelkada in https://github.com/huggingface/trl/pull/992
* make distributed true for multiple process by @allanj in https://github.com/huggingface/trl/pull/997
* Fixed wrong trigger for warning by @zabealbe in https://github.com/huggingface/trl/pull/971
* Update how_to_train.md by @halfrot in https://github.com/huggingface/trl/pull/1003
* Adds `requires_grad` to input for non-quantized peft models by @younesbelkada in https://github.com/huggingface/trl/pull/1006
* [Multi-Adapter PPO] Fix and Refactor reward model adapter  by @mnoukhov in https://github.com/huggingface/trl/pull/982
* Remove duplicate data loading in rl_training.py by @viethoangtranduong in https://github.com/huggingface/trl/pull/1020
* [Document] Minor fixes of sft_trainer document by @mutichung in https://github.com/huggingface/trl/pull/1029
* Update utils.py by @ZihanWang314 in https://github.com/huggingface/trl/pull/1012
* spelling is hard by @grahamannett in https://github.com/huggingface/trl/pull/1043
* Fixing accelerator version function call. by @ParthaEth in https://github.com/huggingface/trl/pull/1056
* [SFT Trainer] precompute packed iterable into a dataset by @lvwerra in https://github.com/huggingface/trl/pull/979
* Update doc CI by @lewtun in https://github.com/huggingface/trl/pull/1060
* Improve PreTrainedModelWrapper._get_current_device by @billvsme in https://github.com/huggingface/trl/pull/1048
* Update doc for the computer_metrics argument of SFTTrainer by @albertauyeung in https://github.com/huggingface/trl/pull/1062
* [`core`] Fix failing tests on main by @younesbelkada in https://github.com/huggingface/trl/pull/1065
* [`SFTTrainer`] Fix Trainer when args is None by @younesbelkada in https://github.com/huggingface/trl/pull/1064
* enable multiple eval datasets by @peter-sk in https://github.com/huggingface/trl/pull/1052
* Add missing `loss_type` in `ValueError` message by @alvarobartt in https://github.com/huggingface/trl/pull/1067
* Add args to SFT example by @lewtun in https://github.com/huggingface/trl/pull/1079
* add local folder support as input for rl_training. by @sywangyi in https://github.com/huggingface/trl/pull/1078
* Make CI happy by @younesbelkada in https://github.com/huggingface/trl/pull/1080
* Removing `tyro` in `sft_llama2.py` by @vwxyzjn in https://github.com/huggingface/trl/pull/1081
* Log arg consistency by @tcapelle in https://github.com/huggingface/trl/pull/1084
* Updated documentation for docs/source/reward_trainer.mdx to import th… by @cm2435 in https://github.com/huggingface/trl/pull/1092
* [Feature] Add Ascend NPU accelerator support  by @statelesshz in https://github.com/huggingface/trl/pull/1096
* `peft_module_casting_to_bf16` util method, `append_concat_token` flag, remove callback `PeftSavingCallback` by @pacman100 in https://github.com/huggingface/trl/pull/1110
* Make prepending of bos token configurable. by @pacman100 in https://github.com/huggingface/trl/pull/1114
* fix gradient checkpointing when using PEFT by @pacman100 in https://github.com/huggingface/trl/pull/1118
* Update `description` in `setup.py` by @alvarobartt in https://github.com/huggingface/trl/pull/1101

## New Contributors

* @alvarobartt made their first contribution in https://github.com/huggingface/trl/pull/977
* @GeekDream-x made their first contribution in https://github.com/huggingface/trl/pull/994
* @MustSave made their first contribution in https://github.com/huggingface/trl/pull/988
* @allanj made their first contribution in https://github.com/huggingface/trl/pull/997
* @zabealbe made their first contribution in https://github.com/huggingface/trl/pull/971
* @viethoangtranduong made their first contribution in https://github.com/huggingface/trl/pull/1020
* @mutichung made their first contribution in https://github.com/huggingface/trl/pull/1029
* @ZihanWang314 made their first contribution in https://github.com/huggingface/trl/pull/1012
* @grahamannett made their first contribution in https://github.com/huggingface/trl/pull/1043
* @ChanderG made their first contribution in https://github.com/huggingface/trl/pull/1045
* @rdk31 made their first contribution in https://github.com/huggingface/trl/pull/1049
* @ParthaEth made their first contribution in https://github.com/huggingface/trl/pull/1056
* @billvsme made their first contribution in https://github.com/huggingface/trl/pull/1048
* @albertauyeung made their first contribution in https://github.com/huggingface/trl/pull/1062
* @peter-sk made their first contribution in https://github.com/huggingface/trl/pull/1052
* @sywangyi made their first contribution in https://github.com/huggingface/trl/pull/1078
* @tcapelle made their first contribution in https://github.com/huggingface/trl/pull/1084
* @cm2435 made their first contribution in https://github.com/huggingface/trl/pull/1092
* @statelesshz made their first contribution in https://github.com/huggingface/trl/pull/1096
* @pacman100 made their first contribution in https://github.com/huggingface/trl/pull/1110

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.4...v0.7.5

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.5)

---

## v0.7.4: v0.7.4: Patch Release
**Published:** 2023-11-10

# Patch Release

This release is a patch release that addresses an issue for users that have TRL installed without PEFT

## What's Changed

* Release: v0.7.3 by @younesbelkada in https://github.com/huggingface/trl/pull/965
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/966
* [`core`] Fix peft config typehint by @younesbelkada in https://github.com/huggingface/trl/pull/967
* Pin bnb to <=0.41.1 by @younesbelkada in https://github.com/huggingface/trl/pull/968


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.3...v0.7.4

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.4)

---

## v0.7.3: v0.7.3:`IterativeTrainer`, NEFTune and major bugfixes for `DPOTrainer` and Distributed Training
**Published:** 2023-11-10

# `IterativeTrainer`, NEFTune and major bugfixes for `DPOTrainer` and Distributed Training

In this release we introduce two new features, `IterativeTrainer` from @gaetanlop and NEFTune, together with important bugfixes for distributed training. 

## IterativeTrainer

Iterative fine-tuning is a training method that enables to perform custom actions (generation and filtering for example) between optimization steps. In TRL we provide an easy-to-use API to fine-tune your models in an iterative way in just a few lines of code.

Read more about it here: https://huggingface.co/docs/trl/iterative_sft_trainer

* Introducing the Iterative Trainer by @gaetanlop in https://github.com/huggingface/trl/pull/737

## NEFTune

NEFTune is a technique to boost the performance of chat models and was introduced by the paper [“NEFTune: Noisy Embeddings Improve Instruction Finetuning”](https://arxiv.org/abs/2310.05914) from Jain et al. it consists of adding noise to the embedding vectors during training. According to the abstract of the paper:

* [`SFTTrainer`] Adds NEFTune into `SFTTrainer` by @younesbelkada in https://github.com/huggingface/trl/pull/871
* [`NEFTune`] Make use of forward hooks instead by @younesbelkada in https://github.com/huggingface/trl/pull/889
* Generalize NEFTune for FSDP, DDP, ... by @younesbelkada in https://github.com/huggingface/trl/pull/924
* [`NEFTune`] Make use of forward hooks instead by @younesbelkada in https://github.com/huggingface/trl/pull/889

Read more about it [here](https://huggingface.co/docs/trl/sft_trainer#enhance-models-performances-using-neftune)

## Major bugfixes

Major bugfixes have been addressed to tackle many issues with distributed training and gradient checkpointing.

* [`DPO`] fix DPO + GC issues by @younesbelkada in https://github.com/huggingface/trl/pull/927
* [`core` / `DDP`] Fix RM trainer + DDP + quantization + propagate `gradient_checkpointing_kwargs` in SFT & DPO by @younesbelkada in https://github.com/huggingface/trl/pull/912

## DPOTrainer enhancements and fixes

The DPOTrainer now comes with multiple enhancements and bugfixes! Check them out below

* [DPO] add SLiC hinge loss to DPOTrainer by @kashif in https://github.com/huggingface/trl/pull/866
* Fix DPOTrainer + PEFT  by @younesbelkada in https://github.com/huggingface/trl/pull/941
* [DPO] Merge initial peft model if trainer has a peft_config by @kashif in https://github.com/huggingface/trl/pull/956
* Adds model kwargs to SFT and DPO trainers by @edbeeching in https://github.com/huggingface/trl/pull/951
* fix: dpo trainer ds config by @mengban in https://github.com/huggingface/trl/pull/957
* hotfix for dpo trainer by @mnoukhov in https://github.com/huggingface/trl/pull/919
* Fix dpo_llama2.py by @younesbelkada in https://github.com/huggingface/trl/pull/934

## What's Changed

* Release: v0.7.2 by @younesbelkada in https://github.com/huggingface/trl/pull/863
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/864
* Remove duplicate key in `reward_modeling.py` by @vwxyzjn in https://github.com/huggingface/trl/pull/890
* fix peft_config type by @u2takey in https://github.com/huggingface/trl/pull/883
* fix: remove useless token by @rtrompier in https://github.com/huggingface/trl/pull/896
* [reward_modeling] Cleaning example script by @gaetanlop in https://github.com/huggingface/trl/pull/882
* Fix couple wrong links on lib homepage by @paulbricman in https://github.com/huggingface/trl/pull/908
* Add whiten ops before compute advatanges by @SingL3 in https://github.com/huggingface/trl/pull/887
* Fix broken link/markdown by @osanseviero in https://github.com/huggingface/trl/pull/903
* [Update reward_trainer.py] append PeftSavingCallback if callbacks is not None by @zuoxingdong in https://github.com/huggingface/trl/pull/910
* deactivate MacOS CI by @lvwerra in https://github.com/huggingface/trl/pull/913
* fix stackllama2 sft gradient checkpointing by @nrailg in https://github.com/huggingface/trl/pull/906
* updating PPOTrainer docstring by @lomahony in https://github.com/huggingface/trl/pull/897
* Bump minimum `tyro` version by @brentyi in https://github.com/huggingface/trl/pull/928
* [Feature] Enable Intel XPU  support by @abhilash1910 in https://github.com/huggingface/trl/pull/839
* [`SFTTrainer`] Make sure to not conflict between `transformers` and TRL implementation by @younesbelkada in https://github.com/huggingface/trl/pull/933
* Fix stale bot by @younesbelkada in https://github.com/huggingface/trl/pull/935
* Optionally logging reference response by @vwxyzjn in https://github.com/huggingface/trl/pull/847
* [`CI`] Fix CI with new transformers release by @younesbelkada in https://github.com/huggingface/trl/pull/946
* Fix unwrapping peft models by @kkteru in https://github.com/huggingface/trl/pull/948
* Added support for custom EncoderDecoder models by @ribesstefano in https://github.com/huggingface/trl/pull/911

## New Contributors

* @u2takey made their first contribution in https://github.com/huggingface/trl/pull/883
* @rtrompier made their first contribution in https://github.com/huggingface/trl/pull/896
* @paulbricman made their first contribution in https://github.com/huggingface/trl/pull/908
* @SingL3 made their first contribution in https://github.com/huggingface/trl/pull/887
* @nrailg made their first contribution in https://github.com/huggingface/trl/pull/906
* @lomahony made their first contribution in https://github.com/huggingface/trl/pull/897
* @brentyi made their first contribution in https://github.com/huggingface/trl/pull/928
* @abhilash1910 made their first contribution in https://github.com/huggingface/trl/pull/839
* @kkteru made their first contribution in https://github.com/huggingface/trl/pull/948
* @ribesstefano made their first contribution in https://github.com/huggingface/trl/pull/911
* @mengban made their first contribution in https://github.com/huggingface/trl/pull/957

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.2...v0.7.3

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.3)

---

## v0.7.2: v0.7.2
**Published:** 2023-10-12

# 0.7.2: Flash Attention documentation and Minor bugfixes 

In this release we provide minor bugfixes and smoother user experience for all public classes. We also added some clarification on the documentation on how to use Flash Attention with `SFTTrainer`

## How to use Flash Attention with `SFTTrainer`:

* Update sft_trainer.mdx to highlight Flash Attention features by @younesbelkada in https://github.com/huggingface/trl/pull/807

## What's Changed

* Release: v0.7.1 by @younesbelkada in https://github.com/huggingface/trl/pull/709
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/710
* fix device issue by @backpropper in https://github.com/huggingface/trl/pull/681
* Update docs on gms8k by @vwxyzjn in https://github.com/huggingface/trl/pull/711
* [`Docs`] Fix sft mistakes by @younesbelkada in https://github.com/huggingface/trl/pull/717
* Fix: RuntimeError: 'weight' must be 2-D issue by @jp1924 in https://github.com/huggingface/trl/pull/687
* Add pyproject.toml by @mnoukhov in https://github.com/huggingface/trl/pull/690
* [`core`] Bump peft to 0.4.0 by @younesbelkada in https://github.com/huggingface/trl/pull/720
* Refactor RewardTrainer hyperparameters into dedicated dataclass by @lewtun in https://github.com/huggingface/trl/pull/726
* Fix DeepSpeed ZeRO-3 in PPOTrainer by @lewtun in https://github.com/huggingface/trl/pull/730
* [`SFTTrainer`] Check correctly for condition by @younesbelkada in https://github.com/huggingface/trl/pull/668
* Add epsilon to score normalization by @zfang in https://github.com/huggingface/trl/pull/727
* Enable gradient checkpointing to be disabled for reward modelling by @lewtun in https://github.com/huggingface/trl/pull/725
* [DPO] fixed metrics typo by @kashif in https://github.com/huggingface/trl/pull/743
* Seq2Seq model support for DPO by @gaetanlop in https://github.com/huggingface/trl/pull/586
* [DPO] fix ref_model by @i4never in https://github.com/huggingface/trl/pull/745
* [`core`] Fix import of `randn_tensor` by @younesbelkada in https://github.com/huggingface/trl/pull/751
* Add benchmark CI by @vwxyzjn in https://github.com/huggingface/trl/pull/752
* update to `prepare_model_for_kbit_training` by @mnoukhov in https://github.com/huggingface/trl/pull/728
* benchmark CI fix by @vwxyzjn in https://github.com/huggingface/trl/pull/755
* EOS token processing for multi-turn DPO by @natolambert in https://github.com/huggingface/trl/pull/741
* Extend DeepSpeed integration to ZeRO-{1,2,3} by @lewtun in https://github.com/huggingface/trl/pull/758
* Imrpove benchmark ci by @vwxyzjn in https://github.com/huggingface/trl/pull/760
* [PPOTrainer] - add comment of zero masking (from second query token) by @zuoxingdong in https://github.com/huggingface/trl/pull/763
* Refactor and benchmark by @vwxyzjn in https://github.com/huggingface/trl/pull/662
* Benchmark CI (actual) by @vwxyzjn in https://github.com/huggingface/trl/pull/754
* docs: add initial version of docs for  `PPOTrainer` by @davidberenstein1957 in https://github.com/huggingface/trl/pull/665
* Support fork in benchmark CI by @vwxyzjn in https://github.com/huggingface/trl/pull/764
* Update benchmark.yml by @vwxyzjn in https://github.com/huggingface/trl/pull/773
* Benchmark CI fix by @vwxyzjn in https://github.com/huggingface/trl/pull/775
* Benchmark CI fix by @vwxyzjn in https://github.com/huggingface/trl/pull/776
* Update benchmark.yml by @vwxyzjn in https://github.com/huggingface/trl/pull/777
* Update benchmark.yml by @vwxyzjn in https://github.com/huggingface/trl/pull/778
* Update benchmark.yml by @vwxyzjn in https://github.com/huggingface/trl/pull/779
* Update benchmark.yml by @vwxyzjn in https://github.com/huggingface/trl/pull/780
* Update benchmark.yml by @vwxyzjn in https://github.com/huggingface/trl/pull/781
* Update benchmark.yml by @vwxyzjn in https://github.com/huggingface/trl/pull/782
* Ensure `RewardConfig` is backwards compatible by @lewtun in https://github.com/huggingface/trl/pull/748
* Temp benchmark ci dir by @vwxyzjn in https://github.com/huggingface/trl/pull/765
* Changed the default value of the `log_with` argument by @filippobistaffa in https://github.com/huggingface/trl/pull/792
* Add default Optim to DPO example by @natolambert in https://github.com/huggingface/trl/pull/759
* Add margin to RM training by @jvhoffbauer in https://github.com/huggingface/trl/pull/719
* [`DPO`] Revert "Add default Optim to DPO example (#759)" by @younesbelkada in https://github.com/huggingface/trl/pull/799
* Add deepspeed experiment by @vwxyzjn in https://github.com/huggingface/trl/pull/795
* [`Docs`] Clarify PEFT docs by @younesbelkada in https://github.com/huggingface/trl/pull/797
* Fix docs bug on  sft_trainer.mdx by @younesbelkada in https://github.com/huggingface/trl/pull/808
* [`PPOTrainer`] Fixes ppo trainer generate nit by @younesbelkada in https://github.com/huggingface/trl/pull/798
* Allow passing the token_ids as instruction_template in DataCollatorForCompletionOnlyLM by @devxpy in https://github.com/huggingface/trl/pull/749
* init custom eval loop for further DPO evals by @natolambert in https://github.com/huggingface/trl/pull/766
* Add RMSProp back to DPO by @natolambert in https://github.com/huggingface/trl/pull/821
* [DPO] add option for compute_metrics in DPOTrainer by @kashif in https://github.com/huggingface/trl/pull/822
* Small fixes to the PPO trainer doc and script. by @namin in https://github.com/huggingface/trl/pull/811
* Unify sentiment documentation by @vwxyzjn in https://github.com/huggingface/trl/pull/803
* Fix DeepSpeed ZeRO-{1,2} for DPOTrainer by @lewtun in https://github.com/huggingface/trl/pull/825
* Set trust remote code to false by default by @lewtun in https://github.com/huggingface/trl/pull/833
* [MINOR:TYPOS] Update README.md by @cakiki in https://github.com/huggingface/trl/pull/829
* Clarify docstrings, help messages, assert messages in merge_peft_adapter.py by @larekrow in https://github.com/huggingface/trl/pull/838
* add DDPO to index by @lvwerra in https://github.com/huggingface/trl/pull/826
* Raise error in `create_reference_model()` when ZeRO-3 is enabled  by @lewtun in https://github.com/huggingface/trl/pull/840
* Use uniform config by @vwxyzjn in https://github.com/huggingface/trl/pull/817
* Give `lewtun` power by @lvwerra in https://github.com/huggingface/trl/pull/856
* Standardise example scripts by @lewtun in https://github.com/huggingface/trl/pull/842
* Fix version check in import_utils.py by @adampauls in https://github.com/huggingface/trl/pull/853
* dont use get_peft_model if model is already peft by @abhishekkrthakur in https://github.com/huggingface/trl/pull/857
* [`core`] Fix import issues by @younesbelkada in https://github.com/huggingface/trl/pull/859
* Support both old and new diffusers import path by @osanseviero in https://github.com/huggingface/trl/pull/843

## New Contributors

* @backpropper made their first contribution in https://github.com/huggingface/trl/pull/681
* @jp1924 made their first contribution in https://github.com/huggingface/trl/pull/687
* @i4never made their first contribution in https://github.com/huggingface/trl/pull/745
* @zuoxingdong made their first contribution in https://github.com/huggingface/trl/pull/763
* @davidberenstein1957 made their first contribution in https://github.com/huggingface/trl/pull/665
* @filippobistaffa made their first contribution in https://github.com/huggingface/trl/pull/792
* @devxpy made their first contribution in https://github.com/huggingface/trl/pull/749
* @namin made their first contribution in https://github.com/huggingface/trl/pull/811
* @cakiki made their first contribution in https://github.com/huggingface/trl/pull/829
* @larekrow made their first contribution in https://github.com/huggingface/trl/pull/838
* @adampauls made their first contribution in https://github.com/huggingface/trl/pull/853
* @abhishekkrthakur made their first contribution in https://github.com/huggingface/trl/pull/857
* @osanseviero made their first contribution in https://github.com/huggingface/trl/pull/843

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.1...v0.7.2

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.2)

---

## v0.7.1: v0.7.1: Patch release
**Published:** 2023-08-30

# Patch release: fix bug with `PPOTrainer` and `log_stats`

Fixed a bug with `log_stats` of `PPOTrainer` to avoid breaking behaviour

* [`PPOTrainer`] A workaround for failing log_stats by @younesbelkada in https://github.com/huggingface/trl/pull/708

## What's Changed

* Release: v0.7.0 by @younesbelkada in https://github.com/huggingface/trl/pull/706
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/707


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.7.0...v0.7.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.1)

---

## v0.7.0: v0.7.0: Text Environments, Agents & Tools
**Published:** 2023-08-30

# Text environments, LLMs with tools and agents!

<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/textenv.png">
</div>

Text environments provide a learning ground for language agents. It allows a language model to use tools to accomplish a task such as using a Python interpreter to answer math questions or using a search index for trivia questions. Having access to tools allows language models to solve tasks that would be very hard for the models itself but can be trivial for the appropriate tools. 

We are excited to bring to the community a complete set of functionalities and full examples to train LLMs to use tools! 


Check out the documentation page [here](https://huggingface.co/docs/trl/text_environments) and few examples below:
* [fine tune a LLM to learn to use a simple calculator tool](https://github.com/huggingface/trl/blob/main/examples/research_projects/tools/calculator.py)
* [fine tune a LLM to learn to use a Question Answering tool to answer general knowledge questions](https://github.com/huggingface/trl/blob/main/examples/research_projects/tools/triviaqa.py)
* [fine tune a LLM to learn to use a Python interpreter](https://github.com/huggingface/trl/blob/main/examples/research_projects/tools/python_interpreter.py)

## What's Changed

* Release: v0.6.0 by @younesbelkada in https://github.com/huggingface/trl/pull/684
* set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/685
* [DPO] fix DPO ref_model=None by @kashif in https://github.com/huggingface/trl/pull/703
* [Docs] fix example README.md by @kashif in https://github.com/huggingface/trl/pull/705
* TextEnvironments by @lvwerra in https://github.com/huggingface/trl/pull/424


**Full Changelog**: https://github.com/huggingface/trl/compare/v0.6.0...v0.7.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.7.0)

---

## v0.6.0: v0.6.0
**Published:** 2023-08-25

# DDPO for diffusion models

We are excited to welcome the first RLHF + diffusion models algorithm to refine the generations from diffusion models. 
Read more about it directly [in the docs](https://huggingface.co/docs/trl/ddpo_trainer).

| Before | After DDPO finetuning |
| --- | --- |
| <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/pre_squirrel.png"/></div> |  <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/post_squirrel.png"/></div> |
|  <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/pre_starfish.png"/></div> |  <div style="text-align: center"><img src="https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/post_starfish.png"/></div> |

* Denoising Diffusion Policy Optimization by @metric-space in https://github.com/huggingface/trl/pull/508

# Bug fixes and other enhancements

The release also comes with multiple bug fixes reported and/or led by the community, check out the commit history below

## What's Changed

* Release: v0.5.0 by @younesbelkada in https://github.com/huggingface/trl/pull/607
* Set dev version by @younesbelkada in https://github.com/huggingface/trl/pull/608
* [`Modeling`] Add token support for `hf_hub_download` by @younesbelkada in https://github.com/huggingface/trl/pull/604
* Add docs explaining logged metrics by @vwxyzjn in https://github.com/huggingface/trl/pull/616
* [DPO] stack-llama-2 training scripts by @kashif in https://github.com/huggingface/trl/pull/611
* Use log_with argument in SFT example by @hitorilabs in https://github.com/huggingface/trl/pull/620
* Allow already tokenized sequences for `response_template` in `DataCollatorForCompletionOnlyLM` by @ivsanro1 in https://github.com/huggingface/trl/pull/622
* Improve docs by @lvwerra in https://github.com/huggingface/trl/pull/612
* Move repo by @lvwerra in https://github.com/huggingface/trl/pull/628
* Add score scaling/normalization/clipping by @zfang in https://github.com/huggingface/trl/pull/560
* Disable dropout in DPO Training by @NouamaneTazi in https://github.com/huggingface/trl/pull/639
* Add checks on backward batch size by @vwxyzjn in https://github.com/huggingface/trl/pull/651
* Resolve various typos throughout the docs by @tomaarsen in https://github.com/huggingface/trl/pull/654
* Update README.md by @Santosh-Gupta in https://github.com/huggingface/trl/pull/657
* Allow for ref_model=None in DPOTrainer by @vincentmin in https://github.com/huggingface/trl/pull/640
* Add more args to SFT example by @photomz in https://github.com/huggingface/trl/pull/642
* Handle potentially long sequences with DataCollatorForCompletionOnlyLM by @tannonk in https://github.com/huggingface/trl/pull/644
* [`sft_llama2`] Add check of arguments by @younesbelkada in https://github.com/huggingface/trl/pull/660
* Fix DPO blogpost thumbnail by @lvwerra in https://github.com/huggingface/trl/pull/673
* propagating eval_batch_size to TrainingArguments by @rahuljha in https://github.com/huggingface/trl/pull/675
* [`CI`] Fix unmutable `TrainingArguments` issue by @younesbelkada in https://github.com/huggingface/trl/pull/676
* Update sft_llama2.py by @msaad02 in https://github.com/huggingface/trl/pull/678
* fix PeftConfig loading from a remote repo. by @w32zhong in https://github.com/huggingface/trl/pull/649
* Simplify immutable TrainingArgs fix using `dataclasses.replace` by @tomaarsen in https://github.com/huggingface/trl/pull/682

## New Contributors

* @hitorilabs made their first contribution in https://github.com/huggingface/trl/pull/620
* @ivsanro1 made their first contribution in https://github.com/huggingface/trl/pull/622
* @zfang made their first contribution in https://github.com/huggingface/trl/pull/560
* @NouamaneTazi made their first contribution in https://github.com/huggingface/trl/pull/639
* @Santosh-Gupta made their first contribution in https://github.com/huggingface/trl/pull/657
* @vincentmin made their first contribution in https://github.com/huggingface/trl/pull/640
* @photomz made their first contribution in https://github.com/huggingface/trl/pull/642
* @tannonk made their first contribution in https://github.com/huggingface/trl/pull/644
* @rahuljha made their first contribution in https://github.com/huggingface/trl/pull/675
* @msaad02 made their first contribution in https://github.com/huggingface/trl/pull/678
* @w32zhong made their first contribution in https://github.com/huggingface/trl/pull/649

**Full Changelog**: https://github.com/huggingface/trl/compare/v0.5.0...v0.6.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.6.0)

---

## v0.5.0: v0.5.0
**Published:** 2023-08-02

# v0.5.0 DPOTrainer and multiple bug fixes on PPOTrainer and SFTTrainer

This release includes multiple important bugfixes (SFTTrainer, PPOTrainer), the release also extends the current `DataCollatorForCompletionOnlyLM` to support chat-like training. 

## DPO Trainer

The DPO algorithm (Direct Policy Optimization) has been introduced by Rafailov et al. in [this paper](https://arxiv.org/abs/2305.18290) and introduces a way of performing RL training without having to rely on a reward model. The DPOTrainer is now part of TRL library for anyone that wants to use it thanks to the amazing contributors! 

* DPO Trainer by @kashif in https://github.com/lvwerra/trl/pull/416
* [DPO] make sure all the concated batches are on same device by @kashif in https://github.com/lvwerra/trl/pull/528
* [DPO] remove response/pairs from the DPO side by @kashif in https://github.com/lvwerra/trl/pull/540
* [DPO] remove unnecessary batch size arg to Collator by @kashif in https://github.com/lvwerra/trl/pull/554
* [`DPO`] Resolve logging for DPOTrainer by @tomaarsen in https://github.com/lvwerra/trl/pull/570

## What's Changed

* Reward trainer multi-gpu eval bug by @rlindskog in https://github.com/lvwerra/trl/pull/513
* Use local process index for `_get_current_device()` by @lewtun in https://github.com/lvwerra/trl/pull/515

## Extending the `DataCollatorForCompletionOnlyLM`

You can now mask out the users prompts in the `DataCollatorForCompletionOnlyLM` data collator and train only on chat completions. Check out the PR below or the appropriate section on the documentation to learn more about it!

* Introducing DataCollatorForChatCompletionOnlyLM by @gaetanlop in https://github.com/lvwerra/trl/pull/456

## Important bug fixes

Multiple bugs on the supported trainers have been raised by the community and fixed in the below PRs

* [`core`] Fix offline case by @younesbelkada in https://github.com/lvwerra/trl/pull/538
* Relax reward trainer constraint by @younesbelkada in https://github.com/lvwerra/trl/pull/539
* ADD: num_proc to SFTTrainer by @BramVanroy in https://github.com/lvwerra/trl/pull/547
* [`SFTTrainer`] Add warning for wrong padding_side by @younesbelkada in https://github.com/lvwerra/trl/pull/550
* Minor typo and whitespace fixes by @tmm1 in https://github.com/lvwerra/trl/pull/559
* [`SFTTrainer`] Add epochs and num steps on CLI by @younesbelkada in https://github.com/lvwerra/trl/pull/562
* Add `DataCollatorForCompletionOnlyLM` in the docs by @younesbelkada in https://github.com/lvwerra/trl/pull/565
* Add comment to explain how the sentiment pipeline is used to run the … by @jvhoffbauer in https://github.com/lvwerra/trl/pull/555
* Fix model output dim in reward trainer example by @liutianlin0121 in https://github.com/lvwerra/trl/pull/566
* Computes the KL penalty using the entire distribution by @edbeeching in https://github.com/lvwerra/trl/pull/541
* Add missing max_seq_length arg to example sft_trainer.py by @SharkWipf in https://github.com/lvwerra/trl/pull/585
* [`PPO`] fix corner cases with PPO batch size and forward_batch_size by @younesbelkada in https://github.com/lvwerra/trl/pull/563
* Update the example sft_trainer.py by @ZeusFSX in https://github.com/lvwerra/trl/pull/587
* docs: Replace SFTTrainer with RewardTrainer in comment by @tomaarsen in https://github.com/lvwerra/trl/pull/589
* Fix comparison in DataCollatorForCompletionOnlyLM (#588) by @RyujiTamaki in https://github.com/lvwerra/trl/pull/594
* refactor grad accum by @vwxyzjn in https://github.com/lvwerra/trl/pull/546

## Big refactor of examples and documentation

The examples and documentation has been refactored, check the PRs below for more details

* [`examples`] Big refactor of examples and documentation by @younesbelkada in https://github.com/lvwerra/trl/pull/509
* [`examples`] Fix sentiment nit by @younesbelkada in https://github.com/lvwerra/trl/pull/517
* [`examples`] make the sft script more modulable by @younesbelkada in https://github.com/lvwerra/trl/pull/543
* Add `use_auth_token` arg to sft_trainer example by @corey-lambda in https://github.com/lvwerra/trl/pull/544


## New Contributors

* @rlindskog made their first contribution in https://github.com/lvwerra/trl/pull/513
* @corey-lambda made their first contribution in https://github.com/lvwerra/trl/pull/544
* @tmm1 made their first contribution in https://github.com/lvwerra/trl/pull/559
* @jvhoffbauer made their first contribution in https://github.com/lvwerra/trl/pull/555
* @liutianlin0121 made their first contribution in https://github.com/lvwerra/trl/pull/566
* @SharkWipf made their first contribution in https://github.com/lvwerra/trl/pull/585
* @ZeusFSX made their first contribution in https://github.com/lvwerra/trl/pull/587
* @gaetanlop made their first contribution in https://github.com/lvwerra/trl/pull/456
* @RyujiTamaki made their first contribution in https://github.com/lvwerra/trl/pull/594

**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.4.7...v0.5.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.5.0)

---

## v0.4.7: v0.4.7
**Published:** 2023-07-13

# Patch release: `SFTTrainer` and `PPOTrainer` bug fixes

## What's Changed

* Make shuffle optional by @lopez-hector in https://github.com/lvwerra/trl/pull/457
* Pre-commit by @vwxyzjn in https://github.com/lvwerra/trl/pull/448
* Debug the tortuous logic in `_prepare_dataset` function by @BeibinLi in https://github.com/lvwerra/trl/pull/464
* [`CI`] Fix CI RM by @younesbelkada in https://github.com/lvwerra/trl/pull/468
* Update sft_trainer.py by @JulesGM in https://github.com/lvwerra/trl/pull/474
* Refactor README by @younesbelkada in https://github.com/lvwerra/trl/pull/460
* add ratio threshold to avoid spikes by @lvwerra in https://github.com/lvwerra/trl/pull/488
* fix typo in reward_modeling.py by @csyourui in https://github.com/lvwerra/trl/pull/494
* FIX: contributing guidelines command by @BramVanroy in https://github.com/lvwerra/trl/pull/493
* Remove padding in batched generation. by @lvwerra in https://github.com/lvwerra/trl/pull/487
* Adds some options to stabilize the KL penalty by @edbeeching in https://github.com/lvwerra/trl/pull/486
* correctly implement gradient checkpointing to multi-adapter example by @mnoukhov in https://github.com/lvwerra/trl/pull/479
* Disable mlm by default in DataCollatorForCompletionOnlyLM, add ignore_index and docstring by @BramVanroy in https://github.com/lvwerra/trl/pull/476
* Use `float` instead of `double` to avoid issues with MPS device by @younesbelkada in https://github.com/lvwerra/trl/pull/499
* [`PPOTrainer`] Add prefix tuning support by @younesbelkada in https://github.com/lvwerra/trl/pull/501
* [`PPOTrainer`] Add prompt tuning support on TRL by @younesbelkada in https://github.com/lvwerra/trl/pull/500
* [`SFTTrainer`] Fix the sequence length check of `SFTTrainer` by @younesbelkada in https://github.com/lvwerra/trl/pull/512

## New Contributors

* @lopez-hector made their first contribution in https://github.com/lvwerra/trl/pull/457
* @BeibinLi made their first contribution in https://github.com/lvwerra/trl/pull/464
* @csyourui made their first contribution in https://github.com/lvwerra/trl/pull/494
* @BramVanroy made their first contribution in https://github.com/lvwerra/trl/pull/493

**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.4.6...v0.4.7

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.4.7)

---

## v0.4.6: v0.4.6
**Published:** 2023-06-23

# Patch release

Patch release to fix a bug on google colab with PPOTrainer & PPOConfig + wandb

## What's Changed

* Fix google colab issue by @younesbelkada in https://github.com/lvwerra/trl/pull/459


**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.4.5...v0.4.6

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.4.6)

---

## v0.4.5: v0.4.5
**Published:** 2023-06-23

# Patch release 1 - `SFTTrainer` enhancements and fixes

This patch release adds multiple fixes for the SFTTrainer and enhancements. Another patch release is coming for fixing an issue with PPOTrainer and Google Colab combined with wandb logging

## What's Changed

* Add slurm utility by @vwxyzjn in https://github.com/lvwerra/trl/pull/412
* Enable autotag feature w/ wandb by @vwxyzjn in https://github.com/lvwerra/trl/pull/411
* [doc build] Use secrets by @mishig25 in https://github.com/lvwerra/trl/pull/420
* Update test_reward_trainer.py by @younesbelkada in https://github.com/lvwerra/trl/pull/421
* best-of-n sampler class by @metric-space in https://github.com/lvwerra/trl/pull/375
* handle the offline case by @younesbelkada in https://github.com/lvwerra/trl/pull/431
* Fix correct gradient accumulation by @younesbelkada in https://github.com/lvwerra/trl/pull/407
* Drop support for Python 3.7 by @younesbelkada in https://github.com/lvwerra/trl/pull/441
* [`SFTTrainer`] Relax dataset constraints by @younesbelkada in https://github.com/lvwerra/trl/pull/442
* [`SFTTrainer`] Fix non packed dataset by @younesbelkada in https://github.com/lvwerra/trl/pull/444
* [`core`] Add stale bot by @younesbelkada in https://github.com/lvwerra/trl/pull/447
* [`SFTTrainer`] Introducing `DataCollatorForCompletionOnlyLM` by @younesbelkada in https://github.com/lvwerra/trl/pull/445
* [`ConstantLengthDataset`] Fix packed dataset issue by @younesbelkada in https://github.com/lvwerra/trl/pull/452
* Update accelerate arg passthrourgh for tensorboard logging to reflect logging_dir deprecation. by @jganitkevitch in https://github.com/lvwerra/trl/pull/437
* Multi adapter RL (MARL) - a single model for RM & Value Head by @younesbelkada in https://github.com/lvwerra/trl/pull/373

## New Contributors

* @jganitkevitch made their first contribution in https://github.com/lvwerra/trl/pull/437

**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.4.4...v0.4.5

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.4.5)

---

## v0.4.4: v0.4.4
**Published:** 2023-06-08

## Patch release

* [`core`] unpin accelerate by @younesbelkada in https://github.com/lvwerra/trl/pull/418


**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.4.3...v0.4.4

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.4.4)

---

## v0.4.3: v0.4.3
**Published:** 2023-06-08

## 0.4.3 Patch release

Patch release - pin accelerate version

* Skip flaky test until next transformers release by @younesbelkada in https://github.com/lvwerra/trl/pull/410
* Pin accelerate version by @younesbelkada in https://github.com/lvwerra/trl/pull/414


**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.4.2...v0.4.3

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.4.3)

---

## v0.4.2: v0.4.2
**Published:** 2023-06-07

# QLoRA RLHF, SFT Trainer and RewardTrainer

A new version of TRL that includes training larger models using QLoRA (4 bit quantization through bitsandbytes), brand new classes `RewardTrainer` and `SFTTrainer` to easily conduct your RLHF projects end-to-end! 

## Introducing `SFTTrainer` and `RewardTrainer`

Use the brand new trainer to easily train your reward model and supervised fine-tuned (SFT) model with few lines of code!

* [`core`] officially support SFT (Supervised Finetuning)  by @younesbelkada in https://github.com/lvwerra/trl/pull/323
* [`SFT`] Fix sft issues by @younesbelkada in https://github.com/lvwerra/trl/pull/336
* [`docs`] fix SFT doc by @younesbelkada in https://github.com/lvwerra/trl/pull/367
* [`core`] Officially Support Reward Modeling by @younesbelkada in https://github.com/lvwerra/trl/pull/303
* Resolve broken evaluation/prediction for RewardTrainer by @tomaarsen in https://github.com/lvwerra/trl/pull/404

## QLoRA integration

Pass 4bit models directly into `PPOTrainer` for more memory efficient training

* [`core`] Add 4bit QLora by @younesbelkada in https://github.com/lvwerra/trl/pull/383
* [`bnb`] fix 4 bit SFT by @younesbelkada in https://github.com/lvwerra/trl/pull/396

## Updated StackLlama example

Great work by @mnoukhov that managed to fix the issues related with StackLlama and the new versions of `accelerate`, `peft` and `transformers`. The completely reproducible examples below:

* StackLLaMA: correctly merge peft model by @mnoukhov in https://github.com/lvwerra/trl/pull/398
* StackLlama: fixed RL training and added args by @mnoukhov in https://github.com/lvwerra/trl/pull/400
* Fixed some type annotations of trl.trainer.PPoTrainer by @JulesGM in https://github.com/lvwerra/trl/pull/392
* StackLLaMA: fix supervised finetuning and reward model training by @mnoukhov in https://github.com/lvwerra/trl/pull/399

## Bug fixes and improvements

* [`core`] refactor peft API by @younesbelkada in https://github.com/lvwerra/trl/pull/231
* Batched generation by @lvwerra in https://github.com/lvwerra/trl/pull/228
* Reduce memory consumption in batched_forward_pass by @ohashi56225 in https://github.com/lvwerra/trl/pull/234
* [`core`] Add warning when negative KL by @younesbelkada in https://github.com/lvwerra/trl/pull/239
* adds early stopping by @edbeeching in https://github.com/lvwerra/trl/pull/238
* PPO config __init__ is bloated  by @GauravVirmani in https://github.com/lvwerra/trl/pull/241
* feat(ci): enable `pip` cache by @SauravMaheshkar in https://github.com/lvwerra/trl/pull/198
* Improve logging for PPO + Docs page by @natolambert in https://github.com/lvwerra/trl/pull/243
* Fix typo by @heya5 in https://github.com/lvwerra/trl/pull/253
* Using batched generate in sentiment scripts by @GauravVirmani in https://github.com/lvwerra/trl/pull/249
* [`core`] Fix DeepSpeed zero-3 issue by @younesbelkada in https://github.com/lvwerra/trl/pull/182
* [`distributed`] Fix early stopping and DP by @younesbelkada in https://github.com/lvwerra/trl/pull/254
* [`core`] Fix ds issue by @younesbelkada in https://github.com/lvwerra/trl/pull/260
* Add LlaMa in tests + `create_reference_model` by @younesbelkada in https://github.com/lvwerra/trl/pull/261
* Use active model to generate response in example on README (#269) by @rmill040 in https://github.com/lvwerra/trl/pull/271
* stack-llama by @edbeeching in https://github.com/lvwerra/trl/pull/273
* Adding pointer back to Meta's LLaMA. by @meg-huggingface in https://github.com/lvwerra/trl/pull/277
* fix doc string problem in ppo trainer loss function by @thuwyh in https://github.com/lvwerra/trl/pull/279
* Add LLaMA tutorial to docs by @natolambert in https://github.com/lvwerra/trl/pull/278
* Fix swapped helper texts by @philipp-classen in https://github.com/lvwerra/trl/pull/284
* fix typo in gpt2-sentiment.ipynb by @eltociear in https://github.com/lvwerra/trl/pull/293
* add functionality to push best models to the hub during training by @Bearnardd in https://github.com/lvwerra/trl/pull/275
* Small improvements / fixes to toxicity example by @natolambert in https://github.com/lvwerra/trl/pull/266
* Fix arguments description by @lvzii in https://github.com/lvwerra/trl/pull/298
* [`t5`] Fix negative kl issue by @younesbelkada in https://github.com/lvwerra/trl/pull/262
* Log Token distribution of Query / Response by @natolambert in https://github.com/lvwerra/trl/pull/295
* clean examples folder by @natolambert in https://github.com/lvwerra/trl/pull/294
* fixed typo in error message by @soerenarlt in https://github.com/lvwerra/trl/pull/312
* fix DS for peft ref_model in ppo trainer by @halfrot in https://github.com/lvwerra/trl/pull/309
* [`CI`] Fix broken tests by @younesbelkada in https://github.com/lvwerra/trl/pull/318
* [`Docs`] Add details on multi-GPU / multi-node by @younesbelkada in https://github.com/lvwerra/trl/pull/320
* Give a key to the wandb PPOConfig config entry by @JulesGM in https://github.com/lvwerra/trl/pull/315
* added doc for using torch.distributed.launch/run by @oroojlooy in https://github.com/lvwerra/trl/pull/324
* Fix argument's description by @vinhkhuc in https://github.com/lvwerra/trl/pull/339
* stack_llama: update instructions in README, fix broken _get_submodules and save tokenizer by @teticio in https://github.com/lvwerra/trl/pull/358
* stack_llama: add parameter to control max_length (to mitigate OOM errors) by @teticio in https://github.com/lvwerra/trl/pull/359
* [`PPO`] Relax negative KL constraint by @younesbelkada in https://github.com/lvwerra/trl/pull/352
* [`PPOTrainer`] Fix tensorboard issue by @younesbelkada in https://github.com/lvwerra/trl/pull/330
* 140/best n sampling by @metric-space in https://github.com/lvwerra/trl/pull/326
* Fix bug when loading local peft model by @Opdoop in https://github.com/lvwerra/trl/pull/342
* add is_trainable in kwargs by @Opdoop in https://github.com/lvwerra/trl/pull/363
* Remove obsolete layer_norm_names parameter and add peft>=0.3.0 to requirements by @teticio in https://github.com/lvwerra/trl/pull/366
* Delete test_training.py by @younesbelkada in https://github.com/lvwerra/trl/pull/371
* [`core`] Fix warning issue by @younesbelkada in https://github.com/lvwerra/trl/pull/377
* Update customization.mdx by @binganao in https://github.com/lvwerra/trl/pull/390
* fix dataloader typo in ppo_trainer.py by @LZY-the-boys in https://github.com/lvwerra/trl/pull/389
* from_pretrain with peft adapter on the hub (# 379) by @glerzing in https://github.com/lvwerra/trl/pull/380
* keep state_dict kwargs instead of popping it in save_pretrained by @rizar in https://github.com/lvwerra/trl/pull/393
* Remove unused imports in docs. by @vwxyzjn in https://github.com/lvwerra/trl/pull/406

## New Contributors

* @ohashi56225 made their first contribution in https://github.com/lvwerra/trl/pull/234
* @GauravVirmani made their first contribution in https://github.com/lvwerra/trl/pull/241
* @SauravMaheshkar made their first contribution in https://github.com/lvwerra/trl/pull/198
* @heya5 made their first contribution in https://github.com/lvwerra/trl/pull/253
* @rmill040 made their first contribution in https://github.com/lvwerra/trl/pull/271
* @thuwyh made their first contribution in https://github.com/lvwerra/trl/pull/279
* @philipp-classen made their first contribution in https://github.com/lvwerra/trl/pull/284
* @Bearnardd made their first contribution in https://github.com/lvwerra/trl/pull/275
* @lvzii made their first contribution in https://github.com/lvwerra/trl/pull/298
* @soerenarlt made their first contribution in https://github.com/lvwerra/trl/pull/312
* @halfrot made their first contribution in https://github.com/lvwerra/trl/pull/309
* @oroojlooy made their first contribution in https://github.com/lvwerra/trl/pull/324
* @vinhkhuc made their first contribution in https://github.com/lvwerra/trl/pull/339
* @teticio made their first contribution in https://github.com/lvwerra/trl/pull/358
* @metric-space made their first contribution in https://github.com/lvwerra/trl/pull/326
* @Opdoop made their first contribution in https://github.com/lvwerra/trl/pull/342
* @binganao made their first contribution in https://github.com/lvwerra/trl/pull/390
* @LZY-the-boys made their first contribution in https://github.com/lvwerra/trl/pull/389
* @glerzing made their first contribution in https://github.com/lvwerra/trl/pull/380
* @rizar made their first contribution in https://github.com/lvwerra/trl/pull/393
* @mnoukhov made their first contribution in https://github.com/lvwerra/trl/pull/398
* @tomaarsen made their first contribution in https://github.com/lvwerra/trl/pull/404
* @vwxyzjn made their first contribution in https://github.com/lvwerra/trl/pull/406

**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.4.1...v0.4.2

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.4.2)

---

## v0.4.1: v0.4.1
**Published:** 2023-03-17

# Large models training, Naive Pipeline Parallelism, `peft` Data Parallelism support and distributed training bug fixes

This release includes a set of features and bug fixes to scale up your RLHF experiments for much larger models leveraging `peft` and `bitsandbytes`.

## Naive Pipeline Parallelism support 

* Let's support naive Pipeline Parallelism by @younesbelkada in https://github.com/lvwerra/trl/pull/210

We introduce a new paradigm in `trl` , termed as Naive Pipeline Parallelism, to fit large scale models on your training setup and apply RLHF on them. This feature uses `peft` to train adapters and `bitsandbytes` to reduce the memory foot print of your active model

![image](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/images/trl-npp.png)

## `peft` Data Parallelism support

* [`peft`] Fix DP issues by @younesbelkada in https://github.com/lvwerra/trl/pull/221
* [`core`] fix DP issue by @younesbelkada in https://github.com/lvwerra/trl/pull/222

There were some bugs with respect to `peft` integration and DP. This release includes the bug fixes to enable multi-GPU training using `accelerate` + DDP (DIstributed Data Parallel)

## Memory optimization

Your training runs can be now much more memory efficient thanks to few tricks / bug fixes:
Now `PPOConfig` also supports the flag `optimize_cuda_cache` (set to `False` by default) to avoid increasing CUDA memory issues

* Grad accumulation and memory bugfix by @edbeeching in https://github.com/lvwerra/trl/pull/220
* adds a missing detach to the ratio by @edbeeching in https://github.com/lvwerra/trl/pull/224

## Pytorch 2.0 fixes

This release also includes minor fixes related to PyTorch 2.0 release

* [`test`] attempt to fix CI test for PT 2.0 by @younesbelkada in https://github.com/lvwerra/trl/pull/225


## What's Changed

* adds sentiment example for a 20b model by @edbeeching in https://github.com/lvwerra/trl/pull/208
* Update README.md blog post link by @TeamDman in https://github.com/lvwerra/trl/pull/212
* spell mistakes by @k-for-code in https://github.com/lvwerra/trl/pull/213
* spell corrections by @k-for-code in https://github.com/lvwerra/trl/pull/214
* Small changes when integrating into H4 by @natolambert in https://github.com/lvwerra/trl/pull/216


## New Contributors
* @TeamDman made their first contribution in https://github.com/lvwerra/trl/pull/212
* @k-for-code made their first contribution in https://github.com/lvwerra/trl/pull/213

**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.4.0...v0.4.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.4.1)

---

## v0.4.0: v0.4.0
**Published:** 2023-03-09

# `v0.4.0`: `peft` integration 

Apply RLHF and fine-tune your favorite large model on consumer GPU using `peft` and `trl` ! Share also easily your trained RLHF adapters on the Hub with few lines of code

With this integration you can train `gpt-neo-x` (20B parameter model - 40GB in `bfloat16`) on a 24GB consumer GPU! 

## What's Changed

* Allow running evaluate-toxicity with cpu by @jordimas in https://github.com/lvwerra/trl/pull/195
* [`core`] Fix quality issue by @younesbelkada in https://github.com/lvwerra/trl/pull/197
* Add 1.12.1 torch compatibility in sum method by @PanchenkoYehor in https://github.com/lvwerra/trl/pull/190
* `peft` integration by @edbeeching in https://github.com/lvwerra/trl/pull/163
* [`core`] Update dependency by @younesbelkada in https://github.com/lvwerra/trl/pull/206

## New Contributors

* @PanchenkoYehor made their first contribution in https://github.com/lvwerra/trl/pull/190

**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.3.1...v0.4.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.4.0)

---

## v0.3.1: v0.3.1
**Published:** 2023-03-02

## What's Changed
* Clarifications of acronyms and initialisms by @meg-huggingface in https://github.com/lvwerra/trl/pull/185
* Update detoxifying_a_lm.mdx by @younesbelkada in https://github.com/lvwerra/trl/pull/186
* Fix reference to example by @jordimas in https://github.com/lvwerra/trl/pull/184

## New Contributors
* @meg-huggingface made their first contribution in https://github.com/lvwerra/trl/pull/185
* @jordimas made their first contribution in https://github.com/lvwerra/trl/pull/184

**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.3.0...v0.3.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.3.1)

---

## v0.3.0: v0.3.0
**Published:** 2023-03-01

## What's Changed
* fix style, typos, license by @natolambert in https://github.com/lvwerra/trl/pull/103
* fix re-added file by @natolambert in https://github.com/lvwerra/trl/pull/116
* add citation by @natolambert in https://github.com/lvwerra/trl/pull/124
* add manual seeding for RL experiments by @natolambert in https://github.com/lvwerra/trl/pull/118
* add `set_seed` to __init__.py by @lvwerra in https://github.com/lvwerra/trl/pull/127
* update docs with Seq2seq models, set_seed, and create_reference_model by @lvwerra in https://github.com/lvwerra/trl/pull/128
* [`bug`] Update gpt2-sentiment.py by @younesbelkada in https://github.com/lvwerra/trl/pull/132
* Fix Sentiment control notebook by @lvwerra in https://github.com/lvwerra/trl/pull/126
* realign values by @lvwerra in https://github.com/lvwerra/trl/pull/137
* Change unclear variables & fix typos by @natolambert in https://github.com/lvwerra/trl/pull/134
* Feat/reward summarization example by @TristanThrush in https://github.com/lvwerra/trl/pull/115
* [`core`] Small refactor of forward pass by @younesbelkada in https://github.com/lvwerra/trl/pull/136
* [`tests`] Add correct repo name by @younesbelkada in https://github.com/lvwerra/trl/pull/138
* fix forward batching for seq2seq and right padding models. by @lvwerra in https://github.com/lvwerra/trl/pull/139
* fix bug in batched_forward_pass by @ArvinZhuang in https://github.com/lvwerra/trl/pull/144
* [`core`] Add `torch_dtype` support by @younesbelkada in https://github.com/lvwerra/trl/pull/147
* [`core`] Fix dataloader issue by @younesbelkada in https://github.com/lvwerra/trl/pull/154
* [`core`] enable `bf16` training by @younesbelkada in https://github.com/lvwerra/trl/pull/156
* [`core`] fix saving multi-gpu by @younesbelkada in https://github.com/lvwerra/trl/pull/157
* Added imports by @BirgerMoell in https://github.com/lvwerra/trl/pull/159
* Add CITATION.cff by @kashif in https://github.com/lvwerra/trl/pull/169
* [Doc] Add how to use Lion optimizer by @younesbelkada in https://github.com/lvwerra/trl/pull/152
* policy kl [old | new] by @kashif in https://github.com/lvwerra/trl/pull/168
* add minibatching by @lvwerra in https://github.com/lvwerra/trl/pull/153
* fix bugs in tutorial by @shizhediao in https://github.com/lvwerra/trl/pull/175
* [`core`] Add `max_grad_norm` support by @younesbelkada in https://github.com/lvwerra/trl/pull/177
* Add toxcitiy example by @younesbelkada in https://github.com/lvwerra/trl/pull/162
* [`Docs`] Fix barplot by @younesbelkada in https://github.com/lvwerra/trl/pull/181

## New Contributors
* @natolambert made their first contribution in https://github.com/lvwerra/trl/pull/103
* @ArvinZhuang made their first contribution in https://github.com/lvwerra/trl/pull/144
* @BirgerMoell made their first contribution in https://github.com/lvwerra/trl/pull/159
* @kashif made their first contribution in https://github.com/lvwerra/trl/pull/169
* @shizhediao made their first contribution in https://github.com/lvwerra/trl/pull/175

**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.2.1...v0.3.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.3.0)

---

## v0.2.1: v0.2.1
**Published:** 2023-01-25

## What's Changed
* Update customization.mdx by @younesbelkada in https://github.com/lvwerra/trl/pull/109
* add `datasets` as a dependancy by @lvwerra in https://github.com/lvwerra/trl/pull/110
* [Docs] Add hlinks to scripts & notebooks by @younesbelkada in https://github.com/lvwerra/trl/pull/111
* Fix `Mapping` in core for Python 3.10 by @lvwerra in https://github.com/lvwerra/trl/pull/112


**Full Changelog**: https://github.com/lvwerra/trl/compare/v0.2.0...v0.2.1

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.2.1)

---

## v0.2.0: v0.2.0
**Published:** 2023-01-25

## Highlights
* General decoder model support in addition to GPT-2 in https://github.com/lvwerra/trl/pull/53
* Encoder-decoder model support (such as T5) in https://github.com/lvwerra/trl/pull/93
* New, shiny docs with the `doc-builder` in https://github.com/lvwerra/trl/pull/59
* `push_to_hub` with PPOTrainer in https://github.com/lvwerra/trl/pull/68
* Simple reference model creation with layer sharing in https://github.com/lvwerra/trl/pull/61

## What's Changed

* Remove `nbdev` dependency by @younesbelkada in https://github.com/lvwerra/trl/pull/52
* Adds github actions and dummy test by @edbeeching in https://github.com/lvwerra/trl/pull/55
* Update README.md by @Keith-Hon in https://github.com/lvwerra/trl/pull/51
* Update README.md by @TristanThrush in https://github.com/lvwerra/trl/pull/49
* Adds Python highlighting to the code block by @JulesGM in https://github.com/lvwerra/trl/pull/45
* `xxxForCausalLM` support by @younesbelkada in https://github.com/lvwerra/trl/pull/53
* [`VHead`] Fix slow convergence issue by @younesbelkada in https://github.com/lvwerra/trl/pull/60
* add docbuilder skeleton by @lvwerra in https://github.com/lvwerra/trl/pull/59
* fix docs workflow by @lvwerra in https://github.com/lvwerra/trl/pull/63
* `accelerate` integration by @younesbelkada in https://github.com/lvwerra/trl/pull/58
* add create_reference_model by @lvwerra in https://github.com/lvwerra/trl/pull/61
* Improve Makefile and code quality by @lvwerra in https://github.com/lvwerra/trl/pull/62
* Relax requirements by @lvwerra in https://github.com/lvwerra/trl/pull/66
* modeling - change namings by @younesbelkada in https://github.com/lvwerra/trl/pull/65
* [`PPOTrainer`] make the reference model optional by @younesbelkada in https://github.com/lvwerra/trl/pull/67
* Improvements 1a by @edbeeching in https://github.com/lvwerra/trl/pull/70
* update GitHub actions to `main` by @lvwerra in https://github.com/lvwerra/trl/pull/77
* [core] refactor `step` method by @younesbelkada in https://github.com/lvwerra/trl/pull/76
* [`PPOTrainer`] Support generic optimizers by @younesbelkada in https://github.com/lvwerra/trl/pull/78
* Update sentiment_tuning.mdx by @eltociear in https://github.com/lvwerra/trl/pull/69
* Remove references to "listify_batch" by @xiaoyesoso in https://github.com/lvwerra/trl/pull/81
* Collater -> collator by @LysandreJik in https://github.com/lvwerra/trl/pull/88
* Model as kwarg in pipeline by @LysandreJik in https://github.com/lvwerra/trl/pull/89
* Small typo correction by @LysandreJik in https://github.com/lvwerra/trl/pull/87
* [API] Make `dataset` attribute optional by @younesbelkada in https://github.com/lvwerra/trl/pull/85
* [Doc] Improve docs by @younesbelkada in https://github.com/lvwerra/trl/pull/91
* [core] Push `v_head` when using `AutoModelForCausalLMWithValueHead` by @younesbelkada in https://github.com/lvwerra/trl/pull/86
* [core] remove `wandb` dependency by @younesbelkada in https://github.com/lvwerra/trl/pull/92
* add logo by @lvwerra in https://github.com/lvwerra/trl/pull/95
* Encoder-Decoder models support by @younesbelkada in https://github.com/lvwerra/trl/pull/93
* Fix docs hyperlinks by @lewtun in https://github.com/lvwerra/trl/pull/98
* [API] LR scheduler support by @younesbelkada in https://github.com/lvwerra/trl/pull/96
* Version should have `dev0` unless it is a release version by @mishig25 in https://github.com/lvwerra/trl/pull/99
* [core] improve API by @younesbelkada in https://github.com/lvwerra/trl/pull/97
* Add push to Hub for PPOTrainer by @lewtun in https://github.com/lvwerra/trl/pull/68
* [`core`] Advise to use `fbs=1` by @younesbelkada in https://github.com/lvwerra/trl/pull/102
* [Doc] New additions by @younesbelkada in https://github.com/lvwerra/trl/pull/105
* restructure examples by @lvwerra in https://github.com/lvwerra/trl/pull/107
* Fix nits & missing things by @younesbelkada in https://github.com/lvwerra/trl/pull/108
* Convert notebook 05 by @edbeeching in https://github.com/lvwerra/trl/pull/80

## New Contributors
* @lvwerra made their first contribution in https://github.com/lvwerra/trl/pull/2
* @vblagoje made their first contribution in https://github.com/lvwerra/trl/pull/16
* @dependabot made their first contribution in https://github.com/lvwerra/trl/pull/26
* @younesbelkada made their first contribution in https://github.com/lvwerra/trl/pull/52
* @edbeeching made their first contribution in https://github.com/lvwerra/trl/pull/55
* @Keith-Hon made their first contribution in https://github.com/lvwerra/trl/pull/51
* @TristanThrush made their first contribution in https://github.com/lvwerra/trl/pull/49
* @JulesGM made their first contribution in https://github.com/lvwerra/trl/pull/45
* @eltociear made their first contribution in https://github.com/lvwerra/trl/pull/69
* @xiaoyesoso made their first contribution in https://github.com/lvwerra/trl/pull/81
* @LysandreJik made their first contribution in https://github.com/lvwerra/trl/pull/88
* @lewtun made their first contribution in https://github.com/lvwerra/trl/pull/98
* @mishig25 made their first contribution in https://github.com/lvwerra/trl/pull/99

**Full Changelog**: https://github.com/lvwerra/trl/commits/v0.2.0

[View on GitHub](https://github.com/huggingface/trl/releases/tag/v0.2.0)

---

