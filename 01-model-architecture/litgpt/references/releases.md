# Releases

Version history for this repository (27 releases).

## v0.5.11: v0.5.11
**Published:** 2025-09-09

## What's Changed

* Add Qwen3 2507 Thinking models by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/2110
* Add MLA by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/2113
* Add Qwen3 2507 Instruct models by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/2125
* finetune_lora upgrades by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/2086
* Use torch autograd integration for backward testing by @t-vi in https://github.com/Lightning-AI/litgpt/pull/2123
* bump PT 2.8 & install with `--upgrade-strategy greedy` by @Borda in https://github.com/Lightning-AI/litgpt/pull/2107
* build(deps): update transformers requirement from <4.54,>=4.51.3 to >=4.51.3,<4.57 by @dependabot[bot] in https://github.com/Lightning-AI/litgpt/pull/2120
* build(deps): update huggingface-hub requirement from <0.33,>=0.30 to >=0.30,<0.35 by @dependabot[bot] in https://github.com/Lightning-AI/litgpt/pull/2119
* build(deps): update bitsandbytes requirement from <0.43,>=0.42 to >=0.42,<0.47 by @dependabot[bot] in https://github.com/Lightning-AI/litgpt/pull/2069

---

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.10...v0.5.11

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.11)

---

## v0.5.10: v0.5.10
**Published:** 2025-08-13

## What's Changed
* qwen2.5 long context by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1933
* debug installing `torch` for Thunder by @Borda in https://github.com/Lightning-AI/litgpt/pull/2072
* Fix in `convert_hf_checkpoint` related to Gemma 3 by @mseeger in https://github.com/Lightning-AI/litgpt/pull/2062
* debug some failing standalone tests with compiler by @Borda in https://github.com/Lightning-AI/litgpt/pull/2075
* Qwen3 MoE by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/2060
* Deferring import of torch in config to allow faster import by @JackUrb in https://github.com/Lightning-AI/litgpt/pull/2079
* update bug-report/issue with reproducing in Studio by @Borda in https://github.com/Lightning-AI/litgpt/pull/2081
* ci: show the longest tests for improvement by @Borda in https://github.com/Lightning-AI/litgpt/pull/2083
* docs: Add documentation for OpenAI-compatible API in LitGPT deployment by @bhimrazy in https://github.com/Lightning-AI/litgpt/pull/2082
* doc: add `n_query_groups` to attention notation table by @raishish in https://github.com/Lightning-AI/litgpt/pull/2092
* doc: add comments for clarifying query / KV groups by @raishish in https://github.com/Lightning-AI/litgpt/pull/2093
* Complete pending todos in testing by @raishish in https://github.com/Lightning-AI/litgpt/pull/2088
* pin: restrict datasets version to <4.0.0 for compatibility by @Borda in https://github.com/Lightning-AI/litgpt/pull/2095
* add/debug Lit CI by @Borda in https://github.com/Lightning-AI/litgpt/pull/2094
* add `api_path` parameter to cli options to allow custom API endpoint configuration by @botirk38 in https://github.com/Lightning-AI/litgpt/pull/2080
* Fix commands in pretrain.md by @H4kor in https://github.com/Lightning-AI/litgpt/pull/2097

## New Contributors
* @JackUrb made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2079
* @raishish made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2092
* @botirk38 made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2080
* @H4kor made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2097

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.9...v0.5.10

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.10)

---

## v0.5.9: v0.5.9
**Published:** 2025-06-12

## What's Changed
* input_pos_maxp1 as a Python integer by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/2016
* add testing for py3.12 & py3.13 by @Borda in https://github.com/Lightning-AI/litgpt/pull/2025
* ci: update guardian for PRs by @Borda in https://github.com/Lightning-AI/litgpt/pull/2043
* Add devcontainer by @twsl in https://github.com/Lightning-AI/litgpt/pull/2035
* Remove dependency from config to utils by @lukemerrick in https://github.com/Lightning-AI/litgpt/pull/2034
* fix: Add fallback chat template by @andyland in https://github.com/Lightning-AI/litgpt/pull/2040
* Add optional sys prompt by @twsl in https://github.com/Lightning-AI/litgpt/pull/2036
* fix: Pretraining text files with recent litdata versions by @andyland in https://github.com/Lightning-AI/litgpt/pull/2048
* tests: mark `test_evaluate_script` as flaky by @Borda in https://github.com/Lightning-AI/litgpt/pull/2049
* simplify the GPU testing flow by @Borda in https://github.com/Lightning-AI/litgpt/pull/2053
* ci: extend testing with `ubuntu-24.04` by @Borda in https://github.com/Lightning-AI/litgpt/pull/2056
* Remove litserve version constraint by @twsl in https://github.com/Lightning-AI/litgpt/pull/2055
* req: pin `bitsandbytes>=0.45.2,<0.45.5` by @Borda in https://github.com/Lightning-AI/litgpt/pull/2057
* ci: use Thunder dev images for testing by @Borda in https://github.com/Lightning-AI/litgpt/pull/2054
* Update spacing in README.md by @Borda in https://github.com/Lightning-AI/litgpt/pull/2058
* Transformers version bump by @KaelanDt in https://github.com/Lightning-AI/litgpt/pull/2029
* Qwen3 Dense by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/2044
* phi-4 reasoning models by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/2047
* adding logger args by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1973
* Qwen3 MoE Preliminary: add intermediate_size argument to MLP modules by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/2046
* OLMo 2 by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1897
* bump: testing with PT 2.7.1 by @Borda in https://github.com/Lightning-AI/litgpt/pull/2063
* Add Dependabot for Pip & GitHub Actions by @Borda in https://github.com/Lightning-AI/litgpt/pull/2066
* build(deps): bump litdata from 0.2.45 to 0.2.49 by @dependabot in https://github.com/Lightning-AI/litgpt/pull/2068

## New Contributors
* @twsl made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2035
* @lukemerrick made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2034
* @prabhuteja12 made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2027
* @andyland made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2040
* @KaelanDt made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2029

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.8...v0.5.9

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.9)

---

## v0.5.8: v0.5.8
**Published:** 2025-05-06

Many great updates!

## What's Changed
* add missing r1 prompt style by @ali-alshaar7 in https://github.com/Lightning-AI/litgpt/pull/1929
* fix incremental save for PyTorch 2.6 by @t-vi in https://github.com/Lightning-AI/litgpt/pull/1928
* fix: resolve failing CI by @Borda in https://github.com/Lightning-AI/litgpt/pull/1944
* handle wrapped thundermodules in generate by @t-vi in https://github.com/Lightning-AI/litgpt/pull/1955
* fix skip condition by @t-vi in https://github.com/Lightning-AI/litgpt/pull/1956
* ci: use HF cache by @Borda in https://github.com/Lightning-AI/litgpt/pull/1958
* ci: split HF caching by @Borda in https://github.com/Lightning-AI/litgpt/pull/1960
* bump: PT 2.6 + `bitsandbytes` & standalone tests by @Borda in https://github.com/Lightning-AI/litgpt/pull/1959
* prune whitespaces for code readability by @Borda in https://github.com/Lightning-AI/litgpt/pull/1962
* fixing various typos in examples & tutorials by @Borda in https://github.com/Lightning-AI/litgpt/pull/1963
* fix `n_query_groups` for llama-3.1-405b by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1946
* tests: make flaky test due to connection issues by @Borda in https://github.com/Lightning-AI/litgpt/pull/1964
* Fix:  incorrect gradient accumulation steps bug by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1947
* fix: use default `num_nodes=1` for back-compatibility by @Borda in https://github.com/Lightning-AI/litgpt/pull/1967
* Do not wrap LoRA layers with FSDP by @janEbert in https://github.com/Lightning-AI/litgpt/pull/1538
* Speculative decoding: Base implementation by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1938
* Better clarity on SFT dataset attributes by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1970
* Enforce Consistent Formatting and Validation for YAML Files by @Borda in https://github.com/Lightning-AI/litgpt/pull/1977
* Apply Standard Formatting and Fix Import & Test Name Issues by @Borda in https://github.com/Lightning-AI/litgpt/pull/1981
* Setting `config.sliding_window_layer_stride` explicity by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1972
* feat: add linear rope type by @k223kim in https://github.com/Lightning-AI/litgpt/pull/1982
* feat: update tests for transformers 4.50.2 by @k223kim in https://github.com/Lightning-AI/litgpt/pull/1983
* fix: `test_tokenizer_against_hf` by @Borda in https://github.com/Lightning-AI/litgpt/pull/1984
* feat: replace sliding window type with offset by @k223kim in https://github.com/Lightning-AI/litgpt/pull/1989
* ci: with `pull_request_target` by @Borda in https://github.com/Lightning-AI/litgpt/pull/1992
* Phi4 mini by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1949
* aggregate `val_loss` by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1971
* feat: add local base freq for rope by @k223kim in https://github.com/Lightning-AI/litgpt/pull/1993
* test: flexible wait for serve start by @Borda in https://github.com/Lightning-AI/litgpt/pull/1996
* fix: replace sliding window configuration parameters to sliding windows indices by @k223kim in https://github.com/Lightning-AI/litgpt/pull/1995
* QwQ-32B by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1952
* feat: run thunder tests as part of LitGPT CI by @deependujha in https://github.com/Lightning-AI/litgpt/pull/1975
* try pyupgrade-up py38 by @Borda in https://github.com/Lightning-AI/litgpt/pull/1999
* [1/4] feat: add gemma 3 27b by @k223kim in https://github.com/Lightning-AI/litgpt/pull/1998
* [2/4] add gemma 3 1b by @k223kim in https://github.com/Lightning-AI/litgpt/pull/2000
* [3/4] feat: add gemma 3 4b by @k223kim in https://github.com/Lightning-AI/litgpt/pull/2001
* feat: add gemma-3-12b by @k223kim in https://github.com/Lightning-AI/litgpt/pull/2002
* Add resume for adapter_v2, enable continued finetuning for adapter by @altria-zewei-wang in https://github.com/Lightning-AI/litgpt/pull/1354
* Fix/loading gemma 3 1b by @pquadri in https://github.com/Lightning-AI/litgpt/pull/2004
* feat: add gemma 3 in readme and tutorials by @k223kim in https://github.com/Lightning-AI/litgpt/pull/2005
* add borda as codeowner by @t-vi in https://github.com/Lightning-AI/litgpt/pull/2007
* example for full finetuning with python code by @astrobdr in https://github.com/Lightning-AI/litgpt/pull/1331
* feat: add tests for gemma3 by @k223kim in https://github.com/Lightning-AI/litgpt/pull/2006
* building tutorials as mkdocs pages by @Borda in https://github.com/Lightning-AI/litgpt/pull/2011
* Add mlflow logger support by @topikachu in https://github.com/Lightning-AI/litgpt/pull/1985
* fix support for `litserve>0.2.4` by @ali-alshaar7 in https://github.com/Lightning-AI/litgpt/pull/1994
* Cast tensors in KVCache only when needed by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/2017
* feat: load only text weights from multimodal gemma by @pquadri in https://github.com/Lightning-AI/litgpt/pull/2008
* Feature: Adds support for OpenAISpec in litgpt serve by @bhimrazy in https://github.com/Lightning-AI/litgpt/pull/1943
* drop upper bounds in dependencies by @t-vi in https://github.com/Lightning-AI/litgpt/pull/2022

## New Contributors
* @syntheticgio made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1939
* @deependujha made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1975
* @altria-zewei-wang made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1354
* @pquadri made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2004
* @astrobdr made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1331
* @topikachu made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1985
* @bhimrazy made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1943
* @Lynsoo made their first contribution in https://github.com/Lightning-AI/litgpt/pull/2018

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.7...v0.5.8

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.8)

---

## v0.5.7: v0.5.7
**Published:** 2025-01-27

## What's Changed
* Add Deepseek r1 distill llama models by @ali-alshaar7 in https://github.com/Lightning-AI/litgpt/pull/1922

## New Contributors
* @ali-alshaar7 made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1922

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.6...v0.5.7


[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.7)

---

## v0.5.6: v0.5.6
**Published:** 2025-01-24

## What has changed

- Phi-4 by @ysjprojects 
- Thunder compatibility by @t-vi 


[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.6)

---

## v0.5.5: v0.5.5
**Published:** 2025-01-08

## What's Changed
* Post-release setup for 0.5.5.dev1 by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1885
* Falcon3 by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1881
* ChatML prompt template by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1882
* Small fixes and refactoring by @mseeger in https://github.com/Lightning-AI/litgpt/pull/1861
* Drop interleave placement in QKV matrix by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1013
* Bump PyTorch, PyTorch-Lightning and BnB versions by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1893
* Pin version of mistune in check links workflow by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1895
* Skip converting .safetensors to .bin by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1853
* Some improvements for KV caching by @mseeger in https://github.com/Lightning-AI/litgpt/pull/1891
* added query-key norm to accomodate OLMo2 by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1894
* Improve HF download speed by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1899
* Bump version for 0.5.5 release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1901

## New Contributors
* @mseeger made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1861

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.4...v0.5.5

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.5)

---

## v0.5.4: v0.5.4
**Published:** 2024-12-23

## What's Changed
* 0.5.3 post release setup by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1817
* Add cff file by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1818
* Deprecate Support for Dolly, Nous-Hermes, Redpajama-Incite, Vicuna and H2O Danube Models. by @ParagEkbote in https://github.com/Lightning-AI/litgpt/pull/1821
* Adding OLMo by @aflah02 in https://github.com/Lightning-AI/litgpt/pull/1827
* Adding Qwen2.5 by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1834
* Restore SlimPajama preprocessing code by @aflah02 in https://github.com/Lightning-AI/litgpt/pull/1840
* Add QwQ-32B-Preview by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1844
* Add Mixtral-8x22B by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1845
* add Llama-3.3-70B-Instruct by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1859
* add Salamandra by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1857
* Qwen2.5: fix block size for Coder series by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1856
* fix: add missing"," by @vra in https://github.com/Lightning-AI/litgpt/pull/1855
* fix llama3.3 readme url by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1862
* Set torch.load(..., `weights_only=False`) in litgpt/api.py by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1874
* Add Qwen2.5 math by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1863
* Add SmolLM2 by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1848
* Add Mistral-Large-Instruct-2411 by @ysjprojects in https://github.com/Lightning-AI/litgpt/pull/1876
* Bump version for 0.5.4 release by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1883
* Temporary remove Thunder to make a release by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1884

## New Contributors
* @ParagEkbote made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1821
* @ysjprojects made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1834
* @vra made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1855

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.3...v0.5.4

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.4)

---

## v0.5.3: v0.5.3
**Published:** 2024-10-29

## What's Changed
* Post-release setup for 0.5.3.dev1 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1799
* Add Phi 3 128k model by @deveworld in https://github.com/Lightning-AI/litgpt/pull/1800
* Add token counts to compute performance by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1801
* Fixed the issue that precision is always "32-true". by @jianpingw in https://github.com/Lightning-AI/litgpt/pull/1802
* Add Nvidia Llama 3.1 70B Nemotron weights by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1803
* Choose evaluation example from test set by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1804
* Pretrain tok sec by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1805
* typo in convert_to_litgpt command by @wasifferoze in https://github.com/Lightning-AI/litgpt/pull/1807
* Move distributed all_reduce import into a function by @IvanYashchuk in https://github.com/Lightning-AI/litgpt/pull/1810
* Remove hardcoded 32-precision conversion by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1814

## New Contributors
* @deveworld made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1800
* @jianpingw made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1802
* @wasifferoze made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1807
* @IvanYashchuk made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1810

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.2...v0.5.3

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.3)

---

## v0.5.2: v0.5.2
**Published:** 2024-10-22

## What's Changed
* Fix prompt style usage in evaluation example by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1790
* Pin to PyTorch 2.4.1 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1796
* Fix step iteration bug in finetuning scripts by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1794
* Housekeeping: remove unused code and fix tests by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1795


**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.1...v0.5.2

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.2)

---

## v0.5.1: v0.5.1
**Published:** 2024-10-10

## What's Changed
* v0.5.0 post release setup by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1774
* Be more specific about missign RoPE parameters by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1781
* Use correct Llama 3.1 and 3.2 context lengths by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1779
* Fixing Llama 3.1 and 3.2 Maximum Context Length by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1782
* Use more realistic RoPE tests by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1785
* AMD (MI250X) support by @TensorTemplar in https://github.com/Lightning-AI/litgpt/pull/1775
* Tidy up RoPE by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1786
* Bump version for 0.5.1 bugfix release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1787

## New Contributors
* @TensorTemplar made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1775

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.5.0...v0.5.1

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.1)

---

## v0.5.0: v0.5.0
**Published:** 2024-10-04

## What's Changed
* Post 0.4.13 release set up by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1755
* Add missing explanation on how to use a finetuned model by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1756
* Bump lightning version to latest stable release (2.4.0) by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1765
* Improve rope by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1745
* Add bnb.nn.StableEmbedding for quantized training by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1770
* [fix][1760] Added fix for the missing `context` key issue in dolly! by @pytholic in https://github.com/Lightning-AI/litgpt/pull/1766
* Fix Llama 3.2 tokenizer by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1772

## New Contributors
* @pytholic made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1766

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.13...v0.5.0

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.5.0)

---

## v0.4.13: v0.4.13
**Published:** 2024-09-27

## What's Changed
* Make 0.4.13.dev1 version by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1722
* Enable MPS support for LitGPT by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1724
* Simplify MPS support by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1726
* Add Chainlit Studio by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1728
* Fixing the tokenizer for slimpajama data preparation by @tomaslaz in https://github.com/Lightning-AI/litgpt/pull/1734
* Add pretrain conversion by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1735
* Typo fix and formatting improvements in API Trainer docs by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1736
* bump macos to m1 by @t-vi in https://github.com/Lightning-AI/litgpt/pull/1725
* Improve filepath handling in unit tests by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1737
* Add a more informative message in case text exceeds context size by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1738
* Update Thunder README.md by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1740
* Add sliding window attention to Mistral and Phi 3 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1741
* Extend context length for sliding window tests by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1742
* Fix jsonarparse version by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1748
* Update RoPE tests by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1746
* Make json parsing more robust by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1749
* Support for optimizers which don't have "fused" parameter such as grokadamw and 8bit bnb by @mtasic85 in https://github.com/Lightning-AI/litgpt/pull/1744
* Increase rtol and atol in Gemma 2 for macOS by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1751
* Repair json files by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1752
* Llama 3.2 weights by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1750
* Bump version to 0.4.13 for new release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1753
* Temporarily take out thunder dependency for deployment by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1754

## New Contributors
* @tomaslaz made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1734
* @mtasic85 made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1744

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.12...v0.4.13

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.13)

---

## v0.4.12: v0.4.12
**Published:** 2024-09-11

## What's Changed
* Preserve eos in encoding when max_seq_length = -1 by @sanderland in https://github.com/Lightning-AI/litgpt/pull/1694
* Add `batched_next_token()` and `batched_sample()` by @apaz-cli in https://github.com/Lightning-AI/litgpt/pull/1693
* bump thunder dependency to main by @t-vi in https://github.com/Lightning-AI/litgpt/pull/1701
* add support for batched input_pos to model by @t-vi in https://github.com/Lightning-AI/litgpt/pull/1700
* Add `batched_generate_fn()` by @apaz-cli in https://github.com/Lightning-AI/litgpt/pull/1702
* minor Readme update/typos by @Borda in https://github.com/Lightning-AI/litgpt/pull/1710
* Fix device Error in Decode Stream by @Motsepe-Jr in https://github.com/Lightning-AI/litgpt/pull/1707
* Temporarily remove MPS device support by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1718
* Add uvloop as a dependency by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1719
* Change default precision on macOS by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1720
* Bump for v0.4.12 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1721

## New Contributors
* @Motsepe-Jr made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1707

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.11...v0.4.12

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.12)

---

## v0.4.11: v0.4.11
**Published:** 2024-08-23

## What's Changed
* Add distribute=None to python-api.md by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1676
* Make LitGPT LLM API compatible with PyTorch Lightning Trainer 1/2 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1667
* Auto device handling in LLM API by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1677
* Fix KV cache issue in LLM API by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1678
* Improved benchmark utils by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1679
* Add PR benchmark util for internal use by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1680
* Added git hash to benchmark utility. by @apaz-cli in https://github.com/Lightning-AI/litgpt/pull/1681
* Spelling fix by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1685
* Add Microsoft Phi 3.5 checkpoint by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1687
* Update check_nvlink_connectivity by @sanderland in https://github.com/Lightning-AI/litgpt/pull/1684
* Make number of generated tokens consistent with CLI by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1690
* Avoid error when executing benchmark util outside a git folder by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1691
* Combine `generate()` functions by @apaz-cli in https://github.com/Lightning-AI/litgpt/pull/1675
* Bumb version to 0.4.11 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1695
* Fix falcon prompt template by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1696

## New Contributors
* @sanderland made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1684

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.10...v0.4.11

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.11)

---

## v0.4.10: v0.4.10
**Published:** 2024-08-14

## What's Changed
* Support Tensor Parallel in Python API by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1661
* Swap old Llama model with Phi-3 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1666
* Update azure-gpu-test.yml by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1669
* Support the refactored API in litgpt serve by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1668
* Multi-gpu serving by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1670
* Add Mistral Large 123B by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1673
* Bumb version to 0.4.10 for next release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1674


**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.9...v0.4.10

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.10)

---

## v0.4.9: v0.4.9
**Published:** 2024-08-08

## What's Changed

* Update LitServe version and tests by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1654
* Support for using large models in the Python API via sequential generation by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1637
* Add a PyTorch Lightning example by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1656
* Refactor Python API to introduce new distribute method (part of a larger refactor for PTL support) by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1657
* Fix some issues with circular and relative imports by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1658
* Optionally return benchmark info in Python API by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1660
* Bumb version for 0.4.9 release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1664


**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.8...v0.4.9

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.9)

---

## v0.4.8: v0.4.8
**Published:** 2024-08-02

## What's Changed
* Adds unit test to test for parity between streaming and non-streaming API by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1650
* Add Gemma 2 2B by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1651
* Pin litserve version by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1652
* Version bumb for Gemma 2 2B release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1653


**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.7...v0.4.8

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.8)

---

## v0.4.7: v0.4.7
**Published:** 2024-08-01

## What's Changed
* Apply prompt style for tp.py and sequentially.py by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1629
* Fix prompt docstring in Python API by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1635
* Update windows cpu-tests.yml by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1630
* Remove NumPy < 2.0 pin by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1631
* Fix kv-cache issue in Python API streaming mode by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1633
* Updates installation requirements to install minimal required packages for basic use by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1634
* Faster safetensors conversion when downloading model by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1624
* Add Sebastian as code owner by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1641
* Add missing super() call in data modules by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1639
* Update Lightning version to 2.4.0 pre by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1640
* Add tunable kvcache with error handling for nonsense inputs. by @apaz-cli in https://github.com/Lightning-AI/litgpt/pull/1636
* Use Python API in serve code by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1644
* Fix autodownload + conversion issue by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1645
* Properly clear kv-cache by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1647
* Fix error raising where max_returned_tokens > max_seq_length_setting by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1648
* Add quantization support to litgpt serve by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1646
* Bump for 0.4.7 release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1649


**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.6...v0.4.7

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.7)

---

## v0.4.6: v0.4.6
**Published:** 2024-07-24

## What's Changed
* Change default top_k to 50 everywhere for consistency by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1592
* Fix kv-cache clearing in Python API and Serve by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1596
* dynamic KV Cache batching by @aniketmaurya in https://github.com/Lightning-AI/litgpt/pull/1600
* Remove non-used eos_id in Python API by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1594
* Add quantization test and revert lightning version by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1605
* Dynamically set kv-cache size in serve by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1602
* Update LitData version and restore previous LitData assertions in tests by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1609
* Gemma 2: `9b` and `27b` versions by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1545
* Update config hub table qlora sections by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1611
* max_returned_tokens -> max_new_tokens by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1612
* Add warning about pretrain preprocessing by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1618
* Print warning about unsupported repo_ids by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1617
* Restore capability to load alternative weights by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1620
* Enable unbalanced number of layers in sequential generation by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1623
* Llama 3.1 8B and 70B checkpoints by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1619
* Add Llama 3.1 405B config by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1622
* Bumb version to 0.4.6 for next release (Gemma 2 and Llama 3.1) by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1626


**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.5...v0.4.6

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.6)

---

## v0.4.5: v0.4.5
**Published:** 2024-07-17

## What's Changed
* Update LitData to latest version 0.2.16 by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1573
* Fixed warning during model download and conversion. by @bhargavyagnik in https://github.com/Lightning-AI/litgpt/pull/1574
* Bump lightning version by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1579
* chat/base.py: extend checkpoint_dir before accessing it by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1575
* Chat: dynamically set kv-cache size by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1583
* Add slow interconnect warning by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1577
* Mathstral checkpoints by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1587
* Make sure quantization is used correctly by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1585
* Improve mathstral tests by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1589
* Fixes an issue where the LitGPT Python API was consuming too much memory by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1590
* Increase version number for 0.4.5 release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1591

## New Contributors
* @bhargavyagnik made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1574

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.4...v0.4.5

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.5)

---

## v0.4.4: v0.4.4
**Published:** 2024-07-11

## What's Changed
* Tokenizer: fix issue when decoding a single token at a time by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1559
* Filter out false positive PyTorch scheduler warning by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1562
* Add an explanation on how to evaluate on a custom test set by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1564
* Remove duplicated bos_token for CodeLlama by @alealv in https://github.com/Lightning-AI/litgpt/pull/1566
* Fix SFTDataset sample generation by @alealv in https://github.com/Lightning-AI/litgpt/pull/1568
* Add Python API section to 0 to LitGPT docs by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1570
* Fix multi-turn prompting error handling and extra newline by @apaz-cli in https://github.com/Lightning-AI/litgpt/pull/1569
* Add automatic downloading to CLI by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1571
* Bumb version for 0.4.4 release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1572

## New Contributors
* @alealv made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1566
* @apaz-cli made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1569

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.3...v0.4.4

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.4)

---

## v0.4.3: v0.4.3
**Published:** 2024-07-05

## What's Changed
* Fix Twine Build Issue by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1530
* Fix Mixtral prompt style by @janEbert in https://github.com/Lightning-AI/litgpt/pull/1531
* Raise warning when loading a large model on a CPU device by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1532
* Tutorial `convert_from_litgpt` doc fix to remove `output_dir` command argument by @jentfoo in https://github.com/Lightning-AI/litgpt/pull/1533
* Test against eager attention on GPTNeoX by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1537
* Bump to 0.4.3.dev0 version by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1536
* Fix for LM Eval harness 0.4.3 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1542
* Add `Phi-3-mini-4k-instruct` checkpoint by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1341
* Update phi-3 prompt template by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1544
* Add progress bar for weight loading by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1534
* Remove redundant code in checkpoint conversion script by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1547
* Update how evaluate tasks are listed by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1550
* Fix generate issue in finetuning script when finetuning with small datasets by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1551
* Tokenizer: check that `bos` and `eos` tokens are applied w.r.t. to a provided config/args by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1527
* Add num_nodes argument by @janEbert in https://github.com/Lightning-AI/litgpt/pull/1546
* Add Phi-3 Configs by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1553
* Update to version 0.4.3 for new release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1554

## New Contributors
* @janEbert made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1531
* @jentfoo made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1533

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.2...v0.4.3

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.3)

---

## v0.4.2: v0.4.2
**Published:** 2024-06-25

## What's Changed
* Disable "validating ..." comment if initial_validation is False by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1499
* Add simple issue templates by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1501
* Add note about dependencies by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1502
* Update test_readme.py by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1505
* Print message when unnecessary can't be deleted by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1507
* Bumb version by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1509
* LitGPT Python API v1 by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1463
* Fix pip upgrade command on windows by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1513
* add `LLM.generate` with streaming by @aniketmaurya in https://github.com/Lightning-AI/litgpt/pull/1511
* Fix and update Python API docs by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1514
* Update LLM import and make stream/non-stream option consistent by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1515
* Extend Python API to initialize model with random weights by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1512
* Add Python API download option by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1516
* Add Rope group size as constexpr to fix thunder tests by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1520
* Explain how to access pretraining dataset options by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1510
* Bump thunder by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1519
* Add automatic checkpoint downloading in Python API by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1521
* Tokenizer: prefer HF Tokenizer by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1518
* iterate response from streaming server by @aniketmaurya in https://github.com/Lightning-AI/litgpt/pull/1523
* Add new python API to readme by @williamFalcon in https://github.com/Lightning-AI/litgpt/pull/1525
* Remove `init="local"` option by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1524
* Readme improvements by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1526
* Add `evaluate` to Readme by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1528


**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.1...v0.4.2

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.2)

---

## v0.4.1: v0.4.1
**Published:** 2024-06-17

## What's Changed
* Fix typos in Download Model Weights documentation by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1477
* 'Add models' docs by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1478
* Exclude prompt from generated response by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1485
* Make litgpt serve consistent with LitServe by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1466
* add multi turn prompt by @aniketmaurya in https://github.com/Lightning-AI/litgpt/pull/1487
* Add docs explaining the streaming mode by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1484
* Fix Mistral 7B v0.3 weight download by @davmacario in https://github.com/Lightning-AI/litgpt/pull/1486
* Fixes for latest lm_eval by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1489
* bugfix: LLM generated text with containing `{KEY}` would result in error by @aniketmaurya in https://github.com/Lightning-AI/litgpt/pull/1490
* Add SAMBA to project gallery by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1493
* Add `resume="auto"` by @awaelchli in https://github.com/Lightning-AI/litgpt/pull/1492
* Fix imports by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1491
* Pin NumPy version by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1495
* Version bump for new release to fix NumPy issue by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1496


**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.4.0...v0.4.1

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.1)

---

## v0.4.0: v0.4.0
**Published:** 2024-06-10

## What's Changed
* Set litdata < 0.2.6 by @carmocca in https://github.com/Lightning-AI/litgpt/pull/1400
* Remove per-file CLIs by @carmocca in https://github.com/Lightning-AI/litgpt/pull/1397
* Simillar -> Similar by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1405
* LitData: set `iterate_over_all` to False for `CombinedStreamingDataset` by @Andrei-Aksionov in https://github.com/Lightning-AI/litgpt/pull/1404
* Allow multiline prompts by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1279
* Explain dataset options by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1407
* Support `no_sync` with Thunder FSDP by @carmocca in https://github.com/Lightning-AI/litgpt/pull/1414
* Minimal Python example by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1410
* Fix bug where LitData doesn't use seed by @bradfordlynch in https://github.com/Lightning-AI/litgpt/pull/1425
* Add prompt style mapping for llama3 by @davmacario in https://github.com/Lightning-AI/litgpt/pull/1406
* Simplify code by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1429
* OptimizerArgs by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1409
* Fix optimizer init with fused=True by @carmocca in https://github.com/Lightning-AI/litgpt/pull/1434
* Fix learning rate calculation in pretrain by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1435
* Align readme by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1438
* Pin litdata by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1440
* Fix README.md alignment by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1439
* Update README.md for one last time by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1442
* A more centered look by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1449
* New CLI by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1437
* Update error message by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1453
* Explain how to list all available models by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1455
* Detect tensor cores by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1456
* Check checkpoint_dir and add `checkpoints` to path by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1454
* Add MicroLlama training support by @keeeeenw in https://github.com/Lightning-AI/litgpt/pull/1457
* Streaming for serving with chat's generate function by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1426
* Fix sequence length bug by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1462
* Add `lr_warmup_steps`, `max_steps` values validation by @shenxiangzhuang in https://github.com/Lightning-AI/litgpt/pull/1460
* Fix issue where path in merge_lora is overwritten by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1465
* Option to skip expensive final validation by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1372
* Allow batch size "auto" setting in evaluate by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1469
* Warn users when there is a bnb mismatch by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1468
* Allow batch argument with batch recomputation by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1470
* LitGPT Python API draft by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1459
* Bump version for PyPI release by @rasbt in https://github.com/Lightning-AI/litgpt/pull/1476

## New Contributors
* @bradfordlynch made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1425
* @davmacario made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1406
* @keeeeenw made their first contribution in https://github.com/Lightning-AI/litgpt/pull/1457

**Full Changelog**: https://github.com/Lightning-AI/litgpt/compare/v0.3.1...v0.4.0

[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.0)

---

## v0.4.0.dev0: Development release 0.4.0.dev0
**Published:** 2024-06-04
**Pre-release**

## What's Changed
* Streaming for serving with chat's generate function by @rasbt in [#1426](https://github.com/Lightning-AI/litgpt/pull/1426)
* Add MicroLlama training support by @keeeeenw in [#1457](https://github.com/Lightning-AI/litgpt/pull/1457)
* Check checkpoint_dir and add `checkpoints` to path by @rasbt in [#1454](https://github.com/Lightning-AI/litgpt/pull/1454)
* Detect tensor cores by @rasbt in [#1456](https://github.com/Lightning-AI/litgpt/pull/1456)
* Explain how to list all available models by @rasbt in [#1455](https://github.com/Lightning-AI/litgpt/pull/1455)
* Update error message by @rasbt in [#1453](https://github.com/Lightning-AI/litgpt/pull/1453)
* New CLI by @rasbt in [#1437](https://github.com/Lightning-AI/litgpt/pull/1437)
* A more centered look by @rasbt in [#1449](https://github.com/Lightning-AI/litgpt/pull/1449)
* Update README.md for one last time by @rasbt in [#1442](https://github.com/Lightning-AI/litgpt/pull/1442)
* Fix README.md alignment by @rasbt in [#1439](https://github.com/Lightning-AI/litgpt/pull/1439)
* Pin litdata by @rasbt in [#1440](https://github.com/Lightning-AI/litgpt/pull/1440)
* Align readme by @rasbt in [#1438](https://github.com/Lightning-AI/litgpt/pull/1438)
* Fix learning rate calculation in pretrain by @rasbt in [#1435](https://github.com/Lightning-AI/litgpt/pull/1435)
* Fix optimizer init with fused=True by @carmocca in [#1434](https://github.com/Lightning-AI/litgpt/pull/1434)
* OptimizerArgs by @rasbt in [#1409](https://github.com/Lightning-AI/litgpt/pull/1409)
* Simplify code by @rasbt in [#1429](https://github.com/Lightning-AI/litgpt/pull/1429)
* Add prompt style mapping for llama3 by @davmacario in [#1406](https://github.com/Lightning-AI/litgpt/pull/1406)
* Fix bug where LitData doesn't use seed by @bradfordlynch in [#1425](https://github.com/Lightning-AI/litgpt/pull/1425)
* Minimal Python example by @rasbt in [#1410](https://github.com/Lightning-AI/litgpt/pull/1410)
* Support `no_sync` with Thunder FSDP by @carmocca in [#1414](https://github.com/Lightning-AI/litgpt/pull/1414)
* Explain dataset options by @rasbt in [#1407](https://github.com/Lightning-AI/litgpt/pull/1407)
* Allow multiline prompts by @rasbt in [#1279](https://github.com/Lightning-AI/litgpt/pull/1279)
* LitData: set `iterate_over_all` to False for `CombinedStreamingDataset` by @Andrei-Aksionov in [#1404](https://github.com/Lightning-AI/litgpt/pull/1404)
* Simillar -> Similar by @rasbt in [#1405](https://github.com/Lightning-AI/litgpt/pull/1405)
* Remove per-file CLIs by @carmocca in [#1397](https://github.com/Lightning-AI/litgpt/pull/1397)
* Set litdata < 0.2.6 by @carmocca in [#1400](https://github.com/Lightning-AI/litgpt/pull/1400)

**Full Changelog**: [View on GitHub](https://github.com/Lightning-AI/litgpt/commits/0.4.0.dev0)


[View on GitHub](https://github.com/Lightning-AI/litgpt/releases/tag/v0.4.0.dev0)

---

