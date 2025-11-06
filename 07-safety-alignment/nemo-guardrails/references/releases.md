# Releases

Version history for this repository (27 releases).

## v0.18.0: 
**Published:** 2025-11-06

## What's Changed

This release introduces BotThinking events, allowing you to apply guardrails to an LLM's reasoning traces. We added support for embedding providers: Azure OpenAI, Cohere, and Google. We also added an integration with Cisco AI Defense. For performance, in-memory caching can be used for Nemoguard [content-safety](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety), [topic-control](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control), and [jailbreak](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect) models to reduce latency. Other additions include automatic provider inference for LangChain LLMs and custom HTTP header support for ChatNVIDIA.

Reasoning trace extraction is refactored to use LangChain's `additional_kwargs`. Traces are no longer prepended to the bot's message but are now in a separate `reasoning` field. Please update any custom parsing logic. Additionally, `stream_async` will now correctly raise an error if output rail streaming is disabled.

We've fixed several stability bugs, especially for parallel execution. This includes ensuring the stop flag is now correctly set for policy violations and exceptions in parallel mode. We also added config validation at creation time for content-safety and topic-control rails. A fallback for legacy `<think>` tag extraction and a capitalization fix for `Snowflake` are also included.

We removed support for Python 3.9, ahead of its EOL in October 2025. We invested in code quality by adding type-annotations and pre-commit checks across a majority of the codebase.

### 🚀 Features

- *(bot-thinking)* Implement BotThinking events to process reasoning traces in Guardrails ([#1431](https://github.com/NVIDIA/NeMo-Guardrails/issues/1431)), ([#1432](https://github.com/NVIDIA/NeMo-Guardrails/issues/1432)), ([#1434](https://github.com/NVIDIA/NeMo-Guardrails/issues/1434)).
- *(embeddings)* Add Azure OpenAI embedding provider ([#702](https://github.com/NVIDIA/NeMo-Guardrails/issues/702)).
- *(embeddings)* Add Cohere embedding integration ([#1305](https://github.com/NVIDIA/NeMo-Guardrails/issues/1305)).
- *(embeddings)* Add Google embedding integration ([#1304](https://github.com/NVIDIA/NeMo-Guardrails/issues/1304)).
- *(library)* Add Cisco AI Defense integration ([#1433](https://github.com/NVIDIA/NeMo-Guardrails/issues/1433)).
- *(cache)* Add in-memory LFU caches for content-safety, topic-control, and jailbreak detection models ([#1436](https://github.com/NVIDIA/NeMo-Guardrails/issues/1436)), ([#1456](https://github.com/NVIDIA/NeMo-Guardrails/issues/1456)),  ([#1457](https://github.com/NVIDIA/NeMo-Guardrails/issues/1457)), ([#1458](https://github.com/NVIDIA/NeMo-Guardrails/issues/1458)).
- *(llm)* Add automatic provider inference for LangChain LLMs ([#1460](https://github.com/NVIDIA/NeMo-Guardrails/issues/1460)).
- *(llm)* Add custom HTTP headers support to ChatNVIDIA provider ([#1461](https://github.com/NVIDIA/NeMo-Guardrails/issues/1461)).

### 🐛 Bug Fixes

- *(config)* Validate content safety and topic control configs at creation time ([#1450](https://github.com/NVIDIA/NeMo-Guardrails/issues/1450)).
- *(jailbreak)* Capitalization of `Snowflake` in use of `snowflake-arctic-embed-m-long` name. ([#1464](https://github.com/NVIDIA/NeMo-Guardrails/issues/1464)).
- *(runtime)* Ensure stop flag is set for policy violations in parallel rails ([#1467](https://github.com/NVIDIA/NeMo-Guardrails/issues/1467)).
- *(llm)* [**breaking**] Extract reasoning traces to separate field instead of prepending ([#1468](https://github.com/NVIDIA/NeMo-Guardrails/issues/1468)).
- *(streaming)* [**breaking**] Raise error when stream_async used with disabled output rails streaming ([#1470](https://github.com/NVIDIA/NeMo-Guardrails/issues/1470)).
- *(llm)* Add fallback extraction for reasoning traces from <think> tags ([#1474](https://github.com/NVIDIA/NeMo-Guardrails/issues/1474)).
- *(runtime)* Set stop flag for exception-based rails in parallel mode ([#1487](https://github.com/NVIDIA/NeMo-Guardrails/issues/1487)).

### 🚜 Refactor

- [**breaking**] Replace reasoning trace extraction with LangChain additional_kwargs ([#1427](https://github.com/NVIDIA/NeMo-Guardrails/issues/1427))

### 📚 Documentation

- *(examples)* Add Nemoguard in-memory cache configuration example ([#1459](https://github.com/NVIDIA/NeMo-Guardrails/issues/1459)), ([#1480](https://github.com/NVIDIA/NeMo-Guardrails/issues/1480)).
- Add guide for bot reasoning guardrails ([#1479](https://github.com/NVIDIA/NeMo-Guardrails/issues/1479)).
- Update LLM reasoning traces configuration ([#1483](https://github.com/NVIDIA/NeMo-Guardrails/issues/1483)).

### 🧪 Testing

- Add mock embedding provider tests ([#1446](https://github.com/NVIDIA/NeMo-Guardrails/issues/1446))
- *(cli)* Add comprehensive CLI test suite and reorganize files ([#1339](https://github.com/NVIDIA/NeMo-Guardrails/issues/1339))
- Skip FastEmbed tests when not in live mode ([#1462](https://github.com/NVIDIA/NeMo-Guardrails/issues/1462))
- Fix flaky stats logging interval timing test ([#1463](https://github.com/NVIDIA/NeMo-Guardrails/issues/1463))
- Restore test that was skipped due to Colang 2.0 serialization issue ([#1449](https://github.com/NVIDIA/NeMo-Guardrails/issues/1449))

### ⚙️ Miscellaneous Tasks

- Resolve PyPI publish workflow trigger and reliability issues ([#1443](https://github.com/NVIDIA/NeMo-Guardrails/issues/1443))
- Fix sparse checkout for publish pypi workflow ([#1444](https://github.com/NVIDIA/NeMo-Guardrails/issues/1444))
- Drop Python 3.9 support ahead of October 2025 EOL ([#1426](https://github.com/NVIDIA/NeMo-Guardrails/issues/1426))
- *(types)* Add type-annotations and pre-commit checks for tracing ([#1388](https://github.com/NVIDIA/NeMo-Guardrails/issues/1388)), logging ([#1395](https://github.com/NVIDIA/NeMo-Guardrails/issues/1395)), kb  ([#1385](https://github.com/NVIDIA/NeMo-Guardrails/issues/1385)), cli ([#1380](https://github.com/NVIDIA/NeMo-Guardrails/issues/1380)), embeddings ([#1383](https://github.com/NVIDIA/NeMo-Guardrails/issues/1383)), server ([#1397](https://github.com/NVIDIA/NeMo-Guardrails/issues/1397)), and llm ([#1394](https://github.com/NVIDIA/NeMo-Guardrails/issues/1394)) code.
- Update insert licenser pe-commit-hooks to use current year ([#1452](https://github.com/NVIDIA/NeMo-Guardrails/issues/1452)).
- *(library)* Remove unused vllm requirements.txt files ([#1466](https://github.com/NVIDIA/NeMo-Guardrails/issues/1466)).


[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.18.0)

---

## v0.17.0: v0.17.0
**Published:** 2025-10-09

## What's Changed

In this release, we add support for integrating with LangGraph and tool calling, to build safe and controlled multi-agent workflows. LangGraph enables you to create sophisticated agent architectures with state management, conditional routing, and tool calling, while NeMo Guardrails provides the safety layer to ensure responsible AI behavior.
We also enhanced our support for integrating with LangChain RunnableRails. This release supports synchronous and asynchronous operations, streaming, and batch processing.

Our newest community integration comes from Trend Micro, whose [AI Guard](https://docs.trendmicro.com/en-us/documentation/article/trend-vision-one-ai-scanner-ai-guard) can now be accessed through NeMo Guardrails to scan for harmful content generation, sensitive information leakage, and prompt injections.

### 🚀 Features

- *(tool-calling)* Add tool call passthrough support in LLMRails ([#1364](https://github.com/NVIDIA/NeMo-Guardrails/issues/1364))
- *(runnable-rails)* Complete rewrite of RunnableRails with full LangChain Runnable protocol support ([#1366](https://github.com/NVIDIA/NeMo-Guardrails/issues/1366), [#1369](https://github.com/NVIDIA/NeMo-Guardrails/issues/1369), [#1370](https://github.com/NVIDIA/NeMo-Guardrails/issues/1370), [#1405](https://github.com/NVIDIA/NeMo-Guardrails/issues/1405))
- *(tool-rails)* Add support for tool output rails and validation ([#1382](https://github.com/NVIDIA/NeMo-Guardrails/issues/1382))
- *(tool-rails)* Implement tool input rails for tool message validation and processing ([#1386](https://github.com/NVIDIA/NeMo-Guardrails/issues/1386))
- *(library)* Add Trend Micro Vision One AI Application Security community integration ([#1355](https://github.com/NVIDIA/NeMo-Guardrails/issues/1355))
- *(llm)* Pass llm params directly ([#1387](https://github.com/NVIDIA/NeMo-Guardrails/issues/1387))

### 🐛 Bug Fixes

- *(jailbreak)* Handle URL joining with/without trailing slashes ([#1346](https://github.com/NVIDIA/NeMo-Guardrails/issues/1346))
- *(logging)* Handle missing id and task in verbose logs ([#1343](https://github.com/NVIDIA/NeMo-Guardrails/issues/1343))
- *(library)* Fix import package declaration to new cleanlab-tlm name ([#1401](https://github.com/NVIDIA/NeMo-Guardrails/issues/1401))
- *(logging)* Add "Tool" type to message sender labeling ([#1412](https://github.com/NVIDIA/NeMo-Guardrails/issues/1412))
- *(logging)* Correct message type formatting in logs ([#1416](https://github.com/NVIDIA/NeMo-Guardrails/issues/1416))

### 🚜 Refactor

- *(llm)* Remove LLMs isolation for actions ([#1408](https://github.com/NVIDIA/NeMo-Guardrails/issues/1408))

### 📚 Documentation

- *(examples)* Add NeMoGuard safety rails config example for Colang 1.0 ([#1365](https://github.com/NVIDIA/NeMo-Guardrails/issues/1365))
- Add hardware reqs ([#1411](https://github.com/NVIDIA/NeMo-Guardrails/issues/1411))
- Add tools integration guide ([#1414](https://github.com/NVIDIA/NeMo-Guardrails/issues/1414))
- *(langgraph)* Add integration guide for LangGraph ([#1422](https://github.com/NVIDIA/NeMo-Guardrails/issues/1422))
- *(langchain)* Update with full support and add tool calling guide … ([#1419](https://github.com/NVIDIA/NeMo-Guardrails/issues/1419))
- *(langgraph)* Clarify tool examples and replace calculate_math with multiply ([#1439](https://github.com/NVIDIA/NeMo-Guardrails/issues/1439))

### ⚙️ Miscellaneous Tasks

- *(docs)* Update v0.16.0 release date in changelog ([#1377](https://github.com/NVIDIA/NeMo-Guardrails/issues/1377))
- *(docs)* Add link to demo.py script in Getting-Started section ([#1399](https://github.com/NVIDIA/NeMo-Guardrails/issues/1399))
- *(types)* Type-clean rails (86 errors) ([#1396](https://github.com/NVIDIA/NeMo-Guardrails/issues/1396))
- *(jailbreak-detection)* Update transformers and torch ([#1417](https://github.com/NVIDIA/NeMo-Guardrails/issues/1417))
- *(types)* Type-clean /actions (189 errors) ([#1361](https://github.com/NVIDIA/NeMo-Guardrails/issues/1361))
- *(docs)* Update repository owner ([#1425](https://github.com/NVIDIA/NeMo-Guardrails/issues/1425))


[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.17.0)

---

## v0.16.0: v0.16.0
**Published:** 2025-09-05

## [0.16.0] - 2025-09-05

### Key Features

- Enhanced tracing system with [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). To learn more, refer to [](tracing). For usage examples, refer to the following notebooks
  - [Tracing Guardrails Quickstart](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/docs/getting-started/8-tracing/1_tracing_quickstart.ipynb)
  - [Tracing Guardrails with Jaeger](https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/docs/getting-started/8-tracing/2_tracing_with_jaeger.ipynb)
- Community integration with [GuardrailsAI](https://www.guardrailsai.com/) and [Pangea AI Guard](https://pangea.cloud/services/ai-guard).

### Other Changes

- Added documentation about using KV cache reuse for LLM-based NemoGuard NIMs. By using KV cache reuse, you can improve the performance of LLM-based NemoGuard NIMs where the system prompt is the same for all calls up to the point where user query and LLM response are injected. To learn more, refer to [](kv-cache-reuse).

----------

### 🚀 Features

- *(llmrails)* Support method chaining by returning self from LLMRails.register_* methods ([#1296](https://github.com/NVIDIA/NeMo-Guardrails/issues/1296))
- Add Pangea AI Guard community integration ([#1300](https://github.com/NVIDIA/NeMo-Guardrails/issues/1300))
- *(llmrails)* Isolate LLMs only for configured actions ([#1342](https://github.com/NVIDIA/NeMo-Guardrails/issues/1342))
- Enhance tracing system with OpenTelemetry semantic conventions ([#1331](https://github.com/NVIDIA/NeMo-Guardrails/issues/1331))
- Add GuardrailsAI community integration ([#1298](https://github.com/NVIDIA/NeMo-Guardrails/issues/1298))

### 🐛 Bug Fixes

- *(models)* Suppress langchain_nvidia_ai_endpoints warnings ([#1371](https://github.com/NVIDIA/NeMo-Guardrails/issues/1371))
- *(tracing)* Respect the user-provided log options regardless of tracing configuration
- *(config)* Ensure adding RailsConfig objects handles None values ([#1328](https://github.com/NVIDIA/NeMo-Guardrails/issues/1328))
- *(config)* Add handling for config directory with `.yml`/`.yaml` extension ([#1293](https://github.com/NVIDIA/NeMo-Guardrails/issues/1293))
- *(colang)* Apply guardrails transformations to LLM inputs and bot outputs. ([#1297](https://github.com/NVIDIA/NeMo-Guardrails/issues/1297))
- *(topic_safety)* Handle InternalEvent objects in topic safety actions for Colang 2.0 ([#1335](https://github.com/NVIDIA/NeMo-Guardrails/issues/1335))
- *(prompts)* Prevent IndexError when LLM provided via constructor with empty models config ([#1334](https://github.com/NVIDIA/NeMo-Guardrails/issues/1334))
- *(llmrails)* Handle LLM models without model_kwargs field in isolation ([#1336](https://github.com/NVIDIA/NeMo-Guardrails/issues/1336))
- *(llmrails)* Move LLM isolation setup to after KB initialization ([#1348](https://github.com/NVIDIA/NeMo-Guardrails/issues/1348))

### 🚜 Refactor

- *(llm)* Move get_action_details_from_flow_id from llmrails.py to utils.py ([#1341](https://github.com/NVIDIA/NeMo-Guardrails/issues/1341))

### 📚 Documentation

- Integrate with multilingual NIM ([#1354](https://github.com/NVIDIA/NeMo-Guardrails/issues/1354))
- *(tracing)* Update tracing notebooks with VDR feedback ([#1376](https://github.com/NVIDIA/NeMo-Guardrails/issues/1376))
- Add kv cache reuse documentation ([#1330](https://github.com/NVIDIA/NeMo-Guardrails/issues/1330))
- *(examples)* Add Colang 2.0 example for sensitive data detection ([#1301](https://github.com/NVIDIA/NeMo-Guardrails/issues/1301))
- Add extra slash to jailbreak detect nim_base_url([#1345](https://github.com/NVIDIA/NeMo-Guardrails/issues/1345))
- Add tracing notebook ([#1337](https://github.com/NVIDIA/NeMo-Guardrails/issues/1337))
- Jaeger tracing notebook ([#1353](https://github.com/NVIDIA/NeMo-Guardrails/issues/1353))
- *(examples)* Add NeMoGuard rails config for colang 2 ([#1289](https://github.com/NVIDIA/NeMo-Guardrails/issues/1289))
- *(tracing)* Add OpenTelemetry span format guide ([#1350](https://github.com/NVIDIA/NeMo-Guardrails/issues/1350))
- Add GuardrailsAI integration user guide and example ([#1357](https://github.com/NVIDIA/NeMo-Guardrails/issues/1357))

### 🧪 Testing

- *(jailbreak)* Add missing pytest.mark.asyncio decorators ([#1352](https://github.com/NVIDIA/NeMo-Guardrails/issues/1352))

### ⚙️ Miscellaneous Tasks

- *(docs)* Rename test_csl.py to csl.py ([#1347](https://github.com/NVIDIA/NeMo-Guardrails/issues/1347))

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.16.0)

---

## v0.15.0: v0.15.0
**Published:** 2025-08-08

## What's Changed

This release introduces performance improvements with parallel rails execution capabilities for both standard and streaming scenarios, enhanced tracing infrastructure using OpenTelemetry API, and several critical bug fixes for streaming functionality. The new parallel rails execution in Colang 1.0 can reduce latency when multiple rails are configured, as they now run concurrently rather than sequentially.

The migration to OpenTelemetry provides standardized observability with better integration into existing monitoring stacks, enabling detailed insights into LLM calls, rail execution times, and token usage across your guardrails. We intend to expand this observability in future release with adapting semantic conventions for Generative AI Systems.
Critical streaming fixes address word spacing issues in output rails (where text was being concatenated without proper delimiters) and restore accurate token counting for streaming LLM calls. These enhancements collectively deliver a more performant and observable guardrails system while maintaining full backward compatibility for existing Colang 1 configurations.

### 🚀 Features

- *(tracing)* [**breaking**] Update tracing to use otel api ([#1269](https://github.com/NVIDIA/NeMo-Guardrails/issues/1269))
- *(streaming)* Implement parallel streaming output rails execution ([#1263](https://github.com/NVIDIA/NeMo-Guardrails/issues/1263), [#1324](https://github.com/NVIDIA/NeMo-Guardrails/pull/1324))
- *(streaming)* Support external async token generators ([#1286](https://github.com/NVIDIA/NeMo-Guardrails/issues/1286))
- Support parallel rails execution ([#1234](https://github.com/NVIDIA/NeMo-Guardrails/issues/1234), [#1323](https://github.com/NVIDIA/NeMo-Guardrails/pull/1323))

### 🐛 Bug Fixes

- *(streaming)* Resolve word concatenation in streaming output rails ([#1259](https://github.com/NVIDIA/NeMo-Guardrails/issues/1259))
- *(streaming)* Enable token usage tracking for streaming LLM calls ([#1264](https://github.com/NVIDIA/NeMo-Guardrails/issues/1264), [#1285](https://github.com/NVIDIA/NeMo-Guardrails/issues/1285))
- *(tracing)* Prevent mutation of user options when tracing is enabled ([#1273](https://github.com/NVIDIA/NeMo-Guardrails/issues/1273))
- *(rails)* Prevent LLM parameter contamination in rails ([#1306](https://github.com/NVIDIA/NeMo-Guardrails/issues/1306))

### 📚 Documentation

- Release notes 0.14.1 ([#1272](https://github.com/NVIDIA/NeMo-Guardrails/issues/1272))
- Update guardrails-library.md to include Clavata as a third party API ([#1294](https://github.com/NVIDIA/NeMo-Guardrails/issues/1294))
- *(streaming)* Add section on token usage tracking ([#1282](https://github.com/NVIDIA/NeMo-Guardrails/issues/1282))
- Add parallel rail section and split config page ([#1295](https://github.com/NVIDIA/NeMo-Guardrails/issues/1295))
- Show complete prompts.yml content in getting started tutorial ([#1311](https://github.com/NVIDIA/NeMo-Guardrails/issues/1311))
- *(tracing)* Update and streamline tracing guide ([#1307](https://github.com/NVIDIA/NeMo-Guardrails/issues/1307))

### ⚙️ Miscellaneous Tasks

- *(dependabot)* Remove dependabot configuration ([#1281](https://github.com/NVIDIA/NeMo-Guardrails/issues/1281))
- *(CI)* Add release workflow ([#1309](https://github.com/NVIDIA/NeMo-Guardrails/issues/1309), [#1318](https://github.com/NVIDIA/NeMo-Guardrails/issues/1318))

## New Contributors
* @ashwini made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1294

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.14.1...v0.15.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.15.0)

---

## v0.14.1: v0.14.1
**Published:** 2025-07-08

## What's Changed

This patch release addresses several important bug fixes and improvements, particularly focusing on jailbreak detection, LLMRails configuration handling, and content safety compatibility issues.


### 🚀 Features

- *(jailbreak)* Add direct API key configuration support ([#1260](https://github.com/NVIDIA/NeMo-Guardrails/issues/1260))

### 🐛 Bug Fixes

- *(jailbreak)* Lazy load jailbreak detection dependencies ([#1223](https://github.com/NVIDIA/NeMo-Guardrails/issues/1223))
- *(llmrails)* Constructor LLM should not skip loading other config models ([#1221](https://github.com/NVIDIA/NeMo-Guardrails/issues/1221), [#1247](https://github.com/NVIDIA/NeMo-Guardrails/issues/1247), [#1250](https://github.com/NVIDIA/NeMo-Guardrails/issues/1250), [#1258](https://github.com/NVIDIA/NeMo-Guardrails/issues/1258))
- *(content_safety)* Replace try-except with iterable unpacking for policy violations ([#1207](https://github.com/NVIDIA/NeMo-Guardrails/issues/1207))
- *(jailbreak)* Pin numpy==1.23.5 for scikit-learn compatibility ([#1249](https://github.com/NVIDIA/NeMo-Guardrails/issues/1249))
- *(output_parsers)* Iterable unpacking compatibility in content safety parsers ([#1242](https://github.com/NVIDIA/NeMo-Guardrails/issues/1242))

### 📚 Documentation

- More heading levels so RNs resolve links ([#1228](https://github.com/NVIDIA/NeMo-Guardrails/issues/1228))
- Update docs version ([#1219](https://github.com/NVIDIA/NeMo-Guardrails/issues/1219))
- Fix jailbreak detection build instructions ([#1248](https://github.com/NVIDIA/NeMo-Guardrails/issues/1248))
- Change ABC bot link at docs ([#1261]([#1248](https://github.com/NVIDIA/NeMo-Guardrails/issues/1261)))

### 🧪 Testing

- Fix async test failures in cache embeddings and buffer strategy tests ([#1237](https://github.com/NVIDIA/NeMo-Guardrails/issues/1237))
- *(content_safety)* Add tests for content safety actions ([#1240](https://github.com/NVIDIA/NeMo-Guardrails/issues/1240))

### ⚙️ Miscellaneous Tasks

- Update pre-commit-hooks to v5.0.0 ([#1238](https://github.com/NVIDIA/NeMo-Guardrails/issues/1238))

## New Contributors
* @jeffreyscarpenter made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1223
* @gcr made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1207
* @bwook00 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1261

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.14.0...v0.14.1

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.14.1)

---

## v0.14.0: v0.14.0
**Published:** 2025-06-11

## What's Changed

This release enhances support for advanced reasoning models, including the NVIDIA Nemotron family such as Llama 3.1 Nemotron Ultra 253B V1 and DeepSeek-r1. We have added new protections against certain code-injection security concerns, with an output guardrail based on YARA, a widely-adopted technology in the cybersecurity space for malware detection. 

From the community, a new 3rd-party guardrail connects to [Clavata](https://www.clavata.ai/)’s customized LLM content moderation.
Among other improvements and bug fixes, this release also now supports Python 3.13.

### 🚀 Features

- Change topic following prompt to allow chitchat ([#1097](https://github.com/NVIDIA/NeMo-Guardrails/issues/1097))
- Validate model name configuration ([#1084](https://github.com/NVIDIA/NeMo-Guardrails/issues/1084))
- Add support for langchain partner and community chat models ([#1085](https://github.com/NVIDIA/NeMo-Guardrails/issues/1085))
- Add fuzzy find provider capability to cli ([#1088](https://github.com/NVIDIA/NeMo-Guardrails/issues/1088))
- Add code injection detection to guardrails library ([#1091](https://github.com/NVIDIA/NeMo-Guardrails/issues/1091))
- Add clavata community integration ([#1027](https://github.com/NVIDIA/NeMo-Guardrails/issues/1027))
- Implement validation to forbid dialog rails with reasoning traces ([#1137](https://github.com/NVIDIA/NeMo-Guardrails/issues/1137))
- Load yara lazily to avoid action dispatcher error ([#1162](https://github.com/NVIDIA/NeMo-Guardrails/issues/1162))
- Add support for system messages to RunnableRails ([#1106](https://github.com/NVIDIA/NeMo-Guardrails/issues/1106))
- Add api_key_env_var to Model, pass in kwargs to langchain initializer ([#1142](https://github.com/NVIDIA/NeMo-Guardrails/issues/1142))
- Add inline YARA rules support ([#1164](https://github.com/NVIDIA/NeMo-Guardrails/issues/1164))
- [**breaking**] Add support for preserving and optionally applying guardrails to reasoning traces ([#1145](https://github.com/NVIDIA/NeMo-Guardrails/issues/1145))
- Prevent reasoning traces from contaminating LLM prompt history ([#1169](https://github.com/NVIDIA/NeMo-Guardrails/issues/1169))
- Add RailException support to injection detection and improve error handling ([#1178](https://github.com/NVIDIA/NeMo-Guardrails/issues/1178))
- Add Nemotron model support with message-based prompts ([#1199](https://github.com/NVIDIA/NeMo-Guardrails/issues/1199))

### 🐛 Bug Fixes

- Correct task name for self_check_facts ([#1040](https://github.com/NVIDIA/NeMo-Guardrails/issues/1040))
- Error in LLMRails with tracing enabled ([#1103](https://github.com/NVIDIA/NeMo-Guardrails/issues/1103))
- Self check output colang 1 flow ([#1126](https://github.com/NVIDIA/NeMo-Guardrails/issues/1126))
- Use ValueError in TaskPrompt to resolve TypeError raised by Pydantic ([#1132](https://github.com/NVIDIA/NeMo-Guardrails/issues/1132))
- Correct dialog rails activation logic ([#1161](https://github.com/NVIDIA/NeMo-Guardrails/issues/1161))
- Allow reasoning traces when embeddings_only is True ([#1170](https://github.com/NVIDIA/NeMo-Guardrails/issues/1170))
- Prevent explain_info overwrite during stream_async ([#1194](https://github.com/NVIDIA/NeMo-Guardrails/issues/1194))
- Colang 2 issues in community integrations ([#1140](https://github.com/NVIDIA/NeMo-Guardrails/issues/1140))
- Ensure proper asyncio task cleanup in test_streaming_handler.py ([#1182](https://github.com/NVIDIA/NeMo-Guardrails/issues/1182))

### 🚜 Refactor

- Reorganize HuggingFace provider structure ([#1083](https://github.com/NVIDIA/NeMo-Guardrails/issues/1083))
- Remove support for deprecated nemollm engine ([#1076](https://github.com/NVIDIA/NeMo-Guardrails/issues/1076))
- [**breaking**] Remove deprecated return_context argument ([#1147](https://github.com/NVIDIA/NeMo-Guardrails/issues/1147))
- Rename `remove_thinking_traces` field to `remove_reasoning_traces` ([#1176](https://github.com/NVIDIA/NeMo-Guardrails/issues/1176))
- Update deprecated field handling  for remove_thinking_traces ([#1196](https://github.com/NVIDIA/NeMo-Guardrails/issues/1196))
- Introduce END_OF_STREAM sentinel and update handling ([#1185](https://github.com/NVIDIA/NeMo-Guardrails/issues/1185))

### 📚 Documentation

- Remove markup from code block ([#1081](https://github.com/NVIDIA/NeMo-Guardrails/issues/1081))
- Replace img tag with Markdown images ([#1087](https://github.com/NVIDIA/NeMo-Guardrails/issues/1087))
- Remove NeMo Service (nemollm) documentation ([#1077](https://github.com/NVIDIA/NeMo-Guardrails/issues/1077))
- Update cleanlab integration description ([#1080](https://github.com/NVIDIA/NeMo-Guardrails/issues/1080))
- Add providers fuzzy search cli command ([#1089](https://github.com/NVIDIA/NeMo-Guardrails/issues/1089))
- Clarify purpose of model parameters field in configuration guide ([#1181](https://github.com/NVIDIA/NeMo-Guardrails/issues/1181))
- Output rails are supported with streaming ([#1007](https://github.com/NVIDIA/NeMo-Guardrails/issues/1007))
- Add mention of Nemotron ([#1200](https://github.com/NVIDIA/NeMo-Guardrails/issues/1200))
- Fix output rail doc ([#1159](https://github.com/NVIDIA/NeMo-Guardrails/issues/1159))
- Revise GS example in getting started doc ([#1146](https://github.com/NVIDIA/NeMo-Guardrails/issues/1146))
- Possible update to injection detection ([#1144](https://github.com/NVIDIA/NeMo-Guardrails/issues/1144))

### ⚙️ Miscellaneous Tasks

- Dynamically set version using importlib.metadata ([#1072](https://github.com/NVIDIA/NeMo-Guardrails/issues/1072))
- Add link to topic control config and prompts ([#1098](https://github.com/NVIDIA/NeMo-Guardrails/issues/1098))
- Reorganize GitHub workflows for better test coverage ([#1079](https://github.com/NVIDIA/NeMo-Guardrails/issues/1079))
- Add summary jobs for workflow branch protection ([#1120](https://github.com/NVIDIA/NeMo-Guardrails/issues/1120))
- Add Adobe Analytics configuration ([#1138](https://github.com/NVIDIA/NeMo-Guardrails/issues/1138))
- Fix and revert poetry lock to its stable state ([#1133](https://github.com/NVIDIA/NeMo-Guardrails/issues/1133))
- Add Codecov integration to workflows ([#1143](https://github.com/NVIDIA/NeMo-Guardrails/issues/1143))
- Add Python 3.12 and 3.13 test jobs to gitlab workflow ([#1171](https://github.com/NVIDIA/NeMo-Guardrails/issues/1171))
- Identify OS packages to install in contribution guide([#1136](https://github.com/NVIDIA/NeMo-Guardrails/issues/1136))
- Remove Got It AI from ToC in 3rd party docs([#1213](https://github.com/NVIDIA/NeMo-Guardrails/issues/1213))

## New Contributors
* @thisthat made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1103
* @jwmueller made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1080
* @ilias-t made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1027
* @smruthi33 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1106
* @andompesta made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1194

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.13.0...v0.14.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.14.0)

---

## v0.13.0: v0.13.0
**Published:** 2025-03-25

## What's Changed

This release adds support for multimodal input and output rails, allowing users to check for the safety of both text and image prompts and vision-language model outputs. We introduce new support for models with reasoning traces, such as Deepseek-R1, and we have added support for the NemoGuard JailbreakDetect NIM.
This release adds a new integration with Fiddler Guardrails, providing access to Fiddler’s low-latency hosted guardrail models for content safety, hallucination detection, and more. We also added support for Python 3.12 . 




### 🚀 Features

- Support models with reasoning traces ([#996](https://github.com/NVIDIA/NeMo-Guardrails/issues/996))  by @trebedea 
- Add SHA-256 hashing option ([#988](https://github.com/NVIDIA/NeMo-Guardrails/issues/988)) by @mdambski 
- Add Fiddler Guardrails integration ([#964](https://github.com/NVIDIA/NeMo-Guardrails/issues/964), [#1043](https://github.com/NVIDIA/NeMo-Guardrails/issues/1043)) by @copperstick6 
- Add generation metadata to streaming chunks ([#1011](https://github.com/NVIDIA/NeMo-Guardrails/issues/1011)) by @Pouyanpi 
- Improve alpha to beta bot migration ([#878](https://github.com/NVIDIA/NeMo-Guardrails/issues/878)) by @schuellc-nvidia 
- Support multimodal input and output rails ([#1033](https://github.com/NVIDIA/NeMo-Guardrails/issues/1033)) by @Pouyanpi 
- Add support for NemoGuard JailbreakDetect NIM.  ([#1038](https://github.com/NVIDIA/NeMo-Guardrails/issues/1038)) by @erickgalinkin 
- Set default start and end reasoning tokens ([#1050](https://github.com/NVIDIA/NeMo-Guardrails/issues/1050)) by @Pouyanpi 
- Improve output rails error handling for SSE format ([#1058](https://github.com/NVIDIA/NeMo-Guardrails/issues/1058)) by @Pouyanpi 

### 🐛 Bug Fixes

- Ensure parse_task_output is called after all llm_call invocations ([#1047](https://github.com/NVIDIA/NeMo-Guardrails/issues/1047)) by @Pouyanpi 
- Handle exceptions in generate_events to propagate errors in streaming ([#1012](https://github.com/NVIDIA/NeMo-Guardrails/issues/1012)) by @Pouyanpi 
- Ensure output rails streaming is enabled explicitly ([#1045](https://github.com/NVIDIA/NeMo-Guardrails/issues/1045)) by @Pouyanpi 
- Improve multimodal prompt length calculation for base64 images ([#1053](https://github.com/NVIDIA/NeMo-Guardrails/issues/1053)) by @Pouyanpi 

### 🚜 Refactor

- Move startup and shutdown logic to lifespan in server  ([#999](https://github.com/NVIDIA/NeMo-Guardrails/issues/999)) by @xiaobo8204 

### 📚 Documentation

- Add multimodal rails documentation ([#1061](https://github.com/NVIDIA/NeMo-Guardrails/issues/1061)) by @mikemckiernan 
- Add content safety tutorial ([#1042](https://github.com/NVIDIA/NeMo-Guardrails/issues/1042)) by @tgasser-nv 
- Revise reasoning model info ([#1062](https://github.com/NVIDIA/NeMo-Guardrails/issues/1062)) by @mikemckiernan 
- Consider new GS experience ([#1005](https://github.com/NVIDIA/NeMo-Guardrails/issues/1005)) by @mikemckiernan 
- Restore deleted configuration files ([#963](https://github.com/NVIDIA/NeMo-Guardrails/issues/963)) by @Pouyanpi 

### ⚙️ Miscellaneous Tasks

- Add Python 3.12 support ([#984](https://github.com/NVIDIA/NeMo-Guardrails/issues/984)) by @Pouyanpi 

## New Contributors
* @mdambski made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/988
* @copperstick6 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/964
* @xiaobo8204 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/999
* @tgasser-nv made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/1042

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.12.0...v0.13.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.13.0)

---

## v0.12.0: v0.12.0
**Published:** 2025-02-26

## What's Changed

This release adds support for streaming LLM content through output rails for more responsive, real-time interactions . It also includes a new integration with Prompt Security, as well as updates to our integrations with ActiveFence and Private AI.

### 🚀 Features

- Support Output Rails Streaming ([#966](https://github.com/NVIDIA/NeMo-Guardrails/issues/966), [#1003](https://github.com/NVIDIA/NeMo-Guardrails/issues/1003)) by @Pouyanpi
- Add unified output mapping for actions ([#965](https://github.com/NVIDIA/NeMo-Guardrails/issues/965)) by @Pouyanpi
- Add output rails support to activefence integration ([#940](https://github.com/NVIDIA/NeMo-Guardrails/issues/940)) by @noamlevy81
- Add Prompt Security integration ([#920](https://github.com/NVIDIA/NeMo-Guardrails/issues/920)) by @lior-ps
- Add pii masking capability to PrivateAI integration ([#901](https://github.com/NVIDIA/NeMo-Guardrails/issues/901)) by @letmerecall
- Add embedding_params to BasicEmbeddingsIndex  ([#898](https://github.com/NVIDIA/NeMo-Guardrails/issues/898)) by @Pouyanpi 
- Add score threshold to AnalyzerEngine ([#845](https://github.com/NVIDIA/NeMo-Guardrails/issues/845)) by @Pouyanpi

### 🐛 Bug Fixes

- Fix dependency resolution issues in AlignScore Dockerfile([#1002](https://github.com/NVIDIA/NeMo-Guardrails/issues/1002), [#982](https://github.com/NVIDIA/NeMo-Guardrails/issues/982)) by @Pouyanpi
- Fix JailbreakDetect docker files([#981](https://github.com/NVIDIA/NeMo-Guardrails/issues/981), [#1001](https://github.com/NVIDIA/NeMo-Guardrails/pull/1001)) by @erickgalinkin
- Fix TypeError from attempting to unpack already-unpacked dictionary ([#959](https://github.com/NVIDIA/NeMo-Guardrails/issues/959)) by @erickgalinkin
- Fix token stats usage in LLM call info ([#953](https://github.com/NVIDIA/NeMo-Guardrails/issues/953)) by @trebedea
- Handle unescaped quotes in generate_value using safe_eval ([#946](https://github.com/NVIDIA/NeMo-Guardrails/issues/946)) by @milk333445
- Handle non-relative file paths ([#897](https://github.co/NVIDIA/NeMo-Guardrails/issues/897)) by @Pouyanpi

### 📚 Documentation

- Output streaming ([#976](https://github.com/NVIDIA/NeMo-Guardrails/issues/976)) by @mikemckiernan
- Fix typos with oauthtoken ([#957](https://github.com/NVIDIA/NeMo-Guardrails/issues/957)) by @Pouyanpi
- Fix broken link in prompt security ([#978](https://github.com/NVIDIA/NeMo-Guardrails/issues/978)) by @lior-ps
- Update advanced user guides per v0.11.1 doc release ([#937](https://github.com/NVIDIA/NeMo-Guardrails/issues/937)) by @Pouyanpi

### ⚙️ Miscellaneous Tasks

- Tolerate prompt in code blocks ([#1004](https://github.com/NVIDIA/NeMo-Guardrails/issues/1004)) by @mikemckiernan
- Update YAML indent to use two spaces ([#1009](https://github.com/NVIDIA/NeMo-Guardrails/issues/1009)) by @mikemckiernan

## New Contributors

- @milk333445 made their first contribution in [#946](https://github.com/NVIDIA/NeMo-Guardrails/pull/946)
- @lior-ps made their first contribution in [#920](https://github.com/NVIDIA/NeMo-Guardrails/pull/920)

**Full Changelog**: <https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.11.1...v0.12.0>

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.12.0)

---

## v0.11.1: v0.11.1
**Published:** 2025-01-16

This release adds support for three new NVIDIA [NemoGuard NIM microservices](https://blogs.nvidia.com/blog/nemo-guardrails-nim-microservices/). These are new safeguard models, developed by NVIDIA, that help AI agents operate at scale while maintaining controlled behavior:

- [Content safety NIM microservice](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-content-safety) that safeguards AI against generating biased or harmful outputs, ensuring responses align with ethical standards.
- [Topic control NIM microservice](https://build.nvidia.com/nvidia/llama-3_1-nemoguard-8b-topic-control) that keeps conversations focused on approved topics, avoiding digression or inappropriate content.
- [Jailbreak detection NIM microservice](https://build.nvidia.com/nvidia/nemoguard-jailbreak-detect) that adds protection against jailbreak attempts, helping maintain AI integrity in adversarial scenarios.

## What's Changed

### Added

- **ContentSafety**: Add ContentSafety NIM connector ([#930](https://github.com/NVIDIA/NeMo-Guardrails/pull/930)) by @prasoonvarshney
- **TopicControl**: Add TopicControl NIM connector ([#930](https://github.com/NVIDIA/NeMo-Guardrails/pull/930)) by @makeshn
- **JailbreakDetect**: Add jailbreak detection NIM connector ([#930](https://github.com/NVIDIA/NeMo-Guardrails/pull/930)) by @erickgalinkin

### Changed

- **AutoAlign Integration**: Add further enhancements and refactoring to AutoAlign integration ([#867](https://github.com/NVIDIA/NeMo-Guardrails/pull/867)) by @KimiJL

### Fixed

- **PrivateAI Integration**: Fix Incomplete URL substring sanitization Error ([#883](https://github.com/NVIDIA/NeMo-Guardrails/pull/883)) by @NJ-186
- **Content Safety**: Fix typo in content safety prompts ([#873](https://github.com/NVIDIA/NeMo-Guardrails/pull/873)) by @krishna-sreeraj-tw

### Documentation

- **NVIDIA Blueprint**: Add Safeguarding AI Virtual Assistant NIM Blueprint NemoGuard NIMs ([#932](https://github.com/NVIDIA/NeMo-Guardrails/pull/932)) by @abodhankar

- **ActiveFence Integration**: Fix flow definition in community docs ([#890](https://github.com/NVIDIA/NeMo-Guardrails/pull/890)) by @noamlevy81

## New Contributors
* @krishna-sreeraj-tw made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/873
* @KimiJL made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/867
* @NJ-186 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/883
* @noamlevy81 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/890

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.11.0...v0.11.1

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.11.1)

---

## v0.11.0: v0.11.0
**Published:** 2024-11-19

This release introduces new features and essential bug fixes. Key highlights of this release include support for **Observability**, **Private AI**, and **Patronus Evaluate API** Integrations, along with an upgrade to **Langchain 3**. Additionally, we have dropped support for Python 3.8 as it has reached its End of Life. You can now exclude specific files from being included in your RailsConfig by using a **.railsignore** file. Once this file is placed, its rules apply to all files within its directory and all of its subdirectories.

## What's Changed

### Added

- **Observability**: Add observability support with support for different backends ([#844](https://github.com/NVIDIA/NeMo-Guardrails/pull/844)) by @Pouyanpi
- **Private AI Integration**: Add Private AI Integration ([#815](https://github.com/NVIDIA/NeMo-Guardrails/pull/815)) by @letmerecall
- **Patronus Evaluate API Integration**: Patronus Evaluate API Integration ([#834](https://github.com/NVIDIA/NeMo-Guardrails/pull/834)) by @varjoshi
- **railsignore**: Add support for .railsignore file ([#790](https://github.com/NVIDIA/NeMo-Guardrails/pull/790)) by @ajanitshimanga
- **Passthrough Support**: Add passthrough support in Colang 2 (#779) by @Pouyanpi

### Changed

- **Sandboxed Environment in Jinja2**: Add sandboxed environment in Jinja2 ([#799](https://github.com/NVIDIA/NeMo-Guardrails/pull/799)) by @Pouyanpi
- **Langchain 3 support**: Upgrade LangChain to Version 0.3 ([#784](https://github.com/NVIDIA/NeMo-Guardrails/pull/784)) by @Pouyanpi
- **Python 3.8**: Drop support for Python 3.8 ([#803](https://github.com/NVIDIA/NeMo-Guardrails/pull/803)) by @Pouyanpi
- **vllm**: Bump vllm from 0.2.7 to 0.5.5 for llama_guard and patronusai([#836](https://github.com/NVIDIA/NeMo-Guardrails/pull/836))
- **Colang 2.0-beta.5**: Upgrade Colang 2 version including new attention library besides other features and bugfixes ([Colang 2 Changelog](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/CHANGELOG-Colang.md))

### Fixed
- **Colang**: Multiline string values in interaction history prompting ([#765](https://github.com/NVIDIA/NeMo-Guardrails/pull/765)) by @radinshayanfar
- **Guardrails Library documentation**": Fix a typo in guardrails library documentation ([#793](https://github.com/NVIDIA/NeMo-Guardrails/pull/793)) by @vedantnaik19
- **Contributing Guide**: Fix incorrect folder name & pre-commit setup in CONTRIBUTING.md ([#800](https://github.com/NVIDIA/NeMo-Guardrails/pull/800)) by @SSK-14
- **Contributing Guide**: Added correct Python command version in documentation([#801](https://github.com/NVIDIA/NeMo-Guardrails/pull/801)) by @ravinder-tw
- **retrieve chunk action**: Fix presence of new line in retrieve chunk action ([#809](https://github.com/NVIDIA/NeMo-Guardrails/pull/809)) by @Pouyanpi
- **Standard Library import**: Fix guardrails standard library import path in Colang 2.0 ([#835](https://github.com/NVIDIA/NeMo-Guardrails/pull/835)) by @Pouyanpi
- **AlignScore Dockerfile**: Add nltk's punkt_tab in align_score Dockerfile ([#841](https://github.com/NVIDIA/NeMo-Guardrails/pull/841)) by @yonromai
- **Eval dependencies**: Make pandas version constraint explicit for eval optional dependency ([#847](https://github.com/NVIDIA/NeMo-Guardrails/pull/847)) by @Pouyanpi
- **tests**: Mock PromptSession to prevent console error ([#851](https://github.com/NVIDIA/NeMo-Guardrails/pull/851)) by @Pouyanpi
- **Streaming**: Handle multiple output parsers in generation ([#854](https://github.com/NVIDIA/NeMo-Guardrails/pull/854)) by @Pouyanpi
- **Colang**: Nested if else construct parsing ([#833](https://github.com/NVIDIA/NeMo-Guardrails/pull/833)) by @radinshayanfar
- **User Guide**: Update role from bot to assistant ([#852](https://github.com/NVIDIA/NeMo-Guardrails/pull/852)) by @Pouyanpi

### Documentation
- **Installation Guide**: Update optional dependencies install ([#853](https://github.com/NVIDIA/NeMo-Guardrails/pull/853)) by @Pouyanpi
- **Documentation Restructuring**: Restructure the docs and several style enhancements ([#855](https://github.com/NVIDIA/NeMo-Guardrails/pull/855)) by @Pouyanpi
- **Got It AI deprecation**: Add deprecation notice for Got It AI integration ([#857](https://github.com/NVIDIA/NeMo-Guardrails/pull/857)) by @mlmonk

## New Contributors
* @ravinder-tw made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/801
* @SSK-14 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/800
* @ajanitshimanga made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/790
* @letmerecall made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/815
* @emmanuel-ferdman made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/751
* @yonromai made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/841

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.10.1...v0.11.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.11.0)

---

## v0.10.1: v0.10.1
**Published:** 2024-10-02

## What's Changed
* Fix/colang 2/flow continuation prompt by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/775
* Remove the user intent from llm library module by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/777

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.10.0...v0.10.1

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.10.1)

---

## v0.10.0: v0.10.0
**Published:** 2024-09-27

This release introduces enhancements to elevate your AI development experience. Key highlights of this release include the introduction of a **Content Safety Module**, which aids in the integration of different content safety models, a new **Migration Tool** that facilitates migration from Colang 1.0 or Colang 2-alpha to the latest version of Colang 2 by handling a significant part of the conversion, support for the **NeMo Guardrails standard library** in Colang 2.x, added support for **Rails Exception Handling** within both Colang 1.0 and 2.x, and the addition of a **Debugging Tool** for Colang 2.x.

This update also includes numerous bug fixes and performance improvements contributed by our community.
 

## What's Changed
* bug fix by @saradiazdelser in https://github.com/NVIDIA/NeMo-Guardrails/pull/392
* Fixed broken links by @buvnswrn in https://github.com/NVIDIA/NeMo-Guardrails/pull/670
* Feature/colang 2/standard library updates by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/669
* Fix: hallucination-check  by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/679
* Add the `relevant_chunks` to the gpt-3.5 general prompt template. by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/678
* Feature/colang 2/flow event bugfix by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/672
* Feat/content safety by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/674
* feat(embeddings): add search threshold to BasicEmbeddingsIndex by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/620
* Doc/llm per task by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/676
* Check that flow names don't start with keywords by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/637
* Feature/colang 2/deactivate keyword by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/673
* Fix/nvidia ai endpoints streaming by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/654
* Fix/issue 183 non openai hallucination check by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/681
* feat(llmrails): Add serialization support for LLMRails by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/627
* New Evaluation Tooling by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/677
* fix:  streamlit import error by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/686
* Fix/ issue 589 override prompt self check facts by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/621
* Initial work on support for exceptions. by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/384
* refactor(api): Update Pydantic validators by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/688
* Feat/migration tool by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/624
* Refactor/migrate standard library by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/625
* Fix/output parser deprecation warning by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/691
* Colang 2.0 - Guardrails Library by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/689
* Fix/langchain_nvidia_ai_endpoints patch by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/697
* Revert back the abc bot Colang v1 and move the v2 version to separate config. by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/698
* Fix/colang 2 runtime issues by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/699
* fix: change 'send event' to 'send' by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/701
* Fix a small bug in the generate_flow action for Colang 2. by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/710
* Fix/output parser validation by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/704
* Feaure/colang 2/expose bot config by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/703
* Fix: pass config and kwargs to passthrough_fn runnable for issue 665 by @vpr1995 in https://github.com/NVIDIA/NeMo-Guardrails/pull/695
* [Feature] Add Cleanlab's Trustworthiness Score  by @AshishSardana in https://github.com/NVIDIA/NeMo-Guardrails/pull/572
* Feature/colang 2/llm chat interface by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/709
* Feat/add relevant chunk support to colang 2 by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/708
* Fix/rails exception migration by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/705
* fix(migration): replace hyphens and apostrophes by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/725
* Fix/llm flow continuation generation by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/724
* Fix/cli server command by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/723
* Fix/cache embeddings filesystem by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/722
* feat(library): Migrate cleanlab to colang 2 and add exception handling by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/714
* Docs/community cleanlab by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/713
* Feature/colang debug library by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/560
* Feature/extend debug cli by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/717
* Feature/colang 2/interaction loop priorties by @schuellc-nvidia in https://github.com/NVIDIA/NeMo-Guardrails/pull/712
* Fix the detection of the triggering flow id by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/728
* Add support for tags in the Eval UI. by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/731
* feat: Add support for embeddings only with search threshold by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/733
* Fix: Process all outgoing events by @sklinglernv in https://github.com/NVIDIA/NeMo-Guardrails/pull/732
* Fix the scene form and choice flows in the Colang 2 standard library by @sklinglernv in https://github.com/NVIDIA/NeMo-Guardrails/pull/741
* Feat/embedding only support colang 2 by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/737
* docs: Add note for rails exception handling in Colang 2.x by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/744
* feat: add relevant chunks prompts  by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/745
* Fix multiline LLM output syntax error for dynamic flow generation by @radinshayanfar in https://github.com/NVIDIA/NeMo-Guardrails/pull/748
* Feature/gcp moderation by @kauabh in https://github.com/NVIDIA/NeMo-Guardrails/pull/727
* Feat/migration tool sample converstaion syntax conversion by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/764
* chore: Upgrade langchain-core and jinja by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/766
* Update changelog by @Pouyanpi in https://github.com/NVIDIA/NeMo-Guardrails/pull/769

## New Contributors
* @saradiazdelser made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/392
* @buvnswrn made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/670
* @vpr1995 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/695
* @AshishSardana made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/572
* @sklinglernv made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/732
* @radinshayanfar made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/748
* @kauabh made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/727

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.9.1.1...v0.10.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.10.0)

---

## v0.9.1.1: Release v0.9.1.1
**Published:** 2024-07-26

This patch release fixes bug #651 introduced in 0.9.1. 

### Fixed

* [#650](https://github.com/NVIDIA/NeMo-Guardrails/pull/650) Fix gpt-3.5-turbo-instruct prompts #651.

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.9.1...v0.9.1.1

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.9.1.1)

---

## v0.9.1: Release v0.9.1
**Published:** 2024-07-25

This release introduces three new integrations ([Got It AI](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/community/gotitai.md), [AutoAlign](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/community/auto-align.md) and [Patronus Lynx](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/community/patronus-lynx.md)), streamlined NIM and NVIDIA API Catalog integration, support for registering custom embedding models, improvements and fixes to [Colang 2.0](https://github.com/NVIDIA/NeMo-Guardrails/blob/v0.9.1/CHANGELOG-Colang.md#20-beta2---unreleased), and many other bug fixes. This release also includes better out-of-the-box support for Llama-3 and Llama-3.1 models. 

## What's Changed

### Added

* Colang version [2.0-beta.2](https://github.com/NVIDIA/NeMo-Guardrails/blob/v0.9.1/CHANGELOG-Colang.md#20-beta2---unreleased)
* [#370](https://github.com/NVIDIA/NeMo-Guardrails/pull/370) Add Got It AI's Truthchecking service for RAG applications by @mlmonk.
* [#543](https://github.com/NVIDIA/NeMo-Guardrails/pull/543) Integrating AutoAlign's guardrail library with NeMo Guardrails by @abhijitpal1247.
* [#566](https://github.com/NVIDIA/NeMo-Guardrails/pull/566) Autoalign factcheck examples by @abhijitpal1247.
* [#518](https://github.com/NVIDIA/NeMo-Guardrails/pull/518) Docs: add example config for using models with ollama by @vedantnaik19.
* [#538](https://github.com/NVIDIA/NeMo-Guardrails/pull/538) Support for `--default-config-id` in the server.
* [#539](https://github.com/NVIDIA/NeMo-Guardrails/pull/539) Support for `LLMCallException`.
* [#548](https://github.com/NVIDIA/NeMo-Guardrails/pull/548) Support for custom embedding models.
* [#617](https://github.com/NVIDIA/NeMo-Guardrails/pull/617) NVIDIA AI Endpoints embeddings.
* [#462](https://github.com/NVIDIA/NeMo-Guardrails/pull/462) Support for calling embedding models from langchain-nvidia-ai-endpoints.
* [#622](https://github.com/NVIDIA/NeMo-Guardrails/pull/622) Patronus Lynx Integration.

### Changed

* [#597](https://github.com/NVIDIA/NeMo-Guardrails/pull/597) Make UUID generation predictable in debug-mode.
* [#603](https://github.com/NVIDIA/NeMo-Guardrails/pull/603) Improve chat cli logging.
* [#551](https://github.com/NVIDIA/NeMo-Guardrails/pull/551) Upgrade to Langchain 0.2.x by @nicoloboschi.
* [#611](https://github.com/NVIDIA/NeMo-Guardrails/pull/611) Change default templates.
* [#545](https://github.com/NVIDIA/NeMo-Guardrails/pull/545) NVIDIA API Catalog and NIM documentation update.
* [#463](https://github.com/NVIDIA/NeMo-Guardrails/pull/463) Do not store pip cache during docker build by @don-attilio.
* [#629](https://github.com/NVIDIA/NeMo-Guardrails/pull/629) Move community docs to separate folder.
* [#647](https://github.com/NVIDIA/NeMo-Guardrails/pull/647) Documentation updates.
* [#648](https://github.com/NVIDIA/NeMo-Guardrails/pull/648) Prompt improvements for Llama-3 models.

### Fixed

* [#482](https://github.com/NVIDIA/NeMo-Guardrails/pull/482) Update README.md by @curefatih.
* [#530](https://github.com/NVIDIA/NeMo-Guardrails/pull/530) Improve the test serialization test to make it more robust.
* [#570](https://github.com/NVIDIA/NeMo-Guardrails/pull/570) Add support for FacialGestureBotAction by @elisam0.
* [#550](https://github.com/NVIDIA/NeMo-Guardrails/pull/550) Fix issue #335 - make import errors visible.
* [#547](https://github.com/NVIDIA/NeMo-Guardrails/pull/547) Fix LLMParams bug and add unit tests (fixes #158).
* [#537](https://github.com/NVIDIA/NeMo-Guardrails/pull/537) Fix directory traversal bug.
* [#536](https://github.com/NVIDIA/NeMo-Guardrails/pull/536) Fix issue #304 NeMo Guardrails packaging.
* [#539](https://github.com/NVIDIA/NeMo-Guardrails/pull/539) Fix bug related to the flow abort logic in Colang 1.0 runtime.
* [#612](https://github.com/NVIDIA/NeMo-Guardrails/pull/612) Follow-up fixes for the default prompt change.
* [#585](https://github.com/NVIDIA/NeMo-Guardrails/pull/585) Fix Colang 2.0 state serialization issue.
* [#486](https://github.com/NVIDIA/NeMo-Guardrails/pull/486) Fix select model type and custom prompts task.py by @cyun9601.
* [#487](https://github.com/NVIDIA/NeMo-Guardrails/pull/487) Fix custom prompts configuration manual.md.
* [#479](https://github.com/NVIDIA/NeMo-Guardrails/pull/479) Fix static method and classmethod action decorators by @piotrm0.
* [#544](https://github.com/NVIDIA/NeMo-Guardrails/pull/544) Fix issue #216 bot utterance.
* [#616](https://github.com/NVIDIA/NeMo-Guardrails/pull/616) Various fixes.
* [#623](https://github.com/NVIDIA/NeMo-Guardrails/pull/623) Fix path traversal check.



## New Contributors
* @mlmonk made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/370
* @rgstephens made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/504
* @abhijitpal1247 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/543
* @nicoloboschi made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/551
* @elisam0 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/570
* @vedantnaik19 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/518
* @cyun9601 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/486
* @don-attilio made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/463
* @varjoshi made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/622

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.9.0...v0.9.1

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.9.1)

---

## v0.9.0: Release v0.9.0
**Published:** 2024-05-10

This release introduces [Colang 2.0](https://docs.nvidia.com/nemo/guardrails/colang_2/overview.html), the next version of Colang, and a revamped [NeMo Guardrails Documentation](https://docs.nvidia.com/nemo-guardrails). 

Colang 2.0 brings a more solid foundation for building complex guardrail configurations (with better parallelism support), advanced RAG orchestration (e.g., with multi-query, contextual relevance check), agents (e.g., driving business process logic), and multi-modal LLM-driven interaction (e.g., interactive avatars). Colang 2.0 is a complete overhaul of the Colang language and runtime, and key enhancements include:

- A more powerful flow engine supporting multiple parallel flows and advanced pattern matching over the stream of events.
- Adoption of terminology and syntax akin to Python to reduce the learning curve for new developers.
- A standard library and an import mechanism to streamline development.
- Explicit entry point through the `main` flow and explicit activation of flows.
- Smaller set of core abstractions: flows, events, and actions.
- The new _generation operator_ (`...`).
- Asynchronous action execution.

**NOTE**: The version of Colang included in v0.8.* is referred to as Colang 2.0-alpha. In v0.9.0, Colang 2.0 moved to Beta, which we refer to as Colang 2.0-beta. We expect Colang 2.0 to go out of Beta and replace Colang 1.0 as the default option in NeMo Guardrails v0.11.0.

Current limitations include not being able to use the [Guardrails Library](https://docs.nvidia.com/nemo/guardrails/user_guides/guardrails-library.html) from within Colang 2.0 and no support for generation options (e.g., logs, activated rails). These limitations will be addressed in v0.10.0 and v0.11.0, along with additional features and example guardrail configurations. 

To get started with Colang 2.0, if you’ve used Colang 1.0 before, you should check out [What’s Changed](https://docs.nvidia.com/nemo/guardrails/colang_2/whats-changed.html) page. If not, you can get started with the [Hello World](https://docs.nvidia.com/nemo/guardrails/colang_2/getting_started/hello-world.html) example.

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.8.3...v0.9.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.9.0)

---

## v0.8.3: Release v0.8.3
**Published:** 2024-04-18

This minor release updates the NVIDIA API Catalog integration documentation and fixes two bugs. 

## What's Changed

### Changed

* [#453](https://github.com/NVIDIA/NeMo-Guardrails/pull/453) Update documentation for NVIDIA API Catalog example.

### Fixed

* [#382](https://github.com/NVIDIA/NeMo-Guardrails/pull/382) Fix issue with `lowest_temperature` in self-check and hallucination rails.
* [#454](https://github.com/NVIDIA/NeMo-Guardrails/pull/454) Redo fix for #385.
* [#442](https://github.com/NVIDIA/NeMo-Guardrails/pull/442) Fix README type by @dileepbapat.

## New Contributors
* @dileepbapat made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/442

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.8.2...v0.8.3

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.8.3)

---

## v0.8.2: Release v0.8.2
**Published:** 2024-04-01

This minor release adds support for integrating NeMo Guardrails with NVIDIA AI Endpoints and Vertex AI. It also introduces the research overview page, which guides the development of future guardrails. Last but not least, it adds another round of improvements for Colang 2.0 and multiple getting-started examples. 

Colang 2.0 is the next version of Colang and will replace Colang 1.0 in a future release. It adds a more powerful flow engine, improved syntax, multi-modal support, parallelism for actions and flows, a standard library of flows, and more. This release still targets alpha testers and does not include the new documentation, which will be added in 0.9.0. Colang 2.0 and 1.0 will be supported side-by-side until Colang 1.0 is deprecated and removed. 

## What's Changed

### Added

* [#402](https://github.com/NVIDIA/NeMo-Guardrails/pull/402) Integrate Vertex AI Models into Guardrails by @aishwaryap.
* [#403](https://github.com/NVIDIA/NeMo-Guardrails/pull/403) Add support for NVIDIA AI Endpoints by @patriciapampanelli
* [#396](https://github.com/NVIDIA/NeMo-Guardrails/pull/396) Docs/examples nv ai foundation models.
* [#438](https://github.com/NVIDIA/NeMo-Guardrails/pull/438) Add research roadmap documentation.

### Changed

* [#389](https://github.com/NVIDIA/NeMo-Guardrails/pull/389) Expose the `verbose` parameter through `RunnableRails` by @d-mariano.
* [#415](https://github.com/NVIDIA/NeMo-Guardrails/pull/415) Enable `print(...)` and `log(...)`.
* [#389](https://github.com/NVIDIA/NeMo-Guardrails/pull/389) Expose verbose arg in RunnableRails by @d-mariano.
* [#414](https://github.com/NVIDIA/NeMo-Guardrails/pull/414) Feature/colang march release.
* [#416](https://github.com/NVIDIA/NeMo-Guardrails/pull/416) Refactor and improve the verbose/debug mode.
* [#418](https://github.com/NVIDIA/NeMo-Guardrails/pull/418) Feature/colang flow context sharing.
* [#425](https://github.com/NVIDIA/NeMo-Guardrails/pull/425) Feature/colang meta decorator.
* [#427](https://github.com/NVIDIA/NeMo-Guardrails/pull/427) Feature/colang single flow activation.
* [#426](https://github.com/NVIDIA/NeMo-Guardrails/pull/426) Feature/colang 2.0 tutorial.
* [#428](https://github.com/NVIDIA/NeMo-Guardrails/pull/428) Feature/Standard library and examples.
* [#431](https://github.com/NVIDIA/NeMo-Guardrails/pull/431) Feature/colang various improvements.
* [#433](https://github.com/NVIDIA/NeMo-Guardrails/pull/433) Feature/Colang 2.0 improvements: generate_async support, stateful API.

### Fixed

* [#412](https://github.com/NVIDIA/NeMo-Guardrails/pull/412) Fix #411 - explain rails not working for chat models.
* [#413](https://github.com/NVIDIA/NeMo-Guardrails/pull/413) Typo fix: Comment in llm_flows.co by @habanoz.
* [#420](https://github.com/NVIDIA/NeMo-Guardrails/pull/430) Fix typo for hallucination message.

## New Contributors
* @aishwaryap made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/402
* @patriciapampanelli made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/403
* @habanoz made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/413
* @d-mariano made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/389

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.8.1...v0.9.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.8.2)

---

## v0.8.1: Release v0.8.1
**Published:** 2024-03-15

This minor release mainly focuses on fixing Colang 2.0 parser and runtime issues. It fixes a bug related to logging the prompt for chat models in verbose mode and a small issue in the installation guide. It also adds an example of using streaming with a custom action. 

## What's Changed

### Added

- [#377](https://github.com/NVIDIA/NeMo-Guardrails/pull/377) Add example for streaming from custom action.

### Changed

- [#380](https://github.com/NVIDIA/NeMo-Guardrails/pull/380) Update installation guide for OpenAI usage.
- [#401](https://github.com/NVIDIA/NeMo-Guardrails/pull/401) Replace YAML import with new import statement in multi-modal example.

### Fixed

- [#398](https://github.com/NVIDIA/NeMo-Guardrails/pull/398) Colang parser fixes and improvements.
- [#394](https://github.com/NVIDIA/NeMo-Guardrails/pull/394) Fixes and improvements for Colang 2.0 runtime.
- [#381](https://github.com/NVIDIA/NeMo-Guardrails/pull/381) Fix typo by @serhatgktp.
- [#379](https://github.com/NVIDIA/NeMo-Guardrails/pull/379) Fix missing prompt in verbose mode for chat models.
- [#400](https://github.com/NVIDIA/NeMo-Guardrails/pull/400) Fix Authorization header showing up in logs for NeMo LLM.

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.8.0...v0.8.1

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.8.1)

---

## v0.8.0: Release v0.8.0
**Published:** 2024-02-28

This release adds three main new features:

1. A new type of input rail that uses a set of [jailbreak heuristics](./docs/user_guides/guardrails-library.md#jailbreak-detection-heuristics). More heuristics will be added in the future. 
2. Support for [generation options](./docs/user_guides/advanced/generation-options.md) allowing fine-grained control on what types of rails should be triggered, what data should be returned and what logging information should be included in the response. 
3. Support for making API calls to the guardrails server using [multiple configuration ids](./docs/user_guides/multi_config_api). 

This release also improves the support for working with embeddings (better async support, batching and caching), adds support for stop tokens per task template, and adds streaming support for HuggingFace pipelines. Last but not least, this release includes the core implementation for Colang 2.0 as a preview for early testing (version `0.9.0` will include documentation and examples). 

## What's Changed

### Added

- [#292](https://github.com/NVIDIA/NeMo-Guardrails/pull/292) [Jailbreak heuristics](./docs/user_guides/guardrails-library.md#jailbreak-detection-heuristics) by @erickgalinkin.
- [#256](https://github.com/NVIDIA/NeMo-Guardrails/pull/256) Support [generation options](./docs/user_guides/advanced/generation-options.md).
- [#307](https://github.com/NVIDIA/NeMo-Guardrails/pull/307) Added support for multi-config api calls by @makeshn.
- [#293](https://github.com/NVIDIA/NeMo-Guardrails/pull/293) Adds configurable stop tokens by @zmackie.
- [#334](https://github.com/NVIDIA/NeMo-Guardrails/pull/334) Colang 2.0 - Preview by @schuellc.
- [#208](https://github.com/NVIDIA/NeMo-Guardrails/pull/208) Implement cache embeddings (resolves #200) by @Pouyanpi.
- [#331](https://github.com/NVIDIA/NeMo-Guardrails/pull/331) Huggingface pipeline streaming by @trebedea.

Documentation:
- [#311](https://github.com/NVIDIA/NeMo-Guardrails/pull/311) Update documentation to demonstrate the use of output rails when using a custom RAG by @niels-garve.
- [#347](https://github.com/NVIDIA/NeMo-Guardrails/pull/347) Add [detailed logging docs](./docs/user_guides/detailed_logging) by @erickgalinkin.
- [#354](https://github.com/NVIDIA/NeMo-Guardrails/pull/354) [Input and output rails only guide](./docs/user_guides/input_output_rails_only) by @trebedea.
- [#359](https://github.com/NVIDIA/NeMo-Guardrails/pull/359) Added [user guide for jailbreak detection heuristics](./docs/user_guides/jailbreak_detection_heuristics) by @makeshn.
- [#363](https://github.com/NVIDIA/NeMo-Guardrails/pull/363) Add [multi-config API call user guide](./docs/user_guides/multi_config_api).
- [#297](https://github.com/NVIDIA/NeMo-Guardrails/pull/297) Example configurations for using only the guardrails, without LLM generation.

### Changed

- [#309](https://github.com/NVIDIA/NeMo-Guardrails/pull/309) Change the paper citation from ArXiV to EMNLP 2023 by @manuelciosici
- [#319](https://github.com/NVIDIA/NeMo-Guardrails/pull/319) Enable embeddings model caching.
- [#267](https://github.com/NVIDIA/NeMo-Guardrails/pull/267) Make embeddings computing async and add support for batching.
- [#281](https://github.com/NVIDIA/NeMo-Guardrails/pull/281) Follow symlinks when building knowledge base by @piotrm0.
- [#280](https://github.com/NVIDIA/NeMo-Guardrails/pull/280) Add more information to results of `retrieve_relevant_chunks` by @piotrm0.
- [#332](https://github.com/NVIDIA/NeMo-Guardrails/pull/332) Update docs for batch embedding computations.
- [#244](https://github.com/NVIDIA/NeMo-Guardrails/pull/244) Docs/edit getting started by @DougAtNvidia.
- [#333](https://github.com/NVIDIA/NeMo-Guardrails/pull/333) Follow-up to PR 244.
- [#341](https://github.com/NVIDIA/NeMo-Guardrails/pull/341) Updated 'fastembed' version to 0.2.2 by @NirantK.

### Fixed

- [#286](https://github.com/NVIDIA/NeMo-Guardrails/pull/286) Fixed #285 - using the same evaluation set given a random seed for topical rails by @trebedea.
- [#336](https://github.com/NVIDIA/NeMo-Guardrails/pull/336) Fix #320. Reuse the asyncio loop between sync calls.
- [#337](https://github.com/NVIDIA/NeMo-Guardrails/pull/337) Fix stats gathering in a parallel async setup.
- [#342](https://github.com/NVIDIA/NeMo-Guardrails/pull/342) Fixes OpenAI embeddings support.
- [#346](https://github.com/NVIDIA/NeMo-Guardrails/pull/346) Fix issues with KB embeddings cache, bot intent detection and config ids validator logic.
- [#349](https://github.com/NVIDIA/NeMo-Guardrails/pull/349) Fix multi-config bug, asyncio loop issue and cache folder for embeddings.
- [#350](https://github.com/NVIDIA/NeMo-Guardrails/pull/350) Fix the incorrect logging of an extra dialog rail.
- [#358](https://github.com/NVIDIA/NeMo-Guardrails/pull/358) Fix Openai embeddings async support.
- [#362](https://github.com/NVIDIA/NeMo-Guardrails/pull/362) Fix the issue with the server being pointed to a folder with a single config.
- [#352](https://github.com/NVIDIA/NeMo-Guardrails/pull/352) Fix a few issues related to jailbreak detection heuristics.
- [#356](https://github.com/NVIDIA/NeMo-Guardrails/pull/356) Redo followlinks PR in new code by @piotrm0.

## New Contributors
* @manuelciosici made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/309
* @erickgalinkin made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/292
* @trebedea made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/286
* @piotrm0 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/281
* @Pouyanpi made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/208
* @niels-garve made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/311
* @zmackie made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/293
* @DougAtNvidia made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/244
* @NirantK made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/341
* @makeshn made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/359

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.7.1...v0.8.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.8.0)

---

## v0.7.1: Release v0.7.1
**Published:** 2024-02-01

## What's Changed
* Replace SentenceTransformers with FastEmbed by @drazvan in https://github.com/NVIDIA/NeMo-Guardrails/pull/288


**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.7.0...v0.7.1

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.7.1)

---

## v0.7.0: Release v0.7.0
**Published:** 2024-01-31

This release adds three new features: support for Llama Guard, improved LangChain integration, and support for server-side threads. It also adds support for Python 3.11 and solves the issue with pinned dependencies (e.g., `langchain>=0.1.0,<2.0`, `typer>=0.7.0`). Last but not least, it includes multiple feature and security-related fixes.

## What's Changed

### Added

- [#254](https://github.com/NVIDIA/NeMo-Guardrails/pull/254) Support for [Llama Guard input and output content moderation](./docs/user_guides/guardrails-library.md#llama-guard-based-content-moderation).
- [#253](https://github.com/NVIDIA/NeMo-Guardrails/pull/253) Support for [server-side threads](./docs/user_guides/server-guide.md#threads).
- [#235](https://github.com/NVIDIA/NeMo-Guardrails/pull/235) Improved [LangChain integration](docs/user_guides/langchain/langchain-integration.md) through `RunnableRails`.
- [#190](https://github.com/NVIDIA/NeMo-Guardrails/pull/190) Add [example](./examples/notebooks/generate_events_and_streaming.ipynb) for using `generate_events_async` with streaming.
- Support for Python 3.11.

### Changed

- [#240](https://github.com/NVIDIA/NeMo-Guardrails/pull/240) Switch to pyproject.
- [#276](https://github.com/NVIDIA/NeMo-Guardrails/pull/276) Upgraded Typer to 0.9.

### Fixed

- [#239](https://github.com/NVIDIA/NeMo-Guardrails/pull/239) Fixed logging issue where `verbose=true` flag did not trigger expected log output.
- [#228](https://github.com/NVIDIA/NeMo-Guardrails/pull/228) Fix docstrings for various functions.
- [#242](https://github.com/NVIDIA/NeMo-Guardrails/pull/242) Fix Azure LLM support.
- [#225](https://github.com/NVIDIA/NeMo-Guardrails/pull/225) Fix annoy import, to allow using without.
- [#209](https://github.com/NVIDIA/NeMo-Guardrails/pull/209) Fix user messages missing from prompt.
- [#261](https://github.com/NVIDIA/NeMo-Guardrails/pull/261) Fix small bug in `print_llm_calls_summary`.
- [#252](https://github.com/NVIDIA/NeMo-Guardrails/pull/252) Fixed duplicate loading for the default config.
- Fixed the dependencies pinning, allowing a wider range of dependencies versions.
- Fixed sever security issues related to uncontrolled data used in path expression and information exposure through an exception.

## New Contributors

* @spehl-max made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/239
* @rajveer43 made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/228
* @smartestrobotdai made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/242
* @prasoonvarshney made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/269
* @eneadodi made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/276
* @baggiponte made their first contribution in https://github.com/NVIDIA/NeMo-Guardrails/pull/240

**Full Changelog**: https://github.com/NVIDIA/NeMo-Guardrails/compare/v0.6.1...v0.7.0

[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.7.0)

---

## v0.6.1: Release v0.6.1
**Published:** 2023-12-20

This patch release upgrades two dependencies (`langchain` and `httpx`) and replaces the deprecated `text-davinci-003` model with `gpt-3.5-turbo-instruct` in all configurations and examples.

---

### Added

- Support for `--version` flag in the CLI.

### Changed

- Upgraded `langchain` to `0.0.352`.
- Upgraded `httpx` to `0.24.1`.
- Replaced deprecated `text-davinci-003` model with `gpt-3.5-turbo-instruct`.

### Fixed

- [#191](https://github.com/NVIDIA/NeMo-Guardrails/pull/191): Fix chat generation chunk issue.





[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.6.1)

---

## v0.6.0: Release v0.6.0
**Published:** 2023-12-13

This release builds on the feedback received over the last few months and brings many improvements and new features. It is also the first **beta** release for NeMo Guardrails. Equally important, this release is the first to include LLM vulnerability scan results for one of the sample bots. 

Release highlights include: 

- Better configuration and support for input, output, dialog, retrieval, and execution rails.
- Ability to reduce the overall latency using `single_call` mode or `embeddings_only` mode for dialog rails.
- Support for streaming.
- First version of the Guardrails Library.
- Fast fact-checking using AlignScore.
- Updated Getting Started guide.
- Docker image for easy deployment.

Detailed changes are included below. 

---

### Added

- Support for [explicit definition](./docs/user_guides/configuration-guide.md#guardrails-definitions) of input/output/retrieval rails.
- Support for [custom tasks and their prompts](docs/user_guides/advanced/prompt-customization.md#custom-tasks-and-prompts).
- Support for fact-checking [using AlignScore](./docs/user_guides/guardrails-library.md#alignscore).
- Support for [NeMo LLM Service](./docs/user_guides/configuration-guide.md#nemo-llm-service) as an LLM provider.
- Support for making a single LLM call for both the guardrails process and generating the response (by setting `rails.dialog.single_call.enabled` to `True`).
- Support for [sensitive data detection](./docs/user_guides/guardrails-library.md#sensitive-data-detection) guardrails using Presidio.
- [Example](./examples/configs/llm/hf_pipeline_llama2) using NeMo Guardrails with the LLaMa2-13B model.
- [Dockerfile](./Dockerfile) for building a Docker image.
- Support for [prompting modes](./docs/user_guides/advanced/prompt-customization.md) using `prompting_mode`.
- Support for [TRT-LLM](./docs/user_guides/configuration-guide.md#trt-llm) as an LLM provider.
- Support for [streaming](./docs/user_guides/advanced/streaming.md) the LLM responses when no output rails are used.
- [Integration](./docs/user_guides/guardrails-library.md#active-fence) of ActiveFence ActiveScore API as an input rail.
- Support for `--prefix` and `--auto-reload` in the [guardrails server](./docs/user_guides/server-guide.md).
- Support for loading a configuration from dictionary, i.e. `RailsConfig.from_content(config=...)`.
- Guidance on [LLM support](./docs/user_guides/llm-support.md).
- Support for `LLMRails.explain()` (see the [Getting Started](./docs/getting_started) guide for sample usage).

### Changed

- Allow context data directly in the `/v1/chat/completion` using messages with the type `"role"`.
- Allow calling a subflow whose name is in a variable, e.g. `do $some_name`.
- Allow using actions which are not `async` functions.
- Disabled pretty exceptions in CLI.
- Upgraded dependencies.
- Updated the [Getting Started Guide](./docs/getting_started).
- Main [README](./README.md) now provides more details.
- Merged original examples into a single [ABC Bot](./examples/bots/abc) and removed the original ones.
- Documentation improvements.

### Fixed

- Fix going over the maximum prompt length using the `max_length` attribute in [Prompt Templates](./docs/user_guides/advanced/prompt-customization.md#prompt-templates).
- Fixed problem with `nest_asyncio` initialization.
- [#144](https://github.com/NVIDIA/NeMo-Guardrails/pull/144) Fixed TypeError in logging call.
- [#121](https://github.com/NVIDIA/NeMo-Guardrails/pull/109) Detect chat model using openai engine.
- [#109](https://github.com/NVIDIA/NeMo-Guardrails/pull/109) Fixed minor logging issue.
- Parallel flow support.
- Fix `HuggingFacePipeline` bug related to LangChain version upgrade.


[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.6.0)

---

## v0.5.0: Release v0.5.0
**Published:** 2023-09-04
**Pre-release**

This release adds support for custom embedding search providers (not using Annoy/SentenceTransformers) and support for OpenAI embeddings for the default embedding search provider. This release adds an advanced example for using multiple knowledge bases (i.e., a tabular and regular one). This release also fixes an old issue related to using the `generate` method inside an async environment (e.g., a notebook) and includes multiple small fixes. Detailed change log below.

---

### Added

- Support for [custom configuration data](./docs/user_guide/configuration-guide.md#custom-data).
- Example for using [custom LLM and multiple KBs](./examples/multi_kb/README.md)
- Support for [`PROMPTS_DIR`](./docs/user_guide/advanced/prompt-customization.md#prompt-configuration).
- [#101](https://github.com/NVIDIA/NeMo-Guardrails/pull/101) Support for [using OpenAI embeddings](./docs/user_guide/configuration-guide.md#the-embeddings-model) models in addition to SentenceTransformers.
- First set of end-to-end QA tests for the example configurations.
- Support for configurable [embedding search providers](./docs/user_guide/advanced/embedding-search-providers.md)

### Changed

- Moved to using `nest_asyncio` for [implementing the blocking API](./docs/user_guide/advanced/nested-async-loop.md). Fixes [#3](https://github.com/NVIDIA/NeMo-Guardrails/issues/3) and [#32](https://github.com/NVIDIA/NeMo-Guardrails/issues/32).
- Improved event property validation in `new_event_dict`.
- Refactored imports to allow installing from source without Annoy/SentenceTransformers (would need a custom embedding search provider to work).

### Fixed

- Fixed when the `init` function from `config.py` is called to allow custom LLM providers to be registered inside.
- [#93](https://github.com/NVIDIA/NeMo-Guardrails/pull/93): Removed redundant `hasattr` check in `nemoguardrails/llm/params.py`.
- [#91](https://github.com/NVIDIA/NeMo-Guardrails/issues/91): Fixed how default context variables are initialized.


[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.5.0)

---

## v0.4.0: Release v0.4.0
**Published:** 2023-08-03
**Pre-release**

This release focused on multiple areas: 

1. Extending the guardrails interface to support generic events.
2. Adding experimental support for running a red teaming process. 
3. Adding experimental support for `vicuna-7b-v1.3` and `mpt-7b-instruct`.
4. Extending Colang 1.0 with support for bot message instructions and using variables inside bot message definitions.
5. Fixing several bugs reported by the community.

Detailed change log below.

---

## Added

- [Event-based API](./docs/user_guide/advanced/event-based-api.md) for guardrails.
- Support for message with type "event" in [`LLMRails.generate_async`](./docs/api/nemoguardrails.rails.llm.llmrails.md#method-llmrailsgenerate_async).
- Support for [bot message instructions](docs/user_guide/advanced/bot-message-instructions.md).
- Support for [using variables inside bot message definitions](./docs/user_guide/colang-language-syntax-guide.md#bot-messages-with-variables).
- Support for `vicuna-7b-v1.3` and `mpt-7b-instruct`.
- Topical evaluation results for `vicuna-7b-v1.3` and `mpt-7b-instruct`.
- Support to use different models for different LLM tasks.
- Support for [red-teaming](docs/user_guide/advanced/red-teaming.md) using challenges.
- Support to disable the Chat UI when running the server using `--disable-chat-ui`.
- Support for accessing the API request headers in server mode.
- Support to [enable CORS settings](docs/user_guide/server-guide.md#cors) for the guardrails server.

## Changed

- Changed the naming of the internal events to align to the upcoming UMIM spec (Unified Multimodal Interaction Management).
- If there are no user message examples, the bot messages examples lookup is disabled as well.

## Fixed

- [#58](https://github.com/NVIDIA/NeMo-Guardrails/issues/58): Fix install on Mac OS 13.
- [#55](https://github.com/NVIDIA/NeMo-Guardrails/issues/55): Fix bug in example causing config.py to crash on computers with no CUDA-enabled GPUs.
- Fixed the model name initialization for LLMs that use the `model` kwarg.
- Fixed the Cohere prompt templates.
- [#55](https://github.com/NVIDIA/NeMo-Guardrails/issues/83): Fix bug related to LangChain callbacks initialization.
- Fixed generation of "..." on value generation.
- Fixed the parameters type conversion when invoking actions from colang (previously everything was string).
- Fixed `model_kwargs` property for the `WrapperLLM`.
- Fixed bug when `stop` was used inside flows.
- Fixed Chat UI bug when an invalid guardrails configuration was used.


[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.4.0)

---

## v0.3.0: Release v0.3.0.
**Published:** 2023-06-30
**Pre-release**

This release focuses on enhancing the support to integrate additional LLMs with NeMo Guardrails. It adds the ability to customize the prompt for various LLMs, including support for completion and chat models. This release adds examples for using the HuggingFace pipeline and inference endpoints. Last but not least, this release provides an initial evaluation of the core prompting technique and some of the rails. 

### Added

- Support for defining [subflows](docs/user_guide/colang-language-syntax-guide.md#subflows).
- Improved support for [customizing LLM prompts](docs/user_guide/advanced/prompt-customization.md)
  - Support for using filters to change how variables are included in a prompt template.
  - Output parsers for prompt templates.
  - The `verbose_v1` formatter and output parser to be used for smaller models that don't understand Colang very well in a few-shot manner.
  - Support for including context variables in prompt templates.
  - Support for chat models i.e. prompting with a sequence of messages.
- Experimental support for allowing the LLM to generate [multi-step flows](docs/user_guide/configuration-guide.md#multi-step-generation).
- Example of using Llama Index from a guardrails configuration (#40).
- [Example](examples/llm/hf_endpoint) for using HuggingFace Endpoint LLMs with a guardrails configuration.
- [Example](examples/llm/hf_pipeline_dolly) for using HuggingFace Pipeline LLMs with a guardrails configuration.
- Support to alter LLM parameters passed as `model_kwargs` in LangChain.
- CLI tool for running evaluations on the different steps (e.g., canonical form generation, next steps, bot message) and on existing rails implementation (e.g., moderation, jailbreak, fact-checking, and hallucination).
- [Initial evaluation](nemoguardrails/eval/README.md) results for `text-davinci-003` and `gpt-3.5-turbo`.
- The `lowest_temperature` can be set through the guardrails config (to be used for deterministic tasks).

### Changed

- The core templates now use Jinja2 as the rendering engines.
- Improved the internal prompting architecture, now using an LLM Task Manager.

### Fixed

- Fixed bug related to invoking a chain with multiple output keys.
- Fixed bug related to tracking the output stats.
- #51: Bug fix - avoid str concat with None when logging user_intent.
- #54: Fix UTF-8 encoding issue and add embedding model configuration.


[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.3.0)

---

## v0.2.0: Release v0.2.0
**Published:** 2023-06-01
**Pre-release**



[View on GitHub](https://github.com/NVIDIA-NeMo/Guardrails/releases/tag/v0.2.0)

---

