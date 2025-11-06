# Releases

Version history for this repository (115 releases).

## ray-2.51.1: Release-2.51.1
**Published:** 2025-11-01

- Reuse previous metadata if transferring the same tensor list with `nixl` (https://github.com/ray-project/ray/pull/58309)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.51.1)

---

## ray-2.51.0: Ray-2.51.0
**Published:** 2025-10-29

# Release Highlights

**Ray Train:**
- Ray Train v2 is now enabled by default! Ray Train v2 provides usability and stability improvements, as well as new features. For more details, see the [REP](https://github.com/ray-project/enhancements/blob/main/reps/2024-10-18-train-tune-api-revamp/2024-10-18-train-tune-api-revamp.md)  and [Migration Guide](https://github.com/ray-project/ray/issues/49454). To disable Ray Train v2, set the environment variable `RAY_TRAIN_V2_ENABLED=0`.


**Ray Serve:**

- Application-level autoscaling: Introduces custom autoscaling policies that operate across all deployments in an application, enabling coordinated scaling decisions based on aggregate metrics. This is a significant advancement over per-deployment autoscaling, allowing for more intelligent resource management at the application level.
- Enhanced autoscaling capabilities with replica-level metrics: Wires up `AutoscalingContext` with `total_running_requests`, `total_queued_requests`, and `total_num_requests`, plus adds support for min, max, and time-weighted average aggregation functions. These improvements give users fine-grained control to implement sophisticated custom autoscaling policies based on real-time workload metrics.

# Ray Libraries
## Ray Data
🎉 New Features:

- Added enhanced support for Unity Catalog integration (#57954, #58049)
- New expression evaluator infrastructure for improved query optimization (#57778, #57855)
- Support for SaveMode in write operations (#57946)
- Added approximate quantile aggregator (#57598)
- MCAP datasource support for robotics data (#55716)
- Callback-based stat computation for preprocessors and ValueCounter (#56848)
- Support for multiple download URIs with improved error handling (#57775)

💫 Enhancements:

- Improved projection pushdown handling with renamed columns (#58033, #58037, #58040, #58071)
- Enhanced hash-shuffle performance with better retry policies (#57572)
- Streamlined concurrency parameter semantics (#57035)
- Improved execution progress rendering (#56992)
- Better handling of empty columns in pandas blocks (#57740)
- Enhanced support for complex data types and column operations (#57271)
- Reduced memory usage with improved streaming generator backpressure (#57688)
- Enhanced preemption testing and utilities (#57883)
- Improved Download operator display names (#57773)
- Better handling of variable-shaped tensors and tensor columns (#57240)
- Optimized aggregator execution with out-of-order processing by default (#57753)

🔨 Fixes:

- Fixed renamed columns to be appropriately dropped from output (#58040, #58071)
- Fixed handling of renames in projection pushdown (#58033, #58037)
- Fixed vLLMEngineStage field name inconsistency for images (#57980)
- Fixed driver hang during streaming generator block metadata retrieval (#56451)
- Fixed retry policy for hash-shuffle tasks (#57572)
- Fixed prefetch loop to avoid blocking on fetches (#57613)
- Fixed empty projection handling (#57740)
- Fixed errors with concatenation of mixed pyarrow native and extension types (#56811)

📖 Documentation:

- Updated document embedding benchmark to use canonical Ray Data API (#57977)
- Improved concurrency-related documentation (#57658)
- Updated preprocessing and data handling examples

## Ray Train

🎉 New features
- Turn on Train v2 by default (#57857)
- Top-level `ray.train` aliases for public APIs (#57758)

💫 Enhancements
- Raise clear errors when mixing v1/v2 APIs (#57570)
- JAX backend: add `jax.distributed.shutdown()` for `JaxBackend` (#57802)
- Update `TrainingFailedError` module (#57865)
- Improve deprecation handling when `ray.train` methods are called from `ray.tune` (#57810)
- Enable deprecation warnings for legacy XGBoost/LightGBM trainers (#57280)

🔨 Fixes
- Fix `ControllerError` triggered by `after_worker_group_poll_status` errors (#57869)
- Fix `iter_torch_batches` use of `ray.train.torch.get_device` outside Train (#57816)
- Fix exception-queue race condition in `ThreadRunner` (#57249)

📖 Documentation
- Add validation and details to checkpoint docs (#57065)

🏗 Architecture / tests
- Enable Train v2 across test suites; migrate remaining tests and isolate/disable stragglers (#56868, #57256, #57534, #57722, #57764)
- Isolate circular-dependency tests and resolve circular imports (#57710, #56921)
- Replace Checkpoint Manager Pydantic v2 APIs with v1 (#57147)
- Bump test timeouts (`test_util`, `torch_trainer`) (#57939, #57873)

## Ray Tune
💫 Enhancements:

- Updated release tests to import from tune (#57956)
- Better integration with Train V2 backend

## Ray Serve
🎉 New Features:

- Application-level autoscaling. Introduces support for custom autoscaling policies that operate across all deployments in an application, enabling coordinated scaling decisions based on aggregate metrics. ([#57535](https://github.com/ray-project/ray/pull/57535), [#57548](https://github.com/ray-project/ray/pull/57548), [#57637](https://github.com/ray-project/ray/pull/57637), [#57756](https://github.com/ray-project/ray/pull/57756))
- Autoscaling metrics aggregation functions. Adds support for min, max, and time-weighted average aggregation over timeseries data, providing more flexible autoscaling control. ([#56871](https://github.com/ray-project/ray/pull/56871))
- Enhanced autoscaling context with replica-level metrics. Wires up AutoscalingContext constructor arguments to expose total_running_requests, total_queued_requests, and total_num_requests for use in custom autoscaling policies. ([#57202](https://github.com/ray-project/ray/pull/57202))
- Multiple task consumers in a single application. Ray Serve applications can now run multiple task consumer deployments concurrently. ([#56618](https://github.com/ray-project/ray/pull/56618))

💫 Enhancements:

- Reconfigure invoked on replica rank changes. The reconfigure method now receives both user_config and rank parameters when ranks change, enabling replicas to adapt their configuration dynamically. ([#57091](https://github.com/ray-project/ray/pull/57091))
- Celery adapter configuration improvements. Added default serializer and new configuration fields to enhance Celery integration flexibility. ([#56707](https://github.com/ray-project/ray/pull/56707))
- AutoscalingContext promoted to public API. The autoscaling context is now officially part of the public API with comprehensive documentation. ([#57600](https://github.com/ray-project/ray/pull/57600))
- Async inference telemetry. Added telemetry tracking to monitor the number of replicas using asynchronous inference. ([#57665](https://github.com/ray-project/ray/pull/57665))
- Rank logging verbosity reduced. Changed seven rank-related INFO logs to DEBUG level, reducing log noise during normal operations. ([#57831](https://github.com/ray-project/ray/pull/57831))
- Controller logging optimized. Removed expensive debug logs from the controller that were costly in large clusters. ([#57813](https://github.com/ray-project/ray/pull/57813))

🔨 Fixes:

- Max constructor retry count test fixed for Windows. Adjusted test resource requirements to account for Windows process creation overhead compared to Linux forking. ([#57541](https://github.com/ray-project/ray/pull/57541))
- Streaming test stability improvements. Added synchronization mechanisms to prevent chunk coalescing and rechunking, eliminating test flakiness. ([#57592](https://github.com/ray-project/ray/pull/57592), [#57728](https://github.com/ray-project/ray/pull/57728))
- Autoscaling test deflaking. Fixed race conditions in application-level autoscaling tests and removed flaky min aggregation test scenario. ([#57784](https://github.com/ray-project/ray/pull/57784), [#57967](https://github.com/ray-project/ray/pull/57967))
- State API usage test corrected. Fixed a unit test that was broken but not running in CI. ([#56948](https://github.com/ray-project/ray/pull/56948))
- Controller recovery logging condition fixed. Updated test condition to properly verify debug and JSON logs after controller recovery. ([#57568](https://github.com/ray-project/ray/pull/57568))

📖 Documentation:

- Custom autoscaling documentation. Added comprehensive guide for implementing custom autoscaling policies with examples and best practices. ([#57600](https://github.com/ray-project/ray/pull/57600))
- Replica ranks documentation. Documented the replica rank feature, including how ranks are assigned and how to use them in reconfigure methods. ([#57649](https://github.com/ray-project/ray/pull/57649))
- Application-level autoscaling guide. Added documentation explaining how to configure and use application-level autoscaling policies. ([#57756](https://github.com/ray-project/ray/pull/57756))
- Autoscaling documentation improvements. Updated serve autoscaling docs with clearer explanations and examples. ([#57652](https://github.com/ray-project/ray/pull/57652))
- Performance flags documentation. Documented performance-related configuration flags for Ray Serve. ([#57845](https://github.com/ray-project/ray/pull/57845))
- Metrics documentation fix. Corrected ray_serve_deployment_queued_queries metric name discrepancy in documentation. ([#57629](https://github.com/ray-project/ray/pull/57629))
- AutoscalingContext import added to examples. Fixed missing import statement in custom autoscaling policy example. ([#57876](https://github.com/ray-project/ray/pull/57876))
- App builder guide typo corrected. Fixed command syntax error in typed application builder example. ([#57634](https://github.com/ray-project/ray/pull/57634))
- Celery filesystem broker note. Added warning about using filesystem as a broker in Celery workers. ([#57686](https://github.com/ray-project/ray/pull/57686))
- Async inference alpha stage warning. Added notice that async inference is in alpha stage. ([#57268](https://github.com/ray-project/ray/pull/57268))

🏗 Architecture refactoring:

- Autoscaling control moved to application state. Migrated autoscaling control loop from deployment state to application state, preparing for application-level autoscaling. ([#57548](https://github.com/ray-project/ray/pull/57548))
- Async capability enum removed. Cleaned up unused async capability enum from codebase. ([#57666](https://github.com/ray-project/ray/pull/57666))


## Ray Serve/Data LLM
🎉 New Features:

- Updated vLLM to 0.11.0 and Nixl to 0.6.0 (#57201)
- Video processor support for multimodal pipelines (#56785)
- Enhanced callback API for engine customization (#57257)
- Unified and extended builder configuration for LLM deployments (#57724)

💫 Enhancements:

- Protocol-based typing improvements and cleaner inheritance structure (#57743)
- Better engine metrics enabled by default (#57615)
- Simplified NIXL dependency management in ray-llm images (#57706)
- Per-stage map kwargs for LLM processor preprocessing/postprocessing (#57826)
- Improved architecture documentation (#57830)
- Better code structure alignment with architectural design (#57889)
- Enhanced multimodal support with Deepseek compatibility (#56906)

🔨 Fixes:

- Fixed NIXL limitations with proper exception handling (#58159)
- Improved runai_streamer for vLLM 0.10.2+ integration (#56906)

📖 Documentation:

- Added comprehensive architecture documentation for Ray Serve LLM (#57830)
- Reorganized LLM documentation with improved navigation (#57787)
- Added benchmark page for performance reference (#57960)
- Converted quick-start guide to MyST Markdown (#57782)
- Better organization of Ray Serve LLM documentation (#57181?)

## RLlib
🎉 New Features:

- Prometheus metrics support for selected RLlib components (#57932)
- Enhanced support for complex observations in SingleAgentEpisode (#57017)

💫 Enhancements:

- LINT improvements with enabled ruff imports for rllib/utils (#56737)
- Better type hints for learner_connector (#57673)
- Improved throughput metrics to avoid biasing (#57215)

🔨 Fixes:

- Fixed segment_tree.py edge case (#57599)
- Fixed small bug in type hints (#57673)

## Ray Core
🎉 New Features:

- Enhanced Ray Direct Transport (RDT) with improved NIXL integration and garbage collection (#57671, #57603, #58159)
- Cgroups support improvements with better system resource management (#57776, #57864, #57731, #58017, #58028, #58064)
- Fault-tolerant RPC improvements for better distributed reliability (#57786, #57861)
- Exponential backoff for retryable gRPCs (#56568)

💫 Enhancements:

- Migrated from STATS to metric interface in RPC components (#57926)
- Improved histogram metrics midpoint calculation (#57948)
- Made FreeObjects non-fatal for better error handling (#57550)
- Enhanced ReleaseUnusedBundles fault tolerance (#57786)
- Made DrainRaylet and ShutdownRaylet fault tolerant (#57861)
- Better error handling for metric and event exporter agent (#57925)
- Improved raylet shutdown process and file organization (#57817)
- Reporter agent can now get PID via RPC to raylet (#57004)
- Enhanced ray.get thread safety (#57911)
- Configurable proto naming during event JSON conversion (#57705)
- Better handling of detached actor restarts (#57931)
- Improved lease rescheduling in local lease manager during node draining (#57834)

🔨 Fixes:

- Fixed "RayEventRecorder::StartExportingEvents() should be called only once" error (#57917)
- Fixed deadlock when cancelling stale requests on in-order actors (#57746)
- Fixed raylet shutdown races (#57198)
- Fixed log monitor seeking bug after log rotation (#56902)
- Deflaked multiple test suites for better CI reliability
- Fixed various memory and resource management issues
- Better handling of actor and task failures

📖 Documentation:

- Added JaxTrainer API overview to Ray docs (#57182)
- Fixed various typos and documentation issues
- Updated autoscaling and system configuration guides
- Enhanced SLURM documentation with symmetric-run support (#56775)

🏗 Architecture refactoring:

- Dashboard API server subprocesses moved into system cgroup (#57864)
- Driver moved into workers cgroup for better isolation (#57776)
- Improved worker-raylet interface separation (#57804)
- Better plasma store provider architecture

## Dashboard
💫 Enhancements:

- Added percentage usage graphs for resources (#57549)
- Introduced sub-tabs with full Grafana dashboard embeds on Metrics tab (#57561)
- Added queued blocks to operator panels (#57739)
- Improved operator metrics logging for better clarity (#57702)
- Better filtering and display in job lists

🔨 Fixes:

- Fixed filtering issue in job list (#56946)
- Fixed incomplete card content on overview page (#56947)
- Filtered out ANSI escape codes from logs (#53370)

## Autoscaler
🎉 New Features:

- KubeRay autoscaling support with top-level Resources and Labels fields (#57260)
- Bundle label selector support in request_resources SDK (#54843)
- Application Gateway for Containers as ingress for Ray clusters on Azure

💫 Enhancements:

- Azure improvements: Cleaning up extra resources (MSI, VNET, NSG) during cluster teardown (#57610)
- Updated defaults for Azure cluster templates (#57716)
- Better availability zone support for Azure node pools (#55532)
- Hello world release tests for Azure and GCE (#57597, #57695)
- Improved cluster resource state handling to fix over-provisioning (#57130)

🔨 Fixes:

- Fixed autoscaler state synchronization issues (#57010)
- Better handling of node state information (#57130)
- Improved timeout handling for patch requests (#56605)

Thank you to everyone who contributed to this release!
Special thanks to all the contributors who helped make Ray 2.51.0 possible through bug fixes, features, documentation improvements, and testing efforts.

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.51.0)

---

## ray-2.50.1: Ray-2.50.1
**Published:** 2025-10-18

Ray Core: Fix deadlock when cancelling stale requests on in-order actors (#57746)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.50.1)

---

## ray-2.50.0: Ray-2.50.0
**Published:** 2025-10-10

# Release Highlights
**Ray Data:** 
This release offers many updates to Ray Data, including: 
- The default shuffle strategy is now changed from sort-based to hash-based. This will result in much lower peak memory usage and improved shuffle performance for aggregations.
- We’ve added a new expression API enables predicate-based filtering, UDF transformations with `with_column`, and column aliasing for more powerful data transformations
- Ray Data LLM has a number of new enhancements for multimodal data pipelines, including multi-node tensor and pipeline parallelism support per replica and ability to share vLLM engines across processors.


**Ray Core:** 

Alpha release of **Ray Direct Transport** (formerly GPU objects) - simply enable it by adding the `tensor_transport` parameter to the existing native Ray Core API. This keeps GPU data in GPU memory until a transfer is needed, avoiding expensive serialization and copies to and from the Ray object store. It uses efficient data transports such as collective communication libraries (GLOO or NCCL) or point-to-point RDMA (via NVIDIA’s NIXL) to transfer data directly between devices, including both CPUs and GPUs.

**Ray Train:**

Local mode support for multi-process training with `torchrun`, enhanced checkpoint management with new upload modes and validation functions


**Ray Serve:**
- Async Inference alpha release -  New Ray Serve APIs for supporting long-running asynchronous inference tasks, such as for video or large document processing. Includes capabilities for using different message brokers, adapters like `celery` and DLQ.
- Support for replica ranks - Replica level ranks are added for supporting large-model inference use-cases such as wide Data Parallel and Expert Parallel setups.
- FastAPI factory pattern support - Enables using FastAPI plugins that are not serializable via cloudpickle.  
- Throughput optimizations - Enable these using the `RAY_SERVE_THROUGHPUT_OPTIMIZED` environment variable. 

**RLLib:**
Add `StepFailedRecreateEnv` exception for users with unsatisfiable environments

**Ray Serve/Data LLM:**

Improvements to multi node serving, loading models from remote storages, and sharing resources for efficiency (fractional gpus, sharing gpus on a data pipeline with shared stages)

# Ray Libraries
## Ray Data

🎉 New Features:

- **Expression and Filtering API:** New expression API enables predicate-based filtering, UDF transformations with with_column, and column aliasing for more powerful data transformations (#56716, #56313, #56550, #55915, #55788, #56193, #56596)
- Added support for projection pushdown into Parquet reads (#56500)
- New download expression enables efficient loading of data from columns containing URIs with improved performance and error handling (#55824, #56462, #56294, #56852, #57146)
- New `explain()` API provides insights into dataset execution plans (#55482)
- Added `streaming_train_test_split` to avoid materialization for train/test splits (#56803)
- **Ray Data LLM:**
  - Enabled multi-node tensor and pipeline parallelism for LLM processing (#56779)
  - Added `chat_template_kwargs` parameter for customizing chat templates (#56490)
  - Added support for OpenAI's nested image URL format in multimodal pipelines (#56584)
  - vLLM engines can now be shared across sequential processors for better resource utilization (#55179)
- Enhanced Dataset.stats() output with input/output row counts per operator (#56040)
- Added new metrics for task duration, inputs per task, and output blocks (#56958, #56379)
- Time to first batch metric for better iteration performance monitoring (#55758)
- Added type-specific aggregators for numerical, categorical, and vector columns (#56610)
- Added fine-grained concurrency controls with `max_task_concurrency` and resource allocation options (#56370, #56381)

💫 Enhancements:

- **Join and shuffle improvements:** 
  - Default shuffle strategy changed from sort-based to hash-based for better performance (#55510)
  - Improved groupby performance with sort-shuffle pull-based approach (#57014)
  - Improved join operations with new abstractions (#57022, #56945, #55759)
- **Tensor type handling improvements:** 
  - Improved compatibility between PyArrow native types, extension types, and pandas Arrow dtypes (#57566, #57176, #57057)
  - Joins now supported with list/tensor non-key columns (#5648)
  - Enhanced support for variable-shaped tensor arrays with different dimensions (#57240, #56918, #56457)
  - Added serialization/deserialization for PyArrow Extension Arrays (#51972)
- Removing Parquet metadata fetching in ParquetDatasource (#56105)
- Resource requirements (num_cpus/gpus, memory) are now top-level parameters in most APIs for easier configuration (#56419)
- zip() operator now supports combining multiple datasets, not just pairs (#56524)
- Concurrency parameter now accepts tuples for more flexible configuration (#55867)
- Write operations now use iterators instead of accumulating blocks in memory (#57108)
- Reduced memory usage for OneHotEncoder (#56565)
- Reduced memory usage for schema unification (#55880)
- Eliminated unnecessary block copying and double execution of arrow conversions (#56569, #56793)
- Improved Parquet encoding ratio estimation (#56268)
- Enabled per-block limiting for Limit operator (#55239)
- Optimized schema handling with deduplication and removed unnecessary unification (#55854, #55926)
- Improved issue detection with event emission instead of just logs (#55717)
- Better metric organization and external queue metric handling (#55495, #56604)
- New backpressure policy based on downstream processing capacity (#55463)

🔨 Fixes:

- Fixed streaming executor to properly drain output queues (#56941)
- Improved resource management and reservation for operators (#56319, #57123)
- Fixed retry logic for hash shuffle operations (#57575)
- Fix split_blocks produce empty blocks (#57085)
- Initialize datacontext after setting src_fn_name in actor worker (#57117)
- Fix mongo datasource collStats invocation (#57027)
- Fixing empty projection handling in ParquetDataSource (#56299)
- Fix UnboundLocalError when read_parquet with columns and no partitioning (#55820)
- Fix high memory usage with FileBasedDatasource & ParquetDatasource when using a large number of files (#55978)
- [llm] Fixed LLM processor deployment with Ray Serve (#57061)
- [llm] Fixed multimodal image extraction when system prompts are absent (#56435)
- Ignore metadata for pandas block (#56402)
- Remove metadata for hashing + truncate warning logs (#56093)


📖 Documentation:

- Error in ray.data.groupby example in docs. (#57036)
- Update on ray.data.Dataset.map() type hints. (#52455)
- Small typo fix. (#56560)
- Fix a typo. (#56587)
- Fix documentation for new execution options resource limits assignment. (#56051)
- Fix broken code snippets in user guides. (#55519)
- Add Autoscaling Config for Context docs. (#55712)
- Make object store tuning tips consistent with other pages. (#56705)
- New example of how to perform batch inference with embedding models (#56027)


## Ray Train

🎉 New Features:

- **Local mode support for Ray Train V2**
  - Add local mode support to Ray Train v2 (num_workers=0). (#55487)
  - Add PyTorch local mode support for multi-process training with torchrun. (#56218)
- **Async checkpoint and validation for Ray Train**
  - Add checkpoint_upload_mode to ray.train.report. (#55637)
  - Add checkpoint_upload_function to ray.train.report. (#56208)
  - Add validate_function and validate_config to ray.train.report. (#56360)
  - Add ray.train.get_all_reported_checkpoints method. (#54555)

💫 Enhancements:

- **Ray Train V2 Migration**
  - Implement BaseWorkerGroup for V1/V2 compatibility. (#57151)
- Train Controller is always actor + fix tune integration to enable this. (#55556)
- Refactor AcceleratorSetupCallback to use before_init_train_context. (#56509)
- Move collective implementations to train_fn_utils. (#55689)
- **Ray Train Framework support enhancements**
  - Add hf trainer support for dictionary of datasets. (#56484)
  - Add usage tag key for JaxTrainer. (#55887)
- Add Torch process group shutdown timeout. (#56182)
- Ray Train disables blocking get inside async warning. (#56757)
- ThreadRunner captures exceptions from nested threads. (#55756)
- Abort reconciliation thread catches ray.util.state.get_actor exception. (#56600)
- **Ray Data Integration**
  - Minor rework of get_dataset_shard. (#55825)
  - Create a deepcopy of the data context on the split coordinator process. (#56211)
  - Enable debug logging; fix default actor_locality_enabled. (#56632)
- Refactor call_with_retry into shared library and use it to retry checkpoint upload. (#56608)
- Remove Placement Group on Train Run Abort. (#56011)

🔨 Fixes:

- Fix LightGBM v2 callbacks for Tune only usage. (#57042)
- Ignore tensorflow test for py312. (#56244)
- Revising test_jax_trainer flaky test. (#56854)
- Fix test_jax_trainer imports. (#55799)
- Fix test_jax_trainer::test_minimal_multihost Flaky Test. (#56548)
- Disable drop_last flag to fix division by zero in torch dataloader baselines. (#56395)
- Preload a subset of modules for torch dataloader forkserver multiprocessing. (#56343)

📖 Documentation:

- Add checkpoint_upload_mode to checkpoint docs. (#56860)
- Add get_all_reported_checkpoints and ReportedCheckpoint to API docs. (#56174)
- Fix typo for Instantiating in ray train doc. (#55826)

🏗 Architecture refactoring:

- Release tests for ray train local mode. (#56862)
- Migrate tune_rllib_connect_test & tune_cloud_long_running_cloud_storage to ray train v2. (#56844)
- Add v2 multinode persistence release test. (#56856)
- Attach a quick checkpoint when reporting metrics. (#56718)
- Upgrade tune_torch_benchmark to v2. (#56804)
- Move tune_with_frequent_pausing to Ray Train v2 and tune_tests folder. (#56799)
- Migrate xgboost/lgbm benchmarks to train V2. (#56792)

## Ray Tune

🎉 New Features:

- Trigger Checkpointing via Trial / Tuner Callback. (#55527)

💫 Enhancements:

- Improve _PBTTrialState for dev/debugging usage. (#56890)
- Enable Train V2 in Tune unit tests and examples. (#56816)
- Enable Train v2 in doc examples. (#56820)
- Reintroduce keras tune callback. (#57121)

🔨 Fixes:

- Increase tune checkpoint test latency threshold. (#56251)
- Remove a bunch of low-signal/redundant train/air/tune tests. (#56477)
- Remove tune_air_oom test. (#57089)


## Ray Serve

🎉 New Features:

- Add tests and DLQ business logic for async inference. (#55608)
- Foundation work for aggregating metrics on controller. (#55568)
- Include custom metrics method and report to controller. (#56005)
- Add post scaling api. (#56135)
- Introduce deployment rank manager. (#55729)
- Integrated deployment ranks with deployment state. (#55829)
- Add rank and world size in replica context. (#55827)
- Added ssl to ray serve. (#55228)
- Custom parameter for downscaling to zero. (#56573)
- Add optional APIType filter to /api/serve/applications/ endpoint. (#56458)
- Make deployment retry configurable. (#56530)

💫 Enhancements:

- Aggregate autoscaling metrics on controller. (#56306)
- Update metrics_utils for future global metrics aggregation in controller. (#55568)
- Use deployment method in access logs for replicas. (#56829)
- Cache router metrics. (#55897)
- Allow same event loop handle shutdown from sync context. (#55551)
- Additional deps to start with prometheus. (#57155)
- Require prefix RAY_SERVE_ for env vars + value verification. (#55864)
- Record queued metrics on timeseries. (#57024)
- Add throughput opt env var for serve. (#55804)
- Fix None pending Request. (#54775)
- Omit unnecessary newlines in the config generated by serve build app:app. (#56609)
- Expose actor name for target group api. (#56738)

🔨 Fixes:

- Fix proxy lua dependency in dockerfile. (#57221)
- Fix non thread safe asyncio task creation in router. (#56124)
- Fix throughput optimized benchmarks. (#56173)
- Move ingress validation for multiple fastapi deployment into client. (#56706)
- Explicitly close choose_replicas_with_backoff async generator. (#56357)
- Fix buffered logging reusing request context. (#56094)
- Use default gc frequency for proxy. (#56511)
- Fixing deployment scoped custom autoscaling. (#56192)

📖 Documentation:

- Stable links for Ray serve. (#56241)
- Add document for using fastapi factory pattern in serve. (#56607)
- Add documentation for async inference (#56453)

🏗 Architecture refactoring:

- Add microbenchmark for throughput optimized configuration. (#55900)
- Only checkpoint controller state when it is confirmed that target state has changed (#55848)
- Proxy Actor Interface. (#56288)
- Allow ProxyActor to return true/false for health check. (#56660)


## Ray Serve/Data LLM

🎉 New Features:

- Score API Integration for Serve LLM. (#55914)
- Add start/stop_profile method to LLMServer. (#55920)
- Add prefix cache hit rate to Serve LLM dashboard. (#55675)
- Configure aggregation interval for dashboard. (#56591)

💫 Enhancements:

- Bump vLLM to 0.10.2. (#56535)
- Vllm bump -> 0.10.1.1. (#56099)
- Refactor: Improve Deployment Builder Ergonomics and Code Organization. (#57181)
- Fix build_llm_processor for ServeDeploymentProcessor. (#57061)
- Allow setting data_parallel_size=1 in engine_kwargs. (#55750)
- Allow tuple for concurrency arg. (#55867)
- Fix multimodal image extraction when no system prompt is present. (#56435)
- Support azure and abfss in LLM config. (#56441)
- Support custom s3 endpoint when downloading models from remote. (#55458)
- Skip safetensor file downloads for runai streamer mode. (#55662)
- Support colocating local DP ranks in DPRankAssigner. (#55720)
- Adjust LLM engine timing logic. (#55595)
- Fixed DP DSV3 issues. (#55802)
- Gracefully return timeouts as HTTPException. (#56264)
- Remove upstreamed workarounds 1/3. (#54512)

🔨 Fixes:

- Changed LMCache dependency to use 0.3.3 to avoid regressions in the release test. (#56104)
- Fix doc test for Working with LLMs guide. (#55917)
- Fix sglang byod on release. (#55885)

📖 Documentation:

- Add gpt oss deployment example. (#56400)
- Add serve llm example to index page + other minor fix. (#56788)
- Example serve llm deployment. (#55819)
- Fix serve llm examples. (#56382)
- Docs: serve llm deployment examples refinement. (#56287)
- Add example of serving a VLLM model on fractional gpu. (#57197)
- Add main pytest code snippet to those tests that were missing it. (#57167)


## RLlib

🎉 New Features:

- Add StepFailedRecreateEnv exception. (#55146)

💫 Enhancements:

- Add tags to envrunner calls, count in flight requests in ActorManager. (#56930, #56953)
- Add spaces in case only offline data is used. (#56141)
- Add Footsies environment and tests. (#55041)

🔨 Fixes:

- Fix failing env step in MultiAgentEnvRunner. (#55567)
- Fix Metrics/Stats lifetime count and throughput measurement for async remote actors. (#56047)
- Fixes Implementation of Shared Encoder. (#54571)
- Fix MetricsLogger/Stats throughput bugs. (#55696)

📖 Documentation:

- [RLlib] [DOC] Fix documentation typos and grammatical issues in RLlib docs (#56130)
- Update rllib-env.rst - typo. (#56140)
- Fixing typo in the RLlib documentation. (#55752)
- Fix formatting of class references. (#55764)

🏗 Architecture refactoring:

- Remove checkpoint release tests. (#57105)
- Remove long_running_apex test. (#57097)
- LINT: Enable ruff imports for multiple directories in rllib. (#56736)
- Upgrade g3 to g4 machine for aws release test. (#56248)


## Ray Core

🎉 New Features:

- **Alpha release of [Ray Direct Transport](https://docs.ray.io/en/master/ray-core/direct-transport.html)**
  - Support ray.put() and ray.get() with nixl in gpu objects. (#56146)
  - Support using ray.get with nixl to retrieve data from GPU object refs created by remote tasks. (#56559)
  - Support tensor transfer from outside owners of actors. (#56485)
  - Automatically enable tensor transport for the actor if any method specifies one. (#55324)
  - Support cpu tensor transfer with NIXL in GPU Objects. (#55793)
  - Handle multiple transfers of the same object to an actor. (#55628)
  - Support NIXL as tensor transport backend. (#54459)
  - Add a user-facing call to wait for tensor to be freed. (#55076)
  - Always write to GPUObjectStore to avoid _get_tensor_meta() from hanging indefinitely. (#55433)
  - Add warning when GPU object refs passed back to the same actor. (#55639)
  - Avoid triggering a KeyError by the GPU object GC callback for intra-actor communication. (#54556)
- Enable autoscaler v2 on clusters launched by the cluster launcher. (#55865)
- Ray Symmetric Run Script and ray symmetric-run command. (#55111, #56497)

💫 Enhancements:

- **Ray Event Export**
  - GCS AddEvent support. (#55528)
  - Actor event: add proto schema. (#56221)
  - Node event: add proto schema and send node events to the aggregator. (#56031, #56426)
  - Job event: add schema for driver job event and send job events to the aggregator. (#55032, #55213)
  - Emit actor events to Event aggregator. (#56617)
  - Export node event by default. (#56810)
- **Security**
  - Bind ray internal servers to the specified node ip instead of 0.0.0.0. (#55178, #55210, #55298, #55484)
  - Bind dashboard agent http server to localhost in addition to the node ip. (#55910)
  - Bind dashboard agent grpc to specified ip instead of 0.0.0.0. (#55732)
  - Bind runtime env agent and dashboard agent http server to specified ip instead of 0.0.0.0. (#55431)
- **RPC network fault tolerance**
  - Making core worker pub sub RPCs fault tolerant. (#56436)
  - Make RequestWorkerLease RPC Fault Tolerant. (#56191)
  - Making ReturnWorkerLease Idempotent. (#56073)
  - Making CancelWorkerLease RPC Fault Tolerant. (#56195)
  - Make Free Objects RPC Fault Tolerant. (#56293)
  - Make PinObjectIDs RPC Fault Tolerant. (#56443)
  - Make Unsubscribe Idempotent. (#57546)
  - Core Worker GetObjStatus GRPC Fault Tolerance. (#54567)
- Not overriding accelerator id env vars when num_accelerators is 0 or not set. (#54928)
- Migrate metric collection from opencensus to opentelemetry. (#53098, #53740)
- Add per worker process group and deprecate process subreaper in favor of cleanup using process group. (#56476)
- Add node_id validation in NodeAffinitySchedulingStrategy. (#56708)
- Add io_context metrics to gcs and raylet. (#55762)
- Modify RedisDelKeyPrefixSync to use the Redis SCAN command instead of KEYS. (#56907)
- Add error_type to job failures. (#55578)
- Add PID to structured logs for tasks and actors. (#55176)
- Log actor name when warning about excess queueing. (#57124)
- Output the error log on the driver side if the failed task will still retry. (#56472)
- Prometheus http service discovery API. (#55656)
- Add node ip in runtime env error message to improve debug observability. (#56837)
- Fallback unserializable exceptions to their string representation. (#55476)
- Introduce new exception type for un-pickleable exceptions. (#55878)
- Improve docs for custom serialization for exceptions + add test. (#56156)
- Add a warning when returning an object w/ num_returns=0. (#56213)
- Adding ability to specify availability zones for ray cluster node pools on Azure. (#55532)
- Query for supported Microsoft.Network/virtualNetworks API versions instead of relying on resource_client.DEFAULT_API_VERSION. (#54874)
- Loosen Ray self-dependency check to allow matching versions. (#57019)
- Add support for pip_install_options for pip. (#53551)
- Proper typing for ObjectRef. (#55566)

🔨 Fixes:

- Use subscription id from azure profile if not provided in config during AzureNodeProvider init. (#56640)
- Always create standard public IP addresses (basic sku is deprecated). (#57131)
- Fix: bug with config key pairs when launching worker nodes. (#57107)
- If azure cluster launcher keypair doesnt exist create one automatically + doc typo fix. (#54596)
- Fix "objects_valid" for the case that multiple instances of the same task are storing returns. (#54904)
- Fix objects_valid check failure with except from BaseException. (#55602)
- Preserve err type in case of task cancellation due to actor death. (#57538)
- Fix checking for uv existence during ray_runtime setup. (#54141)
- Prevent sending SIGTERM after calling Worker::MarkDead. (#54377)
- Fixed the bug where the head was unable to submit tasks after redis is turned on. (#54267)
- Fix possible race by checking node cache status instead of just subscription. (#54745)
- Fix get actor timeout multiplier. (#54525)
- Use a temporary file to share default worker path in runtime env. (#53653)
- Fix check fail when task buffer periodical runner runs before RayEvent is initialized. (#55249)
- Patch grpc with RAY_num_grpc_threads to control grpc thread count. (#54988)
- Fix HandleRefRemoved thread safety. (#56445)
- Fix error handling for plasma put errors. (#56070)
- Fix batching logic in CoreWorkerPlasmaStoreProvider::Get. (#56041)
- Fix RAY_CHECK failure during shutdown due to plasma store race condition. (#55367)
- Fix autoscaler RAY_CHECK when GcsAutoscalerStateManager is out of sync with NodeManager. (#57010)
- Fix bug where inflight requests are not taken into account by retryable. (#57142)
- A timeout should be set when submitting patch requests for autoscaler. (#56605)
- Fix the bug in memray regarding the default configuration of -o {output_file_path}. (#56732)
- Fixed the issue of RemoveActorNameFromRegistry being called repeatedly. (#54955)
- Fixed an issue where the command executed when use_podman=true and run_env=None was not prefixed with podman exec. (#56619)
- Fix data race when using async gpu to gpu transfer. (#57112)
- Retry + Make FreeActorObject idempotent. (#56447)
- Fix check crash on gpu obj free if driver knows actor is dead. (#56404)
- Handle system errors with a background monitor thread. (#56513)
- Fix GPU metrics. (#56009)
- Don't disconnect worker client on OBOD unless the worker is dead. (#57185)
- Prevent stale GET request being registered if its lease was cleared. (#56766)
- Drop messages received after ClientConnection::Close. (#56240)
- Fix cancel race that leads to RAY_CHECK it->second.submitted_task_ref_count > 0. (#56123)
- Reorder asyncio actor shutdown to terminate asyncio thread first. (#56827)
- Fix actor import error message for async actors. (#55722)
- Allow task manager access with submitter mutex + unify retry. (#56216)
- Fix bug in restore_from_path such that connector states are also restored on remote EnvRunners. (#54672)
- Fix S3 access issue in AKS. (#56358)
- Add S3 public bucket fallback to handle NoCredentialsError. (#56334)
- Fix ABFSS (Azure Blob File System Secure) protocol support problems during E2E test. (#56188)
- Ray cluster commands (up, attach, status, etc) updates to work on Windows. (#54982)
- Update cluster scheduler to handle label selector hard node id constraint. (#56235)

📖 Documentation:

- Added guide on using type hints with Ray Core. (#55013)
- Lifecycle of a task. (#55496)
- Add OSS Document for Task Events. (#56203)
- Fix Missing Events Issue in Task Events. (#55916)
- Add docs for asyncio and object mutability. (#56790)
- Update getting started and set up document for ray on vsphere. (#56954)
- Docfix - rst annotation showing up in render. (#57104)
- Add threading requirement to NodeProvider interface. (#56349)
- Add guidance for matching Ray and Python versions with uv envs. (#56597)
- Fix documentation typos, grammar, and terminology inconsistencies. (#56066, #56067, #56068, #56069, #56128, #56129, #56130, #56131, #56132, #56272, #56273, #56274, #56275, #56277, #56278, #56279)
- Update SLURM docs with symmetric-run. (#56775)
- Update Kueue integration documentation to include RayService & RayCluster support. (#56781)
- Application Gateway for Containers as ingress to access Ray Cluster. (#56574)
- Update DLAMI Information in aws.md. (#55702)

## Dashboard

💫 Enhancements:

- Use pynvml for GPU metrics. (#56000)
- Default dashboard usability improvements. (#55620)
- Make Ray Train Dashboard Panel Ids Static. (#55559)
- Small fixes to Metrics Tab for kube-ray clusters. (#57149)
- Add metadata to indicate full dashboard embedding is supported. (#56077)
- Use ray node id instead of ip for profilinglink. (#55439)
- Fix grafana dashboard generation bug. (#56346)
- Catch OSError when detecting the GPU. (#56158)

🔨 Fixes:

- Removed references to a deleted Data metrics panel. (#55478)
- Fix typo in memory_utils and adjust display formatting for clarity. (#56217)


## Ray Images

🎉 New Features:

- Add support for building and publishing ray-extra images. (#56543)
- Add ray-llm and ray-ml extra images. (#56800)
- Build ray-extra images for aarch64. (#56818)
- Add slim image to the image build matrix. (#55723)

💫 Enhancements:

- Add haproxy binary, for ray serve use. (#56845)
- Add ~/.local/bin to PATH in slim image. (#56920)
- Remove slim's dependency on normal bases. (#56544)
- Add label for ray version and commit. (#56493)
- Refactor apt package installation. (#55701)
- Allow using explicit base type. (#56545)
- Add extra-test stage in image building. (#55725)
- Add test rules for image building files. (#56554)
- Add ray-llm image type check. (#56542)
- Unify label and tag conventions. (#56189)
- GKE GPU compat paths: PATH, LD_LIBRARY_PATH (temporarily). (#55569)
- Stop publishing ray-ml images. (#57070)
- Stop building and releasing x86 osx wheels. (#57077)

📖 Documentation:

- Update latest Docker dependencies for 2.49.0 release. (#55966)
- Update latest Docker dependencies for 2.49.2 release. (#56760)

## Wheels and images

💫 Enhancements:

- Use bazel run to generate files required for the wheel and testing (#55957, #56527, #56969, #56004, #55928)
- Ban click 8.3.0. (#56789)
- Upgrade protobuf to v4. (#54496)
- Add adlfs[abfs] into image (#56084)
- Upgrade boto3 to 1.29.x. (#56363)
- Upgrading orjson to 3.9.15. (#55972)
- Update spdlog to 15.3. (#56711)

## Thanks!

Thank you to everyone who contributed to this release! 

@alexeykudinkin, @richardliaw, @nrghosh, @ljstrnadiii, @Daraan, @kouroshHakha, @Bye-legumes, @kamil-kaczmarek, @jugalshah291, @sampan-s-nayak, @jjyao, @Evelynn-V, @gangsf, @omatthew98, @TimothySeah, @kshanmol, @goutamvenkat-anyscale, @axreldable, @jiangwu300, @simonsays1980, @400Ping, @JasonLi1909, @chuang0221, @weiliango, @Myasuka, @win5923, @liulehui, @khluu, @ok-scale, @eicherseiji, @tianyi-ge, @MengjinYan, @kevin85421, @Yevet, @orangeQWJ, @vie-serendipity, @edoakes, @wyhong3103, @israbbani, @vickytsang, @HassamSheikh, @acrewdson, @czgdp1807, @daiping8, @carolynwang, @thc1006, @jeffreyjeffreywang, @Stack-Attack, @Catch-Bull, @elliot-barn, @Levi080513, @BestVIncent, @dragongu, @jmajety-dev, @jcarlson212, @tohtana, @abrarsheikh, @crypdick, @Yicheng-Lu-llll, @ZacAttack, @justinvyu, @lk-chen, @alanwguo, @mcoder6425, @my-vegetable-has-exploded, @yancanmao, @arcyleung, @rjpower, @codope, @harshit-anyscale, @dayshah, @stephanie-wang, @KaisennHu, @ryanaoleary, @saihaj, @mattip, @rueian, @Kunchd, @pavitrabhalla, @owenowenisme, @Aydin-ab, @gvspraveen, @minerharry, @JackGammack, @jpatra72, @coqian, @zcin, @dstrodtman, @aslonnie, @ahao-anyscale, @GuyStone, @iamjustinhsu, @seanlaii, @ruisearch42, @akyang-anyscale, @ArturNiederfahrenhorst, @bveeramani, @OneSizeFitsQuorum, @xinyuangui2, @sb-hakunamatata, @22quinn, @Sparks0219, @sven1977, @snehachhabria, @dioptre, @nadongjun, @eric-higgins-ai, @marosset, @MatthewCWeston, @pcmoritz, @can-anyscale, @pimdh, @roshankathawate, @matthewdeng, @martinbomio, @GokuMohandas, @alimaazamat, @ali-corpo, @landscapepainter, @Qiaolin-Yu, @vaishdho1, @avigyabb, @srinathk10, @tannerdwood

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.50.0)

---

## ray-2.49.2: Ray-2.49.2
**Published:** 2025-09-19

There is no difference between 2.49.2 and 2.49.1, though we needed a patch version for other out of band reasons. To fill the awkward blankness, here is a haiku about Ray:

Summit drawing near
Ray advances, step by step
Scaling without end

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.49.2)

---

## ray-2.49.1: Ray-2.49.1
**Published:** 2025-09-03

- Ray Dashboard: Fix issue where GPU metrics are missing (#56006)
- Ray Data: Fixed regression in handling very large schemas (#56058)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.49.1)

---

## ray-2.49.0: Ray-2.49.0
**Published:** 2025-08-26

# Release Highlights
**Ray Data:** 

- We’ve implemented a variety of performance enhancements, including improved actor/node autoscaling with budget-aware decisions; faster/more accurate shuffle accounting; reduced Parquet metadata footprint; and out-of-order execution for higher throughput.
- We’ve also implemented anti/semi joins, stratified train_test_split, and added Snowflake connectors.

**Ray Core:** 

- Performance/robustness cleanups around GCS publish path and raylet internals; simpler OpenTelemetry flagging; new user-facing API to wait for GPU tensor free; plus assorted test/infra tidy-ups

**Ray Train:**

- We’ve introduced a new JaxTrainer with SPMD support for TPUs.

**Ray Serve:**

- Custom Autoscaling per Deployment Serve now supports user-defined autoscaling policies via AutoscalingContext and AutoscalingPolicy, enabling fine-grained scaling logic at the deployment level. This is part of a large effort where we are adding support for autoscaling based on custom metrics in Serve, see [this RFC](https://github.com/ray-project/ray/issues/41135#issuecomment-3156717488) for more details.
- Async Inference (Initial Support): Ray Serve introduces asynchronous inference execution, laying the foundation for better throughput and latency in async workloads. Please see [this RFC](https://github.com/ray-project/ray/issues/54652) for more details.
- Major Performance Gains: This version of ray serve brings double digit % performance improvements both in throughput and latency. See release notes for more details.

**Ray Serve/Data LLM:**

- We’ve refactored Ray Serve LLM to be fully compatible with the default vllm serve and also now supports vLLM=0.10.
- We’ve added a prefix cache-aware router with PrefixCacheAffinityRouter for optimized cache utilization; dynamic cache management via reset prefix cache remote methods; enhanced LMCacheConnectorV1 with kv_transfer_config support.

# Ray Libraries
## Ray Data
🎉 New Features:

- Wrapped batch indices in a BatchMetadata object to make per-batch metadata explicit. (#55643)
- Added support for Anti/Semi Join types. (#55272)
- Introduced an Issue Detection Framework. (#55155)
- Added an option to enable out-of-order execution for better performance. (#54504)
- Introduced a StreamingSplit logical operator for DAG rewrite. (#54994)
- Added a stratify parameter to train_test_split. (#54624)
- Added Snowflake connectors. (#51429)
- Updated Hudi integration to support incremental query. (#54301)
- Added an Actor location tracker. (#54590)
- Added BundleQueue.has_next. (#54710)
- Made DEFAULT_OBJECT_STORE_MEMORY_LIMIT_FRACTION configurable. (#54873)
- Added Expression support & a with_columns API. (#54322)
- Allocate GPU resources in ResourceManager. (#54445)

💫 Enhancements:

- Decoupled actor and node autoscaling; autoscaling now also considers budget. (#55673, #54902)
- Faster hash-shuffle resource usage calculation; more accurate shuffle progress totals. (#55503, #55543)
- Reduced Parquet metadata storage usage. (#54821)
- Export API improvements: refresh dataset/operator state, sanitize metadata, and truncate exported metadata. (#55355, #55379, #55216, #54623)
- Metrics & observability: task metric improvements, external-buffer block-count metric, row-based metrics, clearer operator names in logs, single debug log when aggregators are ready. (#55429, #55022, #54693, #52949, #54483)
- Dashboard: added “Max Bytes to Read” panel/budget, panels for blocks-per-task and bytes-per-block, and streaming executor duration. (#55024, #55020, #54614)
- Planner/execution & infra cleanups: ExecutionResources and StatsManager cleanup, planner interface refactor, node trackers init, removed ray.get in _MapWorker ctor, removed target_shuffle_max_block_size. (#54694, #55400, #55018, #54665, #54734, #55158)
- Behavior/interop tweaks: map_batches defaults to row_modification=False and avoids pushing past limit; limited operator pushdown; prefetch for PandasJSONDatasource; use cloudpickle for Arrow tensor extension ser/des; bumped Arrow to 21.0; schema warning tone change. (#54992, #54457, #54667, #54831, #55426, #54630)
- Removed randomize-blocks reorder rule for more stable behavior. (#55278)

🔨 Fixes:

- AutoscalingActorPool now properly downscales after execution. (#55565)
- StatsManager handles StatsActor loss on disconnect. (#55163)
- Handle missing chunks key when Databricks UC query returns zero rows. (#54526)
- Handle empty fragments in sampling when num_row_groups=0. (#54822)
- Restored handling of PyExtensionType to keep compatibility with previously written datasets. (#55498)
- Prevent negative resource budget when concurrency exceeds the global limit; fixed resource-manager log calculation. (#54986, #54878)
- Default write_parquet warning removed; handled unhashable types in OneHotEncoding. (#54864, #54863)
- Overwrite mode now maps to the correct Arrow behavior for parallel writes. (#55118)
- Added back from_daft Arrow-version checks. (#54907)
- Pandas chained in-place assignment warning resolved. (#54486)
- Test stability/infra: fixed flaky tests, adjusted bounds and sizes, added additional release tests/chaos variants for image workloads, increased join test size, adjusted sorting release test to produce 1 GB blocks. (#55485, #55489, #54806, #55120, #54716, #55402, #54971)

📖 Documentation:

- Added a user guide for aggregations. (#53568)
- Added a code snippet in docs for partitioned writes. (#55002)
- Updated links to Lance documentation. (#54836)

## Ray Train
🎉 New Features:

- Introduced JaxTrainer with SPMD support on TPUs (#55207)

💫 Enhancements:

- ray.train.get_dataset_shard now lazily configures dataset sharding for better startup behavior (#55230)
- Clearer worker error logging (#55222)
- Fail fast when placement group requirements can never be satisfied (#54402)
- New ControllerError surfaced and handled via failure policy for improved resiliency (#54801, #54833)
- TrainStateActor periodically checks controller health and aborts when necessary (#53818)

🔨 Fixes:

- Resolve circular import in ray.train.v2.lightning.lightning_utils (#55668)
- Fix XGBoost v2 callback behavior (#54787)
- Suppress a spurious type error (#50994)
- Reduce test flakiness: remove randomness and bump a data-integration test size (#55315, #55633)

📖 Documentation:

- New LightGBMTrainer user guide (#54492)
- Fix code-snippet syntax highlighting (#54909)
- Minor correction in experiment-tracking guide comment (#54605)

🏗 Architecture refactoring:

- Public Train APIs routed through TrainFnUtils for consistency (#55226)
- LoggingManager utility for Train logging (#55121)
- Convert DEFAULT variables from strings to bools (#55581)

## Ray Tune
🎉 New Features:

- Add video FPS support to WandbLoggerCallback (#53638)

💫 Enhancements:

- Typing: reset_config now explicitly returns bool (#54581)
- CheckpointManager supports recording scoring metric only (#54642)

🔨 Fixes:

- Fix XGBoost v2 callback integration (#54787)
- Correct type for RunConfig.progress_reporter (#48439)

📖 Documentation:

- Minor fixes (#55125, #54942)

## Ray Serve
🎉 New Features:

- Async inference support in Ray Serve (initial phase). Provides basic asynchronous inference execution, with follow-up work planned for failed/unprocessed queues and additional tests.[ #54824](https://github.com/ray-project/ray/pull/54824)
- Per-deployment custom autoscaling controls. Introduces AutoscalingContext and AutoscalingPolicy classes, enabling user-defined autoscaling strategies at the deployment level.[ #55253](https://github.com/ray-project/ray/pull/55253)
- Same event loop router. Adds option to run the Serve router in the same event loop as the proxy, yielding ~17% throughput improvement.[ #55030](https://github.com/ray-project/ray/pull/55030)

💫 Enhancements:

- Async get_current_servable_instance(). Converts the FastAPI dependency to async def, removing threadpool overhead and boosting performance: 35% higher RPS and reduced latency.[ #55457](https://github.com/ray-project/ray/pull/55457)
- Access log optimization. Cached contexts in request path logging improved request throughput by ~16% with lower average latency.[ #55166](https://github.com/ray-project/ray/pull/55166)
- Batching improvements. Default batch wait timeout increased from 0.0s to 0.01s (10ms) to enable meaningful batching.[ #55126](https://github.com/ray-project/ray/pull/55126)
- HTTP receive refactor. Cleaned up handling of replica-side HTTP receive tasks.[ #54543](https://github.com/ray-project/ray/pull/54543) /[ #54565](https://github.com/ray-project/ray/pull/54565)
- Configurable replica router backoff. Added knobs for retry/backoff control when routing to replicas.[ #54723](https://github.com/ray-project/ray/pull/54723)
- Autoscaling ergonomics. Marked per-deployment autoscaling metrics push interval config as deprecated for consistency.[ #55102](https://github.com/ray-project/ray/pull/55102)
- Health check & env var safety. Introduced warnings for invalid/zero/negative environment variable values, with migration path planned for Ray 2.50.0.[ #55464](https://github.com/ray-project/ray/pull/55464),[ #54944](https://github.com/ray-project/ray/pull/54944)
- Improved CLI UX. serve config now prints No configuration was found. instead of an empty string.[ #54767](https://github.com/ray-project/ray/pull/54767)

🔨 Fixes:

- Removed brittle ray._private dependency usage.[ #55659](https://github.com/ray-project/ray/pull/55659)
- HTTP route test fixes. Migrated to get_application_url() to avoid hardcoded URLs, reducing flakiness on Windows.[ #55623](https://github.com/ray-project/ray/pull/55623),[ #54974](https://github.com/ray-project/ray/pull/54974),[ #54924](https://github.com/ray-project/ray/pull/54924),[ #54911](https://github.com/ray-project/ray/pull/54911),[ #54704](https://github.com/ray-project/ray/pull/54704),[ #54903](https://github.com/ray-project/ray/pull/54903),[ #54882](https://github.com/ray-project/ray/pull/54882),[ #54877](https://github.com/ray-project/ray/pull/54877),[ #54631](https://github.com/ray-project/ray/pull/54631),[ #53933](https://github.com/ray-project/ray/pull/53933)
- Semaphore bug fix. Corrected race where more workers than allowed could acquire the semaphore.[ #55147](https://github.com/ray-project/ray/pull/55147)
- LongPollClient cancellation. Prevented spurious cancellation of listen_for_change.[ #54832](https://github.com/ray-project/ray/pull/54832)
- Backpressure error code. gRPC now returns RESOURCE_EXHAUSTED instead of UNAVAILABLE on overload.[ #54537](https://github.com/ray-project/ray/pull/54537)
- Logging improvements. Added request IDs to proxy access logs; avoided duplicate shutdown log lines.[ #54657](https://github.com/ray-project/ray/pull/54657),[ #54534](https://github.com/ray-project/ray/pull/54534)
- Test stability. Various waits, deflakes, and sync fixes across Serve tests.[ #54794](https://github.com/ray-project/ray/pull/54794),[ #54522](https://github.com/ray-project/ray/pull/54522),[ #54585](https://github.com/ray-project/ray/pull/54585)

📖 Documentation:

- Unexpected queuing behavior. Documented quirks in handle request queuing.[ #54542](https://github.com/ray-project/ray/pull/54542)

🏗 Architecture refactoring:

- Router/handle internals refactored for clarity and future feature expansion.[ #55635](https://github.com/ray-project/ray/pull/55635)
- Model composition benchmarks. Added benchmarking to track performance of common composition patterns.[ #55549](https://github.com/ray-project/ray/pull/55549)
- Constants refactor. Utility functions moved out of constants.py for better readability and stricter env var validation.[ #54944](https://github.com/ray-project/ray/pull/54944),[ #55464](https://github.com/ray-project/ray/pull/55464)
- Ray internals migration. Moved usage, ray_option_utils, and selected constants from _private to _common.[ #54915](https://github.com/ray-project/ray/pull/54915),[ #54578](https://github.com/ray-project/ray/pull/54578)

## Ray Serve/Data LLM
🎉 New Features:

- Prefix cache-aware router with PrefixCacheAffinityRouter for optimized cache utilization. ([#55218](https://github.com/ray-project/ray/pull/55218), [#55588](https://github.com/ray-project/ray/pull/55588))
- Reset prefix cache remote method for dynamic cache management. ([#55658](https://github.com/ray-project/ray/pull/55658))
- LMCacheConnectorV1 support for kv_transfer_config to enhance key-value transfer configurations. ([#54579](https://github.com/ray-project/ray/pull/54579))
- LLMServer and LLMEngine major refactor for 100% vLLM serve frontend compatibility. ([#54554](https://github.com/ray-project/ray/pull/54554))

💫 Enhancements:

- vLLM engine upgrade to version 0.10.0 with improved performance and compatibility. ([#55067](https://github.com/ray-project/ray/pull/55067))
- Enhanced error handling for invalid model_id parameters with clearer error messages. ([#55589](https://github.com/ray-project/ray/pull/55589))
- Improved telemetry handling with better race condition management for push operations. ([#55558](https://github.com/ray-project/ray/pull/55558))
- Optimized deployment defaults with better configuration values to prevent bottlenecks. ([#54696](https://github.com/ray-project/ray/pull/54696))
- LoRA workflow improvements with refactored downloading and utility functions. ([#54946](https://github.com/ray-project/ray/pull/54946))
- LLMServer refactor to synchronous initialization for better reliability. ([#54835](https://github.com/ray-project/ray/pull/54835))
- Mistral tokenizer support for tekken tokenizer compatibility. ([#54666](https://github.com/ray-project/ray/pull/54666))
- Smart batching logic that skips batching when batch_interval_ms == 0. ([#54751](https://github.com/ray-project/ray/pull/54751))
- Dashboard enhancements with improved LLM metrics and monitoring capabilities. ([#54797](https://github.com/ray-project/ray/pull/54797))

🔨 Fixes:

- Pyright linting corrections for Ray Serve LLM examples. ([#55284](https://github.com/ray-project/ray/pull/55284))
- Test stability improvements for DeepSeek model and vLLM engine processor tests. ([#55401](https://github.com/ray-project/ray/pull/55401), [#55120](https://github.com/ray-project/ray/pull/55120))
- Serialization fixes for ChatCompletionRequest tool_calls ValidatorIterator objects. ([#55538](https://github.com/ray-project/ray/pull/55538))

📖 Documentation:

- Prefix cache router documentation with comprehensive usage examples. ([#55218](https://github.com/ray-project/ray/pull/55218))
- Multi-LoRA documentation improvements with clearer setup instructions. ([#54788](https://github.com/ray-project/ray/pull/54788))
- STRICT_PACK strategy FAQ documentation explaining data.llm packing behavior. ([#55505](https://github.com/ray-project/ray/pull/55505))

🏗 Architecture refactoring:

- Docker image optimizations with UCX and NCCL updates, plus GKE GPU operator compatibility paths. ([#54598](https://github.com/ray-project/ray/pull/54598), [#55206](https://github.com/ray-project/ray/pull/55206))

## RLlib
🎉 New Features:

- Implemented Implicit Q-Learning (IQL). (#55304, #55422)
- DreamerV3 is now available in PyTorch. (#45463, #55140)
- Discrete actions support for SAC. (#53982)

💫 Enhancements:

- Upgraded RLlink protocol for external env/simulator training. (#53550)
- Performance improvements in Offline RL API through switching to `iter_torch_batches`. (#54277)
- Added an example for curriculum learning in Atari Pong. (#55304)

🔨 Fixes:

- Corrected `TensorType` handling. (#55694)
- Fixed a bug with multi-learner setups in Offline RL API. (#55693)
- Addressed `ImportError` in Atari examples. (#54967)
- Fixed some bugs in the docs for IQL and CQL. (#55614)
- Increased default timesteps on two experiments. (#54185)
- Fixed `TorchMultiCategorical.to_deterministic` when having different number of categories and logits with time dimension. (#54414)
- Added missing documentation for `SACConfig`'s `training()`. (#53918)
- Fixed bug in `restore_from_path` such that connector states are also restored on remote `EnvRunner`s. (#54672)
- Fixed missing support for `config.count_steps_by = "agent_steps"`. (#54885)
- Added missing colon to `CUBLAS_WORKSPACE_CONFIG`. (#53913)
- Removed `rllib_contrib` completely from RLlib. (#55182)

🏗 Architecture refactoring:

- Deprecated TensorFlow support from new API stack. (#55042)
- Deprecated input/output specs from `RLModule`. (#55141)
- Deprecated `--enable-new-api-stack` flag from all scripts. (#54853, #54702)

## Ray Core
🎉 New Features:

- [[Core] Add AcceleratorManager implementation for Rebellions NPU (](https://github.com/ray-project/ray/commit/5a1ddabee85676c51fc044e05f6a63de4cb0e35b)[#53985](https://github.com/ray-project/ray/pull/53985)[)](https://github.com/ray-project/ray/commit/5a1ddabee85676c51fc044e05f6a63de4cb0e35b)

💫 Enhancements:

- [[core][gpu-objects] Garbage collection (](https://github.com/ray-project/ray/commit/9392c25ffe6d391f7d08108ceeb0a637dd89ac4b)[#53911](https://github.com/ray-project/ray/pull/53911)[)](https://github.com/ray-project/ray/commit/9392c25ffe6d391f7d08108ceeb0a637dd89ac4b)
- [[core] Support pip_install_options for pip (](https://github.com/ray-project/ray/commit/4054c9c3368d0e0b45ff19b4bbcb8d967d3f9026)[#53551](https://github.com/ray-project/ray/pull/53551)[)](https://github.com/ray-project/ray/commit/4054c9c3368d0e0b45ff19b4bbcb8d967d3f9026)
- [[core][gpu-objects] Move data transfers to a background thread (](https://github.com/ray-project/ray/commit/c2db8f0ae934759ee1e66d6ed900a5058d0ff1c0)[#54256](https://github.com/ray-project/ray/pull/54256)[)](https://github.com/ray-project/ray/commit/c2db8f0ae934759ee1e66d6ed900a5058d0ff1c0)
- [core][gpu-objects] Pass tensor_transport to store_task_errors even if the actor task throws an exception ([#55427](https://github.com/ray-project/ray/pull/55427))
- [[core][gpu-objects] Exception handling for application errors (](https://github.com/ray-project/ray/commit/64f32074a34b5bf7c7879be664957482f17a9de8)[#55442](https://github.com/ray-project/ray/pull/55442)[)](https://github.com/ray-project/ray/commit/64f32074a34b5bf7c7879be664957482f17a9de8)
- [core][gpu-object] Add a user-facing call to wait for tensor to be freed ([#55076](https://github.com/ray-project/ray/pull/55076))
- [Core] Bind ray internal servers to the specified node ip instead of 0.0.0.0 which improves the security ([#55178](https://github.com/ray-project/ray/pull/55178), [#55210](https://github.com/ray-project/ray/pull/55210), [#55298](https://github.com/ray-project/ray/pull/55298))
- [core] Fallback unserializable exceptions to their string representation ([#55476](https://github.com/ray-project/ray/pull/55476))

🔨 Fixes:

- [core] Fix objects_valid check failure with except from BaseException (#55602)
- [core][gpu-objects] Avoid triggering a KeyError by the GPU object GC callback for intra-actor communication ([#54556](https://github.com/ray-project/ray/pull/54556))
- [[core] fix checking for uv existence during ray_runtime setup (](https://github.com/ray-project/ray/commit/7547481269fce9546ef31329957f7b115a60684b)[#54141](https://github.com/ray-project/ray/pull/54141)[)](https://github.com/ray-project/ray/commit/7547481269fce9546ef31329957f7b115a60684b)
- [core][autoscaler][v1] add heartbeat timeout logic to determine node activity status ([#54030](https://github.com/ray-project/ray/pull/54030))
- [[core] prevent sending SIGTERM after calling Worker::MarkDead (](https://github.com/ray-project/ray/commit/1de822fd76ebaf2fb93fe8e311f73af0157e4fc8)[#54377](https://github.com/ray-project/ray/pull/54377)[)](https://github.com/ray-project/ray/commit/1de822fd76ebaf2fb93fe8e311f73af0157e4fc8)
- [Core] Fixed the bug where the head was unable to submit tasks after redis is turned on. ([#54267](https://github.com/ray-project/ray/pull/54267))
- [Core] [Azure] query for supported Microsoft.Network/virtualNetworks API versions instead of relying on resource_client.DEFAULT_API_VERSION ([#54874](https://github.com/ray-project/ray/pull/54874))
- [core] Fix possible race by checking node cache status instead of just subscription ([#54745](https://github.com/ray-project/ray/pull/54745))
- [core] Fix get actor timeout multiplier ([#54525](https://github.com/ray-project/ray/pull/54525))
- [core]: Use a temporary file to share default worker path in runtime env ([#53653](https://github.com/ray-project/ray/pull/53653))
- [core] Fix check fail when task buffer periodical runner runs before RayEvent is initialized ([#55249](https://github.com/ray-project/ray/pull/55249))
- [core] Patch grpc with RAY_num_grpc_threads to control grpc thread count ([#54988](https://github.com/ray-project/ray/pull/54988))
- [core][gpu-objects] Always write to GPUObjectStore to avoid _get_tensor_meta() from hanging indefinitely. ([#55433](https://github.com/ray-project/ray/pull/55433))
- [[Core] Core Worker GetObjStatus GRPC Fault Tolerance (](https://github.com/ray-project/ray/commit/ec0fa600e0e84bce7816a3b896e5d2436e8a415a)[#54567](https://github.com/ray-project/ray/pull/54567)[)](https://github.com/ray-project/ray/commit/ec0fa600e0e84bce7816a3b896e5d2436e8a415a) 

📖 Documentation:

- Added guide on using type hints with Ray Core. (#55013)

🏗 Architecture refactoring:

- Migrate metric collection from opencensus to opentelemetry (#53098, #53740)

## Dashboard
💫 Enhancements:

- Grafana: new Operator filter for Data; Prometheus adds `RayNodeType` label on for nodes. (#55493, #55192)

🔨 Fixes:

- Removed references to a deleted Data metrics panel. (#55478)

## Ray Images
🎉 New Features:

- Added cuda 12.6 based images (#55425, #55444)

💫 Enhancements:

- Upgraded protobuf to v4 (#54496)

## Docs
💫 Enhancements:

- KubeRay docs: added InteractiveMode quick-start details; expanded Core type-hints guidance; Serve LLM example coverage; Data LLM batching FAQ (#55570, #55284)

🔨 Fixes:

- Various formatting/mis-highlighting and lints across Train/Tune/Serve LLM docs. (#55284, #54763)


## Thanks!

Thank you to everyone who contributed to this release! 
@pavitrabhalla, @Daraan, @Sparks0219, @daiping8, @abrarsheikh, @sven1977, @Toshaksha, @bveeramani, @MengjinYan, @GokuMohandas, @codope, @nadongjun, @SolitaryThinker, @matthewdeng, @elliot-barn, @isimluk, @avibasnet31, @OneSizeFitsQuorum, @Future-Outlier, @marosset, @jackfrancis, @kshanmol, @eicherseiji, @dayshah, @iamjustinhsu, @Qiaolin-Yu, @goutamvenkat-anyscale, @Yicheng-Lu-llll, @yantarou, @rclough, @zcin, @NeilGirdhar, @VarunBhandary, @400Ping, @akshay-anyscale, @vickytsang, @xushiyan, @JasonLi1909, @n-elia, @simonsays1980, @dragongu, @Kishanthan, @ruisearch42, @jectpro7, @TimothySeah, @liulehui, @rueian, @HollowMan6, @akyang-anyscale, @axreldable, @czgdp1807, @alanwguo, @justinvyu, @ok-scale, @my-vegetable-has-exploded, @landscapepainter, @fscnick, @machichima, @mpashkovskii, @ZacAttack, @gvspraveen, @sword865, @lmsh7, @Ziy1-Tan, @rebel-scottlee, @sampan-s-nayak, @coqian, @can-anyscale, @Bye-legumes, @win5923, @MortalHappiness, @angelinalg, @khluu, @aslonnie, @krishnakalyan3, @minosvasilias, @x-tong, @xinyuangui2, @raulchen, @Yangruipis, @edoakes, @kevin85421, @wingkitlee0, @Fokko, @cristianjd, @srinathk10, @owenowenisme, @JoshKarpel, @MengqingCao, @leopardracer, @westonpace, @LeslieWongCV, @VassilisVassiliadis, @crypdick, @alexeykudinkin, @mjacar, @kunling-anyscale, @saihaj, @kouroshHakha, @ema-pe, @markjm, @avigyabb, @dshepelev15, @mauvilsa, @omatthew98, @nrghosh, @ryanaoleary, @Aydin-ab, @lk-chen, @stephanie-wang, @harshit-anyscale, @jjyao, @bullgom, @Yevet, @israbbani

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.49.0)

---

## ray-2.48.0: Ray-2.48.0
**Published:** 2025-07-18

# Release Highlights
- **Ray Data**: This release features a new Delta Lake and Unity Catalog integration and performance improvements to various reading/writing operators.
- **Ray Core**: Enhanced GPU object support with intra-process communication and improved Autoscaler v2 functionality
- **Ray Train**: Improved hardware metrics integration with Grafana and enhanced collective operations support
- **Ray Serve LLM**: This release features early proof of concept for prefill-decode disaggregation deployment and LLM-aware request routing such as prefix-cache aware routing. 
- **Ray Data LLM**: Improved throughput and CPU memory utilization for ray data workers. 

# Ray Libraries

## Ray Data

🎉 New Features:
- Add reading from Delta Lake tables and Unity Catalog integration (#53701)
- Enhanced pin_memory support in iter_torch_batches (#53792)
- Add pin_memory to iter_torch_batches (#53792)

💫 Enhancements:

- Re-enabled sorting in Ray Data tests with performance improvements (#54475)
- Enhanced handling of mismatched columns and pandas.NA values (#53861, #53859)
- Improved read_text trailing newline semantics (#53860)
- Optimized backpressure handling with policy-based resource management (#54376)
- Enhanced write_parquet with support for both partition_by and row limits (#53930)
- Prevent filename collisions on write operations (#53890)
- Improved execution performance for One Hot encoding in preprocessors (#54022)

🔨 Fixes:

- Fixing map_groups issues (#54462)
- Prevented Op fusion for streaming repartition to avoid performance degradation (#54469)
- Fixed ActorPool autoscaler scaling up logic (#53983)
- Resolved empty dataset repartitioning issues (#54107)
- Fixed PyArrow overflow handling in data processing (#53971, #54390)
- Fixed IcebergDatasink to properly generate individual file uuids (#52956)
- Avoid OOMs with read_json(..., lines=True) (#54436)
- Handle HuggingFace parquet dataset resolve URLs (#54146)
- Fixed BlockMetadata derivation for Read operator (#53908)

📖 Documentation:

- Updated AggregateFnV2 documentation to clarify finalize method (#53835)
- Improved preprocessor and vectorizer API documentation


## Ray Train

🎉 New Features:

- Added broadcast_from_rank_zero and barrier collective operations (#54066)
- Enhanced hardware metrics integration with Grafana dashboards (#53218)
- Added support for dynamically loading callbacks via environment variables (#54233)

💫 Enhancements:

- Improved checkpoint population from before_init_train_context (#54453)
- Enhanced controller state logging and metrics (#52805)
- Added structured logging environment variable support (#52952)
- Improved handling of Noop scaling decisions for smoother scaling logic (#53180)
- Logging of controller state transitions to aid in debugging and analysis (#53344)

🔨 Fixes:

- Fixed GPU tensor reporting in ray.train.report (#53725)
- Enhanced move_tensors_to_device utility for complex tensor structures (#53109)
- Improved worker health check error handling with trace information (#53626)
- Fixed GPU transfer support for non-contiguous tensors (#52548)
- Force abort on SIGINT spam and do not abort finished runs (#54188)

📖 Documentation:

- Updated beginner PyTorch example (#54124)
- Added documentation for ray.train.collective APIs (#54340)
- Added a note about PyTorch DataLoader's multiprocessing and forkserver usage (#52924)
- Fixed various docstring format and indentation issues (#52855, #52878)
- Added note that ray.train.report API docs should mention optional checkpoint_dir_name (#54391)

🏗 Architecture refactoring:

- Removed subclass relationship between RunConfig and RunConfigV1 (#54293)
- Enhanced error handling for finished training runs (#54188)
- Deduplicated ML doctest runners in CI for efficiency (#53157)
- Converted isort configuration to Ruff for consistency (#52869)


## Ray Tune

💫 Enhancements:

- Updated test_train_v2_integration to use the correct RunConfig (#52882)

🔨 Fixes:

- Fixed RayTaskError serialization logic (#54396)
- Improved experiment restore timeout handling (#53387)

📖 Documentation:

- Replaced session.report with tune.report and corrected import paths (#52801)
- Removed outdated graphics cards reference in docs (#52922)
- Fixed various docstring format issues (#52879)


## Ray Serve

🎉 New Features:

- Added RouterConfig field to DeploymentConfig for custom RequestRouter configuration (#53870)
- Added support for implementing custom request routing algorithms (#53251)

💫 Enhancements:

- Enhanced FastAPI ingress deployment validation for multiple deployments (#53647)
- Optimized get_live_deployments performance (#54454)
- Progress towards making ray.serve.llm compatible with vLLM serve frontend (#54481, #54443, #54440)

🔨 Fixes:

- Fixed deployment scheduler issues with component scheduling (#54479)
- Fixed runtime_env validation for py_modules (#53186)
- Added descriptive error message when deployment name is not found (#45181)

📖 Documentation:

- Added troubleshooting guide for DeepSeek/multi-node GPU deployment on KubeRay (#54229)
- Updated the guide on serving models with Triton Server in Ray Serve
- Added documentation for custom request routing algorithms
- Added custom request router docs (#53511)

🏗 Architecture refactoring:

- Remove indirection layers of node initialization (#54481)
- Incremental refactor of LLMEngine (#54443)
- Remove random v0 logic from serve endpoints (#54440)
- Remove usage of internal_api.memory_summary() (#54417)
- Remove usage of ray._private.state (#54140)

## Ray Serve/Data LLM
🎉 New Features

- Support separate deployment config for PDProxy in PrefixAwareReplicaSet (#53935)
- Support for prefix-aware request router (#52725)

💫 Enhancements

- Log engine stats after each batch task is done. (#54360)
- Decouple max_tasks_in_flight from max_concurrent_batches (#54362)
- Make llm serve endpoints compatible with vLLM serve frontend, including streaming, tool_code, and health check support (#54440)
- Remove botocore dependency in Ray Serve LLM (#54156)
- Update vLLM version to 0.9.2 (#54407)

🔨 Fixes

- Fix health check in prefill disagg (#53937)
- Fix doc to only support int concurrency (#54196)
- Fix vLLM batch test by changing to Pixtral (#53744)
- Fix pickle error with remote code models in vLLM Ray workloads (#53868)
- Adaption of the change of vllm.PoolingOutput (#54467)

📖 Documentation

- Ray serve/lora doc fix (#53553)
- Add Ray serve/LLM doc (#52832)
- Add a doc snippet to inform users about existing diffs between vLLM and Ray Serve LLM behavior in some APIs like streaming, tool_code, and health check (#54123)
- Troubleshooting DeepSeek/multi-node GPU deployment on KubeRay (#54229)

🏗 Architecture refactoring

- Make llm serve endpoints compatible with vLLM serve frontend, including streaming, tool_code, and health check support (#54490)
- Prefix-aware scheduler [2/N] Configure PrefixAwareReplicaSet to correctly handle the number of available GPUs for each worker and to ensure efficient GPU utilization in vLLM (#53192)
- Organize spread out utils.py (#53722)
- Remove ImageRetriever class and related tests from the LLM serving codebase. (#54018)
- Return a batch of rows in the udf instead of row by row (#54329)


## RLlib
🎉 New Features:

- Implemented Offline Policy Evaluation (OPE) via Importance Sampling (#53702)
- Enhanced ConnectorV2 ObservationPreprocessor APIs with multi-agent support (#54209)
- Add GPU inference to offline evaluation (#52718)

💫 Enhancements:

- Enhanced MetricsLogger to handle tensors in state management (#53514)
- Improved env seeding in EnvRunners with deterministic training example rewrite (#54039)
- Cleanup of meta learning classes and examples (#52680)

🔨 Fixes:

- Fixed EnvRunner restoration when no local EnvRunner is available (#54091)
- Fixed shapes in explained_variance for recurrent policies (#54005)
- Resolved device check issues in Learner implementation (#53706)
- Enhanced numerical stability in MeanStdFilter (#53484)
- Fixed weight synching in offline evaluation (#52757)
- Fixed bug in split_and_zero_pad utility function (#52818)

📖 Documentation:

- Do-over of examples for connector pipelines (#52604)
- Remove "new API stack" banner from all RLlib docs pages as it's now the default (#54282)

## Ray Core
🎉 New Features:

- Enhanced GPU object support with intra-process communication (#53798)
- Integrated single-controller collective APIs with GPU objects (#53720)
- Added support for ray.get on driver process for GPU objects (#53902)
- Supporting allreduce on list of input nodes in compiled graphs (#51047)
- Add single-controller API for ray.util.collective and torch gloo backend (#53319)

💫 Enhancements:

- Improved autoscaler v2 functionality with cloud instance ID reusing (#54397)
- Enhanced cluster task manager with better resource management (#54413)
- Upgraded OpenTelemetry SDK for better observability (#53745)
- Improved actor scheduling to prevent deadlocks in ordered actors (#54034)
- Enhanced get_max_resources_from_cluster_config functionality (#54455)
- Use std::move in cluster task manager constructor (#54413)
- Improve status messages and add comments about stale seq_no handling (#54470)
- uv run integration is now enabled by default (#53060)

🔨 Fixes:

- Fixed race conditions in object eviction and repinning for recovery (#53934)
- Resolved GCS crash issues on duplicate MarkJobFinished RPCs (#53951)
- Enhanced actor restart handling on node failures (#54088)
- Improved reference counting during worker graceful shutdown (#53002)
- Fix race condition when canceling task that hasn't started yet (#52703)
- Fix the issue where a valid RestartActor rpc is ignored (#53330)
- Fixed "Check failed: it->second.num_retries_left == -1" error (#54116)
- Fix detached actor being unexpectedly killed (#53562)

📖 Documentation:

- Enhanced troubleshooting guides and API documentation
- Updated reStructuredText formatting on Resources page (#53882)
- Fix working code snippets (#52748)
- Add doc for running KubeRay dashboard (#53830)
- Add antipattern for nested ray.get (#43184)

🏗 Architecture refactoring:

- Delete old skipped tests and unused code (#54427)
- Consolidate TaskManager interface (#54317)
- Move dependencies of NodeManager to main.cc for better testability (#53782)
- Use smart pointer in logging.cc (#54351)
- Delete event_label and unused environment variables (#54378, #54095)
- Remove actor task path in normal task submitter (#53996)
- Rename GcsFunctionManager and use fake in test (#53973)


## Dashboard

🎉 New Features:

- Add dynolog for on-demand GPU profiling for Torch training (#53191)

💫 Enhancements:

- Added TPU usage metrics to reporter agent (#53678)
- Enhanced GPU profiling manager IP address retrieval (#53807)
- Improved configurability of Grafana dashboard parameters (#53236)
- Add configurability of 'orgId' param for requesting Grafana dashboards (#53236)

🔨 Fixes:

- Fixed Grafana dashboard dropdowns for data and train dashboards (#52752)
- Resolved daylight savings time issues in dashboard (#52755)
- Fix retrieving IP address from the GPUProfilingManager on the dashboard agent (#53807)

Docs
🎉 New Features:

- New end-to-end examples:
  - Multi-modal AI pipeline (#52342)
  - Xgboost tutorial (#52383)
  - Audio transcription and LLM as judge curation (#53189)
  - LLM training and inference (#53415)
  - Scalable video processing (#50965)

💫 Enhancements:

- Add pydoclint to pre-commit (#52974)
- Add vale to pre-commit (#53564)

Breaking Changes

- Removed deprecated ray.workflow package (#53612)
- Removed deprecated storage parameter from ray.init (#53669)
- Removed deprecated ray start CLI options (#53675)
- Removed experimental "array" library (#54105)
- Remove dask from byod 3.9 deps (#54521)

Dependencies & Build

- Added uv binary v0.7.19 for improved package management (#54437)
- Upgraded datasets in release tests (#54425)
- Enhanced wheel building process with single bazel call optimization (#54476)
- Fixed uv run parser for handling extra arguments (#54488)
- Upgrade h11, requests, starlette, jinja2, pyopenssl and cryptography
- Generate multi-arch image indexes (#52816)


## Thanks!

Thank you to everyone who contributed to this release! 
@kouroshHakha, @davidwagnerkc, @MengjinYan, @minerharry, @simonsays1980, @Myasuka, @noemotiovon, @goutamvenkat-anyscale, @harshit-anyscale, @jugalshah291, @tianyi-ge, @sven1977, @crypdick, @JohnsonKuan, @lk-chen, @richardsliu, @alexeykudinkin, @EagleLo, @soffer-anyscale, @zcin, @AdrienVannson, @nilsmelchert, @raulchen, @jujipotle, @DrehanM, @vigneshka, @Ziy1-Tan, @Blaze-DSP, @ArthurBook, @GokuMohandas, @walkoss, @bveeramani, @edoakes, @omatthew98, @SeanQuant, @CheyuWu, @cszhu, @win5923, @kevin85421, @angelinalg, @iamjustinhsu, @eicherseiji, @kunling-anyscale, @vickytsang, @MortalHappiness, @aslonnie, @psr-ai, @sbhat98, @anyadontfly, @marwan116, @cristianjd, @2niuhe, @codope, @fscnick, @ryanaoleary, @srinathk10, @TimothySeah, @han-steve, @Future-Outlier, @Syulin7, @Qiaolin-Yu, @elliot-barn, @JoshKarpel, @dayshah, @can-anyscale, @ok-scale, @mattip, @SolitaryThinker, @owenowenisme, @nehiljain, @GeneDer, @rnkrtt, @israbbani, @DriverSong, @sinalallsite, @pcmoritz, @akyang-anyscale, @xinyuangui2, @nrghosh, @davidxia, @rueian, @stephanie-wang, @jjyao, @chris-ray-zhang, @czgdp1807, @justinvyu, @Daraan, @landscapepainter, @troychiu, @khluu, @hipudding, @ruisearch42, @robertnishihara, @ArturNiederfahrenhorst, @abrarsheikh, @alanwguo, @HollowMan6, @ran1995data, @matthewdeng

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.48.0)

---

## ray-2.47.1: Ray-2.47.1
**Published:** 2025-06-18

Ray 2.47.1 fixed an issue where Ray failed to start on Mac (https://github.com/ray-project/ray/pull/53807)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.47.1)

---

## ray-2.47.0: Ray-2.47.0
**Published:** 2025-06-12

# Release Highlights
* Prefill disaggregation is now supported in initial support in Ray Serve LLM (#53092). This is critical for production LLM serving use cases.
* Ray Data features a variety of performance improvements (locality-based scheduling, non-blocking execution) as well as improvements to observability, preprocessors, and other stability fixes.
* Ray Serve now features custom request routing algorithms, which is critical for high throughput traffic for large model use cases.


# Ray Libraries

## Ray Data

🎉 New Features:
- Add save modes support to file data sinks (#52900)
- Added flattening capability to the Concatenator preprocessor to support output vectorization use cases (#53378)


💫 Enhancements:
- Re-enable Actor locality-based scheduling. This PR also improves algorithms for ranking the locations for the bundle. (#52861)
- Disable blocking pipeline by default until Actor Pool fully scales up to min actors (#52754)
- Progress bar and dashboard improvements to show name of partial functions properly(#52280)

🔨 Fixes:
- Make Ray Data `from_torch` respect Dataset len (#52804)
- Fixing flaky aggregation test (#53383)
- Fix race condition bug in fault tolerance by disabling `on_exit` hook (#53249)
- Fix `move_tensors_to_device` utility for the list/tuple[tensor] case (#53109)
- Fix `ActorPool` scaling to avoid scaling down when the input queue is empty (#53009)
- Fix internal queues accounting for all Operators w/ an internal queue (#52806)
- Fix backpressure for `FileBasedDatasource`. This fixes potential OOMs for workloads using `FileBasedDatasources` (#52852)

📖 Documentation:
- Fix working code snippets (#52748)
- Improve AggregateFnV2 docstrings and examples (#52911)
- Improved documentation for vectorizers and API visibility in Data (#52456)


## Ray Train

🎉 New Features:
- Added support for configuring Ray Train worker actor runtime environments. (#52421)
- Included Grafana panel data in Ray Train export for improved monitoring. (#53072)
- Introduced a structured logging environment variable to standardize log formats. (#52952)
- Added metrics for `TrainControllerState` to enhance observability. (#52805)

💫 Enhancements:
- Logging of controller state transitions to aid in debugging and analysis. (#53344)
- Improved handling of `Noop` scaling decisions for smoother scaling logic. (#53180)

🔨 Fixes:
- Improved `move_tensors_to_device utility` to correctly handle `list` / `tuple` of tensors. (#53109)
- Fixed GPU transfer support for non-contiguous tensors. (#52548)
- Increased timeout in `test_torch_device_manager` to reduce flakiness. (#52917)

📖 Documentation:
- Added a note about PyTorch DataLoader’s multiprocessing and forkserver usage. (#52924)
- Fixed various docstring format and indentation issues. (#52855, #52878)
- Removed unused "configuration-overview" documentation page. (#52912)
- General typo corrections. (#53048)

🏗 Architecture refactoring:
- Deduplicated ML doctest runners in CI for efficiency. (#53157)
- Converted isort configuration to Ruff for consistency. (#52869)
- Removed unused `PARALLEL_CI` blocks and combined imports. (#53087, #52742)

## Ray Tune

💫 Enhancements:
- Updated `test_train_v2_integration` to use the correct `RunConfig`. (#52882)

📖 Documentation:
- Replaced `session.report` with `tune.report` and corrected import paths. (#52801)
- Removed outdated graphics cards reference in docs. (#52922)
- Fixed various docstring format issues. (#52879)

## Ray Serve

🎉 New Features:
- Added support for implementing custom request routing algorithms. (#53251)
- Introduced an environment variable to prioritize custom resources during deployment scheduling. (#51978)

💫 Enhancements:
- The ingress API now accepts a builder function in addition to an ASGI app object. (#52892)

🔨 Fixes:
- Fixed `runtime_env` validation for `py_modules`. (#53186)
- Disallowed special characters in Serve deployment and application names. (#52702)
- Added a descriptive error message when a deployment name is not found. (#45181)

📖 Documentation:
- Updated the guide on serving models with Triton Server in Ray Serve.
- Added documentation for custom request routing algorithms.

## Ray Serve/Data LLM

🎉 New Features:
- Added initial support for prefill decode disaggregation (#53092)
- Expose vLLM Metrics to `serve.llm` API (#52719)
- Embedding API (#52229)

💫 Enhancements:
- Allow setting `name_prefix` in `build_llm_deployment` (#53316)
- Minor bug fix for 53144: stop tokens cannot be null (#53288)
- Add missing `repetition_penalty` vLLM sampling parameter (#53222)
- Mitigate the serve.llm streaming overhead by properly batching stream chunks (#52766)
- Fix test_batch_vllm leaking resources by using larger `wait_for_min_actors_s`

🔨 Fixes:
- `LLMRouter.check_health()` should check `LLMServer.check_health()` (#53358)
- Fix runtime passthrough and auto-executor class selection (#53253)
- Update `check_health` return type (#53114)
- Bug fix for duplication of `<bos>` token (#52853)
- In stream batching, first part of the stream was always consumed and not streamed back from the router (#52848)

## RLlib

🎉 New Features:
- Add GPU inference to offline evaluation. (#52718)

💫 Enhancements:
- Do-over of examples for connector pipelines. (#52604)
- Cleanup of meta learning classes and examples. (#52680)

🔨 Fixes:
- Fixed weight synching in offline evaluation. (#52757)
- Fixed bug in `split_and_zero_pad` utility function (related to complex structures vs simple values or `np.arrays`). (#52818)

## Ray Core

💫 Enhancements:
- `uv run` integration is now enabled by default, so you don't need to set the `RAY_RUNTIME_ENV_HOOK` any more (#53060). If you rely on the previous behavior where `uv run` only runs the Ray driver but not the workers in the uv environment, you can switch back to the old behavior by setting the `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` environment variable.
- Record gcs process metrics (#53171)

🔨 Fixes:
- Improvements for using `RuntimeEnv` in the Job Submission API. (#52704)
- Close unused pipe file descriptor of child processes of Raylet (#52700)
- Fix race condition when canceling task that hasn't started yet (#52703)
- Implement a thread pool and call the CPython API on all threads within the same concurrency group (#52575)
- cgraph: Fix execution schedules with collective operations (#53007)
- cgraph: Fix scalar tensor serialization edge case with `serialize_to_numpy_or_scalar` (#53160)
- Fix the issue where a valid `RestartActor` rpc is ignored (#53330)
- Fix reference counter crashes during worker graceful shutdown (#53002)

## Dashboard

🎉 New Features:
- train: Add dynolog for on-demand GPU profiling for Torch training (#53191)

💫 Enhancements:
- Add configurability of 'orgId' param for requesting Grafana dashboards (#53236)

🔨 Fixes:
- Fix Grafana dashboards dropdowns for data and train dashboard (#52752)
- Fix dashboard for daylight savings (#52755)

## Ray Container Images

💫 Enhancements:
- Upgrade `h11` (#53361), `requests`, `starlette`, `jinja2` (#52951), `pyopenssl` and `cryptography` (#52941)
- Generate multi-arch image indexes (#52816)

##  Docs

🎉 New Features:
- End-to-end example: Entity recognition with LLMs (#52342) - new end-to-end example
- End-to-end example: xgboost tutorial (#52383)
- End-to-end tutorial for audio transcription and LLM as judge curation (#53189)

💫 Enhancements:
- Adds pydoclint to pre-commit (#52974)

## Thanks!

Thank you to everyone who contributed to this release! 

@NeilGirdhar, @ok-scale, @JiangJiaWei1103, @brandonscript, @eicherseiji, @ktyxx, @MichalPitr, @GeneDer, @rueian, @khluu, @bveeramani, @ArturNiederfahrenhorst, @c8ef, @lk-chen, @alanwguo, @simonsays1980, @codope, @ArthurBook, @kouroshHakha, @Yicheng-Lu-llll, @jujipotle, @aslonnie, @justinvyu, @machichima, @pcmoritz, @saihaj, @wingkitlee0, @omatthew98, @can-anyscale, @nadongjun, @chris-ray-zhang, @dizer-ti, @matthewdeng, @ryanaoleary, @janimo, @crypdick, @srinathk10, @cszhu, @TimothySeah, @iamjustinhsu, @mimiliaogo, @angelinalg, @gvspraveen, @kevin85421, @jjyao, @elliot-barn, @xingyu-long, @LeoLiao123, @thomasdesr, @ishaan-mehta, @noemotiovon, @hipudding, @davidxia, @omahs, @MengjinYan, @dengwxn, @MortalHappiness, @alhparsa, @emmanuel-ferdman, @alexeykudinkin, @KunWuLuan, @dev-goyal, @sven1977, @akyang-anyscale, @GokuMohandas, @raulchen, @abrarsheikh, @edoakes, @JoshKarpel, @bhmiller, @seanlaii, @ruisearch42, @dayshah, @Bye-legumes, @petern48, @richardliaw, @rclough, @israbbani, @jiwq

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.47.0)

---

## ray-2.46.0: Ray-2.46.0
**Published:** 2025-05-07

# Release Highlights
The 2.46 Ray release comes with a couple core highlights:

1. Ray Data now supports **hash shuffling for repartition and aggregations, along with support for joins**. This enables many new data processing workloads to be run on Ray Data. Please give it a try and let us know if you have any feedback!
2. Ray Serve LLM now **supports vLLM v1** to be forward-compatible with upcoming vLLM releases. This also opens up significant performance improvements that come with vLLM's v1 refactor.
3. There is a new **Train Grafana dashboard** which provides in-depth metrics on Grafana for better metrics on training workloads.

# Ray Libraries
## Ray Data
🎉 New Features:
- Adding support for hash-shuffle based repartitioning and aggregations (#52664)
- Added support for Joins (using hash-shuffle) (#52728)
- [LLM] vLLM support upgrades to 0.8.5 (#52344)

💫 Enhancements:
- Add memory attribute to ExecutionResources (#51127)
- Support ray_remote_args for read_tfrecords #52450
- [data.dashboard] Skip reporting internal metrics (#52666)
- Add PhysicalOperator.min_max_resource_usage_bounds (#52502)
- Speed up printing the schema (#52612)
- [data.dashboard] Dataset logger for worker (#52706) 
- Support new pyiceberg version (#51744) 
- Support num_cpus, memory, concurrency, batch_size for preprocess (#52574)

🔨 Fixes:
- Handle Arrow Array null types in to_numpy (#52572)
- Fix S3 serialization wrapper compatibility with RetryingPyFileSystem (#52568)
- Fixing Optimizer to apply rules until plan stabilize; (#52663)
- Fixing FuseOperators rule to properly handle the case of transformations drastically changing size of the dataset (#52570)

📖 Documentation:
- [LLM] Improve concurrency settings, improve prompt to achieve better throughput (#52634)

## Ray Train
🎉 New Features:
- Add initial Train Grafana dashboard (#52709)

💫 Enhancements:
- Lazily import torch FSDP for ray.train.torch module to improve performance and reduce unnecessary dependencies (#52707)
- Deserialize the user-defined training function directly on workers, improving efficiency (#52684)

🔨 Fixes:
- Fixed error when no arguments are passed into TorchTrainer (#52693)

📖 Documentation:
- Added new XGBoostTrainer user guide (#52355)

🏗 Architecture refactoring:
- Re-enabled isort for python/ray/train to maintain code formatting consistency (#52717)

## Ray Tune
📖 Documentation:
- Fixed typo in Ray Tune PyTorch Lightning docs (#52756)

## Ray Serve
💫 Enhancements:
- [LLM] Refactor LLMServer and LLMEngine to not diverge too much from vllm chat formatting logic ([#52597](https://github.com/ray-project/ray/pull/52597))
- Bump vllm from 0.8.2 to 0.8.5 in /python ([#52344](https://github.com/ray-project/ray/pull/52344))
- [LLM] Add router replicas and batch size to llm config ([#52655](https://github.com/ray-project/ray/pull/52655))

🔨 Fixes:
- Request cancellation not propagating correctly across deployments (#52591)
- BackpressureError not properly propagated in FastAPI ingress deployments (#52397)
- Hanging issue when awaiting deployment responses (#52561)
- [Serve.llm] made Ray Serve LLM compatible with vLLM v1 ([#52668](https://github.com/ray-project/ray/pull/52668))

📖 Documentation:
- [Serve][LLM] Add doc for deploying DeepSeek ([#52592](https://github.com/ray-project/ray/pull/52592))

## RLLib
🎉 New Features:
- Offline Evaluation with loss function for Offline RL pipeline. Introduces three new callbacks, `on_offline_evaluate_start`, `on_offline_evaluate_end`, `on_offline_eval_runners_recreated` (#52308)

💫 Enhancements:
- New `custom_data` attribute for `SingleAgentEpisode` and `MultiAgentEpisode` to store custom metrics. Deprecates `add|get_temporary_timestep_data()` (#52603)

# Ray Core
💫 Enhancements:
- Only get serialization context once for all .remote args (#52690)
- Add grpc server success and fail count metric (#52711)

🔨 Fixes:
- Fix open leak for plasma store memory (shm/fallback) by workers (#52622)
- Assure closing of unused pipe for dashboard subprocesses (#52678)
- Expand protection against dead processes in reporter agent (#52657)
- [cgraph] Separate metadata and data in cross-node shared memory transport (#52619)
- Fix JobID check for detached actor tasks (#52405)
- Fix potential log loss of tail_job_logs (#44709)

🏗 Architecture refactoring:
- Cancel tasks when an owner dies instead of checking if an owner is dead during scheduling (#52516)
- Unify GcsAioClient and GcsClient (#52735)
- Remove worker context dependency from the task receiver (#52740)

## Dashboard
🎉 New Features:
- Ray Train Grafana Dashboard added with a few built-in metrics. More to come.

## Thanks!
Thank you to everyone who contributed to this release! 
@kevin85421, @edoakes, @wingkitlee0, @alexeykudinkin, @chris-ray-zhang, @sophie0730, @zcin, @raulchen, @matthewdeng, @abrarsheikh, @popojk, @Jay-ju, @ruisearch42, @eicherseiji, @lk-chen, @justinvyu, @dayshah, @kouroshHakha, @NeilGirdhar, @omatthew98, @ishaan-mehta, @davidxia, @ArthurBook, @GeneDer, @srinathk10, @dependabot[bot], @JoshKarpel, @aslonnie, @khluu, @can-anyscale, @israbbani, @saihaj, @MortalHappiness, @alanwguo, @bveeramani, @iamjustinhsu, @Ziy1-Tan, @xingyu-long, @simonsays1980, @fscnick, @chuang0221, @sven1977, @jjyao

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.46.0)

---

## ray-2.45.0: Ray-2.45.0
**Published:** 2025-04-29

# Ray Core

💫 Enhancements
- Make Object Store Fallback Directory configurable ([#51189](https://github.com/ray-project/ray/pull/51189)).
- [cgraph] Support `with_tensor_transport(transport='shm')` ([#51872](https://github.com/ray-project/ray/pull/51872)).
- [cgraph] Support reduce scatter and all gather collective for GPU communicator in compiled graph ([#50624](https://github.com/ray-project/ray/pull/50624)).

🔨 Fixes
- Make sure `KillActor` RPC with `force_kill=True` can actually kill the threaded actor ([#51414](https://github.com/ray-project/ray/pull/51414)).
- [Autoscaler] Do not remove idle nodes for upcoming placement groups ([#51122](https://github.com/ray-project/ray/pull/51122)).
- Threaded actors get stuck forever if they receive two exit signals ([#51582](https://github.com/ray-project/ray/pull/51582)).
- [cgraph] Fix illegal memory access of cgraph when used in PP ([#51734](https://github.com/ray-project/ray/pull/51734)).
- Avoid resubmitted actor tasks from hanging indefinitely ([#51904](https://github.com/ray-project/ray/pull/51904)).
- Fix interleaved placement group creation process due to node failure ([#52202](https://github.com/ray-project/ray/pull/52202)).
- Flush task events in `CoreWorker::Shutdown` instead of `CoreWorker::Disconnect` ([#52374](https://github.com/ray-project/ray/pull/52374)).

🏗 Architecture refactoring
- Split dashboard single process into multiple processes to improve stability and avoid interference between different heads ([#51282](https://github.com/ray-project/ray/pull/51282), [#51489](https://github.com/ray-project/ray/pull/51489), [#51555](https://github.com/ray-project/ray/pull/51555), [#51507](https://github.com/ray-project/ray/pull/51507), [#51587](https://github.com/ray-project/ray/pull/51587), [#51553](https://github.com/ray-project/ray/pull/51553), [#51676](https://github.com/ray-project/ray/pull/51676), [#51733](https://github.com/ray-project/ray/pull/51733), [#51809](https://github.com/ray-project/ray/pull/51809), [#51877](https://github.com/ray-project/ray/pull/51877), [#51876](https://github.com/ray-project/ray/pull/51876), [#51980](https://github.com/ray-project/ray/pull/51980), [#52114](https://github.com/ray-project/ray/pull/52114)).

# Ray Libraries

## Ray Data

🎉 New Features
- **New ClickHouse sink** via `Dataset.write_clickhouse()` ([#50377](https://github.com/ray-project/ray/pull/50377))
- Support `ray_remote_args_fn` in `Dataset.groupby().map_groups()` to set per-group runtime env and resource hints ([#51236](https://github.com/ray-project/ray/pull/51236))
- Expose `Dataset.name` / `set_name` as a public API for easier lineage tracking ([#51076](https://github.com/ray-project/ray/pull/51076))
- Allow async callable classes in `Dataset.flat_map()` ([#51180](https://github.com/ray-project/ray/pull/51180))
- Introduce **Ruleset** abstraction for rule-based query optimisation ([#51558](https://github.com/ray-project/ray/pull/51558))
- Add seamless conversion from Daft DataFrame to Ray Dataset ([#51531](https://github.com/ray-project/ray/pull/51531))
- Improved support for line-delimited JSONL reading in `read_json()` ([#52083](https://github.com/ray-project/ray/pull/52083))
- Provide `Dataset.export_metadata()` for schema & stats snapshots ([#52227](https://github.com/ray-project/ray/pull/52227))

💫 Enhancements
- **Improved performance of sorting and sort-shuffle based operations** (by more than 5x in benchmarks) ([#51943](https://github.com/ray-project/ray/pull/51943/files))
- Metrics: number of map-actor workers **alive / pending / restarting** ([#51082](https://github.com/ray-project/ray/pull/51082))
- Continuous memory-usage polling per map task ([#51324](https://github.com/ray-project/ray/pull/51324))
- Auto-tune map-task memory based on output size ([#51536](https://github.com/ray-project/ray/pull/51536))
- More informative back-pressure progress bar ([#51697](https://github.com/ray-project/ray/pull/51697))
- Faster `RefBundle.get_cached_location()` lookup ([#52097](https://github.com/ray-project/ray/pull/52097))
- Speed-up for `PandasBlock.size_bytes()` ([#52510](https://github.com/ray-project/ray/pull/52510))
- Expanded `BlockColumnAccessor` utilities and ops ([#51326](https://github.com/ray-project/ray/pull/51326), [#51571](https://github.com/ray-project/ray/pull/51571))

🔨 Fixes
- Correct `MapTransformFn.__eq__` equality check ([#51434](https://github.com/ray-project/ray/pull/51434))
- Persist unresolved wildcard paths in `FileBasedDataSource` ([#51424](https://github.com/ray-project/ray/pull/51424))
- Repair Hugging Face dynamic-module loading on workers ([#51488](https://github.com/ray-project/ray/pull/51488))
- Prevent HTTP URLs from being expanded by `_expand_paths` ([#50178](https://github.com/ray-project/ray/pull/50178))
- Fix Databricks host-URL parsing in Delta datasource ([#49926](https://github.com/ray-project/ray/pull/49926))
- Restore reproducibility of `Dataset.random_sample()` ([#51401](https://github.com/ray-project/ray/pull/51401))
- Correct `RandomAccessDataset.multiget()` return values ([#51421](https://github.com/ray-project/ray/pull/51421))
- Ensure executor shutdown after schema fetch to avoid leaked actors ([#52379](https://github.com/ray-project/ray/pull/52379))
- Repair streaming shutdown regression ([#52509](https://github.com/ray-project/ray/pull/52509))
- Honour minimum resource reservation in `ResourceManager` ([#52226](https://github.com/ray-project/ray/pull/52226))

📖 Documentation
- Clarified shuffle-section wording ([#51289](https://github.com/ray-project/ray/pull/51289))
- Documented concurrency semantics in API reference ([#51963](https://github.com/ray-project/ray/pull/51963))
- Updated Ray Data guides for the 2.45 release ([#52082](https://github.com/ray-project/ray/pull/52082))

## Ray Train
🎉 New Features
- Fold `v2.LightGBMTrainer` API into the public trainer class as an alternate constructor ([#51265](https://github.com/ray-project/ray/pull/51265)).

💫 Enhancements
- Use the user-defined function name as the training thread name ([#52514](https://github.com/ray-project/ray/pull/52514)).
- Upgrade LightGBM to version 4.6.0 ([#52410](https://github.com/ray-project/ray/pull/52410)).
- Adjust test size further for better results ([#52283](https://github.com/ray-project/ray/pull/52283)).
- Log errors raised by workers during training ([#52223](https://github.com/ray-project/ray/pull/52223)).
- Add worker group setup finished log to track progress ([#52120](https://github.com/ray-project/ray/pull/52120)).
- Change `test_telemetry` to medium size ([#52178](https://github.com/ray-project/ray/pull/52178)).
- Improve dataset name observability for better tracking ([#52059](https://github.com/ray-project/ray/pull/52059)).
- Differentiate between train v1 and v2 export data for clarity ([#51728](https://github.com/ray-project/ray/pull/51728)).
- Include scheduling status detail to improve debugging ([#51480](https://github.com/ray-project/ray/pull/51480)).
- Move train library usage check to `Trainer` initialization ([#50966](https://github.com/ray-project/ray/pull/50966)).

🔨 Fixes
- Separate `OutputSplitter._locality_hints` from `actor_locality_enabled` and `locality_with_output` ([#52005](https://github.com/ray-project/ray/pull/52005)).
- Fix print redirection to handle new lines correctly ([#51542](https://github.com/ray-project/ray/pull/51542)).
- Mark `RunAttempt` workers as dead after completion to avoid stale states ([#51540](https://github.com/ray-project/ray/pull/51540)).
- Fix `setup_wandb` `rank_zero_only` logic ([#52381](https://github.com/ray-project/ray/pull/52381)).

📖 Documentation
- Add links to the Train v2 migration guide in the Train API pages ([#51924](https://github.com/ray-project/ray/pull/51924)).

🏗 Architecture refactoring
- Replace AMD device environment variable with `HIP_VISIBLE_DEVICES` ([#51104](https://github.com/ray-project/ray/pull/51104)).
- Remove unnecessary string literal splits ([#47360](https://github.com/ray-project/ray/pull/47360)).

## Ray Tune

📖 Documentation
- Improve Tune documentation structure ([#51684](https://github.com/ray-project/ray/pull/51684)).
- Fix syntax errors in Ray Tune example `pbt_ppo_example.ipynb` ([#51626](https://github.com/ray-project/ray/pull/51626)).

## Ray Serve

🎉 New Features
- Add request timeout sec for gRPC ([#52276](https://github.com/ray-project/ray/pull/52276)).
- [Serve.llm] `ray.llm` support custom accelerators ([#51359](https://github.com/ray-project/ray/pull/51359)).

💫 Enhancements
- Improve Serve deploy ignore behavior ([#49336](https://github.com/ray-project/ray/pull/49336)).
- [Serve.llm] Telemetry GPU type fallback to cluster hardware when unspecified ([#52003](https://github.com/ray-project/ray/pull/52003)).

🔨 Fixes
- Fix multiplex fallback logic during burst requests ([#51389](https://github.com/ray-project/ray/pull/51389)).
- Don't stop retrying replicas when a deployment is scaling back up from zero ([#51600](https://github.com/ray-project/ray/pull/51600)).
- Remove `RAY_SERVE_ENABLE_QUEUE_LENGTH_CACHE` flag ([#51649](https://github.com/ray-project/ray/pull/51649)).
- Remove `RAY_SERVE_EAGERLY_START_REPLACEMENT_REPLICAS` flag ([#51722](https://github.com/ray-project/ray/pull/51722)).
- Unify request cancellation errors ([#51768](https://github.com/ray-project/ray/pull/51768)).
- Catch timeout error when checking if proxy is dead ([#52002](https://github.com/ray-project/ray/pull/52002)).
- Suppress cancelled errors in proxy ([#52423](https://github.com/ray-project/ray/pull/52423)).
- [Serve.llm] Fix loading model from remote storage and add docs ([#51617](https://github.com/ray-project/ray/pull/51617)).
- [Serve.llm] Fix `ServeReplica` deployment failure for DeepSeek ([#51989](https://github.com/ray-project/ray/pull/51989)).
- [Serve.llm] Check `GPUType` enum value rather than enum itself ([#52037](https://github.com/ray-project/ray/pull/52037)).
- [Serve.llm] Fix inconsistent v0/v1 config passed to vLLM ([#52185](https://github.com/ray-project/ray/pull/52185)).
- [Serve.llm] Fix V0/V1 auto-resolution more broadly ([#52357](https://github.com/ray-project/ray/pull/52357)).
- [Serve.llm] Fix getting prompt limit for input-too-long error ([#52315](https://github.com/ray-project/ray/pull/52315)).

📖 Documentation
- Update deployment status docs ([#51610](https://github.com/ray-project/ray/pull/51610)).
- Update engine kwargs in LoRA instructions ([#52203](https://github.com/ray-project/ray/pull/52203)).
- Remove outdated docs about request retries ([#52438](https://github.com/ray-project/ray/pull/52438)).
- [Serve.llm] Add gen config related doc ([#51572](https://github.com/ray-project/ray/pull/51572)).
- [Serve.llm] Fix loading model from remote storage and add docs ([#51617](https://github.com/ray-project/ray/pull/51617)).
- [Serve.llm] Update Ray Serve LLM docs ([#51867](https://github.com/ray-project/ray/pull/51867)).
- [Serve.llm] Update model_source doc ([#52054](https://github.com/ray-project/ray/pull/52054)).
- [Serve.llm] Add doc example on using `tokenizer_pool_size` ([#52356](https://github.com/ray-project/ray/pull/52356)).
- [Serve.llm] Add API links ([#52384](https://github.com/ray-project/ray/pull/52384)).

## RLlib

🎉 New Features
- Meta-learning, differentiable learner ([#51859](https://github.com/ray-project/ray/pull/51859), [#52494](https://github.com/ray-project/ray/pull/52494)).

💫 Enhancements
- Offline RL:
  - Add configuration parameter `ignore_final_observation` and automatic episode ID generation ([#52107](https://github.com/ray-project/ray/pull/52107)).
  - BC performance improvements and adjustments to new `Learner.update` logic ([#51425](https://github.com/ray-project/ray/pull/51425)).
- Better handling of tensors in `MetricsLogger` and `Stats` ([#52334](https://github.com/ray-project/ray/pull/52334)).
- Make min/max env steps per evaluation sample call configurable for `duration="auto"` ([#51637](https://github.com/ray-project/ray/pull/51637)).
- Access to all episode chunks in EnvRunner callback `on_episode_end` ([#52252](https://github.com/ray-project/ray/pull/52252)).

🔨 Fixes
- Various smaller fixes ([#52503](https://github.com/ray-project/ray/pull/52503), [#51371](https://github.com/ray-project/ray/pull/51371), [#51816](https://github.com/ray-project/ray/pull/51816), [#51621](https://github.com/ray-project/ray/pull/51621), [#44700](https://github.com/ray-project/ray/pull/44700), [#51522](https://github.com/ray-project/ray/pull/51522), [#51406](https://github.com/ray-project/ray/pull/51406)).

🏗 Architecture refactoring
- Remove local `EnvRunner` (optional) ([#52091](https://github.com/ray-project/ray/pull/52091)).

## Dashboard

💫 Enhancements
- [Core/Dashboard] Add CPU and memory metrics for subprocesses ([#52208](https://github.com/ray-project/ray/pull/52208)).

🔨 Fixes
- [Observability] Fix metrics in the core Grafana dashboard to sum by labels ([#51894](https://github.com/ray-project/ray/pull/51894)).
- [Dashboard] Remove millisecond padding in `DurationText` component ([#52395](https://github.com/ray-project/ray/pull/52395)).
- [Core][Dashboard-Agent] Surface the root cause of job submission failure when the dashboard agent fails to launch HTTP server ([#51869](https://github.com/ray-project/ray/pull/51869)).

----
Many thanks to all those who contributed to this release!
@samhallam-reverb, @vickytsang, @anyadontfly, @zhaoch23, @bryant1410, @khluu, @akyang-anyscale, @angelinalg, @RocketRider, @wingkitlee0, @robertnishihara, @liuxsh9, @KepingYan, @SumanthRH, @emmanuel-ferdman, @ashwinsnambiar, @ArturNiederfahrenhorst, @KPCOFGS, @Bye-legumes, @400Ping, @dayshah, @aslonnie, @justinvyu, @rugggg, @zhiqiwangebay, @comaniac, @thusoy, @JDarDagran, @chuang0221, @davidxia, @tnixon, @israbbani, @win5923, @leibovitzgil, @simonsays1980, @machichima, @VamshikShetty, @zcin, @lk-chen, @abrarsheikh, @edoakes, @alexeykudinkin, @ruisearch42, @tespent, @jecsand838, @sijieamoy, @can-anyscale, @JonDum, @jyakaranda, @nadongjun, @d-miketa, @MortalHappiness, @kevin85421, @Ziy1-Tan, @matthewdeng, @crypdick, @hongpeng-guo, @richardliaw, @Qiaolin-Yu, @bhmiller, @soffer-anyscale, @kenchung285, @nishi-t, @Drice1999, @ryanaoleary, @chris-ray-zhang, @MengjinYan, @saihaj, @jjyao, @jaganmolleti7, @iamjustinhsu, @fscnick, @pcmoritz, @Jay-ju, @westonpace, @han-steve, @GeneDer, @denadai2, @thomasdesr, @jaychia, @raulchen, @omatthew98, @srinathk10, @alanwguo, @rueian, @akshay-anyscale, @bveeramani, @dentiny, @dhakshin32, @kouroshHakha, @sven1977

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.45.0)

---

## ray-2.44.1: Ray-2.44.1
**Published:** 2025-03-27

There is no difference between 2.44.1 and 2.44.0, though we needed a patch version for other out of band reasons. To fill the awkward blankness, here is a haiku about Ray:

Under screen-lit skies
A ray of bliss in each patch
Joy at any scale


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.44.1)

---

## ray-2.44.0: Ray-2.44.0
**Published:** 2025-03-21

# Release Highlights

- This release features Ray Compiled Graph (beta). Ray Compiled Graph gives you a classic Ray Core-like API, but with (1) less than 50us system overhead for workloads that repeatedly execute the same task graph; and (2) native support for GPU-GPU communication via NCCL. Ray Compiled Graph APIs simplify high-performance multi-GPU workloads such as LLM inference and training. The beta release refines the API, enhances stability, and adds or improves features like visualization, profiling and experimental GPU compute/computation overlap. For more information, refer to Ray documentation: https://docs.ray.io/en/latest/ray-core/compiled-graph/ray-compiled-graph.html
- The experimental Ray Workflows library has been deprecated and will be removed in a future version of Ray. Ray Workflows has been marked experimental since its inception and hasn’t been maintained due to the Ray team focusing on other priorities. If you are using Ray Workflows, we recommend pinning your Ray version to 2.44.

# Ray Libraries

## Ray Data

🎉 New Features:
- Add Iceberg write support through pyiceberg[ (](https://github.com/ray-project/ray/commit/5e26c7fc3866921ce97db876136e04271dabf8b4)[#50590](https://github.com/ray-project/ray/pull/50590)[)](https://github.com/ray-project/ray/commit/5e26c7fc3866921ce97db876136e04271dabf8b4) 
- [LLM] Various feature enhancements to Ray Data LLM, including LoRA support #50804 and structured outputs #50901

💫 Enhancements:
- Add dataset/operator state, progress, total metrics ([#50770](https://github.com/ray-project/ray/pull/50770))
- Make chunk combination threshold configurable ([#51200](https://github.com/ray-project/ray/pull/51200))
- Store average memory use per task in OpRuntimeMetrics ([#51126](https://github.com/ray-project/ray/pull/51126))
- Avoid unnecessary conversion to Numpy when creating Arrow/Pandas blocks ([#51238](https://github.com/ray-project/ray/pull/51238))
- Append-mode API for preprocessors -- #50848, #50847, #50642, #50856, #50584. Note that vectorizers and hashers now output a single column instead 1 column per feature. In the near future, we will be graduating preprocessors to *beta*.

🔨 Fixes:
- Fixing Map Operators to avoid unconditionally overriding generator's back-pressure configuration ([#50900](https://github.com/ray-project/ray/pull/50900))
- Fix filter expr equating negative numbers ([#50932](https://github.com/ray-project/ray/pull/50932))
- Fix error message for `override_num_blocks` when reading from a HuggingFace Dataset  ([#50998](https://github.com/ray-project/ray/pull/50998))
- Make num_blocks in repartition optional ([#50997](https://github.com/ray-project/ray/pull/50997))
- Always pin the seed when doing file-based random shuffle ([#50924](https://github.com/ray-project/ray/pull/50924))
- Fix `StandardScaler` to handle `NaN` stats ([#51281](https://github.com/ray-project/ray/pull/51281))

## Ray Train

🎉 New Features:
- Implement state export API (#50622, #51085, #51177)

💫 Enhancements:
- Folded v2.XGBoostTrainer API into the public trainer class as an alternate constructor (#50045)
- Created a default ScalingConfig if one is not provided to the trainer (#51093)
- Improved TrainingFailedError message (#51199)
- Utilize FailurePolicy factory (#51067)

🔨 Fixes:
- Fixed trainer import deserialization when captured within a Ray task (#50862)
- Fixed serialize import test for Python 3.12 (#50963)
- Fixed RunConfig deprecation message in Tune being emitted in trainer.fit usage (#51198)

📖 Documentation: 
- [Train V2] Updated API references (#51222)
- [Train V2] Updated persistent storage guide (#51202)
- [Train V2] Updated user guides for metrics, checkpoints, results, and experiment tracking (#51204)
- [Train V2] Added updated Train + Tune user guide (#51048)
- [Train V2] Added updated fault tolerance user guide (#51083)
- Improved HF Transformers example (#50896)
- Improved Train DeepSpeed example (#50906)
- Use correct mean and standard deviation norm values in image tutorials (#50240)

🏗 Architecture refactoring:
- Deprecated Torch AMP wrapper utilities (#51066)
- Hid private functions of train context to avoid abuse (#50874)
- Removed ray storage dependency and deprecated RAY_STORAGE env var configuration option (#50872)
- Moved library usage tests out of core (#51161)

## Ray Tune

📖 Documentation:
- Various improvements to Tune Pytorch CIFAR tutorial (#50316)
- Various improvements to the Ray Tune XGBoost tutorial (#50455)
- Various enhancements to Tune Keras example (#50581)
- Minor improvements to Hyperopt tutorial (#50697)
- Various improvements to LightGBM tutorial (#50704)
- Fixed non-runnable Optuna tutorial (#50404)
- Added documentation for Asynchronous HyperBand Example in Tune (#50708)
- Replaced reuse actors example with a fuller demonstration (#51234)
- Fixed broken PB2/RLlib example (#51219)
- Fixed typo and standardized equations across the two APIs (#51114)
- Improved PBT example (#50870)
- Removed broken links in documentation (#50995, #50996)

🏗 Architecture refactoring:
- Removed ray storage dependency and deprecated RAY_STORAGE env var configuration option (#50872)
- Moved library usage tests out of core (#51161)

## Ray Serve

🎉 New Features:
- Faster bulk imperative Serve Application deploys ([#49168](https://github.com/ray-project/ray/pull/49168))
- [LLM] Add gen-config ([#51235](https://github.com/ray-project/ray/pull/51235))

💫 Enhancements:
- Clean up shutdown behavior of serve ([#51009](https://github.com/ray-project/ray/pull/51009))
- Add `additional_log_standard_attrs` to serve logging config ([#51144](https://github.com/ray-project/ray/pull/51144))
- [LLM] remove `asyncache` and `cachetools` from dependencies ([#50806](https://github.com/ray-project/ray/pull/50806))
- [LLM] remove `backoff` dependency ([#50822](https://github.com/ray-project/ray/pull/50822))
- [LLM] Remove `asyncio_timeout` from `ray[llm]` deps on python<3.11 ([#50815](https://github.com/ray-project/ray/pull/50815))
- [LLM] Made JSON validator a singleton and `jsonref` packages lazy imported ([#50821](https://github.com/ray-project/ray/pull/50821))
- [LLM] Reuse `AutoscalingConfig` and `DeploymentConfig` from Serve ([#50871](https://github.com/ray-project/ray/pull/50871))
- [LLM] Use `pyarrow` FS for cloud remote storage interaction ([#50820](https://github.com/ray-project/ray/pull/50820))
- [LLM] Add usage telemetry for `serve.llm` ([#51221](https://github.com/ray-project/ray/pull/51221))

🔨 Fixes:
- Exclude redirects from request error count ([#51130](https://github.com/ray-project/ray/pull/51130))
- [LLM] Fix the wrong `device_capability` issue in vllm on quantized models ([#51007](https://github.com/ray-project/ray/pull/51007))
- [LLM] add `gen-config` related data file to the package ([#51347](https://github.com/ray-project/ray/pull/51347))

📖 Documentation:
- [LLM] Fix quickstart serve LLM docs ([#50910](https://github.com/ray-project/ray/pull/50910))
- [LLM] update `build_openai_app` to include yaml example ([#51283](https://github.com/ray-project/ray/pull/51283))
- [LLM] remove old vllm+serve doc ([#51311](https://github.com/ray-project/ray/pull/51311))

## RLlib

💫 Enhancements:
- APPO/IMPALA accelerate:
  - `LearnerGroup` should not pickle remote functions on each update-call; Refactor `LearnerGroup` and `Learner` APIs. ([#50665](https://github.com/ray-project/ray/pull/50665))
  - `EnvRunner` sync enhancements. ([#50918](https://github.com/ray-project/ray/pull/50918)[)](https://github.com/ray-project/ray/commit/02d4a3a51127f8470f9f422cc7f58dce73a6f520)
  - Various other speedups: [#51302](https://github.com/ray-project/ray/pull/51302), [#50923](https://github.com/ray-project/ray/pull/50923), [#50919](https://github.com/ray-project/ray/pull/50919), [#50791](https://github.com/ray-project/ray/pull/50791)
- Unify namings for actor managers' outstanding in-flight requests metrics. ([#51159](https://github.com/ray-project/ray/pull/51159))
- Add timers to env step, forward pass, and complete connector pipelines runs. ([#51160](https://github.com/ray-project/ray/pull/51160))

🔨 Fixes:
- Multi-agent env vectorization:
  - Fix `MultiAgentEnvRunner` env check bug. ([#50891](https://github.com/ray-project/ray/pull/50891)[)](https://github.com/ray-project/ray/commit/f4ab3439d4eb734f69fd8cc13b3d74d0e724864b)
  - Add `single_action_space` and `single_observation_space` to `VectorMultiAgentEnv`. ([#51096](https://github.com/ray-project/ray/pull/51096))
- Other fixes: [#51255](https://github.com/ray-project/ray/pull/51255), [#50920](https://github.com/ray-project/ray/pull/50920), [#51369](https://github.com/ray-project/ray/pull/51369)

📖 Documentation:
- Smaller fixes: [#51015](https://github.com/ray-project/ray/pull/51015), [#51219](https://github.com/ray-project/ray/pull/51219)

# Ray Core and Ray Clusters

## Ray Core

🎉 New Features:
- Enhanced `uv` support ([#51233](https://github.com/ray-project/ray/pull/51233))

💫 Enhancements:
- Made infeasible task errors much more obvious ([#45909](https://github.com/ray-project/ray/issues/45909))
- Log rotation for workers, runtime env agent, and dashboard agent ([#50759](https://github.com/ray-project/ray/pull/50759), [#50877](https://github.com/ray-project/ray/pull/50877), [#50909](https://github.com/ray-project/ray/pull/50909))
- Support customizing gloo timeout ([#50223](https://github.com/ray-project/ray/pull/50223))
- Support torch profiling in Compiled Graph ([#51022](https://github.com/ray-project/ray/pull/51022))
- Change default tensor deserialization in Compiled Graph ([#50778](https://github.com/ray-project/ray/pull/50778))
- Use current node id if no node is specified on ray drain-node ([#51134](https://github.com/ray-project/ray/pull/51134))

🔨 Fixes:
- Fixed an issue where the raylet continued to have high CPU overhead after a job was terminated ([#49999](https://github.com/ray-project/ray/issues/49999)).
- Fixed compiled graph buffer release issues ([#50434](https://github.com/ray-project/ray/pull/50434))
- Improved logic for `ray.wait` on object store objects ([#50680](https://github.com/ray-project/ray/pull/50680))
- Ray metrics performing validation the same validation as Prometheus for invalid names ([#40586](https://github.com/ray-project/ray/issues/40586))
- Make executor a long-running Python thread ([#51016](https://github.com/ray-project/ray/pull/51016))
- Fix plasma client memory leak ([#51051](https://github.com/ray-project/ray/pull/51051))
- Fix using `ray.actor.exit_actor()` from within an async background thread ([#49451](https://github.com/ray-project/ray/issues/49451))
- Fix UV hook to support Ray Job submission ([#51150](https://github.com/ray-project/ray/pull/51150))
- Fix resource leakage after ray job is finished ([#49999](https://github.com/ray-project/ray/issues/49999))
- Use the correct way to check whether an actor task is running ([#51158](https://github.com/ray-project/ray/pull/51158))
- Controllably destroy CUDA events in GPUFuture’s (Compiled Graph) ([#51090](https://github.com/ray-project/ray/pull/51090))
- Avoid creating a thread pool with 0 threads ([#50837](https://github.com/ray-project/ray/pull/50837))
- Fix the logic to calculate the number of workers based on the TPU version ([#51227](https://github.com/ray-project/ray/pull/51227))

📖 Documentation:
- Updated error message and anti-pattern when forking new processes in worker processes ([#50705](https://github.com/ray-project/ray/pull/50705))
- Compiled Graph API Documentation ([#50754](https://github.com/ray-project/ray/pull/50754))
- Doc for nsight and torch profile for Compiled Graph ([#51037](https://github.com/ray-project/ray/pull/51037))
- Compiled Graph Troubleshooting Doc ([#51030](https://github.com/ray-project/ray/pull/51030))
- Completion of of Compiled Graph Docs ([#51206](https://github.com/ray-project/ray/pull/51206))
- Updated `jemalloc` profiling doc ([#51031](https://github.com/ray-project/ray/pull/51031))
- Add information about standard Python logger attributes ([#51038](https://github.com/ray-project/ray/pull/51038))
- Add description for named placement groups to require a namespace ([#51285](https://github.com/ray-project/ray/pull/51285))
- Deprecation warnings for Ray Workflows and cluster-wide storage ([#51309](https://github.com/ray-project/ray/pull/51309))

## Ray Clusters

🎉 New Features:
- Add cuda 12.8 images ([#51210](https://github.com/ray-project/ray/pull/51210))

💫 Enhancements:
- Add Pod names to the output of `ray status -v` ([#51192](https://github.com/ray-project/ray/pull/51192))

🔨 Fixes:
- Fix autoscaler v1 crash from infeasible strict spread placement groups ([#39691](https://github.com/ray-project/ray/issues/39691))

🏗 Architecture refactoring:
- Refactor autoscaler v2 log formatting ([#49350](https://github.com/ray-project/ray/pull/49350))
- Update yaml example for `CoordinatorSenderNodeProvider` ([#51292](https://github.com/ray-project/ray/pull/51292))

## Dashboard

🎉 New Features:
- Discover TPU logs on the Ray Dashboard ([#47737](https://github.com/ray-project/ray/pull/47737))

🔨 Fixes:
- Return the correct error message when trying to kill non-existent actors ([#51341](https://github.com/ray-project/ray/pull/51341))

----
Many thanks to all those who contributed to this release!
@crypdick, @rueian, @justinvyu, @MortalHappiness, @CheyuWu, @GeneDer, @dayshah, @lk-chen, @matthewdeng, @co63oc, @win5923, @sven1977, @akshay-anyscale, @ShaochenYu-YW, @gvspraveen, @bveeramani, @jakac, @VamshikShetty, @raulchen, @PaulFenton, @elimelt, @comaniac, @qinyiyan, @ruisearch42, @nadongjun, @AndyUB, @israbbani, @hongpeng-guo, @laysfire, @alexeykudinkin, @Drice1999, @harborn, @scottsun94, @abrarsheikh, @martinbomio, @MengjinYan, @HollowMan6, @orcahmlee, @kenchung285, @csy1204, @noemotiovon, @jujipotle, @davidxia, @kevin85421, @hcc429, @edoakes, @kouroshHakha, @omatthew98, @alanwguo, @farridav, @aslonnie, @simonsays1980, @pcmoritz, @terraflops1048576, @JoshKarpel, @SumanthRH, @sijieamoy, @zcin, @can-anyscale, @akyang-anyscale, @angelinalg, @saihaj, @jjyao, @anmscale, @ryanaoleary, @dentiny, @jimmyxie-figma, @stephanie-wang, @khluu, @maofagui

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.44.0)

---

## ray-2.43.0: Ray-2.43.0
**Published:** 2025-02-27

# Highlights
- This release features new modules in Ray Serve and Ray Data for integration with large language models, marking the first step of addressing [#50639](https://github.com/ray-project/ray/issues/50639). Existing Ray Data and Ray Serve have limited support for LLM deployments, where users have to manually configure and manage the underlying LLM engine. In this release, we offer APIs for both batch inference and serving of LLMs within Ray in `ray.data.llm` and `ray.serve.llm`. See the below notes for more details. These APIs are marked as **alpha** -- meaning they may change in future releases without a deprecation period.
- Ray Train V2 is available to try starting in Ray 2.43! Run your next Ray Train job with the `RAY_TRAIN_V2_ENABLED=1` environment variable. See [the migration guide](https://github.com/ray-project/ray/issues/49454) for more information.
- A new integration with `uv run` that allows easily specifying Python dependencies for both driver and workers in a consistent way and enables quick iterations for development of Ray applications ([#50160](https://github.com/ray-project/ray/pull/50160), [50462](https://github.com/ray-project/ray/pull/50462)), check out our [blog post](https://www.anyscale.com/blog/uv-ray-pain-free-python-dependencies-in-clusters)

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>
🎉 New Features:
- *Ray Data LLM*: We are introducing a new module in Ray Data for batch inference with LLMs (currently marked as **alpha**). It offers a new `Processor` abstraction that interoperates with existing Ray Data pipelines. This abstraction can be configured two ways:
  - Using the `vLLMEngineProcessorConfig`, which configures vLLM to load model replicas for high throughput model inference
  - Using the `HttpRequestProcessorConfig`, which sends HTTP requests to an OpenAI-compatible endpoint for inference. 
  - Documentation for these features can be [found here.](https://docs.ray.io/en/master/data/working-with-llms.html)
- Implement accurate memory accounting for `UnionOperator` ([#50436](https://github.com/ray-project/ray/pull/50436))
- Implement accurate memory accounting for all-to-all operations ([#50290](https://github.com/ray-project/ray/pull/50290))

💫 Enhancements:
- Support class constructor args for filter() ([#50245](https://github.com/ray-project/ray/pull/50245))
- Persist ParquetDatasource metadata. ([#50332](https://github.com/ray-project/ray/pull/50332))
- Rebasing `ShufflingBatcher` onto `try_combine_chunked_columns` ([#50296](https://github.com/ray-project/ray/pull/50296))
- Improve warning message if required dependency isn't installed ([#50464](https://github.com/ray-project/ray/pull/50464))
- Move data-related test logic out of core tests directory ([#50482](https://github.com/ray-project/ray/pull/50482))
- Pass executor as an argument to ExecutionCallback ([#50165](https://github.com/ray-project/ray/pull/50165))
- Add operator id info to task+actor ([#50323](https://github.com/ray-project/ray/pull/50323))
- Abstracting common methods, removing duplication in `ArrowBlockAccessor`, `PandasBlockAccessor` ([#50498](https://github.com/ray-project/ray/pull/50498))
- Warn if map UDF is too large ([#50611](https://github.com/ray-project/ray/pull/50611))
- Replace `AggregateFn` with `AggregateFnV2`, cleaning up Aggregation infrastructure ([#50585](https://github.com/ray-project/ray/pull/50585))
- Simplify Operator.__repr__ ([#50620](https://github.com/ray-project/ray/pull/50620))
- Adding in `TaskDurationStats` and `on_execution_step` callback ([#50766](https://github.com/ray-project/ray/pull/50766))
- Print Resource Manager stats in release tests ([#50801](https://github.com/ray-project/ray/pull/50801))

🔨 Fixes:
- Fix invalid escape sequences in `grouped_data.py` docstrings ([#50392](https://github.com/ray-project/ray/pull/50392))
- Deflake `test_map_batches_async_generator` ([#50459](https://github.com/ray-project/ray/pull/50459))
- Avoid memory leak with `pyarrow.infer_type` on datetime arrays ([#50403](https://github.com/ray-project/ray/pull/50403))
- Fix parquet partition cols to support tensors types ([#50591](https://github.com/ray-project/ray/pull/50591))
- Fixing aggregation protocol to be appropriately associative ([#50757](https://github.com/ray-project/ray/pull/50757))

📖 Documentation:
- Remove "Stable Diffusion Batch Prediction with Ray Data" example ([#50460](https://github.com/ray-project/ray/pull/50460))

## Ray Train<a id="ray-train"></a>
🎉 New Features:
- Ray Train V2 is available to try starting in Ray 2.43! Run your next Ray Train job with the `RAY_TRAIN_V2_ENABLED=1` environment variable. See [the migration guide](https://github.com/ray-project/ray/issues/49454) for more information.

💫 Enhancements:
- Add a training ingest benchmark release test ([#50019](https://github.com/ray-project/ray/pull/50019), [#50299](https://github.com/ray-project/ray/pull/50299)) with a fault tolerance variant ([#50399](https://github.com/ray-project/ray/pull/50399))
- Add telemetry for Trainer usage in V2 ([#50321](https://github.com/ray-project/ray/pull/50321))
- Add pydantic as a `ray[train]` extra install ([#46682](https://github.com/ray-project/ray/pull/46682))
- Add state tracking to train v2 to make run status, run attempts, and training worker metadata observable ([#50515](https://github.com/ray-project/ray/pull/50515))

🔨 Fixes:
- Increase doc test parallelism ([#50326](https://github.com/ray-project/ray/pull/50326))
- Disable TF test for py312 ([#50382](https://github.com/ray-project/ray/pull/50382))
- Increase test timeout to deflake ([#50796](https://github.com/ray-project/ray/pull/50796))

📖 Documentation:
- Add missing xgboost pip install in example ([#50232](https://github.com/ray-project/ray/pull/50232))

🏗 Architecture refactoring:
- Add deprecation warnings pointing to a migration guide for Ray Train V2 ([#49455](https://github.com/ray-project/ray/pull/49455), [#50101](https://github.com/ray-project/ray/pull/50101), [#50322](https://github.com/ray-project/ray/pull/50322))
- Refactor internal Train controller state management ([#50113](https://github.com/ray-project/ray/pull/50113), [#50181](https://github.com/ray-project/ray/pull/50181), [#50388](https://github.com/ray-project/ray/pull/50388))

## Ray Tune<a id="ray-tune"></a>
🔨 Fixes:
- Fix worker node failure test ([#50109](https://github.com/ray-project/ray/pull/50109))

📖 Documentation:
- Update all doc examples off of ray.train imports ([#50458](https://github.com/ray-project/ray/pull/50458))
- Update all ray/tune/examples off of ray.train imports ([#50435](https://github.com/ray-project/ray/pull/50435))
- Fix typos in persistent storage guide ([#50127](https://github.com/ray-project/ray/pull/50127))
- Remove Binder notebook links in Ray Tune docs ([#50621](https://github.com/ray-project/ray/pull/50621))

🏗 Architecture refactoring:
- Update RLlib to use ray.tune imports instead of ray.air and ray.train ([#49895](https://github.com/ray-project/ray/pull/49895))

## Ray Serve<a id="ray-serve"></a>
🎉 New Features:
- *Ray Serve LLM*: We are introducing a new module in Ray Serve to easily integrate open source LLMs in your Ray Serve deployment, currently marked as **alpha**. This opens up a powerful capability of composing complex applications with multiple LLMs, which is a use case in emerging applications like agentic workflows. Ray Serve LLM offers a couple core components, including:
  - `VLLMService`: A prebuilt deployment that offers a full-featured vLLM engine integration, with support for features such as LoRA multiplexing and multimodal language models.
  - `LLMRouter`: An out-of-the-box OpenAI compatible model router that can route across multiple LLM deployments.
  - Documentation can be found at https://docs.ray.io/en/releases-2.43.0/serve/llm/overview.html

💫 Enhancements:
- Add `required_resources` to REST API ([#50058](https://github.com/ray-project/ray/pull/50058))

🔨 Fixes:
- Fix batched requests hanging after cancellation ([#50054](https://github.com/ray-project/ray/pull/50054))
- Properly propagate backpressure error ([#50311](https://github.com/ray-project/ray/pull/50311))

## RLlib<a id="rllib"></a>
🎉 New Features:
- Added env vectorization support for multi-agent (new API stack). ([#50437](https://github.com/ray-project/ray/pull/50437))

💫 Enhancements:
- APPO/IMPALA various acceleration efforts. Reached 100k ts/sec on Atari benchmark with 400 EnvRunners and 16 (multi-node) GPU Learners: [#50760](https://github.com/ray-project/ray/pull/50760), [#50162](https://github.com/ray-project/ray/pull/50162), [#50249](https://github.com/ray-project/ray/pull/50249), [#50353](https://github.com/ray-project/ray/pull/50353), [#50368](https://github.com/ray-project/ray/pull/50368), [#50379](https://github.com/ray-project/ray/pull/50379), [#50440](https://github.com/ray-project/ray/pull/50440), [#50477](https://github.com/ray-project/ray/pull/50477), [#50527](https://github.com/ray-project/ray/pull/50527), [#50528](https://github.com/ray-project/ray/pull/50528), [#50600](https://github.com/ray-project/ray/pull/50600), [#50309](https://github.com/ray-project/ray/pull/50309)
- Offline RL:
  - Remove all weight synching to `eval_env_runner_group` from the training steps. ([#50057](https://github.com/ray-project/ray/pull/50057))
  - Enable single-learner/multi-learner GPU training. ([#50034](https://github.com/ray-project/ray/pull/50034))
  - Remove reference to MARWILOfflinePreLearner in `OfflinePreLearner` docstring. ([#50107](https://github.com/ray-project/ray/pull/50107))
  - Add metrics to multi-agent replay buffers. ([#49959](https://github.com/ray-project/ray/pull/49959)[)](https://github.com/ray-project/ray/commit/00de19036cfcd125012711658833124edaf66c53)

🔨 Fixes:
- Fix SPOT preemption tolerance for large AlgorithmConfig: Pass by reference to RolloutWorker ([#50688](https://github.com/ray-project/ray/pull/50688))
- `on_workers/env_runners_recreated` callback would be called twice. ([#50172](https://github.com/ray-project/ray/pull/50172))
- `default_resource_request`: aggregator actors missing in placement group for local Learner. ([#50219](https://github.com/ray-project/ray/pull/50219), [#50475](https://github.com/ray-project/ray/pull/50475))

📖 Documentation:
- Docs re-do (new API stack):
  - Rewrite/enhance "getting started" rst page. ([#49950](https://github.com/ray-project/ray/pull/49950))
  - Remove rllib-models.rst and fix broken html links. ([#49966](https://github.com/ray-project/ray/pull/49966), [#50126](https://github.com/ray-project/ray/pull/50126))

# Ray Core and Ray Clusters

## Ray Core<a id="ray-core"></a>
💫 Enhancements:
- [Core] Enable users to configure python standard log attributes for structured logging (#49871)
- [Core] Prestart worker with runtime env (#49994) 
- [compiled graphs] Support experimental_compile(_default_communicator=comm) (#50023)
- [Core] ray.util.Queue Empty and Full exceptions extend queue.Empty and Full (#50261)
- [Core] Initial port of Ray to Python 3.13 (#47984)

🔨 Fixes:
- [Core] Ignore stale ReportWorkerBacklogRequest (#50280)
- [Core] Fix check failure due to negative available resource (#50517)

## Ray Clusters <a id="ray-clusters"></a>
📖 Documentation:
- Update the KubeRay docs to v1.3.0.

## Ray Dashboard <a id="ray-dashboard"></a>
🎉 New Features:
- Additional filters for job list page ([#50283](https://github.com/ray-project/ray/pull/50283))

# Thanks

Thank you to everyone who contributed to this release! 🥳
@liuxsh9, @justinrmiller, @CheyuWu, @400Ping, @scottsun94, @bveeramani, @bhmiller, @tylerfreckmann, @hefeiyun, @pcmoritz, @matthewdeng, @dentiny, @erictang000, @gvspraveen, @simonsays1980, @aslonnie, @shorbaji, @LeoLiao123, @justinvyu, @israbbani, @zcin, @ruisearch42, @khluu, @kouroshHakha, @sijieamoy, @SergeCroise, @raulchen, @anson627, @bluenote10, @allenyin55, @martinbomio, @rueian, @rynewang, @owenowenisme, @Betula-L, @alexeykudinkin, @crypdick, @jujipotle, @saihaj, @EricWiener, @kevin85421, @MengjinYan, @chris-ray-zhang, @SumanthRH, @chiayi, @comaniac, @angelinalg, @kenchung285, @tanmaychimurkar, @andrewsykim, @MortalHappiness, @sven1977, @richardliaw, @omatthew98, @fscnick, @akyang-anyscale, @cristianjd, @Jay-ju, @spencer-p, @win5923, @wxsms, @stfp, @letaoj, @JDarDagran, @jjyao, @srinathk10, @edoakes, @vincent0426, @dayshah, @davidxia, @DmitriGekhtman, @GeneDer, @HYLcool, @gameofby, @can-anyscale, @ryanaoleary, @eddyxu

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.43.0)

---

## ray-2.42.1: Ray-2.42.1
**Published:** 2025-02-11

## Ray Data<a id="ray-data"></a>

🔨 Fixes:

- Fixes incorrect assertion (#50210)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.42.1)

---

## ray-2.42.0: Ray-2.42.0
**Published:** 2025-02-05

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>
🎉 New Features:
- Added read_audio and read_video ([#50016](https://github.com/ray-project/ray/pull/50016))

💫 Enhancements:
- Optimized multi-column groupbys ([#45667](https://github.com/ray-project/ray/pull/45667))
- Included Ray user-agent in BigQuery client construction ([#49922](https://github.com/ray-project/ray/pull/49922))

🔨 Fixes:
- Fixed bug that made read tasks non-deterministic ([#49897](https://github.com/ray-project/ray/pull/49897))

🗑️ Deprecations:
- Deprecated num_rows_per_file in favor of min_rows_per_file ([#49978](https://github.com/ray-project/ray/pull/49978))

## Ray Train<a id="ray-train"></a>
💫 Enhancements:
- Add Train v2 user-facing callback interface (#49819)
- Add TuneReportCallback for propagating intermediate Train results to Tune (#49927)

## Ray Tune<a id="ray-tune"></a>
📖 Documentation:
- Fix BayesOptSearch docs (#49848)

## Ray Serve<a id="ray-serve"></a>
💫 Enhancements:
- Cache metrics in replica and report on an interval ([#49971](https://github.com/ray-project/ray/pull/49971))
- Cache expensive calls to inspect.signature ([#49975](https://github.com/ray-project/ray/pull/49975))
- Remove extra pickle serialization for gRPCRequest ([#49943](https://github.com/ray-project/ray/pull/49943))
- Shared LongPollClient for Routers ([#48807](https://github.com/ray-project/ray/pull/48807))
- DeploymentHandle API is now stable ([#49840](https://github.com/ray-project/ray/pull/49840))

🔨 Fixes:
- Fix batched requests hanging after request cancellation bug ([#50054](https://github.com/ray-project/ray/pull/50054))

## RLlib<a id="rllib"></a>
💫 Enhancements:
- Add metrics to replay buffers. ([#49822](https://github.com/ray-project/ray/pull/49822))
- Enhance node-failure tolerance (new API stack). ([#50007](https://github.com/ray-project/ray/pull/50007))
- MetricsLogger cleanup throughput logic. ([#49981](https://github.com/ray-project/ray/pull/49981))
- Split AddStates... connectors into 2 connector pieces (`AddTimeDimToBatchAndZeroPad` and `AddStatesFromEpisodesToBatch`) ([#49835](https://github.com/ray-project/ray/pull/49835))

🔨 Fixes:
- Old API stack IMPALA/APPO: Re-introduce mixin-replay-buffer pass, even if `replay-ratio=0` (fixes a memory leak). ([#49964](https://github.com/ray-project/ray/pull/49964))
- Fix MetricsLogger race conditions. ([#49888](https://github.com/ray-project/ray/pull/49888))
- APPO/IMPALA: Bug fix for > 1 Learner actor. ([#49849](https://github.com/ray-project/ray/pull/49849))

📖 Documentation:
- New MetricsLogger API rst page. ([#49538](https://github.com/ray-project/ray/pull/49538))
- Move "new API stack" info box right below page titles for better visibility. ([#49921](https://github.com/ray-project/ray/pull/49921))
- Add example script for how to log custom metrics in `training_step()`. ([#49976](https://github.com/ray-project/ray/pull/49976))
- Enhance/redo autoregressive action distribution example. ([#49967](https://github.com/ray-project/ray/pull/49967))
- Make the "tiny CNN" example RLModule run with APPO (by implementing `TargetNetAPI`) ([#49825](https://github.com/ray-project/ray/pull/49825))

# Ray Core and Ray Clusters

## Ray Core<a id="ray-core"></a>
💫 Enhancements:
- Only get single node info rather then all when needed ([#49727](https://github.com/ray-project/ray/pull/49727))
- Introduce with_tensor_transport API ([#49753](https://github.com/ray-project/ray/pull/49753))

🔨 Fixes:
- Fix tqdm manager thread safe [#50040](https://github.com/ray-project/ray/pull/50040)

## Ray Clusters <a id="ray-clusters"></a>
🔨 Fixes:
- Fix token expiration for ray autoscaler ([#48481](https://github.com/ray-project/ray/pull/48481))

# Thanks

Thank you to everyone who contributed to this release! 🥳 
@wingkitlee0, @saihaj, @win5923, @justinvyu, @kevin85421, @edoakes, @cristianjd, @rynewang, @richardliaw, @LeoLiao123, @alexeykudinkin, @simonsays1980, @aslonnie, @ruisearch42, @pcmoritz, @fscnick, @bveeramani, @mattip, @till-m, @tswast, @ujjawal-khare, @wadhah101, @nikitavemuri, @akshay-anyscale, @srinathk10, @zcin, @dayshah, @dentiny, @LydiaXwQ, @matthewdeng, @JoshKarpel, @MortalHappiness, @sven1977, @omatthew98

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.42.0)

---

## ray-2.41.0: Ray-2.41.0
**Published:** 2025-01-23

# Highlights

- Major update of RLlib docs and example scripts for the new API stack.

# Ray Libraries

## Ray Data

🎉 New Features:

- Expression support for filters (#49016) 
- Support `partition_cols` in `write_parquet` (#49411)
- Feature: implement multi-directional sort over Ray Data datasets (#49281)

💫 Enhancements:

- Use dask 2022.10.2 (#48898) 
- Clarify schema validation error (#48882)
- Raise `ValueError` when the data sort key is `None` (#48969) 
- Provide more messages when webdataset format is error (#48643) 
- Upgrade Arrow version from 17 to 18 (#48448)
- Update `hudi` version to 0.2.0 (#48875) 
- `webdataset`: expand JSON objects into individual samples (#48673)
- Support passing kwargs to map tasks. (#49208) 
- Add `ExecutionCallback` interface (#49205) 
- Add seed for read files (#49129)
- Make `select_columns` and `rename_columns` use Project operator (#49393)

🔨 Fixes:

- Fix partial function name parsing in `map_groups` (#48907) 
- Always launch one task for `read_sql` (#48923) 
- Reimplement of fix memory pandas (#48970)
- `webdataset`: flatten return args (#48674) 
- Handle `numpy > 2.0.0` behaviour in `_create_possibly_ragged_ndarray` (#48064)
- Fix `DataContext` sealing for multiple datasets. (#49096) 
- Fix `to_tf` for `List` types (#49139) 
- Fix type mismatch error while mapping nullable column (#49405) 
- Datasink: support passing write results to `on_write_completes` (#49251)
- Fix `groupby` hang when value contains `np.nan` (#49420)
- Fix bug where `file_extensions` doesn't work with compound extensions (#49244)
- Fix map operator fusion when concurrency is set (#49573) 

## Ray Train

🎉 New Features:

- Output JSON structured log files for system and application logs (#49414)
- Add support for AMD ROCR_VISIBLE_DEVICES (#49346)

💫 Enhancements:

- Implement Train Tune API Revamp REP (#49376, #49467, #49317, #49522)

🏗 Architecture refactoring:

- LightGBM: Rewrite `get_network_params` implementation (#49019)

## Ray Tune

🎉 New Features:

- Update `optuna_search` to allow users to configure optuna storage (#48547)

🏗 Architecture refactoring:

- Make changes to support Train Tune API Revamp REP (#49308, #49317, #49519)

## Ray Serve

💫 Enhancements:

- Improved request_id generation to reduce proxy CPU overhead (#49537)
- Tune GC threshold by default in proxy (#49720)
- Use `pickle.dumps` for faster serialization from `proxy` to `replica` (#49539)

🔨 Fixes:

- Handle nested ‘=’ in serve run arguments (#49719)
- Fix bug when `ray.init()` is called multiple times with different `runtime_envs` (#49074)

🗑️ Deprecations:

- Adds a warning that the default behavior for sync methods will change in a future release. They will be run in a threadpool by default. You can opt into this behavior early by setting `RAY_SERVE_RUN_SYNC_IN_THREADPOOL=1`. (#48897)

## RLlib

🎉 New Features:

- Add support for external Envs to new API stack: New example script and custom tcp-capable EnvRunner. (#49033)

💫 Enhancements:

- Offline RL:
  - Add sequence sampling to `EpisodeReplayBuffer`. (#48116)
  - Allow incomplete `SampleBatch` data and fully compressed observations. (#48699)
  - Add option to customize `OfflineData`. (#49015)
  - Enable offline training without specifying an environment. (#49041)
  - Various fixes: #48309, #49194, #49195
- APPO/IMPALA acceleration (new API stack):
  - Add support for `AggregatorActors` per Learner. (#49284)
  - Auto-sleep time AND thread-safety for MetricsLogger. (#48868)
  - Activate APPO cont. actions release- and CI tests (HalfCheetah-v1 and Pendulum-v1 new in `tuned_examples`). (#49068)
  - Add "burn-in" period setting to the training of stateful RLModules. (#49680)
- Callbacks API: Add support for individual lambda-style callbacks. (#49511)
- Other enhancements: #49687, #49714, #49693, #49497, #49800, #49098

📖 Documentation:

- New example scripts:
  - How to write a custom algorithm (VPG) from scratch. (#49536)
  - How to customize an offline data pipeline. (#49046)
  - GPUs on EnvRunners. (#49166)
  - Hierarchical training. (#49127)
  - Async gym vector env. (#49527)
  - Other fixes and enhancements: #48988, #49071
- New/rewritten html pages:
  - Rewrite checkpointing page. (#49504)
  - New scaling guide. (#49528)
  - New callbacks page. (#49513)
  - Rewrite `RLModule` page. (#49387)
  - New AlgorithmConfig page and redo `package_ref` page for algo configs. (#49464)
  - Rewrite offline RL page. (#48818)
  - Rewrite “key concepts" rst page. (#49398)
  - Rewrite RL environments pages. (#49165, #48542)
  - Fixes and enhancements: #49465, #49037, #49304, #49428, #49474, #49399, #49713, #49518
  
🔨 Fixes:

- Add `on_episode_created` callback to SingleAgentEnvRunner. (#49487)
- Fix `train_batch_size_per_learner` problems. (#49715)
- Various other fixes: #48540, #49363, #49418, #49191

🏗 Architecture refactoring:

- RLModule: Introduce `Default[algo]RLModule` classes (#49366, #49368)
- Remove RLlib dependencies from setup.py; add `ormsgpack` (#49489)

🗑️ Deprecations:

- #49488, #49144

# Ray Core and Ray Clusters

## Ray Core

💫 Enhancements:

- Add `task_name`, `task_function_name` and `actor_name` in Structured Logging (#48703)
- Support redis/valkey authentication with username (#48225)
- Add v6e TPU Head Resource Autoscaling Support (#48201)
- compiled graphs: Support all driver and actor read combinations (#48963) 
- compiled graphs: Add ascii based CG visualization (#48315) 
- compiled graphs: Add ray[cg] pip install option (#49220)
- Allow uv cache at installation (#49176)
- Support != Filter in GCS for Task State API (#48983)
- compiled graphs: Add CPU-based NCCL communicator for development (#48440)
- Support gcs and raylet log rotation (#48952)
- compiled graphs: Support `nsight.nvtx` profiling (#49392)

🔨 Fixes:

- autoscaler: Health check logs are not visible in the autoscaler container's stdout (#48905)
- Only publish `WORKER_OBJECT_EVICTION` when the object is out of scope or manually freed (#47990)
- autoscaler: Autoscaler doesn't scale up correctly when the KubeRay RayCluster is not in the goal state (#48909)
- autoscaler: Fix incorrectly terminating nodes misclassified as idle in autoscaler v1 (#48519)
- compiled graphs: Fix the missing dependencies when num_returns is used (#49118)
- autoscaler: Fuse scaling requests together to avoid overloading the Kubernetes API server (#49150)
- Fix bug to support S3 pre-signed url for `.whl` file (#48560)
- Fix data race on gRPC client context (#49475)
- Make sure draining node is not selected for scheduling (#49517)

## Ray Clusters

💫 Enhancements:

- Azure: Enable accelerated networking as a flag in azure vms (#47988)

📖 Documentation:

- Kuberay: Logging: Add Fluent Bit `DaemonSet` and Grafana Loki to "Persist KubeRay Operator Logs" (#48725)
- Kuberay: Logging: Specify the Helm chart version in "Persist KubeRay Operator Logs" (#48937)

Dashboard

💫 Enhancements:

- Add instance variable to many default dashboard graphs (#49174)
- Display duration in milliseconds if under 1 second. (#49126)
- Add `RAY_PROMETHEUS_HEADERS` env for carrying additional headers to Prometheus (#49353)
- Document about the `RAY_PROMETHEUS_HEADERS` env for carrying additional headers to Prometheus (#49700)

🏗 Architecture refactoring:

- Move `memray` dependency from default to observability (#47763)
- Move `StateHead`'s methods into free functions. (#49388)

## Thanks

@raulchen, @alanwguo, @omatthew98, @xingyu-long, @tlinkin, @yantzu, @alexeykudinkin, @andrewsykim, @win5923, @csy1204, @dayshah, @richardliaw, @stephanie-wang, @gueraf, @rueian, @davidxia, @fscnick, @wingkitlee0, @KPostOffice, @GeneDer, @MengjinYan, @simonsays1980, @pcmoritz, @petern48, @kashiwachen, @pfldy2850, @zcin, @scottjlee, @Akhil-CM, @Jay-ju, @JoshKarpel, @edoakes, @ruisearch42, @gorloffslava, @jimmyxie-figma, @bthananjeyan, @sven1977, @bnorick, @jeffreyjeffreywang, @ravi-dalal, @matthewdeng, @angelinalg, @ivanthewebber, @rkooo567, @srinathk10, @maresb, @gvspraveen, @akyang-anyscale, @mimiliaogo, @bveeramani, @ryanaoleary, @kevin85421, @richardsliu, @hartikainen, @coltwood93, @mattip, @Superskyyy, @justinvyu, @hongpeng-guo, @ArturNiederfahrenhorst, @jecsand838, @Bye-legumes, @hcc429, @WeichenXu123, @martinbomio, @HollowMan6, @MortalHappiness, @dentiny, @zhe-thoughts, @anyadontfly, @smanolloff, @richo-anyscale, @khluu, @xushiyan, @rynewang, @japneet-anyscale, @jjyao, @sumanthratna, @saihaj, @aslonnie 

Many thanks to all those who contributed to this release! 


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.41.0)

---

## ray-2.40.0: Ray-2.40.0
**Published:** 2024-12-04

# Ray Libraries
## Ray Data
🎉 New Features:
- Added read_hudi (https://github.com/ray-project/ray/pull/46273)

💫 Enhancements:
- Improved performance of DelegatingBlockBuilder (https://github.com/ray-project/ray/pull/48509)
- Improved memory accounting of pandas blocks (https://github.com/ray-project/ray/pull/46939)

🔨 Fixes:
- Fixed bug where you can’t specify a schema with write_parquet (https://github.com/ray-project/ray/issues/48630)
- Fixed bug where to_pandas errors if your dataset contains Arrow and pandas blocks (https://github.com/ray-project/ray/pull/48583)
- Fixed bug where map_groups doesn’t work with pandas data (https://github.com/ray-project/ray/pull/48287)
- Fixed bug where write_parquet errors if your data contains nullable fields (https://github.com/ray-project/ray/pull/48478)
- Fixed bug where “Iteration Blocked Time” charts looks incorrect (https://github.com/ray-project/ray/pull/48618)
- Fixed bug where unique fails with null values (https://github.com/ray-project/ray/pull/48750)
- Fixed bug where “Rows Outputted” is 0 in the Data dashboard (https://github.com/ray-project/ray/pull/48745)
- Fixed bug where methods like drop_columns cause spilling (https://github.com/ray-project/ray/pull/48140)
- Fixed bug where async map tasks hang (https://github.com/ray-project/ray/pull/48861)

🗑️ Deprecations:
- Deprecated read_parquet_bulk https://github.com/ray-project/ray/pull/48691
- Deprecated iter_tf_batches https://github.com/ray-project/ray/pull/48693
- Deprecated meta_provider parameter of read functions (https://github.com/ray-project/ray/pull/48690)
- Deprecated to_torch (https://github.com/ray-project/ray/pull/48692)

## Ray Train
🔨 Fixes:
- Fix StartTracebackWithWorkerRank serialization (#48548)

📖 Documentation:
- Add example for fine-tuning Llama3.1 with AWS Trainium (#48768)

## Ray Tune
🔨 Fixes:
- Remove the `clear_checkpoint` function during Trial restoration error handling. (#48532)

## Ray Serve
🎉 New Features:
- Initial version of local_testing_mode ([#48477](https://github.com/ray-project/ray/pull/48477))

💫 Enhancements:
- Handle multiple changed objects per LongPollHost.listen_for_change RPC ([#48803](https://github.com/ray-project/ray/pull/48803/files))
- Add more nuanced checks for http proxy status errors ([#47896](https://github.com/ray-project/ray/pull/47896))
- Improve replica access log messages to include HTTP status info and better resemble standard log format ([#48819](https://github.com/ray-project/ray/pull/48819))
- Propagate replica constructor error to deployment status message and print num retries left ([#48531](https://github.com/ray-project/ray/pull/47896))

🔨 Fixes:
- Pending requests that are cancelled before they were assigned to a replica now also return a serve.RequestCancelledError ([#48496](https://github.com/ray-project/ray/pull/48496))

## RLlib
💫 Enhancements:
- Release test enhancements. ([#45803](https://github.com/ray-project/ray/pull/45803), [#48681](https://github.com/ray-project/ray/pull/48681))
- Make opencv-python-headless default over opencv-python ([#48776](https://github.com/ray-project/ray/pull/48776)[)](https://github.com/ray-project/ray/commit/aaac19c8307038021dd96ffc4c2e616fbbf14896)
- Reverse learner queue behavior of IMPALA/APPO (consume oldest batches first, instead of newest, BUT drop oldest batches if queue full). ([#48702](https://github.com/ray-project/ray/pull/48702))

🔨 Fixes:
- Fix torch scheduler stepping and reporting. ([#48125](https://github.com/ray-project/ray/pull/48125)[)](https://github.com/ray-project/ray/commit/ec9775d86fbf7eb93358d95268e9f62e53f790bd)
- Fix accumulation of results over n training_step calls within same iteration (new API stack). ([#48136](https://github.com/ray-project/ray/pull/48136))
- Various other fixes: [#48563](https://github.com/ray-project/ray/pull/48563), [#48314](https://github.com/ray-project/ray/pull/48314), [#48698](https://github.com/ray-project/ray/pull/48698), [#48869](https://github.com/ray-project/ray/pull/48869).

📖 Documentation:
- Upgrade examples script overview page (new API stack). ([#48526](https://github.com/ray-project/ray/pull/48526)[)](https://github.com/ray-project/ray/commit/d39c9df1b69ba0451abff7075963c3a6e2824c9c)
- Enable RLlib + Serve example in CI and translate to new API stack. ([#48687](https://github.com/ray-project/ray/pull/48687))

🏗 Architecture refactoring:
- Switch new API stack on by default, APPO, IMPALA, BC, MARWIL, and CQL. ([#48516](https://github.com/ray-project/ray/pull/48516), [#48599](https://github.com/ray-project/ray/pull/48599)[)](https://github.com/ray-project/ray/commit/03ea4f6663fafaf64b8d10ac8db8e962302be561)
- Various APPO enhancements (new API stack): Circular buffer [(](https://github.com/ray-project/ray/commit/05915c1b389ab0bada23217a3cb2768311d1184b)[#48798](https://github.com/ray-project/ray/pull/48798)), minor loss math fixes ([#48800](https://github.com/ray-project/ray/pull/48800)), target network update logic ([#48802](https://github.com/ray-project/ray/pull/48802)), smaller cleanups ([#48844](https://github.com/ray-project/ray/pull/48844)).
- Remove `rllib_contrib` from repo. ([#48565](https://github.com/ray-project/ray/pull/48565)[)](https://github.com/ray-project/ray/commit/d2de98323f0848fec2dbeb61bbd39b507b9c97d8)

# Ray Core and Ray Clusters

## Ray Core
🎉 New Features:
- [Core] uv runtime env support ([#48479](https://github.com/ray-project/ray/pull/48479), [#48486](https://github.com/ray-project/ray/pull/48486), [#48611](https://github.com/ray-project/ray/pull/48611), [#48619](https://github.com/ray-project/ray/pull/48619), [#48632](https://github.com/ray-project/ray/pull/48632), [#48634](https://github.com/ray-project/ray/pull/48634), [#48637](https://github.com/ray-project/ray/pull/48637), [#48670](https://github.com/ray-project/ray/pull/48670), [#48731](https://github.com/ray-project/ray/pull/48731))
- [Core] GCS FT with redis sentinel (#47335)

💫 Enhancements:
- [CompiledGraphs] Refine schedule visualization (#48594)

🔨 Fixes:
- [CompiledGraphs] Don't persist input_nodes in _CollectiveOperation to avoid wrong understanding about DAGs (#48463)
- [Core] Fix Ascend NPU discovery to support 8+ cards per node (#48543)
- [Core] Make Placement Group Wildcard and Indexed Resource Assignments Consistent (#48088)
- [Core] Stop the GRPC server before Shut down the Object Store (#48572)

## Ray Clusters
🔨 Fixes:
- [KubeRay]: Fix ConnectionError on Autoscaler CR lookups in K8s clusters with custom DNS for Kubernetes API. ([#48541](https://github.com/ray-project/ray/pull/48541))

## Dashboard
💫 Enhancements:
- Add global UTC timezone button in navbar with local storage (#48510)
- Add memory graphs optimized for OOM debugging (#48530) 
- Improve tasks/actors metric naming and add graph for running tasks (#48528)
add actor pid to dashboard (#48791)

🔨 Fixes:
- Fix Placement Group Table table cells overflow (#47323)
- Fix Rows Outputted being zero on Ray Data Dashboard (#48745) 
- fix confusing dataset operator name (#48805)

# Thanks
Thanks to all those who contributed to this release! 
@rynewang, @rickyyx, @bveeramani, @marwan116, @simonsays1980, @dayshah, @dentiny, @KepingYan, @mimiliaogo, @kevin85421, @SeaOfOcean, @stephanie-wang, @mohitjain2504, @azayz, @xushiyan, @richardliaw, @can-anyscale, @xingyu-long, @kanwang, @aslonnie, @MortalHappiness, @jjyao, @SumanthRH, @matthewdeng, @alexeykudinkin, @sven1977, @raulchen, @andrewsykim, @zcin, @nadongjun, @hongpeng-guo, @miguelteixeiraa, @saihaj, @khluu, @ArturNiederfahrenhorst, @ryanaoleary, @ltbringer, @pcmoritz, @JoshKarpel, @akyang-anyscale, @frances720, @BeingGod, @edoakes, @Bye-legumes, @Superskyyy, @liuxsh9, @MengjinYan, @ruisearch42, @scottjlee, @angelinalg

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.40.0)

---

## ray-2.39.0: Ray-2.39.0
**Published:** 2024-11-13

# Ray Libraries

## Ray Data

🔨 Fixes:
- Fixed InvalidObjectError edge case with Dataset.split() (https://github.com/ray-project/ray/pull/48130)
- Made Concatenator preserve order of concatenated columns (https://github.com/ray-project/ray/pull/47997)

📖 Documentation:
- Improved documentation around Parquet column and predicate pushdown (https://github.com/ray-project/ray/pull/48095)
- Marked num_rows_per_file parameter of write APIs as experimental (https://github.com/ray-project/ray/pull/48208)
- One hot encoder now returns an encoded vector (https://github.com/ray-project/ray/pull/48173)
- transform_batch no longer fails on missing columns (https://github.com/ray-project/ray/pull/48137)

🏗 Architecture refactoring:
- Dataset.count() now uses a Count logical operator (https://github.com/ray-project/ray/pull/48126)

🗑 Deprecations:
- Removed long-deprecated set_progress_bars (https://github.com/ray-project/ray/pull/48203)

## Ray Train

🔨 Fixes:
- Safely check if the storage filesystem is `pyarrow.fs.S3FileSystem` (#48216)

## Ray Tune

🔨 Fixes:
- Safely check if the storage filesystem is `pyarrow.fs.S3FileSystem` (#48216)

## Ray Serve

💫 Enhancements:
- Cancelled requests now return a serve.RequestCancelledError (https://github.com/ray-project/ray/pull/48444)
- Exposed application source in app details model (https://github.com/ray-project/ray/pull/45522)

🔨 Fixes:
- Basic HTTP deployments will now return “Internal Server Error” instead of a traceback to match FastAPI behavior (https://github.com/ray-project/ray/pull/48491)
- Fixed an issue where high values of max_ongoing_requests couldn’t be reached due to an interaction with core’s max_concurrency (https://github.com/ray-project/ray/pull/48274)
- Fixed an edge case where pending requests were not canceled properly (https://github.com/ray-project/ray/pull/47873)
- Removed deprecated API to set route_prefix per-deployment (https://github.com/ray-project/ray/pull/48223)

📖 Documentation:
- Added ProxyStatus model to reference docs (https://github.com/ray-project/ray/pull/48299)
- Added ApplicationStatus model to reference docs (https://github.com/ray-project/ray/pull/48220)

## RLlib

💫 Enhancements:
- Upgrade to gymnasium==1.0.0 (support new API for vector env resets). ([#48443](https://github.com/ray-project/ray/pull/48443), [#45328](https://github.com/ray-project/ray/pull/45328))
- Add off-policy'ness metric to new API stack. ([#48227](https://github.com/ray-project/ray/pull/48227))
- Validate episodes before adding them to the buffer. ([#48083](https://github.com/ray-project/ray/pull/48083))

📖 Documentation:
- New example script for custom metrics on `EnvRunners` (using `MetricsLogger` API on the new stack). ([#47969](https://github.com/ray-project/ray/pull/47969))
- Do-over: New RLlib index page. ([#48285](https://github.com/ray-project/ray/pull/48285), [#48442](https://github.com/ray-project/ray/pull/48442))
- Do-over: Example script for AutoregressiveActionsRLM. ([#47972](https://github.com/ray-project/ray/pull/47972))

🏗 Architecture refactoring:
- New API stack on by default for PPO. ([#48284](https://github.com/ray-project/ray/pull/48284))
- Change config.fault_tolerance default behavior (from `recreate_failed_env_runners=False` to `True`). ([#48286](https://github.com/ray-project/ray/pull/48286))

🔨 Fixes:
- Various bug and CI fixes: [#47993](https://github.com/ray-project/ray/pull/47993), [#48450](https://github.com/ray-project/ray/pull/48450), [#48213](https://github.com/ray-project/ray/pull/48213)
- Cleanup `evaluation` folder ([#48493](https://github.com/ray-project/ray/pull/48493))

## Ray Core

🎉 New Features:
- [CompiledGraphs] Support all reduce collective in aDAG ([#47621](https://github.com/ray-project/ray/pull/47621))
- [CompiledGraphs] Add visualization of compiled graphs ([#47958](https://github.com/ray-project/ray/pull/47958))

💫 Enhancements:
- [**Distributed Debugger**] The distributed debugger can now be used without having to set RAY_DEBUG=1, see https://github.com/ray-project/ray/pull/48301 and https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html. If you want to restore the previous behavior and use the CLI based debugger, you need to set RAY_DEBUG=legacy.
- [Core] Add more infos to each breakpoint for ray debug CLI ([#48202](https://github.com/ray-project/ray/pull/48202))
- [Core] Add demands info to GCS debug state ([#48115](https://github.com/ray-project/ray/pull/48115))
- [Core] Add PENDING_ACTOR_TASK_ARGS_FETCH and PENDING_ACTOR_TASK_ORDERING_OR_CONCURRENCY TaskStatus ([#48242](https://github.com/ray-project/ray/pull/48242))
- [Core] Add metrics ray_io_context_event_loop_lag_ms. ([#47989](https://github.com/ray-project/ray/pull/47989))
- [Core] Better log format when show the disk size ([#46869](https://github.com/ray-project/ray/pull/46869))
- [CompiledGraphs] Support asyncio.gather on multiple CompiledDAGFutures ([#47860](https://github.com/ray-project/ray/pull/47860))
- [CompiledGraphs] Raise an exception if a leaf node is found during compilation ([#47757](https://github.com/ray-project/ray/pull/47757))


🔨 Fixes:
- [Core] Posts CoreWorkerMemoryStore callbacks onto io_context to fix deadlock ([#47833](https://github.com/ray-project/ray/pull/47833))

## Dashboard

🔨 Fixes:
- [Dashboard] Reworking dashboard_max_actors_to_cache to RAY_maximum_gcs_destroyed_actor_cached_count ([#48229](https://github.com/ray-project/ray/pull/48229))

# Thanks
Many thanks to all those who contributed to this release! 

@akyang-anyscale, @rkooo567, @bveeramani, @dayshah, @martinbomio, @khluu, @justinvyu, @slfan1989, @alexeykudinkin, @simonsays1980, @vigneshka, @ruisearch42, @rynewang, @scottjlee, @jjyao, @JoshKarpel, @win5923, @MengjinYan, @MortalHappiness, @ujjawal-khare-27, @zcin, @ccoulombe, @Bye-legumes, @dentiny, @stephanie-wang, @LeoLiao123, @dengwxn, @richo-anyscale, @pcmoritz, @sven1977, @omatthew98, @GeneDer, @srinathk10, @can-anyscale, @edoakes, @kevin85421, @aslonnie, @jeffreyjeffreywang, @ArturNiederfahrenhorst

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.39.0)

---

## ray-2.38.0: Ray-2.38.0
**Published:** 2024-10-23

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉 New Features:
- Add `Dataset.rename_columns` (#47906)
- Basic structured logging (#47210)

💫 Enhancements:
- Add `partitioning` parameter to `read_parquet` (#47553)
- Add `SERVICE_UNAVAILABLE` to list of retried transient errors (#47673)
- Re-phrase the streaming executor current usage string (#47515)
- Remove ray.kill in ActorPoolMapOperator (#47752)
- Simplify and consolidate progress bar outputs (#47692)
- Refactor `OpRuntimeMetrics` to support properties (#47800)
- Refactor `plan_write_op` and `Datasink`s (#47942)
- Link `PhysicalOperator` to its `LogicalOperator` (#47986)
- Allow specifying both `num_cpus` and `num_gpus` for map APIs (#47995)
- Allow specifying insertion index when registering custom plan optimization `Rule`s (#48039)
- Adding in better framework for substituting logging handlers (#48056)

🔨 Fixes:
- Fix bug where Ray Data incorrectly emits progress bar warning (#47680)
- Yield remaining results from async `map_batches` (#47696)
- Fix event loop mismatch with async map (#47907)
- Make sure `num_gpus` provide to Ray Data is appropriately passed to `ray.remote` call (#47768)
- Fix unequal partitions when grouping by multiple keys (#47924)
- Fix reading multiple parquet files with ragged ndarrays (#47961)
- Removing unneeded test case (#48031)
- Adding in better json checking in test logging (#48036)
- Fix bug with inserting custom optimization rule at index 0 (#48051)
- Fix logging output from `write_xxx` APIs (#48096)

📖 Documentation:
- Add docs section for Ray Data progress bars (#47804)
- Add reference to parquet predicate pushdown (#47881)
- Add tip about how to understand map_batches format (#47394)

## Ray Train<a id="ray-train"></a>

🏗 Architecture refactoring:
- Remove deprecated mosaic and sklearn trainer code (#47901)

## Ray Tune<a id="ray-tune"></a>

🔨 Fixes:
- Fix WandbLoggerCallback to reuse actors upon restore (#47985)

## Ray Serve<a id="ray-serve"></a>

🔨 Fixes:
- Stop scheduling task early when requests have been canceled (#47847)

## RLlib<a id="rllib"></a>

🎉 New Features:
- Enable cloud checkpointing. (#47682)

💫 Enhancements:
- PPO on new API stack now shuffles batches properly before each epoch. (#47458)
- Other enhancements: #47705, #47501, #47731, #47451, #47830, #47970, #47157

🔨 Fixes:
- Fix spot node preemption problem (RLlib now run stably with EnvRunner workers on spot nodes) (#47940)
- Fix action masking example. (#47817)
- Various other fixes: #47973, #46721, #47914, #47880, #47304, #47686

🏗 Architecture refactoring:
- Switch on new API stack by default for SAC and DQN. (#47217)
- Remove Tf support on new API stack for PPO/IMPALA/APPO (only DreamerV3 on new API stack remains with tf now). (#47892)
- Discontinue support for "hybrid" API stack (using RLModule + Learner, but still on RolloutWorker and Policy) (#46085)
- RLModule (new API stack) refinements: #47884, #47885, #47889, #47908, #47915, #47965, #47775

📖 Documentation:
- Add new API stack migration guide. (#47779)
- New API stack example script: BC pre training, then PPO finetuning using same RLModule class. (#47838)
- New API stack: Autoregressive actions example. (#47829)
- Remove old API stack connector docs entirely. (#47778)

# Ray Core and Ray Clusters
## Ray Core <a id="ray-core"></a>

🎉 New Features:
- CompiledGraphs: support multi readers in multi node when DAG is created from an actor (#47601)

💫 Enhancements:
- Add a flag to raise exception for out of band serialization of `ObjectRef` (#47544)
- Store each GCS table in its own Redis Hash (#46861)
- Decouple create worker vs pop worker request. (#47694)
- Add metrics for GCS jobs (#47793)

🔨 Fixes:
- Fix broken dashboard cluster page when there are dead nodes (#47701)
- Fix the `ray_tasks{State="PENDING_ARGS_FETCH"}` metric counting (#47770)
- Separate the attempt_number with the task_status in memory summary and object list (#47818)
- Fix object reconstruction hang on arguments pending creation (#47645)
- Fix check failure: `sync_reactors_.find(reactor->GetRemoteNodeID()) == sync_reactors_.end()` (#47861)
- Fix check failure `RAY_CHECK(it != current_tasks_.end())`; (#47659)

📖 Documentation:
- KubeRay docs: Add docs for YuniKorn Gang scheduling #47850 

## Dashboard<a id="dashboard"></a>

💫 Enhancements:
- Performance improvements for large scale clusters (#47617)

🔨 Fixes:
- Placement group and required resources not showing correctly in dashboard (#47754)

# Thanks

Many thanks to all those who contributed to this release!
@GeneDer, @rkooo567, @dayshah, @saihaj, @nikitavemuri, @bill-oconnor-anyscale, @WeichenXu123, @can-anyscale, @jjyao, @edoakes, @kekulai-fredchang, @bveeramani, @alexeykudinkin, @raulchen, @khluu, @sven1977, @ruisearch42, @dentiny, @MengjinYan, @Mark2000, @simonsays1980, @rynewang, @PatricYan, @zcin, @sofianhnaide, @matthewdeng, @dlwh, @scottjlee, @MortalHappiness, @kevin85421, @win5923, @aslonnie, @prithvi081099, @richardsliu, @milesvant, @omatthew98, @Superskyyy, @pcmoritz

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.38.0)

---

## ray-2.37.0: Ray-2.37.0
**Published:** 2024-09-24

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>
💫 Enhancements:
- Simplify custom metadata provider API (#47575)
- Change counts of metrics to rates of metrics (#47236)
- Throw exception for non-streaming HF datasets with "override_num_blocks" argument (#47559)
- Refactor custom optimizer rules (#47605)

🔨 Fixes:
- Remove ineffective retry code in `plan_read_op` (#47456)
- Fix incorrect pending task size if outputs are empty (#47604)

## Ray Train<a id="ray-train"></a>
💫 Enhancements:
- Update run status and add stack trace to `TrainRunInfo` (#46875)

## Ray Serve<a id="ray-serve"></a>
💫 Enhancements:
- Allow control of some serve configuration via env vars ([#47533](https://github.com/ray-project/ray/pull/47533))
- [serve] Faster detection of dead replicas ([#47237](https://github.com/ray-project/ray/pull/47237))

🔨 Fixes:
- [Serve] fix component id logging field ([#47609](https://github.com/ray-project/ray/pull/47609))

## RLlib<a id="rllib"></a>
💫 Enhancements:
- New API stack:
  - Add restart-failed-env option to EnvRunners. ([#47608](https://github.com/ray-project/ray/pull/47608)[)](https://github.com/ray-project/ray/commit/e75f5e7aa950e30097a0323f4baf14d90b1b6b9b)
  - Offline RL: Store episodes in state form. ([#47294](https://github.com/ray-project/ray/pull/47294)[)](https://github.com/ray-project/ray/commit/aa7179a6fa24a0d95a1c9b85014bfb322d3447e6)
  - Offline RL: Replace GAE in MARWILOfflinePreLearner with `GeneralAdvantageEstimation` connector in learner pipeline. ([#47532](https://github.com/ray-project/ray/pull/47532))
  - Off-policy algos: Add episode sampling to EpisodeReplayBuffer. ([#47500](https://github.com/ray-project/ray/pull/47500))
  - RLModule APIs: Add `SelfSupervisedLossAPI` for RLModules[ that bri](https://github.com/ray-project/ray/commit/f422376cda3ae0dc52fc7686df3b1cb03342be7f)ng their own loss and `InferenceOnlyAPI`. ([#47581](https://github.com/ray-project/ray/pull/47581), [#47572](https://github.com/ray-project/ray/pull/47572))

## Ray Core<a id="ray-core"></a>
💫 Enhancements:
- [aDAG] Allow custom NCCL group for aDAG (#47141)
- [aDAG] support buffered input (#47272)
- [aDAG] Support multi node multi reader (#47480)
- [Core] Make is_gpu, is_actor, root_detached_id fields late bind to workers. (#47212)
- [Core] Reconstruct actor to run lineage reconstruction triggered actor task (#47396)
- [Core] Optimize GetAllJobInfo API for performance (#47530)

🔨 Fixes:
- [aDAG] Fix ranks ordering for custom NCCL group (#47594)

## Ray Clusters<a id="ray-clusters"></a>
📖 Documentation:
- [KubeRay] add a guide for deploying vLLM with RayService (#47038) 

# Thanks

Many thanks to all those who contributed to this release!
@ruisearch42, @andrewsykim, @timkpaine, @rkooo567, @WeichenXu123, @GeneDer, @sword865, @simonsays1980, @angelinalg, @sven1977, @jjyao, @woshiyyya, @aslonnie, @zcin, @omatthew98, @rueian, @khluu, @justinvyu, @bveeramani, @nikitavemuri, @chris-ray-zhang, @liuxsh9, @xingyu-long, @peytondmurray, @rynewang

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.37.0)

---

## ray-2.36.1: Ray-2.36.1
**Published:** 2024-09-23

## Ray Core<a id="ray-core"></a>
🔨 Fixes:
- Fix broken dashboard cluster page when there are dead nodes (#47701)
- Fix broken dashboard worker page (#47714)



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.36.1)

---

## ray-2.36.0: Ray-2.36.0
**Published:** 2024-09-17

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>
💫 Enhancements:
- Remove limit on number of tasks launched per scheduling step (#47393)
- Allow user-defined Exception to be caught. (#47339)

🔨 Fixes:
- Display pending actors separately in the progress bar and not count them towards running resources (#46384)
- Fix bug where `arrow_parquet_args` aren't used (#47161)
- Skip empty JSON files in `read_json()` (#47378)
- Remove remote call for initializing `Datasource` in `read_datasource()` (#47467)
- Remove dead `from_*_operator` modules (#47457)
- Release test fixes
- Add `AWS ACCESS_DENIED` as retryable exception for multi-node Data+Train benchmarks (#47232)
- Get AWS credentials with boto (#47352)
- Use worker node instead of head node for `read_images_comparison_microbenchmark_single_node` release test (#47228)

📖 Documentation:
- Add docstring to explain `Dataset.deserialize_lineage` (#47203)
- Add a comment explaining the bundling behavior for `map_batches` with default batch_size (#47433)

## Ray Train<a id="ray-train"></a>

💫 Enhancements:
- Decouple device-related modules and add Huawei NPU support to Ray Train (#44086)

🔨 Fixes:
- Update TORCH_NCCL_ASYNC_ERROR_HANDLING env var (#47292)

📖 Documentation:
- Add missing Train public API reference (#47134)


## Ray Tune<a id="ray-tune"></a>
📖 Documentation:
- Add missing Tune public API references (#47138)


## Ray Serve<a id="ray-serve"></a>
💫 Enhancements:
- Mark proxy as unready when its routers are aware of zero replicas (#47002)
- Setup default serve logger (#47229)

🔨 Fixes:
- Allow get_serve_logs_dir to run outside of Ray's context (#47224)
- Use serve logger name for logs in serve (#47205)

📖 Documentation:
- [HPU] [Serve] [experimental] Add vllm HPU support in vllm example (#45893)

🏗 Architecture refactoring:
- Remove support for nested DeploymentResponses (#47209)

## RLlib<a id="rllib"></a>
🎉 New Features:
- New API stack: Add CQL algorithm. ([#47000](https://github.com/ray-project/ray/pull/47000), [#47402](https://github.com/ray-project/ray/pull/47402))
- New API stack: Enable GPU and multi-GPU support for DQN/SAC/CQL. ([#47179](https://github.com/ray-project/ray/pull/47179))

💫 Enhancements:
- New API stack: Offline RL enhancements: [#47195](https://github.com/ray-project/ray/pull/47195), [#47359](https://github.com/ray-project/ray/pull/47359)
- Enhance new API stack stability: [#46324](https://github.com/ray-project/ray/pull/46324), [#47196](https://github.com/ray-project/ray/pull/47196), [#47245](https://github.com/ray-project/ray/pull/47245), [#47279](https://github.com/ray-project/ray/pull/47279)
- Fix large batch size for synchronous algos (e.g. PPO) after EnvRunner failures. ([#47356](https://github.com/ray-project/ray/pull/47356))
- Add torch.compile config options to old API stack. ([#47340](https://github.com/ray-project/ray/pull/47340)[)](https://github.com/ray-project/ray/commit/78402bc1fde669cb4015f9393106acb144ac45bf)
- Add kwargs to torch.nn.parallel.DistributedDataParallel ([#47276](https://github.com/ray-project/ray/pull/47276))
- Enhanced CI stability: [#47197](https://github.com/ray-project/ray/pull/47197), [#47249](https://github.com/ray-project/ray/pull/47249)

📖 Documentation:
- New API stack example scripts:
  - Float16 training example script. ([#47362](https://github.com/ray-project/ray/pull/47362))
  - Mixed precision training example script ([#47116](https://github.com/ray-project/ray/pull/47116))
  - ModelV2 -> RLModule wrapper for migrating to new API stack. ([#47425](https://github.com/ray-project/ray/pull/47425))
- Remove "new API stack experimental" hint from docs. ([#47301](https://github.com/ray-project/ray/pull/47301))

🏗 Architecture refactoring:
- Remove 2nd Learner ConnectorV2 pass from PPO ([#47401](https://github.com/ray-project/ray/pull/47401))
- Add separate learning rates for policy and alpha to SAC. ([#47078](https://github.com/ray-project/ray/pull/47078))

🔨 Fixes:
- Various bug fixes: [#47401](https://github.com/ray-project/ray/pull/47401), [#47194](https://github.com/ray-project/ray/pull/47194), [#47259](https://github.com/ray-project/ray/pull/47259), [#47271](https://github.com/ray-project/ray/pull/47271), [#47277](https://github.com/ray-project/ray/pull/47277), [#47382](https://github.com/ray-project/ray/pull/47382)

## Ray Core<a id="ray-core"></a>
💫 Enhancements:
- [ADAG] Raise proper error message for nccl within the same actor (#47250)
- [[ADAG] Support multi-read of the same shm channel (](https://github.com/ray-project/ray/commit/c9c150a1f0460589e76415710afa5e940d834311)[#47311](https://github.com/ray-project/ray/pull/47311)[)](https://github.com/ray-project/ray/commit/c9c150a1f0460589e76415710afa5e940d834311)
- [Log why core worker is not idle during HandleExit (](https://github.com/ray-project/ray/commit/43250f4481849ae0233b96e68637b294dc43748d)[#47300](https://github.com/ray-project/ray/pull/47300)[)](https://github.com/ray-project/ray/commit/43250f4481849ae0233b96e68637b294dc43748d)
- Add PREPARED state for placement groups in GCS for better fault tolerance. ([#46858](https://github.com/ray-project/ray/pull/46858))

🔨 Fixes:
- Fix ray_unintentional_worker_failures_total to only count unintentional worker failures (#47368)
- Fix runtime env race condition when uploading the same package concurrently (#47482)

## Dashboard<a id="dashboard"></a>
🔨 Fixes:
- Performance optimizations for dashboard backend logic ([#47392](https://github.com/ray-project/ray/pull/47392)) ([#47367](https://github.com/ray-project/ray/pull/47367)) ([#47160](https://github.com/ray-project/ray/pull/47160)) (#47213)
- Refactor to simplify dashboard backend logic ([#47324](https://github.com/ray-project/ray/pull/47324))

## Docs<a id="dashboard"></a>

💫 Enhancements:
- Add sphinx-autobuild and documentation for make local (#47275): Speed up of local docs builds with `make local`.
- Add Algolia search to docs ([#46477](https://github.com/ray-project/ray/pull/46477))
- Update PyTorch Mnist Training doc for KubeRay 1.2.0 ([#47321](https://github.com/ray-project/ray/pull/47321))
- Life-cycle of documentation [policy](https://docs.ray.io/en/latest/ray-contribute/api-policy.html) of Ray APIs

# Thanks

Many thanks to all those who contributed to this release!
@GeneDer, @Bye-legumes, @nikitavemuri, @kevin85421, @MortalHappiness, @LeoLiao123, @saihaj, @rmcsqrd, @bveeramani, @zcin, @matthewdeng, @raulchen, @mattip, @jjyao, @ruisearch42, @scottjlee, @can-anyscale, @khluu, @aslonnie, @rynewang, @edoakes, @zhanluxianshen, @venkatram-dev, @c21, @allenyin55, @alexeykudinkin, @snehakottapalli, @BitPhinix, @hongchaodeng, @dengwxn, @liuxsh9, @simonsays1980, @peytondmurray, @KepingYan, @bryant1410, @woshiyyya, @sven1977

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.36.0)

---

## ray-2.35.0: Ray-2.35.0
**Published:** 2024-08-28

**Notice**: Starting from this release, `pip install ray[all]` will not include `ray[cpp]`, and will not install the respective `ray-cpp` package. To install everything that includes `ray-cpp`, one can use `pip install ray[cpp-all]` instead.

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>
🎉 New Features:
- Upgrade supported Arrow version from 16 to 17 (#47034)
- Add support for reading from Iceberg (#46889)

💫 Enhancements:
- Various Progress Bar UX improvements (#46816, #46801, #46826, #46692, #46699, #46974, #46928, #47029, #46924, #47120, #47095, #47106)
- Try get `size_bytes` from metadata and consolidate metadata methods (#46862)
- Improve warning message when read task is large (#46942)
- Extend API to enable passing sample weights via ray.dataset.to_tf (#45701)
- Add a parameter to allow overriding LanceDB scanner options (#46975)
- Add failure retry logic for read_lance (#46976)
- Clarify warning for reading old Parquet data (#47049)
- Move datasource implementations to `_internal` subpackage (#46825)
- Handle logs from tensor extensions (#46943)

🔨 Fixes:
- Change type of `DataContext.retried_io_errors` from tuple to list (#46884)
- Make Parquet tests more robust and expose Parquet logic (#46944)
- Change pickling log level from warning to debug (#47032)
- Add validation for shuffle arg (#47055)
- Fix validation bug when size=0 in ActorPoolStrategy (#47072)
- Fix exception in async map (#47110)
- Fix wrong metrics group for `Object Store Memory` metrics on Ray Data Dashboard (#47170)
- Handle errors in SplitCoordinator when generating a new epoch (#47176)

📖 Documentation:
- Auto-gen GroupedData api (#46925)
- Fix signature of `Rule.plan` (#47094)

## Ray Train<a id="ray-train"></a>
💫 Enhancements:
- [train] Updates to support xgboost==2.1.0 (#46667)
- [train] Add hardware stats (#46719)

## Ray Tune<a id="ray-tune"></a>
🔨 Fixes:
- [RLlib; Tune] Fix WandB metric overlap after restore from checkpoint. (#46897)

## Ray Serve<a id="ray-serve"></a>
💫 Enhancements:
- Improved handling of replica death and replica unavailability in deployment handle routers before controller restarts replica (#47008)
- Eagerly create routers in proxy for better GCS fault tolerance (#47031)
- Immediately send ping in router when receiving new replica set (#47053)

🏗 Architecture refactoring:
- Deprecate passing arguments that contain `DeploymentResponses` in nested objects to downstream deployment handle calls (#46806)

## RLlib<a id="rllib"></a>

🎉 New Features:
- Offline RL on the new API stack:
  - Record offline data (#46818, #47046, #47133, #47155) and support to directly read from episodes. (#46865)
  - RLUnplugged example. (#46792)
  - Progress on BC/MARWIL migration: #44970, #47154, #46799
  - Progress on CQL migration:  #46969, #47105

💫 Enhancements:
- Add ObservationPreprocessor (ConnectorV2). (#47077)

🔨 Fixes:
- New API stack: Fix IMPALA/APPO + LSTM for single- and multi-GPU. (#47132, #47158)
- Various bug fixes: #46898, #47047, #46963, #47021, #46897
- Add more control to Algorithm.add_module/policy methods. (#46932, #46836)

📖 Documentation:
- Example scripts for new API stack:
  - Curiosity (inverse dynamics model-based) RLModule example. (#46841)
  - Add example script for Env with protobuf observation space. (#47071)
- New API stack documentation:
  - Cleanup old API stack docs (rllib-dev.rst). (#47172)
  - Episodes (SingleAgentEpisode). (#46985)
  - Redo rllib-algorithms.rst page. (#46916)

🏗 Architecture refactoring:
- Rename MultiAgent...RLModule... into MultiRL...Module for more generality. (#46840)
- Add learner_only flag to RLModuleConfig/Spec and simplify creation of RLModule specs from algo-config. (#46900)

## Ray Core<a id="ray-core"></a>
💫 Enhancements:
- Emit total lineage bytes metrics (#46725)
- Adding accelerator type H100 (#46823)
- More structured logging in core worker (#46906)
- Change all callbacks to move to save copies. (#46971) 
- Add ray[adag] option to pip install (#47009) 

🔨 Fixes:
- Fix dashboard process reporting on windows (#45578) 
- Fix Ray-on-Spark cluster crashing bug when user cancels cell execution (#46899)
- Fix PinExistingReturnObject segfault by passing owner_address (#46973)
- Fix raylet CHECK failure from runtime env creation failure. (#46991)
- Fix typo in memray command (#47006)
- [ADAG] Fix for asyncio outputs (#46845)

📖 Documentation:
- Clarify behavior of placement_group_capture_child_tasks in docs (#46885)
- Update ray.available_resources() docstring (#47018) 

🏗 Architecture refactoring:
- Async APIs for the New GcsClient. (#46788) 
- Replace GCS stubs in the dashboard to use NewGcsAioClient. (#46846) 

## Dashboard<a id="dashboard"></a>

💫 Enhancements: 
- Polish and minor improvements to the Serve page (#46811)

🔨 Fixes: 
- Fix CPU/GPU/RAM not being reported correctly on Windows (#44578)

## Docs<a id="dashboard"></a>

💫 Enhancements: 
- Add more information about developer tooling for docs contributions (#46636), including `esbonio` section

🔨 Fixes: 
- Use PyData Sphinx theme version switcher (#46936)

# Thanks

Many thanks to all those who contributed to this release!
@simonsays1980, @bveeramani, @tungh2, @zcin, @xingyu-long, @WeichenXu123, @aslonnie, @MaxVanDijck, @can-anyscale, @galenhwang, @omatthew98, @matthewdeng, @raulchen, @sven1977, @shrekris-anyscale, @deepyaman, @alexeykudinkin, @stephanie-wang, @kevin85421, @ruisearch42, @hongchaodeng, @khluu, @alanwguo, @hongpeng-guo, @saihaj, @Superskyyy, @tespent, @slfan1989, @justinvyu, @rynewang, @nikitavemuri, @amogkam, @mattip, @dev-goyal, @ryanaoleary, @peytondmurray, @edoakes, @venkatajagannath, @jjyao, @cristianjd, @scottjlee, @Bye-legumes

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.35.0)

---

## ray-2.34.0: Release 2.34.0 Notes
**Published:** 2024-07-31

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>
💫 Enhancements:
  - Add better support for UDF returns from list of datetime objects (#46762)
  
🔨 Fixes:
  - Remove read task warning if size bytes not set in metadata (#46765)

📖 Documentation:
  - Fix read_tfrecords() docstring to display tfx-bsl tip (#46717)
  - Update Dataset.zip() docs (#46757)


## Ray Train<a id="ray-train"></a>
🔨 Fixes:
  - Sort workers by node ID rather than by node IP (#46163)

🏗 Architecture refactoring:
  - Remove dead RayDatasetSpec (#46764)

## RLlib<a id="rllib"></a>

🎉 New Features:
  - Offline RL support on new API stack:
     - Initial design for Ray-Data based offline RL Algos (on new API stack). (#44969)
     - Add user-defined schemas for data loading. (#46738)
     - Make data pipeline better configurable and tuneable for users. (#46777)

💫 Enhancements:
- Move DQN into the TargetNetworkAPI (and deprecate `RLModuleWithTargetNetworksInterface`). (#46752)

🔨 Fixes:
- Numpy version fix: Rename all np.product usage to np.prod (#46317)

📖 Documentation:
- Examples for new API stack: Add 2 (count-based) curiosity examples. (#46737)
- Remove RLlib CLI from docs (soon to be deprecated and replaced by python API). (#46724)

🏗 Architecture refactoring:
- Cleanup, rename, clarify: Algorithm.workers/evaluation_workers, local_worker(), etc.. (#46726)

# Ray Core<a id="ray-core"></a>

🏗 Architecture refactoring:
- New python GcsClient binding (#46186)




Many thanks to all those who contributed to this release! @KyleKoon, @ruisearch42, @rynewang, @sven1977, @saihaj, @aslonnie, @bveeramani, @akshay-anyscale, @kevin85421, @omatthew98, @anyscalesam, @MaxVanDijck, @justinvyu, @simonsays1980, @can-anyscale, @peytondmurray, @scottjlee

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.34.0)

---

## ray-2.33.0: Ray-2.33.0
**Published:** 2024-07-25

# Ray Libraries<a id="ray-libraries"></a>

# Ray Core<a id="ray-core"></a>

💫 Enhancements:
- Add "last exception" to error message when GCS connection fails in ray.init() (#46516)

🔨 Fixes:
- Add object back to memory store when object recovery is skipped (#46460)
- Task status should start with PENDING_ARGS_AVAIL when retry (#46494)
- Fix ObjectFetchTimedOutError ([#46562](https://github.com/ray-project/ray/pull/46562))
- Make working_dir support files created before 1980 ([#46634](https://github.com/ray-project/ray/pull/46634))
- Allow full path in conda runtime env. ([#45550](https://github.com/ray-project/ray/pull/45550))
- Fix worker launch time formatting in state api ([#43516](https://github.com/ray-project/ray/pull/43516))
- 
## Ray Data<a id="ray-data"></a>
🎉 New Features:
- Deprecate Dataset.get_internal_block_refs() (#46455)
- Add read API for reading Databricks table with Delta Sharing (#46072)
- Add support for objects to Arrow blocks (#45272)

💫 Enhancements:
- Change offsets to int64 and change to LargeList for ArrowTensorArray (#45352)
- Prevent from_pandas from combining input blocks (#46363)
- Update Dataset.count() to avoid unnecessarily keeping `BlockRef`s in-memory (#46369)
- Use Set to fix inefficient iteration over Arrow table columns (#46541)
- Add AWS Error UNKNOWN to list of retried write errors (#46646)
- Always print traceback for internal exceptions (#46647)
- Allow unknown estimate of operator output bundles and `ProgressBar` totals (#46601)
- Improve filesystem retry coverage (#46685)

🔨 Fixes:
- Replace lambda mutable default arguments (#46493)

📖 Documentation:
- Auto-generate Dataset API documentation (#46557)
- Update outdated ExecutionPlan docstring (#46638)


## Ray Train<a id="ray-train"></a>
💫 Enhancements:
- Update run status and actor status for train runs. (#46395)

🔨 Fixes:
- Replace lambda default arguments (#46576)

📖 Documentation:
- Add MNIST training using KubeRay doc page (#46123)
- Add example of pre-training Llama model on Intel Gaudi (#45459)
- Fix tensorflow example by using ScalingConfig (#46565)

## Ray Tune<a id="ray-tune"></a>
🔨 Fixes:
- Replace lambda default arguments (#46596)

## Ray Serve<a id="ray-serve"></a>

🎉 New Features:
- Fully deprecate `target_num_ongoing_requests_per_replica` and `max_concurrent_queries`, respectively replaced by `max_ongoing_requests` and `target_ongoing_requests` (#46392 and #46427)
- Configure the task launched by the controller to build an application with Serve’s logging config (#46347)

## RLlib<a id="rllib"></a>

💫 Enhancements:
- Moving sampling coordination for `batch_mode=complete_episodes` to `synchronous_parallel_sample`. (#46321)
- Enable complex action spaces with stateful modules. (#46468)

🏗 Architecture refactoring:
- Enable multi-learner setup for hybrid stack BC. (#46436)
- Introduce Checkpointable API for RLlib components and subcomponents. (#46376)

🔨 Fixes:
- Replace Mapping typehint with Dict: #46474

📖 Documentation:
- More example scripts for new API stack: Two separate optimizers (w/ different learning rates). (#46540) and custom loss function. (#46445)

## Dashboard<a id="dashboard"></a>

🔨 Fixes:
- Task end time showing the incorrect time (#46439)
- Events Table rows having really bad spacing (#46701)
- UI bugs in the serve dashboard page (#46599)

# Thanks

Many thanks to all those who contributed to this release!

@alanwguo, @hongchaodeng, @anyscalesam, @brucebismarck, @bt2513, @woshiyyya, @terraflops1048576, @lorenzoritter, @omrishiv, @davidxia, @cchen777, @nono-Sang, @jackhumphries, @aslonnie, @JoshKarpel, @zjregee, @bveeramani, @khluu, @Superskyyy, @liuxsh9, @jjyao, @ruisearch42, @sven1977, @harborn, @saihaj, @zcin, @can-anyscale, @veekaybee, @chungen04, @WeichenXu123, @GeneDer, @sergey-serebryakov, @Bye-legumes, @scottjlee, @rynewang, @kevin85421, @cristianjd, @peytondmurray, @MortalHappiness, @MaxVanDijck, @simonsays1980, @mjovanovic9999

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.33.0)

---

## ray-2.32.0: Ray-2.32.0
**Published:** 2024-07-10

# Highlight: aDAG Developer Preview <a id="highlilght"></a>

This is a new Ray Core specific feature called Ray accelerated DAGs (aDAGs). 

- aDAGs give you a Ray Core-like API but with extensibility to pre-compile execution paths across pre-allocated resources on a Ray Cluster to possible benefits for optimization on throughput and latency. Some practical examples include:
    - Up to 10x lower task execution time on single-node.
    - Native support for GPU-GPU communication, via NCCL.
- This is still very early, but please reach out on #ray-core on Ray Slack to learn more!

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

💫 Enhancements:
- Support async callable classes in `map_batches()` (#46129)

🔨 Fixes:
- Ensure `InputDataBuffer` doesn't free block references (#46191) 
- `MapOperator.num_active_tasks` should exclude pending actors (#46364)
- Fix progress bars being displayed as partially completed in Jupyter notebooks (#46289)

📖 Documentation:
- Fix docs: `read_api.py` docstring (#45690) 
- Correct API annotation for `tfrecords_datasource` (#46171) 
- Fix broken links in `README` and in `ray.data.Dataset` (#45345) 

## Ray Train<a id="ray-train"></a>

📖 Documentation:
- Update PyTorch Data Ingestion User Guide (#45421)

## Ray Serve<a id="ray-serve"></a>

💫 Enhancements:
- Optimize `ServeController.get_app_config()` (#45878)
- Change default for max and target ongoing requests (#45943)
- Integrate with Ray structured logging (#46215)
- Allow configuring handle cache size and controller max concurrency (#46278)
- Optimize `DeploymentDetails.deployment_route_prefix_not_set()` (#46305)

## RLlib<a id="rllib"></a>

🎉 New Features:
- APPO on new API stack (w/ `EnvRunners`). (#46216)

💫 Enhancements:
- Stability: APPO, SAC, and DQN activate multi-agent learning tests (#45542, #46299)
- Make Tune trial ID available in `EnvRunners` (and callbacks). (#46294)
- Add `env-` and `agent_steps` to custom evaluation function. (#45652)
- Remove default-metrics from Algorithm (tune does NOT error anymore if any stop-metric is missing). (#46200)

🔨 Fixes:
- Various bug fixes: #45542

📖 Documentation:
- Example for new API stack: Offline RL (BC) training on single-agent, while evaluating w/ multi-agent setup. (#46251)
- Example for new API stack: Custom RLModule with an LSTM. (#46276)

# Ray Core<a id="ray-core"></a>

🎉 New Features:
- aDAG Developer Preview.

💫 Enhancements:
- Allow env setup logger encoding (#46242)
- ray list tasks filter state and name on GCS side (#46270)
- Log ray version and ray commit during GCS start (#46341)

🔨 Fixes:
- Decrement lineage ref count of an actor when the actor task return object reference is deleted (#46230)
- Fix negative ALIVE actors metric and introduce IDLE state (#45718)
- `psutil` process attr `num_fds` is not available on Windows (#46329)

## Dashboard<a id="dashboard"></a>

🎉 New Features:
- Added customizable refresh frequency for metrics on Ray Dashboard (#44037)

💫 Enhancements:
- Upgraded to MUIv5 and React 18 (#45789)

🔨 Fixes:
- Fix for multi-line log items breaking log viewer rendering (#46391)
- Fix for UI inconsistency when a job submission creates more than one Ray job. (#46267)
- Fix filtering by job id for tasks API not filtering correctly. (#45017)

## Docs<a id="docs"></a>

🔨 Fixes:
- Re-enabled automatic cross-reference link checking for Ray documentation, with Sphinx nitpicky mode (#46279)
- Enforced naming conventions for public and private APIs to maintain accuracy, starting with Ray Data API documentation (#46261)

📖 Documentation:
- Upgrade Python 3.12 support to alpha, marking the release of the Ray wheel to PyPI and conducting a sanity check of the most critical tests.

# Thanks

Many thanks to all those who contributed to this release!

@stephanie-wang, @MortalHappiness, @aslonnie, @ryanaoleary, @jjyao, @jackhumphries, @nikitavemuri, @woshiyyya, @JoshKarpel, @ruisearch42, @sven1977, @alanwguo, @GeneDer, @saihaj, @raulchen, @liuxsh9, @khluu, @cristianjd, @scottjlee, @bveeramani, @zcin, @simonsays1980, @SumanthRH, @davidxia, @can-anyscale, @peytondmurray, @kevin85421

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.32.0)

---

## ray-2.31.0: Ray-2.31.0
**Published:** 2024-06-26

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🔨 Fixes:
- Fixed bug where `preserve_order` doesn’t work with file reads ([#46135](https://github.com/ray-project/ray/pull/46135))

📖 Documentation:
- Added documentation for `dataset.Schema` ([#46170](https://github.com/ray-project/ray/pull/46170))

## Ray Train<a id="ray-train"></a>

💫 Enhancements:
- Add API for Ray Train run stats (#45711)

## Ray Tune<a id="ray-tune"></a>

💫 Enhancements:
- Missing stopping criterion should not error (just warn). (#45613)

📖 Documentation:
- Fix broken references in Ray Tune documentation (#45233)

## Ray Serve<a id="ray-serve"></a>

**WARNING**: the following default values will change in Ray 2.32:
  - Default for `max_ongoing_requests` will change from 100 to 5.
  - Default for `target_ongoing_requests` will change from 1 to 2.

💫 Enhancements:
- Optimize DeploymentStateManager.get_deployment_statuses ([#45872](https://github.com/ray-project/ray/pull/45872))

🔨 Fixes:
- Fix logging error on passing traceback object into exc_info ([#46105](https://github.com/ray-project/ray/pull/46105))
- Run __del__ even if constructor is still in-progress ([#45882](https://github.com/ray-project/ray/pull/45882))
- Spread replicas with custom resources in torch tune serve release test ([#46093](https://github.com/ray-project/ray/pull/46093))
- [1k release test] don't run replicas on head node ([#46130](https://github.com/ray-project/ray/pull/46130))

📖 Documentation:
- Remove todo since issue is fixed ([#45941](https://github.com/ray-project/ray/pull/45941))

## RLlib<a id="rllib"></a>

🎉 New Features:
- IMPALA runs on the new API stack (with EnvRunners and ConnectorV2s). ([#42085](https://github.com/ray-project/ray/pull/42085))
- SAC/DQN: Prioritized multi-agent episode replay buffer. ([#45576](https://github.com/ray-project/ray/pull/45576)[)](https://github.com/ray-project/ray/commit/a7aa5e4c7bc4654b5ce710bf3551e3775470b57b)

💫 Enhancements:
- New API stack stability: Add systematic CI learning tests for all possible combinations of: [PPO|IMPALA] + [1CPU|2CPU|1GPU|2GPU] + [single-agent|multi-agent]. ([#46162](https://github.com/ray-project/ray/pull/46162), [#46161](https://github.com/ray-project/ray/pull/46161))

📖 Documentation:
- New API stack: Example script for action masking [(](https://github.com/ray-project/ray/commit/9d661677f942ccd7dfbbf2c2838e5dad9f50d12e)[#46146](https://github.com/ray-project/ray/pull/46146))
- New API stack: PyFlight example script cleanup ([#45956](https://github.com/ray-project/ray/pull/45956)[)](https://github.com/ray-project/ray/commit/af45a8999b5d9ca09054de29c2bdd0b5b02184b5)
- Old API stack: Enhanced ONNX example (+LSTM). ([#43592](https://github.com/ray-project/ray/pull/43592)[)](https://github.com/ray-project/ray/commit/d15204f5a0d5f2fa70935efa4935a7a61d8aafd1)

# Ray Core and Ray Clusters<a id="ray-core-and-ray-clusters"></a>

## Ray Core<a id="ray-core"></a>

💫 Enhancements:
- [runtime-env] automatically infer worker path when starting worker in container ([#42304](https://github.com/ray-project/ray/pull/42304))

🔨 Fixes:
- On GCS restart, destroy not forget the unused workers. Fixing PG leaks. (#45854)
- Cancel lease requests before returning a PG bundle ([#45919](https://github.com/ray-project/ray/pull/45919))
- Fix boost fiber stack overflow (#46133)


# Thanks

Many thanks to all those who contributed to this release!

@jjyao, @kevin85421, @vincent-pli, @khluu, @simonsays1980, @sven1977, @rynewang, @can-anyscale, @richardsliu, @jackhumphries, @alexeykudinkin, @bveeramani, @ruisearch42, @shrekris-anyscale, @stephanie-wang, @matthewdeng, @zcin, @hongchaodeng, @ryanaoleary, @liuxsh9, @GeneDer, @aslonnie, @peytondmurray, @Bye-legumes, @woshiyyya, @scottjlee, @JoshKarpel

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.31.0)

---

## ray-2.30.0: Ray-2.30.0
**Published:** 2024-06-20

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

💫 Enhancements:
- Improve fractional CPU/GPU formatting (#45673)
- Use sampled fragments to estimate Parquet reader batch size (#45749)
- Refactoring ParquetDatasource and metadata fetching logic (#45728, #45727, #45733, #45734, #45767)
- Refactor planner.py (#45706)


## Ray Tune<a id="ray-tune"></a>
💫 Enhancements:
- Change the behavior of a missing stopping criterion metric to warn instead of raising an error. This enables the use case of reporting different sets of metrics on different iterations (ex: a separate set of training and validation metrics). ([#45613](https://github.com/ray-project/ray/pull/45613))

## Ray Serve<a id="ray-serve"></a>

💫 Enhancements:
- Create internal request id to track request objects ([#45761](https://github.com/ray-project/ray/pull/45761))

## RLLib<a id="rllib"></a>

💫 Enhancements:
- Stability: [DreamerV3 weekly release test](https://github.com/ray-project/ray/commit/4adb78b2bf3c968f88f72ae9064189b846833230) ([#45654](https://github.com/ray-project/ray/pull/45654)); [Add "official" benchmark script for Atari PPO benchmarks](https://github.com/ray-project/ray/commit/d49f15b1112e67d15a80d696249f587ea7b95b57). ([#45697](https://github.com/ray-project/ray/pull/45697))
- Enhance env-rendering callback (#45682)

🔨 Fixes:
- Bug fix in new MetricsLogger API: [EMA stats w/o window would lead to infinite l](https://github.com/ray-project/ray/commit/cbb1634a23ff4b59090f43dd853cf437e19fc0c8)ist mem-leak. ([#45752](https://github.com/ray-project/ray/pull/45752))
- Various other bug fixes: ([#45819](https://github.com/ray-project/ray/pull/45819), [#45820](https://github.com/ray-project/ray/pull/45820), #[45683](https://github.com/ray-project/ray/pull/45683), [#45651](https://github.com/ray-project/ray/pull/45651), [#45753](https://github.com/ray-project/ray/pull/45753))

📖 Documentation:
- Re-do `examples` overview page (new API stack): [#45382](https://github.com/ray-project/ray/pull/45382)
  - PyFlyt QuadX WayPoints example [#44758](https://github.com/ray-project/ray/pull/44758), [#45956](https://github.com/ray-project/ray/pull/45956)
  - RLModule inference on new API stack ([#45831](https://github.com/ray-project/ray/pull/45831), [#45845](https://github.com/ray-project/ray/pull/45845))
  - How to resume a tune.Tuner.fit() experiment from checkpoint. ([#45681](https://github.com/ray-project/ray/pull/45681))
  - Custom RLModule (tiny CNN): [#45774](https://github.com/ray-project/ray/pull/45774)
  - Connector examples docstrings ([#45864](https://github.com/ray-project/ray/pull/45864))
- Old API stack examples: [#43592](https://github.com/ray-project/ray/pull/43592), [#45829](https://github.com/ray-project/ray/pull/45829)

## Ray Core<a id="ray-core"></a>
🎉 New Features:
- Alpha release of job level [logging configuration](https://docs.ray.io/en/master/ray-core/api/doc/ray.LoggingConfig.html#ray.LoggingConfig): users can now config the user logging to be logfmt format with logging context attached. (#45344)

💫 Enhancements:
- Integrate amdsmi in AMDAcceleratorManager (#44572)

🔨 Fixes:
- Fix the C++ GcsClient Del not respecting del_by_prefix (#45604)
- Fix exit handling of FiberState threads (#45834) 

## Dashboard<a id="ray-dashboard"></a>
💫 Enhancements:
- Parse out json logs (#45853) 

Many thanks to all those who contributed to this release: @liuxsh9, @peytondmurray, @pcmoritz, @GeneDer, @saihaj, @khluu, @aslonnie, @yucai, @vickytsang, @can-anyscale, @bthananjeyan, @raulchen, @hongchaodeng, @x13n, @simonsays1980, @peterghaddad, @kevin85421, @rynewang, @angelinalg, @jjyao, @BenWilson2, @jackhumphries, @zcin, @chris-ray-zhang, @c21, @shrekris-anyscale, @alanwguo, @stephanie-wang, @Bye-legumes, @sven1977, @WeichenXu123, @bveeramani, @nikitavemuri



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.30.0)

---

## ray-2.24.0: Ray-2.24.0
**Published:** 2024-06-06

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉 New Features:
- Allow user to configure timeout for actor pool (#45508)
- Add override_num_blocks to from_pandas and perform auto-partition (#44937)
- Upgrade Arrow version to 16 in CI (#45565)

💫 Enhancements:
- Clarify that num_rows_per_file isn't strict (#45529)
- Record more telemetry for newly added datasources (#45647)
- Avoid pickling LanceFragment when creating read tasks for Lance (#45392)

## Ray Train<a id="ray-train"></a>

📖 Documentation:
- [HPU] Add example of Stable Diffusion fine-tuning and serving on Intel Gaudi (#45217)
- [HPU] Add example of Llama-2 fine-tuning on Intel Gaudi (#44667)


## Ray Tune<a id="ray-tune"></a>
🏗 Architecture refactoring:
- Improve excessive syncing warning and deprecate TUNE_RESULT_DIR, RAY_AIR_LOCAL_CACHE_DIR, local_dir (#45210)

## Ray Serve<a id="ray-serve"></a>

💫 Enhancements:
- Clean up Serve proxy files ([#45486](https://github.com/ray-project/ray/pull/45486))

📖 Documentation:
- vllm example to serve llm models ([#45430](https://github.com/ray-project/ray/pull/45430))

## RLLib<a id="rllib"></a>

💫 Enhancements:
- DreamerV3 on tf: Bug fix, so it can run again with tf==2.11.1 (2.11.0 is not available anymore) (#45419); Added weekly release test for DreamerV3.
- Added support for multi-agent off-policy algorithms (DQN and SAC) in the new (#45182)
- Config option for APPO/IMPALA to change number of GPU-loader threads (#45467)

🔨 Fixes:
- Various MetricsLogger bug fixes (#45543, #45585, #45575)
- Other fixes: #45588, #45617, #45517, #45465

📖 Documentation:
- Example script for new API stack: How-to restore 1 of n agents from a checkpoint. (#45462)
- Example script for new API stack: Autoregressive action module. #45525

## Ray Core<a id="ray-core"></a>

💫 Enhancements:
- Improve node death observability (#45320, #45357, #45533, #45644, #45497)
- Ray c++ backend structured logging (#44468)

🔨 Fixes:
- Fix worker crash when getting actor name from runtime context (#45194)
- log dedup should not dedup number only lines (#45385)

📖 Documentation:
- Improve doc for `--object-store-memory` to describe how the default value is set ([#45301](https://github.com/ray-project/ray/pull/45301))

## Dashboard<a id="ray-dashboard"></a>
🔨 Fixes:
- Move Job package uploading to another thread to unblock the event loop. ([#45282](https://github.com/ray-project/ray/pull/45282))

Many thanks to all those who contributed to this release: @maxliuofficial, @simonsays1980, @GeneDer, @dudeperf3ct, @khluu, @justinvyu, @andrewsykim, @Catch-Bull, @zcin, @bveeramani, @rynewang, @angelinalg, @matthewdeng, @jjyao, @kira-lin, @harborn, @hongchaodeng, @peytondmurray, @aslonnie, @timkpaine, @982945902, @maxpumperla, @stephanie-wang, @ruisearch42, @alanwguo, @can-anyscale, @c21, @Atry, @KamenShah, @sven1977, @raulchen



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.24.0)

---

## ray-2.23.0: Ray-2.23.0
**Published:** 2024-05-22

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉 New Features:
- Add support for using GPUs with map_groups (#45305)
- Add support for using actors with map_groups (#45310)

💫 Enhancements:
- Refine exception handling from arrow data conversion (#45294)

🔨 Fixes:
- Fix Ray databricks UC reader with dynamic Databricks notebook scope token (#45153)
- Fix bug where you can't return objects and array from UDF (#45287 )
- Fix bug where map_groups triggers execution during input validation (#45314)

## Ray Tune<a id="ray-tune"></a>
🔨 Fixes:
- [tune] Fix PB2 scheduler error resulting from trying to sort by Trial objects (#45161)

## Ray Serve<a id="ray-serve"></a>
🔨 Fixes:
- Log application unhealthy errors at error level instead of warning level ([#45211](https://github.com/ray-project/ray/pull/45211))

## RLLib<a id="rllib"></a>
💫 Enhancements:
- Examples and `tuned_examples` learning test for new API stack are now “self-executable” (don’t require a third-party script anymore to run them). + WandB support. ([#45023](https://github.com/ray-project/ray/pull/45023))

🔨 Fixes:
- Fix result dict “spam” (duplicate, deprecated keys, e.g. “sampler_results” dumped into top level). ([#45330](https://github.com/ray-project/ray/pull/45330))

📖 Documentation:
- Add example for training with fractional GPUs on new API stack. ([#45379](https://github.com/ray-project/ray/pull/45379))
- Cleanup examples folder and remove deprecated sub directories. ([#45327](https://github.com/ray-project/ray/pull/45327))

## Ray Core<a id="ray-core"></a>
💫 Enhancements:
- [Logs] Add runtime env started logs to job driver ([#45255](https://github.com/ray-project/ray/pull/45255))
- `ray.util.collective` support `torch.bfloat16` ([#39845](https://github.com/ray-project/ray/pull/39845))
- [Core] Better propagate node death information ([#45128](https://github.com/ray-project/ray/pull/45128))

🔨 Fixes:
- [Core] Fix worker process leaks after job finishes ([#44214](https://github.com/ray-project/ray/pull/44214))


Many thanks to all those who contributed to this release: @hongchaodeng, @khluu, @antoni-jamiolkowski, @ameroyer, @bveeramani, @can-anyscale, @WeichenXu123, @peytondmurray, @jackhumphries, @kevin85421, @jjyao, @robcaulk, @rynewang, @scottsun94, @swang, @GeneDer, @zcin, @ruisearch42, @aslonnie, @angelinalg, @raulchen, @ArthurBook, @sven1977, @wuxibin89


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.23.0)

---

## ray-2.22.0: Ray-2.22.0
**Published:** 2024-05-14

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉 New Features:
- Add function to dynamically generate `ray_remote_args` for Map APIs (#45143)
- Allow manually setting resource limits for training jobs (#45188)

💫 Enhancements:
- Introduce abstract interface for data autoscaling (#45002)
- Add debugging info for `SplitCoordinator` (#45226)

🔨 Fixes:
- Don’t show `AllToAllOperator` progress bar if the disable flag is set (#45136)
- Don't load Arrow `PyExtensionType` by default (#45084)
- Don't raise batch size error if `num_gpus=0` (#45202)

## Ray Train<a id="ray-train"></a>

💫 Enhancements:
- [XGBoost][LightGBM] Update RayTrainReportCallback to only save checkpoints on rank 0 (#45083)

# Ray Core<a id="ray-core"></a>

🔨 Fixes:
- Fix the cpu percentage metrics for dashboard process (#45124)

## Dashboard<a id="dashboard"></a>

💫 Enhancements:
- Improvements to log viewer so line numbers do not get selected when copying text.
- Improvements to the log viewer to avoid unnecessary re-rendering which causes text selection to clear.


Many thanks to all those who contributed to this release: @justinvyu, @simonsays1980, @chris-ray-zhang, @kevin85421, @angelinalg, @rynewang, @brycehuang30, @alanwguo, @jjyao, @shaikhismail, @khluu, @can-anyscale, @bveeramani, @jrosti, @WeichenXu123, @MortalHappiness, @raulchen, @scottjlee, @ruisearch42, @aslonnie, @alexeykudinkin


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.22.0)

---

## ray-2.21.0: Ray-2.21.0
**Published:** 2024-05-08

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉 New features:
- Add `read_lance` API to read Lance Dataset (#45106)

🔨 Fixes:
- Retry RaySystemError application errors (#45079)

📖 Documentation:
- Fix broken references in data documentation (#44956)

## Ray Train<a id="ray-train"></a>

📖 Documentation:
- Fix broken links in Train documentation (#44953)

## Ray Tune<a id="ray-tune"></a>

📖 Documentation:
- Update Hugging Face example to add reference (#42771)

🏗 Architecture refactoring:
- Remove deprecated `ray.air.callbacks` modules (#45104)

## Ray Serve<a id="ray-serve"></a>

💫 Enhancements:
- Allow methods to pass type @serve.batch type hint (#45004)
- Allow configuring Serve control loop interval (#45063)

🔨 Fixes:
- Fix bug with controller failing to recover for autoscaling deployments (#45118)
- Fix control+c after serve run doesn't shutdown serve components (#45087)
- Fix lightweight update max ongoing requests (#45006)



## RLlib<a id="rllib"></a>

🎉 New Features:
- New MetricsLogger API now fully functional on the new API stack (working now also inside Learner classes, i.e. loss functions). ([#44995](https://github.com/ray-project/ray/pull/44995), [#45109](https://github.com/ray-project/ray/pull/45109))

💫 Enhancements:
- Renamings and cleanups (toward new API stack and more consistent naming schemata): WorkerSet -> EnvRunnerGroup, DEFAULT_POLICY_ID -> DEFAULT_MODULE_ID, config.rollouts() -> config.env_runners(), etc.. ([#45022](https://github.com/ray-project/ray/pull/45022), [#44920](https://github.com/ray-project/ray/pull/44920))
- Changed behavior of `EnvRunnerGroup.foreach_worker…` methods to new defaults: `mark_healthy=True` (used to be False) and `healthy_only=True` (used to be False). ([#44993](https://github.com/ray-project/ray/pull/44993))
- Fix `get_state()/from_state()` methods in SingleAgent- and MultiAgentEpisodes. ([#45012](https://github.com/ray-project/ray/pull/45012))

🔨 Fixes:
- Bug fix for (torch) global_norm clipping overflow problem: ([#45055](https://github.com/ray-project/ray/pull/45055))
- Various bug- and test case fixes: [#45030](https://github.com/ray-project/ray/pull/45030), [#45031](https://github.com/ray-project/ray/pull/45031), [#45070](https://github.com/ray-project/ray/pull/45070), [#45053](https://github.com/ray-project/ray/pull/45053), [#45110](https://github.com/ray-project/ray/pull/45110), [#45111](https://github.com/ray-project/ray/pull/45111)

📖 Documentation:
- Example scripts using the MetricsLogger for env rendering and recording w/ WandB: [#45073](https://github.com/ray-project/ray/pull/45073), [#45107](https://github.com/ray-project/ray/pull/45107)

# Ray Core<a id="ray-core"></a>

🔨 Fixes:
- Fix `ray.init(logging_format)` argument is ignored (#45037)
- Handle unserializable user exception (#44878)
- Fix dashboard process event loop blocking issues (#45048, #45047)

## Dashboard<a id="dashboard"></a>
🔨 Fixes:
- Fix Nodes page sorting not working correctly.
- Add back “actors per page” UI control in the actors page.

Many thanks to all those who contributed to this release: @rynewang, @can-anyscale, @scottsun94, @bveeramani, @ceddy4395, @GeneDer, @zcin, @JoshKarpel, @nikitavemuri, @stephanie-wang, @jackhumphries, @matthewdeng, @yash97, @simonsays1980, @peytondmurray, @evalaiyc98, @c21, @alanwguo, @shrekris-anyscale, @kevin85421, @hongchaodeng, @sven1977, @st--, @khluu

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.21.0)

---

## ray-2.20.0: Ray-2.20.0
**Published:** 2024-05-01

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

💫 Enhancements:
- Dedupe repeated schema during `ParquetDatasource` metadata prefetching (#44750)
- Update `map_groups` implementation to better handle large outputs (#44862)
- Deprecate `prefetch_batches` arg of `iter_rows` and change default value (#44982)
- Adding in default behavior to false for creating dirs on s3 writes (#44972)
- Make internal UDF names more descriptive (#44985)
- Make `name` a required argument for `AggregateFn` (#44880)

📖 Documentation:
- Add key concepts to and revise "Data Internals" page (#44751)


## Ray Train<a id="ray-train"></a>

💫 Enhancements:
- Setup XGBoost `CommunicatorContext` automatically (#44883)
- Track Train Run Info with `TrainStateActor` (#44585)

📖 Documentation:
- Add documentation for `accelerator_type` (#44882)
- Update Ray Train example titles (#44369)

## Ray Tune<a id="ray-tune"></a>

💫 Enhancements:
- Remove trial table when running Ray Train in a Jupyter notebook (#44858)
- Clean up temporary checkpoint directories for class Trainables (ex: RLlib) (#44366)

📖 Documentation:
- Fix minor doc format issues (#44865)
- Remove outdated ScalingConfig references (#44918)

## Ray Serve<a id="ray-serve"></a>

💫 Enhancements:
- Handle push metric interval is now configurable with environment variable RAY_SERVE_HANDLE_METRIC_PUSH_INTERVAL_S (#32920)
- Improve performance of developer API serve.get_app_handle (#44812)

🔨 Fixes:
- Fix memory leak in handles for autoscaling deployments (the leak happens when 
- RAY_SERVE_COLLECT_AUTOSCALING_METRICS_ON_HANDLE=1) (#44877)


## RLlib<a id="rllib"></a>

🎉 New Features:
- Introduce `MetricsLogger`, a unified API for users of RLlib to log custom metrics and stats in all of RLlib’s components (Algorithm, EnvRunners, and Learners). Rolled out for new API stack for Algorithm (`training_step`) and EnvRunners (custom callbacks). `Learner` (custom loss functions) support in progress. [#44888](https://github.com/ray-project/ray/pull/44888), [#44442](https://github.com/ray-project/ray/pull/44442)
- Introduce “inference-only” (slim) mode for RLModules that run inside an EnvRunner (and thus don’t require value-functions or target networks): [#44797](https://github.com/ray-project/ray/pull/44797)

💫 Enhancements:
- MultiAgentEpisodeReplayBuffer for new API stack (preparation for multi-agent support of SAC and DQN): [#44450](https://github.com/ray-project/ray/pull/44450)
- AlgorithmConfig cleanup and renaming of properties and methods for better consistency/transparency: [#44896](https://github.com/ray-project/ray/pull/44896)

🔨 Fixes:
- Various minor bug fixes: [#44989](https://github.com/ray-project/ray/pull/44989), [#44988](https://github.com/ray-project/ray/pull/44988), [#44891](https://github.com/ray-project/ray/pull/44891), [#44898](https://github.com/ray-project/ray/pull/44898), [#44868](https://github.com/ray-project/ray/pull/44868), [#44867](https://github.com/ray-project/ray/pull/44867), [#44845](https://github.com/ray-project/ray/pull/44845)

## Ray Core and Ray Clusters<a id="ray-core"></a>

💫 Enhancements:
- Report GCS internal pubsub buffer metrics and cap message size (#44749)

🔨 Fixes:
- Fix task submission never return when network partition happens (#44692)
- Fix incorrect use of ssh port forward option. (#44973)
- Make sure dashboard will exit if grpc server fails (#44928)
- Make sure dashboard agent will exit if grpc server fails (#44899)

Thanks @can-anyscale, @hongchaodeng, @zcin, @marwan116, @khluu, @bewestphal, @scottjlee, @andrewsykim, @anyscalesam, @MortalHappiness, @justinvyu, @JoshKarpel, @woshiyyya, @rynewang, @Abirdcfly, @omatthew98, @sven1977, @marcelocarmona, @rueian, @mattip, @angelinalg, @aslonnie, @matthewdeng, @abizjakpro, @simonsays1980, @jjyao, @terraflops1048576, @hongpeng-guo, @stephanie-wang, @bw-matthew, @bveeramani, @ruisearch42, @kevin85421, @Tongruizhe

Many thanks to all those who contributed to this release!

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.20.0)

---

## ray-2.12.0: Ray-2.12.0
**Published:** 2024-04-25

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉  New Features:

- Store Ray Data logs in special subdirectory (#44743)

💫 Enhancements:
- Add in `local_read` option to `from_torch` (#44752)

🔨 Fixes:
- Fix the config to disable progress bar (#44342)

📖 Documentation:
- Clarify deprecated Datasource docstrings (#44790)

## Ray Train<a id="ray-train"></a>

🔨 Fixes:
- Disable gathering the full state dict in `RayFSDPStrategy` for `lightning>2.1` (#44569)

## Ray Tune<a id="ray-tune"></a>

💫 Enhancements:

- Remove spammy log for "new output engine" (#44824)
- Enable isort (#44693)

## Ray Serve<a id="ray-serve"></a>

🔨 Fixes:
- [Serve] fix getting attributes on stdout during Serve logging redirect ([#44787](https://github.com/ray-project/ray/pull/44787))

## RLlib<a id="rllib"></a>

🎉 New Features:

- Support of images and video logging in WandB (env rendering example script for the new API stack coming up). ([#43356](https://github.com/ray-project/ray/pull/43356))

💫 Enhancements:

- Better support and separation-of-concerns for `model_config_dict` in new API stack. ([#44263](https://github.com/ray-project/ray/pull/44263))
- Added example script to pre-train an `RLModule` in single-agent fashion, then bring checkpoint into multi-agent setup and continue training. ([#44674](https://github.com/ray-project/ray/pull/44674))
- More `examples` scripts got translated from the old- to the new API stack: Curriculum learning, custom-gym-env, etc..: ([#44706](https://github.com/ray-project/ray/pull/44706), [#44707](https://github.com/ray-project/ray/pull/44707), [#44735](https://github.com/ray-project/ray/pull/44735), [#44841](https://github.com/ray-project/ray/pull/44841))

## Ray Core and Ray Clusters<a id="ray-core"></a>

🔨 Fixes:
- Fix GetAllJobInfo `is_running_tasks` is not returning the correct value when driver starts ray (#44459)

# Thanks

Many thanks to all those who contributed to this release!
@can-anyscale, @hongpeng-guo, @sven1977, @zcin, @shrekris-anyscale, @liuxsh9, @jackhumphries, @GeneDer, @woshiyyya, @simonsays1980, @omatthew98, @andrewsykim, @n30111, @architkulkarni, @bveeramani, @aslonnie, @alexeykudinkin, @WeichenXu123, @rynewang, @matthewdeng, @angelinalg, @c21

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.12.0)

---

## ray-2.11.0: Ray-2.11.0
**Published:** 2024-04-17

# Release Highlights<a id="release-highlights"></a>

- [data] Support reading Avro files with `ray.data.read_avro`
- [train] Added experimental support for AWS Trainium (Neuron) and Intel HPU.

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉  New Features:

- Support reading Avro files with `ray.data.read_avro` (#43663)

💫 Enhancements:
- Pin `ipywidgets==7.7.2` to enable Data progress bars in VSCode Web (#44398)
- Change log level for ignored exceptions (#44408)

🔨 Fixes:
- Change Parquet encoding ratio lower bound from 2 to 1 (#44470)
- Fix throughput time calculations for metrics (#44138)
- Fix nested ragged `numpy.ndarray` (#44236)
- Fix Ray debugger incompatibility caused by trimmed error stack trace (#44496)

📖 Documentation:
- Update "Data Loading and Preprocessing" doc (#44165)
- Move imports into `TFPRedictor` in batch inference example (#44434)

## Ray Train<a id="ray-train"></a>

🎉 New Features:

- Add experimental support for AWS Trainium (Neuron) (#39130)
- Add experimental support for Intel HPU (#43343)

💫 Enhancements:

- Log a deprecation warning for local_dir and related environment variables (#44029)
- Enforce xgboost>=1.7 for XGBoostTrainer usage (#44269)

🔨 Fixes:

- Fix ScalingConfig(accelerator_type) to request an appropriate resource amount (#44225)
- Fix maximum recursion issue when serializing exceptions  (#43952)
- Remove base config deepcopy when initializing the trainer actor (#44611)

🏗 Architecture refactoring:

- Remove deprecated `BatchPredictor` (#43934)

## Ray Tune<a id="ray-tune"></a>

💫 Enhancements:

- Add support for new style lightning import (#44339)
- Log a deprecation warning for local_dir and related environment variables (#44029)

🏗 Architecture refactoring:

- Remove scikit-optimize search algorithm (#43969)

## Ray Serve<a id="ray-serve"></a>

🔨 Fixes:
- Dynamically-created applications will no longer be deleted when a config is PUT via the REST API ([#44476](https://github.com/ray-project/ray/pull/44476)).
- Fix `_to_object_ref` memory leak ([#43763](https://github.com/ray-project/ray/pull/43763))
- Log warning to reconfigure `max_ongoing_requests` if `max_batch_size` is less than `max_ongoing_requests` ([#43840](https://github.com/ray-project/ray/pull/43840))
- Deployment fails to start with `ModuleNotFoundError` in Ray 3.10 ([#44329](https://github.com/ray-project/ray/issues/44329))
    - This was fixed by reverting the original core changes on the `sys.path` behavior. Revert "[core] If there's working_dir, don't set _py_driver_sys_path." ([#44435](https://github.com/ray-project/ray/pull/44435))
- The `batch_queue_cls` parameter is removed from the `@serve.batch` decorator ([#43935](https://github.com/ray-project/ray/pull/43935))

## RLlib<a id="rllib"></a>

🎉 New Features:

- New API stack: **DQN Rainbow** is now available for single-agent ([#43196](https://github.com/ray-project/ray/pull/43196), [#43198](https://github.com/ray-project/ray/pull/43198), [#43199](https://github.com/ray-project/ray/pull/43199))
- **`PrioritizedEpisodeReplayBuffer`** is available for **off-policy learning using the EnvRunner API** (`SingleAgentEnvRunner`) and supports random n-step sampling ([#42832](https://github.com/ray-project/ray/pull/42832), [#43258](https://github.com/ray-project/ray/pull/43258), [#43458](https://github.com/ray-project/ray/pull/43458), [#43496](https://github.com/ray-project/ray/pull/43496), [#44262](https://github.com/ray-project/ray/pull/44262))

💫 Enhancements:

- **Restructured `examples/` folder**; started moving example scripts to the new API stack ([#44559](https://github.com/ray-project/ray/pull/44559), [#44067](https://github.com/ray-project/ray/pull/44067), [#44603](https://github.com/ray-project/ray/pull/44603))
- **Evaluation do-over: Deprecate `enable_async_evaluation` option** (in favor of existing `evaluation_parallel_to_training` setting). ([#43787](https://github.com/ray-project/ray/pull/43787))
- Add: **`module_for` API to MultiAgentEpisode** (analogous to `policy_for` API of the old Episode classes). ([#44241](https://github.com/ray-project/ray/pull/44241))
- All **`rllib_contrib`** old stack algorithms have been removed from `rllib/algorithms` ([#43656](https://github.com/ray-project/ray/pull/43656))

🔨 Fixes:

- New API stack: Multi-GPU + multi-agent has been fixed. This completes support for any combinations of the following on the new API stack: [single-agent, multi-agent] vs [0 GPUs, 1 GPU, >1GPUs] vs [any number of EnvRunners] ([#44420](https://github.com/ray-project/ray/pull/44420), [#44664](https://github.com/ray-project/ray/pull/44664), [#44594](https://github.com/ray-project/ray/pull/44594), [#44677](https://github.com/ray-project/ray/pull/44677), [#44082](https://github.com/ray-project/ray/pull/44082), [#44669](https://github.com/ray-project/ray/pull/44669), [#44622](https://github.com/ray-project/ray/pull/44622))
- Various other bug fixes: [#43906](https://github.com/ray-project/ray/pull/43906), [#43871](https://github.com/ray-project/ray/pull/43871), [#44000](https://github.com/ray-project/ray/pull/44000), [#44340](https://github.com/ray-project/ray/pull/44340), [#44491](https://github.com/ray-project/ray/pull/44491), [#43959](https://github.com/ray-project/ray/pull/43959), [#44043](https://github.com/ray-project/ray/pull/44043), [#44446](https://github.com/ray-project/ray/pull/44446), [#44040](https://github.com/ray-project/ray/pull/44040)

📖 Documentation:
- [Re-announced new API stack in alpha stage](https://docs.ray.io/en/master/rllib/rllib-new-api-stack.html) ([#44090](https://github.com/ray-project/ray/pull/44090)).

## Ray Core and Ray Clusters<a id="ray-core"></a>

🎉 New Features:

- Added Ray check-open-ports CLI for checking potential open ports to the public (#44488)

💫 Enhancements:

- Support nodes sharing the same spilling directory without conflicts. (#44487)
- Create two subclasses of `RayActorError` to distinguish between actor died (`ActorDiedError`) and actor temporarily unavailable ([`ActorUnavailableError`](https://docs.ray.io/en/master/ray-core/fault_tolerance/actors.html#unavailable-actors)) cases.

🔨 Fixes:

- Fixed the `ModuleNotFound` issued introduced in 2.10 (#44435)
- Fixed an issue where agent process is using too much CPU (#44348)
- Fixed race condition in multi-threaded actor creation (#44232)
- Fixed several streaming generator bugs (#44079, #44257, #44197)
- Fixed an issue where user exception raised from tasks cannot be subclassed (#44379)

## Dashboard <a id="dashboard"></a>

💫 Enhancements:

- Add serve controller metrics to serve system dashboard page (#43797)
- Add Serve Application rows to Serve top-level deployments details page (#43506)
- [Actor table page enhancements] Include "NodeId", "CPU", "Memory", "GPU", "GRAM" columns in the actor table page. Add sort functionality to resource utilization columns. Enable searching table by "Class" and "Repr". (#42588) (#42633) (#42788)

🔨 Fixes:

- Fix default sorting of nodes in Cluster table page to first be by "Alive" nodes, then head nodes, then alphabetical by node ID. (#42929)
- Fix bug where the Serve Deployment detail page fails to load if the deployment is in "Starting" state (#43279)

## Docs <a id="docs"></a>

💫 Enhancements:

- Landing page refreshes its look and feel. (#44251)

# Thanks

Many thanks to all those who contributed to this release!

@aslonnie, @brycehuang30, @MortalHappiness, @astron8t-voyagerx, @edoakes, @sven1977, @anyscalesam, @scottjlee, @hongchaodeng, @slfan1989, @hebiao064, @fishbone, @zcin, @GeneDer, @shrekris-anyscale, @kira-lin, @chappidim, @raulchen, @c21, @WeichenXu123, @marian-code, @bveeramani, @can-anyscale, @mjd3, @justinvyu, @jackhumphries, @Bye-legumes, @ashione, @alanwguo, @Dreamsorcerer, @KamenShah, @jjyao, @omatthew98, @autolisis, @Superskyyy, @stephanie-wang, @simonsays1980, @davidxia, @angelinalg, @architkulkarni, @chris-ray-zhang, @kevin85421, @rynewang, @peytondmurray, @zhangyilun, @khluu, @matthewdeng, @ruisearch42, @pcmoritz, @mattip, @jerome-habana, @alexeykudinkin

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.11.0)

---

## ray-2.10.0: Ray-2.10.0
**Published:** 2024-03-21

# Release Highlights<a id="release-highlights"></a>
Ray 2.10 release brings important stability improvements and enhancements to Ray Data, with Ray Data becoming generally available (GA).

- [Data] Ray Data becomes generally available with stability improvements in streaming execution, reading and writing data, better tasks concurrency control, and debuggability improvement with dashboard, logging and metrics visualization.
- [RLlib] “**New API Stack**” officially announced as alpha for PPO and SAC.
- [Serve] Added a default autoscaling policy set via `num_replicas=”auto”` ([#42613](https://github.com/ray-project/ray/issues/42613)).
- [Serve] Added support for active load shedding via `max_queued_requests` ([#42950](https://github.com/ray-project/ray/issues/42950)).
- [Serve] Added replica queue length caching to the DeploymentHandle scheduler ([#42943](https://github.com/ray-project/ray/pull/42943)).
  - This should improve overhead in the Serve proxy and handles.
  - `max_ongoing_requests (max_concurrent_queries)` is also now strictly enforced ([#42947](https://github.com/ray-project/ray/issues/42947)).
  - If you see any issues, please report them on GitHub and you can disable this behavior by setting: `RAY_SERVE_ENABLE_QUEUE_LENGTH_CACHE=0`.
- [Serve] Renamed the following parameters. Each of the old names will be supported for another release before removal.
  - `max_concurrent_queries` -> `max_ongoing_requests`
  - `target_num_ongoing_requests_per_replica` -> `target_ongoing_requests`
  - `downscale_smoothing_factor` -> `downscaling_factor`
  - `upscale_smoothing_factor` -> `upscaling_factor`
- [Core] [Autoscaler v2](https://docs.ray.io/en/master/cluster/kubernetes/user-guides/configuring-autoscaling.html#kuberay-autoscaler-v2) is in alpha and can be tried out with Kuberay. It has improved observability and stability compared to v1.
- [Train] Added support for accelerator types via `ScalingConfig(accelerator_type)`.
- [Train] Revamped the `XGBoostTrainer` and `LightGBMTrainer` to no longer depend on `xgboost_ray` and `lightgbm_ray`. A new, more flexible API will be released in a future release.
- [Train/Tune] Refactored local staging directory to remove the need for `local_dir` and `RAY_AIR_LOCAL_CACHE_DIR`.

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉 New Features:
- Streaming execution stability improvement to avoid memory issue, including per-operator resource reservation, streaming generator output buffer management, and better runtime resource estimation (#43026, #43171, #43298, #43299, #42930, #42504)
- Metadata read stability improvement to avoid AWS transient error, including retry on application-level exception, spread tasks across multiple nodes, and configure retry interval (#42044, #43216, #42922, #42759).
- Allow tasks concurrency control for read, map, and write APIs (#42849, #43113, #43177, #42637)
- Data dashboard and statistics improvement with more runtime metrics for each components (#43790, #43628, #43241, #43477, #43110, #43112)
- Allow to specify application-level error to retry for actor task (#42492)
- Add `num_rows_per_file` parameter to file-based writes (#42694)
- Add `DataIterator.materialize` (#43210)
- Skip schema call in `DataIterator.to_tf` if `tf.TypeSpec` is provided (#42917)
- Add option to append for `Dataset.write_bigquery` (#42584)
- Deprecate legacy components and classes (#43575, #43178, #43347, #43349, #43342, #43341, #42936, #43144, #43022, #43023)

💫 Enhancements:

- Restructure stdout logging for better readability (#43360)
- Add a more performant way to read large TFRecord datasets (#42277)
- Modify `ImageDatasource` to use `Image.BILINEAR` as the default image resampling filter (#43484)
- Reduce internal stack trace output by default (#43251)
- Perform incremental writes to Parquet files (#43563)
- Warn on excessive driver memory usage during shuffle ops (#42574)
- Distributed reads for `ray.data.from_huggingface` (#42599)
- Remove `Stage` class and related usages (#42685)
- Improve stability of reading JSON files to avoid PyArrow errors (#42558, #42357)

🔨 Fixes:

- Turn off actor locality by default (#44124)
- Normalize block types before internal multi-block operations (#43764)
- Fix memory metrics for `OutputSplitter` (#43740)
- Fix race condition issue in `OpBufferQueue` (#43015)
- Fix early stop for multiple `Limit` operators. (#42958)
- Fix deadlocks caused by `Dataset.streaming_split` for job hanging (#42601)

📖 Documentation:

- Revamp Ray Data documentation for GA (#44006, #44007, #44008, #44098, #44168, #44093, #44105)

## Ray Train<a id="ray-train"></a>

🎉 New Features:

- Add support for accelerator types via `ScalingConfig(accelerator_type)` for improved worker scheduling (#43090)

💫 Enhancements:

- Add a backend-specific context manager for `train_func` for setup/teardown logic (#43209)
- Remove `DEFAULT_NCCL_SOCKET_IFNAME` to simplify network configuration (#42808)
- Colocate Trainer with rank 0 Worker for to improve scheduling behavior (#43115)

🔨 Fixes:

- Enable scheduling workers with `memory` resource requirements (#42999)
- Make path behavior OS-agnostic by using `Path.as_posix` over `os.path.join` (#42037)
- [Lightning] Fix resuming from checkpoint when using `RayFSDPStrategy` (#43594)
- [Lightning] Fix deadlock in `RayTrainReportCallback` (#42751)
- [Transformers] Fix checkpoint reporting behavior when `get_latest_checkpoint` returns None (#42953)

📖 Documentation:

- Enhance docstring and user guides for `train_loop_config` (#43691)
- Clarify in `ray.train.report` docstring that it is not a barrier (#42422)
- Improve documentation for `prepare_data_loader` shuffle behavior and `set_epoch` (#41807)

🏗 Architecture refactoring:

- Simplify XGBoost and LightGBM Trainer integrations. Implemented `XGBoostTrainer` and `LightGBMTrainer` as `DataParallelTrainer`. Removed dependency on `xgboost_ray` and `lightgbm_ray`. (#42111, #42767, #43244, #43424)
- Refactor local staging directory to remove the need for `local_dir` and `RAY_AIR_LOCAL_CACHE_DIR`. Add isolation between driver and distributed worker artifacts so that large files written by workers are not uploaded implicitly. Results are now only written to `storage_path`, rather than having another copy in the user’s home directory (`~/ray_results`). (#43369, #43403, #43689)
- Split overloaded `ray.train.torch.get_device` into another `get_devices` API for multi-GPU worker setup (#42314)
- Refactor restoration configuration to be centered around `storage_path` (#42853, #43179)
- Deprecations related to `SyncConfig` (#42909)
- Remove deprecated `preprocessor` argument from Trainers (#43146, #43234)
- Hard-deprecate `MosaicTrainer` and remove `SklearnTrainer` (#42814)


## Ray Tune<a id="ray-tune"></a>

💫 Enhancements:

- Increase the minimum number of allowed pending trials for faster auto-scaleup (#43455)
- Add support to `TBXLogger` for logging images (#37822)
- Improve validation of `Experiment(config)` to handle RLlib `AlgorithmConfig` (#42816, #42116)

🔨 Fixes:

- Fix `reuse_actors` error on actor cleanup for function trainables (#42951)
- Make path behavior OS-agnostic by using Path.as_posix over `os.path.join` (#42037)

📖 Documentation:

- Minor documentation fixes (#42118, #41982)

🏗 Architecture refactoring:

- Refactor local staging directory to remove the need for `local_dir` and `RAY_AIR_LOCAL_CACHE_DIR`. Add isolation between driver and distributed worker artifacts so that large files written by workers are not uploaded implicitly. Results are now only written to `storage_path`, rather than having another copy in the user’s home directory (`~/ray_results`). (#43369, #43403, #43689)
- Deprecations related to `SyncConfig` and `chdir_to_trial_dir` (#42909)
- Refactor restoration configuration to be centered around `storage_path` (#42853, #43179)
- Add back `NevergradSearch` (#42305)
- Clean up invalid `checkpoint_dir` and `reporter` deprecation notices (#42698)

## Ray Serve<a id="ray-serve"></a>

🎉 New Features:

- Added support for active load shedding via `max_queued_requests` ([#42950](https://github.com/ray-project/ray/issues/42950)).
- Added a default autoscaling policy set via `num_replicas=”auto”` ([#42613](https://github.com/ray-project/ray/issues/42613)).

🏗 API Changes:

- Renamed the following parameters. Each of the old names will be supported for another release before removal.
  - `max_concurrent_queries` to `max_ongoing_requests`
  - `target_num_ongoing_requests_per_replica` to `target_ongoing_requests`
  - `downscale_smoothing_factor` to `downscaling_factor`
  - `upscale_smoothing_factor` to `upscaling_factor`
- **WARNING**: the following default values will change in Ray 2.11:
  - Default for `max_ongoing_requests` will change from 100 to 5.
  - Default for `target_ongoing_requests` will change from 1 to 2.

💫 Enhancements:

- Add `RAY_SERVE_LOG_ENCODING` env to set the global logging behavior for Serve ([#42781](https://github.com/ray-project/ray/pull/42781)).
- Config Serve's gRPC proxy to allow large payload ([#43114](https://github.com/ray-project/ray/pull/43114)).
- Add blocking flag to serve.run() ([#43227](https://github.com/ray-project/ray/pull/43227)).
- Add actor id and worker id to Serve structured logs ([#43725](https://github.com/ray-project/ray/pull/43725)).
- Added replica queue length caching to the DeploymentHandle scheduler ([#42943](https://github.com/ray-project/ray/pull/42943)).
  - This should improve overhead in the Serve proxy and handles.
  - `max_ongoing_requests` (`max_concurrent_queries`) is also now strictly enforced ([#42947](https://github.com/ray-project/ray/issues/42947)).
  - If you see any issues, please report them on GitHub and you can disable this behavior by setting: `RAY_SERVE_ENABLE_QUEUE_LENGTH_CACHE=0`.
- Autoscaling metrics (tracking ongoing and queued metrics) are now collected at deployment handles by default instead of at the Serve replicas ([#42578](https://github.com/ray-project/ray/pull/42578)).
  - This means you can now set `max_ongoing_requests=1` for autoscaling deployments and still upscale properly, because requests queued at handles are properly taken into account for autoscaling.
  - You should expect deployments to upscale more aggressively during bursty traffic, because requests will likely queue up at handles during bursts of traffic.
  - If you see any issues, please report them on GitHub and you can switch back to the old method of collecting metrics by setting the environment variable `RAY_SERVE_COLLECT_AUTOSCALING_METRICS_ON_HANDLE=0`
- Improved the downscaling behavior of smoothing_factor for low numbers of replicas ([#42612](https://github.com/ray-project/ray/issues/42612)).
- Various logging improvements ([#43707](https://github.com/ray-project/ray/pull/43707), [#43708](https://github.com/ray-project/ray/pull/43708), [#43629](https://github.com/ray-project/ray/pull/43629), [#43557](https://github.com/ray-project/ray/pull/43557)).
- During in-place upgrades or when replicas become unhealthy, Serve will no longer wait for old replicas to gracefully terminate before starting new ones ([#43187](https://github.com/ray-project/ray/pull/43187)). New replicas will be eagerly started to satisfy the target number of healthy replicas.
  - This new behavior is on by default and can be turned off by setting `RAY_SERVE_EAGERLY_START_REPLACEMENT_REPLICAS=0`

🔨 Fixes:

- Fix deployment route prefix override by default route prefix from serve run cli ([#43805](https://github.com/ray-project/ray/pull/43805)).
- Fixed a bug causing batch methods to hang upon cancellation ([#42593](https://github.com/ray-project/ray/issues/42593)).
- Unpinned FastAPI dependency version ([#42711](https://github.com/ray-project/ray/issues/42711)).
- Delay proxy marking itself as healthy until it has routes from the controller ([#43076](https://github.com/ray-project/ray/issues/43076)).
- Fixed an issue where multiplexed deployments could go into infinite backoff ([#43965](https://github.com/ray-project/ray/issues/43965)).
- Silence noisy `KeyError` on disconnects ([#43713](https://github.com/ray-project/ray/pull/43713)).
- Fixed the prometheus counter metrics emitted as gauge bug ([#43795](https://github.com/ray-project/ray/pull/43795), [#43901](https://github.com/ray-project/ray/pull/43901)).
  - All the serve counter metrics are emitted as counters with _total suffix. The old gauge metrics are still emitted for compatibility.

📖 Documentation:

- Update serve logging config docs ([#43483](https://github.com/ray-project/ray/pull/43483)).
- Added documentation for `max_replicas_per_node` ([#42743](https://github.com/ray-project/ray/pull/42743)).

## RLlib<a id="rllib"></a>

🎉 New Features:

- The **“new API stack”** is now in alpha stage and available for **PPO single-** (#42272) and **multi-agent** and for **SAC single-agent** ([#42571](https://github.com/ray-project/ray/pull/42571), [#42570](https://github.com/ray-project/ray/pull/42570), [#42568](https://github.com/ray-project/ray/pull/42568))
  - **ConnectorV2 API** ([#43669](https://github.com/ray-project/ray/pull/43669), [#43680](https://github.com/ray-project/ray/pull/43680), [#43040](https://github.com/ray-project/ray/pull/43040), [#41074](https://github.com/ray-project/ray/pull/41074), [#41212](https://github.com/ray-project/ray/pull/41212))
  - **Episode APIs** (SingleAgentEpisode and MultiAgentEpisode) ([#42009](https://github.com/ray-project/ray/pull/42009), [#43275](https://github.com/ray-project/ray/pull/43275), [#42296](https://github.com/ray-project/ray/pull/42296), [#43818](https://github.com/ray-project/ray/pull/43818), [#41631](https://github.com/ray-project/ray/pull/41631))
  - **EnvRunner APIs** (SingleAgentEnvRunner and MultiAgentEnvRunner) ([#41558](https://github.com/ray-project/ray/pull/41558), [#41825](https://github.com/ray-project/ray/pull/41825), [#42296](https://github.com/ray-project/ray/pull/42296), [#43779](https://github.com/ray-project/ray/pull/43779))
- In preparation of **DQN** on the new API stack: PrioritizedEpisodeReplayBuffer ([#43258](https://github.com/ray-project/ray/pull/43258), [#42832](https://github.com/ray-project/ray/pull/42832))

💫 Enhancements:

- **Old API Stack cleanups:**
  - Move `SampleBatch` column names (e.g. `SampleBatch.OBS`) into new class (`Columns`). ([#43665](https://github.com/ray-project/ray/pull/43665))
  - Remove old exec_plan API code. ([#41585](https://github.com/ray-project/ray/pull/41585))
  - Introduce `OldAPIStack` decorator ([#43657](https://github.com/ray-project/ray/pull/43657))
  - **RLModule API:** Add functionality to define kernel and bias initializers via config. ([#42137](https://github.com/ray-project/ray/pull/42137))
- **Learner/LearnerGroup APIs**: 
  - Replace Learner/LearnerGroup specific config classes (e.g. `LearnerHyperparameters`) with `AlgorithmConfig`. ([#41296](https://github.com/ray-project/ray/pull/41296)) 
  - Learner/LearnerGroup: Allow updating from Episodes. ([#41235](https://github.com/ray-project/ray/pull/41235))
- In preparation of **DQN** on the new API stack: ([#43199](https://github.com/ray-project/ray/pull/43199), [#43196](https://github.com/ray-project/ray/pull/43196))

🔨 Fixes:

- New API Stack bug fixes: Fix `policy_to_train` logic ([#41529](https://github.com/ray-project/ray/pull/41529)), fix multi-APU for PPO on the new API stack. ([#44001](https://github.com/ray-project/ray/pull/44001)), Issue 40347: ([#42090](https://github.com/ray-project/ray/pull/42090))
- Other fixes: MultiAgentEnv would NOT call env.close() on a failed sub-env ([#43664](https://github.com/ray-project/ray/pull/43664)), Issue 42152 ([#43317](https://github.com/ray-project/ray/pull/43317)), issue 42396: ([#43316](https://github.com/ray-project/ray/pull/43316)), issue 41518 ([#42011](https://github.com/ray-project/ray/pull/42011)), issue 42385 ([#43313](https://github.com/ray-project/ray/pull/43313))

📖 Documentation:

- New API Stack examples: Self-play and league-based self-play ([#43276](https://github.com/ray-project/ray/pull/43276)), MeanStdFilter (for both single-agent and multi-agent) ([#43274](https://github.com/ray-project/ray/pull/43274)), Prev-actions/prev-rewards for multi-agent ([#43491](https://github.com/ray-project/ray/pull/43491))
- Other docs fixes and enhancements: ([#43438](https://github.com/ray-project/ray/pull/43438), [#41472](https://github.com/ray-project/ray/pull/41472), [#42117](https://github.com/ray-project/ray/pull/42177), [#43458](https://github.com/ray-project/ray/pull/43458))

# Ray Core and Ray Clusters<a id="ray-core-and-ray-clusters"></a>

## Ray Core<a id="ray-core"></a>

🎉 New Features:

- [Autoscaler v2](https://docs.ray.io/en/master/cluster/kubernetes/user-guides/configuring-autoscaling.html#kuberay-autoscaler-v2) is in alpha and can be tried out with Kuberay.
- Introduced [subreaper](https://docs.ray.io/en/master/ray-core/user-spawn-processes.html) to prevent leaks of sub-processes created by user code. (#42992)

💫 Enhancements:

- Ray state api `get_task()` now accepts ObjectRef (#43507)
- Add an option to disable task tracing for task/actor (#42431)
- Improved object transfer throughput. (#43434) 
- Ray client now compares the Ray and Python version for compatibility with the remote Ray cluster. (#42760)

🔨 Fixes:

- Fixed several bugs for streaming generator (#43775, #43772, #43413)
- Fixed Ray counter metrics emitted as gauge bug (#43795)
- Fixed a bug where empty resource task doesn’t work with placement group (#43448)
- Fixed a bug where CPU resource is not released for a blocked worker inside placement group (#43270)
- Fixed GCS crashes when PG commit phase failed due to node failure (#43405)
- Fixed a bug where Ray memory monitor prematurely kill tasks (#43071)
- Fixed placement group resource leak (#42942)
- Upgraded cloudpickle to 3.0 which fixes the incompatibility with dataclasses (#42730)

📖 Documentation:

- Updated the doc for Ray accelerators support (#41849)

## Ray Clusters<a id="ray-clusters"></a>

💫 Enhancements:

- [spark] Add `heap_memory` param for `setup_ray_cluster` API, and change default value of per ray worker node config, and change default value of ray head node config for global Ray cluster (#42604)
- [spark] Add global mode for ray on spark cluster (#41153)

🔨 Fixes:

- [VSphere] Only deploy ovf to first host of cluster  (#42258)

# Thanks

Many thanks to all those who contributed to this release!

@ronyw7, @xsqian, @justinvyu, @matthewdeng, @sven1977, @thomasdesr, @veryhannibal, @klebster2, @can-anyscale, @simran-2797, @stephanie-wang, @simonsays1980, @kouroshHakha, @Zandew, @akshay-anyscale, @matschaffer-roblox, @WeichenXu123, @matthew29tang, @vitsai, @Hank0626, @anmyachev, @kira-lin, @ericl, @zcin, @sihanwang41, @peytondmurray, @raulchen, @aslonnie, @ruisearch42, @vszal, @pcmoritz, @rickyyx, @chrislevn, @brycehuang30, @alexeykudinkin, @vonsago, @shrekris-anyscale, @andrewsykim, @c21, @mattip, @hongchaodeng, @dabauxi, @fishbone, @scottjlee, @justina777, @surenyufuz, @robertnishihara, @nikitavemuri, @Yard1, @huchen2021, @shomilj, @architkulkarni, @liuxsh9, @Jocn2020, @liuyang-my, @rkooo567, @alanwguo, @KPostOffice, @woshiyyya, @n30111, @edoakes, @y-abe, @martinbomio, @jiwq, @arunppsg, @ArturNiederfahrenhorst, @kevin85421, @khluu, @JingChen23, @masariello, @angelinalg, @jjyao, @omatthew98, @jonathan-anyscale, @sjoshi6, @gaborgsomogyi, @rynewang, @ratnopamc, @chris-ray-zhang, @ijrsvt, @scottsun94, @raychen911, @franklsf95, @GeneDer, @madhuri-rai07, @scv119, @bveeramani, @anyscalesam, @zen-xu, @npuichigo


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.10.0)

---

## ray-2.9.3: Ray-2.9.3
**Published:** 2024-02-22

This patch release contains fixes for Ray Core, Ray Data, and Ray Serve.

# Ray Core

🔨 Fixes:

- Fix protobuf breaking change by adding a compat layer. ([#43172](https://github.com/ray-project/ray/pull/43172))
- Bump up task failure logs to warnings to make sure failures could be troubleshooted ([#43147](https://github.com/ray-project/ray/pull/43147))
- Fix placement group leaks ([#42942](https://github.com/ray-project/ray/pull/42942))

# Ray Data

🔨 Fixes:

- Skip `schema` call in `to_tf` if `tf.TypeSpec` is provided ([#42917](https://github.com/ray-project/ray/pull/42917))
- Skip recording memory spilled stats when get_memory_info_reply is failed ([#42824](https://github.com/ray-project/ray/pull/42824))

# Ray Serve

🔨 Fixes:

- Fixing DeploymentStateManager qualifying replicas as running prematurely ([#43075](https://github.com/ray-project/ray/pull/43075))

# Thanks

Many thanks to all those who contributed to this release!

@rynewang, @GeneDer, @alexeykudinkin, @edoakes, @c21, @rkooo567

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.9.3)

---

## ray-2.9.2: Ray-2.9.2
**Published:** 2024-02-06

This patch release contains fixes for Ray Core, Ray Data, and Ray Serve.

# Ray Core
🔨 Fixes:
- Fix out of disk test on release branch (https://github.com/ray-project/ray/pull/42724)
   
# Ray Data
🔨 Fixes:
- Fix failing huggingface test (https://github.com/ray-project/ray/pull/42727)
- Fix deadlocks caused by streaming_split (https://github.com/ray-project/ray/pull/42601) (https://github.com/ray-project/ray/pull/42755)
- Fix locality config not being respected in DataConfig (https://github.com/ray-project/ray/pull/42204
https://github.com/ray-project/ray/pull/42204) (https://github.com/ray-project/ray/pull/42722)
- Stability & accuracy improvements for Data+Train benchmark (https://github.com/ray-project/ray/pull/42027)
- Add retry for _sample_fragment during `ParquetDatasource._estimate_files_encoding_ratio()` (https://github.com/ray-project/ray/pull/42759) (https://github.com/ray-project/ray/pull/42774)
- Skip recording memory spilled stats when get_memory_info_reply is failed (https://github.com/ray-project/ray/pull/42824) (https://github.com/ray-project/ray/pull/42834)

# Ray Serve
🔨 Fixes:
- Pin the fastapi & starlette version to avoid breaking proxy (https://github.com/ray-project/ray/pull/42740
https://github.com/ray-project/ray/pull/42740)
- Fix IS_PYDANTIC_2 logic for pydantic<1.9.0 (https://github.com/ray-project/ray/pull/42704) (https://github.com/ray-project/ray/pull/42708)
- fix missing message body for json log formats (https://github.com/ray-project/ray/pull/42729) (https://github.com/ray-project/ray/pull/42874)

# Thanks

Many thanks to all those who contributed to this release!

@c21, @raulchen, @can-anyscale, @edoakes, @peytondmurray, @scottjlee, @aslonnie, @architkulkarni, @GeneDer, @Zandew, @sihanwang41



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.9.2)

---

## ray-2.9.1: Ray-2.9.1
**Published:** 2024-01-19

This patch release contains fixes for Ray Core, Ray Data, and Ray Serve.

# Ray Core
🔨 Fixes:
- Adding debupgy as the ray debugger (#42311)
- Fix task events profile events per task leak (#42248) 
- Make sure redis sync context and async context connect to the same redis instance (#42040)
   
# Ray Data
🔨 Fixes:
- [Data] Retry write if error during file clean up (#42326)

# Ray Serve
🔨 Fixes:
- Improve handling the websocket server disconnect scenario (#42130) 
- Fix pydantic config documentation (#42216)
- Address issues under high network delays:
    - Enable setting queue length response deadline via environment variable (#42001)
    - Add exponential backoff for queue_len_response_deadline_s (#42041)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.9.1)

---

## ray-2.9.0: Ray-2.9.0
**Published:** 2023-12-21

# Release Highlights<a id="release-highlights"></a>

- This release contains fixes for the Ray Dashboard. Additional context can be found here: <https://www.anyscale.com/blog/update-on-ray-cves-cve-2023-6019-cve-2023-6020-cve-2023-6021-cve-2023-48022-cve-2023-48023> 
- Ray Train has now upgraded support for spot node preemption -- allowing Ray Train to handle preemption node failures differently than application errors.
- Ray is now compatible with Pydantic versions <2.0.0 and >=2.5.0, addressing a piece of user feedback we’ve consistently received.
- The Ray Dashboard now has a page for Ray Data to monitor real-time execution metrics.
- [Streaming generator](https://docs.ray.io/en/latest/ray-core/ray-generator.html) is now officially a public API (#41436, #38784). Streaming generator allows writing streaming applications easily on top of Ray via Python generator API and has been used for Ray Serve and Ray data for several releases. See the [documentation](https://docs.ray.io/en/master/ray-core/ray-generator.html) for details. 
- We’ve added experimental support for new accelerators: Intel GPU (#38553), Intel Gaudi Accelerators (#40561), and Huawei Ascend NPU (#41256).

# Ray Libraries<a id="ray-libraries"></a>

## Ray Data<a id="ray-data"></a>

🎉 New Features:
* Add the dashboard for Ray Data to monitor real-time execution metrics and log file for debugging (<https://docs.ray.io/en/master/data/monitoring-your-workload.html>).
* Introduce `concurrency` argument to replace `ComputeStrategy` in map-like APIs (#41461)
* Allow task failures during execution (#41226)
* Support PyArrow 14.0.1 (#41036)
* Add new API for reading and writing Datasource (<https://github.com/ray-project/ray/issues/40296>)
* Enable group-by over multiple keys in datasets (#37832)
* Add support for multiple group keys in `map_groups` (#40778)

💫 Enhancements:
- Optimize `OpState.outqueue_num_blocks` (#41748)
- Improve stall detection for `StreamingOutputsBackpressurePolicy` (#41637)
- Enable read-only Datasets to be executed on new execution backend (#41466, #41597)
- Inherit block size from downstream ops (#41019)
- Use runtime object memory for scheduling (#41383)
- Add retries to file writes (#41263)
- Make range datasource streaming (#41302)
- Test core performance metrics (#40757)
- Allow `ConcurrencyCapBackpressurePolicy._cap_multiplier` to be set to 1.0 (#41222)
- Create `StatsManager` to manage `_StatsActor` remote calls (#40913)
- Expose `max_retry_cnt` parameter for `BigQuery` Write (#41163)
- Add rows outputted to data metrics (#40280)
- Add fault tolerance to remote tasks (#41084)
- Add operator-level dropdown to ray data overview (#40981)
- Avoid slicing too-small blocks (#40840)
- Ray Data jobs detail table (#40756)
- Update default shuffle block size to 1GB (#40839)
- Log progress bar to data logs (#40814)
- Operator level metrics (#40805)

🔨 Fixes:
- Partial fix for `Dataset.context` not being sealed after creation (#41569)
- Fix the issue that `DataContext` is not propagated when using `streaming_split` (#41473)
- Fix Parquet partition filter bug (#40947)
- Fix split read output blocks (#41070)
- Fix `BigQueryDatasource `fault tolerance bugs (#40986)

📖 Documentation:
- Add example of how to read and write custom file types (#41785)
- Fix `ray.data.read_databricks_tables` doc (#41366)
- Add `read_json` docs example for setting PyArrow block size when reading large files (#40533)
- Add `AllToAllAPI` to dataset methods (#40842)


## Ray Train<a id="ray-train"></a>

🎉 New Features:
- Support reading `Result` from cloud storage (#40622)

💫 Enhancements:
- Sort local Train workers by GPU ID (#40953)
- Improve logging for Train worker scheduling information (#40536)
- Load the latest unflattened metrics with `Result.from_path` (#40684)
- Skip incrementing failure counter on preemption node died failures (#41285)
- Update TensorFlow `ReportCheckpointCallback` to delete temporary directory (#41033)

🔨 Fixes:
- Update config dataclass repr to check against None (#40851)
- Add a barrier in Lightning `RayTrainReportCallback` to ensure synchronous reporting. (#40875)
- Restore Tuner and `Result`s properly from moved storage path (#40647)

📖 Documentation:
- Improve torch, lightning quickstarts and migration guides + fix torch restoration example (#41843)
- Clarify error message when trying to use local storage for multi-node distributed training and checkpointing (#41844)
- Copy edits and adding links to docstrings (#39617)
- Fix the missing ray module import in PyTorch Guide (#41300)
- Fix typo in lightning_mnist_example.ipynb (#40577)
- Fix typo in deepspeed.rst (#40320)

🏗 Architecture refactoring:
- Remove Legacy Trainers (#41276)


## Ray Tune<a id="ray-tune"></a>

🎉 New Features:
- Support reading `Result` from cloud storage (#40622)

💫 Enhancements:
- Skip incrementing failure counter on preemption node died failures (#41285)

🔨 Fixes:
- Restore Tuner and `Result`s properly from moved storage path (#40647)

📖 Documentation:
- Remove low value Tune examples and references to them  (#41348)
- Clarify when to use `MLflowLoggerCallback` and `setup_mlflow` (#37854)

🏗 Architecture refactoring:
- Delete legacy `TuneClient`/`TuneServer` APIs (#41469)
- Delete legacy `Searcher`s (#41414)
- Delete legacy persistence utilities (`air.remote_storage`, etc.) (#40207)


## Ray Serve<a id="ray-serve"></a>

🎉 New Features:
- Introduce logging config so that users can set different logging parameters for different applications & deployments.
- Added gRPC context object into gRPC deployments for user to set custom code and details back to the client.
- Introduce a runtime environment feature that allows running applications in different containers with different images. This feature is experimental and a new guide can be found in the Serve docs.

💫 Enhancements:
- Explicitly handle gRPC proxy task cancellation when the client dropped a request to not waste compute resources. 
- Enable async `__del__` in the deployment to execute custom clean up steps.
- Make Ray Serve compatible with Pydantic versions <2.0.0 and >=2.5.0.

🔨 Fixes:
- Fixed gRPC proxy streaming request latency metrics to include the entire lifecycle of the request, including the time to consume the generator.
- Fixed gRPC proxy timeout request status from CANCELLED to DEADLINE_EXCEEDED.
- Fixed previously Serve shutdown spamming log files with logs for each event loop to only log once on shutdown.
- Fixed issue during batch requests when a request is dropped, the batch loop will be killed and not processed any future requests.
- Updating replica log filenames to only include POSIX-compliant characters (removed the “#” character).
- Replicas will now be gracefully shut down after being marked unhealthy due to health check failures instead of being force killed.
  - This behavior can be toggled using the environment variable RAY_SERVE_FORCE_STOP_UNHEALTHY_REPLICAS=1, but this is planned to be removed in the near future. If you rely on this behavior, please file an issue on github.


## RLlib<a id="rllib"></a>

🎉 New Features:
- New API stack (in progress):
  - New `MultiAgentEpisode` class introduced. Basis for upcoming multi-agent EnvRunner, which will replace RolloutWorker APIs. (#40263, #40799)
  - PPO runs with new `SingleAgentEnvRunner` (w/o Policy/RolloutWorker APIs). CI learning tests added. (#39732, #41074, #41075)
  - By default: PPO reverted to use old API stack by default, for now. Pending feature-completion of new API stack (incl. multi-agent, RNN support, new EnvRunners, etc..). (#40706)
- Old API stack:
  - APPO/IMPALA: Enable using 2 separate optimizers for policy and vs (and 2 learning rates) on the old API stack. (#40927)
  - Added `on_workers_recreated` callback to Algorithm, which is triggered after workers have failed and been restarted. (#40354)

💫 Enhancements:
- Old API stack and `rllib_contrib` cleanups: #40939, #40744, #40789, #40444, #37271

🔨 Fixes:
- Restoring from a checkpoint from an older wheel (where `AlgorithmConfig.rl_module_spec` was NOT a “@property” yet) breaks when trying to load from this checkpoint. (#41157)
- SampleBatch slicing crashes when using tf + SEQ_LENS + zero-padding. (#40905)
- Other fixes: #39978, #40788, #41168, #41204

📖 Documentation:
- Updated codeblocks in RLlib. (#37271)


# Ray Core and Ray Clusters<a id="ray-core-and-ray-clusters"></a>

## Ray Core<a id="ray-core"></a>

🎉 New Features:
- [Streaming generator](https://docs.ray.io/en/master/ray-core/ray-generator.html) is now officially a public API (#41436, #38784). Streaming generator allows writing streaming applications easily on top of Ray via Python generator API and has been used for Ray serve and Ray data for several releases. See the [documentation](https://docs.ray.io/en/master/ray-core/ray-generator.html) for details. 
  - As part of the change, num_returns=”dynamic” is planning to be deprecated, and its return type is changed from `ObjectRefGenerator` -> “DynamicObjectRefGenerator”
- Add experimental accelerator support for new hardwares.
  - Add experimental support for Intel GPU (#38553)
  - Add experimental support for Intel Gaudi Accelerators (#40561)
  - Add experimental support for Huawei Ascend NPU (#41256)
- Add the initial support to run MPI based code on top of Ray.(#40917, #41349)

💫 Enhancements:
- Optimize next/anext performance for streaming generator (#41270)
- Make the number of connections and thread number of the object manager client tunable. (#41421)
- Add `__ray_call__` default actor method (#41534)

🔨 Fixes:
- Fix NullPointerException cause by raylet id is empty when get actor info in java worker (#40560)
- Fix a bug where SIGTERM is ignored to worker processes (#40210)
- Fix mmap file leak. (#40370)
- Fix the lifetime issue in Plasma server client releasing object. (#40809)
- Upgrade grpc from 1.50.2 to 1.57.1 to include security fixes (#39090)
- Fix the bug where two head nodes are shown from ray list nodes (#40838)
- Fix the crash when the GCS address is not valid. (#41253)
- Fix the issue of unexpectedly high socket usage in ray core worker processes. (#41121)
- Make worker_process_setup_hook work with strings instead of Python functions (#41479)


## Ray Clusters<a id="ray-clusters"></a>

💫 Enhancements:
- Stability improvements for the vSphere cluster launcher
- Better CLI output for cluster launcher

🔨 Fixes:
- Fixed `run_init` for TPU command runner

📖Documentation:
- Added missing steps and simplified YAML in top-level clusters quickstart
- Clarify that job entrypoints run on the head node by default and how to override it


## Dashboard<a id="dashboard"></a>

💫 Enhancements:
  - Improvements to the Ray Data Dashboard
    - Added Ray Data-specific overview on jobs page, including a table view with Dataset-level metrics
    - Added operator-level metrics granularity to drill down on Dataset operators
    - Added additional metrics for monitoring iteration over Datasets

# Docs<a id="docs"></a>
🎉 New Features:
- Updated to Sphinx version 7.1.2. Previously, the docs build used Sphinx 4.3.2. Upgrading to a recent version provides a more modern user experience while fixing many long standing issues. Let us know how you like the upgrade or any other docs issues on your mind, on the Ray Slack #docs channel.

# Thanks

Many thanks to all those who contributed to this release!

@justinvyu, @zcin, @avnishn, @jonathan-anyscale, @shrekris-anyscale, @LeonLuttenberger, @c21, @JingChen23, @liuyang-my, @ahmed-mahran, @huchen2021, @raulchen, @scottjlee, @jiwq, @z4y1b2, @jjyao, @JoshTanke, @marxav, @ArturNiederfahrenhorst, @SongGuyang, @jerome-habana, @rickyyx, @rynewang, @batuhanfaik, @can-anyscale, @allenwang28, @wingkitlee0, @angelinalg, @peytondmurray, @rueian, @KamenShah, @stephanie-wang, @bryanjuho, @sihanwang41, @ericl, @sofianhnaide, @RaffaGonzo, @xychu, @simonsays1980, @pcmoritz, @aslonnie, @WeichenXu123, @architkulkarni, @matthew29tang, @larrylian, @iycheng, @hongchaodeng, @rudeigerc, @rkooo567, @robertnishihara, @alanwguo, @emmyscode, @kevin85421, @alexeykudinkin, @michaelhly, @ijrsvt, @ArkAung, @mattip, @harborn, @sven1977, @liuxsh9, @woshiyyya, @hahahannes, @GeneDer, @vitsai, @Zandew, @evalaiyc98, @edoakes, @matthewdeng, @bveeramani


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.9.0)

---

## ray-2.8.1: Ray-2.8.1
**Published:** 2023-12-01

# Release Highlights
The Ray 2.8.1 patch release contains fixes for the Ray Dashboard.

Additional context can be found here: https://www.anyscale.com/blog/update-on-ray-cves-cve-2023-6019-cve-2023-6020-cve-2023-6021-cve-2023-48022-cve-2023-48023

# Ray Dashboard
🔨 Fixes:

[core][state][log] Cherry pick changes to prevent state API from reading files outside the Ray log directory (#41520)
[Dashboard] Migrate Logs page to use state api. (#41474) (#41522)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.8.1)

---

## ray-2.8.0: Ray-2.8.0
**Published:** 2023-11-03

# Release Highlights
This release features stability improvements and API clean-ups across the Ray libraries.

- In Ray Serve, we are deprecating the previously experimental DAG API for deployment graphs. Model composition will be supported through [deployment handles](https://docs.ray.io/en/latest/serve/model_composition.html) providing more flexibility and stability. The previously deprecated Ray Serve 1.x APIs have also been removed. We’ve also added a new Java APIs that aligns with the Ray Serve 2.x APIs. More API changes in the release notes below.
- In RLlib, we’ve moved 24 algorithms into `rllib_contrib` (still available within RLlib for Ray 2.8).
- We’ve added support for PyTorch-compatible input files shuffling for Ray Data. This allows users to randomly shuffle input files for better model training accuracy. This release also features new Ray Data datasources for Databricks and BigQuery. 
- On the Ray Dashboard, we’ve added new metrics for Ray Data in the Metrics tab. This allows users to monitor Ray Data workload including real time metrics of cluster memory, CPU, GPU, output data size, etc. See [the doc](https://docs.ray.io/en/master/data/performance-tips.html#monitoring-your-application) for more details.
- Ray Core now supports profiling GPU tasks or actors using Nvidia Nsight. See [the documentation](https://docs.ray.io/en/master/ray-observability/user-guides/profiling.html?highlight=nsight#nsight-system-profiler) for instructions.
- We fixed 2 critical bugs raised by many kuberay / ML library users, including a child process leak issue from Ray worker that leaks the GPU memory (#40182) and an job page excessive loading time issue when Ray HA cluster restarts a head node (#40742)
- Python 3.7 support is officially deprecated from Ray.

# Ray Libraries
## Ray Data
🎉 New Features:
- Add support for shuffling input files (#40154)
- Support streaming read of PyTorch dataset (#39554)
- Add BigQuery datasource (#37380)
- Add Databricks table / SQL datasource (#39852)
- Add inverse transform functionality to LabelEncoder (#37785)
- Add function arg params to `Dataset.map` and `Dataset.flat_map` (#40010)

💫Enhancements:
- Hard deprecate `DatasetPipeline` (#40129)
- Remove `BulkExecutor` code path (#40200)
- Deprecate extraneous `Dataset` parameters and methods (#40385)
- Remove legacy iteration code path (#40013)
- Implement streaming output backpressure (#40387)
- Cap op concurrency with exponential ramp-up (#40275)
- Store ray dashboard metrics in `_StatsActor` (#40118)
- Slice output blocks to respect target block size (#40248)
- Drop columns before grouping by in `Dataset.unique()` (#40016)
- Standardize physical operator runtime metrics (#40173)
- Estimate blocks for limit and union operator (#40072)
- Store bytes spilled/restored after plan execution (#39361)
- Optimize `sample_boundaries` in `SortTaskSpec` (#39581)
- Optimization to reduce ArrowBlock building time for blocks of size 1 (#38833)

🔨 Fixes:
- Fix bug where `_StatsActor` errors with `PandasBlock` (#40481)
- Remove deprecated `do_write` (#40422)
- Improve error message when reading HTTP files (#40462)
- Add flag to skip `get_object_locations` for metrics (#39884)
- Fall back to fetch files info in parallel for multiple directories (#39592)
- Replace deprecated `.pieces` with updated `.fragments` (#39523)
- Backwards compatibility for `Preprocessor` that have been fit in older versions (#39173)
- Removing unnecessary data copy in `convert_udf_returns_to_numpy` (#39188)
- Do not eagerly free root `RefBundles` (#39016)

📖Documentation:
- Remove out-of-date Data examples (#40127)
- Remove unused and outdated source examples (#40271)

## Ray Train
🎉 New Features:
- Add initial support for scheduling workers on neuron_cores (#39091)

💫Enhancements:
- Update PyTorch Lightning import path to support both `pytorch_lightning` and `lightning` (#39841, #40266)
- Propagate driver `DataContext` to `RayTrainWorkers` (#40116)

🔨 Fixes:
- Fix error propagation for as_directory if to_directory fails (#40025)

📖Documentation:
- Update checkpoint hierarchy documentation for RayTrainReportCallbacks. (#40174)
- Update Lightning RayDDPStrategy docstring (#40376)

🏗 Architecture refactoring:
- Deprecate `LightningTrainer`, `AccelerateTrainer`, `TransformersTrainer (#40163)
- Clean up legacy persistence mode code paths (#39921, #40061, #40069, #40168)
- Deprecate legacy `DatasetConfig` (#39963)
- Remove references to `DatasetPipeline` (#40159)
- Enable isort (#40172)

## Ray Tune
💫Enhancements:
- Separate storage checkpoint index bookkeeping (#39927, #40003)
- Raise an error if `Tuner.restore()` is called on an instance (#39676)
🏗 Architecture refactoring:
- Clean up legacy persistence mode code paths (#39918, #40061, #40069,  #40168, #40175, #40192, #40181, #40193)
- Migrate TuneController tests (#39704)
- Remove TuneRichReporter (#40169)
- Remove legacy Ray Client tests (#40415)

## Ray Serve
💫Enhancements:
- The single-app configuration format for the [Serve Config](https://docs.ray.io/en/master/serve/production-guide/config.html#serve-in-production-config-file) (i.e. the Serve Config without the ‘applications’ field) has been deprecated in favor of the new configuration format.
Both single-app configuration and DAG API will be removed in 2.9.
- The Serve REST API is now accessible through the dashboard port, which defaults to `8265`.
- Accessing the Serve REST API through the dashboard agent port (default `52365`) is deprecated. The support will be removed in a future version.
- Ray job error tracebacks are now logged in the job driver log for easier access when jobs fail during start up.
- Deprecated single-application config file
- Deprecated DAG API: `InputNode` and `DAGDriver`
- Removed deprecated Deployment 1.x APIs: `Deployment.deploy()`, `Deployment.delete()`, `Deployment.get_handle()`
- Removed deprecated 1.x API: `serve.get_deployment` and `serve.list_deployments`
- New Java API supported (aligns with Ray Serve 2.x API)

🔨 Fixes:
- The `dedicated_cpu` and `detached` options in `serve.start()` have been fully disallowed.
- Error will be raised when users pass invalid gRPC service functions and fail early.
- The proxy’s readiness check now uses a linear backoff to avoid getting stuck in an infinite loop if it takes longer than usual to start.
- `grpc_options` on `serve.start()` was only allowing a `gRPCOptions` object in Ray 2.7.0. Dictionaries are now allowed to be used as`grpc_options` in the `serve.start()` call.

## RLlib
💫Enhancements:
- `rllib_contrib` algorithms (A2C, A3C, AlphaStar #36584, AlphaZero #36736, ApexDDPG #36596, ApexDQN #36591, ARS #36607, Bandits #36612, CRR #36616, DDPG, DDPPO #36620, Dreamer(V1), DT #36623, ES #36625, LeelaChessZero #36627, MA-DDPG #36628, MAML, MB-MPO #36662, PG #36666, QMix #36682, R2D2, SimpleQ #36688, SlateQ #36710, and TD3 #36726) all produce warnings now if used. See [here](https://github.com/ray-project/ray/tree/master/rllib_contrib#rllib-contrib) for more information on the `rllib_contrib` efforts. (36620, 36628, 3
- Provide msgpack checkpoint translation utility to convert checkpoint into msgpack format for being able to move in between python versions (#38825).

🔨 Fixes:
- Issue 35440 (JSON output writer should include INFOS #39632)
- Issue 39453 (PettingZoo wrappers should use correct multi-agent dict spaces #39459)
- Issue 39421 (Multi-discrete action spaces not supported in new stack #39534)
- Issue 39234 (Multi-categorical distribution bug #39464)
#39654, #35975, #39552, #38555
# Ray Core and Ray Clusters
## Ray Core
🎉 New Features:
- Python 3.7 support is officially deprecated from Ray.
- Supports profiling GPU tasks or actors using Nvidia Nsight. See [the doc](https://docs.ray.io/en/master/ray-observability/user-guides/profiling.html?highlight=nsight#nsight-system-profiler) for instructions.
- Ray on spark autoscaling is officially supported from Ray 2.8. See the [REP](https://github.com/ray-project/enhancements/pull/43) for more details.
💫Enhancements:
- IDLE node information in detail is available from ray status -v (#39638)
- Adding a new accelerator to Ray is simplified with a new accelerator interface. See the in-flight [REP](https://github.com/ray-project/enhancements/blob/9936d231b4403cdbceb754a674395ffcf9a586e5/reps/2023-10-13-accelerator-support.md) for more details (#40286).
- Typing_extensions is removed from a dependency requirement because Python 3.7 support is deprecated. (#40336)
- Ray state API supports case insensitive match. (#34577)
- `ray start --runtime-env-agent-port` is officially supported. (#39919)
- Driver exit code is available fromjob info (#39675)

🔨 Fixes:
- Fixed a worker leak when Ray is used with placement group because Ray didn’t handle SIGTERM properly (#40182)
- Fixed an issue job page loading takes a really long time when Ray HA cluster restarts a head node (#40431)
- [core] loosen the check on release object (#39570)
- [Core] ray init sigterm (#39816)
- [Core] Non Unit Instance fractional value fix (#39293)
- [Core]: Enable get_actor_name for actor runtime context (#39347)
- [core][streaming][python] Fix asyncio.wait coroutines args deprecated warnings #40292 

📖Documentation:
- The Ray streaming generator doc (alpha) is officially available at https://docs.ray.io/en/master/ray-core/ray-generator.html 

## Ray Clusters
💫Enhancements:
- Enable GPU support for vSphere cluster launcher (#40667)

📖Documentation:
- Setup RBAC by KubeRay Helm chart
- KubeRay upgrade documentation
- RayService high availability

🔨 Fixes:
- Assorted fixes for vSphere cluster launcher (#40487, #40516, #40655)

## Dashboard
🎉 New Features:
- New metrics for ray data can be found in the Metrics tab.
🔨 Fixes:
- Fix bug where download log button did not download all logs for actors.

## Thanks
Many thanks to all who contributed to this release!

@scottjlee, @chappidim, @alexeykudinkin, @ArturNiederfahrenhorst, @stephanie-wang, @chaowanggg, @peytondmurray, @maxpumperla, @arvind-chandra, @iycheng, @JalinWang, @matthewdeng, @wfangchi, @z4y1b2, @alanwguo, @Zandew, @kouroshHakha, @justinvyu, @yuanchen8911, @vitsai, @hongchaodeng, @allenwang28, @caozy623, @ijrsvt, @omus, @larrylian, @can-anyscale, @joncarter1, @ericl, @lejara, @jjyao, @Ox0400, @architkulkarni, @edoakes, @raulchen, @bveeramani, @sihanwang41, @WeichenXu123, @zcin, @Codle, @dimakis, @simonsays1980, @cadedaniel, @angelinalg, @luv003, @JingChen23, @xwjiang2010, @rynewang, @Yicheng-Lu-llll, @scrivy, @michaelhly, @shrekris-anyscale, @xxnwj, @avnishn, @woshiyyya, @aslonnie, @amogkam, @krfricke, @pcmoritz, @liuyang-my, @jonathan-anyscale, @rickyyx, @scottsun94, @richardliaw, @rkooo567, @stefanbschneider, @kevin85421, @c21, @sven1977, @GeneDer, @matthew29tang, @RocketRider, @LaynePeng, @samhallam-reverb, @scv119, @huchen2021

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.8.0)

---

## ray-2.7.1: Ray-2.7.1
**Published:** 2023-10-09

# Release Highlights



* Ray Serve: 
    * Added an `application` tag to the `ray_serve_num_http_error_requests` metric
    * Fixed a bug where no data shows up on the `Error QPS per Application` panel in the Ray Dashboard
* RLlib:
    * DreamerV3: Bug fix enabling support for continuous actions.
* Ray Train:
    * Fix a bug where setting a local storage path on Windows errors ([#39951](https://github.com/ray-project/ray/pull/39951))
* Ray Tune:
    * Fix a broken `Trial.node_ip` property ([#40028](https://github.com/ray-project/ray/pull/40028))
* Ray Core:
    * Fixes a segfault when a streaming generator and actor cancel is used together
    * Fix autoscaler sdk accidentally initialize ray worker leading to leaked driver showing up in the dashboard.
    * Added a new user guide and fixes for the vSphere cluster launcher.
    * Fixed a bug where `ray start `would occasionally fail with `ValueError: `acceleratorType` should match v(generation)-(cores/chips).`
* Dashboard:
    * Improvement on cluster page UI
    * Fix a bug that overview page UI will crash


# Ray Libraries


## Ray Serve

🔨 Fixes:



* Fixed a bug where no data shows up on the `Error QPS per Application` panel in the Ray Dashboard


## RLlib

🔨 Fixes:



* DreamerV3: Bug fix enabling support for continuous actions ([39751](https://github.com/ray-project/ray/issues/39751)).


# Ray Core and Ray Clusters

🔨 Fixes:



* Fixed Ray cluster stability on a high latency environment

Thanks

Many thanks to all those who contributed to this release!

@chaowanggg, @allenwang28, @shrekris-anyscale, @GeneDer, @justinvyu, @can-anyscale, @edoakes, @architkulkarni, @rkooo567, @rynewang, @rickyyx, @sven1977


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.7.1)

---

## ray-2.7.0: Ray-2.7.0
**Published:** 2023-09-17

# Release Highlights

Ray 2.7 release brings important stability improvements and enhancements to Ray libraries, with Ray Train and Ray Serve becoming generally available. Ray 2.7 is accompanied with a GA release of KubeRay.



* Following user feedback, we are rebranding “Ray AI Runtime (AIR)” to “Ray AI Libraries”. Without reducing any of the underlying functionality of the original Ray AI runtime vision as put forth in Ray 2.0, the underlying namespace (ray.air) is consolidated into ray.data, ray.train, and ray.tune. This change reduces the friction for new machine learning (ML) practitioners to quickly understand and leverage Ray for their production machine learning use cases. 
* With this release, Ray Serve and Ray Train’s Pytorch support are becoming Generally Available -- indicating that the core APIs have been marked stable and that both libraries have undergone significant production hardening. 
* In Ray Serve, we are introducing a new backwards-compatible `DeploymentHandle` API to unify various existing Handle APIs, a high performant gRPC proxy to serve gRPC requests through Ray Serve, along with various stability and usability improvements.
* In Ray Train, we are consolidating various Pytorch-based trainers into the TorchTrainer, reducing the amount of refactoring work new users needed to scale existing training scripts. We are also introducing a new train.Checkpoint API, which provides a consolidated way of interacting with remote and local storage, along with various stability and usability improvements.
* In Ray Core, we’ve added initial integrations with TPUs and AWS accelerators, enabling Ray to natively detect these devices and schedule tasks/actors onto them. Ray Core also officially now supports actor task cancellation and has an experimental streaming generator that supports streaming response to the caller. 

Take a look at our [refreshed documentation](https://docs.ray.io/en/releases-2.7.0) and the [Ray 2.7 migration guide](https://docs.google.com/document/d/1J-09US8cXc-tpl2A1BpOrlHLTEDMdIJp6Ah1ifBUw7Y/view#heading=h.3eeweptnwn6p) and let us know your feedback!


# Ray Libraries


## Ray AIR

🏗 Architecture refactoring:



* **Ray AIR namespace**: We are sunsetting the "Ray AIR" concept and namespace (#39516, #38632, #38338, #38379, #37123, #36706, #37457, #36912, #37742, #37792, #37023). The changes follow the proposal outlined in [this REP](https://github.com/ray-project/enhancements/pull/36).
* **Ray Train Preprocessors, Predictors**: We now recommend using Ray Data instead of Preprocessors (#38348, #38518, #38640, #38866) and Predictors (#38209). 


## Ray Data

🎉 New Features:



* In this release, we’ve integrated the Ray Core streaming generator API by default, which allows us to reduce memory footprint throughout the data pipeline (#37736).
* Avoid unnecessary data buffering between `Read` and `Map` operator (zero-copy fusion) (#38789)
* Add `Dataset.write_images` to write images (#38228)
* Add `Dataset.write_sql()` to write SQL databases (#38544)
* Support sort on multiple keys (#37124)
* Support reading and writing JSONL file format (#37637)
* Support class constructor args for `Dataset.map()` and `flat_map()` (#38606)
* Implement streamed read from Hugging Face Dataset (#38432)

💫Enhancements:



* Read data with multi-threading for `FileBasedDataSource` (#39493)
* Optimization to reduce `ArrowBlock` building time for blocks of size 1 (#38988)
* Add `partition_filter` parameter to `read_parquet `(#38479)
* Apply limit to `Dataset.take()` and related methods (#38677)
* Postpone `reader.get_read_tasks` until execution (#38373)
* Lazily construct metadata providers (#38198)
* Support writing each block to a separate file (#37986)
* Make `iter_batches` an Iterable (#37881)
* Remove default limit on `Dataset.to_pandas()` (#37420)
* Add `Dataset.to_dask()` parameter to toggle consistent metadata check (#37163)
* Add `Datasource.on_write_start` (#38298)
* Remove support for `DatasetDict` as input into `from_huggingface()` (#37555)

🔨 Fixes:



* Backwards compatibility for `Preprocessor` that have been fit in older versions (#39488)
* Do not eagerly free root `RefBundles` (#39085)
* Retry open files with exponential backoff (#38773)
* Avoid passing `local_uri` to all non-Parquet data sources (#38719)
* Add `ctx` parameter to `Datasource.write` (#38688)
* Preserve block format on `map_batches` over empty blocks (#38161)
* Fix args and kwargs passed to `ActorPool` `map_batches` (#38110)
* Add `tif` file extension to `ImageDatasource` (#38129)
* Raise error if PIL can't load image (#38030)
* Allow automatic handling of string features as byte features during TFRecord serialization (#37995)
* Remove unnecessary file system wrapping (#38299)
* Remove `_block_udf` from `FileBasedDatasource` reads (#38111)

📖Documentation:



* Standardize API references (#37015, #36980, #37007, #36982, etc)


## Ray Train

🤝 API Changes



* **Ray Train and Ray Tune Checkpoints:** Introduced a new `train.Checkpoint` class that unifies interaction with remote storage such as S3, GS, and HDFS. The changes follow the proposal in [[REP35] Consolidated persistence API for Ray Train/Tune](https://github.com/ray-project/enhancements/pull/35) (#38452, #38481, #38581, #38626, #38864, #38844)
* **Ray Train with PyTorch Lightning:** Moving away from the LightningTrainer in favor of the TorchTrainer as the recommended way of running distributed PyTorch Lightning.  The changes follow the proposal outlined in [[REP37] [Train] Unify Torch based Trainers on the TorchTrainer API](https://github.com/ray-project/enhancements/pull/37) (#37989)
* **Ray Train with Hugging Face Transformers/Accelerate:** Moving away from the TransformersTrainer/AccelerateTrainer in favor of the TorchTrainer as the recommended way of running distributed Hugging Face Transformers and Accelerate.  The changes follow the proposal outlined in [[REP37] [Train] Unify Torch based Trainers on the TorchTrainer API](https://github.com/ray-project/enhancements/pull/37) (#38083, #38295)
* Deprecated `preprocessor` arg to `Trainer` (#38640)
* Removed deprecated `Result.log_dir` (#38794) 

💫Enhancements:



* Various improvements and fixes for the console output of Ray Train and Tune (#37572, #37571, #37570, #37569, #37531, #36993)
* Raise actionable error message for missing dependencies (#38497)
* Use posix paths throughout library code (#38319)
* Group consecutive workers by IP (#38490)
* Split all Ray Datasets by default (#38694)
* Add static Trainer methods for getting tree-based models (#38344)
* Don't set rank-specific local directories for Train workers (#38007)

🔨 Fixes:



* Fix trainer restoration from S3 (#38251)

🏗 Architecture refactoring:



* Updated internal usage of the new Checkpoint API (#38853, #38804, #38697, #38695, #38757, #38648, #38598, #38617, #38554, #38586, #38523, #38456, #38507, #38491, #38382, #38355, #38284, #38128, #38143, #38227, #38141, #38057, #38104, #37888, #37991, #37962, #37925, #37906, #37690, #37543, #37475, #37142, #38855, #38807, #38818, #39515, #39468, #39368, #39195, #39105, #38563, #38770, #38759, #38767, #38715, #38709, #38478, #38550, #37909, #37613, #38876, #38868, #38736, #38871, #38820, #38457)

📖Documentation:



* Restructured the Ray Train documentation to make it easier to find relevant content (#37892, #38287, #38417, #38359)
* Improved examples, references, and navigation items (#38049, #38084, #38108, #37921, #38391, #38519, #38542, #38541, #38513, #39510, #37588, #37295, #38600, #38582, #38276, #38686, #38537, #38237, #37016)
* Removed outdated examples (#38682, #38696, #38656, #38374, #38377, #38441, #37673, #37657, #37067)


## Ray Tune

🤝 API Changes



* **Ray Train and Ray Tune Checkpoints:** Introduced a new `train.Checkpoint` class that unifies interaction with remote storage such as S3, GS, and HDFS. The changes follow the proposal in [[REP35] Consolidated persistence API for Ray Train/Tune](https://github.com/ray-project/enhancements/pull/35) (#38452, #38481, #38581, #38626, #38864, #38844)
* Removed deprecated `Result.log_dir` (#38794) 

💫Enhancements:



* Various improvements and fixes for the console output of Ray Train and Tune (#37572, #37571, #37570, #37569, #37531, #36993)
* Raise actionable error message for missing dependencies (#38497)
* Use posix paths throughout library code (#38319)
* Improved the PyTorchLightning integration (#38883, #37989, #37387, #37400)
* Improved the XGBoost/LightGBM integrations (#38558, #38828)

🔨 Fixes:



* Fix hyperband r calculation and stopping (#39157)
* Replace deprecated np.bool8 (#38495)
* Miscellaneous refactors and fixes (#38165, #37506, #37181, #37173)

🏗 Architecture refactoring:



* Updated internal usages of the new Checkpoint API (#38853, #38804, #38697, #38695, #38757, #38648, #38598, #38617, #38554, #38586, #38523, #38456, #38507, #38491, #38382, #38355, #38284, #38128, #38143, #38227, #38141, #38057, #38104, #37888, #37991, #37962, #37925, #37906, #37690, #37543, #37475, #37142, #38855, #38807, #38818, #39515, #39468, #39368, #39195, #39105, #38563, #38770, #38759, #38767, #38715, #38709, #38478, #38550, #37909, #37613, #38876, #38868, #38736, #38871, #38820, #38457)
* Removed legacy TrialRunner/Executor (#37927)


## Ray Serve

🎉 New Features:



* Added keep_alive_timeout_s to Serve config file to allow users to configure HTTP proxy’s duration to keep idle connections alive when no requests are ongoing.
* Added gRPC proxy to serve gRPC requests through Ray Serve. It comes with feature parity with HTTP while offering better performance. Also, replaces the previous experimental gRPC direct ingress. 
* Ray 2.7 introduces a new `DeploymentHandle` API that will replace the existing `RayServeHandle` and `RayServeSyncHandle` APIs in a future release. You are encouraged to migrate to the new API to avoid breakages in the future. To opt in, either use `handle.options(use_new_handle_api=True)` or set the global environment variable `export RAY_SERVE_ENABLE_NEW_HANDLE_API=1`. See https://docs.ray.io/en/latest/serve/model_composition.html for more details.
* Added a new API `get_app_handle` that gets a handle used to send requests to an application. The API uses the new `DeploymentHandle` API.
* Added a new developer API `get_deployment_handle` that gets a handle that can be used to send requests to any deployment in any application.
* Added replica placement group support.
* Added a new API `serve.status` which can be used to get the status of proxies and Serve applications (and their deployments and replicas). This is the pythonic equivalent of the CLI `serve status`.
* A `--reload` option has been added to the `serve run` CLI.
* Support X-Request-ID in http header

💫Enhancements:



* Downstream handlers will now be canceled when the HTTP client disconnects or an end-to-end timeout occurs.
* Ray Serve is now “generally available,” so the core APIs have been marked stable.
    * `serve.start` and `serve.run` have a few small changes and deprecations in preparation for this, see [https://docs.ray.io/en/latest/serve/api/index.html](https://docs.ray.io/en/latest/serve/api/index.html) for details.
* Added a new metric (`ray_serve_num_ongoing_http_requests`) to track the number of ongoing requests in each proxy
* Add `RAY_SERVE_MULTIPLEXED_MODEL_ID_MATCHING_TIMEOUT_S` flag to wait until the model matching.
* Reduce the multiplexed model id information publish interval.
* Add Multiplex metrics into dashboard 
* Added metrics to track controller restarts and control loop progress
    * [https://github.com/ray-project/ray/pull/38177](https://github.com/ray-project/ray/pull/38177)
    * [https://github.com/ray-project/ray/pull/38000](https://github.com/ray-project/ray/pull/38000)
* Various stability, flexibility, and performance enhancements to Ray Serve’s autoscaling.
    * [https://github.com/ray-project/ray/pull/38107](https://github.com/ray-project/ray/pull/38107)
    * [https://github.com/ray-project/ray/pull/38034](https://github.com/ray-project/ray/pull/38034)
    * [https://github.com/ray-project/ray/pull/38267](https://github.com/ray-project/ray/pull/38267)
    * [https://github.com/ray-project/ray/pull/38349](https://github.com/ray-project/ray/pull/38349)
    * [https://github.com/ray-project/ray/pull/38351](https://github.com/ray-project/ray/pull/38351)

🔨 Fixes:



* Fixed a memory leak in Serve components by upgrading gRPC: [https://github.com/ray-project/ray/issues/38591](https://github.com/ray-project/ray/issues/38591).
* Fixed a memory leak due to `asyncio.Event`s not being removed in the long poll host: [https://github.com/ray-project/ray/pull/38516](https://github.com/ray-project/ray/pull/38516).
* Fixed a bug where bound deployments could not be passed within custom objects: [https://github.com/ray-project/ray/issues/38809](https://github.com/ray-project/ray/issues/38809).
* Fixed a bug where all replica handles were unnecessarily broadcasted to all proxies every minute: [https://github.com/ray-project/ray/pull/38539](https://github.com/ray-project/ray/pull/38539).
* Fixed a bug where `ray_serve_deployment_queued_queries` wouldn’t decrement when clients disconnected:[ https://github.com/ray-project/ray/pull/37965](https://github.com/ray-project/ray/pull/37965).

📖Documentation:



* Added docs for how to use keep_alive_timeout_s in the Serve config file.
* Added usage and examples for serving gRPC requests through Serve’s gRPC proxy.
* Added example for passing deployment handle responses by reference.
* Added a Ray Serve Autoscaling guide to the Ray Serve docs that goes over basic configurations and autoscaling examples. Also added an Advanced Ray Serve Autoscaling guide that goes over more advanced configurations and autoscaling examples.
* Added docs explaining how to debug memory leaks in Serve.
* Added docs that explain how Serve cancels disconnected requests and how to handle those disconnections.


## RLlib

🎉 New Features:



* In Ray RLlib, we have implemented Google’s new [DreamerV3](https://github.com/ray-project/ray/tree/master/rllib/algorithms/dreamerv3), a sample-efficient, model-based, and hyperparameter hassle-free algorithm. It solves a wide variety of challenging reinforcement learning environments out-of-the-box (e.g. the MineRL diamond challenge), for arbitrary observation- and action-spaces as well as dense and sparse reward functions.

💫Enhancements:



* Added support for Gymnasium 0.28.1 [(#35698](https://github.com/ray-project/ray/pull/35698))
* Dreamer V3 tuned examples and support for “XL” Dreamer models ([#38461](https://github.com/ray-project/ray/pull/38461))
* Added an action masking example for RL Modules ([#38095](https://github.com/ray-project/ray/pull/38095))

🔨 Fixes:



* Multiple fixes to DreamerV3 ([#37979](https://github.com/ray-project/ray/pull/37979)) ([#38259](https://github.com/ray-project/ray/pull/38259)) ([#38461](https://github.com/ray-project/ray/pull/38461)) ([#38981](https://github.com/ray-project/ray/pull/38981))
* Fixed TorchBinaryAutoregressiveDistribution.sampled_action_logp() returning probs not log probs. ([#37240](https://github.com/ray-project/ray/pull/37240))
* Fix a bug in Multi-Categorical distribution. It should use logp and not log_p. ([#36814](https://github.com/ray-project/ray/pull/36814))
* Index tensors in slate epsilon greedy properly so SlateQ does not fail on multiple GPUs ([#37481](https://github.com/ray-project/ray/pull/37481))
* Removed excessive deprecation warnings in exploration related files ([#37404](https://github.com/ray-project/ray/pull/37404))
* Fixed missing agent index in policy input dict on environment reset ([#37544](https://github.com/ray-project/ray/pull/37544))

📖Documentation:



* Added docs for DreamerV3 [(#37978](https://github.com/ray-project/ray/pull/37978))
* Added docs on torch.compile usage ([#37252](https://github.com/ray-project/ray/pull/37252))
* Added docs for the Learner API [(#37729](https://github.com/ray-project/ray/pull/37729))
* Improvements to Catalogs and RL Modules docs + Catalogs improvements ([#37245](https://github.com/ray-project/ray/pull/37245))
* Extended our metrics and callbacks example to showcase how to do custom summarisation on custom metrics ([#37292](https://github.com/ray-project/ray/pull/37292))


# Ray Core and Ray Clusters


## Ray Core

🎉 New Features:
* [Actor task cancelation](https://docs.ray.io/en/master/ray-core/actors.html#cancelling-actor-tasks) is officially supported. 
* The experimental streaming generator is now available. It means the yielded output is sent to the caller before the task is finished and overcomes the [limitation from `num_returns="dynamic"` generator](https://docs.ray.io/en/latest/ray-core/tasks/generators.html#limitations). The API could be used by specifying `num_returns="streaming"`. The API has been used for Ray data and Ray serve to support streaming use cases. [See the test script](https://github.com/ray-project/ray/blob/0d6bc79bbba400e91346a021279501e05940b51e/python/ray/tests/test_streaming_generator.py#L123) to learn how to use the API. The documentation will be available in a few days.

💫Enhancements:
* Minimal Ray installation `pip install ray` doesn't require the Python grpcio dependency anymore. 
* [Breaking change] `ray job submit` now exits with `1` if the job fails instead of `0`. To get the old behavior back, you may use `ray job submit ... || true` . ([#38390](https://github.com/ray-project/ray/pull/38390))
* [Breaking change] `get_assigned_resources` in pg will return the name of the original resources instead of formatted name (#37421)
* [Breaking change] Every env var specified via `${ENV_VAR} ` now can be replaced. Previous versions only supported limited number of env vars. (#36187)
* [Java] Update Guava package (#38424)
* [Java] Update Jackson Databind XML Parsing (#38525)
* [Spark] Allow specifying CPU / GPU / Memory resources for head node of Ray cluster on spark  (#38056)

🔨 Fixes:
* [Core] Internal gRPC version is upgraded from 1.46.6 to 1.50.2, which fixes the memory leak issue
* [Core] Bind jemalloc to raylet and GCS (#38644) to fix memory fragmentation issue
* [Core] Previously, when a ray is started with `ray start --node-ip-address=...`, the driver also had to specify `ray.init(_node_ip_address)`. Now Ray finds the node ip address automatically. (#37644)
* [Core] Child processes of workers are cleaned up automatically when a raylet dies (#38439)
* [Core] Fix the issue where there are lots of threads created when using async actor (#37949)
* [Core] Fixed a bug where tracing did not work when an actor/task was defined prior to calling `ray.init`: [https://github.com/ray-project/ray/issues/26019](https://github.com/ray-project/ray/issues/26019)
* Various other bug fixes
    * [Core] loosen the check on release object (#39570)
    * [Core][agent] fix the race condition where the worker process terminated during the get_all_workers call #37953
    * [Core]Fix PG leakage caused by GCS restart when PG has not been successfully remove after the job died (#35773)
    * [Core]Fix internal_kv del api bug in client proxy mode (#37031)
    * [Core] Pass logs through if sphinx-doctest is running (#36306)
    * [Core][dashboard] Make intentional ray system exit from worker exit non task failing (#38624)
    * [Core][dashboard] Add worker pid to task info  (#36941)
    * [Core] Use 1 thread for all fibers for an actor scheduling queue. (#37949)
    * [runtime env] Fix Ray hangs when nonexistent conda environment is specified #28105 (#34956)

## Ray Clusters

💫Enhancements:



* New Cluster Launcher for vSphere [#37815](https://github.com/ray-project/ray/pull/37815)
* TPU pod support for cluster launcher [#37934](https://github.com/ray-project/ray/pull/37934)

📖Documentation:



* The KubeRay documentation has been moved to [https://docs.ray.io/en/latest/cluster/kubernetes/index.html](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)  from its old location at [https://ray-project.github.io/kuberay/](https://ray-project.github.io/kuberay/).
* New guide: GKE Ingress on KubeRay ([39073](https://github.com/ray-project/ray/pull/39073))
* New tutorial: Cloud storage from GKE on KubeRay [#38858](https://github.com/ray-project/ray/pull/38858)
* New tutorial: Batch inference tutorial using KubeRay RayJob CR [#38857](https://github.com/ray-project/ray/pull/38857)
* New benchmarks for RayService custom resource on KubeRay [#38647](https://github.com/ray-project/ray/pull/38647)
* New tutorial: Text summarizer using NLP with RayService [#38647](https://github.com/ray-project/ray/pull/38647)

Thanks

Many thanks to all those who contributed to this release!

@simran-2797, @can-anyscale, @akshay-anyscale, @c21, @EdwardCuiPeacock, @rynewang, @volks73, @sven1977, @alexeykudinkin, @mattip, @Rohan138, @larrylian, @DavidYoonsik, @scv119, @alpozcan, @JalinWang, @peterghaddad, @rkooo567, @avnishn, @JoshKarpel, @tekumara, @zcin, @jiwq, @nikosavola, @seokjin1013, @shrekris-anyscale, @ericl, @yuxiaoba, @vymao, @architkulkarni, @rickyyx, @bveeramani, @SongGuyang, @jjyao, @sihanwang41, @kevin85421, @ArturNiederfahrenhorst, @justinvyu, @pleaseupgradegrpcio, @aslonnie, @kukushking, @94929, @jrosti, @MattiasDC, @edoakes, @PRESIDENT810, @cadedaniel, @ddelange, @alanwguo, @noahjax, @matthewdeng, @pcmoritz, @richardliaw, @vitsai, @Michaelvll, @tanmaychimurkar, @smiraldr, @wfangchi, @amogkam, @crypdick, @WeichenXu123, @darthhexx, @angelinalg, @chaowanggg, @GeneDer, @xwjiang2010, @peytondmurray, @z4y1b2, @scottsun94, @chappidim, @jovany-wang, @jaidisido, @krfricke, @woshiyyya, @Shubhamurkade, @ijrsvt, @scottjlee, @kouroshHakha, @allenwang28, @raulchen, @stephanie-wang, @iycheng


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.7.0)

---

## ray-2.6.3: Ray-2.6.3
**Published:** 2023-08-15

The Ray 2.6.3 patch release contains fixes for Ray Serve, and Ray Core streaming generators. 

## Ray Core
🔨 Fixes:
* [Core][Streaming Generator] Fix memory leak from the end of object stream object #38152 (#38206)

## Ray Serve
🔨 Fixes:
* [Serve] Fix `serve run` help message (#37859) (#38018)
* [Serve] Decrement `ray_serve_deployment_queued_queries` when client disconnects (#37965) (#38020)


## RLib
📖 Documentation:
* [RLlib][docs] Learner API Docs (#37729) (#38137)





[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.6.3)

---

## ray-2.6.2: Ray-2.6.2
**Published:** 2023-08-03

The Ray 2.6.2 patch release contains a critical fix for ray's logging setup, as well fixes for Ray Serve, Ray Data, and Ray Job. 

## Ray Core
🔨 Fixes:
* [Core] Pass logs through if sphinx-doctest is running (#36306) (#37879)
* [cluster-launcher] Pick GCP cluster launcher tests and fix  (#37797)


## Ray Serve
🔨 Fixes:
* [Serve] Apply `request_timeout_s` from Serve config to the cluster (#37884) (#37903)

## Ray Air
🔨 Fixes:
* [air] fix pyarrow lazy import (#37670) (#37883)


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.6.2)

---

## ray-2.6.1: Ray-2.6.1
**Published:** 2023-07-24

The Ray 2.6.1 patch release contains a critical fix for cluster launcher, and compatibility update for Ray Serve protobuf definition with python 3.11, as well doc improvements. 

⚠️  Cluster launcher in Ray 2.6.0 fails to start multi-node clusters. Please update to 2.6.1 if you plan to use 2.6.0 cluster launcher.


## Ray Core
🔨 Fixes:
* [core][autoscaler] Fix env variable overwrite not able to be used if the command itself uses the env #37675


## Ray Serve
🔨 Fixes:
* [serve] Cherry-pick Serve enum to_proto fixes for Python 3.11 #37660

## Ray Air
📖Documentation:
* [air][doc] Update docs to reflect head node syncing deprecation #37475


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.6.1)

---

## ray-2.6.0: Ray-2.6.0
**Published:** 2023-07-21

# Release Highlights

* **Serve**: Better streaming support -- In this release, Support for HTTP streaming response and WebSockets is now on by default. Also, `@serve.batch`-decorated methods can stream responses.
* **Train and Tune**: Users are now expected to provide cloud storage or NFS path for distributed training or tuning jobs instead of a local path. This means that results written to different worker machines will not be directly synced to the head node. Instead, this will raise an error telling you to switch to one of the recommended alternatives: cloud storage or NFS. Please see https://github.com/ray-project/ray/issues/37177 if you have questions.
* **Data**: We are introducing a new streaming integration of Ray Data and Ray Train. This allows streaming data ingestion for model training, and enables per-epoch data preprocessing. The DatasetPipeline API is also being deprecated in favor of Dataset with streaming execution.
* **RLlib**: Public alpha release for the new multi-gpu Learner API that is less complex and more powerful compared to our previous solution ([blogpost](https://www.anyscale.com/blog/introducing-rllib-multi-gpu-stack-for-cost-efficient-scalable-multi-gpu-rl)). This is used under PPO algorithm by default.

# Ray Libraries

## Ray AIR

🎉 **New Features**:
* Added support for restoring Results from local trial directories. (#35406)

💫 **Enhancements**:
* [Train/Tune] Disable Train/Tune syncing to head node  (#37142)
* [Train/Tune] Introduce new console output progress reporter for Train and Tune (#35389, #36154, #36072, #35770, #36764, #36765, #36156, #35977)
* [Train/Data]  New Train<>Data streaming integration (#35236, #37215, #37383)

🔨 **Fixes**:
* Pass on KMS-related kwargs for s3fs (#35938)
* Fix infinite recursion in log redirection (#36644)
* Remove temporary checkpoint directories after restore (#37173)
* Removed actors that haven't been started shouldn't be tracked (#36020)
* Fix bug in execution for actor re-use (#36951)
* Cancel `pg.ready()` task for pending trials that end up reusing an actor (#35748)
* Add case for `Dict[str, np.array]` batches in `DummyTrainer` read bytes calculation (#36484)

📖 **Documentation**:
* Remove experimental features page, add github issue instead (#36950)
* Fix batch format in `dreambooth` example (#37102) 
* Fix Checkpoint.from_checkpoint docstring (#35793)

🏗 **Architecture refactoring**:
* Remove deprecated mlflow and wandb integrations (#36860, #36899)
* Move constants from tune/results.py to air/constants.py (#35404)
* Clean up a few checkpoint related things. (#35321)

## Ray Data

🎉 **New Features**:
* New streaming integration of Ray Data and Ray Train. This allows streaming data ingestion for model training, and enables per-epoch data preprocessing. (#35236)
* Enable execution optimizer by default (#36294, #35648, #35621, #35952)
* Deprecate DatasetPipeline (#35753)
* Add `Dataset.unique()` (#36655, #36802)
* Add option for parallelizing post-collation data batch operations in `DataIterator.iter_batches()` (#36842) (#37260)
* Enforce strict mode batch format for `DataIterator.iter_batches()` (#36686)
* Remove `ray.data.range_arrow()` (#35756)

💫 **Enhancements**:
* Optimize block prefetching (#35568)
* Enable isort for data directory (#35836)
* Skip writing a file for an empty block in `Dataset.write_datasource()` (#36134)
* Remove shutdown logging from StreamingExecutor (#36408)
* Spread map task stages by default for arg size <50MB (#36290)
* Read->SplitBlocks to ensure requested read parallelism is always met (#36352)
* Support partial execution in `Dataset.schema()` with new execution plan optimizer (#36740)
* Propagate iter stats for `Dataset.streaming_split()` (#36908)
* Cache the computed schema to avoid re-executing (#37103)

🔨 **Fixes**:
* Support sub-progress bars on AllToAllOperators with optimizer enabled (#34997)
* Fix DataContext not propagated properly for `Dataset.streaming_split()` operator
* Fix edge case in empty bundles with `Dataset.streaming_split()` (#36039)
* Apply Arrow table indices mapping on HuggingFace Dataset prior to reading into Ray Data (#36141)
* Fix issues with combining use of `Dataset.materialize()` and `Dataset.streaming_split()` (#36092)
* Fix quadratic slowdown when locally shuffling tensor extension types (#36102)
* Make sure progress bars always finish at 100% (#36679)
* Fix wrong output order of `Dataset.streaming_split()` (#36919)
* Fix the issue that StreamingExecutor is not shutdown when the iterator is not fully consumed (#36933)
* Calculate stage execution time in StageStatsSummary from `BlockMetadata` (#37119)

📖 **Documentation**:
* Standardize Data API ref (#36432, #36937)
* Docs for working with PyTorch (#36880)
* Split "Consuming data" guide (#36121)
* Revise "Loading data" (#36144)
* Consolidate Data user guides (#36439)

🏗 **Architecture refactoring**:
* Remove simple blocks representation (#36477)

## Ray Train

🎉 **New Features**:
* LightningTrainer support DeepSpeedStrategy (#36165)

💫 **Enhancements**:
* Unify Lightning and AIR CheckpointConfig (#36368)
* Add support for custom pipeline class in TransformersPredictor (#36494)

🔨 **Fixes**:
* Fix Deepspeed device ranks check in Lightning 2.0.5 (#37387)
* Clear stale lazy checkpointing markers on all workers. (#36291)

📖 **Documentation**:
* Migrate Ray Train `code-block` to `testcode`. (#36483)

🏗 Architecture refactoring:
* Deprecate `BatchPredictor` (#36947, #37178) 

## Ray Tune

🔨 **Fixes**:

* Optuna: Update distributions to use new APIs (#36704)
* BOHB: Fix nested bracket processing (#36568)
* Hyperband: Fix scheduler raising an error for good `PENDING` trials (#35338)
* Fix param space placeholder injection for numpy/pandas objects (#35763)
* Fix result restoration with Ray Client (#35742)
* Fix trial runner/controller whitelist attributes (#35769)

📖 **Documentation**:
* Remove missing example from Tune "Other examples" (#36691)

🏗 **Architecture refactoring**:
* Remove `tune/automl` (#35557)
* Remove hard-deprecated modules from structure refactor (#36984)
* Remove deprecated mlflow and wandb integrations (#36860, #36899)
* Move constants from tune/results.py to air/constants.py (#35404)
* Deprecate redundant syncing related parameters (#36900)
* Deprecate legacy modules in `ray.tune.integration` (#35160)

## Ray Serve

💫 **Enhancements**:
* Support for HTTP streaming response and WebSockets is now on by default.
* `@serve.batch`-decorated methods can stream responses.
* `@serve.batch` settings can be reconfigured dynamically.
* Ray Serve now uses “power of two random choices” routing. This improves enforcement of `max_concurrent_queries` and tail latencies under load.

🔨 **Fixes**:
* Fixed the bug previously unable to use a custom module named after “utils”.
* Fixed serve downscaling issue by adding a new draining state to the http proxy. This helps http proxies to not take new requests when there are no replicas on the node and prevents interruption on the ongoing requests when the node is downscaled. Also, enables downscaling to happen when the requests use Ray’s object store which is blocking downscaling of the node.
* Fixed non-atomic shutdown logic. Serve shutdown will be run in the background and not require the client to wait for the shutdown to complete. And won’t be interrupted when the client is force killed.

## RLlib

🎉 **New Features**:
* Public alpha release for the new multi-gpu Learner API that is less complex and more powerful than the old training stack ([blogpost](https://www.anyscale.com/blog/introducing-rllib-multi-gpu-stack-for-cost-efficient-scalable-multi-gpu-rl)). This is used under PPO algorithm by default.
* Added RNN support on the new RLModule API
* Added TF-version of DreamerV3 ([link](https://github.com/ray-project/ray/tree/master/rllib/algorithms/dreamerv3)). The comprehensive results will be published soon.
* Added support for torch 2.x compile method in sampling from environment

💫 **Enhancements**:
* Added an Example on how to do pretraining with BC and then continuing finetuning with PPO ([example](https://github.com/ray-project/ray/blob/master/rllib/examples/learner/train_w_bc_finetune_w_ppo.py))
* RLlib deprecation Notices (algorithm/, evaluation/, execution/, models/jax/) (#[36826](https://github.com/ray-project/ray/pull/36826))
* Enable eager_tracing=True by default. (#[36556](https://github.com/ray-project/ray/pull/36556))

🔨 **Fixes**:
* Fix bug in Multi-Categorical distribution. It should use logp and not log_p. (#[36814](https://github.com/ray-project/ray/pull/36814))
* Fix LSTM + Connector bug: StateBuffer restarting states on every in_eval() call. (#[36774](https://github.com/ray-project/ray/pull/36774))

🏗 **Architecture refactoring**:
* Multi-GPU Learner API

## Ray Core

🎉 **New Features**:
* [Core][Streaming Generator] Cpp interfaces and implementation (#35291)
* [Core][Streaming Generator] Streaming Generator. Support Core worker APIs + cython generator interface. (#35324)
* [Core][Streaming Generator] Streaming Generator. E2e integration (#35325)
* [Core][Streaming Generator] Support async actor and async generator interface. (#35584)
* [Core][Streaming Generator] Streaming Generator. Support the basic retry/lineage reconstruction (#35768)
* [Core][Streaming Generator] Allow to raise an exception to avoid check failures. (#35766)
* [Core][Streaming Generator] Fix a reference leak when a stream is deleted with out of order writes. (#35591)
* [Core][Streaming Generator] Fix a reference leak when pinning requests are received after refs are consumed. (#35712)
* [Core][Streaming Generator] Handle out of order report when retry (#36069)
* [Core][Streaming Generator] Make it compatible with wait (#36071)
* [Core][Streaming Generator] Remove busy waiting (#36070)
* [Core][Autoscaler v2] add test for node provider (#35593)
* [Core][Autoscaler v2] add unit tests for NodeProviderConfig (#35590)
* [Core][Autoscaler v2] test ray-installer (#35875)
* [Core][Autoscaler v2] fix too many values to unpack (expected 2) bug (#36231)
* [Core][Autoscaler v2] Add idle time information to Autoscaler endpoint. (#36918)
* [Core][Autoscaler v2] Cherry picks change to Autoscaler intereface (#37407)
* [Core][Autoscaler v2] Fix idle time duration when node resource is not updated periodically (#37121) (#37175)
* [Core][Autoscaler v2] Fix pg id serialization with hex rather than binary for cluster state reporting #37132 (#37176)
* [Core][Autoscaler v2] GCS Autoscaler V2: Add instance id to ray [3/x] (#35649)
* [Core][Autoscaler v2] GCS Autoscaler V2: Add node type name to ray (#36714)
* [Core][Autoscaler v2] GCS Autoscaler V2: Add placement group's gang resource requests handling [4/x] (#35970)
* [Core][Autoscaler v2] GCS Autoscaler V2: Handle ReportAutoscalingState (#36768)
* [Core][Autoscaler v2] GCS Autoscaler V2: Interface [1/x] (#35549)
* [Core][Autoscaler v2] GCS Autoscaler V2: Node states and resource requests [2/x] (#35596)
* [Core][Autoscaler v2] GCS Autoscaler V2: Support Autoscaler.sdk.request_resources [5/x] (#35846)
* [Core][Autoscaler v2] Ray status interface [1/x] (#36894)
* [Core][Autoscaler v2] Remove usage of grpcio from Autoscaler SDK (#36967)
* [Core][Autoscaler v2] Update Autoscaler proto for default enum value (#36962)
* [Core][Autoscalerv2] Update Autoscaler.proto / instance_manager.proto dependency (#36116)

💫 **Enhancements**:
* [Core] Make some grpcio imports lazy (#35705)
* [Core] Only instantiate gcs channels on driver (#36389)
* [Core] Port GcsSubscriber to Cython (#35094)
* [Core] Print out warning every 1s when sched_cls_id is greater than 100 (#35629)
* [Core] Remove attrs dependency (#36270)
* [Core] Remove dataclasses requirement (#36218)
* [Core] Remove grpcio from Ray minimal dashboard (#36636)
* [Core] Remove grpcio import from usage_lib (#36542)
* [Core] remove import thread (#36293)
* [Core] Remove Python grpcio from check_health (#36304)
* [Core] Retrieve the token from GCS server [4/n] (#37003) (#37294)
* [Core] Retry failed redis request (#35249)
* [Core] Sending ReportWorkerFailure after the process died. (#35320)
* [Core] Serialize auto-inits (#36127)
* [Core] Support auto-init ray for get_runtime_context() (#35903)
* [Core] Suppress harmless ObjectRefStreamEndOfStreamError when using asyncio (#37062) (#37200)
* [Core] Unpin grpcio and make Ray run on mac M1 out of the box (#35932)
* [Core] Add a better error message for health checking network failures (#36957) (#37366)
* [Core] Add ClusterID to ClientCallManager [2/n] (#36526)
* [Core] Add ClusterID token to GCS server [3/n] (#36535)
* [Core] Add ClusterID token to GRPC server [1/n] (#36517)
* [Core] Add extra metrics for workers (#36973)
* [Core] Add get_worker_id() to runtime context (#35967)
* [Core] Add logs for Redis leader discovery for observability. (#36108)
* [Core] Add metrics for object size distribution in object store (#37005) (#37110)
* [Core] Add resource idle time to resource report from node. (#36670)
* [Core] Check that temp_dir must be absolute path. (#36431)
* [Core] Clear CPU affinity for worker processes (#36816)
* [Core] Delete object spilling dead code path. (#36286)
* [Core] Don't drop rpc status in favor of reply status (#35530)
* [Core] Feature flag actor task logs with off by default (#35921)
* [Core] Graceful handling of returning bundles when node is removed (#34726)
* [Core] Graceful shutdown in TaskEventBuffer destructor (#35857)
* [Core] Guarantee the ordering of put ActorTaskSpecTable and ActorTable (#35683)
* [Core] Introduce fail_on_unavailable option for hard NodeAffinitySchedulingStrategy (#36718)
* [Core] Make “import” ray work without grpcio (#35737)
* [Core][dashboard] Add task name in task log magic token (#35377)
* [Core][deprecate run_function_on_all_workers 3/n] delete run_function_on_all_workers (#30895)
* [Core][devex] Move ray/util build targets to separate build files (#36598)
* [Core][logging][ipython] Fix log buffering when consecutive runs within ray log dedup window (#37134) (#37174)
* [Core][Logging] Switch worker_setup_hook to worker_process_setup_hook (#37247) (#37463)
* [Core][Metrics] Use Autoscaler-emitted metrics for pending/active/failed nodes. (#35884)
* [Core][state] Record file offsets instead of logging magic token to track task log (#35572)
* [CI] [Minimal install] Check python version in minimal install (#36887)
* [CI] second try of fixing vllm example in CI #36712
* [CI] skip vllm_example #36665
* [CI][Core] Add more visbility into state api stress test (#36465)
* [CI][Doc] Add windows 3.11 wheel support in doc and CI #37297 (#37302)
* [CI][py3.11] Build python wheels on mac os for 3.11 (#36185)
* [CI][python3.11] windows 3.11 wheel build
* [CI][release] Add mac 3.11 wheels to release scripts (#36396)
* [CI] Update state api scale test (#35543)
* [Release Test] Fix dask on ray 1tb sort failure. (#36905)
* [Release Test] Make the cluster name unique for cluster launcher release tests (#35801)
* [Test] Deflakey gcs fault tolerance test in mac os (#36471)
* [Test] Deflakey pubsub integration_test (#36284)
* [Test] Change instance type to r5.8xlarge for dask_on_ray_1tb_sort (#37321) (#37409)
* [Test] Move generators test to large (#35747)
* [Test][Core] Handled the case where memories is empty for dashboard test (#35979)

🔨 **Fixes**:
* [Core] Fix GCS FD usage increase regression. (#35624)
* [Core] Fix issues with worker churn in WorkerPool (#36766)
* [Core] Fix proctitle for generator tasks (#36928)
* [Core] Fix ray.timeline() (#36676)
* [Core] Fix raylet memory leak in the wrong setup. (#35647)
* [Core] Fix test_no_worker_child_process_leaks (#35840)
* [Core] Fix the GCS crash when connecting to a redis cluster with TLS (#36916)
* [Core] Fix the race condition where grpc requests are handled while c… (#37301)
* [Core] Fix the recursion error when async actor has lots of deserialization. (#35494)
* [Core] Fix the segfault from Opencensus upon shutdown (#36906) (#37311)
* [Core] Fix the unnecessary logs (#36931) (#37313)
* [Core] Add a special head node resource and use it to pin the serve controller to the head node (#35929)
* [Core] Add debug log for serialized object size (#35992)
* [Core] Cache schema and test (#37103) (#37201)
* [Core] Fix 'ray stack' on macOS (#36100)
* [Core] Fix a wrong metrics setup link from the doc. (#37312) (#37367)
* [Core] Fix lint (#35844)(#36739)
* [Core] Fix literalinclude path (#35660)
* [Core] Fix microbenchmark (#35823)
* [Core] Fix single_client_wait_1k perf regression (#35614)
* [Core] Get rid of shared_ptr for GcsNodeManager (#36738)
* [Core] Remove extra step in M1 installation instructions (#36029)
* [Core] Remove unnecessary AsyncGetResources in NodeManager::NodeAdded (#36412)
* [Core] Unskip test_Autoscaler_shutdown_node_http_everynode (#36420)
* [Core] Unskip test_get_release_wheel_url for mac (#36430)

📖 **Documentation**:
* [Doc] Clarify that session can also mean a ray cluster (#36422)
* [Doc] Fix doc build on M1 (#35689)
* [Doc] Fix documentation failure due to typing_extensions (#36732)
* [Doc] Make doc code snippet testable [3/n] (#35407)
* [Doc] Make doc code snippet testable [4/n] (#35506)
* [Doc] Make doc code snippet testable [5/n] (#35562)
* [Doc] Make doc code snippet testable [7/n] (#36960)
* [Doc] Make doc code snippet testable [8/n] (#36963)
* [Doc] Some instructions on how to size the head node (#36429)
* [Doc] Fix doc for runtime-env-auth (#36421)
* [Doc][dashboard][state] Promote state api and dashboard usage in Core user guide. (#35760)
* [Doc][python3.11] Update mac os wheels built link (#36379)
* [Doc] [typo] Rename acecelerators.md to accelerators.md (#36500)

Many thanks to all those who contributed to this release!

@ericl, @ArturNiederfahrenhorst, @sihanwang41, @scv119, @aslonnie, @bluecoconut, @alanwguo, @krfricke, @frazierprime, @vitsai, @amogkam, @GeneDer, @jovany-wang, @gjoliver, @simran-2797, @rkooo567, @shrekris-anyscale, @kevin85421, @angelinalg, @maxpumperla, @kouroshHakha, @Yard1, @chaowanggg, @justinvyu, @fantow, @Catch-Bull, @cadedaniel, @ckw017, @hora-anyscale, @rickyyx, @scottsun94, @XiaodongLv, @SongGuyang, @RocketRider, @stephanie-wang, @inpefess, @peytondmurray, @sven1977, @matthewdeng, @ijrsvt, @MattiasDC, @richardliaw, @bveeramani, @rynewang, @woshiyyya, @can-anyscale, @omus, @eax-anyscale, @raulchen, @larrylian, @Deegue, @Rohan138, @jjyao, @iycheng, @akshay-anyscale, @edoakes, @zcin, @dmatrix, @bryant1410, @WanNJ, @architkulkarni, @scottjlee, @JungeAlexander, @avnishn, @harisankar95, @pcmoritz, @wuisawesome, @mattip

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.6.0)

---

## ray-2.5.1: Ray-2.5.1
**Published:** 2023-06-21

The Ray 2.5.1 patch release adds wheels for MacOS for Python 3.11.
It also contains fixes for multiple components, along with fixes for our documentation.

## Ray Train
🔨 Fixes:
* Don't error on eventual success when running with auto-recovery (#36266)

## Ray Core

🎉 New Features:
* Build Python wheels on Mac OS for Python 3.11 (#36373)

🔨 Fixes:
* [Autoscaler] Fix a bug that can cause undefined behavior when clusters attempt to scale up aggressively. (#36241)
* Fix mypy error: Module "ray" does not explicitly export attribute "remote" (#36356)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.5.1)

---

## ray-2.5.0: Ray-2.5.0
**Published:** 2023-06-08

The Ray 2.5 release features focus on a number of enhancements and improvements across the Ray ecosystem, including:

* Training LLMs with Ray Train: New support for checkpointing distributed models, and Pytorch Lightning FSDP to enable training large models on Ray Train’s LightningTrainer
* LLM applications with Ray Serve & Core: New support for streaming responses and model multiplexing
* Improvements to Ray Data: In 2.5, [strict mode ](https://docs.ray.io/en/master/data/faq.html#migrating-to-strict-mode)is enabled by default. This means that schemas are required for all Datasets, and standalone Python objects are no longer supported. Also, the default batch format is fixed to NumPy, giving better performance for [batch inference](https://docs.ray.io/en/master/data/batch_inference.html). 
* RLlib enhancements: New support for multi-gpu training, along with [ray-project/rllib-contrib](https://github.com/ray-project/rllib-contrib) to contain the community contributed algorithms
* Core enhancements: Enable new feature of lightweight resource broadcasting to improve reliability and scalability. Add many enhancements for Core reliability, logging, scheduler, and worker process.

# Ray Libraries
## Ray AIR

💫Enhancements:
* Experiment restore stress tests (#33706)
* Context-aware output engine
  * Add parameter columns to status table (#35388)
  * Context-aware output engine: Add docs, experimental feature docs, prepare default on (#35129)
  * Fix trial status at end (more info + cut off) (#35128)
  * Improve leaked mentions of Tune concepts (#35003)
  * Improve passed time display (#34951)
  * Use flat metrics in results report, use Trainable._progress_metrics (#35035)
  * Print experiment information at experiment start  (#34952)
  * Print single trial config + results as table (#34788)
  * Print out worker ip for distributed train workers. (#33807)
  * Minor fix to print configuration on start. (#34575)
  * Check `air_verbosity` against None. (#33871)
  * Better wording for empty config. (#33811)
* Flatten config and metrics before passing to mlflow (#35074)
* Remote_storage: Prefer fsspec filesystems over native pyarrow (#34663)
* Use filesystem wrapper to exclude files from upload (#34102)
* GCE test variants for air_benchmark and air_examples (#34466)
* New storage path configuration
  * Add `RunConfig.storage_path` to replace `SyncConfig.upload_dir` and `RunConfig.local_dir`. (#33463)
  * Use Ray storage URI as default storage path, if configured [no_early_kickoff] (#34470)
  * Move to new storage_path API in tests and examples (#34263)

🔨 Fixes:
* Store unflattened metrics in _TrackedCheckpoint (#35658) (#35706)
* Fix `test_tune_torch_get_device_gpu` race condition (#35004)
* Deflake test_e2e_train_flow.py (#34308)
* Pin deepspeed version for now to unblock ci. (#34406)
* Fix AIR benchmark configuration link failure. (#34597)
* Fix unused config building function in lightning MNIST example.

📖Documentation:
* Change doc occurrences of ray.data.Dataset to ray.data.Datastream (#34520)
* DreamBooth example: Fix code for batch size > 1 (#34398)
* Synced tabs in AIR getting started (#35170)
* New Ray AIR link for try it out (#34924)
* Correctly Render the Enumerate Numbers in `convert_torch_code_to_ray_air` (#35224)


## Ray Data Processing

🎉 New Features:
* Implement Strict Mode and enable it by default.
* Add column API to Dataset (#35241) 
* Configure progress bars via DataContext (#34638)
* Support using concurrent actors for ActorPool (#34253)
* Add take_batch API for collecting data in the same format as iter_batches and map_batches (#34217)

💫Enhancements:
* Improve map batches error message for strict mode migration (#35368)
* Improve docstring and warning message for from_huggingface (#35206)
* Improve notebook widget display (#34359)
* Implement some operator fusion logic for the new backend (#35178 #34847)
* Use wait based prefetcher by default (#34871) 
* Implement limit physical operator (#34705 #34844) 
* Require compute spec to be explicitly spelled out #34610
* Log a warning if the batch size is misconfigured in a way that would grossly reduce parallelism for actor pool. (#34594)
* Add alias parameters to the aggregate function, and add quantile fn (#34358)
* Improve repr for Arrow Table and pandas types (#34286 #34502)
* Defer first block computation when reading a Datasource with schema information in metadata (#34251)
* Improve handling of KeyboardInterrupt (#34441)
* Validate aggregation key in Aggregate LogicalOperator (#34292)
* Add usage tag for which block formats are used (#34384)
* Validate sort key in Sort LogicalOperator (#34282)
* Combine_chunks before chunking pyarrow.Table block into batches (#34352)
* Use read stage name for naming Data-read tasks on Ray Dashboard (#34341)
* Update path expansion warning (#34221)
* Improve state initialization for ActorPoolMapOperator (#34037)

🔨 Fixes:
* Fix ipython representation (#35414) 
* Fix bugs in handling of nested ndarrays (and other complex object types) (#35359)
* Capture the context when the dataset is first created (#35239)
* Cooperatively exit producer threads for iter_batches (#34819)
* Autoshutdown executor threads when deleted (#34811)
* Fix backpressure when reading directly from input datasource (#34809)
* Fix backpressure handling of queued actor pool tasks (#34254)
* Fix row count after applying filter (#34372)
* Remove unnecessary setting of global logging level to INFO when using Ray Data (#34347)
* Make sure the tf and tensor iteration work in dataset pipeline (#34248)
* Fix '_unwrap_protocol' for Windows systems (#31296) 

📖Documentation:
* Add batch inference object detection example (#35143)
* Refine batch inference doc (#35041)


## Ray Train
🎉 New Features:
* Experimental support for distributed checkpointing (#34709)

💫Enhancements:
* LightningTrainer: Enable prog bar (#35350)
* LightningTrainer enable checkpoint full dict with FSDP strategy (#34967)
* Support FSDP Strategy for LightningTrainer (#34148)

🔨 Fixes:
* Fix HuggingFace -> Transformers wrapping logic (#35276, #35284)
* LightningTrainer always resumes from the latest AIR checkpoint during restoration. (#35617) (#35791)
* Fix lightning trainer devices setting (#34419)
* TorchCheckpoint: Specifying pickle_protocol in `torch.save()` (#35615) (#35790)

📖Documentation:
* Improve visibility of Trainer restore and stateful callback restoration (#34350)
* Fix rendering of diff code-blocks (#34355)
* LightningTrainer Dolly V2 FSDP Fine-tuning Example (#34990)
* Update LightningTrainer MNIST example. (#34867)
* LightningTrainer Advanced Example (#34082, #34429)

🏗 Architecture refactoring:
* Restructure `ray.train` HuggingFace modules (#35270) (#35488)
* rename _base_dataset to _base_datastream (#34423)


## Ray Tune
🎉 New Features:
* Ray Tune's new execution path is now enabled per default (#34840, #34833)

💫Enhancements:
* Make `Tuner.restore(trainable=...) a required argument (#34982)
* Enable `tune.ExperimentAnalysis` to pull experiment checkpoint files from the cloud if needed (#34461)
* Add support for nested hyperparams in PB2 (#31502)
* Release test for durable multifile checkpoints (#34860)
* GCE variants for remaining Tune tests (#34572)
* Add tune frequent pausing release test. (#34501)
* Add PyArrow to ray[tune] dependencies (#34397)
* Fix new execution backend for BOHB (#34828)
* Add tune frequent pausing release test. (#34501)

🔨 Fixes:
* Set config on trial restore (#35000)
* Fix `test_tune_torch_get_device_gpu` race condition (#35004)
* Fix a typo in `tune/execution/checkpoint_manager` state serialization. (#34368)
* Fix tune_scalability_network_overhead by adding `--smoke-test`. (#34167)
* Fix lightning_gpu_tune_.* release test (#35193)

📖Documentation:
* Fix Tune tutorial (#34660)
* Fix typo in Tune restore guide (#34247)

🏗 Architecture refactoring:
* Use Ray-provided `tabulate` package (#34789)


## Ray Serve
🎉 New Features:
* Add support for json logging format.(#35118)
* Add experimental support for model multiplexing.(#35399, #35326)
* Added experimental support for HTTP StreamingResponses. (#35720)
* Add support for application builders & arguments (#34584)

💫Enhancements:
* Add more bucket size for histogram metrics. (#35242).
* Add route information into the custom metrics. (#35246)
* Add HTTPProxy details to Serve Dashboard UI (#35159)
* Add status_code to http qps & latency (#35134)
* Stream Serve logs across different drivers (#35070)
* Add health checking for http proxy actors (#34944)
* Better surfacing of errors in serve status (#34773)
* Enable TLS on gRPCIngress if RAY_USE_TLS is on (#34403)
* Wait until replicas have finished recovering (with timeout) to broadcast `LongPoll` updates (#34675)
* Replace `ClassNode` and `FunctionNode` with `Application` in top-level Serve APIs (#34627)

🔨 Fixes:
* Set `app_msg` to empty string by default (#35646)
* Fix dead replica counts in the stats. (#34761) 
* Add default app name (#34260)
* gRPC Deployment schema check & minor improvements (#34210)

📖Documentation:
* Clean up API reference and various docstrings (#34711)
* Clean up `RayServeHandle` and `RayServeSyncHandle` docstrings & typing (#34714)


## RLlib
🎉 New Features:
* Migrating approximately ~25 of the 30 algorithms from RLlib into [rllib_contrib](https://github.com/ray-project/ray/tree/master/rllib_contrib). You can review the REP [here](https://github.com/ray-project/enhancements/blob/main/reps/2023-04-28-remove-algorithms-from-rllib.md). This release we have covered A3C and MAML. 
* The APPO / IMPALA and PPO are all moved to the new Learner and RLModule stack. 
* The RLModule now supports Checkpointing.([#34717](https://github.com/ray-project/ray/pull/34717) [#34760](https://github.com/ray-project/ray/pull/34760))

💫Enhancements:
* Introduce experimental larger than GPU train batch size feature for torch (#34189)
* Change occurences of "_observation_space_in_preferred_format" to "_obs_space_in_preferred_format" (#33907)
* Add a flag to allow disabling initialize_loss_from_dummy_batch logit. (#34208)
* Remove check specs from default Model forward code path to improve performance (#34877)
* Remove some specs from encoders to smoothen dev experience (#34911)

🔨 Fixes:
* Fix MultiCallbacks class: To be used only with utility function that returns a class to use in the config. (#33863)
* Fix test backward compatibility test for RL Modules (#33857)
* Don't serialize config in Policy states (unless needed for msgpack-type checkpoints). (#33865)
* DM control suite wrapper fix: dtype of obs needs to be pinned to float32. (#33876)
* In the Json_writer convert all non string keys to keys (#33896)
* Fixed a bug with kl divergence calculation of torch.Dirichlet distribution within RLlib (#34209)
* Change broken link in parameter_noise.py (#34231)
* Fixed bug in restoring a gpu trained algorithm (#35024)
* Fix IMPALA/APPO when using multi GPU setup and Multi-Agent Env (#35120)

📖Documentation:
* Add examples and docs for Catalog (#33898)

# Ray Core and Ray Clusters
## Ray Core
🎉 New Features:
* Support both sync and async actor generator interface. (#35584 #35708 #35324 #35656 #35803 #35794 #35707)

💫Enhancements:
* [Scheduler] Introduce spill_on_unavailable option for soft NodeAffinitySchedulingStrategy (#34224)
* [Data] Use wait based prefetcher by default (#34871)
* [Reliability] During GCS restarts, grpc based resource broadcaster should only add ALIVE nodes during initialization (#35349)
* [Reliability] Guarantee the ordering of put ActorTaskSpecTable and ActorTable (#35683) (#35718)
* [Reliability] Graceful handling of returning bundles when node is removed (#34726) (#35542)
* [Reliability] Task backend - marking tasks failed on worker death (#33818)
* [Reliability] Task backend - Add worker dead info to failed tasks when job exits. (#34166)
* [Logging] Make ray.get(timeout=0) to throw timeout error (#35126)
* [Logging] Provide good error message if the factional resource precision is beyond 0.0001 (#34590)
* [Logging] Add debug logs to show UpdateResourceUsage rpc source (#35062)
* [Logging] Add actor_id as an attribute of RayActorError when the actor constructor fails (#34958)
* [Logging] Worker startup hook (#34738)
* [Worker] Partially addresses ray child process leaks by killing all child processes in the CoreWorker shutdown sequence. (#33976)
* [Worker] Change worker niceness in job submission environment (#34727)
* Shorten the membership checking time to 5 seconds. (#34769)
* [Syncer] Remove spammy logs. (#34654)
* [Syncer] Delete disconnected node view in ray syncer when connection is broken. (#35312)
* [Syncer] Turn on ray syncer again. (#35116)
* [Syncer] Start ray syncer reconnection after a delay (#35115)
* Serialize requests in the redis store client. (#35123)
* Reduce self alive check from 60s to 5s. (#34992)
* Add object owner and copy metrics to node stats (#35119)
* Start the synchronization connection after receiving all nodes info. (#34645)
* Improve the workflow finding Redis leader. (#34108)
* Make execute_after accept chrono (#35099)
* Lazy import autoscaler + don't import opentelemetry unless setup hook (#33964)

🔨 Fixes:
* [pubsub] Handle failures when publish fails. (#33115)
* Convert gcs port read from env variable from str to int (#34482)
* Fix download_wheels.sh wheel urls (#34616)
* Fix ray start command output (#34081)
* Fetch_local once for each object ref (#34884)
* Combine_chunks before chunking pyarrow.Table block into batches (#34352)
* Replace deprecated usage of get_runtime_context().node_id (#34874)
* Fix std::move without std namespace (#34149)
* Fix the recursion error when an async actor has lots of deserialization. (#35494) (#35532)
* Fix async actor shutdown issue when exit_actor is used (#32407)
* [Event] Fix incorrect event timestamp (#34402)
* [Metrics] Fix shared memory is not displayed properly (#34301)
* Fix GCS FD usage increase regression. (#35624) (#35738)
* Fix raylet memory leak in the wrong setup. (#35647) (#35673)
* Retry failed redis request (#35249) (#35481)
* Add more messages when accessing a dead actor. (#34697)
* Fix the placement group stress test regression. (#34192)
* Mark Raylet unhealthy if GCS can't recognize it. (#34087)
* Remove multiple core workers in one process 2/n (#34942)
* Remove python 3.6 support (#34373 #34416)

📖Documentation:
* Make doc code snippet testable (#35274 #35057)
* Revamp ray core api reference [1/n] (#34428)
* Add Ray core fault tolerance guide for GCS and node (#33446)
* Ray Debugging Doc Part 1 (OOM) (#34309)
* Rewrite the placement group documentation (#33518)


## Ray Clusters
💫Enhancements:
* [Docker] [runtime env] Bump boto3 version from 1.4.8 to 1.26.82, add pyOpenSSL and cryptography (#33273)
* [Jobs] Fix race condition in supervisor actor creation and add timeout for pending jobs (#34223)
* [Release test] [Cluster launcher] Add gcp minimal and full cluster launcher release test (#34878)
* [Release test] [Cluster launcher] Add release test for aws `example-full.yaml` (#34487)

📖Documentation:
* [runtime env] Clarify conditions for local `pip` and `conda` requirements files (#34071)
* [KubeRay] Provide GKE instructions in KubeRay example (#33339)
* [KubeRay] Update KubeRay doc for release v0.5.0 (#34178)


## Dashboard
💫Enhancements:
* Feature flag task logs recording (#34056)
* Fix log proxy not loading non test/plain files. (#33870)
* [no_early_kickoff] Make dashboard address connectable from remote nodes when not set to 127.0.0.1 (localhost) (#35027)
* [state][job] Supporting job listing(getting) and logs from state API (#35124)
* [state][ci] Fix stress_test_state_api_scale (#35332)
* [state][dashboard][log] Fix subdirectory log getting (#35283)
* [state] Push down filtering to GCS for listing/getting task from state api (#35109)(#34433)
* [state] Task log - Improve log tailing from log_client and support tailing from offsets [2/4] (#28188)
* [state] Use `--err` flag to query stderr logs from worker/actors instead of `--suffix=err` (#34300)
* [state][no_early_kickoff] Make state api return results that are strongly typed (#34297)
* [state] Efficient get/list actors with filters on some high-cardinality fields (#34348)
* [state] Fix list nodes test in test_state_api.py (#34349)
* [state] Add head node flag `is_head_node` to state API and GcsNodeInfo (#34299)
* Make actor tasks' name default to <actor_repr>.<task_name> (#35371)
* Task backend GC policy - worker update [1/3] (#34896)
* [state] Support task logs from state API (#35101)

## Known Issues
* A bug in the Autoscaler can cause undefined behaviour when clusters attempt to scale up aggressively. This is fixed in following releases, as well as post-release on [the 2.5.0 branch](https://github.com/ray-project/ray/tree/releases/2.5.0) (#36482).


Many thanks to all those who contributed to this release!

@vitsai, @XiaodongLv, @justinvyu, @Dan-Yeh, @dependabot[bot], @alanwguo, @grimreaper, @yiwei00000, @pomcho555, @ArturNiederfahrenhorst, @maxpumperla, @jjyao, @ijrsvt, @sven1977, @Yard1, @pcmoritz, @c21, @architkulkarni, @jbedorf, @amogkam, @ericl, @jiafuzha, @clarng, @shrekris-anyscale, @matthewdeng, @gjoliver, @jcoffi, @edoakes, @ethanabrooks, @iycheng, @Rohan138, @angelinalg, @Linniem, @aslonnie, @zcin, @wuisawesome, @Catch-Bull, @woshiyyya, @avnishn, @jjyyxx, @jianoaix, @bveeramani, @sihanwang41, @scottjlee, @YQ-Wang, @mattip, @can-anyscale, @xwjiang2010, @fedassembly, @joncarter1, @robin-anyscale, @rkooo567, @DACUS1995, @simran-2797, @ProjectsByJackHe, @zen-xu, @ashahab, @larrylian, @kouroshHakha, @raulchen, @sofianhnaide, @scv119, @nathan-az, @kevin85421, @rickyyx, @Sahar-E, @krfricke, @chaowanggg, @peytondmurray, @cadedaniel

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.5.0)

---

## ray-2.3.1: Ray-2.3.1
**Published:** 2023-03-27

The Ray 2.3.1 patch release contains fixes for multiple components:

## Ray Data Processing
* Support different number of blocks/rows per block in `zip()` (https://github.com/ray-project/ray/pull/32795)

## Ray Serve
* Revert `serve run` to use Ray Client instead of Ray Jobs (https://github.com/ray-project/ray/pull/32976)
* Fix issue with `max_concurrent_queries` being ignored when autoscaling (https://github.com/ray-project/ray/pull/32772 and https://github.com/ray-project/ray/pull/33022)
## Ray Core
* Write Ray address even if Ray node is started with `--block` (https://github.com/ray-project/ray/pull/32961)
* Fix Ray on Spark running on layered virtualenv python environment (https://github.com/ray-project/ray/pull/32996)
## Dashboard
* Fix disk metric showing double the actual value (https://github.com/ray-project/ray/pull/32674)


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.3.1)

---

## ray-2.3.0: Ray-2.3.0
**Published:** 2023-02-24

# Release Highlights



* The streaming backend for Ray Datasets is in Developer Preview. It is designed to enable terabyte-scale ML inference and training workloads. Please contact us if you'd like to try it out on your workload, or you can find the preview guide here: https://docs.google.com/document/d/1BXd1cGexDnqHAIVoxTnV3BV0sklO9UXqPwSdHukExhY/edit
* New Information Architecture (**Beta**): We’ve restructured the [Ray dashboard](https://docs.ray.io/en/master/ray-core/ray-dashboard.html) to be organized around user personas and workflows instead of entities. 
* Ray-on-Spark is now available (Preview)!: You can launch Ray clusters on Databricks and Spark clusters and run Ray applications. Check out the [documentation](https://docs.ray.io/en/releases-2.3.0/cluster/vms/user-guides/community/spark.html) to learn more.

# Ray Libraries


## Ray AIR

💫Enhancements:



* Add `set_preprocessor` method to `Checkpoint` (#31721)
* Rename Keras callback and its parameters to be more descriptive (#31627)
* Deprecate MlflowTrainableMixin in favor of setup_mlflow() function (#31295)
* W&B
    * Have train_loop_config logged as a config (#31901)
    * Allow users to exclude config values with WandbLoggerCallback (#31624)
    * Rename WandB `save_checkpoints` to `upload_checkpoints` (#31582)
    * Add hook to get project/group for W&B integration (#31035, 31643)
    * Use Ray actors instead of multiprocessing for WandbLoggerCallback (#30847)
    * Update `WandbLoggerCallback` example (#31625)
* Predictor
    * Place predictor kwargs in object store (#30932)
    * Delegate BatchPredictor stage fusion to Datasets (#31585)
    * Rename `DLPredictor.call_model` `tensor` parameter to `inputs` (#30574)
    * Add `use_gpu` to `HuggingFacePredictor` (#30945)
* Checkpoints
    * Various `Checkpoint` improvements (#30948)
    * Implement lazy checkpointing for same-node case (#29824)
    * Automatically strip "module." from state dict (#30705)
    * Allow user to pass model to `TensorflowCheckpoint.get_model` (#31203)

🔨 Fixes:



* Fix and improve support for HDFS remote storage. (#31940)
* Use specified Preprocessor configs when using stream API. (#31725)
* Support nested Chain in BatchPredictor (#31407)

📖Documentation:



* Restructure API References (#32535)
* API Deprecations (#31777, #31867)
* Various fixes to docstrings, documentation, and examples (#30782, #30791)

🏗 Architecture refactoring:



* Use NodeAffinitySchedulingPolicy for scheduling (#32016)
* Internal resource management refactor (#30777, #30016)


## Ray Data Processing

🎉 New Features:



* Lazy execution by default (#31286)
* Introduce streaming execution backend (#31579) 
* Introduce DatasetIterator (#31470)
* Add per-epoch preprocessor (#31739)
* Add TorchVisionPreprocessor (#30578)
* Persist Dataset statistics automatically to log file (#30557)

💫Enhancements:



* Async batch fetching for map_batches (#31576)
* Add informative progress bar names to map_batches (#31526)
* Provide an size bytes estimate for mongodb block (#31930)
* Add support for dynamic block splitting to actor pool (#31715)
* Improve str/repr of Dataset to include execution plan (#31604)
* Deal with nested Chain in BatchPredictor (#31407)
* Allow MultiHotEncoder to encode arrays (#31365)
* Allow specify batch_size when reading Parquet file (#31165)
* Add zero-copy batch API for `ds.map_batches()` (#30000)
* Text dataset should save texts in ArrowTable format (#30963)
* Return ndarray dicts for single-column tabular datasets (#30448)
* Execute randomize_block_order eagerly if it's the last stage for ds.schema() (#30804)

🔨 Fixes:



* Don't drop first dataset when peeking DatasetPipeline (#31513)
* Handle np.array(dtype=object) constructor for ragged ndarrays (#31670)
* Emit warning when starting Dataset execution with no CPU resources available (#31574)
* Fix the bug of eagerly clearing up input blocks (#31459)
* Fix Imputer failing with categorical dtype (#31435)
* Fix schema unification for Datasets with ragged Arrow arrays (#31076)
* Fix Discretizers transforming ignored cols (#31404)
* Fix to_tf when the input feature_columns is a list. (#31228)
* Raise error message if user calls Dataset.__iter__ (#30575)

📖Documentation:



* Refactor Ray Data API documentation (#31204)
* Add seealso to map-related methods (#30579)


## Ray Train

🎉 New Features:



* Add option for per-epoch preprocessor (#31739)

💫Enhancements:



* Change default `NCCL_SOCKET_IFNAME` to blacklist `veth` (#31824)
* Introduce DatasetIterator for bulk and streaming ingest (#31470)
* Clarify which `RunConfig` is used when there are multiple places to specify it (#31959)
* Change `ScalingConfig` to be optional for `DataParallelTrainer`s if already in Tuner `param_space` (#30920)

🔨 Fixes:



* Use specified `Preprocessor` configs when using stream API. (#31725)
* Fix off-by-one AIR Trainer checkpoint ID indexing on restore (#31423)
* Force GBDTTrainer to use distributed loading for Ray Datasets (#31079)
* Fix bad case in ScalingConfig->RayParams (#30977)
* Don't raise TuneError on `fail_fast="raise"` (#30817)
* Report only once in `SklearnTrainer` (#30593)
* Ensure GBDT PGFs match passed ScalingConfig (#30470)

📖Documentation:



* Restructure API References (#32535)
* Remove Ray Client references from Train docs/examples (#32321)
* Various fixes to docstrings, documentation, and examples (#29463, #30492, #30543, #30571, #30782, #31692, #31735)

🏗 Architecture refactoring:



* API Deprecations (#31763)


## Ray Tune

💫Enhancements:



* Improve trainable serialization error (#31070)
* Add support for Nevergrad optimizer with extra parameters (#31015)
* Add timeout for experiment checkpoint syncing to cloud (#30855)
* Move `validate_upload_dir` to Syncer (#30869)
* Enable experiment restore from moved cloud uri (#31669)
* Save and restore stateful callbacks as part of experiment checkpoint (#31957)

🔨 Fixes:



* Do not default to reuse_actors=True when mixins are used (#31999)
* Only keep cached actors if search has not ended (#31974)
* Fix best trial in ProgressReporter with nan (#31276)
* Make ResultGrid return cloud checkpoints (#31437)
* Wait for final experiment checkpoint sync to finish (#31131)
* Fix CheckpointConfig validation for function trainables (#31255)
* Fix checkpoint directory assignment for new checkpoints created after restoring a function trainable (#31231)
* Fix `AxSearch` save and nan/inf result handling (#31147)
* Fix `AxSearch` search space conversion for fixed list hyperparameters (#31088)
* Restore searcher and scheduler properly on `Tuner.restore` (#30893)
* Fix progress reporter `sort_by_metric` with nested metrics (#30906)
* Don't raise TuneError on `fail_fast="raise"` (#30817)
* Fix duplicate printing when trial is done (#30597)

📖Documentation:



* Restructure API references (#32449)
* Remove Ray Client references from Tune docs/examples (#32321)
* Various fixes to docstrings, documentation, and examples (#29581, #30782, #30571, #31045, #31793, #32505)

🏗 Architecture refactoring:



* Deprecate passing a custom trial executor (#31792)
* Move signal handling into separate method (#31004)
* Update staged resources in a fixed counter for faster lookup (#32087)
* Rename `overwrite_trainable` argument in Tuner restore to `trainable` (#32059)


## Ray Serve

🎉 New Features:



* Serve python API to support multi application (#31589)

💫Enhancements:



* Add exponential backoff when retrying replicas (#31436)
* Enable Log Rotation on Serve (#31844)
* Use tasks/futures for asyncio.wait (#31608)
* Change target_num_ongoing_requests_per_replica to positive float (#31378)

🔨 Fixes:



* Upgrade deprecated calls (#31839)
* Change Gradio integration to take a builder function to avoid serialization issues (#31619)
* Add initial health check before marking a replica as RUNNING (#31189)

📖Documentation:



* Document end-to-end timeout in Serve (#31769)
* Document Gradio visualization (#28310)


## RLlib

🎉 New Features:



* Gymnasium is now supported. ([Notes](https://docs.google.com/document/u/0/d/1lxYK1dI5s0Wo_jmB6V6XiP-_aEBsXDykXkD1AXRase4/edit))
* Connectors are now activated by default (#31693, 30388, 31618, 31444, 31092)
* Contribution of LeelaChessZero algorithm for playing chess in a MultiAgent env. (#31480)

💫Enhancements:



* [RLlib] Error out if action_dict is empty in MultiAgentEnv.  (#32129)
* [RLlib] Upgrade tf eager code to no longer use `experimental_relax_shapes` (but `reduce_retracing` instead). (#29214)
* [RLlib] Reduce SampleBatch counting complexity (#30936)
* [RLlib] Use PyTorch vectorized max() and sum() in SampleBatch.__init__ when possible (#28388)
* [RLlib] Support multi-gpu CQL for torch (tf already supported). (#31466)
* [RLlib] Introduce IMPALA off_policyness test with GPU (#31485)
* [RLlib] Properly serialize and restore StateBufferConnector states for policy stashing (#31372)
* [RLlib] Clean up deprecated concat_samples calls (#31391)
* [RLlib] Better support MultiBinary spaces by treating Tuples as superset of them in ComplexInputNet. (#28900)
* [RLlib] Add backward compatibility to MeanStdFilter to restore from older checkpoints. (#30439)
* [RLlib] Clean up some signatures for compute_actions. (#31241)
* [RLlib] Simplify logging configuration. (#30863)
* [RLlib] Remove native Keras Models. (#30986)
* [RLlib] Convert PolicySpec to a readable format when converting to_dict(). (#31146)
* [RLlib] Issue 30394: Add proper `__str__()` method to PolicyMap. (#31098)
* [RLlib] Issue 30840: Option to only checkpoint policies that are trainable. (#31133)
* [RLlib] Deprecate (delete) `contrib` folder. (#30992)
* [RLlib] Better behavior if user does not specify stopping condition in RLLib CLI. (#31078)
* [RLlib] PolicyMap LRU cache enhancements: Swap out policies (instead of GC'ing and recreating) + use Ray object store (instead of file system). (#29513)
* [RLlib] `AlgorithmConfig.overrides()` to replace `multiagent->policies->config` and `evaluation_config` dicts. (#30879)
* [RLlib] `deprecation_warning(.., error=True)` should raise `ValueError`, not `DeprecationWarning`. (#30255)
* [RLlib] Add `gym.spaces.Text` serialization. (#30794)
* [RLlib] Convert `MultiAgentBatch` to `SampleBatch` in offline_rl.py. (#30668)
* [RLlib; Tune] Make `Algorithm.train()` return Tune-style config dict (instead of AlgorithmConfig object). (#30591)

🔨 Fixes:



* [RLlib] Fix waterworld example and test (#32117)
* [RLlib] Change Waterworld v3 to v4 and reinstate indep. MARL test case w/ pettingzoo. (#31820)
* [RLlib] Fix OPE checkpointing. Save method name in configuration dict. (#31778)
* [RLlib] Fix worker state restoration. (#31644)
* [RLlib] Replace ordinary pygame imports by `try_import_..()`. (#31332)
* [RLlib] Remove crude VR checks in agent collector. (#31564)
* [RLlib] Fixed the 'RestoreWeightsCallback' example script. (#31601)
* [RLlib] Issue 28428: QMix not working w/ GPUs. (#31299)
* [RLlib] Fix using yaml files with empty stopping conditions. (#31363)
* [RLlib] Issue 31174: Move all checks into AlgorithmConfig.validate() (even simple ones) to avoid errors when using tune hyperopt objects. (#31396)
* [RLlib] Fix `tensorflow_probability` imports. (#31331)
* [RLlib] Issue 31323: BC/MARWIL/CQL do work with multi-GPU (but config validation prevents them from running in this mode). (#31393)
* [RLlib] Issue 28849: DT fails with num_gpus=1. (#31297)
* [RLlib] Fix `PolicyMap.__del__()` to also remove a deleted policy ID from the internal deque. (#31388)
* [RLlib] Use `get_model_v2()` instead of `get_model()` with MADDPG. (#30905)
* [RLlib] Policy mapping fn can not be called with keyword arguments. (#31141)
* [RLlib] Issue 30213: Appending RolloutMetrics to sampler outputs should happen after(!) all callbacks (such that custom metrics for last obs are still included). (#31102)
* [RLlib] Make convert_to_torch tensor adhere to docstring. (#31095)
* [RLlib] Fix convert to torch tensor (#31023)
* [RLlib] Issue 30221: random policy does not handle nested spaces. (#31025)
* [RLlib] Fix crashing remote envs example (#30562)
* [RLlib] Recursively look up the original space from obs_space (#30602)

📖Documentation:



* [RLlib; docs] Change links and references in code and docs to "Farama foundation's gymnasium" (from "OpenAI gym"). (#32061)


# Ray Core and Ray Clusters


## Ray Core

🎉 New Features:



* Task Events Backend: Ray aggregates all submitted task information to provide better observability (#31840, #31761, #31278, #31247, #31316, #30934, #30979, #31207, #30867, #30829, #31524, #32157). This will back up features like task state API, advanced progress bar, and Ray timeline.

💫Enhancements:



* Remote generator now works for ray actors and ray clients (#31700, #31710).
* Revamp default scheduling strategy, improve worker startup performance up to 8x for embarrassingly parallel workloads (#31934, #31868).
* Worker code clean up and allow workers lazy bind to jobs (#31836, #31846, #30349, #31375).
* A single Ray cluster can scale up to 2000 nodes and 20k actors(#32131, #30131, #31939, #30166, #30460, #30563).
* [Out-of-memory prevention enhancement](https://docs.ray.io/en/master/ray-core/scheduling/ray-oom-prevention.html) is now GA with more robust worker killing policies and better user experiences (#32217, #32361, #32219, #31768, #32107, #31976, #31272, #31509, #31230).

🔨 Fixes:



* Improve garbage collection upon job termination (#32127, #31155)
* Fix opencensus protobuf bug (#31632)
* Support python 3.10 for runtime_env conda (#30970)
* Fix crashes and memory leaks (#31640, #30476, #31488, #31917, #30761, #31018)

📖Documentation:



* Deprecation (#31845, #31140, #31528)


## Ray Clusters

🎉 New Features:
* Ray-on-Spark is now available as Preview! (#28771, #31397, #31962) 

💫Enhancements:



* [observability] Better memory formatting for `ray status` and autoscaler (#32337) 
* [autoscaler] Add flag to disable periodic cluster status log. (#31869) 

🔨 Fixes:



* [observability][autoscaler] Ensure pending nodes is reset to 0 after scaling (#32085)
* Make ~/.bashrc optional in cluster launcher commands (#32393)

📖Documentation:



* Improvements to job submission
* Remove references to Ray Client


## Dashboard

🎉 New Features:



* New Information Architecture (beta): We’ve restructured the Ray dashboard to be organized around user personas and workflows instead of entities. For developers, the jobs and actors tab will be most useful. For infrastructure engineers, the cluster tab may be more valuable.
* Advanced progress bar: Tasks visualization that allow you to see the progress of all your ray tasks
* Timeline view: We’ve added a button to download detailed timeline data about your ray job. Then, one can click a link and use the perfetto open-source visualization tool to visualize the timeline data.
* More metadata tables. You can now see placement groups, tasks, actors, and other information related to your jobs.

📖Documentation:



* We’ve restructured the documentation to make the dashboard documentation more prominent
* We’ve improved the documentation around setting up Prometheus and Grafana for metrics.

Many thanks to all those who contributed to this release!

@minerharry, @scottsun94, @iycheng, @DmitriGekhtman, @jbedorf, @krfricke, @simonsays1980, @eltociear, @xwjiang2010, @ArturNiederfahrenhorst, @richardliaw, @avnishn, @WeichenXu123, @Capiru, @davidxia, @andreapiso, @amogkam, @sven1977, @scottjlee, @kylehh, @yhna940, @rickyyx, @sihanwang41, @n30111, @Yard1, @sriram-anyscale, @Emiyalzn, @simran-2797, @cadedaniel, @harelwa, @ijrsvt, @clarng, @pabloem, @bveeramani, @lukehsiao, @angelinalg, @dmatrix, @sijieamoy, @simon-mo, @jbesomi, @YQ-Wang, @larrylian, @c21, @AndreKuu, @maxpumperla, @architkulkarni, @wuisawesome, @justinvyu, @zhe-thoughts, @matthewdeng, @peytondmurray, @kevin85421, @tianyicui-tsy, @cassidylaidlaw, @gvspraveen, @scv119, @kyuyeonpooh, @Siraj-Qazi, @jovany-wang, @ericl, @shrekris-anyscale, @Catch-Bull, @jianoaix, @christy, @MisterLin1995, @kouroshHakha, @pcmoritz, @csko, @gjoliver, @clarkzinzow, @SongGuyang, @ckw017, @ddelange, @alanwguo, @Dhul-Husni, @Rohan138, @rkooo567, @fzyzcjy, @chaokunyang, @0x2b3bfa0, @zoltan-fedor, @Chong-Li, @crypdick, @jjyao, @emmyscode, @stephanie-wang, @starpit, @smorad, @nikitavemuri, @zcin, @tbukic, @ayushthe1, @mattip


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.3.0)

---

## ray-2.2.0: Ray-2.2.0
**Published:** 2022-12-13

# Release Highlights

Ray 2.2 is a stability-focused release, featuring stability improvements across many Ray components. 

* [Ray Jobs API](https://docs.ray.io/en/releases-2.2.0/cluster/running-applications/job-submission/index.html#ray-jobs-api) is now GA. The Ray Jobs API allows you to submit locally developed applications to a remote Ray Cluster for execution. It simplifies the experience of packaging, deploying, and managing a Ray application.
* [Ray Dashboard](https://docs.ray.io/en/releases-2.2.0/ray-core/ray-dashboard.html#ray-dashboard) has received a number of improvements, such as the ability to see cpu flame graphs of your Ray workers and new metrics for memory usage.
* The [Out-Of-Memory (OOM) Monitor](https://docs.ray.io/en/releases-2.2.0/ray-core/scheduling/ray-oom-prevention.html) is now enabled by default. This will increase the stability of memory-intensive applications on top of Ray.
* [Ray Data] we’ve heard numerous users report that when files are too large, Ray Data can have out-of-memory or performance issues. In this release, we’re enabling [dynamic block splitting](https://docs.ray.io/en/releases-2.2.0/data/dataset-internals.html#execution-memory) by default, which will address the above issues by avoiding holding too much data in memory.

# Ray Libraries

## Ray AIR

🎉 New Features:
* Add a NumPy first path for Torch and TensorFlow Predictors (#28917)

💫Enhancements:
* Suppress "NumPy array is not writable" error in torch conversion (#29808)
* Add node rank and local world size info to session (#29919)

🔨 Fixes:
* Fix MLflow database integrity error (#29794)
* Fix ResourceChangingScheduler dropping PlacementGroupFactory args (#30304)
* Fix bug passing 'raise' to FailureConfig (#30814)
* Fix reserved CPU warning if no CPUs are used (#30598)

📖Documentation:
* Fix examples and docs to specify batch_format in BatchMapper (#30438)

🏗 Architecture refactoring:
* Deprecate Wandb mixin (#29828)
* Deprecate Checkpoint.to_object_ref and Checkpoint.from_object_ref (#30365)

## Ray Data Processing

🎉 New Features:
* Support all PyArrow versions released by Apache Arrow (#29993, #29999)
* Add `select_columns()` to select a subset of columns (#29081)
* Add `write_tfrecords()` to write TFRecord files (#29448)
* Support MongoDB data source (#28550)
* Enable dynamic block splitting by default (#30284)
* Add `from_torch()` to create dataset from Torch dataset (#29588)
* Add `from_tf()` to create dataset from TensorFlow dataset (#29591)
* Allow to set `batch_size` in `BatchMapper` (#29193)
* Support read/write from/to local node file system (#29565)

💫Enhancements:
* Add `include_paths` in `read_images()` to return image file path (#30007)
* Print out Dataset statistics automatically after execution (#29876)
* Cast tensor extension type to opaque object dtype in `to_pandas()` and `to_dask()` (#29417)
* Encode number of dimensions in variable-shaped tensor extension type (#29281)
* Fuse AllToAllStage and OneToOneStage with compatible remote args (#29561)
* Change `read_tfrecords()` output from Pandas to Arrow format (#30390)
* Handle all Ray errors in task compute strategy (#30696)
* Allow nested Chain preprocessors (#29706)
* Warn user if missing columns and support `str` exclude in `Concatenator` (#29443)
* Raise ValueError if preprocessor column doesn't exist (#29643)

🔨 Fixes:
* Support custom resource with remote args for `random_shuffle()` (#29276)
* Support custom resource with remote args for `random_shuffle_each_window()` (#29482)
* Add PublicAPI annotation to preprocessors (#29434)
* Tensor extension column concatenation fixes (#29479)
* Fix `iter_batches()` to not return empty batch (#29638)
* Change `map_batches()` to fetch input blocks on-demand (#29289)
* Change `take_all()` to not accept limit argument (#29746)
* Convert between block and batch correctly for `map_groups()` (#30172)
* Fix `stats()` call causing Dataset schema to be unset (#29635)
* Raise error when `batch_format` is not specified for `BatchMapper` (#30366)
* Fix ndarray representation of single-element ragged tensor slices (#30514)

📖Documentation:
* Improve `map_batches()` documentation about execution model and UDF pickle-ability requirement (#29233)
* Improve `to_tf()` docstring (#29464)

## Ray Train

🎉 New Features:
* Added MosaicTrainer (#29237, #29620, #29919)

💫Enhancements:
* Fast fail upon single worker failure (#29927)
* Optimize checkpoint conversion logic (#29785) 

🔨 Fixes:
* Propagate DatasetContext to training workers (#29192)
* Show correct error message on training failure (#29908) 
* Fix prepare_data_loader with enable_reproducibility (#30266) 
* Fix usage of NCCL_BLOCKING_WAIT (#29562)

📖Documentation:
* Deduplicate Train examples (#29667)

🏗 Architecture refactoring:
* Hard deprecate train.report (#29613)
* Remove deprecated Train modules (#29960)
* Deprecate old prepare_model DDP args #30364

## Ray Tune

🎉 New Features:
* Make `Tuner.restore` work with relative experiment paths (#30363)
* `Tuner.restore` from a local directory that has moved (#29920)

💫Enhancements:
* `with_resources` takes in a `ScalingConfig` (#30259)
* Keep resource specifications when nesting `with_resources` in `with_parameters` (#29740)
* Add `trial_name_creator` and `trial_dirname_creator` to `TuneConfig` (#30123)
* Add option to not override the working directory (#29258)
* Only convert a `BaseTrainer` to `Trainable` once in the Tuner (#30355)
* Dynamically identify PyTorch Lightning Callback hooks (#30045)
* Make `remote_checkpoint_dir` work with query strings (#30125)
* Make cloud checkpointing retry configurable (#30111)
* Sync experiment-checkpoints more often (#30187)
* Update generate_id algorithm (#29900)

🔨 Fixes:
* Catch SyncerCallback failure with dead node (#29438)
* Do not warn in BayesOpt w/ Uniform sampler (#30350)    
* Fix `ResourceChangingScheduler` dropping PGF args (#30304)
* Fix Jupyter output with Ray Client and `Tuner` (#29956)
* Fix tests related to `TUNE_ORIG_WORKING_DIR` env variable (#30134)

📖Documentation:
* Add user guide for analyzing results (using `ResultGrid` and `Result`) (#29072)    
* Tune checkpointing and Tuner restore docfix (#29411)
* Fix and clean up PBT examples (#29060)
* Fix TrialTerminationReporter in docs (#29254)

🏗 Architecture refactoring:
* Remove hard deprecated SyncClient/Syncer (#30253)
* Deprecate Wandb mixin, move to `setup_wandb()` function (#29828)

## Ray Serve

🎉 New Features:
* Guard for high latency requests (#29534)
* Java API Support ([blog](http://anyscale-staging.herokuapp.com/blog/flexible-cross-language-distributed-model-inference-framework-ray-serve-with))

💫Enhancements:
* Serve K8s HA benchmarking (#30278)
* Add method info for http metrics (#29918)

🔨 Fixes:
* Fix log format error (#28760)
* Inherit previous deployment num_replicas (29686)
* Polish serve run deploy message (#29897)
* Remove calling of get_event_loop from python 3.10

## RLlib

🎉 New Features:
* Fault tolerant, elastic WorkerSets: An asynchronous Ray Actor manager class is now used inside all of RLlib’s Algorithms, adding fully flexible fault tolerance to rollout workers and workers used for evaluation. If one or more workers (which are Ray actors) fails - e.g. due to a SPOT instance going down - the RLlib Algorithm will now flexibly wait it out and periodically try to recreate the failed workers. In the meantime, only the remaining healthy workers are used for sampling and evaluation. (#29938, #30118, #30334, #30252, #29703, #30183, #30327, #29953)

💫Enhancements:
* RLlib CLI: A new and enhanced RLlib command line interface (CLI) has been added, allowing for automatically downloading example configuration files, python-based config files (defining an AlgorithmConfig object to use), better interoperability between training and evaluation runs, and many more. For a detailed overview of what has changed, check out the [new CLI documentation](https://docs.ray.io/en/releases-2.2.0/rllib/rllib-cli.html). (#29204, #29459, #30526, #29661, #29972)
* Checkpoint overhaul: Algorithm checkpoints and Policy checkpoints are now more cohesive and transparent. All checkpoints are now characterized by a directory (with files and maybe sub-directories), rather than a single pickle file; Both Algorithm and Policy classes now have a utility static method (`from_checkpoint()`) for directly instantiating instances from a checkpoint directory w/o knowing the original configuration used or any other information (having the checkpoint is sufficient). [For a detailed overview, see here](https://docs.ray.io/en/releases-2.2.0/rllib/rllib-saving-and-loading-algos-and-policies.html). (#28812, #29772, #29370, #29520, #29328)
* A new metric for APPO/IMPALA/PPO has been added that measures off-policy’ness: The difference in number of grad-updates the sampler policy has received thus far vs the trained policy’s number of grad-updates thus far. (#29983)

🏗 Architecture refactoring:
* AlgorithmConfig classes: All of RLlib’s Algorithms, RolloutWorkers, and other important classes now use AlgorithmConfig objects under the hood, instead of python config dicts. It is no longer recommended (however, still supported) to create a new algorithm (or a Tune+RLlib experiment) using a python dict as configuration. For more details on how to convert your scripts to the [new AlgorithmConfig design, see here](https://docs.ray.io/en/releases-2.2.0/rllib/rllib-training.html#configuring-rllib-algorithms). (#29796, #30020, #29700, #29799, #30096, #29395, #29755, #30053, #29974, #29854, #29546, #30042, #29544, #30079, #30486, #30361)
* Major progress was made on the new Connector API and making sure it can be used (tentatively) with the “config.rollouts(enable_connectors=True)” flag. Will be fully supported, across all of RLlib’s algorithms, in Ray 2.3. (#30307, #30434, #30459, #30308, #30332, #30320, #30383, #30457, #30446, #30024, #29064, #30398, #29385, #30481, #30241, #30285, #30423, #30288, #30313, #30220, #30159)
* Progress was made on the upcoming RLModule/RLTrainer/RLOptimizer APIs. (#30135, #29600, #29599, #29449, #29642)

🔨 Fixes:
* Various bug fixes: #25925, #30279, #30478, #30461, #29867, #30099, #30185, #29222, #29227, #29494, #30257, #29798, #30176, #29648, #30331

📖Documentation:
* [RLlib CLI](https://docs.ray.io/en/releases-2.2.0/rllib/rllib-cli.html), [Checkpoint overhaul](https://docs.ray.io/en/releases-2.2.0/rllib/rllib-saving-and-loading-algos-and-policies.html), [AlgorithmConfigs](https://docs.ray.io/en/releases-2.2.0/rllib/rllib-training.html#configuring-rllib-algorithms)
* Minor fixes: #29261, #29752

# Ray Core and Ray Clusters

## Ray Core

🎉 New Features:
* [Out-of-memory monitor](https://docs.ray.io/en/releases-2.2.0/ray-core/scheduling/ray-oom-prevention.html) is now Beta and is enabled by default. 

💫Enhancements:
* The [Ray Jobs API](https://docs.ray.io/en/releases-2.2.0/cluster/running-applications/job-submission/index.html#ray-jobs-api) has graduated from Beta to GA. This means Ray Jobs will maintain API backward compatibility. 
* Run Ray job entrypoint commands (“driver scripts”) on worker nodes by specifying `entrypoint_num_cpus`, `entrypoint_num_gpus`, or `entrypoint_resources`. (#28564, #28203)
* (Beta) OpenAPI spec for Ray Jobs REST API (#30417)
* Improved Ray health checking mechanism. The fix will reduce the frequency of GCS marking raylets fail mistakenly when it is overloaded. (#29346, #29442, #29389, #29924)

🔨 Fixes:
* Various fixes for hanging / deadlocking (#29491, #29763, #30371, #30425)
* Set OMP_NUM_THREADS to `num_cpus` required by task/actors by default (#30496)
* set worker non recyclable if gpu is envolved by default (#30061)

📖Documentation:
* General improvements of Ray Core docs, including design patterns and tasks.

## Ray Clusters

💫Enhancements:
* Stability improvements for Ray Autoscaler / KubeRay Operator integration. (#29933 , #30281, #30502)

## Dashboard

🎉 New Features:
* Additional improvements from the default metrics dashboard. We now have actor, placement group, and per component memory usage breakdown. [You can see details from the doc](https://docs.ray.io/en/releases-2.2.0/ray-observability/ray-metrics.html#system-metrics).
* New profiling feature using py-spy under the hood. You can click buttons to see stack trace or cpu flame graphs of your workers.
* Autoscaler and job events are available from the dashboard. You can also access the same data using `ray list cluster-events`. 

🔨 Fixes:
* Stability improvements from the dashboard
* Dashboard now works at large scale cluster! It is tested with 250 nodes and 10K+ actors (which matches the [Ray scalability envelope](https://github.com/ray-project/ray/tree/master/release/benchmarks)). 
  * Smarter api fetching logic. We now wait for the previous API to finish before sending a new API request when polling for new data.
  * Fix [agent memory leak](https://github.com/ray-project/ray/issues/29199) and high CPU usage.

💫Enhancements:
* General improvements to the progress bar. You can now see progress bars for each task name if you drill into the job details.
* More metadata is available in the jobs and actors tables.
* There is now a feedback button embedded into the dashboard. Please submit any bug reports or suggestions!

Many thanks to all those who contributed to this release!

@shrekris-anyscale, @rickyyx, @scottjlee, @shogohida, @liuyang-my, @matthewdeng, @wjrforcyber, @linusbiostat, @clarkzinzow, @justinvyu, @zygi, @christy, @amogkam, @cool-RR, @jiaodong, @EvgeniiTitov, @jjyao, @ilee300a, @jianoaix, @rkooo567, @mattip, @maxpumperla, @ericl, @cadedaniel, @bveeramani, @rueian, @stephanie-wang, @lcipolina, @bparaj, @JoonHong-Kim, @avnishn, @tomsunelite, @larrylian, @alanwguo, @VishDev12, @c21, @dmatrix, @xwjiang2010, @thomasdesr, @tiangolo, @sokratisvas, @heyitsmui, @scv119, @pcmoritz, @bhavika, @yzs981130, @andraxin, @Chong-Li, @clarng, @acxz, @ckw017, @krfricke, @kouroshHakha, @sijieamoy, @iycheng, @gjoliver, @peytondmurray, @xcharleslin, @DmitriGekhtman, @andreichalapco, @vitrioil, @architkulkarni, @simon-mo, @ArturNiederfahrenhorst, @sihanwang41, @pabloem, @sven1977, @avivhaber, @wuisawesome, @jovany-wang, @Yard1

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.2.0)

---

## ray-2.1.0: Ray-2.1.0
**Published:** 2022-11-08

# Release Highlights

* Ray AI Runtime (AIR)
  * Better support for Image-based workloads.
    * Ray Datasets `read_images()` API for loading data.
    * Numpy-based API for user-defined functions in Preprocessor.
  * Ability to read TFRecord input.
    * Ray Datasets `read_tfrecords()` API to read TFRecord files.
* Ray Serve: 
  * Add support for gRPC endpoint (alpha release). Instead of using an HTTP server, Ray Serve supports gRPC protocol and users can bring their own schema for their use case.
* RLlib:
  * Introduce decision transformer (DT) algorithm.
  * New hook for callbacks with `on_episode_created()`.
  * Learning rate schedule to SimpleQ and PG.
* Ray Core:
  * Ray [OOM prevention](https://docs.ray.io/en/master/ray-core/scheduling/ray-oom-prevention.html) (alpha release).
  * Support dynamic generators as task return values.
* Dashboard:
  * Time series metrics support.
  * Export configuration files can be used in Prometheus or Grafana instances.
  * New progress bar in job detail view.

# Ray Libraries

## Ray AIR

💫Enhancements:
* Improve readability of training failure output (#27946,  #28333, #29143)
* Auto-enable GPU for Predictors (#26549)
* Add ability to create TorchCheckpoint from state dict (#27970)
* Add ability to create TensorflowCheckpoint from saved model/h5 format (#28474)
* Add attribute to retrieve URI from Checkpoint (#28731)
* Add all allowable types to WandB Callback (#28888)

🔨 Fixes:
* Handle nested metrics properly as scoring attribute (#27715)
* Fix serializability of Checkpoints (#28387, #28895, #28935)

📖Documentation:
* Miscellaneous updates to documentation and examples (#28067, #28002, #28189, #28306, #28361, #28364, #28631, #28800)

🏗 Architecture refactoring:
* Deprecate Checkpoint.to_object_ref and Checkpoint.from_object_ref (#28318)
* Deprecate legacy train/tune functions in favor of Session (#28856)

## Ray Data Processing

🎉 New Features:
* Add read_images (#29177)
* Add read_tfrecords (#28430)
* Add NumPy batch format to Preprocessor and `BatchMapper` (#28418)
* Ragged tensor extension type (#27625)
* Add KBinsDiscretizer Preprocessor (#28389)

💫Enhancements:
* Simplify to_tf interface (#29028)
* Add metadata override and inference in `Dataset.to_dask()` (#28625)
* Prune unused columns before aggregate (#28556)
* Add Dataset.default_batch_format (#28434)
* Add partitioning parameter to read_ functions (#28413)
* Deprecate "native" batch format in favor of "default" (#28489)
* Support None partition field name (#28417)
* Re-enable Parquet sampling and add progress bar (#28021)
* Cap the number of stats kept in StatsActor and purge in FIFO order if the limit exceeded (#27964)
* Customized serializer for Arrow JSON ParseOptions in read_json (#27911)
* Optimize groupby/mapgroups performance (#27805)
* Improve size estimation of image folder data source (#27219)
* Use detached lifetime for stats actor (#25271)
* Pin _StatsActor to the driver node (#27765)
* Better error message for partition filtering if no file found (#27353)
* Make Concatenator deterministic (#27575)
* Change FeatureHasher input schema to expect token counts (#27523)
* Avoid unnecessary reads when truncating a dataset with `ds.limit()` (#27343)
* Hide tensor extension from UDFs (#27019)
* Add __repr__ to AIR classes (#27006)

🔨 Fixes:
* Add upper bound to pyarrow version check (#29674) (#29744)
* Fix map_groups to work with different output type (#29184)
* read_csv not filter out files by default (#29032)
* Check columns when adding rows to TableBlockBuilder (#29020)
* Fix the peak memory usage calculation (#28419)
* Change sampling to use same API as read Parquet (#28258)
* Fix column assignment in Concatenator for Pandas 1.2. (#27531)
* Doing partition filtering in reader constructor (#27156)
* Fix split ownership (#27149)

📖Documentation:
* Clarify dataset transformation. (#28482)
* Update map_batches documentation (#28435)
* Improve docstring and doctest for read_parquet (#28488)
* Activate dataset doctests (#28395)
* Document using a different separator for read_csv (#27850)
* Convert custom datetime column when reading a CSV file (#27854)
* Improve preprocessor documentation (#27215)
* Improve `limit()` and `take()` docstrings (#27367)
* Reorganize the tensor data support docs (#26952)
* Fix nyc_taxi_basic_processing notebook (#26983)

## Ray Train

🎉 New Features:
* Add FullyShardedDataParallel support to TorchTrainer (#28096)

💫Enhancements:
* Add rich notebook repr for DataParallelTrainer (#26335)
* Fast fail if training loop raises an error on any worker (#28314)
* Use torch.encode_data with HorovodTrainer when torch is imported (#28440)
* Automatically set NCCL_SOCKET_IFNAME to use ethernet (#28633)
* Don't add Trainer resources when running on Colab (#28822)
* Support large checkpoints and other arguments (#28826)

🔨 Fixes:
* Fix and improve HuggingFaceTrainer (#27875, #28154, #28170, #28308,  #28052)
* Maintain dtype info in LightGBMPredictor (#28673)
* Fix prepare_model (#29104)
* Fix `train.torch.get_device()` (#28659)

📖Documentation:
* Clarify LGBM/XGB Trainer documentation (#28122)
* Improve Hugging Face notebook example (#28121)
* Update Train API reference and docs (#28192)
* Mention FSDP in HuggingFaceTrainer docs (#28217)

🏗 Architecture refactoring:
* Improve Trainer modularity for extensibility (#28650)

## Ray Tune

🎉 New Features:
* Add `Tuner.get_results()` to retrieve results after restore (#29083)

💫Enhancements:
* Exclude files in sync_dir_between_nodes, exclude temporary checkpoints (#27174)
* Add rich notebook output for Tune progress updates (#26263)
* Add logdir to W&B run config (#28454)
* Improve readability for long column names in table output (#28764)
* Add functionality to recover from latest available checkpoint (#29099)
* Add retry logic for restoring trials (#29086)

🔨 Fixes:
* Re-enable progress metric detection (#28130)
* Add timeout to retry_fn to catch hanging syncs (#28155)
* Correct PB2’s beta_t parameter implementation (#28342)
* Ignore directory exists errors to tackle race conditions (#28401)
* Correctly overwrite files on restore (#28404)
* Disable pytorch-lightning multiprocessing per default (#28335)
* Raise error if scheduling an empty PlacementGroupFactory#28445
* Fix trial cleanup after x seconds, set default to 600 (#28449)
* Fix trial checkpoint syncing after recovery from other node (#28470)
* Catch empty hyperopt search space, raise better Tuner error message (#28503)
* Fix and optimize sample search algorithm quantization logic (#28187)
* Support tune.with_resources for class methods (#28596)
* Maintain consistent Trial/TrialRunner state when pausing and resuming trial with PBT (#28511)
* Raise better error for incompatible gcsfs version (#28772)
* Ensure that exploited in-memory checkpoint is used by trial with PBT (#28509)
* Fix Tune checkpoint tracking for minimizing metrics (#29145)

📖Documentation:
* Miscelleanous documentation fixes (#27117, #28131, #28210, #28400,  #28068, #28809)
* Add documentation around trial/experiment checkpoint (#28303)
* Add basic parallel execution guide for Tune (#28677)
* Add example PBT notebook (#28519)

🏗 Architecture refactoring:
* Store SyncConfig and CheckpointConfig in Experiment and Trial (#29019)

## Ray Serve

🎉 New Features:
* Added gRPC direct ingress support [alpha version] (#28175)
* Serve cli can provide kubernetes formatted output (#28918)
* Serve cli can provide user config output without default value (#28313)

💫Enhancements:
* Enrich more benchmarks 
* image objection with resnet50 mode with image preprocessing (#29096)
* gRPC vs HTTP inference performance (#28175)
* Add health check metrics to reflect the replica health status (#29154)

🔨 Fixes:
* Fix memory leak issues during inference (#29187)
* Fix unexpected http options omit warning when using serve cli to start the ray serve (#28257)
* Fix unexpected long poll exceptions (#28612)

📖Documentation:
* Add e2e fault tolerance instructions (#28721)
* Add Direct Ingress instructions (#29149)
* Bunch of doc improvements on “dev workflow”, “custom resources”,  “serve cli” etc (#29147, #28708, #28529, #28527)

## RLlib

🎉 New Features:
* Decision Transformer (DT) Algorithm added (#27890, #27889, #27872, #27829).
* Callbacks now have a new hook `on_episode_created()`. (#28600)
* Added learning rate schedule to SimpleQ and PG. (#28381)

💫Enhancements:
* Soft target network update is now supported by all off-policy algorithms (e.g DQN, DDPG, etc.) (#28135)
* Stop RLlib from "silently" selecting atari preprocessors. (#29011)
* Improved offline RL and off-policy evaluation performance (#28837, #28834, #28593, #28420, #28136, #28013, #27356, #27161, #27451).
* Escalated old deprecation warnings to errors (#28807, #28795, #28733, #28697).
* Others: #27619, #27087.

🔨 Fixes:
* Various bug fixes: #29077, #28811, #28637, #27785, #28703, #28422, #28405, #28358, #27540, #28325, #28357, #28334, #27090, #28133, #27981, #27980, #26666, #27390, #27791, #27741, #27424, #27544, #27459, #27572, #27255, #27304, #26629, #28166, #27864, #28938, #28845, #28588, #28202, #28201, #27806

📖Documentation:
* Connectors. (#27528)
* Training step API. (#27344)
* Others: #28299, #27460

## Ray Workflows

🔨 Fixes:
* Fixed the object loss due to driver exit (#29092)
* Change the name in step to task_id (#28151)

# Ray Core and Ray Clusters

## Ray Core

🎉 New Features:
* Ray [OOM prevention](https://docs.ray.io/en/master/ray-core/scheduling/ray-oom-prevention.html) feature alpha release! If your Ray jobs suffer from OOM issues, please give it a try. 
* Support dynamic generators as task return values. (#29082 #28864 #28291)

💫Enhancements:
* Fix spread scheduling imbalance issues (#28804 #28551 #28551)
* Widening range of grpcio versions allowed (#28623)
* Support encrypted redis connection. (#29109)
* Upgrade redis from 6.x to 7.0.5. (#28936)
* Batch ScheduleAndDispatchTasks calls (#28740)

🔨 Fixes:
* More robust spilled object deletion (#29014)
* Fix the initialization/destruction order between reference_counter_ and node change subscription (#29108)
* Suppress the logging error when python exits and actor not deleted (#27300)
* Mark `run_function_on_all_workers` as deprecated until we get rid of this (#29062)
* Remove unused args for default_worker.py (#28177)
* Don't include script directory in sys.path if it's started via python -m (#28140)
* Handling edge cases of max_cpu_fraction argument (#27035)
* Fix out-of-band deserialization of actor handle (#27700)
* Allow reuse of cluster address if Ray is not running (#27666)
* Fix a uncaught exception upon deallocation for actors (#27637)
* Support placement_group=None in PlacementGroupSchedulingStrategy (#27370)

📖Documentation:
* [Ray 2.0 white paper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview) is published.
* Revamp ray core docs  (#29124 #29046 #28953 #28840 #28784 #28644 #28345 #28113  #27323 #27303)
* Fix cluster docs (#28056 #27062)
* CLI Reference Documentation Revamp (#27862)

## Ray Clusters

💫Enhancements:
* Distinguish Kubernetes deployment stacks (#28490)

📖Documentation:
* State intent to remove legacy Ray Operator (#29178)
* Improve KubeRay migration notes (#28672)
* Add FAQ for cluster multi-tenancy support (#29279)

## Dashboard

🎉 New Features:
* Time series metrics are now built into the dashboard
* Ray now exports some default configuration files which can be used for your Prometheus or Grafana instances. This includes default metrics which show common information important to your Ray application.
* New progress bar is shown in the job detail view. You can see how far along your ray job is.

🔨 Fixes:
* Fix to prometheus exporter producing a slightly incorrect format.
* Fix several performance issues and memory leaks

📖Documentation:
* Added additional documentation on the new time series and the metrics page

Many thanks to all those who contributed to this release!

@sihanwang41, @simon-mo, @avnishn, @MyeongKim, @markrogersjr, @christy, @xwjiang2010, @kouroshHakha, @zoltan-fedor, @wumuzi520, @alanwguo, @Yard1, @liuyang-my, @charlesjsun, @DevJake, @matteobettini, @jonathan-conder-sm, @mgerstgrasser, @guidj, @JiahaoYao, @Zyiqin-Miranda, @jvanheugten, @aallahyar, @SongGuyang, @clarng, @architkulkarni, @Rohan138, @heyitsmui, @mattip, @ArturNiederfahrenhorst, @maxpumperla, @vale981, @krfricke, @DmitriGekhtman, @amogkam, @richardliaw, @maldil, @zcin, @jianoaix, @cool-RR, @kira-lin, @gramhagen, @c21, @jiaodong, @sijieamoy, @tupui, @ericl, @anabranch, @se4ml, @suquark, @dmatrix, @jjyao, @clarkzinzow, @smorad, @rkooo567, @jovany-wang, @edoakes, @XiaodongLv, @klieret, @rozsasarpi, @scottsun94, @ijrsvt, @bveeramani, @chengscott, @jbedorf, @kevin85421, @nikitavemuri, @sven1977, @acxz, @stephanie-wang, @PaulFenton, @WangTaoTheTonic, @cadedaniel, @nthai, @wuisawesome, @rickyyx, @artemisart, @peytondmurray, @pingsutw, @olipinski, @davidxia, @stestagg, @yaxife, @scv119, @mwtian, @yuanchi2807, @ntlm1686, @shrekris-anyscale, @cassidylaidlaw, @gjoliver, @ckw017, @hakeemta, @ilee300a, @avivhaber, @matthewdeng, @afarid, @pcmoritz, @Chong-Li, @Catch-Bull, @justinvyu, @iycheng

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.1.0)

---

## ray-2.0.1: Ray-2.0.1
**Published:** 2022-10-22

The Ray 2.0.1 patch release contains dependency upgrades and fixes for multiple components:

- Upgrade grpcio version to 1.32 ([#28025](https://github.com/ray-project/ray/pull/28025))
- Upgrade redis version to 7.0.5 ([#28936](https://github.com/ray-project/ray/pull/28936))
- Fix segfault when using runtime environments ([#28409](https://github.com/ray-project/ray/pull/28409)) 
- Increase RPC timeout for dashboard ([#28330](https://github.com/ray-project/ray/pull/28330))
- Set correct path when using `python -m` ([#28140](https://github.com/ray-project/ray/pull/28140))
- [Autoscaler] Fix autoscaling for 0 CPU head node ([#26813](https://github.com/ray-project/ray/pull/26813))
- [Serve] Allow code in private remote Git URIs to be imported ([#28250](https://github.com/ray-project/ray/pull/28250))
- [Serve] Allow `host` and `port` in Serve config ([#27026](https://github.com/ray-project/ray/pull/27026)) 
- [RLlib] Evaluation supports asynchronous rollout (single slow eval worker will not block the overall evaluation progress). ([#27390](https://github.com/ray-project/ray/pull/27390))
- [Tune] Fix hang during checkpoint synchronization ([#28155](https://github.com/ray-project/ray/pull/28155))
- [Tune] Fix trial restoration from different IP ([#28470](https://github.com/ray-project/ray/pull/28470)) 
- [Tune] Fix custom synchronizer serialization ([#28699](https://github.com/ray-project/ray/pull/28699)) 
- [Workflows] Replace deprecated `name` option with `task_id` ([#28151](https://github.com/ray-project/ray/pull/28151))

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.0.1)

---

## ray-2.0.0: Ray-2.0.0
**Published:** 2022-08-23

# Release Highlights

Ray 2.0 is an exciting release with enhancements to all libraries in the Ray ecosystem. With this major release, we take strides towards our goal of making distributed computing scalable, unified, and open.

Towards these goals, Ray 2.0 features new capabilities for unifying the machine learning (ML) ecosystem, improving Ray's production support, and making it easier than ever for ML practitioners to use Ray's libraries.

**Highlights:**



* [Ray AIR](https://docs.ray.io/en/releases-2.0.0/ray-air/getting-started.html), a scalable and unified toolkit for ML applications, is now in Beta. 
* ​​Ray now supports [natively shuffling 100TB or more of data](https://docs.ray.io/en/releases-2.0.0/data/performance-tips.html#enabling-push-based-shuffle) with the Ray Datasets library.
* [KubeRay](https://docs.ray.io/en/releases-2.0.0/cluster/kubernetes/index.html), a toolkit for running Ray on Kubernetes, is now in Beta. This replaces the legacy Python-based Ray operator.
* [Ray Serve’s Deployment Graph API ](https://docs.ray.io/en/releases-2.0.0/serve/model_composition.html#serve-model-composition)is a new and easier way to build, test, and deploy an inference graph of deployments. This is released as Beta in 2.0.

A migration guide for all the different libraries can be found here: [Ray 2.0 Migration Guide](https://docs.google.com/document/d/12ODPbhEzeyDRUt8ehHDiKCFoxksPWJOUNEicGhNxtRg/edit#).


# Ray Libraries


## Ray AIR

Ray AIR is now in beta. Ray AIR builds upon Ray’s libraries to enable end-to-end machine learning workflows and applications on Ray. You can install all dependencies needed for Ray AIR via `pip install -u "ray[air]"`.

🎉 **New Features:**


* Predictors:
    * BatchPredictors now have support for scalable inference on GPUs.
    * All Predictors can now be constructed from pre-trained models, allowing you to easily scale batch inference with trained models from common ML frameworks.
    * ray.ml.predictors has been moved to the Ray Train namespace (ray.train).
* Preprocessing: New preprocessors and API changes on Ray Datasets now make feature processing easier to do on AIR. See the Ray Data release notes for more details.
* New features for Datasets/Train/Tune/Serve can be found in the corresponding library release notes for more details.

💫 **Enhancements:**


* Major package refactoring is included in this release.
    * ray.ml is renamed to ray.air.
    * ray.ml.preprocessors have been moved to ray.data.
        * train_test_split is now a new method of ray.data.Dataset (#27065)
    * ray.ml.trainers have been moved to ray.train (#25570)
    * ray.ml.predictors has been moved to ray.train.
    * ray.ml.config has been moved to ray.air.config (#25712). 
    * Checkpoints are now framework-specific -- meaning that each Trainer generates its own Framework-specific Checkpoint class. See Ray Train for more details.
    * ModelWrappers have been renamed to PredictorDeployments.
* API stability annotations have been added (#25485)
* Train/Tune now have the same reporting and checkpointing API -- see the Train notes for more details (#26303)
* ScalingConfigs are now Dataclasses not Dict types 
* Many AIR examples, benchmarks, and documentation pages were added in this release. The Ray AIR documentation will cover breadth of usage (end to end workflows across different libraries) while library-specific documentation will cover depth (specific features of a specific library).

🔨 **Fixes:**


* Many documentation examples were previously untested. This release fixes those examples and adds them to the CI.
* Predictors:
    * Torch/Tensorflow Predictors have correctness fixes (#25199, #25190, #25138, #25136)
    * Update `KerasCallback` to work with `TensorflowPredictor` (#26089)
    * Add streaming BatchPredictor support (#25693)
    * Add `predict_pandas` implementation (#25534)
    * Add `_predict_arrow` interface for Predictor (#25579)
    * Allow creating Predictor directly from a UDF (#26603)
    * Execute GPU inference in a separate stage in BatchPredictor (#26616, #27232, #27398)
    * Accessors for preprocessor in Predictor class (#26600)
    * [AIR] Predictor `call_model` API for unsupported output types (#26845)


## Ray Data Processing

🎉 **New Features:**


* Add ImageFolderDatasource (#24641)
* Add the NumPy batch format for batch mapping and batch consumption (#24870)
* Add iter_torch_batches() and iter_tf_batches() APIs (#26689)
* Add local shuffling API to iterators (#26094)
* Add drop_columns() API (#26200)
* Add randomize_block_order() API (#25568)
* Add random_sample() API (#24492)
* Add support for len(Dataset) (#25152)
* Add UDF passthrough args to map_batches() (#25613)
* Add Concatenator preprocessor (#26526)
* Change range_arrow() API to range_table() (#24704)

💫 **Enhancements:**


* Autodetect dataset parallelism based on available resources and data size (#25883)
* Use polars for sorting (#25454)
* Support tensor columns in to_tf() and to_torch() (#24752)
* Add explicit resource allocation option via a top-level scheduling strategy (#24438)
* Spread actor pool actors evenly across the cluster by default (#25705)
* Add ray_remote_args to read_text() (#23764)
* Add max_epoch argument to iter_epochs() (#25263)
* Add Pandas-native groupby and sorting (#26313)
* Support push-based shuffle in groupby operations (#25910)
* More aggressive memory releasing for Dataset and DatasetPipeline (#25461, #25820, #26902, #26650)
* Automatically cast tensor columns on Pandas UDF outputs (#26924)
* Better error messages when reading from S3 (#26619, #26669, #26789)
* Make dataset splitting more efficient and stable (#26641, #26768, #26778)
* Use sampling to estimate in-memory data size for Parquet data source (#26868)
* De-experimentalized lazy execution mode (#26934)

🔨 **Fixes:**


* Fix pipeline pre-repeat caching (#25265)
* Fix stats construction for from_*() APIs (#25601)
* Fixes label tensor squeezing in to_tf() (#25553)
* Fix stage fusion between equivalent resource args (fixes BatchPredictor) (#25706)
* Fix tensor extension string formatting (repr) (#25768)
* Workaround for unserializable Arrow JSON ReadOptions (#25821)
* Make ActorPoolStrategy kill pool of actors if exception is raised (#25803)
* Fix max number of actors for default actor pool strategy (#26266)
* Fix byte size calculation for non-trivial tensors (#25264)


## Ray Train

Ray Train has received a major expansion of scope with Ray 2.0. 

In particular, the Ray Train module now contains:

1. Trainers 
2. Predictors
3. Checkpoints

for common different ML frameworks including Pytorch, Tensorflow, XGBoost, LightGBM, HuggingFace, and Scikit-Learn. These API help provide end-to-end usage of Ray libraries in Ray AIR workflows.

🎉 **New Features:**


* The Trainer API is now deprecated for the new Ray AIR Trainers API. Trainers for Pytorch, Tensorflow, Horovod, XGBoost, and LightGBM are now in Beta.  (#25570)
* ML framework-specific Predictors have been moved into the `ray.train` namespace. This provides streamlined API for offline and online inference of Pytorch, Tensorflow, XGBoost models and more. (#25769 #26215, #26251, #26451, #26531, #26600, #26603, #26616, #26845)
* ML framework-specific checkpoints are introduced. Checkpoints are consumed by Predictors to load model weights and information. (#26777, #25940, #26532, #26534)

💫 **Enhancements:** 


* Train and Tune now use the same reporting and checkpointing API (#24772, #25558) 
* Add tunable ScalingConfig dataclass (#25712) 
* Randomize block order by default to avoid hotspots (#25870)
* Improve checkpoint configurability and extend results (#25943)
* Improve prepare_data_loader to support multiple batch data types (#26386)
* Discard returns of train loops in Trainers (#26448) 
* Clean up logs, reprs, warning s(#26259, #26906, #26988, #27228, #27519)

📖 **Documentation:**


* Update documentation to use new Train API (#25735) 
* Update documentation to use session API (#26051, #26303) 
* Add Trainer user guide and update Trainer docs (#27570, #27644, #27685)
* Add Predictor documentation (#25833)
* Replace to_torch with iter_torch_batches (#27656) 
* Replace to_tf with iter_tf_batches (#27768) 
* Minor doc fixes (#25773, #27955)

🏗 **Architecture refactoring:**


* Clean up ray.train package (#25566)
* Mark Trainer interfaces as Deprecated (#25573)

🔨 **Fixes:**


* An issue with GPU ID detection and assignment was fixed. (#26493)
* Fix AMP for models with a custom `__getstate__` method (#25335)
* Fix transformers example for multi-gpu (#24832)
* Fix ScalingConfig key validation (#25549)
* Fix ResourceChangingScheduler integration (#26307) 
* Fix auto_transfer cuda device (#26819) 
* Fix BatchPredictor.predict_pipelined not working with GPU stage (#27398)
* Remove rllib dependency from tensorflow_predictor (#27688)


## Ray Tune

🎉 **New Features:**


* The Tuner API is the new way of running Ray Tune experiments. (#26987, #26987, #26961, #26931, #26884, #26930)
* Ray Tune and Ray Train now have the same API for reporting (#25558)
* Introduce tune.with_resources() to specify function trainable resources (#26830)
* Add Tune benchmark for AIR (#26763, #26564)
* Allow Tuner().restore() from cloud URIs (#26963)
* Add top-level imports for Tuner, TuneConfig, move CheckpointConfig (#26882)
* Add resume experiment options to Tuner.restore() (#26826)
* Add checkpoint_frequency/checkpoint_at_end arguments to CheckpointConfig (#26661)
* Add more config arguments to Tuner (#26656)
* Better error message for Tune nested tasks / actors (#25241)
* Allow iterators in tune.grid_search (#25220)
* Add `get_dataframe()` method to result grid, fix config flattening (#24686)

💫 **Enhancements:**


* Expose number of errored/terminated trials in ResultGrid (#26655)
* remove fully_executed from Tune. (#25750)
* Exclude in remote storage upload (#25544)
* Add `TempFileLock` (#25408)
* Add annotations/set scope for Tune classes (#25077)

📖 **Documentation:**


* Improve Tune + Datasets documentation (#25389)
* Tune examples better navigation, minor fixes (#24733)

🏗 **Architecture refactoring:**


* Consolidate checkpoint manager 3: Ray Tune (#24430)
* Clean up ray.tune scope (remove stale objects in __all__) (#26829)

🔨 **Fixes:**


* Fix k8s release test + node-to-node syncing (#27365)
* Fix Tune custom syncer example (#27253)
* Fix tune_cloud_aws_durable_upload_rllib_* release tests (#27180)
* Fix test_tune (#26721)
* Larger head node for tune_scalability_network_overhead weekly test (#26742)
* Fix tune-sklearn notebook example (#26470)
* Fix reference to `dataset_tune` (#25402)
* Fix Tune-Pytorch-CIFAR notebook example (#26474)
* Fix documentation testing (#26409)
* Fix `set_tune_experiment` (#26298)
* Fix GRPC resource exhausted test for tune trainables (#24467)


## Ray Serve

🎉 **New Features:**


* We are excited to introduce you to the 2.0 API centered around multi-model composition API, operation API, and production stability. (#26310,#26507,#26217,#25932,#26374,#26901,#27058,#24549,#24616,#27479,#27576,#27433,#24306,#25651,#26682,#26521,#27194,#27206,#26804,#25575,#26574)
    * Deployment Graph API is the new API for model composition. It provides a declarative layer on top of the 1.x deployment API to help you author performant inference pipeline easily. (#27417,#27420,#24754,#24435,#24630,#26573,#27349,#24404,#25424,#24418,#27815,#27844,#25453,#24629)
    * We introduced a new [K8s native way to deploy Ray Serve](https://ray-project.github.io/kuberay/guidance/rayservice/). Along with a brand new REST API to perform deployment, update, and configure. (#25935,#27063,#24814,#26093,#25213,#26588,#25073,#27000,#27444,#26578,#26652,#25610,#25502,#26096,#24265,#26177,#25861,#25691,#24839,#27498,#27561,#25862,#26347)
    * Serve can now survive Ray GCS failure. This used to be a single-point-of-failure in Ray Serve's architecture. Now, when the GCS goes down, Serve can continue to Serve traffic. We recommend you to [try out this feature](https://ray-project.github.io/kuberay/guidance/gcs-ft/) and give us feedback! (#25633,#26107,#27608,#27763,#27771,#25478,#25637,#27526,#27674,#26753,#26797,#24560,#26685,#26734,#25987,#25091,#24934)
* Autoscaling has been promoted to stable. Additionally, we added a scale to zero support. (#25770,#25733,#24892,#26393)
* The documentation has been revamped. Check them at rayserve.org (#24414,#26211,#25786,#25936,#26029,#25830,#24760,#24871,#25243,#25390,#25646,#24657,#24713,#25270,#25808,#24693,#24736,#24524,#24690,#25494)

💫 **Enhancements:**


* Serve natively supports deploying predictor and checkpoints from Ray AI Runtime (#26026,#25003,#25537,#25609,#25962,#26494,#25688,#24512,#24417)
* Serve now supports scaling Gradio application (#27560)
* Java Client API, marking the complete alpha release Java API (#22726)
* Improved out-of-box performance by using uvicorn with uvloop (#25027)


## RLlib

🎉 **New Features:**


* In 2.0, RLlib is introducing an object-oriented configuration API instead of using a python dict for algorithm configuration (#24332, #24374, #24375, #24376, #24433, #24576, #24650, #24577, #24339, #24687, #24775, #24584, #24583, #24853, #25028, #25059, #25065, #25066, #25067, #25256, #25255, #25278, #25279)
* RLlib is introducing a Connectors API (alpha). Connectors are a new component that handles transformations on inputs and outputs of a given RL policy.  (#25311, #25007, #25923, #25922, #25954, #26253, #26510, #26645, #26836, #26803, #26998, #27016)
* New improvements to off-policy estimators, including a new Doubly-Robust Off-Policy Estimator implementation (#24384, #25107, #25056, #25899, #25911, #26279, #26893)
* CRR Algorithm (#25459, #25667, #25905, #26142, #26304, #26770, #27161)
* Feature importance evaluation for offline RL (#26412)
* RE3 exploration algorithm TF2 framework support (#25221)
* Unified replay Buffer API (#24212, #24156, #24473, #24506, #24866, #24683, #25841, #25560, #26428)

💫 **Enhancements:**


* Improvements to RolloutWorker / Env fault tolerance (#24967, #26134, #26276, #26809)
* Upgrade gym to 0.23 (#24171), Bump gym dep to 0.24 (#26190)
* Agents has been renamed to Algorithms (#24511, #24516, #24739, #24797, #24841, #24896, #25014, #24579, #25314, #25346, #25366, #25539, #25869)
* Execution Plan API is now deprecated. Training step function API is the new way of specifying RLlib algorithms (#23454, #24488, #2450, #24212, #24165, #24545, #24507, #25076, #25624, #25924, #25856, #25851, #27344, #24423)
* Policy V2 subclassing implementation migration (#24742, #24746, #24914, #25117, #25203, #25078, #25254, #25384, #25585, #25871, #25956, #26054)
* Allow passing **kwargs to action distribution. (#24692)
* Deprecation: Replace remaining evaluation_num_episodes with `evaluation_duration`. (#26000)

🔨 **Fixes:**


* Multi-GPU learner thread key error in MA-scenarios (#24382)
* Add release learning tests for SlateQ (#24429)
* APEX-DQN replay buffer config validation fix. (#24588)
* Automatic sequencing in function timeslice_along_seq_lens_with_overlap (#24561)
* Policy Server/Client metrics reporting fix (#24783)
* Re-establish dashboard performance tests. (#24728)
* Bandit tf2 fix (+ add tf2 to test cases). (#24908)
* Fix estimated buffer size in replay buffers. (#24848)
* Fix RNNSAC example failing on CI + fixes for recurrent models for other Q Learning Algos. (#24923)
* Curiosity bug fix. (#24880)
* Auto-infer different agents' spaces in multi-agent env. (#24649)
* Fix the bug “WorkerSet.stop() will raise error if `self._local_worker` is None (e.g. in evaluation worker sets)”. (#25332)
* Fix Policy global timesteps being off by init sample batch size. (#25349)
* Disambiguate timestep fragment storage unit in replay buffers. (#25242)
* Fix the bug where on GPU, sample_batch.to_device() only converts the device and does not convert float64 to float32. (#25460)
* Fix faulty usage of get_filter_config in ComplexInputNextwork` (#25493)`
* Custom resources per worker should get added to default_resource_request (#24463)
* Better default values for training_intensity and `target_network_update_freq` for R2D2. (#25510)
* Fix multi agent environment checks for observations that contain only some agents' obs each step. (#25506)
* Fixes PyTorch grad clipping logic and adds grad clipping to QMIX. (#25584)
* Discussion 6432: Automatic train_batch_size calculation fix. (#25621)
* Added meaningful error for multi-agent failure of SampleCollector in case no agent steps in episode. (#25596)
* Replace torch.range with torch.arange. (#25640)\
* Fix the bug where there is no gradient clipping in QMix. (#25656[)](https://github.com/ray-project/ray/commit/c3645928caf8495a2849e21f1bf0e131409d9f99)
* Fix sample batch concatination. (#25572)
*  Fix action_sampler_fn call in TorchPolicyV2 (obs_batch instead of `input_dict` arg). (#25877)
* Fixes logging of all of RLlib's Algorithm names as warning messages. (#25840)
* IMPALA/APPO multi-agent mix-in-buffer fixes (plus MA learningt ests). (#25848)
* Move offline input into replay buffer using rollout ops in CQL. (#25629)
* Include SampleBatch.T column in all collected batches. (#25926)
* Add timeout to filter synchronization. (#25959)
* SimpleQ PyTorch Multi GPU fix (#26109)
* IMPALA and APPO metrics fixes; remove deprecated `async_parallel_requests` utility. (#26117)
* Added 'episode.hist_data' to the 'atari_metrics' to nsure that custom metrics of the user are kept in postprocessing when using Atari environments. (#25292)
* Make the dataset and json readers batchable (#26055)
* Fix Issue 25696: Output writers not working w/ multiple workers. (#25722)
* Fix all the erroneous on_trainer_init warning. (#26433)
* In env check, step only expected agents. (#26425)
* Make DQN update_target use only trainable variables. (#25226)
* Fix FQE Policy call (#26671)
* Make queue placement ops blocking (#26581)
* Fix memory leak in APEX_DQN (#26691)
* Fix MultiDiscrete not being one-hotted correctly (#26558)
* Make IOContext optional for DatasetReader (#26694)
* Make sure we step() after adding init_obs. (#26827)
* Fix ModelCatalog for nested complex inputs (#25620)
* Use compress observations where replay buffers and image obs are used in tuned examples (#26735)
* Fix SampleBatch.split_by_episode to use dones if episode id is not available (#26492)
* Fix torch None conversion in `torch_utils.py::convert_to_torch_tensor`. (#26863)
* Unify gnorm mixin for tf and torch policies. (#26102)


## Ray Workflows

🎉 **New Features:**


* Support ray client (#26702)
* Http event is supported (#26010)
* Support retry_exceptions (#26913)
* Support queuing in workflow (#24697)
* Make status indexed (#24767)

🔨 **Fixes:**


* Push logs to drivers correctly (#24490)
* Make resume no side effect (#26918)
* Make the max_retries aligned with ray (#26350)

🏗 **Architecture refactoring:**


* Rewrite workflow execution engine (#25618)
* Simplify the resume flow (#24594)
* Deprecate step and use bind (#26232)
* Deprecate virtual actor (#25394)
* Refactor the exception processing (#26398)


# Ray Core and Ray Clusters


## Ray Core

🎉 **New Features:**


* Ray State API is now at alpha. You can access the live information of tasks, actors, objects, placement groups, and etc. through Ray CLI (summary / list / get) and Python SDK. See the [Ray State API documentation](https://docs.ray.io/en/master/ray-observability/state/state-api.html) for more information.
* Support generators for tasks with multiple return values (#25247)
* Support GCS Fault tolerance.(#24764, #24813, #24887, #25131, #25126, #24747, #25789, #25975, #25994, #26405, #26421, #26919)

💫 **Enhancements:**


* Allow failing new tasks immediately while the actor is restarting (#22818)
* Add more accurate worker exit (#24468)
* Allow user to override global default for max_retries (#25189)
* Export additional metrics for workers and Raylet memory (#25418)
* Push message to driver when a Raylet dies (#25516)
* Out of Disk prevention (#25370)
* ray.init defaults to an existing Ray instance if there is one (#26678)
* Reconstruct manually freed objects (#27567)

🔨 **Fixes:**



* Fix a task cancel hanging bug (#24369)
* Adjust worker OOM scores to prioritize the raylet during memory pressure (#24623)
* Fix pull manager deadlock due to object reconstruction (#24791)
* Fix bugs in data locality aware scheduling (#25092)
* Fix node affinity strategy when resource is empty (#25344)
* Fix object transfer resend protocol (#26349)

🏗 **Architecture refactoring:**



* Raylet and GCS schedulers share the same code (#23829)
* Remove multiple core workers in one process (#24147, #25159)


## Ray Clusters

🎉 **New Features:**


* The KubeRay operator is now the preferred tool to run Ray on Kubernetes.
    * Ray Autoscaler + KubeRay operator integration is now beta.

💫 **Enhancements:**


* Check out the [newly revamped docs](https://docs.ray.io/en/releases-2.0.0/cluster/getting-started.html)!

🔨 **Fixes:**



* Previously deprecated fields, `head_node`, `worker_nodes`, `head_node_type`, `default_worker_node_type`, `autoscaling_mode`, `target_utilization_fraction` are removed. Check out the [migration guide ](https://docs.google.com/document/d/1Rz-UGz-RHK6iKSX3xt2VioQnX-kC20d5n8pbRN7KGME/edit#)to learn how to migrate to the new versions. 


## Ray Client

🎉 **New Features:**


* Support for configuring request metadata for client gRPC (#24946)

💫 **Enhancements:**


* Remove 2 GiB size limit on remote function arguments (#24555)

🔨 **Fixes:**


* Fix excessive memory usage when submitting large remote arguments (#24477)


## Dashboard

🎉 **New Features:**



* The new dashboard UI is now to default dashboard. Please leave any feedback about the dashboard on Github Issues or Discourse! You can still go to the legacy dashboard UI by clicking “Back to legacy dashboard”.
* New Dashboard UI now shows all ray jobs. This includes jobs submitted via the job submission API and jobs launched from python scripts via ray.init().
* New Dashboard UI now shows worker nodes in the main node tab
* New Dashboard UI now shows more information in the actors tab

Breaking changes:



* The job submission list_jobs API endpoint, CLI command, and SDK function now returns a list of jobs instead of a dictionary from id to job.
* The Tune tab is no longer in the new dashboard UI. It is still available in the legacy dashboard UI but will be removed.
* The memory tab is no longer in the new dashboard UI. It is still available in the legacy dashboard UI but will be removed.

🔨 **Fixes:**


* We reduced the memory usage of the dashboard. We are no longer caching logs and we cache a maximum of 1000 actors. As a result of this change, node level logs can no longer be accessed in the legacy dashboard.
* Jobs status error message now properly truncates logs to 10 lines. We also added a max characters of 20000 to avoid passing too much data. 

Many thanks to all those who contributed to this release!

@ujvl, @xwjiang2010, @EricCousineau-TRI, @ijrsvt, @waleedkadous, @captain-pool, @olipinski, @danielwen002, @amogkam, @bveeramani, @kouroshHakha, @jjyao, @larrylian, @goswamig, @hanming-lu, @edoakes, @nikitavemuri, @enori, @grechaw, @truelegion47, @alanwguo, @sychen52, @ArturNiederfahrenhorst, @pcmoritz, @mwtian, @vakker, @c21, @rberenguel, @mattip, @robertnishihara, @cool-RR, @iamhatesz, @ofey404, @raulchen, @nmatare, @peterghaddad, @n30111, @fkaleo, @Riatre, @zhe-thoughts, @lchu-ibm, @YoelShoshan, @Catch-Bull, @matthewdeng, @VishDev12, @valtab, @maxpumperla, @tomsunelite, @fwitter, @liuyang-my, @peytondmurray, @clarkzinzow, @VeronikaPolakova, @sven1977, @stephanie-wang, @emjames, @Nintorac, @suquark, @javi-redondo, @xiurobert, @smorad, @brucez-anyscale, @pdames, @jjyyxx, @dmatrix, @nakamasato, @richardliaw, @juliusfrost, @anabranch, @christy, @Rohan138, @cadedaniel, @simon-mo, @mavroudisv, @guidj, @rkooo567, @orcahmlee, @lixin-wei, @neigh80, @yuduber, @JiahaoYao, @simonsays1980, @gjoliver, @jimthompson5802, @lucasalavapena, @zcin, @clarng, @jbn, @DmitriGekhtman, @timgates42, @charlesjsun, @Yard1, @mgelbart, @wumuzi520, @sihanwang41, @ghost, @jovany-wang, @siavash119, @yuanchi2807, @tupui, @jianoaix, @sumanthratna, @code-review-doctor, @Chong-Li, @FedericoGarza, @ckw017, @Makan-Ar, @kfstorm, @flanaman, @WangTaoTheTonic, @franklsf95, @scv119, @kvaithin, @wuisawesome, @jiaodong, @mgerstgrasser, @tiangolo, @architkulkarni, @MyeongKim, @ericl, @SongGuyang, @avnishn, @chengscott, @shrekris-anyscale, @Alyetama, @iycheng, @rickyyx, @krfricke, @sijieamoy, @kimikuri, @czgdp1807, @michalsustr

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-2.0.0)

---

## ray-1.13.0: Ray-1.13.0
**Published:** 2022-06-09

## Highlights:

- Python 3.10 support is now in alpha.
- Ray [usage stats collection](https://docs.ray.io/en/master/cluster/usage-stats.html) is now on by default  (guarded by an opt-out prompt).
- Ray Tune can now synchronize Trial data from worker nodes via the object store (without rsync!)
- Ray Workflow comes with a new API and is integrated with Ray DAG.

## Ray Autoscaler
💫Enhancements: 

- CI tests for KubeRay autoscaler integration (#23365, #23383, #24195) 
- Stability enhancements for KubeRay autoscaler integration (#23428)

🔨 Fixes: 

- Improved GPU support in KubeRay autoscaler integration (#23383)
- Resources scheduled with the node affinity strategy are not reported to the autoscaler (#24250)

## Ray Client
💫Enhancements:

- Add option to configure ray.get with >2 sec timeout (#22165)
- Return `None` from internal KV for non-existent keys (#24058)

🔨 Fixes:

- Fix deadlock by switching to `SimpleQueue` on Python 3.7 and newer in async `dataclient` (#23995)

## Ray Core
🎉 New Features:

- Ray [usage stats collection](https://docs.ray.io/en/master/cluster/usage-stats.html) is now on by default  (guarded by an opt-out prompt)
- Alpha support for python 3.10 (on Linux and Mac)
- Node affinity scheduling strategy (#23381)
- Add metrics for disk and network I/O (#23546)
- Improve exponential backoff when connecting to the redis (#24150)
- Add the ability to inject a setup hook for customization of runtime_env on init (#24036)
- Add a utility to check GCS / Ray cluster health (#23382)

🔨 Fixes:

- Fixed internal storage S3 bugs (#24167)
- Ensure "get_if_exists" takes effect in the decorator. (#24287)
- Reduce memory usage for Pubsub channels that do not require total memory cap (#23985)
- Add memory buffer limit in publisher for each subscribed entity (#23707)
- Use gRPC instead of socket for GCS client health check (#23939)
- Trim size of Reference struct (#23853)
- Enable debugging into pickle backend (#23854)

🏗 Architecture refactoring:

- Gcs storage interfaces unification (#24211)
- Cleanup pickle5 version check (#23885)
- Simplify options handling (#23882)
- Moved function and actor importer away from pubsub (#24132)
- Replace the legacy ResourceSet & SchedulingResources at Raylet (#23173)
- Unification of AddSpilledUrl and UpdateObjectLocationBatch RPCs (#23872)
- Save task spec in separate table  (#22650)


## Ray Datasets
🎉 New Features:

- Performance improvement: the aggregation computation is vectorized (#23478)
- Performance improvement: bulk parquet file reading is optimized with the fast metadata provider (#23179)
- Performance improvement: more efficient move semantics for Datasets block processing (#24127)
- Supports Datasets lineage serialization (aka out-of-band serialization) (#23821, #23931, #23932)
- Supports native Tensor views in map processing for pure-tensor datasets (#24812)
- Implemented push-based shuffle (#24281)

🔨 Fixes:

- Documentation improvement: Getting Started page (#24860)
- Documentation improvement: FAQ (#24932)
- Documentation improvement: End to end examples (#24874)
- Documentation improvement: Feature guide - Creating Datasets (#24831)
- Documentation improvement: Feature guide - Saving Datasets (#24987)
- Documentation improvement: Feature guide - Transforming Datasets (#25033)
- Documentation improvement: Datasets APIs docstrings (#24949)
- Performance: fixed block prefetching (#23952) 
- Fixed zip() for Pandas dataset (#23532) 

🏗 Architecture refactoring:

- Refactored LazyBlockList (#23624)
- Added path-partitioning support for all content types (#23624)
- Added fast metadata provider and refactored Parquet datasource (#24094)

## RLlib
🎉 New Features:

- Replay buffer API: First algorithms are using the new replay buffer API, allowing users to define and configure their own custom buffers or use RLlib’s built-in ones: SimpleQ, DQN (#24164, #22842, #23523, #23586)

🏗 Architecture refactoring:

- More algorithms moved into the training iteration function API (no longer using execution plans). Users can now more easily read, develop, and debug RLlib’s algorithms: A2C, APEX-DQN, CQL, DD-PPO, DQN, MARWIL + BC, PPO, QMIX , SAC, SimpleQ, SlateQ, Trainers defined in examples folder. (#22937, #23420, #23673, #24164, #24151, #23735, #24157, #23798, #23906, #24118, #22842, #24166, #23712). This will be fully completed and documented with Ray 2.0.
- Make RolloutWorkers (optionally) recoverable after failure via the new `recreate_failed_workers=True` config flag. (#23739)
- POC for new TrainerConfig objects (instead of python config dicts): PPOConfig (for PPOTrainer) and PGConfig (for PGTrainer). (#24295, #23491)
- Hard-deprecate `build_trainer()` (trainer_templates.py): All custom Trainers should now sub-class from any existing `Trainer` class. (#23488)

💫Enhancements:

- Add support for complex observations in CQL. (#23332)
- Bandit support for tf2. (#22838)
- Make actions sent by RLlib to the env immutable. (#24262)
- Memory leak finding toolset using tracemalloc + CI memory leak tests. (#15412)
- Enable DD-PPO to run on Windows. (#23673)

🔨 Fixes:

- APPO eager fix (APPOTFPolicy gets wrapped `as_eager()` twice by mistake). (#24268)
- CQL gets stuck when deprecated `timesteps_per_iteration` is used (use `min_train_timesteps_per_reporting` instead). (#24345)
- SlateQ runs on GPU (torch). (#23464)
- Other bug fixes: #24016, #22050, #23814, #24025, #23740, #23741, #24006, #24005, #24273, #22010, #24271, #23690, #24343, #23419, #23830, #24335, #24148, #21735, #24214, #23818, #24429

## Ray Workflow
🎉 New Features:

- Workflow step is deprecated (#23796, #23728, #23456, #24210)

🔨 Fixes:

- Fix one bug where max_retries is not aligned with ray core’s max_retries. (#22903)

🏗 Architecture refactoring:

- Integrate ray storage in workflow (#24120)

## Tune
🎉 New Features:

- Add RemoteTask based sync client (#23605) (rsync not required anymore!)
- Chunk file transfers in cross-node checkpoint syncing (#23804)
- Also interrupt training when SIGUSR1 received (#24015)
- reuse_actors per default for function trainables (#24040)
- Enable AsyncHyperband to continue training for last trials after max_t (#24222)

💫Enhancements:

- Improve testing (#23229
- Improve docstrings (#23375)
- Improve documentation (#23477, #23924)
- Simplify trial executor logic (#23396
- Make `MLflowLoggerUtil` copyable (#23333)
- Use new Checkpoint interface internally (#22801)
- Beautify Optional typehints (#23692)
- Improve missing search dependency info (#23691)
- Skip tmp checkpoints in analysis and read iteration from metadata (#23859)
- Treat checkpoints with nan value as worst (#23862)
- Clean up base ProgressReporter  API (#24010)
- De-clutter log outputs in trial runner (#24257)
- hyperopt searcher to support tune.choice([[1,2],[3,4]]). (#24181)

🔨Fixes:

- Optuna should ignore additional results after trial termination (#23495)
- Fix PTL multi GPU link (#23589)
- Improve Tune cloud release tests for durable storage (#23277)
- Fix tensorflow distributed trainable docstring (#23590)
- Simplify experiment tag formatting, clean directory names (#23672)
- Don't include nan metrics for best checkpoint (#23820)
- Fix syncing between nodes in placement groups (#23864)
- Fix memory resources for head bundle (#23861)
- Fix empty CSV headers on trial restart (#23860)
- Fix checkpoint sorting with nan values (#23909)
- Make Timeout stopper work after restoring in the future (#24217)
- Small fixes to tune-distributed for new restore modes (#24220)

## Train
**Most distributed training enhancements will be captured in the new Ray AIR category!**

🔨Fixes:

- Copy resources_per_worker to avoid modifying user input
- Fix `train.torch.get_device()` for fractional GPU or multiple GPU per worker case (#23763)
- Fix multi node horovod bug (#22564)
- Fully deprecate Ray SGD v1 (#24038)
- Improvements to fault tolerance (#22511)
- MLflow start run under correct experiment (#23662)
- Raise helpful error when required backend isn't installed (#23583)
- Warn pending deprecation for `ray.train.Trainer` and `ray.tune` DistributedTrainableCreators (#24056)

📖Documentation:

- add FAQ (#22757)

## Ray AIR
🎉 New Features:

- `HuggingFaceTrainer` & `HuggingFacePredictor` (#23615, #23876)
- `SklearnTrainer` & `SklearnPredictor` (#23803, #23850)
- `HorovodTrainer` (#23437)
- `RLTrainer` & `RLPredictor` (#23465, #24172)
- `BatchMapper` preprocessor (#23700)
- `Categorizer` preprocessor (#24180)
- `BatchPredictor` (#23808)

💫Enhancements:

- Add `Checkpoint.as_directory()` for efficient checkpoint fs processing (#23908)
- Add `config` to `Result`, extend `ResultGrid.get_best_config` (#23698)
- Add Scaling Config validation (#23889)
- Add tuner test. (#23364)
- Move storage handling to pyarrow.fs.FileSystem (#23370)
- Refactor `_get_unique_value_indices` (#24144)
- Refactor `most_frequent` `SimpleImputer` (#23706)
- Set name of Trainable to match with Trainer #23697
- Use checkpoint.as_directory() instead of cleaning up manually (#24113)
- Improve file packing/unpacking (#23621)
- Make Dataset ingest configurable (#24066)
-  Remove postprocess_checkpoint (#24297)

🔨Fixes:

- Better exception handling (#23695)
- Do not deepcopy RunConfig (#23499)
- reduce unnecessary stacktrace  (#23475)
- Tuner should use `run_config` from Trainer per default (#24079)
- Use custom fsspec handler for GS (#24008)

📖Documentation:

- Add distributed `torch_geometric` example (#23580)
- GNN example cleanup (#24080)

## Serve
🎉 New Features:

- Serve logging system was revamped! Access log is now turned on by default. (#23558)
- New Gradio notebook example for Ray Serve deployments (#23494)
- Serve now includes full traceback in deployment update error message (#23752)

💫Enhancements:

- Serve Deployment Graph was enhanced with performance fixes and structural clean up. (#24199, #24026, #24065, #23984)
- End to end tutorial for deployment graph (#23512, #22771, #23536)
- `input_schema` is now renamed as `http_adapter` for usability (#24353, #24191)
- Progress towards a declarative REST API (#23232, #23481)
- Code cleanup and refactoring (#24067, #23578, #23934, #23759)
- Protobuf based controller API for cross language client (#23004)

🔨Fixes:

- Handle `None` in `ReplicaConfig`'s `resource_dict` (#23851)
- Set `"memory"` to `None` in `ray_actor_options` by default (#23619)
- Make `serve.shutdown()` shutdown remote Serve applications (#23476)
- Ensure replica reconfigure runs after allocation check (#24052)
- Allow cloudpickle serializable objects as init args/kwargs (#24034)
- Use controller namespace when getting actors (#23896)

## Dashboard
🔨Fixes:

- Add toggle to enable showing node disk usage on K8s (#24416, #24440)
- Add job submission id as field to job snapshot (#24303)


Thanks
Many thanks to all those who contributed to this release!
@matthewdeng, @scv119, @xychu, @iycheng, @takeshi-yoshimura, @iasoon, @wumuzi520, @thetwotravelers, @maxpumperla, @krfricke, @jgiannuzzi, @kinalmehta, @avnishn, @dependabot[bot], @sven1977, @raulchen, @acxz, @stephanie-wang, @mgelbart, @xwjiang2010, @jon-chuang, @pdames, @ericl, @edoakes, @gjoseph92, @ddelange, @bkasper, @sriram-anyscale, @Zyiqin-Miranda, @rkooo567, @jbedorf, @architkulkarni, @osanseviero, @simonsays1980, @clarkzinzow, @DmitriGekhtman, @ashione, @smorad, @andenrx, @mattip, @bveeramani, @chaokunyang, @richardliaw, @larrylian, @Chong-Li, @fwitter, @shrekris-anyscale, @gjoliver, @simontindemans, @silky, @grypesc, @ijrsvt, @daikeshi, @kouroshHakha, @mwtian, @mesjou, @sihanwang41, @PavelCz, @czgdp1807, @jianoaix, @GuillaumeDesforges, @pcmoritz, @arsedler9, @n30111, @kira-lin, @ckw017, @max0x7ba, @Yard1, @XuehaiPan, @lchu-ibm, @HJasperson, @SongGuyang, @amogkam, @liuyang-my, @WangTaoTheTonic, @jovany-wang, @simon-mo, @dynamicwebpaige, @suquark, @ArturNiederfahrenhorst, @jjyao, @KepingYan, @jiaodong, @frosk1


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.13.0)

---

## ray-1.12.1: Ray-1.12.1
**Published:** 2022-05-16

Patch release with the following fixes:

- **Ray now works on Google Colab again!** The bug with memory limit fetching when running Ray in a container is now fixed (https://github.com/ray-project/ray/pull/23922).
- `ray-ml` Docker images for CPU will start being built again after they were stopped in Ray 1.9 (https://github.com/ray-project/ray/pull/24266).
- [Train/Tune] Start MLflow run under the correct experiment for Ray Train and Ray Tune integrations (https://github.com/ray-project/ray/pull/23662).
- [RLlib] Fix for APPO in eager mode (https://github.com/ray-project/ray/pull/24268).
- [RLlib] Fix Alphastar for TF2 and tracing enabled (https://github.com/ray-project/ray/commit/c5502b2aa57376b26408bb297ff68696c16f48f1).
- [Serve] Fix replica leak in anonymous namespaces (https://github.com/ray-project/ray/pull/24311).

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.12.1)

---

## ray-1.11.1: Ray-1.11.1
**Published:** 2022-05-10

Patch release including fixes for the following issues:

- Ray Job Submission not working with remote `working_dir` URLs in their runtime environment (https://github.com/ray-project/ray/pull/22018)
- Ray Tune + MLflow integration failing to set MLflow experiment ID (https://github.com/ray-project/ray/pull/23662)
- Dependencies for `gym` not pinned, leading to version incompatibility issues (https://github.com/ray-project/ray/pull/23705)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.11.1)

---

## ray-1.12.0: Ray-1.12.0
**Published:** 2022-04-08

## Highlights
- Ray AI Runtime (AIR), an open-source toolkit for building end-to-end ML applications on Ray, is now in Alpha. AIR is an effort to unify the experience of using different Ray libraries (Ray Data, Train, Tune, Serve, RLlib). You can find more information on the [docs](https://docs.ray.io/en/master/ray-air/getting-started.html) or on the [public RFC](https://github.com/ray-project/ray/issues/22488).
  - Getting involved with Ray AIR. We’ll be holding office hours, development sprints, and other activities as we get closer to the Ray AIR Beta/GA release. Want to join us? Fill out this [short form](https://forms.gle/wCCdbaQDtgErYycT6)!
- Ray [usage data collection](https://github.com/ray-project/ray/issues/20857) is now off by default. If you have any questions or concerns, please comment [on the RFC](https://github.com/ray-project/ray/issues/20857).
- New algorithms are added to RLlib: SlateQ & Bandits (for recommender systems use cases) and AlphaStar (multi-agent, multi-GPU w/ league-based self-play)
- Ray Datasets: new lazy execution model with automatic task fusion and memory-optimizing move semantics; first-class support for Pandas DataFrame blocks; efficient random access datasets.


## Ray Autoscaler

🎉 New Features
- Support cache_stopped_nodes on Azure (#21747)
- AWS Cloudwatch support (#21523) 

💫 Enhancements
- Improved documentation and standards around built in autoscaler node providers. (#22236, 22237)
- Improved KubeRay support (#22987, #22847, #22348, #22188)
- Remove redis requirement (#22083)

🔨 Fixes
- No longer print infeasible warnings for internal placement group resources. Placement groups which cannot be satisfied by the autoscaler still trigger warnings. (#22235)
- Default ami’s per AWS region are updated/fixed. (#22506)
- GCP node termination updated (#23101)
- Retry legacy k8s operator on monitor failure (#22792)
- Cap min and max workers for manually managed on-prem clusters (#21710)
- Fix initialization artifacts (#22570)
- Ensure initial scaleup with high upscaling_speed isn't limited. (#21953)

## Ray Client

🎉 New Features:
- ray.init has consistent return value in client mode and driver mode #21355

💫Enhancements:
- Gets and puts are streamed to support arbitrary object sizes #22100, #22327

🔨 Fixes:
- Fix ray client object ref releasing in wrong context #22025


## Ray Core

🎉 New Features
- RuntimeEnv:
  - Support setting timeout for runtime_env setup. (#23082)
  - Support setting pip_check and pip_version for runtime_env. (#22826, #23306)
  - env_vars will take effect when the pip install command is executed. (temporarily ineffective in conda) (#22730)
  - Support strongly-typed API ray.runtime.RuntimeEnv to define runtime env. (#22522)
  - Introduce [virtualenv](https://github.com/pypa/virtualenv) to isolate the pip type runtime env. (#21801,#22309)
- Raylet shares fate with the dashboard agent. And the dashboard agent will stay alive when it catches the port conflicts. (#22382,#23024)
- Enable dashboard in the minimal ray installation (#21896)
- Add task and object reconstruction status to ray memory cli tools(#22317)

🔨 Fixes
- Report only memory usage of pinned object copies to improve scaledown. (#22020)
- Scheduler: 
  - No spreading if a node is selected for lease request due to locality. (#22015)
  - Placement group scheduling: Non-STRICT_PACK PGs should be sorted by resource priority, size (#22762)
  - Round robin during spread scheduling (#21303)
- Object store:  
  - Increment ref count when creating an ObjectRef to prevent object from going out of scope (#22120)
  - Cleanup handling for nondeterministic object size during transfer (#22639)
  - Fix bug in fusion for spilled objects (#22571)
  - Handle IO worker failures correctly (#20752)
- Improve ray stop behavior (#22159)
- Avoid warning when receiving too much logs from a different job (#22102)
- Gcs resource manager bug fix and clean up. (#22462, #22459)
- Release GIL when running `parallel_memcopy()` / `memcpy()` during serializations. (#22492)
- Fix registering serializer before initializing Ray. (#23031)

🏗 Architecture refactoring
- Ray distributed scheduler refactoring: (#21927, #21992,  #22160, #22359, #22722, #22817, #22880, #22893, #22885, #22597, #22857, #23124)
- Removed support for bootstrapping with Redis.


## Ray Data Processing
🎉 New Features
- Big Performance and Stability Improvements:
  - Add lazy execution mode with automatic stage fusion and optimized memory reclamation via block move semantics (#22233, #22374, #22373, #22476)
  - Support for random access datasets, providing efficient random access to rows via binary search (#22749)
  - Add automatic round-robin load balancing for reading and shuffle reduce tasks, obviating the need for the `_spread_resource_prefix` hack (#21303)
- More Efficient Tabular Data Wrangling:
  - Add first-class support for Pandas blocks, removing expensive Arrow <-> Pandas conversion costs (#21894)
  - Expose `TableRow` API + minimize copies/type-conversions on row-based ops (#22305)
- Groupby + Aggregations Improvements:
  - Support mapping over groupby groups (#22715)
  - Support ignoring nulls in aggregations (#20787)
- Improved Dataset Windowing:
  - Support windowing a dataset by bytes instead of number of blocks (#22577)
  - Batch across windows in `DatasetPipeline`s (#22830)
- Better Text I/O:
  - Support streaming snappy compression for text files (#22486)
  - Allow for custom decoding error handling in `read_text()` (#21967)
  - Add option for dropping empty lines in `read_text()` (#22298)
- New Operations:
  - Add `add_column()` utility for adding derived columns (#21967)
- Support for metadata provider callback for read APIs (#22896)
- Support configuring autoscaling actor pool size (#22574)

🔨 Fixes
- Force lazy datasource materialization in order to respect `DatasetPipeline` stage boundaries (#21970)
- Simplify lifetime of designated block owner actor, and don’t create it if dynamic block splitting is disabled (#22007)
- Respect 0 CPU resource request when using manual resource-based load balancing (#22017)
- Remove batch format ambiguity by always converting Arrow batches to Pandas when `batch_format=”native”` is given (#21566)
- Fix leaked stats actor handle due to closure capture reference counting bug (#22156)
- Fix boolean tensor column representation and slicing (#22323)
- Fix unhandled empty block edge case in shuffle (#22367)
- Fix unserializable Arrow Partitioning spec (#22477)
- Fix incorrect `iter_epochs()` batch format (#22550)
- Fix infinite `iter_epochs()` loop on unconsumed epochs (#22572)
- Fix infinite hang on `split()` when `num_shards < num_rows` (#22559)
- Patch Parquet file fragment serialization to prevent metadata fetching (#22665)
- Don’t reuse task workers for actors or GPU tasks (#22482)
- Pin pipeline executor actors to driver node to allow for lineage-based fault tolerance for pipelines (#​​22715)
- Always use non-empty blocks to determine schema (#22834)
- API fix bash (#22886)
- Make label_column optional for `to_tf()` so it can be used for inference (#22916)
- Fix `schema()` for `DatasetPipeline`s (#23032)
- Fix equalized split when `num_splits == num_blocks` (#23191)

💫 Enhancements
- Optimize Parquet metadata serialization via batching (#21963)
- Optimize metadata read/write for Ray Client (#21939)
- Add sanity checks for memory utilization (#22642)

🏗 Architecture refactoring
- Use threadpool to submit `DatasetPipeline` stages (#22912)

## RLlib

🎉 New Features
- New “AlphaStar” algorithm: A parallelized, multi-agent/multi-GPU learning algorithm, implementing league-based self-play. (#21356, #21649)
- SlateQ algorithm has been re-tested, upgraded (multi-GPU capable, TensorFlow version), and bug-fixed (added to weekly learning tests). (#22389, #23276, #22544, #22543, #23168, #21827, #22738)
- Bandit algorithms: Moved into `agents` folder as first-class citizens, TensorFlow-Version, unified w/ other agents’ APIs. (#22821, #22028, #22427, #22465, #21949, #21773, #21932, #22421)
- ReplayBuffer API (in progress): Allow users to customize and configure their own replay buffers and use these inside custom or built-in algorithms. (#22114, #22390, #21808)
- Datasets support for RLlib: Dataset Reader/Writer and documentation. (#21808, #22239, #21948)

🔨 Fixes
- Fixed memory leak in SimpleReplayBuffer. (#22678)
- Fixed Unity3D built-in examples: Action bounds from -inf/inf to -1.0/1.0. (#22247)
- Various bug fixes. (#22350, #22245, #22171, #21697, #21855, #22076, #22590, #22587, #22657, #22428, #23063, #22619, #22731, #22534, #22074, #22078, #22641, #22684, #22398, #21685)

🏗 Architecture refactoring
- A3C: Moved into new `training_iteration` API (from `exeution_plan` API). Lead to a ~2.7x performance increase on a Atari + CNN + LSTM benchmark. (#22126, #22316)	
- Make `multiagent->policies_to_train` more flexible via callable option (alternative to providing a list of policy IDs). (#20735)

💫Enhancements:
- Env pre-checking module now active by default. (#22191)
- Callbacks: Added `on_sub_environment_created` and `on_trainer_init` callback options. (#21893, #22493)
- RecSim environment wrappers: Ability to use google’s RecSim for recommender systems more easily w/ RLlib algorithms (3 RLlib-ready example environments). (#22028, #21773, #22211)
- MARWIL loss function enhancement (exploratory term for stddev). (#21493)

📖Documentation:
- Docs enhancements: Setup-dev instructions; Ray datasets integration. (#22239)
- Other doc enhancements and fixes. (#23160, #23226, #22496, #22489, #22380)


## Ray Workflow
🎉 New Features:
- Support skip checkpointing.

🔨 Fixes:
- Fix an issue where the event loop is not set.


## Tune
🎉 New Features:
- Expose new checkpoint interface to users ([#22741](https://github.com/ray-project/ray/pull/22741))

💫Enhancements:
- Better error msg for grpc resource exhausted error. (#22806)
- Add CV support for XGB/LGBM Tune callbacks (#22882)
- Make sure tune.run can run inside worker thread (https://github.com/ray-project/ray/commit/b8c28d1f2beb7a141f80a5fd6053c8e8520718b9)[#22566](https://github.com/ray-project/ray/pull/22566)[)](https://github.com/ray-project/ray/commit/b8c28d1f2beb7a141f80a5fd6053c8e8520718b9)
- Add Trainable.postprocess_checkpoint (#22973)
[Trainables will now know TUNE_ORIG_WORKING_DIR (](https://github.com/ray-project/ray/commit/f5995dccdf0ab4012e511c3379b19f06f1d307b5)[#22803](https://github.com/ray-project/ray/pull/22803)[)](https://github.com/ray-project/ray/commit/f5995dccdf0ab4012e511c3379b19f06f1d307b5) 
- Retry cloud sync up/down/delete on fail (#22029)
- Support functools.partial names and treat as function in registry (#21518)

🔨Fixes:
- [Cleanup incorrectly formatted strings (Part 2: Tune) (](https://github.com/ray-project/ray/commit/761f927720586403c642dc62fa510c033fd7ffd5)[#23129](https://github.com/ray-project/ray/pull/23129)[)](https://github.com/ray-project/ray/commit/761f927720586403c642dc62fa510c033fd7ffd5)
- fix error handling for fail_fast case. (#22982)
- Remove Trainable.update_resources (#22471)
- Bump flaml from 0.6.7 to 0.9.7 in /python/requirements/ml (#22071)
- Fix analysis without registered trainable (#21475)
- Update Lightning examples to support PTL 1.5 (#20562)
- Fix WandbTrainableMixin config for rllib trainables (#22063)
- [wandb] Use resume=False per default ([#21892](https://github.com/ray-project/ray/pull/21892))

🏗 Refactoring:
- [Move resource updater out of trial executor (](https://github.com/ray-project/ray/commit/cc1728120f7d49b0016d190971bc8056d3245c5d)[#23178](https://github.com/ray-project/ray/pull/23178)[)](https://github.com/ray-project/ray/commit/cc1728120f7d49b0016d190971bc8056d3245c5d) 
- [Preparation for deadline schedulers (](https://github.com/ray-project/ray/commit/4a15c6f8f3ebb634f7cef967a097f621462e4e50)[#22006](https://github.com/ray-project/ray/pull/22006)[)](https://github.com/ray-project/ray/commit/4a15c6f8f3ebb634f7cef967a097f621462e4e50)
- [Single wait refactor. (](https://github.com/ray-project/ray/commit/323511b716416088859967686c71889ef8425204)[#21852](https://github.com/ray-project/ray/pull/21852)[)](https://github.com/ray-project/ray/commit/323511b716416088859967686c71889ef8425204)
 
📖Documentation:
- Tune docs overhaul (first part) (#22112)
- [Tune overhaul part II (](https://github.com/ray-project/ray/commit/372c620f58c2269dccd5a871a72aebb9df76e32c)[#22656](https://github.com/ray-project/ray/pull/22656)[)](https://github.com/ray-project/ray/commit/372c620f58c2269dccd5a871a72aebb9df76e32c) 
- Note TPESampler performance issues in docs (#22545)
- hyperopt notebook (#22315)

## Train
🎉 New Features
- Integration with PyTorch profiler. Easily enable the pytorch profiler with Ray Train to profile training and visualize stats in Tensorboard (#22345).
- Automatic pipelining of host to device transfer. While training is happening on one batch of data, the next batch of data is concurrently being moved from CPU to GPU (#22716, #22974)
- Automatic Mixed Precision. Easily enable PyTorch automatic mixed precision during training (#22227).

💫 Enhancements
- Add utility function to enable reproducibility for Pytorch training (#22851)
- Add initial support for metrics aggregation (#22099)
- Add support for `trainer.best_checkpoint` and `Trainer.load_checkpoint_path`. You can now directly access the best in memory checkpoint, or load an arbitrary checkpoint path to memory. (#22306)

🔨 Fixes
- Add a utility function to turn off TF autosharding (#21887)
- Fix fault tolerance for Tensorflow training (#22508)
- Train utility methods (`train.report()`, etc.) can now be called outside of a Train session (#21969)
- Fix accuracy calculation for CIFAR example (#22292)
- Better error message for placement group time out (#22845)

📖 Documentation
- Update docs for ray.train.torch import (#22555)
- Clarify shuffle documentation in `prepare_data_loader` (#22876)
- Denote `train.torch.get_device` as a Public API (#22024)
- Minor fixes on Ray Train user guide doc (#22379)


## Serve
🎉 New Features
- [Deployment Graph API](https://docs.ray.io/en/master/serve/deployment-graph.html) is now in alpha. It provides a way to build, test and deploy complex inference graph composed of many deployments. (#23177, #23252, #23301, #22840, #22710, #22878, #23208, #23290, #23256, #23324, #23289, #23285, #22473, #23125, #23210)
- New experimental REST API and CLI for creating and managing deployments. (
#22839, #22257, #23198, #23027, #22039, #22547, #22578, #22611, #22648, #22714, #22805, #22760, #22917, #23059, #23195, #23265, #23157, #22706, #23017, #23026, #23215)
- New sets of [HTTP adapters](https://docs.ray.io/en/master/serve/http-servehandle.html#http-adapters) making it easy to build simple application, as well as [Ray AI Runtime model wrappers](https://docs.ray.io/en/master/ray-air/getting-started.html#air-serve-integration) in alpha. (#22913, #22914, #22915, #22995)
- New `health_check` API for end to end user provided health check. (#22178, #22121, #22297)

🔨 Fixes
- Autoscaling algorithm will now relingquish most idle nodes when scaling down (#22669)
- Serve can now manage Java replicas (#22628)
- Added a hands-on self-contained MLflow and Ray Serve deployment example (#22192)
- Added `root_path` setting to `http_options` (#21090)
- Remove `shard_key`, `http_method`, and `http_headers` in `ServeHandle` (#21590)


## Dashboard
🔨Fixes:
- Update CPU and memory reporting in kubernetes. (#21688)

## Thanks
Many thanks to all those who contributed to this release!
@edoakes, @pcmoritz, @jiaodong, @iycheng, @krfricke, @smorad, @kfstorm, @jjyyxx, @rodrigodelazcano, @scv119, @dmatrix, @avnishn, @fyrestone, @clarkzinzow, @wumuzi520, @gramhagen, @XuehaiPan, @iasoon, @birgerbr, @n30111, @tbabej, @Zyiqin-Miranda, @suquark, @pdames, @tupui, @ArturNiederfahrenhorst, @ashione, @ckw017, @siddgoel, @Catch-Bull, @vicyap, @spolcyn, @stephanie-wang, @mopga, @Chong-Li, @jjyao, @raulchen, @sven1977, @nikitavemuri, @jbedorf, @mattip, @bveeramani, @czgdp1807, @dependabot[bot], @Fabien-Couthouis, @willfrey, @mwtian, @SlowShip, @Yard1, @WangTaoTheTonic, @Wendi-anyscale, @kaushikb11, @kennethlien, @acxz, @DmitriGekhtman, @matthewdeng, @mraheja, @orcahmlee, @richardliaw, @dsctt, @yupbank, @Jeffwan, @gjoliver, @jovany-wang, @clay4444, @shrekris-anyscale, @jwyyy, @kyle-chen-uber, @simon-mo, @ericl, @amogkam, @jianoaix, @rkooo567, @maxpumperla, @architkulkarni, @chenk008, @xwjiang2010, @robertnishihara, @qicosmos, @sriram-anyscale, @SongGuyang, @jon-chuang, @wuisawesome, @valiantljk, @simonsays1980, @ijrsvt

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.12.0)

---

## ray-1.11.0: Ray-1.11.0
**Published:** 2022-03-09

## Highlights

🎉 Ray no longer starts Redis by default. Cluster metadata previously stored in Redis is stored in the GCS now.

## Ray Autoscaler

🎉 New Features
- AWS Cloudwatch dashboard support #20266 

💫 Enhancements
- Kuberay autoscaler prototype #21086

🔨 Fixes
- Ray.autoscaler.sdk import issue #21795

## Ray Core

🎉 New Features
- Set actor died error message in ActorDiedError #20903
- Event stats is enabled by default #21515

🔨 Fixes
- Better support for nested tasks
- Fixed 16GB mac perf issue by limit the plasma store size to 2GB #21224
- Fix `SchedulingClassInfo.running_tasks` memory leak #21535
- Round robin during spread scheduling #19968

🏗 Architecture refactoring
- Refactor scheduler resource reporting public APIs #21732
- Refactor ObjectManager wait logic to WaitManager #21369

## Ray Data Processing
🎉 New Features
- More powerful to_torch() API, providing more control over the GPU batch format. (#21117)

🔨 Fixes
- Fix simple Dataset sort generating only 1 non-empty block. (#21588)
- Improve error handling across sorting, groupbys, and aggregations. (#21610, #21627)
- Fix boolean tensor column representation and slicing. (#22358)

## RLlib

🎉 New Features
- Better utils for flattening complex inputs and enable prev-actions for LSTM/attention for complex action spaces. (#21330)
- `MultiAgentEnv` pre-checker (#21476)
- Base env pre-checker. (#21569)

🔨 Fixes
- Better defaults for QMix (#21332)
- Fix contrib/MADDPG + pettingzoo coop-pong-v4. (#21452)
- Fix action unsquashing causes inf/NaN actions for unbounded action spaces. (#21110)
- Ignore PPO KL-loss term completely if kl-coeff == 0.0 to avoid NaN values (#21456)
- `unsquash_action` and `clip_action` (when None) cause wrong actions computed by `Trainer.compute_single_action`. (#21553)
- Conv2d default filter tests and add default setting for 96x96 image obs space. (#21560)
- Bing back and fix offline RL(BC & MARWIL) learning tests. (#21574, #21643)
- SimpleQ should not use a prio. replay buffer. (#21665)
- Fix video recorder env wrapper. Added test case. (#21670)

🏗 Architecture refactoring
- Decentralized multi-agent learning (#21421)
- Preparatory PR for multi-agent multi-GPU learner (alpha-star style) (#21652)

## Ray Workflow
🔨 Fixes
- Fixed workflow recovery issue due to a bug of dynamic output #21571

## Tune
🎉 New Features
- It is now possible to load all evaluated points from an experiment into a Searcher (#21506)
- Add CometLoggerCallback (#20766) 

💫 Enhancements
- Only sync the checkpoint folder instead of the entire trial folder for cloud checkpoint. (#21658)
- Add test for heterogeneous resource request deadlocks (#21397)
- Remove unused `return_or_clean_cached_pg` (#21403)
- Remove `TrialExecutor.resume_trial` (#21225)
- Leave only one canonical way of stopping a trial (#21021)

🔨 Fixes
- Replace deprecated `running_sanity_check` with `sanity_checking` in PTL integration (#21831)
- Fix loading an `ExperimentAnalysis` object without a registered `Trainable` (#21475)
- Fix stale node detection bug (#21516)
- Fixes to allow `tune/tests/test_commands.py` to run on Windows (#21342)
- Deflake PBT tests (#21366)
- Fix dtype coercion in `tune.choice` (#21270)

📖 Documentation
- Fix typo in `schedulers.rst` (#21777)

## Train
🎉 New Features
- Add PrintCallback (#21261)
- Add MLflowLoggerCallback(#20802)

💫 Enhancements
- Refactor Callback implementation (#21468, #21357, #21262)

🔨 Fixes
- Fix Dataloader (#21467)

📖 Documentation
- Documentation and example fixes (#​​21761, #21689, #21464)

## Serve
🎉 New Features
- Checkout our revampt end-to-end [tutorial](https://docs.ray.io/en/master/serve/end_to_end_tutorial.html) that walks through the deployment journey! (#20765)

🔨 Fixes
- Warn when serve.start() with different options (#21562)
- Detect http.disconnect and cancel requests properly (#21438)

Thanks
Many thanks to all those who contributed to this release!
@isaac-vidas, @wuisawesome, @stephanie-wang, @jon-chuang, @xwjiang2010, @jjyao, @MissiontoMars, @qbphilip, @yaoyuan97, @gjoliver, @Yard1, @rkooo567, @talesa, @czgdp1807, @DN6, @sven1977, @kfstorm, @krfricke, @simon-mo, @hauntsaninja, @pcmoritz, @JamieSlome, @chaokunyang, @jovany-wang, @sidward14, @DmitriGekhtman, @ericl, @mwtian, @jwyyy, @clarkzinzow, @hckuo, @vakker, @HuangLED, @iycheng, @edoakes, @shrekris-anyscale, @robertnishihara, @avnishn, @mickelliu, @ndrwnaguib, @ijrsvt, @Zyiqin-Miranda, @bveeramani, @SongGuyang, @n30111, @WangTaoTheTonic, @suquark, @richardliaw, @qicosmos, @scv119, @architkulkarni, @lixin-wei, @Catch-Bull, @acxz, @benblack769, @clay4444, @amogkam, @marin-ma, @maxpumperla, @jiaodong, @mattip, @isra17, @raulchen, @wilsonwang371, @carlogrisetti, @ashione, @matthewdeng

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.11.0)

---

## ray-1.10.0: Ray-1.10.0
**Published:** 2022-02-04

## Highlights

- 🎉 Ray Windows support is now in beta – a significant fraction of the Ray test suite is now passing on Windows. We are eager to learn about your experience with Ray 1.10 on Windows, please file issues you encounter at https://github.com/ray-project/ray/issues. In the upcoming releases we will spend more time on making Ray Serve and Runtime Environment tests pass on Windows and on polishing things.

## Ray Autoscaler
💫Enhancements:
- Add autoscaler update time to prometheus metrics (#20831)
- Fewer non terminated nodes calls in autoscaler update (#20359, #20623)

🔨 Fixes: 
- GCP TPU autoscaling fix (#20311)
- Scale-down stability fix (#21204)
- Report node launch failure in driver logs (#20814)


## Ray Client
💫Enhancements
- Client task options are encoded with pickle instead of json (#20930)


## Ray Core
🎉 New Features:
- `runtime_env`’s `pip` field now installs pip packages in your existing environment instead of installing them in a new isolated environment. (#20341)

🔨 Fixes:
- Fix bug where specifying runtime_env conda/pip per-job using local requirements file using Ray Client on a remote cluster didn’t work (#20855)
- Security fixes for `log4j2` – the `log4j2` version has been bumped to 2.17.1 (#21373)

💫Enhancements:
- Allow runtime_env working_dir and py_modules to be pathlib.Path type (#20853, #20810)
- Add environment variable to skip local runtime_env garbage collection (#21163)
- Change runtime_env error log to debug log (#20875)
- Improved reference counting for runtime_env resources (#20789)

🏗 Architecture refactoring:
- Refactor runtime_env to use protobuf for multi-language support (#19511)

📖Documentation:
- Add more comprehensive runtime_env documentation (#20222, #21131, #20352)


## Ray Data Processing
🎉 New Features:
- Added stats framework for debugging Datasets performance (#20867, #21070)
- [Dask-on-Ray] New config helper for enabling the Dask-on-Ray scheduler (#21114)

💫Enhancements:
- Reduce memory usage during when converting to a Pandas DataFrame (#20921)

🔨 Fixes:
- Fix slow block evaluation when splitting (#20693)
- Fix boundary sampling concatenation on non-uniform blocks (#20784)
- Fix boolean tensor column slicing (#20905)

🏗 Architecture refactoring:
- Refactor table block structure to support more tabular block formats (#20721)


## RLlib

🎉 New Features:
- Support for RE3 exploration algorithm (for tf only). (#19551)
- Environment pre-checks, better failure behavior and enhanced environment API. (#20481, #20832, #20868, #20785, #21027, #20811)

🏗 Architecture refactoring:
- Evaluation: Support evaluation setting that makes sure `train` doesn't ever have to wait for `eval` to finish (b/c of long episodes). (#20757); Always attach latest eval metrics. (#21011)
- Soft-deprecate `build_trainer()` utility function in favor of sub-classing `Trainer` directly (and overriding some of its methods). (#20635, #20636, #20633, #20424, #20570, #20571, #20639, #20725)
- Experimental no-flatten option for actions/prev-actions. (#20918)
- Use `SampleBatch` instead of an input dict whenever possible. (#20746)
- Switch off `Preprocessors` by default for `PGTrainer` (experimental). (#21008)
- Toward a Replay Buffer API (cleanups; docstrings; renames; move into `rllib/execution/buffers` dir) (#20552)

📖Documentation:
- Overhaul of auto-API reference pages. (#19786, #20537, #20538, #20486, #20250)
- README and RLlib landing page overhaul (#20249).
- Added example containing code to compute an adapted (time-dependent) GAE used by the PPO algorithm (#20850).

🔨 Fixes:
- Smaller fixes and enhancements: #20704, #20541, #20793, #20743.


## Tune
🎉 New Features:
- Introduce TrialCheckpoint class, making checkpoint down/upload easie (#20585)
- Add random state to `BasicVariantGenerator` (#20926)
- Multi-objective support for Optuna (#20489)

💫Enhancements:
- Add `set_max_concurrency` to Searcher API (#20576)
- Allow for tuples in _split_resolved_unresolved_values. (#20794)
- Show the name of training func, instead of just ImplicitFunction. (#21029)
- Enforce one future at a time for any given trial at any given time. (#20783)
move `on_no_available_trials` to a subclass under `runner` (#20809)
- Clean up code  (#20555, #20464, #20403, #20653, #20796, #20916, #21067)
- Start restricting TrialRunner/Executor interface exposures. (#20656)
- TrialExecutor should not take in Runner interface. (#20655)


🔨Fixes:
- Deflake test_tune_restore.py (#20776)
- Fix best_trial_str for nested custom parameter columns (#21078)
- Fix checkpointing error message on K8s (#20559)
- Fix testResourceScheduler and testMultiStepRun. (#20872)
- Fix tune cloud tests for function and rllib trainables (#20536)
- Move _head_bundle_is_empty after conversion (#21039)
- Elongate test_trial_scheduler_pbt timeout. (#21120)


## Train
🔨Fixes:
- Ray Train environment variables are automatically propagated and do not need to be manually set on every node (#20523)
- Various minor fixes and improvements (#20952, #20893, #20603, #20487)
📖Documentation:
- Update saving/loading checkpoint docs (#20973). Thanks @jwyyy!
- Various minor doc updates (#20877, #20683)


## Serve
💫Enhancements:
- Add validation to Serve AutoscalingConfig class (#20779)
- Add Serve metric for HTTP error codes (#21009)

🔨Fixes:
- No longer create placement group for deployment with no resources (#20471)
- Log errors in deployment initialization/configuration user code (#20620)


## Jobs
🎉 New Features:
- Logs can be streamed from job submission server with `ray job logs` command (#20976)
- Add documentation for ray job submission (#20530)
- Propagate custom headers field to JobSubmissionClient and apply to all requests (#20663)

🔨Fixes:
- Fix job serve accidentally creates local ray processes instead of connecting (#20705)

💫Enhancements:
- [Jobs] Update CLI examples to use the same setup (#20844)

## Thanks
Many thanks to all those who contributed to this release!

@dmatrix, @suquark, @tekumara, @jiaodong, @jovany-wang, @avnishn, @simon-mo, @iycheng, @SongGuyang, @ArturNiederfahrenhorst, @wuisawesome, @kfstorm, @matthewdeng, @jjyao, @chenk008, @Sertingolix, @larrylian, @czgdp1807, @scv119, @duburcqa, @runedog48, @Yard1, @robertnishihara, @geraint0923, @amogkam, @DmitriGekhtman, @ijrsvt, @kk-55, @lixin-wei, @mvindiola1, @hauntsaninja, @sven1977, @Hankpipi, @qbphilip, @hckuo, @newmanwang, @clay4444, @edoakes, @liuyang-my, @iasoon, @WangTaoTheTonic, @fgogolli, @dproctor, @gramhagen, @krfricke, @richardliaw, @bveeramani, @pcmoritz, @ericl, @simonsays1980, @carlogrisetti, @stephanie-wang, @AmeerHajAli, @mwtian, @xwjiang2010, @shrekris-anyscale, @n30111, @lchu-ibm, @Scalsol, @seonggwonyoon, @gjoliver, @qicosmos, @xychu, @iamhatesz, @architkulkarni, @jwyyy, @rkooo567, @mattip, @ckw017, @MissiontoMars, @clarkzinzow



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.10.0)

---

## ray-1.9.2: Ray-1.9.2
**Published:** 2022-01-11

Patch release to bump the `log4j` version from `2.16.0` to `2.17.0`.  This resolves the security issue [CVE-2021-45105](https://github.com/advisories/GHSA-p6xc-xr62-6r2g).

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.9.2)

---

## ray-1.9.1: Ray-1.9.1
**Published:** 2021-12-22

Patch release to bump the `log4j2` version from `2.14` to `2.16`.  This resolves the security vulnerabilities  https://nvd.nist.gov/vuln/detail/CVE-2021-44228 and https://nvd.nist.gov/vuln/detail/CVE-2021-45046. 

No library or core changes included.

Thanks @seonggwonyoon and @ijrsvt for contributing the fixes!

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.9.1)

---

## ray-1.9.0: Ray-1.9.0
**Published:** 2021-12-03

## Highlights

- Ray Train is now in beta! If you are using Ray Train, we’d love to hear your feedback [here](https://docs.google.com/forms/d/e/1FAIpQLSfI3asn-m1cQSIbdrk_cd6qYenZvt-eNTVfTwba3SVhmHcHIg/viewform)!
- Ray Docker images for multiple CUDA versions are now provided (#19505)! You can specify a `-cuXXX` suffix to pick a specific version.
  - `ray-ml:cpu` images are now deprecated. The `ray-ml` images are only built for GPU.
- Ray Datasets now supports groupby and aggregations! See the [groupby API](https://docs.ray.io/en/master/data/package-ref.html#ray.data.Dataset.groupby) and [GroupedDataset](https://docs.ray.io/en/master/data/package-ref.html#groupeddataset-api) docs for usage.
- We are making continuing progress in improving Ray stability and usability on Windows. We encourage you to try it out and report feedback or issues at https://github.com/ray-project/ray/issues.
- We are launching a Ray Job Submission server + CLI & SDK clients to make it easier to submit and monitor Ray applications when you don’t want an active connection using Ray Client. This is currently in alpha, so the APIs are subject to change, but please test it out and file issues / leave feedback on GitHub & discuss.ray.io!


## Ray Autoscaler
💫Enhancements: 
- Graceful termination of Ray nodes prior to autoscaler scale down (#20013)
- Ray Clusters on AWS are colocated in one Availability Zone to reduce costs & latency (#19051)

## Ray Client
🔨 Fixes: 
- ray.put on a list of of objects now returns a single object ref (​​#19737)

## Ray Core
🎉 New Features: 
- Support remote file storage for runtime_env (#20280, #19315)
- Added ray job submission client, cli and rest api (#19567, #19657, #19765, #19845, #19851, #19843, #19860, #19995, #20094, #20164, #20170, #20192, #20204)

💫Enhancements: 
- Garbage collection for runtime_env (#20009, #20072)
- Improved logging and error messages for runtime_env (#19897, #19888, #18893)

🔨 Fixes: 
- Fix runtime_env hanging issues (#19823) 
- Fix specifying runtime env in @ray.remote decorator with Ray Client (#19626)
- Threaded actor / core worker / named actor race condition fixes (#19751, #19598, #20178, #20126)

📖Documentation: 
- New page “Handling Dependencies”
- New page “Ray Job Submission: Going from your laptop to production”

## Ray Java
API Changes:
- Fully supported namespace APIs. ([Check out the namespace for more information.](https://docs.ray.io/en/latest/namespaces.html)) #19468 #19986 #20057 
- Removed global named actor APIs and global placement group APIs. #20219 #20135
- Added timeout parameter for `Ray.Get()` API. #20282

Note: 
- Use `Ray.getActor(name, namespace)` API to get a named actor between jobs instead of `Ray.getGlobalActor(name)`.
- Use `PlacementGroup.getPlacementGroup(name, namespace)` API to get a placement group between jobs instead of `PlacementGroup.getGlobalPlacementGroup(name)`.

## Ray Datasets
🎉 New Features:
- Added groupby and aggregations (#19435, #19673, #20010, #20035, #20044, #20074)
- Support custom write paths (#19347)

🔨 Fixes:
- Support custom CSV write options (#19378)

🏗 Architecture refactoring:
- Optimized block compaction (#19681)

## Ray Workflow
🎉 New Features:
- Workflow right now support events (#19239)
- Allow user to specify metadata for workflow and steps (#19372)
- Allow in-place run a step if the resources match (#19928)

🔨 Fixes:
- Fix the s3 path issue (#20115)

## RLlib
🏗 Architecture refactoring:
- “framework=tf2” + “eager_tracing=True” is now (almost) as fast as “framework=tf”. A check for tf2.x eager re-traces has been added making sure re-tracing does not happen outside the initial function calls. All CI learning tests (CartPole, Pendulum, FrozenLake) are now also run as framework=tf2. (#19273, #19981, #20109)
- Prepare deprecation of `build_trainer`/`build_(tf_)?policy` utility functions. Instead, use sub-classing of `Trainer` or `Torch|TFPolicy`. POCs done for `PGTrainer`, `PPO[TF|Torch]Policy`. (#20055, #20061)
- V-trace (APPO & IMPALA): Don’t drop last ts can be optionally switch on. The default is still to drop it, but this may be changed in a future release. (#19601)
- Upgrade to gym 0.21. (#19535)

🔨 Fixes:
- Minor bugs/issues fixes and enhancements: #19069, #19276, #19306, #19408, #19544, #19623, #19627, #19652, #19693, #19805, #19807, #19809, #19881, #19934, #19945,  #20095, #20128, #20134, #20144, #20217, #20283, #20366, #20387

📖Documentation:
- RLlib main page (“RLlib in 60sec”) overhaul. (#20215, #20248, #20225, #19932, #19982)
- Major docstring cleanups in preparation for complete overhaul of API reference pages. (#19784, #19783, #19808, #19759, #19829, #19758, #19830)
- Other documentation enhancements. (#19908, #19672, #20390)


## Tune

💫Enhancements:
- Refactored and improved experiment analysis (#20197, #20181)
- Refactored cloud checkpointing API/SyncConfig (#20155, #20418, #19632, #19641, #19638, #19880, #19589, #19553, #20045, #20283)
- Remove magic results (e.g. config) before calculating trial result metrics (#19583)
- Removal of tech debt (#19773, #19960, #19472, #17654)
- Improve testing (#20016, #20031, #20263, #20210, #19730
- Various enhancements (#19496, #20211)

🔨Fixes:
- Documentation fixes (#20130, #19791)
- Tutorial fixes (#20065, #19999)
- Drop 0 value keys from PGF (#20279)
- Fix shim error message for scheduler (#19642)
- Avoid looping through _live_trials twice in _get_next_trial. (#19596)
- clean up legacy branch in update_avail_resources. (#20071)
- fix Train/Tune integration on Client (#20351)

## Train

Ray Train is now in Beta! The beta version includes various usability improvements for distributed PyTorch training and checkpoint management, support for [Ray Client](https://docs.ray.io/en/master/cluster/ray-client.html), and an [integration with Ray Datasets](https://docs.ray.io/en/master/train/user_guide.html#distributed-data-ingest-ray-datasets) for distributed data ingest.

Check out the docs [here](https://docs.ray.io/en/latest/train/train.html), and the migration guide from Ray SGD to Ray Train [here](https://docs.ray.io/en/latest/train/migration-guide.html). If you are using Ray Train, we’d love to hear your feedback [here](https://docs.google.com/forms/d/e/1FAIpQLSfI3asn-m1cQSIbdrk_cd6qYenZvt-eNTVfTwba3SVhmHcHIg/viewform)!

🎉 New Features:
- New `train.torch.prepare_model(...)` and `train.torch.prepare_data_loader(...)` [API](https://docs.ray.io/en/master/train/user_guide.html#update-training-function) to automatically handle preparing your PyTorch model and DataLoader for distributed training (#20254).
- Checkpoint management and support for custom checkpoint strategies (#19111). 
- Easily [configure](https://docs.ray.io/en/master/train/user_guide.html#configuring-checkpoints) what and how many checkpoints to save to disk.
- Support for [Ray Client](https://docs.ray.io/en/master/cluster/ray-client.html) (#20123, #20351).

💫Enhancements:
- Simplify workflow for training with a single worker (#19814).
- [Ray Placement Groups](https://docs.ray.io/en/master/placement-group.html) are used for scheduling the training workers (#20091). 
- `PACK` strategy is used by default but can be changed by setting the `TRAIN_ENABLE_WORKER_SPREAD` environment variable.
- Automatically unwrap Torch DDP model and convert to CPU when saving a model as checkpoint (#20333).

🔨Fixes:
- Fix `HorovodBackend` to automatically detect NICs- thanks @tgaddair! (#19533).

📖Documentation:
- Denote public facing APIs with beta stability (#20378)
- Doc updates (#20271)

## Serve
We would love to hear from you! Fill out the [Ray Serve survey here](https://forms.gle/zg4gDS84z8wTpKBLA).

🎉 New Features:
- New `checkpoint_path` configuration allows Serve to save its internal state to external storage (disk, S3, and GCS) and [recover upon failure](https://docs.ray.io/en/master/serve/deployment.html#failure-recovery). (#19166, #19998, #20104)
- [Replica autoscaling](https://docs.ray.io/en/master/serve/core-apis.html#autoscaling) is ready for testing out! (#19559, #19520)
- Native [Pipeline API for model composition](https://docs.ray.io/en/master/serve/pipeline.html) is ready for testing as well! 

🔨Fixes:
- Serve deployment functions or classes can take no parameters (#19708)
- Replica slow start message is improved. You can now see whether it is slow to allocate resources or slow to run constructor. (#19431)
- `pip install ray[serve]` will now install `ray[default]` as well. (#19570)

🏗 Architecture refactoring:
- The terminology of “backend” and “endpoint” are officially deprecated in favor of “deployment”. (#20229, #20085, #20040, #20020, #19997, #19947, #19923, #19798).
- Progress towards Java API compatibility (#19463).

## Dashboard
- Ray Dashboard is now enabled on Windows! (#19575)

## Thanks
Many thanks to all those who contributed to this release!
@krfricke, @stefanbschneider, @ericl, @nikitavemuri, @qicosmos, @worldveil, @triciasfu, @AmeerHajAli, @javi-redondo, @architkulkarni, @pdames, @clay4444, @mGalarnyk, @liuyang-my, @matthewdeng, @suquark, @rkooo567, @mwtian, @chenk008, @dependabot[bot], @iycheng, @jiaodong, @scv119, @oscarknagg, @Rohan138, @stephanie-wang, @Zyiqin-Miranda, @ijrsvt, @roireshef, @tkaymak, @simon-mo, @ashione, @jovany-wang, @zenoengine, @tgaddair, @11rohans, @amogkam, @zhisbug, @lchu-ibm, @shrekris-anyscale, @pcmoritz, @yiranwang52, @mattip, @sven1977, @Yard1, @DmitriGekhtman, @ckw017, @WangTaoTheTonic, @wuisawesome, @kcpevey, @kfstorm, @rhamnett, @renos, @TeoZosa, @SongGuyang, @clarkzinzow, @avnishn, @iasoon, @gjoliver, @jjyao, @xwjiang2010, @dmatrix, @edoakes, @czgdp1807, @heng2j, @sungho-joo, @lixin-wei

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.9.0)

---

## ray-1.8.0: Ray-1.8.0
**Published:** 2021-11-02

## Highlights
- Ray SGD has been rebranded to Ray Train! The new documentation landing page can be found [here](https://docs.ray.io/en/master/train/train.html).
- Ray Datasets is now in beta! The beta release includes a new integration with Ray Train yielding scalable ML ingest for distributed training. Check out the docs [here](https://docs.ray.io/en/master/data/dataset.html), try it out for your ML ingest and batch inference workloads, and let us know how it goes!
- This Ray release supports Apple Silicon (M1 Macs). [Check out the installation instructions for more information!](https://docs.ray.io/en/master/installation.html#apple-silicon-support)
## Ray Autoscaler
🎉 New Features:
- Fake multi-node mode for autoscaler testing (#18987) 

💫Enhancements: 
- Improve unschedulable task warning messages by integrating with the autoscaler (#18724)

## Ray Client
💫Enhancements
- Use async rpc for remote call and actor creation (#18298)

## Ray Core
💫Enhancements
- Eagerly install job-level runtime_env (#19449, #17949)
 
🔨 Fixes:
- Fixed resource demand reporting for infeasible 1-CPU tasks (#19000)
- Fixed printing Python stack trace in Python worker (#19423)
- Fixed macOS security popups (#18904)
- Fixed thread safety issues for coreworker (#18902, #18910, #18913 #19343)
- Fixed placement group performance and resource leaking issues (#19277, #19141, #19138, #19129, #18842, #18652)
- Improve unschedulable task warning messages by integrating with the autoscaler (#18724)
- Improved Windows support (#19014, #19062, #19171, #19362)
- Fix runtime_env issues (#19491, #19377, #18988)

## Ray Data
Ray Datasets is now in beta! The beta release includes a new integration with Ray Train yielding scalable ML ingest for distributed training. It supports repeating and rewindowing pipelines, zipping two pipelines together, better cancellation of Datasets workloads, and many performance improvements. Check out the docs [here](https://docs.ray.io/en/master/data/dataset.html), try it out for your ML ingest and batch inference workloads, and let us know how it goes!

🎉 New Features:
- Ray Train integration (#17626)
- Add support for repeating and rewindowing a DatasetPipeline (#19091)
- .iter_epochs() API for iterating over epochs in a DatasetPipeline (#19217)
- Add support for zipping two datasets together (#18833)
- Transformation operations are now cancelled when one fails or the entire workload is killed (#18991)
- Expose from_pandas()/to_pandas() APIs that accept/return plain Pandas DataFrames (#18992)
- Customize compression, read/write buffer size, metadata, etc. in the IO layer (#19197)
- Add spread resource prefix for manual round-robin resource-based task load balancing

💫Enhancements:
- Minimal rows are now dropped when doing an equalized split (#18953)
- Parallelized metadata fetches when reading Parquet datasets (#19211)

🔨 Fixes:
- Tensor columns now properly support table slicing (#19534)
- Prevent Datasets tasks from being captured by Ray Tune placement groups (#19208)
- Empty datasets are properly handled in most transformations (#18983)

🏗 Architecture refactoring:
- Tensor dataset representation changed to a table with a single tensor column (#18867)

## RLlib

🎉 New Features:
- Allow n-step > 1 and prioritized replay for R2D2 and RNNSAC agents. (18939)

🔨 Fixes:
- Fix memory leaks in TF2 eager mode. (#19198)
- Faster worker spaces inference if specified through configuration. (#18805)
- Fix bug for complex obs spaces containing Box([2D shape]) and discrete components. (#18917)
- Torch multi-GPU stats not protected against race conditions. (#18937)
- Fix SAC agent with dict space. (#19101)
- Fix A3C/IMPALA in multi-agent setting. (#19100)

🏗 Architecture refactoring:
- Unify results dictionary returned from Trainer.train() across agents regardless of (tf or pytorch, multi-agent, multi-gpu, or algos that use >1 SGD iterations, e.g. ppo) (#18879)

## Ray Workflow

🎉 New Features:
- Introduce workflow.delete (#19178)

🔨Fixes:
- Fix the bug which allow workflow step to be executed multiple times (#19090)

🏗 Architecture refactoring:
- Object reference serialization is decoupled from workflow storage (#18328)


## Tune

🎉 New Features:
- PBT: Add burn-in period (#19321)

💫Enhancements:
- Optional forcible trial cleanup, return default autofilled metrics even if Trainable doesn't report at least once (#19144)
- Use queue to display JupyterNotebookReporter updates in Ray client (#19137)
- Add resume="AUTO" and enhance resume error messages (#19181)
- Provide information about resource deadlocks, early stopping in Tune docs (#18947)
- Fix HEBOSearch installation docs (#18861)
- OptunaSearch: check compatibility of search space with evaluated_rewards (#18625)
- Add `save` and `restore` methods for searchers that were missing it & test (#18760)
- Add documentation for reproducible runs (setting seeds) (#18849)
- Depreciate `max_concurrent` in `TuneBOHB` (#18770)
- Add `on_trial_result` to ConcurrencyLimiter (#18766)
- Ensure arguments passed to tune `remote_run` match (#18733)
- Only disable ipython in remote actors (#18789)

🔨Fixes:
- Only try to sync driver if sync_to_driver is actually enabled (#19589)
- sync_client: Fix delete template formatting (#19553)
- Force no result buffering for hyperband schedulers (#19140)
- Exclude trial checkpoints in experiment sync (#19185)
- Fix how durable trainable is retained in global registry (#19223, #19184)
- Ensure `loc` column in progress reporter is filled (#19182)
- Deflake PBT Async test (#19135)
- Fix `Analysis.dataframe()` documentation and enable passing of `mode=None` (#18850)

## Ray Train (SGD)

Ray SGD has been rebranded to Ray Train! The new documentation landing page can be found [here](https://docs.ray.io/en/master/train/train.html). Ray Train is integrated with Ray Datasets for distributed data loading while training, documentation available [here](https://docs.ray.io/en/master/train/user_guide.html#distributed-data-ingest-ray-datasets).

🎉 New Features:
- Ray Datasets Integration (#17626)

🔨Fixes:
- Improved support for multi-GPU training (#18824, #18958)
- Make actor creation async (#19325)

📖Documentation:
- Rename Ray SGD v2 to Ray Train (#19436)
- Added migration guide from Ray SGD v1 (#18887)

## Serve
🎉 New Features:
- Add ability to recover from a checkpoint on cluster failure (#19125)
- Support kwargs to deployment constructors (#19023)

🔨Fixes:
- Fix asyncio compatibility issue (#19298)
- Catch spurious ConnectionErrors during shutdown (#19224)
- Fix error with uris=None in runtime_env (#18874)
- Fix shutdown logic with exit_forever (#18820)

🏗 Architecture refactoring:
- Progress towards Serve autoscaling (#18793, #19038, #19145)
- Progress towards Java support (#18630)
- Simplifications for long polling (#19154, #19205)

## Dashboard
🎉 New Features:
- Basic support for the dashboard on Windows (#19319)

🔨Fixes:
- Fix healthcheck issue causing the dashboard to crash under load (#19360)
- Work around aiohttp 4.0.0+ issues (#19120)

🏗 Architecture refactoring:
- Improve dashboard agent retry logic (#18973)

## Thanks
Many thanks to all those who contributed to this release!
@rkooo567, @lchu-ibm, @scv119, @pdames, @suquark, @antoine-galataud, @sven1977, @mvindiola1, @krfricke, @ijrsvt, @sighingnow, @marload, @jmakov, @clay4444, @mwtian, @pcmoritz, @iycheng, @ckw017, @chenk008, @jovany-wang, @jjyao, @hauntsaninja, @franklsf95, @jiaodong, @wuisawesome, @odp, @matthewdeng, @duarteocarmo, @czgdp1807, @gjoliver, @mattip, @richardliaw, @max0x7ba, @Jasha10, @acxz, @xwjiang2010, @SongGuyang, @simon-mo, @zhisbug, @ccssmnn, @Yard1, @hazeone, @o0olele, @froody, @robertnishihara, @amogkam, @sasha-s, @xychu, @lixin-wei, @architkulkarni, @edoakes, @clarkzinzow, @DmitriGekhtman, @avnishn, @liuyang-my, @stephanie-wang, @Chong-Li, @ericl, @juliusfrost, @carlogrisetti

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.8.0)

---

## ray-1.6.0: Ray-1.6.0
**Published:** 2021-08-23

## Highlights

* [Runtime Environments](https://docs.ray.io/en/releases-1.6.0/advanced.html#runtime-environments) are ready for general use!  This feature enables you to dynamically specify per-task, per-actor and per-job dependencies, including a working directory, environment variables, pip packages and conda environments. Install it with `pip install -U 'ray[default]'`.
* Ray Dataset is now in alpha! Dataset is an interchange format for distributed datasets, powered by Arrow. You can also use it for a basic Ray native data processing experience. [Check it out here. ](https://docs.ray.io/en/releases-1.6.0/data/dataset.html)
* [Ray Lightning](https://github.com/ray-project/ray_lightning) v0.1 has been released! You can install it via `pip install ray-lightning`. Ray Lightning is a library of PyTorch Lightning plugins for distributed training using Ray. Features:
  * Enables quick and easy parallel training
  * Supports [PyTorch DDP](https://github.com/ray-project/ray_lightning#pytorch-distributed-data-parallel-plugin-on-ray), [Horovod](https://github.com/ray-project/ray_lightning#horovod-plugin-on-ray), and [Sharded DDP with Fairscale](https://github.com/ray-project/ray_lightning#model-parallel-sharded-training-on-ray)
  * Integrates with [Ray Tune for hyperparameter optimization](https://github.com/ray-project/ray_lightning#hyperparameter-tuning-with-ray-tune) and is compatible with [Ray Client](https://github.com/ray-project/ray_lightning#multi-node-training-from-your-laptop)
* `pip install ray` now has a significantly reduced set of dependencies. Features such as the dashboard, the cluster launcher, runtime environments, and observability metrics may require `pip install -U 'ray[default]'` to be enabled. Please report any issues on Github if this is an issue!

## Ray Autoscaler

🎉 New Features:

* The Ray autoscaler now supports TPUs on GCP. Please refer to this example for spinning up a [simple TPU cluster](https://github.com/ray-project/ray/blob/releases/1.6.0/python/ray/autoscaler/gcp/tpu.yaml). (#17278)

💫Enhancements:

* Better AWS networking configurability (#17236 #17207 #14080)
* Support for running autoscaler without NodeUpdaters (#17194, #17328)

🔨 Fixes:

* Code clean up and corrections to downscaling policy (#17352)
* Docker file sync fix (#17361)  

## Ray Client

💫Enhancements:

* Updated docs for client server ports and ray.init(ray://) (#17003, #17333)
* Better error handling for deserialization failures (#17035)

🔨 Fixes:

* Fix for server proxy not working with non-default redis passwords (#16885)

## Ray Core

🎉 New Features:

* [Runtime Environments](https://docs.ray.io/en/releases-1.6.0/advanced.html#runtime-environments) are ready for general use!
  * Specify a working directory to upload your local files to all nodes in your cluster.
  * Specify different conda and pip dependencies for your tasks and actors and have them installed on the fly.

🔨 Fixes:

* Fix plasma store bugs for better data processing stability (#16976, #17135, #17140, #17187, #17204, #17234, #17396, #17550)
* Fix a placement group bug where CUDA_VISIBLE_DEVICES were not properly detected (#17318)
* Improved Ray stacktrace messages. (#17389)
* Improved GCS stability and scalability (#17456, #17373, #17334, #17238, #17072)

🏗 Architecture refactoring:

* Plasma store refactor for better testability and extensibility. (#17332, #17313, #17307)  

## Ray Data Processing

Ray Dataset is now in alpha! Dataset is an interchange format for distributed datasets, powered by Arrow. You can also use it for a basic Ray native data processing experience. [Check it out here. ](https://docs.ray.io/en/releases-1.6.0/data/dataset.html)  

## RLLib

🎉 New Features:

* Support for RNN/LSTM models with SAC (new agent: "RNNSAC"). Shoutout to ddworak94! (#16577)
* Support for ONNX model export (tf and torch). (#16805)
* Allow Policies to be added to/removed from a Trainer on-the-fly. (#17566)  

🔨 Fixes:

* Fix for view requirements captured during compute actions test pass. Shoutout to Chris Bamford (#15856)
* Issues: 17397, 17425, 16715, 17174\. When on driver, Torch|TFPolicy should not use `ray.get_gpu_ids()` (b/c no GPUs assigned by ray). (#17444)

* Other bug fixes: #15709, #15911, #16083, #16716, #16744, #16896, #16999, #17010, #17014, #17118, #17160, #17315, #17321, #17335, #17341, #17356, #17460, #17543, #17567, #17587  

🏗 Architecture refactoring:

* CV2 to Skimage dependency change (CV2 still supported). Shoutout to Vince Jankovics. (#16841)
* Unify tf and torch policies wrt. multi-GPU handling: PPO-torch is now 33% faster on Atari and 1 GPU. (#17371)
* Implement all policy maps inside RolloutWorkers to be LRU-caches so that a large number of policies can be added on-the-fly w/o running out of memory. (#17031)
* Move all tf static-graph code into DynamicTFPolicy, such that policies can be deleted and their tf-graph is GC'd. (#17169)
* Simplify multi-agent configs: In most cases, creating dummy envs (only to retrieve spaces) are no longer necessary. (#16565, #17046)  

📖Documentation:

* Examples scripts do-over (shoutout to Stefan Schneider for this initiative).
* Example script: League-based self-play with "open spiel" env. (#17077)
* Other doc improvements: #15664 (shoutout to kk-55), #17030, #17530

## Tune

🎉 New Features:

* Dynamic trial resource allocation with ResourceChangingScheduler (#16787)
* It is now possible to use a define-by-run function to generate a search space with OptunaSearcher (#17464)

💫Enhancements:

* String names of searchers/schedulers can now be used directly in tune.run (#17517)
* Filter placement group resources if not in use (progress reporting) (#16996)
* Add unit tests for flatten_dict (#17241)

🔨Fixes:

* Fix HDFS sync down template (#17291)
* Re-enable TensorboardX without Torch installed (#17403)

📖Documentation:

* LightGBM integration (#17304)
* Other documentation improvements: #17407 (shoutout to amavilla), #17441, #17539, #17503

## SGD

🎉 New Features:

* We have started initial development on a new RaySGD v2! We will be rolling it out in a future version of Ray. See the documentation [here](https://docs.ray.io/en/master/raysgd/v2/raysgd.html#sgd-v2-docs). (#17536, #17623, #17357, #17330, #17532, #17440, #17447, #17300, #17253)

💫Enhancements:

* Placement Group support for TorchTrainer (#17037)  

## Serve

🎉 New Features:

* Add Ray API stability annotations to Serve, marking many `serve.\*` APIs as `Stable` (#17295)
* Support `runtime_env`'s `working_dir` for Ray Serve (#16480)

🔨Fixes:

* Fix FastAPI's response_model not added to class based view routes (#17376)
* Replace `backend` with `deployment` in metrics & logging (#17434)

🏗Stability Enhancements:

* Run Ray Serve with multi & single deployment large scale (1K+ cores) test running nightly (#17310, #17411, #17368, #17026, #17277)

## Thanks
Many thanks to all who contributed to this release:

@suquark, @xwjiang2010, @clarkzinzow, @kk-55, @mGalarnyk, @pdames, @Souphis, @edoakes, @sasha-s, @iycheng, @stephanie-wang, @antoine-galataud, @scv119, @ericl, @amogkam, @ckw017, @wuisawesome, @krfricke, @vakker, @qingyun-wu, @Yard1, @juliusfrost, @DmitriGekhtman, @clay4444, @mwtian, @corentinmarek, @matthewdeng, @simon-mo, @pcmoritz, @qicosmos, @architkulkarni, @rkooo567, @navneet066, @dependabot[bot], @jovany-wang, @kombuchafox, @thomasjpfan, @kimikuri, @Ivorforce, @franklsf95, @MissiontoMars, @lantian-xu, @duburcqa, @ddworak94, @ijrsvt, @sven1977, @kira-lin, @SongGuyang, @kfstorm, @Rohan138, @jamesmishra, @amavilla, @fyrestone, @lixin-wei, @stefanbschneider, @jiaodong, @richardliaw, @WangTaoTheTonic, @chenk008, @Catch-Bull, @Bam4d

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.6.0)

---

## ray-1.5.2: Ray-1.5.2
**Published:** 2021-08-12

Cherrypick release to address RLlib issue, no library or core changes included.

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.5.2)

---

## ray-1.5.1: Ray-1.5.1
**Published:** 2021-07-31

Cherrypick release to address a few external integration and documentation issues, no library or core changes included.

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.5.1)

---

## ray-1.5.0: Ray-1.5.0 
**Published:** 2021-07-26

# Ray 1.5.0 Release Note

## Highlight
- Ray Datasets is now in alpha (https://docs.ray.io/en/master/data/dataset.html)
- LightGBM on Ray is now in beta (https://github.com/ray-project/lightgbm_ray). 
  - enables multi-node and multi-GPU training
  - integrates seamlessly with distributed hyperparameter optimization library Ray Tune
  - comes with fault tolerance handling mechanisms, and
  - supports distributed dataframes and distributed data loading

## Ray Autoscaler
🎉 New Features:
- Aliyun support (#15712)

💫 Enhancements:
- [Kubernetes] Operator refactored to use Kopf package (#15787) 
- Flag to control config bootstrap for rsync (#16667)
 - Prometheus metrics for Autoscaler (#16066, #16198)
 - Allows launching in subnets where public IP assignments off by default (#16816)

🔨 Fixes:
- [Kubernetes] Fix GPU=0 resource handling (#16887)
- [Kubernetes] Release docs updated with K8s test instructions (#16662)
- [Kubernetes] Documentation update (#16570) 
- [Kubernetes] All official images set to rayproject/ray:latest (#15988 #16205)
- [Local] Fix bootstrapping ray at a given static set of ips (#16202, #16281)
- [Azure] Fix Azure Autoscaling Failures (#16640)
- Handle node type key change / deletion (#16691)
- [GCP] Retry GCP BrokenPipeError (#16952)

## Ray Client
🎉 New Features:
- Client integrations with major Ray Libraries (#15932, #15996, #16103, #16034, #16029, #16111, #16301)
- Client Connect now returns a context that has`disconnect` and can be used as a context manager (#16021)

💫 Enhancements:
- Better support for multi-threaded client-side applications (#16731, #16732)
- Improved error messages and warnings when misusing Ray Client (#16454, #16508, #16588, #16163)
- Made Client Object & Actor refs a subclass of their non-client counterparts (#16110)

🔨 Fixes:
- `dir()` Works for client-side Actor Handles (#16157)
- Avoid server-side time-outs  (#16554)
- Various fixes to the client-server proxy (#16040, #16038, #16057, #16180)

## Ray Core
🎉 New Features:
- Ray dataset alpha is available! 

🔨 Fixes:
- Fix various Ray IO layer issues that fixes hanging & high memory usage (#16408, #16422, #16620, #16824, #16791, #16487, #16407, #16334, #16167, #16153, #16314, #15955, #15775)
- Namespace now properly isolates placement groups (#16000)
- More efficient object transfer for spilled objects (#16364, #16352)

🏗 Architecture refactoring:
- From Ray 1.5.0, liveness of Ray jobs are guaranteed as long as there’s enough disk space in machines with the “fallback allocator” mechanism which allocates plasma objects to the disk directly when objects cannot be created in memory or spilled to the disk.

## RLlib
🎉 New Features:
- Support for adding/deleting Policies to a Trainer on-the-fly (#16359, #16569, #16927).
- Added new “input API” for customizing offline datasets (shoutout to Julius F.). (#16957)
- Allow for external env PolicyServer to listen on n different ports (given n rollout workers); No longer require creating an env on the server side to get env’s spaces. (#16583).

🔨 Fixes:
- CQL: Bug fixes and clean-ups (fixed iteration count). (#16531, #16332)
- D4RL: #16721
- ensure curiosity exploration actions are passed in as tf tensors (shoutout to Manny V.).  (#15704)
- Other bug fixes and cleanups: #16162 and #16309 (shoutout to Chris B.), #15634, #16133, #16860, #16813, #16428, #16867, #16354, #16218, #16118, #16429, #16427, #16774, #16734, #16019, #16171, #16830, #16722

📖 Documentation and testing:
- #16311, #15908, #16271, #16080, #16740, #16843

🏗 Architecture refactoring:
- All RLlib algos operating on Box action spaces now operate on normalized actions by default (ranging from -1.0 to 1.0). This enables PG-style algos to learn in skewed action spaces. (#16531)

## Tune
🎉 New Features:
- New integration with LightGBM via Tune callbacks (#16713).
- New cost-efficient HPO searchers (BlendSearch and CFO) available from the FLAML library (https://github.com/microsoft/FLAML). (#16329)

💫 Enhancements:
- Pass in configurations that have already been evaluated separately to Searchers. This is useful for warm-starting or for meta-searchers, for example (#16485)
- Sort trials in reporter table by metric (#16576)
- Add option to keep random values constant over grid search (#16501)
- Read trial results from json file (#15915)

🔨 Fixes:
- Fix infinite loop when using ``Searcher`` that limits concurrency internally in conjunction with a ``ConcurrencyLimiter`` (#16416)
- Allow custom sync configuration with ``DurableTrainable`` (#16739)
- Logger fixes. W&B: #16806, #16674, #16839. MLflow: #16840
- Various bug fixes: #16844, #16017, #16575, #16675, #16504, #15811, #15899, #16128, #16396, #16695, #16611

📖 Documentation and testing:
- Use BayesOpt for quick start example (#16997)
- #16793, #16029, #15932, #16980, #16450, #16709, #15913, #16754, #16619

## SGD
🎉 New Features:
- Torch native mixed precision is now supported! (#16382)

🔨 Fixes:
- Use target label count for training batch size (#16400)

📖 Documentation and testing:
- #15999, #16111, #16301, #16046

## Serve
💫 Enhancements: UX improvements (#16227, #15909), Improved logging (#16468)
🔨 Fixes: Fix shutdown logic (#16524), Assorted bug fixes (#16647, #16760, #16783)
📖 Documentation and testing: #16042, #16631, #16759, #16786

## Thanks 
Many thanks to all who contributed to this release:

@Tonyhao96, @simon-mo, @scv119, @Yard1, @llan-ml, @xcharleslin, @jovany-wang, @ijrsvt, @max0x7ba, @annaluo676, @rajagurunath, @zuston, @amogkam, @yorickvanzweeden, @mxz96102, @chenk008, @Bam4d, @mGalarnyk, @kfstorm, @crdnb, @suquark, @ericl, @marload, @jiaodong, @thexiang, @ellimac54, @qicosmos, @mwtian, @jkterry1, @sven1977, @howardlau1999, @mvindiola1, @stefanbschneider, @juliusfrost, @krfricke, @matthewdeng, @zhuangzhuang131419, @brandonJY, @Eleven1Liu, @nikitavemuri, @richardliaw, @iycheng, @stephanie-wang, @HuangLED, @clarkzinzow, @fyrestone, @asm582, @qingyun-wu, @ckw017, @yncxcw, @DmitriGekhtman, @benjamindkilleen, @Chong-Li, @kathryn-zhou, @pcmoritz, @rodrigodelazcano, @edoakes, @dependabot[bot], @pdames, @frenkowski, @loicsacre, @gabrieleoliaro, @achals, @thomasjpfan, @rkooo567, @dibgerge, @clay4444, @architkulkarni, @lixin-wei, @ConeyLiu, @WangTaoTheTonic, @AnnaKosiorek, @wuisawesome, @gramhagen, @zhisbug, @franklsf95, @vakker, @jenhaoyang, @liuyang-my, @chaokunyang, @SongGuyang, @tgaddair


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.5.0)

---

## ray-1.4.1: Ray-1.4.1
**Published:** 2021-06-30

# Release 1.4.1 Notes


## Ray Python Wheels

Python 3.9 wheels (Linux / MacOS / Windows) are available ([#16347](https://github.com/ray-project/ray/pull/16347) [#16586](https://github.com/ray-project/ray/pull/16586))


## Ray Autoscaler

🔨 Fixes: On-prem bug resolved ([#16281](https://github.com/ray-project/ray/pull/16281))


## Ray Client

💫Enhancements: 



*   Add warnings when many tasks scheduled ([#16454](https://github.com/ray-project/ray/pull/16454))
*   Better error messages ([#16163](https://github.com/ray-project/ray/pull/16163))

🔨 Fixes: 



*   Fix gRPC Timeout Options ([#16554](https://github.com/ray-project/ray/pull/16554))
*   Disconnect on dataclient error ([#16588](https://github.com/ray-project/ray/pull/16588))


## Ray Core

🔨 Fixes: 



*   Runtime Environments 
    *   Docs ([#16290](https://github.com/ray-project/ray/pull/16290))
    *   Bug fixes ([#16475](https://github.com/ray-project/ray/pull/16475), [#16535](https://github.com/ray-project/ray/pull/16535), [#16378](https://github.com/ray-project/ray/pull/16378))
    *   Logging improvement ([#16516](https://github.com/ray-project/ray/pull/16516))
*   Fix race condition leading to failed imports [#16278](https://github.com/ray-project/ray/pull/16278)
*   Don't broadcast empty resources data ([#16104](https://github.com/ray-project/ray/pull/16104))
*   Fix async actor lost object bug ([#16414](https://github.com/ray-project/ray/pull/16414))
*   Always report job timestamps in milliseconds ([#16455](https://github.com/ray-project/ray/pull/16455), [#16545](https://github.com/ray-project/ray/pull/16545), [#16548](https://github.com/ray-project/ray/pull/16548))
*   Multi-node placement group and job config bug fixes ([#16345](https://github.com/ray-project/ray/pull/16345))
*   Fix bug in task dependency management for duplicate args ([#16365](https://github.com/ray-project/ray/pull/16365))
*   Unify Python and core worker ids ([#16712](https://github.com/ray-project/ray/pull/16712)) 


## Dask

💫Enhancements: Dask 2021.06.1 support ([#16547](https://github.com/ray-project/ray/pull/16547))


## Tune

💫Enhancements: Support object refs in with_params ([#16753](https://github.com/ray-project/ray/pull/16753))

## Serve

🔨Fixes: Ray serve shutdown goes through Serve controller ([#16524](https://github.com/ray-project/ray/pull/16524))

## Java

🔨Fixes: Upgrade dependencies to fix CVEs ([#16650](https://github.com/ray-project/ray/pull/16650), [#16657](https://github.com/ray-project/ray/pull/16657))
 
## Documentation

* Runtime Environments ([#16290](https://github.com/ray-project/ray/pull/16290))
* Feature contribution [Tune] ([#16477](https://github.com/ray-project/ray/pull/16477))
* Ray design patterns and anti-patterns ([#16478](https://github.com/ray-project/ray/pull/16477))
* PyTorch Lightning ([#16484](https://github.com/ray-project/ray/pull/16484))
* Ray Client ([#16497](https://github.com/ray-project/ray/pull/16497))
* Ray Deployment ([#16538](https://github.com/ray-project/ray/pull/16477))
* Dask version compatibility ([#16595](https://github.com/ray-project/ray/pull/16595))
 
## CI

Move wheel and Docker image upload from Travis to Buildkite ([#16138](https://github.com/ray-project/ray/pull/16138) [#16241](https://github.com/ray-project/ray/pull/16138))
 
## Thanks
Many thanks to all those who contributed to this release!

 
@rkooo567, @clarkzinzow, @WangTaoTheTonic, @ckw017, @stephanie-wang, @Yard1, @mwtian, @jovany-wang, @jiaodong, @wuisawesome, @krfricke, @architkulkarni, @ijrsvt, @simon-mo, @DmitriGekhtman, @amogkam, @richardliaw


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.4.1)

---

## ray-1.4.0: Ray-1.4.0
**Published:** 2021-06-07

# Release 1.4.0 Notes

## Ray Autoscaler

### 🎉 New Features:
* Support Helm Chart for deploying Ray on Kubernetes
* Key Autoscaler metrics are now exported via Prometheus!

### 💫Enhancements
* Better error messages when a node fails to come online

### 🔨 Fixes:
* Stability and interface fixes for Kubernetes deployments.
* Fixes to Azure NodeProvider

## Ray Client

### 🎉 New Features:
* Complete API parity with non-client mode
* Experimental ClientBuilder API (docs here)
* Full Asyncio support

### 💫Enhancements
* Keep Alive for Messages for long lived connections
* Improved pickling error messages

### 🔨 Fixes:
* Client Disconnect can be called multiple times
* Client Reference Equality Check
* Many bug fixes and tests for the complete ray API!

## Ray Core

### 🎉 New Features:
* Namespaces ([check out the docs](https://docs.ray.io/en/master/namespaces.html))! Note: this may be a breaking change if you’re using detached actors (set ray.init(namespace=””) for backwards compatible behavior). 

### 🔨 Fixes:
* Support increment by arbitrary number with ray.util.metrics.Counter 
* Various bug fixes for the placement group APIs including the GPU assignment bug (#15049).

### 🏗 Architecture refactoring:
* Increase the efficiency and robustness of resource reporting

## Ray Data Processing

### 🔨 Fixes:
* Various bug fixes for better stability (#16063, #14821, #15669, #15757, #15431, #15426, #15034, #15071, #15070, #15008, #15955)
* Fixed a critical bug where the driver uses excessive memory usage when there are many objects in the cluster (#14322).
* Dask on Ray and Modin can now be run with Ray client

### 🏗 Architecture refactoring:
* Ray 100TB shuffle results: https://github.com/ray-project/ray/issues/15770
* More robust memory management subsystem is in progress (#15157, #15027)

## RLlib

### 🎉 New Features:
* PyTorch multi-GPU support (#14709, #15492, #15421).
* CQL TensorFlow support (#15841).
* Task-settable Env/Curriculum Learning API (#15740).
* Support for native tf.keras Models (no ModelV2 required) (#14684, #15273).
* Trainer.train() and Trainer.evaluate() can run in parallel (optional) (#15040, #15345).

### 💫Enhancements and documentation:
* CQL: Bug fixes and confirmed MuJoCo benchmarks (#15814, #15603, #15761).
* Example for differentiable neural computer (DNC) network (#14844, 15939).
* Added support for int-Box action spaces. (#15012)
* DDPG/TD3/A[23]C/MARWIL/BC: Code cleanup and type annotations. (#14707).
* Example script for restoring 1 agent out of n 
* Examples for fractional GPU usage. (15334)
* Enhanced documentation page describing example scripts and blog posts (15763).
* Various enhancements/test coverage improvements: 15499, 15454, 15335, 14865, 15525, 15290, 15611, 14801, 14903, 15735, 15631, 

### 🔨 Fixes:
* Memory Leak in multi-agent environment (#15815). Shoutout to Bam4d!
* DDPG PyTorch GPU bug. (#16133)
* Simple optimizer should not be used by default for tf+MA (#15365)
* Various bug fixes: #15762, 14843, 15042, 15427, 15871, 15132, 14840, 14386, 15014, 14737, 15015, 15733, 15737, 15736, 15898, 16118, 15020, 15218, 15451, 15538, 15610, 15326, 15295, 15762, 15436, 15558, 15937

### 🏗 Architecture refactoring:
* Remove atari dependency (#15292).
* `Trainer._evaluate()` renamed to `Trainer.evaluate()` (backward compatible); `Trainer.evaluate()` can be called even w/o evaluation worker set, if `create_env_on_driver=True` (#15591).

## Tune

### 🎉 New Features:
* ASHA scheduler now supports save/restore. (#15438)
* Add HEBO to search algorithm shim function (#15468)
* Add SkoptSearcher/Bayesopt Searcher restore functionality (#15075)

### 💫Enhancements:
* We now document scalability best practices (k8s, scalability thresholds). You can [find this here](https://docs.ray.io/en/master/tune/api_docs/scalability.html) (#14566)
* You can now set the result buffer_length via tune.run - this helps with trials that report too frequently. (#15810)
* Support numpy types in TBXlogger (#15760)
* Add `max_concurrent` option to BasicVariantGenerator (#15680)
* Add `seed` parameter to OptunaSearch (#15248)
* Improve BOHB/ConfigSpace dependency check (#15064)

### 🔨Fixes:
* Reduce default number of maximum pending trials to max(16, cluster_cpus) (#15628)
* Return normalized checkpoint path (#15296)
* Escape paths before globbing in TrainableUtil.get_checkpoints_paths (#15368)
* Optuna Searcher: Set correct Optuna TrialState on trial complete (#15283)
* Fix type annotation in tune.choice (#15038)
* Avoid system exit error by using `del` when cleaning up actors (#15687)

## Serve

### 🎉 New Features:
* As of Ray 1.4, Serve has a new API centered around the concept of “Deployments.” Deployments offer a more streamlined API and can be declaratively updated, which should improve both development and production workflows. The existing APIs have not changed from Ray 1.4 and will continue to work until Ray 1.5, at which point they will be removed (see the package reference if you’re not sure about a specific API). Please see the [migration guide](https://docs.google.com/document/d/1Tgm-bHz6au0B8F_Ps0SLPXh9oyw8pIaGWKWunnK-Kuw/edit#) for details on how to update your existing Serve application to use this new API.
* New `serve.deployment` API: `@serve.deployment, serve.get_deployments, serve.list_deployments` (#14935, #15172, #15124, #15121, #14953, #15152, #15821)
* New `serve.ingress(fastapi_app)` API (#15445, 15441, 14858)
* New `@serve.batch` decorator in favor of legacy `max_batch_size` in backend config (#15065)
* `serve.start()` is now idempotent (#15148)
* Added support for `handle.method_name.remote()` (#14831)

### 🔨Fixes:
* Rolling updates for redeployments (#14803)
* Latency improvement by using pickle (#15945)
* Controller and HTTP proxy uses `num_cpus=0` by default (#15000)
* Health checking in the controller instead of using `max_restarts` (#15047)
* Use longest prefix matching for path routing (#15041)

## Dashboard

### 🎉New Features:
* Experimental [OpenTelemetry support](https://docs.ray.io/en/master/ray-tracing.html). (#16028,#14872,#15742).

### 🔨Fixes:
* Add object store memory column (#15697)
* Add object store stats to dashboard API. (#15677)
* Remove disk data from the dashboard when running on K8s. (#14676)
* Fix reported dashboard ip when using 0.0.0.0 (#15506)

## Thanks
Many thanks to all those who contributed to this release!

@clay4444, @Fabien-Couthouis, @mGalarnyk, @smorad, @ckw017, @ericl, @antoine-galataud, @pleiadesian, @DmitriGekhtman, @robertnishihara, @Bam4d, @fyrestone, @stephanie-wang, @kfstorm, @wuisawesome, @rkooo567, @franklsf95, @micahtyong, @WangTaoTheTonic, @krfricke, @hegdeashwin, @devin-petersohn, @qicosmos, @edoakes, @llan-ml, @ijrsvt, @richardliaw, @Sertingolix, @ffbin, @simjay, @AmeerHajAli, @simon-mo, @tom-doerr, @sven1977, @clarkzinzow, @mxz96102, @SebastianBo1995, @amogkam, @iycheng, @sumanthratna, @Catch-Bull, @pcmoritz, @architkulkarni, @stefanbschneider, @tgaddair, @xcharleslin, @cthoyt, @fcardoso75, @Jeffwan, @mvindiola1, @michaelzhiluo, @rlan, @mwtian, @SongGuyang, @YeahNew, @kathryn-zhou, @rfali, @jennakwon06, @Yeachan-Heo

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.4.0)

---

## ray-1.3.0: Ray-1.3.0
**Published:** 2021-04-22

# Release v1.3.0 Notes

## Highlights
* We are now testing and publishing Ray's scalability limits with each release, see: https://github.com/ray-project/ray/tree/releases/1.3.0/benchmarks
* Ray Client is now usable by default with any Ray cluster started by the Ray Cluster Launcher. 

## Ray Cluster Launcher

### 💫Enhancements: 
* Observability improvements (#14816, #14608)
* Worker nodes no longer killed on autoscaler failure (#14424)
* Better validation for min_workers and max_workers (#13779)
* Auto detect memory resource for AWS and K8s (#14567)
* On autoscaler failure, propagate error message to drivers (#14219)
* Avoid launching GPU nodes when the workload only has CPU tasks (#13776)
* Autoscaler/GCS compatibility (#13970, #14046, #14050)
* Testing (#14488, #14713)
* Migration of configs to multi-node-type format (#13814, #14239)
* Better config validation (#14244, #13779)
* Node-type max workers defaults infinity (#14201)

### 🔨 Fixes:
* AWS configuration (#14868, #13558, #14083, #13808)
* GCP configuration (#14364, #14417)
* Azure configuration (#14787, #14750, #14721)
* Kubernetes (#14712, #13920, #13720, #14773, #13756, #14567, #13705, #14024, #14499, #14593, #14655)
* Other (#14112, #14579, #14002, #13836, #14261, #14286, #14424, #13727, #13966, #14293, #14293, #14718, #14380, #14234, #14484)

## Ray Client

### 💫Enhancements: 
* Version checks for Python and client protocol (#13722, #13846, #13886, #13926, #14295) 
* Validate server port number (#14815) 
* Enable Ray client server by default (#13350, #13429, #13442) 
* Disconnect ray upon client deactivation (#13919)
* Convert Ray objects to Ray client objects (#13639)
* Testing (#14617, #14813, #13016, #13961, #14163, #14248, #14630, #14756, #14786)
* Documentation (#14422, #14265)

### 🔨 Fixes: 

* Hook runtime context (#13750) 
* Fix mutual recursion (#14122)
* Set gRPC max message size (#14063)
* Monitor stream errors (#13386)
* Fix dependencies (#14654)
* Fix `ray.get` ctrl-c (#14425)
* Report error deserialization errors (#13749)
* Named actor refcounting fix (#14753)
* RayTaskError serialization (#14698) 
* Multithreading fixes (#14701)

## Ray Core

### 🎉 New Features:
* We are now testing and publishing Ray's scalability limits with each release. Check out https://github.com/ray-project/ray/tree/releases/1.3.0/benchmarks. 
* [alpha] Ray-native Python-based collective communication primitives for Ray clusters with distributed CPUs or GPUs. 

### 🔨 Fixes:
* Ray is now using c++14. 
* Fixed high CPU breaking raylets with heartbeat missing errors (#13963, #14301)
* Fixed high CPU issues from raylet during object transfer (#13724)
* Improvement in placement group APIs including better Java support (#13821, #13858, #13582, #15049, #13821)

## Ray Data Processing

### 🎉 New Features:
* Object spilling is turned on by default. Check out the [documentation](https://docs.ray.io/en/master/memory-management.html#object-spilling).
* Dask-on-Ray and Spark-on-Ray are fully ready to use. Please [try them out](https://docs.ray.io/en/master/raydp.html) and give us feedback!
* Dask-on-Ray is now compatible with Dask 2021.4.0.
* Dask-on-Ray now works natively with [`dask.persist()`](https://docs.dask.org/en/latest/api.html#dask.persist).

### 🔨 Fixes:
* Various improvements in object spilling and memory management layer to support large scale data processing (#13649, #14149, #13853, #13729, #14222, #13781, #13737, #14288, #14578, #15027)
* `lru_evict` flag is now deprecated. Recommended solution now is to use object spilling. 

### 🏗 Architecture refactoring:
* Various architectural improvements in object spilling and memory management. For more details, check out the [whitepaper](https://docs.google.com/document/d/1lAy0Owi-vPz2jEqBSaHNQcy2IBSDEHyXNOQZlGuj93c/edit#heading=h.61xcnifjkb6v).
* Locality-aware scheduling is turned on by default.
* Moved from centralized GCS-based object directory protocol to decentralized owner-to-owner protocol, yielding better cluster scalability.

## RLlib

### 🎉 New Features:
* R2D2 implementation for torch and tf. (#13933)
* PlacementGroup support (all RLlib algos now return PlacementGroupFactory from Trainer.default_resource_request). (#14289)
* Multi-GPU support for tf-DQN/PG/A2C. (#13393)

### 💫Enhancements:
* Documentation: Update documentation for Curiosity's support of continuous actions (#13784); CQL documentation (#14531)
* Attention-wrapper works with images and supports prev-n-actions/rewards options. (#14569)
* `rllib rollout` runs in parallel by default via Trainer’s evaluation worker set. (#14208)
* Add env rendering (customizable) and video recording options (for non-local mode; >0 workers; +evaluation-workers) and episode media logging. (#14767, #14796)
* Allow SAC to use custom models as Q- or policy nets and deprecate "state-preprocessor" for image spaces. (#13522)
* Example Scripts: Add coin game env + matrix social dilemma env + tests and examples (shoutout to Maxime Riché!). (#14208); Attention net (#14864); Serve + RLlib. (#14416); Env seed (#14471); Trajectory view API (enhancements and tf2 support). (#13786); Tune trial + checkpoint selection. (#14209)
* DDPG: Add support for simplex action space. (#14011)
* Others: `on_learn_on_batch` callback allows custom metrics. (#13584); Add `TorchPolicy.export_model()`. (#13989)

### 🔨 Fixes: 
* Trajectory View API bugs (#13646, #14765, #14037, #14036, #14031, #13555)
* Test cases (#14620, #14450, #14384, #13835, #14357, #14243)
* Others (#13013, #14569, #13733, #13556, #13988, #14737, #14838, #15272, #13681, #13764, #13519, #14038, #14033, #14034, #14308, #14243)

### 🏗 Architecture refactoring:
* Remove all non-trajectory view API code. (#14860)
* Obsolete UsageTrackingDict in favor of SampleBatch. (#13065)

## Tune

### 🎉 New Features:
* We added a new searcher `HEBOSearcher` (#14504, #14246, #13863, #14427)
* Tune is now natively compatible with the Ray Client (#13778, #14115, #14280)
* Tune now uses Ray’s Placement Groups underneath the hood. This will enable much faster autoscaling and training (for distributed trials) (#13906, #15011, #14313)

### 💫Enhancements:
* Checkpointing improvements (#13376, #13767)
* Optuna Search Algorithm improvements (#14731, #14387)
* tune.with_parameters now works with Class API (#14532)

### 🔨Fixes:
* BOHB & Hyperband fixes (#14487, #14171)
* Nested metrics improvements (#14189, #14375, #14379)
* Fix non-deterministic category sampling (#13710)
* Type hints (#13684)
* Documentation (#14468, #13880, #13740)
* Various issues and bug fixes (#14176, #13939, #14392, #13812, #14781, #14150, #14850, #14118, #14388, #14152, #13825, #13936)

## SGD
* Add fault tolerance during worker startup (#14724)

## Serve

### 🎉 New Features:
* Added metadata to default logger in backend replicas (#14251)
* Added more metrics for ServeHandle stats (#13640)
* Deprecated system-level batching in favor of @serve.batch (#14610, #14648)
* Beta support for Serve with Ray client (#14163)
* Use placement groups to bypass autoscaler throttling (#13844)
* Deprecate client-based API in favor of process-wide singleton (#14696)
* Add initial support for FastAPI ingress (#14754)

### 🔨 Fixes:
* Fix ServeHandle serialization (#13695)

### 🏗 Architecture refactoring:
* Refactor BackendState to support backend versioning and add more unit testing (#13870, #14658, #14740, #14748)
* Optimize long polling to be per-key (#14335)

## Dashboard

### 🎉 New Features:
* Dashboard now supports being served behind a reverse proxy. (#14012)
* Disk and network metrics are added to prometheus. (#14144)

### 💫Enhancements:
* Better CPU & memory information on K8s. (#14593, #14499)
* Progress towards a new scalable dashboard. (#13790, #11667, #13763,#14333)

## Thanks

Many thanks to all those who contributed to this release:
@geraint0923, @iycheng, @yurirocha15, @brian-yu, @harryge00, @ijrsvt, @wumuzi520, @suquark, @simon-mo, @clarkzinzow, @RaphaelCS, @FarzanT, @ob, @ashione, @ffbin, @robertnishihara, @SongGuyang, @zhe-thoughts, @rkooo567, @Ezra-H, @acxz, @clay4444, @QuantumMecha, @jirkafajfr, @wuisawesome, @Qstar, @guykhazma, @devin-petersohn, @jeroenboeye, @ConeyLiu, @dependabot[bot], @fyrestone, @micahtyong, @javi-redondo, @Manuscrit, @mxz96102, @EscapeReality846089495, @WangTaoTheTonic, @stanislav-chekmenev, @architkulkarni, @Yard1, @tchordia, @zhisbug, @Bam4d, @niole, @yiranwang52, @thomasjpfan, @DmitriGekhtman, @gabrieleoliaro, @jparkerholder, @kfstorm, @andrew-rosenfeld-ts, @erikerlandson, @Crissman, @raulchen, @sumanthratna, @Catch-Bull, @chaokunyang, @krfricke, @raoul-khour-ts, @sven1977, @kathryn-zhou, @AmeerHajAli, @jovany-wang, @amogkam, @antoine-galataud, @tgaddair, @randxie, @ChaceAshcraft, @ericl, @cassidylaidlaw, @TanjaBayer, @lixin-wei, @lena-kashtelyan, @cathrinS, @qicosmos, @richardliaw, @rmsander, @jCrompton, @mjschock, @pdames, @barakmich, @michaelzhiluo, @stephanie-wang, @edoakes









[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.3.0)

---

## ray-1.2.0: Release ray-1.2.0
**Published:** 2021-02-13

# Release v1.2.0 Notes

## Highlights

* Ray client is now in beta! Check out more details here: https://docs.ray.io/en/master/ray-client.html
XGBoost-Ray is now in beta! Check out more details about this project at https://github.com/ray-project/xgboost_ray.
* Check out the Serve migration guide: https://docs.google.com/document/d/1CG4y5WTTc4G_MRQGyjnb_eZ7GK3G9dUX6TNLKLnKRAc/edit
* Ray’s C++ support is now in beta: https://docs.ray.io/en/master/#getting-started-with-ray
* An alpha version of object spilling is now available: https://docs.ray.io/en/master/memory-management.html#object-spilling

## Ray Autoscaler

### 🎉 New Features:

* A new autoscaler output format in monitor.log (#12772, #13561) 
* Piping autoscaler events to driver logs (#13434)

### 💫Enhancements
* Full support of ray.autoscaler.sdk.request_resources() API (https://docs.ray.io/en/master/cluster/autoscaling.html?highlight=request_resources#ray.autoscaler.sdk.request_resources) .
* Make placement groups bypass max launch limit (#13089)
* [K8s] Retry getting home directory in command runner. (#12925)
* [docker] Pull if image is not present (#13136)
* [Autoscaler] Ensure ubuntu is owner of docker host mount folder (#13579)
### 🔨 Fixes:
* Many autoscaler bug fixes (#12952, #12689, #13058, #13671, #13637, #13588, #13505, #13154, #13151, #13138, #13008, #12980, #12918, #12829, #12714, #12661, #13567, #13663, #13623, #13437, #13498, #13472, #13392, #12514, #13325, #13161, #13129, #12987, #13410, #12942, #12868, #12866, #12865, #12098, #12609)

## RLLib
### 🎉 New Features:
* Fast Attention Nets (using the trajectory view API) (#12753).
* Attention Nets: Full PyTorch support (#12029).
* Attention Nets: Support auto-wrapping around default- or custom models by specifying “use_attention=True” in the model’s config. * * * This works completely analogously now to “use_lstm=True”. (#11698)
* New Offline RL Algorithm: CQL (based on SAC) (#13118).
* MAML: Discrete actions support (added CartPole mass test case).
* Support Atari framestacking via the trajectory view API (#13315).
* Support for D4RL environments/benchmarks (#13550).
* Preliminary work on JAX support (#13077, #13091).

### 💫 Enhancements:
* Rollout lengths: Allow unit to be configured as “agent_steps” in multi-agent settings (default: “env_steps”) (#12420).
* TFModelV2: Soft-deprecate register_variables and unify var names wrt TorchModelV2 (#13339, #13363).

### 📖 Documentation:
* Added documentation on Model building API (#13260, #13261).
* Added documentation for the trajectory view API. (#12718)
* Added documentation for SlateQ (#13266).
* Readme.md documentation for almost all algorithms in rllib/agents (#12943, #13035).
* Type annotations for the “rllib/execution” folder (#12760, #13036).

### 🔨 Fixes:
* MARWIL and BC: Add grad-clipping config option to stabilize learning (#13455).
* A3C: Solve PyTorch- and TF-eager async race condition between calling model and its value function (#13467).
* Various issues- and bug fixes (#12619, #12682, #12704, #12706, #12708, #12765, #12786, #12787, #12793, #12832, #12844, #12846, #12915, #12941, #13039, #13040, #13064, #13083, #13121, #13126, #13237, #13238, #13308, #13332, #13397, #13459, #13553).
###🏗 Architecture refactoring:
* Env directory has been cleaned up and is now divided in: Core part (rllib/env) with all basic env classes, and rllib/env/wrappers containing third-party wrapper classes (Atari, Unity3D, etc..) (#13082).

## Tune

### 🎉  New Features:

* Ray Tune has updated and improved its integration with MLflow. See [this blog post for details](https://medium.com/distributed-computing-with-ray/ray-mlflow-taking-distributed-machine-learning-applications-to-production-103f5505cb88) (#12840, #13301, #13533)

### 💫 Enhancements

* Ray Tune now uses ray.cloudpickle underneath the hood, allowing you to checkpoint large models (>4GB) (#12958).
* Using the 'reuse_actors' flag can now speed up training for general Trainable API usage.  (#13549)
* Ray Tune will now automatically buffer results from trainables, allowing you to use an arbitrary reporting frequency on your training functions. (#13236)
* Ray Tune now has a variety of experiment stoppers (#12750)
* Ray Tune now supports an integer loguniform search space distribution (#12994)
* Ray Tune now has an initial support for the Ray placement group API.  (#13370)
* The Weights and Bias integration (`WandbLogger`) now also accepts wandb.data_types.Video (#13169)
* The Hyperopt integration (`HyperoptSearch`) can now directly accept category variables instead of indices (#12715)
* Ray Tune now supports experiment checkpointing when using grid search (#13357)


### 🔨Fixes and Updates

* The Optuna integration was updated to support the 2.4.0 API while maintaining backwards compatibility (#13631)
* All search algorithms now support `points_to_evaluate` (#12790, #12916)
* PBT Transformers example was updated and improved (#13174, #13131)
* The scikit-optimize integration was improved (#12970)
* Various bug fixes (#13423, #12785, #13171, #12877, #13255, #13355)

## SGD

### 🔨Fixes and Updates

* Fix Docstring for `as_trainable` (#13173)
* Fix process group timeout units (#12477)
* Disable Elastic Training by default when using with Tune (#12927)

## Serve

### 🎉  New Features:
* Ray Serve backends now accept a Starlette request object instead of a Flask request object (#12852).  This is a breaking change, so please read the migration guide.
* Ray Serve backends now have the option of returning a Starlette Response object (#12811, #13328).  This allows for more customizable responses, including responses with custom status codes.
* [Experimental] The new Ray Serve MLflow plugin makes it easy to deploy your MLflow models on Ray Serve.  It comes with a Python API and a command-line interface.
* Using “ImportedBackend” you can now specify a backend based on a class that is installed in the Python environment that the workers will run in, even if the Python environment of the driver script (the one making the Serve API calls) doesn’t have it installed (#12923).

💫  Enhancements:
* Dependency management using conda no longer requires the driver script to be running in an activated conda environment (#13269).
* Ray ObjectRef can now be used as argument to `serve_handle.remote(...)`. (#12592)
* Backends are now shut down gracefully. You can set the graceful timeout in BackendConfig. (#13028)

📖  Documentation:
* A tutorial page has been added for integrating Ray Serve with your existing FastAPI web server or with your existing AIOHTTP web server (#13127).
* Documentation has been added for Ray Serve metrics (#13096).









[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.2.0)

---

## ray-1.1.0: ray-1.1.0
**Published:** 2020-12-24

# Ray 1.1.0
## Ray Core
### 🎉 New Features:
- Progress towards supporting a Ray client
- Descendent tasks are cancelled when the calling task is cancelled
### 🔨 Fixes:
- Improved object broadcast robustness
- Improved placement group support
### 🏗 Architecture refactoring:
- Progress towards the new scheduler backend

## RLlib
### 🎉 New Features:
- SUMO simulator integration (rllib/examples/simulators/sumo/). Huge thanks to Lara Codeca! (#11710)
- SlateQ Algorithm added for PyTorch. Huge thanks to Henry Chen! (#11450)
- MAML extension for all Models, except recurrent ones. (#11337)
- Curiosity Exploration Module for tf1.x/2.x/eager. (#11945)
- Minimal JAXModelV2 example. (#12502)
### 🔨 Fixes:
- Fix RNN learning for tf2.x/eager. (#11720)
- LSTM prev-action/prev-reward settable separately and prev-actions are now one-hot’d. (#12397)
- PyTorch LR schedule not working. (#12396)
- Various PyTorch GPU bug fixes. (#11609)
- SAC loss not using prio. replay weights in critic’s loss term. (#12394)
- Fix epsilon-greedy Exploration for nested action spaces. (#11453)
### 🏗 Architecture refactoring:
- Trajectory View API on by default (faster PG-type algos by ~20% (e.g. PPO on Atari)). (#11717, #11826, #11747, and #11827)

## Tune
### 🎉 New Features:
- Loggers can now be passed as objects to tune.run. The new ExperimentLogger abstraction was introduced for all loggers, making it much easier to configure logging behavior. (#11984, #11746, #11748, #11749)
- The tune verbosity was refactored into four levels: 0: Silent, 1: Only experiment-level logs, 2: General trial-level logs, 3: Detailed trial-level logs (default) (#11767, #12132, #12571)
- Docker and Kubernetes autoscaling environments are detected automatically, automatically utilizing the correct checkpoint/log syncing tools (#12108)
- Trainables can now easily leverage Tensorflow DistributedStrategy! (#11876)

### 💫 Enhancements
- Introduced a new serialization debugging utility (#12142)
- Added a new lightweight Pytorch-lightning example (#11497, #11585)
- The BOHB search algorithm can be seeded with a random state (#12160)
- The default anonymous metrics can be used automatically if a `mode` is set in tune.run (#12159).
- Added HDFS as Cloud Sync Client (#11524)
- Added xgboost_ray integration (#12572)
- Tune search spaces can now be passed to search algorithms on initialization, not only via tune.run (#11503)
- Refactored and added examples (#11931)
- Callable accepted for register_env (#12618)
- Tune search algorithms can handle/ignore infinite and NaN numbers (#11835)
- Improved scalability for experiment checkpointing (#12064)
- Nevergrad now supports points_to_evaluate (#12207)
- Placement group support for distributed training (#11934)

### 🔨 Fixes:
- Fixed with_parameters behavior to avoid serializing large data in scope (#12522)
- TBX logger supports None (#12262)
- Better error when `metric` or `mode` unset in search algorithms (#11646)
- Better warnings/exceptions for fail_fast='raise' (#11842)
- Removed some bottlenecks in trialrunner (#12476)
- Fix file descriptor leak by syncer  and Tensorboard (#12590, #12425)
- Fixed validation for search metrics (#11583)
- Fixed hyperopt randint limits (#11946)

## Serve
### 🎉 New Features:
- You can start backends in different conda environments! See more in the [dependency management doc](https://docs.ray.io/en/master/serve/advanced.html#dependency-management). (#11743)
- You can add a optional `reconfigure` method to your Servable to allow [reconfiguring](https://docs.ray.io/en/master/serve/advanced.html#reconfiguring-backends-experimental) backend replicas at runtime. (#11709)
### 🔨Fixes:
- Set serve.start(http_host=None) to disable HTTP servers. If you are only using ServeHandle, this option lowers resource usage. (#11627)
- Flask requests will no longer create reference cycles. This means peak memory usage should be lower for high traffic scenarios. (#12560)
### 🏗 Architecture refactoring:
- Progress towards a goal state driven Serve controller. (#12369,#11792,#12211,#12275,#11533,#11822,#11579,#12281)
- Progress towards faster and more efficient ServeHandles. (#11905, #12019, #12093)

## Ray Cluster Launcher (Autoscaler)
### 🎉 New Features:
- A new Kubernetes operator: https://docs.ray.io/en/master/cluster/k8s-operator.html
### 💫 Enhancements
- Containers do not run with root user as the default (#11407)
- SHM-Size is auto-populated when using the containers (#11953)
### 🔨 Fixes:
- Many autoscaler bug fixes (#11677, #12222, #11458, #11896, #12123, #11820, #12513, #11714, #12512, #11758, #11615, #12106, #11961, #11674, #12028, #12020, #12316, #11802, #12131, #11543, #11517, #11777, #11810, #11751, #12465, #11422)

## SGD
### 🎉 New Features:
- Easily customize your torch.DistributedDataParallel configurations by passing in a `ddp_args` field into `TrainingOperator.register` (#11771).
### 🔨 Fixes:
- `TorchTrainer` now properly scales up to more workers if more resources become available (#12562)
### 📖 Documentation:
- The new callback API for using Ray SGD with Tune is now documented (#11479)
- Pytorch Lightning + Ray SGD integration is now documented (#12440)

## Dashboard
### 🔨 Fixes:
- Fixed bug that prevented viewing the logs for cluster workers
- Fixed bug that caused "Logical View" page to crash when opening a list of actors for a given class.
### 🏗 Architecture refactoring:
- Dashboard runs on a new backend architecture that is more scalable and well-tested. The dashboard should work on ~100 node clusters now, and we're working on lifting scalability to constraints to support even larger clusters.

## Thanks
Many thanks to all those who contributed to this release:
@bartbroere, @SongGuyang, @gramhagen, @richardliaw, @ConeyLiu, @weepingwillowben, @zhongchun, @ericl, @dHannasch, @timurlenk07, @kaushikb11, @krfricke, @desktable, @bcahlit, @rkooo567, @amogkam, @micahtyong, @edoakes, @stephanie-wang, @clay4444, @ffbin, @mfitton, @barakmich, @pcmoritz, @AmeerHajAli, @DmitriGekhtman, @iamhatesz, @raulchen, @ingambe, @allenyin55, @sven1977, @huyz-git, @yutaizhou, @suquark, @ashione, @simon-mo, @raoul-khour-ts, @Leemoonsoo, @maximsmol, @alanwguo, @kishansagathiya, @wuisawesome, @acxz, @gabrieleoliaro, @clarkzinzow, @jparkerholder, @kingsleykuan, @InnovativeInventor, @ijrsvt, @lasagnaphil, @lcodeca, @jiajiexiao, @heng2j, @wumuzi520, @mvindiola1, @aaronhmiller, @robertnishihara, @WangTaoTheTonic, @chaokunyang, @nikitavemuri, @kfstorm, @roireshef, @fyrestone, @viotemp1, @yncxcw, @karstenddwx, @hartikainen, @sumanthratna, @architkulkarni, @michaelzhiluo, @UWFrankGu, @oliverhu, @danuo, @lixin-wei


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.1.0)

---

## ray-1.0.1.post1: ray-1.0.1.post1
**Published:** 2020-11-19

Patch release containing the following changes:
- https://github.com/ray-project/ray/commit/bcc92f59fdcd837ccc5a560fe37bdf0619075505 Fix dashboard crashing on multi-node clusters.
- https://github.com/ray-project/ray/pull/11600 Add the cluster_name to docker file mounts directory prefix.

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.0.1.post1)

---

## ray-1.0.1: ray-1.0.1
**Published:** 2020-11-10

# Ray 1.0.1

Ray 1.0.1 is now officially released!

## Highlights

* If you're migrating from Ray < 1.0.0, be sure to check out the [1.0 Migration Guide](https://github.com/ray-project/ray/discussions/11482).
* Autoscaler is now **docker by default**.
* RLLib features multiple new environments.
* Tune supports population based bandits, checkpointing in Docker, and multiple usability improvements. 
* SGD supports PyTorch Lightning 
* All of Ray's components and libraries have improved performance, scalability, and stability.

## Core
* [1.0 Migration Guide](https://github.com/ray-project/ray/discussions/11482).
* Many bug fixes and optimizations in GCS.
* Polishing of the Placement Group API. 
* Improved Java language support

## RLlib 
* Added documentation for Curiosity exploration module (#11066).
* Added RecSym environment wrapper (#11205).
* Added Kaggle’s football environment (multi-agent) wrapper (#11249).
* Multiple bug fixes: GPU related fixes for SAC (#11298), MARWIL, all example scripts run on GPU (#11105), lifted limitation on 2^31 timesteps (#11301), fixed eval workers for ES and ARS (#11308), fixed broken no-eager-no-workers mode (#10745).
* Support custom MultiAction distributions (#11311).
* No environment is created on driver (local worker) if not necessary (#11307).
* Added simple SampleCollector class for Trajectory View API (#11056).
* Code cleanup: Docstrings and type annotations for Exploration classes (#11251), DQN (#10710), MB-MPO algorithm, SAC algorithm (#10825).


## Serve
* API: Serve will error when `serve_client` is serialized. (#11181)
* Performance: `serve_client.get_handle("endpoint")` will now get a handle to nearest node, increasing scalability in distributed mode. (#11477)
* Doc: Added FAQ page and updated architecture page (#10754, #11258)
* Testing: New distributed tests and benchmarks are added (#11386)
* Testing: Serve now run on Windows (#10682)

## SGD
* Pytorch Lightning integration is now supported (#11042)
* Support `num_steps` continue training (#11142)
* Callback API for SGD+Tune  (#11316)

## Tune
* New Algorithm: Population-based Bandits (#11466)
* `tune.with_parameters()`, a wrapper function to pass arbitrary objects through the object store to trainables (#11504)
* Strict metric checking - by default, Tune will now error if a result dict does not include the optimization metric as a key. You can disable this with TUNE_DISABLE_STRICT_METRIC_CHECKING (#10972)
* Syncing checkpoints between multiple Docker containers on a cluster is now supported with the `DockerSyncer`  (#11035)
* Added type hints (#10806)
* Trials are now dynamically created (instead of created up front) (#10802)
* Use `tune.is_session_enabled()` in the Function API to toggle between Tune and non-tune code  (#10840)
* Support hierarchical search spaces for hyperopt (#11431)
* Tune function API now also supports `yield` and `return` statements (#10857)
* Tune now supports callbacks with `tune.run(callbacks=...` (#11001)
* By default, the experiment directory will be dated (#11104)
* Tune now supports `reuse_actors` for function API, which can largely accelerate tuning jobs.

## Thanks

We thank all the contributors for their contribution to this release!

@acxz, @Gekho457, @allenyin55, @AnesBenmerzoug, @michaelzhiluo, @SongGuyang, @maximsmol, @WangTaoTheTonic, @Basasuya, @sumanthratna, @juliusfrost, @maxco2, @Xuxue1, @jparkerholder, @AmeerHajAli, @raulchen, @justinkterry, @herve-alanaai, @richardliaw, @raoul-khour-ts, @C-K-Loan, @mattearllongshot, @robertnishihara, @internetcoffeephone, @Servon-Lee, @clay4444, @fangyeqing, @krfricke, @ffbin, @akotlar, @rkooo567, @chaokunyang, @PidgeyBE, @kfstorm, @barakmich, @amogkam, @edoakes, @ashione, @jseppanen, @ttumiel, @desktable, @pcmoritz, @ingambe, @ConeyLiu, @wuisawesome, @fyrestone, @oliverhu, @ericl, @weepingwillowben, @rkube, @alanwguo, @architkulkarni, @lasagnaphil, @rohitrawat, @ThomasLecat, @stephanie-wang, @suquark, @ijrsvt, @VishDev12, @Leemoonsoo, @scottwedge, @sven1977, @yiranwang52, @carlos-aguayo, @mvindiola1, @zhongchun, @mfitton, @simon-mo


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.0.1)

---

## ray-1.0.0: ray-1.0.0
**Published:** 2020-09-30

# Ray 1.0

We're happy to announce the release of Ray 1.0, an important step towards the goal of providing a universal API for distributed computing.

To learn more about Ray 1.0, check out our [blog post](https://www.anyscale.com/blog/announcing-ray-1-0) and [whitepaper](https://docs.ray.io/en/master/whitepaper.html).

## Ray Core
- The ray.init() and `ray start` commands have been cleaned up to remove deprecated arguments
- The Ray Java API is now stable
- Improved detection of Docker CPU limits 
- Add support and documentation for Dask-on-Ray and MARS-on-Ray: https://docs.ray.io/en/master/ray-libraries.html
- Placement groups for fine-grained control over scheduling decisions: https://docs.ray.io/en/latest/placement-group.html.
- New architecture whitepaper: https://docs.ray.io/en/master/whitepaper.html

### Autoscaler
- Support for multiple instance types in the same cluster: https://docs.ray.io/en/master/cluster/autoscaling.html
- Support for specifying GPU/accelerator type in `@ray.remote`

### Dashboard & Metrics
- Improvements to the memory usage tab and machine view
- The dashboard now supports visualization of actor states
- Support for Prometheus metrics reporting: https://docs.ray.io/en/latest/ray-metrics.html

## RLlib
- Two Model-based RL algorithms were added: MB-MPO (“Model-based meta-policy optimization”) and “Dreamer”. Both algos were benchmarked and are performing comparably to the respective papers’ reported results.
- A “Curiosity” (intrinsic motivation) module was added via RLlib’s Exploration API and benchmarked on a sparse-reward Unity3D environment (Pyramids).
- Added documentation for the Distributed Execution API.
- Removed (already soft-deprecated) APIs: Model(V1) class, Trainer config keys, some methods/functions. Where you would see a warning previously when using these, there will be an error thrown now.
- Added DeepMind Control Suite examples.

## Tune

**Breaking changes:**
-  Multiple tune.run parameters have been deprecated: `ray_auto_init, run_errored_only, global_checkpoint_period, with_server` (#10518)
- `tune.run(upload_dir, sync_to_cloud, sync_to_driver, sync_on_checkpoint` have been moved to `tune.SyncConfig` [[docs](https://docs.ray.io/en/releases-1.0.0/tune/tutorials/tune-distributed.html#syncing)] (#10518)

**New APIs:**
- `mode, metric, time_budget` parameters for tune.run (#10627, #10642)
- Search Algorithms now share a uniform API: (#10621, #10444). You can also use the new `create_scheduler/create_searcher` shim layer to create search algorithms/schedulers via string, reducing boilerplate code (#10456).
- Native callbacks for: [MXNet, Horovod, Keras, XGBoost, PytorchLightning](https://docs.ray.io/en/releases-1.0.0/tune/api_docs/integration.html) (#10533, #10304, #10509, #10502, #10220)
- PBT runs can be replayed with PopulationBasedTrainingReplay scheduler (#9953)
- Search Algorithms are saved/resumed automatically (#9972)
- New Optuna Search Algorithm [docs](https://docs.ray.io/en/releases-1.0.0/tune/api_docs/suggestion.html#optuna-tune-suggest-optuna-optunasearch) (#10044)
- Tune now can sync checkpoints across Kubernetes pods (#10097)
- Failed trials can be rerun with `tune.run(resume="run_errored_only")` (#10060)

**Other Changes:**
- Trial outputs can be saved to file via `tune.run(log_to_file=...)` (#9817)
- Trial directories can be customized, and default trial directory now includes trial name (#10608, #10214)
- Improved Experiment Analysis API (#10645)
- Support for Multi-objective search via SigOpt Wrapper (#10457, #10446)
- BOHB Fixes (#10531, #10320)
- Wandb improvements + RLlib compatibility (#10950, #10799, #10680, #10654, #10614, #10441, #10252, #8521)
- Updated documentation for FAQ, Tune+serve, search space API, lifecycle (#10813, #10925, #10662, #10576, #9713, #10222, #10126, #9908)


## RaySGD: 
* Creator functions are subsumed by the TrainingOperator API (#10321)
* Training happens on actors by default (#10539)

## Serve

- [`serve.client` API](https://docs.ray.io/en/master/serve/deployment.html#lifetime-of-a-ray-serve-instance) makes it easy to appropriately manage lifetime for multiple Serve clusters. (#10460) 
- Serve APIs are fully typed. (#10205, #10288)
- Backend configs are now typed and validated via Pydantic. (#10559, #10389)
- Progress towards application level backend autoscaler. (#9955, #9845, #9828)
- New [architecture page](https://docs.ray.io/en/master/serve/architecture.html) in documentation. (#10204)

## Thanks
We thank all the contributors for their contribution to this release!

@MissiontoMars, @ijrsvt, @desktable, @kfstorm, @lixin-wei, @Yard1, @chaokunyang, @justinkterry, @pxc, @ericl, @WangTaoTheTonic, @carlos-aguayo, @sven1977, @gabrieleoliaro, @alanwguo, @aryairani, @kishansagathiya, @barakmich, @rkube, @SongGuyang, @qicosmos, @ffbin, @PidgeyBE, @sumanthratna, @yushan111, @juliusfrost, @edoakes, @mehrdadn, @Basasuya, @icaropires, @michaelzhiluo, @fyrestone, @robertnishihara, @yncxcw, @oliverhu, @yiranwang52, @ChuaCheowHuan, @raphaelavalos, @suquark, @krfricke, @pcmoritz, @stephanie-wang, @hekaisheng, @zhijunfu, @Vysybyl, @wuisawesome, @sanderland, @richardliaw, @simon-mo, @janblumenkamp, @zhuohan123, @AmeerHajAli, @iamhatesz, @mfitton, @noahshpak, @maximsmol, @weepingwillowben, @raulchen, @09wakharet, @ashione, @henktillman, @architkulkarni, @rkooo567, @zhe-thoughts, @amogkam, @kisuke95, @clarkzinzow, @holli, @raoul-khour-ts

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-1.0.0)

---

## ray-0.8.7: ray-0.8.7
**Published:** 2020-08-13

Highlight
---------
- Ray is moving towards 1.0! It has had several important naming changes.
    - `ObjectID`s are now called `ObjectRef`s because they are not just IDs. 
    - The Ray Autoscaler is now called the Ray Cluster Launcher. The autoscaler will be a module of the Ray Cluster Launcher.
- The Ray Cluster Launcher now has a much cleaner and concise output style. Try it out with `ray up --log-new-style`. The new output style will be enabled by default (with opt-out) in a later release.
- Windows is now officially supported by RLlib. Multi node support for Windows is still in progress.

Cluster Launcher/CLI (formerly autoscaler)
--------------------------------------------
- **Highlight:** This release contains a new colorful, concise output style for `ray up` and `ray down`, available with the `--log-new-style` flag. It will be enabled by default (with opt-out) in a later release. Full output style coverage for Cluster Launcher commands will also be available in a later release. (#9322, #9943, #9960, #9690)
- Documentation improvements (with guides and new sections) (#9687
- Improved Cluster launcher docker support (#9001, #9105, #8840)
- Ray now has Docker images available on Docker hub. Please check out the [ray image](https://hub.docker.com/u/rayproject/ray) (#9732, #9556, #9458, #9281)
- Azure improvements (#8938)
- Improved on-prem cluster autoscaler (#9663)
- Add option for continuous sync of file mounts (#9544)
- Add `ray status` debug tool and `ray --version` (#9091, #8886).
- `ray memory` now also supports redis_password (#9492)
- Bug fixes for the Kubernetes cluster launcher mode (#9968)
- __Various improvements:__ disabling the cluster config cache (#8117), Python API requires keyword arguments (#9256), removed fingerprint checking for SSH (#9133), Initial support for multiple worker types (#9096), various changes to the internal node provider interface (#9340, #9443)

Core
-----
- Support Python type checking for Ray tasks (#9574)
- Rename ObjectID => ObjectRef (#9353)
- New GCS Actor manager on by default (#8845, #9883, #9715, #9473, #9275)
- Worker towards placement groups (#9039) 
- Plasma store process is merged with raylet (#8939, #8897)
- Option to automatically reconstruct objects stored in plasma after a failure. See the [documentation](https://docs.ray.io/en/master/fault-tolerance.html#objects) for more information. (#9394, #9557, #9488)
- Many bug fixes.

RLlib
-----
- New algorithm: __“Model-Agnostic Meta-Learning” (MAML)__. An algo that learns and generalizes well across a __distribution__ of environments.
- New algorithm: __“Model-Based Meta-Policy-Optimization” (MB-MPO)__. Our first __model-based RL algo__.
- __Windows__ is now __officially supported__ by RLlib.
- __Native TensorFlow 2.x support__. Use framework=”tf2” in your config to tap into TF2’s full potential. Also: SAC, DDPG, DQN Rainbow, ES, and ARS now run in TF1.x Eager mode.
- __DQN PyTorch__ support for full Rainbow setup (including distributional DQN).
- __Python type hints__ for Policy, Model, Offline, Evaluation, and Env classes.
- __Deprecated “Policy Optimizer”__ package (in favor of new distributed execution API).
- Enhanced __test coverage__ and __stability__.
- __Flexible multi-agent replay modes__ and `replay_sequence_length`. We now allow a) storing sequences (over time) in replay buffers and retrieving “lock-stepped” multi-agent samples.
- Environments: __Unity3D soccer game__ (tuned example/benchmark) and __DM Control__ Suite wrapper and examples.
- Various __Bug fixes__: QMIX not learning, DDPG torch bugs, IMPALA learning rate updates, PyTorch custom loss, PPO not learning MuJoCo due to action clipping bug, DQN w/o dueling layer error.

Tune
-----

- **API Changes**:
    - The Tune Function API now supports checkpointing and is now usable with all search and scheduling algorithms! (#8471, #9853, #9517)
    - The Trainable class API has renamed many of its methods to be public (#9184)
- You can now stop experiments upon convergence with Bayesian Optimization (#8808)
- `DistributedTrainableCreator`, a simple wrapper for distributed parameter tuning with multi-node DistributedDataParallel models (#9550, #9739)
- New integration and tutorial for using Ray Tune with __Weights and Biases__ (Logger and native API) (#9725)
- Tune now provides a Scikit-learn compatible wrapper for hyperparameter tuning (#9129)
- __New tutorials__ for integrations like __XGBoost__ (#9060), __multi GPU PyTorch__ (#9338), __PyTorch Lightning__ (#9151, #9451), and __Huggingface-Transformers__ (#9789)
- CLI Progress reporting improvements (#8802, #9537, #9525)
- Various __bug fixes__: handling of NaN values (#9381), Tensorboard logging improvements (#9297, #9691, #8918), enhanced cross-platform compatibility (#9141), re-structured testing (#9609), documentation reorganization and versioning (#9600, #9427, #9448)

RaySGD
--------
- Variable worker CPU requirements (#8963)
- Simplified cuda visible device setting (#8775)

Serve
------
- Horizontal scalability: Serve will now start one HTTP server per Ray node. (#9523)
- Various performance improvement matching Serve to FastAPI (#9490,#8709, #9531, #9479 ,#9225, #9216, #9485)
- API changes
    - `serve.shadow_traffic(endpoint, backend, fraction)` duplicates and sends a fraction of the incoming traffic to a specific backend. (#9106)
    - `serve.shutdown()` cleanup the current Serve instance in Ray cluster. (#8766)
    - Exception will be raised if `num_replicas` exceeds the maximum resource in the cluster (#9005)
- Added doc examples for how to perform metric [monitoring](https://docs.ray.io/en/master/serve/advanced.html#monitoring) and [model composition](https://docs.ray.io/en/master/serve/advanced.html#composing-multiple-models).

Dashboard
-----------
- __Configurable Dashboard Port__: The port on which the dashboard will run is now configurable using the argument `--dashboard-port` and the argument `dashboard_port` to `ray.init`
- __GPU monitoring improvements__
  - For machines with more than one GPU, the GPU and GRAM utilization is now broken out on a per-GPU basis. 
  - Assignments to physical GPUs are now shown at the worker level.
- __Sortable Machine View__: It is now possible to sort the machine view by almost any of its columns by clicking next to the title. In addition, whereas the workers are normally grouped by node, you can now ungroup them if you only want to see details about workers.
- __Actor Search Bar__: It is possible to search for actors by their title now (this is the class name of the actor in python in addition to the arguments it received.)
- __Logical View UI Updates__: This includes things like color-coded names for each of the actor states, a more grid-like layout, and tooltips for the various data.
- __Sortable Memory View__: Like the machine view, the memory view now has sortable columns and can be grouped / ungrouped by node.

Windows Support
------------------
- Improve GPU detection (#9300)
- Work around msgpack issue on PowerPC64LE (#9140)

Others
-------
- Ray Streaming Library Improvements (#9240, #8910, #8780)
- Java Support Improvements (#9371, #9033, #9037, #9032, #8858, #9777, #9836, #9377)
- Parallel Iterator Improvements (#8964, #8978)


Thanks
-------
We thank the following contributors for their work on this release:
@jsuarez5341, @amitsadaphule, @krfricke, @williamFalcon, @richardliaw, @heyitsmui, @mehrdadn, @robertnishihara, @gabrieleoliaro, @amogkam, @fyrestone, @mimoralea, @edoakes, @andrijazz, @ElektroChan89, @kisuke95, @justinkterry, @SongGuyang, @barakmich, @bloodymeli, @simon-mo, @TomVeniat, @lixin-wei, @alanwguo, @zhuohan123, @michaelzhiluo, @ijrsvt, @pcmoritz, @LecJackS, @sven1977, @ashione, @JerryLeeCS, @raphaelavalos, @stephanie-wang, @ruifangChen, @vnlitvinov, @yncxcw, @weepingwillowben, @goulou, @acmore, @wuisawesome, @gramhagen, @anabranch, @internetcoffeephone, @Alisahhh, @henktillman, @deanwampler, @p-christ, @Nicolaus93, @WangTaoTheTonic, @allenyin55, @kfstorm, @rkooo567, @ConeyLiu, @09wakharet, @piojanu, @mfitton, @KristianHolsheimer, @AmeerHajAli, @pdames, @ericl, @VishDev12, @suquark, @stefanbschneider, @raulchen, @dcfidalgo, @chappers, @aaarne, @chaokunyang, @sumanthratna, @clarkzinzow, @BalaBalaYi, @maximsmol, @zhongchun, @wumuzi520, @ffbin

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.8.7)

---

## ray-0.8.6: ray-0.8.6
**Published:** 2020-06-24

Highlight
---------
- Experimental support for Windows is now available for single node Ray usage. Check out the Windows section below for known issues and other details.
- Have you had troubles monitoring GPU or memory usage while you used Ray? The Ray dashboard now supports the GPU monitoring and a memory view.
- Want to use RLlib with Unity? RLlib officially supports the Unity3D adapter! Please check out the [documentation](https://docs.ray.io/en/master/rllib-env.html?highlight=unity#external-agents-and-applications).
- Ray Serve is ready for feedback! We've gotten feedback from many users, and Ray Serve is already being used in production. Please reach out to us with your use cases, ideas, documentation improvements, and feedback. We'd love to hear from you. Please do so on the Ray Slack and join #serve! Please see the Serve section below for more details.

Core
-----
- We’ve introduced a new feature to automatically retry failed actor tasks after an actor has been restarted by Ray (by specifying `max_restarts` in `@ray.remote`). Try it out with `max_task_retries=-1` where -1 indicates that the system can retry the task until it succeeds.

#### API Change
- To enable automatic restarts of a failed actor, you must now use `max_restarts` in the `@ray.remote` decorator instead of `max_reconstructions`. You can use -1 to indicate infinity, i.e., the system should always restart the actor if it fails unexpectedly.
- We’ve merged the named and detached actor APIs. To create an actor that will survive past the duration of its job (a “detached” actor), specify `name=<str>` in its remote constructor (`Actor.options(name='<str>').remote()`). To delete the actor, you can use `ray.kill`.

RLlib
-----
- PyTorch: IMPALA PyTorch version and all `rllib/examples` scripts now work for either TensorFlow or PyTorch (`--torch` command line option).
- Switched to using distributed execution API by default (replaces Policy Optimizers) for all algorithms.
- Unity3D adapter (supports all Env types: multi-agent, external env, vectorized) with example scripts for running locally or in the cloud.
- Added support for variable length observation Spaces ("Repeated").
- Added support for arbitrarily nested action spaces.
- Added experimental GTrXL (Transformer/Attention net) support to RLlib + learning tests for PPO and IMPALA.
- QMIX now supports complex observation spaces.

#### API Change
- Retire `use_pytorch` and `eager` flags in configs and replace these with `framework=[tf|tfe|torch]`.
- Deprecate PolicyOptimizers in favor of the new distributed execution API.
- Retired support for Model(V1) class. Custom Models should now only use the ModelV2 API. There is still a warning when using ModelV1, which will be changed into an error message in the next release.
- Retired TupleActions (in favor of arbitrarily nested action Spaces).

Ray Tune / RaySGD
-------------------
- There is now a Dataset API for handling large datasets with RaySGD. (#7839)
- You can now filter by an average of the last results using the `ExperimentAnalysis` tool (#8445).
- BayesOptSearch received numerous contributions, enabling preliminary random search and warm starting. (#8541, #8486, #8488)

#### API Changes
- `tune.report` is now the right way to use the Tune function API. `tune.track` is deprecated (#8388)

Serve
------
- New APIs to inspect and manage Serve objects:
	- `serve.list_backends` and `serve.list_endpoints` (#8737)
	- `serve.delete_backend` and `serve.delete_endpoint` (#8252, #8256)
- `serve.create_endpoint` now requires specifying the backend directly. You can remove `serve.set_traffic` if there's only one backend per endpoint. (#8764)
- `serve.init` API cleanup, the following options were removed:
	- `blocking`, `ray_init_kwargs`, `start_server` (#8747, #8447, #8620)
- `serve.init` now supports namespacing with `name`. You can run multiple serve clusters with different names on the same ray cluster. (#8449)
- You can specify session affinity when splitting traffic with backends using `X-SERVE-SHARD-KEY` HTTP header. (#8449)
- Various documentation improvements. Highlights:
	- A new section on how to perform A/B testing and incremental rollout (#8741)
	- Tutorial for batch inference (#8490)
	- Instructions for specifying GPUs and resources (#8495)

Dashboard / Metrics
---------------------
- The Machine View of the dashboard now shows information about GPU utilization such as:
    - Average GPU/GRAM utilization at a node and cluster level
    - Worker-level information about how many GPUs each worker is assigned as well as its GRAM use.
- The dashboard has a new Memory View tab that should be very useful for debugging memory issues. It has:
    - Information about objects in the Ray object store, including size and call-site
    - Information about reference counts and what is keeping an object pinned in the Ray object store.

#### Small changes
- IDLE workers get automatically sorted to the end of the worker list in the Machine View

Autoscaler
-----------
- Improved logging output. Errors are more clearly propagated and excess output has been reduced. (#7198, #8751, #8753)
- Added support for k8s services. 

#### API Changes
- `ray up` accepts remote URLs that point to the desired cluster YAML. (#8279)

Windows support
------------------
- Windows wheels are now available for basic experimental usage (via `ray.init()`).
- Windows support is currently unstable. Unusual, unattended, or production usage is *not* recommended.
- Various functionality may still lack support, including Ray Serve, Ray SGD, the autoscaler, the dashboard, non-ASCII file paths, etc.
- Please check the latest nightly wheels & known issues (#9114), and let us know if any issue you encounter has not yet been addressed.
- Wheels are available for Python 3.6, 3.7, and 3.8. (#8369)
- redis-py has been patched for Windows sockets. (#8386)

Others
-------
- Moving towards highly available Ray (#8650, #8639, #8606, #8601, #8591, #8442)
- Java Support (#8730, #8640, #8637)
- Ray streaming improvements (#8612, #8594, #7464)
- Parallel iterator improvements (#8140, #7931, #8712)

Thanks
------
We thank the following contributors for their work on this release:
@pcmoritz, @akharitonov, @devanderhoff, @ffbin, @anabranch, @jasonjmcghee, @kfstorm, @mfitton, @alecbrick, @simon-mo, @konichuvak, @aniryou, @wuisawesome, @robertnishihara, @ramanNarasimhan77, @09wakharet, @richardliaw, @istoica, @ThomasLecat, @sven1977, @ceteri, @acxz, @iamhatesz, @JarnoRFB, @rkooo567, @mehrdadn, @thomasdesr, @janblumenkamp, @ujvl, @edoakes, @maximsmol, @krfricke, @amogkam, @gehring, @ijrsvt, @internetcoffeephone, @LucaCappelletti94, @chaokunyang, @WangTaoTheTonic, @fyrestone, @raulchen, @ConeyLiu, @stephanie-wang, @suquark, @ashione, @Coac, @JosephTLucas, @ericl, @AmeerHajAli, @pdames

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.8.6)

---

## ray-0.8.5: Ray 0.8.5
**Published:** 2020-05-07

Highlight
---------
- You can now cancel remote tasks using the `ray.cancel` API. 
- PyTorch is now a first-class citizen in RLlib! We've achieved parity between TensorFlow and PyTorch.
- Did you struggle to find good example code for Ray ML libraries? We wrote more examples for Ray SGD and Ray Serve.
   - Ray serve: [Keras/Tensorflow](https://docs.ray.io/en/master/rayserve/tutorials/tensorflow-tutorial.html), [PyTorch](https://docs.ray.io/en/master/rayserve/tutorials/pytorch-tutorial.html), [Scikit-Learn](https://docs.ray.io/en/master/rayserve/tutorials/sklearn-tutorial.html).
   - Ray SGD: New [Semantic Segmentation](https://github.com/ray-project/ray/tree/master/python/ray/util/sgd/torch/examples/segmentation) and [HuggingFace GLUE Fine-tuning](https://github.com/ray-project/ray/tree/master/python/ray/util/sgd/torch/examples/transformers) Examples. 

Core
-----
- Task cancellation is now available for locally submitted tasks. (#7699)
- Experimental support for recovering objects that were lost from the Ray distributed memory store. You can try this out by setting `lineage_pinning_enabled: 1` in the internal config. (#7733)

RLlib
-----
- PyTorch support has now reached parity with TensorFlow. (#7926, #8188, #8120, #8101, #8106, #8104, #8082, #7953, #7984, #7836, #7597, #7797)
- Improved callbacks API. (#6972)
- Enable Ray distributed reference counting. (#8037)
- Work towards customizable distributed training workflows. (#7958, #8077)

Tune
-----
- Documentation has improved with a new format. (#8083, #8201, #7716)
- Search algorithms are refactored to make them easier to extend, deprecating `max_concurrent` argument. (#7037, #8258, #8285)
- TensorboardX errors are now handled safely. (#8174)
- Bug fix in PBT checkpointing. (#7794)
- New ZOOpt search algorithm added. (#7960)

Serve
------
- Improved APIs.
    - Add delete_endpoint and delete_backend. (#8252, #8256)
    - Use dictionary to update backend config. (#8202)
- Added overview section to the documentation.
- Added tutorials for serving models in Tensorflow/Keras, PyTorch, and Scikit-Learn.
- Made serve clusters tolerant to process failures. (#8116, #8008,#7970,#7936)

SGD
-----
- New Semantic Segmentation and HuggingFace GLUE Fine-tuning Examples. (#7792, #7825)
- Fix GPU Reservations in SLURM usage. (#8157)
- Update learning rate scheduler stepping parameter. (#8107)
- Make serialization of data creation optional. (#8027)
- Automatic DDP wrapping is now optional. (#7875)

Others Projects
----------------
- Progress towards the highly available and fault tolerant control plane. (#8144, #8119, #8145, #7909, #7949, #7771, #7557, #7675) 
- Progress towards the Ray streaming library. (#8044, #7827, #7955, #7961, #7348)
- Autoscaler improvement. (#8178, #8168, #7986, #7844, #7717)
- Progress towards Java support. (#8014)
- Progress towards the Window compatibility. (#8237, #8186)
- Progress towards cross language support. (#7711)


Thanks
------
We thank the following contributors for their work on this release:

@simon-mo, @robertnishihara, @BalaBalaYi, @ericl, @kfstorm, @tirkarthi, @nflu, @ffbin, @chaokunyang, @ijrsvt, @pcmoritz, @mehrdadn, @sven1977, @iamhatesz, @nmatthews-asapp, @mitchellstern, @edoakes, @anabranch, @billowkiller, @eisber, @ujvl, @allenyin55, @yncxcw, @deanwampler, @DavidMChan, @ConeyLiu, @micafan, @rkooo567, @datayjz, @wizardfishball, @sumanthratna, @ashione, @marload, @stephanie-wang, @richardliaw, @jovany-wang, @MissiontoMars, @aannadi, @fyrestone, @JarnoRFB, @wumuzi520, @roireshef, @acxz, @gramhagen, @Servon-Lee, @ClarkZinzow, @mfitton, @maximsmol, @janblumenkamp, @istoica

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.8.5)

---

## ray-0.8.4: Ray 0.8.4
**Published:** 2020-04-02

Highlight
----------
- Add Python 3.8 support. (#7754)

Core
----
- Fix asycnio actor deserialization. (#7806)
- Fix importing Pyarrow lead to symbol collison segfault. (#7568)
- `ray memory` will collect statistics from all nodes. (#7721)
- Pin lineage of plasma objects that are still in scope. (#7690)

RLlib
-----
- Add contextual bandit algorithms. (#7642)
- Add parameter noise exploration API. (#7772)
- Add [scaling guide](https://ray.readthedocs.io/en/latest/rllib-training.html#scaling-guide). (#7780)
- Enable restore keras model from h5 file. (#7482)
- Store tf-graph by default when doing `Policy.export_model()`. (#7759)
- Fix default policy overrides torch policy. (#7756, #7769)

RaySGD
----
- BREAKING: Add new API for tuning TorchTrainer using Tune. (#7547)
- BREAKING: Convert the head worker to a local model. (#7746)
- Added a new API for save/restore.  (#7547)
- Add tqdm support to TorchTrainer. (#7588)

Tune
------
- Add sorted columns and TensorBoard to Tune tab. (#7140)
- Tune experiments can now be cancelled via the REST client. (#7719)
- `fail_fast` enables experiments to fail quickly. (#7528)
- override the IP retrieval process if needed. (#7705) 
- TensorBoardX nested dictionary support. (#7705) 

Serve
-----
- Performance improvements:
	- Push route table updates to HTTP proxy. (#7774)
	- Improve serialization. (#7688)
- Add async methods support for serve actors. (#7682)
- Add multiple method support for serve actors. (#7709)
	- You can specify HTTP methods in `serve.create_backend(..., methods=["GET", "POST"])`.
	- The ability to specify which actor method to execute in HTTP through `X-SERVE-CALL-METHOD` header or in `RayServeHandle` through `handle.options("method").remote(...)`.

Others
------
- Progress towards highly available control plane. (#7822, #7742)
- Progress towards Windows compatibility. (#7740, #7739, #7657)
- Progress towards Ray Streaming library. (#7813)
- Progress towards metrics export service. (#7809)
- Basic C++ worker implementation. (#6125)


Thanks
------
We thank the following contributors for their work on this release:

@carlbalmer, @BalaBalaYi, @saurabh3949, @maximsmol, @SongGuyang, @istoica, @pcmoritz, @aannadi, @kfstorm, @ijrsvt, @richardliaw, @mehrdadn, @wumuzi520, @cloudhan, @edoakes, @mitchellstern, @robertnishihara, @hhoke, @simon-mo, @ConeyLiu, @stephanie-wang, @rkooo567, @ffbin, @ericl, @hubcity, @sven1977


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.8.4)

---

## ray-0.8.3: Ray 0.8.3
**Published:** 2020-03-25

Highlights
--------
- Autoscaler has added Azure Support. (#7080, #7515, #7558, #7494)
	- Ray autoscaler helps you launch a distributed ray cluster using a single command line call!
	- It works on Azure, AWS, GCP, Kubernetes, Yarn, Slurm and local nodes.
- Distributed reference counting is turned on by default. (#7628, #7337)
	- This means all ray objects are tracked and garbage collected only when all references go out of scope. It can be turned off with: `ray.init(_internal_config=json.dumps({"distributed_ref_counting_enabled": 0}))`.
	- When the object store is full with objects that are still in scope, you can turn on least-recently-used eviction to force remove objects using `ray.init(lru_evict=True)`. 
- A new command `ray memory` is added to help debug memory usage: (#7589)
	- It shows all object IDs that are in scope, their reference types, sizes and creation site.
        - Read more in the docs: https://ray.readthedocs.io/en/latest/memory-management.html.
```
> ray memory
-----------------------------------------------------------------------------------------------------
 Object ID                                Reference Type       Object Size   Reference Creation Site
=====================================================================================================
; worker pid=51230
ffffffffffffffffffffffff0100008801000000  PINNED_IN_MEMORY            8231   (deserialize task arg) __main__..sum_task
; driver pid=51174
45b95b1c8bd3a9c4ffffffff010000c801000000  USED_BY_PENDING_TASK           ?   (task call) memory_demo.py:<module>:13
ffffffffffffffffffffffff0100008801000000  USED_BY_PENDING_TASK        8231   (put object) memory_demo.py:<module>:6
ef0a6c221819881cffffffff010000c801000000  LOCAL_REFERENCE                ?   (task call) memory_demo.py:<module>:14
-----------------------------------------------------------------------------------------------------
```

API change
----------
- Change `actor.__ray_kill__()` to `ray.kill(actor)`. (#7360)
- Deprecate `use_pickle` flag for serialization. (#7474)
- Remove `experimental.NoReturn`. (#7475)
- Remove `experimental.signal API`. (#7477)

Core
----
- Add Apache 2 license header to C++ files. (#7520)
- Reduce per worker memory usage to 50MB. (#7573)
- Option to fallback to LRU on OutOfMemory. (#7410)
- Reference counting for actor handles. (#7434)
- Reference counting for returning object IDs created by a different process. (#7221)
- Use `prctl(PR_SET_PDEATHSIG)` on Linux instead of reaper. (#7150)
- Route asyncio plasma through raylet instead of direct plasma connection. (#7234)
- Remove static concurrency limit from gRPC server. (#7544)
- Remove `get_global_worker()`, `RuntimeContext`. (#7638)
- Fix known issues from 0.8.2 release:
	- Fix passing duplicate by-reference arguments. (#7306)
	- Fix Raise gRPC message size limit to 100MB. (#7269)


RLlib
-----
- New features:
	- Exploration API improvements. (#7373, #7314, #7380)
	- SAC: add discrete action support. (#7320, #7272)
	- Add high-performance external application connector. (#7641)
- Bug fix highlights:
	- PPO torch memory leak and unnecessary torch.Tensor creation and gc'ing. (#7238)
	- Rename sample_batch_size => rollout_fragment_length. (#7503)
	- Fix bugs and speed up SegmentTree.

Tune
----
- Integrate Dragonfly optimizer. (#5955)
- Fix HyperBand errors. (#7563)
- Access Trial Name, Trial ID inside trainable. (#7378)
- Add a new `repeater` class for high variance trials. (#7366)
- Prevent deletion of checkpoint from user-initiated restoration. (#7501)

Libraries
---------
- [Parallel Iterators] Allow for operator chaining after repartition. (#7268)
- [Parallel Iterators] Repartition functionality. (#7163)
- [Serve] `@serve.route` returns a handle, add `handle.scale`, `handle.set_max_batch_size`.  (#7569)
- [RaySGD] PyTorchTrainer --> TorchTrainer. (#7425)
- [RaySGD] Custom training API. (#7211)
- [RaySGD] Breaking User API changes: (#7384)
  - `data_creator` fed to TorchTrainer now must return a dataloader rather than datasets.
  - TorchTrainer automatically sets "DistributedSampler" if a DataLoader is returned.
  - `data_loader_config` and `batch_size` are no longer parameters for TorchTrainer.
  - TorchTrainer parallelism is now set by `num_workers`.
  - All TorchTrainer args now must be named parameters.

Java
----
- New Java actor API (#7414)
	- `@RayRemote` annotation is removed.
	- Instead of `Ray.call(ActorClass::method, actor)`, the new API is `actor.call(ActorClass::method)`.
- Allow passing internal config from raylet to Java worker. (#7532)
- Enable direct call by default. (#7408)
- Pass large object by reference. (#7595)

Others
------
- Progress towards Ray Streaming, including a Python API. (#7070, #6755, #7152, #7582)
- Progress towards GCS Service for GCS fault tolerance. (#7292, #7592, #7601, #7166)
- Progress towards cross language call between Java and Python. (#7614, #7634)
- Progress towards Windows compatibility. (#7529, #7509, #7658, #7315)
- Improvement in K8s Operator. (#7521, #7621, #7498, #7459, #7622)
- New documentation for Ray Dashboard. (#7304)

Known issues
--------------
- Ray currently doesn't work on Python 3.5.0, but works on 3.5.3 and above. 

Thanks
------
We thank the following contributors for their work on this release:
@rkooo567, @maximsmol, @suquark, @mitchellstern, @micafan, @ClarkZinzow, @Jimpachnet, @mwbrulhardt, @ujvl, @chaokunyang, @robertnishihara, @jovany-wang, @hyeonjames, @zhijunfu, @datayjz, @fyrestone, @eisber, @stephanie-wang, @allenyin55, @BalaBalaYi, @simon-mo, @thedrow, @ffbin, @amogkam, @TisonKun, @richardliaw, @ijrsvt, @wumuzi520, @mehrdadn, @raulchen, @landcold7, @ericl, @edoakes, @sven1977, @ashione, @jorenretel, @gramhagen, @kfstorm, @anthonyhsyu, @pcmoritz


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.8.3)

---

## ray-0.8.2: Ray 0.8.2
**Published:** 2020-02-24

## Highlights
- Pyarrow is no longer vendored. Ray directly uses the C++ Arrow API. You can use any version of pyarrow with ray. (#7233)
- The dashboard is turned on by default. It shows node and process information, actor information, and Ray Tune trials information. You can also use `ray.show_in_webui` to display custom messages for actors. Please try it out and send us feedback! (#6705, #6820, #6822, #6911, #6932, #6955, #7028, #7034) 
- We have made progress on distributed reference counting (behind a feature flag). You can try it out with `ray.init(_internal_config=json.dumps({"distributed_ref_counting_enabled": 1}))`. It is designed to help manage memory using precise distributed garbage collection. (#6945, #6946, #7029, #7075, #7218, #7220, #7222, #7235, #7249)

## Breaking changes
- Many experimental Ray libraries are moved to the util namespace. (#7100)
	- `ray.experimental.multiprocessing` => `ray.util.multiprocessing`
	- `ray.experimental.joblib` => `ray.util.joblib`
	- `ray.experimental.iter` => `ray.util.iter`
	- `ray.experimental.serve` => `ray.serve`
	- `ray.experimental.sgd` => `ray.util.sgd`
- Tasks and actors are cleaned up if their owner process dies. (#6818)
- The `OMP_NUM_THREADS` environment variable defaults to 1 if unset. This improves training performance and reduces resource contention. (#6998)
- We now vendor `psutil` and `setproctitle` to support turning the dashboard on by default. Running `import psutil` after `import ray` will use the version of psutil that ships with Ray. (#7031)

## Core
- The Python raylet client is removed. All raylet communication now goes through the core worker. (#6018)
- Calling `delete()` will not delete objects in the in-memory store. (#7117)
- Removed vanilla pickle serialization for task arguments. (#6948)
- Fix bug passing empty bytes into Python tasks. (#7045)
- Progress toward next generation ray scheduler. (#6913)
- Progress toward service based global control store (GCS). (#6686, #7041)

## RLlib 
- Improved PyTorch support, including a PyTorch version of PPO. (#6826, #6770)
- Added distributed SGD for PPO. (#6918, #7084)
- Added an exploration API for controlling epsilon greedy and stochastic exploration. (#6974, #7155)
- Fixed schedule values going negative past the end of the schedule. (#6971, #6973)
- Added support for histogram outputs in TensorBoard. (#6942)
- Added support for parallel and customizable evaluation step. (#6981)

## Tune
- Improved Ax Example. (#7012)
- Process saves asynchronously. (#6912)
- Default to tensorboardx and include it in requirements. (#6836)
- Added experiment stopping api. (#6886)
- Expose progress reporter to users. (#6915)
- Fix directory naming regression. (#6839)
- Handles nan case for asynchyperband. (#6916)
- Prevent memory checkpoints from breaking trial fault tolerance. (#6691)
- Remove keras dependency. (#6827)
- Remove unused tf loggers. (#7090)
- Set correct path when deleting checkpoint folder. (#6758)
- Support callable objects in variant generation. (#6849)

## Autoscaler 
- Ray nodes now respect docker limits. (#7039)
- Add `--all-nodes` option to rsync-up. (#7065)
- Add port-forwarding support for attach. (#7145)
- For AWS, default to latest deep learning AMI. (#6922)
- Added 'ray dashboard' command to proxy ray dashboard in remote machine. (#6959)

## Utility libraries
- Support of scikit-learn with Ray joblib backend. (#6925)
- Parallel iterator support local shuffle. (#6921)
- [Serve] support no http headless services. (#7010)
- [Serve] refactor router to use Ray asyncio support. (#6873)
- [Serve] support composing arbitrary dags. (#7015)
- [RaySGD] support fp16 via PyTorch apex. (#7061)
- [RaySGD] refactor PyTorch sgd documentation. (#6910)
- Improvement in Ray Streaming. (#7043, #6666, #7071)

## Other improvements
- Progress toward Windows compatibility. (#6882, #6823)
- Ray Kubernetes operator improvements. (#6852, #6851, #7091)
- Java support for concurrent actor calls API. (#7022)
- Java support for direct call for normal tasks. (#7193)
- Java support for cross language Python invocation. (#6709)
- Java support for cross language serialization for actor handles. (#7134)

## Known issue
- Passing the same ObjectIDs multiple time as arguments currently doesn't work. (#7296)
- Tasks can exceed gRPC max message size. (#7263)

## Thanks
We thank the following contributors for their work on this release:
@mitchellstern, @hugwi, @deanwampler, @alindkhare, @ericl, @ashione, @fyrestone, @robertnishihara, @pcmoritz, @richardliaw, @yutaizhou, @istoica, @edoakes, @ls-daniel, @BalaBalaYi, @raulchen, @justinkterry, @roireshef, @elpollouk, @kfstorm, @Bassstring, @hhbyyh, @Qstar, @mehrdadn, @chaokunyang, @flying-mojo, @ujvl, @AnanthHari, @rkooo567, @simon-mo, @jovany-wang, @ijrsvt, @ffbin, @AmeerHajAli, @gaocegege, @suquark, @MissiontoMars, @zzyunzhi, @sven1977, @stephanie-wang, @amogkam, @wuisawesome, @aannadi, @maximsmol


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.8.2)

---

## ray-0.8.1: ray-0.8.1
**Published:** 2020-01-27

Ray 0.8.1 Release Notes
=======================

Highlights
----------
- `ObjectID`s corresponding to `ray.put()` objects and task returns are now reference counted locally in Python and when passed into a remote task as an argument. `ObjectID`s that have a nonzero reference count will not be evicted from the object store. Note that references for `ObjectID`s passed into remote tasks inside of other objects (e.g., `f.remote((ObjectID,))` or `f.remote([ObjectID])`) are not currently accounted for. (#6554)
- `asyncio` actor support: actors can now define `async def` method and Ray will run multiple method invocations in the same event loop. The maximum concurrency level can be adjusted with `ActorClass.options(max_concurrency=2000).remote()`.
- `asyncio` `ObjectID` support: Ray ObjectIDs can now be directly awaited using the Python API. `await my_object_id` is similar to `ray.get(my_object_id)`, but allows context switching to make the operation non-blocking. You can also convert an `ObjectID` to a `asyncio.Future` using `ObjectID.as_future()`.
- Added experimental parallel iterators API (#6644, #6726): `ParallelIterator`s can be used to more convienently load and process data into Ray actors. See the [documentation](https://ray.readthedocs.io/en/latest/iter.html) for details.
- Added multiprocessing.Pool API (#6194): Ray now supports the `multiprocessing.Pool` API out of the box, so you can scale existing programs up from a single node to a cluster by only changing the import statment. See the [documentation](https://ray.readthedocs.io/en/latest/multiprocessing.html) for details.

Core
----
- Deprecated Python 2 (#6581, #6601, #6624, #6665)
- Fixed bug when failing to import remote functions or actors with args and kwargs (#6577)
- Many improvements to the dashboard (#6493, #6516, #6521, #6574, #6590, #6652, #6671, #6683, #6810)
- Progress towards Windows compatibility (#6446, #6548, #6653, #6706)
- Redis now binds to localhost and has a password set by default (#6481)
- Added `actor.__ray_kill__()` to terminate actors immediately (#6523)
- Added 'ray stat' command for debugging (#6622)
- Added documentation for fault tolerance behavior (#6698)
- Treat static methods as class methods instead of instance methods in actors (#6756)

RLlib
-----
- DQN distributional model: Replace all legacy tf.contrib imports with tf.keras.layers.xyz or tf.initializers.xyz (#6772)
- SAC site changes (#6759)
- PG unify/cleanup tf vs torch and PG functionality test cases (tf + torch) (#6650)
- SAC for Mujoco Environments (#6642)
- Tuple action dist tensors not reduced properly in eager mode (#6615)
- Changed foreach_policy to foreach_trainable_policy (#6564)
- Wrapper for the dm_env interface (#6468)

Tune
----
- Get checkpoints paths for a trial after tuning (#6643)
- Async restores and S3/GCP-capable trial FT (#6376)
- Usability errors PBT (#5972)
- Demo exporting trained models in pbt examples (#6533)
- Avoid duplication in TrialRunner execution (#6598)
- Update params for optimizer in reset_config (#6522)
- Support Type Hinting for py3 (#6571)

Other Libraries
---------------
- [serve] Pluggable Queueing Policy (#6492)
- [serve] Added BackendConfig (#6541)
- [sgd] Fault tolerance support for pytorch + revamp documentation (#6465)

Thanks
-------

We thank the following contributors for their work on this release:

@chaokunyang, @Qstar, @simon-mo, @wlx65003, @stephanie-wang, @alindkhare, @ashione, @harrisonfeng, @JingGe, @pcmoritz, @zhijunfu, @BalaBalaYi, @kfstorm, @richardliaw, @mitchellstern, @michaelzhiluo, @ziyadedher, @istoica, @EyalSel, @ffbin, @raulchen, @edoakes, @chenk008, @frthjf, @mslapek, @gehring, @hhbyyh, @zzyunzhi, @zhu-eric, @MissiontoMars, @sven1977, @walterddr, @micafan, @inventormc, @robertnishihara, @ericl, @ZhongxiaYan, @mehrdadn, @jovany-wang, @ujvl, @bharatpn


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.8.1)

---

## ray-0.8.0: Ray 0.8.0 Release Notes
**Published:** 2019-12-18

Ray 0.8.0 Release Notes
=======================

This is the first release with gRPC direct calls enabled by default for both tasks and actors, which substantially improves task submission performance.

Highlights
----------
- Enable gRPC direct calls by default (#6367). In this mode, actor tasks are sent directly from actor to actor over gRPC; the Raylet only coordinates actor creation. Similarly, with tasks, tasks are submitted directly from worker to worker over gRPC; the Raylet only coordinates the scheduling decisions. In addition, small objects (<100KB in size) are no longer placed in the object store. They are inlined into task submissions and returns when possible.

Note: in some cases, reconstruction of large evicted objects is not possible with direct calls. To revert to the 0.7.7 behaviour, you can set the environment variable `RAY_FORCE_DIRECT=0`.

Core
----
- [Dashboard] Add remaining features from old dashboard (#6489)
- Ray Kubernetes Operator Part 1: readme, structure, config and CRD realted file (#6332)
- Make sure numpy >= 1.16.0 is installed for fast pickling support (#6486)
- Avoid workers starting with the same random seed (#6471)
- Properly handle a forwarded task that gets forwarded back (#6271)

RLlib
-----
- (Bug Fix): Remove the extra 0.5 in the Diagonal Gaussian entropy (#6475)
- AlphaZero and Ranked reward implementation (#6385)

Tune
-----
- Add example and tutorial for DCGAN (#6400)
- Report trials by state fairly (#6395)
- Fixed bug in PBT where initial trial result is empty. (#6351)

Other Libraries
---------------
- [sgd] Add support for multi-model multi-optimizer training (#6317)
- [serve] Added deadline awareness (#6442)
- [projects] Return parameters for a command (#6409)
- [streaming] Streaming data transfer and python integration (#6185)


Thanks
-------

We thank the following contributors for their work on this release:

@zplizzi, @istoica, @ericl, @mehrdadn, @walterddr, @ujvl, @alindkhare, @timgates42, @chaokunyang, @eugenevinitsky, @kfstorm, @Maltimore, @visatish, @simon-mo, @AmeerHajAli, @wumuzi520, @robertnishihara, @micafan, @pcmoritz, @zhijunfu, @edoakes, @sytelus, @ffbin, @richardliaw, @Qstar, @stephanie-wang, @Coac, @mitchellstern, @MissiontoMars, @deanwampler, @hhbyyh, @raulchen

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.8.0)

---

## ray-0.7.7: ray-0.7.7
**Published:** 2019-12-16

Ray 0.7.7 Release Notes
=======================

Highlights
----------

- Remote functions and actors now support kwargs and positionals (#5606).
- `ray.get` now supports a `timeout` argument (#6107). If the object isn't available before the timeout passes, a `RayTimeoutError` is raised.
- Ray now supports [detached actors](https://ray.readthedocs.io/en/latest/advanced.html#detached-actors) (#6036), which persist beyond the lifetime of the script that creates them and can be referred to by a user-defined name.
- Added [documentation](https://ray.readthedocs.io/en/latest/deploy-on-yarn.html) for how to deploy Ray on YARN clusters using [Skein](https://jcrist.github.io/skein/) (#6119, #6173).
- The Ray scheduler now attempts to schedule tasks fairly to avoid starvation (#5851).

Core
----

- Progress towards a new backend architecture where tasks and actor tasks are submitted directly between workers. #5783, #5991, #6040, #6054, #6075, #6088, #6122, #6147, #6171, #6177, #6118, #6188, #6259, #6277
- Progress towards Windows compatibility. #6071, #6204, #6205, #6282
- Now using cloudpickle_fast for serialization by default, which supports more types of Python objects without sacrificing performance. #5658, #5805, #5960, #5978
- Various bugfixes. #5946, #6175, #6176, #6231, #6253, #6257, #6276, 

RLlib
-----

- Now using pytorch's function to see if gpu is available. #5890
- Fixed APEX priorities returning zero all the time. #5980
- Fixed leak of TensorFlow assign operations in DQN/DDPG. #5979
- Fixed choosing the wrong neural network model for Atari in 0.7.5. #6087
- Added large scale regression test for RLlib. #6093
- Fixed and added test for LR annealing config. #6101
- Reduced log verbosity. #6154
- Added a microbatch optimizer with an A2C example. #6161

Tune
-----

- Search algorithms now use early stopped trials for optimization. #5651
- Metrics are now outputted via a tabular format. Errors are outputted on a separate table. #5822
- In the distributed setting, checkpoints are now deleted automatically post-sync using an rsync flag. Checkpoints on the driver are garbage collected according to the policy defined by the user. #5877
- A much faster ExperimentAnalysis tool. #5962
- Trial executor callbacks now take in a “Runner” parameter. #5868
- Fixed `queue_trials` so to enable cluster autoscaling with a CPU-Only Head Node. #5900
- Added a TensorBoardX logger. #6133

Other Libraries
---------------

- Serving: Progress towards a new Ray serving library. #5854, #5886, #5894, #5929, #5937, #5961, #6051

Thanks
-------

We thank the following contributors for their amazing contributions:

@zhuohan123, @jovany-wang, @micafan, @richardliaw, @waldroje, @mitchellstern, @visatish, @mehrdadn, @istoica, @ericl, @adizim, @simon-mo, @lsklyut, @zhu-eric, @pcmoritz, @hhbyyh, @suquark, @sotte, @hershg, @pschafhalter, @stackedsax, @edoakes, @mawright, @stephanie-wang, @ujvl, @ashione, @couturierc, @AdamGleave, @robertnishihara, @DaveyBiggers, @daiyaanarfeen, @danyangz, @AmeerHajAli, @mimoralea

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.7.7)

---

## ray-0.7.6: ray-0.7.6
**Published:** 2019-10-24

Ray 0.7.6 Release Notes
=======================

Highlights
----------

- The Ray autoscaler now supports Kubernetes as a backend (#5492). This makes it possible to start a Ray cluster on top of your existing Kubernetes cluster with a simple shell command. 
  + Please see the Kubernetes section of the [autoscaler documentation](https://ray.readthedocs.io/en/latest/autoscaling.html) to get started.
  + This is a new feature and may be rough around the edges. If you run into problems or have suggestions for how to improve Ray on Kubernetes, please file an issue.

- The Ray cluster dashboard has been revamped (#5730, #5857) to improve the UI and include logs and error messages. More improvements will be coming in the near future.
  + You can try out the dashboard by starting Ray with `ray.init(include_webui=True)` or `ray start --include-webui`.
  + Please let us know if you have suggestions for what would be most useful to you in the new dashboard.

Core
----

- Progress towards refactoring the Python worker on top of the core worker. #5750, #5771, #5752
- Fix an issue in local mode where multiple actors didn't work properly. #5863
- Fix class attributes and methods for actor classes. #5802
- Improvements in error messages and handling. #5782, #5746, #5799
- Serialization improvements. #5841, #5725
- Various documentation improvements. #5801, #5792, #5414, #5747, #5780, #5582

RLlib
-----

- Added a link to BAIR blog posts in the documentation. #5762
- Tracing for eager tensorflow policies with `tf.function`. #5705

Tune
-----

- Improved MedianStoppingRule. #5402
- Add PBT + Memnn example. #5723
- Add support for function-based stopping condition. #5754
- Save/Restore for Suggestion Algorithms. #5719
- TensorBoard HParams for TF2.0. #5678

Other Libraries
---------------

- Serving: Progress towards a new Ray serving library. #5849, #5850, #5852

Thanks
-------

We thank the following contributors for their amazing contributions:

@hershg, @JasonWayne, @kfstorm, @richardliaw, @batzner, @vakker, @robertnishihara, @stephanie-wang, @gehring, @edoakes, @zhijunfu, @pcmoritz, @mitchellstern, @ujvl, @simon-mo, @ecederstrand, @mawright, @ericl, @anthonyhsyu, @suquark, @waldroje


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.7.6)

---

## ray-0.7.5: ray-0.7.5
**Published:** 2019-09-25

# Ray 0.7.5 Release Notes

## Ray API
- Objects created with `ray.put()` are now reference counted. #5590
- Add internal `pin_object_data()` API. #5637
- Initial support for pickle5. #5611
- Warm up Ray on `ray.init()`. #5685
- `redis_address` passed to `ray.init` is now just `address`. #5602

## Core
- Progress towards a common C++ core worker. #5516, #5272, #5566, #5664
- Fix log monitor stall with many log files. #5569
- Print warnings when tasks are unschedulable. #5555
- Take into account resource queue lengths when autoscaling #5702, #5684

## Tune
- TF2.0 TensorBoard support. #5547, #5631
- `tune.function()` is now deprecated. #5601

## RLlib
- Enhancements for TF eager support. #5625, #5683, #5705
- Fix DDPG regression. #5626

## Other Libraries
- Complete rewrite of experimental serving library. #5562
- Progress toward Ray projects APIs. #5525, #5632, #5706
- Add TF SGD implementation for training. #5440 
- Many documentation improvements and bugfixes.


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.7.5)

---

## ray-0.7.4: ray-0.7.4
**Published:** 2019-09-05

Ray 0.7.4 Release Notes
=================

Highlights
----------

- There were many **documentation improvements** (#5391, #5389, #5175). As we continue to improve the documentation we value your feedback through the “Doc suggestion?” link at the top of the [documentation](https://ray.readthedocs.io/en/latest/). Notable improvements:
  + We’ve added guides for best practices using TensorFlow and PyTorch.
  + We’ve revamped the Walkthrough page for Ray users, providing a better experience for beginners.
  + We’ve revamped guides for using Actors and inspecting internal state.

- Ray supports **memory limits** now to ensure memory-intensive applications run predictably and reliably. You 
   can activate them through the `ray.remote` decorator:
   ```python
   @ray.remote(
       memory=2000 * 1024 * 1024,
       object_store_memory=200 * 1024 * 1024)
   class SomeActor(object):
       def __init__(self, a, b):
           pass
   ```
   You can set limits for the heap and the object store, see the [documentation](https://ray.readthedocs.io/en/latest/memory-management.html).

- There is now preliminary support for **projects**, see the the [project documentation](https://ray.readthedocs.io/en/latest/projects.html). Projects allow you to 
   package your code and easily share it with others, ensuring a reproducible cluster setup. To get started, you 
   can run
   ```shell
   # Create a new project.
   ray project create <project-name>
   # Launch a session for the project in the current directory.
   ray session start
   # Open a console for the given session.
   ray session attach
   # Stop the given session and all of its worker nodes.
   ray session stop
   ```
   Check out the [examples](https://github.com/ray-project/ray/tree/f1dcce5a472fba1c77c4aa023589689efbfeb4f6/python/ray/projects/examples). This is an actively developed new feature so we appreciate your feedback!

**Breaking change:** The `redis_address` parameter was renamed to `address` (#5412, #5602) and the former will be removed in the future.

Core
-----

- Move Java bindings on top of the core worker #5370
- Improve log file discoverability #5580
- Clean up and improve error messages #5368, #5351

RLlib
-----

- Support custom action space distributions #5164
- Add TensorFlow eager support #5436
- Add autoregressive KL #5469
- Autoregressive Action Distributions #5304
- Implement MADDPG agent #5348
- Port Soft Actor-Critic on Model v2 API #5328
- More examples: Add CARLA community example #5333 and rock paper scissors multi-agent example #5336
- Moved RLlib to top level directory #5324

Tune
-----

- Experimental Implementation of the BOHB algorithm #5382
- Breaking change: Nested dictionary results are now flattened for CSV writing: `{“a”: {“b”: 1}} => {“a/b”: 1}` #5346
- Add Logger for MLFlow #5438
- TensorBoard support for TensorFlow 2.0 #5547
- Added examples for XGBoost and LightGBM #5500
- HyperOptSearch now has warmstarting #5372

Other Libraries
---------------

- SGD: Tune interface for Pytorch MultiNode SGD #5350
- Serving: The old version of ray.serve was deprecated #5541
- Autoscaler: Fix ssh control path limit #5476
- Dev experience: Ray CI tracker online at https://ray-travis-tracker.herokuapp.com/

Various fixes: Fix log monitor issues #4382 #5221 #5569, the top-level ray directory was cleaned up #5404

Thanks
-------

We thank the following contributors for their amazing contributions:

@jon-chuang, @lufol, @adamochayon, @idthanm, @RehanSD, @ericl, @michaelzhiluo, @nflu, @pengzhenghao, @hartikainen, @wsjeon, @raulchen, @TomVeniat, @layssi, @jovany-wang, @llan-ml, @ConeyLiu, @mitchellstern, @gregSchwartz18, @jiangzihao2009, @jichan3751, @mhgump, @zhijunfu, @micafan, @simon-mo, @richardliaw, @stephanie-wang, @edoakes, @akharitonov, @mawright, @robertnishihara, @lisadunlap, @flying-mojo, @pcmoritz, @jredondopizarro, @gehring, @holli, @kfstorm



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.7.4)

---

## ray-0.7.3: ray-0.7.3
**Published:** 2019-08-04

Ray 0.7.3 Release Note
======================

Highlights
----------
-   RLlib [ModelV2API](https://ray.readthedocs.io/en/latest/rllib-models.html) is ready to use. It improves support for Keras and RNN models, as well as allowing object-oriented reuse of variables. ModelV1 API is deprecated. No migration is needed.
-   `ray.experimental.sgd.pytorch.PyTorchTrainer` is ready for early adopters. Checkout the documentation [here](https://ray.readthedocs.io/en/latest/distributed_training.html). We welcome your feedback!
    ```python
    model_creator = lambda config: YourPyTorchModel()
    data_creator = lambda config: YourTrainingSet(), YourValidationSet()

    trainer = PyTorchTrainer(
        model_creator,
        data_creator,
        optimizer_creator=utils.sgd_mse_optimizer,
        config={"lr": 1e-4},
        num_replicas=2,
        resources_per_replica=Resources(num_gpus=1),
        batch_size=16,
        backend="auto")

    for i in range(NUM_EPOCHS):
        trainer.train()
    ```
-   You can query all the clients that have performed `ray.init` to connect to the current cluster with `ray.jobs()`. #5076
    ```python
    >>> ray.jobs()
    [{'JobID': '02000000',
      'NodeManagerAddress': '10.99.88.77',
      'DriverPid': 74949,
      'StartTime': 1564168784,
      'StopTime': 1564168798},
     {'JobID': '01000000',
      'NodeManagerAddress': '10.99.88.77',
      'DriverPid': 74871,
      'StartTime': 1564168742}]
    ```

Core
----
- Improvement on memory storage handling. #5143, #5216, #4893
- Improved workflow:
    - Debugging tool `local_mode` now behaves more consistently. #5060
    - Improved KeyboardInterrupt Exception Handling, stack trace reduced from 115 lines to 22 lines. #5237
-   Ray core:
    - Experimental direct actor call. #5140, #5184
    - Improvement in core worker, the shared module between Python and Java. #5079, #5034, #5062
    - GCS (global control store) was refactored. #5058, #5050

RLlib
-----
- Finished port of all major RLlib algorithms to builder pattern #5277, #5258, #5249
- `learner_queue_timeout` can be configured for async sample optimizer. #5270
- `reproducible_seed` can be used for reproducible experiments. #5197
- Added entropy coefficient decay to IMPALA, APPO and PPO #5043

Tune:
-----
- **Breaking:** `ExperimentAnalysis` is now returned by default from `tune.run`. To obtain a list of trials, use `analysis.trials`. #5115
- **Breaking:** Syncing behavior between head and workers can now be customized (`sync_to_driver`). Syncing behavior (`upload_dir`) between cluster and cloud is now separately customizable (`sync_to_cloud`). This changes the structure of the uploaded directory - now `local_dir` is synced with `upload_dir`. #4450
- Introduce `Analysis` and `ExperimentAnalysis` objects. `Analysis` object will now return all trials in a folder; `ExperimentAnalysis` is a subclass that returns all trials of an experiment. #5115
- Add missing argument `tune.run(keep_checkpoints_num=...)`. Enables only keeping the last N checkpoints. #5117
- Trials on failed nodes will be prioritized in processing. #5053
- Trial Checkpointing is now more flexible. #4728
- Add system performance tracking for gpu, ram, vram, cpu usage statistics - toggle with `tune.run(log_sys_usage=True)`. #4924
- Experiment checkpointing frequency is now less frequent and can be controlled with `tune.run(global_checkpoint_period=...)`. #4859

Autoscaler
----------
- Add a `request_cores` function for manual autoscaling. You can now manually request resources for the autoscaler. #4754
- Local cluster:
    - More readable example yaml with comments. #5290

    - Multiple cluster name is supported. #4864

- Improved logging with AWS NodeProvider. `create_instance` call will be logged. #4998

Others Libraries:
-----------------
- SGD:
    - Example for Training. #5292
    - Deprecate old distributed SGD implementation. #5160
- Kuberentes: Ray namespace added for k8s. #4111
- Dev experience: Add linting pre-push hook. #5154

Thanks:
-------

We thank the following contributors for their amazing contributions:

@joneswong, @1beb, @richardliaw, @pcmoritz, @raulchen, @stephanie-wang, @jiangzihao2009, @LorenzoCevolani, @kfstorm, @pschafhalter, @micafan, @simon-mo, @vipulharsh, @haje01, @ls-daniel, @hartikainen, @stefanpantic, @edoakes, @llan-ml, @alex-petrenko, @ztangent, @gravitywp, @MQQ, @dulex123, @morgangiraud, @antoine-galataud, @robertnishihara, @qxcv, @vakker, @jovany-wang, @zhijunfu, @ericl

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.7.3)

---

## ray-0.7.2: ray-0.7.2
**Published:** 2019-07-03

Core
----
- Improvements
  - Continue moving the worker code to C++. #5031, #4966, #4922, #4899, #5032, #4996, #4875
  - Add a hash table data structure to the Redis modules. #4911
  - Use gRPC for communication between node managers. #4968, #5023, #5024
- Python
   - `@ray.remote` now inherits the function docstring. #4985
   - Remove `typing` module from setup.py `install_requirements`. #4971
- Java
  - Allow users to set JVM options at actor creation time. #4970
- Internal
  - Refactor IDs: `DriverID` -> `JobID`, change all ID functions to camel case. #4964, #4896
  - Improve organization of directory structure. #4898
- Peformance
  - Get task object dependencies in parallel from object store. #4775
  - Flush lineage cache on task submission instead of execution. #4942
  - Remove debug check for uncommitted lineage. #5038

Tune
----
- Add directional metrics for components. #4120, #4915
- Disallow setting `resources_per_trial` when it is already configured. #4880
- Make PBT Quantile fraction configurable. #4912

RLlib
-----
- Add QMIX mixer parameters to optimizer param list. #5014
- Allow Torch policies access to full action input dict in `extra_action_out_fn`. #4894
- Allow access to batches prior to postprocessing. #4871
- Throw error if `sample_async` is used with pytorch for A3C. #5000
- Patterns & User Experience
  - Rename `PolicyEvaluator` => `RolloutWorker`. #4820
  - Port remainder of algorithms to `build_trainer()` pattern. #4920
  - Port DQN to `build_tf_policy()` pattern. #4823
- Documentation
  - Add docs on how to use TF eager execution. #4927
  - Add preprocessing example to offline documentation. #4950

Other Libraries
---------------
- Add support for distributed training with PyTorch. #4797, #4933
- Autoscaler will kill workers on exception. #4997
- Fix handling of non-integral timeout values in `signal.receive`. #5002

Thanks
-----
We thank the following contributors for their amazing contributions: @jiangzihao2009, @raulchen, @ericl, @hershg, @kfstorm, @kiddyboots216, @jovany-wang, @pschafhalter, @richardliaw, @robertnishihara, @stephanie-wang, @simon-mo, @zhijunfu, @ls-daniel, @ajgokhale, @rueberger, @suquark, @guoyuhong, @jovany-wang, @pcmoritz, @hartikainen, @timonbimon, @TianhongDai



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.7.2)

---

## ray-0.7.1: ray-0.7.1
**Published:** 2019-06-23

Core
----
- Change global state API. #4857
  - `ray.global_state.client_table()` -> `ray.nodes()`
  - `ray.global_state.task_table()` -> `ray.tasks()`
  - `ray.global_state.object_table()` -> `ray.objects()`
  - `ray.global_state.chrome_tracing_dump()` -> `ray.timeline()`
  - `ray.global_state.cluster_resources()` -> `ray.cluster_resources()`
  - `ray.global_state.available_resources()` -> `ray.available_resources()`
- Export remote functions lazily. #4898
- Begin moving worker code to C++. #4875, #4899, #4898
- Upgrade arrow to latest master. #4858
- Upload wheels to S3 under `<branch-name>/<commit-id>`. #4949
- Add hash table to Redis-Module. #4911
- Initial support for distributed training with PyTorch. #4797

Tune
----
- Disallow setting `resources_per_trial` when it is already configured. #4880
- Initial experiment tracking support. #4362

RLlib
-----
- Begin deprecating Python 2 support in RLlib. #4832
- TensorFlow 2 compatibility. #4802
- Allow Torch policies access to full action input dict in `extra_action_out_fn`. #4894
- Allow access to batches prior to postprocessing. #4871
- Port algorithms to `build_trainer()` pattern. #4823
- Rename `PolicyEvaluator` -> `RolloutWorker`. #4820
- Rename `PolicyGraph` -> `Policy`, move from evaluation/ to policy/. #4819
- Support continuous action distributions in IMPALA/APPO. #4771

(Revision: 6/23/2019 - Accidentally included commits that were not part of the release.)

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.7.1)

---

## ray-0.7.0: ray-0.7.0
**Published:** 2019-05-18

Core
----
- Backend bug fixes. #4766, #4763, #4605
- Add experimental API for creating resources at runtime. #3742

Tune
----
- Post-Experiment Tools. #4351
- Add Ax to Tune. #4731
- Tune bug fixes. #4733, #4659, #4747

RLlib
-----
- Remove dependency on TensorFlow. #4764
- TD3/DDPG improvements and MuJoCo benchmarks. #4694
- Evaluation mode implementation for rllib.Trainer class. #4647
- Replace ray.get() with ray_get_and_free() to automatically free object store memory. #4586
- RLLib bug fixes. #4736, #4735, #4652, #4630

Autoscaler
----------
- Add an aggressive autoscaling flag. #4285
- Autoscalar bug fixes. #4782, #4653


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.7.0)

---

## ray-0.6.6: ray-0.6.6
**Published:** 2019-04-19

Core
----
- Add `delete_creating_tasks` option for `internal.free()` #4588

Tune
----
- Add filter flag for Tune CLI. #4337
- Better handling of ``tune.function`` in global checkpoint. #4519
- Add compatibility to nevergrad 0.2.0+. #4529
- Add `--columns` flag for CLI. #4564
- Add checkpoint eraser. #4490
- Fix checkpointing for Gym types. #4619

RLlib
-----
- Report sampler performance metrics. #4427
- Ensure stats are consistently reported across all algos. #4445
- Cleanup ``TFPolicyGraph``. #4478
- Make batch timeout for remote workers tunable. #4435
- Fix inconsistent weight assignment operations in ``DQNPolicyGraph``. #4504
- Add support for LR schedule to DQN/APEX. #4473
- Add option for RNN state and value estimates to span episodes. #4429
- Create a combination of ``ExternalEnv`` and ``MultiAgentEnv``, called ``ExternalMutliAgentEnv``. #4200
- Support ``prev_state``/``prev_action`` in rollout and fix multiagent. #4565
- Support torch device and distributions. #4553

Java
----
- TestNG outputs more verbose error messages. #4507
- Implement ``GcsClient``. #4601
- Avoid unnecessary memory copy and addd a benchmark. #4611

Autoscaler
----------
- Add support for separate docker containers on head and worker nodes. #4537
- Add an aggressive autoscaling flag. #4285


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.6.6)

---

## ray-0.6.5: ray-0.6.5
**Published:** 2019-03-25

Core
----
- Build system fully converted to Bazel. #4284, #4280, #4281
- Introduce a set data structure in the GCS. #4199
- Make all arguments to `_remote()` optional. #4305
- Improve object transfer latency by setting `TCP_NODELAY` on all TCP connections. #4318
- Add beginning of experimental serving module. #4095
- Remove Jupyter notebook based UI. #4301
- Add `ray timeline` command line command for dumping Chrome trace. #4239

Tune
----
- Add custom field for serializations. #4237
- Begin adding Tune CLI. #3983, #4321, #4322
- Add optimization to reuse actors. #4218
- Add warnings if the Tune event loop gets clogged. #4353
- Switch preferred API from `tune.run_experiments` to `tune.run`. #4234
- Make the logging from the function API consistent and predictable. #4011

RLlib
-----
- **Breaking:** Flip sign of entropy coefficient in A2C and Impala. #4374
- Add option to continue training even if some workers crash. #4376
- Add asynchronous remote workers. #4253
- Add callback accessor for raw observations. #4212

Java
----
- Improve single-process mode. #4245, #4265
- Package native dependencies into jar. #4367
- Initial support for calling Python functions from Java. #4166

Autoscaler
----------
- Restore error messages for setup errors. #4388

Known Issues
------------
- Object broadcasts on large clusters are inefficient. #2945

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.6.5)

---

## ray-0.6.4: ray-0.6.4
**Published:** 2019-03-06

Breaking
---------
- Removed `redirect_output` and `redirect_worker_output` from `ray.init`, removed deprecated `_submit` method. #4025
- Move `TensorFlowVariables` to `ray.experimental.tf_utils`. #4145

Core
------
- Stream worker logging statements to driver by default. #3892
- Added experimental ray signaling mechanism, see the [documentation](https://ray.readthedocs.io/en/latest/signals.html). #3624
- Make Bazel the default build system. #3898
- Preliminary experimental streaming API for Python. #4126
- Added web dashboard for monitoring node resource usage. #4066
- Improved propagation of backend errors to user. #4039
- Many improvements for the Java frontend. #3687, #3978, #4014, #3943, #3839, #4038, #4039, #4063, #4100, #4179, #4178 
- Support for dataclass serialization. #3964
- Implement actor checkpointing. #3839
- First steps toward cross-language invocations. #3675
- Better defaults for Redis memory usage. #4152

Tune
------
- **Breaking**: Introduce ability to turn off default logging. Deprecates custom_loggers. #4104
- Support custom resources. #2979
- Add initial parameter suggestions for HyperOpt. #3944
- Add scipy-optimize to Tune. #3924
- Add Nevergrad. #3985
- Add number of trials to the trial runner logger. #4068
- Support RESTful API for the webserver. #4080
- Local mode support. #4138
- Dynamic resources for trials. #3974

RLlib
-------
- Basic infrastructure for off-policy estimation. #3941
- Add simplex action space and Dirichlet action distribution. #4070
- Exploration with parameter space noise. #4048
- Custom supervised loss API. #4083
- Add torch policy gradient implementation. #3857 

Autoscaler and Cluster Setup
--------------------------------------
- Add docker run option (e.g. to support nvidia-docker). #3921

Modin
-----
- Upgrade Modin to 0.3.1, see the [release notes](https://github.com/modin-project/modin/releases/tag/v0.3.1). #4058

Known Issues
------------
- Object broadcasts on large clusters are inefficient. #2945
- IMPALA is broken #4329

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.6.4)

---

## ray-0.6.3: ray-0.6.3
**Published:** 2019-03-06

Core
------
- Initial work on porting the build system to Bazel. #3918, #3806, #3867, #3842
- Allow starting Ray processes inside valgrind, gdb, tmux. #3824, #3847
- Stability improvements and bug fixes. #3861, #3962, #3958, #3855, #3736, #3822, #3821, #3925
- Convert Python C extensions to Cython. #3541
- `ray start` can now be used to start Java workers. #3838, #3852
- Enable LZ4 compression in `pyarrow` build. #3931
- Update Redis to version 5.0.3. #3886
- Use one memory-mapped file for Plasma store. #3871, 

Tune
------
- Support for BayesOpt. #3864
- Support for SigOpt. #3844
- Support executing infinite recovery retries for a trial. #3901
- Support `export_formats` option to export policy graphs. #3868
- Cluster and logging improvements. #3906

RLlib
-------
- Support for Asynchronous Proximal Policy Optimization (APPO). #3779
- Support for MARWIL. #3635
- Support for evaluation option in DQN. #3835
- Bug fixes. #3865, #3810, #3938
- Annotations for API stability. #3808

Autoscaler and Cluster Setup
--------------------------------------
- Faster cluster launch and update. #3720
- Bug fixes. #3916, #3860, #3937, #3782, #3969
- Kubernetes configuration improvements. #3875, #3909

Modin
-----
- Update Modin to 0.3.0. #3936 
    - [Modin 0.3.0 release notes](https://github.com/modin-project/modin/releases/tag/v0.3.0)

Known Issues
------------
- Object broadcasts on large clusters are inefficient. #2945 

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.6.3)

---

## ray-0.6.2: ray-0.6.2
**Published:** 2019-01-17

## Breaking Changes
- The `timeout` argument of `ray.wait` now uses seconds instead of milliseconds. #3706

## Core
- Limit default redis max memory to 10GB. #3630
- Define a `Node` class to manage Ray processes. #3733
- Garbage collection of actor dummy objects. #3593
- Split profile table among many keys in the GCS. #3676
- Automatically try to figure out the memory limit in a docker container. #3605
- Improve multi-threading support. #3672
- Push a warning to all users when large number of workers have been started. #3645
- Refactor code `ray.ObjectID` code. #3674

## Tune
- Change log handling for Tune. #3661
- Tune now supports resuming from cluster failure. #3309, #3725, #3657, #3681
- Support Configuration Merging for Suggestion Algorithms. #3584
- Support nested PBT mutations. #3455

## RLlib
- Add starcraft multiagent env as example. #3542
- Allow development without needing to compile Ray. #3623
- Documentation for I/O API and multi-agent improvements. #3650
- Export policy model checkpoint. #3637
- Refactor PyTorch custom model support. #3634

## Autoscaler
- Add an initial_workers option. #3530
- Add kill and get IP commands to CLI for testing. #3731
- GCP allow manual network configuration. #3748

# Known Issues:
- Object broadcasts on large clusters are inefficient. #2945


[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.6.2)

---

## ray-0.6.1: ray-0.6.1
**Published:** 2018-12-24

## Core
- Added experimental option to limit Redis memory usage. #3499
- Added option for restarting failed actors. #3332
- Fixed Plasma TensorFlow operator memory leak. #3448
- Fixed compatibility issue with TensorFlow and PyTorch. #3574
- Miscellaneous code refactoring and cleanup. #3563 #3564 #3461 #3511
- Documentation. #3427 #3535 #3138
- Several stability improvements. #3592 #3597

## RLlib
- Multi-GPU support for Multi-agent PPO. #3479
- Unclipped actions are sent to learner. #3496
- `rllib rollout` now also preprocesses observations. #3512
- Basic Offline Data API added. #3473
- Improvements to metrics reporting in DQN. #3491
- AsyncSampler no longer auto-concats. #3556
- QMIX Implementation (Experimental). #3548
- IMPALA performance improvements. #3402
- Better error messages. #3444
- PPO performance improvements. #3552

## Autoscaler
- Bug fixes, botocore handling fix. #3454 #3503 #3447

## Ray Tune
- Lambdas now require `tune.function` wrapper. #3457
- Custom loggers, sync functions, and trial names are now supported. #3465
- Improvements to fault tolerance. #3414
- Variant Generator docs clarification. #3583
- `trial_resources` now renamed to `resources_per_trial`. #3580

## Modin
- Modin 0.2.5 is now bundled with Ray
  - `import modin` after `import ray`
  - [Modin 0.2.5 release notes](https://github.com/modin-project/modin/releases/tag/v0.2.5)
- Greater than memory support for object store. #3450 

## Known Issues
- Object broadcasts on large clusters are inefficient. #2945

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.6.1)

---

## ray-0.6.0: ray-0.6.0
**Published:** 2018-12-01

## Breaking Changes
- Renamed `_submit` to `_remote`. #3321
- Object store memory capped at 20GB by default. #3243
- Now `ray.global_state.client_table()` returns a list instead of a dictionary.
- Renamed `ray.global_state.dump_catapult_trace` to `ray.global_state.chrome_tracing_dump`.

## Known Issues
- The Plasma TensorFlow operator leaks memory. #3404
- Object broadcasts on large clusters are inefficient. #2945
- Ape-X leaks memory. #3452
- Action clipping can impede learning (please set clip_actions: False as a workaround) #3496

## Core
- New raylet backend on by default and legacy backend removed. #3020 #3121
- Support for Python 3.7. #2546
- Support for fractional resources (e.g., GPUs). 
- Added `ray stack` for improved debugging (to get stack traces of Python processes on current node). #3213
- Better error messages for low-memory conditions. #3323
- Log file names reorganized under `/tmp/ray/`. #2862
- Improved timeline visualizations. #2306 #3255 

## Modin
- Modin is shipped with Ray. After running `import ray` you can run `import modin`. #3109

## RLlib
- Multi agent support for Ape-X and IMPALA. #3147
- Multi GPU support for IMPALA. #2766
- TD3 optimizations for DDPG. #3353
- Support for Dict and Tuple observation spaces. #3051
- Support for parametric and variable-length action spaces. #3384
- Support batchnorm layers. #3369
- Support custom metrics. #3144

## Autoscaler
- Added `ray submit` for submitting scripts to clusters. #3312
- Added `--new` flag for ray attach. #2973
- Added option to allow private IPs only. #3270 

## Tune
- Support for fractional GPU allocations for trials. #3169
- Better checkpointing and setup. #2889
- Memory tracking and notification. #3298 
- Bug fixes for `SearchAlgorithm`s. #3081 
- Add a `raise_on_failed_trial` flag in run_experiments. #2915
- Better handling of node failures. #3238

## Training
- Experimental support for distributed SGD. #2858 #3033

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.6.0)

---

## ray-0.5.3: ray-0.5.3
**Published:** 2018-09-28

## API
- Add `ray.is_initialized()` to check if `ray.init()` has been called. #2818

## Fixes and Improvements
- Fix issue in which `ray stop` fails to kill plasma object store. #2850
- Remove dependence on `psutil`. #2892

## RLlib
- Set better default for VF clip PPO parameter to avoid silent performance degradation. #2921
- Reward clipping should default to off for non-Atari environments. #2904
- Fix LSTM failing to train on truncated sequences. #2898

## Tune
- Fixed a small bug in trial pausing and cleaned up error messages. #2815

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.5.3)

---

## ray-0.5.2: ray-0.5.2
**Published:** 2018-08-29

## Breaking Changes
- Local mode has changed from `ray.init(driver_mode=ray.PYTHON_MODE)` to `ray.init(local_mode=True)` to improve clarity.

## Autoscaler and Cluster Setup
- Added many convenience commands such as `ray up`, `ray attach`, `ray exec`, and `ray rsync` to simplify launching jobs with Ray.
- Added experimental support for local/on-prem clusters.

## RLlib
- Added the IMPALA algorithm.
- Added the ARS algorithm.
- Added the A2C variant of A3C.
- Added support for distributional DQN.
- Made improvements to multiagent support.
- Added support for model-based rollouts and custom policies.
- Added initial set of reference Atari results.

## Tune
- `SearchAlgorithm`s can now be used separately from `TrialScheduler`s and are found in `ray.tune.suggest`.
- All `TrialScheduler`s have been consolidated under `ray.tune.schedulers`.
- Minor API changes:
    - For `Experiment` configuration, `repeat` has been renamed to `num_samples`.
    - Now, `register_trainable` is handled implicitly.

[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.5.2)

---

## ray-0.5.0: ray-0.5.0
**Published:** 2018-07-07



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.5.0)

---

## ray-0.4.0: ray-0.4.0
**Published:** 2018-03-27



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.4.0)

---

## ray-0.3.1: ray-0.3.1
**Published:** 2018-02-04



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.3.1)

---

## ray-0.3.0: ray-0.3.0
**Published:** 2017-11-28



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.3.0)

---

## ray-0.2.2: ray-0.2.2
**Published:** 2017-11-02



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.2.2)

---

## ray-0.2.1: ray-0.2.1
**Published:** 2017-10-01



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.2.1)

---

## ray-0.2.0: ray-0.2.0
**Published:** 2017-08-31



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.2.0)

---

## ray-0.1.2: ray-0.1.2
**Published:** 2017-06-27



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.1.2)

---

## ray-0.1.1: ray-0.1.1
**Published:** 2017-06-27



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.1.1)

---

## ray-0.1.0: ray-0.1.0
**Published:** 2017-05-21



[View on GitHub](https://github.com/ray-project/ray/releases/tag/ray-0.1.0)

---

