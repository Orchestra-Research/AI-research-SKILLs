# Releases

Version history for this repository (170 releases).

## 2.5.6: Lightning v2.5.6
**Published:** 2025-11-05

# Changes in `2.5.6`

<a name="changelog-pytorch"></a>

## PyTorch Lightning

<details open><summary>Changed</summary>

- Add `name()` function to accelerator interface (([#21325](https://github.com/Lightning-AI/pytorch-lightning/pull/21325)))

</details>

<details open><summary>Removed</summary>

- Remove support for deprecated and archived lightning-habana package ([#21327](https://github.com/Lightning-AI/pytorch-lightning/pull/21327))

</details>


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.6)

---

## 2.5.5: Lightning v2.5.5
**Published:** 2025-09-05

# Changes in `2.5.5`

<a name="changelog-pytorch"></a>

## PyTorch Lightning

<details open><summary>Changed</summary>

- Include `exclude_frozen_parameters` to `DeepSpeedStrategy` ([#21060](https://github.com/Lightning-AI/pytorch-lightning/pull/21060))
- Include `PossibleUserWarning` that is raised if modules are in eval mode when training starts ([#21146](https://github.com/Lightning-AI/pytorch-lightning/pull/21146))

</details>

<details open><summary>Fixed</summary>

- Fixed `LightningCLI` not using `ckpt_path` hyperparameters to instantiate classes ([#21116](https://github.com/Lightning-AI/pytorch-lightning/pull/21116))
- Fixed callbacks by defer step/time-triggered `ModelCheckpoint` saves until validation metrics are available ([#21106](https://github.com/Lightning-AI/pytorch-lightning/pull/21106))
- Fixed with adding a missing device id for pytorch 2.8 ([#21105](https://github.com/Lightning-AI/pytorch-lightning/pull/21105))
- Fixed `TQDMProgressBar` not resetting correctly when using both a finite and iterable dataloader ([#21147](https://github.com/Lightning-AI/pytorch-lightning/pull/21147))
- Fixed cleanup of temporary files from `Tuner` on crashes ([#21162](https://github.com/Lightning-AI/pytorch-lightning/pull/21162))

</details>

<a name="changelog-fabric"></a>

## Lightning Fabric

<details open><summary>Changed</summary>

- Include `exclude_frozen_parameters` to `DeepSpeedStrategy` ([#21060](https://github.com/Lightning-AI/pytorch-lightning/pull/21060))
- Let `_get_default_process_group_backend_for_device` support more hardware platforms (
    [#21057](https://github.com/Lightning-AI/pytorch-lightning/pull/21057), [#21093](https://github.com/Lightning-AI/pytorch-lightning/pull/21093))

</details>

<details open><summary>Fixed</summary>

- Fixed with adding a missing device id for pytorch 2.8 ([#21105](https://github.com/Lightning-AI/pytorch-lightning/pull/21105))
- Respecting `verbose=False` in `seed_everything` when no seed is provided ([#21161](https://github.com/Lightning-AI/pytorch-lightning/pull/21161))

</details>


</br>

**Full commit list**: [2.5.4 -> 2.5.5](https://github.com/Lightning-AI/pytorch-lightning/compare/2.5.4...2.5.5)

<a name="contributors"></a>

# Contributors

We thank **all folks** who submitted issues, features, fixes and doc changes. It's the only way we can **collectively** make Lightning :zap: better for everyone, nice job!

In particular, we would like to thank the authors of the pull-requests above, in no particular order:

@Borda, @KAVYANSHTYAGI, @littlebullGit, @mauvilsa, @SkafteNicki, @taozhiwei

Thank you :heart: and we hope you'll keep them coming!


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.5)

---

## 2.5.4: Lightning v2.5.4
**Published:** 2025-08-29

# Changes in `2.5.4`

<a name="changelog-pytorch"></a>

## PyTorch Lightning

<details open><summary>Fixed</summary>

- Fixed `AsyncCheckpointIO` snapshots tensors to avoid race with parameter mutation ([#21079](https://github.com/Lightning-AI/pytorch-lightning/pull/21079))
- Fixed `AsyncCheckpointIO` threadpool exception if calling fit or validate more than one ([#20952](https://github.com/Lightning-AI/pytorch-lightning/pull/20952))
- Fixed learning rate not being correctly set after using `LearningRateFinder` callback ([#21068](https://github.com/Lightning-AI/pytorch-lightning/pull/21068))
- Fixed misalignment column while using rich model summary in `DeepSpeedstrategy` ([#21100](https://github.com/Lightning-AI/pytorch-lightning/pull/21100))
- Fixed `RichProgressBar` crashing when sanity checking using val dataloader with 0 len ([#21108](https://github.com/Lightning-AI/pytorch-lightning/pull/21108))

</details>

<a name="changelog-fabric"></a>

## Lightning Fabric

<details open><summary>Changed</summary>

- Added support for NVIDIA H200 GPUs in `get_available_flops` ([#20913](https://github.com/Lightning-AI/pytorch-lightning/pull/21119))

</details>


</br>

**Full commit list**: [2.5.3 -> 2.5.4](https://github.com/Lightning-AI/pytorch-lightning/compare/2.5.3...2.5.4)

<a name="contributors"></a>

# Contributors

We thank **all folks** who submitted issues, features, fixes and doc changes. It's the only way we can **collectively** make Lightning :zap: better for everyone, nice job!

In particular, we would like to thank the authors of the pull-requests above, in no particular order:

@fnhirwa, @GdoongMathew, @jjh42, @littlebullGit, @SkafteNicki

Thank you :heart: and we hope you'll keep them coming!


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.4)

---

## 2.5.3: Lightning v2.5.3
**Published:** 2025-08-13

# Notable changes in this release

<a name="changelog-pytorch"></a>

## PyTorch Lightning

<details open><summary>Changed</summary>

- Added `save_on_exception` option to `ModelCheckpoint` Callback ([#20916](https://github.com/Lightning-AI/pytorch-lightning/pull/20916))
- Allow `dataloader_idx_` in log names when `add_dataloader_idx=False` ([#20987](https://github.com/Lightning-AI/pytorch-lightning/pull/20987))
- Allow returning `ONNXProgram` when calling `to_onnx(dynamo=True)` ([#20811](https://github.com/Lightning-AI/pytorch-lightning/pull/20811))
- Extended support for general mappings being returned from `training_step` when using manual optimization ([#21011](https://github.com/Lightning-AI/pytorch-lightning/pull/21011))

</details>

<details open><summary>Fixed</summary>

- Fixed Allowing trainer to accept CUDAAccelerator instance as accelerator with FSDP strategy ([#20964](https://github.com/Lightning-AI/pytorch-lightning/pull/20964))
- Fixed progress bar console clearing for Rich `14.1+` ([#21016](https://github.com/Lightning-AI/pytorch-lightning/pull/21016))
- Fixed `AdvancedProfiler` to handle nested profiling actions for Python 3.12+ ([#20809](https://github.com/Lightning-AI/pytorch-lightning/pull/20809))
- Fixed `rich` progress bar error when resume training ([#21000](https://github.com/Lightning-AI/pytorch-lightning/pull/21000))
- Fixed double iteration bug when resumed from a checkpoint. ([#20775](https://github.com/Lightning-AI/pytorch-lightning/pull/20775))
- Fixed support for more dtypes in `ModelSummary` ([#21034](https://github.com/Lightning-AI/pytorch-lightning/pull/21034))
- Fixed metrics in `RichProgressBar` being updated according to user provided `refresh_rate` ([#21032](https://github.com/Lightning-AI/pytorch-lightning/pull/21032))
- Fixed `save_last` behavior in the absence of validation ([#20960](https://github.com/Lightning-AI/pytorch-lightning/pull/20960))
- Fixed integration between `LearningRateFinder` and `EarlyStopping` ([#21056](https://github.com/Lightning-AI/pytorch-lightning/pull/21056))
- Fixed gradient calculation in `lr_finder` for `mode="exponential"`  ([#21055](https://github.com/Lightning-AI/pytorch-lightning/pull/21055))
- Fixed `save_hyperparameters` crashing with `dataclasses` using `init=False` fields ([#21051](https://github.com/Lightning-AI/pytorch-lightning/pull/21051))

</details>

<a name="changelog-fabric"></a>

## Lightning Fabric

<details open><summary>Changed</summary>

- Enable "auto" for `devices` and `accelerator` as CLI arguments ([#20913](https://github.com/Lightning-AI/pytorch-lightning/pull/20913))
- Raise ValueError when seed is `out-of-bounds` or `cannot be cast to int` ([#21029](https://github.com/Lightning-AI/pytorch-lightning/pull/21029))

</details>

<details open><summary>Fixed</summary>

- Fixed remove extra `name` parameter in accelerator registry decorator ([#20975](https://github.com/Lightning-AI/pytorch-lightning/pull/20975))
- Fixed XLA strategy to add support for `global_ordinal`, `local_ordinal`, `world_size` which came instead of deprecated methods ([#20852](https://github.com/Lightning-AI/pytorch-lightning/issues/20852))

</details>


</br>

**Full commit list**: [2.5.2 -> 2.5.3](https://github.com/Lightning-AI/pytorch-lightning/compare/2.5.2...2.5.3)

<a name="contributors"></a>

# Contributors

We thank **all folks** who submitted issues, features, fixes and doc changes. It's the only way we can **collectively** make Lightning :zap: better for everyone, nice job!

In particular, we would like to thank the authors of the pull-requests above, in no particular order:

@baskrahmer, @bhimrazy, @deependujha, @fnhirwa, @GdoongMathew, @jonathanking, @relativityhd, @rittik9, @SkafteNicki, @sudiptob2, @vsey, @YgLK

Thank you :heart: and we hope you'll keep them coming!

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.3)

---

## 2.5.2: Lightning v2.5.2
**Published:** 2025-06-20

# Notable changes in this release

<a name="changelog-pytorch"></a>

## PyTorch Lightning

<details open><summary>Changed</summary>

- Add `enable_autolog_hparams` argument to Trainer ([#20593](https://github.com/Lightning-AI/pytorch-lightning/pull/20593))
- Add `toggled_optimizer(optimizer)` method to the LightningModule, which is a context manager version of `toggle_optimize` and `untoggle_optimizer` ([#20771](https://github.com/Lightning-AI/pytorch-lightning/pull/20771))
- For cross-device local checkpoints, instruct users to install `fsspec>=2025.5.0` if unavailable ([#20780](https://github.com/Lightning-AI/pytorch-lightning/pull/20780))
- Check param is of `nn.Parameter` type for pruning sanitization ([#20783](https://github.com/Lightning-AI/pytorch-lightning/pull/20783))

</details>

<details open><summary>Fixed</summary>

- Fixed `save_hyperparameters` not working correctly with `LightningCLI` when there are parsing links applied on instantiation ([#20777](https://github.com/Lightning-AI/pytorch-lightning/pull/20777))
- Fixed `logger_connector` has an edge case where step can be a float ([#20692](https://github.com/Lightning-AI/pytorch-lightning/pull/20692))
- Fixed Synchronize SIGTERM Handling in DDP to Prevent Deadlocks ([#20825](https://github.com/Lightning-AI/pytorch-lightning/pull/20825))
- Fixed case-sensitive model name ([#20661](https://github.com/Lightning-AI/pytorch-lightning/pull/20661))
- CLI: resolve jsonargparse deprecation warning ([#20802](https://github.com/Lightning-AI/pytorch-lightning/pull/20802))
- Fix: move `check_inputs` to the target device if available during `to_torchscript` ([#20873](https://github.com/Lightning-AI/pytorch-lightning/pull/20873))
- Fixed progress bar display to correctly handle iterable dataset and `max_steps` during training ([#20869](https://github.com/Lightning-AI/pytorch-lightning/pull/20869))
- Fixed problem for silently supporting `jsonnet` ([#20899](https://github.com/Lightning-AI/pytorch-lightning/pull/20899))

</details>

<a name="changelog-fabric"></a>

## Lightning Fabric

<details open><summary>Changed</summary>

- Ensure correct device is used for autocast when mps is selected as Fabric accelerator ([#20876](https://github.com/Lightning-AI/pytorch-lightning/pull/20876))

</details>

<details open><summary>Removed</summary>

- Fix: `TransformerEnginePrecision` conversion for layers with `bias=False` ([#20805](https://github.com/Lightning-AI/pytorch-lightning/pull/20805))

</details>

</br>

**Full commit list**: [2.5.1 -> 2.5.2](https://github.com/Lightning-AI/pytorch-lightning/compare/2.5.1...2.5.2)

<a name="contributors"></a>

# Contributors

We thank **all folks** who submitted issues, features, fixes, and doc changes. It's the only way we can **collectively** make Lightning :zap: better for everyone, nice job!

In particular, we would like to thank the authors of the pull-requests above, in no particular order:

@adamjstewart, @Armannas, @bandpooja, @Borda, @chanokin, @duydl, @GdoongMathew, @KAVYANSHTYAGI, @mauvilsa, @muthissar, @rustamzh, @siemdejong

Thank you :heart: and we hope you'll keep them coming!


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.2)

---

## 2.5.1.post0: Lightning v2.5.1.post
**Published:** 2025-04-25

**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.5.1...2.5.1.post0

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.1.post0)

---

## 2.5.1: Lightning v2.5.1
**Published:** 2025-03-19

# Changes

<a name="changelog-pytorch"></a>

## PyTorch Lightning

<details open><summary>Changed</summary>

- Allow LightningCLI to use a customized argument parser class ([#20596](https://github.com/Lightning-AI/pytorch-lightning/pull/20596))
- Change `wandb` default x-axis to `tensorboard`'s `global_step` when `sync_tensorboard=True` ([#20611](https://github.com/Lightning-AI/pytorch-lightning/pull/20611))
- Added a new `checkpoint_path_prefix` parameter to the MLflow logger which can control the path to where the MLflow artifacts for the model checkpoints are stored ([#20538](https://github.com/Lightning-AI/pytorch-lightning/pull/20538))
- CometML logger was updated to support the recent Comet SDK ([#20275](https://github.com/Lightning-AI/pytorch-lightning/pull/20275))
- bump: testing with latest `torch` 2.6 ([#20509](https://github.com/Lightning-AI/pytorch-lightning/pull/20509))

</details>

<details open><summary>Fixed</summary>

- Fixed CSVLogger logging hyperparameter at every write which increases latency  ([#20594](https://github.com/Lightning-AI/pytorch-lightning/pull/20594))
- Fixed OverflowError when resuming from checkpoint with an iterable dataset ([#20565](https://github.com/Lightning-AI/pytorch-lightning/issues/20565))
- Fixed swapped `_R_co` and `_P` to prevent type error ([#20508](https://github.com/Lightning-AI/pytorch-lightning/issues/20508))
- Always call `WandbLogger.experiment` first in `_call_setup_hook` to ensure `tensorboard` logs can sync to `wandb` ([#20610](https://github.com/Lightning-AI/pytorch-lightning/pull/20610))
- Fixed TBPTT example ([#20528](https://github.com/Lightning-AI/pytorch-lightning/pull/20528))
- Fixed test compatibility as AdamW became a subclass of Adam ([#20574](https://github.com/Lightning-AI/pytorch-lightning/pull/20574))
- Fixed file extension of model checkpoints uploaded by NeptuneLogger ([#20581](https://github.com/Lightning-AI/pytorch-lightning/pull/20581))
- Reset trainer variable `should_stop` when `fit` is called ([#19177](https://github.com/Lightning-AI/pytorch-lightning/pull/19177))
- Fixed making `WandbLogger` upload models from all `ModelCheckpoint` callbacks, not just one ([#20191](https://github.com/Lightning-AI/pytorch-lightning/pull/20191))
- Error when logging to MLFlow deleted experiment ([#20556](https://github.com/Lightning-AI/pytorch-lightning/pull/20556))

</details>

<a name="changelog-fabric"></a>

## Lightning Fabric

<details open><summary>Changed</summary>

- Added logging support for a list of dicts without collapsing to a single key ([#19957](https://github.com/Lightning-AI/pytorch-lightning/issues/19957))
- bump: testing with latest `torch` 2.6 ([#20509](https://github.com/Lightning-AI/pytorch-lightning/pull/20509))

</details>

<details open><summary>Removed</summary>

- Removed legacy support for `lightning run model`; use `fabric run` instead. ([#20588](https://github.com/Lightning-AI/pytorch-lightning/pull/20588))

</details>


</br>

**Full commit list**: [2.5.0 -> 2.5.1](https://github.com/Lightning-AI/pytorch-lightning/compare/2.5.0...2.5.1)

<a name="contributors"></a>

# Contributors

We thank **all folks** who submitted issues, features, fixes and doc changes. It's the only way we can **collectively** make Lightning :zap: better for everyone, nice job!

In particular, we would like to thank the authors of the pull-requests above, in no particular order:

@benglewis, @Borda, @cgebbe, @duydl, @haifeng-jin, @japdubengsub, @justusschock, @lantiga, @mauvilsa, @millskyle, @ringohoffman, @ryan597, @senarvi, @TresYap

Thank you :heart: and we hope you'll keep them coming!


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.1)

---

## 2.5.0.post0: Lightning v2.5 post0
**Published:** 2024-12-21

**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.5.0...2.5.0.post0

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.0.post0)

---

## 2.5.0: Lightning v2.5
**Published:** 2024-12-20

[Lightning AI](https://lightning.ai) :zap: is excited to announce the release of Lightning 2.5. 

Lightning 2.5 comes with improvements on several fronts, with **zero** API changes. Our users love it stable, we keep it stable :smile:.

Talking about love :heart:, the `lightning`, `pytorch-lightning` and `lightning-fabric` packages are collectively getting more than **10M downloads per month** :open_mouth:, for a total of over **180M downloads** :exploding_head: since the early days . It's incredible to see PyTorch Lightning getting such a strong adoption across the industry and the sciences.

Release 2.5 embraces PyTorch 2.5, and it marks some of its more recent directions as officially supported, namely tensor subclass-based APIs like [Distributed Tensors](https://pytorch.org/docs/stable/distributed.tensor.html) and [TorchAO](https://pytorch.org/blog/pytorch-native-architecture-optimization/), in combination with `torch.compile`.

Here's a couple of examples:

<details><summary>Distributed FP8 transformer with PyTorch Lightning</summary>

Full example [here](https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/pytorch/fp8_distributed_transformer)

```python
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.demos import Transformer, WikiText2
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.model = None

    def configure_model(self):
        if self.model is not None:
            return

        with torch.device("meta"):
            model = Transformer(
                vocab_size=self.vocab_size,
                nlayers=16,
                nhid=4096,
                ninp=1024,
                nhead=32,
            )

        float8_config = Float8LinearConfig(
            # pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly  # noqa
            pad_inner_dim=True,
        )

        def module_filter_fn(mod: torch.nn.Module, fqn: str):
            # we skip the decoder because it typically vocabulary size
            # is not divisible by 16 as required by float8
            return fqn != "decoder"

        convert_to_float8_training(model, config=float8_config, module_filter_fn=module_filter_fn)

        for module in model.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                fully_shard(module, mesh=self.device_mesh)

        fully_shard(model, mesh=self.device_mesh)

        self.model = torch.compile(model)

    def training_step(self, batch):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

def train():
    L.seed_everything(42)

    dataset = WikiText2()
    train_dataloader = DataLoader(dataset, num_workers=8, batch_size=1)

    model = LanguageModel(vocab_size=dataset.vocab_size)

    mp_strategy = ModelParallelStrategy(
        data_parallel_size=4,
        tensor_parallel_size=1,
    )

    trainer = L.Trainer(strategy=mp_strategy, max_steps=100, precision="bf16-true", accumulate_grad_batches=8)

    trainer.fit(model, train_dataloader)

    trainer.print(torch.cuda.memory_summary())

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    train()
```

</details>

<details><summary>Distributed FP8 transformer with Fabric</summary>

Full example [here](https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/fabric/fp8_distributed_transformer)

```python
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.strategies import ModelParallelStrategy
from lightning.pytorch.demos import Transformer, WikiText2
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from tqdm import tqdm

def configure_model(model: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
    float8_config = Float8LinearConfig(
        # pip install -U --index-url <https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/> triton-nightly  # noqa
        pad_inner_dim=True,
    )

    def module_filter_fn(mod: torch.nn.Module, fqn: str):
        # we skip the decoder because it typically vocabulary size
        # is not divisible by 16 as required by float8
        return fqn != "decoder"

    convert_to_float8_training(model, config=float8_config, module_filter_fn=module_filter_fn)

    for module in model.modules():
        if isinstance(module, (torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer)):
            fully_shard(module, mesh=device_mesh)

    fully_shard(model, mesh=device_mesh)

    return torch.compile(model)

def train():
    L.seed_everything(42)

    batch_size = 8
    micro_batch_size = 1

    max_steps = 100

    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=8, batch_size=micro_batch_size)

    with torch.device("meta"):
        model = Transformer(
            vocab_size=dataset.vocab_size,
            nlayers=16,
            nhid=4096,
            ninp=1024,
            nhead=32,
        )

    strategy = ModelParallelStrategy(data_parallel_size=4, tensor_parallel_size=1, parallelize_fn=configure_model)

    fabric = L.Fabric(precision="bf16-true", strategy=strategy)
    fabric.launch()

    model = fabric.setup(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = fabric.setup_optimizers(optimizer)

    dataloader = fabric.setup_dataloaders(dataloader)

    iterable = tqdm(enumerate(dataloader), total=len(dataloader)) if fabric.is_global_zero else enumerate(dataloader)

    steps = 0

    for i, batch in iterable:
        input, target = batch

        is_accumulating = i % (batch_size // micro_batch_size) != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            output = model(input, target)
            loss = F.nll_loss(output, target.view(-1))
            fabric.backward(loss)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            steps += 1

        if fabric.is_global_zero:
            iterable.set_postfix_str(f"train_loss={loss.item():.2f}")

        if steps == max_steps:
            break

    fabric.print(torch.cuda.memory_summary())

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    train()
```

</details>

As these examples show, it's now easier than ever to take your PyTorch Lightning module and run it with **FSDP2 and/or tensor parallelism in FP8 precision**, using the `ModelParallelStrategy` we introduced in 2.4.

Also note the use of distributed tensor APIs, TorchAO APIs, and `torch.compile` directly in the `configure_model` hook (or in the parallelize function in Fabric's `ModelParallelStrategy`), as opposed to the `LightningModule` as a whole. The advantage with this approach is that you can just **copy-paste the parallelize functions** that come with native PyTorch models directly in `configure_model` and get the same effect, no head-scratching involved :nerd_face:.

Talking about head scratching, we also made a pass at the PyTorch Lightning internals and **hardened** the parts where we keep track of **progress counters** during training, validation, testing, as well as learning rate scheduling, in relation to **resuming from checkpoints**. We now made sure there are no (to the best of our knowledge) edge cases where stopping and resuming from checkpoints can change the sequence of loops or other internal states. **Fault tolerance for the win** :partying_face:!

Alright! Feel free to take a look at the **full changelog** below.

And of course: the best way to use PyTorch Lightning and Fabric is through [Lightning Studio](https://lightning.ai/) :zap:. Access GPUs, train models, deploy and more with **zero setup**. Focus on data and models - not infrastructure.

<a name="changelog"></a>

# Changes

<a name="changelog-pytorch"></a>

## PyTorch Lightning

<details open><summary>Added</summary>

- Added `step` parameter to `TensorBoardLogger.log_hyperparams` to visualize changes during training ([#20176](https://github.com/Lightning-AI/pytorch-lightning/pull/20176))
- Added `str` method to datamodule ([#20301](https://github.com/Lightning-AI/pytorch-lightning/pull/20301))
- Added timeout to DeepSpeedStrategy ([#20474](https://github.com/Lightning-AI/pytorch-lightning/pull/20474))
- Added doc for Truncated Back-Propagation Through Time ([#20422](https://github.com/Lightning-AI/pytorch-lightning/pull/20422))
- Added FP8 + FSDP2 + torch.compile examples for PyTorch Lightning ([#20440](https://github.com/Lightning-AI/pytorch-lightning/pull/20440))
- Added profiling to `Trainer.save_checkpoint` ([#20405](https://github.com/Lightning-AI/pytorch-lightning/pull/20405))
- Added after_instantiate_classes hook to CLI ([#20401](https://github.com/Lightning-AI/pytorch-lightning/pull/20401))

</details>

<details open><summary>Changed</summary>

- Updated checkpointing documentation to mark `resume_from_checkpoint` as deprecated ([#20477](https://github.com/Lightning-AI/pytorch-lightning/pull/20477))
- Made plugin type checks more flexible ([#20186](https://github.com/Lightning-AI/pytorch-lightning/pull/20186))
- Changed seeding NumPy using `np.random.SeedSequence()` in `pl_worker_init_function()` to robustly seed NumPy-dependent dataloader workers ([#20369](https://github.com/Lightning-AI/pytorch-lightning/pull/20369))
- Allowed callbacks to be restored not just during training ([#20403](https://github.com/Lightning-AI/pytorch-lightning/pull/20403))
- Changed LightningCLI tests to account for future fix in jsonargparse ([#20372](https://github.com/Lightning-AI/pytorch-lightning/pull/20372))
- Bumped PyTorch to version `2.5` ([#20351](https://github.com/Lightning-AI/pytorch-lightning/pull/20351))
- Decoupled checkpoint artifact path from model artifact path ([#20325](https://github.com/Lightning-AI/pytorch-lightning/pull/20325))
- Updated BitsAndBytes version ([#20313](https://github.com/Lightning-AI/pytorch-lightning/pull/20313))
- Changed merging of hparams when logging to ignore parameter names that start with an underscore `_` ([#20221](https://github.com/Lightning-AI/pytorch-lightning/pull/20221))
- Re-enabled passing `BytesIO` as path in `.to_onnx()` ([#20172](https://github.com/Lightning-AI/pytorch-lightning/pull/20172))

</details>

<details open><summary>Removed</summary>

- Removed `List[int]` as input type for Trainer when `accelerator="cpu"` ([#20399](https://github.com/Lightning-AI/pytorch-lightning/pull/20399))

</details>

<details open><summary>Fixed</summary>

- Fixed UnboundLocalError when using the predict method with return_predictions=False. ([#20484](https://github.com/Lightning-AI/pytorch-lightning/pull/20484))
- Fixed use of `convert_module` in FSDP to avoid using more memory than necessary during initialization ([#20323](https://github.com/Lightning-AI/pytorch-lightning/pull/20323))
- Fixed TypeError in `configure_optimizers` when running with `ReduceLROnPlateau` ([#20471](https://github.com/Lightning-AI/pytorch-lightning/pull/20471))
- Fixed return type in `configure_optimizers` example ([#20420](https://github.com/Lightning-AI/pytorch-lightning/pull/20420))
- Fixed in ncorrect URI prefix stripping in MLFlowLogger ([#20365](https://github.com/Lightning-AI/pytorch-lightning/pull/20365))
- Fixed shuffling behavior when using a custom sampler in data module ([#20327](https://github.com/Lightning-AI/pytorch-lightning/pull/20327))
- Ensured restarting from checkpoints leads to consistent internal counters compared to uninterrupted training ([#20379](https://github.com/Lightning-AI/pytorch-lightning/pull/20379))
- Fixed LightningCLI failing when both module and data module save hyperparameters due to conflicting internal `_class_path` parameter ([#20221](https://github.com/Lightning-AI/pytorch-lightning/pull/20221))

</details>

<a name="changelog-fabric"></a>

## Lightning Fabric

<details open><summary>Added</summary>

- Added `step` parameter to `TensorBoardLogger.log_hyperparams` to visualize changes during training ([#20176](https://github.com/Lightning-AI/pytorch-lightning/pull/20176))
- Added timeout to DeepSpeedStrategy ([#20474](https://github.com/Lightning-AI/pytorch-lightning/pull/20474))
- Added FP8 + FSDP2 + torch.compile examples for Fabric ([#20440](https://github.com/Lightning-AI/pytorch-lightning/pull/20440))
- Added RTX 4080 super to chips dictionary ([#20285](https://github.com/Lightning-AI/pytorch-lightning/pull/20285))
- Added device property to lazy load functionality ([#20183](https://github.com/Lightning-AI/pytorch-lightning/pull/20183))
- Added `ddp_find_unused_parameters_true` alias in Fabric's DDPStrategy ([#20125](https://github.com/Lightning-AI/pytorch-lightning/pull/20125))

</details>

<details open><summary>Changed</summary>

- Changed seeding NumPy using `np.random.SeedSequence()` in `pl_worker_init_function()` to robustly seed NumPy-dependent dataloader workers ([#20369](https://github.com/Lightning-AI/pytorch-lightning/pull/20369))
- Bumped PyTorch to version `2.5` ([#20351](https://github.com/Lightning-AI/pytorch-lightning/pull/20351))
- Update BitsAndBytes version ([#20313](https://github.com/Lightning-AI/pytorch-lightning/pull/20313))

</details>

<details open><summary>Removed</summary>

- Nothing to see here :smile:

</details>

<details open><summary>Fixed</summary>

- Fixed use of `convert_module` in FSDP to avoid using more memory than necessary during initialization ([#20323](https://github.com/Lightning-AI/pytorch-lightning/pull/20323))

</details>

</br>

**Full commit list**: [2.4.0 -> 2.5.0](https://github.com/Lightning-AI/pytorch-lightning/compare/2.4.0...2.5.0)

<a name="contributors"></a>

# Contributors

We thank **all folks** who submitted issues, features, fixes and doc changes. It's the only way we can **collectively** make Lightning :zap: better for everyone, nice job!

In particular, we would like to thank the authors of the pull-requests above, in no particular order:

@ringohoffman @MrWhatZitToYaa @jedyang97 @chualanagit @lantiga @AlessandroW @kazuar @t-vi @01AbhiSingh @WangYue0000 @amorehead @EricCousineau-TRI @mauvilsa @Borda @pete-mcelroy @ali-alshaar7 @GdoongMathew @farhadrgh @tshu-w @LukasSalchow @awindmann @dadwadw233 @qingquansong

Thank you :heart: and we hope you'll keep them coming!


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.0)

---

## 2.5.0rc0: Lightning 2.5 RC
**Published:** 2024-12-12
**Pre-release**



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.5.0rc0)

---

## 2.4.0: Lightning v2.4
**Published:** 2024-08-07

[Lightning AI](https://lightning.ai) :zap: is excited to announce the release of Lightning 2.4. This is mainly a compatibility upgrade for PyTorch 2.4 and Python 3.12, with a sprinkle of a few features and bug fixes.

**Did you know?** The Lightning philosophy extends beyond a boilerplate-free deep learning framework: We've been hard at work bringing you [Lightning Studio](https://lightning.ai/). Code together, prototype, train, deploy, host AI web apps. All from your browser, with zero setup.

<a name="changelog"></a>
# Changes

<a name="changelog-pytorch"></a>
## PyTorch Lightning

<details open><summary>Added</summary>

- Made saving non-distributed checkpoints fully atomic ([#20011](https://github.com/Lightning-AI/pytorch-lightning/pull/20011))
- Added `dump_stats` flag to `AdvancedProfiler` ([#19703](https://github.com/Lightning-AI/pytorch-lightning/issues/19703))
- Added a flag `verbose` to the `seed_everything()` function ([#20108](https://github.com/Lightning-AI/pytorch-lightning/pull/20108))
- Added support for PyTorch 2.4 ([#20010](https://github.com/Lightning-AI/pytorch-lightning/pull/20010))
- Added support for Python 3.12 ([20078](https://github.com/Lightning-AI/pytorch-lightning/pull/20078))
- The `TQDMProgressBar` now provides an option to retain prior training epoch bars ([#19578](https://github.com/Lightning-AI/pytorch-lightning/pull/19578))
- Added the count of modules in train and eval mode to the printed `ModelSummary` table ([#20159](https://github.com/Lightning-AI/pytorch-lightning/pull/20159))

</details>

<details open><summary>Changed</summary>

- Triggering KeyboardInterrupt (Ctrl+C) during `.fit()`, `.evaluate()`, `.test()` or `.predict()` now terminates all processes launched by the Trainer and exits the program ([#19976](https://github.com/Lightning-AI/pytorch-lightning/pull/19976))
- Changed the implementation of how seeds are chosen for dataloader workers when using `seed_everything(..., workers=True)` ([#20055](https://github.com/Lightning-AI/pytorch-lightning/pull/20055))
- NumPy is no longer a required dependency ([#20090](https://github.com/Lightning-AI/pytorch-lightning/issues/20090))

</details>

<details open><summary>Removed</summary>

- Removed support for PyTorch 2.1 ([#20009](https://github.com/Lightning-AI/lightning/pull/20009))
- Removed support for Python 3.8 ([#20071](https://github.com/Lightning-AI/lightning/pull/20071))

</details>

<details open><summary>Fixed</summary>

- Avoid LightningCLI saving hyperparameters with `class_path` and `init_args` since this would be a breaking change ([#20068](https://github.com/Lightning-AI/pytorch-lightning/pull/20068))
- Fixed an issue that would cause too many printouts of the seed info when using `seed_everything()` ([#20108](https://github.com/Lightning-AI/pytorch-lightning/pull/20108))
- Fixed `_LoggerConnector`'s `_ResultMetric` to move all registered keys to the device of the logged value if needed ([#19814](https://github.com/Lightning-AI/pytorch-lightning/issues/19814))
- Fixed `_optimizer_to_device` logic for special 'step' key in optimizer state causing performance regression ([#20019](https://github.com/Lightning-AI/lightning/pull/20019))
- Fixed parameter counts in `ModelSummary` when model has distributed parameters (DTensor) ([#20163](https://github.com/Lightning-AI/pytorch-lightning/pull/20163))

</details>

<a name="changelog-fabric"></a>
## Lightning Fabric

<details open><summary>Added</summary>

- Made saving non-distributed checkpoints fully atomic ([#20011](https://github.com/Lightning-AI/pytorch-lightning/pull/20011))
- Added a flag `verbose` to the `seed_everything()` function ([#20108](https://github.com/Lightning-AI/pytorch-lightning/pull/20108))
- Added support for PyTorch 2.4 ([#20028](https://github.com/Lightning-AI/pytorch-lightning/pull/20028))
- Added support for Python 3.12 ([20078](https://github.com/Lightning-AI/pytorch-lightning/pull/20078))

</details>

<details open><summary>Changed</summary>

- Changed the implementation of how seeds are chosen for dataloader workers when using `seed_everything(..., workers=True)` ([#20055](https://github.com/Lightning-AI/pytorch-lightning/pull/20055))
- NumPy is no longer a required dependency ([#20090](https://github.com/Lightning-AI/pytorch-lightning/issues/20090))

</details>

<details open><summary>Removed</summary>

- Removed support for PyTorch 2.1 ([#20009](https://github.com/Lightning-AI/lightning/pull/20009))
- Removed support for Python 3.8 ([#20071](https://github.com/Lightning-AI/lightning/pull/20071))

</details>

<details open><summary>Fixed</summary>

- Fixed an attribute error when loading a checkpoint into a quantized model using the `_lazy_load()` function ([#20121](https://github.com/Lightning-AI/lightning/pull/20121))
- Fixed `_optimizer_to_device` logic for special 'step' key in optimizer state causing performance regression ([#20019](https://github.com/Lightning-AI/lightning/pull/20019))

</details>

</br>

**Full commit list**: [2.3.0 -> 2.4.0](https://github.com/Lightning-AI/pytorch-lightning/compare/2.3.0...2.4.0)

<a name="contributors"></a>
# Contributors

We thank all our contributors who submitted pull requests for features, bug fixes and documentation updates.

### New Contributors
* @SamuelLarkin made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19969
* @liambsmith made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19986
* @EtayLivne made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19915
* @elmuz made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19998
* @swyo made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19982
* @corwinjoy made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/20011
* @omahs made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19979
* @linbo0518 made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/20040
* @01AbhiSingh made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/20055
* @K-H-Ismail made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/20099
* @adosar made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/20146
* @jojje made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19578

### Did you know?

Chuck Norris can solve NP-hard problems in polynomial time. In fact, any problem is easy when Chuck Norris solves it.

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.4.0)

---

## 2.3.3: Patch release v2.3.3
**Published:** 2024-07-08

This release removes the code from the main `lightning` package that was reported in [CVE-2024-5980](https://github.com/advisories/GHSA-mr7h-w2qc-ffc2).

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.3.3)

---

## 2.3.2: Patch release v2.3.2
**Published:** 2024-07-04

Includes a minor bugfix that avoids a conflict with the entrypoint command with another package [#20041](https://github.com/Lightning-AI/pytorch-lightning/pull/20041).



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.3.2)

---

## 2.3.1: Patch release v2.3.1
**Published:** 2024-06-27

Includes minor bugfixes and stability improvements.


**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.3.0...2.3.1

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.3.1)

---

## 2.3.0: Lightning v2.3: Tensor Parallelism and 2D Parallelism
**Published:** 2024-06-13

[Lightning AI](https://lightning.ai) is excited to announce the release of Lightning 2.3 :zap:

**Did you know?** The Lightning philosophy extends beyond a boilerplate-free deep learning framework: We've been hard at work bringing you [Lightning Studio](https://lightning.ai/). Code together, prototype, train, deploy, host AI web apps. All from your browser, with zero setup.

This release introduces experimental support for Tensor Parallelism and 2D Parallelism, [PyTorch 2.3](https://pytorch.org/blog/pytorch2-3/) support, and several bugfixes and stability improvements.


- [Highlights](#highlights)
    - [Tensor Parallelism (beta)](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#highlights-tensor-parallel)
    - [2D Parallelism (beta)](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#highlights-2d-parallel)
    - [Training Mode in Model Summary](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#highlights-model-summary)
    - [Special Forward Methods in Fabric](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#highlights-forward-methods)
- [Notable Changes](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#bc-changes)
- [Full Changelog](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#changelog)
    - [PyTorch Lightning](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#changelog-pytorch)
    - [Lightning Fabric](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#changelog-fabric)
- [Contributors](https://github.com/Lightning-AI/lightning/releases/tag/2.3.0#contributors)


<a name="highlights"></a>
# Highlights

<a name="highlights-tensor-parallel"></a>
## Tensor Parallelism (beta)

Tensor parallelism (TP) is a technique that splits up the computation of selected layers across GPUs to save memory and speed up distributed models. To enable TP as well as other forms of parallelism, we introduce a `ModelParallelStrategy` for both Lightning Trainer and Fabric. Under the hood, TP is enabled through new experimental PyTorch APIs like [DTensor](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md) and [`torch.distributed.tensor.parallel`](https://pytorch.org/docs/stable/distributed.tensor.parallel.html).

### PyTorch Lightning

Enabling TP in a model with PyTorch Lightning requires you to implement the `LightningModule.configure_model()` method where you convert selected layers of a model to paralellized layers. This is an advanced feature, because it requires a deep understanding of the model architecture. Open the [tutorial Studio](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning) to learn the basics of Tensor Parallelism.

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

&nbsp;

```python
import lightning as L
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module


# 1. Implement the `configure_model()` method in LightningModule
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FeedForward(8192, 8192)

    def configure_model(self):
        # Lightning will set up a `self.device_mesh` for you
        tp_mesh = self.device_mesh["tensor_parallel"]
        # Use PyTorch's distributed tensor APIs to parallelize the model
        plan = {
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        }
        parallelize_module(self.model, tp_mesh, plan)

    def training_step(self, batch):
        ...


# 2. Create the strategy
strategy = ModelParallelStrategy()

# 3. Configure devices and set the strategy in Trainer
trainer = L.Trainer(accelerator="cuda", devices=2, strategy=strategy)
trainer.fit(...)

```

<details><summary>Full training example (requires at least 2 GPUs).</summary>
    
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module

import lightning as L
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.pytorch.strategies import ModelParallelStrategy


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = FeedForward(8192, 8192)

    def configure_model(self):
        if self.device_mesh is None:
            return

        # Lightning will set up a `self.device_mesh` for you
        tp_mesh = self.device_mesh["tensor_parallel"]
        # Use PyTorch's distributed tensor APIs to parallelize the model
        plan = {
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        }
        parallelize_module(self.model, tp_mesh, plan)

    def training_step(self, batch):
        output = self.model(batch)
        loss = output.sum()
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-3)

    def train_dataloader(self):
        # Trainer configures the sampler automatically for you such that
        # all batches in a tensor-parallel group are identical
        dataset = RandomDataset(8192, 64)
        return torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=2)


strategy = ModelParallelStrategy()
trainer = L.Trainer(
    accelerator="cuda",
    devices=2,
    strategy=strategy,
    max_epochs=1,
)

model = LitModel()
trainer.fit(model)

trainer.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
```

</details>

</br>

### Lightning Fabric

Applying TP in a model with Fabric requires you to implement a special function where you convert selected layers of a model to paralellized layers. This is an advanced feature, because it requires a deep understanding of the model architecture. Open the [tutorial Studio](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-lightning-fabric) to learn the basics of Tensor Parallelism.

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-lightning-fabric">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

&nbsp;

```python
import lightning as L
from lightning.fabric.strategies import ModelParallelStrategy
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module


# 1. Implement the parallelization function for your model
def parallelize_feedforward(model, device_mesh):
    # Lightning will set up a device mesh for you
    tp_mesh = device_mesh["tensor_parallel"]
    # Use PyTorch's distributed tensor APIs to parallelize the model
    plan = {
        "w1": ColwiseParallel(),
        "w2": RowwiseParallel(),
        "w3": ColwiseParallel(),
    }
    parallelize_module(model, tp_mesh, plan)
    return model


# 2. Pass the parallelization function to the strategy
strategy = ModelParallelStrategy(parallelize_fn=parallelize_feedforward)

# 3. Configure devices and set the strategy in Fabric
fabric = L.Fabric(accelerator="cuda", devices=2, strategy=strategy)
fabric.launch()
```


<details><summary>Full training example (requires at least 2 GPUs).</summary>
    
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.parallel import parallelize_module

import lightning as L
from lightning.pytorch.demos.boring_classes import RandomDataset
from lightning.fabric.strategies import ModelParallelStrategy


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def parallelize_feedforward(model, device_mesh):
    # Lightning will set up a device mesh for you
    tp_mesh = device_mesh["tensor_parallel"]
    # Use PyTorch's distributed tensor APIs to parallelize the model
    plan = {
        "w1": ColwiseParallel(),
        "w2": RowwiseParallel(),
        "w3": ColwiseParallel(),
    }
    parallelize_module(model, tp_mesh, plan)
    return model


strategy = ModelParallelStrategy(parallelize_fn=parallelize_feedforward)
fabric = L.Fabric(accelerator="cuda", devices=2, strategy=strategy)
fabric.launch()

# Initialize the model
model = FeedForward(8192, 8192)
model = fabric.setup(model)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
optimizer = fabric.setup_optimizers(optimizer)

# Define dataset/dataloader
dataset = RandomDataset(8192, 64)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
dataloader = fabric.setup_dataloaders(dataloader)

# Simplified training loop
for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = output.sum()
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    fabric.print(f"Iteration {i} complete")

fabric.print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
```

</details>

</br>


<a name="highlights-2d-parallel"></a>
## 2D Parallelism (beta)

Tensor Parallelism by itself can be very effective for efficient inference of very large models. For training, TP is typically combined with other forms of parallelism, such as FSDP, to increase throughput and scalability on large clusters with 100s of GPUs. The new `ModelParallelStrategy` in this release supports the combination of TP + FSDP, which is referred to as 2D parallelism.

For an introduction to this feature, please also refer to the tutorial Studios ([PyTorch Lightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning), [Lightning Fabric](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-lightning-fabric)). At the moment, the PyTorch team is reimplementing FSDP under the name [FSDP2](https://github.com/pytorch/pytorch/issues/114299) with the aim to make it compose well with other parallelisms such as TP. Therefore, for the experimental 2D parallelism support, you'll need to switch to using FSDP2 with the new `ModelParallelStrategy`. Please refer to our docs ([PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/advanced/model_parallel/tp_fsdp.html), [Lightning Fabric](https://lightning.ai/docs/fabric/latest/advanced/model_parallel/tp_fsdp.html)) and stay tuned for future releases as these APIs mature.


<a name="highlights-model-summary"></a>
## Training Mode in Model Summary

The model summary table that gets displayed when you run `Trainer.fit()` now contains a new column "Mode" that shows the training mode each layer is in ([#19468](https://github.com/Lightning-AI/lightning/pull/19468)).

```
  | Name                 | Type            | Params | Mode 
-----------------------------------------------------------------
0 | model                | Sam             | 93.7 M | train
1 | model.image_encoder  | ImageEncoderViT | 89.7 M | eval 
2 | model.prompt_encoder | PromptEncoder   | 6.2 K  | train
3 | model.mask_decoder   | MaskDecoder     | 4.1 M  | train
-----------------------------------------------------------------
93.7 M    Trainable params
0         Non-trainable params
93.7 M    Total params
374.942   Total estimated model params size (MB)
```

A module in PyTorch is always either in `train` (default) or `eval` mode.
This improvement should give users more visibility into the state of their model and help debug issues, for example when you need to make sure certain layers of the model are frozen.


<a name="highlights-forward-methods"></a>
## Special Forward Methods in Fabric

Until now, Lightning Fabric warned the user in case the forward pass of the model or a subset of its modules was conducted through methods other than the dedicated `forward` method of the PyTorch module. The reason for this is that PyTorch needs to run special hooks in case of DDP/FSDP and other strategies to function properly, and not running through the real `forward` method would skip these hooks and lead to correctness issues.

In Lightning Fabric 2.3, we added a [feature to explicitly mark alternative forward methods](https://lightning.ai/docs/fabric/latest/api/wrappers.html#using-methods-other-than-forward-for-computation) so that Fabric can add the necessary rerouting behind the scenes:

```python
import lightning as L

fabric = L.Fabric(devices=2, strategy="ddp")
fabric.launch()

model = MyModel()
model = fabric.setup(model)

# OK: Calling the model directly
output = model(input)

# ERROR: Calling another method that calls forward indirectly
prediction = model.generate(input)

# New: Mark special forward methods explicitly before using them
model.mark_forward_method(model.generate)

# OK: Now can use `model.generate()` in DDP/FSDP without issues
prediction = model.generate(input)
```

Find the [full example](https://lightning.ai/docs/fabric/latest/api/wrappers.html#using-methods-other-than-forward-for-computation) and more details in our docs.


<a name="bc-changes"></a>
# Notable Changes

The 2.0 series of Lightning releases guarantees core API stability: No name changes, argument renaming, hook removals etc. on core interfaces (Trainer, LightningModule, etc.) unless a feature is specifically marked experimental. Here we list a few behavioral changes made in places where the change was justified if it significantly improves the user experience, improves performance, or fixes the correctness of a feature. These changes will likely not impact most users.

### Skipping the training step in DDP

It is no longer allowed to skip `training_step()` by returning `None` in distributed training ([#19918](https://github.com/Lightning-AI/pytorch-lightning/pull/19918)). The following usage was previously possible but would result in unpredictable hangs and timeouts in distributed training:

```python
def training_step(self, batch):
    loss = ...
    if loss.isnan():
        # No longer allowed in multi-GPU!
        # Raises error in Lightning >= 2.3
        return None
    return loss
```

We decided to raise an error if the user attempts to return `None` when running in a multi-GPU setting. 


## Miscellaneous Changes

- Dropped support for PyTorch 1.13 ([#19300](https://github.com/Lightning-AI/lightning/pull/19300)). With every new Lightning release, we add official support for the latest PyTorch stable version and drop the oldest version in our support window.
- The `prepare_data()` hook in `LightningModule` and `LightningDataModule` is now subject to a barrier without timeout to avoid long-running tasks to be interrupted ([#19448](https://github.com/Lightning-AI/lightning/pull/19448)). Similarly, also in Fabric the `Fabric.rank_zero_first` context manager now uses an infinite barrier ([#19448](https://github.com/Lightning-AI/lightning/pull/19448)).

<a name="changelog"></a>
# CHANGELOG

<a name="changelog-pytorch"></a>
## PyTorch Lightning

<details><summary>Added</summary>

- The `ModelSummary` and `RichModelSummary` callbacks now display the training mode of each layer in the column "Mode" ([#19468](https://github.com/Lightning-AI/lightning/pull/19468))
- Added `load_from_checkpoint` support for `LightningCLI` when using dependency injection ([#18105](https://github.com/Lightning-AI/lightning/pull/18105))
- Added robust timer duration parsing with an informative error message when parsing fails ([#19513](https://github.com/Lightning-AI/pytorch-lightning/pull/19513))
- Added `on_exception` hook to `LightningDataModule` ([#19601](https://github.com/Lightning-AI/pytorch-lightning/pull/19601))
- Added support for PyTorch 2.3 ([#19708](https://github.com/Lightning-AI/pytorch-lightning/pull/19708))
- Added `ModelParallelStrategy` to support 2D parallelism ([#19878](https://github.com/Lightning-AI/pytorch-lightning/pull/19878), [#19888](https://github.com/Lightning-AI/pytorch-lightning/pull/19888))
- Added a call to `torch.distributed.destroy_process_group` in atexit handler if process group needs destruction ([#19931](https://github.com/Lightning-AI/pytorch-lightning/pull/19931))
- Added support for configuring hybrid-sharding by passing a tuple for the `FSDPStrategy(device_mesh=...)` argument ([#19504](https://github.com/Lightning-AI/pytorch-lightning/pull/19504))

</details>

<details><summary>Changed</summary>
    
- The `prepare_data()` hook in `LightningModule` and `LightningDataModule` is now subject to a barrier without timeout to avoid long-running tasks to be interrupted ([#19448](https://github.com/Lightning-AI/lightning/pull/19448))
- Relaxed the requirement for custom batch samplers to expose `drop_last` for prediction ([#19678](https://github.com/Lightning-AI/pytorch-lightning/pull/19678))
- It is no longer allowed to skip `training_step()` by returning `None` in distributed training ([#19918](https://github.com/Lightning-AI/pytorch-lightning/pull/19918))

</details>

<details><summary>Removed</summary>

- Removed the Bagua integration (`Trainer(strategy="bagua")`) ([#19445](https://github.com/Lightning-AI/lightning/pull/19445))
- Removed support for PyTorch 1.13 ([#19706](https://github.com/Lightning-AI/lightning/pull/19706))


</details>

<details><summary>Fixed</summary>

- Fixed a matrix shape mismatch issue when running a model loaded from a quantized checkpoint (bitsandbytes) ([#19886](https://github.com/Lightning-AI/lightning/pull/19886))
- Fixed `WandbLogger.log_hyperparameters()` raising an error if hyperparameters are not JSON serializable ([#19769](https://github.com/Lightning-AI/pytorch-lightning/pull/19769))
- Fixed an issue with the LightningCLI not being able to set the `ModelCheckpoint(save_last=...)` argument ([#19808](https://github.com/Lightning-AI/pytorch-lightning/pull/19808))
- Fixed an issue causing ValueError for certain object such as TorchMetrics when dumping hyperparameters to YAML ([#19804](https://github.com/Lightning-AI/pytorch-lightning/pull/19804))
- Fixed resetting `epoch_loop.restarting` to avoid full validation run after `LearningRateFinder` ([#19818](https://github.com/Lightning-AI/pytorch-lightning/issues/19818))


</details>

<a name="changelog-fabric"></a>
## Lightning Fabric

<details><summary>Added</summary>

- Added sanitization for classes before logging them as hyperparameters ([#19771](https://github.com/Lightning-AI/pytorch-lightning/pull/19771))
- Enabled consolidating distributed checkpoints through `fabric consolidate` in the new CLI ([#19560](https://github.com/Lightning-AI/pytorch-lightning/pull/19560))
- Added the ability to explicitly mark forward methods in Fabric via `_FabricModule.mark_forward_method()` ([#19690](https://github.com/Lightning-AI/pytorch-lightning/pull/19690))
- Added support for PyTorch 2.3 ([#19708](https://github.com/Lightning-AI/pytorch-lightning/pull/19708))
- Added `ModelParallelStrategy` to support 2D parallelism ([#19846](https://github.com/Lightning-AI/pytorch-lightning/pull/19846), [#19852](https://github.com/Lightning-AI/pytorch-lightning/pull/19852), [#19870](https://github.com/Lightning-AI/pytorch-lightning/pull/19870), [#19872](https://github.com/Lightning-AI/pytorch-lightning/pull/19872))
- Added a call to `torch.distributed.destroy_process_group` in atexit handler if process group needs destruction ([#19931](https://github.com/Lightning-AI/pytorch-lightning/pull/19931))
- Added support for configuring hybrid-sharding by passing a tuple for the `FSDPStrategy(device_mesh=...)` argument ([#19504](https://github.com/Lightning-AI/pytorch-lightning/pull/19504))

</details>

<details><summary>Changed</summary>

- Renamed `lightning run model` to `fabric run` ([#19442](https://github.com/Lightning-AI/pytorch-lightning/pull/19442), [#19527](https://github.com/Lightning-AI/pytorch-lightning/pull/19527))
- The `Fabric.rank_zero_first` context manager now uses a barrier without timeout to avoid long-running tasks to be interrupted ([#19448](https://github.com/Lightning-AI/lightning/pull/19448))
- Fabric now raises an error if you forget to call `fabric.backward()` when it is needed by the strategy or precision selection ([#19447](https://github.com/Lightning-AI/lightning/pull/19447), [#19493](https://github.com/Lightning-AI/lightning/pull/19493))
- `_BackwardSyncControl` can now control what to do when gradient accumulation is disabled ([#19577](https://github.com/Lightning-AI/lightning/pull/19577))

</details>

<details><summary>Removed</summary>

- Removed support for PyTorch 1.13 ([#19706](https://github.com/Lightning-AI/lightning/pull/19706))

</details>

<details><summary>Fixed</summary>

- Fixed a matrix shape mismatch issue when running a model loaded from a quantized checkpoint (bitsandbytes) ([#19886](https://github.com/Lightning-AI/lightning/pull/19886))

</details>

</br>

**Full commit list**: [2.2.0 -> 2.3.0](https://github.com/Lightning-AI/lightning/compare/2.2.0...2.3.0)

<a name="contributors"></a>
# Contributors

We thank all our contributors who submitted pull requests for features, bug fixes and documentation updates.

### New Contributors
* @cauyxy made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19437
* @mwip made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19518
* @kylebgorman made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19513
* @kashif made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19520
* @ash0ts made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19451
* @dimitri-voytan made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19524
* @ankitgola005 made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19615
* @invisprints made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19629
* @kvenkman made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19465
* @fnhirwa made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19640
* @inyong37 made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19677
* @clumsy made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19601
* @judidoko made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19692
* @Lunamos made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19701
* @dominicgkerr made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19727
* @daavoo made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19774
* @Peiffap made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19805
* @IvanYashchuk made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19926
* @ringohoffman made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19904
* @afspies made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19847
* @fedebotu made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19822
* @mariovas3 made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19808
* @Bhavay-2001 made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19947
* @V0XNIHILI made their first contribution in https://github.com/Lightning-AI/pytorch-lightning/pull/19771

### Did you know?

Chuck Norris is a big fan and daily user of Lightning Studio.



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.3.0)

---

## 2.2.5: Patch release v2.2.5
**Published:** 2024-05-22

## PyTorch Lightning + Fabric

### Fixed

- Fixed a matrix shape mismatch issue when running a model loaded from a quantized checkpoint (bitsandbytes) ([#19886](https://github.com/Lightning-AI/lightning/pull/19886))


----

**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.2.4...2.2.5

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.5)

---

## 2.2.4: Patch release v2.2.4
**Published:** 2024-05-01

## App

### Fixed

- Fixed HTTPClient retry for flow/work queue ([#19837](https://github.com/Lightning-AI/pytorch-lightning/pull/19837))


## PyTorch

No Changes.

## Fabric

No Changes.


**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.2.3...2.2.4

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.4)

---

## 2.2.3: Patch release v2.2.3
**Published:** 2024-04-23

## PyTorch

### Fixed

- Fixed `WandbLogger.log_hyperparameters()` raising an error if hyperparameters are not JSON serializable ([#19769](https://github.com/Lightning-AI/pytorch-lightning/pull/19769))


## Fabric

No Changes.


**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.2.2...2.2.3


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.3)

---

## 2.2.2: Patch release v2.2.2
**Published:** 2024-04-11

## PyTorch

### Fixed

- Fixed an issue causing a TypeError when using `torch.compile` as a decorator ([#19627](https://github.com/Lightning-AI/pytorch-lightning/pull/19627))
- Fixed a KeyError when saving a FSDP sharded checkpoint and setting `save_weights_only=True` ([#19524](https://github.com/Lightning-AI/pytorch-lightning/pull/19524))

## Fabric

### Fixed


- Fixed an issue causing a TypeError when using `torch.compile` as a decorator ([#19627](https://github.com/Lightning-AI/pytorch-lightning/pull/19627))
- Fixed issue where some model methods couldn't be monkeypatched after being Fabric wrapped ([#19705](https://github.com/Lightning-AI/pytorch-lightning/pull/19705))
- Fixed an issue causing weights to be reset in `Fabric.setup()` when using FSDP ([#19755](https://github.com/Lightning-AI/pytorch-lightning/pull/19755))


**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.2.1...2.2.2

## Contributors

@ankitgola005 @awaelchli @Borda @carmocca @dmitsf @dvoytan-spark @fnhirwa 



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.2)

---

## 2.2.1: Patch release v2.2.1
**Published:** 2024-03-04

## PyTorch

### Fixed

- Fixed an issue with CSVLogger trying to append to file from a previous run when the version is set manually ([#19446](https://github.com/Lightning-AI/lightning/pull/19446))
- Fixed the divisibility check for `Trainer.accumulate_grad_batches` and `Trainer.log_every_n_steps` in ThroughputMonitor ([#19470](https://github.com/Lightning-AI/lightning/pull/19470))
- Fixed support for Remote Stop and Remote Abort with NeptuneLogger ([#19130](https://github.com/Lightning-AI/pytorch-lightning/pull/19130))
- Fixed infinite recursion error in precision plugin graveyard ([#19542](https://github.com/Lightning-AI/pytorch-lightning/pull/19542))


## Fabric

### Fixed

- Fixed an issue with CSVLogger trying to append to file from a previous run when the version is set manually ([#19446](https://github.com/Lightning-AI/lightning/pull/19446))




**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.2.0post...2.2.1

## Contributors
@Raalsky @awaelchli @carmocca @Borda


_If we forgot someone due to not matching commit email with GitHub account, let us know :]_



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.1)

---

## 2.2.0.post0: Minor release correction
**Published:** 2024-02-12

**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.2.0...2.2.0.post0

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.0.post0)

---

## 2.2.0: Lightning v2.2
**Published:** 2024-02-07

[Lightning AI](https://lightning.ai) is excited to announce the release of Lightning 2.2 :zap:

**Did you know?** The Lightning philosophy extends beyond a boilerplate-free deep learning framework: We've been hard at work bringing you [Lightning Studio](https://lightning.ai/). Code together, prototype, train, deploy, host AI web apps. All from your browser, with zero setup.

While our previous release was packed with many big new features, this time around we're rolling out mainly improvements based on feedback from the community. And of course, as the name implies, this release fully supports the latest [PyTorch 2.2](https://pytorch.org/blog/pytorch2-2/) :tada:


- [Highlights](#highlights)
    - [Monitoring Throughput](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#highlights-throughput)
    - [Improved Handling of Evaluation Mode](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#highlights-eval)
    - [Converting FSDP Checkpoints](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#highlights-consolidate-fsdp)
    - [Improvements to Compiling DDP/FSDP in Fabric](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#highlights-compile)
    - [Saving and Loading DataLoader State](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#highlights-dataloader-state)
    - [Non-strict Checkpoint Loading in Trainer](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#highlights-non-strict)
- [Notable Changes](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#bc-changes)
- [Full Changelog](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#changelog)
    - [PyTorch Lightning](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#changelog-pytorch)
    - [Lightning Fabric](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#changelog-fabric)
- [Contributors](https://github.com/Lightning-AI/lightning/releases/tag/2.2.0#contributors)


<a name="highlights"></a>
# Highlights

<a name="highlights-throughput"></a>
## Monitoring Throughput

Lightning now has built-in utilities to measure throughput metrics such as batches/sec, samples/sec and Model FLOP Utilization (MFU) ([#18848](https://github.com/Lightning-AI/lightning/pull/18848)).

**Trainer:**

For the Trainer, this comes in form of a `ThroughputMonitor` callback. In order to track samples/sec, you need to provide a function to tell the monitor how to extract the batch dimension from your input. Furthermore, if you want to track MFU, you can provide a sample forward pass and the `ThroughputMonitor` will automatically estimate the utilization based on the hardware you are running on:

```python
import lightning as L
from lightning.pytorch.callbacks import ThroughputMonitor
from lightning.fabric.utilities.throughput import measure_flops


class MyModel(LightningModule):
    def setup(self, stage):
        with torch.device("meta"):
            model = MyModel()

        def sample_forward():
            batch = torch.randn(..., device="meta")
            return model(batch)

        self.flops_per_batch = measure_flops(model, sample_forward, loss_fn=torch.Tensor.sum)


throughput = ThroughputMonitor(
    batch_size_fn=lambda batch: batch.size(0),
    # optional, if your samples have a length (like number of tokens)
    sample_fn=lambda batch: batch.size(1)
)
trainer = L.Trainer(log_every_n_steps=10, callbacks=throughput, logger=...)
model = MyModel()
trainer.fit(model)

```

The results get automatically sent to the logger if one is configured on the Trainer.

**Fabric:**

For Fabric, the `ThroughputMonitor` is a simple utility object on which you call `.update()` and `compute_and_log()` during the training loop:

```python
import lightning as L
from lightning.fabric.utilities import ThroughputMonitor


fabric = L.Fabric(logger=...)
throughput = ThroughputMonitor(fabric)

t0 = time()
for batch_idx, batch in enumerate(train_dataloader):
    do_work()
    torch.cuda.synchronize()  # required or else time() won't be correct
    throughput.update(
        time=(time() - t0), 
        batches=batch_idx, 
        samples=(batch_idx * batch_size)
    )
    if batch_idx % 10 == 0:
        throughput.compute_and_log(step=batch_idx)
```

Check out [our TinyLlama LLM pretraining script](https://github.com/Lightning-AI/lit-gpt/blob/6150d04ff3b199ddefbe55e58d593ecae587b9d9/pretrain/tinyllama.py) for a full example using Fabric's `ThroughputMonitor`. 

The troughput utilities can report:
- batches per second (per process and across process)
- samples per second (per process and across process)
- items per second (e.g. tokens) (per process and across process)
- flops per second (per process and across process)
- model flops utilization (MFU) (per process)
- total time, total samples, total batches, and total items (per process)


<a name="highlights-eval"></a>
## Improved Handling of Evaluation Mode

When you train a model and have validation enabled, the Trainer automatically calls `.eval()` when transitioning to the validation loop, and `.train()` when validation ends. Until now, this had the unfortunate side effect that any submodules in your LightningModule that were in evaluation mode get reset to train mode. In Lightning 2.2, the Trainer now captures the mode of every submodule before switching to validation, and restores the mode the modules were in when validation ends ([#18951](https://github.com/Lightning-AI/lightning/pull/18951), [#18951](https://github.com/Lightning-AI/lightning/pull/18951), [#18951](https://github.com/Lightning-AI/lightning/pull/18951)). This improvement will help users avoid silent correctness bugs and removes boilerplate code for managing frozen layers.


```python
import lightning as L


class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.trainable_module = ...
        
        # This will now stay in eval mode
        self.frozen_module = ...
        self.frozen_module.eval()
        
    def training_step(self, batch):
        # Previously, modules were all in train mode
        # Now: Modules are in mode they were set up with
        assert self.trainable_module.training
        assert not self.frozen_module.training
        ...
        
    def validation_step(self, batch):
        # All modules are in eval mode
        ...
    
    
model = LitModel()
trainer = L.Trainer()
trainer.fit(model)
```

If you have overridden any of the `LightningModule.on_{validation,test,predict}_model_{eval,train}` hooks, they will still get called and execute your custom logic, but they are no longer required if you added them to preserve the eval mode of frozen modules.

> [!IMPORTANT]
> In some libraries, for example HuggingFace, models are created in evaluation mode by default (e.g. `HFModel.from_pretrained(...)`). Starting from 2.2, you will have to set `.train()` on these models if you intend to train them.


<a name="highlights-consolidate-fsdp"></a>
## Converting FSDP Checkpoints

In the previous release, we introduced distributed checkpointing with FSDP to speed up saving and loading checkpoints for big models. These checkpoints are in a special format saved in a folder with shards from each GPU in a separate file. While these checkpoints can be loaded back with Lightning Trainer or Fabric very easily, they aren't easy to load or process externally. In Lightning 2.2, we introduced a CLI utility that lets you consolidate the checkpoint folder to a single file that can be loaded in raw PyTorch with `torch.load()` for example ([#19213](https://github.com/Lightning-AI/lightning/pull/19213)).

Given you saved a distributed checkpoint, you can then convert it like so:

```bash
# For Trainer checkpoints:
python -m lightning.pytorch.utilities.consolidate_checkpoint path/to/my/checkpoint


# For Fabric checkpoints:
python -m lightning.fabric.utilities.consolidate_checkpoint path/to/my/checkpoint
```

Read more about distributed checkpointing in our documentation: [Trainer](https://lightning.ai/docs/pytorch/2.2.0/common/checkpointing_expert.html#convert-a-distributed-checkpoint), [Fabric](https://lightning.ai/docs/fabric/2.2.0/guide/checkpoint/distributed_checkpoint.html#convert-a-distributed-checkpoint).


<a name="highlights-compile"></a>
## Improvements to Compiling DDP/FSDP in Fabric

PyTorch 2.0+ introduced `torch.compile`, a powerful tool to speed up your models without changing the code.
We now added [a comprehensive guide how to use `torch.compile`](https://lightning.ai/docs/fabric/2.2.0/advanced/compile.html) correctly with tips and tricks to help you troubleshoot common issues. On top of that, `Fabric.setup()` will now reapply `torch.compile` on top of DDP/FSDP if you are enabling these strategies ([#19280](https://github.com/Lightning-AI/lightning/pull/19280)).

```python
import lightning as L

# Select a distributed strategy (DDP, FSDP, ...)
fabric = L.Fabric(strategy="ddp", devices=8)

# Compile your model before `.setup()`
model = torch.compile(model)

# Now automatically handles compiling also over DDP/FSDP
model = fabric.setup(model)

# You can opt-out if it is causing trouble
model = fabric.setup(model, _reapply_compile=False)
```

You might see fewer graph breaks, but there won't be any significant speed-ups with this. We introduced this mainly to make Fabric ready for future improvements from PyTorch to optimizing distributed operations.


<a name="highlights-dataloader-state"></a>
## Saving and Loading DataLoader State

If you use a dataloader/iterable that implements the `.state_dict()` and `.load_state_dict()` interface, the Trainer will now automatically save and load their state in the checkpoint ([#19361](https://github.com/Lightning-AI/lightning/pull/19361)).

```python
import lightning as L


class MyDataLoader:
    """A dataloader that implements the 'stateful' interface."""
    
    def state_dict(self):
        # Return a dictionary with state
        return {"batches_fetched": ...}
    
    def load_state_dict(self, state_dict):
        # Load the state from the checkpoint
        self.batches_fetched = state_dict["batches_fetched"]


model = ...
dataloader = MyDataLoader()
trainer = L.Trainer()

# Saves checkpoints that include the dataloader state
trainer.fit(model, dataloader)

# When you resume training, the dataloader can now load its state
trainer.fit(model, dataloader, ckpt_path="path/to/my/checkpoint")
```

Note that the standard [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html) does not support this stateful interface. This feature only works on loaders that implement these two methods. A dataloader that supports full fault-tolerance will be included in our upcoming release of Lightning Data - a library to optimize data preprocessing and streaming in the cloud. Stay tuned!

<a name="highlights-non-strict"></a>
## Non-strict Checkpoint Loading in Trainer

A feature that has been requested for a long time by the community is non-strict checkpoint loading. By default, a checkpoint in PyTorch is loaded with `strict=True` to ensure all keys in the saved checkpoint match what's in the model's state dict.
However, in some use cases it might make sense to exclude certain weights from being included in the checkpoint. When resuming training, the user would then be required to set `strict=False`, which wasn't configurable until now.

You can now set the attribute `strict_loading=False` on your LightningModule if you want to allow loading partial checkpoints ([#19404](https://github.com/Lightning-AI/lightning/pull/19404)).

```python
import lightning as L

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        # This model only trains the decoder, we don't save the encoder
        self.encoder = from_pretrained(...).requires_grad_(False)
        self.decoder = Decoder()
        
        # Set to False because we only care about the decoder
        self.strict_loading = False
    
    def state_dict(self):
        # Don't save the encoder, it is not being trained
        return {k: v for k, v in super().state_dict().items() if "encoder" not in k}

...

trainer = L.Trainer()
model = LitModel()

# Will load weights with `.load_state_dict(strict=model.strict_loading)`
trainer.fit(model, ckpt_path="path/to/checkpoint")
```

Full documentation [here](https://lightning.ai/docs/pytorch/2.2.0/common/checkpointing_advanced.html#resume-from-a-partial-checkpoint).



<a name="bc-changes"></a>
# Notable Changes

The 2.0 series of Lightning releases guarantees core API stability: No name changes, argument renaming, hook removals etc. on core interfaces (Trainer, LightningModule, etc.) unless a feature is specifically marked experimental. Here we list a few behavioral changes made in places where the change was justified if it significantly improves the user experience, improves performance, or fixes the correctness of a feature. These changes will likely not impact most users.

<a name="highlights-save-last"></a>
## ModelCheckpoint's save-last Feature

In Lightning 2.1, we made the `ModelCheckpoint(..., save_last=True)` feature save a symbolic link to the last saved checkpoint instead of rewriting the checkpoint ([#18748](https://github.com/Lightning-AI/lightning/pull/18748)). This time saver is especially useful for large models who take a while to save. However, many users were confused by the new behavior and wanted it turned off, saving a copy instead of a symbolic link like before. In Lightning 2.2, we are reverting this decision and make the linking opt-in ([#19191](https://github.com/Lightning-AI/lightning/pull/19191)):

```python
from lightning.pytorch.callbacks import ModelCheckpoint

# In 2.1 saves a symbolic link "last.ckpt" to the last checkpoint saved
# In 2.2 saves "last.ckpt" as a copy of the last checkpoint saved
checkpoint = ModelCheckpoint("./my_checkpoints", save_last=True)

# You can opt-in to save a symlink (if possible)
checkpoint = ModelCheckpoint("./my_checkpoints", save_last="link")
```


<a name="highlights-seed-everything"></a>
## Removed Problematic Default Seeding

The `seed_everything(x)` utility function is useful to set the seed for several libraries like PyTorch, NumPy and Python in a single line of code. However, until now you were allowed to omit passing a seeding value, in which case the function picked a seed value *randomly*. In certain cases, for example when processes are launched externally (e.g., SLURM, torchelastic etc.), this default behavior is dangerous because each process will independently choose a random seed. This can affect sampling, randomized validation splits, and other behaviors that rely on each process having the same seed. In 2.2, we removed this default behavior and default to a seed value 0 ([#18846](https://github.com/Lightning-AI/lightning/pull/18846)):

```python
from lightning.pytorch.utilities import seed_everything

# Set the random seed for PyTorch, NumPy, Python etc.
seed_everything(42)

# Not setting a value now defaults to 0
seed_everything()
```

In the unlikely event that you relied on the previous behavior, you now have to choose the seed randomly yourself:

```python
seed_everything(random.randint(0, 1000000))
```


## Miscellaneous Changes

- Dropped support for PyTorch 1.12 ([#19300](https://github.com/Lightning-AI/lightning/pull/19300))
- The columns in the `metrics.csv` file produced by `CSVLogger` are now sorted alphabetically ([#19159](https://github.com/Lightning-AI/lightning/pull/19159))
- Added support for meta-device initialization and materialization of 4-bit Bitsandbytes layers ([#19150](https://github.com/Lightning-AI/lightning/pull/19150))
- Added `TransformerEnginePrecision(fallback_compute_dtype=)` to control the dtype of operations that don't support fp8 ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- We renamed the `TransformerEnginePrecision(dtype=)` argument to `weights_dtype` and made it required ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- The `LightningModule.load_from_checkpoint()` function now calls `.configure_model()` on the model if it is overridden, to ensure all layers can be loaded from the checkpoint ([#19036](https://github.com/Lightning-AI/lightning/pull/19036))


<a name="changelog"></a>
# CHANGELOG

<a name="changelog-pytorch"></a>
## PyTorch Lightning

<details><summary>Added</summary>
    
- Added `lightning.pytorch.callbacks.ThroughputMonitor` to track throughput and log it ([#18848](https://github.com/Lightning-AI/lightning/pull/18848))
- The Trainer now restores the training mode set through `.train()` or `.eval()` on a submodule-level when switching from validation to training ([#18951](https://github.com/Lightning-AI/lightning/pull/18951))
- Added support for meta-device initialization and materialization of 4-bit Bitsandbytes layers ([#19150](https://github.com/Lightning-AI/lightning/pull/19150))
- Added `TransformerEnginePrecision(fallback_compute_dtype=)` to control the dtype of operations that don't support fp8 ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- Added the option `ModelCheckpoint(save_last='link')` to create a symbolic link for the 'last.ckpt' file ([#19191](https://github.com/Lightning-AI/lightning/pull/19191))
- Added a utility function and CLI to consolidate FSDP sharded checkpoints into a single file ([#19213](https://github.com/Lightning-AI/lightning/pull/19213))
- The TQDM progress bar now respects the env variable `TQDM_MINITERS` for setting the refresh rate ([#19381](https://github.com/Lightning-AI/lightning/pull/19381))
- Added support for saving and loading stateful training DataLoaders ([#19361](https://github.com/Lightning-AI/lightning/pull/19361))
- Added shortcut name `strategy='deepspeed_stage_1_offload'` to the strategy registry ([#19075](https://github.com/Lightning-AI/lightning/pull/19075))
- Added support for non-strict state-dict loading in Trainer via the new `LightningModule.strict_loading = True | False` attribute ([#19404](https://github.com/Lightning-AI/lightning/pull/19404))

</details>

<details><summary>Changed</summary>
    
- `seed_everything()` without passing in a seed no longer randomly selects a seed, and now defaults to `0` ([#18846](https://github.com/Lightning-AI/lightning/pull/18846))
- The `LightningModule.on_{validation,test,predict}_model_{eval,train}` now only get called if they are overridden by the user ([#18951](https://github.com/Lightning-AI/lightning/pull/18951))
- The `Trainer.fit()` loop no longer calls `LightningModule.train()` at the start; it now preserves the user's configuration of frozen layers ([#18951](https://github.com/Lightning-AI/lightning/pull/18951))
- The `LightningModule.load_from_checkpoint()` function now calls `.configure_model()` on the model if it is overridden, to ensure all layers can be loaded from the checkpoint ([#19036](https://github.com/Lightning-AI/lightning/pull/19036))
- Restored usage of `step` parameter when logging metrics with `NeptuneLogger` ([#19126](https://github.com/Lightning-AI/pytorch-lightning/pull/19126))
- Changed the `TransformerEnginePrecision(dtype=)` argument to `weights_dtype` and made it required ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- The columns in the `metrics.csv` file produced by `CSVLogger` are now sorted alphabetically ([#19159](https://github.com/Lightning-AI/lightning/pull/19159))
- Reverted back to creating a checkpoint copy when `ModelCheckpoint(save_last=True)` instead of creating a symbolic link ([#19191](https://github.com/Lightning-AI/lightning/pull/19191))

</details>


<details><summary>Deprecated</summary>
    
- Deprecated all precision plugin classes under `lightning.pytorch.plugins` with the suffix `Plugin` in the name ([#18840](https://github.com/Lightning-AI/lightning/pull/18840))

</details>

<details><summary>Removed</summary>
    
- Removed support for PyTorch 1.12 ([#19300](https://github.com/Lightning-AI/lightning/pull/19300))

</details>

<details><summary>Fixed</summary>
    
- Fixed issue where the `precision="transformer-engine"` argument would not replace layers by default ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- Fixed issue where layers created in `LightningModule.setup` or `LightningModule.configure_model` wouldn't get converted when using the Bitsandbytes or TransformerEngine plugins ([#19061](https://github.com/Lightning-AI/lightning/pull/19061))
- Fixed the input validation logic in `FSDPStrategy` to accept a `device_mesh` ([#19392](https://github.com/Lightning-AI/lightning/pull/19392))

</details>

<a name="changelog-fabric"></a>
## Lightning Fabric

<details><summary>Added</summary>
  
- Added `lightning.fabric.utilities.ThroughputMonitor` and `lightning.fabric.utilities.Throughput` to track throughput and log it ([#18848](https://github.com/Lightning-AI/lightning/pull/18848))
- Added `lightning.fabric.utilities.AttributeDict` for convenient dict-attribute access to represent state in script ([#18943](https://github.com/Lightning-AI/lightning/pull/18943))
- Added support for meta-device initialization and materialization of 4-bit Bitsandbytes layers ([#19150](https://github.com/Lightning-AI/lightning/pull/19150))
- Added `TransformerEnginePrecision(fallback_compute_dtype=)` to control the dtype of operations that don't support fp8 ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- Added support for clipping gradients by value with FSDP ([#19236](https://github.com/Lightning-AI/lightning/pull/19236))
- Added a utility function and CLI to consolidate FSDP sharded checkpoints into a single file ([#19213](https://github.com/Lightning-AI/lightning/pull/19213))
- Added support for re-compiling the model inside `Fabric.setup()` over the FSDP/DDP wrappers ([#19280](https://github.com/Lightning-AI/lightning/pull/19280))

</details>

<details><summary>Changed</summary>
    
- `seed_everything()` without passing in a seed no longer randomly selects a seed, and now defaults to `0` ([#18846](https://github.com/Lightning-AI/lightning/pull/18846))
- Changed the `TransformerEnginePrecision(dtype=)` argument to `weights_dtype` and made it required ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- The columns in the `metrics.csv` file produced by `CSVLogger` are now sorted alphabetically ([#19159](https://github.com/Lightning-AI/lightning/pull/19159))

</details>

<details><summary>Removed</summary>

- Removed support for PyTorch 1.12 ([#19300](https://github.com/Lightning-AI/lightning/pull/19300))

</details>

<details><summary>Fixed</summary>

- Fixed parsing of v100s GPUs in `get_available_flops` ([#18952](https://github.com/Lightning-AI/lightning/pull/18952))
- Fixed issue where the `precision="transformer-engine"` argument would not replace layers by default ([#19082](https://github.com/Lightning-AI/lightning/pull/19082))
- Fixed the input validation logic in `FSDPStrategy` to accept a `device_mesh` ([#19392](https://github.com/Lightning-AI/lightning/pull/19392))

</details>

</br>

**Full commit list**: [2.1.0 -> 2.2.0](https://github.com/Lightning-AI/lightning/compare/2.1.0...2.2.0)

<a name="contributors"></a>
# Contributors

Everyone who contributed between 2.1 and 2.2, in no particular order:

### Veteran

@nik777 @Raalsky @wouterzwerink @AleksanderWWW @awaelchli @nohalon @ioangatop @Borda @ethanwharris @BoringDonut @mauvilsa @parambharat @tchaton @ryan597 @adamjstewart @rasbt @carmocca


### New

@hiaoxui @VictorPrins @jaswon @AMHermansen @JalinWang @MF-FOOM @unacanal @Jamim @harishb00 @asingh9530 @dipta007 @daturkel @jerrymannil @mjbommar @shenmishajing @paganpasta @lauritsf @andyland @mathematicalmichael

### Did you know?

Chuck Norris is a big fan and daily user of PyTorch Lightning.



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.0)

---

## 2.2.0.rc0: Lightning 2.2 Release Candidate
**Published:** 2024-02-01
**Pre-release**

This is a preview release for Lightning 2.2.0.

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.2.0.rc0)

---

## 2.1.4: Minor patch release v2.1.4
**Published:** 2024-02-01

## Fabric

### Fixed

- Fixed an issue preventing Fabric to run on CPU when the system's CUDA driver is outdated or broken ([#19234](https://github.com/Lightning-AI/lightning/pull/19234))
- Fixed typo in kwarg in SpikeDetection ([#19282](https://github.com/Lightning-AI/lightning/pull/19282))


---

## PyTorch

### Fixed

- Fixed `Trainer` not expanding the `default_root_dir` if it has the `~` (home) prefix ([#19179](https://github.com/Lightning-AI/lightning/pull/19179))
- Fixed warning for Dataloader if `num_workers=1` and CPU count is 1 ([#19224](https://github.com/Lightning-AI/lightning/pull/19224))
- Fixed `WandbLogger.watch()` method annotation to accept `None` for the log parameter ([#19237](https://github.com/Lightning-AI/lightning/pull/19237))
- Fixed an issue preventing the Trainer to run on CPU when the system's CUDA driver is outdated or broken ([#19234](https://github.com/Lightning-AI/lightning/pull/19234))
- Fixed an issue with the ModelCheckpoint callback not saving relative symlinks with `ModelCheckpoint(save_last="link")` ([#19303](https://github.com/Lightning-AI/lightning/pull/19303))
- Fixed issue where the `_restricted_classmethod_impl` would incorrectly raise a TypeError on inspection rather than on call ([#19332](https://github.com/Lightning-AI/lightning/pull/19332))
- Fixed exporting `__version__` in `__init__` ([#19221](https://github.com/Lightning-AI/lightning/pull/19221))


---

**Full Changelog**: https://github.com/Lightning-AI/pytorch-lightning/compare/2.1.3...2.1.4

## Contributors

@andyland @asingh9530 @awaelchli @Borda @daturkel @dipta007 @lauritsf @mjbommar @shenmishajing @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.1.4)

---

## 2.1.3: Minor patch release v2.1.3
**Published:** 2023-12-21

## App

### Changed

- Lightning App: Use the batch get endpoint (#19180)
- Drop starsessions from App's requirements (#18470)
- Optimize loading time for chunks to be there (#19109)

---

## Data

### Added

- Add fault tolerance `StreamingDataset` (#19052)
- Add numpy support for the `StreamingDataset` (#19050)
- Add fault tolerance for the `StreamingDataset` (#19049)
- Add direct s3 support to the `StreamingDataset` (#19044)
- Add disk usage check before downloading files (#19041)

### Changed

- Cleanup chunks right away if the dataset doesn't fit within the cache in `StreamingDataset` (#19168)
- `StreamingDataset` improve deletion strategy (#19118)
- Improve `StreamingDataset` Speed (#19114)
- Remove time in the Data Processor progress bar (#19108)
- Optimize loading time for chunks to be there (#19109)
- Resolve path for `StreamingDataset` (#19094)
- Make input dir in `DataProcessor` required (#18910)
- Remove the `LightningDataset` relying on un-maintained torchdata (#19019)

### Fixed

- Resolve checkpointing for the Streaming Dataset (#19123)
- Resolve Item Loader bugs (#19017)

---

## Fabric

### Fixed

- Avoid moving the model to device if `move_to_device=False` is passed (#19152)
- Fixed broadcast at initialization in `MPIEnvironment` (#19074)

---

## PyTorch

### Changed

- `LightningCLI` no longer allows setting a normal class instance as default. A `lazy_instance` can be used instead (#18822)

### Fixed

- Fixed checks for local file protocol due to fsspec changes in 2023.10.0 (#19023)
- Fixed automatic detection of 'last.ckpt' files to respect the extension when filtering (#17072)
- Fixed an issue where setting `CHECKPOINT_JOIN_CHAR` or `CHECKPOINT_EQUALS_CHAR` would only work on the `ModelCheckpoint` class but not on an instance (#19054)
- Fixed `ModelCheckpoint` not expanding the `dirpath` if it has the `~` (home) prefix (#19058)
- Fixed handling checkpoint dirpath suffix in NeptuneLogger (#18863)
- Fixed an edge case where `ModelCheckpoint` would alternate between versioned and unversioned filename (#19064)
- Fixed broadcast at initialization in `MPIEnvironment` (#19074)
- Fixed the tensor conversion in `self.log` to respect the default dtype (#19046)

---

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/2.1.2...2.1.3

## Contributors

@AleksanderWWW, @awaelchli, @borda, @carmocca, @dependabot[bot], @mauvilsa, @MF-FOOM, @tchaton, @yassersouri

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.1.3)

---

## 2.1.2: Minor patch release v2.1.2
**Published:** 2023-11-15

## App

### Changed

- Forced plugin server to use localhost (#18976)
- Enabled bundling additional files into app source (#18980)
- Limited rate of requests to http queue (#18981)

---

## Fabric

### Fixed

- Fixed precision default from environment (#18928)

---

## PyTorch

### Fixed

- Fixed an issue causing permission errors on Windows when attempting to create a symlink for the "last" checkpoint (#18942)
- Fixed an issue where Metric instances from `torchmetrics` wouldn't get moved to the device when using FSDP (#18954)
- Fixed an issue preventing the user to `Trainer.save_checkpoint()` an FSDP model when `Trainer.test/validate/predict()` ran after `Trainer.fit()` (#18992)

---

## Contributors

@awaelchli, @carmocca, @ethanwharris, @tchaton 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/2.1.1...2.1.2

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.1.2)

---

## 2.1.1: Minor patch release v2.1.1
**Published:** 2023-11-06

## App

### Added

- add flow `fail()` (#18883)

### Fixed

- Fix failing lightning cli entry point (#18821)

---

## Fabric

### Changed

- Calling a method other than `forward` that invokes submodules is now an error when the model is wrapped (e.g., with DDP) (#18819)

### Fixed

- Fixed false-positive warnings about method calls on the Fabric-wrapped module (#18819)
- Refined the FSDP saving logic and error messaging when the path exists (#18884)
- Fixed layer conversion under `Fabric.init_module()` context manager when using the `BitsandbytesPrecision` plugin (#18914)

---

## PyTorch

### Fixed

- Fixed an issue when replacing an existing `last.ckpt` file with a symlink (#18793)
- Fixed an issue when `BatchSizeFinder` `steps_per_trial` parameter ends up defining how many validation batches to run during the entire training (#18394)
- Fixed an issue saving the `last.ckpt` file when using `ModelCheckpoint` on a remote filesystem, and no logger is used (#18867)
- Refined the FSDP saving logic and error messaging when the path exists (#18884)
- Fixed an issue parsing the version from folders that don't include a version number in `TensorBoardLogger` and `CSVLogger` (#18897)

---

## Contributors

@awaelchli, @borda, @BoringDonut, @carmocca, @hiaoxui, @ioangatop, @nohalon, @rasbt, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/2.1.0...2.1.1

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.1.1)

---

## 2.1.0: Lightning 2.1: Train Bigger, Better, Faster
**Published:** 2023-10-12

[Lightning AI](https://lightning.ai) is excited to announce the release of Lightning 2.1 :zap: It's the culmination of work from 79 contributors who have worked on features, bug-fixes, and documentation for a total of over 750+ commits since v2.0.

The theme of 2.1 is "bigger, better, faster": **Bigger** because training large multi-billion parameter models has gotten even more efficient thanks to FSDP, efficient initialization and sharded checkpointing improvements, **better** because it's easier than ever to scale models without making substantial code changes or installing third-party packages and **faster** because it leverages the latest hardware features to speed up training in low-bit precision thanks to new precision plugins like bitsandbytes and transformer engine.
And of course, as the name implies, this release fully leverages the latest features in [PyTorch 2.1](https://pytorch.org/blog/pytorch-2-1/) :tada: 


- [Highlights](#highlights)
    - [Improvements To Large-Scale Training With FSDP](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#highlights-fsdp)
    - [True Half-Precision](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#highlights-half-precision)
    - [Bitsandbytes Quantization](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#highlights-bitsandbytes)
    - [Transformer Engine](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#highlights-transformer-engine)
    - [Lightning on TPU Goes Brrr](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#highlights-tpu)
    - [Granular Control Over Checkpoints in Fabric](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#highlights-fabric-checkpoints)
- [Backward Incompatible Changes](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#bc-changes)
    - [PyTorch Lightning](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#bc-changes-pytorch)
    - [Lightning Fabric](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#bc-changes-fabric)
- [Full Changelog](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#changelog)
    - [PyTorch Lightning](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#changelog-pytorch)
    - [Lightning Fabric](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#changelog-fabric)
    - [Lightning App](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#changelog-app)
- [Contributors](https://github.com/Lightning-AI/lightning/releases/tag/2.1.0#contributors)


<a name="highlights"></a>
# Highlights

<a name="highlights-fsdp"></a>
## Improvements To Large-Scale Training With FSDP

The FSDP strategy for training large billion-parameter models gets substantial improvements and new features in Lightning 2.1, both in Trainer and Fabric (in case you didn't know, [Fabric](https://lightning.ai/docs/fabric/stable) is the latest addition to the Lightning family of tools to scale models without the boilerplate code).
FSDP is now more user-friendly to configure, has memory management and speed improvements, and we have a brand new end-to-end user guide with best practices ([Trainer](https://lightning.ai/docs/pytorch/latest/advanced/model_parallel/fsdp.html), [Fabric](https://lightning.ai/docs/fabric/latest/advanced/model_parallel/fsdp.html)).


### Efficient Saving and Loading of Large Checkpoints

When training large billion-parameter models with FSDP, saving and resuming training, or even just loading model parameters for finetuning can be challenging, as users are are often plagued by out-of-memory errors and speed bottlenecks.

In 2.1, we made several improvements. Starting with saving checkpoints, we added support for distributed/sharded checkpoints, enabled through the setting `state_dict_type` in the strategy ([#18364](https://github.com/Lightning-AI/lightning/pull/18364), [#18358](https://github.com/Lightning-AI/lightning/pull/18358)):


**Trainer:**
```python
import lightning as L
from lightning.pytorch.strategies import FSDPStrategy

# Default used by the strategy
strategy = FSDPStrategy(state_dict_type="full")

# Enable saving distributed checkpoints
strategy = FSDPStrategy(state_dict_type="sharded")

trainer = L.Trainer(strategy=strategy, ...)
```

**Fabric:**
```python
import lightning as L
from lightning.fabric.strategies import FSDPStrategy

# Saving distributed checkpoints is the default
strategy = FSDPStrategy(state_dict_type="sharded")

# Save consolidated (single file) checkpoints
strategy = FSDPStrategy(state_dict_type="full")

fabric = L.Fabric(strategy=strategy, ...)
```

Distributed checkpoints are the fastest and most memory efficient way to save the state of very large models.
The distributed checkpoint format also makes it efficient to load these checkpoints back for resuming training in parallel, and it reduces the impact on CPU memory usage significantly. Furthermore, we've also introduced lazy-loading for non-distributed checkpoints ([#18150](https://github.com/Lightning-AI/lightning/pull/18150), [#18379](https://github.com/Lightning-AI/lightning/pull/18379)), which greatly reduces the impact on CPU memory usage when loading a consolidated (single-file) checkpoint (e.g. for finetuning). Learn more about these features in our FSDP guides ([Trainer](https://lightning.ai/docs/pytorch/latest/advanced/model_parallel/fsdp.html), [Fabric](https://lightning.ai/docs/fabric/latest/advanced/model_parallel/fsdp.html)).


### Fast and Memory-Optimized Initialization

A major challenge that users face when working with large models such as LLMs is dealing with the extreme memory requirements. Even something as simple as instantiating a model becomes non-trivial if the model is so large it won't fit in a single GPU or even a single machine. In Lightning 2.1, we are introducing empty-weights initialization through the `Fabric.init_module()` ([#17462](https://github.com/Lightning-AI/lightning/pull/17462), [#17627](https://github.com/Lightning-AI/lightning/pull/17627)) and `Trainer.init_module()`/`LightningModule.configure_model()` ([#18004](https://github.com/Lightning-AI/lightning/pull/18004), [#18004](https://github.com/Lightning-AI/lightning/pull/18004), [#18385](https://github.com/Lightning-AI/lightning/pull/18385)) methods:


**Trainer:**
```python
import lightning as L

class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Delay initialization of model to `configure_model()`

    def configure_model(self):
        # Model initialized in correct precision and weights on meta-device
        self.model = ...

    ...

trainer = L.Trainer(strategy="fsdp", ...)
trainer.fit(model)
```

**Fabric:**
```python
import lightning as L

fabric = L.Fabric(strategy="fsdp", ...)

# Model initialized in correct precision and weights on meta-device
with fabric.init_module(empty_init=True):
    model = ...
    

# You can also initialize buffers and tensors directly on device and dtype
with fabric.init_tensor():
    model.mask.create()
    model.kv_cache.create()
    x = torch.randn(4, 128)

# Materialization and sharding of model happens inside here
model = fabric.setup(model)
```

Read more about this new feature and its other benefits in our docs ([Trainer](https://lightning.ai/docs/pytorch/latest/advanced/model_init.html), [Fabric](https://lightning.ai/docs/fabric/latest/advanced/model_init.html)).


### User-Friendly Configuration

We made it super easy to configure the sharding- and activation-checkpointing policy when you want to auto-wrap particular layers of your model for advanced control ([#18045](https://github.com/Lightning-AI/lightning/pull/18045), [#18084](https://github.com/Lightning-AI/lightning/pull/18084)).

```diff
  import lightning as L
  from lightning.pytorch.strategies import FSDPStrategy
- from torch.distributed.fsdp.wrap import ModuleWrapPolicy

- strategy = FSDPStrategy(auto_wrap_policy=ModuleWrapPolicy({MyTransformerBlock}))
+ strategy = FSDPStrategy(auto_wrap_policy={MyTransformerBlock})
  trainer = L.Trainer(strategy=strategy, ...)
```

Furthermore, the sharding strategy can now be conveniently set with a string value ([#18087](https://github.com/Lightning-AI/lightning/pull/18087)):

```diff
  import lightning as L
  from lightning.pytorch.strategies import FSDPStrategy
- from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

- strategy = FSDPStrategy(sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
+ strategy = FSDPStrategy(sharding_strategy="SHARD_GRAD_OP")
  trainer = L.Trainer(strategy=strategy, ...)
```
You no longer need to remember the long PyTorch imports! Fabric also supports all these improvements shown above.

<a name="highlights-half-precision"></a>
## True Half-Precision

Lightning now supports true half-precision for training and inference with all built-in strategies ([#18193](https://github.com/Lightning-AI/lightning/pull/18193), [#18217](https://github.com/Lightning-AI/lightning/pull/18217), [#18213](https://github.com/Lightning-AI/lightning/pull/18213), [#18219](https://github.com/Lightning-AI/lightning/pull/18219)). With this setting, the memory required to store the model weights is only half of what is normally needed when running with float32. In addition, you get the same speed benefits as mixed precision training (`precision="16-mixed"`) has:

```python
import lightning as L

# default
trainer = L.Trainer(precision="32-true")

# train with model weights in `torch.float16`
trainer = L.Trainer(precision="16-true")

# train with model weights in `torch.bfloat16`
# (if hardware supports it)
trainer = L.Trainer(precision="bf16-true")
```

The same settings are also available in Fabric! We recommend to try bfloat16 training (`precision="bf16-true"`) as it is often more numerically stable than regular 16-bit precision (`precision="16-true"`).

<a name="highlights-bitsandbytes"></a>
## Bitsandbytes Quantization

With the new [Bitsandbytes precision plugin](https://lightning.ai/docs/pytorch/latest/common/precision_intermediate.html#quantization-via-bitsandbytes) [#18655](https://github.com/Lightning-AI/lightning/pull/18655), you can now quantize your model for significant memory savings during training, finetuning, or inference with a selection of several state-of-the-art quantization algorithms (int8, fp4, nf4 and more). For the first time, Trainer and Fabric make [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) easy to use for general models.


**Trainer:**
```python
import lightning as L
from lightning.pytorch.plugins import BitsandbytesPrecisionPlugin

# this will pick out the compute dtype automatically, by default `bfloat16`
precision = BitsandbytesPrecisionPlugin("nf4-dq")
trainer = L.Trainer(plugins=precision)
```

**Fabric:**
```python
import lightning as L
from lightning.fabric.plugins import BitsandbytesPrecision

# this will pick out the compute dtype automatically, by default `bfloat16`
precision = BitsandbytesPrecision("nf4-dq")
trainer = L.Fabric(plugins=precision)
```

[Learn more!](https://lightning.ai/docs/pytorch/latest/common/precision_intermediate.html#quantization-via-bitsandbytes)

<a name="highlights-transformer-engine"></a>
## Transformer Engine

The [Transformer Engine by NVIDIA](https://docs.nvidia.com/deeplearning/transformer-engine) is a library for accelerating transformer layers on the new Hopper (H100) generation of GPUs. With the integration in Lightning Trainer and Fabric ([#17597](https://github.com/Lightning-AI/lightning/pull/17597), [#18459](https://github.com/Lightning-AI/lightning/pull/18459)), you have easy access to the 8-bit mixed precision for significant speed ups:

**Trainer:**
```python
import lightning as L

# Select 8-bit mixed precision via TransformerEngine, with model weights in float16
trainer = L.Trainer(precision="transformer-engine-float16")
```

**Fabric:**
```python
import lightning as L

# Select 8-bit mixed precision via TransformerEngine, with model weights in float16
fabric = L.Fabric(precision="transformer-engine-float16")
```

More configuration options are available through the respective plugins in [Trainer](https://lightning.ai/docs/pytorch/latest/common/precision_intermediate.html#float8-mixed-precision-via-nvidia-s-transformerengine) and [Fabric](https://lightning.ai/docs/fabric/latest/fundamentals/precision.html#float8-mixed-precision-via-nvidia-s-transformerengine).


<a name="highlights-tpu"></a>
## Lightning on TPU Goes Brrr

Lightning 2.1 runs on the latest generation of TPU hardware on Google Cloud! TPU-v4 and TPU-v5 ([#17227](https://github.com/Lightning-AI/lightning/pull/17227)) are now fully supported both in Fabric and Trainer and run using the new PjRT runtime by default ([#17352](https://github.com/Lightning-AI/lightning/pull/17352)). PjRT is the runtime used by Jax and has shown an average improvement of 35% on benchmarks.

**Trainer:**
```python
import lightning as L

trainer = L.Trainer(accelerator="tpu", devices=8)
model = MyModel()
trainer.fit(model)  # uses PjRT if available
```

**Fabric:**
```python
import lightning as L


def train(fabric):
    ...

fabric = L.Fabric(accelerator="tpu")
fabric.launch(train)  # uses PjRT if available
```

And what's even more exciting, you can now scale massive multi-billion parameter models on TPUs using FSDP ([#17421](https://github.com/Lightning-AI/lightning/pull/17421)).

```python
import lightning as L
from lightning.fabric.strategies import XLAFSDPStrategy

strategy = XLAFSDPStrategy(
    # Most arguments from the PyTorch native FSDP strategy are also available here!
    auto_wrap_policy={Block},
    activation_checkpointing_policy={Block},
    state_dict_type="full",
    sequential_save=True,
)
    
fabric = L.Fabric(devices=8, strategy=strategy)
fabric.launch(finetune)
```
You can find a full end-to-end finetuning example script in our [Lit-GPT repository](https://github.com/Lightning-AI/lit-gpt/blob/main/xla/finetune/adapter.py). The new XLA-FSDP strategy is experimental and currently only available in Fabric. Support in the Trainer will follow in the future.


<a name="highlights-fabric-checkpoints"></a>
## Granular Control Over Checkpoints in Fabric

Several improvements for checkpoint saving and loading have landed in Fabric, enabling more fine-grained control over what is saved/loaded while reducing boilerplate code:

1. There is a new `Fabric.load_raw()` method with which you can load model- or optimizer state-dicts saved externally by a non-Fabric application (e.g., raw PyTorch) ([#18049](https://github.com/Lightning-AI/lightning/pull/18049))

    ```python
    import lightning as L
    
    fabric = L.Fabric()
    model = MyModel()

    # A model weights file saved by your friend who doesn't use Fabric
    fabric.load_raw("path/to/model.pt", model)

    # Equivalent to this:
    # model.load_state_dict(torch.load("path/to/model.pt"))
    ```

2. A new parameter `Fabric.load(..., strict=True|False)` to disable strict loading ([#17645](https://github.com/Lightning-AI/lightning/pull/17645))

    ```python
    import lightning as L
    
    fabric = L.Fabric()
    model = MyModel()
    state = {"model": model}

    # strict loading is the default
    fabric.load("path/to/checkpoint.ckpt", state, strict=True)

    # disable strict loading
    fabric.load("path/to/checkpoint.ckpt", state, strict=False)
    ```

3. A new parameter `Fabric.save(..., filter=...)` that enables you to exclude certain parameters of your model without writing boilerplate code for it ([#17845](https://github.com/Lightning-AI/lightning/pull/17845))


    ```python
    import lightning as L
    
    fabric = L.Fabric()
    model, optimizer = ...

    state = {"model": model, "optimizer": optimizer, "foo": 123}

    # save only the weights that match a pattern
    filter = {"model": lambda k, v: "weight" in k}
    fabric.save("path/to/checkpoint.ckpt", state, filter=filter)
    ```

You can read more about the new options in our [checkpoint guide](https://lightning.ai/docs/fabric/latest/guide/checkpoint.html).


<a name="bc-changes"></a>
# Backward Incompatible Changes

The release of PyTorch Lightning 2.0 was a big step into a new chapter: It brought a more polished API and removed a lot of legacy code and outdated as well as experimental features, at the cost of a long list of breaking changes resulting in more work needed than usual to upgrade from 1.9 to 2.0. Moving forward, we promised to maintain full backward compatibility of our public core APIs to guarantee a smooth upgrade experience for everyone, and with 2.1 we are happy to deliver on this promise. A few exceptions were made in places where the change was justified if it significantly improves the user experience, improves performance, or fixes the correctness of a feature. These changes will likely not impact most users.



<a name="bc-changes-pytorch"></a>
## PyTorch Lightning

### TPU/XLA Changes

When selecting device indices via `devices=[i]`, the Trainer now selects the i-th TPU core (0-based, previously it was 1-based) ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))

**Before:**
```python
# Selects the first TPU core (1-based index)
trainer = Trainer(accelerator="tpu", devices=[1])
```

**Now:**
```python
# Selects the second TPU core (0-based index)
trainer = Trainer(accelerator="tpu", devices=[1])
```

### Multi-GPU in Jupyter Notebooks

Due to lack of reliability, Trainer now only runs on one GPU instead of all GPUs in a Jupyter notebook if `devices="auto"` (default) ([#18291](https://github.com/Lightning-AI/lightning/pull/18291))


**Before:**
```python
import lightning as L

# In Jupyter notebooks, this would select all available GPUs (DDP)
trainer = L.Trainer(accelerator="cuda", devices="auto")
```

**Now:**
```python
# In Jupyter notebooks, this now selects only one GPU (the first)
trainer = L.Trainer(accelerator="cuda", devices="auto")

# You can still explicitly select multiple
trainer = L.Trainer(accelerator="cuda", devices=8)
```

### Device Access in Setup Hook

- During `LightningModule.setup()`, the `self.device` now returns the device the module *will be placed on* instead of `cpu` ([#18021](https://github.com/Lightning-AI/lightning/pull/18021))

**Before:**
```python
def setup(self, stage):
    # CPU regardless of the accelerator used
    print(self.device)
```

**Now:**
```python
def setup(self, stage):
    # CPU/CUDA/MPS/XLA depending on accelerator
    print(self.device)
```
    
### Miscellaneous Changes

- `self.log`ed tensors are now kept in the original device to reduce unnecessary host-to-device synchronizations ([#17334](https://github.com/Lightning-AI/lightning/pull/17334))
- The `FSDPStrategy` now loads checkpoints after the `configure_model`/`configure_sharded_model` hook ([#18358](https://github.com/Lightning-AI/lightning/pull/18358))
- The `FSDPStrategy.load_optimizer_state_dict` and `FSDPStrategy.load_model_state_dict` are a no-op now ([#18358](https://github.com/Lightning-AI/lightning/pull/18358))
- Removed experimental support for `torchdistx` due to a lack of project maintenance ([#17995](https://github.com/Lightning-AI/lightning/pull/17995))
- Dropped support for PyTorch 1.11 ([#18691](https://github.com/Lightning-AI/lightning/pull/18691))


<a name="bc-changes-fabric"></a>
## Lightning Fabric

We thank the community for the amazing feedback we got for [Fabric](https://lightning.ai/docs/fabric/stable/) so far - keep it coming. The list of breaking changes is short and won't affect the vast majority of users.

### Sharding Context Manager in Fabric.run()

We removed automatic sharding support with `Fabric.run` or using `fabric.launch(fn)`. This only impacts FSDP and DeepSpeed strategy users who use this way of launching. Please note that `Fabric.run` is a legacy construct from the `LightningLite` days, and is not recommended today. Please instantiate your large FSDP or DeepSpeed model under the newly added `fabric.init_module` context manager ([#17832](https://github.com/Lightning-AI/lightning/pull/17832)).

**Before:**
```python
import lightning as L

def train(fabric):
    # FSDP's `enable_wrap` context or `deepspeed.zero.Init()`
    # were applied automaticaly here
    model = LargeModel()
    ...
        
fabric = L.Fabric()
fabric.launch(train)
```

**Now:**
```python
def train(fabric):
    # Use `init_module` explicitly to apply these context managers
    with fabric.init_module():
        model = LargeModel()
    ...
```

### Multi-GPU in Jupyter Notebooks

Due to lack of reliability, Fabric now only runs on one GPU instead of all GPUs in a Jupyter notebook if `devices="auto"` (default) ([#18291](https://github.com/Lightning-AI/lightning/pull/18291))


**Before:**
```python
import lightning as L

# In Jupyter notebooks, this would select all available GPUs (DDP)
fabric = L.Fabric(accelerator="cuda", devices="auto")
```

**Now:**
```python
# In Jupyter notebooks, this now selects only one GPU (the first)
fabric = L.Fabric(accelerator="cuda", devices="auto")

# You can still explicitly select multiple
fabric = L.Fabric(accelerator="cuda", devices=8)
```



<a name="changelog"></a>
# CHANGELOG

<a name="changelog-pytorch"></a>
## PyTorch Lightning

<details><summary>Added</summary>

- Added `metrics_format` attribute to `RichProgressBarTheme` class ([#18373](https://github.com/Lightning-AI/lightning/pull/18373))
- Added `CHECKPOINT_EQUALS_CHAR` attribute to `ModelCheckpoint` class ([#17999](https://github.com/Lightning-AI/lightning/pull/17999))
- Added `**summarize_kwargs` to `ModelSummary` and `RichModelSummary` callbacks ([#16788](https://github.com/Lightning-AI/lightning/pull/16788))
- Added support for the `max_size_cycle|max_size|min_size` iteration modes during evaluation ([#17163](https://github.com/Lightning-AI/lightning/pull/17163))
- Added support for the TPU-v4 architecture ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Added support for XLA's new PJRT runtime ([#17352](https://github.com/Lightning-AI/lightning/pull/17352))
- Check for invalid TPU device inputs ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Added `XLAStrategy(sync_module_states=bool)` to control whether to broadcast the parameters to all devices ([#17522](https://github.com/Lightning-AI/lightning/pull/17522))
- Added support for multiple optimizer parameter groups when using the FSDP strategy ([#17309](https://github.com/Lightning-AI/lightning/pull/17309))
- Enabled saving the full model state dict when using the `FSDPStrategy` ([#16558](https://github.com/Lightning-AI/lightning/pull/16558))
- Update `LightningDataModule.from_datasets` to support arbitrary iterables ([#17402](https://github.com/Lightning-AI/lightning/pull/17402))
- Run the DDP wrapper in a CUDA stream ([#17334](https://github.com/Lightning-AI/lightning/pull/17334))
- Added `SaveConfigCallback.save_config` to ease use cases such as saving the config to a logger ([#17475](https://github.com/Lightning-AI/lightning/pull/17475))
- Enabled optional file versioning of model checkpoints ([#17320](https://github.com/Lightning-AI/lightning/pull/17320))
- Added the process group timeout argument `FSDPStrategy(timeout=...)` for the FSDP strategy ([#17274](https://github.com/Lightning-AI/lightning/pull/17274))
- Added `FSDPStrategy(activation_checkpointing_policy=...)` to customize the layer policy for automatic activation checkpointing (requires torch>=2.1) ([#18045](https://github.com/Lightning-AI/lightning/pull/18045))
- Added CLI option `--map-to-cpu` to the checkpoint upgrade script to enable converting GPU checkpoints on a CPU-only machine ([#17527](https://github.com/Lightning-AI/lightning/pull/17527))
- Added non-layer param count to the model summary ([#17005](https://github.com/Lightning-AI/lightning/pull/17005))
- Updated `LearningRateMonitor` to log monitored values to `trainer.callback_metrics` ([#17626](https://github.com/Lightning-AI/lightning/pull/17626))
- Added `log_weight_decay` argument to `LearningRateMonitor` callback ([#18439](https://github.com/Lightning-AI/lightning/pull/18439))
- Added `Trainer.print()` to print on local rank zero only ([#17980](https://github.com/Lightning-AI/lightning/pull/17980))
- Added `Trainer.init_module()` context manager to instantiate large models efficiently directly on device, dtype ([#18004](https://github.com/Lightning-AI/lightning/pull/18004))
  * Creates the model parameters in the desired dtype (`torch.float32`, `torch.float64`) depending on the 'true' precision choice in `Trainer(precision='32-true'|'64-true')`
- Added the `LightningModule.configure_model()` hook to instantiate large models efficiently directly on device, dtype, and with sharding support ([#18004](https://github.com/Lightning-AI/lightning/pull/18004))
  * Handles initialization for FSDP models before wrapping and the Zero stage 3 initialization for DeepSpeed before sharding
- Added support for meta-device initialization with `Trainer.init_module(empty_init=True)` in FSDP ([#18385](https://github.com/Lightning-AI/lightning/pull/18385))
- Added `lightning.pytorch.plugins.PrecisionPlugin.module_init_context()` and `lightning.pytorch.strategies.Strategy.tensor_init_context()` context managers to control model and tensor instantiation ([#18004](https://github.com/Lightning-AI/lightning/pull/18004))
- Automatically call `xla_model.mark_step()` before saving checkpoints with XLA ([#17882](https://github.com/Lightning-AI/lightning/pull/17882))
- Added a callback for spike-detection ([#18014](https://github.com/Lightning-AI/lightning/pull/18014))
- Added the ability to set the `torch.distributed.fsdp.ShardingStrategy` via string in `FSDPStrategy` ([#18087](https://github.com/Lightning-AI/lightning/pull/18087))
- Improved error messages when attempting to load a DeepSpeed checkpoint at an invalid path ([#17795](https://github.com/Lightning-AI/lightning/pull/17795))
- Allowed accessing rank information in the main process before processes are launched when using the `XLAStrategy` ([#18194](https://github.com/Lightning-AI/lightning/pull/18194))
- Added support for true half-precision training via `Trainer(precision="16-true"|"bf16-true")` ([#18193](https://github.com/Lightning-AI/lightning/pull/18193), [#18217](https://github.com/Lightning-AI/lightning/pull/18217), [#18213](https://github.com/Lightning-AI/lightning/pull/18213), [#18219](https://github.com/Lightning-AI/lightning/pull/18219))
- Added automatic process cleanup to avoid zombie child processes and stalls when exceptions are raised ([#18218](https://github.com/Lightning-AI/lightning/pull/18218))
- Added validation of user input for `devices` and `num_nodes` when running with `SLURM` or `TorchElastic` ([#18292](https://github.com/Lightning-AI/lightning/pull/18292))
- Added support for saving checkpoints with either full state-dict or sharded state dict via `FSDPStrategy(state_dict_type="full"|"sharded")` ([#18364](https://github.com/Lightning-AI/lightning/pull/18364))
- Added support for loading sharded/distributed checkpoints in FSDP ([#18358](https://github.com/Lightning-AI/lightning/pull/18358))
- Made the text delimiter in the rich progress bar configurable ([#18372](https://github.com/Lightning-AI/lightning/pull/18372))
- Improved the error messaging and instructions when handling custom batch samplers in distributed settings ([#18402](https://github.com/Lightning-AI/lightning/pull/18402))
- Added support for mixed 8-bit precision as `Trainer(precision="transformer-engine")` using [Nvidia's Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine) ([#18459](https://github.com/Lightning-AI/lightning/pull/18459))
- Added support for linear layer quantization with `Trainer(plugins=BitsandbytesPrecision())` using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) ([#18655](https://github.com/Lightning-AI/lightning/pull/18655))
- Added support for passing the process group to the `FSDPStrategy` ([#18583](https://github.com/Lightning-AI/lightning/pull/18583))
- Enabled the default process group configuration for FSDP's hybrid sharding ([#18583](https://github.com/Lightning-AI/lightning/pull/18583))
- Added `lightning.pytorch.utilities.suggested_max_num_workers` to assist with setting a good value in distributed settings ([#18591](https://github.com/Lightning-AI/lightning/pull/18591))
- Improved the `num_workers` warning to give a more accurate upper limit on the `num_workers` suggestion ([#18591](https://github.com/Lightning-AI/lightning/pull/18591))
- Added `lightning.pytorch.utilities.is_shared_filesystem` utility function to automatically check whether the filesystem is shared between machines ([#18586](https://github.com/Lightning-AI/lightning/pull/18586))
- Added support for returning an object of type `Mapping` from `LightningModule.training_step()` ([#18657](https://github.com/Lightning-AI/lightning/pull/18657))
- Added the hook `LightningModule.on_validation_model_zero_grad()` to allow overriding the behavior of zeroing the gradients before entering the validation loop ([#18710](https://github.com/Lightning-AI/lightning/pull/18710))

</details>

<details><summary>Changed</summary>

- Changed default metric formatting from `round(..., 3)` to `".3f"` format string in `MetricsTextColumn` class ([#18483](https://github.com/Lightning-AI/lightning/pull/18483))
- Removed the limitation to call `self.trainer.model.parameters()` in `LightningModule.configure_optimizers()` ([#17309](https://github.com/Lightning-AI/lightning/pull/17309))
- `Trainer(accelerator="tpu", devices=[i])"` now selects the i-th TPU core (0-based, previously it was 1-based) ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Allow using iterable-style datasets with TPUs ([#17331](https://github.com/Lightning-AI/lightning/pull/17331))
- Increased the minimum XLA requirement to 1.13 ([#17368](https://github.com/Lightning-AI/lightning/pull/17368))
- `self.log`ed tensors are now kept in the original device to reduce unnecessary host-to-device synchronizations ([#17334](https://github.com/Lightning-AI/lightning/pull/17334))
- Made the run initialization in `WandbLogger` lazy to avoid creating artifacts when the CLI is used ([#17573](https://github.com/Lightning-AI/lightning/pull/17573))
- Simplified redirection of `*_step` methods in strategies by removing the `_LightningModuleWrapperBase` wrapper module ([#17531](https://github.com/Lightning-AI/lightning/pull/17531))
- Support kwargs input for LayerSummary ([#17709](https://github.com/Lightning-AI/lightning/pull/17709))
- Dropped support for `wandb` versions older than 0.12.0 in `WandbLogger` ([#17876](https://github.com/Lightning-AI/lightning/pull/17876))
- During `LightningModule.setup()`, the `self.device` now returns the device the module will be placed on instead of `cpu` ([#18021](https://github.com/Lightning-AI/lightning/pull/18021))
- Increased the minimum supported `wandb` version for `WandbLogger` from 0.12.0 to 0.12.10 ([#18171](https://github.com/Lightning-AI/lightning/pull/18171))
- The input tensors now get cast to the right precision type before transfer to the device ([#18264](https://github.com/Lightning-AI/lightning/pull/18264))
- Improved the formatting of emitted warnings ([#18288](https://github.com/Lightning-AI/lightning/pull/18288))
- Broadcast and reduction of tensors with XLA-based strategies now preserve the input's device ([#18275](https://github.com/Lightning-AI/lightning/pull/18275))
- The `FSDPStrategy` now loads checkpoints after the `configure_model`/`configure_sharded_model` hook ([#18358](https://github.com/Lightning-AI/lightning/pull/18358))
- The `FSDPStrategy.load_optimizer_state_dict` and `FSDPStrategy.load_model_state_dict` are a no-op now ([#18358](https://github.com/Lightning-AI/lightning/pull/18358))
- The `Trainer.num_val_batches`, `Trainer.num_test_batches` and `Trainer.num_sanity_val_batches` now return a list of sizes per dataloader instead of a single integer ([#18441](https://github.com/Lightning-AI/lightning/pull/18441))
- The `*_step(dataloader_iter)` flavor now no longer takes the `batch_idx` in the signature ([#18390](https://github.com/Lightning-AI/lightning/pull/18390))
- Calling `next(dataloader_iter)` now returns a triplet `(batch, batch_idx, dataloader_idx)` ([#18390](https://github.com/Lightning-AI/lightning/pull/18390))
- Calling `next(combined_loader)` now returns a triplet `(batch, batch_idx, dataloader_idx)` ([#18390](https://github.com/Lightning-AI/lightning/pull/18390))
- Due to lack of reliability, Trainer now only runs on one GPU instead of all GPUs in a Jupyter notebook if `devices="auto"` (default) ([#18291](https://github.com/Lightning-AI/lightning/pull/18291))
- Made the `batch_idx` argument optional in `validation_step`, `test_step` and `predict_step` to maintain consistency with `training_step` ([#18512](https://github.com/Lightning-AI/lightning/pull/18512))
- The `TQDMProgressBar` now consistently shows it/s for the speed even when the iteration time becomes larger than one second ([#18593](https://github.com/Lightning-AI/lightning/pull/18593))
- The `LightningDataModule.load_from_checkpoint` and `LightningModule.load_from_checkpoint` methods now raise an error if they are called on an instance instead of the class ([#18432](https://github.com/Lightning-AI/lightning/pull/18432))
- Enabled launching via `torchrun` in a SLURM environment; the `TorchElasticEnvironment` now gets chosen over the `SLURMEnvironment` if both are detected ([#18618](https://github.com/Lightning-AI/lightning/pull/18618))
- If not set by the user, Lightning will set `OMP_NUM_THREADS` to `num_cpus / num_processes` when launching subprocesses (e.g. when DDP is used) to avoid system overload for CPU-intensive tasks ([#18677](https://github.com/Lightning-AI/lightning/pull/18677))
- The `ModelCheckpoint` no longer deletes files under the save-top-k mechanism when resuming from a folder that is not the same as the current checkpoint folder ([#18750](https://github.com/Lightning-AI/lightning/pull/18750))
- The `ModelCheckpoint` no longer deletes the file that was passed to `Trainer.fit(ckpt_path=...)` ([#18750](https://github.com/Lightning-AI/lightning/pull/18750))
- Calling `trainer.fit()` twice now raises an error with strategies that spawn subprocesses through `multiprocessing` (ddp_spawn, xla) ([#18776](https://github.com/Lightning-AI/lightning/pull/18776))
- The `ModelCheckpoint` now saves a symbolic link if `save_last=True` and `save_top_k != 0` ([#18748](https://github.com/Lightning-AI/lightning/pull/18748))

</details>


<details><summary>Deprecated</summary>

- Deprecated the `SingleTPUStrategy` (`strategy="single_tpu"`) in favor of `SingleDeviceXLAStrategy` (`strategy="single_xla"`) ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUAccelerator` in favor of `XLAAccelerator` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUPrecisionPlugin` in favor of `XLAPrecisionPlugin` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUBf16PrecisionPlugin` in favor of `XLABf16PrecisionPlugin` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `Strategy.post_training_step` method ([#17531](https://github.com/Lightning-AI/lightning/pull/17531))
- Deprecated the `LightningModule.configure_sharded_model` hook in favor of `LightningModule.configure_model` ([#18004](https://github.com/Lightning-AI/lightning/pull/18004))
- Deprecated the `LightningDoublePrecisionModule` wrapper in favor of calling `Trainer.precision_plugin.convert_input()` ([#18209](https://github.com/Lightning-AI/lightning/pull/18209))

    
</details>

<details><summary>Removed</summary>

- Removed the `XLAStrategy.is_distributed` property. It is always True ([#17381](https://github.com/Lightning-AI/lightning/pull/17381))
- Removed the `SingleTPUStrategy.is_distributed` property. It is always False ([#17381](https://github.com/Lightning-AI/lightning/pull/17381))
- Removed experimental support for `torchdistx` due to a lack of project maintenance ([#17995](https://github.com/Lightning-AI/lightning/pull/17995))
- Removed support for PyTorch 1.11 ([#18691](https://github.com/Lightning-AI/lightning/pull/18691))

    
</details>

<details><summary>Fixed</summary>

- Fixed an issue with reusing the same model across multiple trainer stages when using the `DeepSpeedStrategy` ([#17531](https://github.com/Lightning-AI/lightning/pull/17531))
- Fixed the saving and loading of FSDP optimizer states ([#17819](https://github.com/Lightning-AI/lightning/pull/17819))
- Fixed FSDP re-applying activation checkpointing when the user had manually applied it already ([#18006](https://github.com/Lightning-AI/lightning/pull/18006))
- Fixed issue where unexpected exceptions would leave the default torch dtype modified when using true precision settings ([#18500](https://github.com/Lightning-AI/lightning/pull/18500))
- Fixed issue where not including the `batch_idx` argument in the `training_step` would disable gradient accumulation ([#18619](https://github.com/Lightning-AI/lightning/pull/18619))
- Fixed the replacement of callbacks returned in `LightningModule.configure_callbacks` when the callback was a subclass of an existing Trainer callback ([#18508](https://github.com/Lightning-AI/lightning/pull/18508))
- Fixed `Trainer.log_dir` not returning the correct directory for the `CSVLogger` ([#18548](https://github.com/Lightning-AI/lightning/pull/18548))
- Fixed redundant input-type casting in FSDP precision ([#18630](https://github.com/Lightning-AI/lightning/pull/18630))
- Fixed numerical issues when reducing values in low precision with `self.log` ([#18686](https://github.com/Lightning-AI/lightning/pull/18686))
- Fixed an issue that would cause the gradients to be erased if validation happened in the middle of a gradient accumulation phase ([#18710](https://github.com/Lightning-AI/lightning/pull/18710))
- Fixed redundant file writes in `CSVLogger` ([#18567](https://github.com/Lightning-AI/lightning/pull/18567))
- Fixed an issue that could lead to checkpoint files being deleted accidentally when resuming training ([#18750](https://github.com/Lightning-AI/lightning/pull/18750))


</details>

<a name="changelog-fabric"></a>
## Lightning Fabric

<details><summary>Added</summary>
  
- Added support for the TPU-v4 architecture ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Added support for XLA's new PJRT runtime ([#17352](https://github.com/Lightning-AI/lightning/pull/17352))
- Added support for Fully Sharded Data Parallel (FSDP) training with XLA ([#18126](https://github.com/Lightning-AI/lightning/pull/18126), [#18424](https://github.com/Lightning-AI/lightning/pull/18424), [#18430](https://github.com/Lightning-AI/lightning/pull/18430))
- Check for invalid TPU device inputs ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Added `XLAStrategy(sync_module_states=bool)` to control whether to broadcast the parameters to all devices ([#17522](https://github.com/Lightning-AI/lightning/pull/17522))
- Added support for joint setup of model and optimizer with FSDP ([#17305](https://github.com/Lightning-AI/lightning/pull/17305))
- Added support for handling multiple parameter groups in optimizers set up with FSDP ([#17305](https://github.com/Lightning-AI/lightning/pull/17305))
- Added support for saving and loading sharded model and optimizer state with `FSDPStrategy` ([#17323](https://github.com/Lightning-AI/lightning/pull/17323))
- Added a warning when calling methods on `_FabricModule` that bypass the strategy-specific wrappers ([#17424](https://github.com/Lightning-AI/lightning/pull/17424))
- Added `Fabric.init_tensor()` context manager to instantiate tensors efficiently directly on device and dtype ([#17488](https://github.com/Lightning-AI/lightning/pull/17488))
- Added `Fabric.init_module()` context manager to instantiate large models efficiently directly on device, dtype, and with sharding support ([#17462](https://github.com/Lightning-AI/lightning/pull/17462))
  * Creates the model parameters in the desired dtype (`torch.float32`, `torch.float64`, `torch.float16`, or `torch.bfloat16`) depending on the 'true' precision choice in `Fabric(precision='32-true'|'64-true'|'16-true'|'bf16-true')`
  * Handles initialization for FSDP models before wrapping and the Zero stage 3 initialization for DeepSpeed before sharding
- Added support for empty weight initialization with `Fabric.init_module(empty_init=True)` for checkpoint loading ([#17627](https://github.com/Lightning-AI/lightning/pull/17627))
- Added support for meta-device initialization with `Fabric.init_module(empty_init=True)` in FSDP ([#18122](https://github.com/Lightning-AI/lightning/pull/18122))
- Added `lightning.fabric.plugins.Precision.module_init_context()` and `lightning.fabric.strategies.Strategy.module_init_context()` context managers to control model and tensor instantiation ([#17462](https://github.com/Lightning-AI/lightning/pull/17462))
- `lightning.fabric.strategies.Strategy.tensor_init_context()` context manager to instantiate tensors efficiently directly on device and dtype ([#17607](https://github.com/Lightning-AI/lightning/pull/17607))
- Run the DDP wrapper in a CUDA stream ([#17334](https://github.com/Lightning-AI/lightning/pull/17334))
- Added support for true half-precision as `Fabric(precision="16-true"|"bf16-true")` ([#17287](https://github.com/Lightning-AI/lightning/pull/17287))
- Added support for mixed 8-bit precision as `Fabric(precision="transformer-engine")` using [Nvidia's Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine) ([#17597](https://github.com/Lightning-AI/lightning/pull/17597))
- Added support for linear layer quantization with `Fabric(plugins=BitsandbytesPrecision())` using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) ([#18655](https://github.com/Lightning-AI/lightning/pull/18655))
- Added error messaging for missed `.launch()` when it is required ([#17570](https://github.com/Lightning-AI/lightning/pull/17570))
- Added support for saving checkpoints with either full state-dict or sharded state dict via `FSDPStrategy(state_dict_type="full"|"sharded")` ([#17526](https://github.com/Lightning-AI/lightning/pull/17526))
- Added support for loading a full-state checkpoint file into a sharded model ([#17623](https://github.com/Lightning-AI/lightning/pull/17623))
- Added support for calling hooks on a LightningModule via `Fabric.call` ([#17874](https://github.com/Lightning-AI/lightning/pull/17874))
- Added the parameter `Fabric.load(..., strict=True|False)` to enable non-strict loading of partial checkpoint state ([#17645](https://github.com/Lightning-AI/lightning/pull/17645))
- Added the parameter `Fabric.save(..., filter=...)` to enable saving a partial checkpoint state ([#17845](https://github.com/Lightning-AI/lightning/pull/17845))
- Added support for loading optimizer states from a full-state checkpoint file ([#17747](https://github.com/Lightning-AI/lightning/pull/17747))
- Automatically call `xla_model.mark_step()` before saving checkpoints with XLA ([#17882](https://github.com/Lightning-AI/lightning/pull/17882))
- Automatically call `xla_model.mark_step()` after `optimizer.step()` with XLA ([#17883](https://github.com/Lightning-AI/lightning/pull/17883))
- Added support for all half-precision modes in FSDP precision plugin ([#17807](https://github.com/Lightning-AI/lightning/pull/17807))
- Added `FSDPStrategy(activation_checkpointing_policy=...)` to customize the layer policy for automatic activation checkpointing (requires torch>=2.1) ([#18045](https://github.com/Lightning-AI/lightning/pull/18045))
- Added a callback for spike-detection ([#18014](https://github.com/Lightning-AI/lightning/pull/18014))
- Added the ability to set the `torch.distributed.fsdp.ShardingStrategy` via string in `FSDPStrategy` ([#18087](https://github.com/Lightning-AI/lightning/pull/18087))
- Improved error messages when attempting to load a DeepSpeed checkpoint at an invalid path ([#17795](https://github.com/Lightning-AI/lightning/pull/17795))
- Added `Fabric.load_raw()` for loading raw PyTorch state dict checkpoints for model or optimizer objects ([#18049](https://github.com/Lightning-AI/lightning/pull/18049))
- Allowed accessing rank information in the main process before processes are launched when using the `XLAStrategy` ([#18194](https://github.com/Lightning-AI/lightning/pull/18194))
- Added automatic process cleanup to avoid zombie child processes and stalls when exceptions are raised ([#18218](https://github.com/Lightning-AI/lightning/pull/18218))
- Added validation of user input for `devices` and `num_nodes` when running with `SLURM` or `TorchElastic` ([#18292](https://github.com/Lightning-AI/lightning/pull/18292))
- Improved the error messaging and instructions when handling custom batch samplers in distributed settings ([#18402](https://github.com/Lightning-AI/lightning/pull/18402))
- Added support for saving and loading stateful objects other than modules and optimizers ([#18513](https://github.com/Lightning-AI/lightning/pull/18513))
- Enabled the default process group configuration for FSDP's hybrid sharding ([#18583](https://github.com/Lightning-AI/lightning/pull/18583))
- Added `lightning.fabric.utilities.suggested_max_num_workers` to assist with setting a good value in distributed settings ([#18591](https://github.com/Lightning-AI/lightning/pull/18591))
- Added `lightning.fabric.utilities.is_shared_filesystem` utility function to automatically check whether the filesystem is shared between machines ([#18586](https://github.com/Lightning-AI/lightning/pull/18586))
- Removed support for PyTorch 1.11 ([#18691](https://github.com/Lightning-AI/lightning/pull/18691))
- Added support for passing the argument `.load_state_dict(..., assign=True|False)` on Fabric-wrapped modules in PyTorch 2.1 or newer ([#18690](https://github.com/Lightning-AI/lightning/pull/18690))

</details>

<details><summary>Changed</summary>

- Allow using iterable-style datasets with TPUs ([#17331](https://github.com/Lightning-AI/lightning/pull/17331))
- Increased the minimum XLA requirement to 1.13 ([#17368](https://github.com/Lightning-AI/lightning/pull/17368))
- Fabric argument validation now only raises an error if conflicting settings are set through the CLI ([#17679](https://github.com/Lightning-AI/lightning/pull/17679))
- DataLoader re-instantiation is now only performed when a distributed sampler is required ([#18191](https://github.com/Lightning-AI/lightning/pull/18191))
- Improved the formatting of emitted warnings ([#18288](https://github.com/Lightning-AI/lightning/pull/18288))
- Broadcast and reduction of tensors with XLA-based strategies now preserve the input's device ([#18275](https://github.com/Lightning-AI/lightning/pull/18275))
- Due to lack of reliability, Fabric now only runs on one GPU instead of all GPUs in a Jupyter notebook if `devices="auto"` (default) ([#18291](https://github.com/Lightning-AI/lightning/pull/18291))
- Enabled launching via `torchrun` in a SLURM environment; the `TorchElasticEnvironment` now gets chosen over the `SLURMEnvironment` if both are detected ([#18618](https://github.com/Lightning-AI/lightning/pull/18618))
- If not set by the user, Lightning will set `OMP_NUM_THREADS` to `num_cpus / num_processes` when launching subprocesses (e.g. when DDP is used) to avoid system overload for CPU-intensive tasks ([#18677](https://github.com/Lightning-AI/lightning/pull/18677))

</details>

<details><summary>Deprecated</summary>

- Deprecated the `DDPStrategy.is_distributed` property. This strategy is distributed by definition ([#17381](https://github.com/Lightning-AI/lightning/pull/17381))
- Deprecated the `SingleTPUStrategy` (`strategy="single_tpu"`) in favor of `SingleDeviceXLAStrategy` (`strategy="single_xla"`) ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUAccelerator` in favor of `XLAAccelerator` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUPrecision` in favor of `XLAPrecision` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))
- Deprecated the `TPUBf16Precision` in favor of `XLABf16Precision` ([#17383](https://github.com/Lightning-AI/lightning/pull/17383))

</details>

<details><summary>Removed</summary>

- Removed automatic sharding support with `Fabric.run` or using `fabric.launch(fn)`. This only impacts FSDP and DeepSpeed strategy users. Please instantiate your module under the newly added `fabric.init_module` context manager ([#17832](https://github.com/Lightning-AI/lightning/pull/17832))
- Removed the unsupported `checkpoint_io` argument from the `FSDPStrategy` ([#18192](https://github.com/Lightning-AI/lightning/pull/18192))

</details>

<details><summary>Fixed</summary>

- Fixed issue where running on TPUs would select the wrong device index ([#17227](https://github.com/Lightning-AI/lightning/pull/17227))
- Removed the need to call `.launch()` when using the DP-strategy (`strategy="dp"`) ([#17931](https://github.com/Lightning-AI/lightning/pull/17931))
- Fixed FSDP re-applying activation checkpointing when the user had manually applied it already ([#18006](https://github.com/Lightning-AI/lightning/pull/18006))
- Fixed FSDP re-wrapping the module root when the user had manually wrapped the model ([#18054](https://github.com/Lightning-AI/lightning/pull/18054))
- Fixed issue where unexpected exceptions would leave the default torch dtype modified when using true precision settings ([#18500](https://github.com/Lightning-AI/lightning/pull/18500))
- Fixed redundant input-type casting in FSDP precision ([#18630](https://github.com/Lightning-AI/lightning/pull/18630))
- Fixed an issue with `find_usable_cuda_devices(0)` incorrectly returning a list of devices ([#18722](https://github.com/Lightning-AI/lightning/pull/18722))
- Fixed redundant file writes in `CSVLogger` ([#18567](https://github.com/Lightning-AI/lightning/pull/18567))

</details>

<a name="changelog-app"></a>
## Lightning App

<details><summary>Added</summary>

- Allow customizing `gradio` components with lightning colors ([#17054](https://github.com/Lightning-AI/lightning/pull/17054))

</details>

<details><summary>Changed</summary>
    
- Changed `LocalSourceCodeDir` cache_location to not use home in some certain cases ([#17491](https://github.com/Lightning-AI/lightning/pull/17491))

</details>

<details><summary>Removed</summary>
    
- Remove cluster commands from the CLI ([#18151](https://github.com/Lightning-AI/lightning/pull/18151))

</details>


**Full commit list**: https://github.com/Lightning-AI/lightning/compare/2.0.0...2.1.0

<a name="contributors"></a>
# Contributors

### Veteran

@adamjstewart @akreuzer @ethanwharris @dmitsf @lantiga @nicolai86 @pl-ghost @carmocca @awaelchli @justusschock @edenlightning @belerico @lightningforever @nisheethlahoti @tchaton @yurijmikhalevich @mauvilsa @rlizzo @rusmux @yhl48 @Liyang90 @jerome-habana @JustinGoheen @Borda @speediedan @SkafteNicki @dcfidalgo

### New

@saryazdi @parambharat @kshitij12345 @woqidaideshi @colehawkins @md-121 @gkroiz @idc9 @BoringDonut @OmerShubi @ishandutta0098 @ryan597 @leng-yue @alicanb @One-sixth @santurini @SpirinEgor @KogaiIrina @shanmugamr1992 @janeyx99 @asmith26 @dingusagar @AleksanderWWW @strawberrypie @solyaH @kaczmarj @voidful @water-vapor @bkiat1123 @rhiga2 @baskrahmer @felipewhitaker @mukhery @Quasar-Kim @robieta @one-matrix @jere357 @schmidt-ai @schuhschuh @anio @rjarun8 @callumhay @minhlong94 @klieret @giorgioskij @shihaoyin @JonathanRayner @NripeshN @marcimarc1 @bilelomrani1 @NikolasWolke @0x404 @quintenroets @Borodin @amorehead @SebastianGer @ioangatop @Tribhuvan0 @f0k @sameertantry @kwsp @nik777 @matsumotosan

### Did you know?

When Chuck Norris trains a neural network, it not only learns, but it also gains the ability to defend itself from adversarial attacks by roundhouse kicking them into submission.


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.1.0)

---

## 2.1.0.rc1: Feature teaser
**Published:** 2023-10-10
**Pre-release**

:rabbit: 

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.1.0.rc1)

---

## 2.0.9.post0: Hotfix for Conda package
**Published:** 2023-09-28



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.9.post0)

---

## 2.0.9: Weekly patch release
**Published:** 2023-09-14

## App

### Fixed

- Replace LightningClient with import from lightning_cloud (#18544)

---

## Fabric

### Fixed

- Fixed an issue causing the `_FabricOptimizer.state` to remain outdated after loading with `load_state_dict` (#18488)

---

## PyTorch

### Fixed

- Fixed an issue that wouldn't prevent the user to set the `log_model` parameter in `WandbLogger` via the LightningCLI (#18458)
- Fixed the display of `v_num` in the progress bar when running with `Trainer(fast_dev_run=True)` (#18491)
- Fixed `UnboundLocalError` when running with `python -O` (#18496)
- Fixed visual glitch with the TQDM progress bar leaving the validation bar incomplete before switching back to the training display (#18503)
- Fixed false positive warning about logging interval when running with `Trainer(fast_dev_run=True)` (#18550)

---

## Contributors

@awaelchli, @borda, @justusschock, @SebastianGer

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.9)

---

## 2.0.8: Weekly patch release
**Published:** 2023-08-30

## App

## Changed

- Change top folder (#18212)
- Remove `_handle_is_headless` calls in app run loop (#18362)

### Fixed

- refactor path to root preventing circular import (#18357)

---

## Fabric

### Changed

- On XLA, avoid setting the global rank before processes have been launched as this will initialize the PJRT computation client in the main process (#16966)

### Fixed

- Fixed model parameters getting shared between processes when running with `strategy="ddp_spawn"` and `accelerator="cpu"`; this has a necessary memory impact, as parameters are replicated for each process now (#18238)
- Removed false positive warning when using `fabric.no_backward_sync` with XLA strategies (#17761)
- Fixed issue where Fabric would not initialize the global rank, world size, and rank-zero-only rank after initialization and before launch (#16966)
- Fixed FSDP full-precision `param_dtype` training (`16-mixed`, `bf16-mixed` and `32-true` configurations) to avoid FSDP assertion errors with PyTorch < 2.0 (#18278)

---

## PyTorch

### Changed

- On XLA, avoid setting the global rank before processes have been launched as this will initialize the PJRT computation client in the main process (#16966)
- Fix inefficiency in rich progress bar (#18369)

### Fixed

- Fixed FSDP full-precision `param_dtype` training (`16-mixed` and `bf16-mixed` configurations) to avoid FSDP assertion errors with PyTorch < 2.0 (#18278)
- Fixed an issue that prevented the use of custom logger classes without an `experiment` property defined (#18093)
- Fixed setting the tracking uri in `MLFlowLogger` for logging artifacts to the MLFlow server (#18395)
- Fixed redundant `iter()` call to dataloader when checking dataloading configuration (#18415)
- Fixed model parameters getting shared between processes when running with `strategy="ddp_spawn"` and `accelerator="cpu"`; this has a necessary memory impact, as parameters are replicated for each process now (#18238)
- Properly manage `fetcher.done` with `dataloader_iter` (#18376)

---

## Contributors

@awaelchli, @Borda, @carmocca, @quintenroets, @rlizzo, @speediedan, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.8)

---

## 2.0.7: Weekly patch release
**Published:** 2023-08-16

## App

### Changed

- Removed the top-level import `lightning.pdb`; import `lightning.app.pdb` instead (#18177)
- Client retries forever (#18065)

### Fixed

- Fixed an issue that would prevent the user to set the multiprocessing start method after importing lightning (#18177)

---

## Fabric

### Changed

- Disabled the auto-detection of the Kubeflow environment (#18137)

### Fixed

- Fixed issue where DDP subprocesses that used Hydra would set hydra's working directory to current directory (#18145)
- Fixed an issue that would prevent the user to set the multiprocessing start method after importing lightning (#18177)
- Fixed an issue with `Fabric.all_reduce()` not performing an inplace operation for all backends consistently (#18235)

---

## PyTorch

### Added

- Added `LightningOptimizer.refresh()` to update the `__dict__` in case the optimizer it wraps has changed its internal state (#18280)

### Changed

- Disabled the auto-detection of the Kubeflow environment (#18137))

### Fixed

- Fixed a `Missing folder` exception when using a Google Storage URL as a `default_root_dir` (#18088)
- Fixed an issue that would prevent the user to set the multiprocessing start method after importing lightning (#18177)
- Fixed the gradient unscaling logic if the training step skipped backward (by returning `None`) (#18267)
- Ensure that the closure running inside the optimizer step has gradients enabled, even if the optimizer step has it disabled (#18268)
- Fixed an issue that could cause the `LightningOptimizer` wrapper returned by `LightningModule.optimizers()` have different internal state than the optimizer it wraps (#18280)


---

## Contributors

@0x404, @awaelchli, @bilelomrani1, @borda, @ethanwharris, @nisheethlahoti

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.7)

---

## 2.0.6: Minor patch release
**Published:** 2023-07-24

## 2.0.6

### App

- Fixed handling a `None` request in the file orchestration queue (#18111)

---

### Fabric

- Fixed `TensorBoardLogger.log_graph` not unwrapping the `_FabricModule` (#17844)

---

### PyTorch

- `LightningCLI` not saving correctly `seed_everything` when `run=True` and `seed_everything=True` (#18056)
- Fixed validation of non-PyTorch LR schedulers in manual optimization mode (#18092)
- Fixed an attribute error for `_FaultTolerantMode` when loading an old checkpoint that pickled the enum (#18094)


---

## Contributors

@awaelchli, @lantiga, @mauvilsa, @shihaoyin

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.6)

---

## 2.0.5: Minor patch release
**Published:** 2023-07-10

## App

### Added

- plugin: store source app (#17892)
- added colocation identifier (#16796)
- Added exponential backoff to HTTPQueue put (#18013)
- Content for plugins (#17243)

### Changed

- Save a reference to created tasks, to avoid tasks disappearing (#17946)

---

## Fabric

### Added

- Added validation against misconfigured device selection when using the DeepSpeed strategy (#17952)

### Changed

- Avoid info message when loading 0 entry point callbacks (#17990)

### Fixed

- Fixed the emission of a false-positive warning when calling a method on the Fabric-wrapped module that accepts no arguments (#17875)
- Fixed check for FSDP's flat parameters in all parameter groups (#17914)
- Fixed automatic step tracking in Fabric's CSVLogger (#17942)
- Fixed an issue causing the `torch.set_float32_matmul_precision` info message to show multiple times (#17960)
- Fixed loading model state when `Fabric.load()` is called after `Fabric.setup()` (#17997)

---

## PyTorch

### Fixed

- Fixed delayed creation of experiment metadata and checkpoint/log dir name when using `WandbLogger` (#17818)
- Fixed incorrect parsing of arguments when augmenting exception messages in DDP (#17948)
- Fixed an issue causing the `torch.set_float32_matmul_precision` info message to show multiple times (#17960)
- Added missing `map_location` argument for the `LightningDataModule.load_from_checkpoint` function (#17950)
- Fix support for `neptune-client` (#17939)


---

## Contributors

@anio, @awaelchli, @borda, @ethanwharris, @lantiga, @nicolai86, @rjarun8, @schmidt-ai, @schuhschuh, @wouterzwerink, @yurijmikhalevich

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.5)

---

## 2.0.4: Minor patch release
**Published:** 2023-06-22

## App

### Fixed

- bumped several dependencies to address security vulnerabilities.

---

## Fabric

### Fixed

- Fixed validation of parameters of `plugins.precision.MixedPrecision` (#17687)
- Fixed an issue with HPU imports leading to performance degradation  (#17788)

---

## PyTorch

### Changed

- Changes to the `NeptuneLogger` (#16761):
  * It now supports neptune-client 0.16.16 and neptune >=1.0, and we have replaced the `log()` method with `append()` and `extend()`.
  * It now accepts a namespace `Handler` as an alternative to `Run` for the `run` argument. This means that you can call it `NeptuneLogger(run=run["some/namespace"])` to log everything to the `some/namespace/` location of the run.

### Fixed

- Fixed validation of parameters of `plugins.precision.MixedPrecisionPlugin` (#17687)
- Fixed deriving default map location in `LightningModule.load_from_checkpoint` when there is an extra state (#17812)


---

## Contributors

@akreuzer, @awaelchli, @borda, @jerome-habana, @kshitij12345

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.4)

---

## 2.0.3: Minor patch release
**Published:** 2023-06-07

## App

### Added

- Added the property `LightningWork.public_ip` that exposes the public IP of the `LightningWork` instance (#17742)
- Add missing python-multipart dependency (#17244)

### Changed

- Made type hints public (#17100)

### Fixed

- Fixed `LightningWork.internal_ip` that was mistakenly exposing the public IP instead; now exposes the private/internal IP address (#17742)
- Fixed resolution of the latest version in CLI (#17351)
- Fixed property raised instead of returned (#17595)
- Fixed get project (#17617, #17666)

---

## Fabric

### Added

- Added support for `Callback` registration through entry points (#17756)

### Changed

- Made type hints public (#17100)
- Support compiling a module after it was set up by Fabric (#17529)

### Fixed

- Fixed computing the next version folder in `CSVLogger` (#17139)
- Fixed inconsistent settings for FSDP Precision (#17670)

---

## PyTorch

### Changed

- Made type hints public (#17100)

### Fixed

- `CombinedLoader` only starts DataLoader workers when necessary when operating in sequential mode (#17639)
- Fixed a potential bug with uploading model checkpoints to Neptune.ai by uploading files from stream (#17430)
- Fixed signature inspection of decorated hooks (#17507)
- The `WandbLogger` no longer flattens dictionaries in the hyperparameters logged to the dashboard (#17574)
- Fixed computing the next version folder in `CSVLogger` (#17139)
- Fixed a formatting issue when the filename in `ModelCheckpoint` contained metrics that were substrings of each other (#17610)
- Fixed `WandbLogger` ignoring the `WANDB_PROJECT` environment variable (#16222)
- Fixed inconsistent settings for FSDP Precision ([#17670](https://github.com/Lightning-AI/lightning/issues/17670))
- Fixed an edge case causing overlapping samples in DDP when no global seed is set (#17713)
- Fallback to module available check for mlflow (#17467)
- Fixed LR finder max val batches (#17636)
- Fixed multithreading checkpoint loading (#17678)

---

## Contributors

@adamjstewart, @AleksanderWWW, @awaelchli, @baskrahmer, @bkiat1123, @borda, @carmocca, @ethanwharris, @leng-yue, @lightningforever, @manangoel99, @mukhery, @Quasar-Kim, @water-vapor, @yurijmikhalevich

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.3)

---

## 2.0.2: Minor patch release: App jobs
**Published:** 2023-04-24

## App

### Fixed

- Resolved Lightning App with remote storage (#17426)
- Fixed `AppState`, streamlit example (#17452)

---

## Fabric

### Changed

- Enable precision autocast for LightningModule step methods in Fabric (#17439)

### Fixed

- Fixed an issue with `LightningModule.*_step` methods bypassing the DDP/FSDP wrapper (#17424)
- Fixed device handling in `Fabric.setup()` when the model has no parameters (#17441)

---

## PyTorch

### Fixed

- Fixed  `Model.load_from_checkpoint("checkpoint.ckpt", map_location=map_location)` would always return model on CPU (#17308)
- Fixed Sync module states during non-fit (#17370)
- Fixed an issue that caused `num_nodes` not to be set correctly for `FSDPStrategy` (#17438)
---

## Contributors

@awaelchli, @borda, @carmocca, @ethanwharris, @ryan597, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.2)

---

## 1.9.5: Minor patch release
**Published:** 2023-04-12

## App

### Changed

- Added `healthz` endpoint to plugin server (#16882)
- System customization syncing for jobs run (#16932)

---

## Fabric

### Changed

- Let `TorchCollective` works on the `torch.distributed` WORLD process group by default (#16995)

### Fixed

- fixed for all `_cuda_clearCublasWorkspaces` on teardown (#16907)
- Improved the error message for installing tensorboard or tensorboardx (#17053)

---

## PyTorch

### Changed

- Changed to the `NeptuneLogger` (#16761):
  * It now supports neptune-client 0.16.16 and neptune >=1.0, and we have replaced the `log()` method with `append()` and `extend()`.
  * It now accepts a namespace `Handler` as an alternative to `Run` for the `run` argument. This means that you can call it like `NeptuneLogger(run=run["some/namespace"])` to log everything to the `some/namespace/` location of the run.
- Allow `sys.argv` and args in `LightningCLI` (#16808)
- Moveed HPU broadcast override to the HPU strategy file (#17011)

### Depercated

- Removed registration of `ShardedTensor` state dict hooks in `LightningModule.__init__` with `torch>=2.1` (#16892)
- Removed the `lightning.pytorch.core.saving.ModelIO` class interface (#16974)

### Fixed

- Fixed `num_nodes` not being set for `DDPFullyShardedNativeStrategy` (#17160)
- Fixed parsing the precision config for inference in `DeepSpeedStrategy` (#16973)
- Fixed the availability check for `rich` that prevented Lightning to be imported in Google Colab (#17156)
- Fixed for all `_cuda_clearCublasWorkspaces` on teardown (#16907)
- The `psutil` package is now required for CPU monitoring (#17010)
- Improved the error message for installing tensorboard or tensorboardx (#17053)

---

## Contributors

@awaelchli, @belerico, @carmocca, @colehawkins, @dmitsf, @Erotemic, @ethanwharris, @kshitij12345, @borda

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.9.5)

---

## 2.0.1.post0: 2.0.1 appendix
**Published:** 2023-04-11

## App

### Fixed

- Fix frontend hosts when running with multi-process in the cloud ([#17324](https://github.com/Lightning-AI/lightning/pull/17324))

---

## Fabric

No changes.

---

## PyTorch

### Fixed

- Make the `is_picklable` function more robust ([#17270](https://github.com/Lightning-AI/lightning/pull/17270))

---

## Contributors

@eng-yue @ethanwharris @Borda @awaelchli @carmocca

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.1.post0)

---

## 2.0.1: 2.0.1 patch release
**Published:** 2023-03-30

## App

No changes

---

## Fabric

### Changed

- Generalized `Optimizer` validation to accommodate both FSDP 1.x and 2.x ([#16733](https://github.com/Lightning-AI/lightning/pull/16733))

---

## PyTorch

### Changed

- Pickling the `LightningModule` no longer pickles the `Trainer` ([#17133](https://github.com/Lightning-AI/lightning/pull/17133))
- Generalized `Optimizer` validation to accommodate both FSDP 1.x and 2.x ([#16733](https://github.com/Lightning-AI/lightning/pull/16733))
- Disable `torch.inference_mode` with `torch.compile` in PyTorch 2.0 ([#17215](https://github.com/Lightning-AI/lightning/pull/17215))

### Fixed

- Fixed issue where pickling the module instance would fail with a DataLoader error ([#17130](https://github.com/Lightning-AI/lightning/pull/17130))
- Fixed WandbLogger not showing "best" aliases for model checkpoints when `ModelCheckpoint(save_top_k>0)` is used ([#17121](https://github.com/Lightning-AI/lightning/pull/17121))
- Fixed the availability check for `rich` that prevented Lightning to be imported in Google Colab ([#17156](https://github.com/Lightning-AI/lightning/pull/17156))
- Fixed parsing the precision config for inference in `DeepSpeedStrategy` ([#16973](https://github.com/Lightning-AI/lightning/pull/16973))
- Fixed issue where `torch.compile` would fail when logging to WandB ([#17216](https://github.com/Lightning-AI/lightning/pull/17216))

---

## Contributors

@Borda @williamFalcon @lightningforever @adamjstewart @carmocca @tshu-w @saryazdi @parambharat @awaelchli @colehawkins @woqidaideshi @md-121 @yhl48 @gkroiz @idc9 @speediedan 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.1)

---

## 2.0.0: Lightning 2.0: Fast, Flexible, Stable
**Published:** 2023-03-15

[Lightning AI](https://lightning.ai) is excited to announce the release of Lightning 2.0 :zap:


- [Highlights](#highlights)
- [Backward Incompatible Changes](#bc-changes)
    - [PyTorch](#bc-changes-pytorch)
    - [Fabric](#bc-changes-fabric)
- [Full Changelog](#changelog)
    - [PyTorch](#changelog-pytorch)
    - [Fabric](#changelog-fabric)
    - [App](#changelog-app)
- [Contributors](#contributors)

Over the last couple of years PyTorch Lightning has become the preferred deep learning framework for researchers and ML developers around the world, with close to 50 million downloads and 18k OSS projects, from top universities to leading labs.

With the help of over 800 contributors, we have added many features and functionalities to make it the most complete research toolkit possible, but some of these changes also introduced issues:

- API changes to the trainer
- Trainer code became harder to follow
- Many integrations made Lightning appear bloated
- The trainer became harder to customize / takes away what I instead need to tweak / have control over.

To make the research experience better, we are introducing 2.0:

- No API changes - *We commit to backward compatibility in the 2.0 series*
- Simplified abstraction layers, removed legacy functionality, integrations out of the main repo. This improves the project's readability and debugging experience.
- Introducing **Fabric**. Scale any PyTorch model with just a few lines of code. Read-on!

<a name="highlights"></a>
# Highlights


## PyTorch 2.0 and `torch.compile`

Lightning 2.0 is best friends with PyTorch 2.0. You can `torch.compile` your LightningModules now!

```python
import torch
import lightning as L

model = LitModel()
# This will compile forward and {training,validation,test,predict}_step 
compiled_model = torch.compile(model)

trainer = L.Trainer()
trainer.fit(compiled_model)
```

PyTorch reports that on average, "models runs 43% faster in training on an NVIDIA A100 GPU. At Float32 precision, it runs 21% faster on average and at AMP Precision it runs 51% faster on average" ([source](https://pytorch.org/get-started/pytorch-2.0/)). If you want to learn more about `torch.compile` and how such speedups can be achieved, read the [official PyTorch 2.0 blog post](https://pytorch.org/get-started/pytorch-2.0/).


## Automatic accelerator selection ([#16847](https://github.com/Lightning-AI/lightning/pull/16847))

The `Trainer` now chooses `accelerator="auto", strategy="auto", devices="auto"` as defaults. This automatically detects the best hardware on your system (TPUs, GPUs, Apple Silicon, etc.) and chooses as many devices as are available.

```python
import lightning as L

# Selects accelerator, devices and strategy automatically!
trainer = L.Trainer()

# Same as:
trainer = L.Trainer(accelerator="auto", strategy="auto", devices="auto")
```

For example, on a 8-GPU server, this will implicitly select `Trainer(accelerator="cuda", strategy="ddp", devices=8)`.


## Support for arbitrary iterables ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))

Previously, the Trainer supported DataLoader-like iterables. However, with this release, users can now work with any iterable that implements the Python iterable definition. This includes custom data structures, such as user-defined classes and generators, as well as built-in Python objects.

To use this new feature, return any iterable (or collection of iterables) from the dataloader hooks. 

```python
def train_dataloader(self):
    return DataLoader(...)
    return list(range(1000))
    
    # pass loaders as a dict. This will create batches like this:
    # {'a': batch_from_loader_a, 'b': batch_from_loader_b}
    return {"a": DataLoader(...), "b": DataLoader(...)}
    
    # pass loaders as list. This will create batches like this:
    # [batch_from_dl_1, batch_from_dl_2]
    return [DataLoader(...), DataLoader(...)]
    
    # arbitrary nesting
    # {'a': [batch_from_dl_1, batch_from_dl_2], 'b': [batch_from_dl_3, batch_from_dl_4]}
    return {"a": [dl1, dl2], "b": [dl3, dl4]}
```

Read our [data section](https://lightning.ai/docs/pytorch/latest/data/data.html) for more information.

## Redesigned multi-dataloader support ([#16743](https://github.com/Lightning-AI/lightning/pull/16743), [#16784](https://github.com/Lightning-AI/lightning/pull/16784), [#16939](https://github.com/Lightning-AI/lightning/pull/16939))

Lightning automatically collates the batches from multiple iterables based on a "mode". This is done with our newly revamped [`CombinedLoader`](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.utilities.combined_loader.html) class. 

```python
from lightning.pytorch.utilities import CombinedLoader

iterables = {"a": DataLoader(), "b": DataLoader()}
# Lightning uses this under the hood, but this way you can change the "mode"
combined_loader = CombinedLoader(iterables, mode="min_size")

model = ...
trainer = Trainer()
trainer.fit(model, combined_loader)
```

The following modes are supported:
  * `min_size`: stops after the shortest iterable (the one with the lowest number of items) is done.
  * `max_size_cycle`: stops after the longest iterable (the one with most items) is done, while cycling through the rest of the iterables.
  * `max_size`: stops after the longest iterable (the one with most items) is done, while returning None for the exhausted iterables.
  * `sequential`: completely consumes ecah iterable sequentially, and returns a triplet `(data, idx, iterable_idx)`

If you have a need for a different "mode", feel free to open a feature request! Adding new modes is now very simplified. These improvements also allowed us to simplify the trainer's loops by abstracting this logic inside the `CombinedLoader`.

## Barebones Trainer mode ([#16854](https://github.com/Lightning-AI/lightning/pull/16854))

A new Trainer [argument](https://lightning.ai/docs/pytorch/latest/common/trainer.html#lightning.pytorch.trainer.Trainer.params.barebones) `Trainer(barebones=...)` was added (default is False) to disable all features that may impact the raw speed of the training loop. This allows users to quickly and fairily compare the runtime of a Lightning script with a raw PyTorch script. 

This is how you enable it:
```python
import lightning as L

# Default: False
trainer = L.Trainer(barebones=True)
```
A message informs about the changed settings:

```
You are running in `Trainer(barebones=True)` mode. All features that may impact raw speed have been disabled to facilitate analyzing the Trainer overhead. Specifically, the following features are deactivated:
 - Checkpointing: `Trainer(enable_checkpointing=True)`
 - Progress bar: `Trainer(enable_progress_bar=True)`
 - Model summary: `Trainer(enable_model_summary=True)`
 - Logging: `Trainer(logger=True)`, `Trainer(log_every_n_steps>0)`, `LightningModule.log(...)`, `LightningModule.log_dict(...)`
 - Sanity checking: `Trainer(num_sanity_val_steps>0)`
 - Development run: `Trainer(fast_dev_run=True)`
 - Anomaly detection: `Trainer(detect_anomaly=True)`
 - Profiling: `Trainer(profiler=...)`
```

Tip: This feature is also very useful for unit testing!


## Better progress bar ([#16695](https://github.com/Lightning-AI/lightning/pull/16695))

Based on feedback from users, we decided to separate the training progress bar from the validation bar. This greatly improves the time estimates (since validation is usually faster) and resolves confusion around the total batches being processed in an epoch.

This is how the bar looked in versions before 2.0:

```python
Epoch 3:  21%|██        | 28/128 [00:36<01:32, 23.12it/s, loss=0.163]
Validation DataLoader 0:  38%|███      | 12/32 [00:12<00:20,  1.01s/it]
```
Note how the total batches (128) is the sum of the training batches (32) and the three validation runs (3 x 32). And this is how the progress bar looks like now:

```python
Epoch 3:  50%|█████     | 16/32 [00:36<01:32, 23.12it/s]
Validation DataLoader 0:  38%|███      | 12/32 [00:12<00:20,  1.01s/it]
```
Note how the batch counts are now separate. The training progress bar pauses until validation is completed.


<a name="highlights-fabric"></a>
## Lightning Fabric

Lightning 2.0 is the official release for Lightning Fabric :tada:

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/PyTorch-to-Fabric-Spectrum-2.svg"
     alt="Fabric spans across a large spectrum - from raw PyTorch all the way to high-level PyTorch Lightning"
     style="margin-top: 50px; margin-bottom: 50px" />

Fabric is the fast and lightweight way to scale PyTorch models without boilerplate code.

- Easily switch from running on CPU to GPU (Apple Silicon, CUDA, ...), TPU, multi-GPU or even multi-node training
- State-of-the-art distributed training strategies (DDP, FSDP, DeepSpeed) and mixed precision out of the box
- Handles all the boilerplate device logic for you
- Brings useful tools to help you build a trainer (callbacks, logging, checkpoints, ...)
- Designed with multi-billion parameter models in mind

[:open_book: Go to Fabric documentation :open_book:](https://lightning.ai/docs/fabric/latest)

```diff

  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader, Dataset

+ from lightning.fabric import Fabric

  class PyTorchModel(nn.Module):
      ...

  class PyTorchDataset(Dataset):
      ...

+ fabric = Fabric(accelerator="cuda", devices=8, strategy="ddp")
+ fabric.launch()

- device = "cuda" if torch.cuda.is_available() else "cpu"
  model = PyTorchModel(...)
  optimizer = torch.optim.SGD(model.parameters())
+ model, optimizer = fabric.setup(model, optimizer)
  dataloader = DataLoader(PyTorchDataset(...), ...)
+ dataloader = fabric.setup_dataloaders(dataloader)
  model.train()

  for epoch in range(num_epochs):
      for batch in dataloader:
          input, target = batch
-         input, target = input.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(input)
          loss = loss_fn(output, target)
-         loss.backward()
+         fabric.backward(loss)
          optimizer.step()
          lr_scheduler.step()

```


<a name="bc-changes"></a>
# Backward Incompatible Changes

This section outlines notable changes that are not backward compatible with previous versions. The full list of changes and removals can be found in the [Full Changelog](#changelog) below.

Since 2.0 is a major release, we took the opportunity to take our APIs to the next level and make considerable changes to reduce the backwards incompatible changes in the future. To alleviate this, we will commit to continue supporting the 1.9.x line of releases by doing bug-fix releases with any important fixes that are necessary.

<a name="bc-changes-pytorch"></a>
## PyTorch

### The `*_epoch_end` hooks were removed ([#16520](https://github.com/Lightning-AI/lightning/pull/16520))

Since the very beginning of Lightning, the LightningModule offered the convenient `*_epoch_end` end hooks in which users could reduce metrics collected across the entire epoch to log them. For this to work, Lightning had to store all the outputs returned from the `*_step` methods internally to be able to send them to the `*_epoch_end` hook.

This "silent" accumulation of memory lead to many users scratching their head when they found that after training for 20 hours the epoch crashed randomly with an out-of-memory error. This also meant that avoiding this behaviour required code changes, as simply overriding this hook would force this behaviour, regardless of whether you used the outputs. This was exacerbated by users not knowing the difference between the `on_train_epoch_end` (does not store outputs) and `training_epoch_end` (does store outputs) hooks.

Based on this feedback, we decided to remove this mechanism completely. **Lightning 2.0 favors simplicity and speed over convenience features**.

#### Before:

```python
import lightning as L

class LitModel(L.LightningModule):
    
    def training_step(self, batch, batch_idx):
        ...
        return {"loss": loss, "banana": banana}
    
    # `outputs` is a list of all bananas returned in the epoch
    def training_epoch_end(self, outputs):
        avg_banana = torch.cat(out["banana"] for out in outputs).mean()  
```

#### Now:

```python
import lightning as L

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # 1. Create a list to hold the outputs of `*_step`
        self.bananas = []
    
    def training_step(self, batch, batch_idx):
        ...
        # 2. Add the outputs to the list
        # You should be aware of the implications on memory usage
        self.bananas.append(banana)
        return loss
    
    # 3. Rename the hook to `on_*_epoch_end`
    def on_train_epoch_end(self):
        # 4. Do something with all outputs
        avg_banana = torch.cat(self.bananas).mean()
        # Don't forget to clear the memory for the next epoch!
        self.bananas.clear()
```

The new way makes the user to manage their own list of outputs, and with it also the responsibility of managing the memory correctly. If an OOM happens, the user can identify the issue *in their own code* rather than having to guess what the Trainer does.

The prediction epoch end hook is a [special case](https://github.com/Lightning-AI/lightning/pull/16655) of this change where the outputs argument has been removed from `on_predict_epoch_end(trainer, pl_module)` but can still be accessed via the attribute `trainer.predict_loop.outputs`

You can find a migration guide for this change in this [PR's description](https://github.com/Lightning-AI/lightning/pull/16520).


### Working with multiple optimizers ([#16539](https://github.com/Lightning-AI/lightning/pull/16539))

Lightning 2.0 removed the special optimizer loop that would manage multiple optimizers in the automatic mode in favor of an overall easier to understand and debuggable training loop code.
Training with multiple optimizers is now restricted to the ["manual optimization mode"](https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html):

#### Before:

```python
import lightning as L

class LitModel(L.LightningModule):
    
    def configure_optimizers(self):
        ...
        return optimizer1, optimizer2

    def training_step(self, batch, batch_idx, optimizer_idx)
        if optimizer_idx == 0:
            ...
            return loss0
        if optimizer_idx == 1:
            ...
            return loss1
```


#### Now:

```python
import lightning as L

class LitModel(L. LightningModule):
    
    def __init__(self):
        super().__init__()
        # 1. Switch to manual optimization mode
        self.automatic_optimization = False
    
    def configure_optimizers(self):
        ...
        # 2. Return multiple optimizers, same as before
        return optimizer1, optimizer2

    # 3. Remove the `optimizer_idx` argument from `training_step`
    def training_step(self, batch, batch_idx)
        
        # 4. Grab all optimizers you want to work with
        optimizer1, optimizer2 = self.optimizers()
        ...
        
        # 5. Perform backward manually, step optimzers, etc.
        self.manual_backward(loss0)
        optimizer1.step()
        optimizer1.zero_grad()
        
        # 6. In manual optimization, you don't need to return anything
        return None
```

You may also find the utility methods `self.toggle_optimizer()` and `self.untoggle_optimizer()` useful if you need to restrict parameters that require gradients to a specific optimizer. For a complete example, see our [simple GAN implementation](https://github.com/Lightning-AI/lightning/blob/master/examples/pytorch/domain_templates/generative_adversarial_net.py).


### Truncated backpropagation through time (TBPTT) ([#16172](https://github.com/Lightning-AI/lightning/pull/16172))

Similar to the multi-optimizer loop mentioned above, truncated backpropagation through time (TBPTT) was a loop that added a lot of complexity to the Trainer. Over time, TBPTT has fallen out of fashion and today the demand from users is so low that we decided to drop special support in the framework in favor of simplifying the Trainer. TBPTT can still be done in manual optimization.

#### Before:
```python
import lightning as L

class LitModel(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.truncated_bptt_steps = 10
        self.my_rnn = ...
        
    def training_step(self, batch, batch_idx, hiddens):
        ...
        loss, hiddens = self.my_rnn(..., hiddens)
        ...
        return loss, hiddens
```

#### Now:

```python
import lightning as L

class LitModel(L.LightningModule):

    def __init__(self):
        super().__init__()
        
         # 1. Switch to manual optimization
        self.automatic_optimization = False
        
        self.truncated_bptt_steps = 10
        self.my_rnn = ...
        
    # 2. Remove the `hiddens` argument
    def training_step(self, batch, batch_idx):
        
        # 3. Split the batch in chunks along the time dimension
        split_batches = split_batch(batch, self.truncated_bptt_steps)
        
        hiddens = ...  # 3. Choose the initial hidden state
        for split_batch in range(split_batches):
            # 4. Perform the optimization in a loop
            loss, hiddens = self.my_rnn(split_batch, hiddens)
            self.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # 5. "Truncate"
            hiddens = hiddens.detach()
        
        # 6. Remove the return of `hiddens`
        # Returning loss in manual optimization is not needed
        return None
```


### Working with multiple dataloaders ([#16800](https://github.com/Lightning-AI/lightning/pull/16800), [#16753](https://github.com/Lightning-AI/lightning/pull/16753))

To simplify the Trainer interface and with the goal of simpler iterable support inside the Trainer, we removed the`Trainer(multiple_trainloader_mode=...)` argument. The mode is now agnostic to the trainer stage ("train" previously) and it's easier to debug and understand for the user as the logic is all encapsulated in the `CombinedLoader`

#### Before:

```python
import lightning as L

class LitModel(L.LightningModule):
    def train_dataloader(self):
        return [DataLoader(), DataLoader()]

model = LitModel()
trainer = Trainer(multiple_trainloader_mode="min_size")
trainer.fit(model)
```

#### Now:

```python
import lightning as L
from lightning.pytorch.utilities import CombinedLoader

class LitModel(L.LightningModule):
    def train_dataloader(self):
        iterables = [DataLoader(), DataLoader()]
        return CombinedLoader(iterables, mode="min_size")

model = LitModel()
trainer = Trainer()
trainer.fit(model)
```

Related to this, we cleaned up which hooks need the `dataloader_idx` as an input argument. Now it's only required if you use multiple dataloaders. Don't worry, the Trainer will automatically check if it's required for you and tell you about it.

### Accessing dataloaders ([#16726](https://github.com/Lightning-AI/lightning/pull/16726), [#16800](https://github.com/Lightning-AI/lightning/pull/16800))

In the case that you require access to the `DataLoader` or `Dataset` objects, iterables for each step can be accessed
via the trainer properties `Trainer.train_dataloader`, `Trainer.val_dataloaders`, `Trainer.test_dataloaders`, and `Trainer.predict_dataloaders`.

These properties will match exactly what was returned in your `*_dataloader` hooks or passed to the `Trainer`, meaning that if you returned a dictionary of dataloaders, these will return a dictionary of dataloaders. This wasn't the case in before 2.0:

#### Before:

```python!
# Passing 1 dataloader per stage
train_dataset = trainer.train_dataloader.loaders.dataset
val_dataset = trainer.val_dataloaders[0].dataset

# Passing 2 dataloaders per stage
train_dataset = trainer.train_dataloader.loaders[0].dataset
val_dataset = trainer.val_dataloaders[0].dataset
```

#### Now:

```python
# Passing 1 dataloader per stage
train_dataset = trainer.train_dataloader.dataset
val_dataset = trainer.val_dataloaders.dataset

# Passing 2 dataloaders per stage
train_dataset = trainer.train_dataloader[0].dataset
val_dataset = trainer.val_dataloaders[0].dataset
```


### The Tuner and Trainer broke up ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))

The Tuner and Trainer are no longer together. The two Trainer arguments `Trainer(auto_lr_find=..., auto_scale_batch_size=...)` and the `Trainer.tune()` method were removed to make the Trainer leaner and easier to work with.

#### Before:

```python
import lightning as L

# Tune learning rate
trainer = L.Trainer(auto_lr_find=True)
trainer.tune(model)

# Tune batch size
trainer = L.Trainer(auto_scale_batch_size=True)
trainer.tune(model)

# Fit using tuned settings
trainer.fit(model)
```

#### Now:

```python
import lightning as L

# 1. Create the Trainer
trainer = L.Trainer()

# 2. Create the Tuner
tuner = L.pytorch.tuner.Tuner(trainer)

# 3. Tune learning rate
tuner.lr_find(...)

# 4. Tune batch size
tuner.scale_batch_size(...)

# Fit using tuned settings
trainer.fit(model)
```

You can find more documentation about the tuner [here](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#batch-size-finder).


### Standardized device selection and automation

In Lightning 1.6.0, we simplified the Trainer's signature by collapsing the four accelerator-specific device arguments into a single one called `Trainer(devices=...)`. In 2.0, we are now dropping support for the old `Trainer(gpus=..., tpu_cores=..., ipus=..., num_processes=...)`.

#### Before:

```python
import lightning as L

# Multiple CPU processes
trainer = L.Trainer(num_processes=2)

# GPU selection
trainer = L.Trainer(gpus=4)

# TPU core selection
trainer = L.Trainer(tpu_cores=[1])

# Graphcore IPU devices
trainer = L.Trainer(ipus=1)
```

#### Now:

```python
import lightning as L

# Select devices and accelerator separately
trainer = L.Trainer(accelerator="cpu", devices=2)
trainer = L.Trainer(accelerator="gpu", devices=4)
trainer = L.Trainer(accelerator="tpu", devices=[1])
trainer = L.Trainer(accelerator="ipu", devices=1)

# Or let Lightning detect the accelerator automatically
trainer = L.Trainer(accelerator="auto", devices=2)

# `accelerator="auto"` is the default
trainer = L.Trainer(devices=2)
```

In addition, the `Trainer(auto_select_gpus=...)` also got removed ([#16184](https://github.com/Lightning-AI/lightning/pull/16184)). This was a problematic feature that was not well documented, often misundertood, and lead to DDP stalls due to race conditions in the device selection. It is recommended to use the `devices="auto"` instead or, if the exact previous behavior is desired, use the utility function.

#### Before:

```python
import lightning as L

trainer = L.Trainer(auto_select_gpus=True, devices=2)
```

#### Now:

```python
import lightning as L

# Recommended
trainer = L.Trainer(devices="auto")

# Alternatively, use the utility function (with care!)
from lightning.pytorch.accelerators import find_usable_cuda_devices

trainer = L.Trainer(devices=find_usable_cuda_devices(2), strategy="ddp_spawn")
```


### The slow and clunky data-parallel strategy ([#16748](https://github.com/Lightning-AI/lightning/pull/16748))

PyTorch and Lightning have discouraged the use of the [old-style DataParallel (DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) training for more than two years now. This method of multi-GPU training is slow and has many limitations that impact users, even more so in Lightning. Since DP has fallen out of fashion, and other strategies (DDP, DeepSpeed, etc.) have emerged without the same limitations, Lightning is now dropping DP completely.

#### Before:

```python
import lightning as L

# Enables DP, but with many limitations
trainer = L.Trainer(strategy="dp", devices=8)
```

#### Now:

```python
import lightning as L

# DDP is recommended for multi-GPU training
trainer = L.Trainer(strategy="ddp", devices=8)

# If devices > 1, it selects ddp for you
trainer = L.Trainer(devices=8)

# This is the same
trainer = L.Trainer(strategy="auto", devices=8)
```

Note that the DDP strategy now gets selected automatically when multiple devices are selected.
In Jupyter notebooks (Google Colab, Kaggle, etc.), Lightning will automatically select a fork-based DDP strategy (`strategy="ddp_notebook"`).

By extension, the LightningModule no longer has the `training_step_end()`, `validation_step_end()` and `test_step_end()` hooks because they were only used for reducing the outputs in DP. If you have other code that needs to run at the end of the step, you can migrate it to the corresponding `*_batch_end` hook for example.


### Loop customization has a new face

Loop customization was our attempt to make the Trainer more customizable, hackable, and extensible. The modularity it brought had many pros, but it turned out that many users didn't need customization and favored readability and simplicity instead. Hence, the concept of loop customization was completely removed from 2.0.

For users who like build custom training loops, there is now a new paradigm with Lightning Fabric and "Build your own Trainer" (BYOT). Check out the [Fabric documentation](https://lightning.ai/docs/fabric/latest/) and the [super hackable, 500-lines trainer template](https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/build_your_own_trainer).


### Mixed precision overhaul ([#16783](https://github.com/Lightning-AI/lightning/pull/16783))

Based on feedback, we decided to make the names for precision backends in `Trainer(precision=...)` clearer and less ambiguous. For example, the previous notation `Trainer(precision=16)` (which is still allowed to be used) suggested to some users that all of the weights and tensors would be stored in a `float16` format, which is not true. To solve this misunderstanding, we now distinguish these modes with "true" and "mixed" suffixes in the names:


| Recommended value                 | Short form |
|-----------------------------------|------------|
| "64-true"                         | "64", 64   |
| "32-true"                         | "32", 32   |
| "16-mixed"                        | "16", 16   |
| "bf16-mixed"                      | "bf16"     |

All documentation and examples are now recommending the new, less ambiguous names.


### Apex mixed precision gets replaced with AMP ([#16149](https://github.com/Lightning-AI/lightning/pull/16149))

In addition to reworking the precision settings, we removed the [NVIDIA/Apex](https://github.com/NVIDIA/apex) integration which got deprecated in 1.9.0. Apex itself has deprecated the mixed precision module and recommends the native `torch.amp` module in PyTorch.
For Lightning, this means you should switch to `Trainer(precision="16-mixed")`.

#### Before:

```python
import lightning as L

# This required installing nvidia/apex
trainer = L.Trainer(amp_backend="apex", amp_level="O2")
```

#### Now:

```python
import lightning as L

# Rely on PyTorch's native mixed precision
trainer = L.Trainer(precision="16-mixed")
```


### Native FSDP replaces Fairscale FSDP ([#16400](https://github.com/Lightning-AI/lightning/pull/16400))

With the recent annoucement that FSDP becomes production ready in PyTorch 2.0, we are dropping the support for the experimental Fairscale version of FSDP and go all in on the native implementation instead.

#### Before:

```python
import lightning as L

# Short-hand names (fairscale)
trainer = L.Trainer(strategy="ddp_sharded" | "ddp_fully_sharded" | "fsdp")

# Shorthand names (native PyTorch)
trainer = L.Trainer(strategy="fsdp_native" | "fsdp_native_full_shard_offload")

# Or using the strategy instance
from lightning.pytorch.strategies import DDPShardedStrategy, DDPFullyShardedStrategy

trainer = L.Trainer(strategy=DDPShardedStrategy(...))
trainer = L.Trainer(strategy=DDPFullyShardedStrategy(...))
```

#### Now:

```python
import lightning as L

# Shorthand names (native PyTorch)
trainer = L.Trainer(strategy="fsdp" | "fsdp_cpu_offload")

# Or using the strategy instance
from lightning.pytorch.strategies import FSDPStrategy

trainer = L.Trainer(strategy=FSDPStrategy(...))
```


### Resuming from checkpoints ([#16167](https://github.com/Lightning-AI/lightning/pull/16167))

Resuming from a checkpoint is no longer done by specifying the filename in the Trainer constructor. The `Trainer(resume_from_checkpoint=...)` argument was renamed and moved to the individual Trainer methods.

#### Before:

```python
import lightning as L

trainer = L.Trainer(resume_from_checkpoint="/path/to/checkpoint.ckpt")
trainer.fit(model)
trainer.test(model)
...
```

#### Now:

```python
import lightning as L

# 1. Remove `resume_from_checkpoint` from the Trainer
trainer = L.Trainer()

# 2. Add the path to any of the trainer methods you want to run
trainer.fit(model, ckpt_path="/path/to/checkpoint.ckpt")
trainer.test(model, ckpt_path="/path/to/checkpoint.ckpt")
...
```

We also added support for setting the checkpoint path statefully:

#### Now:

```python
import lightning as L

trainer = L.Trainer()

# set the checkpoint path with a setter
trainer.ckpt_path = "/path/to/checkpoint.ckpt"
trainer.fit(model)
# remember to clear it before continuing
trainer.ckpt_path = None
trainer.test()
...
```

### Logging the loss to the progress bar ([#16192](https://github.com/Lightning-AI/lightning/issues/16192))

In previous versions of Lightning, the Trainer used to automatically compute a running mean of the training loss and log it to the progress bar. We are removing this feature to favor speed over automation. The loss will no longer appear in the progress bar unless the user explicitly adds it.

#### Before:

```
Epoch 8:  53%|█████    | 17/32 [5.13/s, v_num=2, loss=0.5643]
```

#### Now:

```
Epoch 8:  53%|█████    | 17/32 [5.13/s, v_num=2]
```


```python
def training_step(self, batch, batch_idx):
    ...

    # Add this if you need the loss to be displayed in the progress bar
    self.log("loss", loss, prog_bar=True, on_step=True)
    return loss

```


### The brittle argument parsing utilities ([#16708](https://github.com/Lightning-AI/lightning/pull/16708))

In previous versions of Lightning, the `Trainer.add_argparse_args` and `Trainer.from_argparse_args` utility functions helped the user construct a commandline interface using Python's `argparse` library. Over time, new functionality was added to the Trainer that allowed most of the frequently used arguments to accept many different types. This made the arg-parsing utilities very brittle and limited in its scope. Ultimately, we decided to remove it in 2.0 in favor of more robust solutions like [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) or third-party commandline tools ([Click](https://click.palletsprojects.com), [Hydra](https://hydra.cc), etc.).


#### Before:

```python
import lightning as L
import argparse


parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
trainer = L.Trainer.from_argparse_args(args)
...
```


#### Now:

Example using the [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html):

```python
from lightning.pytorch.cli import LightningCLI

LightningCLI(MyLitModel, MyLitDataModule)
```
Run it with `python train.py fit --trainer.max_epochs=2` for example.
Alternatively, you can add the argparse arguments you want manually:
```python
import lightning as L
import argparse

parser = argparse.ArgumentParser()

# 1. Add the arguments you need
parser.add_argument("--accelerator", type=str, default="cuda")
args = parser.parse_args()

# 2. Pass them into the Trainer
trainer = L.Trainer(**vars(args))
...
```


### Gradient norm tracking ([#16745](https://github.com/Lightning-AI/lightning/pull/16745))

Until now, automatic tracking of the gradient norm was possible through the `Trainer(track_grad_norm=...)` argument. This functionality has now moved to a utility function and can be easily added in two lines of code to the LightningModule.

#### Before:

```python
import lightning as L

trainer = L.Trainer(track_grad_norm=2)

# Optionally customize logging in LightningModule
def log_grad_norm(self, grad_norm_dict):
    self.log(...)
```
    
#### Now:

```python
from lightning.pytorch.utilities import grad_norm

# 1. Override LightningModule hook
def on_before_optimizer_step(self, optimizer):
    # 2. Inspect the (unscaled) gradients here
    self.log_dict(grad_norm(self, norm_type=2))
```

This enables users to customize how the gradient norm is computed and logged, without needing to wrangle with the `Trainer` or override the `log_grad_norm` hook.

### Speeding up DDP with `find_unused_parameters` ([#16611](https://github.com/Lightning-AI/lightning/pull/16611))

When your model doesn't use all parameters in the forward-backward pass, PyTorch's DDP wrapper will freak out and inform you about it with an error. For this reason, and for the longest time, Lightning has set `find_unused_parameters=True` so that this error can be avoided. However, depending on the model, this can have significant performance impact (slowing down your training). With Lightning 2.0, we switch this flag back to `find_unused_parameters=False` (default) and favor speed over convenience.


#### Before:

```python
import lightning as L

# Previously, you had to manually override this for speed
trainer = L.Trainer(strategy="ddp_find_unused_parameters_false")
```

#### Now:
```python
import lightning as L

# Now, you get the best speed by default for all ddp variants
trainer = L.Trainer(strategy="ddp")
trainer = L.Trainer(strategy="ddp_spawn")

# We now have these if you need them
trainer = L.Trainer(strategy="ddp_find_unused_parameters_true")
```

However, it can still happen that users run into problems with unused parametes in their model. Lightning now overrides PyTorch's default error message with a custom one to help users resolve the problem.


### Sampler replacement in distributed strategies ([#16829](https://github.com/Lightning-AI/lightning/pull/16829))

We renamed the `Trainer(replace_sampler_ddp=...)` argument to `Trainer(use_distributed_sampler=...)` to communicate that the sampler gets created not only for the DDP strategies, but all distributed strategies that need it. Its function is still the same as before, and most users don't need to change its default value.

#### Before:

```python
import lightning as L

# Hmm, I wonder, does this only apply to DDP?
trainer = L.Trainer(replace_sampler_ddp=True)
```

#### Now:

```python
import lightning as L

# Aha! Lightning uses a distributed sampler by default, got it!
trainer = L.Trainer(use_distributed_sampler=True)
```


### PyTorch 1.10 and Python 3.7 no longer supported ([#16492](https://github.com/Lightning-AI/lightning/pull/16492), [#16579](https://github.com/Lightning-AI/lightning/pull/16579))

Whenever there is a new PyTorch version, or a new Python version, it is time to say good bye to the oldest one we support. With the introduction of PyTorch 2.0, we are dropping support for PyTorch 1.10 to continue our support window of the four latest versions: 1.11, 1.12, 1.13 and 2.0. Similarly, with Lightning 2.0 we support the latest three versions of Python: 3.8, 3.9, and 3.10 (3.11 is coming soon).

If you are interested which range of PyTorch or Python versions a particular Lightning version supports, see our [compatibility matrix](https://lightning.ai/docs/pytorch/latest/versioning.html#compatibility-matrix).


### Removed experimental fault-tolerance support ([#16516](https://github.com/Lightning-AI/lightning/pull/16516), [#16533](https://github.com/Lightning-AI/lightning/pull/16533))

To simplify reading and debugging the codebase, we removed the experimental support for fault-tolerance which was under the `PL_FAULT_TOLERANT_TRAINING=` environment flag. We are looking at ways to re-implement this. If you are interested in this feature, don't hesistate to reach out to us or create a feature request.

Some of the features it included were now ported to stable APIs. One feature is the [new callback](https://github.com/Lightning-AI/lightning/pull/16512) to save a checkpoint on exception:

#### Now:

```python
import lightning as L
from lightning.pytorch.callbacks import OnExceptionCheckpoint

on_exception = OnExceptionCheckpoint(".")
trainer = L.Trainer(callbacks=on_exception)
```

Another [feature](https://github.com/Lightning-AI/lightning/pull/16501) is automatic SIGTERM handling:

#### Now:

```python
import lightning as L

trainer = L.Trainer()

if trainer.received_sigterm:
    ...
```

### Removed support for `self.log()`ing a dictionary ([#16389](https://github.com/Lightning-AI/lightning/pull/16389))

Our logging mechanism previously supported `log("key", {"something": 123})` (not using `log_dict`). However, this added significant complexity to the implementation with little benefit, as these keys could not be monitored by our Callbacks and most logger implementations do not support this notation. If you were using this feature with a compatible logger, you can still publish data directly to the Logger using `self.logger.log_metrics()`.

### Removed `trainer.reset_*_dataloader()` methods ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))

These methods were not intended for public use as they could leave the Trainer in an undefined state. As a result, we have removed them. To achieve the same functionality, you can use the `Trainer(reload_dataloaders_every_n_epochs=...)` argument.

### Removed the `Trainer(move_metrics_to_cpu=True)` argument ([#16358](https://github.com/Lightning-AI/lightning/pull/16358))

This flag was designed to reduce device memory allocation at the end of an epoch. However, due to design issues, it could leave non-CPU runs in a non-functional state. Since the memory savings were minimal compared to other components and users can still manually control their metrics, we decided to remove this flag.

### Separate the Gradient Accumulation Scheduler from Trainer ([#16729](https://github.com/Lightning-AI/lightning/pull/16729))

We removed support for passing a scheduling dictionary to `Trainer(accumulate_grad_batches=...)`. The same functionality can still be achieved by simply passing the callback in. This simplifies the Trainer and the overall validation logic.

Before:
```py
trainer = Trainer(accumulate_grad_batches={"1": 5, "10": 3})
```

Now:
```py
from lightning.pytorch.callbacks import GradientAccumulationScheduler
trainer = Trainer(callbacks=GradientAccumulationScheduler({"1": 5, "10": 3}))
```

<a name="bc-changes-fabric"></a>
## Fabric


### LightningLite is dead, long live Lightning Fabric!

Over time, LightningLite has evolved from a simple onboarding tool to a powerful toolbox enabling users to build performant and hackable trainers. In 1.9.0, we renamed it to [Lightning Fabric](https://lightning.ai/docs/fabric/latest/) and gave users early access to its new features. In 2.0, we are dropping LightningLite from the package completely.

### Before:

```python
# There were two different ways of importing Lite in <= 1.9.0
from pytorch_lighting.lite import LightningLite
from lightning.lite import LightningLite

# You had to subclass LightningLite and implement `run()`
class MyTrainer(LightningLite):
    
    def run(self):
        ...
        self.backward(loss)
        ...

lite = LightningLite(...)
lite.run()
```

### Now:

```python
# 1. Import Fabric directly from the lightning package
import lightning as L

# 2. Instantiate Fabric directly, without subclassing
fabric = L.Fabric(...)

# 3. Use it in your training loop
fabric.backward(loss)
...
```

[Learn more about Fabric and what it can do in the new docs!](https://lightning.ai/docs/fabric/latest/)


<a name="changelog"></a>
# CHANGELOG

<a name="changelog-pytorch"></a>
## PyTorch

<details><summary>Added</summary>


- Added migration logic to warn about checkpoints with apex AMP state ([#16161](https://github.com/Lightning-AI/lightning/pull/16161))
- Added the `Trainer.ckpt_path = ...` setter to statefully set the checkpoint path to load. This can act as a replacement for the removed `Trainer(resume_from_checkpoint=...)` flag ([#16187](https://github.com/Lightning-AI/lightning/pull/16187))
- Added an argument `include_cuda` in `pytorch_lightning.utilities.seed.isolate_rng` to disable managing `torch.cuda`'s rng ([#16423](https://github.com/Lightning-AI/lightning/pull/16423))
- Added `Tuner.lr_find(attr_name=...)` to specify custom learning rate attribute names ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
- Added an `OnExceptionCheckpoint` callback to save a checkpoint on exception ([#16512](https://github.com/Lightning-AI/lightning/pull/16512))
- Added support for running the `MLFlowLogger` with the `mlflow-skinny` package ([16513](https://github.com/Lightning-AI/lightning/pull/16513))
- Added a `Trainer.received_sigterm` property to check whether a SIGTERM signal was received ([#16501](https://github.com/Lightning-AI/lightning/pull/16501))
- Added support for cascading a SIGTERM signal to launched processes after the launching process (rank 0) receives it ([#16525](https://github.com/Lightning-AI/lightning/pull/16525))
- Added a `kill` method to launchers to kill all launched processes ([#16525](https://github.com/Lightning-AI/lightning/pull/16525))
- Added suffix option to DDP strategy names to enable `find_unused_parameters=True`, for example `strategy="ddp_find_unused_parameters_true"` ([#16611](https://github.com/Lightning-AI/lightning/pull/16611))
- Added a new method `Strategy.on_exception` to the strategy base interface ([#16646](https://github.com/Lightning-AI/lightning/pull/16646))
- Added support for `predict_step(dataloader_iter, batch_index)` ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- Added support for arbitrary iterables as dataloaders ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- Added "sequential" mode support to `CombinedLoader` to consume multiple iterables in sequence ([#16743](https://github.com/Lightning-AI/lightning/pull/16743), [#16784](https://github.com/Lightning-AI/lightning/pull/16784))
- Added "max_size" mode support to `CombinedLoader` to consume multiple iterables entirely without cycling ([#16939](https://github.com/Lightning-AI/lightning/pull/16939)
- Added a `Trainer(barebones=True)` argument where all features that may impact raw speed are disabled ([#16854](https://github.com/Lightning-AI/lightning/pull/16854))
- Added support for writing logs remote file systems on `CSVLoggers`. ([#16880](https://github.com/Lightning-AI/lightning/pull/16880))
- Added `DDPStrategy(start_method=...)` argument, defaulting to 'popen' ([#16809](https://github.com/Lightning-AI/lightning/pull/16809))
- Added checks for whether the iterables used by the loops are valid ([#17007](https://github.com/Lightning-AI/lightning/pull/17007))

    
</details>

<details><summary>Changed</summary>


- The Trainer's signal handlers are now registered for `trainer.{validate,test,predict}` ([#17017](https://github.com/Lightning-AI/lightning/pull/17017))
- Renamed `ProgressBarBase` to `ProgressBar` ([#17058](https://github.com/Lightning-AI/lightning/pull/17058))
- The `Trainer` now chooses `accelerator="auto", strategy="auto", devices="auto"` as defaults ([#16847](https://github.com/Lightning-AI/lightning/pull/16847))
- "Native" suffix removal ([#16490](https://github.com/Lightning-AI/lightning/pull/16490))
 * `strategy="fsdp_native"` is now `strategy="fsdp"`
 * `strategy="fsdp_native_full_shard_offload"` is now `strategy="fsdp_cpu_offload"`
 * `pytorch_lightning.strategies.fully_sharded_native.DDPFullyShardedNativeStrategy` is now `pytorch_lightning.strategies.fsdp.FSDPStrategy`
 * `pytorch_lightning.plugins.precision.fsdp_native_native_amp.FullyShardedNativeNativeMixedPrecisionPlugin` is now `pytorch_lightning.plugins.precision.fsdp.FSDPMixedPrecisionPlugin`
 * `pytorch_lightning.plugins.precision.native_amp` is now `pytorch_lightning.plugins.precision.amp`
 * `NativeSyncBatchNorm` is now `TorchSyncBatchNorm`
- Changed the default of `LearningRateFinder(update_attr=...)` and `Tuner.lr_find(update_attr=...)` to `True` ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
- Renamed the `pl.utilities.exceptions.GracefulExitException` to `SIGTERMException` ([#16501](https://github.com/Lightning-AI/lightning/pull/16501))
- The `Callback.on_train_epoch_end` hook now runs after the `LightningModule.on_train_epoch_end` hook for instances of `EarlyStopping` and `Checkpoint` callbacks ([#16567](https://github.com/Lightning-AI/lightning/pull/16567))
- The `LightningModule.{un}toggle_optimizer` methods no longer accept a `optimizer_idx` argument to select the relevant optimizer. Instead, the optimizer object can be passed in directly ([#16560](https://github.com/Lightning-AI/lightning/pull/16560))
- Manual optimization is now required for working with multiple optimizers ([#16539](https://github.com/Lightning-AI/lightning/pull/16539))
- DDP's `find_unused_parameters` now defaults to `False` ([#16611](https://github.com/Lightning-AI/lightning/pull/16611))
- The strategy selected by `accelerator="hpu"` now defaults to `find_unused_parameters=False` ([#16611](https://github.com/Lightning-AI/lightning/pull/16611))
- The main progress bar displayed during training no longer includes the combined progress for validation ([#16695](https://github.com/Lightning-AI/lightning/pull/16695))
- Renamed `TQDMProgressBar.main_progress_bar` to `TQDMProgressBar.train_progress_bar` ([#16695](https://github.com/Lightning-AI/lightning/pull/16695))
- Marked the progress tracking classes as protected ([#17009](https://github.com/Lightning-AI/lightning/pull/17009))
- Marked the `lightning.pytorch.trainer.configuration_validator.verify_loop_configurations` function as protected ([#17009](https://github.com/Lightning-AI/lightning/pull/17009))
- Marked the `lightning.pytorch.utiltiies.distirbuted.register_ddp_comm_hook` function as protected ([#17009](https://github.com/Lightning-AI/lightning/pull/17009))
- Marked `lightning.pytorch.utilities.supporters.CombinedDataset` as protected ([#16714](https://github.com/Lightning-AI/lightning/pull/16714))
- Marked the `{Accelerator,Signal,Callback,Checkpoint,Data,Logger}Connector` classes as protected ([#17008](https://github.com/Lightning-AI/lightning/pull/17008))
- Marked the `lightning.pytorch.trainer.connectors.signal_connector.HandlersCompose` class as protected ([#17008](https://github.com/Lightning-AI/lightning/pull/17008))
- Disabled strict loading in multiprocessing launcher ("ddp_spawn", etc.) when loading weights back into the main process ([#16365](https://github.com/Lightning-AI/lightning/pull/16365))
- Renamed `CombinedLoader.loaders` to `CombinedLoader.iterables` ([#16743](https://github.com/Lightning-AI/lightning/pull/16743))
- Renamed `Trainer(replace_sampler_ddp=...)` to `Trainer(use_distributed_sampler=...)` ([#16829](https://github.com/Lightning-AI/lightning/pull/16829))
- Moved the `CombinedLoader` class from `lightning.pytorch.trainer.supporters` to `lightning.pytorch.combined_loader` ([#16819](https://github.com/Lightning-AI/lightning/pull/16819))
- The top-level loops now own the data sources and combined dataloaders ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- The `trainer.*_dataloader` properties now return what the user returned in their `LightningModule.*_dataloader()` hook ([#16726](https://github.com/Lightning-AI/lightning/pull/16726), [#16800](https://github.com/Lightning-AI/lightning/pull/16800))
- The `dataloader_idx` argument is now optional for the `on_{validation,test,predict}_batch_{start,end}` hooks. Remove it or default it to 0 if you don't use multiple dataloaders ([#16753](https://github.com/Lightning-AI/lightning/pull/16753))
- Renamed `TPUSpawnStrategy` to `XLAStrategy` ([#16781](https://github.com/Lightning-AI/lightning/pull/16781))
- Renamed `strategy='tpu_spawn'` to `strategy='xla'` and `strategy='tpu_spawn_debug'` to `strategy='xla_debug'` ([#16781](https://github.com/Lightning-AI/lightning/pull/16781))
- Changed arguments for precision settings (from [64|32|16|bf16] to ["64-true"|"32-true"|"16-mixed"|"bf16-mixed"]) ([#16783](https://github.com/Lightning-AI/lightning/pull/16783))
- When using multiple devices, the strategy now defaults to "ddp" instead of "ddp_spawn" when none is set ([#16780](https://github.com/Lightning-AI/lightning/pull/16780))
- The selection `Trainer(strategy="ddp_spawn", ...)` no longer falls back to "ddp" when a cluster environment gets detected ([#16780](https://github.com/Lightning-AI/lightning/pull/16780))
- Predict's custom BatchSampler that tracks the batch indices no longer consumes the entire batch sampler at the beginning ([#16826](https://github.com/Lightning-AI/lightning/pull/16826))
- Gradient norm tracking with `track_grad_norm` no longer rounds the norms to 4 digits, but instead logs them at full resolution ([#16877](https://github.com/Lightning-AI/lightning/pull/16877))
- Merged the `DDPSpawnStrategy` into `DDPStrategy` ([#16809](https://github.com/Lightning-AI/lightning/pull/16809))
- The `NeptuneLogger` now requires `neptune>=1.0.0` ([#16888](https://github.com/Lightning-AI/lightning/pull/16888))
- Changed minimum supported version of `rich` from `10.14.0` to `12.13.0` ([#16798](https://github.com/Lightning-AI/lightning/pull/16798))
- Removed the `lightning.pytorch.overrides.torch_distributed.broadcast_object_list` function ([#17011](https://github.com/Lightning-AI/lightning/pull/17011))
- The `ServableModule` is now an abstract interface ([#17000](https://github.com/Lightning-AI/lightning/pull/17000))
- The `psutil` package is now required for CPU monitoring ([#17010](https://github.com/Lightning-AI/lightning/pull/17010))
- The Trainer no longer accepts positional arguments to ([#17022](https://github.com/Lightning-AI/lightning/pull/17022))


</details>

<details><summary>Removed</summary>


- Removed support for PyTorch 1.10 ([#16492](https://github.com/Lightning-AI/lightning/pull/16492))
- Removed support for Python 3.7 ([#16579](https://github.com/Lightning-AI/lightning/pull/16579))
- Removed the `pytorch_lightning.lite` module in favor of `lightning_fabric` ([#15953](https://github.com/Lightning-AI/lightning/pull/15953))
- `nvidia/apex` removal ([#16149](https://github.com/Lightning-AI/lightning/pull/16149))
  * Removed `pytorch_lightning.plugins.NativeMixedPrecisionPlugin` in favor of `pytorch_lightning.plugins.MixedPrecisionPlugin`
  * Removed the `LightningModule.optimizer_step(using_native_amp=...)` argument
  * Removed the `Trainer(amp_backend=...)` argument
  * Removed the `Trainer.amp_backend` property
  * Removed the `Trainer(amp_level=...)` argument
  * Removed the `pytorch_lightning.plugins.ApexMixedPrecisionPlugin` class
  * Removed the `pytorch_lightning.utilities.enums.AMPType` enum
  * Removed the `DeepSpeedPrecisionPlugin(amp_type=..., amp_level=...)` arguments
- Removed `Trainer(strategy='horovod')` support ([#16150](https://github.com/Lightning-AI/lightning/pull/16150))
- `FairScale` removal (in favor of PyTorch's FSDP implementation) ([#16400](https://github.com/Lightning-AI/lightning/pull/16400))
  * Removed the `pytorch_lightning.overrides.fairscale.LightningShardedDataParallel` class
  * Removed the `pytorch_lightning.plugins.precision.fully_sharded_native_amp.FullyShardedNativeMixedPrecisionPlugin` class
  * Removed the `pytorch_lightning.plugins.precision.sharded_native_amp.ShardedNativeMixedPrecisionPlugin` class
  * Removed the `pytorch_lightning.strategies.fully_sharded.DDPFullyShardedStrategy` (fsdp) class
  * Removed the `pytorch_lightning.strategies.sharded.DDPShardedStrategy` (ddp_sharded) class
  * Removed the `pytorch_lightning.strategies.sharded_spawn.DDPSpawnShardedStrategy` (ddp_sharded_spawn) class
- Removed legacy device arguments in Trainer ([#16171](https://github.com/Lightning-AI/lightning/pull/16171))
  * Removed the `Trainer(gpus=...)` argument
  * Removed the `Trainer(tpu_cores=...)` argument
  * Removed the `Trainer(ipus=...)` argument
  * Removed the `Trainer(num_processes=...)` argument
- Removed the deprecated `pytorch_lightning.utilities.AllGatherGrad` class ([#16360](https://github.com/Lightning-AI/lightning/pull/16360))
- Removed the deprecated `resume_from_checkpoint` Trainer argument ([#16167](https://github.com/Lightning-AI/lightning/pull/16167))
- Removed the deprecated `pytorch_lightning.profiler` module ([#16359](https://github.com/Lightning-AI/lightning/pull/16359))
- Removed deadlock detection / process reconciliation (`PL_RECONCILE_PROCESS=1`) ([#16204](https://github.com/Lightning-AI/lightning/pull/16204))
- Removed the `{training,validation,test}_epoch_end` hooks which would retain step outputs in memory. Alternative implementations are suggested by implementing their `on_*_epoch_end` hooks instead ([#16520](https://github.com/Lightning-AI/lightning/pull/16520))
- Removed the `outputs` argument from the `on_predict_epoch_end` hook. You can access them via `trainer.predict_loop.predictions` ([#16655](https://github.com/Lightning-AI/lightning/pull/16655))
- Removed support for the experimental `PL_FAULT_TOLERANT_TRAINING` environment flag ([#16516](https://github.com/Lightning-AI/lightning/pull/16516), [#16533](https://github.com/Lightning-AI/lightning/pull/16533))
- Removed the deprecated `LightningCLI` arguments ([#16380](https://github.com/Lightning-AI/lightning/pull/16380))
  * `save_config_filename`
  * `save_config_overwrite`
  * `save_config_multifile`
  * `description`
  * `env_prefix`
  * `env_parse`
- Removed the deprecated `pl.strategies.utils.on_colab_kaggle` function ([#16437](https://github.com/Lightning-AI/lightning/pull/16437))
- Removed the deprecated code in:
  * `pl.core.mixins` ([#16424](https://github.com/Lightning-AI/lightning/pull/16424))
  * `pl.utilities.distributed` ([#16390](https://github.com/Lightning-AI/lightning/pull/16390))
  * `pl.utilities.apply_func` ([#16413](https://github.com/Lightning-AI/lightning/pull/16413))
  * `pl.utilities.xla_device` ([#16404](https://github.com/Lightning-AI/lightning/pull/16404))
  * `pl.utilities.data` ([#16440](https://github.com/Lightning-AI/lightning/pull/16440))
  * `pl.utilities.device_parser` ([#16412](https://github.com/Lightning-AI/lightning/pull/16412))
  * `pl.utilities.optimizer` ([#16439](https://github.com/Lightning-AI/lightning/pull/16439))
  * `pl.utilities.seed` ([#16422](https://github.com/Lightning-AI/lightning/pull/16422))
  * `pl.utilities.cloud_io` ([#16438](https://github.com/Lightning-AI/lightning/pull/16438))
- Removed the deprecated `Accelerator.setup_environment` method ([#16436](https://github.com/Lightning-AI/lightning/pull/16436))
- Mark the `forward_module` argument as required ([#16386](https://github.com/Lightning-AI/lightning/pull/16386))
  * Removed the deprecated `pl_module` argument from the distributed module wrappers
  * Removed the deprecated `pytorch_lightning.overrides.base.unwrap_lightning_module` function
  * Removed the `pytorch_lightning.overrides.distributed.LightningDistributedModule` class
  * Removed the deprecated `pytorch_lightning.overrides.fairscale.unwrap_lightning_module_sharded` function
  * Removed the `pytorch_lightning.overrides.fairscale.LightningDistributedModule` class
- Removed the deprecated automatic GPU selection ([#16184](https://github.com/Lightning-AI/lightning/pull/16184))
  * Removed the `Trainer(auto_select_gpus=...)` argument
  * Removed the `pytorch_lightning.tuner.auto_gpu_select.{pick_single_gpu,pick_multiple_gpus}` functions
- Removed support for loop customization
  * Removed `Loop.replace()` ([#16361](https://github.com/Lightning-AI/lightning/pull/16361))
  * Removed `Loop.connect()` ([#16384](https://github.com/Lightning-AI/lightning/pull/16384))
  * Removed the `trainer.{fit,validate,test,predict}_loop` properties ([#16384](https://github.com/Lightning-AI/lightning/pull/16384))
  * Removed the default `Loop.run()` implementation ([#16384](https://github.com/Lightning-AI/lightning/pull/16384))
  * The loop classes are now marked as protected ([#16445](https://github.com/Lightning-AI/lightning/pull/16445))
  * The fetching classes are now marked as protected ([#16664](https://github.com/Lightning-AI/lightning/pull/16664))
- The `lightning.pytorch.overrides.distributed.IndexBatchSamplerWrapper` class is now marked as protected ([#16826](https://github.com/Lightning-AI/lightning/pull/16826))
- Removed the `DataLoaderLoop`, `EvaluationEpochLoop`, and `PredictionEpochLoop` classes ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- Removed `trainer.reset_*_dataloader()` methods in favor of `Loop.setup_data()` for the top-level loops ([#16726](https://github.com/Lightning-AI/lightning/pull/16726))
- Removed special support for truncated backpropagation through time (TBPTT) ([#16172](https://github.com/Lightning-AI/lightning/pull/16172))
  * Removed the `LightningModule.truncated_bptt_steps` attribute
  * Removed the `LightningModule.tbptt_split_batch` hook
  * The `LightningModule.training_step` no longer accepts a `hiddens` argument
  * Removed the `pytorch_lightning.loops.batch.TrainingBatchLoop`
  * Removed the `FitLoop.split_idx` property
  * Removed the `LoggerConnector.on_train_split_start` method
- Removed the experimental `PL_INTER_BATCH_PARALLELISM` environment flag ([#16355](https://github.com/Lightning-AI/lightning/pull/16355))
- Removed the `Trainer(move_metrics_to_cpu=True)` argument ([#16358](https://github.com/Lightning-AI/lightning/pull/16358))
- Removed the `LightningModule.precision` attribute ([#16203](https://github.com/Lightning-AI/lightning/pull/16203))
- Removed the automatic addition of a moving average of the `training_step` loss in the progress bar. Use `self.log("loss", ..., prog_bar=True)` instead. ([#16192](https://github.com/Lightning-AI/lightning/issues/16192))
- Removed support for passing a dictionary value to `self.log()` ([#16389](https://github.com/Lightning-AI/lightning/pull/16389))
- Removed `Trainer.model` setter ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
- Removed the argument `Trainer(multiple_trainloader_mode=...)`. You can use `CombinedLoader(..., mode=...)` directly now ([#16800](https://github.com/Lightning-AI/lightning/pull/16800))
- Removed the unused `lightning.pytorch.utilities.finite_checks.print_nan_gradients` function ([#16682](https://github.com/Lightning-AI/lightning/pull/16682))
- Removed the unused `lightning.pytorch.utilities.finite_checks.detect_nan_parameters` function ([#16682](https://github.com/Lightning-AI/lightning/pull/16682))
- Removed the unused `lightning.pytorch.utilities.parsing.flatten_dict` function ([#16744](https://github.com/Lightning-AI/lightning/pull/16744))
- Removed the unused `lightning.pytorch.utilities.metrics.metrics_to_scalars` function ([#16681](https://github.com/Lightning-AI/lightning/pull/16681))
- Removed the unused `lightning.pytorch.utilities.supporters.{SharedCycleIteratorState,CombinedLoaderIterator}` classes ([#16714](https://github.com/Lightning-AI/lightning/pull/16714))
- Tuner removal
  * Removed the deprecated `trainer.tuning` property ([#16379](https://github.com/Lightning-AI/lightning/pull/16379))
  * Removed the deprecated `TrainerFn.TUNING` and `RunningStage.TUNING` enums ([#16379](https://github.com/Lightning-AI/lightning/pull/16379))
  * Removed `Trainer.tune()` in favor of `Tuner(trainer).{lr_find,scale_batch_size}` ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
  * Removed `Trainer(auto_scale_batch_size=...)` in favor of `Tuner(trainer).scale_batch_size()` ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
  * Removed `Trainer(auto_lr_find=...)` in favor of `Tuner(trainer).lr_find()` ([#16462](https://github.com/Lightning-AI/lightning/pull/16462))
- Removed the `on_tpu` argument from `LightningModule.optimizer_step` hook ([#16537](https://github.com/Lightning-AI/lightning/pull/16537))
- Removed the `using_lbfgs` argument from `LightningModule.optimizer_step` hook ([#16538](https://github.com/Lightning-AI/lightning/pull/16538))
- Removed the `Trainer.data_parallel` property. Use `isinstance(trainer.strategy, ParallelStrategy)` instead ([#16703](https://github.com/Lightning-AI/lightning/pull/16703))
- Removed the `Trainer.prediction_writer_callbacks` property ([#16759](https://github.com/Lightning-AI/lightning/pull/16759))
- Removed support for multiple optimizers in automatic optimization mode ([#16539](https://github.com/Lightning-AI/lightning/pull/16539))
  * Removed `opt_idx` argument from `BaseFinetuning.finetune_function` callback method
  * Removed `opt_idx` argument from `Callback.on_before_optimizer_step` callback method
  * Removed `optimizer_idx` as an optional argument in `LightningModule.training_step`
  * Removed `optimizer_idx` argument from `LightningModule.on_before_optimizer_step`
  * Removed `optimizer_idx` argument from `LightningModule.configure_gradient_clipping`
  * Removed `optimizer_idx` argument from `LightningModule.optimizer_step`
  * Removed `optimizer_idx` argument from `LightningModule.optimizer_zero_grad`
  * Removed `optimizer_idx` argument from `LightningModule.lr_scheduler_step`
  * Removed support for declaring optimizer frequencies in the dictionary returned from `LightningModule.configure_optimizers`
  * Removed arguments `optimizer` and `optimizer_idx` from `LightningModule.backward`
  * Removed `optimizer_idx` argument from `PrecisionPlugin.optimizer_step` and all of its overrides in subclasses
  * Removed `optimizer_idx` argument from `PrecisionPlugin.{optimizer_step,backward}` and all of its overrides in subclasses
  * Removed `optimizer_idx` argument from `Strategy.{optimizer_step,backward}` and all of its overrides in subclasses
  * Removed `Trainer.optimizer_frequencies` attribute
- Removed `Strategy.dispatch` ([#16618](https://github.com/Lightning-AI/lightning/pull/16618))
- Removed `PrecisionPlugin.dispatch` ([#16618](https://github.com/Lightning-AI/lightning/pull/16618))
- Removed legacy argparse utilities ([#16708](https://github.com/Lightning-AI/lightning/pull/16708))
  * Removed `LightningDataModule` methods: `add_argparse_args()`, `from_argparse_args()`, `parse_argparser()`, `get_init_arguments_and_types()`
  * Removed class methods from Trainer: `default_attributes()`, `from_argparse_args()`, `parse_argparser()`, `match_env_arguments()`, `add_argparse_args()`
  * Removed functions from `lightning.pytorch.utilities.argparse`: `from_argparse_args()`, `parse_argparser()`, `parse_env_variables()`, `get_init_arguments_and_types()`, `add_argparse_args()`
  * Removed functions from `lightning.pytorch.utilities.parsing`: `import str_to_bool()`, `str_to_bool_or_int()`, `str_to_bool_or_str()`
- Removed support for passing a scheduling dictionary to `Trainer(accumulate_grad_batches=...)` ([#16729](https://github.com/Lightning-AI/lightning/pull/16729))
- Removed support for `DataParallel` (`strategy='dp'`) and the `LightningParallelModule`-Wrapper, ([#16748](https://github.com/Lightning-AI/lightning/pull/16748))
- Removed the unused `lightning.pytorch.utilities.supporters.{SharedCycleIteratorState,CombinedLoaderIterator}` classes ([#16714](https://github.com/Lightning-AI/lightning/pull/16714))
- Removed `ProgressBarBase.{train_batch_idx,val_batch_idx,test_batch_idx,predict_batch_idx}` properties ([#16760](https://github.com/Lightning-AI/lightning/pull/16760))
- Removed the `fit_loop.{min,max}_steps` setters ([#16803](https://github.com/Lightning-AI/lightning/pull/16803))
- Removed the `Trainer(track_grad_norm=...)` argument ([#16745](https://github.com/Lightning-AI/lightning/pull/16745))
- Removed the `LightningModule.log_grad_norm()` hook method ([#16745](https://github.com/Lightning-AI/lightning/pull/16745))
- Removed the `QuantizationAwareTraining` callback ([#16750](https://github.com/Lightning-AI/lightning/pull/16750))
- Removed the `ColossalAIStrategy` and `ColossalAIPrecisionPlugin` in favor of the new [lightning-colossalai](https://github.com/Lightning-AI/lightning-colossalai) package ([#16757](https://github.com/Lightning-AI/lightning/pull/16757), [#16778](https://github.com/Lightning-AI/lightning/pull/16778))
- Removed the `training_step_end`, `validation_step_end`, and `test_step_end` hooks from the `LightningModule` in favor of the `*_batch_end` hooks ([#16791](https://github.com/Lightning-AI/lightning/pull/16791))
- Removed the `lightning.pytorch.strategies.DDPSpawnStrategy` in favor of `DDPStrategy(start_method='spawn')` (merged both classes) ([#16809](https://github.com/Lightning-AI/lightning/pull/16809))
- Removed registration of `ShardedTensor` state dict hooks in `LightningModule.__init__` with `torch>=2.1` ([#16892](https://github.com/Lightning-AI/lightning/pull/16892))
- Removed the `lightning.pytorch.core.saving.ModelIO` class interface ([#16999](https://github.com/Lightning-AI/lightning/pull/16999))
- Removed the unused `lightning.pytorch.utilities.memory.get_model_size_mb` function ([#17001](https://github.com/Lightning-AI/lightning/pull/17001))

    
</details>

<details><summary>Fixed</summary>

- Fixed an issue where `DistributedSampler.set_epoch` wasn't getting called during `trainer.predict` ([#16785](https://github.com/Lightning-AI/lightning/pull/16785), [#16826](https://github.com/Lightning-AI/lightning/pull/16826))
- Fixed an issue with comparing torch versions when using a version of torch built from source ([#17030](https://github.com/Lightning-AI/lightning/pull/17030))
- Improved the error message for installing tensorboard or tensorboardx ([#17053](https://github.com/Lightning-AI/lightning/pull/17053))


</details>

<a name="changelog-fabric"></a>
## Fabric

<details><summary>Added</summary>

- Added `Fabric.all_reduce` ([#16459](https://github.com/Lightning-AI/lightning/pull/16459))
- Added support for saving and loading DeepSpeed checkpoints through `Fabric.save/load()` ([#16452](https://github.com/Lightning-AI/lightning/pull/16452))
- Added support for automatically calling `set_epoch` on the `dataloader.batch_sampler.sampler` ([#16841](https://github.com/Lightning-AI/lightning/pull/16841))
- Added support for writing logs to remote file systems with the `CSVLogger` ([#16880](https://github.com/Lightning-AI/lightning/pull/16880))
- Added support for frozen dataclasses in the optimizer state ([#16656](https://github.com/Lightning-AI/lightning/pull/16656))
- Added `lightning.fabric.is_wrapped` to check whether a module, optimizer, or dataloader was already wrapped by Fabric ([#16953](https://github.com/Lightning-AI/lightning/pull/16953))
    
</details>

<details><summary>Changed</summary>


- Fabric now chooses `accelerator="auto", strategy="auto", devices="auto"` as defaults ([#16842](https://github.com/Lightning-AI/lightning/pull/16842))
- Checkpoint saving and loading redesign ([#16434](https://github.com/Lightning-AI/lightning/pull/16434))
  * Changed the method signatrue of `Fabric.save` and `Fabric.load`
  * Changed the method signature of `Strategy.save_checkpoint` and `Fabric.load_checkpoint`
  * `Fabric.save` accepts a state that can contain model and optimizer references
  * `Fabric.load` can now load state in-place onto models and optimizers
  * `Fabric.load` returns a dictionary of objects that weren't loaded into the state
  * `Strategy.save_checkpoint` and `Fabric.load_checkpoint` are now responsible for accessing the state of the model and optimizers
- `DataParallelStrategy.get_module_state_dict()` and `DDPStrategy.get_module_state_dict()` now correctly extracts the state dict without keys prefixed with 'module' ([#16487](https://github.com/Lightning-AI/lightning/pull/16487))
- "Native" suffix removal ([#16490](https://github.com/Lightning-AI/lightning/pull/16490))
  * `strategy="fsdp_full_shard_offload"` is now `strategy="fsdp_cpu_offload"`
  * `lightning.fabric.plugins.precision.native_amp` is now `lightning.fabric.plugins.precision.amp`
- Enabled all shorthand strategy names that can be supported in the CLI ([#16485](https://github.com/Lightning-AI/lightning/pull/16485))
- Renamed `strategy='tpu_spawn'` to `strategy='xla'` and `strategy='tpu_spawn_debug'` to `strategy='xla_debug'` ([#16781](https://github.com/Lightning-AI/lightning/pull/16781))
- Changed arguments for precision settings (from [64|32|16|bf16] to ["64-true"|"32-true"|"16-mixed"|"bf16-mixed"]) ([#16767](https://github.com/Lightning-AI/lightning/pull/16767))
- The selection `Fabric(strategy="ddp_spawn", ...)` no longer falls back to "ddp" when a cluster environment gets detected ([#16780](https://github.com/Lightning-AI/lightning/pull/16780))
- Renamed `setup_dataloaders(replace_sampler=...)` to `setup_dataloaders(use_distributed_sampler=...)` ([#16829](https://github.com/Lightning-AI/lightning/pull/16829))


</details>

<details><summary>Removed</summary>

- Removed support for PyTorch 1.10 ([#16492](https://github.com/Lightning-AI/lightning/pull/16492))
- Removed support for Python 3.7 ([#16579](https://github.com/Lightning-AI/lightning/pull/16579))

    
</details>

<details><summary>Fixed</summary>

- Fixed issue where the wrapped dataloader `iter()` would be called twice ([#16841](https://github.com/Lightning-AI/lightning/pull/16841))
- Improved the error message for installing tensorboard or tensorboardx ([#17053](https://github.com/Lightning-AI/lightning/pull/17053))

</details>

<a name="changelog-app"></a>
## App

<details><summary>Added</summary>

- Added `--zip` option to the `lightning cp` command to copy content from the Cloud Platform Filesystem as a zipfile
    
</details>

<details><summary>Changed</summary>

- Changed minimum supported version of `rich` from `10.14.0` to `12.13.0` ([#16798](https://github.com/Lightning-AI/lightning/pull/16798))

</details>

<details><summary>Removed</summary>

- Removed support for Python 3.7 ([#16579](https://github.com/Lightning-AI/lightning/pull/16579))
    
</details>


**Full commit list**: https://github.com/Lightning-AI/lightning/compare/1.9.0...2.0.0

<a name="contributors"></a>
# Contributors

### Veteran

@aniketmaurya @Atharva-Phatak @awaelchli @Borda @carmocca @dmitsf @edenlightning @hhsecond @janpawlowskiof @justusschock @krshrimali @leoleoasd @martenlienen @narJH27 @rusmux @SauravMaheshkar @shenoynikhil @tshu-w @wouterzwerink

### New

@coreyjadams @dconathan @RuRo @sergeevii123 @BrianPulfer @akkefa @belerico @tupini07 @jihoonkim2100 @eamonn-zh @lightningforever @mauvilsa @muhammadanas0716 @AdityaKane2001 @dtuit @themattinthehatt @janpawlowskiof @rusmux @Erotemic @janEbert


_If we forgot somebody or you have a suggestion, find [us on Discord](discord.gg/VptPCZkGNa) :zap:_


### Did you know?

Chuck Norris jokes aren't actually jokes, they are the truth. Chuck Norris actually can do all the things people say he can.



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.0)

---

## 1.9.4: Weekly patch release
**Published:** 2023-03-01

## App

### Removed

- Removed implicit ui testing with `testing.run_app_in_cloud` in favor of headless login and app selection ([#16741](https://github.com/Lightning-AI/lightning/pull/16741))

---

## Fabric

### Added

- Added `Fabric(strategy="auto")` support ([#16916](https://github.com/Lightning-AI/lightning/pull/16916))

### Fixed

- Fixed edge cases in parsing device ids using NVML ([#16795](https://github.com/Lightning-AI/lightning/pull/16795))
- Fixed DDP spawn hang on TPU Pods ([#16844](https://github.com/Lightning-AI/lightning/pull/16844))
- Fixed an error when passing `find_usable_cuda_devices(num_devices=-1)` ([#16866](https://github.com/Lightning-AI/lightning/pull/16866))

---

## PyTorch

### Added

- Added `Fabric(strategy="auto")` support. It will choose DDP over DDP-spawn, contrary to `strategy=None` (default) ([#16916](https://github.com/Lightning-AI/lightning/pull/16916))

### Fixed

- Fixed DDP spawn hang on TPU Pods ([#16844](https://github.com/Lightning-AI/lightning/pull/16844))
- Fixed edge cases in parsing device ids using NVML ([#16795](https://github.com/Lightning-AI/lightning/pull/16795))
- Fixed backwards compatibility for `lightning.pytorch.utilities.parsing.get_init_args` ([#16851](https://github.com/Lightning-AI/lightning/pull/16851))

---

## Contributors

@ethanwharris, @carmocca, @awaelchli, @justusschock , @dtuit, @Liyang90 


_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.9.4)

---

## 2.0.0rc0: Lightning 2.0 Release Candidate
**Published:** 2023-02-23
**Pre-release**

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.9.0...2.0.0rc0

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.0.0rc0)

---

## 1.9.3: Weekly patch release
**Published:** 2023-02-21

## App

### Fixed

- Fixed `lightning open` command and improved redirects ([#16794](https://github.com/Lightning-AI/lightning/pull/16794))

---

## Fabric

### Fixed

- Fixed an issue causing a wrong environment plugin to be selected when `accelerator=tpu` and `devices > 1` ([#16806](https://github.com/Lightning-AI/lightning/pull/16806))
- Fixed parsing of defaults for `--accelerator` and `--precision` in Fabric CLI when `accelerator` and `precision` are set to non-default values in the code ([#16818](https://github.com/Lightning-AI/lightning/pull/16818))

---

## PyTorch

### Fixed

- Fixed an issue causing a wrong environment plugin to be selected when `accelerator=tpu` and `devices > 1` ([#16806](https://github.com/Lightning-AI/lightning/pull/16806))

---

## Contributors

@ethanwharris, @carmocca, @awaelchli, @borda, @tchaton, @yurijmikhalevich


_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.9.3)

---

## 1.9.2: Weekly patch release
**Published:** 2023-02-15

## App

### Added

- Added Storage Commands ([#16740](https://github.com/Lightning-AI/lightning/pull/16740))
  * `rm`: Delete files from your Cloud Platform Filesystem
- Added `lightning connect data` to register data connection to private s3 buckets ([#16738](https://github.com/Lightning-AI/lightning/pull/16738))

---

## Fabric

### Fixed

- Fixed an attribute error and improved input validation for invalid strategy types being passed to Fabric ([#16693](https://github.com/Lightning-AI/lightning/pull/16693))

---

## PyTorch

### Changed

- Disabled strict loading in multiprocessing launcher ("ddp_spawn", etc.) when loading weights back into the main process ([#16365](https://github.com/Lightning-AI/lightning/pull/16365))

### Fixed

- Fixed an attribute error and improved input validation for invalid strategy types being passed to Trainer ([#16693](https://github.com/Lightning-AI/lightning/pull/16693))
- Fixed early stopping triggering extra validation runs after reaching `min_epochs` or `min_steps` ([#16719](https://github.com/Lightning-AI/lightning/pull/16719))


---

## Contributors

@akihironitta, @awaelchli, @borda, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.9.2)

---

## 1.9.1: Weekly patch release
**Published:** 2023-02-10

## App

### Added

- Added `lightning open` command (#16482)
- Added experimental support for interruptable GPU in the cloud (#16399)
- Added FileSystem abstraction to simply manipulate files (#16581)
- Added Storage Commands (#16606)
   * `ls`: List files from your Cloud Platform Filesystem
   * `cd`: Change the current directory within your Cloud Platform filesystem (terminal session based)
   * `pwd`: Return the current folder in your Cloud Platform Filesystem
   * `cp`: Copy files between your Cloud Platform Filesystem and local filesystem
- Prevent to `cd` into non-existent folders (#16645)
- Enabled `cp` (upload) at project level (#16631)
- Enabled `ls` and `cp` (download) at project level (#16622)
- Added `lightning connect data` to register data connection to s3 buckets (#16670)
- Added support for running with multiprocessing in the cloud (#16624)
- Initial plugin server (#16523)
- Connect and Disconnect node (#16700)

### Changed

- Changed the default `LightningClient(retry=False)` to `retry=True` (#16382)
- Add support for async predict method in PythonServer and remove torch context (#16453)
- Renamed `lightning.app.components.LiteMultiNode` to `lightning.app.components.FabricMultiNode` (#16505)
- Changed the command `lightning connect` to `lightning connect app` for consistency (#16670)
- Refactor cloud dispatch and update to new API (#16456)
- Updated app URLs to the latest format (#16568)

### Fixed

- Fixed a deadlock causing apps not to exit properly when running locally (#16623)
- Fixed the Drive root_folder not parsed properly (#16454)
- Fixed malformed path when downloading files using `lightning cp` (#16626)
- Fixed app name in URL (#16575)

---

## Fabric

### Fixed

- Fixed error handling for `accelerator="mps"` and `ddp` strategy pairing (#16455)
- Fixed strict availability check for `torch_xla` requirement (#16476)
- Fixed an issue where PL would wrap DataLoaders with XLA's MpDeviceLoader more than once (#16571)
- Fixed the batch_sampler reference for DataLoaders wrapped with XLA's MpDeviceLoader (#16571)
- Fixed an import error when `torch.distributed` is not available (#16658)

---

## Pytorch

### Fixed

- Fixed an unintended limitation for calling `save_hyperparameters` on mixin classes that don't subclass `LightningModule`/`LightningDataModule` (#16369)
- Fixed an issue with `MLFlowLogger` logging the wrong keys with `.log_hyperparams()` (#16418)
- Fixed logging more than 100 parameters with `MLFlowLogger` and long values are truncated (#16451)
- Fixed strict availability check for `torch_xla` requirement (#16476)
- Fixed an issue where PL would wrap DataLoaders with XLA's MpDeviceLoader more than once (#16571)
- Fixed the batch_sampler reference for DataLoaders wrapped with XLA's MpDeviceLoader (#16571)
- Fixed an import error when `torch.distributed` is not available (#16658)

---

## Contributors

@akihironitta, @awaelchli, @borda, @BrianPulfer, @ethanwharris, @hhsecond, @justusschock, @Liyang90, @RuRo, @senarvi, @shenoynikhil, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.9.1)

---

## 1.9.0: Stability and additional improvements
**Published:** 2023-01-17

## App

### Added

- Added a possibility to set up basic authentication for Lightning apps (#16105)

### Changed

- The LoadBalancer now uses internal ip + port instead of URL exposed (#16119)
- Added support for logging in different trainer stages with  `DeviceStatsMonitor`
(#16002)
- Changed `lightning_app.components.serve.gradio` to  `lightning_app.components.serve.gradio_server` (#16201)
- Made cluster creation/deletion async by default (#16185)

### Fixed

- Fixed not being able to run multiple lightning apps locally due to port collision (#15819)
- Avoid `relpath` bug on Windows (#16164)
- Avoid using the deprecated `LooseVersion` (#16162)
- Porting fixes to autoscaler component (#16249)
- Fixed a bug where `lightning login` with env variables would not correctly save the credentials (#16339)

---

## Fabric

### Added

- Added `Fabric.launch()` to programmatically launch processes (e.g. in Jupyter notebook) (#14992)
- Added the option to launch Fabric scripts from the CLI, without the need to wrap the code into the `run` method (#14992)
- Added `Fabric.setup_module()` and `Fabric.setup_optimizers()` to support strategies that need to set up the model before an optimizer can be created (#15185)
- Added support for Fully Sharded Data Parallel (FSDP) training in Lightning Lite (#14967)
- Added `lightning_fabric.accelerators.find_usable_cuda_devices` utility function (#16147)
- Added basic support for LightningModules (#16048)
- Added support for managing callbacks via `Fabric(callbacks=...)` and emitting events through `Fabric.call()` (#16074)
- Added Logger support (#16121)
  * Added `Fabric(loggers=...)` to support different Logger frameworks in Fabric
  * Added `Fabric.log` for logging scalars using multiple loggers
  * Added `Fabric.log_dict` for logging a dictionary of multiple metrics at once
  * Added `Fabric.loggers` and `Fabric.logger` attributes to access the individual logger instances
  * Added support for calling `self.log` and `self.log_dict` in a LightningModule when using Fabric
  * Added access to `self.logger` and `self.loggers` in a LightningModule when using Fabric
- Added `lightning_fabric.loggers.TensorBoardLogger` (#16121)
- Added `lightning_fabric.loggers.CSVLogger` (#16346)
- Added support for a consistent `.zero_grad(set_to_none=...)` on the wrapped optimizer regardless of which strategy is used (#16275)

### Changed

- Renamed the class `LightningLite` to `Fabric` (#15932, #15938)
- The `Fabric.run()` method is no longer abstract (#14992)
- The `XLAStrategy` now inherits from `ParallelStrategy` instead of `DDPSpawnStrategy` (#15838)
- Merged the implementation of `DDPSpawnStrategy` into `DDPStrategy` and removed `DDPSpawnStrategy` (#14952)
- The dataloader wrapper returned from `.setup_dataloaders()` now calls `.set_epoch()` on the distributed sampler if one is used (#16101)
- Renamed `Strategy.reduce` to `Strategy.all_reduce` in all strategies (#16370)
- When using multiple devices, the strategy now defaults to "ddp" instead of "ddp_spawn" when none is set (#16388)

### Removed

- Removed support for FairScale's sharded training (`strategy='ddp_sharded'|'ddp_sharded_spawn'`). Use Fully-Sharded Data Parallel instead (`strategy='fsdp'`) (#16329)

### Fixed

- Restored sampling parity between PyTorch and Fabric dataloaders when using the `DistributedSampler` (#16101)
- Fixes an issue where the error message wouldn't tell the user the real value that was passed through the CLI (#16334)

---

## PyTorch

### Added
- Added support for native logging of `MetricCollection` with enabled compute groups (#15580)
- Added support for custom artifact names in `pl.loggers.WandbLogger` (#16173)
- Added support for DDP with `LRFinder` (#15304)
- Added utilities to migrate checkpoints from one Lightning version to another (#15237)
- Added support to upgrade all checkpoints in a folder using the `pl.utilities.upgrade_checkpoint` script (#15333)
- Add an axes argument `ax` to the `.lr_find().plot()` to enable writing to a user-defined axes in a matplotlib figure (#15652)
- Added `log_model` parameter to `MLFlowLogger` (#9187)
- Added a check to validate that wrapped FSDP models are used while initializing optimizers (#15301)
- Added a warning when `self.log(..., logger=True)` is called without a configured logger (#15814)
- Added support for colossalai 0.1.11 (#15888)
- Added `LightningCLI` support for optimizer and learning schedulers via callable type dependency injection (#15869)
- Added support for activation checkpointing for the `DDPFullyShardedNativeStrategy` strategy (#15826)
- Added the option to set `DDPFullyShardedNativeStrategy(cpu_offload=True|False)` via bool instead of needing to pass a configuration object (#15832)
- Added info message for Ampere CUDA GPU users to enable tf32 matmul precision (#16037)
- Added support for returning optimizer-like classes in `LightningModule.configure_optimizers` (#16189)

### Changed

- Switch from `tensorboard` to `tensorboardx` in `TensorBoardLogger` (#15728)
- From now on, Lightning Trainer and `LightningModule.load_from_checkpoint` automatically upgrade the loaded checkpoint if it was produced in an old version of Lightning (#15237)
- `Trainer.{validate,test,predict}(ckpt_path=...)` no longer restores the `Trainer.global_step` and `trainer.current_epoch` value from the checkpoints - From now on, only `Trainer.fit` will restore this value (#15532)
- The `ModelCheckpoint.save_on_train_epoch_end` attribute is now computed dynamically every epoch, accounting for changes to the validation dataloaders (#15300)
- The Trainer now raises an error if it is given multiple stateful callbacks of the same time with colliding state keys (#15634)
- `MLFlowLogger` now logs hyperparameters and metrics in batched API calls (#15915)
- Overriding the `on_train_batch_{start,end}` hooks in conjunction with taking a `dataloader_iter` in the `training_step` no longer errors out and instead shows a warning (#16062)
- Move `tensorboardX` to extra dependencies. Use the `CSVLogger` by default (#16349)
- Drop PyTorch 1.9 support (#15347)

### Deprecated

- Deprecated `description`, `env_prefix` and `env_parse` parameters in `LightningCLI.__init__` in favour of giving them through `parser_kwargs` (#15651)
- Deprecated `pytorch_lightning.profiler` in favor of `pytorch_lightning.profilers` (#16059)
- Deprecated `Trainer(auto_select_gpus=...)` in favor of `pytorch_lightning.accelerators.find_usable_cuda_devices` (#16147)
- Deprecated `pytorch_lightning.tuner.auto_gpu_select.{pick_single_gpu,pick_multiple_gpus}` in favor of `pytorch_lightning.accelerators.find_usable_cuda_devices` (#16147)
- `nvidia/apex` deprecation (#16039)
  * Deprecated `pytorch_lightning.plugins.NativeMixedPrecisionPlugin` in favor of `pytorch_lightning.plugins.MixedPrecisionPlugin`
  * Deprecated the `LightningModule.optimizer_step(using_native_amp=...)` argument
  * Deprecated the `Trainer(amp_backend=...)` argument
  * Deprecated the `Trainer.amp_backend` property
  * Deprecated the `Trainer(amp_level=...)` argument
  * Deprecated the `pytorch_lightning.plugins.ApexMixedPrecisionPlugin` class
  * Deprecates the `pytorch_lightning.utilities.enums.AMPType` enum
  * Deprecates the `DeepSpeedPrecisionPlugin(amp_type=..., amp_level=...)` arguments
- `horovod` deprecation (#16141)
  * Deprecated `Trainer(strategy="horovod")`
  * Deprecated the `HorovodStrategy` class
- Deprecated `pytorch_lightning.lite.LightningLite` in favor of `lightning.fabric.Fabric` (#16314)
- `FairScale` deprecation (in favor of PyTorch's FSDP implementation) (#16353)
  * Deprecated the `pytorch_lightning.overrides.fairscale.LightningShardedDataParallel` class
  * Deprecated the `pytorch_lightning.plugins.precision.fully_sharded_native_amp.FullyShardedNativeMixedPrecisionPlugin` class
  * Deprecated the `pytorch_lightning.plugins.precision.sharded_native_amp.ShardedNativeMixedPrecisionPlugin` class
  * Deprecated the `pytorch_lightning.strategies.fully_sharded.DDPFullyShardedStrategy` class
  * Deprecated the `pytorch_lightning.strategies.sharded.DDPShardedStrategy` class
  * Deprecated the `pytorch_lightning.strategies.sharded_spawn.DDPSpawnShardedStrategy` class

### Removed

- Removed deprecated `pytorch_lightning.utilities.memory.get_gpu_memory_map` in favor of `pytorch_lightning.accelerators.cuda.get_nvidia_gpu_stats` (#15617)
- Temporarily removed support for Hydra multi-run (#15737)
- Removed deprecated `pytorch_lightning.profiler.base.AbstractProfiler` in favor of `pytorch_lightning.profilers.profiler.Profiler` (#15637)
- Removed deprecated `pytorch_lightning.profiler.base.BaseProfiler` in favor of `pytorch_lightning.profilers.profiler.Profiler` (#15637)
- Removed deprecated code in `pytorch_lightning.utilities.meta` (#16038)
- Removed the deprecated `LightningDeepSpeedModule` (#16041)
- Removed the deprecated `pytorch_lightning.accelerators.GPUAccelerator` in favor of `pytorch_lightning.accelerators.CUDAAccelerator` (#16050)
- Removed the deprecated `pytorch_lightning.profiler.*` classes in favor of `pytorch_lightning.profilers` (#16059)
- Removed the deprecated `pytorch_lightning.utilities.cli` module in favor of `pytorch_lightning.cli` (#16116)
- Removed the deprecated `pytorch_lightning.loggers.base` module in favor of `pytorch_lightning.loggers.logger` (#16120)
- Removed the deprecated `pytorch_lightning.loops.base` module in favor of `pytorch_lightning.loops.loop` (#16142)
- Removed the deprecated `pytorch_lightning.core.lightning` module in favor of `pytorch_lightning.core.module` (#16318)
- Removed the deprecated `pytorch_lightning.callbacks.base` module in favor of `pytorch_lightning.callbacks.callback` (#16319)
- Removed the deprecated `Trainer.reset_train_val_dataloaders()` in favor of `Trainer.reset_{train,val}_dataloader` (#16131)
- Removed support for `LightningCLI(seed_everything_default=None)` (#16131)
- Removed support in LightningLite for FairScale's sharded training (`strategy='ddp_sharded'|'ddp_sharded_spawn'`). Use Fully-Sharded Data Parallel instead (`strategy='fsdp'`) (#16329)

### Fixed

- Enhanced `reduce_boolean_decision` to accommodate `any`-analogous semantics expected by the `EarlyStopping` callback (#15253)
- Fixed the incorrect optimizer step synchronization when running across multiple TPU devices (#16020)
- Fixed a type error when dividing the chunk size in the ColossalAI strategy (#16212)
- Fixed bug where the ``interval`` key of the scheduler would be ignored during manual optimization, making the LearningRateMonitor callback fail to log the learning rate (#16308)
- Fixed an issue with `MLFlowLogger` not finalizing correctly when status code 'finished' was passed (#16340)

---

## Contributors

@1SAA, @akihironitta, @AlessioQuercia, @awaelchli, @bipinKrishnan, @Borda, @carmocca, @dmitsf, @erhoo82, @ethanwharris, @Forbu, @hhsecond, @justusschock, @lantiga, @lightningforever, @Liyang90, @manangoel99, @mauvilsa, @nicolai86, @nohalon, @rohitgr7, @schmidt-jake, @speediedan, @yMayanand

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.9.0)

---

## 1.8.6: Weekly patch release
**Published:** 2022-12-21

## App

### Added

- Added partial support for fastapi `Request` annotation in `configure_api` handlers (#16047)
- Added a nicer UI with URL and examples for the autoscaler component (#16063)
- Enabled users to have more control over scaling out/in intervals (#16093)
- Added more datatypes to the serving component (#16018)
- Added `work.delete` method to delete the work (#16103)
- Added `display_name` property to LightningWork for the cloud (#16095)
- Added `ColdStartProxy` to the AutoScaler (#16094)
- Added status endpoint, enable `ready` (#16075)
- Implemented `ready` for components (#16129)

### Changed

- The default `start_method` for creating Work processes locally on macOS is now 'spawn' (previously 'fork') (#16089)
- The utility `lightning.app.utilities.cloud.is_running_in_cloud` now returns `True` during the loading of the app locally when running with `--cloud` (#16045)
- Updated Multinode Warning (#16091)
- Updated app testing (#16000)
- Changed overwrite to `True` (#16009)
- Simplified messaging in cloud dispatch (#16160)
- Added annotations endpoint (#16159)

### Fixed

- Fixed `PythonServer` messaging "Your app has started" (#15989)
- Fixed auto-batching to enable batching for requests coming even after the batch interval but is in the queue (#16110)
- Fixed a bug where `AutoScaler` would fail with min_replica=0 (#16092
- Fixed a non-thread safe deepcopy in the scheduler (#16114)
- Fixed HTTP Queue sleeping for 1 sec by default if no delta was found (#16114)
- Fixed the endpoint info tab not showing up in the `AutoScaler` UI (#16128)
- Fixed an issue where an exception would be raised in the logs when using a recent version of `streamlit` (#16139)
- Fixed e2e tests (#16146)


---

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.5.post0...1.8.6

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.6)

---

## 1.8.5.post0: Minor patch release
**Published:** 2022-12-16

## App

- Fixed install/upgrade - removing single quote (#16079)
- Fixed bug where components that are re-instantiated several times failed to initialize if they were modifying `self.lightningignore` (#16080)
- Fixed a bug where apps that had previously been deleted could not be run again from the CLI (#16082)


## Pytorch

- Add function to remove checkpoint to allow override for extended classes (#16067)


---

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.5...1.8.5.post0

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.5.post0)

---

## 1.8.5: Weekly patch release
**Published:** 2022-12-15

## App

### Added

- Added `Lightning{Flow,Work}.lightningignores` attributes to programmatically ignore files before uploading to the cloud (#15818)
- Added a progress bar while connecting to an app through the CLI (#16035)
- Support running on multiple clusters (#16016)
- Added guards to cluster deletion from cli (#16053)
- Added creation of the default `.lightningignore` that ignores `venv` (#16056)

### Changed

- Cleanup cluster waiting (#16054)

### Fixed

- Fixed `DDPStrategy` import in app framework (#16029)
- Fixed `AutoScaler` raising an exception when non-default cloud compute is specified (#15991)
- Fixed and improvements of login flow (#16052)
- Fixed the debugger detection mechanism for the lightning App in VSCode (#16068)


## Pytorch

- some minor cleaning


---

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.4.post0...1.8.5

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.5)

---

## 1.8.4.post0: Minor patch release
**Published:** 2022-12-09

## App

- Fixed MultiNode Component to use separate cloud computes (#15965)
- Fixed Registration for CloudComputes of Works in `L.app.structures` (#15964)
- Fixed a bug where auto-upgrading to the latest lightning via the CLI could get stuck in a loop (#15984)


## Pytorch

- Fixed the `XLAProfiler` not recording anything due to mismatching of action names (#15885)


---

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.4...1.8.4.post0

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.4.post0)

---

## 1.8.3.post2: Dependency hotfix
**Published:** 2022-12-09

:robot: 

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.3.post2)

---

## 1.8.4: Weekly patch release
**Published:** 2022-12-08

## App

### Added

- Add `code_dir` argument to tracer run (#15771)
- Added the CLI command `lightning run model` to launch a `LightningLite` accelerated script (#15506)
- Added the CLI command `lightning delete app` to delete a lightning app on the cloud (#15783)
- Added a CloudMultiProcessBackend which enables running a child App from within the Flow in the cloud (#15800)
- Utility for pickling work object safely even from a child process (#15836)
- Added `AutoScaler` component (#15769)
- Added the property `ready` of the LightningFlow to inform when the `Open App` should be visible (#15921)
- Added private work attributed `_start_method` to customize how to start the works (#15923)
- Added a `configure_layout` method to the `LightningWork` which can be used to control how the work is handled in the layout of a parent flow (#15926)
- Added the ability to run a Lightning App or Component directly from the Gallery using `lightning run app organization/name` (#15941)
- Added automatic conversion of list and dict of works and flows to structures (#15961)

### Changed

- The `MultiNode` components now warn the user when running with `num_nodes > 1` locally (#15806)
- Cluster creation and deletion now waits by default [#15458
- Running an app without a UI locally no longer opens the browser (#15875)
- Show a message when `BuildConfig(requirements=[...])` is passed but a `requirements.txt` file is already present in the Work (#15799)
- Show a message when `BuildConfig(dockerfile="...")` is passed but a `Dockerfile` file is already present in the Work (#15799)
- Dropped name column from cluster list (#15721)
- Apps without UIs no longer activate the "Open App" button when running in the cloud (#15875)
- Wait for full file to be transferred in Path / Payload (#15934)

### Removed

- Removed the `SingleProcessRuntime` (#15933)

### Fixed

- Fixed SSH CLI command listing stopped components (#15810)
- Fixed bug when launching apps on multiple clusters (#15484)
- Fixed Sigterm Handler causing thread lock which caused KeyboardInterrupt to hang (#15881)
- Fixed MPS error for multinode component (defaults to cpu on mps devices now as distributed operations are not supported by pytorch on mps) (#15748)
- Fixed the work not stopped when successful when passed directly to the LightningApp (#15801)
- Fixed the PyTorch Inference locally on GPU (#15813)
- Fixed the `enable_spawn` method of the `WorkRunExecutor` (#15812)
- Fixed require/import decorator (#15849)
- Fixed a bug where using `L.app.structures` would cause multiple apps to be opened and fail with an error in the cloud (#15911)
- Fixed PythonServer generating noise on M1 (#15949)
- Fixed multiprocessing breakpoint (#15950)
- Fixed detection of a Lightning App running in debug mode (#15951)
- Fixed `ImportError` on Multinode if package not present (#15963)

## Lite

- Fixed `shuffle=False` having no effect when using DDP/DistributedSampler (#15931)

## Pytorch

### Changed

- Direct support for compiled models (#15922)

### Fixed

- Fixed issue with unsupported torch.inference_mode() on hpu backends (#15918)
- Fixed LRScheduler import for PyTorch 2.0 (#15940)
- Fixed `fit_loop.restarting` to be `False` for lr finder (#15620)
- Fixed `torch.jit.script`-ing a LightningModule causing an unintended error message about deprecated `use_amp` property (#15947)

---

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.3...1.8.4

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.4)

---

## 1.8.3.post1: Hotfix for Python Server
**Published:** 2022-11-25

## App

### Changed

- Fixed the PyTorch Inference locally on GPU ([#15813](https://github.com/Lightning-AI/lightning/pull/15813))

---
**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.3...1.8.3

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.3.post1)

---

## 1.8.3.post0: Hotfix for requirements
**Published:** 2022-11-23



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.3.post0)

---

## 1.8.3: Weekly patch release
**Published:** 2022-11-23

## App

### Changed

- Deduplicate top-level lighting CLI command groups (#15761)
  * `lightning add ssh-key` CLI command has been transitioned to `lightning create ssh-key`
  * `lightning remove ssh-key` CLI command has been transitioned to `lightning delete ssh-key`
- Set Torch inference mode for prediction (#15719)
- Improved `LightningTrainerScript` start-up time (#15751)
- Disable XSRF protection in `StreamlitFrontend` to support upload in localhost (#15684)

### Fixed

- Fixed debugging with VSCode IDE (#15747)
- Fixed setting property to the `LightningFlow` (#15750)

## Lite

### Changed

- Temporarily removed support for Hydra multi-run (#15737)

## Pytorch

### Changed

- Temporarily removed support for Hydra multi-run (#15737)
- Switch from `tensorboard` to `tensorboardx` in `TensorBoardLogger` (#15728)

---
**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.2...1.8.3

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.3)

---

## 1.8.2: Weekly patch release
**Published:** 2022-11-18

## App

### Added

- Added title and description to ServeGradio (#15639)
- Added a friendly error message when attempting to run the default cloud compute with a custom base image configured (#14929)

### Changed

- Improved support for running apps when dependencies aren't installed (#15711)
- Changed the root directory of the app (which gets uploaded) to be the folder containing the app file, rather than any parent folder containing a `.lightning` file (#15654)
- Enabled MultiNode Components to support state broadcasting (#15607)
- Prevent artefactual "running from outside your current environment" error (#15647)
- Rename failed -> error in tables (#15608)

### Fixed

- Fixed race condition to over-write the frontend with app infos (#15398)
- Fixed bi-directional queues sending delta with Drive Component name changes (#15642)
- Fixed CloudRuntime works collection with structures and accelerated multi node startup time (#15650)
- Fixed catimage import (#15712)
- Parse all lines in app file looking for shebangs to run commands (#15714)

## Lite

### Fixed

- Fixed the automatic fallback from `LightningLite(strategy="ddp_spawn", ...)` to `LightningLite(strategy="ddp", ...)` when on an LSF cluster (#15103)

## Pytorch

### Fixed

- Make sure save_dir can be empty str (#15638](https://github.com/PyTorchLightning/pytorch-lightning/issues/15638))
- Fixed the automatic fallback from `Trainer(strategy="ddp_spawn", ...)` to `Trainer(strategy="ddp", ...)` when on an LSF cluster (#15103](https://github.com/PyTorchLightning/pytorch-lightning/issues/15103))

---
**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.1...1.8.2

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.2)

---

## 1.8.1: Weekly patch release
**Published:** 2022-11-10

## App

### Added

- Added the `start` method to the work (#15523)
- Added a `MultiNode` Component to run with distributed computation with any frameworks (#15524)
- Expose `RunWorkExecutor` to the work and provides default ones for the `MultiNode` Component (#15561)
- Added a `start_with_flow` flag to the `LightningWork` which can be disabled to prevent the work from starting at the same time as the flow (#15591)
- Added support for running Lightning App with VSCode IDE debugger (#15590)
- Added `bi-directional` delta updates between the flow and the works (#15582)
- Added `--setup` flag to `lightning run app` CLI command allowing for dependency installation via app comments (#15577)
- Auto-upgrade / detect environment mis-match from the CLI (#15434)
- Added Serve component (#15609)


### Changed

- Changed the `flow.flows` to be recursive wont to align the behavior with the `flow.works` (#15466)
- The `params` argument in `TracerPythonScript.run` no longer prepends `--` automatically to parameters (#15518)
- Only check versions / env when not in the cloud (#15504)
- Periodically sync database to the drive (#15441)
- Slightly safer multi node (#15538)
- Reuse existing commands when running connect more than once (#15471)

### Fixed

- Fixed writing app name and id in connect.txt file for the command CLI (#15443)
- Fixed missing root flow among the flows of the app (#15531)
- Fixed bug with Multi Node Component and add some examples (#15557)
- Fixed a bug where payload would take a very long time locally (#15557)
- Fixed an issue with the `lightning` CLI taking a long time to error out when the cloud is not reachable (#15412)

## Lite

### Fixed

- Fix an issue with the SLURM `srun` detection causing permission errors (#15485)
- Fixed the import of `lightning_lite` causing a warning 'Redirects are currently not supported in Windows or MacOs' (#15610)

## PyTorch

### Fixed

- Fixed `TensorBoardLogger` not validating the input array type when logging the model graph (#15323)
- Fixed an attribute error in `ColossalAIStrategy` at import time when `torch.distributed` is not available (#15535)
- Fixed an issue when calling `fs.listdir` with file URI instead of path in `CheckpointConnector` (#15413)
- Fixed an issue with the `BaseFinetuning` callback not setting the `track_running_stats` attribute for batch normaliztion layers (#15063)
- Fixed an issue with `WandbLogger(log_model=True|'all)` raising an error and not being able to serialize tensors in the metadata (#15544)
- Fixed the gradient unscaling logic when using `Trainer(precision=16)` and fused optimizers such as `Adam(..., fused=True)` (#15544)
- Fixed model state transfer in multiprocessing launcher when running multi-node (#15567)
- Fixed manual optimization raising `AttributeError` with Bagua Strategy (#12534)
- Fixed the import of `pytorch_lightning` causing a warning 'Redirects are currently not supported in Windows or MacOs' (#15610)

---
**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.0...1.8.1

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.1)

---

## 1.8.0.post1: Minor pkg stability fix
**Published:** 2022-11-02

## What's Changed
* Implement freeze batchnorm with freezing track running stats by @PososikTeam in https://github.com/Lightning-AI/lightning/pull/15063
* Pkg: fix parsing versions by @Borda in https://github.com/Lightning-AI/lightning/pull/15401
* Remove pytest as a requirement to run app by @manskx in https://github.com/Lightning-AI/lightning/pull/15449

## New Contributors
* @PososikTeam made their first contribution in https://github.com/Lightning-AI/lightning/pull/15063

**Full Changelog**: https://github.com/Lightning-AI/lightning/compare/1.8.0...1.8.0.post1

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.0.post1)

---

## 1.8.0:  Lightning 1.8: Colossal-AI Strategy, Commands and Secrets for Apps, FSDP Improvements and More!
**Published:** 2022-11-01

The core team is excited to announce the release of Lightning 1.8 :zap:

- [Highlights](#highlights)
- [Backward Incompatible Changes](#bc-changes)
- [Deprecations](#deprecations)
- [Full Changelog](#changelog)
- [Contributors](#contributors)

Lightning v1.8 is the culmination of work from 52 contributors who have worked on features, bug-fixes, and documentation for a total of over 550+ commits since v1.7.

<a name="highlights"></a>
## Highlights

### Colossal-AI

[Colossal-AI](https://github.com/hpcaitech/ColossalAI) focuses on improving efficiency when training large-scale AI models with billions of parameters. With the new Colossal-AI strategy in Lightning 1.8, you can train existing models like GPT-3 with up to half as many GPUs as usually needed. You can also train models up to twice as big with the same number of GPUs, saving you significant cost. Here is how you use it: 


```python 
# Select the strategy with good defaults
trainer = Trainer(strategy="colossalai")

# or tune parameters to your liking
from lightning.pytorch.strategies import ColossalAIStrategy

trainer = Trainer(strategy=ColossalAIStrategy(placement_policy="cpu", ...))
```

You can find Colossal-AI's benchmarks with Lightning on GPT-2 [here](https://github.com/hpcaitech/ColossalAI-Pytorch-lightning/tree/main/benchmark).

Under the hood, Colossal-AI implements different parallelism algorithms that are especially interesting for the development of SOTA transformer models:

- Data Parallelism
- Pipeline Parallelism
- 1D, 2D, 2.5D, 3D Tensor Parallelism
- Sequence Parallelism
- Zero Redundancy Optimization


Learn how to install and use Colossal-AI effectively with Lightning [here](https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html?highlight=colossal-ai#colossal-ai).


**NOTE:** This strategy is marked as **experimental**. Stay tuned for more updates in the future.


### Secrets for Lightning Apps

Introducing encrypted secrets ([#14612](https://github.com/Lightning-AI/lightning/pull/14612)), a feature requested by Lightning App users :tada:!

Encrypted secrets allow you to securely pass private data to your apps, like API keys, access tokens, database passwords, or other credentials, without exposing them in your code.

1. Add a secret to your Lightning account in lightning.ai (read more [here](https://lightning.ai/lightning-docs/glossary/secrets.html))
2. Add an environment variable to your app to read the secret:

    ```python 
    # somewhere in your Flow or Work:
    GitHubComponent(api_token=os.environ["API_TOKEN"])
    ```

3. Pass the secret to your app run with the following command:

    ```bash
    lightning run app app.py --cloud --secret API_TOKEN=github_api_token
    ```

These secrets are encrypted and stored in the Lightning database. Nothing except your app can access the value.

**NOTE:** This is an **experimental** feature.


### CLI Commands for Lightning Apps

Introducing CLI commands for apps ([#13602](https://github.com/Lightning-AI/lightning/pull/13602))!
As a Lightning App builder, if you want to easily create a CLI interface for users to interract with your app, then this is for you.

Here is an example where users can dynamically create notebooks from the CLI.
All you need to do is implement the `configure_commands` hook on the `LightningFlow`:

```python 
import lightning as L
from commands.notebook.run import RunNotebook


class Flow(L.LightningFlow):
    ...

    def configure_commands(self):
        # Return a list of dictionaries with commands:
        return [{"run notebook": RunNotebook(method=self.run_notebook)}]


app = L.LightningApp(Flow())
```

Once the app is running with `lightning run app app.py`, you can connect to the app with the following command:

```bash
lightning connect {app name} -y
```

and run the command that was configured:

```bash
lightning run notebook --name=my_notebook_name
```
<a style="visibility:hidden">For a full tutorial and running example, visit our docs. TODO: add to docs</a>
**NOTE:** This is an **experimental** feature.


### Auto-wrapping for FSDP Strategy

In Lightning v1.7, we introduced an integration for PyTorch FSDP in the form of our FSDP strategy, which allows you to train huge models with billions of parameters sharded across hundreds of GPUs and machines. 

```python
# Native FSDP implementation
trainer = Trainer(strategy="fsdp_native")
```

We are continuing to improve the support for this feature by adding automatic wrapping of layers for use cases where the model fits into CPU memory, but not into GPU memory ([#14383](https://github.com/Lightning-AI/lightning/issues/14383)).

Here are some examples:


**Case 1:** Model is so large that it does not fit into CPU memory. 
Construct your layers in the `configure_sharded_model` hook and wrap the large ones you want to shard across GPUs:

```python 
class MassiveModel(LightningModule):
    ...
    
    # Create model here and wrap the large layers for sharding
    def configure_sharded_model(self):
        for i, layer in enumerate(self.block):
            self.block[i] = wrap(layer)
        ...
```

**Case 2:** Model fits into CPU memory, but not into GPU memory. In Lightning v1.8, you no longer need to do anything special here, as we can automatically wrap the layers for you using FSDP's policy:

```python
model = MassiveModel()
trainer = Trainer(
    accelerator="gpu", 
    devices=8, 
    strategy="fsdp_native",  # or strategy="fsdp" for fairscale
    precision=16
)

# Automatically wraps the layers here:
trainer.fit(model)
```


**Case 3:** Model fits into GPU memory. No action required, use any strategy you want. 


**Note:** if you want to manually wrap layers for more control, you can still do that!

Read more about FSDP and how layer wrapping works in our [docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#pytorch-fully-sharded-training).


### New Tuner Callbacks

In this release, we focused on Tuner improvements and introduced two new callbacks that can help you customize the batch size finder and learning rate finder as per your use case.

#### Batch Size Finder ([#11089](https://github.com/PyTorchLightning/pytorch-lightning/pull/11089))

1. You can customize the `BatchSizeFinder` callback to run at different epochs. This feature is useful while fine-tuning models since you can't always use the same batch size after unfreezing the backbone.

    ```python
    from lightning.pytorch.callbacks import BatchSizeFinder


    class FineTuneBatchSizeFinder(BatchSizeFinder):
        def __init__(self, milestones, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.milestones = milestones

        def on_fit_start(self, *args, **kwargs):
            return

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                self.scale_batch_size(trainer, pl_module)


    trainer = Trainer(callbacks=[FineTuneBatchSizeFinder(milestones=(5, 10))])
    trainer.fit(...)
    ```

2. Run batch size finder for `validate`/`test`/`predict`.

    ```python
    from lightning.pytorch.callbacks import BatchSizeFinder


    class EvalBatchSizeFinder(BatchSizeFinder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def on_fit_start(self, *args, **kwargs):
            return

        def on_test_start(self, trainer, pl_module):
            self.scale_batch_size(trainer, pl_module)


    trainer = Trainer(callbacks=[EvalBatchSizeFinder()])
    trainer.test(...)
    ```

#### Learning Rate Finder ([#13802](https://github.com/PyTorchLightning/pytorch-lightning/pull/13802))

You can now use the `LearningRateFinder` callback to run at different intervals. This feature is useful when fine-tuning models, for example.

```python
from lightning.pytorch.callbacks import LearningRateFinder


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

trainer = Trainer(callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))])
trainer.fit(...)
```


### LightningCLI Improvements


Even though the `LightningCLI` class is designed to help in the implementation of command line tools, there are instances when it might be more desirable to run directly from Python. In Lightning 1.8, you can now do this ([#14596](https://github.com/Lightning-AI/lightning/pull/14596)):

```python 
from lightning.pytorch.cli import LightningCLI

def cli_main(args):
    cli = LightningCLI(MyModel, ..., args=args)
    ...
```

Anywhere in your program, you can now call the CLI directly:

```python
cli_main(["--trainer.max_epochs=100", "--model.encoder_layers=24"])
```

[Learn about all features of the LightningCLI!](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html)


### Improvements to the SLURM Support

Multi-node training on a SLURM cluster has been supported since the inception of Lightning Trainer, and has seen several improvements over time thanks to many community contributions. And we just keep going! In this release, we've added two quality of life improvements:

- The preemption/termination signal is now configurable ([#14626](https://github.com/Lightning-AI/lightning/pull/14626)):


    ```python
    # the default signal is SIGUSR1
    trainer = Trainer(plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)])

    # customize it for your cluster
    trainer = Trainer(plugins=[SLURMEnvironment(requeue_signal=signal.SIGHUP)])
    ```

- Automatic requeuing of jobs now also works for array jobs ([#15040](https://github.com/Lightning-AI/lightning/pull/15040))! Array jobs are a convenient way to group/launch several scripts at once. When the SLURM scheduler interrupts your jobs, Lightning will save a checkpoint, resubmit a new job, and, once the scheduler allocates resources, the Trainer will resume from where it left off.


Read more about our SLURM integration [here](https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html).


<a name="bc-changes"></a>
## Backward Incompatible Changes

This section outlines notable changes that are not backward compatible with previous versions. The full list of changes and removals can be found in the CHANGELOG below.

### Callback hooks for loading and saving checkpoints

The signature and behavior of the `on_load_checkpoint` and `on_save_checkpoint` callback hooks have changed ([#14835](https://github.com/Lightning-AI/lightning/pull/14835)):

Before:
```python
def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    ...
    # previously, we were able to return state here
    return state

def on_load_checkpoint(self, trainer, pl_module, callback_state):
    # previously, only the state for this callback was passed in as argument
    ...
```


Now:
```python
def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    ...
    # returning a value here is no longer supported
    # you can modify the checkpoint dict directly
    return None


def state_dict(self):
    ...
    # Now, return state from this new method
    return state


def on_load_checkpoint(self, trainer, pl_module, checkpoint):
    # previously, only the state for this callback was passed in as argument
    ...
    
    
def load_state_dict(self, state):
    # Now, the state for this callback gets passed to this new method
    ...
```


### DataModule hooks for loading and saving checkpoints

The `on_save_checkpoint` and `on_load_checkpoint` hooks on the `LightningDataModule` have been removed in favor of the `state_dict` and `load_state_dict` methods:

```diff 
-def on_save_checkpoint(self, checkpoint):
-    checkpoint["banana"] = self.banana
+def state_dict(self):
+    return dict(banana=self.banana)


-def on_load_checkpoint(self, checkpoint):
-    self.banana = checkpoint["banana"]
+def load_state_dict(self, state):
+    self.banana = state["banana"]
```


### Callback hooks

We removed some `Callback` hooks that were ambiguous to use Removed deprecated callback hooks ([#14834](https://github.com/Lightning-AI/lightning/pull/14834)):


| Old name                     | New name                       |
|------------------------------|--------------------------------|
| `on_batch_start`             | `on_train_batch_start`         |
| `on_batch_end`               | `on_train_batch_end`           |
| `on_epoch_start`             | `on_train_epoch_start`         | 
| `on_epoch_start`             | `on_validation_epoch_start`    | 
| `on_epoch_start`             | `on_test_epoch_start`          | 
| `on_pretrain_routine_start`  | `on_fit_start`                 | 



### Trainer Device Attributes

We cleaned up the properties related to device indices ([#14829](https://github.com/Lightning-AI/lightning/pull/14829)).

The attributes `Trainer.{devices,gpus,num_gpus,ipus,tpu_cores,num_processes,root_gpu,data_parallel_device_ids}` have been removed in favor of accelerator-agnostic attributes:

```python
trainer = Trainer(...)

# access the number of devices the trainer uses on this machine ...
print(trainer.num_devices)

# ... or the device IDs
print(trainer.device_ids)
```



### Setting the torch-distributed backend

In previous versions of Lightning, switching between the "gloo" and "nccl" backends for multi-GPU, multi-node training was possible through setting an environment variable like so:

```bash
PL_TORCH_DISTRIBUTED_BACKEND="gloo" python train.py
```

But not all strategies support changing the backend in this way.
From now on, the backend has to be set in the code ([#14693](https://github.com/Lightning-AI/lightning/pull/14693)):

```python
trainer = Trainer(strategy=DDPStrategy(process_group_backend="gloo"))
```

The default remains "nccl", and you should choose "gloo" only for debugging purposes.




### Logging with multiple loggers

Logging with multiple loggers can be super useful (and super easy with Lightning). For example, you could be using one logger to record sensitive image logs to a hosted MLFlow server within your organization, and at the same time log loss curves online to WandB. 

```python
trainer = Trainer(
    loggers=[WandbLogger(...), MLFlowLogger(...)]
)
```

Here are two major changes that apply when using multiple loggers in 1.8:

- Checkpoints and profiler reports no longer go to a strange folder with a long, hard to remember name ([#14325](https://github.com/Lightning-AI/lightning/pull/14325)). From now on, these arifacts will land in the version folder of the **first** logger in the list. 

- The loggers used to be wrapped by a `LoggerCollection` object, so that when you accessed `trainer.logger` you could log to all of them simultaneously. However, this "magic" caused confusion and errors among users and we decided to simplify this ([#14283](https://github.com/Lightning-AI/lightning/pull/14283)):

    ```python
    # now returns the first logger in the list
    print(trainer.logger)

    # access all loggers in a list with plural
    loggers = trainer.loggers

    for logger in loggers:
        logger.do_something()
    ```


<a name="deprecations"></a>
## Deprecations

**Why is Lightning deprecating APIs in every release?**

Many users have this question, and it is a fair one! Deprecations are a normal part of API evolution in all software. We continually improve Lightning, which means we make APIs like class names, methods, hooks and arguments clear, easy to remember, and general enough to adopt more functionality in the future. Sometimes we have to let old things go to build new and better products.

Learn more about our deprecation window [here](https://pytorch-lightning.readthedocs.io/en/stable/governance.html#api-evolution).

So far, we have followed the pattern of removing deprecated functionality and APIs after two minor versions of deprecation. From Lightning 1.8 onward, we will additionaly convert warnings to error messages after the deprecation phase ends. This way, we can greatly improve the upgrade experience with helpful messages for users who skip more than two minor Lightning versions. The exception to this rule are experimental features, which are marked as such in our documentation.

Here is a summary of major deprecations introduced in 1.8:

| API                                                                                                                      | Removal version | Alternative                                     |
|--------------------------------------------------------------------------------------------------------------------------|-----------------|-------------------------------------------------|
| Argument `Trainer(amp_level=...)`                                                        | 1.10             | `Trainer(plugins=[ApexMixedPrecisionPlugin(amp_level=...)])`   |
| Function `unwrap_lightning_module`                                                             | 1.10             | `Strategy.lightning_module`          |
| Function `unwrap_lightning_module_sharded`                                                              | 1.10             | `Strategy.lightning_module`          |
| Import `pl.core.mixins.DeviceDtypeModuleMixin`                                                           | 1.10             | No longer supported |
| Argument `LightningCLI(save_config_filename=...)`                                    | 1.10             | `LightningCLI(save_config_kwargs=dict(config_filename=...))`         |
| Argument `LightningCLI(save_config_overwrite=...)`                                    | 1.10             | `LightningCLI(save_config_kwargs=dict(overwrite=...))`         |
| Argument `LightningCLI(save_config_multifile=...)`                                    | 1.10             | `LightningCLI(save_config_kwargs=dict(multifile=...))`         |
| Enum `TrainerFn.TUNING`                                    | 1.10             | No longer supported         |
| Enum `RunningStage.TUNING`                                    | 1.10             | No longer supported         |
| Attribute `Trainer.tuning`                                    | 1.10             | No longer supported         |


<a name="changelog"></a>
## CHANGELOG

### Lightning App

<details><summary>Added</summary>

- Added `load_state_dict` and `state_dict` hooks for `LightningFlow` components ([#14100](https://github.com/Lightning-AI/lightning/pull/14100))
- Added a `--secret` option to CLI to allow binding secrets to app environment variables when running in the cloud ([#14612](https://github.com/Lightning-AI/lightning/pull/14612))
- Added support for running the works without cloud compute in the default container ([#14819](https://github.com/Lightning-AI/lightning/pull/14819))
- Added an HTTPQueue as an optional replacement for the default redis queue ([#14978](https://github.com/Lightning-AI/lightning/pull/14978)
- Added support for configuring flow cloud compute ([#14831](https://github.com/Lightning-AI/lightning/pull/14831))
- Added support for adding descriptions to commands either through a docstring or the `DESCRIPTION` attribute ([#15193](https://github.com/Lightning-AI/lightning/pull/15193)
- Added a try / catch mechanism around request processing to avoid killing the flow ([#15187](https://github.com/Lightning-AI/lightning/pull/15187)
- Added an Database Component ([#14995](https://github.com/Lightning-AI/lightning/pull/14995)
- Added authentication to HTTP queue ([#15202](https://github.com/Lightning-AI/lightning/pull/15202))
- Added support to pass a `LightningWork` to the `LightningApp` ([#15215](https://github.com/Lightning-AI/lightning/pull/15215)
- Added support getting CLI help for connected apps even if the app isn't running ([#15196](https://github.com/Lightning-AI/lightning/pull/15196)
- Added support for adding requirements to commands and installing them when missing when running an app command ([#15198](https://github.com/Lightning-AI/lightning/pull/15198)
- Added Lightning CLI Connection to be terminal session instead of global ([#15241](https://github.com/Lightning-AI/lightning/pull/15241)
- Added support for managing SSH-keys via CLI ([#15291](https://github.com/Lightning-AI/lightning/pull/15291))
- Add a `JustPyFrontend` to ease UI creation with `https://github.com/justpy-org/justpy` ([#15002](https://github.com/Lightning-AI/lightning/pull/15002))
- Added a layout endpoint to the Rest API and enable to disable pulling or pushing to the state ([#15367](https://github.com/Lightning-AI/lightning/pull/15367)
- Added support for functions for `configure_api` and `configure_commands` to be executed in the Rest API process ([#15098](https://github.com/Lightning-AI/lightning/pull/15098)
- Added support to start lightning app on cloud without needing to install dependencies locally ([#15019](https://github.com/Lightning-AI/lightning/pull/15019)
    
</details>


<details><summary>Changed</summary>
    
- Improved the show logs command to be standalone and re-usable ([#15343](https://github.com/Lightning-AI/lightning/pull/15343)
- Removed the `--instance-types` option when creating clusters ([#15314](https://github.com/Lightning-AI/lightning/pull/15314))

</details>

<details><summary>Fixed</summary>


- Fixed an issue when using the CLI without arguments ([#14877](https://github.com/Lightning-AI/lightning/pull/14877))
- Fixed a bug where the upload files endpoint would raise an error when running locally ([#14924](https://github.com/Lightning-AI/lightning/pull/14924))
- Fixed BYOC cluster region selector -> hiding it from help since only us-east-1 has been tested and is recommended ([#15277]https://github.com/Lightning-AI/lightning/pull/15277)
- Fixed a bug when launching an app on multiple clusters ([#15226](https://github.com/Lightning-AI/lightning/pull/15226))
- Fixed a bug with a default CloudCompute for Lightning flows ([#15371](https://github.com/Lightning-AI/lightning/pull/15371))

    
</details>


### Lightning Trainer

<details><summary>Added</summary>

- Added support for requeueing slurm array jobs ([#15040](https://github.com/Lightning-AI/lightning/pull/15040))
- Added native AMP support for `ddp_fork` (and associated alias strategies) with CUDA GPUs ([#14983](https://github.com/Lightning-AI/lightning/pull/14983))
- Added `BatchSizeFinder` callback ([#11089](https://github.com/Lightning-AI/lightning/pull/11089))
- Added `LearningRateFinder` callback ([#13802](https://github.com/Lightning-AI/lightning/pull/13802))
- Tuner now supports a new `method` argument which will determine when to run the `BatchSizeFinder`: one of `fit`, `validate`, `test` or `predict` ([#11089](https://github.com/Lightning-AI/lightning/pull/11089))
- Added prefix to log message in `seed_everything` with rank info ([#14031](https://github.com/Lightning-AI/lightning/pull/14031))
- Added support for auto wrapping for `DDPFullyShardedNativeStrategy` ([#14252](https://github.com/Lightning-AI/lightning/pull/14252))
- Added support for passing extra init-parameters to the `LightningDataModule.from_datasets` ([#14185](https://github.com/Lightning-AI/lightning/pull/14185))
- Added support for saving sharded optimizer state dict outside of `DDPShardedStrategy` ([#14208](https://github.com/Lightning-AI/lightning/pull/14208))
- Added support for auto wrapping for `DDPFullyShardedStrategy` ([#14383](https://github.com/Lightning-AI/lightning/pull/14383))
- Integrate the `lightning_utilities` package (
  [#14475](https://github.com/Lightning-AI/lightning/pull/14475),
  [#14537](https://github.com/Lightning-AI/lightning/pull/14537),
  [#14556](https://github.com/Lightning-AI/lightning/pull/14556),
  [#14558](https://github.com/Lightning-AI/lightning/pull/14558),
  [#14575](https://github.com/Lightning-AI/lightning/pull/14575),
  [#14620](https://github.com/Lightning-AI/lightning/pull/14620))
- Added `args` parameter to `LightningCLI` to ease running from within Python ([#14596](https://github.com/Lightning-AI/lightning/pull/14596))
- Added `WandbLogger.download_artifact` and `WandbLogger.use_artifact` for managing artifacts with Weights and Biases ([#14551](https://github.com/Lightning-AI/lightning/pull/14551))
- Added an option to configure the signal SLURM sends when a job is preempted or requeued ([#14626](https://github.com/Lightning-AI/lightning/pull/14626))
- Added a warning when the model passed to `LightningLite.setup()` does not have all parameters on the same device ([#14822](https://github.com/Lightning-AI/lightning/pull/14822))
- The `CometLogger` now flags the Comet Experiments as being created from Lightning for analytics purposes ([#14906](https://github.com/Lightning-AI/lightning/pull/14906))
- Introduce `ckpt_path="hpc"` keyword for checkpoint loading ([#14911](https://github.,com/Lightning-AI/lightning/pull/14911))
- Added a more descriptive error message when attempting to fork processes with pre-initialized CUDA context ([#14709](https://github.com/Lightning-AI/lightning/pull/14709))
- Added support for custom parameters in subclasses of `SaveConfigCallback` ([#14998](https://github.com/Lightning-AI/lightning/pull/14998))
- Added `inference_mode` flag to Trainer to let users enable/disable inference mode during evaluation ([#15034](https://github.com/Lightning-AI/lightning/pull/15034))
- Added `LightningLite.no_backward_sync` for control over efficient gradient accumulation with distributed strategies ([#14966](https://github.com/Lightning-AI/lightning/pull/14966))
- Added a sanity check that scripts are executed with the `srun` command in SLURM and that environment variables are not conflicting ([#15011](https://github.com/Lightning-AI/lightning/pull/15011))
- Added an error message when attempting to launch processes with `python -i` and an interactive-incompatible strategy ([#15293](https://github.com/Lightning-AI/lightning/pull/15293))


</details>

<details><summary>Changed</summary>

- The `Trainer.{fit,validate,test,predict,tune}` methods now raise a useful error message if the input is not a `LightningModule` ([#13892](https://github.com/Lightning-AI/lightning/pull/13892))
- Raised a `MisconfigurationException` if batch transfer hooks are overriden with `IPUAccelerator` ([#13961](https://github.com/Lightning-AI/lightning/pull/13961))
- Replaced the unwrapping logic in strategies with direct access to unwrapped `LightningModule` ([#13738](https://github.com/Lightning-AI/lightning/pull/13738))
- Enabled `on_before_batch_transfer` for `DPStrategy` and `IPUAccelerator` ([#14023](https://github.com/Lightning-AI/lightning/pull/14023))
- When resuming training with Apex enabled, the `Trainer` will now raise an error ([#14341](https://github.com/Lightning-AI/lightning/pull/14341))
- Included `torch.cuda` rng state to the aggregate `_collect_rng_states()` and `_set_rng_states()` ([#14384](https://github.com/Lightning-AI/lightning/pull/14384))
- Changed `trainer.should_stop` to not stop in between an epoch and run until `min_steps/min_epochs` only ([#13890](https://github.com/Lightning-AI/lightning/pull/13890))
- The `pyDeprecate` dependency is no longer installed ([#14472](https://github.com/Lightning-AI/lightning/pull/14472))
- When using multiple loggers, by default checkpoints and profiler output now get saved to the log dir of the first logger in the list ([#14325](https://github.com/Lightning-AI/lightning/pull/14325))
- In Lightning Lite, state-dict access to the module wrapper now gets passed through to the original module reference ([#14629](https://github.com/Lightning-AI/lightning/pull/14629))
- Removed fall-back to `LightningEnvironment` when number of SLURM tasks does not correspond to number of processes in Trainer ([#14300](https://github.com/Lightning-AI/lightning/pull/14300))
- Aligned DDP and DDPSpawn strategies in setting up the environment ([#11073](https://github.com/Lightning-AI/lightning/pull/11073))
- Integrated the Lite Precision plugins into the PL Precision plugins - the base class in PL now extends the `lightning_lite.precision.Precision` base class ([#14798](https://github.com/Lightning-AI/lightning/pull/14798))
  * The `PrecisionPlugin.backward` signature changed: The `closure_loss` argument was renamed to `tensor`
  * The `PrecisionPlugin.{pre_,post_}backward` signature changed: The `closure_loss` argument was renamed to `tensor` and moved as the first argument
  * The `PrecisionPlugin.optimizer_step` signature changed: The `model`, `optimizer_idx` and `closure` arguments need to be passed as keyword arguments now
- Trainer queries the CUDA devices through NVML if available to avoid initializing CUDA before forking, which eliminates the need for the `PL_DISABLE_FORK` environment variable introduced in v1.7.4 ([#14631](https://github.com/Lightning-AI/lightning/pull/14631))
- The `MLFlowLogger.finalize()` now sets the status to `FAILED` when an exception occurred in `Trainer`, and sets the status to `FINISHED` on successful completion ([#12292](https://github.com/Lightning-AI/lightning/pull/12292))
- It is no longer needed to call `model.double()` when using `precision=64` in Lightning Lite ([#14827](https://github.com/Lightning-AI/lightning/pull/14827))
- HPC checkpoints are now loaded automatically only in slurm environment when no specific value for `ckpt_path` has been set ([#14911](https://github.com/Lightning-AI/lightning/pull/14911))
- The `Callback.on_load_checkpoint` now gets the full checkpoint dictionary and the `callback_state` argument was renamed `checkpoint` ([#14835](https://github.com/Lightning-AI/lightning/pull/14835))
- Moved the warning about saving nn.Module in `save_hyperparameters()` to before the deepcopy ([#15132](https://github.com/Lightning-AI/lightning/pull/15132))
- To avoid issues with forking processes, from PyTorch 1.13 and higher, Lightning will directly use the PyTorch NVML-based check for `torch.cuda.device_count` and from PyTorch 1.14 and higher, Lightning will configure PyTorch to use a NVML-based check for `torch.cuda.is_available`. ([#15110](https://github.com/Lightning-AI/lightning/pull/15110), [#15133](https://github.com/Lightning-AI/lightning/pull/15133))
- The `NeptuneLogger` now uses `neptune.init_run` instead of the deprecated `neptune.init` to initialize a run ([#15393](https://github.com/Lightning-AI/lightning/pull/15393))


</details>

<details><summary>Deprecated</summary>


- Deprecated `LightningDeepSpeedModule` ([#14000](https://github.com/Lightning-AI/lightning/pull/14000))
- Deprecated `amp_level` from `Trainer` in favour of passing it explictly via precision plugin ([#13898](https://github.com/Lightning-AI/lightning/pull/13898))
- Deprecated the calls to `pytorch_lightning.utiltiies.meta` functions in favor of built-in https://github.com/pytorch/torchdistx support ([#13868](https://github.com/Lightning-AI/lightning/pull/13868))
- Deprecated the `unwrap_lightning_module` and `unwrap_lightning_module_sharded` utility functions in favor of accessing the unwrapped `LightningModule` on the strategy directly ([#13738](https://github.com/Lightning-AI/lightning/pull/13738))
- Deprecated the `pl_module` argument in `LightningParallelModule`, `LightningDistributedModule`, `LightningShardedDataParallel`, `LightningBaguaModule` and `LightningDeepSpeedModule` wrapper classes ([#13738](https://github.com/Lightning-AI/lightning/pull/13738))
- Deprecated the `on_colab_kaggle` function ([#14247](https://github.com/Lightning-AI/lightning/pull/14247))
- Deprecated the internal `pl.core.mixins.DeviceDtypeModuleMixin` class ([#14511](https://github.com/Lightning-AI/lightning/pull/14511), [#14548](https://github.com/Lightning-AI/lightning/pull/14548))
- Deprecated all functions in `pytorch_lightning.utilities.xla_device` ([#14514](https://github.com/Lightning-AI/lightning/pull/14514), [#14550](https://github.com/Lightning-AI/lightning/pull/14550))
  * Deprecated the internal `inner_f` function
  * Deprecated the internal `pl_multi_process` function
  * Deprecated the internal `XLADeviceUtils.xla_available` staticmethod
  * Deprecated the `XLADeviceUtils.tpu_device_exists` staticmethod in favor of `pytorch_lightning.accelerators.TPUAccelerator.is_available()`
- Deprecated `pytorch_lightning.utilities.distributed.tpu_distributed` in favor of `lightning_lite.accelerators.tpu.tpu_distributed` ([#14550](https://github.com/Lightning-AI/lightning/pull/14550))
- Deprecated all functions in `pytorch_lightning.utilities.cloud_io` in favor of `lightning_lite.utilities.cloud_io` ([#14515](https://github.com/Lightning-AI/lightning/pull/14515))
- Deprecated the functions in `pytorch_lightning.utilities.apply_func` in favor of `lightning_utilities.core.apply_func` ([#14516](https://github.com/Lightning-AI/lightning/pull/14516), [#14537](https://github.com/Lightning-AI/lightning/pull/14537))
- Deprecated all functions in `pytorch_lightning.utilities.device_parser` ([#14492](https://github.com/Lightning-AI/lightning/pull/14492), [#14753](https://github.com/Lightning-AI/lightning/pull/14753))
  * Deprecated the `pytorch_lightning.utilities.device_parser.determine_root_gpu_device` in favor of `lightning_lite.utilities.device_parser.determine_root_gpu_device`
  * Deprecated the `pytorch_lightning.utilities.device_parser.parse_gpu_ids` in favor of `lightning_lite.utilities.device_parser.parse_gpu_ids`
  * Deprecated the `pytorch_lightning.utilities.device_parser.is_cuda_available` in favor of `lightning_lite.accelerators.cuda.is_cuda_available`
  * Deprecated the `pytorch_lightning.utilities.device_parser.num_cuda_devices` in favor of `lightning_lite.accelerators.cuda.num_cuda_devices`
  * Deprecated the `pytorch_lightning.utilities.device_parser.parse_cpu_cores` in favor of `lightning_lite.accelerators.cpu.parse_cpu_cores`
  * Deprecated the `pytorch_lightning.utilities.device_parser.parse_tpu_cores` in favor of `lightning_lite.accelerators.tpu.parse_tpu_cores`
  * Deprecated the `pytorch_lightning.utilities.device_parser.parse_hpus` in favor of `pytorch_lightning.accelerators.hpu.parse_hpus`
- Deprecated duplicate `SaveConfigCallback` parameters in `LightningCLI.__init__`: `save_config_kwargs`, `save_config_overwrite` and `save_config_multifile`. New `save_config_kwargs` parameter should be used instead ([#14998](https://github.com/Lightning-AI/lightning/pull/14998))
- Deprecated `TrainerFn.TUNING`, `RunningStage.TUNING` and `trainer.tuning` property ([#15100](https://github.com/Lightning-AI/lightning/pull/15100))
- Deprecated custom `pl.utilities.distributed.AllGatherGrad` implementation in favor of PyTorch's ([#15364](https://github.com/Lightnign-AI/lightning/pull/15364))

    
</details>

<details><summary>Removed</summary>

- Removed the deprecated `Trainer.training_type_plugin` property in favor of `Trainer.strategy` ([#14011](https://github.com/Lightning-AI/lightning/pull/14011))
- Removed all deprecated training type plugins ([#14011](https://github.com/Lightning-AI/lightning/pull/14011))
- Removed the deprecated `DDP2Strategy` ([#14026](https://github.com/Lightning-AI/lightning/pull/14026))
- Removed the deprecated `DistributedType` and `DeviceType` enum classes ([#14045](https://github.com/Lightning-AI/lightning/pull/14045))
- Removed deprecated support for passing the `rank_zero_warn` warning category positionally ([#14470](https://github.com/Lightning-AI/lightning/pull/14470))
- Removed the legacy and unused `Trainer.get_deprecated_arg_names()` ([#14415](https://github.com/Lightning-AI/lightning/pull/14415))
- Removed the deprecated `on_train_batch_end(outputs)` format when multiple optimizers are used and TBPTT is enabled ([#14373](https://github.com/Lightning-AI/lightning/pull/14373))
- Removed the deprecated `training_epoch_end(outputs)` format when multiple optimizers are used and TBPTT is enabled ([#14373](https://github.com/Lightning-AI/lightning/pull/14373))
- Removed the experimental `pytorch_lightning.utiltiies.meta` functions in favor of built-in https://github.com/pytorch/torchdistx support ([#13868](https://github.com/Lightning-AI/lightning/pull/13868))
- Removed the deprecated `LoggerCollection`; `Trainer.logger` and `LightningModule.logger` now returns the first logger when more than one gets passed to the Trainer ([#14283](https://github.com/Lightning-AI/lightning/pull/14283))
- Removed the deprecated the `trainer.lr_schedulers` ([#14408](https://github.com/Lightning-AI/lightning/pull/14408))
- Removed the deprecated `LightningModule.{on_hpc_load,on_hpc_save}` hooks in favor of the general purpose hooks `LightningModule.{on_load_checkpoint,on_save_checkpoint}` ([#14315](https://github.com/Lightning-AI/lightning/pull/14315))
- Removed deprecated support for old torchtext versions ([#14375](https://github.com/Lightning-AI/lightning/pull/14375))
- Removed deprecated support for the old `neptune-client` API in the `NeptuneLogger` ([#14727](https://github.com/Lightning-AI/lightning/pull/14727))
- Removed the deprecated `weights_save_path` Trainer argumnent and `Trainer.weights_save_path` property ([#14424](https://github.com/Lightning-AI/lightning/pull/14424))
- Removed the deprecated ([#14471](https://github.com/Lightning-AI/lightning/pull/14471))
  * `pytorch_lightning.utilities.distributed.rank_zero_only` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_only`
  * `pytorch_lightning.utilities.distributed.rank_zero_debug` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_debug`
  * `pytorch_lightning.utilities.distributed.rank_zero_info` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_info`
  * `pytorch_lightning.utilities.warnings.rank_zero_warn` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_warn`
  * `pytorch_lightning.utilities.warnings.rank_zero_deprecation` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_deprecation`
  * `pytorch_lightning.utilities.warnings.LightningDeprecationWarning` in favor of `pytorch_lightning.utilities.rank_zero.LightningDeprecationWarning`
- Removed deprecated `Trainer.num_processes` attribute in favour of `Trainer.num_devices` ([#14423](https://github.com/Lightning-AI/lightning/pull/14423))
- Removed the deprecated `Trainer.data_parallel_device_ids` hook in favour of `Trainer.device_ids` ([#14422](https://github.com/Lightning-AI/lightning/pull/14422))
- Removed the deprecated class `TrainerCallbackHookMixin` ([#14401](https://github.com/Lightning-AI/lightning/14401))
- Removed the deprecated `BaseProfiler` and `AbstractProfiler` classes ([#14404](https://github.com/Lightning-AI/lightning/pull/14404))
- Removed the deprecated way to set the distributed backend via the environment variable `PL_TORCH_DISTRIBUTED_BACKEND`, in favor of setting the `process_group_backend` in the strategy constructor ([#14693](https://github.com/Lightning-AI/lightning/pull/14693))
- Removed deprecated callback hooks ([#14834](https://github.com/Lightning-AI/lightning/pull/14834))
  * `Callback.on_configure_sharded_model` in favor of `Callback.setup`
  * `Callback.on_before_accelerator_backend_setup` in favor of `Callback.setup`
  * `Callback.on_batch_start` in favor of `Callback.on_train_batch_start`
  * `Callback.on_batch_end` in favor of `Callback.on_train_batch_end`
  * `Callback.on_epoch_start` in favor of `Callback.on_{train,validation,test}_epoch_start`
  * `Callback.on_epoch_end` in favor of `Callback.on_{train,validation,test}_epoch_end`
  * `Callback.on_pretrain_routine_{start,end}` in favor of `Callback.on_fit_start`
- Removed the deprecated device attributes `Trainer.{devices,gpus,num_gpus,ipus,tpu_cores}` in favor of the accelerator-agnostic `Trainer.num_devices` ([#14829](https://github.com/Lightning-AI/lightning/pull/14829))
- Removed the deprecated `LightningIPUModule` ([#14830](https://github.com/Lightning-AI/lightning/pull/14830))
- Removed the deprecated `Logger.agg_and_log_metrics` hook in favour of `Logger.log_metrics` and the `agg_key_funcs` and `agg_default_func` arguments. ([#14840](https://github.com/Lightning-AI/lightning/pull/14840))
- Removed the deprecated precision plugin checkpoint hooks `PrecisionPlugin.on_load_checkpoint` and `PrecisionPlugin.on_save_checkpoint` ([#14833](https://github.com/Lightning-AI/lightning/pull/14833))
- Removed the deprecated `Trainer.root_gpu` attribute in favor of `Trainer.strategy.root_device` ([#14829](https://github.com/Lightning-AI/lightning/pull/14829))
- Removed the deprecated `Trainer.use_amp` and `LightningModule.use_amp` attributes ([#14832](https://github.com/Lightning-AI/lightning/pull/14832))
- Removed the deprecated callback hooks `Callback.on_init_start` and `Callback.on_init_end` ([#14867](https://github.com/Lightning-AI/lightning/pull/14867))
- Removed the deprecated `Trainer.run_stage` in favor of `Trainer.{fit,validate,test,predict}` ([#14870](https://github.com/Lightning-AI/lightning/pull/14870))
- Removed the deprecated `SimpleProfiler.profile_iterable` and `AdvancedProfiler.profile_iterable` attributes ([#14864](https://github.com/Lightning-AI/lightning/pull/14864))
- Removed the deprecated `Trainer.verbose_evaluate` ([#14884](https://github.com/Lightning-AI/lightning/pull/14884))
- Removed the deprecated `Trainer.should_rank_save_checkpoint` ([#14885](https://github.com/Lightning-AI/lightning/pull/14885))
- Removed the deprecated `TrainerOptimizersMixin` ([#14887](https://github.com/Lightning-AI/lightning/pull/14887))
- Removed the deprecated `Trainer.lightning_optimizers` ([#14889](https://github.com/Lightning-AI/lightning/pull/14889))
- Removed the deprecated `TrainerDataLoadingMixin` ([#14888](https://github.com/Lightning-AI/lightning/pull/14888))
- Removed the deprecated `Trainer.call_hook` in favor of `Trainer._call_callback_hooks`, `Trainer._call_lightning_module_hook`, `Trainer._call_ttp_hook`, and `Trainer._call_accelerator_hook` ([#14869](https://github.com/Lightning-AI/lightning/pull/14869))
- Removed the deprecated `Trainer.{validated,tested,predicted}_ckpt_path` ([#14897](https://github.com/Lightning-AI/lightning/pull/14897))
- Removed the deprecated `device_stats_monitor_prefix_metric_keys` ([#14890](https://github.com/Lightning-AI/lightning/pull/14890))
- Removed the deprecated `LightningDataModule.on_save/load_checkpoint` hooks ([#14909](https://github.com/Lightning-AI/lightning/pull/14909))
- Removed support for returning a value in `Callback.on_save_checkpoint` in favor of implementing `Callback.state_dict` ([#14835](https://github.com/Lightning-AI/lightning/pull/14835))

    
</details>

<details><summary>Fixed</summary>


- Fixed an issue with `LightningLite.setup()` not setting the `.device` attribute correctly on the returned wrapper ([#14822](https://github.com/Lightning-AI/lightning/pull/14822))
- Fixed an attribute error when running the tuner together with the `StochasticWeightAveraging` callback ([#14836](https://github.com/Lightning-AI/lightning/pull/14836))
- Fixed MissingFieldException in offline mode for the `NeptuneLogger()` ([#14919](https://github.com/Lightning-AI/lightning/pull/14919))
- Fixed wandb `save_dir` is overridden by `None` `dir` when using CLI ([#14878](https://github.com/Lightning-AI/lightning/pull/14878))
- Fixed a missing call to `LightningDataModule.load_state_dict` hook while restoring checkpoint using `LightningDataModule.load_from_checkpoint` ([#14883](https://github.com/Lightning-AI/lightning/pull/14883))
- Fixed torchscript error with containers of LightningModules ([#14904](https://github.com/Lightning-AI/lightning/pull/14904))
- Fixed reloading of the last checkpoint on run restart ([#14907](https://github.com/Lightning-AI/lightning/pull/14907))
- `SaveConfigCallback` instances should only save the config once to allow having the `overwrite=False` safeguard when using `LightningCLI(..., run=False)` ([#14927](https://github.com/Lightning-AI/lightning/pull/14927))
- Fixed an issue with terminating the trainer profiler when a `StopIteration` exception is raised while using an `IterableDataset` ([#14940](https://github.com/Lightning-AI/lightning/pull/14945))
- Do not update on-plateau schedulers when reloading from an end-of-epoch checkpoint ([#14702](https://github.com/Lightning-AI/lightning/pull/14702))
- Fixed `Trainer` support for PyTorch built without distributed support ([#14971](https://github.com/Lightning-AI/lightning/pull/14971))
- Fixed batch normalization statistics calculation in `StochasticWeightAveraging` callback ([#14866](https://github.com/Lightning-AI/lightning/pull/14866))
- Avoided initializing optimizers during deepspeed inference ([#14944](https://github.com/Lightning-AI/lightning/pull/14944))
- Fixed `LightningCLI` parse_env and description in subcommands ([#15138](https://github.com/Lightning-AI/lightning/pull/15138))
- Fixed an exception that would occur when creating a `multiprocessing.Pool` after importing Lightning ([#15292](https://github.com/Lightning-AI/lightning/pull/15292))
- Fixed a pickling error when using `RichProgressBar` together with checkpointing ([#15319](https://github.com/Lightning-AI/lightning/pull/15319))
- Fixed the `RichProgressBar` crashing when used with distributed strategies ([#15376](https://github.com/Lightning-AI/lightning/pull/15376))
- Fixed an issue with `RichProgressBar` not resetting the internal state for the sanity check progress ([#15377](https://github.com/Lightning-AI/lightning/pull/15377))
- Fixed an issue with DataLoader re-instantiation when the attribute is an array and the default value of the corresponding argument changed ([#15409](https://github.com/Lightning-AI/lightning/pull/15409))


</details>

**Full commit list**: https://github.com/PyTorchLightning/pytorch-lightning/compare/1.7.0...1.8.0

<a name="contributors"></a>
## Contributors

### Veteran

@akihironitta @ananthsub @AndresAlgaba @ar90n @Atharva-Phatak @awaelchli @BongYang @Borda @carmocca @dependabot @donlapark @ethanwharris @Felonious-Spellfire @hhsecond @jerome-habana @JustinGoheen @justusschock @kaushikb11 @krishnakalyan3 @krshrimali @luca-medeiros @manangoel99 @manskx @mauvilsa @MrShevan @nicolai86 @nmiculinic @otaj @Queuecumber @rlizzo @rohitgr7 @rschireman @SeanNaren @speediedan @tchaton @tshu-w


### New

@Birch-san @clementpoiret @HalestormAI @thongonary @alecmerdler @adam-lightning @yurijmikhalevich @lijm1358 @robert-s-lee @panos-is @kacperlukawski @alro923 @dmitsf @Anner-deJong @cschell @nishantb06 @Callidior @j0rd1smit @MarcSkovMadsen @KralaBenjamin @robertomest @daniel347x @pierocor @datumbox @nohalon @pritamsoni-hsr @nandwalritik @gilfree @ritsuki1227 @christopher-nguyen-re @JulesGM @jgbos @dconathan @jsr-p @NeoKish @Blaizzy @suyash-811 @alexkuzmik @ziyadsheeba @geoffrey-g-delhomme @amrutha1098 @AlessioQuercia @ver217 @Helias @zxvix @1SAA @fabiofumarola @luca3rd @kimpty @PaulLerner @rbracco @wouterzwerink


_If we forgot somebody or you have a suggestion, find [support here](https://www.pytorchlightning.ai/support) :zap:_

#### Did you know?

Chuck Norris can write functions of infinite recursion ... and have them return.


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.8.0)

---

## App/0.7.0: Apps's secrets & meta tags
**Published:** 2022-10-20

## [0.7.0] - 2022-10-20

### Added

- Add `--secret` option to CLI to allow binding Secrets to app environment variables when running in the cloud ([#14612](https://github.com/Lightning-AI/lightning/pull/14612))
- Added support for adding descriptions to commands either through a docstring or the `DESCRIPTION` attribute ([#15193](https://github.com/Lightning-AI/lightning/pull/15193)
- Added option to add custom meta tags to the UI container ([#14915](https://github.com/Lightning-AI/lightning/pull/14915))
- Added support to pass a `LightningWork` to the `LightningApp` ([#15215](https://github.com/Lightning-AI/lightning/pull/15215)

### Changed

- Allowed root path to run the app on `/path` ([#14972](https://github.com/Lightning-AI/lightning/pull/14972))

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/App/0.7.0)

---

## app/0.6.3: App with meta tags
**Published:** 2022-10-07

## [0.6.3] - 2022-10-07

### Added

- Added option to add custom meta tags to the UI container (#14915)

### Changed

- Allowed root path to run the app on `/path` (#14972)


### Contributors

@pritamsoni-hsr

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.6.3)

---

## 1.7.7: PyTorch Lightning 1.7.7: Standard patch release
**Published:** 2022-09-22

## [1.7.7] - 2022-09-22

### Fixed

- Fixed the availability check for the neptune-client package ([#14714](https://github.com/Lightning-AI/lightning/pull/14714))
- Break HPU Graphs into two parts (forward + backward as one and optimizer as another) for better performance ([#14656](https://github.com/Lightning-AI/lightning/pull/14656))
- Fixed torchscript error with ensembles of LightningModules ([#14657](https://github.com/Lightning-AI/lightning/pull/14657), [#14724](https://github.com/Lightning-AI/lightning/pull/14724))
- Fixed an issue with `TensorBoardLogger.finalize` creating a new experiment when none was created during the Trainer's execution ([#14762](https://github.com/Lightning-AI/lightning/pull/14762))
- Fixed `TypeError` on import when `torch.distributed` is not available ([#14809](https://github.com/Lightning-AI/lightning/pull/14809))

### Contributors

@awaelchli @Borda @carmocca @dependabot @otaj @raoakarsha

_If we forgot someone due to not matching commit email with GitHub account, let us know_ :)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.7.7)

---

## app/0.6.2: Minor patch release
**Published:** 2022-09-22

## [0.6.2] - 2022-09-22

### Changed

- Improved Lightning App connect logic by disconnecting automatically (#14532)
- Improved the error message when the `LightningWork` is missing the `run` method (#14759)
- Improved the error message when the root `LightningFlow` passed to `LightningApp` is missing the `run` method (#14760)

### Fixed

- Fixed a bug where the uploaded command file wasn't properly parsed (#14532)
- Fixed an issue where custom property setters were not being used `LightningWork` class (#14259)
- Fixed an issue where some terminals would display broken icons in the PL app CLI (#14226)


### Contributors

@awaelchli, @borda, @pranjaldatta, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.6.2)

---

## app/0.6.1: Memory fixes inbound!
**Published:** 2022-09-19

## [0.6.1] - 2022-09-19

### Added

- Add support to upload files to the Drive through an asynchronous `upload_file` endpoint (#14703)

### Changed

- Application storage prefix moved from `app_id` to `project_id/app_id` (#14583)
- LightningCloud client calls to use keyword arguments instead of positional arguments (#14685)

### Fixed

- Making `threadpool` non-default from LightningCloud client  (#14757)
- Resolved a bug where the state change detection using DeepDiff won't work with Path, Drive objects (#14465)
- Resolved a bug where the wrong client was passed to collect cloud logs (#14684)
- Resolved the memory leak issue with the Lightning Cloud package and bumped the requirements to use the latest version (#14697)
- Fixing 5000 log line limitation for Lightning AI BYOC cluster logs (#14458)
- Fixed a bug where the uploaded command file wasn't properly parsed (#14532)
- Resolved `LightningApp(..., debug=True)` (#14464)


## Contributors

@dmitsf @hhsecond @tchaton @nohalon @krshrimali @pritamsoni-hsr @nmiculinic @ethanwharris @yurijmikhalevich @Felonious-Spellfire @otaj @Borda 

_If we forgot someone due to not matching commit email with GitHub account, let us know_ :)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.6.1)

---

## 1.7.6: PyTorch Lightning 1.7.6: Standard patch release
**Published:** 2022-09-13

## [1.7.6] - 2022-09-13

### Changed

- Improved the error messaging when passing `Trainer.method(model, x_dataloader=None)` with no module-method implementations available ([#14614](https://github.com/Lightning-AI/lightning/pull/14614))

### Fixed

- Reset the dataloaders on OOM failure in batch size finder to use the last successful batch size ([#14372](https://github.com/Lightning-AI/lightning/pull/14372))
- Fixed an issue to keep downscaling the batch size in case there hasn't been even a single successful optimal batch size with `mode="power"` ([#14372](https://github.com/Lightning-AI/lightning/pull/14372))
- Fixed an issue where `self.log`-ing a tensor would create a user warning from PyTorch about cloning tensors ([#14599](https://github.com/Lightning-AI/lightning/pull/14599))
- Fixed compatibility when `torch.distributed` is not available ([#14454](https://github.com/Lightning-AI/lightning/pull/14454))

### Contributors

@akihironitta @awaelchli @Borda @carmocca @dependabot @krshrimali @mauvilsa @pierocor @rohitgr7 @wangraying

_If we forgot someone due to not matching commit email with GitHub account, let us know_ :)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.7.6)

---

## app/0.6.0: BYOC cluster management
**Published:** 2022-09-08

## [0.6.0] - 2022-09-08

### Added

- Introduce lightning connect (#14452)
- Adds `PanelFrontend` to easily create complex UI in Python (#13531)
- Add support for Lightning App Commands through the `configure_commands` hook on `LightningFlow` and `ClientCommand`  (#13602)
- Add support for Lightning AI BYOC cluster management (#13835)
- Add support to see Lightning AI BYOC cluster logs (#14334)
- Add support to run Lightning apps on Lightning AI BYOC clusters (#13894)
- Add support for listing Lightning AI apps (#13987)
- Adds `LightningTrainingComponent`. `LightningTrainingComponent` orchestrates multi-node training in the cloud (#13830)
- Add support for printing application logs using CLI `lightning show logs <app_name> [components]` (#13634)
- Add support for Lightning API through the `configure_api` hook on the `LightningFlow` and the `Post`, `Get`, `Delete`, `Put` with HttpMethods (#13945)
- Added a warning when `configure_layout` returns URLs configured with HTTP instead of HTTPS (#14233)
- Add `--app_args` support from the CLI (#13625)

### Changed

- Default values and parameter names for Lightning AI BYOC cluster management (#14132)
- Run the flow only if the state has changed from the previous execution (#14076)
- Increased DeepDiff's verbose level to properly handle dict changes (#13960)
- Setup: added requirement freeze for the next major version (#14480)

### Fixed

- Unification of app template: moved `app.py` to root dir for `lightning init app <app_name>` template (#13853)
- Fixed an issue with `lightning --version` command (#14433)
- Fixed imports of collections.abc for py3.10 (#14345)

### Contributors

@adam-lightning, @awaelchli, @Borda, @dmitsf, @manskx, @MarcSkovMadsen, @nicolai86, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.6.0)

---

## 1.7.5: PyTorch Lightning 1.7.5: Standard patch release
**Published:** 2022-09-07

## [1.7.5] - 2022-09-06

### Fixed

- Squeezed tensor values when logging with `LightningModule.log` ([#14489](https://github.com/Lightning-AI/lightning/pull/14489))
- Fixed `WandbLogger` `save_dir` is not set after creation ([#14326](https://github.com/Lightning-AI/lightning/pull/14326))
- Fixed `Trainer.estimated_stepping_batches` when maximum number of epochs is not set ([#14317](https://github.com/Lightning-AI/lightning/pull/14317))

### Contributors

@carmocca @dependabot @robertomest @rohitgr7 @tshu-w

_If we forgot someone due to not matching commit email with GitHub account, let us know_ :)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.7.5)

---

## 1.7.4: PyTorch Lightning 1.7.4: Standard patch release
**Published:** 2022-08-31

## [1.7.4] - 2022-08-31

### Added

- Added an environment variable `PL_DISABLE_FORK` that can be used to disable all forking in the Trainer ([#14319](https://github.com/Lightning-AI/lightning/issues/14319))

### Fixed

- Fixed `LightningDataModule` hparams parsing ([#12806](https://github.com/PyTorchLightning/pytorch-lightning/pull/12806))
- Reset epoch progress with batch size scaler ([#13846](https://github.com/Lightning-AI/lightning/pull/13846))
- Fixed restoring the trainer after using `lr_find()` so that the correct LR schedule is used for the actual training ([#14113](https://github.com/Lightning-AI/lightning/pull/14113))
- Fixed incorrect values after transferring data to an MPS device ([#14368](https://github.com/Lightning-AI/lightning/pull/14368))

### Contributors

@rohitgr7 @tanmoyio @justusschock @cschell @carmocca @Callidior @awaelchli @j0rd1smit @dependabot @Borda @otaj

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.7.4)

---

## 1.7.3: PyTorch Lightning 1.7.3: Standard patch release
**Published:** 2022-08-25

## [1.7.3] - 2022-08-25

### Fixed

- Fixed an assertion error when using a `ReduceOnPlateau` scheduler with the Horovod strategy ([#14215](https://github.com/Lightning-AI/lightning/pull/14215))
- Fixed an `AttributeError` when accessing `LightningModule.logger` and the Trainer has multiple loggers ([#14234](https://github.com/Lightning-AI/lightning/pull/14234))
- Fixed wrong num padding for `RichProgressBar` ([#14296](https://github.com/Lightning-AI/lightning/pull/14296))
- Added back support for `log`ging in the `configure_gradient_clipping` hook after unintended removal in v1.7.2 ([#14298](https://github.com/Lightning-AI/lightning/issues/14298))
- Fixed an issue to avoid the impact of sanity check on `reload_dataloaders_every_n_epochs` for validation ([#13964](https://github.com/Lightning-AI/lightning/pull/13964))


### Contributors

@awaelchli @Borda @carmocca @dependabot @kaushikb11 @otaj @rohitgr7


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.7.3)

---

## app/0.5.7: Dependency hotfix
**Published:** 2022-08-22

## [0.5.7] - 2022-08-22

### Changed

- Release LAI docs as stable (#14250)
- Compatibility for Python 3.10

### Fixed

- Pinning `starsessions` to 1.x (#14333)
- Parsed local package versions (#13933)

### Contributors

@borda, @hhsecond, @manskx 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.5.7)

---

## app/0.5.6: Minor patch release
**Published:** 2022-08-18

## [0.5.6] - 2022-08-16

### Fixed

- Resolved a bug where the `install` command was not installing the latest version of an app/component by default (#14181)


### Contributors

@manskx

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.5.6)

---

## 1.7.2: PyTorch Lightning 1.7.2: Standard patch release
**Published:** 2022-08-17

## [1.7.2] - 2022-08-17

### Added

- Added `FullyShardedNativeNativeMixedPrecisionPlugin` to handle precision for `DDPFullyShardedNativeStrategy` ([#14092](https://github.com/Lightning-AI/lightning/pull/14092))
- Added profiling to these hooks: `on_before_batch_transfer`, `transfer_batch_to_device`, `on_after_batch_transfer`, `configure_gradient_clipping`, `clip_gradients` ([#14069](https://github.com/Lightning-AI/lightning/pull/14069))

### Changed

- Updated compatibility for LightningLite to run with the latest DeepSpeed 0.7.0 ([13967](https://github.com/Lightning-AI/lightning/pull/13967))
- Raised a `MisconfigurationException` if batch transfer hooks are overriden with `IPUAccelerator` ([13961](https://github.com/Lightning-AI/lightning/pull/13961))
- The default project name in `WandbLogger` is now "lightning_logs" ([#14145](https://github.com/Lightning-AI/lightning/pull/14145))
- The `WandbLogger.name` property no longer returns the name of the experiment, and instead returns the project's name ([#14145](https://github.com/Lightning-AI/lightning/pull/14145))

### Fixed

- Fixed a bug that caused spurious `AttributeError` when multiple `DataLoader` classes are imported ([#14117](https://github.com/Lightning-AI/lightning/pull/14117))
- Fixed epoch-end logging results not being reset after the end of the epoch ([#14061](https://github.com/Lightning-AI/lightning/pull/14061))
- Fixed saving hyperparameters in a composition where the parent class is not a `LightningModule` or `LightningDataModule` ([#14151](https://github.com/Lightning-AI/lightning/pull/14151))
- Fixed epoch-end logging results not being reset after the end of the epoch ([#14061](https://github.com/Lightning-AI/lightning/pull/14061))
- Fixed the device placement when `LightningModule.cuda()` gets called without specifying a device index and the current cuda device was not 0 ([#14128](https://github.com/Lightning-AI/lightning/pull/14128))
- Avoided false positive warning about using `sync_dist` when using torchmetrics ([#14143](https://github.com/Lightning-AI/lightning/pull/14143))
- Avoid `metadata.entry_points` deprecation warning on Python 3.10 ([#14052](https://github.com/Lightning-AI/lightning/pull/14052))
- Avoid raising the sampler warning if num_replicas=1 ([#14097](https://github.com/Lightning-AI/lightning/pull/14097))
- Fixed resuming from a checkpoint when using Stochastic Weight Averaging (SWA) ([#9938](https://github.com/Lightning-AI/lightning/pull/9938))
- Avoided requiring the FairScale package to use precision with the fsdp native strategy ([#14092](https://github.com/Lightning-AI/lightning/pull/14092))
- Fixed an issue in which the default name for a run in `WandbLogger` would be set to the project name instead of a randomly generated string ([#14145](https://github.com/Lightning-AI/lightning/pull/14145))
- Fixed not preserving set attributes on `DataLoader` and `BatchSampler` when instantiated inside `*_dataloader` hooks ([#14212](https://github.com/Lightning-AI/lightning/pull/14212))


### Contributors

@adamreeve @akihironitta @awaelchli @Borda @carmocca @dependabot @otaj @rohitgr7


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.7.2)

---

## 1.7.1: PyTorch Lightning 1.7.1: Standard patch release
**Published:** 2022-08-09

## [1.7.1] - 2022-08-09

### Fixed

- Casted only floating point tensors to fp16 with IPUs ([#13983](https://github.com/Lightning-AI/lightning/pull/13983))
- Casted tensors to fp16 before moving them to device with  `DeepSpeedStrategy` ([#14000](https://github.com/Lightning-AI/lightning/pull/14000))
- Fixed the `NeptuneLogger` dependency being unrecognized ([#13988](https://github.com/Lightning-AI/lightning/pull/13988))
- Fixed an issue where users would be warned about unset `max_epochs` even when `fast_dev_run` was set ([#13262](https://github.com/Lightning-AI/lightning/pull/13262))
- Fixed MPS device being unrecognized ([#13992](https://github.com/Lightning-AI/lightning/pull/13992))
- Fixed incorrect `precision="mixed"` being used with `DeepSpeedStrategy` and `IPUStrategy` ([#14041](https://github.com/Lightning-AI/lightning/pull/14041))
- Fixed dtype inference during gradient norm computation ([#14051](https://github.com/Lightning-AI/lightning/pull/14051))
- Fixed a bug that caused `ddp_find_unused_parameters` to be set `False`, whereas the intended default is `True` ([#14095](https://github.com/Lightning-AI/lightning/pull/14095))


### Contributors

@adamjstewart @akihironitta @awaelchli @Birch-san @carmocca @clementpoiret @dependabot @rohitgr7

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.7.1)

---

## app/0.5.5: Week bugfix release
**Published:** 2022-08-09

## [0.5.5] - 2022-08-9

### Deprecated

- Deprecate sheety API (#14004)

### Fixed

- Resolved a bug where the work statuses will grow quickly and be duplicated (#13970)
- Resolved a bug about a race condition when sending the work state through the caller_queue (#14074)
- Fixed Start Lightning App on Cloud if Repo Begins With Name "Lightning" (#14025)


### Contributors

@manskx, @rlizzo, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.5.5)

---

## 1.7.0: PyTorch Lightning 1.7: Apple Silicon support, Native FSDP, Collaborative training, and multi-GPU support with Jupyter notebooks
**Published:** 2022-08-02

The core team is excited to announce the release of PyTorch Lightning 1.7 :zap:

- [Highlights](#highlights)
- [Backward Incompatible Changes](#bc-changes)
- [Deprecations](#deprecations)
- [Full Changelog](#changelog)
- [Contributors](#contributors)

PyTorch Lightning 1.7 is the culmination of work from 106 contributors who have worked on features, bug-fixes, and documentation for a total of over 492 commits since 1.6.0.

<a name="highlights"></a>
# Highlights

### Apple Silicon Support

For those using [PyTorch 1.12](https://pytorch.org/blog/pytorch-1.12-released/#prototype-introducing-accelerated-pytorch-training-on-mac) on M1 or M2 Apple machines, we have created the `MPSAccelerator`. `MPSAccelerator` enables accelerated GPU training on Apple’s Metal Performance Shaders (MPS) as a backend process.

---
**NOTE**

Support for this accelerator is currently marked as **experimental** in PyTorch. Because many operators are still missing, you may run into a few rough edges.

---

```python
# Selects the accelerator
trainer = pl.Trainer(accelerator="mps")

# Equivalent to
from pytorch_lightning.accelerators import MPSAccelerator
trainer = pl.Trainer(accelerator=MPSAccelerator())

# Defaults to "mps" when run on M1 or M2 Apple machines
# to avoid code changes when switching computers
trainer = pl.Trainer(accelerator="gpu")
```

### Native Fully Sharded Data Parallel Strategy

PyTorch 1.12 also added native support for [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api). Previously, PyTorch Lightning enabled this by using the [`fairscale`](https://engineering.fb.com/2021/07/15/open-source/fsdp/) project. You can now choose between both options.

---
**NOTE**

Support for this strategy is marked as **beta** in PyTorch.

---

```python
# Native PyTorch implementation
trainer = pl.Trainer(strategy="fsdp_native")

# Equivalent to
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
trainer = pl.Trainer(strategy=DDPFullyShardedNativeStrategy())

# For reference, FairScale's implementation can be used with
trainer = pl.Trainer(strategy="fsdp")
```

### A Collaborative Training strategy using Hivemind

Collaborative Training solves the need for top-tier multi-GPU servers by allowing you to train across unreliable machines such as local ones or even preemptible cloud compute across the Internet.

Under the hood, we use [Hivemind](https://github.com/learning-at-home/hivemind). This provides de-centralized training across the Internet.

```python
from pytorch_lightning.strategies import HivemindStrategy

trainer = pl.Trainer(
    strategy=HivemindStrategy(target_batch_size=8192), 
    accelerator="gpu", 
    devices=1
)
```

For more information, check out the [docs](https://pytorch-lightning.readthedocs.io/en/1.7.0/strategies/hivemind.html).

### Distributed support in Jupyter Notebooks

So far, the only multi-GPU strategy supported in [Jupyter notebooks](https://jupyter.org/) (including [Grid.ai](https://www.grid.ai/), [Google Colab](https://research.google.com/colaboratory/), and [Kaggle](https://www.kaggle.com/), for example) has been the [Data-Parallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) (DP) strategy (`strategy="dp"`). DP, however, has several limitations that often obstruct users' workflows. It can be slow, it's incompatible with [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/), it doesn't persist state changes on replicas, and it's difficult to use with non-primitive input- and output structures.

In this release, we've added support for Distributed Data Parallel in Jupyter notebooks using the fork mechanism to address these shortcomings. This is only available for MacOS and Linux (sorry Windows!).

---
**NOTE**

This feature is **experimental**.

---

This is how you use multi-device in notebooks now:

```python
# Train on 2 GPUs in a Jupyter notebook
trainer = pl.Trainer(accelerator="gpu", devices=2)

# Can be set explicitly
trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp_notebook")

# Can also be used in non-interactive environments
trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp_fork")
```

By default, the Trainer detects the interactive environment and selects the right strategy for you. Learn more in the [full documentation](https://pytorch-lightning.readthedocs.io/en/1.7.0/accelerators/gpu_intermediate.html?highlight=fork#distributed-data-parallel-fork). 

### Versioning of "last" checkpoints

If a run is configured to save to the same directory as a previous run and `ModelCheckpoint(save_last=True)` is enabled, the "last" checkpoint is now versioned with a simple `-v1` suffix to avoid overwriting the existing "last" checkpoint. This mimics the behaviour for checkpoints that monitor a metric.

### Automatically reload the "last" checkpoint

In certain scenarios, like when running in a cloud spot instance with fault-tolerant training enabled, it is useful to load the latest available checkpoint. It is now possible to pass the string `ckpt_path="last"` in order to load the latest available checkpoint from the set of existing checkpoints.

```python
trainer = Trainer(...)
trainer.fit(..., ckpt_path="last")
```

### Validation every N batches across epochs

In some cases, for example iteration based training, it is useful to run validation after every `N` number of training batches without being limited by the epoch boundary. Now, you can enable validation based on total training batches.

```python
trainer = Trainer(..., val_check_interval=N, check_val_every_n_epoch=None)
trainer.fit(...)
```

For example, given 5 epochs of 10 batches, setting `N=25` would run validation in the 3rd and 5th epoch. 

### CPU stats monitoring

PyTorch Lightning provides the [`DeviceStatsMonitor`](https://pytorch-lightning.readthedocs.io/en/1.7.0/api/pytorch_lightning.callbacks.DeviceStatsMonitor.html#pytorch_lightning.callbacks.DeviceStatsMonitor) callback to monitor the stats of the hardware currently used. However, users often also want to monitor the stats of other hardware. In this release, we have added an option to additionally monitor CPU stats:

```python
from pytorch_lightning.callbacks import DeviceStatsMonitor

# Log both CPU stats and GPU stats
trainer = pl.Trainer(callbacks=DeviceStatsMonitor(cpu_stats=True), accelerator="gpu")

# Log just the GPU stats
trainer = pl.Trainer(callbacks=DeviceStatsMonitor(cpu_stats=False), accelerator="gpu")

# Equivalent to `DeviceStatsMonitor()`
trainer = pl.Trainer(callbacks=DeviceStatsMonitor(cpu_stats=True), accelerator="cpu")
```

The CPU stats are gathered using the [`psutil`](https://github.com/giampaolo/psutil) package.

### Automatic distributed samplers

It is now possible to use custom samplers in a distributed environment without the need to set `replace_ddp_sampler=False` and wrap your sampler manually with the [`DistributedSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler).

### Inference mode support

PyTorch 1.9 introduced [`torch.inference_mode`](https://pytorch.org/docs/stable/generated/torch.inference_mode.html), which is a faster alternative for [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html). Lightning will now use `inference_mode` wherever possible during evaluation.

### Support for warn-level determinism

In Pytorch 1.11, operations that do not have a [deterministic](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms) implementation can be set to throw a warning instead of an error when ran in deterministic mode. This is now supported by our `Trainer`:

```python
trainer = pl.Trainer(deterministic="warn")
```

### LightningCLI improvements

After the latest updates to [`jsonargparse`](https://github.com/omni-us/jsonargparse), the library supporting the [`LightningCLI`](https://pytorch-lightning.readthedocs.io/en/1.7.0/cli/lightning_cli.html), there's now complete support for shorthand notation. This includes automatic support for shorthand notation to all arguments, not just the ones that are part of the registries, plus support inside configuration files.

```diff
+ # pytorch_lightning==1.7.0
  trainer:
  callbacks:
-   - class_path: pytorch_lightning.callbacks.EarlyStopping
+   - class_path: EarlyStopping
      init_args:
        monitor: "loss"
```

A header with the version that generated the config is now included.

All subclasses for a given base class can be specified by name, so there's no need to explicitly register them. The only requirement is that the module where the subclass is defined is imported prior to parsing. 

```python
from pytorch_lightning.cli import LightningCLI
import my_code.models
import my_code.optimizers

cli = LightningCLI()
# Now use any of the classes:
# python trainer.py fit --model=Model1 --optimizer=CustomOptimizer
```

The new version renders the registries and the `auto_registry` flag, introduced in 1.6.0, unnecessary, so we have deprecated them.

Support was also added for list appending; for example, to add a callback to an existing list that might be already configured:

```diff
$ python trainer.py fit \
-   --trainer.callbacks=EarlyStopping \
+   --trainer.callbacks+=EarlyStopping \
    --trainer.callbacks.patience=5 \
-   --trainer.callbacks=LearningRateMonitor \
+   --trainer.callbacks+=LearningRateMonitor \
    --trainer.callbacks.logging_interval=epoch
```


### Callback registration through entry points

[Entry Points](https://setuptools.pypa.io/en/stable/userguide/entry_point.html) are an advanced feature in Python's setuptools that allow packages to expose metadata to other packages. In Lightning, we allow an arbitrary package to include callbacks that the Lightning Trainer can automatically use when installed, without you having to manually add them to the Trainer. This is useful in production environments where it is common to provide specialized monitoring and logging callbacks globally for every application.

A `setup.py` file for a callbacks plugin package could look something like this:

```python
from setuptools import setup

setup(
    name="my-package",
    version="0.0.1",
    entry_points={
        # Lightning will look for this key here in the environment:
        "pytorch_lightning.callbacks_factory": [
            "monitor_callbacks=factories:my_custom_callbacks_factory"
        ]
    },
)
```

Read more about callback entry points in our [docs](https://pytorch-lightning.readthedocs.io/en/1.7.0/extensions/entry_points.html?highlight=entry%20points).

### Rank-zero only `EarlyStopping` messages

Our [`EarlyStopping`](https://pytorch-lightning.readthedocs.io/en/1.7.0/api/pytorch_lightning.callbacks.EarlyStopping.html) callback implementation, by default, logs the stopping messages on every rank when it's run in a distributed environment. This was done in case the monitored values were not synchronized. However, some users found this verbose. To avoid this, you can now set a flag:

```python
from pytorch_lightning.callbacks import EarlyStopping

trainer = pl.Trainer(callbacks=EarlyStopping(..., log_rank_zero_only=True))
```

### A base `Checkpoint` class for extra customization

If you want to customize [`ModelCheckpoint`](https://pytorch-lightning.readthedocs.io/en/1.7.0/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) callback, without all the extra functionality this class provides, this release provides an empty class `Checkpoint` for easier inheritance. In all internal code, the check is made against the `Checkpoint` class in order to ensure everything works properly for custom classes.

### Validation now runs in overfitting mode

Setting `overfit_batches=N`, now enables validation and runs `N` number of validation batches during `trainer.fit`.

```python
# Uses 1% of each train & val set
trainer = Trainer(overfit_batches=0.01)

# Uses 10 batches for each train & val set
trainer = Trainer(overfit_batches=10)
```

### Device Stats Monitoring support for HPUs

[`DeviceStatsMonitor`](https://pytorch-lightning.readthedocs.io/en/1.7.0/api/pytorch_lightning.callbacks.DeviceStatsMonitor.html#pytorch_lightning.callbacks.DeviceStatsMonitor) callback can now be used to automatically monitor and log device stats during the training stage with Habana devices.

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor

device_stats = DeviceStatsMonitor()
trainer = Trainer(accelerator="hpu", callbacks=[device_stats])
```

## New Hooks

### `LightningDataModule.load_from_checkpoint`

Now, hyper-parameters from `LightningDataModule` save to checkpoints and reload when training is resumed. And just like you use `LightningModule.load_from_checkpoint` to load a model using a checkpoint filepath, you can now load `LightningDataModule` using the same hook.

```python
# Lad weights without mapping ...
datamodule = MyLightningDataModule.load_from_checkpoint('path/to/checkpoint.ckpt')

# Or load weights and hyperparameters from separate files.
datamodule = MyLightningDataModule.load_from_checkpoint(
    'path/to/checkpoint.ckpt',
    hparams_file='/path/to/hparams_file.yaml'
)

# Override some of the params with new values
datamodule = MyLightningDataModule.load_from_checkpoint(
    'path/to/checkpoint.ckpt',
    batch_size=32,
    num_workers=10,
)
```

## Experimental Features

### ServableModule and its Servable Module Validator Callback

When serving models in production, it generally is a good pratice to ensure that the model can be served and optimzed before starting training to avoid wasting money.

To do so, you can import a `ServableModule` (an `nn.Module`) and add it as an extra base class to your base model as follows:

```python
from pytorch_lightning import LightningModule
from pytorch_lightning.serve import ServableModule

class ProductionReadyModel(LightningModule, ServableModule):
    ...
```

To make your model servable, you would need to implement three hooks:
* `configure_payload`: Describe the format of the payload (data sent to the server).
* `configure_serialization`: Describe the functions used to convert the payload to tensors (de-serialization) and tensors to payload (serialization)
* `serve_step`: The method used to transform the input tensors to a dictionary of prediction tensors.

```python
from pytorch_lightning.serve import ServableModule, ServableModuleValidator

class ProductionReadyModel(LitModule, ServableModule):
    def configure_payload(self):
        # 1: Access the train dataloader and load a single sample.
        image, _ = self.trainer.train_dataloader.loaders.dataset[0]

        # 2: Convert the image into a PIL Image to bytes and encode it with base64
        pil_image = T.ToPILImage()(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("UTF-8")

        payload = {"body": {"x": img_str}}
        return payload

    def configure_serialization(self):
        deserializers = {"x": Image(224, 224).deserialize}
        serializers = {"output": Top1().serialize}
        return deserializers, serializers

    def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"output": self.model(x)}
```

Finally, add the `ServableModuleValidator` callback to the Trainer to validate the model is servable `on_train_start`. This uses a [FastAPI](https://fastapi.tiangolo.com/) server.

```python 
pl_module = ProductionReadyModel()
trainer = Trainer(..., callbacks=[ServableModuleValidator()])
trainer.fit(pl_module)
```

Have a look at the full example [here](https://pytorch-lightning.readthedocs.io/en/1.7.0/deploy/production_advanced.html#validate-a-model-is-servable).

### Asynchronous Checkpointing

You can now save checkpoints asynchronously using the [`AsyncCheckpointIO`](https://pytorch-lightning.readthedocs.io/en/1.7.0/api/pytorch_lightning.plugins.io.AsyncCheckpointIO.html) plugin without blocking your training process. To enable this, you can pass a `AsyncCheckpointIO` plugin to the `Trainer`.

```python
from pytorch_lightning.plugins.io import AsyncCheckpointIO

trainer = Trainer(plugins=[AsyncCheckpointIO()])
```

Have a look at the full example [here](https://pytorch-lightning.readthedocs.io/en/1.7.0/common/checkpointing_expert.html#asynchronous-checkpointing).


<a name="bc-changes"></a>
# Backward Incompatible Changes

This section outlines notable changes that are not backward compatible with previous versions. The full list of changes and removals can be found in the CHANGELOG below.

### Removed support for the DDP2 strategy

The DDP2 strategy, previously known as the DDP2 plugin, has been part of Lightning since its inception. Due to both the technical challenges in maintaining the plugin after [PyTorch's removal of the multi-device support in DistributedDataParallel](https://github.com/pytorch/pytorch/pull/54454), as well as a general lack of interest, we have decided to retire the strategy entirely.

### Do not force metric synchronization on epoch end

In previous versions, metrics logged inside epoch-end hooks were forcefully synced. This makes the `sync_dist` flag irrelevant and causes communication overhead that might be undesired. In this release, we've removed this behaviour and instead warn the user that synchronization might be desired.

<a name="deprecations"></a>
# Deprecations

| API                                                                                                          | Removal version | Alternative                                     |
|--------------------------------------------------------------------------------------------------------------|-----------------|-------------------------------------------------|
| Import `pytorch_lightning.loggers.base.LightningLoggerBase`                                                  | 1.9             | `pytorch_lightning.loggers.logger.Logger`       |
| Import `pytorch_lightning.callbacks.base.Callback`                                                           | 1.9             | `pytorch_lightning.callbacks.callback.Callback` |
| Import `pytorch_lightning.core.lightning.LightningModule`                                                    | 1.9             | `pytorch_lightning.core.module.LightningModule` |
| Import `pytorch_lightning.loops.base.Loop`                                                                   | 1.9             | `pytorch_lightning.loops.loop.Loop`             |
| Import `pytorch_lightning.profiler`                                                                          | 1.9             | `pytorch_lightning.profilers`                   |
| Arguments `Trainer(num_processes=..., gpus=..., tpu_cores=..., ipus=...)`                                    | 2.0             | `Trainer(accelerator=..., devices=...)`         |
| Argument `LightningCLI(seed_everything_default=None)`                                                        | 1.9             | `LightningCLI(seed_everything_default=False)`   |
| Method `Trainer.reset_train_val_dataloaders()`                                                               | 1.9             | `Trainer.reset_{train,val}_dataloader`          |
| Import `pytorch_lightning.utilities.cli` module                                                              | 1.9             | `pytorch_lightning.cli`                         |
| Objects `pytorch_lightning.utilities.cli.{OPTIMIZER,LR_SCHEDULER,MODEL,DATAMODULE,CALLBACK,LOGGER}_REGISTRY` | 1.9             | Not necessary anymore                           |
| Argument `LightningCLI(auto_registry=...)`                                                                   | 1.9             | Not necessary anymore                           |
| Argument `Trainer(strategy="ddp2")` and class `pytorch_lightning.strategies.DDP2Strategy`                    | 1.8             | No longer supported                             |

<a name="changelog"></a>
# CHANGELOG

<details><summary>Added</summary>


-  Added ``ServableModule`` and its associated callback called ``ServableModuleValidator`` to ensure the model can served ([#13614](https://github.com/Lightning-AI/lightning/pull/13614))
-  Converted validation loop config warnings to `PossibleUserWarning` ([#13377](https://github.com/Lightning-AI/lightning/pull/13377))
- Added a flag named `log_rank_zero_only` to `EarlyStopping` to disable logging to non-zero rank processes ([#13233](https://github.com/Lightning-AI/lightning/pull/13233))
- Added support for reloading the last checkpoint saved by passing `ckpt_path="last"` ([#12816](https://github.com/Lightning-AI/lightning/pull/12816))
- Added `LightningDataModule.load_from_checkpoint` to support loading datamodules directly from checkpoint ([#12550](https://github.com/Lightning-AI/lightning/pull/12550))
- Added a friendly error message when attempting to call `Trainer.save_checkpoint()` without a model attached ([#12772](https://github.com/Lightning-AI/lightning/pull/12772))
- Added a friendly error message when attempting to use `DeepSpeedStrategy` on unsupported accelerators ([#12699](https://github.com/Lightning-AI/lightning/pull/12699))
- Enabled `torch.inference_mode` for evaluation and prediction ([#12715](https://github.com/Lightning-AI/lightning/pull/12715))
- Added support for setting `val_check_interval` to a value higher than the amount of training batches when `check_val_every_n_epoch=None` ([#11993](https://github.com/Lightning-AI/lightning/pull/11993))
- Include the `pytorch_lightning` version as a header in the CLI config files ([#12532](https://github.com/Lightning-AI/lightning/pull/12532))
- Added support for `Callback` registration through entry points ([#12739](https://github.com/Lightning-AI/lightning/pull/12739))
- Added support for `Trainer(deterministic="warn")` to warn instead of fail when a non-deterministic operation is encountered ([#12588](https://github.com/Lightning-AI/lightning/pull/12588))
- Added profiling to the loops' dataloader `__next__` calls ([#12124](https://github.com/Lightning-AI/lightning/pull/12124))
- Hivemind Strategy
    * Added `CollaborativeStrategy` ([#12842](https://github.com/Lightning-AI/lightning/pull/12842))
    * Renamed `CollaborativeStrategy` to `HivemindStrategy` ([#13388](https://github.com/Lightning-AI/lightning/pull/13388))
    * Removed unnecessary endpoint logic, renamed `collaborative` to `hivemind` ([#13392](https://github.com/Lightning-AI/lightning/pull/13392))
- Include a version suffix for new "last" checkpoints of later runs in the same directory ([#12902](https://github.com/Lightning-AI/lightning/pull/12902))
- Show a better error message when a Metric that does not return a Tensor is logged ([#13164](https://github.com/Lightning-AI/lightning/pull/13164))
- Added missing `predict_dataset` argument in `LightningDataModule.from_datasets` to create predict dataloaders ([#12942](https://github.com/Lightning-AI/lightning/pull/12942))
- Added class name prefix to metrics logged by `DeviceStatsMonitor` ([#12228](https://github.com/Lightning-AI/lightning/pull/12228))
- Automatically wrap custom samplers under a distributed environment by using `DistributedSamplerWrapper` ([#12959](https://github.com/Lightning-AI/lightning/pull/12959))
- Added profiling of `LightningDataModule` hooks ([#12971](https://github.com/Lightning-AI/lightning/pull/12971))
- Added Native FSDP Strategy ([#12447](https://github.com/Lightning-AI/lightning/pull/12447))
- Added breaking of lazy graph across training, validation, test and predict steps when training with habana accelerators to ensure better performance ([#12938](https://github.com/Lightning-AI/lightning/pull/12938))
- Added `Checkpoint` class to inherit from ([#13024](https://github.com/Lightning-AI/lightning/pull/13024))
- Added CPU metric tracking to `DeviceStatsMonitor` ([#11795](https://github.com/Lightning-AI/lightning/pull/11795))
- Added `teardown()` method to `Accelerator` ([#11935](https://github.com/Lightning-AI/lightning/pull/11935))
- Added support for using custom Trainers that don't include callbacks using the CLI ([#13138](https://github.com/Lightning-AI/lightning/pull/13138))
- Added a `timeout` argument to `DDPStrategy` and `DDPSpawnStrategy`. ([#13244](https://github.com/Lightning-AI/lightning/pull/13244), [#13383](https://github.com/Lightning-AI/lightning/pull/13383))
- Added `XLAEnvironment` cluster environment plugin ([#11330](https://github.com/Lightning-AI/lightning/pull/11330))
- Added logging messages to notify when `FitLoop` stopping conditions are met ([#9749](https://github.com/Lightning-AI/lightning/pull/9749))
- Added support for calling unknown methods with `DummyLogger` ([#13224](https://github.com/Lightning-AI/lightning/pull/13224)
- Added support for recursively setting the `Trainer` reference for ensembles of `LightningModule`s ([#13638](https://github.com/Lightning-AI/lightning/pull/13638)
- Added Apple Silicon Support via `MPSAccelerator` ([#13123](https://github.com/Lightning-AI/lightning/pull/13123))
- Added support for DDP Fork ([#13405](https://github.com/Lightning-AI/lightning/pull/13405))
- Added support for async checkpointing ([#13658](https://github.com/Lightning-AI/lightning/pull/13658))
- Added support for HPU Device stats monitor ([#13819](https://github.com/Lightning-AI/lightning/pull/13819))


</details>

<details><summary>Changed</summary>


- `accelerator="gpu"` now automatically selects an available GPU backend (CUDA and MPS currently) ([#13642](https://github.com/Lightning-AI/lightning/pull/13642))
- Enable validation during overfitting ([#12527](https://github.com/Lightning-AI/lightning/pull/12527))
- Added dataclass support to `extract_batch_size` ([#12573](https://github.com/Lightning-AI/lightning/pull/12573))
- Changed checkpoints save path in the case of one logger and user-provided weights_save_path from `weights_save_path/name/version/checkpoints` to `weights_save_path/checkpoints` ([#12372](https://github.com/Lightning-AI/lightning/pull/12372))
- Changed checkpoints save path in the case of multiple loggers and user-provided weights_save_path from `weights_save_path/name1_name2/version1_version2/checkpoints` to `weights_save_path/checkpoints` ([#12372](https://github.com/Lightning-AI/lightning/pull/12372))
- Marked `swa_lrs` argument in `StochasticWeightAveraging` callback as required ([#12556](https://github.com/Lightning-AI/lightning/pull/12556))
- `LightningCLI`'s shorthand notation changed to use jsonargparse native feature ([#12614](https://github.com/Lightning-AI/lightning/pull/12614))
- `LightningCLI` changed to use jsonargparse native support for list append ([#13129](https://github.com/Lightning-AI/lightning/pull/13129))
- Changed `seed_everything_default` argument in the `LightningCLI` to type `Union[bool, int]`. If set to `True` a seed is automatically generated for the parser argument `--seed_everything`. ([#12822](https://github.com/Lightning-AI/lightning/pull/12822), [#13110](https://github.com/Lightning-AI/lightning/pull/13110))
- Make positional arguments required for classes passed into the `add_argparse_args` function. ([#12504](https://github.com/Lightning-AI/lightning/pull/12504))
- Raise an error if there are insufficient training batches when using a float value of `limit_train_batches` ([#12885](https://github.com/Lightning-AI/lightning/pull/12885))
- `DataLoader` instantiated inside a `*_dataloader` hook will not set the passed arguments as attributes anymore ([#12981](https://github.com/Lightning-AI/lightning/pull/12981))
- When a multi-element tensor is logged, an error is now raised instead of silently taking the mean of all elements ([#13164](https://github.com/Lightning-AI/lightning/pull/13164))
- The `WandbLogger` will now use the run name in the logs folder if it is provided, and otherwise the project name  ([#12604](https://github.com/Lightning-AI/lightning/pull/12604))
- Enabled using any Sampler in distributed environment in Lite ([#13646](https://github.com/Lightning-AI/lightning/pull/13646))
- Raised a warning instead of forcing `sync_dist=True` on epoch end ([13364](https://github.com/Lightning-AI/lightning/pull/13364))
- Updated `val_check_interval`(int) to consider total train batches processed instead of `_batches_that_stepped` for validation check during training ([#12832](https://github.com/Lightning-AI/lightning/pull/12832)
- Updated Habana Accelerator's `auto_device_count`, `is_available` & `get_device_name` methods based on the latest torch habana package ([#13423](https://github.com/Lightning-AI/lightning/pull/13423))
- Disallowed using `BatchSampler` when running on multiple IPUs ([#13854](https://github.com/Lightning-AI/lightning/pull/13854))


</details>

<details><summary>Deprecated</summary>


- Deprecated `pytorch_lightning.accelerators.gpu.GPUAccelerator` in favor of `pytorch_lightning.accelerators.cuda.CUDAAccelerator` ([#13636](https://github.com/Lightning-AI/lightning/pull/13636))
- Deprecated `pytorch_lightning.loggers.base.LightningLoggerBase` in favor of `pytorch_lightning.loggers.logger.Logger`, and deprecated `pytorch_lightning.loggers.base` in favor of `pytorch_lightning.loggers.logger` ([#120148](https://github.com/Lightning-AI/lightning/pull/12014))
- Deprecated `pytorch_lightning.callbacks.base.Callback` in favor of `pytorch_lightning.callbacks.callback.Callback` ([#13031](https://github.com/Lightning-AI/lightning/pull/13031))
- Deprecated `num_processes`, `gpus`, `tpu_cores,` and `ipus` from the `Trainer` constructor in favor of using the `accelerator` and `devices` arguments ([#11040](https://github.com/Lightning-AI/lightning/pull/11040))
- Deprecated setting `LightningCLI(seed_everything_default=None)` in favor of `False` ([#12804](https://github.com/Lightning-AI/lightning/issues/12804)).
- Deprecated `pytorch_lightning.core.lightning.LightningModule` in favor of `pytorch_lightning.core.module.LightningModule` ([#12740](https://github.com/Lightning-AI/lightning/pull/12740))
- Deprecated `pytorch_lightning.loops.base.Loop` in favor of `pytorch_lightning.loops.loop.Loop` ([#13043](https://github.com/Lightning-AI/lightning/pull/13043))
- Deprecated `Trainer.reset_train_val_dataloaders()` in favor of `Trainer.reset_{train,val}_dataloader` ([#12184](https://github.com/Lightning-AI/lightning/pull/12184))
- Deprecated LightningCLI's registries in favor of importing the respective package ([#13221](https://github.com/Lightning-AI/lightning/pull/13221))
- Deprecated public utilities in `pytorch_lightning.utilities.cli.LightningCLI` in favor of equivalent copies in `pytorch_lightning.cli.LightningCLI` ([#13767](https://github.com/Lightning-AI/lightning/pull/13767))
- Deprecated `pytorch_lightning.profiler` in favor of `pytorch_lightning.profilers` ([#12308](https://github.com/Lightning-AI/lightning/pull/12308))


</details>

<details><summary>Removed</summary>


- Removed deprecated `IndexBatchSamplerWrapper.batch_indices` ([#13565](https://github.com/Lightning-AI/lightning/pull/13565))
- Removed the deprecated `LightningModule.add_to_queue` and `LightningModule.get_from_queue` method ([#13600](https://github.com/Lightning-AI/lightning/pull/13600))
- Removed deprecated `pytorch_lightning.core.decorators.parameter_validation` from `decorators` ([#13514](https://github.com/Lightning-AI/lightning/pull/13514))
- Removed the deprecated `Logger.close` method ([#13149](https://github.com/Lightning-AI/lightning/pull/13149))
- Removed the deprecated `weights_summary` argument from the `Trainer` constructor ([#13070](https://github.com/Lightning-AI/lightning/pull/13070))
- Removed the deprecated `flush_logs_every_n_steps` argument from the `Trainer` constructor ([#13074](https://github.com/Lightning-AI/lightning/pull/13074))
- Removed the deprecated `process_position` argument from the `Trainer` constructor ([13071](https://github.com/Lightning-AI/lightning/pull/13071))
- Removed the deprecated `checkpoint_callback` argument from the `Trainer` constructor ([#13027](https://github.com/Lightning-AI/lightning/pull/13027))
- Removed the deprecated `on_{train,val,test,predict}_dataloader` hooks from the `LightningModule` and `LightningDataModule` ([#13033](https://github.com/Lightning-AI/lightning/pull/13033))
- Removed the deprecated `TestTubeLogger` ([#12859](https://github.com/Lightning-AI/lightning/pull/12859))
- Removed the deprecated `pytorch_lightning.core.memory.LayerSummary` and `pytorch_lightning.core.memory.ModelSummary` ([#12593](https://github.com/Lightning-AI/lightning/pull/12593))
- Removed the deprecated `summarize` method from the `LightningModule` ([#12559](https://github.com/Lightning-AI/lightning/pull/12559))
- Removed the deprecated `model_size` property from the `LightningModule` class ([#12641](https://github.com/Lightning-AI/lightning/pull/12641))
- Removed the deprecated `stochastic_weight_avg` argument from the `Trainer` constructor ([#12535](https://github.com/Lightning-AI/lightning/pull/12535))
- Removed the deprecated `progress_bar_refresh_rate` argument from the `Trainer` constructor ([#12514](https://github.com/Lightning-AI/lightning/pull/12514))
- Removed the deprecated `prepare_data_per_node` argument from the `Trainer` constructor ([#12536](https://github.com/Lightning-AI/lightning/pull/12536))
- Removed the deprecated `pytorch_lightning.core.memory.{get_gpu_memory_map,get_memory_profile}` ([#12659](https://github.com/Lightning-AI/lightning/pull/12659))
- Removed the deprecated `terminate_on_nan` argument from the `Trainer` constructor ([#12553](https://github.com/Lightning-AI/lightning/pull/12553))
- Removed the deprecated `XLAStatsMonitor` callback ([#12688](https://github.com/Lightning-AI/lightning/pull/12688))
- Remove deprecated `pytorch_lightning.callbacks.progress.progress` ([#12658](https://github.com/Lightning-AI/lightning/pull/12658))
- Removed the deprecated `dim` and `size` arguments from the `LightningDataModule` constructor([#12780](https://github.com/Lightning-AI/lightning/pull/12780))
- Removed the deprecated `train_transforms` argument from the `LightningDataModule` constructor([#12662](https://github.com/Lightning-AI/lightning/pull/12662))
- Removed the deprecated `log_gpu_memory` argument from the `Trainer` constructor ([#12657](https://github.com/Lightning-AI/lightning/pull/12657))
- Removed the deprecated automatic logging of GPU stats by the logger connector ([#12657](https://github.com/Lightning-AI/lightning/pull/12657))
- Removed deprecated `GPUStatsMonitor` callback ([#12554](https://github.com/Lightning-AI/lightning/pull/12554))
- Removed support for passing strategy names or strategy instances to the accelerator Trainer argument ([#12696](https://github.com/Lightning-AI/lightning/pull/12696))
- Removed support for passing strategy names or strategy instances to the plugins Trainer argument ([#12700](https://github.com/Lightning-AI/lightning/pull/12700))
- Removed the deprecated `val_transforms` argument from the `LightningDataModule` constructor ([#12763](https://github.com/Lightning-AI/lightning/pull/12763))
- Removed the deprecated `test_transforms` argument from the `LightningDataModule` constructor ([#12773](https://github.com/Lightning-AI/lightning/pull/12773))
- Removed deprecated `Trainer(max_steps=None)` ([#13591](https://github.com/Lightning-AI/lightning/pull/13591))
- Removed deprecated `dataloader_idx` argument from `on_train_batch_start/end` hooks `Callback` and `LightningModule` ([#12769](https://github.com/Lightning-AI/lightning/pull/12769), [#12977](https://github.com/Lightning-AI/lightning/pull/12977))
- Removed deprecated `get_progress_bar_dict` property from `LightningModule` ([#12839](https://github.com/Lightning-AI/lightning/pull/12839))
- Removed sanity check for multi-optimizer support with habana backends ([#13217](https://github.com/Lightning-AI/lightning/pull/13217))
- Removed the need to explicitly load habana module ([#13338](https://github.com/Lightning-AI/lightning/pull/13338))
- Removed the deprecated `Strategy.post_dispatch()` hook ([#13461](https://github.com/Lightning-AI/lightning/pull/13461))
- Removed deprecated `pytorch_lightning.callbacks.lr_monitor.LearningRateMonitor.lr_sch_names` ([#13353](https://github.com/Lightning-AI/lightning/pull/13353))
- Removed deprecated `Trainer.slurm_job_id` in favor of `SLURMEnvironment.job_id` ([#13459](https://github.com/Lightning-AI/lightning/pull/13459))
- Removed support for the `DDP2Strategy` ([#12705](https://github.com/Lightning-AI/lightning/pull/12705))
- Removed deprecated `LightningDistributed` ([#13549](https://github.com/Lightning-AI/lightning/pull/13549))
- Removed deprecated ClusterEnvironment properties `master_address` and `master_port` in favor of `main_address` and `main_port` ([#13458](https://github.com/Lightning-AI/lightning/pull/13458))
- Removed deprecated ClusterEnvironment methods `KubeflowEnvironment.is_using_kubelfow()`, `LSFEnvironment.is_using_lsf()` and `TorchElasticEnvironment.is_using_torchelastic()` in favor of the `detect()` method ([#13458](https://github.com/Lightning-AI/lightning/pull/13458))
- Removed deprecated `Callback.on_keyboard_interrupt` ([#13438](https://github.com/Lightning-AI/lightning/pull/13438))
- Removed deprecated `LightningModule.on_post_move_to_device` ([#13548](https://github.com/Lightning-AI/lightning/pull/13548))
- Removed `TPUSpawnStrategy.{tpu_local_core_rank,tpu_global_core_rank}` attributes in favor of `TPUSpawnStrategy.{local_rank,global_rank}` ([#11163](https://github.com/Lightning-AI/lightning/pull/11163))
- Removed `SingleTPUStrategy.{tpu_local_core_rank,tpu_global_core_rank}` attributes in favor of `SingleTPUStrategy.{local_rank,global_rank}`([#11163](https://github.com/Lightning-AI/lightning/pull/11163))

</details>

<details><summary>Fixed</summary>


- Improved support for custom `DataLoader`s when instantiated in `*_dataloader` hook ([#12981](https://github.com/Lightning-AI/lightning/pull/12981))
- Allowed custom `BatchSampler`s when instantiated in `*_dataloader` hook [#13640](https://github.com/Lightning-AI/lightning/pull/13640))
- Fixed an issue with unsupported torch.inference_mode() on hpu backends by making it use no_grad ([#13014](https://github.com/Lightning-AI/lightning/pull/13014))
- The model wrapper returned by `LightningLite.setup()` now properly supports pass-through when looking up attributes ([#12597](https://github.com/Lightning-AI/lightning/pull/12597))
- Fixed issue where the CLI fails with certain torch objects ([#13153](https://github.com/Lightning-AI/lightning/pull/13153))
- Fixed ``LightningCLI`` signature parameter resolving for some lightning classes ([#13283](https://github.com/Lightning-AI/lightning/pull/13283))
- Fixed Model Summary when using DeepSpeed Stage 3 ([#13427](https://github.com/Lightning-AI/lightning/pull/13427))
- Fixed `pytorch_lightning.utilities.distributed.gather_all_tensors` to handle tensors of different dimensions ([#12630](https://github.com/Lightning-AI/lightning/pull/12630))
- Fixed the input validation for the accelerator Trainer argument when passed as a string ([#13417](https://github.com/Lightning-AI/lightning/pull/13417))
- Fixed `Trainer.predict(return_predictions=False)` to track prediction's batch_indices ([#13629](https://github.com/Lightning-AI/lightning/pull/13629))
- Fixed and issue that prevented setting a custom `CheckpointIO` plugin with strategies ([#13785](https://github.com/Lightning-AI/lightning/pull/13785))
- Fixed main progress bar counter when `val_check_interval=int` and `check_val_every_n_epoch=None` ([#12832](https://github.com/Lightning-AI/lightning/pull/12832)
- Improved support for custom `ReduceLROnPlateau` scheduler if `reduce_on_plateau` is set by the user in scheduler config ([#13838](https://github.com/Lightning-AI/lightning/pull/13838))
- Used `global_step` while restoring logging step for old checkpoints ([#13645](https://github.com/Lightning-AI/lightning/pull/13645))
- When training with `precision=16` on IPU, the cast has been moved off the IPU onto the host, making the copies from host to IPU cheaper ([#13880](https://github.com/Lightning-AI/lightning/pull/13880))
- Fixed error handling in learning rate finder when not enough data points are available to give a good suggestion ([#13845](https://github.com/Lightning-AI/lightning/pull/13845))
- Fixed an issue that caused the learning rate finder to set the model's learning rate to None when no suggestion was possible ([#13845](https://github.com/Lightning-AI/lightning/pull/13845))
- Fixed an issue causing deterministic algorighms and other globals to get reset in spawned processes ([#13921](https://github.com/Lightning-AI/lightning/pull/13921))
- Fixed default `amp_level` for `DeepSpeedPrecisionPlugin` to `O2` ([#13897](https://github.com/Lightning-AI/lightning/pull/13897))
- Fixed Python 3.10 compatibility for truncated back-propagation through time (TBPTT) ([#13973](https://github.com/Lightning-AI/lightning/pull/13973))
- Fixed `TQDMProgressBar` reset and update to show correct time estimation (2/2) ([#13962](https://github.com/Lightning-AI/lightning/pull/13962))




</details>

**Full commit list**: https://github.com/PyTorchLightning/pytorch-lightning/compare/1.6.0...1.7.0

<a name="contributors"></a>
# Contributors

### Veteran

@akashkw @akihironitta @aniketmaurya @awaelchli @Benjamin-Etheredge @Borda @carmocca @catalys1 @daniellepintz @edenlightning @edward-io @EricWiener @fschlatt @ftorres16 @jerome-habana @justusschock @karthikrangasai @kaushikb11 @krishnakalyan3 @krshrimali @mauvilsa @nikvaessen @otaj @pre-commit-ci @puhuk @raoakarsha @rasbt @rohitgr7 @SeanNaren @s-rog @talregev @tchaton @tshu-w @twsl @weiji14 @williamFalcon @WrRan

### New

@alvitawa @aminst @ankitaS11 @ar90n @Atharva-Phatak @bibhabasumohapatra @BongYang @code-review-doctor @CompRhys @Cyprien-Ricque @dependabot @digital-idiot @DN6 @donlapark @ekagra-ranjan @ethanfurman @gautierdag @georgestein @HallerPatrick @HenryLau0220 @hhsecond @himkt @HMellor @igorgad @inwaves @ishtos @JeroenDelcour @JiahaoYao @jiny419 @jinyoung-lim @JustinGoheen @jxmorris12 @Keiku @kingjuno @lsy643 @luca-medeiros @lukasugar @maciek-pioro @mads-oestergaard @manskx @martinosorb @MohammedAlkhrashi @MrShevan @myxik @naisofly @NathanielDamours @nayoungjun @niberger @nitinramvelraj @nninept @pbsds @Pragyanstha @PrajwalBorkar @Prometheos2 @rampartrange @rhjohnstone @rschireman @samz5320 @Schinkikami @semaphore-egg @shantam-8 @shenoynikhil @sisilmehta2000 @s-kumano @stanbiryukov @talregev @tanmoyio @tkonopka @vumichien @wangherr @yhl48 @YongWookHa

_If we forgot somebody or you have a suggestion, find [support here](https://www.pytorchlightning.ai/support) :zap:_

#### Did you know?

Chuck Norris can unit-test entire applications with a single assert.


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.7.0)

---

## app/0.5.4: Build-in templates
**Published:** 2022-08-01

## [0.5.4] - 2022-08-01

### Changed

- Wrapped imports for traceability (#13924)
- Set version as today (#13906)

### Fixed

- Included app templates to the lightning and app packages (#13731)
- Added UI for installing it all (#13732)
- Fixed build meta pkg flow (#13926)


### Contributors

@Borda, @manskx

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.5.4)

---

## app/0.5.3: Minor bug-fix release
**Published:** 2022-07-26

## [0.5.3] - 2022-07-25

### Changed

- Pruned requirements duplicity (#13739)

### Fixed

- Use correct python version in lightning component template (#13790)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.5.3)

---

## app/0.5.2: Lightning App 0.5.2
**Published:** 2022-07-18

## [0.5.2] - 2022-07-18

### Added

- Update the Lightning App docs (#13537)

### Changed

- Added `LIGHTNING_` prefix to Platform AWS credentials (#13703)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/app/0.5.2)

---

## 1.6.5: PyTorch Lightning 1.6.5: Standard patch release
**Published:** 2022-07-13

## [1.6.5] - 2022-07-13

### Fixed

- Fixed `estimated_stepping_batches` requiring distributed comms in `configure_optimizers` for the `DeepSpeedStrategy` ([#13350](https://github.com/PyTorchLightning/pytorch-lightning/pull/13350))
- Fixed bug with Python version check that prevented use with development versions of Python ([#13420](https://github.com/PyTorchLightning/pytorch-lightning/pull/13420))
- The loops now call `.set_epoch()` also on batch samplers if the dataloader has one wrapped in a distributed sampler ([#13396](https://github.com/PyTorchLightning/pytorch-lightning/pull/13396))
- Fixed the restoration of log step during restart ([#13467](https://github.com/PyTorchLightning/pytorch-lightning/pull/13467))

### Contributors

@adamjstewart @akihironitta @awaelchli @Borda @martinosorb @rohitgr7 @SeanNaren

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.6.5)

---

## 1.6.4: PyTorch Lightning 1.6.4: Standard patch release
**Published:** 2022-06-01

## [1.6.4] - 2022-06-01

### Added

- Added all DDP params to be exposed through hpu parallel strategy ([#13067](https://github.com/PyTorchLightning/pytorch-lightning/pull/13067))

### Changed

- Keep `torch.backends.cudnn.benchmark=False` by default (unlike in v1.6.{0-4}) after speed and memory problems depending on the data used. Please consider tuning `Trainer(benchmark)` manually. ([#13154](https://github.com/PyTorchLightning/pytorch-lightning/pull/13154))
- Prevent modification of `torch.backends.cudnn.benchmark` when `Trainer(benchmark=...)` is not set ([#13154](https://github.com/PyTorchLightning/pytorch-lightning/pull/13154))

### Fixed

- Fixed an issue causing zero-division error for empty dataloaders ([#12885](https://github.com/PyTorchLightning/pytorch-lightning/pull/12885))
- Fixed mismatching default values for the types of some arguments in the DeepSpeed and Fully-Sharded strategies which made the CLI unable to use them ([#12989](https://github.com/PyTorchLightning/pytorch-lightning/pull/12989))
- Avoid redundant callback restore warning while tuning ([#13026](https://github.com/PyTorchLightning/pytorch-lightning/pull/13026))
- Fixed `Trainer(precision=64)` during evaluation which now uses the wrapped precision module ([#12983](https://github.com/PyTorchLightning/pytorch-lightning/pull/12983))
- Fixed an issue to use wrapped `LightningModule` for evaluation during `trainer.fit` for `BaguaStrategy` ([#12983](https://github.com/PyTorchLightning/pytorch-lightning/pull/12983))
- Fixed an issue wrt unnecessary usage of habana mixed precision package for fp32 types ([#13028](https://github.com/PyTorchLightning/pytorch-lightning/pull/13028))
- Fixed the number of references of `LightningModule` so it can be deleted ([#12897](https://github.com/PyTorchLightning/pytorch-lightning/pull/12897))
- Fixed `materialize_module` setting a module's child recursively ([#12870](https://github.com/PyTorchLightning/pytorch-lightning/pull/12870))
- Fixed issue where the CLI could not pass a `Profiler` to the `Trainer` ([#13084](https://github.com/PyTorchLightning/pytorch-lightning/pull/13084))
- Fixed torchelastic detection with non-distributed installations ([#13142](https://github.com/PyTorchLightning/pytorch-lightning/pull/13142))
- Fixed logging's step values when multiple dataloaders are used during evaluation ([#12184](https://github.com/PyTorchLightning/pytorch-lightning/pull/12184))
- Fixed epoch logging on train epoch end ([#13025](https://github.com/PyTorchLightning/pytorch-lightning/pull/13025))
- Fixed `DDPStrategy` and `DDPSpawnStrategy` to initialize optimizers only after moving the module to the device ([#11952](https://github.com/PyTorchLightning/pytorch-lightning/pull/11952))

### Contributors

@akihironitta @ananthsub @ar90n @awaelchli @Borda @carmocca @dependabot @jerome-habana @mads-oestergaard @otaj @rohitgr7 

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.6.4)

---

## 1.6.3: PyTorch Lightning 1.6.3: Standard patch release
**Published:** 2022-05-03

## [1.6.3] - 2022-05-03

### Fixed

- Use only a single instance of `rich.console.Console` throughout codebase ([#12886](https://github.com/PyTorchLightning/pytorch-lightning/pull/12886))
- Fixed an issue to ensure all the checkpoint states are saved in a common filepath with `DeepspeedStrategy` ([#12887](https://github.com/PyTorchLightning/pytorch-lightning/pull/12887))
- Fixed `trainer.logger` deprecation message ([#12671](https://github.com/PyTorchLightning/pytorch-lightning/pull/12671))
- Fixed an issue where sharded grad scaler is passed in when using BF16 with the `ShardedStrategy` ([#12915](https://github.com/PyTorchLightning/pytorch-lightning/pull/12915))
- Fixed an issue wrt recursive invocation of DDP configuration in hpu parallel plugin ([#12912](https://github.com/PyTorchLightning/pytorch-lightning/pull/12912))
- Fixed printing of ragged dictionaries in `Trainer.validate` and `Trainer.test` ([#12857](https://github.com/PyTorchLightning/pytorch-lightning/pull/12857))
- Fixed threading support for legacy loading of checkpoints ([#12814](https://github.com/PyTorchLightning/pytorch-lightning/pull/12814))
- Fixed pickling of `KFoldLoop` ([#12441](https://github.com/PyTorchLightning/pytorch-lightning/pull/12441))
- Stopped `optimizer_zero_grad` from being called after IPU execution ([#12913](https://github.com/PyTorchLightning/pytorch-lightning/pull/12913))
- Fixed `fuse_modules` to be qat-aware for `torch>=1.11` ([#12891](https://github.com/PyTorchLightning/pytorch-lightning/pull/12891))
- Enforced eval shuffle warning only for default samplers in DataLoader ([#12653](https://github.com/PyTorchLightning/pytorch-lightning/pull/12653))
- Enable mixed precision in `DDPFullyShardedStrategy` when `precision=16` ([#12965](https://github.com/PyTorchLightning/pytorch-lightning/pull/12965))
- Fixed `TQDMProgressBar` reset and update to show correct time estimation ([#12889](https://github.com/PyTorchLightning/pytorch-lightning/pull/12889))
- Fixed fit loop restart logic to enable resume using the checkpoint ([#12821](https://github.com/PyTorchLightning/pytorch-lightning/pull/12821))

### Contributors

@akihironitta @carmocca @HMellor @jerome-habana @kaushikb11 @krshrimali @mauvilsa @niberger @ORippler @otaj @rohitgr7 @SeanNaren

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.6.3)

---

## 1.6.2: PyTorch Lightning 1.6.2: Standard patch release
**Published:** 2022-04-27

## [1.6.2] - 2022-04-27

### Fixed

- Fixed `ImportError` when `torch.distributed` is not available. ([#12794](https://github.com/PyTorchLightning/pytorch-lightning/pull/12794))
- When using custom DataLoaders in LightningDataModule, multiple inheritance is resolved properly ([#12716](https://github.com/PyTorchLightning/pytorch-lightning/pull/12716))
- Fixed encoding issues on terminals that do not support unicode characters ([#12828](https://github.com/PyTorchLightning/pytorch-lightning/pull/12828))
- Fixed support for `ModelCheckpoint` monitors with dots ([#12783](https://github.com/PyTorchLightning/pytorch-lightning/pull/12783))

### Contributors

@akihironitta @alvitawa @awaelchli @Borda @carmocca @code-review-doctor @ethanfurman @HenryLau0220 @krshrimali @otaj 

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.6.2)

---

## 1.6.1: PyTorch Lightning 1.6.1: Standard weekly patch release
**Published:** 2022-04-13

## [1.6.1] - 2022-04-13

### Changed

- Support `strategy` argument being case insensitive ([#12528](https://github.com/PyTorchLightning/pytorch-lightning/pull/12528))

### Fixed

- Run main progress bar updates independent of val progress bar updates in `TQDMProgressBar` ([#12563](https://github.com/PyTorchLightning/pytorch-lightning/pull/12563))
- Avoid calling `average_parameters` multiple times per optimizer step ([#12452](https://github.com/PyTorchLightning/pytorch-lightning/pull/12452))
- Properly pass some Logger's parent's arguments to `super().__init__()` ([#12609](https://github.com/PyTorchLightning/pytorch-lightning/pull/12609))
- Fixed an issue where incorrect type warnings appear when the overridden `LightningLite.run` method accepts user-defined arguments ([#12629](https://github.com/PyTorchLightning/pytorch-lightning/pull/12629))
- Fixed `rank_zero_only` decorator in LSF environments ([#12587](https://github.com/PyTorchLightning/pytorch-lightning/pull/12587))
- Don't raise a warning when `nn.Module` is not saved under hparams ([#12669](https://github.com/PyTorchLightning/pytorch-lightning/pull/12669))
- Raise `MisconfigurationException` when the accelerator is available but the user passes invalid `([]/0/"0")` values to the `devices` flag ([#12708](https://github.com/PyTorchLightning/pytorch-lightning/pull/12708))
- Support `auto_select_gpus` with the accelerator and devices API ([#12608](https://github.com/PyTorchLightning/pytorch-lightning/pull/12608))


### Contributors

@akihironitta @awaelchli @Borda @carmocca @kaushikb11 @krshrimali @mauvilsa @otaj @pre-commit-ci @rohitgr7 @semaphore-egg @tkonopka @wayi1


If we forgot someone due to not matching the commit email with the GitHub account, let us know :]


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.6.1)

---

## 1.6.0: PyTorch Lightning 1.6: Support Intel's Habana Accelerator, New efficient DDP strategy (Bagua), Manual Fault-tolerance, Stability and Reliability.
**Published:** 2022-03-29

The core team is excited to announce the PyTorch Lightning 1.6 release ⚡

- [Highlights](#highlights)
- [Backward Incompatible Changes](#bc-changes)
- [Full Changelog](#changelog)
- [Contributors](#contributors)

<a name="highlights"></a>
## Highlights

PyTorch Lightning 1.6 is the work of 99 contributors who have worked on features, bug-fixes, and documentation for a total of over 750 commits since 1.5. This is our most active release yet. Here are some highlights:

### Introducing Intel's Habana Accelerator

Lightning 1.6 now supports the Habana® framework, which includes Gaudi® AI training processors. Their heterogeneous architecture includes a cluster of fully programmable Tensor Processing Cores (TPC) along with its associated development tools and libraries and a configurable Matrix Math engine. 

You can leverage the [Habana](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/hpu.html) hardware to accelerate your Deep Learning training workloads simply by passing:

```python
trainer = pl.Trainer(accelerator="hpu")

# single Gaudi training
trainer = pl.Trainer(accelerator="hpu", devices=1)

# distributed training with 8 Gaudi
trainer = pl.Trainer(accelerator="hpu", devices=8)
```

### The Bagua Strategy

The [Bagua Strategy](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html#bagua) is a deep learning acceleration framework that supports multiple, advanced distributed training algorithms with state-of-the-art system relaxation techniques. Enabling [Bagua](https://github.com/BaguaSys/bagua), which can be considerably faster than vanilla PyTorch DDP, is as simple as:

```python
trainer = pl.Trainer(strategy="bagua")

# or to choose a custom algorithm
trainer = pl.Trainer(strategy=BaguaStrategy(algorithm="gradient_allreduce")  # default
```

### Towards stable Accelerator, Strategy, and Plugin APIs


The `Accelerator`, `Strategy`, and `Plugin` APIs are a core part of PyTorch Lightning. They're where all the distributed boilerplate lives, and we're constantly working to improve both them and the overall PyTorch Lightning platform experience.

In this release, we've made some large changes to achieve that goal. Not to worry, though! The only users affected by these changes are those who use custom implementations of Accelerator and Strategy (`TrainingTypePlugin`) as well as certain Plugins. In particular, we want to highlight the following changes:

- All `TrainingTypePlugin`s have been renamed to `Strategy` ([#11120](https://github.com/PyTorchLightning/pytorch-lightning/pull/11120)). Strategy is a more appropriate name because it encompasses more than simply training communcation. This change is now aligned with the changes we implemented in 1.5, which introduced the new [`strategy` and `devices` flags to the Trainer](https://github.com/PyTorchLightning/pytorch-lightning/releases/tag/1.5.0#strategy-and-devices).


    ```python
    # Before
    from pytorch_lightning.plugins import DDPPlugin

    # New
    from pytorch_lightning.strategies import DDPStrategy
    ```

- The `Accelerator` and `PrecisionPlugin` have moved into `Strategy`. All strategies now take an optional parameter `accelerator` and `precision_plugin` ([#11022](https://github.com/PyTorchLightning/pytorch-lightning/pull/11022), [#10570](https://github.com/PyTorchLightning/pytorch-lightning/pull/10570)).

- Custom Accelerator implementations must now implement two new abstract methods: `is_available()` ([#11797](https://github.com/PyTorchLightning/pytorch-lightning/pull/11797)) and `auto_device_count()` ([#10222](https://github.com/PyTorchLightning/pytorch-lightning/pull/10222)). The latter determines how many devices get used by default when specifying `Trainer(accelerator=..., devices="auto")`.


- We redesigned the process creation for spawn-based strategies such as `DDPSpawnStrategy` and `TPUSpawnStrategy` ([#10896](https://github.com/PyTorchLightning/pytorch-lightning/pull/10896)). All spawn-based strategies now spawn processes immediately upon calling `Trainer.{fit,validate,test,predict}`, which means the hooks/callbacks `prepare_data`, `setup`, `configure_sharded_model` and `teardown` all run under an initialized process group. These changes align the spawn-based strategies with their non-spawn counterparts (such as `DDPStrategy`).



We've also exposed the process group backend for use. For example, you can now easily enable [`fairring`](https://github.com/facebookresearch/fairring) like this:

```python
# Explicitly specify the process group backend if you choose to
ddp = pl.strategies.DDPStrategy(process_group_backend="fairring")
trainer = Trainer(strategy=ddp, accelerator="gpu", devices=8)
```

In a similar fashion, if installing `torch>=1.11`, you can enable [DDP static graph](https://pytorch.org/blog/pytorch-1.11-released/#stable-ddp-static-graph) to apply special runtime optimizations:

```python
trainer = Trainer(devices=4, strategy=DDPStrategy(static_graph=True))
```

### `LightningCLI` improvements


In the previous release, we added shorthand notation support for registered components. In this release, we added a flag to [automatically register](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html#subclass-registration) all available components:

```python
from pytorch_lightning.utilities.cli import LightningCLI

LightningCLI(auto_registry=True)
```

We have also added support for the `ReduceLROnPlateau` scheduler with shorthand notation:

```bash
$ python script.py fit --optimizer=Adam --lr_scheduler=ReduceLROnPlateau --lr_scheduler.monitor=metric_to_track
```

If you need to [customize the learning rate scheduler configuration](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html#optimizers-and-learning-rate-schedulers), you can do so by overriding:
```python
class MyLightningCLI(LightningCLI):
    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, ...}}
```

Finally, loggers are also now configurable with shorthand:

```bash=
$ python script.py fit --trainer.logger=WandbLogger --trainer.logger.name="my_lightning_run"
```

### Control SLURM's re-queueing

We've added the ability to turn the [automatic resubmission](https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html#wall-time-auto-resubmit) on or off when a job gets interrupted by the SLURM controller (via signal handling). Users who prefer to let their code handle the resubmission (for example, when submitit is used) can now pass:

```python
from pytorch_lightning.plugins.environments import SLURMEnvironment

trainer = pl.Trainer(plugins=SLURMEnvironment(auto_requeue=False))
``` 

### Fault-tolerance improvements 

The [Fault-tolerance training](https://pytorch-lightning.readthedocs.io/en/stable/advanced/fault_tolerant_training.html) under manual optimization now tracks optimization progress. We also changed the graceful exit signal from `SIGUSR1` to `SIGTERM` for better support inside cloud instances.
An additional feature we're excited to announce is support for consecutive `trainer.fit()` calls.

```python
trainer = pl.Trainer(max_epochs=2)
trainer.fit(model)

# now, run 2 more epochs
trainer.fit_loop.max_epochs = 4
trainer.fit(model)
```

### Loop customization improvements

The [`Loop`](https://pytorch-lightning.readthedocs.io/en/stable/extensions/loops.html)'s state is now included as part of the checkpoints saved by the library. This enables finer restoration of custom loops.

We've also made it easier to replace Lightning's loops with your own. For example:

```python
class MyCustomLoop(pl.loops.TrainingEpochLoop):
    ...

trainer = pl.Trainer(...)
trainer.fit_loop.replace(epoch_loop=MyCustomLoop)
# Trainer runs the fit loop with your new epoch loop!
trainer.fit(model)
```

### Data-Loading improvements

In previous versions, Lightning required that the `DataLoader` instance set its input arguments as instance attributes. This meant that custom `DataLoader`s also had this hidden requirement. In this release, we do this automatically for the user, easing the passing of custom loaders:

```diff
class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, a=123, *args, **kwargs):
-       # this was required before
-       self.a = a
        super().__init__(*args, **kwargs)

trainer.fit(model, train_dataloader=MyDataLoader())
```

As of this release, Lightning no longer pre-fetches 1 extra batch if it doesn't need to. Previously, doing so would conflict with the internal pre-fetching done by optimized data loaders such as [FFCV's](https://ffcv.io/). You can now define your own pre-fetching value like this:

```python
class MyCustomLoop(pl.loops.FitLoop):
    @property
    def prefetch_batches(self):
        return 7  # lucky number 7

trainer = pl.Trainer(...)
trainer.fit_loop = MyCustomLoop(min_epochs=trainer.min_epochs, max_epochs=trainer.max_epochs)
```

## New Hooks

### `LightningModule.lr_scheduler_step`

Lightning now allows the use of [custom learning rate schedulers](https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html#bring-your-own-custom-learning-rate-schedulers) that aren't natively available in [PyTorch](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). A great example of this is [Timm Schedulers](https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler.py).

When using custom learning rate schedulers relying on an API other than PyTorch's, you can now define the `LightningModule.lr_scheduler_step` with your desired logic.

```python
from timm.scheduler import TanhLRScheduler


class MyLightningModule(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = ...
        scheduler = TanhLRScheduler(optimizer, ...)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
```

### A new stateful API

This release introduces new hooks to standardize all stateful components to use `state_dict` and `load_state_dict`, mimicking the PyTorch API. The new hooks receive their own component's state and replace most usages of the previous `on_save_checkpoint` and `on_load_checkpoint` hooks.

```diff
def MyCallback(pl.Callback):
-   def on_save_checkpoint(self, trainer, pl_module, checkpoint):
-       return {'x': self.x}
    
-   def on_load_checkpoint(self, trainer, pl_module, checkpoint):
-       self.x = x

+   def state_dict(self):
+       return {'x': self.x}
    
+   def load_state_dict(self, checkpoint):
+       self.x = x     
```

## New properties

### `Trainer.estimated_stepping_batches`

You can use built-in `Trainer.estimated_stepping_batches` to compute the total number of stepping batches needed for the complete training.

The property takes gradient accumulation factor and distributed setting into consideration when performing this computation so that you don't have to derive it manually:

```python
class MyLightningModule(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = ...
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```


### `Trainer.num_devices` and `Trainer.device_ids` 

In the past, retrieving the number of devices used, or their IDs, posed a considerable challenge. Additionally, doing so required knowing which property to access based on the current `Trainer` configuration. 

To simplify this process, we've deprecated the per-accelerator properties to have accelerator agnostic properties. For example:

```diff
- num_devices = max(1, trainer.num_gpus, trainer.num_processes)
- if trainer.tpu_cores:
-    num_devices = max(num_devices, trainer.tpu_cores)
+ num_devices = trainer.num_devices
```

## Experimental Features

### Manual Fault-tolerance

[Fault Tolerance](https://pytorch-lightning.readthedocs.io/en/latest/advanced/fault_tolerant_training.html#:~:text=Fault%2Dtolerant%20Training%20is%20an,can%20shutdown%20at%20any%20time.) has limitations that require specific information about your data-loading structure.

It is now possible to resolve those limitations by enabling manual fault tolerance where you can write your own logic and specify how exactly to checkpoint your own datasets and samplers. You can do so using this environment flag:

```shell
$ PL_FAULT_TOLERANT_TRAINING=MANUAL python script.py
```

Check out [this video](https://www.youtube.com/watch?v=-HRh_szyuhE) for a dive into the internals of this flag.

### Customizing the layer synchronization

We introduced a new plugin class for wrapping layers of a model with synchronization logic for multiprocessing.

```python
class MyLayerSync(pl.plugins.LayerSync):
    ...

layer_sync = MyLayerSync(...)
trainer = Trainer(sync_batchnorm=True, plugins=layer_sync, strategy="ddp")
```

### Registering Custom Accelerators

There has been much progress in the field of ML Accelerators, and the list of accelerators is constantly expanding.

We've made it easier for users to try out new accelerators by enabling support for registering custom `Accelerator` classes in Lightning.

```python
from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry

class SOTAAccelerator(Accelerator):
    def __init__(self, x):
        ...

AcceleratorRegistry.register("sota_accelerator", SOTAAccelerator, x=123)
# the following works now:
trainer = Trainer(accelerator="sota_accelerator")
```

<a name="bc-changes"></a>
# Backward Incompatible Changes

Here is a selection of notable changes that are not backward compatible with previous versions. The full list of changes and removals can be found in the CHANGELOG below.

## Drop PyTorch 1.7 support

Following our 4 PyTorch release window, this release supports PyTorch 1.8 to 1.11. Support for PyTorch 1.7 has been removed.

## Drop Python 3.6 support

Following [Python's end-of-life](https://endoflife.date/python), support for Python 3.6 has been removed.

## `AcceleratorConnector` rewrite

To support new accelerator and stategy features, we completely rewrote our internal `AcceleratorConncetor` class. No backwards compatibility was maintained so it is likely to have broken your code if it was using this class.

## Re-define the `current_epoch` boundary

To resolve fault-tolerance issues, we changed where the current epoch value gets increased.

`trainer.current_epoch` is now increased by 1 `on_train_end`. This means that if a model is run for 3 epochs (0, 1, 2), `trainer.current_epoch` will now return 3 instead of 2 after `trainer.fit()`. This can also impact custom callbacks that acess this property inside this hook.

This also impacts checkpoints saved during an epoch (e.g. `on_train_epoch_end`). For example, a `Trainer(max_epochs=1, limit_train_batches=1)` instance that saves a checkpoint will have the `current_epoch=0` value saved instead of `current_epoch=1`.

## Re-define the `global_step` boundary

To resolve fault-tolerance issues, we changed where the global step value gets increased.

Access to `trainer.global_step` during an intra-training validation hook will now correctly return the number of optimizer steps taken already. In pseudocode:

```diff
  training_step()
+ global_step += 1
  validation_if_necessary()
- global_step += 1
```

Saved checkpoints that use the global step value as part of the filename are now increased by 1 for the same reason. A checkpoint saved after 1 step will be now be named `step=1.ckpt` instead of `step=0.ckpt`.

The `trainer.global_step` value will now account for TBPTT or multiple optimizers. Users setting `Trainer({min,max}_steps=...)` under these circumstances will need to adjust their values.

## Removed automatic reduction of outputs in `training_step` when using DataParallel

When using `Trainer(strategy="dp")`, *all* the tensors returned by training_step were previously reduced to a scalar (https://github.com/PyTorchLightning/pytorch-lightning/pull/11594). This behavior was especially confusing when outputs needed to be collected into the `training_epoch_end` hook. 

From now on, outputs are no longer reduced except for the `loss` tensor, unless you implement `training_step_end`, in which case the loss won't get reduced either.

## No longer fallback to CPU with no devices

Previous versions were lenient in that the lack of GPU devices defaulted to running on CPU. This meant that users' code could be running much slower without them ever noticing that it was running on CPU. 

We suggest passing `Trainer(accelerator="auto")` when this leniency is desired.

<a name="changelog"></a>
# CHANGELOG

<details><summary>Added</summary>

- Allow logging to an existing run ID in MLflow with `MLFlowLogger` ([#12290](https://github.com/PyTorchLightning/pytorch-lightning/pull/12290))
- Enable gradient accumulation using Horovod's `backward_passes_per_step` ([#11911](https://github.com/PyTorchLightning/pytorch-lightning/pull/11911))
- Add new `DETAIL` log level to provide useful logs for improving monitoring and debugging of batch jobs ([#11008](https://github.com/PyTorchLightning/pytorch-lightning/pull/11008))
- Added a flag `SLURMEnvironment(auto_requeue=True|False)` to control whether Lightning handles the requeuing ([#10601](https://github.com/PyTorchLightning/pytorch-lightning/pull/10601))
- Fault Tolerant Manual
    * Add `_Stateful` protocol to detect if classes are stateful ([#10646](https://github.com/PyTorchLightning/pytorch-lightning/pull/10646))
    * Add `_FaultTolerantMode` enum used to track different supported fault tolerant modes ([#10645](https://github.com/PyTorchLightning/pytorch-lightning/pull/10645))
    * Add a `_rotate_worker_indices` utility to reload the state according the latest worker ([#10647](https://github.com/PyTorchLightning/pytorch-lightning/pull/10647))
    * Add stateful workers ([#10674](https://github.com/PyTorchLightning/pytorch-lightning/pull/10674))
    * Add an utility to collect the states across processes ([#10639](https://github.com/PyTorchLightning/pytorch-lightning/pull/10639))
    * Add logic to reload the states across data loading components ([#10699](https://github.com/PyTorchLightning/pytorch-lightning/pull/10699))
    * Cleanup some fault tolerant utilities ([#10703](https://github.com/PyTorchLightning/pytorch-lightning/pull/10703))
    * Enable Fault Tolerant Manual Training ([#10707](https://github.com/PyTorchLightning/pytorch-lightning/pull/10707))
    * Broadcast the `_terminate_gracefully` to all processes and add support for DDP ([#10638](https://github.com/PyTorchLightning/pytorch-lightning/pull/10638))
- Added support for re-instantiation of custom (subclasses of) `DataLoaders` returned in the `*_dataloader()` methods, i.e., automatic replacement of samplers now works with custom types of `DataLoader` ([#10680](https://github.com/PyTorchLightning/pytorch-lightning/pull/10680))
- Added a function to validate if fault tolerant training is supported. ([#10465](https://github.com/PyTorchLightning/pytorch-lightning/pull/10465))
- Added a private callback to manage the creation and deletion of fault-tolerance checkpoints ([#11862](https://github.com/PyTorchLightning/pytorch-lightning/pull/11862))
- Show a better error message when a custom `DataLoader` implementation is not well implemented and we need to reconstruct it ([#10719](https://github.com/PyTorchLightning/pytorch-lightning/pull/10719))
- Show a better error message when frozen dataclass is used as a batch ([#10927](https://github.com/PyTorchLightning/pytorch-lightning/pull/10927))
- Save the `Loop`'s state by default in the checkpoint ([#10784](https://github.com/PyTorchLightning/pytorch-lightning/pull/10784))
- Added `Loop.replace` to easily switch one loop for another ([#10324](https://github.com/PyTorchLightning/pytorch-lightning/pull/10324))
- Added support for `--lr_scheduler=ReduceLROnPlateau` to the `LightningCLI` ([#10860](https://github.com/PyTorchLightning/pytorch-lightning/pull/10860))
- Added `LightningCLI.configure_optimizers` to override the `configure_optimizers` return value ([#10860](https://github.com/PyTorchLightning/pytorch-lightning/pull/10860))
- Added `LightningCLI(auto_registry)` flag to register all subclasses of the registerable components automatically ([#12108](https://github.com/PyTorchLightning/pytorch-lightning/pull/12108))
- Added a warning that shows when `max_epochs` in the `Trainer` is not set ([#10700](https://github.com/PyTorchLightning/pytorch-lightning/pull/10700))
- Added support for returning a single Callback from `LightningModule.configure_callbacks` without wrapping it into a list ([#11060](https://github.com/PyTorchLightning/pytorch-lightning/pull/11060))
- Added `console_kwargs` for `RichProgressBar` to initialize inner Console ([#10875](https://github.com/PyTorchLightning/pytorch-lightning/pull/10875))
- Added support for shorthand notation to instantiate loggers with the `LightningCLI` ([#11533](https://github.com/PyTorchLightning/pytorch-lightning/pull/11533))
- Added a `LOGGER_REGISTRY` instance to register custom loggers to the `LightningCLI` ([#11533](https://github.com/PyTorchLightning/pytorch-lightning/pull/11533))
- Added info message when the `Trainer` arguments `limit_*_batches`, `overfit_batches`, or `val_check_interval` are set to `1` or `1.0` ([#11950](https://github.com/PyTorchLightning/pytorch-lightning/pull/11950))
- Added a `PrecisionPlugin.teardown` method ([#10990](https://github.com/PyTorchLightning/pytorch-lightning/pull/10990))
- Added `LightningModule.lr_scheduler_step` ([#10249](https://github.com/PyTorchLightning/pytorch-lightning/pull/10249))
- Added support for no pre-fetching to `DataFetcher` ([#11606](https://github.com/PyTorchLightning/pytorch-lightning/pull/11606))
- Added support for optimizer step progress tracking with manual optimization ([#11848](https://github.com/PyTorchLightning/pytorch-lightning/pull/11848))
- Return the output of the `optimizer.step`. This can be useful for `LightningLite` users, manual optimization users, or users overriding `LightningModule.optimizer_step` ([#11711](https://github.com/PyTorchLightning/pytorch-lightning/pull/11711))
- Teardown the active loop and strategy on exception ([#11620](https://github.com/PyTorchLightning/pytorch-lightning/pull/11620))
- Added a `MisconfigurationException` if user provided `opt_idx` in scheduler config doesn't match with actual optimizer index of its respective optimizer ([#11247](https://github.com/PyTorchLightning/pytorch-lightning/pull/11247))
- Added a `loggers` property to `Trainer` which returns a list of loggers provided by the user ([#11683](https://github.com/PyTorchLightning/pytorch-lightning/pull/11683))
- Added a `loggers` property to `LightningModule` which retrieves the `loggers` property from `Trainer` ([#11683](https://github.com/PyTorchLightning/pytorch-lightning/pull/11683))
- Added support for DDP when using a `CombinedLoader` for the training data ([#11648](https://github.com/PyTorchLightning/pytorch-lightning/pull/11648))
- Added a warning when using `DistributedSampler` during validation/testing ([#11479](https://github.com/PyTorchLightning/pytorch-lightning/pull/11479))
- Added support for `Bagua` training strategy ([#11146](https://github.com/PyTorchLightning/pytorch-lightning/pull/11146))
- Added support for manually returning a `poptorch.DataLoader` in a `*_dataloader` hook ([#12116](https://github.com/PyTorchLightning/pytorch-lightning/pull/12116))
- Added `rank_zero` module to centralize utilities ([#11747](https://github.com/PyTorchLightning/pytorch-lightning/pull/11747))
- Added a `_Stateful` support for `LightningDataModule` ([#11637](https://github.com/PyTorchLightning/pytorch-lightning/pull/11637))
- Added `_Stateful` support for `PrecisionPlugin` ([#11638](https://github.com/PyTorchLightning/pytorch-lightning/pull/11638))
- Added `Accelerator.is_available` to check device availability ([#11797](https://github.com/PyTorchLightning/pytorch-lightning/pull/11797))
- Enabled static type-checking on the signature of `Trainer` ([#11888](https://github.com/PyTorchLightning/pytorch-lightning/pull/11888))
- Added utility functions for moving optimizers to devices ([#11758](https://github.com/PyTorchLightning/pytorch-lightning/pull/11758))
- Added a warning when saving an instance of `nn.Module` with `save_hyperparameters()` ([#12068](https://github.com/PyTorchLightning/pytorch-lightning/pull/12068))
- Added `estimated_stepping_batches` property to `Trainer` ([#11599](https://github.com/PyTorchLightning/pytorch-lightning/pull/11599))
- Added support for pluggable Accelerators ([#12030](https://github.com/PyTorchLightning/pytorch-lightning/pull/12030))
- Added profiling for `on_load_checkpoint`/`on_save_checkpoint` callback and LightningModule hooks ([#12149](https://github.com/PyTorchLightning/pytorch-lightning/pull/12149))
- Added `LayerSync` and `NativeSyncBatchNorm` plugins ([#11754](https://github.com/PyTorchLightning/pytorch-lightning/pull/11754))
- Added optional `storage_options` argument to `Trainer.save_checkpoint()` to pass to custom `CheckpointIO` implementations ([#11891](https://github.com/PyTorchLightning/pytorch-lightning/pull/11891))
- Added support to explicitly specify the process group backend for parallel strategies ([#11745](https://github.com/PyTorchLightning/pytorch-lightning/pull/11745))
- Added `device_ids` and `num_devices` property to `Trainer` ([#12151](https://github.com/PyTorchLightning/pytorch-lightning/pull/12151))
- Added `Callback.state_dict()` and `Callback.load_state_dict()` methods ([#12232](https://github.com/PyTorchLightning/pytorch-lightning/pull/12232))
- Added `AcceleratorRegistry` ([#12180](https://github.com/PyTorchLightning/pytorch-lightning/pull/12180))
- Added support for Habana Accelerator (HPU) ([#11808](https://github.com/PyTorchLightning/pytorch-lightning/pull/11808))
- Added support for dataclasses in `apply_to_collections` ([#11889](https://github.com/PyTorchLightning/pytorch-lightning/pull/11889))

</details>

<details><summary>Changed</summary>

- Drop PyTorch 1.7 support ([#12191](https://github.com/PyTorchLightning/pytorch-lightning/pull/12191)), ([#12432](https://github.com/PyTorchLightning/pytorch-lightning/pull/12432))
- Make `benchmark` flag optional and set its value based on the deterministic flag ([#11944](https://github.com/PyTorchLightning/pytorch-lightning/pull/11944))
- Implemented a new native and rich format in `_print_results` method of the `EvaluationLoop` ([#11332](https://github.com/PyTorchLightning/pytorch-lightning/pull/11332))
- Do not print an empty table at the end of the `EvaluationLoop` ([#12427](https://github.com/PyTorchLightning/pytorch-lightning/pull/12427))
- Set the `prog_bar` flag to False in `LightningModule.log_grad_norm` ([#11472](https://github.com/PyTorchLightning/pytorch-lightning/pull/11472))
- Raised exception in `init_dist_connection()` when torch distributed is not available ([#10418](https://github.com/PyTorchLightning/pytorch-lightning/pull/10418))
- The `monitor` argument in the `EarlyStopping` callback is no longer optional ([#10328](https://github.com/PyTorchLightning/pytorch-lightning/pull/10328))
- Do not fail if batch size could not be inferred for logging when using DeepSpeed ([#10438](https://github.com/PyTorchLightning/pytorch-lightning/pull/10438))
- Raised `MisconfigurationException` when `enable_progress_bar=False` and a progress bar instance has been passed in the callback list ([#10520](https://github.com/PyTorchLightning/pytorch-lightning/pull/10520))
- Moved `trainer.connectors.env_vars_connector._defaults_from_env_vars` to `utilities.argsparse._defaults_from_env_vars` ([#10501](https://github.com/PyTorchLightning/pytorch-lightning/pull/10501))
- Changes in `LightningCLI` required for the new major release of jsonargparse v4.0.0 ([#10426](https://github.com/PyTorchLightning/pytorch-lightning/pull/10426))
- Renamed `refresh_rate_per_second` parameter to `refresh_rate` for `RichProgressBar` signature ([#10497](https://github.com/PyTorchLightning/pytorch-lightning/pull/10497))
- Moved ownership of the `PrecisionPlugin` into `TrainingTypePlugin` and updated all references ([#10570](https://github.com/PyTorchLightning/pytorch-lightning/pull/10570))
- Fault Tolerant relies on `signal.SIGTERM` to gracefully exit instead of `signal.SIGUSR1` ([#10605](https://github.com/PyTorchLightning/pytorch-lightning/pull/10605))
- `Loop.restarting=...` now sets the value recursively for all subloops ([#11442](https://github.com/PyTorchLightning/pytorch-lightning/pull/11442))
- Raised an error if the `batch_size` cannot be inferred from the current batch if it contained a string or was a custom batch object ([#10541](https://github.com/PyTorchLightning/pytorch-lightning/pull/10541))
- The validation loop is now disabled when `overfit_batches > 0` is set in the Trainer ([#9709](https://github.com/PyTorchLightning/pytorch-lightning/pull/9709))
- Moved optimizer related logics from `Accelerator` to `TrainingTypePlugin` ([#10596](https://github.com/PyTorchLightning/pytorch-lightning/pull/10596))
- Moved ownership of the lightning optimizers from the `Trainer` to the `Strategy` ([#11444](https://github.com/PyTorchLightning/pytorch-lightning/pull/11444))
- Moved ownership of the data fetchers from the DataConnector to the Loops ([#11621](https://github.com/PyTorchLightning/pytorch-lightning/pull/11621))
- Moved `batch_to_device` method from `Accelerator` to `TrainingTypePlugin` ([#10649](https://github.com/PyTorchLightning/pytorch-lightning/pull/10649))
- The `DDPSpawnPlugin` no longer overrides the `post_dispatch` plugin hook ([#10034](https://github.com/PyTorchLightning/pytorch-lightning/pull/10034))
- Integrate the progress bar implementation with progress tracking ([#11213](https://github.com/PyTorchLightning/pytorch-lightning/pull/11213))
- The `LightningModule.{add_to_queue,get_from_queue}` hooks no longer get a `torch.multiprocessing.SimpleQueue` and instead receive a list based queue ([#10034](https://github.com/PyTorchLightning/pytorch-lightning/pull/10034))
- Changed `training_step`, `validation_step`, `test_step` and `predict_step` method signatures in `Accelerator` and updated input from caller side ([#10908](https://github.com/PyTorchLightning/pytorch-lightning/pull/10908))
- Changed the name of the temporary checkpoint that the `DDPSpawnPlugin` and related plugins save ([#10934](https://github.com/PyTorchLightning/pytorch-lightning/pull/10934))
- `LoggerCollection` returns only unique logger names and versions ([#10976](https://github.com/PyTorchLightning/pytorch-lightning/pull/10976))
- Redesigned process creation for spawn-based plugins (`DDPSpawnPlugin`, `TPUSpawnPlugin`, etc.) ([#10896](https://github.com/PyTorchLightning/pytorch-lightning/pull/10896))
    * All spawn-based plugins now spawn processes immediately upon calling `Trainer.{fit,validate,test,predict}`
    * The hooks/callbacks `prepare_data`, `setup`, `configure_sharded_model` and `teardown` now run under initialized process group for spawn-based plugins just like their non-spawn counterparts
    * Some configuration errors that were previously raised as `MisconfigurationException`s will now be raised as `ProcessRaisedException` (torch>=1.8) or as `Exception` (torch<1.8)
    * Removed the `TrainingTypePlugin.pre_dispatch()` method and merged it with `TrainingTypePlugin.setup()` ([#11137](https://github.com/PyTorchLightning/pytorch-lightning/pull/11137))
- Changed profiler to index and display the names of the hooks with a new pattern [<base class>]<class>.<hook name> ([#11026](https://github.com/PyTorchLightning/pytorch-lightning/pull/11026))
- Changed `batch_to_device` entry in profiling from stage-specific to generic, to match profiling of other hooks ([#11031](https://github.com/PyTorchLightning/pytorch-lightning/pull/11031))
- Changed the info message for finalizing ddp-spawn worker processes to a debug-level message ([#10864](https://github.com/PyTorchLightning/pytorch-lightning/pull/10864))
- Removed duplicated file extension when uploading model checkpoints with `NeptuneLogger` ([#11015](https://github.com/PyTorchLightning/pytorch-lightning/pull/11015))
- Removed `__getstate__` and `__setstate__` of `RichProgressBar` ([#11100](https://github.com/PyTorchLightning/pytorch-lightning/pull/11100))
- The `DDPPlugin` and `DDPSpawnPlugin` and their subclasses now remove the `SyncBatchNorm` wrappers in `teardown()` to enable proper support at inference after fitting ([#11078](https://github.com/PyTorchLightning/pytorch-lightning/pull/11078))
- Moved ownership of the `Accelerator` instance to the `TrainingTypePlugin`; all training-type plugins now take an optional parameter `accelerator` ([#11022](https://github.com/PyTorchLightning/pytorch-lightning/pull/11022))
- Renamed the `TrainingTypePlugin` to `Strategy` ([#11120](https://github.com/PyTorchLightning/pytorch-lightning/pull/11120))
    * Renamed the `ParallelPlugin` to `ParallelStrategy` ([#11123](https://github.com/PyTorchLightning/pytorch-lightning/pull/11123))
    * Renamed the `DataParallelPlugin` to `DataParallelStrategy` ([#11183](https://github.com/PyTorchLightning/pytorch-lightning/pull/11183))
    * Renamed the `DDPPlugin` to `DDPStrategy` ([#11142](https://github.com/PyTorchLightning/pytorch-lightning/pull/11142))
    * Renamed the `DDP2Plugin` to `DDP2Strategy` ([#11185](https://github.com/PyTorchLightning/pytorch-lightning/pull/11185))
    * Renamed the `DDPShardedPlugin` to `DDPShardedStrategy` ([#11186](https://github.com/PyTorchLightning/pytorch-lightning/pull/11186))
    * Renamed the `DDPFullyShardedPlugin` to `DDPFullyShardedStrategy` ([#11143](https://github.com/PyTorchLightning/pytorch-lightning/pull/11143))
    * Renamed the `DDPSpawnPlugin` to `DDPSpawnStrategy` ([#11145](https://github.com/PyTorchLightning/pytorch-lightning/pull/11145))
    * Renamed the `DDPSpawnShardedPlugin` to `DDPSpawnShardedStrategy` ([#11210](https://github.com/PyTorchLightning/pytorch-lightning/pull/11210))
    * Renamed the `DeepSpeedPlugin` to `DeepSpeedStrategy` ([#11194](https://github.com/PyTorchLightning/pytorch-lightning/pull/11194))
    * Renamed the `HorovodPlugin` to `HorovodStrategy` ([#11195](https://github.com/PyTorchLightning/pytorch-lightning/pull/11195))
    * Renamed the `TPUSpawnPlugin` to `TPUSpawnStrategy` ([#11190](https://github.com/PyTorchLightning/pytorch-lightning/pull/11190))
    * Renamed the `IPUPlugin` to `IPUStrategy` ([#11193](https://github.com/PyTorchLightning/pytorch-lightning/pull/11193))
    * Renamed the `SingleDevicePlugin` to `SingleDeviceStrategy` ([#11182](https://github.com/PyTorchLightning/pytorch-lightning/pull/11182))
    * Renamed the `SingleTPUPlugin` to `SingleTPUStrategy` ([#11182](https://github.com/PyTorchLightning/pytorch-lightning/pull/11182))
    * Renamed the `TrainingTypePluginsRegistry` to `StrategyRegistry` ([#11233](https://github.com/PyTorchLightning/pytorch-lightning/pull/11233))
- Marked the `ResultCollection`, `ResultMetric`, and `ResultMetricCollection` classes as protected ([#11130](https://github.com/PyTorchLightning/pytorch-lightning/pull/11130))
- Marked `trainer.checkpoint_connector` as protected ([#11550](https://github.com/PyTorchLightning/pytorch-lightning/pull/11550))
- The epoch start/end hooks are now called by the `FitLoop` instead of the `TrainingEpochLoop` ([#11201](https://github.com/PyTorchLightning/pytorch-lightning/pull/11201))
- DeepSpeed does not require lightning module zero 3 partitioning ([#10655](https://github.com/PyTorchLightning/pytorch-lightning/pull/10655))
- Moved `Strategy` classes to the `strategies` directory ([#11226](https://github.com/PyTorchLightning/pytorch-lightning/pull/11226))
- Renamed `training_type_plugin` file to `strategy` ([#11239](https://github.com/PyTorchLightning/pytorch-lightning/pull/11239))
- Changed `DeviceStatsMonitor` to group metrics based on the logger's `group_separator` ([#11254](https://github.com/PyTorchLightning/pytorch-lightning/pull/11254))
- Raised `UserWarning` if evaluation is triggered with `best` ckpt and trainer is configured with multiple checkpoint callbacks ([#11274](https://github.com/PyTorchLightning/pytorch-lightning/pull/11274))
- `Trainer.logged_metrics` now always contains scalar tensors, even when a Python scalar was logged ([#11270](https://github.com/PyTorchLightning/pytorch-lightning/pull/11270))
- The tuner now uses the checkpoint connector to copy and restore its state ([#11518](https://github.com/PyTorchLightning/pytorch-lightning/pull/11518))
- Changed `MisconfigurationException` to `ModuleNotFoundError` when `rich` isn't available ([#11360](https://github.com/PyTorchLightning/pytorch-lightning/pull/11360))
- The `trainer.current_epoch` value is now increased by 1 during and after `on_train_end` ([#8578](https://github.com/PyTorchLightning/pytorch-lightning/pull/8578))
- The `trainer.global_step` value now accounts for multiple optimizers and TBPTT splits ([#11805](https://github.com/PyTorchLightning/pytorch-lightning/pull/11805))
- The `trainer.global_step` value is now increased right after the `optimizer.step()` call which will impact users who access it during an intra-training validation hook ([#11805](https://github.com/PyTorchLightning/pytorch-lightning/pull/11805))
- The filename of checkpoints created with `ModelCheckpoint(filename='{step}')` is different compared to previous versions. A checkpoint saved after 1 step will be named `step=1.ckpt` instead of `step=0.ckpt` ([#11805](https://github.com/PyTorchLightning/pytorch-lightning/pull/11805))
- Inherit from `ABC` for `Accelerator`: Users need to implement `auto_device_count` ([#11521](https://github.com/PyTorchLightning/pytorch-lightning/pull/11521))
- Changed `parallel_devices` property in `ParallelStrategy` to be lazy initialized ([#11572](https://github.com/PyTorchLightning/pytorch-lightning/pull/11572))
- Updated `TQDMProgressBar` to run a separate progress bar for each eval dataloader ([#11657](https://github.com/PyTorchLightning/pytorch-lightning/pull/11657))
- Sorted `SimpleProfiler(extended=False)` summary based on mean duration for each hook ([#11671](https://github.com/PyTorchLightning/pytorch-lightning/pull/11671))
- Avoid enforcing `shuffle=False` for eval dataloaders ([#11575](https://github.com/PyTorchLightning/pytorch-lightning/pull/11575))
- When using DP (data-parallel), Lightning will no longer automatically reduce all tensors returned in training_step; it will only reduce the loss unless `training_step_end` is overridden ([#11594](https://github.com/PyTorchLightning/pytorch-lightning/pull/11594))
- When using DP (data-parallel), the `training_epoch_end` hook will no longer receive reduced outputs from `training_step` and instead get the full tensor of results from all GPUs ([#11594](https://github.com/PyTorchLightning/pytorch-lightning/pull/11594))
- Changed default logger name to `lightning_logs` for consistency ([#11762](https://github.com/PyTorchLightning/pytorch-lightning/pull/11762))
- Rewrote `accelerator_connector` ([#11448](https://github.com/PyTorchLightning/pytorch-lightning/pull/11448))
- When manual optimization is used with DDP, we no longer force `find_unused_parameters=True` ([#12425](https://github.com/PyTorchLightning/pytorch-lightning/pull/12425))
- Disable loading dataloades if corresponding `limit_batches=0` ([#11576](https://github.com/PyTorchLightning/pytorch-lightning/pull/11576))
- Removed `is_global_zero` check in `training_epoch_loop` before `logger.save`. If you have a custom logger that implements `save` the Trainer will now call `save` on all ranks by default. To change this behavior add `@rank_zero_only` to your `save` implementation ([#12134](https://github.com/PyTorchLightning/pytorch-lightning/pull/12134))
- Disabled tuner with distributed strategies ([#12179](https://github.com/PyTorchLightning/pytorch-lightning/pull/12179))
- Marked `trainer.logger_connector` as protected ([#12195](https://github.com/PyTorchLightning/pytorch-lightning/pull/12195))
- Move `Strategy.process_dataloader` function call from `fit/evaluation/predict_loop.py` to `data_connector.py` ([#12251](https://github.com/PyTorchLightning/pytorch-lightning/pull/12251))
- `ModelCheckpoint(save_last=True, every_n_epochs=N)` now saves a "last" checkpoint every epoch (disregarding `every_n_epochs`) instead of only once at the end of training ([#12418](https://github.com/PyTorchLightning/pytorch-lightning/pull/12418))
- The strategies that support `sync_batchnorm` now only apply it when fitting ([#11919](https://github.com/PyTorchLightning/pytorch-lightning/pull/11919))
- Avoided fallback on CPU if no devices are provided for other accelerators ([#12410](https://github.com/PyTorchLightning/pytorch-lightning/pull/12410))
- Modified `supporters.py` so that in the accumulator element (for loss) is created directly on the device ([#12430](https://github.com/PyTorchLightning/pytorch-lightning/pull/12430))
- Removed `EarlyStopping.on_save_checkpoint` and `EarlyStopping.on_load_checkpoint` in favor of `EarlyStopping.state_dict` and `EarlyStopping.load_state_dict` ([#11887](https://github.com/PyTorchLightning/pytorch-lightning/pull/11887))
- Removed `BaseFinetuning.on_save_checkpoint` and `BaseFinetuning.on_load_checkpoint` in favor of `BaseFinetuning.state_dict` and `BaseFinetuning.load_state_dict` ([#11887](https://github.com/PyTorchLightning/pytorch-lightning/pull/11887))
- Removed `BackboneFinetuning.on_save_checkpoint` and `BackboneFinetuning.on_load_checkpoint` in favor of `BackboneFinetuning.state_dict` and `BackboneFinetuning.load_state_dict` ([#11887](https://github.com/PyTorchLightning/pytorch-lightning/pull/11887))
- Removed `ModelCheckpoint.on_save_checkpoint` and `ModelCheckpoint.on_load_checkpoint` in favor of `ModelCheckpoint.state_dict` and `ModelCheckpoint.load_state_dict` ([#11887](https://github.com/PyTorchLightning/pytorch-lightning/pull/11887))
- Removed `Timer.on_save_checkpoint` and `Timer.on_load_checkpoint` in favor of `Timer.state_dict` and `Timer.load_state_dict` ([#11887](https://github.com/PyTorchLightning/pytorch-lightning/pull/11887))
- Replaced PostLocalSGDOptimizer with a dedicated model averaging component ([#12378](https://github.com/PyTorchLightning/pytorch-lightning/pull/12378))

</details>

<details><summary>Deprecated</summary>

- Deprecated `training_type_plugin` property in favor of `strategy` in `Trainer` and updated the references ([#11141](https://github.com/PyTorchLightning/pytorch-lightning/pull/11141))
- Deprecated `Trainer.{validated,tested,predicted}_ckpt_path` and replaced with read-only property `Trainer.ckpt_path` set when checkpoints loaded via `Trainer.{fit,validate,test,predict}` ([#11696](https://github.com/PyTorchLightning/pytorch-lightning/pull/11696))
- Deprecated `ClusterEnvironment.master_{address,port}` in favor of `ClusterEnvironment.main_{address,port}` ([#10103](https://github.com/PyTorchLightning/pytorch-lightning/pull/10103))
- Deprecated `DistributedType` in favor of `_StrategyType` ([#10505](https://github.com/PyTorchLightning/pytorch-lightning/pull/10505))
- Deprecated the `precision_plugin` constructor argument from `Accelerator` ([#10570](https://github.com/PyTorchLightning/pytorch-lightning/pull/10570))
- Deprecated `DeviceType` in favor of `_AcceleratorType` ([#10503](https://github.com/PyTorchLightning/pytorch-lightning/pull/10503))
- Deprecated the property `Trainer.slurm_job_id` in favor of the new `SLURMEnvironment.job_id()` method ([#10622](https://github.com/PyTorchLightning/pytorch-lightning/pull/10622))
- Deprecated the access to the attribute `IndexBatchSamplerWrapper.batch_indices` in favor of `IndexBatchSamplerWrapper.seen_batch_indices` ([#10870](https://github.com/PyTorchLightning/pytorch-lightning/pull/10870))
- Deprecated `on_init_start` and `on_init_end` callback hooks ([#10940](https://github.com/PyTorchLightning/pytorch-lightning/pull/10940))
- Deprecated `Trainer.call_hook` in favor of `Trainer._call_callback_hooks`, `Trainer._call_lightning_module_hook`, `Trainer._call_ttp_hook`, and `Trainer._call_accelerator_hook` ([#10979](https://github.com/PyTorchLightning/pytorch-lightning/pull/10979))
- Deprecated `TrainingTypePlugin.post_dispatch` in favor of `TrainingTypePlugin.teardown` ([#10939](https://github.com/PyTorchLightning/pytorch-lightning/pull/10939))
- Deprecated `ModelIO.on_hpc_{save/load}` in favor of `CheckpointHooks.on_{save/load}_checkpoint` ([#10911](https://github.com/PyTorchLightning/pytorch-lightning/pull/10911))
- Deprecated `Trainer.run_stage` in favor of `Trainer.{fit,validate,test,predict}` ([#11000](https://github.com/PyTorchLightning/pytorch-lightning/pull/11000))
- Deprecated `Trainer.lr_schedulers` in favor of `Trainer.lr_scheduler_configs` which returns a list of dataclasses instead of dictionaries ([#11443](https://github.com/PyTorchLightning/pytorch-lightning/pull/11443))
- Deprecated `Trainer.verbose_evaluate` in favor of `EvaluationLoop(verbose=...)` ([#10931](https://github.com/PyTorchLightning/pytorch-lightning/pull/10931))
- Deprecated `Trainer.should_rank_save_checkpoint` Trainer property ([#11068](https://github.com/PyTorchLightning/pytorch-lightning/pull/11068))
- Deprecated `Trainer.lightning_optimizers` ([#11444](https://github.com/PyTorchLightning/pytorch-lightning/pull/11444))
- Deprecated `TrainerOptimizersMixin` and moved functionality to `core/optimizer.py`([#11155](https://github.com/PyTorchLightning/pytorch-lightning/pull/11155))
- Deprecated the `on_train_batch_end(outputs)` format when multiple optimizers are used and TBPTT is enabled ([#12182](https://github.com/PyTorchLightning/pytorch-lightning/pull/12182))
- Deprecated the `training_epoch_end(outputs)` format when multiple optimizers are used and TBPTT is enabled ([#12182](https://github.com/PyTorchLightning/pytorch-lightning/pull/12182))
- Deprecated `TrainerCallbackHookMixin` ([#11148](https://github.com/PyTorchLightning/pytorch-lightning/pull/11148))
- Deprecated `TrainerDataLoadingMixin` and moved functionality to `Trainer` and `DataConnector` ([#11282](https://github.com/PyTorchLightning/pytorch-lightning/pull/11282))
- Deprecated function `pytorch_lightning.callbacks.device_stats_monitor.prefix_metric_keys` ([#11254](https://github.com/PyTorchLightning/pytorch-lightning/pull/11254))
- Deprecated `Callback.on_epoch_start` hook in favour of `Callback.on_{train/val/test}_epoch_start` ([#11578](https://github.com/PyTorchLightning/pytorch-lightning/pull/11578))
- Deprecated `Callback.on_epoch_end` hook in favour of `Callback.on_{train/val/test}_epoch_end` ([#11578](https://github.com/PyTorchLightning/pytorch-lightning/pull/11578))
- Deprecated `LightningModule.on_epoch_start` hook in favor of `LightningModule.on_{train/val/test}_epoch_start` ([#11578](https://github.com/PyTorchLightning/pytorch-lightning/pull/11578))
- Deprecated `LightningModule.on_epoch_end` hook in favor of `LightningModule.on_{train/val/test}_epoch_end` ([#11578](https://github.com/PyTorchLightning/pytorch-lightning/pull/11578))
- Deprecated `on_before_accelerator_backend_setup` callback hook in favour of `setup` ([#11568](https://github.com/PyTorchLightning/pytorch-lightning/pull/11568))
- Deprecated `on_batch_start` and `on_batch_end` callback hooks in favor of `on_train_batch_start` and `on_train_batch_end` ([#11577](https://github.com/PyTorchLightning/pytorch-lightning/pull/11577))
- Deprecated `on_configure_sharded_model` callback hook in favor of `setup` ([#11627](https://github.com/PyTorchLightning/pytorch-lightning/pull/11627))
- Deprecated `pytorch_lightning.utilities.distributed.rank_zero_only` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_only` ([#11747](https://github.com/PyTorchLightning/pytorch-lightning/pull/11747))
- Deprecated `pytorch_lightning.utilities.distributed.rank_zero_debug` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_debug` ([#11747](https://github.com/PyTorchLightning/pytorch-lightning/pull/11747))
- Deprecated `pytorch_lightning.utilities.distributed.rank_zero_info` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_info` ([#11747](https://github.com/PyTorchLightning/pytorch-lightning/pull/11747))
- Deprecated `pytorch_lightning.utilities.warnings.rank_zero_warn` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_warn` ([#11747](https://github.com/PyTorchLightning/pytorch-lightning/pull/11747))
- Deprecated `pytorch_lightning.utilities.warnings.rank_zero_deprecation` in favor of `pytorch_lightning.utilities.rank_zero.rank_zero_deprecation` ([#11747](https://github.com/PyTorchLightning/pytorch-lightning/pull/11747))
- Deprecated `pytorch_lightning.utilities.warnings.LightningDeprecationWarning` in favor of `pytorch_lightning.utilities.rank_zero.LightningDeprecationWarning`
- Deprecated `on_pretrain_routine_start` and `on_pretrain_routine_end` callback hooks in favor of `on_fit_start` ([#11794](https://github.com/PyTorchLightning/pytorch-lightning/pull/11794))
- Deprecated `LightningModule.on_pretrain_routine_start` and `LightningModule.on_pretrain_routine_end` hooks in favor of `on_fit_start` ([#12122](https://github.com/PyTorchLightning/pytorch-lightning/pull/12122))
- Deprecated `agg_key_funcs` and `agg_default_func` parameters from `LightningLoggerBase` ([#11871](https://github.com/PyTorchLightning/pytorch-lightning/pull/11871))
- Deprecated `LightningLoggerBase.update_agg_funcs` ([#11871](https://github.com/PyTorchLightning/pytorch-lightning/pull/11871))
- Deprecated `LightningLoggerBase.agg_and_log_metrics` in favor of `LightningLoggerBase.log_metrics` ([#11832](https://github.com/PyTorchLightning/pytorch-lightning/pull/11832))
- Deprecated passing `weights_save_path` to the `Trainer` constructor in favor of adding the `ModelCheckpoint` callback with `dirpath` directly to the list of callbacks ([#12084](https://github.com/PyTorchLightning/pytorch-lightning/pull/12084))
- Deprecated `pytorch_lightning.profiler.AbstractProfiler` in favor of `pytorch_lightning.profiler.Profiler` ([#12106](https://github.com/PyTorchLightning/pytorch-lightning/pull/12106))
- Deprecated `pytorch_lightning.profiler.BaseProfiler` in favor of `pytorch_lightning.profiler.Profiler` ([#12150](https://github.com/PyTorchLightning/pytorch-lightning/pull/12150))
- Deprecated `BaseProfiler.profile_iterable` ([#12102](https://github.com/PyTorchLightning/pytorch-lightning/pull/12102))
- Deprecated `LoggerCollection` in favor of `trainer.loggers` ([#12147](https://github.com/PyTorchLightning/pytorch-lightning/pull/12147))
- Deprecated `PrecisionPlugin.on_{save,load}_checkpoint` in favor of `PrecisionPlugin.{state_dict,load_state_dict}` ([#11978](https://github.com/PyTorchLightning/pytorch-lightning/pull/11978))
- Deprecated `LightningDataModule.on_save/load_checkpoint` in favor of `state_dict/load_state_dict` ([#11893](https://github.com/PyTorchLightning/pytorch-lightning/pull/11893))
- Deprecated `Trainer.use_amp` in favor of `Trainer.amp_backend` ([#12312](https://github.com/PyTorchLightning/pytorch-lightning/pull/12312))
- Deprecated `LightingModule.use_amp` in favor of `Trainer.amp_backend` ([#12315](https://github.com/PyTorchLightning/pytorch-lightning/pull/12315))
- Deprecated specifying the process group backend through the environment variable `PL_TORCH_DISTRIBUTED_BACKEND` ([#11745](https://github.com/PyTorchLightning/pytorch-lightning/pull/11745))
- Deprecated `ParallelPlugin.torch_distributed_backend` in favor of `DDPStrategy.process_group_backend` property ([#11745](https://github.com/PyTorchLightning/pytorch-lightning/pull/11745))
- Deprecated `ModelCheckpoint.save_checkpoint` in favor of `Trainer.save_checkpoint` ([#12456](https://github.com/PyTorchLightning/pytorch-lightning/pull/12456))
- Deprecated `Trainer.devices` in favor of `Trainer.num_devices` and `Trainer.device_ids` ([#12151](https://github.com/PyTorchLightning/pytorch-lightning/pull/12151))
- Deprecated `Trainer.root_gpu` in favor of `Trainer.strategy.root_device.index` when GPU is used ([#12262](https://github.com/PyTorchLightning/pytorch-lightning/pull/12262))
- Deprecated `Trainer.num_gpus` in favor of `Trainer.num_devices` when GPU is used ([#12384](https://github.com/PyTorchLightning/pytorch-lightning/pull/12384))
- Deprecated `Trainer.ipus` in favor of `Trainer.num_devices` when IPU is used ([#12386](https://github.com/PyTorchLightning/pytorch-lightning/pull/12386))
- Deprecated `Trainer.num_processes` in favor of `Trainer.num_devices` ([#12388](https://github.com/PyTorchLightning/pytorch-lightning/pull/12388))
- Deprecated `Trainer.data_parallel_device_ids` in favor of `Trainer.device_ids` ([#12072](https://github.com/PyTorchLightning/pytorch-lightning/pull/12072))
- Deprecated returning state from `Callback.on_save_checkpoint` in favor of returning state in `Callback.state_dict` for checkpointing ([#11887](https://github.com/PyTorchLightning/pytorch-lightning/pull/11887))
- Deprecated passing only the callback state to `Callback.on_load_checkpoint(callback_state)` in favor of passing the callback state to `Callback.load_state_dict` and in 1.8, passing the entire checkpoint dictionary to `Callback.on_load_checkpoint(checkpoint)` ([#11887](https://github.com/PyTorchLightning/pytorch-lightning/pull/11887))
- Deprecated `Trainer.gpus` in favor of `Trainer.device_ids` or `Trainer.num_devices` ([#12436](https://github.com/PyTorchLightning/pytorch-lightning/pull/12436))
- Deprecated `Trainer.tpu_cores` in favor of `Trainer.num_devices` ([#12437](https://github.com/PyTorchLightning/pytorch-lightning/pull/12437))

</details>

<details><summary>Removed</summary>

- Removed deprecated parameter `method` in `pytorch_lightning.utilities.model_helpers.is_overridden` ([#10507](https://github.com/PyTorchLightning/pytorch-lightning/pull/10507))
- Remove deprecated method `ClusterEnvironment.creates_children` ([#10339](https://github.com/PyTorchLightning/pytorch-lightning/pull/10339))
- Removed deprecated `TrainerModelHooksMixin.is_function_implemented` and `TrainerModelHooksMixin.has_arg` ([#10322](https://github.com/PyTorchLightning/pytorch-lightning/pull/10322))
- Removed deprecated `pytorch_lightning.utilities.device_dtype_mixin.DeviceDtypeModuleMixin` in favor of `pytorch_lightning.core.mixins.device_dtype_mixin.DeviceDtypeModuleMixin` ([#10442](https://github.com/PyTorchLightning/pytorch-lightning/pull/10442))
- Removed deprecated `LightningModule.loaded_optimizer_states_dict` property ([#10346](https://github.com/PyTorchLightning/pytorch-lightning/pull/10346))
- Removed deprecated `Trainer.fit(train_dataloader=)`, `Trainer.validate(val_dataloaders=)`, and `Trainer.test(test_dataloader=)` ([#10325](https://github.com/PyTorchLightning/pytorch-lightning/pull/10325))
- Removed deprecated `has_prepared_data`, `has_setup_fit`, `has_setup_validate`, `has_setup_test`, `has_setup_predict`, `has_teardown_fit`, `has_teardown_validate`, `has_teardown_test` and `has_teardown_predict` datamodule lifecycle properties  ([#10350](https://github.com/PyTorchLightning/pytorch-lightning/pull/10350))
- Removed deprecated `every_n_val_epochs` parameter of ModelCheckpoint ([#10366](https://github.com/PyTorchLightning/pytorch-lightning/pull/10366))
- Removed deprecated `import pytorch_lightning.profiler.profilers` in favor of `import pytorch_lightning.profiler` ([#10443](https://github.com/PyTorchLightning/pytorch-lightning/pull/10443))
- Removed deprecated property `configure_slurm_dpp` from accelerator connector ([#10370](https://github.com/PyTorchLightning/pytorch-lightning/pull/10370))
- Removed deprecated arguments `num_nodes` and `sync_batchnorm` from `DDPPlugin`, `DDPSpawnPlugin`, `DeepSpeedPlugin` ([#10357](https://github.com/PyTorchLightning/pytorch-lightning/pull/10357))
- Removed deprecated property `is_slurm_managing_tasks` from AcceleratorConnector ([#10353](https://github.com/PyTorchLightning/pytorch-lightning/pull/10353))
- Removed deprecated `LightningModule.log(tbptt_reduce_fx, tbptt_reduce_token, sync_dist_op)` ([#10423](https://github.com/PyTorchLightning/pytorch-lightning/pull/10423))
- Removed deprecated `Plugin.task_idx` ([#10441](https://github.com/PyTorchLightning/pytorch-lightning/pull/10441))
- Removed deprecated method `master_params` from PrecisionPlugin ([#10372](https://github.com/PyTorchLightning/pytorch-lightning/pull/10372))
- Removed the automatic detachment of "extras" returned from `training_step`. For example, `return {'loss': ..., 'foo': foo.detach()}` will now be necessary if `foo` has gradients which you do not want to store ([#10424](https://github.com/PyTorchLightning/pytorch-lightning/pull/10424))
- Removed deprecated passthrough methods and properties from `Accelerator` base class:
  * ([#10403](https://github.com/PyTorchLightning/pytorch-lightning/pull/10403))
  * ([#10448](https://github.com/PyTorchLightning/pytorch-lightning/pull/10448))
- Removed deprecated signature for `transfer_batch_to_device` hook. The new argument `dataloader_idx` is now required ([#10480](https://github.com/PyTorchLightning/pytorch-lightning/pull/10480))
- Removed deprecated `utilities.distributed.rank_zero_{warn/deprecation}` ([#10451](https://github.com/PyTorchLightning/pytorch-lightning/pull/10451))
- Removed deprecated `mode` argument from `ModelSummary` class ([#10449](https://github.com/PyTorchLightning/pytorch-lightning/pull/10449))
- Removed deprecated `Trainer.train_loop` property in favor of `Trainer.fit_loop` ([#10482](https://github.com/PyTorchLightning/pytorch-lightning/pull/10482))
- Removed deprecated `Trainer.train_loop` property in favor of `Trainer.fit_loop` ([#10482](https://github.com/PyTorchLightning/pytorch-lightning/pull/10482))
- Removed deprecated `disable_validation` property from Trainer ([#10450](https://github.com/PyTorchLightning/pytorch-lightning/pull/10450))
- Removed deprecated `CheckpointConnector.hpc_load` property in favor of `CheckpointConnector.restore` ([#10525](https://github.com/PyTorchLightning/pytorch-lightning/pull/10525))
- Removed deprecated `reload_dataloaders_every_epoch` from `Trainer` in favour of `reload_dataloaders_every_n_epochs` ([#10481](https://github.com/PyTorchLightning/pytorch-lightning/pull/10481))
- Removed the `precision_plugin` attribute from `Accelerator` in favor of its equivalent attribute `precision_plugin` in the `TrainingTypePlugin` ([#10570](https://github.com/PyTorchLightning/pytorch-lightning/pull/10570))
- Removed `DeepSpeedPlugin.{precision,amp_type,amp_level}` properties ([#10657](https://github.com/PyTorchLightning/pytorch-lightning/pull/10657))
- Removed patching of `on_before_batch_transfer`, `transfer_batch_to_device` and `on_after_batch_transfer` hooks in `LightningModule` ([#10603](https://github.com/PyTorchLightning/pytorch-lightning/pull/10603))
- Removed argument `return_result` from the `DDPSpawnPlugin.spawn()` method ([#10867](https://github.com/PyTorchLightning/pytorch-lightning/pull/10867))
- Removed the property `TrainingTypePlugin.results` and corresponding properties in subclasses ([#10034](https://github.com/PyTorchLightning/pytorch-lightning/pull/10034))
- Removed the `mp_queue` attribute from `DDPSpawnPlugin` and `TPUSpawnPlugin` ([#10034](https://github.com/PyTorchLightning/pytorch-lightning/pull/10034))
- Removed unnecessary `_move_optimizer_state` method overrides from `TPUSpawnPlugin` and `SingleTPUPlugin` ([#10849](https://github.com/PyTorchLightning/pytorch-lightning/pull/10849))
- Removed `should_rank_save_checkpoint` property from `TrainingTypePlugin` ([#11070](https://github.com/PyTorchLightning/pytorch-lightning/pull/11070))
- Removed `model_sharded_context` method from `Accelerator` ([#10886](https://github.com/PyTorchLightning/pytorch-lightning/pull/10886))
- Removed method `pre_dispatch` from the `PrecisionPlugin` ([#10887](https://github.com/PyTorchLightning/pytorch-lightning/pull/10887))
- Removed method `setup_optimizers_in_pre_dispatch` from the `strategies` and achieve the same logic in `setup` and `pre_dispatch` methods ([#10906](https://github.com/PyTorchLightning/pytorch-lightning/pull/10906))
- Removed methods `pre_dispatch`, `dispatch` and `post_dispatch` from the `Accelerator` ([#10885](https://github.com/PyTorchLightning/pytorch-lightning/pull/10885))
- Removed method `training_step`, `test_step`, `validation_step` and `predict_step` from the `Accelerator` ([#10890](https://github.com/PyTorchLightning/pytorch-lightning/pull/10890))
- Removed `TrainingTypePlugin.start_{training,evaluating,predicting}` hooks and the same in all subclasses ([#10989](https://github.com/PyTorchLightning/pytorch-lightning/pull/10989), [#10896](https://github.com/PyTorchLightning/pytorch-lightning/pull/10896))
- Removed `Accelerator.on_train_start` ([#10999](https://github.com/PyTorchLightning/pytorch-lightning/pull/10999))
- Removed support for Python 3.6 ([#11117](https://github.com/PyTorchLightning/pytorch-lightning/pull/11117))
- Removed `Strategy.init_optimizers` in favor of `Strategy.setup_optimizers` ([#11236](https://github.com/PyTorchLightning/pytorch-lightning/pull/11236))
- Removed `profile("training_step_and_backward")` in `Closure` class since we already profile calls `training_step` and `backward` ([#11222](https://github.com/PyTorchLightning/pytorch-lightning/pull/11222))
- Removed `Strategy.optimizer_zero_grad` ([#11246](https://github.com/PyTorchLightning/pytorch-lightning/pull/11246))
- Removed `Strategy.on_gpu` ([#11537](https://github.com/PyTorchLightning/pytorch-lightning/pull/11537))
- Removed `Strategy.on_tpu` property ([#11536](https://github.com/PyTorchLightning/pytorch-lightning/pull/11536))
- Removed the abstract property `LightningLoggerBase.experiment` ([#11603](https://github.com/PyTorchLightning/pytorch-lightning/pull/11603))
- Removed `FitLoop.current_epoch` getter and setter ([#11562](https://github.com/PyTorchLightning/pytorch-lightning/pull/11562))
- Removed access to `_short_id` in `NeptuneLogger` ([#11517](https://github.com/PyTorchLightning/pytorch-lightning/pull/11517))
- Removed `log_text` and `log_image` from the `LightningLoggerBase` API ([#11857](https://github.com/PyTorchLightning/pytorch-lightning/pull/11857))
- Removed calls to `profile("model_forward")` in favor of profiling `training_step` ([#12032](https://github.com/PyTorchLightning/pytorch-lightning/pull/12032))
- Removed `get_mp_spawn_kwargs` from `DDPSpawnStrategy` and `TPUSpawnStrategy` in favor of configuration in the `_SpawnLauncher` ([#11966](https://github.com/PyTorchLightning/pytorch-lightning/pull/11966))
- Removed `_aggregate_metrics`, `_reduce_agg_metrics`, and `_finalize_agg_metrics` from `LightningLoggerBase` ([#12053](https://github.com/PyTorchLightning/pytorch-lightning/pull/12053))
- Removed the `AcceleratorConnector.device_type` property ([#12081](https://github.com/PyTorchLightning/pytorch-lightning/pull/12081))
- Removed `AcceleratorConnector.num_nodes` ([#12107](https://github.com/PyTorchLightning/pytorch-lightning/pull/12107))
- Removed `AcceleratorConnector.has_ipu` property ([#12111](https://github.com/PyTorchLightning/pytorch-lightning/pull/12111))
- Removed `AcceleratorConnector.use_ipu` property ([#12110](https://github.com/PyTorchLightning/pytorch-lightning/pull/12110))
- Removed `AcceleratorConnector.has_tpu` property ([#12109](https://github.com/PyTorchLightning/pytorch-lightning/pull/12109))
- Removed `AcceleratorConnector.use_dp` property ([#12112](https://github.com/PyTorchLightning/pytorch-lightning/pull/12112))
- Removed `configure_sync_batchnorm` from `ParallelStrategy` and all other strategies that inherit from it ([#11754](https://github.com/PyTorchLightning/pytorch-lightning/pull/11754))
- Removed public attribute `sync_batchnorm` from strategies ([#11754](https://github.com/PyTorchLightning/pytorch-lightning/pull/11754))
- Removed `AcceleratorConnector.root_gpu` property ([#12262](https://github.com/PyTorchLightning/pytorch-lightning/pull/12262))
- Removed `AcceleratorConnector.tpu_id` property ([#12387](https://github.com/PyTorchLightning/pytorch-lightning/pull/12387))
- Removed `AcceleratorConnector.num_gpus` property ([#12384](https://github.com/PyTorchLightning/pytorch-lightning/pull/12384))
- Removed `AcceleratorConnector.num_ipus` property ([#12386](https://github.com/PyTorchLightning/pytorch-lightning/pull/12386))
- Removed `AcceleratorConnector.num_processes` property ([#12388](https://github.com/PyTorchLightning/pytorch-lightning/pull/12388))
- Removed `AcceleratorConnector.parallel_device_ids` property ([#12072](https://github.com/PyTorchLightning/pytorch-lightning/pull/12072))
- Removed `AcceleratorConnector.devices` property ([#12435](https://github.com/PyTorchLightning/pytorch-lightning/pull/12435))
- Removed `AcceleratorConnector.parallel_devices` property ([#12075](https://github.com/PyTorchLightning/pytorch-lightning/pull/12075))
- Removed `AcceleratorConnector.tpu_cores` property ([#12437](https://github.com/PyTorchLightning/pytorch-lightning/pull/12437))

</details>

<details><summary>Fixed</summary>

- Fixed an issue where `ModelCheckpoint` could delete last checkpoint from the old directory when `dirpath` has changed during resumed training ([#12225](https://github.com/PyTorchLightning/pytorch-lightning/pull/12225))
- Fixed an issue where `ModelCheckpoint` could delete older checkpoints when `dirpath` has changed during resumed training ([#12045](https://github.com/PyTorchLightning/pytorch-lightning/pull/12045))
- Fixed an issue where `HorovodStrategy.teardown()` did not complete gracefully if an exception was thrown during callback setup [#11752](https://github.com/PyTorchLightning/pytorch-lightning/pull/11752)
- Fixed security vulnerabilities CVE-2020-1747 and CVE-2020-14343 caused by the `PyYAML` dependency ([#11099](https://github.com/PyTorchLightning/pytorch-lightning/pull/11099))
- Fixed security vulnerability "CWE-94: Improper Control of Generation of Code (Code Injection)" ([#12212](https://github.com/PyTorchLightning/pytorch-lightning/pull/12212))
- Fixed logging on `{test,validation}_epoch_end` with multiple dataloaders ([#11132](https://github.com/PyTorchLightning/pytorch-lightning/pull/11132))
- Reset the validation progress tracking state after sanity checking ([#11218](https://github.com/PyTorchLightning/pytorch-lightning/pull/11218))
- Fixed double evaluation bug with fault-tolerance enabled where the second call was completely skipped ([#11119](https://github.com/PyTorchLightning/pytorch-lightning/pull/11119))
- Fixed an issue with the `TPUSpawnPlugin` handling the `XLA_USE_BF16` environment variable incorrectly ([#10990](https://github.com/PyTorchLightning/pytorch-lightning/pull/10990))
- Fixed wrong typehint for `Trainer.lightning_optimizers` ([#11155](https://github.com/PyTorchLightning/pytorch-lightning/pull/11155))
- Fixed the lr-scheduler state not being dumped to checkpoint when using the deepspeed strategy ([#11307](https://github.com/PyTorchLightning/pytorch-lightning/pull/11307))
- Fixed bug that forced overriding `configure_optimizers` with the CLI ([#11672](https://github.com/PyTorchLightning/pytorch-lightning/pull/11672))
- Fixed type promotion when tensors of higher category than float are logged ([#11401](https://github.com/PyTorchLightning/pytorch-lightning/pull/11401))
- Fixed `SimpleProfiler` summary ([#11414](https://github.com/PyTorchLightning/pytorch-lightning/pull/11414))
- No longer set a `DistributedSampler` to the `poptorch.DataLoader` when IPUs are used ([#12114](https://github.com/PyTorchLightning/pytorch-lightning/pull/12114))
- Fixed bug where progress bar was not being disabled when not in rank zero during predict ([#11377](https://github.com/PyTorchLightning/pytorch-lightning/pull/11377))
- Fixed the mid-epoch warning call while resuming training ([#11556](https://github.com/PyTorchLightning/pytorch-lightning/pull/11556))
- Fixed `LightningModule.{un,}toggle_model` when only 1 optimizer is used ([#12088](https://github.com/PyTorchLightning/pytorch-lightning/pull/12088))
- Fixed an issue in `RichProgressbar` to display the metrics logged only on main progress bar ([#11690](https://github.com/PyTorchLightning/pytorch-lightning/pull/11690))
- Fixed `RichProgressBar` progress when refresh rate does not evenly divide the total counter ([#11668](https://github.com/PyTorchLightning/pytorch-lightning/pull/11668))
- Fixed `RichProgressBar` progress validation bar total when using multiple validation runs within a single training epoch ([#11668](https://github.com/PyTorchLightning/pytorch-lightning/pull/11668))
- Configure native Deepspeed schedulers with interval='step' ([#11788](https://github.com/PyTorchLightning/pytorch-lightning/pull/11788)), ([#12031](https://github.com/PyTorchLightning/pytorch-lightning/pull/12031))
- Update `RichProgressBarTheme` styles after detecting light theme on colab ([#10993](https://github.com/PyTorchLightning/pytorch-lightning/pull/10993))
- Fixed passing `_ddp_params_and_buffers_to_ignore` ([#11949](https://github.com/PyTorchLightning/pytorch-lightning/pull/11949))
- Fixed an `AttributeError` when calling `save_hyperparameters` and no parameters need saving ([#11827](https://github.com/PyTorchLightning/pytorch-lightning/pull/11827))
- Fixed environment variable priority for global rank determination ([#11406](https://github.com/PyTorchLightning/pytorch-lightning/pull/11406))
- Fixed an issue that caused the Trainer to produce identical results on subsequent runs without explicit re-seeding ([#11870](https://github.com/PyTorchLightning/pytorch-lightning/pull/11870))
- Fixed an issue that caused the Tuner to affect the random state ([#11870](https://github.com/PyTorchLightning/pytorch-lightning/pull/11870))
- Fixed to avoid common hook warning if no hook is overridden ([#12131](https://github.com/PyTorchLightning/pytorch-lightning/pull/12131))
- Fixed deepspeed keeping old sub-folders in same ckpt path ([#12194](https://github.com/PyTorchLightning/pytorch-lightning/pull/12194))
- Fixed returning logged metrics instead of callback metrics during evaluation ([#12224](https://github.com/PyTorchLightning/pytorch-lightning/pull/12224))
- Fixed the case where `logger=None` is passed to the Trainer ([#12249](https://github.com/PyTorchLightning/pytorch-lightning/pull/12249))
- Fixed bug where the global step tracked by `ModelCheckpoint` was still set even if no checkpoint was saved ([#12418](https://github.com/PyTorchLightning/pytorch-lightning/pull/12418))
- Fixed bug where `ModelCheckpoint` was overriding the `epoch` and `step` logged values ([#12418](https://github.com/PyTorchLightning/pytorch-lightning/pull/12418))
- Fixed bug where monitoring the default `epoch` and `step` values with `ModelCheckpoint` would fail ([#12418](https://github.com/PyTorchLightning/pytorch-lightning/pull/12418))
- Fixed initializing optimizers unnecessarily in `DDPFullyShardedStrategy` ([#12267](https://github.com/PyTorchLightning/pytorch-lightning/pull/12267))
- Fixed check for horovod module ([#12377](https://github.com/PyTorchLightning/pytorch-lightning/pull/12377))
- Fixed logging to loggers with multiple eval dataloaders ([#12454](https://github.com/PyTorchLightning/pytorch-lightning/pull/12454))
- Fixed an issue with resuming from a checkpoint trained with QAT ([#11346](https://github.com/PyTorchLightning/pytorch-lightning/pull/11346))

</details>

**Full commit list**: https://github.com/PyTorchLightning/pytorch-lightning/compare/1.5.0...1.6.0

<a name="contributors"></a>
# Contributors

### Veteran

@akihironitta @ananthsub @awaelchli @Borda @borisdayma @carmocca @daniellepintz @edward-io @ethanwharris @four4fish @jjenniferdai @kaushikb11 @kingyiusuen @kragniz @mauvilsa @ninginthecloud @popfido @rohitgr7 @SeanNaren @speediedan @tchaton @tshu-w @twsl @williamFalcon

### New

@a-gardner1 @abhi-rf @abhinavarora @adamreeve @adamviola @AJSVB @akashkw @amin-nejad @AndresAlgaba @ant0nsc @armanal @bhadreshpsavani @CAIQT @catalys1 @chaddy1004 @chunyang-wen @circlecrystal @Code-Cornelius @Cyber-Machine @dennisbappert @DuYicong515 @edpizzi @franp9am @ftorres16 @ggare-cmu @guyang3532 @Honzys @idiomaticrefactoring @isvogor-foi @jerome-habana @jgibson2 @jlhbaseball15 @jona-0 @JoostvDoorn @josafatburmeister @konstantinjdobler @Kr4is @krishnakalyan3 @krshrimali @lemairecarl @lucmos @manangoel99 @mathemusician @mayeroa @mbortolon97 @NathanGodey @Nesqulck @nithinraok @ORippler @os1ma @peterdudfield @Piyush-97 @puhuk @qqueing @quancs @Raahul-Singh @Raalsky @Rajathbharadwaj @rasbt @rharish101 @rhjohnstone @rjkilpatrick @RobertLaurella @roschly @rsokl @rusty1s @SauravMaheshkar @sethvargo @shabie @shivammehta007 @srb-cv @ThomVett @wangraying @whokilleddb @zredeaux65

_If we forgot someone or have any suggestion, let us know in [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-12iz3cds1-uyyyBYJLiaL2bqVmMN7n~A) :zap:_


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.6.0)

---

## 1.5.10: Standard weekly patch release
**Published:** 2022-02-09

## [1.5.10] - 2022-02-08

### Fixed

- Fixed an issue to avoid validation loop run on restart ([#11552](https://github.com/PyTorchLightning/pytorch-lightning/pull/11552))
- The Rich progress bar now correctly shows the `on_epoch` logged values on train epoch end ([#11689](https://github.com/PyTorchLightning/pytorch-lightning/pull/11689))
- Fixed an issue to make the `step` argument in `WandbLogger.log_image` work ([#11716](https://github.com/PyTorchLightning/pytorch-lightning/pull/11716))
- Fixed `restore_optimizers` for mapping states ([#11757](https://github.com/PyTorchLightning/pytorch-lightning/pull/11757))
- With `DPStrategy`, the batch is not explicitly moved to the device ([#11780](https://github.com/PyTorchLightning/pytorch-lightning/pull/11780))
- Fixed an issue to avoid val bar disappear after `trainer.validate()` ([#11700](https://github.com/PyTorchLightning/pytorch-lightning/pull/11700))
- Fixed supporting remote filesystems with `Trainer.weights_save_path` for fault-tolerant training ([#11776](https://github.com/PyTorchLightning/pytorch-lightning/pull/11776))
- Fixed check for available modules ([#11526](https://github.com/PyTorchLightning/pytorch-lightning/pull/11526))
- Fixed bug where the path for "last" checkpoints was not getting saved correctly which caused newer runs to not remove the previous "last" checkpoint ([#11481](https://github.com/PyTorchLightning/pytorch-lightning/pull/11481))
- Fixed bug where the path for best checkpoints was not getting saved correctly when no metric was monitored which caused newer runs to not use the best checkpoint ([#11481](https://github.com/PyTorchLightning/pytorch-lightning/pull/11481))

### Contributors

@ananthsub @borda @circlecrystal @NathanGodey @nithinraok @rohitgr7


If we forgot someone due to not matching commit email with GitHub account, let us know :]


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.10)

---

## 1.5.9: Standard weekly patch release
**Published:** 2022-01-20

## [1.5.9] - 2022-01-20

### Fixed

- Pinned sphinx-autodoc-typehints with <v1.15 ([#11400](https://github.com/PyTorchLightning/pytorch-lightning/pull/11400))
- Skipped testing with PyTorch 1.7 and Python 3.9 on Ubuntu ([#11217](https://github.com/PyTorchLightning/pytorch-lightning/pull/11217))
- Fixed type promotion when tensors of higher category than float are logged ([#11401](https://github.com/PyTorchLightning/pytorch-lightning/pull/11401))
- Fixed the format of the configuration saved automatically by the CLI's `SaveConfigCallback` ([#11532](https://github.com/PyTorchLightning/pytorch-lightning/pull/11532))

### Changed

- Changed `LSFEnvironment` to use `LSB_DJOB_RANKFILE` environment variable instead of `LSB_HOSTS` for determining node rank and main address ([#10825](https://github.com/PyTorchLightning/pytorch-lightning/pull/10825))
- Disabled sampler replacement when using `IterableDataset` ([#11507](https://github.com/PyTorchLightning/pytorch-lightning/pull/11507))

### Contributors

@ajtritt @akihironitta @carmocca @rohitgr7

If we forgot someone due to not matching commit email with GitHub account, let us know :]


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.9)

---

## 1.5.8: Standard weekly patch release
**Published:** 2022-01-05

## [1.5.8] - 2022-01-05

### Fixed

- Fixed `LightningCLI` race condition while saving the config ([#11199](https://github.com/PyTorchLightning/pytorch-lightning/pull/11199))
- Fixed the default value used with `log(reduce_fx=min|max)` ([#11310](https://github.com/PyTorchLightning/pytorch-lightning/pull/11310))
- Fixed data fetcher selection ([#11294](https://github.com/PyTorchLightning/pytorch-lightning/pull/11294))
- Fixed a race condition that could result in incorrect (zero) values being observed in prediction writer callbacks ([#11288](https://github.com/PyTorchLightning/pytorch-lightning/pull/11288))
- Fixed dataloaders not getting reloaded the correct amount of times when setting `reload_dataloaders_every_n_epochs` and `check_val_every_n_epoch` ([#10948](https://github.com/PyTorchLightning/pytorch-lightning/pull/10948))


### Contributors

@adamviola @akihironitta @awaelchli @Borda @carmocca @edpizzi

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.8)

---

## 1.5.7: Standard weekly patch release
**Published:** 2021-12-21

## [1.5.7] - 2021-12-21

### Fixed

- Fixed `NeptuneLogger` when using DDP ([#11030](https://github.com/PyTorchLightning/pytorch-lightning/pull/11030))
- Fixed a bug to disable logging hyperparameters in logger if there are no hparams ([#11105](https://github.com/PyTorchLightning/pytorch-lightning/issues/11105))
- Avoid the deprecated `onnx.export(example_outputs=...)` in torch 1.10 ([#11116](https://github.com/PyTorchLightning/pytorch-lightning/pull/11116))
- Fixed an issue when torch-scripting a `LightningModule` after training with `Trainer(sync_batchnorm=True)` ([#11078](https://github.com/PyTorchLightning/pytorch-lightning/pull/11078))
- Fixed an `AttributeError` occuring when using a `CombinedLoader` (multiple dataloaders) for prediction ([#11111](https://github.com/PyTorchLightning/pytorch-lightning/pull/11111))
- Fixed bug where `Trainer(track_grad_norm=..., logger=False)` would fail ([#11114](https://github.com/PyTorchLightning/pytorch-lightning/pull/11114))
- Fixed an incorrect warning being produced by the model summary when using `bf16` precision on CPU ([#11161](https://github.com/PyTorchLightning/pytorch-lightning/pull/11161))

### Changed

- DeepSpeed does not require lightning module zero 3 partitioning ([#10655](https://github.com/PyTorchLightning/pytorch-lightning/pull/10655))
- The `ModelCheckpoint` callback now saves and restores attributes `best_k_models`, `kth_best_model_path`, `kth_value`, and `last_model_path` ([#10995](https://github.com/PyTorchLightning/pytorch-lightning/pull/10995))


### Contributors

@awaelchli @borchero @carmocca @guyang3532 @kaushikb11 @ORippler @Raalsky @rohitgr7 @SeanNaren

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_




[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.7)

---

## 1.5.6: Standard weekly patch release
**Published:** 2021-12-15

## [1.5.6] - 2021-12-15

### Fixed

- Fixed a bug where the DeepSpeedPlugin arguments `cpu_checkpointing` and `contiguous_memory_optimization` were not being forwarded to deepspeed correctly ([#10874](https://github.com/PyTorchLightning/pytorch-lightning/issues/10874))
- Fixed an issue with `NeptuneLogger` causing checkpoints to be uploaded with a duplicated file extension ([#11015](https://github.com/PyTorchLightning/pytorch-lightning/issues/11015))
- Fixed support for logging within callbacks returned from `LightningModule` ([#10991](https://github.com/PyTorchLightning/pytorch-lightning/pull/10991))
- Fixed running sanity check with `RichProgressBar` ([#10913](https://github.com/PyTorchLightning/pytorch-lightning/pull/10913))
- Fixed support for `CombinedLoader` while checking for warning raised with eval dataloaders ([#10994](https://github.com/PyTorchLightning/pytorch-lightning/pull/10994))
- The TQDM progress bar now correctly shows the `on_epoch` logged values on train epoch end ([#11069](https://github.com/PyTorchLightning/pytorch-lightning/pull/11069))
- Fixed bug where the TQDM updated the training progress bar during `trainer.validate` ([#11069](https://github.com/PyTorchLightning/pytorch-lightning/pull/11069))


### Contributors

@carmocca @jona-0 @kaushikb11 @Raalsky @rohitgr7 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.6)

---

## 1.5.5: Standard weekly patch release
**Published:** 2021-12-07

## [1.5.5] - 2021-12-07

### Fixed

- Disabled batch_size extraction for torchmetric instances because they accumulate the metrics internally ([#10815](https://github.com/PyTorchLightning/pytorch-lightning/pull/10815))
- Fixed an issue with `SignalConnector` not restoring the default signal handlers on teardown when running on SLURM or with fault-tolerant training enabled ([#10611](https://github.com/PyTorchLightning/pytorch-lightning/pull/10611))
- Fixed `SignalConnector._has_already_handler` check for callable type ([#10483](https://github.com/PyTorchLightning/pytorch-lightning/pull/10483))
- Fixed an issue to return the results for each dataloader separately instead of duplicating them for each ([#10810](https://github.com/PyTorchLightning/pytorch-lightning/pull/10810))
- Improved exception message if `rich` version is less than `10.2.2` ([#10839](https://github.com/PyTorchLightning/pytorch-lightning/pull/10839))
- Fixed uploading best model checkpoint in NeptuneLogger ([#10369](https://github.com/PyTorchLightning/pytorch-lightning/pull/10369))
- Fixed early schedule reset logic in PyTorch profiler that was causing data leak ([#10837](https://github.com/PyTorchLightning/pytorch-lightning/pull/10837))
- Fixed a bug that caused incorrect batch indices to be passed to the `BasePredictionWriter` hooks when using a dataloader with `num_workers > 0` ([#10870](https://github.com/PyTorchLightning/pytorch-lightning/pull/10870))
- Fixed an issue with item assignment on the logger on rank > 0 for those who support it ([#10917](https://github.com/PyTorchLightning/pytorch-lightning/pull/10917))
- Fixed importing `torch_xla.debug` for `torch-xla<1.8` ([#10836](https://github.com/PyTorchLightning/pytorch-lightning/pull/10836))
- Fixed an issue with `DDPSpawnPlugin` and related plugins leaving a temporary checkpoint behind ([#10934](https://github.com/PyTorchLightning/pytorch-lightning/pull/10934))
- Fixed a `TypeError` occuring in the `SingalConnector.teardown()` method ([#10961](https://github.com/PyTorchLightning/pytorch-lightning/pull/10961))


### Contributors

@awaelchli @carmocca @four4fish @kaushikb11 @lucmos @mauvilsa @Raalsky @rohitgr7

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.5)

---

## 1.5.4: Standard weekly patch release
**Published:** 2021-11-30

## [1.5.4] - 2021-11-30

### Fixed

- Fixed support for `--key.help=class` with the `LightningCLI` ([#10767](https://github.com/PyTorchLightning/pytorch-lightning/pull/10767))
- Fixed `_compare_version` for python packages ([#10762](https://github.com/PyTorchLightning/pytorch-lightning/pull/10762))
- Fixed TensorBoardLogger `SummaryWriter` not close before spawning the processes ([#10777](https://github.com/PyTorchLightning/pytorch-lightning/pull/10777))
- Fixed a consolidation error in Lite when attempting to save the state dict of a sharded optimizer ([#10746](https://github.com/PyTorchLightning/pytorch-lightning/pull/10746))
- Fixed the default logging level for batch hooks associated with training from `on_step=False, on_epoch=True` to `on_step=True, on_epoch=False` ([#10756](https://github.com/PyTorchLightning/pytorch-lightning/pull/10756))


### Removed

- Removed PyTorch 1.6 support ([#10367](https://github.com/PyTorchLightning/pytorch-lightning/pull/10367), [#10738](https://github.com/PyTorchLightning/pytorch-lightning/pull/10738))


### Contributors

@awaelchli @carmocca @kaushikb11 @rohitgr7 @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.4)

---

## 1.5.3: Standard weekly patch release
**Published:** 2021-11-24

## [1.5.3] - 2021-11-24

### Fixed

- Fixed `ShardedTensor` state dict hook registration to check if torch distributed is available ([#10621](https://github.com/PyTorchLightning/pytorch-lightning/pull/10621))
- Fixed an issue with `self.log` not respecting a tensor's `dtype` when applying computations ([#10076](https://github.com/PyTorchLightning/pytorch-lightning/pull/10076))
- Fixed LigtningLite `_wrap_init` popping unexisting keys from DataLoader signature parameters ([#10613](https://github.com/PyTorchLightning/pytorch-lightning/pull/10613))
- Fixed signals being registered within threads ([#10610](https://github.com/PyTorchLightning/pytorch-lightning/pull/10610))
- Fixed an issue that caused Lightning to extract the batch size even though it was set by the user in `LightningModule.log` ([#10408](https://github.com/PyTorchLightning/pytorch-lightning/pull/10408))
- Fixed `Trainer(move_metrics_to_cpu=True)` not moving the evaluation logged results to CPU ([#10631](https://github.com/PyTorchLightning/pytorch-lightning/pull/10631))
- Fixed the `{validation,test}_step` outputs getting moved to CPU with `Trainer(move_metrics_to_cpu=True)` ([#10631](https://github.com/PyTorchLightning/pytorch-lightning/pull/10631))
- Fixed signals being registered within threads ([#10610](https://github.com/PyTorchLightning/pytorch-lightning/pull/10610))
- Fixed an issue with collecting logged test results with multiple dataloaders ([#10522](https://github.com/PyTorchLightning/pytorch-lightning/pull/10522))


### Contributors

@ananthsub @awaelchli @carmocca @jiwidi @kaushikb11 @qqueing @rohitgr7 @shabie @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.3)

---

## 1.5.2: Standard weekly patch release
**Published:** 2021-11-16

## [1.5.2] - 2021-11-16

### Fixed

- Fixed `CombinedLoader` and `max_size_cycle` didn't receive a `DistributedSampler` ([#10374](https://github.com/PyTorchLightning/pytorch-lightning/issues/10374))
- Fixed an issue where class or init-only variables of dataclasses were passed to the dataclass constructor in `utilities.apply_to_collection` ([#9702](https://github.com/PyTorchLightning/pytorch-lightning/issues/9702))
- Fixed `isinstance` not working with `init_meta_context`, materialized model not being moved to the device ([#10493](https://github.com/PyTorchLightning/metrics/pull/10493))
- Fixed an issue that prevented the Trainer to shutdown workers when execution is interrupted due to failure([#10463](https://github.com/PyTorchLightning/pytorch-lightning/issues/10463))
- Squeeze the early stopping monitor to remove empty tensor dimensions ([#10461](https://github.com/PyTorchLightning/pytorch-lightning/issues/10461))
- Fixed sampler replacement logic with `overfit_batches` to only replace the sample when `SequentialSampler` is not used ([#10486](https://github.com/PyTorchLightning/pytorch-lightning/issues/10486))
- Fixed scripting causing false positive deprecation warnings ([#10470](https://github.com/PyTorchLightning/pytorch-lightning/pull/10470), [#10555](https://github.com/PyTorchLightning/pytorch-lightning/pull/10555))
- Do not fail if batch size could not be inferred for logging when using DeepSpeed ([#10438](https://github.com/PyTorchLightning/pytorch-lightning/issues/10438))
- Fixed propagation of device and dtype information to submodules of LightningLite when they inherit from `DeviceDtypeModuleMixin` ([#10559](https://github.com/PyTorchLightning/pytorch-lightning/issues/10559))

### Contributors

@a-gardner1 @awaelchli @carmocca @justusschock @Raahul-Singh @rohitgr7 @SeanNaren @tchaton 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.2)

---

## 1.5.1: Standard weekly patch release
**Published:** 2021-11-09


## [1.5.1] - 2021-11-09

### Fixed

- Fixed `apply_to_collection(defaultdict)` ([#10316](https://github.com/PyTorchLightning/pytorch-lightning/issues/10316))
- Fixed failure when `DataLoader(batch_size=None)` is passed ([#10345](https://github.com/PyTorchLightning/pytorch-lightning/issues/10345))
- Fixed interception of `__init__` arguments for sub-classed DataLoader re-instantiation in Lite ([#10334](https://github.com/PyTorchLightning/pytorch-lightning/issues/10334))
- Fixed issue with pickling `CSVLogger` after a call to `CSVLogger.save` ([#10388](https://github.com/PyTorchLightning/pytorch-lightning/pull/10388))
- Fixed an import error being caused by `PostLocalSGD` when `torch.distributed` not available ([#10359](https://github.com/PyTorchLightning/pytorch-lightning/pull/10359))
- Fixed the logging with `on_step=True` in epoch-level hooks causing unintended side-effects. Logging with `on_step=True` in epoch-level hooks will now correctly raise an error ([#10409](https://github.com/PyTorchLightning/pytorch-lightning/pull/10409))
- Fixed deadlocks for distributed training with `RichProgressBar` ([#10428](https://github.com/PyTorchLightning/pytorch-lightning/pull/10428))
- Fixed an issue where the model wrapper in Lite converted non-floating point tensors to float ([#10429](https://github.com/PyTorchLightning/pytorch-lightning/pull/10429))
- Fixed an issue with inferring the dataset type in fault-tolerant training ([#10432](https://github.com/PyTorchLightning/pytorch-lightning/pull/10432))
- Fixed dataloader workers with `persistent_workers` being deleted on every iteration ([#10434](https://github.com/PyTorchLightning/pytorch-lightning/pull/10434))


### Contributors

@EspenHa @four4fish @peterdudfield @rohitgr7 @tchaton @kaushikb11 @awaelchli @Borda @carmocca 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.1)

---

## 1.5.0: PyTorch Lightning 1.5: LightningLite, Fault-Tolerant Training, Loop Customization, Lightning Tutorials, LightningCLI v2, RichProgressBar, CheckpointIO Plugin, and Trainer Strategy Flag
**Published:** 2021-11-02

The PyTorch Lightning team and its community are excited to announce Lightning 1.5, introducing support for LightningLite, Fault-tolerant Training, Loop Customization, Lightning Tutorials, LightningCLI V2, RichProgressBar, CheckpointIO Plugin, Trainer Strategy flag, and more!

- [Highlights](#highlights)
- [Backward Incompatible Changes](#bc-changes)
- [Full Changelog](#changelog)


<a name="highlights"></a>
# Highlights

Lightning 1.5 marks our biggest release yet. Over 60 contributors have worked on features, bugfixes and documentation improvements for a total of 640 commits since v1.4. Here are some highlights:

## Fault-tolerant Training

[Fault-tolerant Training](https://pytorch-lightning.readthedocs.io/en/1.5.0/advanced/fault_tolerant_training.html) is a new internal mechanism that enables PyTorch Lightning to recover from a hardware or software failure. This is particularly interesting while training in the cloud with preemptive instances which can shutdown at any time. Once a Lightning experiment unexpectedly exits, a temporary checkpoint is saved that contains the exact state of all loops and the model. With this new experimental feature, you will be able to restore your training mid-epoch on the exact batch and continue training as if it never got interrupted.

```bash 
PL_FAULT_TOLERANT_TRAINING=1 python train.py
```

## LightningLite

LightningLite enables pure PyTorch users to scale their existing code to any kind of hardware while retaining full control over their own loops and optimization logic.

With just a few lines of code and no large refactoring, you get support for multi-device, multi-node, running on different accelerators (CPU, GPU, TPU), native automatic mixed precision (`half` and `bfloat16`), and double precision, in just a few seconds. And no special launcher required! Check out our [documentation](https://pytorch-lightning.readthedocs.io/en/1.5.0/starter/lightning_lite.html) to find out how you can get one step closer to boilerplate-free research!


```python
class Lite(LightningLite):
    def run(self):
        # Let Lite setup your dataloader(s)
        train_loader = self.setup_dataloaders(torch.utils.data.DataLoader(...))

        model = Net()  # .to() not needed
        optimizer = optim.Adam(model.parameters())
        # Let Lite setup your model and optimizer
        model, optimizer = self.setup(model, optimizer)

        for epoch in range(5):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)  # data is already on the device
                loss = F.nll_loss(output, target)
                self.backward(loss)  # instead of loss.backward()
                optimizer.step()


Lite(accelerator="gpu", devices="auto").run()
```

## Loop Customization

The new Loop API lets advanced users swap out the default gradient descent optimization loop at the core of Lightning with a different optimization paradigm. This is part of our effort to make Lightning the simplest, most flexible framework to take any kind of deep learning research to production.

[Read our comprehensive introduction to loops](https://pytorch-lightning.readthedocs.io/en/1.5.0/extensions/loops.html?highlight=loops)

## New Rich Progress Bar

We integrated with [Rich](https://github.com/willmcgugan/rich) and created a new and improved progress bar for Lightning.
Try it out:

```bash
pip install rich
```

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar

trainer = Trainer(callbacks=[RichProgressBar()])
```

<a name="strategy-and-devices"></a>
## New Trainer Arguments: Strategy and Devices

With the new strategy and devices arguments in the Trainer, it is now easer to switch from one hardware to another.

<div align="center">

| Before                                                | After                                                           |
|-------------------------------------------------------|-----------------------------------------------------------------|
| `Trainer(accelerator="ddp", gpus=2)`                  |  `Trainer(accelerator="gpu", devices=2, strategy="ddp")`        |
| `Trainer(accelerator="ddp_cpu", num_processes=2)`     |  `Trainer(accelerator="cpu", devices=2, strategy="ddp")`        |
| `Trainer(accelerator="tpu_spawn", tpu_cores=8)`       |  `Trainer(accelerator="tpu", devices=8)`                        |

</div>

The new `devices` argument is now agnostic to all accelerators, but the previous arguments `gpus`, `tpu_cores`, `ipus` are still available and work the same as before. In addition, it is now also possible to set `devices="auto"` or `accelerator="auto"` to select the best accelerator available on the hardware.

```python
from pytorch_lightning import Trainer

trainer = Trainer(accelerator="auto", devices="auto")
```

## LightningCLI V2

This release adds support for running not just `Trainer.fit` but any of the `Trainer` entry points!

```bash
python script.py fit
python script.py test
```

LightningCLI now supports registries for callbacks, optimizers, learning rate schedulers, LightningModules and LightningDataModules. This greatly improves the command line experience as only the class names and arguments are required as follows:

```bash
python script.py \
    --trainer.callbacks=EarlyStopping \
    --trainer.callbacks.patience=5 \
    --trainer.callbacks.LearningRateMonitor \
    --trainer.callbacks.logging_interval=epoch \
    --optimizer=Adam \
    --optimizer.lr=0.01 \
    --lr_scheduler=OneCycleLR \
    --lr_scheduler=anneal_strategy=linear
```

We've also added support for a manual mode where the CLI takes care of the instantiation but you have control over the `Trainer` calls:

```python
cli = LightningCLI(MyModel, run=False)
cli.trainer.fit(cli.model)
```

[Try out LightninCLI!](https://pytorch-lightning.readthedocs.io/en/1.5.0/common/lightning_cli.html)

## CheckpointIO Plugins

As part of our commitment to extensibility, we have abstracted the checkpointing logic into a [CheckpointIO](https://pytorch-lightning.readthedocs.io/en/1.5.0/api/pytorch_lightning.plugins.io.CheckpointIO.html?highlight=checkpointio) plugin. This enables users to adapt Lightning to their own infrastructure.

```python
from pytorch_lightning.plugins import CheckpointIO

class CustomCheckpointIO(CheckpointIO):
  
    def save_checkpoint(self, checkpoint, path):
        # put all logic related to saving a checkpoint here

    def load_checkpoint(self, path):
        # put all logic related to loading a checkpoint here

    def remove_checkpoint(self, path):
        # put all logic related to deleting a checkpoint here
```

## BFloat16 Support

PyTorch 1.10 introduces native Automatic Mixed Precision (AMP) support for `torch.bfloat16` on CPU (was already supported for TPUs), enabling higher performance compared with `torch.float16`. Switch to bfloat16 training by setting the argument:

```python
from pytorch_lightning import Trainer

trainer = Trainer(precision="bf16")
```

## Enable Auto Parameters Tying

It is pretty common to share parameters within a model. However, TPUs don't retain shared parameters once moved on the devices. Lightning now supports automatic detection and re-assignement to alleviate this problem from TPUs.

## Infinite Training

Infinite training is now supported by setting `Trainer(max_epochs=-1)` for an unlimited number of epochs, or `Trainer(max_steps=-1)` for an endless epoch.

> Note: you will want to avoid logging with `on_epoch=True` in case of `max_steps=-1`.

## DeepSpeed Stage 1

DeepSpeed is a deep learning training optimization library, providing the means to train massive billion parameter models at scale. Lightning now also supports the DeepSpeed ZeRO Stage 1 protocol that partitions your optimizer states across your GPUs to reduce memory.

```python

from pytorch_lightning import Trainer

trainer = Trainer(gpus=4, strategy="deepspeed_stage_1", precision=16)
trainer.fit(model)
```

For even more memory savings and model sharding advice, check out stage 2 & 3 as well in our [multi-GPU docs](https://pytorch-lightning.readthedocs.io/en/1.5.0/advanced/advanced_gpu.html#deepspeed).


## Gradient Clipping Customization

By overriding the `LightningModule.configure_gradient_clipping` hook, you can customize gradient clipping to your needs:

```python
# Perform gradient clipping on gradients associated with discriminator (optimizer_idx=1) in GAN
def configure_gradient_clipping(
    self,
    optimizer,
    optimizer_idx,
    gradient_clip_val,
    gradient_clip_algorithm
):
    if optimizer_idx == 1:
        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )
```

This means you can now implement state-of-the-art clipping algorithms with Lightning!

## Determinism

Added support for `torch.use_deterministic_algorithms`. Read more about how it works [here](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html). You can enable it by setting:

```python
from pytorch_lightning import Trainer

trainer = Trainer(deterministic=True)
```

## Anomaly Detection

Lightning makes it easier to debug your code, so we've added support for `torch.set_detect_anomaly`. With this, PyTorch detects numerical anomalies like NaN or inf during forward and backward. Read more about anomaly detection [here](https://pytorch.org/docs/stable/autograd.html)

```python
from pytorch_lightning import Trainer

trainer = Trainer(detect_anomaly=True)
```

## DDP Debugging Improvements

Are you having a hard time debugging DDP on your remote machine? Now you can debug DDP locally on the CPU:

```python 
trainer = Trainer(accelerator="cpu", strategy="ddp", devices=2)
```

When everything works, switch back to GPU by changing only the `accelerator`. Check our documentation for more [useful debugging tricks](https://pytorch-lightning.readthedocs.io/en/1.5.0/common/debugging.html).
Note that this will not provide any speed benefits.

## ModelSummary Callback

Generates a summary of all layers in a LightningModule. This currently works with the new `RichProgressBar` callback.

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary

trainer = Trainer(callbacks=[ModelSummary(max_depth=1)])
```

## New Hooks

An `on_exception` Callback hook has been added which allows the user to perform custom exception handling.

```python
class MyCallback(Callback):
    def on_exception(self, trainer, pl_module, exception):
        # whatever you want!
        ...
```

## Experimental Features

### Inter Batch Parallelism

The inter-batch parallelism feature aims at hiding the latency of host-to-device copy of input batches behind computationally intensive operations. In some use case, it can provide training speed up. This feature is experimental and subject to change, hence opt-in through an environment variable.

```bash 
PL_INTER_BATCH_PARALLELISM=1 python train.py
```

### Training Step With DataLoader Iterator

If your `training_step` signature takes a `dataloader_iter`, Lightning would pass it directly. This can be useful for recommendation engine optimization.

### Meta Module

PyTorch 1.10 introduces the meta tensors, tensors without the data. In this continuation, PyTorch Lightning provides an `init_meta_context` context manager and `materialize_module` function to handle large sharded models.

<a name="bc-changes"></a>

# Backward Incompatible Changes

Here is a selection of important changes that are not backward compatible with versions < 1.5. The full list of changes and removals are listed in the changelog at the bottom.

### Parsing of GPU Argument

The interpretation of the `gpus` Trainer argument when provided as a string has changed: `Trainer(gpus="n")` (string) no longer selects the GPU index n and instead selects the first n devices. In order to preserve the old behavior, you will have to change your code to `Trainer(gpus=[n])` (list of indices) or `Trainer(gpus="n,")` (string with comma separated indices).

### Distributed Backend

The argument `distributed_backend` has been removed from the `Trainer` in favor of the new `accelerator` and `strategy` arguments ([#10017](https://github.com/PyTorchLightning/pytorch-lightning/pull/10017)).

```python
# BEFORE
trainer = Trainer(distributed_backend="ddp_spawn", gpus=2)

# NOW
trainer = Trainer(strategy="ddp_spawn", accelerator="gpu", devices=2)
```

### Trainer Argument Defaults

- The default value of the `max_steps` Trainer argument has changed from `None` to -1 ([#9460](https://github.com/PyTorchLightning/pytorch-lightning/pull/9460)). You can no longer specify `Trainer(max_steps=None)` and if you did, you need to change the code to `Trainer(max_steps=-1)`.
- The default value of `accumulate_grad_batches` has changed from 1 to `None` ([#9652](https://github.com/PyTorchLightning/pytorch-lightning/pull/9652)).

### Loading Model Weights
The model weights now get loaded in all cases when the checkpoint path is provided in `Trainer.{validate,test,predict}`, regardless of whether the model instance is provided or not. 

```python

# model reference provided:
trainer.test(model, ckpt_path=None) # use provided model
trainer.test(model, ckpt_path="best") # load best model
trainer.test(model, ckpt_path="my_path") # load path

# model reference not provided
trainer.fit(model)
trainer.test(ckpt_path=None) # load best model (NEW BEHAVIOR!)
trainer.test(ckpt_path="my_path") # load path (NEW BEHAVIOR!)
```
Users who relied on `trainer.test(ckpt_path=None)` to load the latest model need to change their code to `trainer.test(model)` and pass the model reference directly.

### Lightning CLI

All CLI commands now need to include the Trainer method to run as the first command, i.e., one of `fit`, `validate`, `test`, `predict`.

```python
# BEFORE
python script.py     --trainer.max_epochs=123

# NOW
python script.py fit --trainer.max_epochs=123
```

For questions and help regarding CLI, join our [Lightning-CLI Slack channel](https://pytorch-lightning.slack.com/archives/C01URG3M74L).

### Optimizer Hooks

- Executing the `optimizer_closure` is now required when overriding the `optimizer_step` hook ([#9360](https://github.com/PyTorchLightning/pytorch-lightning/pull/9360)). If you relied on the previous behavior, we recommend to switch to Manual Optimization alltogether.
- The `on_before_optimizer_step` hook previously ran before the entire optimization closure, including backward. This was unintended behavior and if you rely on this, move your code to the new on_before_backward` hook.

### Changes in Accelerators and Plugins

Changes in Accelerators and Plugins were made without deprecation due to their experimental state. The API is expected to become stable in 1.6.

Removed attributes and methods:
- `Accelerator.{call_configure_sharded_model_hook, connect_training_type_plugin, connect_precision_plugin, on_reset_*_dataloader, on_train_epoch_end, on_save, post_optimizer_step, update_global_step}`
- `TrainingTypePlugin.{call_configure_sharded_model_hook, on_reset_*_dataloader, on_save, post_optimizer_step, update_global_step}`
- `PrecisionPlugin.{post_optimizer_step}`
- `ParallelPlugin.teardown`

Changed signatures:
- The accelerator and training type plugin `setup` hooks no longer have a `model` argument.

Other changes:
- The base `Plugin` class has been removed. 
- `HorovodPlugin.all_gather` now returns a `torch.Tensor` instead of a list.
- The LightningModule no longer gets wrapped with data-parallel modules when not fitting in `DDPPlugin`, `DDPSpawnPlugin`, `DDPShardedPlugin`, `DDPSpawnShardedPlugin`.

<a name="changelog"></a>

# Full Changelog

### Added

- Added support for monitoring the learning rate without schedulers in `LearningRateMonitor` ([#9786](https://github.com/PyTorchLightning/pytorch-lightning/issues/9786))
- Added registration of `ShardedTensor` state dict hooks in `LightningModule.__init__` if the PyTorch version supports `ShardedTensor` ([#8944](https://github.com/PyTorchLightning/pytorch-lightning/pull/8944))
- Added error handling including calling of `on_keyboard_interrupt()` and `on_exception()` for all entrypoints (fit, validate, test, predict) ([#8819](https://github.com/PyTorchLightning/pytorch-lightning/pull/8819))
- Added a flavor of `training_step` that takes `dataloader_iter` as an argument ([#8807](https://github.com/PyTorchLightning/pytorch-lightning/pull/8807))
- Added a `state_key` property to the `Callback` base class ([#6886](https://github.com/PyTorchLightning/pytorch-lightning/pull/6886))
- Added progress tracking to loops:
    * Integrated `TrainingEpochLoop.total_batch_idx` ([#8598](https://github.com/PyTorchLightning/pytorch-lightning/pull/8598))
    * Added `BatchProgress` and integrated `TrainingEpochLoop.is_last_batch` ([#9657](https://github.com/PyTorchLightning/pytorch-lightning/pull/9657))
    * Avoid optional `Tracker` attributes ([#9320](https://github.com/PyTorchLightning/pytorch-lightning/pull/9320))
    * Reset `current` progress counters when restarting an epoch loop that had already finished ([#9371](https://github.com/PyTorchLightning/pytorch-lightning/pull/9371))
    * Call `reset_on_restart` in the loop's `reset` hook instead of when loading a checkpoint ([#9561](https://github.com/PyTorchLightning/pytorch-lightning/pull/9561))
    * Use `completed` over `processed` in `reset_on_restart` ([#9656](https://github.com/PyTorchLightning/pytorch-lightning/pull/9656))
    * Renamed `reset_on_epoch` to `reset_on_run` ([#9658](https://github.com/PyTorchLightning/pytorch-lightning/pull/9658))
- Added `batch_size` and `rank_zero_only` arguments for `log_dict` to match `log` ([#8628](https://github.com/PyTorchLightning/pytorch-lightning/pull/8628))
- Added a check for unique GPU ids ([#8666](https://github.com/PyTorchLightning/pytorch-lightning/pull/8666))
- Added `ResultCollection` state_dict to the Loop `state_dict` and added support for distributed reload ([#8641](https://github.com/PyTorchLightning/pytorch-lightning/pull/8641))
- Added DeepSpeed collate checkpoint utility function ([#8701](https://github.com/PyTorchLightning/pytorch-lightning/pull/8701))
- Added a `handles_accumulate_grad_batches` property to the training type plugins ([#8856](https://github.com/PyTorchLightning/pytorch-lightning/pull/8856))
- Added a warning to `WandbLogger` when reusing a wandb run ([#8714](https://github.com/PyTorchLightning/pytorch-lightning/pull/8714))
- Added `log_graph` argument for `watch` method of `WandbLogger` ([#8662](https://github.com/PyTorchLightning/pytorch-lightning/pull/8662))
- `LightningCLI` additions:
  * Added `LightningCLI(run=False|True)` to choose whether to run a `Trainer` subcommand ([#8751](https://github.com/PyTorchLightning/pytorch-lightning/pull/8751))
  * Added support to call any trainer function from the `LightningCLI` via subcommands ([#7508](https://github.com/PyTorchLightning/pytorch-lightning/pull/7508))
  * Allow easy trainer re-instantiation ([#7508](https://github.com/PyTorchLightning/pytorch-lightning/pull/9241))
  * Automatically register all optimizers and learning rate schedulers ([#9565](https://github.com/PyTorchLightning/pytorch-lightning/pull/9565))
  * Allow registering custom optimizers and learning rate schedulers without subclassing the CLI ([#9565](https://github.com/PyTorchLightning/pytorch-lightning/pull/9565))
  * Support shorthand notation to instantiate optimizers and learning rate schedulers ([#9565](https://github.com/PyTorchLightning/pytorch-lightning/pull/9565))
  * Support passing lists of callbacks via command line ([#8815](https://github.com/PyTorchLightning/pytorch-lightning/pull/8815))
  * Support shorthand notation to instantiate models ([#9588](https://github.com/PyTorchLightning/pytorch-lightning/pull/9588))
  * Support shorthand notation to instantiate datamodules ([#10011](https://github.com/PyTorchLightning/pytorch-lightning/pull/10011))
  * Added `multifile` option to `LightningCLI` to enable/disable config saving to preserve multiple files structure ([#9073](https://github.com/PyTorchLightning/pytorch-lightning/pull/9073))
- Fault-tolerant training:
    * Added `FastForwardSampler` and `CaptureIterableDataset` injection to data loading utilities ([#8366](https://github.com/PyTorchLightning/pytorch-lightning/pull/8366))
    * Added `DataFetcher` to control fetching flow ([#8890](https://github.com/PyTorchLightning/pytorch-lightning/pull/8890))
    * Added `SharedCycleIteratorState` to prevent infinite loop ([#8889](https://github.com/PyTorchLightning/pytorch-lightning/pull/8889))
    * Added `CaptureMapDataset` for state management in map-style datasets ([#8891](https://github.com/PyTorchLightning/pytorch-lightning/pull/8891))
    * Added Fault Tolerant Training to `DataFetcher` ([#8891](https://github.com/PyTorchLightning/pytorch-lightning/pull/8891))
    * Replaced old prefetch iterator with new `DataFetcher` in training loop ([#8953](https://github.com/PyTorchLightning/pytorch-lightning/pull/8953))
    * Added partial support for global random state fault-tolerance in map-style datasets ([#8950](https://github.com/PyTorchLightning/pytorch-lightning/pull/8950))
    * Converted state to tuple explicitly when setting Python random state ([#9401](https://github.com/PyTorchLightning/pytorch-lightning/pull/9401))
    * Added support for restarting an optimizer loop (multiple optimizers) ([#9537](https://github.com/PyTorchLightning/pytorch-lightning/pull/9537))
    * Added support for restarting within Evaluation Loop ([#9563](https://github.com/PyTorchLightning/pytorch-lightning/pull/9563))
    * Added mechanism to detect that a signal has been sent so the Trainer can gracefully exit ([#9566](https://github.com/PyTorchLightning/pytorch-lightning/pull/9566))
    * Added support for skipping ahead to validation during the auto-restart of fitting ([#9681](https://github.com/PyTorchLightning/pytorch-lightning/pull/9681))
    * Added support for auto-restart if a fault-tolerant checkpoint is available ([#9722](https://github.com/PyTorchLightning/pytorch-lightning/pull/9722))
- Checkpoint saving and loading extensibility:
  * Added `CheckpointIO` plugin to expose checkpoint IO from training type plugin ([#8743](https://github.com/PyTorchLightning/pytorch-lightning/pull/8743))
  * Refactored `CheckpointConnector` to offload validation logic to the `CheckpointIO` plugin ([#9045](https://github.com/PyTorchLightning/pytorch-lightning/pull/9045))
  * Added `remove_checkpoint` to `CheckpointIO` plugin by moving the responsibility out of the `ModelCheckpoint` callback ([#9373](https://github.com/PyTorchLightning/pytorch-lightning/pull/9373))
  * Added `XLACheckpointIO` plugin ([#9972](https://github.com/PyTorchLightning/pytorch-lightning/pull/9972))
- Loop customization:
    * Added `Closure` and `AbstractClosure` classes ([#8642](https://github.com/PyTorchLightning/pytorch-lightning/pull/8642))
    * Refactored `TrainingBatchLoop` and extracted `OptimizerLoop`, splitting off automatic optimization into its own loop ([#9191](https://github.com/PyTorchLightning/pytorch-lightning/pull/9191))
    * Removed `TrainingBatchLoop.backward()`; manual optimization now calls directly into `Accelerator.backward()` and automatic optimization handles backward in new `OptimizerLoop` ([#9265](https://github.com/PyTorchLightning/pytorch-lightning/pull/9265))
    * Extracted `ManualOptimization` logic from `TrainingBatchLoop` into its own separate loop class ([#9266](https://github.com/PyTorchLightning/pytorch-lightning/pull/9266))
    * Added `OutputResult` and `ManualResult` classes ([#9437](https://github.com/PyTorchLightning/pytorch-lightning/pull/9437), [#9424](https://github.com/PyTorchLightning/pytorch-lightning/pull/9424))
    * Marked `OptimizerLoop.backward` as protected ([#9514](https://github.com/PyTorchLightning/pytorch-lightning/pull/9514))
    * Marked `FitLoop.should_accumulate` as protected ([#9515](https://github.com/PyTorchLightning/pytorch-lightning/pull/9515))
    * Marked several methods in `PredictionLoop` as protected: `on_predict_start`, `on_predict_epoch_end`, `on_predict_end`, `on_predict_model_eval` ([#9516](https://github.com/PyTorchLightning/pytorch-lightning/pull/9516))
    * Marked several methods in `EvaluationLoop` as protected: `get_max_batches`, `on_evaluation_model_eval`, `on_evaluation_model_train`, `on_evaluation_start`, `on_evaluation_epoch_start`, `on_evaluation_epoch_end`, `on_evaluation_end`, `reload_evaluation_dataloaders` ([#9516](https://github.com/PyTorchLightning/pytorch-lightning/pull/9516))
    * Marked several methods in `EvaluationEpochLoop` as protected: `on_evaluation_batch_start`, `evaluation_step`, `evaluation_step_end` ([#9516](https://github.com/PyTorchLightning/pytorch-lightning/pull/9516))
    * Added `yielding_training_step` example ([#9983](https://github.com/PyTorchLightning/pytorch-lightning/pull/9983))
- Added support for saving and loading state of multiple callbacks of the same type ([#7187](https://github.com/PyTorchLightning/pytorch-lightning/pull/7187))
- Added DeepSpeed Stage 1 support ([#8974](https://github.com/PyTorchLightning/pytorch-lightning/pull/8974))
- Added `Python dataclass` support for `LightningDataModule` ([#8272](https://github.com/PyTorchLightning/pytorch-lightning/issues/8272))
- Added sanitization of tensors when they get logged as hyperparameters in `TensorBoardLogger` ([#9031](https://github.com/PyTorchLightning/pytorch-lightning/pull/9031))
- Added `InterBatchParallelDataFetcher` ([#9020](https://github.com/PyTorchLightning/pytorch-lightning/pull/9020))
- Added `DataLoaderIterDataFetcher` ([#9020](https://github.com/PyTorchLightning/pytorch-lightning/pull/9020))
- Added `DataFetcher` within `Fit / Evaluation` Loop  ([#9047](https://github.com/PyTorchLightning/pytorch-lightning/pull/9047))
- Added a friendly error message when DDP attempts to spawn new distributed processes with rank > 0 ([#9005](https://github.com/PyTorchLightning/pytorch-lightning/pull/9005))
- Added Rich integration:
    * Added Rich progress bar ([#8929](https://github.com/PyTorchLightning/pytorch-lightning/pull/8929), [#9559](https://github.com/PyTorchLightning/pytorch-lightning/pull/9559))
    * Added Support for iterable datasets ([#9734](https://github.com/PyTorchLightning/pytorch-lightning/pull/9734))
    * Added `RichModelSummary` callback ([#9546](https://github.com/PyTorchLightning/pytorch-lightning/pull/9546))
    * Added `configure_columns` method to `RichProgressBar` ([#10288](https://github.com/PyTorchLightning/pytorch-lightning/pull/10288))
    * Added `leave` argument to `RichProgressBar` ([#10301](https://github.com/PyTorchLightning/pytorch-lightning/pull/10301))
- Added input validation logic for precision ([#9080](https://github.com/PyTorchLightning/pytorch-lightning/pull/9080))
- Added support for CPU AMP autocast ([#9084](https://github.com/PyTorchLightning/pytorch-lightning/pull/9084))
- Added `on_exception` callback hook ([#9183](https://github.com/PyTorchLightning/pytorch-lightning/pull/9183))
- Added a warning to DeepSpeed when inferring batch size ([#9221](https://github.com/PyTorchLightning/pytorch-lightning/pull/9221))
- Added `ModelSummary` callback ([#9344](https://github.com/PyTorchLightning/pytorch-lightning/pull/9344))
- Added `log_images`, `log_text` and `log_table` to `WandbLogger` ([#9545](https://github.com/PyTorchLightning/pytorch-lightning/pull/9545))
- Added `PL_RECONCILE_PROCESS` environment variable to enable process reconciliation regardless of cluster environment settings ([#9389](https://github.com/PyTorchLightning/pytorch-lightning/pull/9389))
- Added `get_device_stats` to the Accelerator interface and added its implementation for GPU and TPU ([#9586](https://github.com/PyTorchLightning/pytorch-lightning/pull/9586))
- Added a warning when an unknown key is encountered in the optimizer configuration, and when `OneCycleLR` is used with `"interval": "epoch"` ([#9666](https://github.com/PyTorchLightning/pytorch-lightning/pull/9666))
- Added `DeviceStatsMonitor` callback ([#9712](https://github.com/PyTorchLightning/pytorch-lightning/pull/9712))
- Added `enable_progress_bar` to the Trainer constructor ([#9664](https://github.com/PyTorchLightning/pytorch-lightning/pull/9664))
- Added `pl_legacy_patch` load utility for loading old checkpoints that have pickled legacy Lightning attributes ([#9166](https://github.com/PyTorchLightning/pytorch-lightning/pull/9166))
- Added support for `torch.use_deterministic_algorithms` ([#9121](https://github.com/PyTorchLightning/pytorch-lightning/pull/9121))
- Added automatic parameters tying for TPUs ([#9525](https://github.com/PyTorchLightning/pytorch-lightning/pull/9525))
- Added support for `torch.autograd.set_detect_anomaly` through `Trainer` constructor argument `detect_anomaly` ([#9848](https://github.com/PyTorchLightning/pytorch-lightning/pull/9848))
- Added `enable_model_summary` flag to Trainer ([#9699](https://github.com/PyTorchLightning/pytorch-lightning/pull/9699))
- Added `strategy` argument to Trainer ([#8597](https://github.com/PyTorchLightning/pytorch-lightning/pull/8597))
- Added `init_meta_context`, `materialize_module` utilities ([#9920](https://github.com/PyTorchLightning/pytorch-lightning/pull/9920))
- Added `TPUPrecisionPlugin` ([#10020](https://github.com/PyTorchLightning/pytorch-lightning/pull/#10020))
- Added `torch.bfloat16` support:
  * Added bfloat16 support for Lightning Trainer ([#9049](https://github.com/PyTorchLightning/pytorch-lightning/pull/9049))
  * Renamed `TPUHalfPrecisionPlugin` to `TPUBf16PrecisionPlugin` ([#10026](https://github.com/PyTorchLightning/pytorch-lightning/pull/10026))
  * Default to `precision=bf16` on CPU when `precision=16` is passed ([#10033](https://github.com/PyTorchLightning/pytorch-lightning/pull/10033))
  * Added support for `torch.autocast` ([#10053](https://github.com/PyTorchLightning/pytorch-lightning/pull/10053))
- Added `kfold` example for loop customization ([#9965](https://github.com/PyTorchLightning/pytorch-lightning/pull/9965))
- LightningLite:
    * Added `PrecisionPlugin.forward_context`, making it the default implementation for all `{train,val,test,predict}_step_context()` methods ([#9988](https://github.com/PyTorchLightning/pytorch-lightning/pull/9988))
    * Added `DDPSpawnPlugin.spawn()` for spawning new processes of a given function ([#10018](https://github.com/PyTorchLightning/pytorch-lightning/pull/10018), [#10022](https://github.com/PyTorchLightning/pytorch-lightning/pull/10022))
    * Added `TrainingTypePlugin.{_setup_model, _setup_optimizer}` methods ([#9994](https://github.com/PyTorchLightning/pytorch-lightning/pull/9994), [#10064](https://github.com/PyTorchLightning/pytorch-lightning/pull/10064))
    * Implemented `DataParallelPlugin._setup_model` ([#10010](https://github.com/PyTorchLightning/pytorch-lightning/pull/10010))
    * Implemented `DeepSpeedPlugin._setup_model_and_optimizers` ([#10009](https://github.com/PyTorchLightning/pytorch-lightning/pull/10009), [#10064](https://github.com/PyTorchLightning/pytorch-lightning/pull/10064))
    * Implemented `{DDPShardedPlugin,DDPShardedSpawnPlugin}._setup_model_and_optimizers` ([#10028](https://github.com/PyTorchLightning/pytorch-lightning/pull/10028), [#10064](https://github.com/PyTorchLightning/pytorch-lightning/pull/10064))
    * Added optional `model` argument to the `optimizer_step` methods in accelerators and plugins ([#10023](https://github.com/PyTorchLightning/pytorch-lightning/pull/10023))
    * Updated precision attributes in `DeepSpeedPlugin` ([#10164](https://github.com/PyTorchLightning/pytorch-lightning/pull/10164))
    * Added the ability to return a result from rank 0 in `DDPSpawnPlugin.spawn` ([#10162](https://github.com/PyTorchLightning/pytorch-lightning/pull/10162))
    * Added `pytorch_lightning.lite` package ([#10175](https://github.com/PyTorchLightning/pytorch-lightning/pull/10175))
    * Added `LightningLite` documentation ([#10043](https://github.com/PyTorchLightning/pytorch-lightning/pull/10043))
    * Added `LightningLite` examples ([#9987](https://github.com/PyTorchLightning/pytorch-lightning/pull/9987))
    * Make the `_LiteDataLoader` an iterator and add supports for custom dataloader ([#10279](https://github.com/PyTorchLightning/pytorch-lightning/pull/10279))
- Added `use_omegaconf` argument to `save_hparams_to_yaml` plugin ([#9170](https://github.com/PyTorchLightning/pytorch-lightning/pull/9170))
- Added `ckpt_path` argument for `Trainer.fit()` ([#10061](https://github.com/PyTorchLightning/pytorch-lightning/pull/10061))
- Added `auto_device_count` method to `Accelerators` ([#10222](https://github.com/PyTorchLightning/pytorch-lightning/pull/10222))
- Added support for `devices="auto"` ([#10264](https://github.com/PyTorchLightning/pytorch-lightning/pull/10264))
- Added a `filename` argument in `ModelCheckpoint.format_checkpoint_name` ([#9818](https://github.com/PyTorchLightning/pytorch-lightning/pull/9818))
- Added support for empty `gpus` list to run on CPU ([#10246](https://github.com/PyTorchLightning/pytorch-lightning/pull/10246))
- Added a warning if multiple batch sizes are found from ambiguous batch ([#10247](https://github.com/PyTorchLightning/pytorch-lightning/pull/10247))


### Changed

- Trainer now raises a `MisconfigurationException` when its methods are called with `ckpt_path="best"` but a checkpoint callback isn't configured ([#9841](https://github.com/PyTorchLightning/pytorch-lightning/pull/9841))
- Setting `Trainer(accelerator="ddp_cpu")` now does not spawn a subprocess if `num_processes` is kept `1` along with `num_nodes > 1` ([#9603](https://github.com/PyTorchLightning/pytorch-lightning/pull/9603))
- Module imports are now catching `ModuleNotFoundError` instead of `ImportError` ([#9867](https://github.com/PyTorchLightning/pytorch-lightning/pull/9867))
- `pytorch_lightning.loggers.neptune.NeptuneLogger` is now consistent with the new [neptune-client](https://github.com/neptune-ai/neptune-client) API; the old [neptune-client](https://github.com/neptune-ai/neptune-client) API is supported by `NeptuneClient` from the [neptune-contrib](https://github.com/neptune-ai/neptune-contrib) repo ([#6867](https://github.com/PyTorchLightning/pytorch-lightning/pull/6867))
- Parsing of `enums` type hyperparameters to be saved in the `haprams.yaml` file by TensorBoard and CSV loggers has been fixed and made in line with how OmegaConf parses it ([#9170](https://github.com/PyTorchLightning/pytorch-lightning/pull/9170))
- Parsing of the `gpus` Trainer argument has changed: `gpus="n"` (str) no longer selects the GPU index n and instead selects the first n devices ([#8770](https://github.com/PyTorchLightning/pytorch-lightning/pull/8770))
- `iteration_count` and other index attributes in the loops has been replaced with progress dataclasses ([#8477](https://github.com/PyTorchLightning/pytorch-lightning/pull/8477))
- The `trainer.lightning_module` reference is now properly set at the very beginning of a run ([#8536](https://github.com/PyTorchLightning/pytorch-lightning/pull/8536))
- The model weights now get loaded in all cases when the checkpoint path gets provided in validate/test/predict, regardless of whether the model instance is provided or not ([#8352](https://github.com/PyTorchLightning/pytorch-lightning/pull/8352))
- The `Trainer` functions `reset_{train,val,test,predict}_dataloader`, `reset_train_val_dataloaders`, and `request_dataloader` `model` argument is now optional ([#8536](https://github.com/PyTorchLightning/pytorch-lightning/pull/8536))
- Saved checkpoints will no longer use the type of a `Callback` as the key to avoid issues with unpickling ([#6886](https://github.com/PyTorchLightning/pytorch-lightning/pull/6886))
- Improved string conversion for `ResultCollection` ([#8622](https://github.com/PyTorchLightning/pytorch-lightning/pull/8622))
- `LightningCLI` changes:
    * `LightningCLI.init_parser` now returns the parser instance ([#8721](https://github.com/PyTorchLightning/pytorch-lightning/pull/8721))
    * `LightningCLI.add_core_arguments_to_parser`, `LightningCLI.parse_arguments` now take a `parser` argument ([#8721](https://github.com/PyTorchLightning/pytorch-lightning/pull/8721))
    * `LightningCLI.instantiate_trainer` now takes a config and a list of callbacks ([#8721](https://github.com/PyTorchLightning/pytorch-lightning/pull/8721))
    * Split `LightningCLI.add_core_arguments_to_parser` into `LightningCLI.add_default_arguments_to_parser` + `LightningCLI.add_core_arguments_to_parser` ([#8721](https://github.com/PyTorchLightning/pytorch-lightning/pull/8721))
- The accelerator and training type plugin `setup` hooks no longer have a `model` argument ([#8536](https://github.com/PyTorchLightning/pytorch-lightning/pull/8536))
- The accelerator and training type plugin `update_global_step` hook has been removed ([#8856](https://github.com/PyTorchLightning/pytorch-lightning/pull/8856))
- The coverage of `self.log`-ing in any `LightningModule` or `Callback` hook has been improved ([#8498](https://github.com/PyTorchLightning/pytorch-lightning/pull/8498))
- `self.log`-ing without a `Trainer` reference now raises a warning instead of an exception ([#9733](https://github.com/PyTorchLightning/pytorch-lightning/pull/9733))
- Removed restrictions in the Trainer that loggers can only log from rank 0; the existing logger behavior has not changed ([#8608](https://github.com/PyTorchLightning/pytorch-lightning/pull/8608))
- `Trainer.request_dataloader` now takes a `RunningStage` enum instance ([#8858](https://github.com/PyTorchLightning/pytorch-lightning/pull/8858))
- Changed `rank_zero_warn` to `NotImplementedError` in the `{train, val, test, predict}_dataloader` hooks that `Lightning(Data)Module` uses ([#9161](https://github.com/PyTorchLightning/pytorch-lightning/pull/9161))
- Moved `block_ddp_sync_behaviour` out of `TrainingBatchLoop` to loop utilities ([#9192](https://github.com/PyTorchLightning/pytorch-lightning/pull/9192))
- Executing the `optimizer_closure` is now required when overriding the `optimizer_step` hook ([#9360](https://github.com/PyTorchLightning/pytorch-lightning/pull/9360))
- Changed logging of `LightningModule` and `LightningDataModule` hyperparameters to raise an exception only if there are colliding keys with different values ([#9496](https://github.com/PyTorchLightning/pytorch-lightning/pull/9496))
- `seed_everything` now fails when an invalid seed value is passed instead of selecting a random seed ([#8787](https://github.com/PyTorchLightning/pytorch-lightning/pull/8787))
- The Trainer now calls `TrainingTypePlugin` collective APIs directly instead of going through the Accelerator reference ([#9677](https://github.com/PyTorchLightning/pytorch-lightning/pull/9677), [#9901](https://github.com/PyTorchLightning/pytorch-lightning/pull/9901))
- The tuner now usees a unique filename to save a temporary checkpoint ([#9682](https://github.com/PyTorchLightning/pytorch-lightning/pull/9682))
- Changed `HorovodPlugin.all_gather` to return a `torch.Tensor` instead of a list ([#9696](https://github.com/PyTorchLightning/pytorch-lightning/pull/9696))
- Changed Trainer connectors to be protected attributes:
    * Configuration Validator ([#9779](https://github.com/PyTorchLightning/pytorch-lightning/pull/9779))
- The `current_epoch` and `global_step` attributes now get restored irrespective of the Trainer task ([#9413](https://github.com/PyTorchLightning/pytorch-lightning/pull/9413))
- Trainer now raises an exception when requesting `amp_level` with native `amp_backend` ([#9755](https://github.com/PyTorchLightning/pytorch-lightning/pull/9755))
- Update the logic to check for accumulation steps with deepspeed ([#9826](https://github.com/PyTorchLightning/pytorch-lightning/pull/9826))
- `pytorch_lightning.utilities.grads.grad_norm` now raises an exception if parameter `norm_type <= 0` ([#9765](https://github.com/PyTorchLightning/pytorch-lightning/pull/9765))
- Updated error message for interactive incompatible plugins ([#9896](https://github.com/PyTorchLightning/pytorch-lightning/pull/9896))
- Moved the `optimizer_step` and `clip_gradients` hook from the `Accelerator` and `TrainingTypePlugin` into the `PrecisionPlugin` ([#10143](https://github.com/PyTorchLightning/pytorch-lightning/pull/10143), [#10029](https://github.com/PyTorchLightning/pytorch-lightning/pull/10029))
- `NativeMixedPrecisionPlugin` and its subclasses now take an optional `GradScaler` instance ([#10055](https://github.com/PyTorchLightning/pytorch-lightning/pull/10055))
- Trainer is now raising a `MisconfigurationException` instead of a warning if `Trainer.{validate/test}` is missing required methods ([#10016](https://github.com/PyTorchLightning/pytorch-lightning/pull/10016))
- Changed default value of the `max_steps` Trainer argument from `None` to -1 ([#9460](https://github.com/PyTorchLightning/pytorch-lightning/pull/9460))
- LightningModule now raises an error when calling `log(on_step=False, on_epoch=False)` ([#10227](https://github.com/PyTorchLightning/pytorch-lightning/pull/10227))
- Quantization aware training observers are now disabled by default during validating/testing/predicting stages ([#8540](https://github.com/PyTorchLightning/pytorch-lightning/pull/8540))
- Raised `MisconfigurationException` when total length of `dataloader` across ranks is zero, and give warning when total length is non-zero, but only local rank length is zero. ([#9827](https://github.com/PyTorchLightning/pytorch-lightning/pull/9827))
- Changed the model size calculation using `ByteCounter` ([#10123](https://github.com/PyTorchLightning/pytorch-lightning/pull/10123))
- Enabled `on_load_checkpoint` for `LightningDataModule` for all `trainer_fn` ([#10238](https://github.com/PyTorchLightning/pytorch-lightning/pull/10238))
- Allowed separate config files for parameters with class type when LightningCLI is in `subclass_mode=False` ([#10286](https://github.com/PyTorchLightning/pytorch-lightning/pull/10286))


### Deprecated

- Deprecated Trainer argument `terminate_on_nan` in favor of `detect_anomaly`([#9175](https://github.com/PyTorchLightning/pytorch-lightning/pull/9175))
- Deprecated `Trainer.terminate_on_nan` public attribute access ([#9849](https://github.com/PyTorchLightning/pytorch-lightning/pull/9849))
- Deprecated `LightningModule.summarize()` in favor of `pytorch_lightning.utilities.model_summary.summarize()` ([#8513](https://github.com/PyTorchLightning/pytorch-lightning/pull/8513))
- Deprecated `LightningModule.model_size` ([#8343](https://github.com/PyTorchLightning/pytorch-lightning/pull/8343))
- Deprecated `DataModule` properties: `train_transforms`, `val_transforms`, `test_transforms`, `size`, `dims` ([#8851](https://github.com/PyTorchLightning/pytorch-lightning/pull/8851))
- Deprecated `add_to_queue`, `get_from_queue` from `LightningModule` in favor of corresponding methods in the `DDPSpawnPlugin` ([#9118](https://github.com/PyTorchLightning/pytorch-lightning/pull/9118))
- Deprecated `LightningModule.get_progress_bar_dict` and `Trainer.progress_bar_dict` in favor of `pytorch_lightning.callbacks.progress.base.get_standard_metrics` and `ProgressBarBase.get_metrics` ([#8985](https://github.com/PyTorchLightning/pytorch-lightning/pull/8985))
- Deprecated `prepare_data_per_node` flag on Trainer and set it as a property of `DataHooks`, accessible in the `LightningModule` and `LightningDataModule` ([#8958](https://github.com/PyTorchLightning/pytorch-lightning/pull/8958))
- Deprecated the `TestTubeLogger` ([#9065](https://github.com/PyTorchLightning/pytorch-lightning/pull/9065))
- Deprecated `on_{train/val/test/predict}_dataloader()` from `LightningModule` and `LightningDataModule` ([#9098](https://github.com/PyTorchLightning/pytorch-lightning/pull/9098))
- Deprecated `on_keyboard_interrupt` callback hook in favor of new `on_exception` hook ([#9260](https://github.com/PyTorchLightning/pytorch-lightning/pull/9260))
- Deprecated passing `process_position` to the `Trainer` constructor in favor of adding the `ProgressBar` callback with `process_position` directly to the list of callbacks ([#9222](https://github.com/PyTorchLightning/pytorch-lightning/pull/9222))
- Deprecated passing `flush_logs_every_n_steps` as a Trainer argument, instead pass it to the logger init if supported ([#9366](https://github.com/PyTorchLightning/pytorch-lightning/pull/9366))
- Deprecated `LightningLoggerBase.close`, `LoggerCollection.close` in favor of `LightningLoggerBase.finalize`, `LoggerCollection.finalize` ([#9422](https://github.com/PyTorchLightning/pytorch-lightning/pull/9422))
- Deprecated passing `progress_bar_refresh_rate` to the `Trainer` constructor in favor of adding the `ProgressBar` callback with `refresh_rate` directly to the list of callbacks, or passing `enable_progress_bar=False` to disable the progress bar ([#9616](https://github.com/PyTorchLightning/pytorch-lightning/pull/9616))
- Deprecated `LightningDistributed` and moved the broadcast logic to `DDPPlugin` and `DDPSpawnPlugin` directly ([#9691](https://github.com/PyTorchLightning/pytorch-lightning/pull/9691))
- Deprecated passing `stochastic_weight_avg` to the `Trainer` constructor in favor of adding the `StochasticWeightAveraging` callback directly to the list of callbacks ([#8989](https://github.com/PyTorchLightning/pytorch-lightning/pull/8989))
- Deprecated Accelerator collective API `barrier`, `broadcast`, and `all_gather` in favor of calling the `TrainingTypePlugin` collective API directly ([#9677](https://github.com/PyTorchLightning/pytorch-lightning/pull/9677))
- Deprecated `checkpoint_callback` from the `Trainer` constructor in favor of `enable_checkpointing` ([#9754](https://github.com/PyTorchLightning/pytorch-lightning/pull/9754))
- Deprecated the `LightningModule.on_post_move_to_device` method ([#9525](https://github.com/PyTorchLightning/pytorch-lightning/pull/9525))
- Deprecated `pytorch_lightning.core.decorators.parameter_validation` in favor of `pytorch_lightning.utilities.parameter_tying.set_shared_parameters` ([#9525](https://github.com/PyTorchLightning/pytorch-lightning/pull/9525))
- Deprecated passing `weights_summary` to the `Trainer` constructor in favor of adding the `ModelSummary` callback with `max_depth` directly to the list of callbacks ([#9699](https://github.com/PyTorchLightning/pytorch-lightning/pull/9699))
- Deprecated `log_gpu_memory`, `gpu_metrics`, and util funcs in favor of `DeviceStatsMonitor` callback ([#9921](https://github.com/PyTorchLightning/pytorch-lightning/pull/9921))
- Deprecated `GPUStatsMonitor` and `XLAStatsMonitor` in favor of `DeviceStatsMonitor` callback ([#9924](https://github.com/PyTorchLightning/pytorch-lightning/pull/9924))
- Deprecated setting `Trainer(max_steps=None)`; To turn off the limit, set `Trainer(max_steps=-1)` (default) ([#9460](https://github.com/PyTorchLightning/pytorch-lightning/pull/9460))
- Deprecated access to the `AcceleratorConnector.is_slurm_managing_tasks` attribute and marked it as protected ([#10101](https://github.com/PyTorchLightning/pytorch-lightning/pull/10101))
- Deprecated access to the `AcceleratorConnector.configure_slurm_ddp` method and marked it as protected ([#10101](https://github.com/PyTorchLightning/pytorch-lightning/pull/10101))
- Deprecated passing `resume_from_checkpoint` to the `Trainer` constructor in favor of `trainer.fit(ckpt_path=)` ([#10061](https://github.com/PyTorchLightning/pytorch-lightning/pull/10061))
- Deprecated `ClusterEnvironment.creates_children()` in favor of `ClusterEnvironment.creates_processes_externally` (property) ([#10106](https://github.com/PyTorchLightning/pytorch-lightning/pull/10106))
- Deprecated `PrecisionPlugin.master_params()` in favor of `PrecisionPlugin.main_params()` ([#10105](https://github.com/PyTorchLightning/pytorch-lightning/pull/10105))
- Deprecated `lr_sch_names` from `LearningRateMonitor` ([#10066](https://github.com/PyTorchLightning/pytorch-lightning/pull/10066))
- Deprecated `ProgressBar` callback in favor of `TQDMProgressBar` ([#10134](https://github.com/PyTorchLightning/pytorch-lightning/pull/10134))


### Removed

- Removed deprecated `metrics` ([#8586](https://github.com/PyTorchLightning/pytorch-lightning/pull/8586/))
- Removed the deprecated `outputs` argument in both the `LightningModule.on_train_epoch_end` and `Callback.on_train_epoch_end` hooks ([#8587](https://github.com/PyTorchLightning/pytorch-lightning/pull/8587))
- Removed the deprecated `TrainerLoggingMixin` class ([#8609](https://github.com/PyTorchLightning/pytorch-lightning/pull/8609))
- Removed the deprecated `TrainerTrainingTricksMixin` class ([#8679](https://github.com/PyTorchLightning/pytorch-lightning/pull/8679))
- Removed the deprecated `optimizer_idx` from `training_step` as an accepted argument in manual optimization ([#8576](https://github.com/PyTorchLightning/pytorch-lightning/pull/8576))
- Removed support for the deprecated `on_save_checkpoint` signature. The hook now takes a `checkpoint` positional parameter ([#8697](https://github.com/PyTorchLightning/pytorch-lightning/pull/8697))
- Removed support for the deprecated `on_load_checkpoint` signature. The hook now takes a `pl_module` positional parameter ([#8697](https://github.com/PyTorchLightning/pytorch-lightning/pull/8697))
- Removed the deprecated `save_function` property in `ModelCheckpoint` ([#8680](https://github.com/PyTorchLightning/pytorch-lightning/pull/8680))
- Removed the deprecated `model` argument from `ModelCheckpoint.save_checkpoint` ([#8688](https://github.com/PyTorchLightning/pytorch-lightning/pull/8688))
- Removed the deprecated `sync_step` argument from `WandbLogger` ([#8763](https://github.com/PyTorchLightning/pytorch-lightning/pull/8763))
- Removed the deprecated `Trainer.truncated_bptt_steps` in favor of `LightningModule.truncated_bptt_steps` ([#8826](https://github.com/PyTorchLightning/pytorch-lightning/pull/8826))
- Removed `LightningModule.write_predictions` and `LightningModule.write_predictions_dict` ([#8850](https://github.com/PyTorchLightning/pytorch-lightning/pull/8850))
- Removed `on_reset_*_dataloader` hooks in TrainingType Plugins and Accelerators ([#8858](https://github.com/PyTorchLightning/pytorch-lightning/pull/8858))
- Removed deprecated `GradInformation` module in favor of `pytorch_lightning.utilities.grads` ([#8831](https://github.com/PyTorchLightning/pytorch-lightning/pull/8831/))
- Removed `TrainingTypePlugin.on_save` and `Accelerator.on_save` ([#9023](https://github.com/PyTorchLightning/pytorch-lightning/pull/9023))
- Removed `{Accelerator,TrainingTypePlugin,PrecisionPlugin}.post_optimizer_step` ([#9746](https://github.com/PyTorchLightning/pytorch-lightning/pull/9746))
- Removed deprecated `connect_precision_plugin` and `connect_training_type_plugin` from `Accelerator` ([#9019](https://github.com/PyTorchLightning/pytorch-lightning/pull/9019))
- Removed `on_train_epoch_end` from `Accelerator` ([#9035](https://github.com/PyTorchLightning/pytorch-lightning/pull/9035))
- Removed `InterBatchProcessor` in favor of `DataLoaderIterDataFetcher` ([#9052](https://github.com/PyTorchLightning/pytorch-lightning/pull/9052))
- Removed `Plugin` in `base_plugin.py` in favor of accessing `TrainingTypePlugin` and `PrecisionPlugin` directly instead ([#9066](https://github.com/PyTorchLightning/pytorch-lightning/pull/9066))
- Removed `teardown` from `ParallelPlugin` ([#8943](https://github.com/PyTorchLightning/pytorch-lightning/pull/8943))
- Removed deprecated `profiled_functions` argument from `PyTorchProfiler` ([#9178](https://github.com/PyTorchLightning/pytorch-lightning/pull/9178))
- Removed deprecated `pytorch_lighting.utilities.argparse_utils` module ([#9166](https://github.com/PyTorchLightning/pytorch-lightning/pull/9166))
- Removed deprecated property `Trainer.running_sanity_check` in favor of `Trainer.sanity_checking` ([#9209](https://github.com/PyTorchLightning/pytorch-lightning/pull/9209))
- Removed deprecated `BaseProfiler.output_filename` arg from it and its descendants in favor of `dirpath` and `filename` ([#9214](https://github.com/PyTorchLightning/pytorch-lightning/pull/9214))
- Removed deprecated property `ModelCheckpoint.period` in favor of `ModelCheckpoint.every_n_epochs` ([#9213](https://github.com/PyTorchLightning/pytorch-lightning/pull/9213))
- Removed deprecated `auto_move_data` decorator ([#9231](https://github.com/PyTorchLightning/pytorch-lightning/pull/9231))
- Removed deprecated property `LightningModule.datamodule` in favor of `Trainer.datamodule` ([#9233](https://github.com/PyTorchLightning/pytorch-lightning/pull/9233))
- Removed deprecated properties `DeepSpeedPlugin.cpu_offload*` in favor of `offload_optimizer`, `offload_parameters` and `pin_memory` ([#9244](https://github.com/PyTorchLightning/pytorch-lightning/pull/9244))
- Removed deprecated property `AcceleratorConnector.is_using_torchelastic` in favor of `TorchElasticEnvironment.is_using_torchelastic()` ([#9729](https://github.com/PyTorchLightning/pytorch-lightning/pull/9729))
- Removed `pytorch_lightning.utilities.debugging.InternalDebugger` ([#9680](https://github.com/PyTorchLightning/pytorch-lightning/pull/9680))
- Removed `call_configure_sharded_model_hook` property from `Accelerator` and `TrainingTypePlugin` ([#9612](https://github.com/PyTorchLightning/pytorch-lightning/pull/9612))
- Removed `TrainerProperties` mixin and moved property definitions directly into `Trainer` ([#9495](https://github.com/PyTorchLightning/pytorch-lightning/pull/9495))
- Removed a redundant warning with `ModelCheckpoint(monitor=None)` callback ([#9875](https://github.com/PyTorchLightning/pytorch-lightning/pull/9875))
- Remove `epoch` from `trainer.logged_metrics` ([#9904](https://github.com/PyTorchLightning/pytorch-lightning/pull/9904))
- Removed `should_rank_save_checkpoint` property from Trainer ([#9433](https://github.com/PyTorchLightning/pytorch-lightning/pull/9433))
- Remove deprecated `distributed_backend` from `Trainer` ([#10017](https://github.com/PyTorchLightning/pytorch-lightning/pull/10017))
- Removed `process_idx` from the `{DDPSpawnPlugin,TPUSpawnPlugin}.new_process` methods ([#10022](https://github.com/PyTorchLightning/pytorch-lightning/pull/10022))
- Removed automatic patching of `{train,val,test,predict}_dataloader()` on the `LightningModule` ([#9764](https://github.com/PyTorchLightning/pytorch-lightning/pull/9764))
- Removed `pytorch_lightning.trainer.connectors.OptimizerConnector` ([#10120](https://github.com/PyTorchLightning/pytorch-lightning/pull/10120))


### Fixed

- Fixed ImageNet evaluation in example ([#10179](https://github.com/PyTorchLightning/pytorch-lightning/pull/10179))
- Fixed an issue with logger outputs not being finalized correctly after prediction runs ([#8685](https://github.com/PyTorchLightning/pytorch-lightning/pull/8685))
- Fixed `move_metrics_to_cpu` moving the loss to CPU while training on device ([#9308](https://github.com/PyTorchLightning/pytorch-lightning/pull/9308))
- Fixed incorrect main progress bar indicator when resuming training mid-epoch ([#9310](https://github.com/PyTorchLightning/pytorch-lightning/pull/9310))
- Fixed an issue with freeing memory of datafetchers during teardown ([#9387](https://github.com/PyTorchLightning/pytorch-lightning/pull/9387))
- Fixed a bug where the training step output needed to be `deepcopy`-ed ([#9349](https://github.com/PyTorchLightning/pytorch-lightning/pull/9349))
- Fixed an issue with freeing memory allocated by the data iterators in `Loop.on_run_end` ([#9386](https://github.com/PyTorchLightning/pytorch-lightning/pull/9386), [#9915](https://github.com/PyTorchLightning/pytorch-lightning/pull/9915))
- Fixed `BasePredictionWriter` not returning the batch indices in a non-distributed setting ([#9432](https://github.com/PyTorchLightning/pytorch-lightning/pull/9432))
- Fixed an error when running in XLA environments with no TPU attached ([#9572](https://github.com/PyTorchLightning/pytorch-lightning/pull/9572))
- Fixed check on torchmetrics logged whose `compute()` output is a multielement tensor ([#9582](https://github.com/PyTorchLightning/pytorch-lightning/pull/9582))
- Fixed gradient accumulation for `DDPShardedPlugin` ([#9122](https://github.com/PyTorchLightning/pytorch-lightning/pull/9122))
- Fixed missing DeepSpeed distributed call ([#9540](https://github.com/PyTorchLightning/pytorch-lightning/pull/9540))
- Fixed an issue with wrapped LightningModule during evaluation; The LightningModule no longer gets wrapped with data-parallel modules when not fitting in `DDPPlugin`, `DDPSpawnPlugin`, `DDPShardedPlugin`, `DDPSpawnShardedPlugin` ([#9096](https://github.com/PyTorchLightning/pytorch-lightning/pull/9096))
- Fixed `trainer.accumulate_grad_batches` to be an int on init. The default value for it is now `None` inside Trainer ([#9652](https://github.com/PyTorchLightning/pytorch-lightning/pull/9652))
- Fixed `broadcast` in `DDPPlugin` and `DDPSpawnPlugin` to respect the `src` input ([#9691](https://github.com/PyTorchLightning/pytorch-lightning/pull/9691))
- Fixed `self.log(on_epoch=True, reduce_fx=sum))` for the `on_batch_start` and `on_train_batch_start` hooks ([#9791](https://github.com/PyTorchLightning/pytorch-lightning/pull/9791))
- Fixed `self.log(on_epoch=True)` for the `on_batch_start` and `on_train_batch_start` hooks ([#9780](https://github.com/PyTorchLightning/pytorch-lightning/pull/9780))
- Fixed restoring training state during `Trainer.fit` only ([#9413](https://github.com/PyTorchLightning/pytorch-lightning/pull/9413))
- Fixed DeepSpeed and Lightning both calling the scheduler ([#9788](https://github.com/PyTorchLightning/pytorch-lightning/pull/9788))
- Fixed missing arguments when saving hyperparameters from the parent class but not from the child class ([#9800](https://github.com/PyTorchLightning/pytorch-lightning/pull/9800))
- Fixed DeepSpeed GPU device IDs ([#9847](https://github.com/PyTorchLightning/pytorch-lightning/pull/9847))
- Reset `val_dataloader` in `tuner/batch_size_scaling` ([#9857](https://github.com/PyTorchLightning/pytorch-lightning/pull/9857))
- Fixed use of `LightningCLI` in computer_vision_fine_tuning.py example ([#9934](https://github.com/PyTorchLightning/pytorch-lightning/pull/9934))
- Fixed issue with non-init dataclass fields in `apply_to_collection` ([#9963](https://github.com/PyTorchLightning/pytorch-lightning/issues/9963))
- Reset `val_dataloader` in `tuner/batch_size_scaling` for binsearch ([#9975](https://github.com/PyTorchLightning/pytorch-lightning/pull/9975))
- Fixed logic to check for spawn in dataloader `TrainerDataLoadingMixin._worker_check` ([#9902](https://github.com/PyTorchLightning/pytorch-lightning/pull/9902))
- Fixed `train_dataloader` getting loaded twice when resuming from a checkpoint during `Trainer.fit()` ([#9671](https://github.com/PyTorchLightning/pytorch-lightning/pull/9671))
- Fixed `LearningRateMonitor` logging with multiple param groups optimizer with no scheduler ([#10044](https://github.com/PyTorchLightning/pytorch-lightning/pull/10044))
- Fixed undesired side effects being caused by `Trainer` patching dataloader methods on the `LightningModule` ([#9764](https://github.com/PyTorchLightning/pytorch-lightning/pull/9764))
- Fixed gradients not being unscaled when clipping or logging the gradient norm ([#9287](https://github.com/PyTorchLightning/pytorch-lightning/pull/9287))
- Fixed `on_before_optimizer_step` getting called before the optimizer closure (including backward) has run ([#10167](https://github.com/PyTorchLightning/pytorch-lightning/pull/10167))
- Fixed monitor value in `ModelCheckpoint` getting moved to the wrong device in a special case where it becomes NaN ([#10118](https://github.com/PyTorchLightning/pytorch-lightning/pull/10118))
- Fixed creation of `dirpath` in `BaseProfiler` if it doesn't exist ([#10073](https://github.com/PyTorchLightning/pytorch-lightning/pull/10073))
- Fixed incorrect handling of sigterm ([#10189](https://github.com/PyTorchLightning/pytorch-lightning/pull/10189))
- Fixed bug where `log(on_step=True, on_epoch=True, sync_dist=True)` wouldn't reduce the value on step ([#10227](https://github.com/PyTorchLightning/pytorch-lightning/pull/10227))
- Fixed an issue with `pl.utilities.seed.reset_seed` converting the `PL_SEED_WORKERS` environment variable to `bool` ([#10099](https://github.com/PyTorchLightning/pytorch-lightning/pull/10099))
- Fixed iterating over a logger collection when `fast_dev_run > 0` ([#10232](https://github.com/PyTorchLightning/pytorch-lightning/pull/10232))
- Fixed `batch_size` in `ResultCollection` not being reset to 1 on epoch end ([#10242](https://github.com/PyTorchLightning/pytorch-lightning/pull/10242))
- Fixed `distrib_type` not being set when training plugin instances are being passed to the Trainer ([#10251](https://github.com/PyTorchLightning/pytorch-lightning/pull/10251))

# Contributors

@adamjstewart @akihironitta @alessiobonfiglio @ananthsub @aphedges @awaelchli @bamblebam @Benjamin-Etheredge @borchero @Borda @borisdayma @bryant1410 @carmocca @cowwoc @daniellepintz @danielykim @edward-io @eladsegal @EricWiener @ethanwharris @four4fish @gau-nernst @hankyul2 @HansolEom @himanshu-dutta @I-iBot @jjenniferdai @jstjohn @justusschock @kainoj @kaushikb11 @kingyiusuen @Knarik1 @low5545 @lsqshr @mauvilsa @michele-arrival @nasnoisaac @ninginthecloud @popfido @pre-commit-ci @PuneetDabral @qmpzzpmq @rohitgr7 @ronif @roshikouhai @s-rog @samlurye @SeanNaren @shnela @sidml @stancld @stfwn @tangbinh @tchaton @thepurpleowl @Tshimanga @twsl @victorjoos @VirajBagal @wayi1 @weiji14 @yifuwang @yopknopixx

_If we forgot someone, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.5.0)

---

## 1.4.9: Standard weekly patch release
**Published:** 2021-09-30

## [1.4.9] - 2021-09-30

- Moved the gradient unscaling in `NativeMixedPrecisionPlugin` from `pre_optimizer_step` to `post_backward` ([#9606](https://github.com/PyTorchLightning/pytorch-lightning/pull/9606))
- Fixed gradient unscaling being called too late, causing gradient clipping and gradient norm tracking to be applied incorrectly ([#9606](https://github.com/PyTorchLightning/pytorch-lightning/pull/9606))
- Fixed `lr_find` to generate same results on multiple calls ([#9704](https://github.com/PyTorchLightning/pytorch-lightning/pull/9704))
- Fixed `reset` metrics on validation epoch end ([#9717](https://github.com/PyTorchLightning/pytorch-lightning/pull/9717))
- Fixed input validation for `gradient_clip_val`, `gradient_clip_algorithm`, `track_grad_norm` and `terminate_on_nan` Trainer arguments ([#9595](https://github.com/PyTorchLightning/pytorch-lightning/pull/9595))
- Reset metrics before each task starts ([#9410](https://github.com/PyTorchLightning/pytorch-lightning/pull/9410))


### Contributors

@rohitgr7 @tchaton 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.9)

---

## 1.4.8: Standard weekly patch release
**Published:** 2021-09-22


## [1.4.8] - 2021-09-22

- Fixed error reporting in DDP process reconciliation when processes are launched by an external agent ([#9389](https://github.com/PyTorchLightning/pytorch-lightning/pull/9389))
- Added PL_RECONCILE_PROCESS environment variable to enable process reconciliation regardless of cluster environment settings (#)([#9389](https://github.com/PyTorchLightning/pytorch-lightning/pull/9389))
- Fixed `add_argparse_args` raising `TypeError` when args are typed as `typing.Generic` in Python 3.6 ([#9554](https://github.com/PyTorchLightning/pytorch-lightning/pull/9554))
- Fixed back-compatibility for saving hyperparameters from a single container and inferring its argument name by reverting [#9125](https://github.com/PyTorchLightning/pytorch-lightning/pull/9125) ([#9642](https://github.com/PyTorchLightning/pytorch-lightning/pull/9642))
### Contributors

@ananthsub @akihironitta @awaelchli @carmocca @tchaton 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.8)

---

## 1.4.7: Standard weekly patch release
**Published:** 2021-09-15

## [1.4.7] - 2021-09-14

- Fixed logging of nan parameters ([#9364](https://github.com/PyTorchLightning/pytorch-lightning/pull/9364))
- Fixed `replace_sampler` missing the batch size under specific conditions ([#9367](https://github.com/PyTorchLightning/pytorch-lightning/pull/9367))
- Pass init args to ShardedDataParallel ([#9483](https://github.com/PyTorchLightning/pytorch-lightning/pull/9483))
- Fixed collision of user argument when using ShardedDDP ([#9512](https://github.com/PyTorchLightning/pytorch-lightning/pull/9512))
- Fixed DeepSpeed crash for RNNs ([#9489](https://github.com/PyTorchLightning/pytorch-lightning/pull/9489))

### Contributors

@asanakoy @awaelchli @borisdayma @carmocca @guotuofeng @justusschock @kaushikb11 @rohitgr7 @SeanNaren

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.7)

---

## 1.4.6: Standard weekly patch release
**Published:** 2021-09-10

## [1.4.6] - 2021-09-10

- Fixed an issues with export to ONNX format when a model has multiple inputs ([#8800](https://github.com/PyTorchLightning/pytorch-lightning/pull/8800))
- Removed deprecation warnings being called for `on_{task}_dataloader` ([#9279](https://github.com/PyTorchLightning/pytorch-lightning/pull/9279))
- Fixed save/load/resume from checkpoint for DeepSpeed Plugin ([#8397](https://github.com/PyTorchLightning/pytorch-lightning/pull/8397), [#8644](https://github.com/PyTorchLightning/pytorch-lightning/pull/8644), [#8627](https://github.com/PyTorchLightning/pytorch-lightning/pull/8627))
- Fixed `EarlyStopping` running on train epoch end when `check_val_every_n_epoch>1` is set ([#9156](https://github.com/PyTorchLightning/pytorch-lightning/pull/9156))
- Fixed an issue with logger outputs not being finalized correctly after prediction runs ([#8333](https://github.com/PyTorchLightning/pytorch-lightning/issues/8333))
- Fixed the Apex and DeepSpeed plugin closure running after the `on_before_optimizer_step` hook ([#9288](https://github.com/PyTorchLightning/pytorch-lightning/issues/9288))
- Fixed the Native AMP plugin closure not running with manual optimization ([#9288](https://github.com/PyTorchLightning/pytorch-lightning/issues/9288))
- Fixed bug where data-loading functions where not getting the correct running stage passed ([#8858](https://github.com/PyTorchLightning/pytorch-lightning/pull/8858))
- Fixed intra-epoch evaluation outputs staying in memory when the respective `*_epoch_end` hook wasn't overridden ([#9261](https://github.com/PyTorchLightning/pytorch-lightning/pull/9261))
- Fixed error handling in DDP process reconciliation when `_sync_dir` was not initialized ([#9267](https://github.com/PyTorchLightning/pytorch-lightning/pull/9267))
- Fixed PyTorch Profiler not enabled for manual optimization ([#9316](https://github.com/PyTorchLightning/pytorch-lightning/pull/9316))
- Fixed inspection of other args when a container is specified in `save_hyperparameters` ([#9125](https://github.com/PyTorchLightning/pytorch-lightning/pull/9125))
- Fixed signature of `Timer.on_train_epoch_end` and `StochasticWeightAveraging.on_train_epoch_end` to prevent unwanted deprecation warnings ([#9347](https://github.com/PyTorchLightning/pytorch-lightning/pull/9347))

### Contributors

@ananthsub @awaelchli @Borda @four4fish @justusschock @kaushikb11 @s-rog @SeanNaren @tangbinh @tchaton @xerus 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.6)

---

## 1.4.5: Standard weekly patch release
**Published:** 2021-09-01


## [1.4.5] - 2021-08-31

- Fixed reduction using `self.log(sync_dict=True, reduce_fx={mean,max})` ([#9142](https://github.com/PyTorchLightning/pytorch-lightning/pull/9142))
- Fixed not setting a default value for `max_epochs` if `max_time` was specified on the `Trainer` constructor ([#9072](https://github.com/PyTorchLightning/pytorch-lightning/pull/9072))
- Fixed the CometLogger, no longer modifies the metrics in place. Instead creates a copy of metrics before performing any operations ([#9150](https://github.com/PyTorchLightning/pytorch-lightning/pull/9150))
- Fixed `DDP` "CUDA error: initialization error" due to a `copy` instead of `deepcopy` on `ResultCollection` ([#9239](https://github.com/PyTorchLightning/pytorch-lightning/pull/9239))

### Contributors

@ananthsub @bamblebam @carmocca @daniellepintz @ethanwharris @kaushikb11 @sohamtiwari3120 @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.5)

---

## 1.4.4: Standard weekly patch release
**Published:** 2021-08-24


## [1.4.4] - 2021-08-24

- Fixed a bug in the binary search mode of auto batch size scaling where exception was raised if the first trainer run resulted in OOM ([#8954](https://github.com/PyTorchLightning/pytorch-lightning/pull/8954))
- Fixed a bug causing logging with `log_gpu_memory='min_max'` not working ([#9013](https://github.com/PyTorchLightning/pytorch-lightning/pull/9013))

### Contributors

@SkafteNicki @eladsegal

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_



[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.4)

---

## 1.4.3: Standard weekly patch release
**Published:** 2021-08-23

## [1.4.3] - 2021-08-17

- Fixed plateau scheduler stepping on incomplete epoch ([#8861](https://github.com/PyTorchLightning/pytorch-lightning/pull/8861))
- Fixed infinite loop with `CycleIterator` and multiple loaders ([#8889](https://github.com/PyTorchLightning/pytorch-lightning/pull/8889))
- Fixed `StochasticWeightAveraging` with a list of learning rates not applying them to each param group ([#8747](https://github.com/PyTorchLightning/pytorch-lightning/issues/8747))
- Restore original loaders if replaced by entrypoint ([#8885](https://github.com/PyTorchLightning/pytorch-lightning/pull/8885))
- Fixed lost reference to `_Metadata` object in `ResultMetricCollection` ([#8932](https://github.com/PyTorchLightning/pytorch-lightning/pull/8932))
- Ensure the existence of `DDPPlugin._sync_dir` in `reconciliate_processes` ([#8939](https://github.com/PyTorchLightning/pytorch-lightning/pull/8939))


### Contributors

@awaelchli @carmocca @justusschock @tchaton @yifuwang 
_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.3)

---

## 1.4.2: Standard weekly patch release
**Published:** 2021-08-11

## [1.4.2] - 2021-08-10

- Fixed recursive call for `apply_to_collection(include_none=False)` ([#8719](https://github.com/PyTorchLightning/pytorch-lightning/pull/8719))
- Fixed truncated backprop through time enablement when set as a property on the LightningModule and not the Trainer ([#8804](https://github.com/PyTorchLightning/pytorch-lightning/pull/8804/))
- Fixed comments and exception message for metrics_to_scalars ([#8782](https://github.com/PyTorchLightning/pytorch-lightning/pull/8782/))
- Fixed typo error in LightningLoggerBase.after_save_checkpoint docstring ([#8737](https://github.com/PyTorchLightning/pytorch-lightning/pull/8737/))


### Contributors

@Aiden-Jeon @ananthsub @awaelchli @edward-io
_If we forgot someone due to not matching commit email with GitHub account, let us know :]_


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.2)

---

## 1.4.1: Standard weekly patch release
**Published:** 2021-08-03

## [1.4.1] - 2021-08-03

- Fixed `trainer.fit_loop.split_idx` always returning `None` (#8601)
- Fixed references for `ResultCollection.extra` (#8622)
- Fixed reference issues during epoch end result collection (#8621)
- Fixed horovod auto-detection when horovod is not installed and the launcher is `mpirun` (#8610)
- Fixed an issue with `training_step` outputs not getting collected correctly for `training_epoch_end` (#8613)
- Fixed distributed types support for CPUs (#8667)
- Fixed a deadlock issue with DDP and torchelastic (#8655)
- Fixed `accelerator=ddp` choice for CPU (#8645)


### Contributors

@awaelchli, @borda, @carmocca, @kaushikb11, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.1)

---

## 1.4.0: TPU Pod Training, IPU Accelerator, DeepSpeed Infinity, Fully Sharded Data Parallel
**Published:** 2021-07-27

Today we are excited to announce Lightning 1.4, introducing support for TPU pods, XLA profiling, IPUs, and new plugins to reach 10+ billion parameters, including Deep Speed Infinity, Fully Sharded Data-Parallel and more!

https://devblog.pytorchlightning.ai/announcing-lightning-1-4-8cd20482aee9

## [1.4.0] - 2021-07-27

### Added

- Added `extract_batch_size` utility and corresponding tests to extract batch dimension from multiple batch types (#8357)
- Added support for named parameter groups in `LearningRateMonitor` (#7987)
- Added `dataclass` support for `pytorch_lightning.utilities.apply_to_collection` (#7935)
- Added support to `LightningModule.to_torchscript` for saving to custom filesystems with `fsspec` (#7617)
- Added `KubeflowEnvironment` for use with the `PyTorchJob` operator in Kubeflow
- Added LightningCLI support for config files on object stores (#7521)
- Added `ModelPruning(prune_on_train_epoch_end=True|False)` to choose when to apply pruning (#7704)
- Added support for checkpointing based on a provided time interval during training (#7515)
- Progress tracking
  * Added dataclasses for progress tracking (#6603, #7574, #8140, #8362)
  * Add `{,load_}state_dict` to the progress tracking dataclasses (#8140)
  * Connect the progress tracking dataclasses to the loops (#8244, #8362)
  * Do not reset the progress tracking dataclasses total counters (#8475)
- Added support for passing a `LightningDataModule` positionally as the second argument to `trainer.{validate,test,predict}` (#7431)
- Added argument `trainer.predict(ckpt_path)` (#7430)
- Added `clip_grad_by_value` support for TPUs (#7025)
- Added support for passing any class to `is_overridden` (#7918)
- Added `sub_dir` parameter to `TensorBoardLogger` (#6195)
- Added correct `dataloader_idx` to batch transfer hooks (#6241)
- Added `include_none=bool` argument to `apply_to_collection` (#7769)
- Added `apply_to_collections` to apply a function to two zipped collections (#7769)
- Added `ddp_fully_sharded` support (#7487)
- Added `should_rank_save_checkpoint` property to Training Plugins (#7684)
- Added `log_grad_norm` hook to `LightningModule` to customize the logging of gradient norms (#7873)
- Added `save_config_filename` init argument to `LightningCLI` to ease resolving name conflicts (#7741)
- Added `save_config_overwrite` init argument to `LightningCLI` to ease overwriting existing config files (#8059)
- Added reset dataloader hooks to Training Plugins and Accelerators (#7861)
- Added trainer stage hooks for Training Plugins and Accelerators (#7864)
- Added the `on_before_optimizer_step` hook (#8048)
- Added IPU Accelerator (#7867)
- Fault-tolerant training
    * Added `{,load_}state_dict` to `ResultCollection` (#7948)
    * Added `{,load_}state_dict` to `Loops` (#8197)
    * Set `Loop.restarting=False` at the end of the first iteration (#8362)
    * Save the loops state with the checkpoint (opt-in) (#8362)
    * Save a checkpoint to restore the state on exception (opt-in) (#8362)
    * Added `state_dict` and `load_state_dict` utilities for `CombinedLoader` + utilities for dataloader (#8364)
- Added `rank_zero_only` to `LightningModule.log` function (#7966)
- Added `metric_attribute` to `LightningModule.log` function (#7966)
- Added a warning if `Trainer(log_every_n_steps)` is a value too high for the training dataloader (#7734)
- Added LightningCLI support for argument links applied on instantiation (#7895)
- Added LightningCLI support for configurable callbacks that should always be present (#7964)
- Added DeepSpeed Infinity Support, and updated to DeepSpeed 0.4.0 (#7234)
- Added support for `torch.nn.UninitializedParameter` in `ModelSummary` (#7642)
- Added support `LightningModule.save_hyperparameters` when `LightningModule` is a dataclass (#7992)
- Added support for overriding `optimizer_zero_grad` and `optimizer_step` when using accumulate_grad_batches (#7980)
- Added `logger` boolean flag to `save_hyperparameters` (#7960)
- Added support for calling scripts using the module syntax (`python -m package.script`) (#8073)
- Added support for optimizers and learning rate schedulers to `LightningCLI` (#8093)
- Added XLA Profiler (#8014)
- Added `PrecisionPlugin.{pre,post}_backward` (#8328)
- Added `on_load_checkpoint` and `on_save_checkpoint` hooks to the `PrecisionPlugin` base class (#7831)
- Added `max_depth` parameter in `ModelSummary` (#8062)
- Added `XLAStatsMonitor` callback (#8235)
- Added `restore` function and `restarting` attribute to base `Loop` (#8247)
- Added `FastForwardSampler` and `CaptureIterableDataset` (#8307)
- Added support for `save_hyperparameters` in `LightningDataModule` (#3792)
- Added the `ModelCheckpoint(save_on_train_epoch_end)` to choose when to run the saving logic (#8389)
- Added `LSFEnvironment` for distributed training with the LSF resource manager `jsrun` (#5102)
- Added support for `accelerator='cpu'|'gpu'|'tpu'|'ipu'|'auto'` (#7808)
- Added `tpu_spawn_debug` to plugin registry (#7933)
- Enabled traditional/manual launching of DDP processes through `LOCAL_RANK` and `NODE_RANK` environment variable assignments (#7480)
- Added `quantize_on_fit_end` argument to `QuantizationAwareTraining` (#8464)
- Added experimental support for loop specialization (#8226)
- Added support for `devices` flag to Trainer (#8440)
- Added private `prevent_trainer_and_dataloaders_deepcopy` context manager on the `LightningModule` (#8472)
- Added support for providing callables to the Lightning CLI instead of types (#8400)


### Changed

- Decoupled device parsing logic from Accelerator connector to Trainer (#8180)
- Changed the `Trainer`'s `checkpoint_callback` argument to allow only boolean values (#7539)
- Log epoch metrics before the `on_evaluation_end` hook (#7272)
- Explicitly disallow calling `self.log(on_epoch=False)` during epoch-only or single-call hooks (#7874)
- Changed these `Trainer` methods to be protected: `call_setup_hook`, `call_configure_sharded_model`, `pre_dispatch`, `dispatch`, `post_dispatch`, `call_teardown_hook`, `run_train`, `run_sanity_check`, `run_evaluate`, `run_evaluation`, `run_predict`, `track_output_for_epoch_end`
- Changed `metrics_to_scalars` to work with any collection or value (#7888)
- Changed `clip_grad_norm` to use `torch.nn.utils.clip_grad_norm_` (#7025)
- Validation is now always run inside the training epoch scope (#7357)
- `ModelCheckpoint` now runs at the end of the training epoch by default (#8389)
- `EarlyStopping` now runs at the end of the training epoch by default (#8286)
- Refactored Loops
    * Moved attributes `global_step`, `current_epoch`, `max/min_steps`, `max/min_epochs`, `batch_idx`, and `total_batch_idx` to TrainLoop (#7437)
    * Refactored result handling in training loop (#7506)
    * Moved attributes `hiddens` and `split_idx` to TrainLoop (#7507)
    * Refactored the logic around manual and automatic optimization inside the optimizer loop (#7526)
    * Simplified "should run validation" logic (#7682)
    * Simplified logic for updating the learning rate for schedulers (#7682)
    * Removed the `on_epoch` guard from the "should stop" validation check (#7701)
    * Refactored internal loop interface; added new classes `FitLoop`, `TrainingEpochLoop`, `TrainingBatchLoop` (#7871, #8077)
    * Removed `pytorch_lightning/trainer/training_loop.py` (#7985)
    * Refactored evaluation loop interface; added new classes `DataLoaderLoop`, `EvaluationLoop`, `EvaluationEpochLoop` (#7990, #8077)
    * Removed `pytorch_lightning/trainer/evaluation_loop.py` (#8056)
    * Restricted public access to several internal functions (#8024)
    * Refactored trainer `_run_*` functions and separate evaluation loops (#8065)
    * Refactored prediction loop interface; added new classes `PredictionLoop`, `PredictionEpochLoop` (#7700, #8077)
    * Removed `pytorch_lightning/trainer/predict_loop.py` (#8094)
    * Moved result teardown to the loops (#8245)
    * Improve `Loop` API to better handle children `state_dict` and `progress` (#8334)
- Refactored logging
    * Renamed and moved `core/step_result.py` to `trainer/connectors/logger_connector/result.py` (#7736)
    * Dramatically simplify the `LoggerConnector` (#7882)
    * `trainer.{logged,progress_bar,callback}_metrics` are now updated on-demand (#7882)
    * Completely overhaul the `Result` object in favor of `ResultMetric` (#7882)
    * Improve epoch-level reduction time and overall memory usage (#7882)
    * Allow passing `self.log(batch_size=...)` (#7891)
    * Each of the training loops now keeps its own results collection (#7891)
    * Remove `EpochResultStore` and `HookResultStore` in favor of `ResultCollection` (#7909)
    * Remove `MetricsHolder` (#7909)
- Moved `ignore_scalar_return_in_dp` warning suppression to the DataParallelPlugin class (#7421)
- Changed the behaviour when logging evaluation step metrics to no longer append `/epoch_*` to the metric name (#7351)
- Raised `ValueError` when a `None` value is `self.log`-ed (#7771)
- Changed `resolve_training_type_plugins` to allow setting `num_nodes` and `sync_batchnorm` from `Trainer` setting (#7026)
- Default `seed_everything(workers=True)` in the `LightningCLI` (#7504)
- Changed `model.state_dict()` in `CheckpointConnector` to allow `training_type_plugin` to customize the model's `state_dict()` (#7474)
- `MLflowLogger` now uses the env variable `MLFLOW_TRACKING_URI` as default tracking URI (#7457)
- Changed `Trainer` arg and functionality from `reload_dataloaders_every_epoch` to `reload_dataloaders_every_n_epochs` (#5043)
- Changed `WandbLogger(log_model={True/'all'})` to log models as artifacts (#6231)
- MLFlowLogger now accepts `run_name` as an constructor argument (#7622)
- Changed `teardown()` in `Accelerator` to allow `training_type_plugin` to customize `teardown` logic (#7579)
- `Trainer.fit` now raises an error when using manual optimization with unsupported features such as `gradient_clip_val` or `accumulate_grad_batches` (#7788)
- Accelerator hooks are called regardless if `LightningModule` overrides the same hooks (#7826)
- Moved profilers to their own file (#7822)
- The `on_after_backward` hook is now called on accumulating iterations. Use the `on_before_optimizer_step` hook to mimic the old behaviour (#8328)
- The mixed precision loss is no longer unscaled before the `on_after_backward` hook. Use the `on_before_optimizer_step` hook to mimic the old behaviour  (#8328)
- The `TrainingTypePlugin.{pre,post}_backward` hooks no longer take the `optimizer, opt_idx, should_accumulate` arguments (#8328)
- The `PrecisionPlugin.backward` hooks no longer returns a value (#8328)
- The `PrecisionPlugin.backward` hooks no longer takes a `should_accumulate` argument (#8328)
- Added the `on_before_backward` hook (#7865)
- `LightningCLI` now aborts with a clearer message if config already exists and disables save config during `fast_dev_run`(#7963)
- Saved the `LightningCLI` config on `setup` and only on the main process (#8017)
- Dropped the `LightningCLI` `ArgumentParser` when pickling (#8017)
- Skip `broadcast` if distributed not initialized for the spawn plugins (#8017)
- `Trainer(resume_from_checkpoint=...)` now restores the model directly after `LightningModule.setup()`, which is before `LightningModule.configure_sharded_model()` (#7652)
- Moved `torch.cuda.set_device()` to enable collective calls earlier in setup (#8312)
- Used XLA utility API to move data to CPU (Single TPU core) (#8078)
- Improved error messages in `replace_sampler` when the `DataLoader` attributes are not included in the signature or the signature is missing optional arguments (#8519)
- Moved `DeviceDtypeModuleMixin` and `HyperparametersMixin` mixin to `core` (#8396)
- Return the `default_root_dir` as the `log_dir` when the logger is a `LoggerCollection` (#8187)


### Deprecated

- Deprecated `LightningModule.loaded_optimizer_states_dict` (#8229)
- Standardized the dataloaders arguments of `trainer.{fit,valdiate,test,tune}` (#7431)
- Deprecated `DataModule` properties: `has_prepared_data`, `has_setup_fit`, `has_setup_validate`, `has_setup_test`, `has_setup_predict`, `has_teardown_fit`, `has_teardown_validate`, `has_teardown_test`, `has_teardown_predict` (#7657)
- Deprecated `TrainerModelHooksMixin` in favor of `pytorch_lightning.utilities.signature_utils` (#7422)
- Deprecated `num_nodes` and `sync_batchnorm` arguments in `DDPPlugin` and `DDPSpawnPlugin` (#7026)
- Deprecated `self.log(sync_dist_op)` in favor of `self.log(reduce_fx)`. (#7891)
- Deprecated `is_overridden(model=...)` in favor of `is_overridden(instance=...)` (#7918)
- Deprecated automatically detaching returned extras with grads (#7994)
- Deprecated default value of `monitor` argument in EarlyStopping callback to enforce `monitor` as a required argument (#7907)
- Deprecated importing `rank_zero_{warn,deprecation}` directly from `pytorch_lightning.utilities.distributed` (#8085)
- Deprecated the use of `CheckpointConnector.hpc_load()` in favor of `CheckpointConnector.restore()` (#7652)
- Deprecated `ModelCheckpoint(every_n_val_epochs)` in favor of `ModelCheckpoint(every_n_epochs)` (#8383)
- Deprecated `DDPPlugin.task_idx` in favor of `DDPPlugin.local_rank` (#8203)
- Deprecated the `Trainer.train_loop` property in favor of `Trainer.fit_loop` (#8025)
- Deprecated the `Trainer.disable_validation` property in favor of `not Trainer.enable_validation` (#8291)
- Deprecated `mode` parameter in `ModelSummary` in favor of `max_depth` (#8062)
- Deprecated `reload_dataloaders_every_epoch` argument of `Trainer` in favor of `reload_dataloaders_every_n_epochs` (#5043)
- Deprecated `distributed_backend` argument for `Trainer` (#8575)


### Removed

- Dropped official support/testing for PyTorch <1.6 (#8288)
- Removed `ProfilerConnector` (#7654)
- Pruned deprecated classif. metrics from `pytorch_lightning.metrics.functional.classification` (#7499)
- Removed deprecated data parallel classes `LightningDataParallel` and `LightningDistributedDataParallel` from `pytorch_lightning.overrides.data_parallel` (#7510)
- Removed deprecated trainer attributes - `get_model` and `accelerator_backend` (#7502)
- Removed support for automatically monitoring the `val_loss` key with `ModelCheckpoint`. Pass your `monitor` of choice to the `ModelCheckpoint` instance instead (#8293)
- Removed support for `self.log(tbptt_reduce_fx)` and `self.log(tbptt_pad_token)`. Please, open a discussion explaining your use-case if you relied on these. (#7644)
- Removed deprecated utils modules `model_utils`, `warning_utils`, `xla_device_utils` and partially `argparse_utils` (#7503)
- Removed `RPCPlugin` and `RPCSequentialPlugin`. If you were successfully using these plugins, please open a GitHub discussion about your use case (#8101)
- Removed deprecated trainer attributes - `on_cpu`, `on_tpu`, `use_tpu`, `on_gpu`, `use_dp`, `use_ddp`, `use_ddp2`, `use_horovod`, `use_single_gpu` (#7501)
- Removed deprecated `optimizer` argument in `LightningModule.manual_backward()`; Toggling optimizers in manual optimization should be done using `LightningModule.{un}toggle_optimizer()` (#8287)
- Removed DeepSpeed FP16 Exception as FP32 is now supported (#8462)
- Removed environment variable `PL_EXP_VERSION` from DDP subprocesses (#7403)


### Fixed

- Fixed the `GPUStatsMonitor` callbacks to use the correct GPU IDs if `CUDA_VISIBLE_DEVICES` set (#8260)
- Fixed `lr_scheduler` checkpointed state by calling `update_lr_schedulers` before saving checkpoints (#7877)
- Fixed ambiguous warning when both overfit and train dataloader shuffling are enabled (#7685)
- Fixed dev debugger memory growing due to tracking events even when disabled (#7875)
- Fixed `None` loss keys getting added in `training_epoch_end` when using manual optimization and not returning a loss (#7772)
- Fixed a bug where `precision=64` with `accelerator='ddp_spawn'` would throw a pickle error (#6924)
- Do not override the existing `epoch` value in `logged_metrics` when already logged by the user (#7982)
- Support for manual optimization with DeepSpeed (#7970)
- Fixed `dataloader_idx` argument value when predicting with only one `DataLoader` (#7941)
- Fixed passing the `stage` argument of `Callback.{setup,teardown}` as a keyword (#7973)
- Fixed metrics generated during `validation sanity checking` are cleaned on end (#8171)
- Fixed `log_gpu_memory` metrics not being added to `logging` when nothing else is logged (#8174)
- Fixed a bug where calling `log` with a `Metric` instance would raise an error if it was a nested attribute of the model (#8181)
- Fixed a bug where using `precision=64` would cause buffers with complex dtype to be cast to real (#8208)
- Fixed `is_overridden` returning true for wrapped functions with no changes (#8296)
- Fixed a bug where `truncated_bptt_steps` would throw an AttributeError when the target RNN has multiple hidden states (#8145)
- Fixed `self.optimizers()` not returning a single optimizer if it had been wrapped (#8326)
- Fixed the `on_after_backward` hook not getting called when using manual optimization and no plugins (#8328)
- Fixed the `LightningModule.backward` hook only getting called with the `apex` plugin when using manual optimization (#8328)
- Fixed moving batch to device before sending it to the `on_*_batch_start`/`on_*_batch_end` callbacks and model hooks (#7378)
- Fixed passing a custom `DDPPlugin` when choosing `accelerator="ddp_cpu"` for the accelerator (#6208)
- Fixed missing call to `LightningModule.untoggle_optimizer` in training loop when running gradient accumulation with multiple optimizers (#8284)
- Fixed hash of LightningEnum to work with value instead of name (#8421).
- Fixed a bug where an extra checkpoint was saved at the end of training if the `val_check_interval` did not align with the number of training batches (#7724)
- Fixed hash of LightningEnum to work with value instead of name(#8421).
- Fixed `move_data_to_device` to return the batch if the object `to` function didn't return `self` (#8433)
- Fixed progress bar updates for Pod Training (#8258)
- Fixed clearing dataloader references before attaching new dataloaders in consecutive `Trainer.{fit,validate,test,predict}´ runs (#8442)
- Fixed memory leaks on GPU by moving `optimizer_states`, `ResultCollection.extra`, `ResultMetric` attributes, and `LoggerConnector` metrics to `cpu`. Also, delete the DDP wrapper on `teardown` (#8490)
- Fixed `SWA` callback using LightningModule `prevent_trainer_and_dataloaders_deepcopy` to avoid OOM (#8472)
- Fixed `ModelPruning` callback `on_save_checkpoint` to avoid making a `deepcopy` potentially leading to OOM (#8472)
- Fixed the sampler replacement logic for `DataLoader`s which do not define all `DataLoader` attributes as `__init__` parameters (#8519)
- Fixed DeepSpeed Windows support (#8488)
- Fixed DeepSpeed not properly setting the trainer `lr_schedulers` attribute (#8527)
- Fixed experiment version and log-dir divergence in DDP when using multiple `Trainer` instances in sequence (#7403)
- Enabled manual optimization for TPUs (#8458)
- Fixed `accumulate_grad_batches` not been recomputed during model reload (#5334)
- Fixed a `TypeError` when wrapping optimizers in the `HorovodPlugin` and running `Trainer.test` (#7840)
- Fixed `BackboneFinetuning` restoration (#8501)
- Fixed `lr_scheduler` with metric (e.g. `torch.optim.lr_scheduler.ReduceLROnPlateau`) when using `automatic_optimization = False` (#7643)
- Fixed `DeepSpeed` breaking with no schedulers ([#8580](https://github.com/PyTorchLightning/pytorch-lightning/pull/8580))


### Contributors

@00sapo @AffineParameter @ajtritt @akihironitta @ananthsub @aniketmaurya @aslisabanci @awaelchli @bamblebam @Borda @borisdayma @carmocca @dalek-who @DavidMChan @davors72 @dcfidalgo @ddrevicky @deepsource-autofix @djthegr8 @edenlightning @edgarriba @eladsegal @ethanwharris @eugeneh101 @fepegar @gaoteng-git @gtauzin @i-aki-y @janhenriklambrechts @jiwidi @justusschock @karthikrangasai @kaushikb11 @loic-beheshti @Lucklyric @ManuelPalermo @mauvilsa @maxoppelt @neggert @nikvaessen @nisheethlahoti @pre-commit-ci @rohitgr7 @ruotianluo @satishjasthi @SeanNaren @shirayu @shuyingsunshine21 @sid-sundrani @Sileadim @simran2905 @stancld @t-vi @tchaton @theblackfly @theodumont @tilman151 @tomy0000000 @tshu-w @vatch123 @WrRan @yifuwang

_If we forgot someone, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.4.0)

---

## 1.3.8: Standard weekly patch release
**Published:** 2021-07-01

## [1.3.8] - 2021-07-01

### Fixed

- Fixed a sync deadlock when checkpointing a `LightningModule` that uses a torchmetrics 0.4 `Metric` ([#8218](https://github.com/PyTorchLightning/pytorch-lightning/pull/8218))
- Fixed compatibility TorchMetrics v0.4 ([#8206](https://github.com/PyTorchLightning/pytorch-lightning/pull/8206))
- Added torchelastic check when sanitizing GPUs ([#8095](https://github.com/PyTorchLightning/pytorch-lightning/pull/8095))
- Fixed a DDP info message that was never shown ([#8111](https://github.com/PyTorchLightning/pytorch-lightning/pull/8111))
- Fixed metrics deprecation message at module import level ([#8163](https://github.com/PyTorchLightning/pytorch-lightning/pull/8163))
- Fixed a bug where an infinite recursion would be triggered when using the `BaseFinetuning` callback on a model that contains a `ModuleDict` ([#8170](https://github.com/PyTorchLightning/pytorch-lightning/pull/8170))
- Added a mechanism to detect `deadlock` for `DDP` when only 1 process trigger an `Exception`. The mechanism will `kill the processes` when it happens ([#8167](https://github.com/PyTorchLightning/pytorch-lightning/pull/8167))
- Fixed NCCL error when selecting non-consecutive device ids ([#8165](https://github.com/PyTorchLightning/pytorch-lightning/pull/8165))
- Fixed SWA to also work with `IterableDataset` ([#8172](https://github.com/PyTorchLightning/pytorch-lightning/pull/8172))


### Contributors

@GabrielePicco @SeanNaren @ethanwharris @carmocca @tchaton @justusschock 

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.8)

---

## 1.3.7post0: Hotfix Patch Release
**Published:** 2021-06-23

## [1.3.7post0] - 2021-06-23

### Fixed

- Fixed backward compatibility of moved functions `rank_zero_warn` and `rank_zero_deprecation` ([#8085](https://github.com/PyTorchLightning/pytorch-lightning/pull/8085))

### Contributors

@kaushikb11 @carmocca 

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.7post0)

---

## 1.3.7: Standard weekly patch release
**Published:** 2021-06-22

## [1.3.7] - 2021-06-22

### Fixed

- Fixed a bug where skipping an optimizer while using amp causes amp to trigger an assertion error ([#7975](https://github.com/PyTorchLightning/pytorch-lightning/pull/7975))
This conversation was marked as resolved by carmocca
- Fixed deprecation messages not showing due to incorrect stacklevel ([#8002](https://github.com/PyTorchLightning/pytorch-lightning/pull/8002), [#8005](https://github.com/PyTorchLightning/pytorch-lightning/pull/8005))
- Fixed setting a `DistributedSampler` when using a distributed plugin in a custom accelerator ([#7814](https://github.com/PyTorchLightning/pytorch-lightning/pull/7814))
- Improved `PyTorchProfiler` chrome traces names ([#8009](https://github.com/PyTorchLightning/pytorch-lightning/pull/8009))
- Fixed moving the best score to device in `EarlyStopping` callback for TPU devices ([#7959](https://github.com/PyTorchLightning/pytorch-lightning/pull/7959))


### Contributors

@yifuwang @kaushikb11 @ajtritt @carmocca @tchaton 

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.7)

---

## 1.3.6: Standard weekly patch release
**Published:** 2021-06-17

## [1.3.6] - 2021-06-15

### Fixed

- Fixed logs overwriting issue for remote filesystems ([#7889](https://github.com/PyTorchLightning/pytorch-lightning/pull/7889))
- Fixed `DataModule.prepare_data` could only be called on the global rank 0 process ([#7945](https://github.com/PyTorchLightning/pytorch-lightning/pull/7945))
- Fixed setting `worker_init_fn` to seed dataloaders correctly when using DDP ([#7942](https://github.com/PyTorchLightning/pytorch-lightning/pull/7942))
- Fixed `BaseFinetuning` callback to properly handle parent modules w/ parameters ([#7931](https://github.com/PyTorchLightning/pytorch-lightning/pull/7931))

### Contributors

@awaelchli @Borda @kaushikb11 @Queuecumber @SeanNaren @senarvi @speediedan

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.6)

---

## 1.3.5: Standard weekly patch release
**Published:** 2021-06-09

## [1.3.5] - 2021-06-08

### Added

- Added warning to Training Step output ([#7779](https://github.com/PyTorchLightning/pytorch-lightning/pull/7779))

### Fixed

- Fixed LearningRateMonitor + BackboneFinetuning ([#7835](https://github.com/PyTorchLightning/pytorch-lightning/pull/7835))
- Minor improvements to `apply_to_collection` and type signature of `log_dict` ([#7851](https://github.com/PyTorchLightning/pytorch-lightning/pull/7851))
- Fixed docker versions ([#7834](https://github.com/PyTorchLightning/pytorch-lightning/pull/7834))
- Fixed sharded training check for fp16 precision ([#7825](https://github.com/PyTorchLightning/pytorch-lightning/pull/7825))
- Fixed support for torch Module type hints in LightningCLI ([#7807](https://github.com/PyTorchLightning/pytorch-lightning/pull/7807))

### Changed

- Move `training_output` validation to after `train_step_end` ([#7868](https://github.com/PyTorchLightning/pytorch-lightning/pull/7868))

### Contributors

@Borda, @justusschock, @kandluis, @mauvilsa, @shuyingsunshine21, @tchaton

*If we forgot someone due to not matching commit email with GitHub account, let us know :]*

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.5)

---

## 1.3.4: Standard weekly patch release
**Published:** 2021-06-03

## [1.3.4] - 2021-06-01

### Fixed

- Fixed info message when max training time reached ([#7780](https://github.com/PyTorchLightning/pytorch-lightning/pull/7780))
- Fixed missing `__len__` method to `IndexBatchSamplerWrapper` ([#7681](https://github.com/PyTorchLightning/pytorch-lightning/pull/7681))

### Contributors

@awaelchli @kaushikb11 

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.4)

---

## 1.3.3: Standard weekly patch release
**Published:** 2021-05-26

## [1.3.3] - 2021-05-26

### Changed

- Changed calling of `untoggle_optimizer(opt_idx)` out of the closure function (#7563)

### Fixed

- Fixed `ProgressBar` pickling after calling `trainer.predict` (#7608)
- Fixed broadcasting in multi-node, multi-gpu DDP using torch 1.7 (#7592)
- Fixed dataloaders are not reset when tuning the model (#7566)
- Fixed print errors in `ProgressBar` when `trainer.fit` is not called (#7674)
- Fixed global step update when the epoch is skipped (#7677)
- Fixed training loop total batch counter when accumulate grad batches was enabled (#7692)

### Contributors

@carmocca @kaushikb11 @ryanking13 @Lucklyric @ajtritt @yifuwang

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.3)

---

## 1.3.2: Standard weekly patch release
**Published:** 2021-05-19

## [1.3.2] - 2021-05-18

### Changed

- `DataModule`s now avoid duplicate `{setup,teardown,prepare_data}` calls for the same stage (#7238)

### Fixed

- Fixed parsing of multiple training dataloaders (#7433)
- Fixed recursive passing of `wrong_type` keyword argument in `pytorch_lightning.utilities.apply_to_collection` (#7433)
- Fixed setting correct `DistribType` for `ddp_cpu` (spawn) backend (#7492)
- Fixed incorrect number of calls to LR scheduler when `check_val_every_n_epoch > 1` (#7032)

### Contributors

@alanhdu @carmocca @justusschock @tkng

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.2)

---

## 1.3.1: Standard weekly patch release
**Published:** 2021-05-11

## [1.3.1] - 2021-05-11

### Fixed

- Fixed DeepSpeed with IterableDatasets (#7362)
- Fixed `Trainer.current_epoch` not getting restored after tuning (#7434)
- Fixed local rank displayed in console log (#7395)

### Contributors

@akihironitta @awaelchli @leezu

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.1)

---

## 1.3.0: Lightning CLI, PyTorch Profiler, Improved Early Stopping
**Published:** 2021-05-06

Today we are excited to announce Lightning 1.3, containing highly anticipated new features including a new Lightning CLI, improved TPU support, integrations such as PyTorch profiler, new early stopping strategies, predict and validate trainer routines, and more.

https://medium.com/pytorch/pytorch-lightning-1-3-lightning-cli-pytorch-profiler-improved-early-stopping-6e0ffd8deb29

## [1.3.0] - 2021-05-06

### Added

- Added support for the `EarlyStopping` callback to run at the end of the training epoch (#6944)
- Added synchronization points before and after `setup` hooks are run (#7202)
- Added a `teardown` hook to `ClusterEnvironment` (#6942)
- Added utils for metrics to scalar conversions (#7180)
- Added utils for NaN/Inf detection for gradients and parameters (#6834)
- Added more explicit exception message when trying to execute `trainer.test()` or `trainer.validate()` with `fast_dev_run=True` (#6667)
- Added `LightningCLI` class to provide simple reproducibility with minimum boilerplate training CLI (#4492, #6862, #7156, #7299)
- Added `gradient_clip_algorithm` argument to Trainer for gradient clipping by value (#6123).
- Added a way to print to terminal without breaking up the progress bar (#5470)
- Added support to checkpoint after training steps in `ModelCheckpoint` callback (#6146)
- Added `TrainerStatus.{INITIALIZING,RUNNING,FINISHED,INTERRUPTED}` (#7173)
- Added `Trainer.validate()` method to perform one evaluation epoch over the validation set (#4948)
- Added `LightningEnvironment` for Lightning-specific DDP (#5915)
- Added `teardown()` hook to LightningDataModule (#4673)
- Added `auto_insert_metric_name` parameter to `ModelCheckpoint` (#6277)
- Added arg to `self.log` that enables users to give custom names when dealing with multiple dataloaders (#6274)
- Added `teardown` method to `BaseProfiler` to enable subclasses defining post-profiling steps outside of `__del__` (#6370)
- Added `setup` method to `BaseProfiler` to enable subclasses defining pre-profiling steps for every process (#6633)
- Added no return warning to predict (#6139)
- Added `Trainer.predict` config validation (#6543)
- Added `AbstractProfiler` interface (#6621)
- Added support for including module names for forward in the autograd trace of `PyTorchProfiler` (#6349)
- Added support for the PyTorch 1.8.1 autograd profiler (#6618)
- Added `outputs` parameter to callback's `on_validation_epoch_end` & `on_test_epoch_end` hooks (#6120)
- Added `configure_sharded_model` hook (#6679)
- Added support for `precision=64`, enabling training with double precision (#6595)
- Added support for DDP communication hooks (#6736)
- Added `artifact_location` argument to `MLFlowLogger` which will be passed to the `MlflowClient.create_experiment` call (#6677)
- Added `model` parameter to precision plugins' `clip_gradients` signature (#6764, #7231)
- Added `is_last_batch` attribute to `Trainer` (#6825)
- Added `LightningModule.lr_schedulers()` for manual optimization  (#6567)
- Added `MpModelWrapper` in TPU Spawn (#7045)
- Added `max_time` Trainer argument to limit training time (#6823)
- Added `on_predict_{batch,epoch}_{start,end}` hooks (#7141)
- Added new `EarlyStopping` parameters `stopping_threshold` and `divergence_threshold` (#6868)
- Added `debug` flag to TPU Training Plugins (PT_XLA_DEBUG) (#7219)
- Added new `UnrepeatedDistributedSampler` and `IndexBatchSamplerWrapper` for tracking distributed predictions (#7215)
- Added `trainer.predict(return_predictions=None|False|True)` (#7215)
- Added `BasePredictionWriter` callback to implement prediction saving (#7127)
- Added `trainer.tune(scale_batch_size_kwargs, lr_find_kwargs)` arguments to configure the tuning algorithms (#7258)
- Added `tpu_distributed` check for TPU Spawn barrier (#7241)
- Added device updates to TPU Spawn for Pod training (#7243)
- Added warning when missing `Callback` and using `resume_from_checkpoint` (#7254)
- DeepSpeed single file saving (#6900)
- Added Training type Plugins Registry (#6982, #7063, #7214, #7224)
- Add `ignore` param to `save_hyperparameters` (#6056)

### Changed

- Changed `LightningModule.truncated_bptt_steps` to be property (#7323)
- Changed `EarlyStopping` callback from by default running `EarlyStopping.on_validation_end` if only training is run. Set `check_on_train_epoch_end` to run the callback at the end of the train epoch instead of at the end of the validation epoch (#7069)
- Renamed `pytorch_lightning.callbacks.swa` to `pytorch_lightning.callbacks.stochastic_weight_avg` (#6259)
- Refactor `RunningStage` and `TrainerState` usage (#4945, #7173)
    * Added `RunningStage.SANITY_CHECKING`
    * Added `TrainerFn.{FITTING,VALIDATING,TESTING,PREDICTING,TUNING}`
    * Changed `trainer.evaluating` to return `True` if validating or testing
- Changed `setup()` and `teardown()` stage argument to take any of `{fit,validate,test,predict}` (#6386)
- Changed profilers to save separate report files per state and rank (#6621)
- The trainer no longer tries to save a checkpoint on exception or run callback's `on_train_end` functions (#6864)
- Changed `PyTorchProfiler` to use `torch.autograd.profiler.record_function` to record functions (#6349)
- Disabled `lr_scheduler.step()` in manual optimization  (#6825)
- Changed warnings and recommendations for dataloaders in `ddp_spawn` (#6762)
- `pl.seed_everything` will now also set the seed on the `DistributedSampler` (#7024)
- Changed default setting for communication of multi-node training using `DDPShardedPlugin` (#6937)
- `trainer.tune()` now returns the tuning result (#7258)
- `LightningModule.from_datasets()` now accepts `IterableDataset` instances as training datasets. (#7503)
- Changed `resume_from_checkpoint` warning to an error when the checkpoint file does not exist (#7075)
- Automatically set `sync_batchnorm` for `training_type_plugin` (#6536)
- Allowed training type plugin to delay optimizer creation (#6331)
- Removed ModelSummary validation from train loop on_trainer_init (#6610)
- Moved `save_function` to accelerator (#6689)
- Updated DeepSpeed ZeRO (#6546, #6752, #6142, #6321)
- Improved verbose logging for `EarlyStopping` callback (#6811)
- Run ddp_spawn dataloader checks on Windows (#6930)
- Updated mlflow with using `resolve_tags` (#6746)
- Moved `save_hyperparameters` to its own function (#7119)
- Replaced `_DataModuleWrapper` with `__new__` (#7289)
- Reset `current_fx` properties on lightning module in teardown (#7247)
- Auto-set `DataLoader.worker_init_fn` with `seed_everything` (#6960)
- Remove `model.trainer` call inside of dataloading mixin (#7317)
- Split profilers module (#6261)
- Ensure accelerator is valid if running interactively (#5970)
- Disabled batch transfer in DP mode (#6098)

### Deprecated

- Deprecated `outputs` in both `LightningModule.on_train_epoch_end` and `Callback.on_train_epoch_end` hooks (#7339)
- Deprecated `Trainer.truncated_bptt_steps` in favor of `LightningModule.truncated_bptt_steps` (#7323)
- Deprecated `outputs` in both `LightningModule.on_train_epoch_end` and `Callback.on_train_epoch_end` hooks (#7339)
- Deprecated `LightningModule.grad_norm` in favor of `pytorch_lightning.utilities.grads.grad_norm` (#7292)
- Deprecated the `save_function` property from the `ModelCheckpoint` callback (#7201)
- Deprecated `LightningModule.write_predictions` and `LightningModule.write_predictions_dict` (#7066)
- Deprecated `TrainerLoggingMixin` in favor of a separate utilities module for metric handling (#7180)
- Deprecated `TrainerTrainingTricksMixin` in favor of a separate utilities module for NaN/Inf detection for gradients and parameters (#6834)
- `period` has been deprecated in favor of `every_n_val_epochs` in the `ModelCheckpoint` callback (#6146)
- Deprecated `trainer.running_sanity_check` in favor of `trainer.sanity_checking` (#4945)
- Deprecated `Profiler(output_filename)` in favor of `dirpath` and `filename` (#6621)
- Deprecated `PytorchProfiler(profiled_functions)` in favor of `record_functions` (#6349)
- Deprecated `@auto_move_data` in favor of `trainer.predict` (#6993)
- Deprecated `Callback.on_load_checkpoint(checkpoint)` in favor of `Callback.on_load_checkpoint(trainer, pl_module, checkpoint)` (#7253)
- Deprecated metrics in favor of `torchmetrics` (#6505, #6530, #6540, #6547, #6515, #6572, #6573, #6584, #6636, #6637, #6649, #6659, #7131)
- Deprecated the `LightningModule.datamodule` getter and setter methods; access them through `Trainer.datamodule` instead (#7168)
- Deprecated the use of `Trainer(gpus="i")` (string) for selecting the i-th GPU; from v1.5 this will set the number of GPUs instead of the index (#6388)

### Removed

- Removed the `exp_save_path` property from the `LightningModule` (#7266)
- Removed training loop explicitly calling `EarlyStopping.on_validation_end` if no validation is run (#7069)
- Removed `automatic_optimization` as a property from the training loop in favor of `LightningModule.automatic_optimization` (#7130)
- Removed evaluation loop legacy returns for `*_epoch_end` hooks (#6973)
- Removed support for passing a bool value to `profiler` argument of Trainer (#6164)
- Removed no return warning from val/test step (#6139)
- Removed passing a `ModelCheckpoint` instance to `Trainer(checkpoint_callback)` (#6166)
- Removed deprecated Trainer argument `enable_pl_optimizer` and `automatic_optimization` (#6163)
- Removed deprecated metrics (#6161)
    * from `pytorch_lightning.metrics.functional.classification` removed `to_onehot`, `to_categorical`, `get_num_classes`, `roc`, `multiclass_roc`, `average_precision`, `precision_recall_curve`, `multiclass_precision_recall_curve`
    * from `pytorch_lightning.metrics.functional.reduction` removed `reduce`, `class_reduce`
- Removed deprecated `ModelCheckpoint` arguments `prefix`, `mode="auto"` (#6162)
- Removed `mode='auto'` from `EarlyStopping` (#6167)
- Removed `epoch` and `step` arguments from `ModelCheckpoint.format_checkpoint_name()`, these are now included in the `metrics` argument (#7344)
- Removed legacy references for magic keys in the `Result` object (#6016)
- Removed deprecated `LightningModule` `hparams` setter (#6207)
- Removed legacy code to log or include metrics in the progress bar by returning them in a dict with the `"log"/"progress_bar"` magic keys. Use `self.log` instead (#6734)
- Removed `trainer.fit()` return value of `1`. It has no return now (#7237)
- Removed `logger_connector` legacy code (#6733)
- Removed unused mixin attributes (#6487)

### Fixed

- Fixed NaN errors in progress bars when training with iterable datasets with no length defined (#7306)
- Fixed attaching train and validation dataloaders when `reload_dataloaders_every_epoch=True` and `num_sanity_val_steps=0` (#7207)
- Added a barrier in the accelerator `teardown` to synchronize processes before execution finishes (#6814)
- Fixed multi-node DDP sub-process launch by using `local_rank` instead of `global_rank` for main process assertion (#7061)
- Fixed incorrect removal of `WORLD_SIZE` environment variable in DDP training when launching with torch distributed/torchelastic (#6942)
- Made the `Plugin.reduce` method more consistent across all Plugins to reflect a mean-reduction by default (#6011)
- Move lightning module to correct device type when using LightningDistributedWrapper (#6070)
- Do not print top-k verbose log with `ModelCheckpoint(monitor=None)` (#6109)
- Fixed `ModelCheckpoint(save_top_k=0, save_last=True)` not saving the `last` checkpoint (#6136)
- Fixed `.teardown(stage='fit')` and `.on_fit_{start,end}()` getting called during `trainer.test` (#6386)
- Fixed LightningModule `all_gather` on cpu tensors (#6416)
- Fixed torch distributed not available in setup hook for DDP (#6506)
- Fixed `trainer.tuner.{lr_find,scale_batch_size}` not setting the `Trainer` state properly (#7258)
- Fixed bug where the learning rate schedulers did not follow the optimizer frequencies (#4868)
- Fixed pickle error checker to now check for `pickle.PickleError` to catch all pickle errors (#6917)
- Fixed a bug where the outputs object passed to `LightningModule.training_epoch_end` was different from the object passed to the `on_train_end_epoch` hook (#6969)
- Fixed a bug where the outputs passed to `train_batch_end` would be listed even when using a single optimizer and no truncated backprop through time steps (#6969)
- Fixed bug for trainer error handling which would cause hang for distributed training (#6864)
- Fixed `self.device` not returning the correct device in replicas of data-parallel (#6414)
- Fixed `lr_find` trying beyond `num_training` steps and suggesting a too high learning rate (#7076)
- Fixed logger creating incorrect version folder in DDP with repeated `Trainer.fit` calls (#7077)
- Fixed metric objects passed directly to `self.log` not being reset correctly (#7055)
- Fixed `CombinedLoader` in distributed settings for validation / testing (#7102)
- Fixed the save_dir in `WandbLogger` when the run was initiated externally (#7106)
- Fixed `num_sanity_val_steps` affecting reproducibility of training data shuffling (#7014)
- Fixed resetting device after `fitting/evaluating/predicting` (#7188)
- Fixed bug where `trainer.tuner.scale_batch_size(max_trials=0)` would not return the correct batch size result (#7262)
- Fixed metrics not being properly logged with `precision=16` and `manual_optimization` (#7228)
- Fixed `BaseFinetuning` properly reloading `optimizer_states` when using `resume_from_checkpoint` (#6891)
- Fixed `parameters_to_ignore` not properly set to DDPWrapper (#7239)
- Fixed parsing of `fast_dev_run=True` with the built-in `ArgumentParser` (#7240)
- Fixed handling an `IterableDataset` that fails to produce a batch at the beginning of an epoch (#7294)
- Fixed `LightningModule.save_hyperparameters()` when attempting to save an empty container (#7268)
- Fixed `apex` not properly instantiated when running with `ddp` (#7274)
- Fixed optimizer `state` not moved to `GPU` (#7277)
- Fixed custom init args for `WandbLogger` (#6989)
- Fixed a bug where an error would be raised if the train dataloader sometimes produced None for a batch (#7342)
- Fixed examples (#6600, #6638, #7096, #7246, #6357, #6476, #6294, #6373, #6088, #7398)
- Resolved schedule step bug for PyTorch Profiler (#6674, #6681)
- Updated logic for checking TPUs availability (#6767)
- Resolve TPU miss rendezvous (#6781)
- Fixed auto-scaling mode when calling tune method on trainer (#7321)
- Fixed finetuning complex models correctly unfreezes (#6880)
- Ensure we set the eval/train flag correctly on accelerator model (#6877)
- Set better defaults for `rank_zero_only.rank` when training is launched with SLURM and torchelastic (#6802)
- Fixed matching the number of outputs of backward with forward for AllGatherGrad (#6625)
- Fixed the `gradient_clip_algorithm` has no effect (#6928)
- Fixed CUDA OOM detection and handling (#6934)
- Fixed `unfreeze_and_add_param_group` expects `modules` rather than `module` (#6822)
- Fixed DPP + SyncBN when move on device (#6838)
- Fixed missing arguments in `lr_find` call (#6784)
- Fixed `set_default_tensor_type` to `torch.DoubleTensor` with precision=64 (#7108)
- Fixed `NeptuneLogger.log_text(step=None)` (#7194)
- Fixed importing torchtext batch (#6365, #6323, #6211)


### Contributors

@akihironitta, @alessiobonfiglio, @amisev, @amogkam, @ananthsub, @ArvinZhuang, @ashleve, @asnorkin, @awaelchli, @BloodAxe, @bmahlbrand, @Borda, @borisdayma, @camruta, @carmocca, @ceshine, @dbonner, @dhkim0225, @EdwardJB, @EliaCereda, @EricCousineau-TRI, @ethanwharris, @FlorianMF, @hemildesai, @ifsheldon, @kaushikb11, @mauvilsa, @maxfrei750, @mesejo, @ramonemiliani93, @rohitgr7, @s-rog, @sadiqj, @scart97, @SeanNaren, @shuyingsunshine21, @SkafteNicki, @SpontaneousDuck, @stllfe, @tchaton, @THasthika, @vballoli

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.3.0)

---

## 1.2.10: Quick patch release
**Published:** 2021-04-23

Fixing missing `packaging` package in dependencies, which was affecting the only installation to a very blank system.

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.10)

---

## 1.2.9: Standard weekly patch release
**Published:** 2021-04-22

## [1.2.9] - 2021-04-20

### Fixed

- Fixed the order to call for world ranks & the `root_device` property in `TPUSpawnPlugin` ([#7074](https://github.com/PyTorchLightning/pytorch-lightning/pull/7074))
- Fixed multi-gpu join for Horovod ([#6954](https://github.com/PyTorchLightning/pytorch-lightning/pull/6954))
- Fixed parsing for pre-release package versions ([#6999](https://github.com/PyTorchLightning/pytorch-lightning/pull/6999))

### Contributors

@irasit @Borda @kaushikb11 

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.9)

---

## 1.2.8: Standard weekly patch release
**Published:** 2021-04-14

## [1.2.8] - 2021-04-14

### Added

- Added TPUSpawn + IterableDataset error message ([#6875](https://github.com/PyTorchLightning/pytorch-lightning/pull/6875))

### Fixed

- Fixed process rank not being available right away after `Trainer` instantiation ([#6941](https://github.com/PyTorchLightning/pytorch-lightning/pull/6941))
- Fixed `sync_dist` for tpus ([#6950](https://github.com/PyTorchLightning/pytorch-lightning/pull/6950))
- Fixed `AttributeError for `require_backward_grad_sync` when running manual optimization with sharded plugin ([#6915](https://github.com/PyTorchLightning/pytorch-lightning/pull/6915))
- Fixed `--gpus` default for parser returned by `Trainer.add_argparse_args` ([#6898](https://github.com/PyTorchLightning/pytorch-lightning/pull/6898))
- Fixed TPU Spawn all gather ([#6896](https://github.com/PyTorchLightning/pytorch-lightning/pull/6896))
- Fixed `EarlyStopping` logic when `min_epochs` or `min_steps` requirement is not met ([#6705](https://github.com/PyTorchLightning/pytorch-lightning/pull/6705))
- Fixed csv extension check ([#6436](https://github.com/PyTorchLightning/pytorch-lightning/pull/6436))
- Fixed checkpoint issue when using Horovod distributed backend ([#6958](https://github.com/PyTorchLightning/pytorch-lightning/pull/6958))
- Fixed tensorboard exception raising ([#6901](https://github.com/PyTorchLightning/pytorch-lightning/pull/6901))
- Fixed setting the eval/train flag correctly on accelerator model ([#6983](https://github.com/PyTorchLightning/pytorch-lightning/pull/6983))
- Fixed DDP_SPAWN compatibility with bug_report_model.py ([#6892](https://github.com/PyTorchLightning/pytorch-lightning/pull/6892))
- Fixed bug where `BaseFinetuning.flatten_modules()` was duplicating leaf node parameters ([#6879](https://github.com/PyTorchLightning/pytorch-lightning/pull/6879))
- Set better defaults for `rank_zero_only.rank` when training is launched with SLURM and torchelastic:
    * Support SLURM and torchelastic global rank environment variables ([#5715](https://github.com/PyTorchLightning/pytorch-lightning/pull/5715))
    * Remove hardcoding of local rank in accelerator connector ([#6878](https://github.com/PyTorchLightning/pytorch-lightning/pull/6878))

### Contributors

@ananthsub @awaelchli @ethanwharris @justusschock @kandluis @kaushikb11 @liob @SeanNaren @skmatz

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.8)

---

## 1.2.7: Standard weekly patch release
**Published:** 2021-04-07

## [1.2.7] - 2021-04-06

### Fixed

- Fixed resolve a bug with omegaconf and `xm.save` (#6741)
- Fixed an issue with IterableDataset when __len__ is not defined (#6828)
- Sanitize None params during pruning (#6836)
- Enforce an epoch scheduler interval when using SWA (#6588)
- Fixed TPU Colab hang issue, post training (#6816])
- Fixed a bug where `TensorBoardLogger` would give a warning and not log correctly to a symbolic link `save_dir` (#6730)


### Contributors

@awaelchli, @ethanwharris, @karthikprasad, @kaushikb11, @mibaumgartner, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.7)

---

## 1.2.6: Standard weekly patch release
**Published:** 2021-03-30

## [1.2.6] - 2021-03-30

### Changed

- Changed the behavior of `on_epoch_start` to run at the beginning of validation & test epoch (#6498)

### Removed

- Removed legacy code to include `step` dictionary returns in `callback_metrics`. Use `self.log_dict` instead. (#6682)

### Fixed

- Fixed `DummyLogger.log_hyperparams` raising a `TypeError` when running with `fast_dev_run=True` (#6398)
- Fixed error on TPUs when there was no `ModelCheckpoint` (#6654)
- Fixed `trainer.test` freeze on TPUs (#6654)
- Fixed a bug where gradients were disabled after calling `Trainer.predict` (#6657)
- Fixed bug where no TPUs were detected in a TPU pod env (#6719)


### Contributors

@awaelchli, @carmocca, @ethanwharris, @kaushikb11, @rohitgr7, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.6)

---

## 1.2.5: Weekly patch release - torchmetrics compatibility
**Published:** 2021-03-24

## [1.2.5] - 2021-03-23

### Changed

- Added Autocast in validation, test and predict modes for Native AMP (#6565)
- Update Gradient Clipping for the TPU Accelerator (#6576)
- Refactored setup for typing friendly (#6590)

### Fixed

- Fixed a bug where `all_gather` would not work correctly with `tpu_cores=8` (#6587)
- Fixed comparing required versions (#6434)
- Fixed duplicate logs appearing in console when using the python logging module (#6275)


### Contributors

@awaelchli, @Borda, @ethanwharris, @justusschock, @kaushikb11

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.5)

---

## 1.2.4: Standard weekly patch release
**Published:** 2021-03-16

## [1.2.4] - 2021-03-16


### Changed

- Changed the default of `find_unused_parameters` back to `True` in DDP and DDP Spawn (#6438)

### Fixed

- Expose DeepSpeed loss parameters to allow users to fix loss instability (#6115)
- Fixed DP reduction with collection (#6324)
- Fixed an issue where the tuner would not tune the learning rate if also tuning the batch size (#4688)
- Fixed broadcast to use PyTorch `broadcast_object_list` and add `reduce_decision` (#6410)
- Fixed logger creating directory structure too early in DDP (#6380)
- Fixed DeepSpeed additional memory use on rank 0 when default device not set early enough (#6460)
- Fixed `DummyLogger.log_hyperparams` raising a `TypeError` when running with `fast_dev_run=True` (#6398)
- Fixed an issue with `Tuner.scale_batch_size` not finding the batch size attribute in the datamodule (#5968)
- Fixed an exception in the layer summary when the model contains torch.jit scripted submodules (#6511)
- Fixed when Train loop config was run during `Trainer.predict` (#6541)

### Contributors
@awaelchli, @kaushikb11, @Palzer, @SeanNaren, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.4)

---

## 1.2.3: Standard weekly patch release
**Published:** 2021-03-09

## [1.2.3] - 2021-03-09

### Fixed

- Fixed `ModelPruning(make_pruning_permanent=True)` pruning buffers getting removed when saved during training (#6073)
- Fixed when `_stable_1d_sort` to work when `n >= N` (#6177)
- Fixed `AttributeError` when `logger=None` on TPU (#6221)
- Fixed PyTorch Profiler with `emit_nvtx` (#6260)
- Fixed `trainer.test` from `best_path` hangs after calling `trainer.fit`  (#6272)
- Fixed `SingleTPU` calling `all_gather` (#6296)
- Ensure we check deepspeed/sharded in multinode DDP (#6297)
- Check `LightningOptimizer` doesn't delete optimizer hooks (#6305)
- Resolve memory leak for evaluation (#6326)
- Ensure that clip gradients is only called if the value is greater than 0 (#6330)
- Fixed `Trainer` not resetting `lightning_optimizers` when calling `Trainer.fit()` multiple times (#6372)

### Contributors
@awaelchli, @carmocca, @Chizuchizu, @frankier, @SeanNaren, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.3)

---

## 1.2.2: Standard weekly patch release
**Published:** 2021-03-05

## [1.2.2] - 2021-03-02

### Added

- Added `checkpoint` parameter to callback's `on_save_checkpoint` hook (#6072)

### Changed

- Changed the order of `backward`, `step`, `zero_grad` to `zero_grad`, `backward`, `step` (#6147)
- Changed default for DeepSpeed CPU Offload to False, due to prohibitively slow speeds at smaller scale (#6262)

### Fixed

- Fixed epoch level schedulers not being called when `val_check_interval < 1.0` (#6075)
- Fixed multiple early stopping callbacks (#6197)
- Fixed incorrect usage of `detach()`, `cpu()`, `to()` (#6216)
- Fixed LBFGS optimizer support which didn't converge in automatic optimization (#6147)
- Prevent `WandbLogger` from dropping values (#5931)
- Fixed error thrown when using valid distributed mode in multi node (#6297)


### Contributors
@akihironitta, @borisdayma, @carmocca, @dvolgyes, @SeanNaren, @SkafteNicki

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.2)

---

## 1.2.1: Standard weekly patch release
**Published:** 2021-02-24

## [1.2.1] - 2021-02-23

### Fixed

- Fixed incorrect yield logic for the amp autocast context manager (#6080)
- Fixed priority of plugin/accelerator when setting distributed mode (#6089)
- Fixed error message for AMP + CPU incompatibility (#6107)

### Contributors
@awaelchli, @SeanNaren, @carmocca

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.1)

---

## 1.2.0: Pruning & Quantization & SWA
**Published:** 2021-02-18

## [1.2.0] - 2021-02-18

### Added

- Added `DataType`, `AverageMethod` and `MDMCAverageMethod` enum in metrics (#5657)
- Added support for summarized model total params size in megabytes (#5590)
- Added support for multiple train loaders (#1959)
- Added `Accuracy` metric now generalizes to Top-k accuracy for (multi-dimensional) multi-class inputs using the `top_k` parameter (#4838)
- Added `Accuracy` metric now enables the computation of subset accuracy for multi-label or multi-dimensional multi-class inputs with the `subset_accuracy` parameter (#4838)
- Added `HammingDistance` metric to compute the hamming distance (loss) (#4838)
- Added `max_fpr` parameter to `auroc` metric for computing partial auroc metric (#3790)
- Added `StatScores` metric to compute the number of true positives, false positives, true negatives and false negatives (#4839)
- Added `R2Score` metric (#5241)
- Added `LambdaCallback` (#5347)
- Added `BackboneLambdaFinetuningCallback` (#5377)
- Accelerator `all_gather` supports collection (#5221)
- Added `image_gradients` functional metric to compute the image gradients of a given input image. (#5056)
- Added `MetricCollection` (#4318)
- Added `.clone()` method to metrics (#4318)
- Added `IoU` class interface (#4704)
- Support to tie weights after moving model to TPU via `on_post_move_to_device` hook
- Added missing val/test hooks in `LightningModule` (#5467)
- The `Recall` and `Precision` metrics (and their functional counterparts `recall` and `precision`) can now be generalized to Recall@K and Precision@K with the use of `top_k` parameter (#4842)
- Added `ModelPruning` Callback (#5618, #5825, #6045)
- Added `PyTorchProfiler` (#5560)
- Added compositional metrics (#5464)
- Added Trainer method `predict(...)` for high performence predictions (#5579)
- Added `on_before_batch_transfer` and `on_after_batch_transfer` data hooks (#3671)
- Added AUC/AUROC class interface (#5479)
- Added `PredictLoop` object (#5752)
- Added `QuantizationAwareTraining` callback (#5706, #6040)
- Added `LightningModule.configure_callbacks` to enable the definition of model-specific callbacks (#5621)
- Added `dim` to `PSNR` metric for mean-squared-error reduction (#5957)
- Added promxial policy optimization template to pl_examples (#5394)
- Added `log_graph` to `CometLogger` (#5295)
- Added possibility for nested loaders (#5404)
- Added `sync_step` to Wandb logger (#5351)
- Added `StochasticWeightAveraging` callback (#5640)
- Added `LightningDataModule.from_datasets(...)` (#5133)
- Added `PL_TORCH_DISTRIBUTED_BACKEND` env variable to select backend (#5981)
- Added `Trainer` flag to activate Stochastic Weight Averaging (SWA) `Trainer(stochastic_weight_avg=True)` (#6038)
- Added DeepSpeed integration (#5954, #6042)

### Changed

- Changed `stat_scores` metric now calculates stat scores over all classes and gains new parameters, in line with the new `StatScores` metric (#4839)
- Changed `computer_vision_fine_tunning` example to use `BackboneLambdaFinetuningCallback` (#5377)
- Changed `automatic casting` for LoggerConnector `metrics` (#5218)
- Changed `iou` [func] to allow float input (#4704)
- Metric `compute()` method will no longer automatically call `reset()` ([#5409](https://github.com/PyTorchLightning/pytorch-lightning/pull/5409/))
- Set PyTorch 1.4 as min requirements, also for testing and examples `torchvision>=0.5` and `torchtext>=0.5` (#5418)
- Changed `callbacks` argument in `Trainer` to allow `Callback` input (#5446)
- Changed the default of `find_unused_parameters` to `False` in DDP (#5185)
- Changed `ModelCheckpoint` version suffixes to start at 1 (#5008)
- Progress bar metrics tensors are now converted to float (#5692)
- Changed the default value for the `progress_bar_refresh_rate` Trainer argument in Google COLAB notebooks to 20 (#5516)
- Extended support for purely iteration-based training (#5726)
- Made `LightningModule.global_rank`, `LightningModule.local_rank` and `LightningModule.logger` read-only properties (#5730)
- Forced `ModelCheckpoint` callbacks to run after all others to guarantee all states are saved to the checkpoint (#5731)
- Refactored Accelerators and Plugins (#5743)
    * Added base classes for plugins (#5715)
    * Added parallel plugins for DP, DDP, DDPSpawn, DDP2 and Horovod (#5714)
    * Precision Plugins (#5718)
    * Added new Accelerators for CPU, GPU and TPU (#5719)
    * Added Plugins for TPU training (#5719)
    * Added RPC and Sharded plugins (#5732)
    * Added missing `LightningModule`-wrapper logic to new plugins and accelerator (#5734)
    * Moved device-specific teardown logic from training loop to accelerator (#5973)
    * Moved accelerator_connector.py to the connectors subfolder (#6033)
    * Trainer only references accelerator (#6039)
    * Made parallel devices optional across all plugins (#6051)
    * Cleaning (#5948, #5949, #5950)
- Enabled `self.log` in callbacks (#5094)
- Renamed xxx_AVAILABLE as protected (#5082)
- Unified module names in Utils (#5199)
- Separated utils: imports & enums (#5256, #5874)
- Refactor: clean trainer device & distributed getters (#5300)
- Simplified training phase as LightningEnum (#5419)
- Updated metrics to use LightningEnum (#5689)
- Changed the seq of `on_train_batch_end`, `on_batch_end` & `on_train_epoch_end`, `on_epoch_end hooks` (#5688)
- Refactored `setup_training` and remove `test_mode` (#5388)
- Disabled training with zero `num_training_batches` when insufficient `limit_train_batches` (#5703)
- Refactored `EpochResultStore` (#5522)
- Update `lr_finder` to check for attribute if not running `fast_dev_run` (#5990)
- LightningOptimizer manual optimizer is more flexible and expose `toggle_model` (#5771)
- `MlflowLogger` limit parameter value length to 250 char (#5893)
- Re-introduced fix for Hydra directory sync with multiple process (#5993)

### Deprecated

- Function `stat_scores_multiple_classes` is deprecated in favor of `stat_scores` (#4839)
- Moved accelerators and plugins to its `legacy` pkg (#5645)
- Deprecated `LightningDistributedDataParallel` in favor of new wrapper module `LightningDistributedModule` (#5185)
- Deprecated `LightningDataParallel` in favor of new wrapper module `LightningParallelModule` (#5670)
- Renamed utils modules (#5199)
    * `argparse_utils` >> `argparse`
    * `model_utils` >> `model_helpers`
    * `warning_utils` >> `warnings`
    * `xla_device_utils` >> `xla_device`
- Deprecated using `'val_loss'` to set the `ModelCheckpoint` monitor (#6012)
- Deprecated `.get_model()` with explicit `.lightning_module` property (#6035)
- Deprecated Trainer attribute `accelerator_backend` in favor of `accelerator` (#6034)

### Removed

- Removed deprecated checkpoint argument `filepath` (#5321)
- Removed deprecated `Fbeta`, `f1_score` and `fbeta_score` metrics (#5322)
- Removed deprecated `TrainResult` (#5323)
- Removed deprecated `EvalResult` (#5633)
- Removed `LoggerStages` (#5673)

### Fixed

- Fixed distributed setting and `ddp_cpu` only with `num_processes>1` (#5297)
- Fixed the saved filename in `ModelCheckpoint` when it already exists (#4861)
- Fixed `DDPHPCAccelerator` hangs in DDP construction by calling `init_device` (#5157)
- Fixed `num_workers` for Windows example (#5375)
- Fixed loading yaml (#5619)
- Fixed support custom DataLoader with DDP if they can be re-instantiated (#5745)
- Fixed repeated `.fit()` calls ignore max_steps iteration bound (#5936)
- Fixed throwing `MisconfigurationError` on unknown mode (#5255)
- Resolve bug with Finetuning (#5744)
- Fixed `ModelCheckpoint` race condition in file existence check (#5155)
- Fixed some compatibility with PyTorch 1.8 (#5864)
- Fixed forward cache (#5895)
- Fixed recursive detach of tensors to CPU (#6007)
- Fixed passing wrong strings for scheduler interval doesn't throw an error (#5923)
- Fixed wrong `requires_grad` state after `return None` with multiple optimizers (#5738)
- Fixed add `on_epoch_end` hook at the end of `validation`, `test` epoch (#5986)
- Fixed missing `process_dataloader` call for `TPUSpawn` when in distributed mode (#6015)
- Fixed progress bar flickering by appending 0 to floats/strings (#6009)
- Fixed synchronization issues with TPU training (#6027)
- Fixed `hparams.yaml` saved twice when using `TensorBoardLogger` (#5953)
- Fixed basic examples (#5912, #5985)
- Fixed `fairscale` compatible with PT 1.8 (#5996)
- Ensured `process_dataloader` is called when `tpu_cores > 1` to use Parallel DataLoader (#6015)
- Attempted SLURM auto resume call when non-shell call fails (#6002)
- Fixed wrapping optimizers upon assignment (#6006)
- Fixed allowing hashing of metrics with lists in their state (#5939)

### Contributors
@alanhdu, @ananthsub, @awaelchli, @Borda, @borisdayma, @carmocca, @ddrevicky, @deng-cy, @ducthienbui97, @justusschock, @kartik4949, @kaushikb11, @manipopopo, @marload, @neighthan, @peblair, @prampey, @pranjaldatta, @rohitgr7, @SeanNaren, @sid-sundrani, @SkafteNicki, @tadejsv, @tchaton, @teddykoker, @titu1994, @yuntai

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.2.0)

---

## 1.1.8: Standard weekly patch release
**Published:** 2021-02-08

## [1.1.8] - 2021-02-08

### Fixed

- Separate epoch validation from step validation (#5208)
- Fixed `toggle_optimizers` not handling all optimizer parameters (#5775)

### Contributors
@ananthsub, @rohitgr7

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.8)

---

## 1.1.7: Standard weekly patch release
**Published:** 2021-02-03

## [1.1.7] - 2021-02-03

### Fixed

- Fixed `TensorBoardLogger` not closing `SummaryWriter` on `finalize` (#5696)
- Fixed filtering of pytorch  "unsqueeze" warning when using DP (#5622)
- Fixed `num_classes` argument in F1 metric (#5663)
- Fixed `log_dir` property (#5537)
- Fixed a race condition in `ModelCheckpoint` when checking if a checkpoint file exists (#5144)
- Remove unnecessary intermediate layers in Dockerfiles (#5697)
- Fixed auto learning rate ordering (#5638)

### Contributors
@awaelchli @guillochon @noamzilo @rohitgr7 @SkafteNicki @sumanthratna

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.7)

---

## 1.1.6: Standard weekly patch release
**Published:** 2021-01-26

## [1.1.6] - 2021-01-26

### Changed

- Increased TPU check timeout from 20s to 100s (#5598)
- Ignored `step` param in Neptune logger's log_metric method (#5510)
- Pass batch outputs to `on_train_batch_end` instead of `epoch_end` outputs (#4369)

### Fixed

- Fixed `toggle_optimizer` to reset `requires_grad` state  (#5574)
- Fixed FileNotFoundError for best checkpoint when using DDP with Hydra (#5629)
- Fixed an error when logging a progress bar metric with a reserved name (#5620)
- Fixed `Metric`'s `state_dict` not included when child modules (#5614)
- Fixed Neptune logger creating multiple experiments when GPUs > 1 (#3256) 
- Fixed duplicate logs appearing in console when using the python logging module (#5509)
- Fixed tensor printing in `trainer.test()` (#5138)
- Fixed not using dataloader when `hparams` present (#4559)

### Contributors
@awaelchli @bryant1410 @lezwon @manipopopo @PiotrJander @psinger @rnett @SeanNaren @swethmandava @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.6)

---

## 1.1.5: Standard weekly patch release
**Published:** 2021-01-21

## [1.1.5] - 2021-01-19

### Fixed

- Fixed a visual bug in the progress bar display initialization (#4579)
- Fixed logging `on_train_batch_end` in a callback with multiple optimizers (#5521)
- Fixed `reinit_scheduler_properties` with correct optimizer (#5519)
- Fixed `val_check_interval` with `fast_dev_run` (#5540)

## Contributors

@awaelchli, @carmocca, @rohitgr7

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.5)

---

## 1.1.4: Standard weekly patch release
**Published:** 2021-01-12

## [1.1.4] - 2021-01-12

### Added

- Add automatic optimization property setter to lightning module ([#5169](https://github.com/PyTorchLightning/pytorch-lightning/pull/5169))

### Changed

- Changed deprecated `enable_pl_optimizer=True` ([#5244](https://github.com/PyTorchLightning/pytorch-lightning/pull/5244))

### Fixed

- Fixed `transfer_batch_to_device` for DDP with `len(devices_ids) == 1` ([#5195](https://github.com/PyTorchLightning/pytorch-lightning/pull/5195))
- Logging only on `not should_accumulate()` during training ([#5417](https://github.com/PyTorchLightning/pytorch-lightning/pull/5417))
- Resolve interpolation bug with Hydra ([#5406](https://github.com/PyTorchLightning/pytorch-lightning/pull/5406)) 
- Check environ before selecting a seed to prevent warning message ([#4743](https://github.com/PyTorchLightning/pytorch-lightning/pull/4743))

### Contributors
@ananthsub, @SeanNaren, @tchaton

If we forgot someone due to not matching commit email with GitHub account, let us know :]

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.4)

---

## 1.1.3: Standard weekly patch release
**Published:** 2021-01-06

## [1.1.3] - 2021-01-05

### Added

- Added a check for optimizer attached to `lr_scheduler` (#5338)
- Added support for passing non-existing `filepaths` to `resume_from_checkpoint` (#4402)

### Changed

- Skip restore from `resume_from_checkpoint` while `testing` (#5161)
- Allowed `log_momentum` for adaptive optimizers in `LearningRateMonitor` (#5333)
- Disabled checkpointing, earlystopping and logging with `fast_dev_run` (#5277)
- Distributed group defaults to `WORLD` if `None` (#5125)

### Fixed

- Fixed `trainer.test` returning non-test metrics (#5214)
- Fixed metric state reset (#5273)
- Fixed `--num-nodes` on `DDPSequentialPlugin` (#5327)
- Fixed invalid value for `weights_summary` (#5296)
- Fixed `Trainer.test` not using the latest `best_model_path` (#5161)
- Fixed existence check for `hparams` not using underlying filesystem (#5250)
- Fixed `LightningOptimizer` AMP bug (#5191)
- Fixed casted key to string in `_flatten_dict` (#5354)

## Contributors

@8greg8, @haven-jeon, @kandluis, @marload, @rohitgr7, @tadejsv, @tarepan, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.3)

---

## 1.1.2: standard weekly patch release
**Published:** 2020-12-23

## Overview

## Detail changes

### Added

- Support number for logging with `sync_dist=True` (#5080)
- Added offset logging step when resuming for Wandb logger (#5050)

### Removed

- `enable_pl_optimizer=False` by default to temporarily fix AMP issues (#5163)

### Fixed

- Metric reduction with Logging (#5150)
- Remove nan loss in manual optimization (#5121)
- Un-balanced logging properly supported (#5119)
- Fix hanging in DDP HPC accelerators (#5157)
- Fix saved filename in `ModelCheckpoint` if it already exists (#4861)
- Fix reset `TensorRunningAccum` (#5106)
- Updated `DALIClassificationLoader` to not use deprecated arguments (#4925)
- Corrected call to `torch.no_grad` (#5124)

## Contributors

@8greg8, @ananthsub, @borisdayma, @gan3sh500, @rohitgr7, @SeanNaren, @tchaton, @VinhLoiIT

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.2)

---

## 1.1.1: standard weekly patch release
**Published:** 2020-12-15

## Overview

## Detail changes

### Added

- Add a notebook example to reach a quick baseline of ~94% accuracy on CIFAR10 using Resnet in Lightning (#4818)

### Changed

- Simplify accelerator steps (#5015)
- Refactor load in checkpoint connector (#4593)

### Removed

- Drop duplicate metrics (#5014)
- Remove beta arg from F1 class and functional (#5076)

### Fixed

- Fixed trainer by default `None` in `DDPAccelerator` (#4915)
- Fixed `LightningOptimizer` to expose optimizer attributes (#5095)
- Do not warn when the `name` key is used in the `lr_scheduler` dict (#5057)
- Check if optimizer supports closure (#4981)
- Extend LightningOptimizer to exposure underlying Optimizer attributes + update doc (#5095)
- Add deprecated metric utility functions back to functional (#5067, #5068)
- Allow any input in `to_onnx` and `to_torchscript` (#4378)
- Do not warn when the name key is used in the `lr_scheduler` dict (#5057)

## Contributors

@Borda, @carmocca, @hemildesai, @rohitgr7, @s-rog, @tarepan, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.1)

---

## 1.1.0: Model Parallelism Training and More Logging Options
**Published:** 2020-12-10

## Overview

Lightning 1.1 is out! You can now train models with twice the parameters and zero code changes with the new sharded model training! We also have a new plugin for sequential model parallelism, more logging options, and a lot of improvements!
Release highlights: https://bit.ly/3gyLZpP

Learn more about sharded training: https://bit.ly/2W3hgI0

## Detail changes

### Added

- Added "monitor" key to saved `ModelCheckpoints` (#4383)
- Added `ConfusionMatrix` class interface (#4348)
- Added multiclass AUROC metric (#4236)
- Added global step indexing to the checkpoint name for a better sub-epoch checkpointing experience (#3807)
- Added optimizer hooks in callbacks (#4379)
- Added option to log momentum (#4384)
- Added `current_score` to `ModelCheckpoint.on_save_checkpoint` (#4721)
- Added logging using `self.log` in train and evaluation for epoch end hooks (#4913)
- Added ability for DDP plugin to modify optimizer state saving (#4675)
- Added casting to python types for NumPy scalars when logging `hparams` (#4647)
- Added `prefix` argument in loggers (#4557)
- Added printing of total num of params, trainable and non-trainable params in ModelSummary (#4521)
- Added `PrecisionRecallCurve, ROC, AveragePrecision` class metric (#4549)
- Added custom `Apex` and `NativeAMP` as `Precision plugins` (#4355)
- Added `DALI MNIST` example (#3721)
- Added `sharded plugin` for DDP for multi-GPU training memory optimizations (#4773)
- Added `experiment_id` to the NeptuneLogger (#3462)
- Added `Pytorch Geometric` integration example with Lightning (#4568)
- Added `all_gather` method to `LightningModule` which allows gradient-based tensor synchronizations for use-cases such as negative sampling. (#5012)
- Enabled `self.log` in most functions (#4969)
- Added changeable extension variable for `ModelCheckpoint` (#4977)


### Changed

- Removed `multiclass_roc` and `multiclass_precision_recall_curve`, use `roc` and `precision_recall_curve` instead (#4549)
- Tuner algorithms will be skipped if `fast_dev_run=True` (#3903)
- WandbLogger does not force wandb `reinit` arg to True anymore and creates a run only when needed (#4648)
- Changed `automatic_optimization` to be a model attribute (#4602)
- Changed `Simple Profiler` report to order by percentage time spent + num calls (#4880)
- Simplify optimization Logic (#4984)
- Classification metrics overhaul (#4837)
- Updated `fast_dev_run` to accept integer representing num_batches (#4629)
- Refactored optimizer (#4658)


### Deprecated

- Deprecated `prefix` argument in `ModelCheckpoint` (#4765)
- Deprecated the old way of assigning hyper-parameters through `self.hparams = ...` (#4813)
- Deprecated `mode='auto'` from `ModelCheckpoint` and `EarlyStopping` (#4695)

### Removed

- Removed `reorder` parameter of the `auc` metric (#5004)

### Fixed

- Added feature to move tensors to CPU before saving (#4309)
- Fixed `LoggerConnector` to have logged metrics on root device in DP (#4138)
- Auto convert tensors to contiguous format when `gather_all` (#4907)
- Fixed `PYTHONPATH` for DDP test model (#4528)
- Fixed allowing logger to support indexing (#4595)
- Fixed DDP and manual_optimization (#4976)

## Contributors

@ananyahjha93, @awaelchli, @blatr, @Borda, @borisdayma, @carmocca, @ddrevicky, @george-gca, @gianscarpe, @irustandi, @janhenriklambrechts, @jeremyjordan, @justusschock, @lezwon, @rohitgr7, @s-rog, @SeanNaren, @SkafteNicki, @tadejsv, @tchaton, @williamFalcon, @zippeurfou

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.1.0)

---

## 1.0.8: standard weekly patch release
**Published:** 2020-11-24

## Detail changes

### Added

- Added casting to python types for numpy scalars when logging `hparams` (#4647)
- Added warning when progress bar refresh rate is less than 20 on Google Colab to prevent crashing (#4654)
- Added `F1` class metric (#4656)

### Changed

- Consistently use `step=trainer.global_step` in `LearningRateMonitor` independently of `logging_interval` (#4376)
- Metric states are no longer as default added to `state_dict` (#4685)
- Renamed class metric `Fbeta` >> `FBeta` (#4656)
- Model summary: add 1 decimal place (#4745)
- Do not override `PYTHONWARNINGS` (#4700)

### Fixed

- Fixed checkpoint `hparams` dict casting when `omegaconf` is available (#4770)
- Fixed incomplete progress bars when total batches not divisible by refresh rate (#4577)
- Updated SSIM metric (#4566)(#4656)
- Fixed batch_arg_name - add `batch_arg_name` to all calls to `_adjust_batch_size`bug (#4812)
- Fixed `torchtext` data to GPU (#4785)
- Fixed a crash bug in MLFlow logger (#4716)

## Contributors

@awaelchli, @jonashaag, @jungwhank, @M-Salti, @moi90, @pgagarinov, @s-rog, @Samyak2, @SkafteNicki, @teddykoker, @ydcjeff

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.8)

---

## 1.0.7: standard weekly patch release
**Published:** 2020-11-17

## Detail changes

### Added

- Added lambda closure to `manual_optimizer_step` (#4618)

### Changed

- Change Metrics `persistent` default mode to `False` (#4685)


### Fixed

- Prevent crash if `sync_dist=True` on CPU (#4626)
- Fixed average pbar Metrics (#4534)
- Fixed `setup` callback hook to correctly pass the LightningModule through (#4608)
- Allowing decorate model init with saving `hparams` inside (#4662)
- Fixed `split_idx` set by `LoggerConnector` in `on_trainer_init` to `Trainer`  (#4697)

## Contributors

@ananthsub, @Borda, @SeanNaren, @SkafteNicki, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.7)

---

## 1.0.6: standard weekly patch release
**Published:** 2020-11-11

## Detail changes

### Added

- Added metrics aggregation in Horovod and fixed early stopping (#3775) 
- Added `manual_optimizer_step` which work with `AMP Native` and `accumulated_grad_batches` (#4485)
- Added `persistent(mode)` method to metrics, to enable and disable metric states being added to `state_dict` (#4482)
- Added congratulations at the end of our notebooks (#4555)

### Changed

- Changed `fsspec` to tuner (#4458)
- Unify sLURM/TorchElastic under backend plugin (#4578, #4580, #4581, #4582, #4583)

### Fixed

- Fixed feature-lack in `hpc_load` (#4526)
- Fixed metrics states being overridden in DDP mode (#4482)
- Fixed `lightning_getattr`, `lightning_hasattr` not finding the correct attributes in datamodule (#4347)
- Fixed automatic optimization AMP by `manual_optimization_step` (#4485)
- Replace `MisconfigurationException` with warning in `ModelCheckpoint` Callback (#4560)
- Fixed logged keys in mlflow logger (#4412)
- Fixed `is_picklable` by catching `AttributeError` (#4508)

## Contributors

@dscarmo, @jtamir, @kazhang, @maxjeblick, @rohitgr7, @SkafteNicki, @tarepan, @tchaton, @tgaddair, @williamFalcon

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.6)

---

## 1.0.5: standard weekly patch release
**Published:** 2020-11-04

## Detail changes

### Added

- Added PyTorch 1.7 Stable support (#3821)
- Added timeout for `tpu_device_exists` to ensure process does not hang indefinitely (#4340)

### Changed

- W&B log in sync with `Trainer` step (#4405)
- Hook `on_after_backward` is called only when `optimizer_step` is being called (#4439)
- Moved `track_and_norm_grad` into `training loop` and called only when `optimizer_step` is being called (#4439)
- Changed type checker with explicit cast of ref_model object (#4457)

### Deprecated

- Deprecated passing `ModelCheckpoint` instance to `checkpoint_callback` Trainer argument (#4336)

### Fixed

- Disable saving checkpoints if not trained (#4372)
- Fixed error using `auto_select_gpus=True` with `gpus=-1` (#4209)
- Disabled training when `limit_train_batches=0` (#4371)
- Fixed that metrics do not store computational graph for all seen data (#4313)
- Fixed AMP unscale for `on_after_backward` (#4439)
- Fixed TorchScript export when module includes Metrics (#4428)
- Fixed CSV logger warning (#4419)
- Fixed skip DDP parameter sync (#4301)

## Contributors

@ananthsub, @awaelchli, @borisdayma, @carmocca, @justusschock, @lezwon, @rohitgr7, @SeanNaren, @SkafteNicki, @ssaru, @tchaton, @ydcjeff

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.5)

---

## 1.0.4: standard weekly patch release
**Published:** 2020-10-27

## Detail changes

### Added

- Added `dirpath` and `filename` parameter in `ModelCheckpoint` (#4213)
- Added plugins docs and DDPPlugin to customize ddp across all accelerators (#4258)
- Added `strict` option to the scheduler dictionary (#3586)
- Added `fsspec` support for profilers (#4162)
- Added autogenerated helptext to `Trainer.add_argparse_args` (#4344)
- Added support for string values in `Trainer`'s `profiler` parameter (#3656)

### Changed

- Improved error messages for invalid `configure_optimizers` returns (#3587)
- Allow changing the logged step value in `validation_step` (#4130)
- Allow setting `replace_sampler_ddp=True` with a distributed sampler already added (#4273)
- Fixed santized parameters for `WandbLogger.log_hyperparams` (#4320)

### Deprecated

- Deprecated `filepath` in `ModelCheckpoint` (#4213)
- Deprecated `reorder` parameter of the `auc` metric (#4237)
- Deprecated bool values in `Trainer`'s `profiler` parameter (#3656)

### Fixed

- Fixed setting device ids in DDP (#4297)
- Fixed synchronization of best model path in `ddp_accelerator` (#4323)
- Fixed `WandbLogger` not uploading checkpoint artifacts at the end of training (#4341)

## Contributors

@ananthsub, @awaelchli, @carmocca, @ddrevicky, @louis-she, @mauvilsa, @rohitgr7, @SeanNaren, @tchaton

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.4)

---

## 1.0.3: standard weekly patch release
**Published:** 2020-10-20

## Detail changes

### Added

- Added persistent flag to `Metric.add_state` (#4195)

### Changed

- Used `checkpoint_connector.hpc_save` in SLURM (#4217)
- Moved base req. to root (#4219)

### Fixed

- Fixed `hparams` assign in init (#4189)
- Fixed overwrite check for model hooks (#4010)

## Contributors

@Borda, @EspenHa, @teddykoker

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.3)

---

## 1.0.2: fixes a major logging bug for val in 1.0
**Published:** 2020-10-15

Fixes the last major bugs for validation logging.
Also removes duplicate charts for metric / metric_loss.
Doing this minor release because correct validation metrics logging is critical.

## Details changes

### Added

- Added trace functionality to the function `to_torchscript` (#4142)

### Changed

- Called `on_load_checkpoint` before loading `state_dict` (#4057)

### Removed

- Removed duplicate metric vs step log for train loop (#4173)

### Fixed

- Fixed the self.log problem in `validation_step()` (#4169)
- Fixed `hparams` saving - save the state when `save_hyperparameters()` is called [in `__init__`] (#4163)
- Fixed runtime failure while exporting `hparams` to yaml (#4158)

## Contributors

@Borda, @NumesSanguis, @rohitgr7, @williamFalcon

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.2)

---

## 1.0.1: minor jit fixes
**Published:** 2020-10-14

Obligatory post 1.0 minor release. Main fix is to make Lightning module fully compatible with Jit (had some edge-cases we had not covered).

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.1)

---

## 1.0.0: 1.0.0 - General availability
**Published:** 2020-10-13

## Overview

...

## Detail changes

- Added Explained Variance Metric + metric fix (#4013)
- Added Metric <-> Lightning Module integration tests (#4008)
- Added parsing OS env vars in `Trainer` (#4022)
- Added classification metrics (#4043)
- Updated explained variance metric (#4024)
- Enabled plugins (#4041)
- Enabled custom clusters (#4048)
- Enabled passing in custom accelerators (#4050)
- Added `LightningModule.toggle_optimizer` (#4058)
- Added `LightningModule.manual_backward` (#4063)

### Changed

- Integrated metrics API with self.log (#3961)
- Decoupled Appex (#4052, #4054, #4055, #4056, #4058, #4060, #4061, #4062, #4063, #4064, #4065)
- Renamed all backends to `Accelerator` (#4066)
- Enabled manual returns (#4089)

### Removed

- Removed `output` argument from `*_batch_end` hooks (#3965, #3966)
- Removed `output` argument from `*_epoch_end` hooks (#3967)
- Removed support for EvalResult and TrainResult (#3968)
- Removed deprecated trainer flags: `overfit_pct`, `log_save_interval`, `row_log_interval` (#3969)
- Removed deprecated early_stop_callback (#3982)
- Removed deprecated model hooks (#3980)
- Removed deprecated callbacks (#3979)
- Removed `trainer` argument in `LightningModule.backward` [#4056)

### Fixed

- Fixed `current_epoch` property update to reflect true epoch number inside `LightningDataModule`, when `reload_dataloaders_every_epoch=True`. (#3974)
- Fixed to print scaler value in progress bar (#4053) 
- Fixed mismatch between docstring and code regarding when `on_load_checkpoint` hook is called (#3996)

## Contributors

@ananyahjha93, @Borda, @edenlightning, @hbredin, @rohitgr7, @SkafteNicki, @teddykoker, @williamFalcon

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/1.0.0)

---

## 0.10.0: Buffer release before 1.0
**Published:** 2020-10-07

This release is a buffer in case 1.0 breaks any compatibility for people who upgrade. 0.10.0 has all the bug fixes and features of 1.0 but is 100% backward compatible. The 1.0 release following in the next 24 hours. 

## Overview

The major changes are:
- Results objects are deprecated (we hated them too haha)
- This means dataflow and logging have been decoupled

To log:
```python
def any_step(...):
   self.log('something', i_computed)
```

Separately, return whatever you want from methods:
```python
def training_step(...):
  return loss
```
or
```python
def training_step(...):
   return {'loss': loss, 'whatever': [1, 'want']}
```

## Detail changes

### Added

- Added new Metrics API. (#3868, [#3921)
- Enable PyTorch 1.7 compatibility (#3541)
- Added `LightningModule.to_torchscript` to support exporting as `ScriptModule` (#3258)
- Added warning when dropping unpicklable `hparams` (#2874)
- Added EMB similarity (#3349)
- Added `ModelCheckpoint.to_yaml` method (#3048)
- Allow `ModelCheckpoint` monitor to be `None`, meaning it will always save ([3630)
- Disabled optimizers setup during testing (#3059)
- Added support for datamodules to save and load checkpoints when training (#3563
- Added support for datamodule in learning rate finder (#3425)
- Added gradient clip test for native AMP (#3754)
- Added dist lib to enable syncing anything across devices (#3762)
- Added `broadcast` to `TPUBackend` (#3814)
- Added `XLADeviceUtils` class to check XLA device type (#3274)

### Changed

- Refactored accelerator backends:
   * moved TPU `xxx_step` to backend (#3118)
   * refactored DDP backend `forward` (#3119)
   * refactored GPU backend `__step` (#3120)
   * refactored Horovod backend (#3121, #3122)
   * remove obscure forward call in eval + CPU backend `___step` (#3123)
   * reduced all simplified forward (#3126)
   * added hook base method (#3127)
   * refactor eval loop to use hooks - use `test_mode` for if so we can split later (#3129)
   * moved `___step_end` hooks (#3130)
   * training forward refactor (#3134)
   * training AMP scaling refactor (#3135)
   * eval step scaling factor (#3136)
   * add eval loop object to streamline eval loop (#3138)
   * refactored dataloader process hook (#3139)
   * refactored inner eval loop (#3141)
   * final inner eval loop hooks (#3154)
   * clean up hooks in `run_evaluation` (#3156)
   * clean up data reset (#3161)
   * expand eval loop out (#3165)
   * moved hooks around in eval loop (#3195)
   * remove `_evaluate` fx (#3197)
   * `Trainer.fit` hook clean up (#3198)
   * DDPs train hooks (#3203)
   * refactor DDP backend (#3204, #3207, #3208, #3209, #3210)
   * reduced accelerator selection (#3211)
   * group prepare data hook (#3212)
   * added data connector (#3285)
   * modular is_overridden (#3290)
   * adding `Trainer.tune()` (#3293)
   * move `run_pretrain_routine` -> `setup_training` (#3294)
   * move train outside of setup training (#3297)
   * move `prepare_data` to data connector (#3307)
   * moved accelerator router (#3309)
   * train loop refactor - moving train loop to own object (#3310, #3312, #3313, #3314)
   * duplicate data interface definition up into DataHooks class (#3344)
   * inner train loop (#3359, #3361, #3362, #3363, #3365, #3366, #3367, #3368, #3369, #3370, #3371, #3372, #3373, #3374, #3375, #3376, #3385, #3388, #3397)
   * all logging related calls in a connector (#3395)
   * device parser (#3400, #3405)
   * added model connector (#3407)
   * moved eval loop logging to loggers (#3408)
   * moved eval loop (#3412[#3408)
   * trainer/separate argparse (#3421, #3428, #3432)
   * move `lr_finder` (#3434)
   * organize args (##3435, #3442, #3447, #3448, #3449, #3456)
   * move specific accelerator code (#3457)
   * group connectors (#3472)
   * accelerator connector methods x/n (#3469, #3470, #3474)
   * merge backends (#3476, #3477, #3478, #3480, #3482)
   * apex plugin (#3502)
   * precision plugins (#3504)
   * Result - make monitor default to `checkpoint_on` to simplify (#3571)
   * reference to the Trainer on the `LightningDataModule` (#3684)
   * add `.log` to lightning module (#3686, #3699, #3701, #3704, #3715)
   * enable tracking original metric when step and epoch are both true (#3685)
   * deprecated results obj, added support for simpler comms (#3681)
   * move backends back to individual files (#3712)
   * fixes logging for eval steps (#3763)
   * decoupled DDP, DDP spawn (#3733, #3766, #3767, #3774, #3802, #3806)
   * remove weight loading hack for ddp_cpu (#3808)
   * separate `torchelastic` from DDP (#3810)
   * separate SLURM from DDP (#3809)
   * decoupled DDP2 (#3816)
   * bug fix with logging val epoch end + monitor (#3812)
   * decoupled DDP, DDP spawn (#3733, #3817, #3819, #3927)
   * callback system and init DDP (#3836)
   * adding compute environments (#3837, [#3842)
   * epoch can now log independently (#3843)
   * test selecting the correct backend. temp backends while slurm and TorchElastic are decoupled (#3848)
   * fixed `init_slurm_connection` causing hostname errors (#3856)
   * moves init apex from LM to apex connector (#3923)
   * moves sync bn to each backend (#3925)
   * moves configure ddp to each backend (#3924)
- Deprecation warning (#3844)
- Changed `LearningRateLogger` to `LearningRateMonitor` (#3251)
- Used `fsspec` instead of `gfile` for all IO (#3320)
    * Swaped `torch.load` for `fsspec` load in DDP spawn backend (#3787)
    * Swaped `torch.load` for `fsspec` load in cloud_io loading (#3692)
    * Added support for `to_disk()` to use remote filepaths with `fsspec` (#3930)
    * Updated model_checkpoint's to_yaml to use `fsspec` open (#3801)
    * Fixed `fsspec` is inconsistant when doing `fs.ls` (#3805)
- Refactor `GPUStatsMonitor` to improve training speed (#3257)
- Changed IoU score behavior for classes absent in target and pred (#3098)
- Changed IoU `remove_bg` bool to `ignore_index` optional int (#3098)
- Changed defaults of `save_top_k` and `save_last` to `None` in ModelCheckpoint (#3680)
- `row_log_interval` and `log_save_interval` are now based on training loop's `global_step` instead of epoch-internal batch index (#3667)
- Silenced some warnings. verified ddp refactors (#3483)
- Cleaning up stale logger tests (#3490)
- Allow `ModelCheckpoint` monitor to be `None` (#3633)
- Enable `None` model checkpoint default (#3669)
- Skipped `best_model_path` if `checkpoint_callback` is `None` (#2962)
- Used `raise .. from ..` to explicitly chain exceptions (#3750)
-  Mocking loggers (#3596, #3617, #3851, #3859, #3884, #3853, #3910, #3889, #3926)
- Write predictions in LightningModule instead of EvalResult [#3882

### Deprecated

- Deprecated `TrainResult` and `EvalResult`, use `self.log` and `self.write` from the `LightningModule` to log metrics and write predictions. `training_step` can now only return a scalar (for the loss) or a dictionary with anything you want. (#3681)
- Deprecate `early_stop_callback` Trainer argument (#3845)
- Rename Trainer arguments `row_log_interval` >> `log_every_n_steps` and `log_save_interval` >> `flush_logs_every_n_steps` (#3748)

### Removed

- Removed experimental Metric API (#3868, #3943, #3949, #3946), listed changes before final removal:
    * Added `EmbeddingSimilarity` metric (#3349, [#3358)
    * Added hooks to metric module interface (#2528)
    * Added error when AUROC metric is used for multiclass problems (#3350)
    * Fixed `ModelCheckpoint` with `save_top_k=-1` option not tracking the best models when a monitor metric is available (#3735)
    * Fixed counter-intuitive error being thrown in `Accuracy` metric for zero target tensor (#3764)
    * Fixed aggregation of metrics (#3517)
    * Fixed Metric aggregation (#3321)
    * Fixed RMSLE metric (#3188)
    * Renamed `reduction` to `class_reduction` in classification metrics (#3322)
    * Changed `class_reduction` similar to sklearn for classification metrics (#3322)
    * Renaming of precision recall metric (#3308)

### Fixed

- Fixed `on_train_batch_start` hook to end epoch early (#3700)
- Fixed `num_sanity_val_steps` is clipped to `limit_val_batches` (#2917)
- Fixed ONNX model save on GPU (#3145)
- Fixed `GpuUsageLogger` to work on different platforms (#3008)
- Fixed auto-scale batch size not dumping `auto_lr_find` parameter (#3151)
- Fixed `batch_outputs` with optimizer frequencies (#3229)
- Fixed setting batch size in `LightningModule.datamodule` when using `auto_scale_batch_size` (#3266)
- Fixed Horovod distributed backend compatibility with native AMP (#3404)
- Fixed batch size auto scaling exceeding the size of the dataset (#3271)
- Fixed getting `experiment_id` from MLFlow only once instead of each training loop (#3394)
- Fixed `overfit_batches` which now correctly disables shuffling for the training loader. (#3501)
- Fixed gradient norm tracking for `row_log_interval > 1` (#3489)
- Fixed `ModelCheckpoint` name formatting ([3164)
- Fixed auto-scale batch size (#3151)
- Fixed example implementation of AutoEncoder (#3190)
- Fixed invalid paths when remote logging with TensorBoard (#3236)
- Fixed change `t()` to `transpose()` as XLA devices do not support `.t()` on 1-dim tensor (#3252)
- Fixed (weights only) checkpoints loading without PL (#3287)
- Fixed `gather_all_tensors` cross GPUs in DDP (#3319)
- Fixed CometML save dir (#3419)
- Fixed forward key metrics (#3467)
- Fixed normalize mode at confusion matrix (replace NaNs with zeros) (#3465)
- Fixed global step increment in training loop when `training_epoch_end` hook is used (#3673)
- Fixed dataloader shuffling not getting turned off with `overfit_batches > 0` and `distributed_backend = "ddp"` (#3534)
- Fixed determinism in `DDPSpawnBackend` when using `seed_everything` in main process (#3335)
- Fixed `ModelCheckpoint` `period` to actually save every `period` epochs (#3630)
- Fixed `val_progress_bar` total with `num_sanity_val_steps` (#3751)
- Fixed Tuner dump: add `current_epoch` to dumped_params (#3261)
- Fixed `current_epoch` and `global_step` properties mismatch between `Trainer` and `LightningModule` (#3785)
- Fixed learning rate scheduler for optimizers with internal state (#3897)
- Fixed `tbptt_reduce_fx` when non-floating tensors are logged (#3796)
- Fixed model checkpoint frequency (#3852)
- Fixed logging non-tensor scalar with result breaks subsequent epoch aggregation (#3855)
- Fixed `TrainerEvaluationLoopMixin` activates `model.train()` at the end (#3858)
- Fixed `overfit_batches` when using with multiple val/test_dataloaders (#3857)
- Fixed enables `training_step` to return `None` (#3862)
- Fixed init nan for checkpointing (#3863)
- Fixed for `load_from_checkpoint` (#2776)
- Fixes incorrect `batch_sizes` when Dataloader returns a dict with multiple tensors (#3668)
- Fixed unexpected signature for `validation_step` (#3947)

## Contributors

@abrahambotros, @akihironitta, @ananthsub, @ananyahjha93, @awaelchli, @Borda, @c00k1ez, @carmocca, @f4hy, @GimmickNG, @jbschiratti, @justusschock, @LeeJZh, @lezwon, @Lucas-Steinmann, @maxjeblick, @monney, @mpariente, @nateraw, @nrupatunga, @patrickorlando, @PhilJd, @rohitgr7, @s-rog, @ShomyLiu, @SkafteNicki, @Sordie, @teddykoker, @tgaddair, @Vozf, @williamFalcon, @XDynames, @ydcjeff

_If we forgot someone due to not matching the commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.10.0)

---

## 0.9.0: synced BatchNorm, DataModules and final API
**Published:** 2020-08-20

## Overview

The newest PyTorch Lightning release includes final API clean-up with better data decoupling and shorter logging syntax.

Were happy to release PyTorch Lightning 0.9 today, which contains many great new features, more bugfixes than any release we ever had, but most importantly it introduced our mostly final API changes! Lightning is being adopted by top researchers and AI labs around the world, and we are working hard to make sure we provide a smooth experience and support for all the latest best practices. 

## Detail changes

### Added

- Added SyncBN for DDP (#2801, #2838)
- Added basic `CSVLogger` (#2721)
- Added SSIM metrics (#2671)
- Added BLEU metrics (#2535)
- Added support to export a model to ONNX format (#2596)
- Added support for `Trainer(num_sanity_val_steps=-1)` to check all validation data before training (#2246)
- Added struct. output:
  * tests for val loop flow (#2605)
  * `EvalResult` support for train and val. loop (#2615, #2651)
  * weighted average in results obj (#2930)
  * fix result obj DP auto reduce (#3013)
- Added class `LightningDataModule` (#2668)
- Added support for PyTorch 1.6 (#2745)
- Added call DataModule hooks implicitly in trainer (#2755)
- Added support for Mean in DDP Sync (#2568)
- Added remaining `sklearn` metrics: `AveragePrecision`, `BalancedAccuracy`, `CohenKappaScore`, `DCG`, `Hamming`, `Hinge`, `Jaccard`, `MeanAbsoluteError`, `MeanSquaredError`, `MeanSquaredLogError`, `MedianAbsoluteError`, `R2Score`, `MeanPoissonDeviance`, `MeanGammaDeviance`, `MeanTweedieDeviance`, `ExplainedVariance` (#2562)
- Added support for `limit_{mode}_batches (int)` to work with infinite dataloader (IterableDataset) (#2840)
- Added support returning python scalars in DP (#1935)
- Added support to Tensorboard logger for OmegaConf `hparams` (#2846)
- Added tracking of basic states in `Trainer` (#2541)
- Tracks all outputs including TBPTT and multiple optimizers (#2890)
- Added GPU Usage Logger (#2932)
- Added `strict=False` for `load_from_checkpoint` (#2819)
- Added saving test predictions on multiple GPUs (#2926)
- Auto log the computational graph for loggers that support this (#3003)
- Added warning when changing monitor and using results obj (#3014)
- Added a hook `transfer_batch_to_device` to the `LightningDataModule` (#3038)

### Changed

- Truncated long version numbers in progress bar (#2594)
- Enabling val/test loop disabling (#2692)
- Refactored into `accelerator` module:
    * GPU training (#2704)
    * TPU training (#2708)
    * DDP(2) backend (#2796)
    * Retrieve last logged val from result by key (#3049)
- Using `.comet.config` file for `CometLogger` (#1913)
- Updated hooks arguments - breaking for `setup` and `teardown` (#2850)
- Using `gfile` to support remote directories (#2164)
- Moved optimizer creation after device placement for DDP backends (#2904](https://github.com/PyTorchLightning/pytorch-lighting/pull/2904))
- Support `**DictConfig` for `hparam` serialization (#2519)
- Removed callback metrics from test results obj (#2994)
- Re-enabled naming metrics in ckpt name (#3060)
- Changed progress bar epoch counting to start from 0 (#3061)

### Deprecated

- Deprecated Trainer attribute `ckpt_path`, which will now be set by `weights_save_path` (#2681)

### Removed

- Removed deprecated: (#2760)
    * core decorator `data_loader`
    * Module hook `on_sanity_check_start` and loading `load_from_metrics`
    * package `pytorch_lightning.logging`
    * Trainer arguments: `show_progress_bar`, `num_tpu_cores`, `use_amp`, `print_nan_grads`
    * LR Finder argument `num_accumulation_steps`

### Fixed

- Fixed `accumulate_grad_batches` for last batch (#2853)
- Fixed setup call while testing (#2624)
- Fixed local rank zero casting (#2640)
- Fixed single scalar return from training (#2587)
- Fixed Horovod backend to scale LR schedlers with the optimizer (#2626)
- Fixed `dtype` and `device` properties not getting updated in submodules (#2657)
- Fixed `fast_dev_run` to run for all dataloaders (#2581)
- Fixed `save_dir` in loggers getting ignored by default value of `weights_save_path` when user did not specify `weights_save_path` (#2681)
- Fixed `weights_save_path` getting ignored when `logger=False` is passed to Trainer (#2681)
- Fixed TPU multi-core and Float16 (#2632)
- Fixed test metrics not being logged with `LoggerCollection` (#2723)
- Fixed data transfer to device when using `torchtext.data.Field` and `include_lengths is True` (#2689)
- Fixed shuffle argument for the distributed sampler (#2789)
- Fixed logging interval (#2694)
- Fixed loss value in the progress bar is wrong when `accumulate_grad_batches > 1` (#2738)
- Fixed correct CWD for DDP sub-processes when using Hydra (#2719)
- Fixed selecting GPUs using `CUDA_VISIBLE_DEVICES` (#2739, #2796)
- Fixed false `num_classes` warning in metrics (#2781)
- Fixed shell injection vulnerability in subprocess call (#2786)
- Fixed LR finder and `hparams` compatibility (#2821)
- Fixed `ModelCheckpoint` not saving the latest information when `save_last=True` (#2881)
- Fixed ImageNet example: learning rate scheduler, number of workers and batch size when using DDP (#2889)
- Fixed apex gradient clipping (#2829)
- Fixed save apex scaler states (#2828)
- Fixed a model loading issue with inheritance and variable positional arguments (#2911)
- Fixed passing `non_blocking=True` when transferring a batch object that does not support it (#2910)
- Fixed checkpointing to remote file paths (#2925)
- Fixed adding `val_step` argument to metrics (#2986)
- Fixed an issue that caused `Trainer.test()` to stall in DDP mode (#2997)
- Fixed gathering of results with tensors of varying shape (#3020)
- Fixed batch size auto-scaling feature to set the new value on the correct model attribute (#3043) 
- Fixed automatic batch scaling not working with half-precision (#3045)
- Fixed setting device to root GPU (#3042)

## Contributors

@ananthsub, @ananyahjha93, @awaelchli, @bkhakshoor, @Borda, @ethanwharris, @f4hy, @groadabike, @ibeltagy, @justusschock, @lezwon, @nateraw, @neighthan, @nsarang, @PhilJd, @pwwang, @rohitgr7, @romesco, @ruotianluo, @shijianjian, @SkafteNicki, @tgaddair, @thschaaf, @williamFalcon, @xmotli02, @ydcjeff, @yukw777, @zerogerc

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.9.0)

---

## 0.8.5: Bug fixes and .test() fix + TPU tests
**Published:** 2020-07-10

## Overview

The point of this release is more bug fixes ahead of v 1.0.0. We now have CI tests on TPU thanks to @zcain117 from Google! :slightly_smiling_face:
This means we fixed many TPU bugs we hadn’t caught before because we had no tests.
In addition, we fixed:
- all the file path errors with loggers (txs @awaelchli)
- pickling errors with loggers (txs @awaelchli)
- fixed all the .test() calls

## Detail changes

### Added

- Added a PSNR metric: peak signal-to-noise ratio (#2483)
- Added functional regression metrics (#2492)

### Removed

- Removed auto val reduce (#2462)

### Fixed

- Flattening Wandb Hyperparameters (#2459)
- Fixed using the same DDP python interpreter and actually running (#2482)
- Fixed model summary input type conversion for models that have input dtype different from model parameters (#2510)
- Made `TensorBoardLogger` and `CometLogger` pickleable (#2518)
- Fixed a problem with `MLflowLogger` creating multiple run folders (#2502)
- Fixed global_step increment (#2455)
- Fixed TPU hanging example (#2488)
- Fixed `argparse` default value bug (#2526)
- Fixed Dice and IoU to avoid NaN by adding small eps (#2545)
- Fixed accumulate gradients schedule at epoch 0 (continued) (#2513)
- Fixed Trainer `.fit()` returning last not best weights in "ddp_spawn" (#2565)
- Fixed passing (do not pass) TPU weights back on test (#2566)
- Fixed DDP tests and `.test()` (#2512, #2570)

## Contributors

@anthonytec2, @awaelchli, @bernardomig, @Borda, @EspenHa, @HHousen, @InCogNiTo124, @rohitgr7, @williamFalcon

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.8.5)

---

## 0.8.4: More bug fixing!
**Published:** 2020-07-01

## Detail changes

### Added

- Added reduce ddp results on eval (#2434)
- Added a warning when an `IterableDataset` has `__len__` defined (#2437)

### Changed

- Enabled no returns from eval (#2446)

### Fixed

- Fixes train outputs (#2428)
- Fixes Conda dependencies (#2412)
- Fixed Apex scaling with decoupled backward (#2433)
- Fixed crashing or wrong displaying progressbar because of missing ipywidgets (#2417)
- Fixed TPU saving dir (fc26078e395f8a001f4c6dd7b3fe7ca202f914a3, 04e68f022fc03dd5f1555ee86dea997d42a448ad)
- Fixed logging on rank 0 only (#2425)

## Contributors

@awaelchli, @Borda, @olineumann, @williamFalcon

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.8.4)

---

## 0.8.3: Bug fixing
**Published:** 2020-06-29

### Fixed

- Fixed AMP wrong call (593837e1da24ff6c942b24ed803fc1496a304609)
- Fixed batch typo (92d1e75b2638a493d9d21ed5fe00a22093888285)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.8.3)

---

## 0.8.2: DDP and Checkpoint bug fixes
**Published:** 2020-06-29
**Pre-release**

## Overview

As we continue to strengthen the codebase with more tests, we’re finally getting rid of annoying bugs that have been around for a bit now. Mostly around the inconsistent checkpoint and early stopping behaviour (amazing work @awaelchli  @jeremyjordan )

### Noteworthy changes:

- Fixed TPU flag parsing
- fixed average_precision metric
- all the checkpoint issues should be gone now (including backward support for old checkpoints)
- DDP + loggers should be fixed

## Detail changes

### Added

- Added TorchText support for moving data to GPU (#2379)

### Changed

- Changed epoch indexing from 0 instead of 1 (#2289)
- Refactor Model `backward` (#2276)
- Refactored `training_batch` + tests to verify correctness (#2327, #2328)
- Refactored training loop (#2336)
- Made optimization steps for hooks (#2363)
- Changed default apex level to 'O2' (#2362)

### Removed

- Moved `TrainsLogger` to Bolts (#2384)

### Fixed

- Fixed parsing TPU arguments and TPU tests (#2094)
- Fixed number batches in case of multiple dataloaders and `limit_{*}_batches` (#1920, #2226)
- Fixed an issue with forward hooks not being removed after model summary (#2298)
- Fix for `load_from_checkpoint()` not working with absolute path on Windows (#2294)
- Fixed an issue how _has_len handles `NotImplementedError` e.g. raised by `torchtext.data.Iterator` (#2293), (#2307)
- Fixed `average_precision` metric (#2319)
- Fixed ROC metric for CUDA tensors (#2304)
- Fixed `average_precision` metric (#2319)
- Fixed lost compatibility with custom datatypes implementing `.to` (#2335)
- Fixed loading model with kwargs (#2387)
- Fixed sum(0) for `trainer.num_val_batches` (#2268)
- Fixed checking if the parameters are a `DictConfig` Object (#2216)
- Fixed SLURM weights saving (#2341)
- Fixed swaps LR scheduler order (#2356)
- Fixed adding tensorboard `hparams` logging test (#2342)
- Fixed use model ref for tear down (#2360)
- Fixed logger crash on DDP (#2388)
- Fixed several issues with early stopping and checkpoint callbacks (#1504, #2391)
- Fixed loading past checkpoints from v0.7.x (#2405)
- Fixed loading model without arguments (#2403)

## Contributors

@airium, @awaelchli, @Borda, @elias-ramzi, @jeremyjordan, @lezwon, @mateuszpieniak, @mmiakashs, @pwl, @rohitgr7, @ssakhavi, @thschaaf, @tridao, @williamFalcon

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.8.2)

---

## 0.8.1: Fixing hooks & hparams
**Published:** 2020-06-19

## Overview

Fixing critical bugs in newly added hooks and `hparams` assignment.
The recommended data following:

1. use `prepare_data` to download and process the dataset.
2. use `setup` to do splits, and build your model internals

## Detail changes

- Fixed the `load_from_checkpoint` path detected as URL bug (#2244)
- Fixed hooks - added barrier (#2245, #2257, #2260)
- Fixed `hparams` - remove frame inspection on `self.hparams` (#2253)
- Fixed setup and on fit calls (#2252)
- Fixed GPU template (#2255)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.8.1)

---

## 0.8.0: Metrics, speed improvements, new hooks and flags
**Published:** 2020-06-19
**Pre-release**

## Overview

Highlights of this release are adding Metric package and new hooks and flags to customize your workflow.

### Major features:

- brand new Metrics package with built-in DDP support (by @justusschock and  @SkafteNicki)
- `hparams` can now be anything! (call `self.save_hyperparameters()` to register anything in the `_init_`
- many speed improvements (how we move data, adjusted some flags & PL now adds 300ms overhead per epoch only!)
- much faster `ddp` implementation. Old one was renamed `ddp_spawn`
- better support for Hydra
- added the overfit_batches flag and corrected some bugs with the `limit_[train,val,test]_batches` flag
- added conda support
- tons of bug fixes :wink:

## Detail changes

### Added

- Added `overfit_batches`, `limit_{val|test}_batches` flags (overfit now uses training set for all three) (#2213)
- Added metrics 
  * Base classes (#1326, #1877)
  * Sklearn metrics classes (#1327)
  * Native torch metrics (#1488, #2062)
  * docs for all Metrics (#2184, #2209)
  * Regression metrics (#2221)
- Added type hints in `Trainer.fit()` and `Trainer.test()` to reflect that also a list of dataloaders can be passed in (#1723)
- Allow dataloaders without sampler field present (#1907)
- Added option `save_last` to save the model at the end of every epoch in `ModelCheckpoint` (#1908)
- Early stopping checks `on_validation_end` (#1458)
- Attribute `best_model_path` to `ModelCheckpoint` for storing and later retrieving the path to the best saved model file (#1799) 
- Speed up single-core TPU training by loading data using `ParallelLoader` (#2033)
- Added a model hook `transfer_batch_to_device` that enables moving custom data structures to the target device (#1756)
- Added [black](https://black.readthedocs.io/en/stable/) formatter for the code with code-checker on pull (#1610)
- Added back the slow spawn ddp implementation as `ddp_spawn` (#2115)
- Added loading checkpoints from URLs (#1667)
- Added a callback method `on_keyboard_interrupt` for handling KeyboardInterrupt events during training (#2134)
- Added a decorator `auto_move_data` that moves data to the correct device when using the LightningModule for inference (#1905)
- Added `ckpt_path` option to `LightningModule.test(...)` to load particular checkpoint (#2190)
- Added `setup` and `teardown` hooks for model (#2229)

### Changed

- Allow user to select individual TPU core to train on (#1729)
- Removed non-finite values from loss in `LRFinder` (#1862)
- Allow passing model hyperparameters as complete kwarg list (#1896) 
- Renamed `ModelCheckpoint`'s attributes `best` to `best_model_score` and `kth_best_model` to `kth_best_model_path` (#1799)
- Re-Enable Logger's `ImportError`s (#1938)
- Changed the default value of the Trainer argument `weights_summary` from `full` to `top` (#2029)
- Raise an error when lightning replaces an existing sampler (#2020)
- Enabled prepare_data from correct processes - clarify local vs global rank (#2166)
- Remove explicit flush from tensorboard logger (#2126)
- Changed epoch indexing from 1 instead of 0 (#2206)

### Deprecated

- Deprecated flags: (#2213)
  * `overfit_pct` in favour of `overfit_batches`
  * `val_percent_check` in favour of `limit_val_batches`
  * `test_percent_check` in favour of `limit_test_batches`
- Deprecated `ModelCheckpoint`'s attributes `best` and `kth_best_model` (#1799)
- Dropped official support/testing for older PyTorch versions <1.3 (#1917)

### Removed

- Removed unintended Trainer argument `progress_bar_callback`, the callback should be passed in by `Trainer(callbacks=[...])` instead (#1855)
- Removed obsolete `self._device` in Trainer (#1849)
- Removed deprecated API (#2073)
   * Packages: `pytorch_lightning.pt_overrides`, `pytorch_lightning.root_module`
   * Modules: `pytorch_lightning.logging.comet_logger`, `pytorch_lightning.logging.mlflow_logger`, `pytorch_lightning.logging.test_tube_logger`, `pytorch_lightning.overrides.override_data_parallel`, `pytorch_lightning.core.model_saving`, `pytorch_lightning.core.root_module`
   * Trainer arguments: `add_row_log_interval`, `default_save_path`, `gradient_clip`, `nb_gpu_nodes`, `max_nb_epochs`, `min_nb_epochs`, `nb_sanity_val_steps`
   * Trainer attributes: `nb_gpu_nodes`, `num_gpu_nodes`, `gradient_clip`, `max_nb_epochs`, `min_nb_epochs`, `nb_sanity_val_steps`, `default_save_path`, `tng_tqdm_dic`

### Fixed

- Run graceful training teardown on interpreter exit (#1631)
- Fixed user warning when apex was used together with learning rate schedulers (#1873)
- Fixed multiple calls of `EarlyStopping` callback (#1863)
- Fixed an issue with `Trainer.from_argparse_args` when passing in unknown Trainer args (#1932)
- Fixed bug related to logger not being reset correctly for model after tuner algorithms (#1933)
- Fixed root node resolution for SLURM cluster with dash in hostname (#1954)
- Fixed `LearningRateLogger` in multi-scheduler setting (#1944)
- Fixed test configuration check and testing (#1804)
- Fixed an issue with Trainer constructor silently ignoring unknown/misspelt arguments (#1820)
- Fixed `save_weights_only` in ModelCheckpoint (#1780)
- Allow use of same `WandbLogger` instance for multiple training loops (#2055)
- Fixed an issue with `_auto_collect_arguments` collecting local variables that are not constructor arguments and not working for signatures that have the instance not named `self` (#2048)
- Fixed mistake in parameters' grad norm tracking (#2012)
- Fixed CPU and hanging GPU crash (#2118)
- Fixed an issue with the model summary and `example_input_array` depending on a specific ordering of the submodules in a LightningModule (#1773)
- Fixed Tpu logging (#2230)
- Fixed Pid port + duplicate `rank_zero` logging (#2140, #2231)

## Contributors

@awaelchli, @baldassarreFe, @Borda, @borisdayma, @cuent, @devashishshankar, @ivannz, @j-dsouza, @justusschock, @kepler, @kumuji, @lezwon, @lgvaz, @LoicGrobol, @mateuszpieniak, @maximsch2, @moi90, @rohitgr7, @SkafteNicki, @tullie, @williamFalcon, @yukw777, @ZhaofengWu

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.8.0)

---

## 0.7.6: Transfer learning, tuning batch size, torchelastic support
**Published:** 2020-05-15

## Overview

Highlights of this release are adding support for TorchElastic enables distributed PyTorch training jobs to be executed in a fault-tolerant and elastic manner; auto-scaling of batch size; new transfer learning example; an option to provide seed to random generators to ensure reproducibility.

## Detail changes

### Added

- Added callback for logging learning rates (#1498)
- Added transfer learning example (for a binary classification task in computer vision) (#1564)
- Added type hints in `Trainer.fit()` and `Trainer.test()` to reflect that also a list of dataloaders can be passed in (#1723).
- Added auto scaling of batch size (#1638)
- The progress bar metrics now also get updated in `training_epoch_end` (#1724)
- Enable `NeptuneLogger` to work with `distributed_backend=ddp` (#1753)
- Added option to provide seed to random generators to ensure reproducibility (#1572)
- Added override for hparams in `load_from_ckpt` (#1797)
- Added support multi-node distributed execution under `torchelastic` (#1811, #1818)
- Added using `store_true` for bool args (#1822, #1842)
- Added dummy logger for internally disabling logging for some features (#1836)

### Changed

- Enable `non-blocking` for device transfers to GPU (#1843)
- Replace mata_tags.csv with hparams.yaml (#1271)
- Reduction when `batch_size < num_gpus` (#1609)
- Updated LightningTemplateModel to look more like Colab example (#1577)
- Don't convert `namedtuple` to `tuple` when transferring the batch to target device (#1589)
- Allow passing `hparams` as a keyword argument to LightningModule when loading from checkpoint (#1639)
- Args should come after the last positional argument (#1807)
- Made DDP the default if no backend specified with multiple GPUs (#1789)

### Deprecated

- Deprecated `tags_csv` in favor of `hparams_file` (#1271)

### Fixed

- Fixed broken link in PR template (#1675)
- Fixed ModelCheckpoint not None checking file path (#1654)
- Trainer now calls `on_load_checkpoint()` when resuming from a checkpoint (#1666)
- Fixed sampler logic for DDP with the iterable dataset (#1734)
- Fixed `_reset_eval_dataloader()` for IterableDataset (#1560)
- Fixed Horovod distributed backend to set the `root_gpu` property (#1669)
- Fixed wandb logger `global_step` affects other loggers (#1492)
- Fixed disabling progress bar on non-zero ranks using Horovod backend (#1709)
- Fixed bugs that prevent LP finder to be used together with early stopping and validation dataloaders (#1676)
- Fixed a bug in Trainer that prepended the checkpoint path with `version_` when it shouldn't (#1748)
- Fixed LR key name in case of param groups in LearningRateLogger (#1719)
- Fixed saving native AMP scaler state (introduced in #1561)
- Fixed accumulation parameter and suggestion method for learning rate finder (#1801)
- Fixed num processes wasn't being set properly and auto sampler was DDP failing (#1819)
- Fixed bugs in semantic segmentation example (#1824)
- Fixed saving native AMP scaler state (#1561, #1777)
- Fixed native AMP + DDP (#1788)
- Fixed `hparam` logging with metrics (#1647)

## Contributors

@ashwinb, @awaelchli, @Borda, @cmpute, @festeh, @jbschiratti, @justusschock, @kepler, @kumuji, @nanddalal, @nathanbreitsch, @olineumann, @pitercl, @rohitgr7, @S-aiueo32, @SkafteNicki, @tgaddair, @tullie, @tw991, @williamFalcon, @ybrovman, @yukw777

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.7.6)

---

## 0.7.5: Critical DDP bug fixes
**Published:** 2020-04-27

We made a few changes to Callbacks to test ops on detached GPU tensors to avoid CPU transfer. However, it made callbacks unpicklable which will crash DDP. 

This release fixes that core issue

### Changed

- Allow logging of metrics together with hparams (#1630)
- Allow metrics logged together with hparams (#1630)

### Removed

- Removed Warning from trainer loop (#1634)

### Fixed

- Fixed ModelCheckpoint not being fixable (#1632)
- Fixed CPU DDP breaking change and DDP change (#1635)
- Tested pickling (#1636)

### Contributors

@justusschock, @quinor, @williamFalcon


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.7.5)

---

## 0.7.4: PyTorch 1.5 support, native PyTorch AMP, speed/memory optimizations and many bug fixes
**Published:** 2020-04-26
**Pre-release**

### Key updates

- PyTorch 1.5  support
- Added Horovod distributed_backend option
- Enable forward compatibility with the native AMP (PyTorch 1.6).
- Support 8-core TPU on Kaggle
- Added ability to customize progress_bar via Callbacks
- Speed/memory optimizations.
- Improved Argparse usability with Trainer
- Docs improvements
- Tons of bug fixes

## Detail changes

### Added

- Added flag `replace_sampler_ddp` to manually disaple sampler replacement in ddp  (#1513)
- Added speed parity tests (max 1 sec difference per epoch)(#1482)
- Added `auto_select_gpus` flag to trainer that enables automatic selection of available GPUs on exclusive mode systems.
- Added learining rate finder (#1347)
- Added support for ddp mode in clusters without SLURM (#1387)
- Added `test_dataloaders` parameter to `Trainer.test()` (#1434)
- Added `terminate_on_nan` flag to trainer that performs a NaN check with each training iteration when set to `True` (#1475)
- Added speed parity tests (max 1 sec difference per epoch)(#1482)
- Added `terminate_on_nan` flag to trainer that performs a NaN check with each training iteration when set to `True`. (#1475)
- Added `ddp_cpu` backend for testing ddp without GPUs (#1158)
- Added [Horovod](http://horovod.ai) support as a distributed backend `Trainer(distributed_backend='horovod')` (#1529)
- Added support for 8 core distributed training on Kaggle TPU's (#1568)
- Added support for native AMP (#1561, [#1580) 

### Changed

- Changed the default behaviour to no longer include a NaN check with each training iteration. (#1475)
- Decoupled the progress bar from trainer. It is a callback now and can be customized or even be replaced entirely (#1450).
- Changed lr schedule step interval behavior to update every backwards pass instead of every forwards pass (#1477)
- Defines shared proc. rank, remove rank from instances (e.g. loggers) (#1408)
- Updated semantic segmentation example with custom u-net and logging (#1371)
- Disabled val and test shuffling (#1600)

### Deprecated

- Deprecated `training_tqdm_dict` in favor of `progress_bar_dict` (#1450).

### Removed

- Removed `test_dataloaders` parameter from `Trainer.fit()` (#1434)

### Fixed

- Added the possibility to pass nested metrics dictionaries to loggers (#1582)
- Fixed memory leak from opt return (#1528)
- Fixed saving checkpoint before deleting old ones (#1453)
- Fixed loggers - flushing last logged metrics even before continue, e.g. `trainer.test()` results (#1459)
- Fixed optimizer configuration when `configure_optimizers` returns dict without `lr_scheduler` (#1443)
- Fixed `LightningModule` - mixing hparams and arguments in `LightningModule.__init__()` crashes load_from_checkpoint() (#1505)
- Added a missing call to the `on_before_zero_grad` model hook (#1493). 
- Allow use of sweeps with WandbLogger (#1512)
- Fixed a bug that caused the `callbacks` Trainer argument to reference a global variable (#1534).
- Fixed a bug that set all boolean CLI arguments from Trainer.add_argparse_args always to True (#1571)
- Fixed do not copy the batch when training on a single GPU (#1576, [#1579)
- Fixed soft checkpoint removing on DDP (#1408)
- Fixed automatic parser bug (#1585)
- Fixed bool conversion from string (#1606)

## Contributors

@alexeykarnachev, @areshytko, @awaelchli, @Borda, @borisdayma, @ethanwharris, @fschlatt, @HenryJia, @Ir1d, @justusschock, @karlinjf, @lezwon, @neggert, @rmrao, @rohitgr7, @SkafteNicki, @tgaddair, @williamFalcon

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.7.4)

---

## 0.7.3: DDP bug fixes
**Published:** 2020-04-10

We had a few (subtle) bugs that affected DDP and a few key things in 0.7.2 so we released 0.7.3 to fix them because they are critical for DDP. sorry about that! still, no API changes, but please do skip straight to  0.7.3 upgrade for those fixes

## Detail changes

### Added

- Added `rank_zero_warn` for warning only in rank 0 (#1428)

### Fixed

- Fixed default `DistributedSampler` for DDP training (#1425)
- Fixed workers warning not on windows (#1430)
- Fixed returning tuple from `run_training_batch` (#1431)
- Fixed gradient clipping (#1438)
- Fixed pretty print (#1441)

## Contributors

@alsrgv, @Borda, @williamFalcon

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.7.3)

---

## 0.7.2: Many bug fixes, added flexibility, parity tests with pytorch and more
**Published:** 2020-04-08
**Pre-release**

## Overview

This release aims at fixing particular issues and improving the user development experience via extending docs, adding typing and supporting python 3.8. In particular, some of the release highlights are:
- Added benchmark for comparing lightning with vanilla implementations
- Extended optimizer support with particular frequency
- Several improvements for loggers such as represent no-primitive types, supporting hierarchical dictionaries for hyper param searchers
- Added model configuration checking before it runs
- Simplify the PL examples structure (shallower and more readable)
- Improved Trainer CLI arguments handling (generalization)
- Two Trainer argument become deprecated: `print_nan_grads` and `show_progress_bar`



## Detail changes

### Added

- Added same step loggers' metrics aggregation (#1278)
- Added parity test between a vanilla MNIST model and lightning model (#1284)
- Added parity test between a vanilla RNN model and lightning model (#1351)
- Added Reinforcement Learning - Deep Q-network (DQN) lightning example (#1232)
- Added support for hierarchical `dict` (#1152)
- Added `TrainsLogger` class (#1122)
- Added type hints to `pytorch_lightning.core` (#946)
- Added support for `IterableDataset` in validation and testing (#1104)
- Added support for non-primitive types in `hparams` for `TensorboardLogger` (#1130)
- Added a check that stops the training when loss or weights contain `NaN` or `inf` values. (#1097) 
- Added support for `IterableDataset` when `val_check_interval=1.0` (default), this will trigger validation at the end of each epoch. (#1283)
- Added `summary` method to Profilers. (#1259)
- Added informative errors if user defined dataloader has zero length (#1280)
- Added testing for python 3.8 (#915)
- Added a `training_epoch_end` method which is the mirror of `validation_epoch_end`. (#1357)
- Added model configuration checking (#1199)
- Added support for optimizer frequencies through `LightningModule.configure_optimizers()` (#1269)
- Added option to run without an optimizer by returning `None` from `configure_optimizers`. (#1279)
- Added a warning when the number of data loader workers is small. (#1378)

### Changed

- Changed (renamed and refactored) `TensorRunningMean` -> `TensorRunningAccum`: running accumulations were generalized. (#1278)
- Changed `progress_bar_refresh_rate` trainer flag to disable progress bar when setting to 0. (#1108)
- Enhanced `load_from_checkpoint` to also forward params to the model (#1307)
- Updated references to self.forward() to instead use the `__call__` interface. (#1211)
- Changed default behaviour of `configure_optimizers` to use no optimizer rather than Adam. (#1279)
- Allow uploading models on W&B (#1339)
- On DP and DDP2 unsqueeze is automated now (#1319)
- Did not always create a DataLoader during reinstantiation, but the same type as before (if a subclass of DataLoader) (#1346)
- Did not interfere with a default sampler (#1318)
- Removed default Adam optimizer (#1317)
- Gave warnings for unimplemented required lightning methods (#1317)
- Made `evaluate` method private >> `Trainer._evaluate(...)`. (#1260)
- Simplify the PL examples structure (shallower and more readable) (#1247)
- Changed min-max GPU memory to be on their own plots (#1358)
- Remove `.item` which causes sync issues (#1254)
- Changed smoothing in TQDM to decrease variability of time remaining between training/eval (#1194)
- Change default logger to a dedicated one (#1064)

### Deprecated

- Deprecated Trainer argument `print_nan_grads` (#1097)
- Deprecated Trainer argument `show_progress_bar` (#1108)

### Removed

- Removed duplicated module `pytorch_lightning.utilities.arg_parse` for loading CLI arguments (#1167)
- Removed wandb logger's `finalize` method (#1193)
- Dropped `torchvision` dependency in tests and added own MNIST dataset class instead (#986)

### Fixed

- Fixed `model_checkpoint` when saving all models (#1359)
- `Trainer.add_argparse_args` classmethod fixed. Now it adds a type for the arguments (#1147)
- Fixed bug related to type cheking of `ReduceLROnPlateau` lr schedulers(#1114)
- Fixed a bug to ensure lightning checkpoints to be backward compatible (#1132)
- Fixed a bug that created an extra dataloader with active `reload_dataloaders_every_epoch` (#1181)
- Fixed all warnings and errors in the docs build process (#1191)
- Fixed an issue where `val_percent_check=0` would not disable validation (#1251)
- Fixed average of incomplete `TensorRunningMean` (#1309)
- Fixed `WandbLogger.watch` with `wandb.init()` (#1311)
- Fixed an issue with early stopping that would prevent it from monitoring training metrics when validation is disabled / not implemented (#1235)
- Fixed a bug that would cause `trainer.test()` to run on the validation set when overloading `validation_epoch_end` and `test_end` (#1353)
- Fixed `WandbLogger.watch` - use of the watch method without importing `wandb` (#1311)
- Fixed `WandbLogger` to be used with 'ddp' - allow reinits in sub-processes (#1149, #1360)
- Made `training_epoch_end` behave like `validation_epoch_end` (#1357)
- Fixed `fast_dev_run` running validation twice (#1365)
- Fixed pickle error from quick patch `__code__` (#1352)
- Fixed memory leak on GPU0 (#1094, #1349)
- Fixed checkpointing interval (#1272)
- Fixed validation and training loops run the partial dataset (#1192)
- Fixed running `on_validation_end` only on main process in DDP (#1125)
- Fixed `load_spawn_weights` only in proc rank 0 (#1385)
- Fixes `use_amp` issue (#1145)
- Fixes using deprecated `use_amp` attribute (#1145)
- Fixed Tensorboard logger error: lightning_logs directory not exists in multi-node DDP on nodes with rank != 0 (#1375)
- Fixed `Unimplemented backend XLA` error on TPU (#1387)

## Contributors

@alexeykarnachev, @amoudgl, @areshytko, @asafmanor, @awaelchli, @bkkaggle, @bmartinn, @Borda, @borisdayma, @cmpute, @djbyrne, @ethanwharris, @gerardrbentley, @jbschiratti, @jeremyjordan, @justusschock, @monney, @mpariente, @pertschuk, @rmrao, @S-aiueo32, @shubhamagarwal92, @SkafteNicki, @sneiman, @tullie, @vanpelt, @williamFalcon, @xingzhaolee

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.7.2)

---

## 0.7.1: Minor deprecation fix
**Published:** 2020-03-13

Monir bug fix with `print` issues and `data_loader` (#1080)

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.7.1)

---

## 0.7.0: TPU support & profiling
**Published:** 2020-03-10
**Pre-release**

## Overview

This is the first joint release between [pytorch-bearer](http://www.pytorchbearer.org) and Lightning, here we come ...

This release adds support for training models on Tensor Processing Units (TPU). We can now train models on GPUs and TPUs by changing a single parameter in `Trainer` (see docs). We are also bringing the flexibility of Bearer into Lightning by allowing for arbitrary user-defined callbacks, see [docs](https://pytorch-lightning.readthedocs.io/en/0.7.0/callbacks.html). 

We are also including a profiler that allows Lightning users to identify training bottlenecks (see [docs](https://pytorch-lightning.readthedocs.io/en/0.7.0/profiler.html)).

This release also includes automatic sampler setup depending on the selected backend, Lightning configures the sampler correctly (no need for user input). 

The loggers have also been extended to support for multiple concurrent loggers to be passed to `Trainer` as an iterable, [docs](https://pytorch-lightning.readthedocs.io/en/0.7.0/loggers.html) and added support for step-based [learning rate scheduling](https://pytorch-lightning.readthedocs.io/en/0.7.0/optimizers.html#learning-rate-scheduling).

At last, lots of bug fixes (see below).

## Detail changes

### Added

- Added automatic sampler setup. Depending on DDP or TPU, lightning configures the sampler correctly (user needs to do nothing)  (#926)
- Added `reload_dataloaders_every_epoch=False` flag for trainer. Some users require reloading data every epoch  (#926)
- Added `progress_bar_refresh_rate=50` flag for trainer. The refresh rate on notebooks  (#926)
- Updated governance docs
- Added a check to ensure that the metric used for early stopping exists before training commences (#542)
- Added `optimizer_idx` argument to `backward` hook (#733)
- Added `entity` argument to `WandbLogger` to be passed to `wandb.init` (#783)
- Added a tool for profiling training runs (#782)
- Improved flexibility for naming of TensorBoard logs, can now set `version` to a `str` to just save to that directory, and use `name=''` to prevent experiment-name directory (#804)
- Added option to specify `step` key when logging metrics (#808)
- Added `train_dataloader`, `val_dataloader` and `test_dataloader` arguments to `Trainer.fit()`, for alternative data parsing (#759)
- Added Tensor Processing Unit (TPU) support (#868)
- Added semantic segmentation example (#751, #876, #881)
- Split callbacks in multiple files (#849)
- Support for user-defined callbacks (#889 and #950)
- Added support for multiple loggers to be passed to `Trainer` as an iterable (e.g. list, tuple, etc.) (#903)
- Added support for step-based learning rate scheduling (#941)
- Added support for logging hparams as `dict` (#1029)
- Checkpoint and early stopping now work without val. step (#1041)
- Support graceful training cleanup after Keyboard Interrupt (#856, #1019)
- Added type hints for function arguments (#912)
- Added default `argparser` for `Trainer` (#952, #1023)
- Added TPU gradient clipping (#963)
- Added max/min number of steps in Trainer (#728)


### Changed

- Changed default TQDM to use `tqdm.auto` for prettier outputs in IPython notebooks (#752)
- Changed `pytorch_lightning.logging` to `pytorch_lightning.loggers` (#767)
- Moved the default `tqdm_dict` definition from Trainer to `LightningModule`, so it can be overridden by the user (#749)
- Moved functionality of `LightningModule.load_from_metrics` into `LightningModule.load_from_checkpoint` (#995)
- Changed Checkpoint path parameter from `filepath` to `dirpath` (#1016)
- Freezed models `hparams` as `Namespace` property (#1029)
- Dropped `logging` config in package init (#1015)
- Renames model steps (#1051)
    * `training_end` >> `training_epoch_end`
    * `validation_end` >> `validation_epoch_end`
    * `test_end` >> `test_epoch_end`
- Refactor dataloading, supports infinite dataloader (#955)
- Create single file in `TensorBoardLogger` (#777)

### Deprecated

- Deprecated `pytorch_lightning.logging` (#767)
- Deprecated `LightningModule.load_from_metrics` in favour of `LightningModule.load_from_checkpoint` (#995, #1079)
- Deprecated `@data_loader` decorator  (#926)
- Deprecated model steps `training_end`, `validation_end` and `test_end` (#1051, #1056)

### Removed

- Removed dependency on `pandas` (#736)
- Removed dependency on `torchvision` (#797)
- Removed dependency on `scikit-learn` (#801)

### Fixed

- Fixed a bug where early stopping `on_end_epoch` would be called inconsistently when `check_val_every_n_epoch == 0` (#743)
- Fixed a bug where the model checkpoint didn't write to the same directory as the logger (#771)
- Fixed a bug where the `TensorBoardLogger` class would create an additional empty log file during fitting (#777)
- Fixed a bug where `global_step` was advanced incorrectly when using `accumulate_grad_batches > 1` (#832)
- Fixed a bug when calling `self.logger.experiment` with multiple loggers (#1009)
- Fixed a bug when calling `logger.append_tags` on a `NeptuneLogger` with a single tag (#1009)
- Fixed sending back data from `.spawn` by saving and loading the trained model in/out of the process (#1017)
- Fixed port collision on DDP (#1010)
- Fixed/tested pass overrides (#918)
- Fixed comet logger to log after train (#892)
- Remove deprecated args to learning rate step function (#890)

## Contributors

@airglow, @akshaykvnit, @AljoSt, @AntixK, @awaelchli, @baeseongsu, @bobkemp, @Borda, @calclavia, @Calysto, @djbyrne, @ethanwharris, @fdelrio89, @hadim, @hanbyul-kim, @jeremyjordan, @kuynzereb, @luiscape, @MattPainter01, @neggert, @onkyo14taro, @peteriz, @shoarora, @SkafteNicki, @smallzzy, @srush, @theevann, @tullie, @williamFalcon, @xeTaiz, @xssChauhan, @yukw777

_If we forgot someone due to not matching commit email with GitHub account, let us know :]_

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.7.0)

---

## 0.6.0: Simplifications & new docs
**Published:** 2020-01-21

This release focused on a ton of bug fixes, small optimizations to training but most importantly, clean new docs!

## Major changes

We have released [New documentation](https://pytorch-lightning.readthedocs.io/en/0.6.0/), please bear with us as we fix broken links and patch in missing pieces. 
This project moved to new org [PyTorchLightning](https://github.com/PyTorchLightning), so no longer the root sits on WilliamFalcon/PyTorchLightning. 
We have added own custom Tensorboard logger as default logger. 
We have upgrade Continues Integration to speed up the automatic testing. 
We have fixed GAN training - supporting multiple optimizers.

## Complete changelog

### Added

- Added support for resuming from a specific checkpoint via `resume_from_checkpoint` argument (#516)
- Added support for `ReduceLROnPlateau` scheduler (#320)
- Added support for Apex mode `O2` in conjunction with Data Parallel (#493)
- Added option (`save_top_k`) to save the top k models in the `ModelCheckpoint` class (#128)
- Added `on_train_start` and `on_train_end` hooks to `ModelHooks` (#598)
- Added `TensorBoardLogger` (#607)
- Added support for weight summary of model with multiple inputs (#543)
- Added `map_location` argument to `load_from_metrics` and `load_from_checkpoint` (#625)
- Added option to disable validation by setting `val_percent_check=0` (#649)
- Added `NeptuneLogger` class (#648)
- Added `WandbLogger` class (#627)


### Changed

- Changed the default progress bar to print to stdout instead of stderr (#531)
- Renamed `step_idx` to `step`, `epoch_idx` to `epoch`, `max_num_epochs` to `max_epochs` and `min_num_epochs` to `min_epochs` (#589)
- Renamed several `Trainer` atributes:  (#567)
	* `total_batch_nb` to `total_batches`,
	* `nb_val_batches` to `num_val_batches`,
	* `nb_training_batches` to `num_training_batches`,
	* `max_nb_epochs` to `max_epochs`,
	* `min_nb_epochs` to `min_epochs`,
	* `nb_test_batches` to `num_test_batches`,
	* and `nb_val_batches` to `num_val_batches` (#567)
- Changed gradient logging to use parameter names instead of indexes (#660)
- Changed the default logger to `TensorBoardLogger` (#609)
- Changed the directory for tensorboard logging to be the same as model checkpointing (#706)


### Deprecated

- Deprecated `max_nb_epochs` and `min_nb_epochs` (#567)
- Deprecated the `on_sanity_check_start` hook in `ModelHooks` (#598)


### Removed

- Removed the `save_best_only` argument from `ModelCheckpoint`, use `save_top_k=1` instead (#128)


### Fixed

- Fixed a bug which ocurred when using Adagrad with cuda (#554)
- Fixed a bug where training would be on the GPU despite setting `gpus=0` or `gpus=[]` (#561)
- Fixed an error with `print_nan_gradients` when some parameters do not require gradient (#579)
- Fixed a bug where the progress bar would show an incorrect number of total steps during the validation sanity check when using multiple validation data loaders (#597)
- Fixed support for PyTorch 1.1.0 (#552)
- Fixed an issue with early stopping when using a `val_check_interval < 1.0` in `Trainer` (#492)
- Fixed bugs relating to the `CometLogger` object that would cause it to not work properly (#481)
- Fixed a bug that would occur when returning `-1` from `on_batch_start` following an early exit or when the batch was `None` (#509)
- Fixed a potential race condition with several processes trying to create checkpoint directories (#530)
- Fixed a bug where batch 'segments' would remain on the GPU when using `truncated_bptt > 1` (#532)
- Fixed a bug when using `IterableDataset` (#547](https://github.com/PyTorchLightning/pytorch-lightning/pull/547))
- Fixed a bug where `.item` was called on non-tensor objects (#602)
- Fixed a bug where `Trainer.train` would crash on an uninitialized variable if the trainer was run after resuming from a checkpoint that was already at `max_epochs` (#608)
- Fixed a bug where early stopping would begin two epochs early (#617)
- Fixed a bug where `num_training_batches` and `num_test_batches` would sometimes be rounded down to zero (#649)
- Fixed a bug where an additional batch would be processed when manually setting `num_training_batches` (#653)
- Fixed a bug when batches did not have a `.copy` method (#701)
- Fixed a bug when using `log_gpu_memory=True` in Python 3.6 (#715)
- Fixed a bug where checkpoint writing could exit before completion, giving incomplete checkpoints (#689)
- Fixed a bug where `on_train_end` was not called when early stopping (#723)


## Contributors

@akhti, @alumae, @awaelchli, @Borda, @borisdayma, @ctlaltdefeat, @dreamgonfly, @elliotwaite, @fdiehl, @goodok, @haossr, @HarshSharma12, @Ir1d, @jakubczakon, @jeffling, @kuynzereb, @MartinPernus, @matthew-z, @MikeScarp, @mpariente, @neggert, @rwesterman, @ryanwongsa, @schwobr, @tullie, @vikmary, @VSJMilewski, @williamFalcon, @YehCF

If we forgot someone due to not matching commit email with GitHub account, let us know :]

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.6.0)

---

## 0.5.3: Generalization!
**Published:** 2019-11-06

# Generalization release
The main focus of this release was on adding flexibility and generalization to support broad research cases.

Next release will be Dec 7th (every 30 days).

## Internal Facebook support
@lorenzoFabbri @tullie @myleott @ashwinb  @shootingsoul @vreis
These features were added to support FAIR, FAIAR and broader ML across other FB teams.

In general, we can expose any part that isn't exposed yet where someone might want to override the lightning implementation.

1. Added truncated back propagation through time support (thanks @tullie).

```python
Trainer(truncated_bptt_steps=2)
```

2. Added iterable datasets. 

```python
# return iterabledataset
def train_dataloader(...):
    ds = IterableDataset(...)
    return Dataloader(ds)

# set validation to a fix number of batches
# (checks val every 100 train epochs)
Trainer(val_check_interval=100)
```

3. Add ability to customize backward and other training parts:
```python
    def backward(self, use_amp, loss, optimizer):
        """
        Override backward with your own implementation if you need to
        :param use_amp: Whether amp was requested or not
        :param loss: Loss is already scaled by accumulated grads
        :param optimizer: Current optimizer being used
        :return:
        """
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
```

3. DDP custom implementation support (override these hooks):
```python
    def configure_ddp(self, model, device_ids):
        """
        Override to init DDP in a different way or use your own wrapper.
        Must return model.
        :param model:
        :param device_ids:
        :return: DDP wrapped model
        """
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        return model

    def init_ddp_connection(self, proc_rank, world_size):
        """
        Connect all procs in the world using the env:// init
        Use the first node as the root address
        """

        # use slurm job id for the port number
        # guarantees unique ports across jobs from same grid search
        try:
            # use the last 4 numbers in the job id as the id
            default_port = os.environ['SLURM_JOB_ID']
            default_port = default_port[-4:]

            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000

        except Exception as e:
            default_port = 12910

        # if user gave a port number, use that one instead
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)

        # figure out the root node addr
        try:
            root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
        except Exception:
            root_node = '127.0.0.2'

        root_node = self.trainer.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)
```

4. Support for your own apex init or implementation.
```python
    def configure_apex(self, amp, model, optimizers, amp_level):
        """
        Override to init AMP your own way
        Must return a model and list of optimizers
        :param amp:
        :param model:
        :param optimizers:
        :param amp_level:
        :return: Apex wrapped model and optimizers
        """
        model, optimizers = amp.initialize(
            model, optimizers, opt_level=amp_level,
        )

        return model, optimizers
```

5. DDP2 implementation (inspired by parlai and @stephenroller).  
DDP2 acts as DP in the node and DDP across nodes. 
As a result, an optional method is introduced ```training_end```  
where you can use the outputs of ```training_step``` (performed on each GPU with a portion of the batch),
to do something with the outputs of all batches on the node (ie: negative sampling).
```python
Trainer(distributed_backend='ddp2')  

def training_step(...):
    # x is 1/nb_gpus of the full batch
    out = model(x)
    return {'out': out}

def training_end(self, outputs):
     # all_outs has outs from ALL gpus 
     all_outs = outputs['out']
     loss = softmax(all_outs)
     return {'loss': loss}
```

## Logging
- More logger diversity including Comet.ml.
- Versioned logs for all loggers.   
- switched from print to logging

## progress bar
- now the progress bar has a full bar for the full train + val epochs and a second bar visible only during val.

## loading
- checkpoints now store hparams
- no need to pass tags.csv to restore state because it lives in the checkpoint.

## Slurm resubmit with apex + ddp
- Fixes issue of ddp restore weights blowing out GPU memory (load on cpu first then GPU).
- Saves apex states automatically and restores it for a checkpoint.

## Refactoring
- internal code made modular through Mixins for ease of readability and to minimize merge conflicts.

## Docs
- Tons of doc improvements.

## Thanks!
Thank you to the amazing contributor community! Especially @neggert  and @Borda for reviewing PRs and taking care of a good number of Github issues. The community is thriving and has really embraced making Lightning better.

Great job everyone!





[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.5.3)

---

## 0.5.1: Simpler interface, new features
**Published:** 2019-10-05

# 0.5.1   

### Simpler interface  
All trainers now have a default logger, early stopping and checkpoint object. To modify the behavior, pass in your own versions of those. 

- Removed collisions with logger versions by tying it to job id.

### Features
- Added new DDP implementation. It uses DP in a node but allows multiple nodes. Useful for models which need negative samples, etc...
```python
Trainer(distributed_backend='ddp2')
```   

- support for LBFGS. If you pass in LBFGS Lightning handles the closure for you automatically.
- No longer need to set master port, Lightning does it for you using the job id.

# Minor changes   
- training_step and validation_end now return two separate dicts, one for the progress bar and one for logging.

- Added options to memory printing: 'min_max' logs only the max/min memory use. 'all' logs all the GPUs on the root node.  

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.5.1)

---

## 0.5.0: API clean up
**Published:** 2019-09-26

This release has breaking API changes. See #124 for all details. 
Syntax changes are:   
```
in trainer options use: train, test, val  
for data: val_dataloader, test_dataloader, train_dataloader
data_batch -> batch    
prog -> progress   
gradient_clip -> gradient_clip_val    
add_log_row_interval -> row_log_interval
```

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.5.0)

---

## 0.4.9: Various ddt improvements
**Published:** 2019-09-16

This release does the following: 

- Moves SLURM resubmit from test-tube to PL (which removes the need for cluster parameter).
- Cluster checkpoint done by Lightning now (not test-tube). Also doesn't require a checkpoint object to restore weights when on cluster.     
- Loads all models on CPU when restoring weights to avoid OOM issues in PyTorch. User now needs to move to GPU manually. However, if using Lightning, lightning will move to correct GPUs automatically.   
- Fixes various subtle bugs in DDP implementation.  
- documentation updates


[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.4.9)

---

## 0.4.4: New features
**Published:** 2019-08-12

- validation_step, val_dataloader are now optional.   
- enabled multiple dataloaders for validation.    
- support for latest test-tube logger optimized for PT 1.2.0.   
- lr_scheduler now activated after epoch    

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.4.4)

---

## 0.4.0: Stable fully-featured release
**Published:** 2019-08-08

# 0.4.0    
0.4.0 is the first public release after a short period testing with public users. Thanks for all the help ironing out bugs to get Lightning to run on everything from notebooks to local to server machines.

#### This release includes:
- Extensively tested code.   
- Cleaner API to accommodate the various research use cases   

#### New features    
- No need for experiment object in trainer.   
- Training continuation (not just weights, but also epoch, global step, etc...) 
    - if the folder the checkpoint callback uses has weights, it loads the last weights automatically.
- training step and validation step don't reduce outputs automatically anymore. This fixes issues with reducing generated outputs for example (images, text).   
- 16-bit can now be used with a single GPU (no DP or DDP in this case). bypasses issue with NVIDIA apex and PT compatibility for DP+16-bit training. 





[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.4.0)

---

## 0.3.6: Simple data loader
**Published:** 2019-07-25

Simplified data loader.

Added a decorator to do lazy loading internally:

Old:  
```python
@property
def tng_dataloader(self):
      if self._tng_dataloader is None:
               self._tng_dataloader = DataLoader(...)
      return self.tng_dataloder
```

Now:

```python
@ptl.data_loader
def tng_dataloader(self):
      return DataLoader(...)
````

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.3.6)

---

## 0.3.5: Tests!
**Published:** 2019-07-25

Fully tested!

Includes:
- Code coverage (99%)
- Full tests that run multiple models in different configs
- Full tests that test specific functionality in trainer.

[View on GitHub](https://github.com/Lightning-AI/pytorch-lightning/releases/tag/0.3.5)

---

