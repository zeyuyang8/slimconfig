# slimconfig

YAML configs merged onto typed dataclass schemas — a lightweight Hydra stand-in in ~300 lines,
built on [OmegaConf](https://omegaconf.readthedocs.io).

Two rules, enforced at load time:

* **Every field is required.** A schema's leaves all default to `MISSING`, so a config has to set
  each one explicitly — a nullable field that is "off" is still written out as `null`, an empty
  collection as `[]`. Nothing is silently inherited.
* **Unknown keys are rejected.** A typo in a YAML key is an error, not a value that goes nowhere.

Plus what a research/experiment runner actually needs: Hydra-style `defaults:` composition, a
`mode` dispatcher, and a run folder that snapshots the exact config it ran with.

## Install

```bash
pip install slimconfig
```

## Load a config

```python
# train.py
import sys
from dataclasses import dataclass, field
from omegaconf import MISSING
from slimconfig import load_config

@dataclass
class Optim:
    lr: float = MISSING
    warmup_steps: int = MISSING

@dataclass
class TrainConfig:
    run_dir: str = MISSING
    model: str = MISSING
    optim: Optim = field(default_factory=Optim)
    resume_from: str | None = MISSING   # "off" must still be spelled `null`

cfg = load_config(TrainConfig, sys.argv[1:])   # -> a real TrainConfig instance
print(cfg.optim.lr)
```

```yaml
# configs/train.yaml
run_dir: runs/${now:%Y%m%d-%H%M%S}
model: llama-3-8b
optim:
  lr: 2.0e-4
  warmup_steps: 100
resume_from: null
```

```bash
python train.py configs/train.yaml                 # a file
python train.py configs/train.yaml optim.lr=1e-4   # ...plus dotted overrides, later wins
```

Leave `warmup_steps` out and the load fails with
`TrainConfig is missing required field(s): optim.warmup_steps` — before anything runs.

## Share configs with `defaults:`

Any YAML may carry a top-level `defaults:` list of paths. Listed files merge first, in order, and
the current file wins on top; composition is recursive, and cycles are caught.

```yaml
# configs/train_7b.yaml
defaults: [configs/train.yaml, configs/optim/cosine.yaml]
model: llama-3-7b
```

Paths resolve **relative to the current working directory** (the project root scripts are launched
from), so one path convention holds wherever the including file lives. Absolute paths work too.

## Interpolation resolvers

On top of OmegaConf's own `${a.b}` interpolation, importing slimconfig registers:

| Resolver | Meaning |
| --- | --- |
| `${now:%Y%m%d-%H%M%S}` | the load time, `strftime`-formatted — one consistent stamp per process |
| `${from_yaml:configs/data.yaml,dataset.name}` | one value read out of *another* config, so a config can track a value another file owns without duplicating it |

## Dispatch on `mode`

For a single entry point that fans out to several jobs, `dispatch` reads `mode`, opens and
snapshots `run_dir`, then calls the matching handler:

```python
# run.py
import sys
from slimconfig import dispatch

MODES = {
    "train": (TrainConfig, train),      # load TrainConfig strictly, call train(cfg)
    "eval": (EvalConfig, evaluate),
    "sweep": run_sweep,                 # bare handler: gets the raw specs, loads its own schema
}
raise SystemExit(dispatch(MODES, sys.argv[1:]))
```

`mode` and `run_dir` are ordinary config keys, so a schema loaded this way declares them itself
(unknown keys are rejected).

## Run folders

`start_run(run_dir, config)` (called for you by `dispatch`) creates the folder and writes:

* `config.yaml` — the fully-resolved config, re-runnable as-is: `python run.py <run_dir>/config.yaml`
* `run_meta.json` — argv, cwd, git commit + dirty flag, start time, host

Everything a run produces goes in that same folder, so a result is never separated from the config
that made it. The snapshot is best-effort — provenance never aborts a run.

## API

| | |
| --- | --- |
| `load_config(schema, specs)` | merge specs onto a dataclass schema → a populated instance |
| `merge_specs(specs)` | merge specs into one unvalidated `DictConfig` |
| `peek(specs, key)` | read one top-level key before choosing a schema |
| `dispatch(modes, specs)` | `mode` → handler, with the run folder opened and snapshotted |
| `start_run(run_dir, config)` | create the run folder, write `config.yaml` + `run_meta.json` |
| `load_mapping_yaml(path)` | one YAML → `DictConfig`, with `defaults:` composed |
| `load_yaml(path)` | one YAML → `dict`, plain PyYAML, no composition |

A *spec* is a YAML file path, a `dotted.key=value` string, or a ready-made mapping/`DictConfig` —
so a caller can merge values it computed at runtime under the same "later wins" rule.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
