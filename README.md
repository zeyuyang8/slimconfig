# slimconfig

YAML configs merged onto typed dataclass schemas — a lightweight Hydra stand-in in ~300 lines,
built on [OmegaConf](https://omegaconf.readthedocs.io).

Two rules, enforced at load time:

* **Every field is required.** A schema's leaves all default to `MISSING`, so a config has to set
  each one explicitly — a nullable field that is "off" is still written out as `null`, an empty
  collection as `[]`. Nothing is silently inherited.
* **Unknown keys are rejected.** A typo in a YAML key is an error, not a value that goes nowhere.

Plus what a research/experiment runner actually needs: Hydra-style `defaults:` composition, a
`mode` dispatcher, and a run folder — holding the config, the log, and the results of one run.

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

## Run a function

`@run(Schema)` makes a function the entry point of a run: it is called with the config named on the
command line, and the folder that config's `run_dir` points at is created, snapshotted, and logged
into around the call.

```python
# train.py
from slimconfig import run

@run(TrainConfig)
def main(cfg: TrainConfig) -> int:
    print(f"training {cfg.model}")             # printed here -> also in <run_dir>/main.log
    torch.save(state, f"{cfg.run_dir}/model.pt")   # results go in the same folder
    return 0

if __name__ == "__main__":
    raise SystemExit(main())                   # specs default to sys.argv[1:]
```

```bash
python train.py configs/train.yaml optim.lr=1e-4
```

```
runs/20260828-114500/     # whatever run_dir says
├── config.yaml           # the exact resolved config — re-run with `python train.py <this>`
├── run_meta.json         # argv, cwd, git commit + dirty flag, start time, host
├── main.log              # everything the run printed
└── model.pt              # …and whatever the function wrote there
```

`main(["configs/train.yaml", "optim.lr=1e-4"])` calls the same entry point from Python. Options:

| | |
| --- | --- |
| `@run(Schema)` | call the function with a loaded `Schema` instance |
| `@run` (bare) | call it with the raw specs — for an entry point that picks its schema off another key and loads its own config |
| `@run(Schema, log="train.log")` | name the log file (`{name}` stands for the function's name; the default is `"{name}.log"`) |
| `@run(Schema, log=None)` | no log file, just the snapshot |

The log captures **stdout only** — progress bars go to stderr, and 45 KB of progress bars is not a
log — appends (a resumed run adds to the folder's history, under a banner naming the invocation),
and tees, so output still reaches the terminal. Child processes hold the real fd 1 and are not
captured.

## Dispatch on `mode`

For a single entry point that fans out to several jobs, `dispatch` reads `mode`, opens and
snapshots `run_dir` (logging to `<mode>.log`), then calls the matching handler:

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

`@run` and `dispatch` open the folder for you. The pieces are also usable on their own, for a
routine that opens a folder the config does not name — one cell of a sweep, say:

```python
from slimconfig import open_run, start_run, tee_stdout

with open_run(cfg, log="eval.log"):       # snapshot + log, run_dir read off cfg
    ...

start_run(cell_dir, cell_cfg)             # just the snapshot, into a folder you chose
with tee_stdout(f"{cell_dir}/eval.log"):  # just the log
    ...
```

Everything a run produces goes in that same folder, so a result is never separated from the config
that made it. The snapshot is best-effort — provenance never aborts a run.

## API

| | |
| --- | --- |
| `@run(schema, log=...)` | make a function the entry point of a run: load, open the folder, log, call |
| `load_config(schema, specs)` | merge specs onto a dataclass schema → a populated instance |
| `merge_specs(specs)` | merge specs into one unvalidated `DictConfig` |
| `peek(specs, key)` | read one top-level key before choosing a schema |
| `dispatch(modes, specs)` | `mode` → handler, with the run folder opened, snapshotted, and logged |
| `open_run(config, log=...)` | context manager: create the folder `config` names, snapshot it, tee the log |
| `start_run(run_dir, config)` | create the run folder, write `config.yaml` + `run_meta.json` |
| `tee_stdout(path, banner=None)` | context manager: also append stdout to `path` |
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
