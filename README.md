# slimconfig

YAML configs merged onto typed dataclass schemas — a lightweight Hydra stand-in in ~300 lines,
built on [OmegaConf](https://omegaconf.readthedocs.io).

Two rules, enforced at load time:

* **Every field is required.** A schema's leaves all default to `MISSING`, so a config has to set
  each one explicitly — a nullable field that is "off" is still written out as `null`, an empty
  collection as `[]`. Nothing is silently inherited.
* **Unknown keys are rejected.** A typo in a YAML key is an error, not a value that goes nowhere.

Plus what a research/experiment runner actually needs: Hydra-style `defaults:` composition, one
launcher, and a run folder — holding the config, the log, and the results of one run.

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
    log: str | None = MISSING
    model: str = MISSING
    optim: Optim = field(default_factory=Optim)
    resume_from: str | None = MISSING   # "off" must still be spelled `null`

cfg = load_config(TrainConfig, sys.argv[1:])   # -> a real TrainConfig instance
print(cfg.optim.lr)
```

```yaml
# configs/train.yaml
run_dir: runs/${now:%Y%m%d-%H%M%S}
log: train.log
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

A run is a config plus the function that consumes it, and `run` is where a script hands over both —
and nothing else. It loads the config, creates and snapshots the run's folder, tees stdout into it,
calls the function, and exits with its status, so a script's `__main__` is one line with no argv
handling and no `SystemExit` of its own.

```python
# train.py
from slimconfig import run

def main(cfg: TrainConfig) -> int:
    print(f"training {cfg.model}")             # printed here -> also in <run_dir>/<cfg.log>
    torch.save(state, f"{cfg.run_dir}/model.pt")   # results go in the same folder
    return 0

if __name__ == "__main__":
    run(main)
```

```bash
python train.py configs/train.yaml optim.lr=1e-4
```

```
runs/20260828-114500/     # whatever run_dir says
├── config.yaml           # the exact resolved config — re-run with `python train.py <this>`
├── metadata.json         # argv, cwd, git commit + dirty flag, start time, host
├── train.log             # everything the run printed — whatever log says
└── model.pt              # …and whatever the function wrote there
```

`run` takes **two** things:

| Input | What it is |
| --- | --- |
| `function` | the routine to run. It takes exactly **one argument, annotated with its config class**, and returns this process's exit status (`None` → 0). The function *is* its config: `run` reads the schema off that annotation, so there is nothing else to name. |
| `config` | the YAML file to load it from. Omitted, it comes off the command line (`<config.yaml> [key=value ...]`) — the usual way a script is launched. |

The config class must declare two fields, and they are the only thing `run` needs beyond the
function: **`run_dir`**, the folder this run owns, and **`log`**, the log file inside it (`null` for
no log — a distributed launch, say, where every rank would append to one file). A schema missing
either is rejected at launch, by name.

Keyword arguments are `key=value` overrides on top, the same ones the command line takes:

```python
run(main, "configs/train.yaml", **{"optim.lr": 1e-4})
```

The log captures **stdout only** — progress bars go to stderr, and 45 KB of progress bars is not a
log — appends (a resumed run adds to the folder's history, under a banner naming the invocation),
and tees, so output still reaches the terminal. Child processes hold the real fd 1 and are not
captured.

## One job per entry point

There is no mode switch and no dispatch table: an entry point is one function, and a script that
runs several kinds of job is several scripts. Each is three lines, and each says in its own name
which routine it is.

```python
# train.py                          # eval.py
from slimconfig import run          from slimconfig import run
from jobs import train              from jobs import evaluate

if __name__ == "__main__":          if __name__ == "__main__":
    run(train)                          run(evaluate)
```

A routine whose config depends on a value inside that config — a schema chosen by `method`, a cell
resolved out of a matrix — resolves it *itself*, from a config class that is still one annotation:
`peek`, `load_config` and `merge_specs` are public for exactly that.

## Run folders

`run` opens the folder for you. The pieces are also usable on their own, for a routine that opens a
second folder of its own — one cell of a sweep, say:

```python
from slimconfig import start_run, tee_stdout

start_run(cell_dir, cell_cfg)             # just the snapshot, into a folder you chose
with tee_stdout(f"{cell_dir}/eval.log"):  # just the log
    ...
```

Everything a run produces goes in that same folder, so a result is never separated from the config
that made it. The snapshot is best-effort — provenance never aborts a run.

## API

| | |
| --- | --- |
| `run(fn[, config], **overrides)` | run this process as one run: config loaded (schema off `fn`'s annotation), folder opened, logged, called, exit with its status |
| `load_config(schema, specs)` | merge specs onto a dataclass schema → a populated instance |
| `merge_specs(specs)` | merge specs into one unvalidated `DictConfig` |
| `peek(specs, key)` | read one top-level key before choosing a schema |
| `start_run(run_dir, config)` | create the run folder, write `config.yaml` + `metadata.json` |
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
