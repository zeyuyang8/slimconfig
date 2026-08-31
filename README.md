# slimconfig

YAML configs merged onto typed dataclass schemas — a lightweight Hydra stand-in in ~500 lines,
built on [OmegaConf](https://omegaconf.readthedocs.io).

Four rules, enforced at load time:

* **A config class is a dataclass of leaves, nested config classes, and tables of one.** Groups are
  *composed* as fields, not inherited as mixins, so every value's name says which group it came from —
  and no field is typed `Any`: one that holds a config names the class it holds.
* **A config file names the class it fills** — `_schema: myproject.train.TrainConfig` — and a
  hierarchical class takes a hierarchical file. Rename the class and its configs break loudly.
* **Every leaf is required.** A schema's leaves all default to `MISSING`, so a config has to set each
  one explicitly — "off" is still written out as `null`, an empty collection as `[]`. Nothing is
  silently inherited, and unknown keys are rejected.
* **Where a run writes is not part of its config.** `run_dir` and `log` are the launcher's, from the
  command line or the script.

Plus what a research/experiment runner actually needs: `defaults:` composition at any depth, one
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
    model: str = MISSING
    optim: Optim = field(default_factory=Optim)   # a nested group: a field, not a base class
    resume_from: str | None = MISSING             # "off" must still be spelled `null`

cfg = load_config(TrainConfig, sys.argv[1:])   # -> a real TrainConfig instance
print(cfg.optim.lr)
```

```yaml
# configs/train.yaml
_schema: train.TrainConfig     # the class this file fills, by dotted import path
model: llama-3-8b
optim:                         # a nested class takes a nested block
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

A nested field must carry `field(default_factory=...)`; a schema that forgets it, or that contains
itself, is rejected by name at launch (`check_schema`). A class defined in the launched script is
named by the script's own name (`train.TrainConfig` above, not `__main__.TrainConfig`) — the
already-running module is used, never imported a second time.

### Groups are composed, not inherited

A config class that needs another one's fields declares a **field** of that type. Inheriting them as
a mixin flattens the borrowed names into the parent's own namespace, so the YAML can no longer say
where a value came from and two mixins can silently collide on a name.

```python
@dataclass
class TrainConfig(Optim, Data):        # ✗ lr and path land in TrainConfig's namespace
    ...

@dataclass
class TrainConfig:                     # ✓ optim.lr and data.path, in the class and in the YAML
    optim: Optim = field(default_factory=Optim)
    data: Data = field(default_factory=Data)
```

### Tables: several of the same group, keyed

A field typed `dict[str, C]` — or `dict[SomeEnum, C]` — for a config class `C` holds **several** of
that group, one per key: per model, per task, per environment. Entries are validated exactly like a
group (unknown key rejected, values type-checked), and an `Enum` key type checks the *keys* too.

```python
@dataclass
class Sweep:
    axes: dict[str, Axis] = MISSING          # a table of Axis
    weights: dict[str, float] = MISSING      # a dict of plain values: just a leaf
```

A table needs no `default_factory` — its entries do not exist until a config file names their keys —
and a `defaults:` mounts on one *entry* (`axes.lr`), never on the table itself, which has no single
class to fill.

Mind the difference when a later spec overrides an earlier one. Mappings merge **key by key**, which is
what you want for a table (`axes`) and wrong for a leaf that holds a set of things (`weights`): setting
`weights: {a: 1, b: 1}` on top of `{a: 1, c: 1}` gives you all three. A layer can add a key to such a
leaf but never drop one, so if two variants need two different sets, give each its own whole value at a
node where only one of them applies — don't stack them.

### Layers: a config that is allowed to say nothing

Sometimes a run is composed at runtime out of several partial configs: a base, then whatever the
per-model and per-method tables say for the cell being run. Type such a field with `partial_of`, not
`Any`:

```python
from slimconfig import partial_of, stated

CellPart = partial_of(RunConfig)                          # every field of RunConfig, none required

@dataclass
class Matrix:
    base: CellPart = field(default_factory=CellPart)
    per_method: dict[Method, CellPart] = MISSING

cell = load_config(RunConfig, [stated(m.base), stated(m.per_method[method])])   # later wins
```

`partial_of(C)` is a real subclass of `C`, so a fragment written for `C` still mounts under it and
every key and value is validated as usual; the one thing it drops is *required*. `stated(layer)`
reads back only the fields that layer actually set, as a plain dict, which is what the resolver
merges. An unset field stays `MISSING` rather than becoming `None`, so a layer that writes
`resume_from: null` still overwrites — `null` is a value, silence is not.

## Share configs with `defaults:`

Any mapping — the file's top level *or any block inside it* — may carry a `defaults:` list of paths,
meaning "this block starts from these files". The listed files are composed in order and the mapping
that wrote `defaults:` wins on top; composition is recursive, and cycles are caught.

```yaml
# configs/optim/cosine.yaml        # configs/train_7b.yaml
_schema: train.Optim               _schema: train.TrainConfig
lr: 2.0e-4                         defaults: [configs/train.yaml]
warmup_steps: 100                  model: llama-3-7b
                                   optim:
                                     defaults: [configs/optim/cosine.yaml]
                                     lr: 1.0e-4      # this file wins
```

Because the mount point is named by the *parent*, a shared fragment states its own fields at its own
top level and does not have to know where it will land. That is what the `_schema:` line buys: the
fragment says it is an `Optim`, the parent says an `Optim` goes under `optim:`, and a fragment
mounted at the wrong block is an error naming both:

```
config file 'configs/optim/cosine.yaml' says it fills train.Optim, but it is being merged onto
`data` of train.TrainConfig, which is train.Data
```

A file may declare a **base** of the class at its mount point — a base states a subset of the fields,
which is exactly what a shared fragment does. The other direction is rejected.

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

def main(cfg: TrainConfig, run_dir: str) -> int:
    print(f"training {cfg.model}")           # printed here -> also in <run_dir>/train.log
    torch.save(state, f"{run_dir}/model.pt") # results go in the same folder
    return 0

if __name__ == "__main__":
    run(main, log="train.log")
```

```bash
python train.py configs/train.yaml optim.lr=1e-4 --run-dir runs/exp1
```

```
runs/exp1/
├── config.yaml           # the exact resolved config, `_schema:` line and all — a config file like
│                         #   any other: `python train.py runs/exp1/config.yaml --run-dir runs/redo`
├── metadata.json         # argv, cwd, run dir, git commit + dirty flag, start time, host
├── train.log             # everything the run printed
└── model.pt              # …and whatever the function wrote there
```

`run` takes the function, optionally the config, and where to write:

| Input | What it is |
| --- | --- |
| `function` | the routine to run. Its **first argument is annotated with its config class** — the function *is* its config, so `run` reads the schema off that annotation and there is nothing else to name. An optional **second argument, `run_dir: str`**, is the folder. Returns this process's exit status (`None` → 0). |
| `config` | the YAML file to load it from. Omitted, it comes off the command line (`<config.yaml> [key=value ...]`) — the usual way a script is launched. |
| `run_dir=` | the folder this run owns: a path, or a **function** returning one — of the loaded config, and of the config file itself if it takes a second argument. `--run-dir PATH` on the command line wins over it; one of the two must say, or the launch stops. |
| `log=` | the log file inside that folder, `None` for no log. `--log NAME` and `--no-log` win over it — the latter is how a distributed launch says "not every rank appends to one file". |

Keyword arguments are `key=value` overrides on top, the same ones the command line takes:

```python
run(main, "configs/train.yaml", run_dir="runs/exp1", **{"optim.lr": 1e-4})
```

An identity-addressed output tree — a folder named after what is in the config — is one function in
code, rather than the same interpolation copied into every config file that lands there:

```python
run(main, run_dir=lambda cfg: f"runs/{cfg.model}/lr{cfg.optim.lr}")
```

The other naming rule worth having — a folder named after the config that produced it, so a result and
the file that asked for it are findable from each other — needs the file, which only the launcher
knows. A run-dir function that takes a second argument gets it (`""` if the config came from
overrides alone):

```python
run(main, run_dir=lambda cfg, config: f"runs/{Path(config).stem}")
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
`peek`, `schema_of`, `load_config` and `merge_specs` are public for exactly that.

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
| `run(fn[, config], *, run_dir, log, **overrides)` | run this process as one run: config loaded (schema off `fn`'s annotation), folder opened, logged, called, exit with its status |
| `load_config(schema, specs)` | merge specs onto a dataclass schema → a populated instance |
| `merge_specs(specs)` | merge specs into one unvalidated `DictConfig` |
| `peek(specs, key)` | read one (dotted) key before choosing a schema |
| `schema_of(path)` | the class a config file was written for, without loading it |
| `start_run(run_dir, config)` | create the run folder, write `config.yaml` + `metadata.json` |
| `tee_stdout(path, banner=None)` | context manager: also append stdout to `path` |
| `compose(path[, node])` | one YAML → the composed `DictConfig` plus every `_schema:` claim in it |
| `load_mapping_yaml(path)` | the same, keeping only the `DictConfig` |
| `load_yaml(path)` | one YAML → `dict`, plain PyYAML, no composition |
| `partial_of(cls[, name])` | the schema of one LAYER of `cls`: a subclass whose fields may be left unset |
| `stated(layer)` | what one layer actually said, as a plain dict |
| `is_partial(cls)` | is this a layer schema? |
| `check_schema(cls)` | reject a config class that cannot be filled from YAML |
| `resolve_schema(dotted)` | import the class a `_schema:` line names |
| `field_schema(root, node)` | the class that belongs at a dotted node of `root` |
| `fields_of(cls)` | each field's shape: `value`, `group`, or `table` (and of what class) |
| `schema_name(cls)` | the dotted path a `_schema:` line would name `cls` by |

A *spec* is a YAML file path, a `dotted.key=value` string, or a ready-made mapping/`DictConfig` —
so a caller can merge values it computed at runtime under the same "later wins" rule. A mapping spec
needs no `_schema:`: values a routine computed are code, and code is already typed.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
