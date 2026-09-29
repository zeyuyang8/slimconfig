# slimconfig — YAML configs onto typed dataclass schemas, a lightweight Hydra stand-in.
#
#     from slimconfig import run
#
#     def main(cfg: MyConfig, run_dir: str) -> int:   # MyConfig: a dataclass of MISSING leaves and
#         ...                                         # nested config classes; results go under run_dir
#
#     if __name__ == "__main__":                      # python main.py config=configs/my.yaml home=runs/x
#         run(main)
#
# Four rules:
#   * a config class is a @dataclass subclassing `Config` of leaves, nested classes and `dict[K, C]` tables.
#   * a config file names its class (`_ > <dotted.path>:`) and sets only that class's fields.
#   * every leaf is required; nothing is silently defaulted, and "off" is spelled `null`.
#   * where a run writes is not config: `config=` and `home=` belong to the launcher; the log is `run.log`.

from .config import compose, load_mapping_yaml, load_yaml
from .partials import is_partial, partial_of, stated
from .paths import project_root, resolve_path
from .runs import Run, run, start_run, tee_stdout
from .schemas import Config, Schema
from .structured import Spec, load_config, merge_specs, peek, schema_of

__version__ = "0.15.0"

# The public API; loader record types (Claim, Composed, Key) stay in slimconfig.config.
__all__ = [
    "Config",
    "Run",
    "Schema",
    "Spec",
    "compose",
    "is_partial",
    "load_config",
    "load_mapping_yaml",
    "load_yaml",
    "merge_specs",
    "partial_of",
    "peek",
    "project_root",
    "resolve_path",
    "run",
    "schema_of",
    "start_run",
    "stated",
    "tee_stdout",
]
