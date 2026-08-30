# slimconfig — YAML configs onto typed dataclass schemas, a lightweight Hydra stand-in.
#
#     from slimconfig import run
#
#     def main(cfg: MyConfig) -> int:             # MyConfig: a dataclass of MISSING fields, two of
#         ...                                     # which are `run_dir` (results go under it) and `log`
#
#     if __name__ == "__main__":                  # python main.py configs/my.yaml
#         run(main)
#
# See config.py for the YAML loader + `defaults:` composition and the ${now:...} / ${from_yaml:...}
# resolvers, structured.py for the typed, all-fields-required schema loader, runs.py for the run folder
# every entry point owns — its config snapshot, its log, its results — and paths.py for the project-root
# rule relative paths resolve against.

from .config import load_mapping_yaml, load_yaml
from .paths import project_root, resolve_path
from .runs import (
    RUN_FIELDS,
    run,
    start_run,
    tee_stdout,
)
from .structured import (
    Spec,
    load_config,
    merge_specs,
    peek,
)

__version__ = "0.5.0"

__all__ = [
    "RUN_FIELDS",
    "Spec",
    "load_config",
    "load_mapping_yaml",
    "load_yaml",
    "merge_specs",
    "peek",
    "project_root",
    "resolve_path",
    "run",
    "start_run",
    "tee_stdout",
]
