# slimconfig — YAML configs onto typed dataclass schemas, a lightweight Hydra stand-in.
#
#     from slimconfig import load_config
#     cfg = load_config(MyConfig, sys.argv[1:])   # MyConfig: a dataclass of MISSING fields
#
# See config.py for the YAML loader + `defaults:` composition and the ${now:...} / ${from_yaml:...}
# resolvers, structured.py for the typed, all-fields-required schema loader, the `mode` dispatcher,
# and the run-folder snapshot, and paths.py for the project-root rule relative paths resolve against.

from .config import load_mapping_yaml, load_yaml
from .paths import project_root, resolve_path
from .structured import (
    Spec,
    dispatch,
    load_config,
    merge_specs,
    peek,
    start_run,
)

__version__ = "0.2.0"

__all__ = [
    "Spec",
    "dispatch",
    "load_config",
    "load_mapping_yaml",
    "load_yaml",
    "merge_specs",
    "peek",
    "project_root",
    "resolve_path",
    "start_run",
]
