# slimconfig — YAML configs onto typed dataclass schemas, a lightweight Hydra stand-in.
#
#     from slimconfig import load_config
#     cfg = load_config(MyConfig, sys.argv[1:])   # MyConfig: a dataclass of MISSING fields
#
# See config.py for the YAML loader + `defaults:` composition and the ${now:...} / ${from_yaml:...}
# resolvers, and structured.py for the typed, all-fields-required schema loader, the `mode` dispatcher,
# and the run-folder snapshot.

from .config import load_mapping_yaml, load_yaml
from .structured import (
    Spec,
    dispatch,
    load_config,
    merge_specs,
    peek,
    start_run,
)

__version__ = "0.1.0"

__all__ = [
    "Spec",
    "dispatch",
    "load_config",
    "load_mapping_yaml",
    "load_yaml",
    "merge_specs",
    "peek",
    "start_run",
]
