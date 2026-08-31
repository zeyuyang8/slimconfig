# slimconfig — YAML configs onto typed dataclass schemas, a lightweight Hydra stand-in.
#
#     from slimconfig import run
#
#     def main(cfg: MyConfig, run_dir: str) -> int:   # MyConfig: a dataclass of MISSING leaves and
#         ...                                         # nested config classes; results go under run_dir
#
#     if __name__ == "__main__":                      # python main.py configs/my.yaml --run-dir runs/x
#         run(main)
#
# Four rules:
#   * a config class is a @dataclass of LEAVES, NESTED CONFIG CLASSES and TABLES of one (`dict[K, C]`) —
#     groups are composed as fields, not inherited as mixins, so a value's name says where it came from,
#     and nothing is typed `Any`: a field that holds a config names its class (schemas.py);
#   * a config FILE names the class it fills (`_schema: <dotted.path>`), and a hierarchical class takes
#     a hierarchical file (structured.py);
#   * every leaf is required — nothing is silently defaulted, "off" is spelled `null` (structured.py);
#   * where a run WRITES is not part of its config: `run_dir` and `log` are the launcher's, from the
#     command line or the script (runs.py).
#
# See config.py for the YAML loader, the `defaults:` composition that works at any depth, and the
# ${now:...} / ${from_yaml:...} resolvers; and paths.py for the project-root rule relative paths
# resolve against.

from .config import Claim, Composed, compose, load_mapping_yaml, load_yaml
from .partials import is_partial, partial_of, stated
from .paths import project_root, resolve_path
from .runs import (
    run,
    start_run,
    tee_stdout,
)
from .schemas import (
    check_schema,
    field_schema,
    fields_of,
    resolve_schema,
    schema_name,
)
from .structured import (
    Spec,
    load_config,
    merge_specs,
    peek,
    schema_of,
)

__version__ = "0.8.2"

__all__ = [
    "Claim",
    "Composed",
    "Spec",
    "check_schema",
    "compose",
    "field_schema",
    "fields_of",
    "is_partial",
    "load_config",
    "load_mapping_yaml",
    "load_yaml",
    "merge_specs",
    "partial_of",
    "peek",
    "project_root",
    "resolve_path",
    "resolve_schema",
    "run",
    "schema_name",
    "schema_of",
    "start_run",
    "stated",
    "tee_stdout",
]
