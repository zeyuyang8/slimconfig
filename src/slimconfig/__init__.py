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
#   * a config class is a @dataclass subclassing `Config`, holding LEAVES, NESTED CONFIG CLASSES and
#     TABLES of one (`dict[K, C]`) — groups are composed as fields, not inherited as mixins, so a
#     value's name says where it came from, and every type hint is one a YAML value can actually have,
#     checked at the `class` statement (schemas.py);
#   * a config FILE names the class it fills (`_schema: <dotted.path>`), and a hierarchical class takes
#     a hierarchical file; every key it sets is a field of that class (structured.py);
#   * every leaf is required — nothing is silently defaulted, "off" is spelled `null` (structured.py);
#   * where a run WRITES is not part of its config: `run_dir` and `log` are the launcher's, from the
#     command line or the script (runs.py).
#
# See config.py for the YAML loader, the `_default:` composition that works at any depth, and the
# ${now:...} / ${from_yaml:...} resolvers; and paths.py for the project-root rule relative paths
# resolve against.

from .config import compose, load_mapping_yaml, load_yaml
from .partials import is_partial, partial_of, stated
from .paths import project_root, resolve_path
from .runs import run, start_run, tee_stdout
from .schemas import Config, Schema
from .structured import Spec, load_config, merge_specs, peek, schema_of

__version__ = "0.11.0"

# What a config-driven script uses. Everything a schema is ASKED is a method of `Schema`, so one name
# comes out of the schema layer instead of a handful of free functions doing one call each; the loader's
# own record types (Claim, Composed, Key) stay in slimconfig.config, where a caller who wants them will
# already be reading compose().
__all__ = [
    "Config",
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
