from importlib import import_module
from pathlib import Path

NODE_CLASS_MAPPINGS        = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_pkg_path = Path(__file__).parent / "nodes"

for file in _pkg_path.glob("*.py"):
    if file.name.startswith("_"):
        continue

    mod_name = f"{__name__}.nodes.{file.stem}"
    module   = import_module(mod_name)

    NODE_CLASS_MAPPINGS.update(
        getattr(module, "NODE_CLASS_MAPPINGS", {})
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
