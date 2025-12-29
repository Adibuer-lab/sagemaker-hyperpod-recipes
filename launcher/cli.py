from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location

from .paths import get_project_root


def main() -> None:
    project_root = get_project_root()
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    main_path = project_root / "main.py"
    spec = spec_from_file_location("aws_hp_launcher_main", main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load launcher entrypoint from {main_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
