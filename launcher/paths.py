from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

_ENV_ROOT_VARS: Tuple[str, ...] = (
    "HYPERPOD_RECIPES_ROOT",
    "RECIPES_ROOT",
    "AWS_HP_RECIPES_ROOT",
    "RECIPES_SRC",
)


def _iter_env_roots() -> Iterable[Tuple[str, Path]]:
    for var in _ENV_ROOT_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            yield var, Path(value).expanduser()


def _normalize_start(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path.parent
    return path


def _has_launcher_dir(path: Path) -> bool:
    return (path / "launcher").is_dir()


def _has_recipes_collection(path: Path) -> bool:
    return (path / "recipes_collection").is_dir()


def _find_root_from(start: Path, require_recipes: bool) -> Optional[Path]:
    start = _normalize_start(start)
    for candidate in [start, *start.parents]:
        if not _has_launcher_dir(candidate):
            continue
        if require_recipes and not _has_recipes_collection(candidate):
            continue
        return candidate
    return None


def _resolve_root(require_recipes: bool) -> Path:
    for var, env_path in _iter_env_roots():
        root = _find_root_from(env_path, require_recipes=require_recipes)
        if root is not None:
            return root
        if require_recipes:
            raise FileNotFoundError(
                f"{var} was set to '{env_path}', but no valid HyperPod recipes root was found. "
                "Set it to the repository root that contains both 'launcher' and 'recipes_collection'."
            )
        raise FileNotFoundError(
            f"{var} was set to '{env_path}', but no valid HyperPod launcher root was found. "
            "Set it to the repository root that contains the 'launcher' directory."
        )

    package_dir = Path(__file__).resolve().parent
    root = _find_root_from(package_dir, require_recipes=require_recipes)
    if root is not None:
        return root

    if require_recipes:
        raise FileNotFoundError(
            "Unable to locate the HyperPod recipes root. "
            "Set HYPERPOD_RECIPES_ROOT or RECIPES_ROOT to the repository root "
            "that contains 'recipes_collection'."
        )

    raise FileNotFoundError(
        "Unable to locate the HyperPod launcher root. "
        "Set HYPERPOD_RECIPES_ROOT or RECIPES_ROOT to the repository root that contains 'launcher'."
    )


def get_project_root() -> Path:
    """Return the repository root that contains the launcher package."""
    return _resolve_root(require_recipes=False)


def get_recipes_root() -> Path:
    """Return the repository root that contains recipes_collection."""
    return _resolve_root(require_recipes=True)


def get_recipes_collection_dir() -> Path:
    return get_recipes_root() / "recipes_collection"


def get_recipe_yaml_path(recipe_file_path: str) -> Path:
    rel_path = recipe_file_path
    if not rel_path.endswith(".yaml"):
        rel_path = f"{rel_path}.yaml"
    return get_recipes_collection_dir() / "recipes" / rel_path


def get_recipe_templatization_dir() -> Path:
    return get_project_root() / "launcher" / "recipe_templatization"


def get_recipe_templatization_path(*parts: str) -> Path:
    return get_recipe_templatization_dir().joinpath(*parts)


def get_launcher_scripts_path() -> Path:
    return (
        get_project_root()
        / "launcher"
        / "nemo"
        / "nemo_framework_launcher"
        / "launcher_scripts"
    )


def resolve_project_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return get_project_root() / path
