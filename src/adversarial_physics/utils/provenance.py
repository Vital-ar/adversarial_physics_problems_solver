"""Capture Git, configuration and runtime provenance for experiments."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(cwd: str | Path | None = None) -> dict[str, Any]:
    """Return current Git commit and working-tree state."""
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
            )
            return completed.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "is_git_repository": commit is not None,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "torch",
                "datasets",
                "huggingface_hub",
                "transformers",
                "aligntune",
                "curatorkit",
                "auditkit",
            ]
        ),
    }
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        else:
            info["gpu"] = None
            info["cuda_version"] = None
    except Exception as exc:  # pragma: no cover - depends on runtime installation
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return info
