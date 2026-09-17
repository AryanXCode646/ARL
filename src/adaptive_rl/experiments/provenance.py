"""Provenance metadata capture for AdaptiveRL experiments.

Collects git commit, branch, working-tree dirty status, platform details,
and package versions to document the experimental execution environment.

IMPORTANT REPRODUCIBILITY NOTE:
Provenance capture records execution environment metadata (Python version,
package versions, git commit, seeds, hardware platform). This metadata facilitates
auditability, debugging, and provenance tracking, but does NOT guarantee bit-for-bit
numerical reproducibility across different hardware architectures, operating systems,
or PyTorch backend versions due to non-deterministic GPU kernel scheduling, CPU floating-point
vectorization differences, and library version variances.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_provenance(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Collect git provenance: full commit SHA, branch, dirty status, or explicit error.

    Args:
        cwd: Optional working directory for git commands.

    Returns:
        Dictionary with keys 'commit', 'branch', 'dirty', and 'error'.
    """
    prov: Dict[str, Any] = {
        "commit": None,
        "branch": None,
        "dirty": None,
        "error": None,
    }
    try:
        res_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if res_sha.returncode == 0:
            prov["commit"] = res_sha.stdout.strip()
        else:
            prov["error"] = (
                res_sha.stderr.strip() or f"git rev-parse returned code {res_sha.returncode}"
            )
            return prov

        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if res_branch.returncode == 0:
            prov["branch"] = res_branch.stdout.strip()

        res_dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if res_dirty.returncode == 0:
            prov["dirty"] = bool(res_dirty.stdout.strip())
    except FileNotFoundError:
        prov["error"] = "git binary not found"
    except subprocess.TimeoutExpired:
        prov["error"] = "git command timed out"
    except Exception as exc:
        prov["error"] = f"git inspection error: {exc}"

    return prov


def get_git_commit(cwd: Optional[Path] = None) -> str:
    """Return the short git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return "unknown"


def get_package_version(package: str) -> str:
    """Return installed version of a package, or 'not_installed'."""
    try:
        return importlib.metadata.version(package)
    except Exception:
        return "not_installed"


def get_platform_info() -> str:
    """Return operating system, OS release, and machine architecture string."""
    return f"{platform.system()} {platform.release()} {platform.machine()}"


def collect_environment_provenance(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Collect complete environment provenance record for experiment manifests."""
    git_prov = get_git_provenance(cwd=cwd)
    return {
        "git_commit": git_prov["commit"],
        "git_branch": git_prov["branch"],
        "git_dirty": git_prov["dirty"],
        "git_error": git_prov["error"],
        "python_version": sys.version,
        "platform_info": get_platform_info(),
        "package_versions": {
            "adaptive-rl": get_package_version("adaptive-rl"),
            "gymnasium": get_package_version("gymnasium"),
            "stable-baselines3": get_package_version("stable-baselines3"),
            "torch": get_package_version("torch"),
            "pydantic": get_package_version("pydantic"),
        },
    }


# Backward-compatible aliases matching manager.py internals
_get_git_provenance = get_git_provenance
_get_git_commit = get_git_commit
_get_package_version = get_package_version
_get_platform_info = get_platform_info
