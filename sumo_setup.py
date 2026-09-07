"""
Locate the SUMO installation and make its Python libraries importable.

Every script in this repo imports this module FIRST, before importing
``traci``, ``sumolib`` or ``sumo_rl``.  It exists because SUMO is a C++
program that ships its Python bindings inside its own install directory
rather than on the normal Python path, so something has to point Python at
them.  Doing that in one place means the rest of the code never has to care
where SUMO lives.

Three installation styles are supported, tried in this order:

1. A SUMO_HOME environment variable, which is how the official Windows /
   Linux installers set things up. Checked first so that an explicitly
   configured system install always wins.
2. ``pip install eclipse-sumo`` - SUMO shipped as an ordinary Python wheel.
   This is what this project uses by default: it needs no administrator
   rights and no system package manager, so a fresh clone works with just
   ``pip install -r requirements.txt``.
3. SUMO binaries already on the system PATH.

Usage:
    import sumo_setup                       # side effect: paths configured
    sumo_setup.binary("sumo")               # -> absolute path to the binary
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = ["SUMO_HOME", "binary", "describe"]


def _from_pip_wheel() -> Path | None:
    """Find SUMO if it was installed with ``pip install eclipse-sumo``."""
    try:
        import sumo  # noqa: F401  (the wheel's marker package)
    except ImportError:
        return None
    root = Path(sumo.__file__).parent
    # The wheel lays out <site-packages>/sumo/{bin,tools,data}
    if (root / "bin").is_dir():
        return root
    return None


def _from_environment() -> Path | None:
    """Find SUMO via the SUMO_HOME variable set by the official installers."""
    raw = os.environ.get("SUMO_HOME")
    if raw and Path(raw).is_dir():
        return Path(raw)
    return None


def _from_path() -> Path | None:
    """Find SUMO by looking for its binaries on the system PATH."""
    exe = shutil.which("sumo") or shutil.which("netconvert")
    if exe is None:
        return None
    # <root>/bin/sumo.exe  ->  <root>
    return Path(exe).resolve().parent.parent


def _discover() -> Path:
    for finder in (_from_environment, _from_pip_wheel, _from_path):
        found = finder()
        if found is not None:
            return found
    raise RuntimeError(
        "Could not find a SUMO installation.\n"
        "Fix it with either of:\n"
        "  pip install eclipse-sumo        (recommended, no admin rights needed)\n"
        r"  set SUMO_HOME=C:\path\to\sumo   (if you used the official installer)"
    )


SUMO_HOME: Path = _discover()

# SUMO's own tooling reads SUMO_HOME at import time, so export it even when we
# discovered the install some other way.
os.environ.setdefault("SUMO_HOME", str(SUMO_HOME))
os.environ["SUMO_HOME"] = str(SUMO_HOME)

# Put <SUMO_HOME>/tools on the import path - that is where traci and sumolib live.
_tools = SUMO_HOME / "tools"
if _tools.is_dir() and str(_tools) not in sys.path:
    sys.path.insert(0, str(_tools))

# Put <SUMO_HOME>/bin on PATH so libsumo's DLLs resolve and so subprocess calls
# to netconvert / sumo work without an absolute path.
_bin = SUMO_HOME / "bin"
if _bin.is_dir():
    os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")


def binary(name: str) -> str:
    """Absolute path to a SUMO executable, e.g. ``binary("netconvert")``."""
    suffix = ".exe" if os.name == "nt" else ""
    candidate = SUMO_HOME / "bin" / f"{name}{suffix}"
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"SUMO binary {name!r} not found under {SUMO_HOME}")


def describe() -> str:
    """One-line human-readable summary, used by the smoke test."""
    return f"SUMO_HOME={SUMO_HOME}"


if __name__ == "__main__":
    print(describe())
    for tool in ("sumo", "netconvert", "duarouter"):
        try:
            print(f"  {tool:12s} -> {binary(tool)}")
        except FileNotFoundError as exc:
            print(f"  {tool:12s} -> MISSING ({exc})")
