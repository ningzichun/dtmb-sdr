"""Locate an installed CUDA runtime for native child processes.

Discovery order:

1. ``DTMB_CUDA_RUNTIME_DIR`` already set — respected as-is.
2. A pip-installed ``nvidia-cuda-runtime-cuNN`` distribution (the
   ``dtmb-sdr[cuda]`` extra installs the cu12 variant).
3. A conda-installed ``cuda-cudart`` under ``$CONDA_PREFIX``.

Discovery selects the runtime major required by the chosen native backend.
"""

from importlib import metadata
import os
from pathlib import Path

# (pip distribution name, Windows runtime filename, POSIX runtime filename)
_CUDA_RUNTIME_CANDIDATES = (
    ("nvidia-cuda-runtime-cu13", "cudart64_13.dll", "libcudart.so.13"),
    ("nvidia-cuda-runtime-cu12", "cudart64_12.dll", "libcudart.so.12"),
)


def _runtime_from_distribution(distribution_name: str, filename: str) -> Path | None:
    try:
        runtime = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None
    for record in runtime.files or ():
        if Path(record).name == filename:
            library = Path(runtime.locate_file(record)).resolve()
            if library.is_file():
                return library.parent
    return None


def _runtime_from_conda_prefix(filename: str) -> Path | None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    if os.name == "nt":
        candidates = (Path(prefix) / "Library" / "bin" / filename,)
    else:
        candidates = (
            Path(prefix) / "lib" / filename,
            Path(prefix) / "targets" / "x86_64-linux" / "lib" / filename,
            Path(prefix) / "targets" / "aarch64-linux" / "lib" / filename,
            Path(prefix) / "targets" / "sbsa-linux" / "lib" / filename,
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.parent
    return None


def _backend_major(executable: str | Path | None) -> int | None:
    configured = os.environ.get("DTMB_CUDA_BACKEND")
    directory = Path(executable).resolve().parent if executable else Path(__file__).resolve().parent / "bin"
    for major in (13, 12):
        names = (f"dtmb_cuda{major}.dll", f"libdtmb_cuda{major}.so", f"libdtmb_cuda{major}.dylib")
        if configured:
            if Path(configured).name in names:
                return major
        elif any((directory / name).is_file() for name in names):
            return major
    return None


def native_cuda_environment(executable: str | Path | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    if environment.get("DTMB_CUDA_RUNTIME_DIR"):
        return environment
    major = _backend_major(executable)
    # With no identifiable backend, retain the wheel's CUDA 12 default.
    distribution = f"nvidia-cuda-runtime-cu{major or 12}"
    for distribution_name, windows_name, linux_name in _CUDA_RUNTIME_CANDIDATES:
        if distribution_name != distribution:
            continue
        filename = windows_name if os.name == "nt" else linux_name
        directory = _runtime_from_distribution(distribution_name, filename)
        if directory is None:
            directory = _runtime_from_conda_prefix(filename)
        if directory is not None:
            environment["DTMB_CUDA_RUNTIME_DIR"] = str(directory)
            return environment
    return environment
