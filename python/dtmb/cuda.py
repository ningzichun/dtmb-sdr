"""Locate the optional pip-installed CUDA 12 runtime for native child processes."""

from importlib import metadata
import os
from pathlib import Path


def native_cuda_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if environment.get("DTMB_CUDA_RUNTIME_DIR"):
        return environment
    try:
        runtime = metadata.distribution("nvidia-cuda-runtime-cu12")
    except metadata.PackageNotFoundError:
        return environment
    filename = "cudart64_12.dll" if os.name == "nt" else "libcudart.so.12"
    for record in runtime.files or ():
        if Path(record).name == filename:
            library = Path(runtime.locate_file(record)).resolve()
            if library.is_file():
                environment["DTMB_CUDA_RUNTIME_DIR"] = str(library.parent)
                break
    return environment
