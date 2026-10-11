"""Runtime discovery must match the selected native backend's CUDA major."""
import os
from pathlib import Path

import pytest
from dtmb import cuda


def backend_name(major):
    return f"dtmb_cuda{major}.dll" if os.name == "nt" else f"libdtmb_cuda{major}.so"


def runtimes(monkeypatch, tmp_path):
    libraries = {}
    for major in (12, 13):
        name = f"cudart64_{major}.dll" if os.name == "nt" else f"libcudart.so.{major}"
        library = tmp_path / f"runtime{major}" / name
        library.parent.mkdir()
        library.touch()
        libraries[major] = library

    class Distribution:
        def __init__(self, major):
            self.library = libraries[major]
            self.files = [Path(self.library.name)]

        def locate_file(self, record):
            return self.library

    monkeypatch.delenv("DTMB_CUDA_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("DTMB_CUDA_BACKEND", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(cuda.metadata, "distribution", lambda name: Distribution(int(name[-2:])))
    return libraries


@pytest.mark.parametrize("major", [12, 13])
def test_two_installed_runtimes_match_backend(monkeypatch, tmp_path, major):
    libraries = runtimes(monkeypatch, tmp_path)
    executable = tmp_path / "native" / "dtmb_core_ldpc_bch_decode"
    executable.parent.mkdir()
    (executable.parent / backend_name(major)).touch()
    environment = cuda.native_cuda_environment(executable)
    assert environment["DTMB_CUDA_RUNTIME_DIR"] == str(libraries[major].parent)
    assert "DTMB_CUDA_RUNTIME_DIR" not in os.environ


def test_newest_available_backend_selects_matching_runtime(monkeypatch, tmp_path):
    libraries = runtimes(monkeypatch, tmp_path)
    for major in (12, 13):
        (tmp_path / backend_name(major)).touch()
    assert cuda.native_cuda_environment(tmp_path / "decoder")["DTMB_CUDA_RUNTIME_DIR"] == str(libraries[13].parent)


def test_explicit_backend_precedes_neighboring_plugin(monkeypatch, tmp_path):
    libraries = runtimes(monkeypatch, tmp_path)
    (tmp_path / backend_name(13)).touch()
    monkeypatch.setenv("DTMB_CUDA_BACKEND", str(tmp_path / backend_name(12)))
    assert cuda.native_cuda_environment(tmp_path / "decoder")["DTMB_CUDA_RUNTIME_DIR"] == str(libraries[12].parent)


def test_explicit_runtime_directory_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("DTMB_CUDA_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(cuda.metadata, "distribution", lambda name: pytest.fail("override ignored"))
    assert cuda.native_cuda_environment()["DTMB_CUDA_RUNTIME_DIR"] == str(tmp_path)


def test_missing_matching_runtime_does_not_use_other_major(monkeypatch, tmp_path):
    runtimes(monkeypatch, tmp_path)
    (tmp_path / backend_name(13)).touch()
    def missing(name):
        raise cuda.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(cuda.metadata, "distribution", missing)
    assert "DTMB_CUDA_RUNTIME_DIR" not in cuda.native_cuda_environment(tmp_path / "decoder")
