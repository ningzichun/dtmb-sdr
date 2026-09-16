import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope="session")
def native_bin(tmp_path_factory):
    explicit = os.environ.get("DTMB_TEST_BIN_DIR")
    if explicit:
        binary_dir = Path(explicit).resolve()
        assert binary_dir.is_dir(), binary_dir
        return binary_dir
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("native")
    subprocess.run(["cmake", "-S", str(root / "core/cpp"), "-B", str(build),
                    "-DDTMB_CORE_BUILD_TESTS=ON", "-DDTMB_CORE_ENABLE_FFTW=OFF",
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--parallel", "4"], check=True)
    subprocess.run(["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure"], check=True)
    return build / "Release" if (build / "Release").is_dir() else build


@pytest.fixture
def native(native_bin):
    def command(name, *args):
        suffix = ".exe" if os.name == "nt" else ""
        executable = native_bin / ("dtmb_core_" + name + suffix)
        assert executable.is_file(), executable
        return [str(executable), *map(str, args)]
    return command
