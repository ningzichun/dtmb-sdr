PYTHON ?= python
CMAKE ?= cmake
BUILD_DIR ?= build/core-cpp

.PHONY: configure build test install

configure:
	$(CMAKE) -S core/cpp -B $(BUILD_DIR) -DDTMB_CORE_BUILD_TESTS=ON

build: configure
	$(CMAKE) --build $(BUILD_DIR) --config Release

test: build
	ctest --test-dir $(BUILD_DIR) -C Release --output-on-failure
	$(PYTHON) -m pytest tests

install:
	$(PYTHON) -m pip install -e .
