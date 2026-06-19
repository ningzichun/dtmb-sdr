# Root convenience front door for the staged DTMB pipeline.
#
# Example:
#   make dtmb pipeline 522mhz 16 14 0 5s
#
# This stays intentionally thin: all stage ownership remains in pipeline.mk.

PYTHON ?= python
CMAKE ?= cmake
DEFAULT_CXX := $(if $(wildcard /opt/homebrew/opt/llvm/bin/clang++),/opt/homebrew/opt/llvm/bin/clang++,c++)
CORE_CXX ?= $(DEFAULT_CXX)
CORE_CPP_BUILD_DIR ?= build/core-cpp
CORE_CPP_BUILD_TYPE ?= Release

.DEFAULT_GOAL := help

DTMB_ARG_GOALS := $(filter-out dtmb pipeline help,$(MAKECMDGOALS))
DTMB_FORWARD_VARIABLES := \
	BANDWIDTH \
	CAPTURE_TUNE_OFFSET_HZ \
	CAPTURE_NAME \
	CAPTURE_GROUP \
	CAPTURE_TAG \
	CAPTURES_DIR \
	CAPTURE_ANALYSIS_DIR \
	CROP_LOWPASS_CUTOFF_HZ \
	CROP_LOWPASS_TRANSITION_HZ \
	CROP_OUTPUT_BANDWIDTH_HZ \
	CROP_OUTPUT_SAMPLE_RATE \
	CROP_WIDEBAND_CAPTURE \
	DD_MAX_HARD_BIT_BIAS \
	DTMB_TARGET \
	FFT_BIN_SHIFT \
	FREQUENCY_DEINTERLEAVER_DIRECTION \
	DATA_CARRIER_ORDER \
	CARRIER_PERMUTATION \
	LOGICAL_POSITION_SHIFT \
	FFMPEG \
	FFPROBE \
	FORCE_DEMAP \
	FORCE_RECEIVE \
	FREQUENCY_SHIFT \
	MAKE_JOBS \
	BODY_WINDOW_FALLBACK \
	BODY_WINDOW_OFFSET \
	PHASE_OFFSET \
	PROFILE \
	RAW_CAPTURES_DIR \
	RECEIVE_FRAMES \
	SYSINFO_FRAMES \
	SYSINFO_INDEX \
	SYSINFO_ORACLE_EQUALIZER \
	SYSINFO_ORACLE_FRAMES \
	TARGET \
	TIMING_BODY_PHASE_BIAS \
	TIMING_BODY_PHASE_BIAS_SEARCH \
	TIMING_BODY_PHASE_BIAS_MIN \
	TIMING_BODY_PHASE_BIAS_MAX \
	TIMING_BODY_PHASE_BIAS_STEP \
	TIMING_MAX_SAMPLES \
	TIMING_POLICY \
	TIMING_TRAJECTORY \
	TIMING_TRAJECTORY_SOURCE
DTMB_FORWARD_ARGS = $(foreach v,$(DTMB_FORWARD_VARIABLES),$(if $(filter undefined,$(origin $(v))),,--set $(v)=$($(v))))

.PHONY: help dtmb pipeline core-test cpp-test cpp-configure

help:
	@echo "DTMB shortcuts:"
	@echo "  make dtmb pipeline 522mhz 16 14 0 5s"
	@echo "      Runs the full HackRF -> Gate A/C -> receive -> TS probe -> visuals path."
	@echo ""
	@echo "Argument order after 'pipeline':"
	@echo "  frequency lna_gain vga_gain amp duration"
	@echo ""
	@echo "Examples:"
	@echo "  make dtmb pipeline 522mhz 16 14 0 5s"
	@echo "  make dtmb pipeline 522.1mhz 16 14 0 6s CAPTURE_TAG=night_test"
	@echo "  make dtmb pipeline 522mhz 16 14 0 5s CAPTURE_NAME=wall_test"
	@echo "  make dtmb pipeline 522mhz 16 14 0 5s CAPTURE_GROUP=manual/wall_test"
	@echo "  make dtmb pipeline 522mhz 16 14 0 5s TARGET=wall-20msps BANDWIDTH=10mhz"
	@echo "  make dtmb pipeline 522mhz 16 16 0 2s TARGET=wall-20msps CROP_WIDEBAND_CAPTURE=1 CAPTURE_TUNE_OFFSET_HZ=5000000"
	@echo "  make dtmb pipeline 522mhz 16 16 0 2s MAKE_JOBS=12"
	@echo "  make core-test"
	@echo "  make -f pipeline.mk sysinfo CAPTURE=captures/raw/foo.ci8 PHASE_OFFSET=4123 BODY_WINDOW_OFFSET=164 FFT_BIN_SHIFT=0"
	@echo "  # raw CI8 goes under captures/raw; derived diagnostics go under captures/analysis."
	@echo "  # wall-7msps defaults to 7.56 Msps with an 8 MHz HackRF baseband filter."
	@echo "  make -f pipeline.mk help"

dtmb:
ifeq ($(filter pipeline,$(MAKECMDGOALS)),)
	@$(MAKE) help
else
	@$(PYTHON) scripts/pipeline/make_dtmb_pipeline.py --make-bin "$(MAKE)" $(DTMB_ARG_GOALS) $(DTMB_FORWARD_ARGS)
endif

pipeline:
ifeq ($(filter dtmb,$(MAKECMDGOALS)),)
	@$(PYTHON) scripts/pipeline/make_dtmb_pipeline.py --make-bin "$(MAKE)" $(DTMB_ARG_GOALS) $(DTMB_FORWARD_ARGS)
else
	@:
endif

core-test: cpp-test

cpp-configure:
	$(CMAKE) -S core/cpp -B $(CORE_CPP_BUILD_DIR) -DCMAKE_CXX_COMPILER="$(CORE_CXX)" -DCMAKE_BUILD_TYPE=$(CORE_CPP_BUILD_TYPE) -DDTMB_CORE_BUILD_TESTS=ON

cpp-test: cpp-configure
	$(CMAKE) --build $(CORE_CPP_BUILD_DIR)
	ctest --test-dir $(CORE_CPP_BUILD_DIR) --output-on-failure

ifneq ($(filter dtmb pipeline,$(MAKECMDGOALS)),)
$(DTMB_ARG_GOALS):
	@:
endif
