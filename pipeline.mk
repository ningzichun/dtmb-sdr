# -----------------------------------------------------------------------------
# DTMB HackRF -> MPEG-TS -> MP4 pipeline (make -j friendly)
#
# Stage graph:
#     captures/raw/<stem>.ci8 -> captures/analysis/<stem>.acquire.json
#                                      (Gate A, PN945 acquisition)
#     .acquire.json -> .gate_a.visual.json + .gate_a.png
#     captures/raw/<stem>.ci8 -> captures/analysis/<stem>.sysinfo.json
#                                      (Gate C, system information)
#     .sysinfo.json -> .sysinfo_oracle.raw.json
#                                      (independent raw SI continuity gate)
#     .ci8 + .sysinfo.json -> .recovered.ts + .receiver.json
#     .recovered.ts -> .probe.json  (ffprobe verdict)
#     .recovered.ts -> .mp4         (ffmpeg transcode, needs probe ok)
#     stage JSONs -> .gate_[b-f].visual.json + .visuals.json
#                  (+ PNGs when matplotlib is available)
#
# Canonical targets (capture + full pipeline):
#     make -f pipeline.mk synthetic               # known-good loopback
#     make -f pipeline.mk wall-20msps             # 20 MSps HackRF capture
#     make -f pipeline.mk wall-7msps              # 7.56 MSps HackRF capture
#
# Diagnostic-only targets (do not produce MP4 unless --require-video is set):
#     make -f pipeline.mk acquire CAPTURE=captures/raw/path/to/file.ci8
#     make -f pipeline.mk sysinfo CAPTURE=captures/raw/path/to/file.ci8
#     make -f pipeline.mk receive CAPTURE=captures/raw/path/to/file.ci8
#
# Parallelism:
#     make -f pipeline.mk -j 3 synthetic wall-20msps wall-7msps
#     Each top-level target is independent; stages inside a target run
#     serially because they share inputs.
# -----------------------------------------------------------------------------

PYTHON            ?= python
PIPELINE_DIR      ?= scripts/pipeline
CAPTURES_DIR      ?= captures
RAW_CAPTURES_DIR  ?= $(CAPTURES_DIR)/raw
CAPTURE_ANALYSIS_DIR ?= $(CAPTURES_DIR)/analysis
CAPTURE_GROUP     ?=
RAW_RUN_DIR       := $(RAW_CAPTURES_DIR)$(if $(strip $(CAPTURE_GROUP)),/$(CAPTURE_GROUP),)
ANALYSIS_RUN_DIR  := $(CAPTURE_ANALYSIS_DIR)$(if $(strip $(CAPTURE_GROUP)),/$(CAPTURE_GROUP),)
FFMPEG            ?= ffmpeg
FFPROBE           ?= ffprobe
FFPLAY            ?= ffplay
HACKRF_TRANSFER   ?= hackrf_transfer
HACKRF_CLOCK      ?= hackrf_clock
HACKRF_SERIAL     ?=
PYTHON_SOURCES    := $(wildcard python/dtmb/*.py)
PIPELINE_SOURCES  := $(wildcard $(PIPELINE_DIR)/*.py)

# Default tunable parameters for real-RF captures. Override on the command
# line, for example: make -f pipeline.mk wall-20msps FREQUENCY=602000000
FREQUENCY         ?= 602000000
DURATION          ?= 6
AMP               ?= 0
LNA_GAIN          ?= 24
VGA_GAIN          ?= 14
BANDWIDTH         ?=
# Fixed-center DTMB captures use the 8 MHz MAX2837 filter to limit ingress
# from adjacent multiplexes, especially the contiguous 586/594/602 MHz group.
CENTERED_CAPTURE_BANDWIDTH ?= 8000000
# The optional offset/crop diagnostic needs a genuinely wide input. At
# 20 MSps, 15 MHz mirrors libhackrf's <=0.75*Fs anti-aliasing default.
WALL_20MSPS_WIDE_BANDWIDTH ?= 15000000
CAPTURE_TUNE_OFFSET_HZ ?= 0
CROP_WIDEBAND_CAPTURE ?= 0
CROP_OUTPUT_SAMPLE_RATE ?= 7560000
CROP_OUTPUT_BANDWIDTH_HZ ?= 8000000
CROP_LOWPASS_CUTOFF_HZ ?= 4200000
CROP_LOWPASS_TRANSITION_HZ ?= 800000
ANTENNA           ?= UHF antenna
LOCATION          ?=
NOTES             ?=
CAPTURE_CLOCK_SOURCE ?= internal
CAPTURE_EXTERNAL_REFERENCE_HZ ?=
CAPTURE_HARDWARE_TRIGGER ?= 0
CAPTURE_TRIGGER_SOURCE ?=
CAPTURE_TRIGGER_TIME_UTC ?=
RECEIVE_FRAMES    ?= 600
SYSINFO_FRAMES    ?= 48
SYSINFO_ORACLE_FRAMES ?= 620
SYSINFO_ORACLE_EQUALIZER ?= none
SYSINFO_INDEX      ?= 22
FREQUENCY_SHIFT    ?= 0
PHASE_OFFSET       ?=
BODY_WINDOW_OFFSET ?= 0
FFT_BIN_SHIFT      ?= 0
FREQUENCY_DEINTERLEAVER_DIRECTION ?= standard
DATA_CARRIER_ORDER ?= normal
CARRIER_PERMUTATION ?= identity
LOGICAL_POSITION_SHIFT ?= 0
BODY_WINDOW_FALLBACK ?= 0
FORCE_DEMAP        ?= 0
FORCE_RECEIVE      ?= 0
DD_MAX_HARD_BIT_BIAS ?= 0
LDPC_BACKEND       ?= python
CPP_BUILD_DIR      ?= build/core-cpp
CPP_CI8_STATS      ?= $(CPP_BUILD_DIR)/dtmb_core_ci8_stats
CPP_PIPE_BUFFER    ?= $(CPP_BUILD_DIR)/dtmb_core_pipe_buffer
CPP_LIVE_MONITOR   ?= $(CPP_BUILD_DIR)/dtmb_core_live_monitor
CPP_CI8_RESAMPLE   ?= $(CPP_BUILD_DIR)/dtmb_core_ci8_resample
CPP_C3780_EXTRACT  ?= $(CPP_BUILD_DIR)/dtmb_core_c3780_extract
CPP_HACKRF_ACQUIRE ?= $(CPP_BUILD_DIR)/dtmb_core_hackrf_acquire
CPP_DEINTERLEAVE_QAM64 ?= $(CPP_BUILD_DIR)/dtmb_core_deinterleave_qam64
CPP_LDPC_DECODER   ?= $(CPP_BUILD_DIR)/dtmb_core_ldpc_bch_decode
CPP_LDPC_H_SCORE   ?= $(CPP_BUILD_DIR)/dtmb_core_ldpc_h_score
CPP_LDPC_H_GATE    ?= $(CPP_BUILD_DIR)/dtmb_core_ldpc_h_gate
CPP_PRE_LDPC_SOURCE_SCORE ?= $(CPP_BUILD_DIR)/dtmb_core_pre_ldpc_source_score
CPP_RESAMPLE_WORKERS ?= 0
CPP_FRONTEND_WORKERS ?= 0
CPP_DEMAP_WORKERS  ?= 0
CPP_LDPC_WORKERS   ?= 0
NATIVE_DEMAP_CHUNK_SYMBOLS ?= 65536
NATIVE_CAPTURE     ?= $(CAPTURE)
NATIVE_INPUT_SAMPLE_RATE ?= 7560000
NATIVE_PREFIX      ?= $(CAPTURE_ANALYSIS_DIR)/native_receive
NATIVE_LLR_OUTPUT  ?= $(NATIVE_PREFIX).llr.f32
NATIVE_TAGGED_PRE_LDPC ?= $(NATIVE_PREFIX).tagged.pre_ldpc.npz
NATIVE_PHASE_OFFSET ?= $(PHASE_OFFSET)
NATIVE_AUTO_SYNC   ?= $(if $(strip $(NATIVE_PHASE_OFFSET)),0,1)
NATIVE_SYNC_FRAMES ?= 300
NATIVE_ACQUISITION_FRAMES ?= 16
NATIVE_AUTO_PHASE_ADJUSTMENT ?= $(if $(filter 7560000,$(NATIVE_INPUT_SAMPLE_RATE)),0,1)
NATIVE_TIMING_SEARCH_RADIUS ?= 0
NATIVE_TIMING_SEARCH_THRESHOLD ?= 0.45
NATIVE_TIMING_TRAJECTORY_INTERVAL_FRAMES ?= 0
NATIVE_TIMING_TRAJECTORY_FIT_POINTS ?= 17
NATIVE_TIMING_TRAJECTORY_MAX_INNOVATION_SAMPLES ?= 2
NATIVE_TIMING_TRAJECTORY_LOCAL_SEARCH ?= 0
NATIVE_TIMING_TRAJECTORY_LOCAL_SEARCH_MIN_IMPROVEMENT ?= 0
NATIVE_TIMING_DIAGNOSTICS ?= 0
NATIVE_FRAME_RESIDUAL_DIAGNOSTICS ?= 0
NATIVE_FRAMES      ?= 760
NATIVE_FREQUENCY_SHIFT_HZ ?= $(FREQUENCY_SHIFT)
NATIVE_SYSTEM_INFO_INDEX ?= $(SYSINFO_INDEX)
NATIVE_PN_ESTIMATOR ?= wideband
NATIVE_PN_WIDEBAND_BLOCK_FRAMES ?= 16
NATIVE_PN_WIDEBAND_DIAGNOSTICS ?= 0
NATIVE_LIVE_TIMING_DIAGNOSTICS ?= $(NATIVE_TIMING_DIAGNOSTICS)
NATIVE_LIVE_FRAME_RESIDUAL_DIAGNOSTICS ?= $(NATIVE_FRAME_RESIDUAL_DIAGNOSTICS)
NATIVE_PN_MMSE     ?= 0.05
NATIVE_NORMALIZATION ?= qam64
NATIVE_SOURCE_CARRIER_RESIDUAL_RULES ?=
NATIVE_SOURCE_CARRIER_RESIDUAL_DIAGNOSTICS_OUT ?=
NATIVE_LIVE_SOURCE_CARRIER_RESIDUAL_DIAGNOSTICS_OUT ?=
NATIVE_INTERLEAVER_MODE ?= mode2
NATIVE_INTERLEAVER_PHASE ?= 0
NATIVE_BRANCH_GAIN_BRANCHES ?=
NATIVE_BRANCH_GAIN_RELIABILITY_THRESHOLD ?= 0.55
NATIVE_BRANCH_GAIN_MIN_SYMBOLS ?= 32
NATIVE_BRANCH_GAIN_SOURCE_FRAME_RANGES ?=
NATIVE_BRANCH_GAIN_DIAGNOSTICS_OUT ?=
NATIVE_BRANCH_GAIN_DIAGNOSTICS ?= 0
NATIVE_SOURCE_FRAME_SYMBOL_GAIN_RULES ?=
NATIVE_SOURCE_CARRIER_SYMBOL_GAIN_RULES ?=
NATIVE_SOURCE_FRAME_AXIS_AFFINE_RULES ?=
NATIVE_SOURCE_FRAME_LLR_SCALE_RULES ?=
NATIVE_SOURCE_CARRIER_LLR_SCALE_RULES ?=
NATIVE_FEC_RATE    ?= 2
NATIVE_CODEWORDS_PER_FRAME ?= 3
NATIVE_LDPC_MAX_ITERATIONS ?= 50
NATIVE_LDPC_ATTENUATION ?=
NATIVE_LDPC_RETRY_LLR_CLIPS ?=
NATIVE_LDPC_RETRY_WEAK_ERASE_FRACTIONS ?=
NATIVE_LDPC_RETRY_ATTENUATIONS ?=
NATIVE_LDPC_RETRY_LAYERED ?= 0
NATIVE_LDPC_DECODE_BATCH_FRAMES ?= 16
NATIVE_LDPC_FRAME_DIAGNOSTICS ?= 0
NATIVE_LDPC_EARLY_SYNDROME_REJECT_RATIO ?= off
NATIVE_LIVE_LDPC_EARLY_SYNDROME_REJECT_RATIO ?= off
NATIVE_HARD_H_GATE ?= 0
NATIVE_HARD_H_GATE_WINDOW_CODEWORDS ?= $(NATIVE_LIVE_H_WINDOW_CODEWORDS)
NATIVE_HARD_H_GATE_STEP_CODEWORDS ?= $(NATIVE_LIVE_H_WINDOW_STEP_CODEWORDS)
NATIVE_HARD_H_GATE_THRESHOLD ?= $(NATIVE_LIVE_H_WINDOW_THRESHOLD)
NATIVE_HARD_H_GATE_FILL_MAX_GAP_FRAMES ?= 0
NATIVE_FORCE_FRAME_RANGES ?=
NATIVE_CODEWORD_LLR_SCALE_RULES ?=
NATIVE_VARIABLE_LLR_SCALE_RULES ?=
NATIVE_VARIABLE_MOD_LLR_SCALE_RULES ?=
NATIVE_REMOVE_DC   ?= 1
NATIVE_CLEAN_FRAMES_ONLY ?= 1
NATIVE_EMIT_CLEAN_CODEWORDS ?= 0
NATIVE_EMIT_BCH_CLEAN_CODEWORDS ?= 0
NATIVE_EMIT_FORCED_CODEWORDS ?= 0
NATIVE_MARK_DISCONTINUITIES ?= 1
NATIVE_INSERT_DISCONTINUITY_PACKETS ?= 1
NATIVE_LIVE_OUTPUT ?= $(CAPTURE_ANALYSIS_DIR)/native_live.ts
NATIVE_LIVE_PREFIX ?= $(CAPTURE_ANALYSIS_DIR)/native_live
NATIVE_LIVE_SOURCE ?= hackrf_transfer
NATIVE_LIVE_INPUT ?= $(NATIVE_CAPTURE)
NATIVE_LIVE_ACQUIRE_MAX_QUEUED_BYTES ?= 67108864
NATIVE_LIVE_PIPE_BUFFER_BYTES ?= 67108864
NATIVE_LIVE_PIPE_BUFFER_CHUNK_BYTES ?= 1048576
NATIVE_LIVE_PIPE_PREFILL_BYTES ?= $(NATIVE_LIVE_PIPE_BUFFER_BYTES)
NATIVE_LIVE_CAPTURE_OUTPUT ?= $(RAW_RUN_DIR)/native_live_$(FREQUENCY)_$(CAPTURE_TAG).ci8
NATIVE_LIVE_SAMPLE_RATE ?= 16000000
NATIVE_LIVE_AUTO_PHASE_ADJUSTMENT ?= $(if $(filter 7560000,$(NATIVE_LIVE_SAMPLE_RATE)),0,1)
NATIVE_LIVE_TIMING_SEARCH_RADIUS ?= 0
NATIVE_LIVE_TIMING_SEARCH_THRESHOLD ?= $(NATIVE_TIMING_SEARCH_THRESHOLD)
NATIVE_LIVE_TIMING_TRAJECTORY_INTERVAL_FRAMES ?= $(NATIVE_TIMING_TRAJECTORY_INTERVAL_FRAMES)
NATIVE_LIVE_TIMING_TRAJECTORY_FIT_POINTS ?= $(NATIVE_TIMING_TRAJECTORY_FIT_POINTS)
NATIVE_LIVE_TIMING_TRAJECTORY_MAX_INNOVATION_SAMPLES ?= $(NATIVE_TIMING_TRAJECTORY_MAX_INNOVATION_SAMPLES)
NATIVE_LIVE_TIMING_TRAJECTORY_LOCAL_SEARCH ?= $(NATIVE_TIMING_TRAJECTORY_LOCAL_SEARCH)
NATIVE_LIVE_TIMING_TRAJECTORY_LOCAL_SEARCH_MIN_IMPROVEMENT ?= $(NATIVE_TIMING_TRAJECTORY_LOCAL_SEARCH_MIN_IMPROVEMENT)
NATIVE_LIVE_SAMPLES ?= 0
NATIVE_LIVE_FRAMES ?= 0
NATIVE_LIVE_BANDWIDTH ?= 12000000
NATIVE_LIVE_MONITOR ?= 0
NATIVE_LIVE_MONITOR_PREFIX ?= $(CAPTURE_ANALYSIS_DIR)/native_live_monitor
NATIVE_LIVE_MONITOR_FFT_SIZE ?= 2048
NATIVE_LIVE_MONITOR_BINS ?= 160
NATIVE_LIVE_MONITOR_INTERVAL_SAMPLES ?= 0
NATIVE_LIVE_MONITOR_MAX_REPORTS ?= 0
NATIVE_LIVE_MONITOR_PN_FRAMES ?= 16
NATIVE_LIVE_MONITOR_PN_WORKERS ?= $(CPP_FRONTEND_WORKERS)
NATIVE_LIVE_LLR_SCAN_OUTPUT_DIR ?= $(CAPTURE_ANALYSIS_DIR)/native_live_llr_scan_$(CAPTURE_TAG)
NATIVE_LIVE_LLR_SCAN_SAMPLES ?= 25000000
NATIVE_LIVE_LLR_SCAN_FRAMES ?= 0
NATIVE_LIVE_H_WINDOW_CODEWORDS ?= 24
NATIVE_LIVE_H_WINDOW_STEP_CODEWORDS ?= 3
NATIVE_LIVE_H_WINDOW_THRESHOLD ?= 0.44
NATIVE_LIVE_STREAM_H_SCORE ?= 0
NATIVE_LIVE_H_SCORE_MODE ?= gate
NATIVE_LIVE_SOURCE_SCORE ?= 0
NATIVE_LIVE_SOURCE_SCORE_TOP ?= 12
NATIVE_LIVE_SOURCE_SCORE_FRAME_GROUP_CAP ?= 4096
NATIVE_LIVE_SOURCE_SCORE_WEAK_LLR_THRESHOLD ?= 1.0
NATIVE_LIVE_SOURCE_SCORE_FEC_FRAME_RANGES ?=
NATIVE_LIVE_SOURCE_SCORE_TRACK_LDPC_VARIABLES ?=
NATIVE_REMOVE_DC_ARG := $(if $(filter 1,$(NATIVE_REMOVE_DC)),--remove-dc,)
NATIVE_CLEAN_FRAMES_ARG := $(if $(filter 1,$(NATIVE_CLEAN_FRAMES_ONLY)),--clean-frames-only,)
NATIVE_EMIT_CLEAN_CODEWORDS_ARG := $(if $(filter 1,$(NATIVE_EMIT_CLEAN_CODEWORDS)),--emit-clean-codewords,)
NATIVE_EMIT_BCH_CLEAN_CODEWORDS_ARG := $(if $(filter 1,$(NATIVE_EMIT_BCH_CLEAN_CODEWORDS)),--emit-bch-clean-codewords,)
NATIVE_EMIT_FORCED_CODEWORDS_ARG := $(if $(filter 1,$(NATIVE_EMIT_FORCED_CODEWORDS)),--emit-forced-codewords,)
NATIVE_MARK_DISCONTINUITIES_ARG := $(if $(filter 1,$(NATIVE_MARK_DISCONTINUITIES)),--mark-discontinuities,)
NATIVE_INSERT_DISCONTINUITY_PACKETS_ARG := $(if $(filter 1,$(NATIVE_INSERT_DISCONTINUITY_PACKETS)),--insert-discontinuity-packets,)
NATIVE_LDPC_EARLY_REJECT_ARG := $(if $(filter-out off none,$(NATIVE_LDPC_EARLY_SYNDROME_REJECT_RATIO)),--early-syndrome-reject-ratio $(NATIVE_LDPC_EARLY_SYNDROME_REJECT_RATIO),)
NATIVE_LIVE_LDPC_EARLY_REJECT_ARG := $(if $(filter-out off none,$(NATIVE_LIVE_LDPC_EARLY_SYNDROME_REJECT_RATIO)),--early-syndrome-reject-ratio $(NATIVE_LIVE_LDPC_EARLY_SYNDROME_REJECT_RATIO),)
NATIVE_LDPC_ATTENUATION_ARG := $(if $(strip $(NATIVE_LDPC_ATTENUATION)),--attenuation $(NATIVE_LDPC_ATTENUATION),)
NATIVE_LDPC_RETRY_ARGS := $(foreach value,$(NATIVE_LDPC_RETRY_LLR_CLIPS),--retry-llr-clip $(value)) $(foreach value,$(NATIVE_LDPC_RETRY_WEAK_ERASE_FRACTIONS),--retry-weak-erase-fraction $(value)) $(foreach value,$(NATIVE_LDPC_RETRY_ATTENUATIONS),--retry-attenuation $(value))
NATIVE_LDPC_RETRY_ARGS += $(if $(filter 1,$(NATIVE_LDPC_RETRY_LAYERED)),--retry-layered,)
NATIVE_HARD_H_GATE_ARG := $(if $(filter 1,$(NATIVE_HARD_H_GATE)),--hard-h-gate-window-codewords $(NATIVE_HARD_H_GATE_WINDOW_CODEWORDS) --hard-h-gate-step-codewords $(NATIVE_HARD_H_GATE_STEP_CODEWORDS) --hard-h-gate-threshold $(NATIVE_HARD_H_GATE_THRESHOLD),)
NATIVE_HARD_H_GATE_FILL_ARG := $(if $(filter 1,$(NATIVE_HARD_H_GATE)),$(if $(filter-out 0,$(NATIVE_HARD_H_GATE_FILL_MAX_GAP_FRAMES)),--hard-h-gate-fill-max-gap-frames $(NATIVE_HARD_H_GATE_FILL_MAX_GAP_FRAMES),),)
NATIVE_FORCE_FRAME_RANGE_ARGS := $(foreach range,$(NATIVE_FORCE_FRAME_RANGES),--force-frame-range $(range))
NATIVE_CODEWORD_LLR_SCALE_ARGS := $(foreach rule,$(NATIVE_CODEWORD_LLR_SCALE_RULES),--llr-scale-codeword $(rule))
NATIVE_VARIABLE_LLR_SCALE_ARGS := $(foreach rule,$(NATIVE_VARIABLE_LLR_SCALE_RULES),--llr-scale-variable $(rule))
NATIVE_VARIABLE_MOD_LLR_SCALE_ARGS := $(foreach rule,$(NATIVE_VARIABLE_MOD_LLR_SCALE_RULES),--llr-scale-variable-mod $(rule))
NATIVE_BRANCH_GAIN_ARG := $(if $(strip $(NATIVE_BRANCH_GAIN_BRANCHES)),--branch-gain-branches $(NATIVE_BRANCH_GAIN_BRANCHES) --branch-gain-reliability-threshold $(NATIVE_BRANCH_GAIN_RELIABILITY_THRESHOLD) --branch-gain-min-symbols $(NATIVE_BRANCH_GAIN_MIN_SYMBOLS) $(foreach range,$(NATIVE_BRANCH_GAIN_SOURCE_FRAME_RANGES),--branch-gain-source-frame-range $(range)) $(if $(strip $(NATIVE_BRANCH_GAIN_DIAGNOSTICS_OUT)),--branch-gain-diagnostics-out $(NATIVE_BRANCH_GAIN_DIAGNOSTICS_OUT),),) $(foreach rule,$(NATIVE_SOURCE_FRAME_SYMBOL_GAIN_RULES),--source-frame-symbol-gain $(rule)) $(foreach rule,$(NATIVE_SOURCE_CARRIER_SYMBOL_GAIN_RULES),--source-carrier-symbol-gain $(rule)) $(foreach rule,$(NATIVE_SOURCE_FRAME_AXIS_AFFINE_RULES),--source-frame-axis-affine $(rule)) $(foreach rule,$(NATIVE_SOURCE_FRAME_LLR_SCALE_RULES),--source-frame-llr-scale $(rule)) $(foreach rule,$(NATIVE_SOURCE_CARRIER_LLR_SCALE_RULES),--source-carrier-llr-scale $(rule))
NATIVE_LDPC_FRAME_DIAGNOSTICS_ARG := $(if $(filter 1,$(NATIVE_LDPC_FRAME_DIAGNOSTICS)),--frame-diagnostics-out $(NATIVE_PREFIX).fec_frames.ndjson,)
NATIVE_LIVE_LDPC_FRAME_DIAGNOSTICS_ARG := $(if $(filter 1,$(NATIVE_LDPC_FRAME_DIAGNOSTICS)),--frame-diagnostics-out $(NATIVE_LIVE_PREFIX).fec_frames.ndjson,)
NATIVE_SYNC_ARG := $(if $(filter 1,$(NATIVE_AUTO_SYNC)),--auto-sync --sync-frames $(NATIVE_SYNC_FRAMES) --acquisition-frames $(NATIVE_ACQUISITION_FRAMES) --auto-phase-adjustment $(NATIVE_AUTO_PHASE_ADJUSTMENT),--phase-offset $(NATIVE_PHASE_OFFSET))
NATIVE_TIMING_TRAJECTORY_ARG := $(if $(filter-out 0,$(NATIVE_TIMING_TRAJECTORY_INTERVAL_FRAMES)),--timing-trajectory-interval-frames $(NATIVE_TIMING_TRAJECTORY_INTERVAL_FRAMES) --timing-trajectory-fit-points $(NATIVE_TIMING_TRAJECTORY_FIT_POINTS) --timing-trajectory-max-innovation-samples $(NATIVE_TIMING_TRAJECTORY_MAX_INNOVATION_SAMPLES) $(if $(filter 1,$(NATIVE_TIMING_TRAJECTORY_LOCAL_SEARCH)),--timing-trajectory-local-search --timing-trajectory-local-search-min-improvement $(NATIVE_TIMING_TRAJECTORY_LOCAL_SEARCH_MIN_IMPROVEMENT),),)
NATIVE_LIVE_TIMING_TRAJECTORY_ARG := $(if $(filter-out 0,$(NATIVE_LIVE_TIMING_TRAJECTORY_INTERVAL_FRAMES)),--timing-trajectory-interval-frames $(NATIVE_LIVE_TIMING_TRAJECTORY_INTERVAL_FRAMES) --timing-trajectory-fit-points $(NATIVE_LIVE_TIMING_TRAJECTORY_FIT_POINTS) --timing-trajectory-max-innovation-samples $(NATIVE_LIVE_TIMING_TRAJECTORY_MAX_INNOVATION_SAMPLES) $(if $(filter 1,$(NATIVE_LIVE_TIMING_TRAJECTORY_LOCAL_SEARCH)),--timing-trajectory-local-search --timing-trajectory-local-search-min-improvement $(NATIVE_LIVE_TIMING_TRAJECTORY_LOCAL_SEARCH_MIN_IMPROVEMENT),),)
NATIVE_TIMING_TRACKING_ARG := $(if $(filter-out 0,$(NATIVE_TIMING_SEARCH_RADIUS)),--timing-search-radius $(NATIVE_TIMING_SEARCH_RADIUS) --timing-search-threshold $(NATIVE_TIMING_SEARCH_THRESHOLD),) $(NATIVE_TIMING_TRAJECTORY_ARG)
NATIVE_LIVE_TIMING_TRACKING_ARG := $(if $(filter-out 0,$(NATIVE_LIVE_TIMING_SEARCH_RADIUS)),--timing-search-radius $(NATIVE_LIVE_TIMING_SEARCH_RADIUS) --timing-search-threshold $(NATIVE_LIVE_TIMING_SEARCH_THRESHOLD),) $(NATIVE_LIVE_TIMING_TRAJECTORY_ARG)
NATIVE_TIMING_DIAGNOSTICS_ARG := $(if $(filter 1,$(NATIVE_TIMING_DIAGNOSTICS)),--timing-diagnostics $(NATIVE_PREFIX).timing.csv,)
NATIVE_LIVE_TIMING_DIAGNOSTICS_ARG := $(if $(filter 1,$(NATIVE_LIVE_TIMING_DIAGNOSTICS)),--timing-diagnostics $(NATIVE_LIVE_PREFIX).timing.csv,)
NATIVE_FRAME_RESIDUAL_DIAGNOSTICS_ARG := $(if $(filter 1,$(NATIVE_FRAME_RESIDUAL_DIAGNOSTICS)),--frame-residual-diagnostics $(NATIVE_PREFIX).frame_residuals.csv,)
NATIVE_LIVE_FRAME_RESIDUAL_DIAGNOSTICS_ARG := $(if $(filter 1,$(NATIVE_LIVE_FRAME_RESIDUAL_DIAGNOSTICS)),--frame-residual-diagnostics $(NATIVE_LIVE_PREFIX).frame_residuals.csv,)
NATIVE_WIDEBAND_DIAGNOSTICS_ARG := $(if $(filter 1,$(NATIVE_PN_WIDEBAND_DIAGNOSTICS)),--pn-wideband-diagnostics $(NATIVE_PREFIX).wideband.csv,)
NATIVE_LIVE_WIDEBAND_DIAGNOSTICS_ARG := $(if $(filter 1,$(NATIVE_PN_WIDEBAND_DIAGNOSTICS)),--pn-wideband-diagnostics $(NATIVE_LIVE_PREFIX).wideband.csv,)
NATIVE_SOURCE_CARRIER_RESIDUAL_ARG := $(if $(strip $(NATIVE_SOURCE_CARRIER_RESIDUAL_RULES)),--source-carrier-residual-diagnostics-out $(if $(strip $(NATIVE_SOURCE_CARRIER_RESIDUAL_DIAGNOSTICS_OUT)),$(NATIVE_SOURCE_CARRIER_RESIDUAL_DIAGNOSTICS_OUT),$(NATIVE_PREFIX).carrier_residuals.json) $(foreach rule,$(NATIVE_SOURCE_CARRIER_RESIDUAL_RULES),--source-carrier-residual $(rule)),)
NATIVE_LIVE_SOURCE_CARRIER_RESIDUAL_ARG := $(if $(strip $(NATIVE_SOURCE_CARRIER_RESIDUAL_RULES)),--source-carrier-residual-diagnostics-out $(if $(strip $(NATIVE_LIVE_SOURCE_CARRIER_RESIDUAL_DIAGNOSTICS_OUT)),$(NATIVE_LIVE_SOURCE_CARRIER_RESIDUAL_DIAGNOSTICS_OUT),$(NATIVE_LIVE_PREFIX).carrier_residuals.json) $(foreach rule,$(NATIVE_SOURCE_CARRIER_RESIDUAL_RULES),--source-carrier-residual $(rule)),)
NATIVE_LIVE_SAMPLES_ARG := $(if $(filter-out 0,$(NATIVE_LIVE_SAMPLES)),-n $(NATIVE_LIVE_SAMPLES),)
NATIVE_LIVE_TRIGGER_ARG := $(if $(filter 1,$(CAPTURE_HARDWARE_TRIGGER)),-H,)
HACKRF_DEVICE_ARG := $(if $(strip $(HACKRF_SERIAL)),-d $(HACKRF_SERIAL),)
HACKRF_CLOCK_DEVICE_ARG := $(if $(strip $(HACKRF_SERIAL)),--device-serial $(HACKRF_SERIAL),)
NATIVE_LIVE_ACQUIRE_SERIAL_ARG := $(if $(strip $(HACKRF_SERIAL)),--serial $(HACKRF_SERIAL),)
NATIVE_LIVE_ACQUIRE_SAMPLES_ARG := $(if $(filter-out 0,$(NATIVE_LIVE_SAMPLES)),--samples $(NATIVE_LIVE_SAMPLES),)
NATIVE_LIVE_CAPTURE_LIMIT_ARG := $(if $(filter-out 0,$(NATIVE_LIVE_SAMPLES)),--samples $(NATIVE_LIVE_SAMPLES),--duration $(DURATION))
NATIVE_LIVE_MONITOR_SERIAL_ARG := $(if $(strip $(HACKRF_SERIAL)),--serial $(HACKRF_SERIAL),)
NATIVE_LIVE_MONITOR_INTERVAL_ARG := $(if $(filter-out 0,$(NATIVE_LIVE_MONITOR_INTERVAL_SAMPLES)),--report-every-samples $(NATIVE_LIVE_MONITOR_INTERVAL_SAMPLES),)
NATIVE_LIVE_MONITOR_MAX_ARG := $(if $(filter-out 0,$(NATIVE_LIVE_MONITOR_MAX_REPORTS)),--max-reports $(NATIVE_LIVE_MONITOR_MAX_REPORTS),)
NATIVE_LIVE_MONITOR_SAMPLE_LIMIT_ARG := $(if $(filter-out 0,$(NATIVE_LIVE_SAMPLES)),--max-samples $(NATIVE_LIVE_SAMPLES),)
NATIVE_LIVE_MONITOR_COMMON_ARGS := --sample-rate $(NATIVE_LIVE_SAMPLE_RATE) --center-frequency $(FREQUENCY) --bandwidth $(NATIVE_LIVE_BANDWIDTH) --amp $(AMP) --lna-gain $(LNA_GAIN) --vga-gain $(VGA_GAIN) --fft-size $(NATIVE_LIVE_MONITOR_FFT_SIZE) --spectrum-bins $(NATIVE_LIVE_MONITOR_BINS) --pn-frames $(NATIVE_LIVE_MONITOR_PN_FRAMES) --pn-workers $(NATIVE_LIVE_MONITOR_PN_WORKERS) $(NATIVE_LIVE_MONITOR_SERIAL_ARG) $(NATIVE_LIVE_MONITOR_INTERVAL_ARG) $(NATIVE_LIVE_MONITOR_MAX_ARG)
NATIVE_LIVE_MONITOR_PIPE := $(if $(filter 1,$(NATIVE_LIVE_MONITOR)),| $(CPP_LIVE_MONITOR) $(NATIVE_LIVE_MONITOR_COMMON_ARGS) --passthrough --json-out $(NATIVE_LIVE_PREFIX).monitor.ndjson - 2>$(NATIVE_LIVE_PREFIX).monitor.log,)
NATIVE_RECEIVE_SOURCE := $(if $(filter 7560000,$(NATIVE_INPUT_SAMPLE_RATE)),$(CPP_C3780_EXTRACT),{ $(CPP_CI8_RESAMPLE) --input-rate $(NATIVE_INPUT_SAMPLE_RATE) --workers $(CPP_RESAMPLE_WORKERS) $(NATIVE_CAPTURE) - 2>$(NATIVE_PREFIX).resample.log || test $$? -eq 141; } | $(CPP_C3780_EXTRACT))
NATIVE_RECEIVE_INPUT := $(if $(filter 7560000,$(NATIVE_INPUT_SAMPLE_RATE)),$(NATIVE_CAPTURE),-)
NATIVE_LIVE_RESAMPLE_PIPE := $(if $(filter 7560000,$(NATIVE_LIVE_SAMPLE_RATE)),,| $(CPP_CI8_RESAMPLE) --input-rate $(NATIVE_LIVE_SAMPLE_RATE) --workers $(CPP_RESAMPLE_WORKERS) - - 2>$(NATIVE_LIVE_PREFIX).resample.log)
NATIVE_LIVE_HACKRF_TRANSFER_SOURCE := $(HACKRF_TRANSFER) -r - $(HACKRF_DEVICE_ARG) -f $(FREQUENCY) -s $(NATIVE_LIVE_SAMPLE_RATE) -b $(NATIVE_LIVE_BANDWIDTH) -a $(AMP) -l $(LNA_GAIN) -g $(VGA_GAIN) $(NATIVE_LIVE_TRIGGER_ARG) $(NATIVE_LIVE_SAMPLES_ARG) 2>$(NATIVE_LIVE_PREFIX).transfer.log
NATIVE_LIVE_NATIVE_ACQUIRE_COMMON := $(CPP_HACKRF_ACQUIRE) --frequency $(FREQUENCY) --sample-rate $(NATIVE_LIVE_SAMPLE_RATE) --bandwidth $(NATIVE_LIVE_BANDWIDTH) --amp $(AMP) --lna-gain $(LNA_GAIN) --vga-gain $(VGA_GAIN) $(NATIVE_LIVE_ACQUIRE_SERIAL_ARG) --max-queued-bytes $(NATIVE_LIVE_ACQUIRE_MAX_QUEUED_BYTES)
NATIVE_LIVE_NATIVE_ACQUIRE_SOURCE := $(NATIVE_LIVE_NATIVE_ACQUIRE_COMMON) $(NATIVE_LIVE_ACQUIRE_SAMPLES_ARG) --output - 2>$(NATIVE_LIVE_PREFIX).acquire.log
NATIVE_LIVE_PIPE_BUFFER_ARGS := --buffer-bytes $(NATIVE_LIVE_PIPE_BUFFER_BYTES) --chunk-bytes $(NATIVE_LIVE_PIPE_BUFFER_CHUNK_BYTES) --prefill-bytes $(NATIVE_LIVE_PIPE_PREFILL_BYTES) --diagnostics-out $(NATIVE_LIVE_PREFIX).source_buffer.json
NATIVE_LIVE_FILE_SOURCE := $(CPP_PIPE_BUFFER) $(NATIVE_LIVE_PIPE_BUFFER_ARGS) $(NATIVE_LIVE_INPUT) -
NATIVE_LIVE_DEVICE_SOURCE_BUFFER_PIPE := | $(CPP_PIPE_BUFFER) $(NATIVE_LIVE_PIPE_BUFFER_ARGS) - -
NATIVE_LIVE_CI8_SOURCE := $(if $(filter file,$(NATIVE_LIVE_SOURCE)),$(NATIVE_LIVE_FILE_SOURCE),$(if $(filter native_acquire,$(NATIVE_LIVE_SOURCE)),$(NATIVE_LIVE_NATIVE_ACQUIRE_SOURCE),$(NATIVE_LIVE_HACKRF_TRANSFER_SOURCE)))
NATIVE_LIVE_CI8_SOURCE_BUFFER_PIPE := $(if $(filter file,$(NATIVE_LIVE_SOURCE)),,$(NATIVE_LIVE_DEVICE_SOURCE_BUFFER_PIPE))
NATIVE_LIVE_CLOCK_ARGS := --clock-source $(CAPTURE_CLOCK_SOURCE) --hackrf-clock $(HACKRF_CLOCK) $(HACKRF_CLOCK_DEVICE_ARG) --log $(NATIVE_LIVE_PREFIX).clkin.log
ifneq ($(strip $(CAPTURE_EXTERNAL_REFERENCE_HZ)),)
NATIVE_LIVE_CLOCK_ARGS += --external-reference-hz $(CAPTURE_EXTERNAL_REFERENCE_HZ)
endif
TIMING_POLICY     ?= fixed
TIMING_TRAJECTORY ?=
TIMING_TRAJECTORY_SOURCE ?= gate-c
TIMING_BODY_PHASE_BIAS ?= 0
TIMING_BODY_PHASE_BIAS_SEARCH ?= 0
TIMING_BODY_PHASE_BIAS_MIN ?= -256
TIMING_BODY_PHASE_BIAS_MAX ?= 256
TIMING_BODY_PHASE_BIAS_STEP ?= 4
TIMING_MAX_SAMPLES ?=
PIPELINE_QUIET    ?= 1
TIMING_ARGS       := --timing-policy $(TIMING_POLICY)
CAPTURE_BANDWIDTH_ARG := --bandwidth $(if $(strip $(BANDWIDTH)),$(BANDWIDTH),$(CENTERED_CAPTURE_BANDWIDTH))
WALL_20MSPS_WIDE_BANDWIDTH_ARG := --bandwidth $(if $(strip $(BANDWIDTH)),$(BANDWIDTH),$(WALL_20MSPS_WIDE_BANDWIDTH))
CAPTURE_CLOCK_ARGS := --clock-source $(CAPTURE_CLOCK_SOURCE)
CAPTURE_CLOCK_ARGS += $(HACKRF_CLOCK_DEVICE_ARG)
ifneq ($(strip $(CAPTURE_EXTERNAL_REFERENCE_HZ)),)
CAPTURE_CLOCK_ARGS += --external-reference-hz $(CAPTURE_EXTERNAL_REFERENCE_HZ)
endif
ifeq ($(CAPTURE_HARDWARE_TRIGGER),1)
CAPTURE_CLOCK_ARGS += --hardware-trigger
endif
ifneq ($(strip $(CAPTURE_TRIGGER_SOURCE)),)
CAPTURE_CLOCK_ARGS += --trigger-source "$(CAPTURE_TRIGGER_SOURCE)"
endif
ifneq ($(strip $(CAPTURE_TRIGGER_TIME_UTC)),)
CAPTURE_CLOCK_ARGS += --trigger-time-utc "$(CAPTURE_TRIGGER_TIME_UTC)"
endif
# Keep the analog/baseband filter at or above DTMB's 7.56 MHz occupied span.
# 7 MHz clips channel edges before ADC; 8 MHz is the nearest HackRF filter.
WALL_7MSPS_BANDWIDTH ?= 8000000
WALL_7MSPS_BANDWIDTH_ARG := --bandwidth $(if $(strip $(BANDWIDTH)),$(BANDWIDTH),$(WALL_7MSPS_BANDWIDTH))
ifneq ($(TIMING_TRAJECTORY),)
ifneq ($(TIMING_TRAJECTORY),auto)
TIMING_ARGS       += --timing-trajectory $(TIMING_TRAJECTORY)
endif
endif
TIMING_TRAJECTORY_PREREQ :=
ifeq ($(TIMING_TRAJECTORY),auto)
TIMING_TRAJECTORY_PREREQ := $(CAPTURE_ANALYSIS_DIR)/%.timing_trajectory.json
endif
TIMING_TRAJECTORY_ARG_FOR = $(if $(filter auto,$(TIMING_TRAJECTORY)),--timing-trajectory $(1).timing_trajectory.json,$(if $(TIMING_TRAJECTORY),--timing-trajectory $(TIMING_TRAJECTORY),))
TIMING_ARGS_FOR = --timing-policy $(TIMING_POLICY) $(call TIMING_TRAJECTORY_ARG_FOR,$(1))
TIMING_TRAJECTORY_ARGS := --source $(TIMING_TRAJECTORY_SOURCE) --body-phase-bias $(TIMING_BODY_PHASE_BIAS)
ifeq ($(TIMING_BODY_PHASE_BIAS_SEARCH),1)
TIMING_TRAJECTORY_ARGS += --search-body-phase-bias --body-phase-bias-min $(TIMING_BODY_PHASE_BIAS_MIN) --body-phase-bias-max $(TIMING_BODY_PHASE_BIAS_MAX) --body-phase-bias-step $(TIMING_BODY_PHASE_BIAS_STEP)
endif
ifneq ($(TIMING_MAX_SAMPLES),)
TIMING_TRAJECTORY_ARGS += --max-samples $(TIMING_MAX_SAMPLES)
endif
DEMAP_FORCE_ARGS   :=
ifeq ($(FORCE_DEMAP),1)
DEMAP_FORCE_ARGS   += --force-demap
endif
RECEIVE_FORCE_ARGS :=
ifeq ($(FORCE_RECEIVE),1)
RECEIVE_FORCE_ARGS += --force-receive
endif
PHASE_OFFSET_ARG :=
ifneq ($(strip $(PHASE_OFFSET)),)
PHASE_OFFSET_ARG += --phase-offset $(PHASE_OFFSET)
endif
BODY_WINDOW_FALLBACK_ARG :=
ifeq ($(BODY_WINDOW_FALLBACK),1)
BODY_WINDOW_FALLBACK_ARG += --body-window-fallback
endif
QUIET_ARG :=
ifeq ($(PIPELINE_QUIET),1)
QUIET_ARG += --quiet
endif

# Capture names. Timestamped so re-running does not clobber a prior run's
# sidecar; `make wall-20msps CAPTURE_TAG=20260509_evening` pins the tag.
CAPTURE_TAG       ?= $(shell $(PYTHON) -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))")
WALL_20MSPS_STEM  := wall_$(FREQUENCY)_amp$(AMP)_lna$(LNA_GAIN)_vga$(VGA_GAIN)_$(CAPTURE_TAG)_20msps
WALL_7MSPS_STEM   := wall_$(FREQUENCY)_amp$(AMP)_lna$(LNA_GAIN)_vga$(VGA_GAIN)_$(CAPTURE_TAG)_7p56msps
WALL_20MSPS_TUNE_FREQUENCY := $(shell $(PYTHON) -c "print(int($(FREQUENCY)) + int($(CAPTURE_TUNE_OFFSET_HZ)))")
WALL_20MSPS_WIDE_CI8 := $(RAW_RUN_DIR)/$(WALL_20MSPS_STEM).wide20.ci8
WALL_20MSPS_CI8   := $(RAW_RUN_DIR)/$(WALL_20MSPS_STEM).ci8
WALL_7MSPS_CI8    := $(RAW_RUN_DIR)/$(WALL_7MSPS_STEM).ci8
WALL_20MSPS_PREFIX := $(ANALYSIS_RUN_DIR)/$(WALL_20MSPS_STEM)
WALL_7MSPS_PREFIX := $(ANALYSIS_RUN_DIR)/$(WALL_7MSPS_STEM)
WALL_20MSPS_MP4   := $(WALL_20MSPS_PREFIX).mp4
WALL_7MSPS_MP4    := $(WALL_7MSPS_PREFIX).mp4
WALL_20MSPS_CROP_JSON := $(WALL_20MSPS_PREFIX).crop.json
VISUAL_MANIFEST_SUFFIXES := .gate_a.visual.json .gate_b.visual.json .gate_c.visual.json .gate_d.visual.json .gate_e.visual.json .gate_f.visual.json .echo.visual.json .visuals.json
WALL_20MSPS_VISUALS := $(foreach suffix,$(VISUAL_MANIFEST_SUFFIXES),$(WALL_20MSPS_PREFIX)$(suffix))
WALL_7MSPS_VISUALS := $(foreach suffix,$(VISUAL_MANIFEST_SUFFIXES),$(WALL_7MSPS_PREFIX)$(suffix))

# Synthetic loopback: use the in-tree demo_input.ts fixture as the
# payload, run TX -> RX -> ffmpeg. This is the only target that asserts
# a playable MP4 as a hard gate, because it must work end-to-end.
SYNTHETIC_INPUT_TS  ?= $(RAW_CAPTURES_DIR)/demo_input.ts
SYNTHETIC_CI8       := $(RAW_CAPTURES_DIR)/synthetic_pipeline.ci8
SYNTHETIC_PREFIX    := $(CAPTURE_ANALYSIS_DIR)/synthetic_pipeline
SYNTHETIC_MP4       := $(SYNTHETIC_PREFIX).mp4
SYNTHETIC_VISUALS   := $(foreach suffix,$(VISUAL_MANIFEST_SUFFIXES),$(SYNTHETIC_PREFIX)$(suffix))
NATIVE_SYNTHETIC_PREFIX := $(CAPTURE_ANALYSIS_DIR)/native_synthetic_pipeline
NATIVE_SYNTHETIC_TS := $(NATIVE_SYNTHETIC_PREFIX).recovered.ts
NATIVE_SYNTHETIC_MP4 := $(NATIVE_SYNTHETIC_PREFIX).mp4

.PHONY: synthetic native-core native-synthetic native-receive native-replay-llr native-live-capture native-live-stream native-live-playback native-live-monitor native-live-llr-scan wall-20msps wall-7msps acquire sysinfo sysinfo-oracle receive demap tag-pre-ldpc llr-health decode-ldpc timing-trajectory video-preflight probe visuals realtime sweep scan live-stream live-synthetic clean-analysis clean-pipeline restructure-captures help

help:
	@echo "DTMB pipeline targets:"
	@echo "  synthetic       Loopback demo_input.ts through TX -> RX -> ffmpeg (gated on MP4)."
	@echo "  native-synthetic  Auto-synchronized flat-channel CI8 -> all-C++ pipe -> ffmpeg gate."
	@echo "  native-receive  Auto/pinned arbitrary-rate PN945 capture -> all-C++ clean-frame TS salvage."
	@echo "  native-replay-llr  Auto/pinned capture -> native LLR + per-bit source-tagged pre-LDPC artifacts."
	@echo "  native-live-capture  Clock-checked bounded native HackRF CI8 capture to disk."
	@echo "  native-live-stream  HackRF CI8 -> optional native resample -> all-C++ auto-sync/decode -> TS sink."
	@echo "  native-live-playback  HackRF CI8 -> all-C++ auto-sync/decode -> low-latency ffplay sink."
	@echo "  native-live-monitor  HackRF CI8 -> native live spectrum + PN945 telemetry NDJSON."
	@echo "  native-live-llr-scan  HackRF center sweep -> native demap -> LLR parity-health ranking."
	@echo "  wall-20msps     Capture HackRF at 20 MSps and run full pipeline + visuals (diagnostic)."
	@echo "  wall-7msps      Capture HackRF at native 7.56 MSps and run full pipeline + visuals (diagnostic)."
	@echo "  realtime        One-shot HackRF -> decode -> ffmpeg with a per-stage verdict."
	@echo "  live-stream     Streaming HackRF -> decode -> ffmpeg (no intermediate files)."
	@echo "  live-synthetic  Streaming loopback: tx -> decode -> ffmpeg (no hardware, no files)."
	@echo "  sweep           Native hackrf_sweep spectrum CSV + DTMB-window ranking."
	@echo "  scan            PN acquisition scan over chosen centers."
	@echo ""
	@echo "Single-stage diagnostics need CAPTURE=path:"
	@echo "  make -f pipeline.mk acquire CAPTURE=captures/raw/foo.ci8"
	@echo "  make -f pipeline.mk sysinfo CAPTURE=captures/raw/foo.ci8"
	@echo "  make -f pipeline.mk sysinfo-oracle CAPTURE=captures/raw/foo.ci8   # raw SI gate"
	@echo "  make -f pipeline.mk receive CAPTURE=captures/raw/foo.ci8"
	@echo "  make -f pipeline.mk demap CAPTURE=captures/raw/foo.ci8   # DSP -> LLR"
	@echo "  make -f pipeline.mk tag-pre-ldpc CAPTURE=captures/raw/foo.ci8   # LLR -> per-bit source tags"
	@echo "  make -f pipeline.mk llr-health CAPTURE=captures/raw/foo.ci8   # score LLR health"
	@echo "  make -f pipeline.mk decode-ldpc CAPTURE=captures/raw/foo.ci8   # LLR -> TS"
	@echo "  make -f pipeline.mk timing-trajectory CAPTURE=captures/raw/foo.ci8   # windowed timing"
	@echo "  make -f pipeline.mk video-preflight CAPTURE=captures/raw/foo.ci8   # video readiness verdict"
	@echo "  make -f pipeline.mk visuals CAPTURE=captures/raw/foo.ci8   # render gate PNG/JSON visuals"
	@echo "  make -f pipeline.mk sysinfo CAPTURE=captures/raw/foo.ci8 PHASE_OFFSET=4123 BODY_WINDOW_OFFSET=164"
	@echo "  make -f pipeline.mk clean-analysis   # delete captures/analysis only"
	@echo ""
	@echo "Gate defaults: SYSINFO_FRAMES=48, SYSINFO_ORACLE_FRAMES=620, SYSINFO_ORACLE_EQUALIZER=none, SYSINFO_INDEX=22."
	@echo "Capture defaults: tune exactly to the multiplex center and use an 8 MHz HackRF baseband filter."
	@echo "Offset/crop mode remains available only for explicit diagnostics; it is not the default capture path."
	@echo "Diagnostic overrides: FORCE_DEMAP=1 and FORCE_RECEIVE=1 bypass Gate C only for explicit experiments."
	@echo "DD guard: DD_MAX_HARD_BIT_BIAS=0.06 rejects biased decision-directed refinement in receive/demap diagnostics."
	@echo "Native playback salvage: NATIVE_MARK_DISCONTINUITIES=1 and NATIVE_INSERT_DISCONTINUITY_PACKETS=1 signal omitted FEC frames in TS."
	@echo "Native FEC salvage: NATIVE_EMIT_CLEAN_CODEWORDS=1 emits clean codeword TS slices from partially dirty FEC frames."
	@echo "Native stream source: set NATIVE_LIVE_SOURCE=file|hackrf_transfer|native_acquire. File mode uses NATIVE_LIVE_INPUT."
	@echo "Native stream buffering: NATIVE_LIVE_PIPE_BUFFER_BYTES caps the shared CI8 source buffer; NATIVE_LIVE_PIPE_PREFILL_BYTES controls startup fill."
	@echo "Native LDPC batching: NATIVE_LDPC_DECODE_BATCH_FRAMES controls chronology-safe selected-frame decode batches."
	@echo "Native LDPC frame diagnostics: set NATIVE_LDPC_FRAME_DIAGNOSTICS=1 to write <prefix>.fec_frames.ndjson."
	@echo "Native live timing: unsmoothed per-frame tracking is disabled by default; use only for explicit diagnostics."
	@echo "Native PN diagnostics: NATIVE_PN_WIDEBAND_DIAGNOSTICS=1 writes <prefix>.wideband.csv channel-model rows."
	@echo "Native live monitor: NATIVE_LIVE_MONITOR=1 writes <prefix>.monitor.ndjson beside native live decode."
	@echo "Native LDPC diagnostic: set NATIVE_LIVE_LDPC_EARLY_SYNDROME_REJECT_RATIO=<ratio> only for explicit failed-frame throughput probes."
	@echo "Native rolling-H decoder gate: NATIVE_HARD_H_GATE=1 enables chronology-safe calibrated soft-decode selection."
	@echo "Timing: TIMING_TRAJECTORY=auto builds and feeds <capture>.timing_trajectory.json into Gate C/demap/receive."
	@echo ""
	@echo "realtime example:"
	@echo "  make -f pipeline.mk realtime FREQUENCY=602000000 LNA_GAIN=24 VGA_GAIN=14 AMP=0 CAPTURE_TAG=20260510_live"
	@echo "  make -f pipeline.mk wall-20msps FREQUENCY=602000000 DURATION=2 BANDWIDTH=12000000"
	@echo ""
	@echo "sweep example:"
	@echo "  make -f pipeline.mk sweep SWEEP_RANGE_MHZ=470:862 SWEEP_BIN_WIDTH=250000"
	@echo ""
	@echo "PN scan example:"
	@echo "  make -f pipeline.mk scan CENTERS=602000000"
	@echo ""
	@echo "live-stream examples:"
	@echo "  make -f pipeline.mk live-synthetic LIVE_OUTPUT=captures/analysis/live.mp4"
	@echo "  make -f pipeline.mk live-stream FREQUENCY=602000000 LIVE_OUTPUT=captures/analysis/live.mp4"
	@echo "  make -f pipeline.mk live-stream FREQUENCY=602000000 LIVE_SINK=udp LIVE_UDP_URL=udp://127.0.0.1:5555"
	@echo "  gmake -f pipeline.mk native-live-playback FREQUENCY=602000000 NATIVE_LIVE_SAMPLE_RATE=16000000 NATIVE_LIVE_BANDWIDTH=12000000"
	@echo ""
	@echo "Root shorthand:"
	@echo "  make dtmb pipeline 602mhz 24 14 0 5s"
	@echo "  make dtmb pipeline 602mhz 24 14 0 5s TARGET=wall-20msps BANDWIDTH=12mhz"

# -----------------------------------------------------------------------------
# Realtime one-shot target (capture + full pipeline + verdict)
# -----------------------------------------------------------------------------
REALTIME_RAW_PREFIX ?= $(RAW_RUN_DIR)/realtime_$(FREQUENCY)_$(CAPTURE_TAG)
REALTIME_PREFIX ?= $(ANALYSIS_RUN_DIR)/realtime_$(FREQUENCY)_$(CAPTURE_TAG)

realtime:
	$(PYTHON) $(PIPELINE_DIR)/realtime.py \
		--frequency $(FREQUENCY) \
		--sample-rate 20000000 \
		$(CAPTURE_BANDWIDTH_ARG) \
		--duration $(DURATION) \
		--amp $(AMP) \
		--lna-gain $(LNA_GAIN) \
		--vga-gain $(VGA_GAIN) \
		$(CAPTURE_CLOCK_ARGS) \
		--antenna "$(ANTENNA)" \
		--location "$(LOCATION)" \
		--receive-frames $(RECEIVE_FRAMES) \
		--sysinfo-frames $(SYSINFO_FRAMES) \
		--sysinfo-oracle-frames $(SYSINFO_ORACLE_FRAMES) \
		--sysinfo-oracle-equalizer $(SYSINFO_ORACLE_EQUALIZER) \
		--sysinfo-index $(SYSINFO_INDEX) \
		$(RECEIVE_FORCE_ARGS) \
		$(TIMING_ARGS) \
		--raw-prefix $(REALTIME_RAW_PREFIX) \
		--output-prefix $(REALTIME_PREFIX) \
		--keep-ci8 \
		--ffmpeg-bin $(FFMPEG) \
		--ffprobe-bin $(FFPROBE)

# -----------------------------------------------------------------------------
# Live streaming entry points (no intermediate files)
# -----------------------------------------------------------------------------
# Defaults shared by both live targets. Override with command-line vars.
LIVE_OUTPUT        ?= $(CAPTURE_ANALYSIS_DIR)/live.mp4
LIVE_SINK          ?= mp4
LIVE_UDP_URL       ?= udp://127.0.0.1:5555?pkt_size=1316
LIVE_DEMO_INPUT_TS ?= $(RAW_CAPTURES_DIR)/demo_input.ts
LIVE_FRAMES        ?= 600

live-synthetic:
	$(PYTHON) $(PIPELINE_DIR)/live_stream.py \
		--source synthetic \
		--input-ts $(LIVE_DEMO_INPUT_TS) \
		--synthetic-loopback \
		--frames $(LIVE_FRAMES) \
		--sink $(LIVE_SINK) \
		--output $(LIVE_OUTPUT) \
		--udp-url $(LIVE_UDP_URL) \
		--ffmpeg-bin $(FFMPEG)

live-stream:
	$(PYTHON) $(PIPELINE_DIR)/live_stream.py \
		--source hackrf \
		--frequency $(FREQUENCY) \
		--hackrf-sample-rate 20000000 \
		--sample-rate 20000000 \
		--amp $(AMP) \
		--lna-gain $(LNA_GAIN) \
		--vga-gain $(VGA_GAIN) \
		--frames $(LIVE_FRAMES) \
		--sink $(LIVE_SINK) \
		--output $(LIVE_OUTPUT) \
		--udp-url $(LIVE_UDP_URL) \
		--ffmpeg-bin $(FFMPEG)

# -----------------------------------------------------------------------------
# Native hackrf_sweep and PN scan helpers
# -----------------------------------------------------------------------------
SWEEP_RANGE_MHZ ?= 470:862
SWEEP_BIN_WIDTH ?= 1000000
SWEEP_COUNT ?= 1
SWEEP_RAW_OUTPUT_DIR ?= $(RAW_CAPTURES_DIR)/sweep_$(CAPTURE_TAG)
SWEEP_OUTPUT_DIR ?= $(CAPTURE_ANALYSIS_DIR)/sweep_$(CAPTURE_TAG)
SWEEP_CSV ?= $(SWEEP_RAW_OUTPUT_DIR)/hackrf_sweep.csv
SWEEP_REPORT ?= $(SWEEP_OUTPUT_DIR)/hackrf_sweep.report.json
SWEEP_WINDOW_BANDWIDTH ?= 8000000
SWEEP_WINDOW_STEP ?= 1000000
SWEEP_TOP ?= 12

sweep:
	$(PYTHON) $(PIPELINE_DIR)/sweep_hackrf.py \
		--range-mhz $(SWEEP_RANGE_MHZ) \
		--bin-width $(SWEEP_BIN_WIDTH) \
		--sweeps $(SWEEP_COUNT) \
		--amp $(AMP) \
		--lna-gain $(LNA_GAIN) \
		--vga-gain $(VGA_GAIN) \
		--output-csv $(SWEEP_CSV) \
		--report-json $(SWEEP_REPORT) \
		--window-bandwidth $(SWEEP_WINDOW_BANDWIDTH) \
		--window-step $(SWEEP_WINDOW_STEP) \
		--top $(SWEEP_TOP)

CENTERS ?= 602000000
NATIVE_LIVE_LLR_SCAN_CENTERS ?= 602000000
SCAN_OUTPUT_DIR ?= $(CAPTURE_ANALYSIS_DIR)/scan_$(CAPTURE_TAG)
SCAN_SAMPLE_RATE ?= 7560000

scan:
	$(PYTHON) $(PIPELINE_DIR)/scan_dtmb.py \
		--centers $(CENTERS) \
		--duration $(DURATION) \
		--sample-rate $(SCAN_SAMPLE_RATE) \
		$(CAPTURE_BANDWIDTH_ARG) \
		--amp $(AMP) \
		--lna-gain $(LNA_GAIN) \
		--vga-gain $(VGA_GAIN) \
		$(CAPTURE_CLOCK_ARGS) \
		--antenna "$(ANTENNA)" \
		--location "$(LOCATION)" \
		--output-dir $(SCAN_OUTPUT_DIR)

# -----------------------------------------------------------------------------
# Synthetic loopback
# -----------------------------------------------------------------------------
synthetic: $(SYNTHETIC_MP4) $(SYNTHETIC_VISUALS)

native-core:
	cmake -S core/cpp -B $(CPP_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DDTMB_CORE_BUILD_TESTS=ON
	cmake --build $(CPP_BUILD_DIR)

native-synthetic: native-core $(SYNTHETIC_CI8)
	mkdir -p $(dir $(NATIVE_SYNTHETIC_PREFIX))
	bash -o pipefail -c '$(CPP_C3780_EXTRACT) --auto-sync --sync-frames 160 --acquisition-frames 16 --workers $(CPP_FRONTEND_WORKERS) --system-info-index 23 $(SYNTHETIC_CI8) - | $(CPP_DEINTERLEAVE_QAM64) --mode mode1 --phase 0 --workers $(CPP_DEMAP_WORKERS) - - | $(CPP_LDPC_DECODER) --fec-rate 3 --alist python/dtmb/data/dtmb_ldpc_rate3.alist --codewords-per-frame 3 --workers $(CPP_LDPC_WORKERS) - $(NATIVE_SYNTHETIC_TS)'
	$(PYTHON) $(PIPELINE_DIR)/transcode.py \
		--input-ts $(NATIVE_SYNTHETIC_TS) \
		--probe-json $(NATIVE_SYNTHETIC_PREFIX).probe.json \
		--output-mp4 $(NATIVE_SYNTHETIC_MP4) \
		--ffmpeg-bin $(FFMPEG) \
		--ffprobe-bin $(FFPROBE) \
		--require-video

native-receive: native-core
	@test -n "$(NATIVE_CAPTURE)" || (echo "set CAPTURE=... or NATIVE_CAPTURE=..." && exit 1)
	@test "$(NATIVE_AUTO_SYNC)" = "1" -o -n "$(NATIVE_PHASE_OFFSET)" || (echo "set NATIVE_AUTO_SYNC=1 or PHASE_OFFSET=..." && exit 1)
	mkdir -p $(dir $(NATIVE_PREFIX))
	bash -o pipefail -c '$(NATIVE_RECEIVE_SOURCE) \
		$(NATIVE_SYNC_ARG) \
		$(NATIVE_TIMING_TRACKING_ARG) \
		$(NATIVE_TIMING_DIAGNOSTICS_ARG) \
		--max-frames $(NATIVE_FRAMES) \
		--frequency-shift-hz $(NATIVE_FREQUENCY_SHIFT_HZ) \
		--equalizer pn \
		--pn-estimator $(NATIVE_PN_ESTIMATOR) \
		--pn-wideband-block-frames $(NATIVE_PN_WIDEBAND_BLOCK_FRAMES) \
		$(NATIVE_WIDEBAND_DIAGNOSTICS_ARG) \
		$(NATIVE_FRAME_RESIDUAL_DIAGNOSTICS_ARG) \
		$(NATIVE_SOURCE_CARRIER_RESIDUAL_ARG) \
		--pn-mmse $(NATIVE_PN_MMSE) \
		$(NATIVE_REMOVE_DC_ARG) \
		--normalization $(NATIVE_NORMALIZATION) \
		--system-info-index $(NATIVE_SYSTEM_INFO_INDEX) \
		$(NATIVE_RECEIVE_INPUT) - 2>$(NATIVE_PREFIX).extract.log \
		| $(CPP_DEINTERLEAVE_QAM64) \
		--mode $(NATIVE_INTERLEAVER_MODE) \
		--phase $(NATIVE_INTERLEAVER_PHASE) \
		$(NATIVE_BRANCH_GAIN_ARG) \
		--workers $(CPP_DEMAP_WORKERS) --chunk-symbols $(NATIVE_DEMAP_CHUNK_SYMBOLS) - - 2>$(NATIVE_PREFIX).demap.log \
		| $(CPP_LDPC_DECODER) \
		--fec-rate $(NATIVE_FEC_RATE) \
		--alist python/dtmb/data/dtmb_ldpc_rate$(NATIVE_FEC_RATE).alist \
		--codewords-per-frame $(NATIVE_CODEWORDS_PER_FRAME) \
		--workers $(CPP_LDPC_WORKERS) \
		--max-iterations $(NATIVE_LDPC_MAX_ITERATIONS) \
		$(NATIVE_LDPC_ATTENUATION_ARG) \
		$(NATIVE_LDPC_RETRY_ARGS) \
		--decode-batch-frames $(NATIVE_LDPC_DECODE_BATCH_FRAMES) \
		$(NATIVE_LDPC_EARLY_REJECT_ARG) \
		$(NATIVE_LDPC_FRAME_DIAGNOSTICS_ARG) \
		$(NATIVE_HARD_H_GATE_ARG) \
		$(NATIVE_HARD_H_GATE_FILL_ARG) \
		$(NATIVE_FORCE_FRAME_RANGE_ARGS) \
		$(NATIVE_CODEWORD_LLR_SCALE_ARGS) \
		$(NATIVE_VARIABLE_LLR_SCALE_ARGS) \
		$(NATIVE_VARIABLE_MOD_LLR_SCALE_ARGS) \
		$(NATIVE_CLEAN_FRAMES_ARG) \
		$(NATIVE_EMIT_CLEAN_CODEWORDS_ARG) \
		$(NATIVE_EMIT_BCH_CLEAN_CODEWORDS_ARG) \
		$(NATIVE_EMIT_FORCED_CODEWORDS_ARG) \
		$(NATIVE_MARK_DISCONTINUITIES_ARG) \
		$(NATIVE_INSERT_DISCONTINUITY_PACKETS_ARG) \
		- $(NATIVE_PREFIX).recovered.ts 2>$(NATIVE_PREFIX).fec.log'
	@rc=0; $(PYTHON) -m dtmb.ts $(NATIVE_PREFIX).recovered.ts --json > $(NATIVE_PREFIX).ts.json || rc=$$?; test $$rc -eq 0 -o $$rc -eq 2

native-replay-llr: native-core
	@test -n "$(NATIVE_CAPTURE)" || (echo "set CAPTURE=... or NATIVE_CAPTURE=..." && exit 1)
	@test "$(NATIVE_AUTO_SYNC)" = "1" -o -n "$(NATIVE_PHASE_OFFSET)" || (echo "set NATIVE_AUTO_SYNC=1 or PHASE_OFFSET=..." && exit 1)
	mkdir -p $(dir $(NATIVE_PREFIX))
	bash -o pipefail -c '$(NATIVE_RECEIVE_SOURCE) \
		$(NATIVE_SYNC_ARG) \
		$(NATIVE_TIMING_TRACKING_ARG) \
		$(NATIVE_TIMING_DIAGNOSTICS_ARG) \
		--max-frames $(NATIVE_FRAMES) \
		--frequency-shift-hz $(NATIVE_FREQUENCY_SHIFT_HZ) \
		--equalizer pn \
		--pn-estimator $(NATIVE_PN_ESTIMATOR) \
		--pn-wideband-block-frames $(NATIVE_PN_WIDEBAND_BLOCK_FRAMES) \
		$(NATIVE_WIDEBAND_DIAGNOSTICS_ARG) \
		$(NATIVE_FRAME_RESIDUAL_DIAGNOSTICS_ARG) \
		$(NATIVE_SOURCE_CARRIER_RESIDUAL_ARG) \
		--pn-mmse $(NATIVE_PN_MMSE) \
		$(NATIVE_REMOVE_DC_ARG) \
		--normalization $(NATIVE_NORMALIZATION) \
		--system-info-index $(NATIVE_SYSTEM_INFO_INDEX) \
		$(NATIVE_RECEIVE_INPUT) - 2>$(NATIVE_PREFIX).extract.log \
		| $(CPP_DEINTERLEAVE_QAM64) \
		--mode $(NATIVE_INTERLEAVER_MODE) \
		--phase $(NATIVE_INTERLEAVER_PHASE) \
		$(NATIVE_BRANCH_GAIN_ARG) \
		--workers $(CPP_DEMAP_WORKERS) --chunk-symbols $(NATIVE_DEMAP_CHUNK_SYMBOLS) - $(NATIVE_LLR_OUTPUT) 2>$(NATIVE_PREFIX).demap.log'
	$(PYTHON) $(PIPELINE_DIR)/tag_pre_ldpc.py \
		--llr $(NATIVE_LLR_OUTPUT) \
		--output $(NATIVE_TAGGED_PRE_LDPC) \
		--source-capture $(NATIVE_CAPTURE) \
		--qam 64qam \
		--fec-rate $(NATIVE_FEC_RATE) \
		--interleaver-mode $(NATIVE_INTERLEAVER_MODE) \
		--interleaver-phase $(NATIVE_INTERLEAVER_PHASE) \
		--bits-per-frame 22464

native-live-capture: native-core
	@test "$(NATIVE_LIVE_SOURCE)" = "hackrf_transfer" -o "$(NATIVE_LIVE_SOURCE)" = "native_acquire" || (echo "NATIVE_LIVE_SOURCE must be hackrf_transfer or native_acquire" && exit 1)
	mkdir -p $(dir $(NATIVE_LIVE_CAPTURE_OUTPUT)) $(dir $(NATIVE_LIVE_PREFIX))
	$(PYTHON) $(PIPELINE_DIR)/check_hackrf_clock.py $(NATIVE_LIVE_CLOCK_ARGS)
	$(NATIVE_LIVE_NATIVE_ACQUIRE_COMMON) \
		$(NATIVE_LIVE_CAPTURE_LIMIT_ARG) \
		--output $(NATIVE_LIVE_CAPTURE_OUTPUT) \
		2>$(NATIVE_LIVE_PREFIX).acquire.log

native-live-stream: native-core
	@test "$(NATIVE_LIVE_SOURCE)" = "file" -o "$(NATIVE_LIVE_SOURCE)" = "hackrf_transfer" -o "$(NATIVE_LIVE_SOURCE)" = "native_acquire" || (echo "NATIVE_LIVE_SOURCE must be file, hackrf_transfer, or native_acquire" && exit 1)
	@test "$(NATIVE_LIVE_SOURCE)" != "file" -o -n "$(NATIVE_LIVE_INPUT)" || (echo "set NATIVE_LIVE_INPUT=... when NATIVE_LIVE_SOURCE=file" && exit 1)
	mkdir -p $(dir $(NATIVE_LIVE_OUTPUT)) $(dir $(NATIVE_LIVE_PREFIX))
	$(if $(filter file,$(NATIVE_LIVE_SOURCE)),true,$(PYTHON) $(PIPELINE_DIR)/check_hackrf_clock.py $(NATIVE_LIVE_CLOCK_ARGS))
	bash -o pipefail -c '$(NATIVE_LIVE_CI8_SOURCE) \
		$(NATIVE_LIVE_CI8_SOURCE_BUFFER_PIPE) \
		$(NATIVE_LIVE_MONITOR_PIPE) \
		| $(CPP_CI8_STATS) --passthrough - 2>$(NATIVE_LIVE_PREFIX).ci8.log \
		$(NATIVE_LIVE_RESAMPLE_PIPE) \
		| $(CPP_C3780_EXTRACT) \
		--auto-sync \
		--sync-frames $(NATIVE_SYNC_FRAMES) \
		--acquisition-frames $(NATIVE_ACQUISITION_FRAMES) \
		--auto-phase-adjustment $(NATIVE_LIVE_AUTO_PHASE_ADJUSTMENT) \
		$(NATIVE_LIVE_TIMING_TRACKING_ARG) \
		$(NATIVE_LIVE_TIMING_DIAGNOSTICS_ARG) \
		--max-frames $(NATIVE_LIVE_FRAMES) \
		--workers $(CPP_FRONTEND_WORKERS) \
		--frequency-shift-hz $(NATIVE_FREQUENCY_SHIFT_HZ) \
		--equalizer pn \
		--pn-estimator $(NATIVE_PN_ESTIMATOR) \
		--pn-wideband-block-frames $(NATIVE_PN_WIDEBAND_BLOCK_FRAMES) \
		$(NATIVE_LIVE_WIDEBAND_DIAGNOSTICS_ARG) \
		$(NATIVE_LIVE_FRAME_RESIDUAL_DIAGNOSTICS_ARG) \
		$(NATIVE_LIVE_SOURCE_CARRIER_RESIDUAL_ARG) \
		--pn-mmse $(NATIVE_PN_MMSE) \
		$(NATIVE_REMOVE_DC_ARG) \
		--normalization $(NATIVE_NORMALIZATION) \
		--system-info-index $(NATIVE_SYSTEM_INFO_INDEX) \
		- - 2>$(NATIVE_LIVE_PREFIX).extract.log \
		| $(CPP_DEINTERLEAVE_QAM64) \
		--mode $(NATIVE_INTERLEAVER_MODE) \
		--phase $(NATIVE_INTERLEAVER_PHASE) \
		$(NATIVE_BRANCH_GAIN_ARG) \
		--workers $(CPP_DEMAP_WORKERS) --chunk-symbols $(NATIVE_DEMAP_CHUNK_SYMBOLS) - - 2>$(NATIVE_LIVE_PREFIX).demap.log \
		| $(CPP_LDPC_DECODER) \
		--fec-rate $(NATIVE_FEC_RATE) \
		--alist python/dtmb/data/dtmb_ldpc_rate$(NATIVE_FEC_RATE).alist \
		--codewords-per-frame $(NATIVE_CODEWORDS_PER_FRAME) \
		--workers $(CPP_LDPC_WORKERS) \
		--max-iterations $(NATIVE_LDPC_MAX_ITERATIONS) \
		$(NATIVE_LDPC_ATTENUATION_ARG) \
		$(NATIVE_LDPC_RETRY_ARGS) \
		--decode-batch-frames $(NATIVE_LDPC_DECODE_BATCH_FRAMES) \
		$(NATIVE_LIVE_LDPC_EARLY_REJECT_ARG) \
		$(NATIVE_LIVE_LDPC_FRAME_DIAGNOSTICS_ARG) \
		$(NATIVE_HARD_H_GATE_ARG) \
		$(NATIVE_HARD_H_GATE_FILL_ARG) \
		$(NATIVE_FORCE_FRAME_RANGE_ARGS) \
		$(NATIVE_CODEWORD_LLR_SCALE_ARGS) \
		$(NATIVE_VARIABLE_LLR_SCALE_ARGS) \
		$(NATIVE_VARIABLE_MOD_LLR_SCALE_ARGS) \
		$(NATIVE_CLEAN_FRAMES_ARG) \
		$(NATIVE_EMIT_CLEAN_CODEWORDS_ARG) \
		$(NATIVE_EMIT_BCH_CLEAN_CODEWORDS_ARG) \
		$(NATIVE_EMIT_FORCED_CODEWORDS_ARG) \
		$(NATIVE_MARK_DISCONTINUITIES_ARG) \
		$(NATIVE_INSERT_DISCONTINUITY_PACKETS_ARG) \
		- $(NATIVE_LIVE_OUTPUT) 2>$(NATIVE_LIVE_PREFIX).fec.log'; \
	rc=$$?; test $$rc -eq 0 -o $$rc -eq 141

native-live-playback: native-core
	@test "$(NATIVE_LIVE_SOURCE)" = "file" -o "$(NATIVE_LIVE_SOURCE)" = "hackrf_transfer" -o "$(NATIVE_LIVE_SOURCE)" = "native_acquire" || (echo "NATIVE_LIVE_SOURCE must be file, hackrf_transfer, or native_acquire" && exit 1)
	@test "$(NATIVE_LIVE_SOURCE)" != "file" -o -n "$(NATIVE_LIVE_INPUT)" || (echo "set NATIVE_LIVE_INPUT=... when NATIVE_LIVE_SOURCE=file" && exit 1)
	mkdir -p $(dir $(NATIVE_LIVE_PREFIX))
	$(if $(filter file,$(NATIVE_LIVE_SOURCE)),true,$(PYTHON) $(PIPELINE_DIR)/check_hackrf_clock.py $(NATIVE_LIVE_CLOCK_ARGS))
	bash -o pipefail -c '$(NATIVE_LIVE_CI8_SOURCE) \
		$(NATIVE_LIVE_CI8_SOURCE_BUFFER_PIPE) \
		$(NATIVE_LIVE_MONITOR_PIPE) \
		| $(CPP_CI8_STATS) --passthrough - 2>$(NATIVE_LIVE_PREFIX).ci8.log \
		$(NATIVE_LIVE_RESAMPLE_PIPE) \
		| $(CPP_C3780_EXTRACT) \
		--auto-sync \
		--sync-frames $(NATIVE_SYNC_FRAMES) \
		--acquisition-frames $(NATIVE_ACQUISITION_FRAMES) \
		--auto-phase-adjustment $(NATIVE_LIVE_AUTO_PHASE_ADJUSTMENT) \
		$(NATIVE_LIVE_TIMING_TRACKING_ARG) \
		--max-frames $(NATIVE_LIVE_FRAMES) \
		--workers $(CPP_FRONTEND_WORKERS) \
		--frequency-shift-hz $(NATIVE_FREQUENCY_SHIFT_HZ) \
		--equalizer pn \
		--pn-estimator $(NATIVE_PN_ESTIMATOR) \
		--pn-wideband-block-frames $(NATIVE_PN_WIDEBAND_BLOCK_FRAMES) \
		$(NATIVE_LIVE_WIDEBAND_DIAGNOSTICS_ARG) \
		$(NATIVE_LIVE_SOURCE_CARRIER_RESIDUAL_ARG) \
		--pn-mmse $(NATIVE_PN_MMSE) \
		$(NATIVE_REMOVE_DC_ARG) \
		--normalization $(NATIVE_NORMALIZATION) \
		--system-info-index $(NATIVE_SYSTEM_INFO_INDEX) \
		- - 2>$(NATIVE_LIVE_PREFIX).extract.log \
		| $(CPP_DEINTERLEAVE_QAM64) \
		--mode $(NATIVE_INTERLEAVER_MODE) \
		--phase $(NATIVE_INTERLEAVER_PHASE) \
		$(NATIVE_BRANCH_GAIN_ARG) \
		--workers $(CPP_DEMAP_WORKERS) --chunk-symbols $(NATIVE_DEMAP_CHUNK_SYMBOLS) - - 2>$(NATIVE_LIVE_PREFIX).demap.log \
		| $(CPP_LDPC_DECODER) \
		--fec-rate $(NATIVE_FEC_RATE) \
		--alist python/dtmb/data/dtmb_ldpc_rate$(NATIVE_FEC_RATE).alist \
		--codewords-per-frame $(NATIVE_CODEWORDS_PER_FRAME) \
		--workers $(CPP_LDPC_WORKERS) \
		--max-iterations $(NATIVE_LDPC_MAX_ITERATIONS) \
		$(NATIVE_LDPC_ATTENUATION_ARG) \
		$(NATIVE_LDPC_RETRY_ARGS) \
		--decode-batch-frames $(NATIVE_LDPC_DECODE_BATCH_FRAMES) \
		$(NATIVE_LIVE_LDPC_EARLY_REJECT_ARG) \
		$(NATIVE_LIVE_LDPC_FRAME_DIAGNOSTICS_ARG) \
		$(NATIVE_HARD_H_GATE_ARG) \
		$(NATIVE_HARD_H_GATE_FILL_ARG) \
		$(NATIVE_FORCE_FRAME_RANGE_ARGS) \
		$(NATIVE_CODEWORD_LLR_SCALE_ARGS) \
		$(NATIVE_VARIABLE_LLR_SCALE_ARGS) \
		$(NATIVE_VARIABLE_MOD_LLR_SCALE_ARGS) \
		$(NATIVE_CLEAN_FRAMES_ARG) \
		$(NATIVE_EMIT_CLEAN_CODEWORDS_ARG) \
		$(NATIVE_EMIT_BCH_CLEAN_CODEWORDS_ARG) \
		$(NATIVE_EMIT_FORCED_CODEWORDS_ARG) \
		$(NATIVE_MARK_DISCONTINUITIES_ARG) \
		$(NATIVE_INSERT_DISCONTINUITY_PACKETS_ARG) \
		- - 2>$(NATIVE_LIVE_PREFIX).fec.log \
		| $(FFPLAY) -hide_banner -loglevel warning -fflags nobuffer -flags low_delay -f mpegts -'; \
	rc=$$?; test $$rc -eq 0 -o $$rc -eq 141

native-live-monitor: native-core
	mkdir -p $(dir $(NATIVE_LIVE_MONITOR_PREFIX))
	$(PYTHON) $(PIPELINE_DIR)/check_hackrf_clock.py --clock-source $(CAPTURE_CLOCK_SOURCE) --hackrf-clock $(HACKRF_CLOCK) $(HACKRF_CLOCK_DEVICE_ARG) --log $(NATIVE_LIVE_MONITOR_PREFIX).clkin.log $(if $(strip $(CAPTURE_EXTERNAL_REFERENCE_HZ)),--external-reference-hz $(CAPTURE_EXTERNAL_REFERENCE_HZ),)
	$(CPP_LIVE_MONITOR) $(NATIVE_LIVE_MONITOR_COMMON_ARGS) $(NATIVE_LIVE_MONITOR_SAMPLE_LIMIT_ARG) --hackrf --json-out $(NATIVE_LIVE_MONITOR_PREFIX).ndjson 2>$(NATIVE_LIVE_MONITOR_PREFIX).monitor.log

native-live-llr-scan: native-core
	$(PYTHON) $(PIPELINE_DIR)/native_live_llr_scan.py \
		--centers $(NATIVE_LIVE_LLR_SCAN_CENTERS) \
		--output-dir $(NATIVE_LIVE_LLR_SCAN_OUTPUT_DIR) \
		--sample-rate $(NATIVE_LIVE_SAMPLE_RATE) \
		--bandwidth $(NATIVE_LIVE_BANDWIDTH) \
		--samples $(NATIVE_LIVE_LLR_SCAN_SAMPLES) \
		--frames $(NATIVE_LIVE_LLR_SCAN_FRAMES) \
		--amp $(AMP) \
		--lna-gain $(LNA_GAIN) \
		--vga-gain $(VGA_GAIN) \
		--clock-source $(CAPTURE_CLOCK_SOURCE) \
		$(if $(strip $(CAPTURE_EXTERNAL_REFERENCE_HZ)),--external-reference-hz $(CAPTURE_EXTERNAL_REFERENCE_HZ),) \
		--hackrf-clock $(HACKRF_CLOCK) \
		--hackrf-transfer $(HACKRF_TRANSFER) \
		--hackrf-acquire $(CPP_HACKRF_ACQUIRE) \
		--capture-source $(NATIVE_LIVE_SOURCE) \
		$(if $(filter file,$(NATIVE_LIVE_SOURCE)),--input $(NATIVE_LIVE_INPUT),) \
		--max-queued-bytes $(NATIVE_LIVE_ACQUIRE_MAX_QUEUED_BYTES) \
		$(HACKRF_CLOCK_DEVICE_ARG) \
		--ci8-stats $(CPP_CI8_STATS) \
		--resampler $(CPP_CI8_RESAMPLE) \
		--extractor $(CPP_C3780_EXTRACT) \
		--demapper $(CPP_DEINTERLEAVE_QAM64) \
		--h-scorer $(CPP_LDPC_H_GATE) \
		$(if $(filter 1,$(NATIVE_LIVE_SOURCE_SCORE)),--source-score --source-scorer $(CPP_PRE_LDPC_SOURCE_SCORE) --source-score-top $(NATIVE_LIVE_SOURCE_SCORE_TOP) --source-score-frame-group-cap $(NATIVE_LIVE_SOURCE_SCORE_FRAME_GROUP_CAP) --source-score-weak-llr-threshold $(NATIVE_LIVE_SOURCE_SCORE_WEAK_LLR_THRESHOLD) --source-score-codewords-per-frame $(NATIVE_CODEWORDS_PER_FRAME) $(foreach range,$(NATIVE_LIVE_SOURCE_SCORE_FEC_FRAME_RANGES),--source-score-fec-frame-range $(range)) $(foreach variable,$(NATIVE_LIVE_SOURCE_SCORE_TRACK_LDPC_VARIABLES),--source-score-track-ldpc-variable $(variable)),) \
		$(if $(strip $(NATIVE_BRANCH_GAIN_BRANCHES)),--branch-gain-branches $(NATIVE_BRANCH_GAIN_BRANCHES) --branch-gain-reliability-threshold $(NATIVE_BRANCH_GAIN_RELIABILITY_THRESHOLD) --branch-gain-min-symbols $(NATIVE_BRANCH_GAIN_MIN_SYMBOLS) $(foreach range,$(NATIVE_BRANCH_GAIN_SOURCE_FRAME_RANGES),--branch-gain-source-frame-range $(range)) $(if $(filter 1,$(NATIVE_BRANCH_GAIN_DIAGNOSTICS)),--branch-gain-diagnostics,),) \
		$(foreach rule,$(NATIVE_SOURCE_FRAME_SYMBOL_GAIN_RULES),--source-frame-symbol-gain $(rule)) \
		$(foreach rule,$(NATIVE_SOURCE_FRAME_AXIS_AFFINE_RULES),--source-frame-axis-affine $(rule)) \
		$(foreach rule,$(NATIVE_SOURCE_FRAME_LLR_SCALE_RULES),--source-frame-llr-scale $(rule)) \
		$(foreach rule,$(NATIVE_SOURCE_CARRIER_LLR_SCALE_RULES),--source-carrier-llr-scale $(rule)) \
		--alist python/dtmb/data/dtmb_ldpc_rate$(NATIVE_FEC_RATE).alist \
		--resample-workers $(CPP_RESAMPLE_WORKERS) \
		--frontend-workers $(CPP_FRONTEND_WORKERS) \
		--demap-workers $(CPP_DEMAP_WORKERS) \
		--frequency-shift-hz $(NATIVE_FREQUENCY_SHIFT_HZ) \
		--system-info-index $(NATIVE_SYSTEM_INFO_INDEX) \
		--fec-rate $(NATIVE_FEC_RATE) \
		--interleaver-mode $(NATIVE_INTERLEAVER_MODE) \
		--interleaver-phase $(NATIVE_INTERLEAVER_PHASE) \
		--timing-search-radius $(NATIVE_LIVE_TIMING_SEARCH_RADIUS) \
		--timing-search-threshold $(NATIVE_LIVE_TIMING_SEARCH_THRESHOLD) \
		--timing-trajectory-interval-frames $(NATIVE_LIVE_TIMING_TRAJECTORY_INTERVAL_FRAMES) \
		--timing-trajectory-fit-points $(NATIVE_LIVE_TIMING_TRAJECTORY_FIT_POINTS) \
		--timing-trajectory-max-innovation-samples $(NATIVE_LIVE_TIMING_TRAJECTORY_MAX_INNOVATION_SAMPLES) \
		$(if $(filter 1,$(NATIVE_LIVE_TIMING_TRAJECTORY_LOCAL_SEARCH)),--timing-trajectory-local-search --timing-trajectory-local-search-min-improvement $(NATIVE_LIVE_TIMING_TRAJECTORY_LOCAL_SEARCH_MIN_IMPROVEMENT),) \
		$(if $(filter 1,$(NATIVE_LIVE_TIMING_DIAGNOSTICS)),--timing-diagnostics,) \
		$(if $(filter 1,$(NATIVE_LIVE_FRAME_RESIDUAL_DIAGNOSTICS)),--frame-residual-diagnostics,) \
		--pn-estimator $(NATIVE_PN_ESTIMATOR) \
		--pn-wideband-block-frames $(NATIVE_PN_WIDEBAND_BLOCK_FRAMES) \
		--pn-mmse $(NATIVE_PN_MMSE) \
		--normalization $(NATIVE_NORMALIZATION) \
		--h-window-codewords $(NATIVE_LIVE_H_WINDOW_CODEWORDS) \
		--h-window-step-codewords $(NATIVE_LIVE_H_WINDOW_STEP_CODEWORDS) \
		--h-window-threshold $(NATIVE_LIVE_H_WINDOW_THRESHOLD) \
		--h-score-mode $(NATIVE_LIVE_H_SCORE_MODE) \
		$(if $(filter 1,$(NATIVE_LIVE_STREAM_H_SCORE)),--stream-h-score,) \
		$(if $(filter 1,$(NATIVE_REMOVE_DC)),,--no-remove-dc)

# Synthesis writes both the CI8 and the sidecar JSON in a single invocation;
# grouped-target syntax (&:) prevents make from running the recipe twice.
$(SYNTHETIC_CI8) $(SYNTHETIC_CI8).json &: $(SYNTHETIC_INPUT_TS) $(PIPELINE_DIR)/synthesize.py $(PYTHON_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/synthesize.py --input-ts $(SYNTHETIC_INPUT_TS) --output $(SYNTHETIC_CI8)

# Synthetic skips sysinfo because we know the mode (64QAM, rate 3, mode 1).
$(SYNTHETIC_PREFIX).recovered.ts $(SYNTHETIC_PREFIX).receiver.json &: $(SYNTHETIC_CI8) $(SYNTHETIC_CI8).json $(PIPELINE_DIR)/receive.py $(PYTHON_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/receive.py \
		--capture $(SYNTHETIC_CI8) \
		--output-ts $(SYNTHETIC_PREFIX).recovered.ts \
		--output-json $(SYNTHETIC_PREFIX).receiver.json \
		--frames $(RECEIVE_FRAMES) \
		--synthetic-loopback

$(SYNTHETIC_PREFIX).probe.json $(SYNTHETIC_MP4) &: $(SYNTHETIC_PREFIX).recovered.ts $(PIPELINE_DIR)/transcode.py
	$(PYTHON) $(PIPELINE_DIR)/transcode.py \
		--input-ts $(SYNTHETIC_PREFIX).recovered.ts \
		--probe-json $(SYNTHETIC_PREFIX).probe.json \
		--output-mp4 $(SYNTHETIC_MP4) \
		--ffmpeg-bin $(FFMPEG) \
		--ffprobe-bin $(FFPROBE) \
		--require-video

# -----------------------------------------------------------------------------
# Wall captures
# -----------------------------------------------------------------------------
# Top-level wall targets aggregate acquire + sysinfo + receive + transcode
# so a single `make -f pipeline.mk wall-20msps` produces every artifact
# worth inspecting. Acquire and sysinfo have no dependency on each other,
# so `make -j 2 wall-20msps` parallelises them naturally.
wall-20msps: $(WALL_20MSPS_PREFIX).acquire.json $(WALL_20MSPS_PREFIX).sysinfo.json $(WALL_20MSPS_PREFIX).sysinfo_oracle.raw.json $(WALL_20MSPS_PREFIX).llr.f32 $(WALL_20MSPS_PREFIX).llr_health.json $(WALL_20MSPS_PREFIX).decoded.ts $(WALL_20MSPS_MP4) $(WALL_20MSPS_VISUALS) $(if $(filter 1,$(CROP_WIDEBAND_CAPTURE)),$(WALL_20MSPS_CROP_JSON),)

ifeq ($(CROP_WIDEBAND_CAPTURE),1)
$(WALL_20MSPS_WIDE_CI8) $(WALL_20MSPS_WIDE_CI8).json &: $(PIPELINE_DIR)/capture_hackrf.py
	$(PYTHON) $(PIPELINE_DIR)/capture_hackrf.py \
		--output $(WALL_20MSPS_WIDE_CI8) \
		--frequency $(WALL_20MSPS_TUNE_FREQUENCY) \
		--sample-rate 20000000 \
		$(WALL_20MSPS_WIDE_BANDWIDTH_ARG) \
		--duration $(DURATION) \
		--amp $(AMP) \
		--lna-gain $(LNA_GAIN) \
		--vga-gain $(VGA_GAIN) \
		$(CAPTURE_CLOCK_ARGS) \
		--antenna "$(ANTENNA), 20MHz wide capture" \
		--location "$(LOCATION)" \
		--notes "$(if $(NOTES),$(NOTES); ,)target_frequency_hz=$(FREQUENCY); tune_offset_hz=$(CAPTURE_TUNE_OFFSET_HZ); workflow=wide20_crop"

$(WALL_20MSPS_CI8) $(WALL_20MSPS_CI8).json $(WALL_20MSPS_CROP_JSON) &: $(WALL_20MSPS_WIDE_CI8) $(WALL_20MSPS_WIDE_CI8).json $(PIPELINE_DIR)/crop_capture.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/crop_capture.py \
		--input $(WALL_20MSPS_WIDE_CI8) \
		--output $(WALL_20MSPS_CI8) \
		--analysis-json $(WALL_20MSPS_CROP_JSON) \
		--target-frequency-hz $(FREQUENCY) \
		--tuner-frequency-hz $(WALL_20MSPS_TUNE_FREQUENCY) \
		--frequency-shift-hz $(shell $(PYTHON) -c "print(-float($(CAPTURE_TUNE_OFFSET_HZ)))") \
		--output-sample-rate $(CROP_OUTPUT_SAMPLE_RATE) \
		--output-bandwidth-hz $(CROP_OUTPUT_BANDWIDTH_HZ) \
		--lowpass-cutoff-hz $(CROP_LOWPASS_CUTOFF_HZ) \
		--lowpass-transition-hz $(CROP_LOWPASS_TRANSITION_HZ)
else
$(WALL_20MSPS_CI8) $(WALL_20MSPS_CI8).json &: $(PIPELINE_DIR)/capture_hackrf.py
	$(PYTHON) $(PIPELINE_DIR)/capture_hackrf.py \
		--output $(WALL_20MSPS_CI8) \
		--frequency $(FREQUENCY) \
		--sample-rate 20000000 \
		$(CAPTURE_BANDWIDTH_ARG) \
		--duration $(DURATION) \
		--amp $(AMP) \
		--lna-gain $(LNA_GAIN) \
		--vga-gain $(VGA_GAIN) \
		$(CAPTURE_CLOCK_ARGS) \
		--antenna "$(ANTENNA)" \
		--location "$(LOCATION)"
endif

wall-7msps: $(WALL_7MSPS_PREFIX).acquire.json $(WALL_7MSPS_PREFIX).sysinfo.json $(WALL_7MSPS_PREFIX).sysinfo_oracle.raw.json $(WALL_7MSPS_PREFIX).llr.f32 $(WALL_7MSPS_PREFIX).llr_health.json $(WALL_7MSPS_PREFIX).decoded.ts $(WALL_7MSPS_MP4) $(WALL_7MSPS_VISUALS)

$(WALL_7MSPS_CI8) $(WALL_7MSPS_CI8).json &: $(PIPELINE_DIR)/capture_hackrf.py
	$(PYTHON) $(PIPELINE_DIR)/capture_hackrf.py \
		--output $(WALL_7MSPS_CI8) \
		--frequency $(FREQUENCY) \
		--sample-rate 7560000 \
		$(WALL_7MSPS_BANDWIDTH_ARG) \
		--duration $(DURATION) \
		--amp $(AMP) \
		--lna-gain $(LNA_GAIN) \
		--vga-gain $(VGA_GAIN) \
		$(CAPTURE_CLOCK_ARGS) \
		--antenna "$(ANTENNA), native 7.56 MSps" \
		--location "$(LOCATION)"

# -----------------------------------------------------------------------------
# Single-stage convenience aliases (use CAPTURE=path/to/file.ci8)
# -----------------------------------------------------------------------------
ifneq ($(CAPTURE),)
CAPTURE_RELATIVE := $(patsubst $(RAW_CAPTURES_DIR)/%.ci8,%,$(CAPTURE))
CAPTURE_ANALYSIS_PREFIX ?= $(if $(filter $(RAW_CAPTURES_DIR)/%.ci8,$(CAPTURE)),$(CAPTURE_ANALYSIS_DIR)/$(CAPTURE_RELATIVE),$(CAPTURE:.ci8=))
CAPTURE_VISUALS := $(foreach suffix,$(VISUAL_MANIFEST_SUFFIXES),$(CAPTURE_ANALYSIS_PREFIX)$(suffix))
acquire: $(CAPTURE_ANALYSIS_PREFIX).acquire.json
sysinfo: $(CAPTURE_ANALYSIS_PREFIX).sysinfo.json
sysinfo-oracle: $(CAPTURE_ANALYSIS_PREFIX).sysinfo_oracle.raw.json
receive: $(CAPTURE_ANALYSIS_PREFIX).recovered.ts $(CAPTURE_ANALYSIS_PREFIX).receiver.json
demap:   $(CAPTURE_ANALYSIS_PREFIX).llr.f32 $(CAPTURE_ANALYSIS_PREFIX).llr.f32.json $(CAPTURE_ANALYSIS_PREFIX).demap.json
tag-pre-ldpc: $(CAPTURE_ANALYSIS_PREFIX).tagged.pre_ldpc.npz $(CAPTURE_ANALYSIS_PREFIX).tagged.pre_ldpc.json
llr-health: $(CAPTURE_ANALYSIS_PREFIX).llr_health.json
decode-ldpc: $(CAPTURE_ANALYSIS_PREFIX).decoded.ts $(CAPTURE_ANALYSIS_PREFIX).decode_ldpc.json
timing-trajectory: $(CAPTURE_ANALYSIS_PREFIX).timing_trajectory.json
video-preflight: $(CAPTURE_ANALYSIS_PREFIX).video_preflight.json
probe:   $(CAPTURE_ANALYSIS_PREFIX).probe.json
visuals: $(CAPTURE_VISUALS)
else
acquire sysinfo sysinfo-oracle receive demap tag-pre-ldpc llr-health decode-ldpc timing-trajectory video-preflight probe visuals:
	@echo "set CAPTURE=captures/raw/path/to/file.ci8" && exit 1
endif

# -----------------------------------------------------------------------------
# Pattern rules (shared by synthetic + wall targets)
# -----------------------------------------------------------------------------
$(CAPTURE_ANALYSIS_DIR)/%.acquire.json: $(RAW_CAPTURES_DIR)/%.ci8 $(RAW_CAPTURES_DIR)/%.ci8.json $(PIPELINE_DIR)/acquire.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/acquire.py --capture $< --output $@ --frequency-shift $(FREQUENCY_SHIFT)
	$(PYTHON) $(PIPELINE_DIR)/clipping_gate.py --capture $< --acquire-json $@ --write-verdict $(CAPTURE_ANALYSIS_DIR)/$*.clipping.json

$(CAPTURE_ANALYSIS_DIR)/%.gate_a.visual.json: $(RAW_CAPTURES_DIR)/%.ci8 $(CAPTURE_ANALYSIS_DIR)/%.acquire.json $(PYTHON_SOURCES)
	$(PYTHON) -m dtmb.gate_visuals gate-a $< --acquire-json $(CAPTURE_ANALYSIS_DIR)/$*.acquire.json --png $(CAPTURE_ANALYSIS_DIR)/$*.gate_a.png --manifest $(CAPTURE_ANALYSIS_DIR)/$*.gate_a.visual.json

$(CAPTURE_ANALYSIS_DIR)/%.sysinfo.json: $(RAW_CAPTURES_DIR)/%.ci8 $(RAW_CAPTURES_DIR)/%.ci8.json $(TIMING_TRAJECTORY_PREREQ) $(PIPELINE_DIR)/sysinfo.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/sysinfo.py --capture $< --output $@ --frames $(SYSINFO_FRAMES) --system-info-index $(SYSINFO_INDEX) --frequency-shift $(FREQUENCY_SHIFT) $(PHASE_OFFSET_ARG) --body-window-offset $(BODY_WINDOW_OFFSET) --fft-bin-shift $(FFT_BIN_SHIFT) --frequency-deinterleaver-direction $(FREQUENCY_DEINTERLEAVER_DIRECTION) --data-carrier-order $(DATA_CARRIER_ORDER) --carrier-permutation $(CARRIER_PERMUTATION) --logical-position-shift $(LOGICAL_POSITION_SHIFT) $(BODY_WINDOW_FALLBACK_ARG) $(call TIMING_ARGS_FOR,$(CAPTURE_ANALYSIS_DIR)/$*) $(QUIET_ARG)

$(CAPTURE_ANALYSIS_DIR)/%.sysinfo_oracle.source.json: $(RAW_CAPTURES_DIR)/%.ci8 $(RAW_CAPTURES_DIR)/%.ci8.json $(TIMING_TRAJECTORY_PREREQ) $(PIPELINE_DIR)/sysinfo.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/sysinfo.py \
		--capture $< \
		--output $@ \
		--frames $(SYSINFO_ORACLE_FRAMES) \
		--system-info-index $(SYSINFO_INDEX) \
		--frequency-shift $(FREQUENCY_SHIFT) \
		$(PHASE_OFFSET_ARG) \
		--body-window-offset $(BODY_WINDOW_OFFSET) \
		--fft-bin-shift $(FFT_BIN_SHIFT) \
		--frequency-deinterleaver-direction $(FREQUENCY_DEINTERLEAVER_DIRECTION) \
		--data-carrier-order $(DATA_CARRIER_ORDER) \
		--carrier-permutation $(CARRIER_PERMUTATION) \
		--logical-position-shift $(LOGICAL_POSITION_SHIFT) \
		--equalizer $(SYSINFO_ORACLE_EQUALIZER) \
		$(call TIMING_ARGS_FOR,$(CAPTURE_ANALYSIS_DIR)/$*) \
		--no-timing-continuity \
		$(QUIET_ARG)

$(CAPTURE_ANALYSIS_DIR)/%.sysinfo_oracle.raw.json: $(CAPTURE_ANALYSIS_DIR)/%.sysinfo_oracle.source.json scripts/probe_sysinfo_oracle.py $(PYTHON_SOURCES)
	$(PYTHON) scripts/probe_sysinfo_oracle.py \
		$< \
		--iq-source raw \
		--omit-frames \
		--json $@

$(CAPTURE_ANALYSIS_DIR)/%.recovered.ts $(CAPTURE_ANALYSIS_DIR)/%.receiver.json &: $(RAW_CAPTURES_DIR)/%.ci8 $(RAW_CAPTURES_DIR)/%.ci8.json $(CAPTURE_ANALYSIS_DIR)/%.sysinfo.json $(CAPTURE_ANALYSIS_DIR)/%.sysinfo_oracle.raw.json $(TIMING_TRAJECTORY_PREREQ) $(PIPELINE_DIR)/receive.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/receive.py \
		--capture $(RAW_CAPTURES_DIR)/$*.ci8 \
		$(PHASE_OFFSET_ARG) \
		--sysinfo $(CAPTURE_ANALYSIS_DIR)/$*.sysinfo.json \
		--sysinfo-oracle $(CAPTURE_ANALYSIS_DIR)/$*.sysinfo_oracle.raw.json \
		--output-ts $(CAPTURE_ANALYSIS_DIR)/$*.recovered.ts \
		--output-json $(CAPTURE_ANALYSIS_DIR)/$*.receiver.json \
		--frames $(RECEIVE_FRAMES) \
		--frequency-shift $(FREQUENCY_SHIFT) \
		--body-window-offset $(BODY_WINDOW_OFFSET) \
		--fft-bin-shift $(FFT_BIN_SHIFT) \
		--frequency-deinterleaver-direction $(FREQUENCY_DEINTERLEAVER_DIRECTION) \
		--data-carrier-order $(DATA_CARRIER_ORDER) \
		--carrier-permutation $(CARRIER_PERMUTATION) \
		--logical-position-shift $(LOGICAL_POSITION_SHIFT) \
		--dd-max-hard-bit-bias $(DD_MAX_HARD_BIT_BIAS) \
		$(RECEIVE_FORCE_ARGS) \
		$(call TIMING_ARGS_FOR,$(CAPTURE_ANALYSIS_DIR)/$*) \
		$(QUIET_ARG)

# demap: run the DSP front end + LLR stream only. Writes <prefix>.llr.f32
# (the post-deinterleaver soft bits) and <prefix>.llr.f32.json (framing
# sidecar telling the decode stage what rate/mode this stream is). This
# is the first half of the decoupled FEC path; decode_ldpc consumes the
# pair and produces the recovered TS.
$(CAPTURE_ANALYSIS_DIR)/%.llr.f32 $(CAPTURE_ANALYSIS_DIR)/%.llr.f32.json $(CAPTURE_ANALYSIS_DIR)/%.demap.json &: $(RAW_CAPTURES_DIR)/%.ci8 $(RAW_CAPTURES_DIR)/%.ci8.json $(CAPTURE_ANALYSIS_DIR)/%.sysinfo.json $(CAPTURE_ANALYSIS_DIR)/%.sysinfo_oracle.raw.json $(TIMING_TRAJECTORY_PREREQ) $(PIPELINE_DIR)/demap.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/demap.py \
		--capture $(RAW_CAPTURES_DIR)/$*.ci8 \
		$(PHASE_OFFSET_ARG) \
		--sysinfo $(CAPTURE_ANALYSIS_DIR)/$*.sysinfo.json \
		--sysinfo-oracle $(CAPTURE_ANALYSIS_DIR)/$*.sysinfo_oracle.raw.json \
		--output-llr $(CAPTURE_ANALYSIS_DIR)/$*.llr.f32 \
		--output-json $(CAPTURE_ANALYSIS_DIR)/$*.demap.json \
		--frames $(RECEIVE_FRAMES) \
		--frequency-shift $(FREQUENCY_SHIFT) \
		--body-window-offset $(BODY_WINDOW_OFFSET) \
		--fft-bin-shift $(FFT_BIN_SHIFT) \
		--frequency-deinterleaver-direction $(FREQUENCY_DEINTERLEAVER_DIRECTION) \
		--data-carrier-order $(DATA_CARRIER_ORDER) \
		--carrier-permutation $(CARRIER_PERMUTATION) \
		--logical-position-shift $(LOGICAL_POSITION_SHIFT) \
		--dd-max-hard-bit-bias $(DD_MAX_HARD_BIT_BIAS) \
		$(DEMAP_FORCE_ARGS) \
		$(call TIMING_ARGS_FOR,$(CAPTURE_ANALYSIS_DIR)/$*) \
		$(QUIET_ARG)

$(CAPTURE_ANALYSIS_DIR)/%.timing_trajectory.json: $(RAW_CAPTURES_DIR)/%.ci8 $(RAW_CAPTURES_DIR)/%.ci8.json $(PIPELINE_DIR)/timing_trajectory.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/timing_trajectory.py \
		--capture $< \
		--output $@ \
		$(TIMING_TRAJECTORY_ARGS) \
		--frames $(RECEIVE_FRAMES) \
		--frequency-shift $(FREQUENCY_SHIFT)

# decode-ldpc: run only the LDPC + BCH + descrambler + TS-lock stage on
# the dumped LLR stream. Kept separate from `receive` so a future backend
# (AFF3CT, liquid-dsp, Python `ldpc`) can swap in without touching any
# DSP. Produces <prefix>.decoded.ts and <prefix>.decode_ldpc.json.
$(CAPTURE_ANALYSIS_DIR)/%.decoded.ts $(CAPTURE_ANALYSIS_DIR)/%.decode_ldpc.json &: $(CAPTURE_ANALYSIS_DIR)/%.llr.f32 $(CAPTURE_ANALYSIS_DIR)/%.llr.f32.json $(PIPELINE_DIR)/decode_ldpc.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/decode_ldpc.py \
		--llr $(CAPTURE_ANALYSIS_DIR)/$*.llr.f32 \
		--output-ts $(CAPTURE_ANALYSIS_DIR)/$*.decoded.ts \
		--output-json $(CAPTURE_ANALYSIS_DIR)/$*.decode_ldpc.json \
		--backend $(LDPC_BACKEND) \
		--cpp-decoder $(CPP_LDPC_DECODER) \
		--cpp-workers $(CPP_LDPC_WORKERS)

$(CAPTURE_ANALYSIS_DIR)/%.llr_health.json: $(CAPTURE_ANALYSIS_DIR)/%.llr.f32 $(CAPTURE_ANALYSIS_DIR)/%.llr.f32.json $(PYTHON_SOURCES)
	-$(PYTHON) -m dtmb.llr_health \
		$(CAPTURE_ANALYSIS_DIR)/$*.llr.f32 \
		--output-json $@

# Add deterministic source-frame/carrier/branch/QAM-plane provenance to each
# post-deinterleaver LLR bit without rerunning the receiver front end.
$(CAPTURE_ANALYSIS_DIR)/%.tagged.pre_ldpc.npz $(CAPTURE_ANALYSIS_DIR)/%.tagged.pre_ldpc.json &: $(CAPTURE_ANALYSIS_DIR)/%.llr.f32 $(CAPTURE_ANALYSIS_DIR)/%.llr.f32.json $(PIPELINE_DIR)/tag_pre_ldpc.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/tag_pre_ldpc.py \
		--llr $(CAPTURE_ANALYSIS_DIR)/$*.llr.f32 \
		--llr-sidecar $(CAPTURE_ANALYSIS_DIR)/$*.llr.f32.json \
		--output $(CAPTURE_ANALYSIS_DIR)/$*.tagged.pre_ldpc.npz

$(CAPTURE_ANALYSIS_DIR)/%.video_preflight.json: $(RAW_CAPTURES_DIR)/%.ci8 $(RAW_CAPTURES_DIR)/%.ci8.json $(CAPTURE_ANALYSIS_DIR)/%.acquire.json $(CAPTURE_ANALYSIS_DIR)/%.sysinfo.json $(CAPTURE_ANALYSIS_DIR)/%.sysinfo_oracle.raw.json $(CAPTURE_ANALYSIS_DIR)/%.timing_trajectory.json $(CAPTURE_ANALYSIS_DIR)/%.demap.json $(CAPTURE_ANALYSIS_DIR)/%.llr_health.json $(PIPELINE_DIR)/video_preflight.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/video_preflight.py \
		--capture $(RAW_CAPTURES_DIR)/$*.ci8 \
		--acquire-json $(CAPTURE_ANALYSIS_DIR)/$*.acquire.json \
		--clipping-json $(CAPTURE_ANALYSIS_DIR)/$*.clipping.json \
		--sysinfo-json $(CAPTURE_ANALYSIS_DIR)/$*.sysinfo.json \
		--sysinfo-oracle-json $(CAPTURE_ANALYSIS_DIR)/$*.sysinfo_oracle.raw.json \
		--timing-trajectory-json $(CAPTURE_ANALYSIS_DIR)/$*.timing_trajectory.json \
		--demap-json $(CAPTURE_ANALYSIS_DIR)/$*.demap.json \
		--llr-health-json $(CAPTURE_ANALYSIS_DIR)/$*.llr_health.json \
		--probe-json $(CAPTURE_ANALYSIS_DIR)/$*.probe.json \
		--output-json $@

# probe.json is the canonical verdict for wall-style captures. It always
# runs ffprobe on the recovered TS and, when probe passes, also invokes
# ffmpeg to produce the MP4 next to it. Failures are recorded in the JSON
# rather than aborting so `make` can continue to the next target.
$(CAPTURE_ANALYSIS_DIR)/%.probe.json: $(CAPTURE_ANALYSIS_DIR)/%.recovered.ts $(PIPELINE_DIR)/transcode.py $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/transcode.py \
		--input-ts $< \
		--probe-json $@ \
		--output-mp4 $(CAPTURE_ANALYSIS_DIR)/$*.mp4 \
		--ffmpeg-bin $(FFMPEG) \
		--ffprobe-bin $(FFPROBE)

# For the wall targets, %.mp4 depends on %.probe.json (which itself builds
# the mp4 when probe passes). We restate the dependency so `make wall-20msps`
# always produces the final inspectable artifact if it can.
$(CAPTURE_ANALYSIS_DIR)/%.mp4: $(CAPTURE_ANALYSIS_DIR)/%.probe.json
	@$(PYTHON) -c "from pathlib import Path; target=Path(r'$@'); probe=Path(r'$(<)'); print(f'[transcode] {target} produced by probe step' if target.is_file() else f'[transcode] {target} not produced; see {probe}')"

$(CAPTURE_ANALYSIS_DIR)/%.visuals.json: $(RAW_CAPTURES_DIR)/%.ci8 $(RAW_CAPTURES_DIR)/%.ci8.json $(CAPTURE_ANALYSIS_DIR)/%.receiver.json $(CAPTURE_ANALYSIS_DIR)/%.probe.json $(PIPELINE_DIR)/visualize.py $(PYTHON_SOURCES) $(PIPELINE_SOURCES)
	$(PYTHON) $(PIPELINE_DIR)/visualize.py --capture $< --summary-json $(CAPTURE_ANALYSIS_DIR)/$*.visuals.json

$(CAPTURE_ANALYSIS_DIR)/%.gate_b.visual.json: $(CAPTURE_ANALYSIS_DIR)/%.visuals.json
	@$(PYTHON) -c "from pathlib import Path; import sys; target=Path(r'$@'); sys.exit(0 if target.exists() else 1)"

$(CAPTURE_ANALYSIS_DIR)/%.gate_c.visual.json: $(CAPTURE_ANALYSIS_DIR)/%.visuals.json
	@$(PYTHON) -c "from pathlib import Path; import sys; target=Path(r'$@'); sys.exit(0 if target.exists() else 1)"

$(CAPTURE_ANALYSIS_DIR)/%.gate_d.visual.json: $(CAPTURE_ANALYSIS_DIR)/%.visuals.json
	@$(PYTHON) -c "from pathlib import Path; import sys; target=Path(r'$@'); sys.exit(0 if target.exists() else 1)"

$(CAPTURE_ANALYSIS_DIR)/%.gate_e.visual.json: $(CAPTURE_ANALYSIS_DIR)/%.visuals.json
	@$(PYTHON) -c "from pathlib import Path; import sys; target=Path(r'$@'); sys.exit(0 if target.exists() else 1)"

$(CAPTURE_ANALYSIS_DIR)/%.gate_f.visual.json: $(CAPTURE_ANALYSIS_DIR)/%.visuals.json
	@$(PYTHON) -c "from pathlib import Path; import sys; target=Path(r'$@'); sys.exit(0 if target.exists() else 1)"

$(CAPTURE_ANALYSIS_DIR)/%.echo.visual.json: $(CAPTURE_ANALYSIS_DIR)/%.visuals.json
	@$(PYTHON) -c "from pathlib import Path; import sys; target=Path(r'$@'); sys.exit(0 if target.exists() else 1)"

clean-analysis:
	$(PYTHON) $(PIPELINE_DIR)/clean_analysis.py --analysis-dir $(CAPTURE_ANALYSIS_DIR)

clean-pipeline: clean-analysis

restructure-captures:
	$(PYTHON) $(PIPELINE_DIR)/restructure_captures.py --captures-dir $(CAPTURES_DIR) --execute

.SECONDARY:
