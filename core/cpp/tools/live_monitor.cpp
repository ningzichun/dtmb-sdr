#include "dtmb/core.hpp"

#include "binary_stdio.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef DTMB_CORE_HAVE_HACKRF
#include <hackrf.h>
#endif

namespace {

volatile std::sig_atomic_t stop_signal = 0;

void handle_stop_signal(int signal_number) {
    stop_signal = signal_number;
}

bool stop_requested() noexcept {
    return stop_signal != 0;
}

std::string_view signal_stop_reason() noexcept {
    if (stop_signal == SIGINT) {
        return "sigint";
    }
#ifdef SIGTERM
    if (stop_signal == SIGTERM) {
        return "sigterm";
    }
#endif
    return "signal";
}

class StopSignalSession {
public:
    StopSignalSession() {
        stop_signal = 0;
        previous_sigint_ = std::signal(SIGINT, handle_stop_signal);
#ifdef SIGTERM
        previous_sigterm_ = std::signal(SIGTERM, handle_stop_signal);
#endif
    }

    StopSignalSession(const StopSignalSession&) = delete;
    StopSignalSession& operator=(const StopSignalSession&) = delete;

    ~StopSignalSession() {
        if (previous_sigint_ != SIG_ERR) {
            (void) std::signal(SIGINT, previous_sigint_);
        }
#ifdef SIGTERM
        if (previous_sigterm_ != SIG_ERR) {
            (void) std::signal(SIGTERM, previous_sigterm_);
        }
#endif
    }

private:
    using SignalHandler = void (*)(int);

    SignalHandler previous_sigint_ = SIG_ERR;
#ifdef SIGTERM
    SignalHandler previous_sigterm_ = SIG_ERR;
#endif
};

struct Options {
    std::size_t sample_rate = dtmb::core::kDtmbSymbolRateSps;
    std::uint64_t center_frequency = 0;
    std::size_t bandwidth = 0;
    bool hackrf_source = false;
    std::string serial;
    int amp_enable = 0;
    std::uint32_t lna_gain = 16;
    std::uint32_t vga_gain = 16;
    std::size_t max_samples = 0;
    std::size_t chunk_samples = 1U << 18U;
    std::size_t report_every_samples = 0;
    std::size_t max_reports = 0;
    std::size_t fft_size = 2048;
    std::size_t spectrum_bins = 160;
    std::size_t pn_frames = 16;
    std::size_t pn_workers = 0;
    bool passthrough = false;
    std::string input_path = "-";
    std::string json_path;
};

struct SpectrumSummary {
    std::vector<float> dbfs;
    float peak_dbfs = 0.0F;
    std::size_t peak_bin = 0;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " --sample-rate N --center-frequency N [--bandwidth N]"
        << " [--hackrf] [--serial SERIAL] [--max-samples N]"
        << " [--amp 0|1] [--lna-gain DB] [--vga-gain DB]"
        << " [--chunk-samples N] [--report-every-samples N] [--max-reports N]"
        << " [--fft-size N] [--spectrum-bins N]"
        << " [--pn-frames N] [--pn-workers N]"
        << " [--passthrough] [--json-out PATH] [input.ci8|-]\n";
}

std::size_t parse_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size() || value == 0) {
        throw std::invalid_argument(std::string(field) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::size_t parse_nonnegative_size(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t parse_u64(const std::string& text, const char* field) {
    std::size_t parsed = 0;
    const auto value = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + field + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1U)) == 0;
}

std::istream& input_stream(
    const std::string& path,
    std::unique_ptr<std::ifstream>& file_holder) {
    if (path.empty() || path == "-") {
        return std::cin;
    }
    file_holder = std::make_unique<std::ifstream>(path, std::ios::binary);
    if (!*file_holder) {
        throw std::runtime_error("failed to open input: " + path);
    }
    return *file_holder;
}

std::ostream& json_stream(
    const std::string& path,
    std::unique_ptr<std::ofstream>& file_holder,
    bool passthrough) {
    if (!path.empty()) {
        file_holder = std::make_unique<std::ofstream>(path);
        if (!*file_holder) {
            throw std::runtime_error("failed to open JSON output: " + path);
        }
        return *file_holder;
    }
    return passthrough ? std::cerr : std::cout;
}

Options parse_args(int argc, char** argv) {
    Options options;
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto need_value = [&](const char* field) -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + field);
            }
            return argv[index];
        };
        if (arg == "--sample-rate") {
            options.sample_rate = parse_size(need_value("--sample-rate"), "sample rate");
        } else if (arg == "--center-frequency" || arg == "--frequency") {
            options.center_frequency = parse_u64(
                need_value(arg.c_str()),
                "center frequency");
        } else if (arg == "--bandwidth") {
            options.bandwidth = parse_size(need_value("--bandwidth"), "bandwidth");
        } else if (arg == "--hackrf") {
            options.hackrf_source = true;
        } else if (arg == "--source") {
            const auto source = need_value("--source");
            if (source == "hackrf") {
                options.hackrf_source = true;
            } else if (source == "stdin" || source == "-") {
                options.input_path = "-";
            } else if (source == "file") {
                // The positional path still selects the file.
            } else {
                throw std::invalid_argument("source must be hackrf, stdin, or file");
            }
        } else if (arg == "--serial") {
            options.serial = need_value("--serial");
        } else if (arg == "--amp") {
            options.amp_enable = static_cast<int>(
                parse_nonnegative_size(need_value("--amp"), "amp"));
        } else if (arg == "--lna-gain") {
            options.lna_gain = static_cast<std::uint32_t>(
                parse_nonnegative_size(need_value("--lna-gain"), "LNA gain"));
        } else if (arg == "--vga-gain") {
            options.vga_gain = static_cast<std::uint32_t>(
                parse_nonnegative_size(need_value("--vga-gain"), "VGA gain"));
        } else if (arg == "--max-samples") {
            options.max_samples = parse_nonnegative_size(
                need_value("--max-samples"),
                "max samples");
        } else if (arg == "--chunk-samples") {
            options.chunk_samples = parse_size(need_value("--chunk-samples"), "chunk samples");
        } else if (arg == "--report-every-samples") {
            options.report_every_samples = parse_size(
                need_value("--report-every-samples"),
                "report interval");
        } else if (arg == "--max-reports") {
            options.max_reports = parse_size(need_value("--max-reports"), "max reports");
        } else if (arg == "--fft-size") {
            options.fft_size = parse_size(need_value("--fft-size"), "FFT size");
        } else if (arg == "--spectrum-bins") {
            options.spectrum_bins = parse_size(need_value("--spectrum-bins"), "spectrum bins");
        } else if (arg == "--pn-frames") {
            options.pn_frames = parse_size(need_value("--pn-frames"), "PN frame count");
        } else if (arg == "--pn-workers") {
            options.pn_workers = parse_nonnegative_size(
                need_value("--pn-workers"),
                "PN worker count");
        } else if (arg == "--passthrough") {
            options.passthrough = true;
        } else if (arg == "--json-out") {
            options.json_path = need_value("--json-out");
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() > 1) {
        throw std::invalid_argument("too many positional inputs");
    }
    if (!positional.empty()) {
        options.input_path = positional.front();
    }
    if (options.hackrf_source && !positional.empty()) {
        throw std::invalid_argument("direct HackRF source does not accept an input path");
    }
    if (options.center_frequency == 0) {
        throw std::invalid_argument("--center-frequency is required");
    }
    if (options.amp_enable != 0 && options.amp_enable != 1) {
        throw std::invalid_argument("--amp must be 0 or 1");
    }
    if (options.lna_gain > 40 || (options.lna_gain % 8U) != 0) {
        throw std::invalid_argument("--lna-gain must be 0..40 dB in 8 dB steps");
    }
    if (options.vga_gain > 62 || (options.vga_gain % 2U) != 0) {
        throw std::invalid_argument("--vga-gain must be 0..62 dB in 2 dB steps");
    }
    if (!is_power_of_two(options.fft_size)) {
        throw std::invalid_argument("--fft-size must be a power of two");
    }
    if (options.spectrum_bins > options.fft_size) {
        throw std::invalid_argument("--spectrum-bins must be <= --fft-size");
    }
    if (options.report_every_samples == 0) {
        options.report_every_samples = std::max(options.chunk_samples, options.sample_rate / 4U);
    }
    return options;
}

void write_all(std::ostream& output, std::span<const std::int8_t> bytes) {
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("failed to write CI8 passthrough stream");
    }
}

void append_ci8_as_cf32(std::span<const std::int8_t> ci8, std::vector<float>& cf32) {
    cf32.resize(ci8.size());
    for (std::size_t index = 0; index < ci8.size(); ++index) {
        cf32[index] = static_cast<float>(ci8[index]);
    }
}

void append_tail(
    std::vector<float>& destination,
    std::span<const float> source,
    std::size_t max_complex_samples) {
    destination.insert(destination.end(), source.begin(), source.end());
    const auto max_values = max_complex_samples * 2U;
    if (destination.size() > max_values) {
        const auto erase_values = destination.size() - max_values;
        destination.erase(destination.begin(), destination.begin() + static_cast<std::ptrdiff_t>(erase_values));
    }
}

std::string iso_utc_now() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

float linear_to_dbfs(double power) {
    constexpr double reference_power = 128.0 * 128.0;
    return static_cast<float>(10.0 * std::log10(std::max(power / reference_power, 1.0e-18)));
}

SpectrumSummary summarize_spectrum(
    std::span<const float> raw_cf32,
    std::size_t fft_size,
    std::size_t spectrum_bins) {
    if (raw_cf32.size() < fft_size * 2U) {
        return {};
    }
    const auto sample_offset = raw_cf32.size() / 2U - fft_size;
    std::vector<float> windowed(fft_size * 2U, 0.0F);
    double window_power = 0.0;
    for (std::size_t sample = 0; sample < fft_size; ++sample) {
        const auto phase = 2.0 * std::numbers::pi_v<double>
            * static_cast<double>(sample)
            / static_cast<double>(fft_size - 1U);
        const auto window = 0.5 - 0.5 * std::cos(phase);
        window_power += window * window;
        windowed[sample * 2U] =
            static_cast<float>(static_cast<double>(raw_cf32[(sample_offset + sample) * 2U]) * window);
        windowed[sample * 2U + 1U] =
            static_cast<float>(static_cast<double>(raw_cf32[(sample_offset + sample) * 2U + 1U]) * window);
    }
    std::vector<float> fft(fft_size * 2U, 0.0F);
    dtmb::core::mixed_radix_fft_forward_cf32(windowed, fft);

    std::vector<double> shifted_power(fft_size, 0.0);
    for (std::size_t bin = 0; bin < fft_size; ++bin) {
        const auto src = (bin + fft_size / 2U) % fft_size;
        const auto real = static_cast<double>(fft[src * 2U]);
        const auto imag = static_cast<double>(fft[src * 2U + 1U]);
        shifted_power[bin] = (real * real + imag * imag) / std::max(window_power, 1.0);
    }

    SpectrumSummary summary;
    summary.dbfs.resize(spectrum_bins);
    summary.peak_dbfs = -180.0F;
    for (std::size_t out_bin = 0; out_bin < spectrum_bins; ++out_bin) {
        const auto first = out_bin * fft_size / spectrum_bins;
        const auto last = std::max(first + 1U, (out_bin + 1U) * fft_size / spectrum_bins);
        double power_sum = 0.0;
        for (std::size_t bin = first; bin < last; ++bin) {
            power_sum += shifted_power[bin];
        }
        const auto dbfs = linear_to_dbfs(power_sum / static_cast<double>(last - first));
        summary.dbfs[out_bin] = dbfs;
        if (dbfs > summary.peak_dbfs) {
            summary.peak_dbfs = dbfs;
            summary.peak_bin = out_bin;
        }
    }
    return summary;
}

void write_float_array(std::ostream& output, std::span<const float> values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << std::fixed << std::setprecision(3) << values[index];
    }
    output << ']';
}

void write_json_string(std::ostream& output, std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    output << '"';
    for (const auto character : value) {
        const auto byte = static_cast<unsigned char>(character);
        switch (byte) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (byte < 0x20U) {
                output << "\\u00" << hex[byte >> 4U] << hex[byte & 0x0FU];
            } else {
                output.put(static_cast<char>(byte));
            }
            break;
        }
    }
    output << '"';
}

std::size_t pn_keep_samples(std::size_t pn_frames) {
    return (pn_frames + 1U) * dtmb::core::kPn945FrameSymbols
        + dtmb::core::kPn945HeaderSymbols;
}

std::optional<dtmb::core::Pn945AcquisitionResult> acquire_pn(
    std::span<const float> symbol_cf32,
    std::size_t pn_frames,
    std::size_t pn_workers) {
    if (symbol_cf32.size() / 2U < pn_keep_samples(pn_frames)) {
        return std::nullopt;
    }
    return dtmb::core::acquire_pn945_cf32(
        symbol_cf32,
        dtmb::core::Pn945AcquisitionOptions{pn_frames, 0.35F, pn_workers});
}

void write_report(
    std::ostream& output,
    const Options& options,
    std::size_t report_index,
    std::size_t total_samples,
    const dtmb::core::Ci8PowerStats& stats,
    const SpectrumSummary& spectrum,
    const std::optional<dtmb::core::Pn945AcquisitionResult>& pn);

struct ProcessingState {
    explicit ProcessingState(const Options& in_options)
        : options(in_options),
          up(compute_up_factor(in_options)),
          down(compute_down_factor(in_options)),
          taps(dtmb::core::square_root_raised_cosine_taps(
              8,
              std::max(up, down),
              0.05F)),
          resampler(up, down, taps, 0, 16'384),
          next_report_sample(default_next_report(in_options)) {
        raw_cf32.reserve(options.chunk_samples * 2U);
        spectrum_window.reserve(options.fft_size * 2U);
        pn_window.reserve(pn_keep_samples(options.pn_frames) * 2U);
    }

    static std::size_t compute_up_factor(const Options& options) {
        const auto gcd = std::gcd(options.sample_rate, dtmb::core::kDtmbSymbolRateSps);
        return dtmb::core::kDtmbSymbolRateSps / gcd;
    }

    static std::size_t compute_down_factor(const Options& options) {
        const auto gcd = std::gcd(options.sample_rate, dtmb::core::kDtmbSymbolRateSps);
        return options.sample_rate / gcd;
    }

    static std::size_t default_next_report(const Options& options) {
        return options.report_every_samples;
    }

    const Options& options;
    const std::size_t up;
    const std::size_t down;
    const std::vector<float> taps;
    dtmb::core::RationalResamplerCf32 resampler;
    std::vector<float> raw_cf32;
    std::vector<float> spectrum_window;
    std::vector<float> pn_window;
    std::vector<float> resampled;
    std::size_t total_samples = 0;
    std::uint64_t total_clip_count_i = 0;
    std::uint64_t total_clip_count_q = 0;
    double total_i2q2_sum = 0.0;
    std::size_t next_report_sample = 0;
    std::size_t reports = 0;
};

bool process_ci8_chunk(
    ProcessingState& state,
    std::ostream& json,
    std::span<const std::int8_t> values,
    std::ostream* passthrough_output) {
    if (values.empty()) {
        return true;
    }
    if ((values.size() % 2U) != 0) {
        throw std::runtime_error("CI8 input byte count is not a whole number of samples");
    }
    if (state.options.max_samples != 0) {
        if (state.total_samples >= state.options.max_samples) {
            return false;
        }
        const auto remaining_samples = state.options.max_samples - state.total_samples;
        const auto remaining_values = remaining_samples * 2U;
        if (values.size() > remaining_values) {
            values = values.first(remaining_values);
        }
    }

    if (passthrough_output != nullptr) {
        write_all(*passthrough_output, values);
    }
    append_ci8_as_cf32(values, state.raw_cf32);
    append_tail(state.spectrum_window, state.raw_cf32, state.options.fft_size);
    state.resampled.clear();
    if (state.up == state.down) {
        append_tail(state.pn_window, state.raw_cf32, pn_keep_samples(state.options.pn_frames));
    } else {
        state.resampler.process(state.raw_cf32, state.resampled);
        append_tail(state.pn_window, state.resampled, pn_keep_samples(state.options.pn_frames));
    }

    const auto stats = dtmb::core::ci8_power_stats(values);
    state.total_samples += stats.sample_count;
    state.total_clip_count_i += stats.clip_count_i;
    state.total_clip_count_q += stats.clip_count_q;
    state.total_i2q2_sum += stats.mean_i2q2 * static_cast<double>(stats.sample_count);
    const auto should_report = state.total_samples >= state.next_report_sample
        || (state.options.max_reports != 0 && state.reports == 0);
    if (should_report) {
        const auto spectrum = summarize_spectrum(
            state.spectrum_window,
            state.options.fft_size,
            state.options.spectrum_bins);
        const auto pn = acquire_pn(
            state.pn_window,
            state.options.pn_frames,
            state.options.pn_workers);
        write_report(
            json,
            state.options,
            state.reports,
            state.total_samples,
            stats,
            spectrum,
            pn);
        ++state.reports;
        state.next_report_sample = state.total_samples + state.options.report_every_samples;
        if (state.options.max_reports != 0 && state.reports >= state.options.max_reports) {
            return false;
        }
    }
    return state.options.max_samples == 0 || state.total_samples < state.options.max_samples;
}

void write_report(
    std::ostream& output,
    const Options& options,
    std::size_t report_index,
    std::size_t total_samples,
    const dtmb::core::Ci8PowerStats& stats,
    const SpectrumSummary& spectrum,
    const std::optional<dtmb::core::Pn945AcquisitionResult>& pn) {
    const auto sample_rate = static_cast<double>(options.sample_rate);
    const auto center = static_cast<double>(options.center_frequency);
    const auto bin_width = sample_rate / static_cast<double>(std::max<std::size_t>(spectrum.dbfs.size(), 1));
    const auto start_hz = center - sample_rate * 0.5;
    const auto peak_hz = start_hz + (static_cast<double>(spectrum.peak_bin) + 0.5) * bin_width;
    const auto elapsed_ms = static_cast<double>(total_samples) * 1000.0 / sample_rate;
    const auto clip_count = stats.clip_count_i + stats.clip_count_q;
    const auto clip_ratio = stats.sample_count == 0
        ? 0.0
        : static_cast<double>(clip_count) / static_cast<double>(stats.sample_count * 2U);
    const auto rms_dbfs = linear_to_dbfs(stats.mean_i2q2);

    output << std::setprecision(10)
           << "{\"schema\":\"dtmb.live_monitor.v1\""
           << ",\"created_utc\":\"" << iso_utc_now() << "\""
           << ",\"report_index\":" << report_index
           << ",\"elapsed_ms\":" << std::fixed << std::setprecision(3) << elapsed_ms
           << ",\"input_samples\":" << total_samples
           << ",\"sample_rate_sps\":" << options.sample_rate
           << ",\"sample_rate_msps\":" << std::fixed << std::setprecision(6)
           << (sample_rate / 1.0e6)
           << ",\"center_frequency_hz\":" << options.center_frequency
           << ",\"center_frequency_mhz\":" << std::fixed << std::setprecision(6)
           << (center / 1.0e6)
           << ",\"bandwidth_hz\":" << options.bandwidth
           << ",\"bandwidth_mhz\":" << std::fixed << std::setprecision(6)
           << (static_cast<double>(options.bandwidth) / 1.0e6)
           << ",\"iq\":{\"rms_dbfs\":" << std::fixed << std::setprecision(3) << rms_dbfs
           << ",\"clip_count_i\":" << stats.clip_count_i
           << ",\"clip_count_q\":" << stats.clip_count_q
           << ",\"clip_ratio\":" << std::fixed << std::setprecision(8) << clip_ratio
           << "},\"spectrum\":{\"fft_size\":" << options.fft_size
           << ",\"bin_count\":" << spectrum.dbfs.size()
           << ",\"start_hz\":" << std::fixed << std::setprecision(3) << start_hz
           << ",\"bin_width_hz\":" << std::fixed << std::setprecision(3) << bin_width
           << ",\"peak_frequency_hz\":" << std::fixed << std::setprecision(3) << peak_hz
           << ",\"peak_dbfs\":" << std::fixed << std::setprecision(3) << spectrum.peak_dbfs
           << ",\"power_dbfs\":";
    write_float_array(output, spectrum.dbfs);
    output << "},\"pn945\":";
    if (pn.has_value()) {
        output << "{\"available\":true"
               << ",\"symbol_rate_sps\":" << dtmb::core::kDtmbSymbolRateSps
               << ",\"phase_offset_symbols\":" << pn->phase_offset
               << ",\"mean_metric\":" << std::fixed << std::setprecision(6) << pn->mean_metric
               << ",\"max_metric\":" << std::fixed << std::setprecision(6) << pn->max_metric
               << ",\"hit_count\":" << pn->hit_count
               << ",\"observed_frames\":" << pn->observed_frames
               << ",\"coarse_cfo_hz\":" << std::fixed << std::setprecision(3) << pn->coarse_cfo_hz
               << ",\"coarse_cfo_valid\":" << (pn->coarse_cfo_valid ? "true" : "false")
               << "}";
    } else {
        output << "{\"available\":false"
               << ",\"symbol_rate_sps\":" << dtmb::core::kDtmbSymbolRateSps
               << ",\"reason\":\"insufficient_symbol_window\"}";
    }
    output << "}\n";
    output.flush();
}

void write_end_report(
    std::ostream& output,
    const ProcessingState& state,
    std::string_view source,
    std::string_view stop_reason,
    std::size_t dropped_bytes = 0,
    bool input_complete = true,
    std::string_view error_message = {}) {
    input_complete = input_complete && dropped_bytes == 0;
    const auto clip_count = state.total_clip_count_i + state.total_clip_count_q;
    const auto clip_ratio = state.total_samples == 0
        ? 0.0
        : static_cast<double>(clip_count) / static_cast<double>(state.total_samples * 2U);
    const auto rms_dbfs = state.total_samples == 0
        ? -180.0
        : linear_to_dbfs(state.total_i2q2_sum / static_cast<double>(state.total_samples));
    const bool ok = error_message.empty()
        && input_complete
        && state.total_samples != 0
        && state.reports != 0
        && clip_ratio <= 0.005;
    const std::string_view verdict = !error_message.empty()
        ? "error"
        : ok ? "ok" : "degraded";

    output << "{\"schema\":\"dtmb.live_monitor.end.v1\""
           << ",\"event\":\"end\""
           << ",\"verdict\":\"" << verdict << '"'
           << ",\"source\":\"" << source << '"'
           << ",\"stop_reason\":\"" << stop_reason << '"'
           << ",\"input_complete\":" << (input_complete ? "true" : "false")
           << ",\"total_samples\":" << state.total_samples
           << ",\"reports\":" << state.reports
           << ",\"dropped_bytes\":" << dropped_bytes
           << ",\"max_samples_requested\":" << state.options.max_samples
           << ",\"max_samples_reached\":"
           << (
               state.options.max_samples != 0
                   && state.total_samples >= state.options.max_samples
               ? "true"
               : "false")
           << ",\"iq\":{\"rms_dbfs\":" << std::fixed << std::setprecision(3) << rms_dbfs
           << ",\"clip_count_i\":" << state.total_clip_count_i
           << ",\"clip_count_q\":" << state.total_clip_count_q
           << ",\"clip_ratio\":" << std::fixed << std::setprecision(8) << clip_ratio
           << ",\"clip_ratio_max_allowed\":0.00500000}";
    if (!error_message.empty()) {
        output << ",\"error\":";
        write_json_string(output, error_message);
    }
    output << "}\n";
    output.flush();
}

#ifdef DTMB_CORE_HAVE_HACKRF

struct HackrfRxQueue {
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::vector<std::int8_t>> chunks;
    std::size_t queued_bytes = 0;
    std::size_t dropped_bytes = 0;
    std::size_t max_queued_bytes = 64U * 1024U * 1024U;
    std::atomic<bool> stop_requested = false;
    std::atomic<std::size_t> samples_seen = 0;
    bool finished = false;
    std::size_t max_samples = 0;
};

void check_hackrf_result(int result, const char* call_name) {
    if (result == HACKRF_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << call_name << " failed: "
            << hackrf_error_name(static_cast<hackrf_error>(result))
            << " (" << result << ")";
    throw std::runtime_error(message.str());
}

int hackrf_rx_callback(hackrf_transfer* transfer) {
    auto* queue = static_cast<HackrfRxQueue*>(transfer->rx_ctx);
    if (queue == nullptr || queue->stop_requested.load(std::memory_order_relaxed)) {
        return 1;
    }
    auto bytes_to_copy = static_cast<std::size_t>(std::max(transfer->valid_length, 0));
    bytes_to_copy -= bytes_to_copy % 2U;
    if (bytes_to_copy == 0) {
        return 0;
    }

    bool should_stop = false;
    if (queue->max_samples != 0) {
        const auto transfer_samples = bytes_to_copy / 2U;
        const auto prior_samples = queue->samples_seen.fetch_add(
            transfer_samples,
            std::memory_order_relaxed);
        if (prior_samples >= queue->max_samples) {
            queue->stop_requested.store(true, std::memory_order_relaxed);
            return 1;
        }
        const auto allowed_samples = queue->max_samples - prior_samples;
        if (allowed_samples < transfer_samples) {
            bytes_to_copy = allowed_samples * 2U;
            should_stop = true;
        } else if (allowed_samples == transfer_samples) {
            should_stop = true;
        }
    } else {
        queue->samples_seen.fetch_add(bytes_to_copy / 2U, std::memory_order_relaxed);
    }

    std::vector<std::int8_t> chunk(bytes_to_copy);
    std::memcpy(chunk.data(), transfer->buffer, bytes_to_copy);
    {
        std::lock_guard<std::mutex> lock(queue->mutex);
        while (queue->queued_bytes + chunk.size() > queue->max_queued_bytes
               && !queue->chunks.empty()) {
            queue->dropped_bytes += queue->chunks.front().size();
            queue->queued_bytes -= queue->chunks.front().size();
            queue->chunks.pop_front();
        }
        queue->queued_bytes += chunk.size();
        queue->chunks.push_back(std::move(chunk));
        if (should_stop) {
            queue->finished = true;
            queue->stop_requested.store(true, std::memory_order_relaxed);
        }
    }
    queue->condition.notify_one();
    return should_stop ? 1 : 0;
}

class HackrfLibrarySession {
public:
    HackrfLibrarySession() {
        check_hackrf_result(hackrf_init(), "hackrf_init");
    }

    HackrfLibrarySession(const HackrfLibrarySession&) = delete;
    HackrfLibrarySession& operator=(const HackrfLibrarySession&) = delete;

    ~HackrfLibrarySession() {
        (void) hackrf_exit();
    }
};

class HackrfDeviceHandle {
public:
    explicit HackrfDeviceHandle(const std::string& serial) {
        int result = HACKRF_SUCCESS;
        if (serial.empty()) {
            result = hackrf_open(&device_);
        } else {
            result = hackrf_open_by_serial(serial.c_str(), &device_);
        }
        check_hackrf_result(result, serial.empty() ? "hackrf_open" : "hackrf_open_by_serial");
    }

    HackrfDeviceHandle(const HackrfDeviceHandle&) = delete;
    HackrfDeviceHandle& operator=(const HackrfDeviceHandle&) = delete;

    ~HackrfDeviceHandle() {
        if (device_ != nullptr) {
            (void) hackrf_close(device_);
        }
    }

    [[nodiscard]] hackrf_device* get() const noexcept {
        return device_;
    }

private:
    hackrf_device* device_ = nullptr;
};

void run_hackrf_source(const Options& options, std::ostream& json) {
    HackrfLibrarySession library;
    HackrfDeviceHandle device(options.serial);
    check_hackrf_result(
        hackrf_set_sample_rate(device.get(), static_cast<double>(options.sample_rate)),
        "hackrf_set_sample_rate");
    if (options.bandwidth != 0) {
        check_hackrf_result(
            hackrf_set_baseband_filter_bandwidth(
                device.get(),
                static_cast<std::uint32_t>(options.bandwidth)),
            "hackrf_set_baseband_filter_bandwidth");
    }
    check_hackrf_result(
        hackrf_set_freq(device.get(), options.center_frequency),
        "hackrf_set_freq");
    check_hackrf_result(
        hackrf_set_amp_enable(device.get(), static_cast<std::uint8_t>(options.amp_enable)),
        "hackrf_set_amp_enable");
    check_hackrf_result(
        hackrf_set_lna_gain(device.get(), options.lna_gain),
        "hackrf_set_lna_gain");
    check_hackrf_result(
        hackrf_set_vga_gain(device.get(), options.vga_gain),
        "hackrf_set_vga_gain");

    HackrfRxQueue queue;
    queue.max_samples = options.max_samples;
    ProcessingState state(options);

    check_hackrf_result(
        hackrf_start_rx(device.get(), hackrf_rx_callback, &queue),
        "hackrf_start_rx");

    bool stopped = false;
    try {
        while (!stop_requested()) {
            std::vector<std::int8_t> chunk;
            {
                std::unique_lock<std::mutex> lock(queue.mutex);
                queue.condition.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return !queue.chunks.empty()
                        || queue.finished
                        || queue.stop_requested.load(std::memory_order_relaxed);
                });
                if (stop_requested()) {
                    queue.stop_requested.store(true, std::memory_order_relaxed);
                    break;
                }
                if (queue.chunks.empty()) {
                    break;
                }
                chunk = std::move(queue.chunks.front());
                queue.queued_bytes -= chunk.size();
                queue.chunks.pop_front();
            }
            if (!process_ci8_chunk(state, json, chunk, nullptr)) {
                queue.stop_requested.store(true, std::memory_order_relaxed);
                break;
            }
        }
    } catch (...) {
        queue.stop_requested.store(true, std::memory_order_relaxed);
        check_hackrf_result(hackrf_stop_rx(device.get()), "hackrf_stop_rx");
        stopped = true;
        throw;
    }
    if (!stopped) {
        check_hackrf_result(hackrf_stop_rx(device.get()), "hackrf_stop_rx");
    }
    if (queue.dropped_bytes != 0) {
        std::cerr << "dtmb_core_live_monitor: dropped "
                  << queue.dropped_bytes
                  << " queued HackRF bytes while producing telemetry\n";
    }
    const auto stop_reason = stop_requested()
        ? signal_stop_reason()
        : (options.max_reports != 0 && state.reports >= options.max_reports)
            ? std::string_view("max_reports")
            : (options.max_samples != 0 && state.total_samples >= options.max_samples)
                ? std::string_view("max_samples")
                : std::string_view("source_complete");
    write_end_report(json, state, "hackrf", stop_reason, queue.dropped_bytes);
}

#else

void run_hackrf_source(const Options&, std::ostream&) {
    throw std::runtime_error(
        "direct HackRF source is not available; rebuild with DTMB_CORE_ENABLE_HACKRF=ON and libusb");
}

#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_args(argc, argv);
        std::unique_ptr<std::ifstream> input_file;
        std::unique_ptr<std::ofstream> json_file;
        auto& json = json_stream(options.json_path, json_file, options.passthrough);
        StopSignalSession signal_session;

        if (options.hackrf_source) {
            run_hackrf_source(options, json);
            return 0;
        }

        dtmb::tools::configure_binary_stdio(options.input_path == "-", options.passthrough);
        auto& input = input_stream(options.input_path, input_file);
        ProcessingState state(options);
        std::vector<std::int8_t> ci8(options.chunk_samples * 2U);
        std::size_t dropped_bytes = 0;

        try {
            while (!stop_requested()) {
                input.read(
                    reinterpret_cast<char*>(ci8.data()),
                    static_cast<std::streamsize>(ci8.size()));
                const auto bytes_read = static_cast<std::size_t>(input.gcount());
                if (bytes_read == 0) {
                    if (stop_requested()) {
                        break;
                    }
                    if (input.bad()) {
                        throw std::runtime_error("failed to read CI8 input");
                    }
                    break;
                }
                if ((bytes_read % 2U) != 0) {
                    dropped_bytes = 1;
                    throw std::runtime_error(
                        "CI8 input byte count is not a whole number of samples");
                }
                const auto values = std::span<const std::int8_t>(ci8.data(), bytes_read);
                if (!process_ci8_chunk(
                        state,
                        json,
                        values,
                        options.passthrough ? &std::cout : nullptr)) {
                    break;
                }
                if (bytes_read < ci8.size()) {
                    if (input.bad()) {
                        throw std::runtime_error("failed to read CI8 input");
                    }
                    break;
                }
            }
        } catch (const std::exception& exc) {
            write_end_report(
                json,
                state,
                options.input_path == "-" ? "stdin" : "file",
                "input_error",
                dropped_bytes,
                false,
                exc.what());
            throw;
        }
        const auto stop_reason = stop_requested()
            ? signal_stop_reason()
            : (options.max_reports != 0 && state.reports >= options.max_reports)
                ? std::string_view("max_reports")
                : (options.max_samples != 0 && state.total_samples >= options.max_samples)
                    ? std::string_view("max_samples")
                    : std::string_view("eof");
        write_end_report(
            json,
            state,
            options.input_path == "-" ? "stdin" : "file",
            stop_reason,
            dropped_bytes);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "dtmb_core_live_monitor: " << exc.what() << '\n';
        return 1;
    }
}
