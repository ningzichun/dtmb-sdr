#include "live_view_host_renderer.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

[[noreturn]] void fail(std::string_view message) {
    throw std::runtime_error(std::string(message));
}

void check(bool condition, std::string_view message) {
    if (!condition) {
        fail(message);
    }
}

void check_contains(std::string_view text, std::string_view expected,
                    std::string_view message) {
    check(text.find(expected) != std::string_view::npos, message);
}

void replace_once(std::string &text, std::string_view from,
                  std::string_view to) {
    const auto offset = text.find(from);
    check(offset != std::string::npos, "test mutation source not found");
    text.replace(offset, from.size(), to);
}

std::string fixture() {
    std::ifstream input(DTMB_LIVE_VIEW_HOST_FIXTURE);
    check(static_cast<bool>(input), "failed to open live-view-host fixture");
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
    };
}

std::string valid_scene_line() {
    std::istringstream lines(fixture());
    std::string line;
    while (std::getline(lines, line)) {
        if (line.find(R"("schema":"dtmb.live_view.scene.v1")") !=
                std::string::npos &&
            line.find(R"("revision":2)") != std::string::npos) {
            return line;
        }
    }
    fail("valid scene line not found in fixture");
}

std::string end_line(std::string_view verdict = "ok",
                     bool input_complete = true) {
    return "{\"schema\":\"dtmb.live_view.scene_end.v1\",\"event\":\"end\","
           "\"verdict\":\"" +
           std::string(verdict) + "\",\"input_complete\":" +
           (input_complete ? "true" : "false") + "}";
}

std::string stream_with_end(std::string_view body,
                            std::string_view verdict = "ok",
                            bool input_complete = true) {
    return std::string(body) + "\n" + end_line(verdict, input_complete) + "\n";
}

struct Result {
    int status = -1;
    std::string frames;
    std::string diagnostics;
};

Result run(std::string_view input_text,
           const dtmb::tools::live_view_host::Options &options = {}) {
    std::istringstream input{std::string(input_text)};
    std::ostringstream frames;
    std::ostringstream diagnostics;
    const int status =
        dtmb::tools::live_view_host::run(input, frames, diagnostics, options);
    return {status, frames.str(), diagnostics.str()};
}

void check_single_frame(const Result &result) {
    const std::string header = "P6\n16 12\n255\n";
    check(result.frames.starts_with(header), "missing PPM frame header");
    check(result.frames.size() == header.size() + 16U * 12U * 3U,
          "unexpected PPM frame size");
}

void test_fixture_recovers_and_renders_visible_panel() {
    const auto result = run(fixture());
    check(result.status == 1, "fixture must propagate degraded status");
    check_single_frame(result);

    const std::string header = "P6\n16 12\n255\n";
    const auto pixel = [&](std::size_t x, std::size_t y, std::size_t channel) {
        return static_cast<unsigned char>(
            result.frames[header.size() + (y * 16U + x) * 3U + channel]);
    };
    check(pixel(15, 0, 0) == 255U, "expected red spectrum background");
    check(pixel(15, 0, 1) == 0U, "unexpected green spectrum channel");
    check(pixel(15, 0, 2) == 0U, "unexpected blue spectrum channel");
    check(pixel(8, 10, 0) == 0U && pixel(8, 10, 1) == 0U &&
              pixel(8, 10, 2) == 0U,
          "hidden panel must remain black");

    check_contains(result.diagnostics,
                   R"("schema":"dtmb.live_view.host_error.v1")",
                   "fixture malformed input must emit host error");
    check_contains(result.diagnostics,
                   R"("schema":"dtmb.live_view.host_frame.v1")",
                   "fixture must emit frame diagnostic");
    check_contains(result.diagnostics, R"("revision":2)",
                   "fixture revision missing");
    check_contains(result.diagnostics,
                   R"("schema":"dtmb.live_view.host_end.v1")",
                   "fixture must emit terminal diagnostic");
    check_contains(result.diagnostics, R"("verdict":"degraded")",
                   "fixture terminal verdict must be degraded");
    check_contains(result.diagnostics, R"("frame_count":1)",
                   "fixture frame count mismatch");
    check_contains(result.diagnostics, R"("no_draw_packet_count":1)",
                   "fixture no-draw count mismatch");
}

void test_control_only_scene_propagates_degraded_terminal() {
    auto input_text = fixture();
    input_text.erase(0, input_text.find('\n') + 1);
    input_text.erase(0, input_text.find('\n') + 1);
    const auto result = run(input_text);

    check(result.status == 1, "degraded control scene must return nonzero");
    check_single_frame(result);
    check_contains(
        result.diagnostics,
        R"("schema":"dtmb.live_view.host_end.v1","event":"end","verdict":"degraded")",
        "degraded control scene must propagate to terminal verdict");
    check_contains(result.diagnostics, R"("no_draw_packet_count":1)",
                   "control-only scene count mismatch");
}

void test_bad_scene_is_transactional_and_later_scene_renders() {
    const auto valid = valid_scene_line();
    auto invalid = valid;
    replace_once(invalid, R"("shader":"solid_color_gles2")",
                 R"("shader":"unsupported")");

    const auto result = run(stream_with_end(invalid + "\n" + valid));
    check(result.status == 1, "recoverable scene error must return degraded");
    check_single_frame(result);
    check_contains(result.diagnostics, "unsupported shader",
                   "unsupported shader diagnostic missing");
    check_contains(result.diagnostics, R"("frame_count":1)",
                   "transactional render frame count mismatch");
}

void test_menu_control_changes_host_frame() {
    const auto closed_scene = valid_scene_line();
    auto open_scene = closed_scene;
    replace_once(open_scene, R"("open":false)", R"("open":true)");

    const auto closed = run(stream_with_end(closed_scene));
    const auto open = run(stream_with_end(open_scene));
    check(closed.status == 0 && open.status == 0,
          "clean menu scenes must return success");
    check(closed.frames.size() == open.frames.size(),
          "menu frame sizes must match");
    check(closed.frames != open.frames, "menu overlay must change host frame");
}

void test_fail_fast_emits_no_partial_frame() {
    dtmb::tools::live_view_host::Options options;
    options.fail_fast = true;
    const auto result = run(fixture(), options);

    check(result.status == 2, "fail-fast malformed input must return error");
    check(result.frames.empty(), "fail-fast must emit no partial frame");
    check_contains(result.diagnostics,
                   R"("schema":"dtmb.live_view.host_error.v1")",
                   "fail-fast error diagnostic missing");
    check_contains(result.diagnostics,
                   R"("schema":"dtmb.live_view.host_end.v1")",
                   "fail-fast terminal diagnostic missing");
    check_contains(result.diagnostics, R"("input_complete":false)",
                   "fail-fast must mark input incomplete");
    check_contains(result.diagnostics, R"("frame_count":0)",
                   "fail-fast frame count mismatch");
}

void test_scene_end_protocol_is_enforced() {
    const auto valid = valid_scene_line();

    const auto missing = run(valid + "\n");
    check(missing.status == 2, "missing scene_end must return error");
    check_contains(missing.diagnostics, "missing dtmb.live_view.scene_end.v1",
                   "missing scene_end diagnostic missing");
    check_contains(missing.diagnostics, R"("input_complete":false)",
                   "missing scene_end must mark input incomplete");

    const auto duplicate =
        run(valid + "\n" + end_line() + "\n" + end_line() + "\n");
    check(duplicate.status == 2, "duplicate scene_end must return error");
    check_contains(duplicate.diagnostics, "duplicate scene_end packet",
                   "duplicate scene_end diagnostic missing");

    const auto post_end =
        run(valid + "\n" + end_line() + "\n" + valid + "\n");
    check(post_end.status == 2, "post-end scene must return error");
    check_contains(post_end.diagnostics, "packet received after scene_end",
                   "post-end packet diagnostic missing");
    check_contains(post_end.diagnostics, R"("frame_count":1)",
                   "post-end scene must not render");

    auto bad_event = end_line();
    replace_once(bad_event, R"("event":"end")", R"("event":"scene")");
    const auto invalid_event = run(valid + "\n" + bad_event + "\n");
    check(invalid_event.status == 2, "invalid scene_end event must fail");
    check_contains(invalid_event.diagnostics, "unsupported event",
                   "invalid scene_end event diagnostic missing");

    auto missing_complete = end_line();
    replace_once(missing_complete, R"(,"input_complete":true)", "");
    const auto invalid_complete = run(valid + "\n" + missing_complete + "\n");
    check(invalid_complete.status == 2,
          "scene_end without input_complete must fail");
    check_contains(invalid_complete.diagnostics, "missing field: input_complete",
                   "missing input_complete diagnostic missing");
}

void test_terminal_verdict_and_exit_status_propagate() {
    const auto valid = valid_scene_line();
    const auto degraded = run(stream_with_end(valid, "degraded"));
    check(degraded.status == 1, "degraded scene_end must return status 1");
    check_contains(degraded.diagnostics, R"("verdict":"degraded")",
                   "degraded scene_end verdict missing");

    const auto error = run(stream_with_end(valid, "error"));
    check(error.status == 2, "error scene_end must return status 2");
    check_contains(error.diagnostics,
                   R"("schema":"dtmb.live_view.host_end.v1","event":"end","verdict":"error")",
                   "error scene_end verdict missing");

    const auto incomplete = run(stream_with_end(valid, "ok", false));
    check(incomplete.status == 2, "incomplete scene_end must return status 2");
    check_contains(incomplete.diagnostics, R"("input_complete":false)",
                   "incomplete scene_end flag missing");
}

void test_contract_and_unsigned_fields_are_validated() {
    const auto valid = valid_scene_line();

    auto coordinate = valid;
    replace_once(coordinate, R"("coordinate_system":"normalized top-left panels")",
                 R"("coordinate_system":"bottom-left")");
    auto result = run(stream_with_end(coordinate));
    check(result.status == 2, "invalid coordinate contract must fail");
    check_contains(result.diagnostics, "unsupported coordinate_system",
                   "coordinate contract diagnostic missing");

    auto blend = valid;
    replace_once(blend, R"("blend":"source-over")", R"("blend":"additive")");
    result = run(stream_with_end(blend));
    check(result.status == 2, "invalid blend contract must fail");
    check_contains(result.diagnostics, "unsupported blend",
                   "blend contract diagnostic missing");

    auto revision = valid;
    replace_once(revision, R"("revision":2)",
                 R"("revision":18446744073709551615)");
    result = run(stream_with_end(revision));
    check(result.status == 2, "unsafe JSON unsigned conversion must fail");
    check_contains(result.diagnostics, "invalid unsigned integer field: revision",
                   "unsigned conversion diagnostic missing");
}

void test_extreme_geometry_is_rejected_or_clipped_safely() {
    const auto valid = valid_scene_line();

    auto rect = valid;
    replace_once(rect, R"("rect":[0,0,1,1])",
                 R"("rect":[1e308,0,1,1])");
    auto result = run(stream_with_end(rect));
    check(result.status == 2, "extreme rectangle must fail");
    check_contains(result.diagnostics, "out-of-range geometry in rect",
                   "extreme rectangle diagnostic missing");

    auto polyline = valid;
    replace_once(polyline, R"("vertices":[[0,0],[1,1]])",
                 R"("vertices":[[0,0],[1e308,1]])");
    result = run(stream_with_end(polyline));
    check(result.status == 2, "extreme polyline must fail");
    check_contains(result.diagnostics, "out-of-range geometry in polyline vertex",
                   "extreme polyline diagnostic missing");

    auto text = valid;
    replace_once(text, R"("anchor":[0.5,1])",
                 R"("anchor":[1e308,1])");
    result = run(stream_with_end(text));
    check(result.status == 2, "extreme text anchor must fail");
    check_contains(result.diagnostics, "out-of-range geometry in anchor",
                   "extreme text diagnostic missing");

    auto nonfinite = valid;
    replace_once(nonfinite, R"("anchor":[0.5,1])",
                 R"("anchor":[1e309,1])");
    result = run(stream_with_end(nonfinite));
    check(result.status == 2, "nonfinite text anchor must fail");
    check_contains(result.diagnostics, "invalid finite number",
                   "nonfinite numeric diagnostic missing");

    auto clipped = valid;
    replace_once(clipped, R"("vertices":[[0,0],[1,1]])",
                 R"("vertices":[[-1000000,0.25],[1000000,0.25]])");
    result = run(stream_with_end(clipped));
    check(result.status == 0, "bounded offscreen polyline must be clipped");
    check_single_frame(result);
}

} // namespace

int main() {
    try {
        test_fixture_recovers_and_renders_visible_panel();
        test_control_only_scene_propagates_degraded_terminal();
        test_bad_scene_is_transactional_and_later_scene_renders();
        test_menu_control_changes_host_frame();
        test_fail_fast_emits_no_partial_frame();
        test_scene_end_protocol_is_enforced();
        test_terminal_verdict_and_exit_status_propagate();
        test_contract_and_unsigned_fields_are_validated();
        test_extreme_geometry_is_rejected_or_clipped_safely();
        std::cout << "dtmb_live_view_host_tests: ok\n";
        return 0;
    } catch (const std::exception &exc) {
        std::cerr << "dtmb_live_view_host_tests: " << exc.what() << '\n';
        return 1;
    }
}
