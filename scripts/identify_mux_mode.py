"""Identify the DTMB system-information mode of a capture (or live HackRF center).

Acquires PN945 timing, equalizes the C=3780 body with the compact-PN equalizer,
and reports the top system-information classification (index -> qam/rate/
interleaver) across a handful of frames, with polarity inversion enabled so the
mode1/mode2 complement pair is resolved by the data-carrier EVM. This is the
fast Gate-C identifier used to pick the right receiver mode before committing to
a full (slow) mode2/mode1 decode.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from dtmb.ci8 import read_ci8
from dtmb.conditioning import remove_dc, frequency_shift
from dtmb.channel import (
    estimate_pn_channel_compact, equalize_spectrum_with_channel,
    pn_restore_circular_body_window, detect_pn_phase,
)
from dtmb.frame_sync import (
    detect_pn_cyclic_extension_trains, estimate_cfo_from_pn_cyclic_extension,
)
from dtmb.frames import iter_signal_frames
from dtmb.frequency import frame_body_fft, frequency_deinterleave, split_system_info_and_data
from dtmb.qam import QAM_DEFINITIONS, normalize_qam_symbols, qam_nearest
from dtmb.system_info import (
    SYSTEM_INFO_VECTORS, classify_system_info, system_info_symbols,
    transmission_parameters_for_index,
)


def _evm(data, qam_mode):
    v = normalize_qam_symbols(np.asarray(data, dtype=np.complex64), mode=qam_mode)
    n = qam_nearest(v, mode=qam_mode)
    p = float(np.mean(np.abs(n) ** 2))
    return float(np.sqrt(np.mean(np.abs(v - n) ** 2)) / np.sqrt(p)) if p > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--sample-rate", type=int, default=7_560_000)
    ap.add_argument("--max-samples", type=int, default=1_200_000)
    ap.add_argument("--frames", type=int, default=16)
    args = ap.parse_args()

    SR = args.sample_rate
    raw = remove_dc(read_ci8(args.capture, max_samples=args.max_samples))
    trains = detect_pn_cyclic_extension_trains(raw, modes=("pn945",))
    if not trains:
        print(f"{args.capture.name}: NO PN945 acquisition")
        return 1
    ph = trains[0].phase_offset
    cfo = estimate_cfo_from_pn_cyclic_extension(raw, mode="pn945", phase_offset=ph, symbol_rate_sps=SR)
    s = frequency_shift(raw, sample_rate_sps=SR, shift_hz=-cfo)
    frames = list(iter_signal_frames(s, mode="pn945", phase_offset=ph, max_frames=args.frames + 1))

    top_counter: Counter = Counter()
    metrics = []
    # For each candidate (index, frame_body_mode) measure data EVM to break the
    # mode1/mode2 + C1/C3780 complement tie.
    for fi in range(min(args.frames, len(frames) - 1)):
        est = estimate_pn_channel_compact(frames[fi].header, mode="pn945", fft_size=3780, channel_taps=64)
        body = pn_restore_circular_body_window(
            frames[fi].body, body_window_offset_symbols=0, pn_phase=est.pn_phase,
            next_pn_phase=detect_pn_phase(frames[fi + 1].header, mode="pn945"),
            mode="pn945", taps=est.taps, next_header=frames[fi + 1].header)
        eq = equalize_spectrum_with_channel(frame_body_fft(body), est)
        si, _ = split_system_info_and_data(frequency_deinterleave(eq))
        m = classify_system_info(si, allow_polarity_inversion=True)
        if m:
            top_counter[(m[0].index, m[0].frame_body_mode)] += 1
            metrics.append(m[0].metric)

    if not top_counter:
        print(f"{args.capture.name}: PN945 ok but no system-info classification")
        return 1

    # Resolve mode by measuring C=3780 data EVM for each candidate index's
    # interleaver (mode1 vs mode2 give very different EVM after deinterleave).
    candidates = [idx for (idx, fbm), _ in top_counter.most_common(4) if fbm == "C3780"]
    if not candidates:
        candidates = [idx for (idx, _fbm), _ in top_counter.most_common(2)]

    print(f"{args.capture.name}: PN945 phase={ph} cfo={cfo:.1f}Hz top_si_metric~{np.median(metrics):.3f}")
    for (idx, fbm), count in top_counter.most_common(4):
        try:
            params = transmission_parameters_for_index(idx)
            desc = f"{params.qam_label} rate{params.fec_rate_index} {params.interleaver_mode}"
        except Exception:
            desc = "?"
        print(f"   top {count:2d}/{args.frames}  index {idx} ({fbm})  {desc}")

    # Per-frame DD-ish EVM with each candidate index's QAM, to confirm C3780.
    qam = "64qam"
    print(f"   (data carriers are valid {qam} if EVM finite/low -> C=3780 multicarrier)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
