import sys
import os
sys.path.insert(0, os.path.abspath('python'))

import numpy as np
from dtmb.fec import encode_frame_bch_ldpc_codewords
from dtmb.qam import qam_modulate
from dtmb.symbol_interleaver import convolutional_interleave
from dtmb.frequency import frequency_interleave
from dtmb.system_info import system_info_symbols
from dtmb.pn import pn_header_symbols_for_body_power

def write_synthetic_fixture(ci8_path, ts_path, frames):
    packet = bytes([0x47]) + bytes(range(1, 188))
    # 4 packets per frame for 4QAM, FEC rate 3
    ts_bytes = packet * (frames * 4) 
    
    with open(ts_path, 'wb') as f:
        f.write(ts_bytes)

    data_chunks = []
    for frame_index in range(frames):
        chunk = ts_bytes[frame_index * 752 : (frame_index + 1) * 752]
        codeword = encode_frame_bch_ldpc_codewords(chunk, fec_rate_index=3)
        data_chunks.append(qam_modulate(codeword, mode="4qam"))
        
    data_stream = convolutional_interleave(
        np.concatenate(data_chunks),
        mode="mode1",
    )
    
    bodies = []
    for frame_index in range(frames):
        logical = np.empty(3780, dtype=np.complex64)
        # Vector 9: 4QAM, FEC rate 3, Mode 1
        logical[:36] = system_info_symbols(
            "01001011111110110001110111100110",
            frame_body_mode="C3780",
        )
        logical[36:] = data_stream[frame_index * 3744 : (frame_index + 1) * 3744]
        body = np.fft.ifft(frequency_interleave(logical)).astype(np.complex64)
        bodies.append(body)

    body_power = float(np.mean([np.mean(np.abs(body) ** 2) for body in bodies]))
    pn_header = pn_header_symbols_for_body_power("pn945", body_power=body_power)
    frame_samples = [np.concatenate((pn_header, body)) for body in bodies]
        
    capture = np.concatenate(frame_samples)
    capture = capture / np.max(np.abs(capture)) * 0.75
    ci8 = np.empty(capture.size * 2, dtype=np.int8)
    ci8[0::2] = np.clip(np.round(capture.real * 127), -128, 127).astype(np.int8)
    ci8[1::2] = np.clip(np.round(capture.imag * 127), -128, 127).astype(np.int8)
    
    with open(ci8_path, 'wb') as f:
        f.write(ci8.tobytes())

if __name__ == '__main__':
    write_synthetic_fixture('captures/synthetic_ts.ci8', 'captures/synthetic_oracle.ts', frames=173)
    print("Generated captures/synthetic_ts.ci8 and captures/synthetic_oracle.ts")
