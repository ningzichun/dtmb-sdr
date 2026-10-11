"""Normative SI-to-data amplitude conversion in the native frontend."""
import subprocess
import numpy as np
import pytest


@pytest.mark.parametrize('qam,profile', [(4, 10), (16, 16), (32, 18), (64, 24)])
def test_explicit_si_normalization_restores_constellation(native, qam, profile):
    vectors = {10: '01001011111110110001110111100110',
               16: '00101101100111010111101110000000',
               18: '01110111001110000010000100100101',
               24: '00010001010111100100011101000011'}
    # Even SI profiles complement the corresponding Appendix G odd row.
    bits = np.r_[np.ones(4, int), 1-np.array([int(v) for v in vectors[profile]])]
    si = (2*bits-1)*4.5*(1+1j)
    mapping = np.arange(3780).reshape(7, 5, 2, 2, 3, 3, 3).transpose(6, 5, 4, 3, 2, 1, 0).ravel()
    positions = (np.array([0, 140, 279, 419])[None, :]+420*np.arange(9)[:, None]).ravel()
    mask = np.ones(3780, bool); mask[positions] = False
    # Fixed Figure 4/5 constellation points; amplitude matters, payload does not.
    points = np.array([-4.5-4.5j, 4.5+4.5j]) if qam == 4 else np.array([-7.5-1.5j, 1.5+4.5j])
    expected = np.resize(points, (4, 3744))
    inserted = np.empty((4, 3780), complex); inserted[:, positions] = si; inserted[:, mask] = expected
    physical = np.empty_like(inserted); physical[:, mapping] = inserted
    bodies = np.fft.ifft(physical, axis=1)
    # Flat equalization does not inspect PN, so a zero header isolates the SI contract.
    samples = np.column_stack((np.zeros((4, 945), complex), bodies)).ravel()*(.71+.43j)
    result = subprocess.run(native('c3780_extract', '--input-format', 'cf32', '--pn-mode', 'pn945',
                                  '--qam', f'{qam}qam', '--system-info-index', profile,
                                  '--normalization', 'system-info', '--equalizer', 'flat', '--workers', 1, '-', '-'),
                            input=samples.astype('<c8').tobytes(), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()
    np.testing.assert_allclose(np.frombuffer(result.stdout, '<c8').reshape(4, 3744), expected,
                               rtol=1e-5, atol=2e-4)
