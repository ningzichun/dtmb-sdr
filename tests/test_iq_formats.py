import subprocess
import sys
import numpy as np
import pytest

@pytest.mark.parametrize('format,dtype,values,expected', [
    ('cu8','u1',[0,128,255,129],[-1,0,127/128,1/128]),
    ('cs8','i1',[-128,0,127,1],[-1,0,127/128,1/128]),
    ('sc16','<i2',[-32768,1,32767,-1],[-1,1/32768,32767/32768,-1/32768]),
    ('ci16','<i2',[-129,129,255,-255],[-129/32768,129/32768,255/32768,-255/32768]),
    ('cf32','<f4',[-.12345,.00001,.5,-.75],[-.12345,.00001,.5,-.75]),
])
def test_native_conversion_preserves_precision(native, format, dtype, values, expected):
    p=subprocess.run(native('ci8_resample','--input-format',format,'--output-format','cf32',
        '--input-rate',7560000,'--output-rate',7560000,'--chunk-samples',1,'-','-'),
        input=np.array(values,dtype=dtype).tobytes(),capture_output=True)
    assert p.returncode==0,p.stderr.decode()
    np.testing.assert_allclose(np.frombuffer(p.stdout,'<f4'),expected,rtol=1e-6)

@pytest.mark.parametrize('rate',[10000000,11520000,12500000,15120000,16000000,20000000,10000019,6000000])
def test_arbitrary_resampling_chunk_invariant_and_tone(native,rate):
    n=20000
    x=(.4*np.exp(2j*np.pi*500000*np.arange(n)/rate)).astype('<c8')
    outputs=[]
    for chunk in [173,65536]:
        p=subprocess.run(native('ci8_resample','--input-format','cf32','--output-format','cf32',
            '--input-rate',rate,'--output-rate',7560000,'--chunk-samples',chunk,'-','-'),input=x.tobytes(),capture_output=True)
        assert p.returncode==0,p.stderr.decode()
        outputs.append(p.stdout)
    assert outputs[0]==outputs[1]
    y=np.frombuffer(outputs[0],'<c8')
    assert len(y)==int(np.ceil(n*7560000/rate))
    expected=.4*np.exp(2j*np.pi*500000*np.arange(len(y))/7560000)
    assert np.max(np.abs(y[100:-100]-expected[100:-100]))<.025

@pytest.mark.parametrize('format,size',[('ci8',1),('cu8',1),('ci16',3),('cf32',7)])
def test_truncated_pairs_fail(native,format,size):
    p=subprocess.run(native('ci8_resample','--input-format',format,'--output-format','cf32','-','-'),input=b'\0'*size,capture_output=True)
    assert p.returncode!=0 and b'incomplete' in p.stderr

def test_nonfinite_float_fails(native):
    p=subprocess.run(native('ci8_resample','--input-format','cf32','--output-format','cf32','-','-'),input=np.array([np.nan,0],'<f4').tobytes(),capture_output=True)
    assert p.returncode!=0 and b'NaN' in p.stderr
