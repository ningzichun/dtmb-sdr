import subprocess
import sys
import numpy as np
import pytest
from synthetic_support import DATA, ci8_frames, symbol_stream

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

@pytest.mark.parametrize('pn',['pn420','pn595','pn945'])
@pytest.mark.parametrize('profile',range(5,11))
def test_4qam_all_profiles_headers_exact_transport(native,pn,profile):
    rate=(profile-5)//2+1; mode='mode1' if profile%2 else 'mode2'
    symbols,bits,expected=symbol_stream(rate,4,mode)
    raw=ci8_frames(symbols,profile,4,pn_mode=pn)
    frontend=subprocess.run(native('c3780_extract','--pn-mode',pn,'--system-info-index',profile,
        '--normalization','qam','--workers',1,'-','-'),input=raw,capture_output=True,timeout=90)
    assert frontend.returncode==0,frontend.stderr.decode()
    demap=subprocess.run(native('deinterleave_qam','--qam','4qam','--mode',mode,'--workers',1,'-','-'),
        input=frontend.stdout,capture_output=True,timeout=90)
    assert demap.returncode==0,demap.stderr.decode()
    np.testing.assert_array_equal(np.frombuffer(demap.stdout,'<f4')<0,bits)
    fec=subprocess.run(native('ldpc_bch_decode','--fec-rate',rate,'--alist',DATA/f'dtmb_ldpc_rate{rate}.alist',
        '--qam','4qam','--workers',1,'--clean-frames-only','--fail-on-unclean-frame','--require-output','-','-'),
        input=demap.stdout,capture_output=True,timeout=90)
    assert fec.returncode==0,fec.stderr.decode()
    assert fec.stdout==expected

@pytest.mark.parametrize('format',['cu8','ci16','cf32'])
def test_frontend_formats_match_ci8(native,format):
    symbols,_,_=symbol_stream(3,4,'mode1')
    raw=ci8_frames(symbols,9,4)
    x=np.frombuffer(raw,'i1')
    encoded=((x.astype(np.int16)+128).astype('u1') if format=='cu8' else
             x.astype('<i2')*256 if format=='ci16' else x.astype('<f4')/128).tobytes()
    def run(data,fmt):
        p=subprocess.run(native('c3780_extract','--input-format',fmt,'--system-info-index',9,
            '--normalization','qam','--workers',1,'-','-'),input=data,capture_output=True,timeout=90)
        assert p.returncode==0,p.stderr.decode()
        return p.stdout
    assert run(encoded,format)==run(raw,'ci8')


@pytest.mark.parametrize('pn',['pn420','pn595','pn945'])
def test_sc16_cli_recovers_4qam_transport(native_bin,pn):
    symbols,_,expected=symbol_stream(3,4,'mode1')
    symbols=np.concatenate((symbols,symbols[-3744:]))
    raw=ci8_frames(symbols,9,4,pn_mode=pn,scheduled=True)
    samples=(np.frombuffer(raw,'i1').astype('<i2')*256).tobytes()
    result=subprocess.run([sys.executable,'-m','dtmb.decode','--input-format','sc16',
        '--input-rate','7560000','--system-info-index','9','--pn-mode',pn,
        '--bin-dir',str(native_bin),'--workers','1','--pipeline-buffer-mib','0'],
        input=samples,capture_output=True,timeout=90)
    assert result.returncode==0,result.stderr.decode()
    assert result.stdout==expected
