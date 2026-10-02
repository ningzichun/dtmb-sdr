import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readIQ, spectrum, inspect } from '../src/diagnostics.js';
import { commands, profile } from '../src/pipeline.js';
import { createDetection, observeDetection, detectionSummary, resolveReceiver } from '../src/pn-detection.js';
import { TransportMetadata, mpegSectionCrc, pcrSyntaxError } from '../src/transport-info.js';
import { commonSampleRates, selectRateGuess } from '../src/sample-rate.js';
test('signed aliases, unsigned centering and low CI16 bits', () => {
  assert.deepEqual([...readIQ(Uint8Array.of(0,128,255,129).buffer,'cu8')],[-1,0,127/128,1/128]);
  assert.deepEqual([...readIQ(Int16Array.of(-32768,1,32767,-1).buffer,'sc16')],[-1,1/32768,32767/32768,-1/32768]);
  assert.throws(()=>readIQ(new ArrayBuffer(3),'ci16'),/incomplete/);
});
test('complex tone occupies the positive-frequency bin with calibrated level', () => {
  const n=2048, iq=new Float32Array(n*2);
  for(let i=0;i<n;i++){iq[2*i]=.5*Math.cos(2*Math.PI*123*i/n);iq[2*i+1]=.5*Math.sin(2*Math.PI*123*i/n);}
  const bins=spectrum(iq), peak=bins.indexOf(Math.max(...bins));
  assert.equal(peak,n/2+123); assert.ok(Math.abs(bins[peak]+6.0206)<.01);
});
test('reports sampled evidence and rejects nonfinite samples', async () => {
  const x=new Float32Array(4096); x[0]=NaN;
  const result=await inspect(new File([x],'invalid.cf32'),{format:'cf32',rate:15120000});
  assert.equal(result.nonfinite,1); assert.match(result.warnings.join(' '),/non-finite/);
  assert.equal(result.file.hashedBytes,x.byteLength); assert.equal(result.file.prefixSha256.length,64);
});
test('all profiles use the same receiver stages and preserve input precision', () => {
  for(let i=3;i<=24;i++){
    const p=profile(i), stages=commands({profile:i,rate:12500000,format:'sc16',pn:'pn595'});
    assert.equal(stages.length,4);assert.ok(stages[0].args.includes('cf32'));
    assert.ok(stages[1].args.includes(p.qam));assert.ok(stages[3].args.includes(`${p.words}`));
    if(i<5) assert.ok(stages[3].args.includes('--nr-frame-phase'));
  }
  assert.throws(()=>profile(2)); assert.throws(()=>commands({profile:9,rate:NaN,pn:'pn595'}));
});

test('PN acquisition and SI lock require observations rather than configured modes', () => {
  const options = { pn: 'pn595', format: 'ci8', rate: 16000000, startByte: 0, endByte: 6400000 };
  const result = createDetection(options);
  observeDetection(result, 'pn_mode=pn595');
  observeDetection(result, 'system_info_index=10');
  assert.equal(result.state, 'not-tested');
  assert.equal(result.systemInformation, 'not-tested');
  for (const line of ['acquisition_hit_count=16', 'acquisition_observed_frames=16',
    'system_info_auto=true', 'system_info_auto_locked=false']) observeDetection(result, line);
  assert.equal(detectionSummary(result, options.pn), 'PN595 acquired; system information not locked.');
  assert.equal(createDetection({ ...options, pn: 'pn420' }).state, 'not-tested');
  result.runStatus = 'complete';
  const candidates = { pn595: result, pn420: { state: 'not-acquired', runStatus: 'failed' }, pn945: { state: 'not-acquired', runStatus: 'failed' } };
  assert.equal(resolveReceiver(candidates).profile, null);
  observeDetection(result, 'system_info_index=10'); observeDetection(result, 'system_info_auto_locked=true');
  assert.equal(resolveReceiver(candidates).pn, 'pn595');
  assert.equal(resolveReceiver(candidates).profile, 10);
  assert.equal(resolveReceiver(candidates, 'pn945', '21').profile, 21);
});

test('sample-rate guessing requires a complete scan and unique or dominant acquisition', () => {
  const candidates = commonSampleRates.map(rate => ({ rate, detections: Object.fromEntries(
    ['pn420', 'pn595', 'pn945'].map(mode => [mode, { state: 'not-acquired', runStatus: 'failed' }])) }));
  const acquired = { state: 'acquired', runStatus: 'complete', native: {
    acquisition_hit_count: '16', acquisition_observed_frames: '16', acquisition_mean_metric: '0.543', sync_hit_threshold: '0.35' } };
  const selected = candidates.find(candidate => candidate.rate === 12500000);
  const alternative = candidates.find(candidate => candidate.rate === 15120000);
  selected.detections.pn595 = acquired;
  assert.equal(selectRateGuess(candidates).rate, 12500000);
  assert.equal(selectRateGuess(candidates.slice(0, -1)).rate, null);
  alternative.detections.pn595 = acquired;
  assert.equal(selectRateGuess(candidates).rate, null);
  alternative.detections.pn595 = { ...acquired, native: { ...acquired.native, acquisition_mean_metric: '0.35' } };
  assert.equal(selectRateGuess(candidates).rate, 12500000);
});

test('advertised PCR fields need valid reserved bits and extension without payload repair', () => {
  const packet = new Uint8Array(188).fill(0xff);
  packet.set([0x47, 1, 0, 0x30, 7, 0x10, 0, 0, 0, 0, 0x7e, 0]);
  assert.equal(pcrSyntaxError(packet), false);
  packet[10] = 0;
  assert.equal(pcrSyntaxError(packet), true);
  packet[10] = 0x7f; packet[11] = 44;
  assert.equal(pcrSyntaxError(packet), true);
  packet[5] = 0;
  assert.equal(pcrSyntaxError(packet), false);
});

test('transport info uses CRC-valid PAT/PMT declarations rather than media decoding', () => {
  assert.equal(mpegSectionCrc(new TextEncoder().encode('123456789')), 0x0376e6e7);
  const packet = (pid, section) => {
    const bytes = new Uint8Array(188).fill(0xff);
    bytes.set([0x47, 0x40 | pid >> 8, pid & 255, 0x10, 0]); bytes.set(section, 5); return bytes;
  };
  const hex = text => Uint8Array.from(text.match(/../g).map(value => parseInt(value, 16)));
  const metadata = new TransportMetadata();
  metadata.add(packet(0, hex('00b00d0001c100000001f0002ab104b2')));
  metadata.add(packet(0x1000, hex('02b01d0001c10000e100f0001be100f0000fe101f0060a04656e67008d829a07')));
  const result = metadata.result();
  assert.equal(result.transportStreamId, 1); assert.equal(result.programs[0].number, 1);
  assert.equal(result.programs[0].pcrPid, 0x100);
  assert.deepEqual(result.programs[0].streams.map(stream => [stream.pid, stream.kind, stream.description, stream.language]),
    [[0x100, 'video', 'H.264 / AVC', undefined], [0x101, 'audio', 'AAC / ADTS', 'eng']]);
  const invalid = new TransportMetadata(), damaged = hex('00b00d0001c100000001f0002ab104b2'); damaged[9] ^= 1;
  invalid.add(packet(0, damaged)); assert.equal(invalid.result().programs.length, 0);
});
