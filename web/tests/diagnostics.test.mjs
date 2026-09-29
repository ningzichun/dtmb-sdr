import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readIQ, spectrum, inspect } from '../src/diagnostics.js';
import { commands, profile } from '../src/pipeline.js';
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
