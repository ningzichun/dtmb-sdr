import { formats } from './diagnostics.js';
import { profile, decode } from './pipeline.js';
import { Spectrum, Waterfall } from './plots.js';
import { preview } from './player.js';
import { defaultManifest, parseManifest, applyEntry, fetchFile, sampleFile, iqFromZip, CorsError } from './samples.js';

const $ = id => document.getElementById(id);
let file, analysis, inspection, running, report, outputURL, build = {}, bins, lastOptions;
let stopPreview;
const spectrum = new Spectrum($('spectrum'));
let waterfall;
try { waterfall = new Waterfall($('waterfall')); } catch (error) { $('status').textContent = error.message; }
spectrum.draw();
for (let i = 3; i <= 24; i++) {
  const p = profile(i), option = document.createElement('option');
  const label = p.qam === '4qam-nr' ? '4QAM-NR' : p.qam.toUpperCase();
  option.value = i; option.textContent = `${label} · FEC ${[0, .4, .6, .8][p.rate]} · ${p.mode.replace('mode', 'IL ')} [${i}]`;
  $('profile').append(option);
}
$('profile').value = '21';
fetch('./build.json').then(r => r.ok ? r.json() : {}).then(data => { build = data; if (data.commit) $('build').textContent = `dtmb-sdr ${data.coreVersion?`v${data.coreVersion} · `:``}${data.commit.slice(0, 8)} · MIT`; }).catch(() => {});

function options() {
  const rate = Number($('rate').value), format = $('format').value, center = Number($('center').value) * 1e6;
  const start = Number($('start').value), duration = Number($('duration').value);
  if (!Number.isSafeInteger(rate) || rate <= 0 || rate > 4294967295) throw Error('Enter a positive sample rate in samples per second.');
  if (![start, duration, center].every(Number.isFinite) || start < 0 || duration < 0 || center < 0) throw Error('Frequency, start and duration must be finite, nonnegative values.');
  const width = formats[format], startByte = Math.floor(start * rate) * width;
  const endByte = duration ? Math.min(file.size, startByte + Math.floor(duration * rate) * width) : file.size;
  if (startByte >= endByte) throw Error('The selected interval is outside the recording.');
  return { rate, format, center, start, duration, startByte, endByte, pn: $('pn').value, profile: Number($('profile').value), tracking: $('tracking').checked };
}
function badge(id, text, kind = '') { $(id).textContent = text; $(id).className = `badge ${kind}`; }
function status(text) { $('status').textContent = text; }
function redraw() { spectrum.draw(bins, Number($('rate').value), Number($('center').value) * 1e6, Number($('floor').value)); if (waterfall) { waterfall.floor = Number($('floor').value); waterfall.draw(); } }
function lock(busy) {
  for (const id of ['file', 'format', 'rate', 'center', 'start', 'duration', 'pn', 'profile', 'tracking']) $(id).disabled = busy;
  $('inspect').disabled = busy || !file; $('decode').disabled = busy || !analysis || analysis.nonfinite > 0;
  $('cancel').hidden = !running;
}
function invalidate() {
  inspection?.terminate(); inspection = null; analysis = null; report = null;
  $('decode').disabled = true; $('report').disabled = true;
  badge('health-badge', 'RECHECK SETTINGS');
  badge('receiver-badge', 'IDLE');
  $('download').hidden = true;
  stopPreview?.(); stopPreview = null; $('preview').hidden = true; $('video').hidden = true;
  if (outputURL) { URL.revokeObjectURL(outputURL); outputURL = null; }
}
function choose(selected) {
  if (running || !selected) return;
  invalidate(); file = selected;
  if (/\.zip$/i.test(file.name)) { unpackZip(file).then(choose).catch(error => { file = null; lock(false); status(error.message); }); return; }
  $('file-name').textContent = file.name; $('file-info').textContent = `${(file.size / 1048576).toFixed(1)} MiB`;
  const suffix = file.name.split('.').pop().toLowerCase();
  const names = { cs8: 'ci8', sc16: 'ci16', cs16: 'ci16' };
  if (formats[suffix]) $('format').value = names[suffix] || suffix;
  lock(false); inspect();
}
$('file').onchange = () => choose($('file').files[0]);
for (const type of ['dragover', 'dragleave', 'drop']) $('dropzone').addEventListener(type, e => {
  e.preventDefault(); $('dropzone').classList.toggle('drag', type === 'dragover');
  if (type !== 'drop') return;
  const dropped = e.dataTransfer.files[0];
  if (dropped) { choose(dropped); return; }
  const link = (e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain')).split('\n')
    .map(line => line.trim()).find(line => /^https:\/\//i.test(line));
  if (link) loadURL(link);
});
for (const id of ['format', 'rate', 'start', 'duration']) $(id).addEventListener('change', () => { invalidate(); lock(false); status('Settings changed. Inspect the selected interval again.'); });
for (const id of ['pn', 'profile', 'tracking']) $(id).addEventListener('change', () => {
  badge('receiver-badge', 'IDLE'); $('download').hidden = true; $('preview').hidden = true;
  stopPreview?.(); stopPreview = null; $('video').hidden = true;
  if (report) { report.receiver = { status: 'not-run' }; report.options = options(); }
});
$('center').onchange = redraw;
$('floor').oninput = () => { $('floor-value').value = `${$('floor').value} dB`; redraw(); };
window.addEventListener('resize', redraw);

function inspect() {
  if (!file || running) return;
  let selected;
  try { selected = options(); } catch (error) { status(error.message); return; }
  invalidate(); lastOptions = selected; lock(true);
  waterfall?.reset(); badge('health-badge', 'INSPECTING'); $('plot-state').textContent = 'Sampling capture';
  inspection = new Worker(new URL('./inspect-worker.js', import.meta.url), { type: 'module' });
  const fail = error => { status(error.message); inspection?.terminate(); inspection = null; badge('health-badge', 'CHECK INPUT', 'warn'); lock(false); };
  inspection.onerror = fail;
  inspection.onmessage = ({ data }) => {
    if (data.type === 'row') { bins = data.bins; redraw(); waterfall?.add(bins, Number($('floor').value)); }
    else if (data.type === 'error') fail(Error(data.message));
    else if (data.type === 'result') {
      analysis = data.result; bins = analysis.spectrum; redraw();
      inspection.terminate(); inspection = null;
      const a = analysis;
      $('rms').replaceChildren(document.createTextNode(a.rmsDbfs.toFixed(1)), unit(' dBFS'));
      $('clipping').replaceChildren(document.createTextNode(a.clippingPercent.toFixed(3)), unit(' %'));
      $('dc').textContent = `${a.dcI.toFixed(3)} / ${a.dcQ.toFixed(3)}`; $('dc').style.fontSize = '15px';
      $('length').replaceChildren(document.createTextNode((file.size / formats[selected.format] / selected.rate).toFixed(2)), unit(' s'));
      $('sample-count').textContent = `${a.samples.toLocaleString()} samples inspected`;
      $('plot-state').textContent = `${(selected.rate / 1e6).toFixed(3)} MS/s`;
      badge('health-badge', a.warnings.length ? 'CHECK CAPTURE' : 'INPUT CHECKED', a.warnings.length ? 'warn' : 'good');
      $('findings').replaceChildren();
      const intro = document.createElement('p'); intro.className = 'empty';
      intro.textContent = a.warnings.length ? 'Review these observations before decoding.' : 'No obvious level, clipping, or I/Q balance problems in the sampled windows.';
      $('findings').append(intro);
      if (a.warnings.length) { const list = document.createElement('ul'); for (const warning of a.warnings) { const li = document.createElement('li'); li.textContent = warning; list.append(li); } $('findings').append(list); }
      const note = document.createElement('p'); note.className = 'hint'; note.textContent = 'This is a sampled input check, not a DTMB validity verdict. Confirm the original export format and sample rate, then decode to check PN lock and clean FEC output.'; $('findings').append(note);
      report = { schema: 'dtmb-web-diagnostic-v1', created: new Date().toISOString(), build, input: a, options: selected, receiver: { status: 'not-run' } };
      $('report').disabled = false; lock(false); status('Capture inspected. Confirm the PN mode and transmission profile, then decode.');
    }
  };
  inspection.postMessage({ file, options: selected });
}
function unit(text) { const el = document.createElement('small'); el.textContent = text; return el; }
$('inspect').onclick = inspect;
$('decode').onclick = async () => {
  if (!analysis || running) return;
  let selected;
  try { selected = options(); } catch (error) { status(error.message); return; }
  stopPreview?.(); stopPreview = null; $('video').hidden = true; $('preview').hidden = true;
  $('log').textContent = ''; $('progress').value = 0; $('download').hidden = true; $('transport').textContent = '';
  document.querySelectorAll('.stages span').forEach(el => el.classList.remove('done'));
  badge('receiver-badge', 'DECODING'); status('Loading receiver and acquiring PN. The first output may take a while.');
  report.options = selected; report.receiver = { status: 'running' };
  let lines = 0;
  running = decode(file, selected, {
    progress: value => { $('progress').value = value; },
    log: (stage, line) => { if (lines++ < 2000) $('log').append(document.createTextNode(`[${stage}] ${line}\n`)); },
    output: bytes => status(`Recovered ${(bytes / 1024).toFixed(1)} KiB of transport. Processing continues…`),
    stage: i => document.querySelectorAll('.stages span')[i].classList.add('done'),
  });
  lock(true);
  const current = running;
  try {
    const result = await current.promise;
    const transport = await checkTransport(result.blob);
    report.receiver = { status: 'complete', bytes: result.bytes, logs: result.logs, transport };
    if (outputURL) URL.revokeObjectURL(outputURL);
    outputURL = URL.createObjectURL(result.blob); $('download').href = outputURL; $('download').download = `${file.name.replace(/\.[^.]+$/, '')}.ts`; $('download').hidden = false;
    $('preview').hidden = false;
    badge('receiver-badge', transport.syncErrors || transport.transportErrors ? 'CHECK TRANSPORT' : 'TS RECOVERED', transport.syncErrors || transport.transportErrors ? 'warn' : 'good');
    status(`Recovered ${(result.bytes / 1024).toFixed(1)} KiB from the selected interval.`);
    $('transport').textContent = `${transport.packets.toLocaleString()} packets · ${transport.syncErrors} sync errors · ${transport.transportErrors} transport error flags. Failed FEC groups are omitted with discontinuity markers. Validate playback with a codec decoder for an error-free-video claim.`;
  } catch (error) {
    report.receiver = { status: error.name === 'AbortError' ? 'cancelled' : 'failed', error: error.message, logs: current.logs };
    badge('receiver-badge', error.name === 'AbortError' ? 'CANCELLED' : 'NO DECODE', 'warn'); status(error.message);
  } finally { running = null; lock(false); }
};
$('cancel').onclick = () => running?.cancel();
$('preview').onclick = async () => {
  if (!outputURL) return;
  try {
    stopPreview?.(); $('video').hidden = false;
    $('preview-status').textContent = 'Preview uses the codecs supported by your browser. A preview is not a full-stream validation.';
    stopPreview = await preview(outputURL, $('video'), text => { $('preview-status').textContent = text; });
  } catch (error) { $('preview-status').textContent = error.message; }
};
async function checkTransport(blob) {
  const reader = blob.stream().getReader(); let tail = new Uint8Array(), packets = 0, syncErrors = 0, transportErrors = 0;
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    const bytes = new Uint8Array(tail.length + value.length); bytes.set(tail); bytes.set(value, tail.length);
    const end = bytes.length - bytes.length % 188;
    for (let i = 0; i < end; i += 188) { packets++; syncErrors += bytes[i] !== 0x47 ? 1 : 0; transportErrors += bytes[i + 1] & 0x80 ? 1 : 0; }
    tail = bytes.slice(end);
  }
  return { packets, syncErrors, transportErrors, trailingBytes: tail.length, scope: 'Packet alignment and TEI checks; not continuity or codec validation.' };
}
$('report').onclick = () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = 'dtmb-diagnostic.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};

async function unpackZip(file) {
  status('Unpacking ZIP...');
  return iqFromZip(file);
}

async function loadSamples() {
  const list = $('samples-list'), state = $('samples-status');
  const manifest = new URL(location.href).searchParams.get('manifest') || defaultManifest;
  try {
    const response = await fetch(manifest);
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    const samples = parseManifest(await response.text());
    list.textContent = '';
    for (const entry of samples) {
      const item = document.createElement('li');
      const name = document.createElement('b'); name.textContent = entry.name;
      const meta = document.createElement('small');
      const rate = Number(entry.sample_rate_sps), center = Number(entry.center_hz);
      meta.textContent = [entry.format, Number.isFinite(rate) ? `${rate / 1e6} MS/s` : '', Number.isFinite(center) ? `${center / 1e6} MHz` : '', entry.pn_mode, entry.profile ? `profile ${entry.profile}` : '', entry.notes].filter(Boolean).join(' · ');
      item.append(name, meta); item.title = entry.url;
      item.onclick = () => pickSample(entry);
      list.append(item);
    }
  } catch (error) { state.textContent = `Sample list unavailable (${error.message}).`; }
}

async function pickSample(entry) {
  if (running) return;
  const fields = { format: $('format'), rate: $('rate'), center: $('center'), start: $('start'), duration: $('duration'), pn: $('pn'), profile: $('profile'), tracking: $('tracking') };
  const errors = applyEntry(entry, fields);
  invalidate();
  try {
    const selected = await sampleFile(entry, status);
    choose(selected);
    if (errors.length) status(`${file.name} loaded. ${errors.join(' ')} Confirm the highlighted fields.`);
  } catch (error) { lock(false); fallbackDownload(entry.url, error); }
}
loadSamples();
async function loadURL(url) {
  if (running || !url) return;
  invalidate();
  try {
    const selected = await fetchFile(url, { onStatus: status });
    choose(selected);
  } catch (error) { lock(false); fallbackDownload(url, error); }
}
function fallbackDownload(url, error) {
  if (error instanceof CorsError) {
    window.open(url, '_blank', 'noopener');
    status('Direct download blocked by CORS. The file is downloading in a new tab — drop it here once saved.');
    return;
  }
  status(error.message);
}
$('url-load').onclick = () => loadURL($('url').value.trim());
$('url').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); $('url-load').click(); } });