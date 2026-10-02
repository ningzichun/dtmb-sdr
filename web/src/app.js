import { formats } from './diagnostics.js';
import { profile, decode, probe } from './pipeline.js';
import { pnModes, detectionLabels, boundedProbeOptions, createDetection, observeDetection, detectionSummary, canReuseDetection, observedProfileIndex, resolveReceiver } from './pn-detection.js';
import { Spectrum, Waterfall } from './plots.js';
import { TransportMetadata, pcrSyntaxError } from './transport-info.js';
import { guessSampleRate } from './sample-rate.js';
import { defaultManifest, parseManifest, applyEntry, fetchFile, sampleFile, iqFromZip, CorsError } from './samples.js';

const $ = id => document.getElementById(id);
let file, analysis, inspection, running, report, outputURL, build = {}, bins, lastOptions;
let detections = {};
let viewedMode;
let progressLabel = '', inputFraction = 0;
let rateGuess;
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
$('profile').value = 'auto';
fetch('./build.json', { cache: 'no-store' }).then(r => r.ok ? r.json() : {}).then(data => { build = data; if (data.commit) $('build').textContent = `dtmb-sdr ${data.coreVersion?`v${data.coreVersion} · `:``}${data.commit.slice(0, 8)} · MIT`; }).catch(() => {});

function receiverSelection() { return resolveReceiver(detections, $('pn').value, $('profile').value); }
function selectionText(selection = receiverSelection()) {
  const pn = selection.pn ? selection.pn.toUpperCase() : 'waiting';
  const transmission = selection.profile ? `profile ${selection.profile}` : 'waiting';
  return `PN ${selection.pnSource}: ${pn} · Profile ${selection.profileSource}: ${transmission}`;
}
function options(requireSelection = false) {
  const rate = Number($('rate').value), format = $('format').value, center = Number($('center').value) * 1e6;
  const start = Number($('start').value), duration = Number($('duration').value);
  if (!Number.isSafeInteger(rate) || rate <= 0 || rate > 4294967295) throw Error('Enter a positive sample rate in samples per second.');
  if (![start, duration, center].every(Number.isFinite) || start < 0 || duration < 0 || center < 0) throw Error('Frequency, start and duration must be finite, nonnegative values.');
  const width = formats[format], startByte = Math.floor(start * rate) * width;
  const endByte = duration ? Math.min(file.size, startByte + Math.floor(duration * rate) * width) : file.size;
  if (startByte >= endByte) throw Error('The selected interval is outside the recording.');
  const selection = receiverSelection();
  if (requireSelection && !selection.ready) throw Error(selection.reason);
  return { rate, format, center, start, duration, startByte, endByte,
    pn: selection.pn, profile: selection.profile, bodyMode: 'auto',
    pnSelection: $('pn').value, profileSelection: $('profile').value,
    tracking: $('tracking').checked };
}
function badge(id, text, kind = '') { $(id).textContent = text; $(id).className = `badge ${kind}`; }
function status(text) { $('status').textContent = text; }
function beginProgress(label) {
  progressLabel = label; inputFraction = 0;
  $('progress').hidden = false; $('progress').removeAttribute('value');
  $('progress-status').dataset.state = 'running';
  $('progress-status').textContent = `${label}: starting receiver…`;
}
function updateProgress(value) {
  inputFraction = value; $('progress').value = value;
  $('progress-status').textContent = `${progressLabel}: ${(value * 100).toFixed(1)}% input read · receiver still processing`;
}
function endProgress(outcome, state) {
  $('progress').hidden = true; $('progress').removeAttribute('value');
  $('progress-status').dataset.state = state;
  $('progress-status').textContent = `${outcome} · ${progressLabel}: ${(inputFraction * 100).toFixed(1)}% input read.`;
}
function redraw() { spectrum.draw(bins, Number($('rate').value), Number($('center').value) * 1e6, Number($('floor').value)); if (waterfall) { waterfall.floor = Number($('floor').value); waterfall.draw(); } }
function lock(busy) {
  for (const id of ['file', 'format', 'rate', 'center', 'start', 'duration', 'pn', 'profile', 'tracking', 'auto-scan-pn']) $(id).disabled = busy;
  const rate = Number($('rate').value), validRate = Number.isSafeInteger(rate) && rate > 0 && rate <= 4294967295;
  $('guess-rate').hidden = $('rate').value.trim() !== '';
  $('guess-rate').disabled = busy || !file;
  $('inspect').disabled = busy || !file || !validRate; $('decode').disabled = busy || !analysis || analysis.nonfinite > 0 || !receiverSelection().ready;
  $('test-pn').disabled = $('scan-pn').disabled = busy || !analysis || analysis.nonfinite > 0;
  $('cancel').hidden = !running;
}
function renderDetection() {
  const configured = $('pn').value, selection = receiverSelection();
  const mode = viewedMode || selection.pn || 'pn945', result = detections[mode], native = result?.native || {};
  const metric = key => native[key] !== undefined && Number.isFinite(Number(native[key])) ? Number(native[key]).toFixed(3) : '—';
  const automatic = resolveReceiver(detections), autoProfile = resolveReceiver(detections, configured, 'auto');
  $('pn').options[0].textContent = automatic.pn ? `Auto · ${automatic.pn.toUpperCase()}` : 'Auto (default)';
  $('profile').options[0].textContent = autoProfile.profile ? `Auto · profile ${autoProfile.profile}` : 'Auto (default)';
  $('receiver-selection').textContent = `${selectionText(selection)}.${selection.reason ? ` ${selection.reason}` : ''}`;
  $('test-pn').textContent = configured === 'auto' ? 'Check Auto selection' : 'Test configured PN';
  $('pn-configured').textContent = configured === 'auto' ? `Auto${selection.pn ? ` → ${selection.pn.toUpperCase()}` : ' (waiting)'}` : `${configured.toUpperCase()} (forced)`;
  $('pn-result-mode').textContent = mode.toUpperCase();
  $('pn-summary').textContent = detectionSummary(result, mode);
  $('pn-state').textContent = detectionLabels[result?.state || 'not-tested'];
  $('pn-hits').textContent = result?.state !== 'not-tested' && native.acquisition_observed_frames !== undefined ? `${native.acquisition_hit_count}/${native.acquisition_observed_frames}` : '—';
  $('pn-correlation').textContent = `${metric('acquisition_mean_metric')} / ${metric('sync_hit_threshold')}`;
  $('pn-peak').textContent = metric('acquisition_max_metric');
  $('pn-offset').textContent = result?.state === 'acquired' && native.coarse_cfo_valid === 'true' ? `${Number(native.coarse_cfo_hz).toFixed(1)} Hz` : 'Unavailable';
  const context = result?.context;
  $('pn-dimensions').textContent = native.pn_header_symbols !== undefined ? `${native.pn_header_symbols} / ${native.pn_frame_symbols}` : '—';
  $('pn-period').textContent = native.pn_frame_symbols !== undefined ? `${(Number(native.pn_frame_symbols) / 7560000 * 1000).toFixed(3)} ms` : '—';
  $('pn-superframe').textContent = native.pn_frames_per_superframe ?? '—';
  const observed = detectedProfile(result);
  $('si-state').textContent = observed ? 'Locked' : result?.systemInformation === 'not-locked' ? 'Not locked' : 'Not tested';
  $('si-profile').textContent = observed ? `Profile ${observed.index}` : '—';
  $('si-body').textContent = observed ? native.frame_body_mode === 'c1' ? 'C=1 · single-carrier' : 'C=3780 · multicarrier' : '—';
  $('si-modulation').textContent = observed ? observed.qam.toUpperCase() : '—';
  $('si-fec').textContent = observed ? [0, .4, .6, .8][observed.rate].toFixed(1) : '—';
  $('si-interleaver').textContent = observed ? observed.mode.replace('mode', 'Mode ') : '—';
  $('si-confidence').textContent = native.system_info_auto === 'true' ? `${metric('system_info_auto_metric')} / ${metric('system_info_auto_margin')}` : '—';
  $('si-observations').textContent = native.system_info_auto_observations ?? '—';
  $('si-settings').textContent = observed ? `Observed in ${mode.toUpperCase()}. ${selectionText(selection)}.${selection.ready && (selection.pn !== mode || selection.profile !== observed.index) ? ' Your override differs from these observations.' : ''}` : 'Profile details appear after system-information lock.';
  $('pn-context').textContent = context ? `${context.format.toUpperCase()} · ${(context.sampleRate / 1e6).toFixed(3)} MS/s · ${context.startSeconds.toFixed(6)}–${context.endSeconds.toFixed(6)} s input window bound (${result.source}).${result.error ? ` ${result.error}` : ''}` : 'No interval analyzed.';
  $('pn-timing').textContent = native.startup_buffered_samples !== undefined ? `Acquisition: ${native.startup_buffered_samples} samples (${(Number(native.startup_buffered_samples) / 7560000).toFixed(6)} s) at 7.56 MS/s after conditioning. Header offset: ${native.acquisition_phase_offset ?? '—'} symbols; header/frame: ${native.pn_header_symbols ?? '—'}/${native.pn_frame_symbols ?? '—'} symbols. SI observations: ${native.system_info_auto_observations ?? '—'}; complete: ${native.system_info_auto_complete ?? '—'}.` : 'Not tested.';
  if (native.frame_body_mode === 'c1') $('pn-timing').textContent += ` C=1: CFO alias ${native.c1_cfo_alias}; applied shift ${native.frequency_shift_hz} Hz; SI timing ${native.c1_si_offset_symbols} symbols; PN fit error ${native.c1_pn_fit_error}.`;
  $('pn-modes').replaceChildren(...pnModes.map(candidate => {
    const item = document.createElement('li');
    const button = document.createElement('button'), candidateProfile = detectedProfile(detections[candidate]);
    button.type = 'button'; button.className = 'pn-mode-choice'; button.setAttribute('aria-pressed', String(candidate === mode));
    button.textContent = `${candidate.toUpperCase()}: ${detectionLabels[detections[candidate]?.state || 'not-tested']}${candidateProfile ? ` · profile ${candidateProfile.index}` : ''}${candidate === selection.pn ? ` (${selection.pnSource === 'auto' ? 'auto selected' : 'forced'})` : ''}`;
    button.onclick = () => { viewedMode = candidate; renderDetection(); };
    item.append(button);
    return item;
  }));
  $('stage-pn').textContent = detectionLabels[result?.state || 'not-tested'];
  $('stage-pn').classList.toggle('done', result?.state === 'acquired');
  $('stage-si').textContent = result?.systemInformation === 'locked' ? `Locked · profile ${native.system_info_index}` : result?.systemInformation === 'not-locked' ? 'Not locked' : 'Not tested';
  $('stage-si').classList.toggle('done', result?.systemInformation === 'locked');
  if (report) {
    report.pnDetection = Object.fromEntries(pnModes.map(candidate => [candidate, detections[candidate] || { mode: candidate, state: 'not-tested', systemInformation: 'not-tested' }]));
    report.selection = { pnSetting: configured, profileSetting: $('profile').value, ...selection };
  }
}
function detectedProfile(result) {
  const index = observedProfileIndex(result);
  return index !== null ? { index, ...profile(index) } : null;
}
function resetReceiver(clearDetections = true) {
  if (clearDetections) { detections = {}; viewedMode = undefined; }
  renderDetection();
  $('stage-fec').textContent = $('stage-ts').textContent = 'Not tested';
  $('stage-fec').className = $('stage-ts').className = '';
  $('transport').textContent = ''; $('progress').hidden = true; $('progress').removeAttribute('value');
  $('progress-status').textContent = ''; $('progress-status').dataset.state = 'idle';
}
function invalidate() {
  inspection?.terminate(); inspection = null; analysis = null; report = null;
  resetReceiver();
  $('decode').disabled = true; $('report').disabled = true;
  badge('health-badge', 'RECHECK SETTINGS');
  badge('receiver-badge', 'IDLE');
  $('download').hidden = true;
  $('ts-info').hidden = true; $('ts-details').hidden = true;
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
  if (rateGuess) { rateGuess = null; $('rate').value = ''; $('rate-status').textContent = 'Enter the recording rate, or guess common rates from PN.'; }
  lock(false);
  if ($('rate').value.trim()) inspect();
  else status('Capture loaded. Enter its sample rate or click Guess to test common rates.');
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
for (const id of ['format', 'start', 'duration']) $(id).addEventListener('change', () => {
  if (id === 'format' && rateGuess) { rateGuess = null; $('rate').value = ''; $('rate-status').textContent = 'Format changed. Enter the recording rate or guess again.'; }
  invalidate(); lock(false); status('Settings changed. Inspect the selected interval again.');
});
$('rate').addEventListener('input', () => {
  rateGuess = null; invalidate(); lock(false); redraw();
  $('rate-status').textContent = $('rate').value.trim() ? 'Using the entered sample rate. You can edit it.' : 'Enter the recording rate, or guess common rates from PN.';
  status($('rate').value.trim() ? 'Sample rate changed. Inspect the selected interval again.' : 'Enter the recording rate or click Guess.');
});
for (const id of ['pn', 'profile', 'tracking']) $(id).addEventListener('change', () => {
  if (id === 'pn') viewedMode = $('pn').value === 'auto' ? receiverSelection().pn || undefined : $('pn').value;
  resetReceiver(id === 'tracking');
  badge('receiver-badge', 'IDLE'); $('download').hidden = true;
  $('ts-info').hidden = true; $('ts-details').hidden = true;
  if (report) { report.receiver = { status: 'not-run' }; report.options = options(); }
  lock(false);
});
$('center').onchange = redraw;
$('floor').oninput = () => { $('floor-value').value = `${$('floor').value} dB`; redraw(); };
window.addEventListener('resize', redraw);
new ResizeObserver(redraw).observe($('spectrum'));

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
      const note = document.createElement('p'); note.className = 'hint'; note.textContent = 'The spectrum checks sampled RF energy. PN acquisition recognizes frame structure; system-information lock identifies the transmission profile. FEC and transport checks run during decoding.'; $('findings').append(note);
      report = { schema: 'dtmb-web-diagnostic-v1', created: new Date().toISOString(), build, input: a, options: selected, sampleRateGuess: rateGuess || null, receiver: { status: 'not-run' } };
      renderDetection(); $('report').disabled = false; lock(false);
      if (a.nonfinite > 0) status('Capture inspected. Fix non-finite input samples before testing PN or decoding.');
      else if ($('auto-scan-pn').checked) testModes(true);
      else status('Capture inspected. Test the configured PN mode, scan all modes, or decode.');
    }
  };
  inspection.postMessage({ file, options: selected });
}
function unit(text) { const el = document.createElement('small'); el.textContent = text; return el; }
$('inspect').onclick = inspect;
$('guess-rate').onclick = async () => {
  if (!file || running || inspection || $('rate').value.trim()) return;
  invalidate(); $('log').textContent = '';
  let current, guessed = false;
  try {
    beginProgress('Sample-rate guess probe');
    current = guessSampleRate(file, { format: $('format').value, start: Number($('start').value), duration: Number($('duration').value) }, {
      candidate: (rate, mode) => {
        const message = `Testing ${(rate / 1e6).toFixed(3)} MS/s · ${mode.toUpperCase()}…`;
        $('rate-status').textContent = message; status(`Guessing common sample rates. ${message}`);
        beginProgress('Sample-rate guess probe');
      },
      progress: updateProgress, log: appendLog,
    });
    running = current; lock(true);
    rateGuess = await current.promise;
    if (rateGuess.rate) {
      $('rate').value = rateGuess.rate; guessed = true;
      const result = rateGuess.candidates.find(candidate => candidate.rate === rateGuess.rate).detections[rateGuess.mode];
      $('rate-status').textContent = `Guessed ${(rateGuess.rate / 1e6).toFixed(3)} MS/s · ${rateGuess.mode.toUpperCase()} ${result.native.acquisition_hit_count}/${result.native.acquisition_observed_frames} headers. Editable; not capture metadata.`;
      status('Sample rate guessed from PN. Inspecting with this rate…');
    } else { $('rate-status').textContent = rateGuess.reason; status(rateGuess.reason); }
    endProgress('Sample-rate guess finished', 'complete');
  } catch (error) {
    rateGuess = { ...current?.evidence, rate: null, complete: false, reason: error.message };
    $('rate-status').textContent = error.message; status(error.message);
    endProgress(error.name === 'AbortError' ? 'Sample-rate guess cancelled' : 'Sample-rate guess stopped', error.name === 'AbortError' ? 'cancelled' : 'failed');
  } finally { running = null; lock(false); }
  if (guessed) inspect();
};
function operation() {
  return { active: null, cancelled: false, cancel() { this.cancelled = true; this.active?.cancel(); } };
}
function appendLog(stage, line) {
  if ($('log').childNodes.length < 2000) $('log').append(document.createTextNode(`[${stage}] ${line}\n`));
}
async function testMode(selected, mode, current) {
  if (current.cancelled) throw new DOMException('Receiver operation cancelled.', 'AbortError');
  const settings = boundedProbeOptions(file, { ...selected, pn: mode, profile: selected.profile ?? 21 });
  const result = createDetection(settings);
  detections[mode] = result; renderDetection();
  status(`Testing ${mode.toUpperCase()} frame structure and system information…`);
  beginProgress(`${mode.toUpperCase()} probe window`);
  const active = probe(file, settings, {
    progress: updateProgress,
    log: (stage, line) => { appendLog(`${mode}/${stage}`, line); if (stage === 'c3780_extract') { observeDetection(result, line); renderDetection(); } },
  });
  current.active = active;
  try { await active.promise; result.runStatus = 'complete'; }
  catch (error) {
    result.runStatus = error.name === 'AbortError' ? 'cancelled' : 'failed'; result.error = error.message;
    if (['AbortError', 'ReceiverBuildError'].includes(error.name)) throw error;
  } finally {
    result.logs = active.logs; result.inputProgress = active.inputProgress; renderDetection();
    endProgress(result.runStatus === 'complete' || result.state === 'not-acquired' ? 'PN check finished' : result.runStatus === 'cancelled' ? 'PN check cancelled' : 'PN check stopped', result.runStatus);
  }
  return result;
}
async function testModes(scan) {
  if (!analysis || running) return;
  let selected;
  try { selected = options(); } catch (error) { status(error.message); return; }
  if (!scan && $('pn').value === 'auto' && !receiverSelection().pn) scan = true;
  const current = operation(); running = current; lock(true);
  const firstMode = selected.pn || 'pn945';
  viewedMode = firstMode;
  if (scan) detections = {};
  renderDetection();
  report.options = selected; $('log').textContent = '';
  try {
    for (const mode of scan ? [firstMode, ...pnModes.filter(mode => mode !== firstMode)] : [firstMode]) await testMode(selected, mode, current);
    if (scan) {
      $('progress-status').dataset.state = 'complete';
      $('progress-status').textContent = `PN scan finished · ${pnModes.filter(mode => detections[mode]?.state !== 'not-tested').length}/3 acquisition results available.`;
      const resolved = receiverSelection().pn;
      viewedMode = resolved && detections[resolved]?.state === 'acquired' ? resolved : pnModes.find(mode => detections[mode]?.state === 'acquired') || firstMode;
      renderDetection();
    }
    report.options = options();
    status(scan ? `PN scan complete. ${detectionSummary(detections[viewedMode], viewedMode)} ${selectionText()}` : detectionSummary(detections[firstMode], firstMode));
  } catch (error) { status(error.message); }
  finally { running = null; lock(false); }
}
$('test-pn').onclick = () => testModes(false);
$('scan-pn').onclick = () => testModes(true);
$('decode').onclick = async () => {
  if (!analysis || running) return;
  let selected;
  try { selected = options(true); } catch (error) { status(error.message); return; }
  $('ts-info').hidden = true; $('ts-details').hidden = true;
  $('log').textContent = ''; $('download').hidden = true; $('transport').textContent = '';
  $('stage-fec').textContent = $('stage-ts').textContent = 'Not tested';
  $('stage-fec').className = $('stage-ts').className = '';
  badge('receiver-badge', 'DECODING'); status('Loading receiver and acquiring PN. The first output may take a while.');
  report.options = selected; report.receiver = { status: 'running' };
  const current = operation(); running = current; lock(true);
  viewedMode = selected.pn; renderDetection();
  let decoder, fecStarted = false, cleanOutput = false;
  try {
    if (!canReuseDetection(detections[selected.pn], boundedProbeOptions(file, selected))) await testMode(selected, selected.pn, current);
    if (current.cancelled) throw new DOMException('Receiver operation cancelled.', 'AbortError');
    status(`${detectionSummary(detections[selected.pn], selected.pn)} Decoding with configured profile ${selected.profile}.`);
    const receiverDetection = createDetection(selected, 'decode');
    beginProgress('Selected decode interval');
    $('stage-fec').textContent = 'Waiting for symbols';
    decoder = decode(file, selected, {
      progress: updateProgress,
      activity: stage => { if (stage === 'ldpc_bch_decode' && !fecStarted) { fecStarted = true; $('stage-fec').textContent = 'Decoding'; } },
      log: (stage, line) => { appendLog(stage, line); if (stage === 'c3780_extract') observeDetection(receiverDetection, line); },
      output: bytes => { cleanOutput = true; $('stage-fec').textContent = 'Clean output received'; status(`Recovered ${(bytes / 1024).toFixed(1)} KiB of transport. Processing continues…`); },
    });
    current.active = decoder;
    report.receiver.detection = receiverDetection;
    const result = await decoder.promise;
    $('stage-ts').textContent = 'Checking packets';
    $('progress').removeAttribute('value'); $('progress-status').textContent = 'Checking recovered transport packets…';
    const transport = await checkTransport(result.blob, current);
    report.receiver = { status: 'complete', bytes: result.bytes, logs: result.logs, transport, detection: receiverDetection, inputProgress: decoder.inputProgress };
    endProgress('Decode finished', 'complete');
    $('stage-fec').textContent = 'Clean output received'; $('stage-fec').className = 'done';
    const transportFailed = transport.syncErrors || transport.transportErrors || transport.trailingBytes || transport.pcrErrors;
    $('stage-ts').textContent = transportFailed ? 'Packet checks failed' : 'Packet checks passed';
    if (outputURL) URL.revokeObjectURL(outputURL);
    outputURL = URL.createObjectURL(result.blob); $('download').href = outputURL; $('download').download = `${file.name.replace(/\.[^.]+$/, '')}.ts`; $('download').hidden = false;
    renderTransportInfo(transport, result.bytes); $('ts-info').hidden = false;
    badge('receiver-badge', transportFailed ? 'CHECK TRANSPORT' : 'TS RECOVERED', transportFailed ? 'warn' : 'good');
    status(`${detectionSummary(detections[selected.pn], selected.pn)} Recovered ${(result.bytes / 1024).toFixed(1)} KiB from the selected interval.`);
    $('transport').textContent = `${transport.packets.toLocaleString()} packets · ${transport.syncErrors} sync errors · ${transport.transportErrors} transport error flags · ${transport.pcrErrors} PCR syntax errors. Failed FEC groups are omitted with discontinuity markers. Validate playback with a codec decoder for an error-free-video claim.`;
  } catch (error) {
    const cancelled = error.name === 'AbortError';
    report.receiver = { ...report.receiver, status: cancelled ? 'cancelled' : 'failed', error: error.message, logs: decoder?.logs, inputProgress: decoder?.inputProgress };
    if (decoder) $('stage-fec').textContent = cleanOutput ? 'Partial clean output; stopped' : !fecStarted ? 'Not reached' : cancelled ? 'Cancelled' : 'No clean output';
    $('stage-ts').textContent = 'Not validated';
    endProgress(cancelled ? 'Decode cancelled' : 'Decode stopped', cancelled ? 'cancelled' : 'failed');
    badge('receiver-badge', error.name === 'AbortError' ? 'CANCELLED' : 'NO DECODE', 'warn');
    const noTransport = /[Nn]o (?:clean transport|transport bytes) recovered/.test(error.message);
    const detail = noTransport ? `No clean transport recovered.${detections[selected.pn]?.state === 'acquired' ? ' Check the configured profile or try a longer interval.' : ''}` : error.message;
    status(`${detectionSummary(detections[selected.pn], selected.pn)} ${detail}`);
  } finally { running = null; lock(false); }
};
$('cancel').onclick = () => running?.cancel();
$('ts-info').onclick = () => { $('ts-details').hidden = false; $('ts-details').open = true; };
function renderTransportInfo(transport, bytes) {
  const root = $('ts-content'), metadata = transport.metadata;
  root.replaceChildren();
  const summary = document.createElement('p'); summary.className = 'hint';
  summary.textContent = `${bytes.toLocaleString()} bytes · ${transport.packets.toLocaleString()} packets · ${metadata.programs.length} programs${metadata.transportStreamId !== null ? ` · transport stream ID ${metadata.transportStreamId}` : ''}.`;
  root.append(summary);
  for (const program of metadata.programs) {
    const heading = document.createElement('h3'); heading.className = 'pn-section-title';
    const pid = value => value === null ? '—' : `0x${value.toString(16).padStart(4, '0')}`;
    heading.textContent = `Program ${program.number} · PMT ${pid(program.pmtPid)} · PCR ${pid(program.pcrPid)}`;
    root.append(heading);
    if (!program.streams.length) {
      const note = document.createElement('p'); note.className = 'hint'; note.textContent = 'No complete CRC-valid PMT received for this program.'; root.append(note);
    } else {
      const table = document.createElement('table'); table.className = 'ts-table';
      const header = document.createElement('tr');
      for (const name of ['PID', 'Type', 'Declaration', 'Language', 'Packets']) { const cell = document.createElement('th'); cell.textContent = name; header.append(cell); }
      table.append(header);
      for (const stream of program.streams) {
        const row = document.createElement('tr');
        for (const value of [pid(stream.pid), stream.kind, `${stream.description} (0x${stream.streamType.toString(16).padStart(2, '0')})`, stream.language || '—', stream.packets.toLocaleString()]) {
          const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
        }
        table.append(row);
      }
      root.append(table);
    }
  }
  if (!metadata.programs.length) {
    const note = document.createElement('p'); note.className = 'hint';
    note.textContent = 'No complete CRC-valid PAT/PMT program information found in the recovered interval.'; root.append(note);
  }
  const pids = document.createElement('p'); pids.className = 'hint';
  pids.textContent = `Observed PIDs: ${metadata.pids.slice(0, 64).map(entry => `0x${entry.pid.toString(16).padStart(4, '0')} (${entry.packets.toLocaleString()} packets)`).join(' · ')}${metadata.pids.length > 64 ? ' · remaining PIDs are included in the report' : ''}.`;
  root.append(pids);
}
async function checkTransport(blob, current) {
  const reader = blob.stream().getReader(); let tail = new Uint8Array(), packets = 0, syncErrors = 0, transportErrors = 0, pcrErrors = 0;
  const metadata = new TransportMetadata();
  while (true) {
    const { value, done } = await reader.read();
    if (current.cancelled) { await reader.cancel(); throw new DOMException('Receiver operation cancelled.', 'AbortError'); }
    if (done) break;
    const bytes = new Uint8Array(tail.length + value.length); bytes.set(tail); bytes.set(value, tail.length);
    const end = bytes.length - bytes.length % 188;
    for (let i = 0; i < end; i += 188) { packets++; syncErrors += bytes[i] !== 0x47 ? 1 : 0; transportErrors += bytes[i + 1] & 0x80 ? 1 : 0; const packet = bytes.subarray(i, i + 188); pcrErrors += pcrSyntaxError(packet) ? 1 : 0; metadata.add(packet); }
    tail = bytes.slice(end);
  }
  return { packets, syncErrors, transportErrors, pcrErrors, trailingBytes: tail.length, metadata: metadata.result(), scope: 'Packet alignment, TEI and PCR syntax checks; not continuity, clock ordering or codec validation.' };
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
  if (rateGuess) { rateGuess = null; $('rate').value = ''; }
  const fields = { format: $('format'), rate: $('rate'), center: $('center'), start: $('start'), duration: $('duration'), pn: $('pn'), profile: $('profile'), tracking: $('tracking') };
  const errors = applyEntry(entry, fields);
  $('rate-status').textContent = entry.sample_rate_sps ? 'Sample rate supplied by sample metadata. You can edit it.' : 'Enter the recording rate, or guess common rates from PN.';
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
lock(false);
