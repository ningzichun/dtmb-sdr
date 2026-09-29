export const defaultManifest = 'https://raw.githubusercontent.com/wiki/ningzichun/dtmb-sdr/Samples.md';
const iqSuffixes = /\.(ci8|cs8|ci16|sc16|cs16|cu8|cf32)$/i;

// The wiki page is a plain Markdown table. Unknown columns are ignored, so the
// page stays readable for humans while the app only uses the known keys.
export function parseManifest(text) {
  const header = [];
  const samples = [];
  for (const raw of String(text).split('\n')) {
    const line = raw.replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').trim();
    if (!line.includes('|')) continue;
    const cells = line.split('|').map(cell => cell.trim()).filter(Boolean);
    if (!cells.length) continue;
    if (cells.every(cell => /^:?-{2,}:?$/.test(cell))) continue;                 // |---|---| separator
    if (cells[0] === 'name' && cells.includes('url')) {
      for (const cell of cells) if (!header.includes(cell)) header.push(cell);  // column order drives mapping
      continue;
    }
    if (!header.length) continue;
    const entry = {};
    for (const [i, column] of header.entries()) entry[column] = cells[i] ?? '';
    if (!entry.name || !entry.url) continue;
    samples.push(entry);
  }
  if (!samples.length) throw Error('No samples found in the manifest.');
  return samples;
}

export function applyEntry(entry, fields) {
  const errors = [];
  if (!/^https:\/\//i.test(entry.url)) errors.push('Sample URL must use https.');
  const names = { cs8: 'ci8', sc16: 'ci16', cs16: 'ci16' };
  let format = (entry.format || '').trim().toLowerCase();
  if (!format) format = ((entry.archive_entry || entry.url).split('.').pop() || '').toLowerCase();
  const value = names[format] || format;
  if (value) {
    if (![...fields.format.options].some(o => o.value === value)) errors.push(`Unknown sample format "${entry.format}".`);
    else fields.format.value = value;
  }
  for (const [column, id, scale] of [['sample_rate_sps', 'rate', 1], ['center_hz', 'center', 1e-6], ['start_s', 'start', 1], ['duration_s', 'duration', 1]]) {
    const text = (entry[column] || '').trim();
    if (!text) continue;
    const number = Number(text) * scale;
    if (!Number.isFinite(number) || number < 0) { errors.push(`Bad ${column} "${text}".`); continue; }
    fields[id].value = number;
  }
  const pn = (entry.pn_mode || '').trim().toLowerCase();
  if (pn) {
    if (![...fields.pn.options].some(o => o.value === pn)) errors.push(`Unknown pn_mode "${entry.pn_mode}".`);
    else fields.pn.value = pn;
  }
  if ((entry.profile || '').trim()) {
    const number = Number(entry.profile);
    if (!Number.isInteger(number) || number < 5 || number > 24) errors.push(`Bad profile "${entry.profile}".`);
    else fields.profile.value = number;
  }
  const tracking = (entry.tracking || '').trim().toLowerCase();
  if (['true', 'false'].includes(tracking)) fields.tracking.checked = tracking === 'true';
  return errors;
}

export function pickEntry(names, wanted) {
  if (wanted) {
    const match = names.find(name => name === wanted || name.split('/').pop() === wanted);
    if (!match) throw Error(`ZIP does not contain "${wanted}".`);
    return match;
  }
  const iq = names.find(name => iqSuffixes.test(name));
  if (!iq) throw Error('No IQ file (.ci8/.cs8/.ci16/.cf32/.cu8) inside the ZIP.');
  return iq;
}

export async function unzipFiles(buffer) {
  const { unzipSync } = await import('./vendor/fflate.js');
  return unzipSync(new Uint8Array(buffer), { filter: file => !file.name.endsWith('/') });
}

export async function iqFromZip(blob, wanted, onStatus) {
  onStatus?.('Unpacking ZIP…');
  const files = await unzipFiles(await blob.arrayBuffer());
  const name = pickEntry(Object.keys(files), wanted);
  const data = files[name];
  return new File([data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)], name.split('/').pop());
}

export class CorsError extends Error {}

async function download(url, onStatus) {
  const response = await fetch(url).catch(error => { throw new CorsError(error.message); });
  if (!response.ok) throw Error(`Download failed: HTTP ${response.status}`);
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body?.getReader) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = []; let received = 0; const started = Date.now();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value); received += value.length;
    const mb = received / 1048576, secs = (Date.now() - started) / 1000;
    onStatus?.(total
      ? `Downloading ${mb.toFixed(1)} / ${(total / 1048576).toFixed(1)} MiB (${Math.round(100 * received / total)}%, ${secs.toFixed(0)} s)…`
      : `Downloading ${mb.toFixed(1)} MiB (${secs.toFixed(0)} s)…`);
  }
  const data = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { data.set(chunk, offset); offset += chunk.length; }
  return data;
}

export async function fetchFile(url, { entry = '', onStatus } = {}) {
  if (!/^https:\/\//i.test(url)) throw Error('Remote files must use https.');
  const name = decodeURIComponent(url.split('/').pop().split('?')[0]) || 'capture.iq';
  const data = await download(url, onStatus);
  if (!/\.zip$/i.test(name)) return new File([data], name);
  return iqFromZip(new Blob([data]), entry, onStatus);
}

export async function sampleFile(entry, onStatus) {
  return fetchFile(entry.url, { entry: entry.archive_entry, onStatus });
}

