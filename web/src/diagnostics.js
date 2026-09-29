export const formats = { cu8: 2, ci8: 2, cs8: 2, ci16: 4, sc16: 4, cs16: 4, cf32: 8 };
export function readIQ(buffer, format) {
  const width = formats[format];
  if (!width || buffer.byteLength % width) throw Error('Input ends in an incomplete I/Q pair. Check the format or capture export.');
  const view = new DataView(buffer), out = new Float32Array(buffer.byteLength / width * 2);
  for (let n = 0; n < out.length; n++) {
    out[n] = width === 8 ? view.getFloat32(n * 4, true) : width === 4 ? view.getInt16(n * 2, true) / 32768 : format === 'cu8' ? (view.getUint8(n) - 128) / 128 : view.getInt8(n) / 128;
  }
  return out;
}

export function spectrum(iq) {
  const n = iq.length / 2;
  if (n < 2 || n & (n - 1)) throw Error('FFT size must be a power of two.');
  const re = new Float64Array(n), im = new Float64Array(n);
  let norm = 0;
  for (let i = 0; i < n; i++) {
    const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
    norm += w; re[i] = iq[i * 2] * w; im[i] = iq[i * 2 + 1] * w;
  }
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let size = 2; size <= n; size *= 2) {
    const step = -2 * Math.PI / size;
    for (let start = 0; start < n; start += size) {
      for (let j = 0; j < size / 2; j++) {
        const a = start + j, b = a + size / 2, c = Math.cos(j * step), s = Math.sin(j * step);
        const r = re[b] * c - im[b] * s, q = re[b] * s + im[b] * c;
        re[b] = re[a] - r; im[b] = im[a] - q; re[a] += r; im[a] += q;
      }
    }
  }
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const j = (k + n / 2) % n;
    out[k] = 10 * Math.log10(Math.max(1e-14, (re[j] ** 2 + im[j] ** 2) / norm ** 2));
  }
  return out;
}

export function assess({ samples, powerI, powerQ, sumI, sumQ, clipped, nonfinite, minimum, maximum }, rate, format) {
  const rms = Math.sqrt((powerI + powerQ) / Math.max(1, 2 * samples));
  const dcI = sumI / Math.max(1, samples), dcQ = sumQ / Math.max(1, samples);
  const warnings = [];
  if (!samples) warnings.push('No complete samples in this interval.');
  if (nonfinite) warnings.push(`${nonfinite} non-finite components: the selected format may be wrong or the recording is damaged.`);
  if (rms < 1e-6) warnings.push('No measurable signal. Check the file and receiver gain.');
  else if (rms < 0.008) warnings.push('Very low recorded level. Check gain and whether the correct sample format is selected.');
  if (clipped / Math.max(1, samples * 2) > 0.001) warnings.push('More than 0.1% of sampled components touch full scale. Check capture gain and format.');
  if (Math.hypot(dcI, dcQ) > rms * 0.25) warnings.push('Large DC offset or asymmetric bytes. Check format, I/Q export, and receiver DC correction.');
  if (samples && (powerI === 0 || powerQ === 0 || Math.max(powerI, powerQ) > 10 * Math.min(powerI, powerQ))) warnings.push('I/Q power is badly imbalanced. This may be real-only data or an incorrect sample format.');
  if (rate < 7_938_000 && rate !== 7_560_000) warnings.push('This rate may omit part of an 8 MHz DTMB channel. Resampling cannot restore missing bandwidth.');
  return { samples, format, rate, rmsDbfs: 20 * Math.log10(Math.max(rms, 1e-12)), dcI, dcQ, clippedComponents: clipped, clippingPercent: 100 * clipped / Math.max(1, samples * 2), nonfinite, minimum, maximum, warnings };
}

export async function inspect(file, options, onRow = () => {}) {
  const width = formats[options.format];
  if (!width) throw Error('Choose a supported sample format.');
  if (file.size % width) throw Error(`File size is not divisible by ${width} bytes per complex sample.`);
  const start = options.startByte || 0, end = options.endByte ?? file.size;
  const fft = 2048, available = Math.floor((end - start) / width / fft);
  if (!available) throw Error('Choose an interval with at least 2048 complex samples.');
  const rows = Math.min(128, available), windows = [];
  const stats = { samples: 0, powerI: 0, powerQ: 0, sumI: 0, sumQ: 0, clipped: 0, nonfinite: 0, minimum: Infinity, maximum: -Infinity };
  const avgPower = new Float64Array(fft);
  const rail = width === 2 ? 127 / 128 : width === 4 ? 32767 / 32768 : 1;
  for (let row = 0; row < rows; row++) {
    const index = rows === 1 ? 0 : Math.floor(row * (available - 1) / (rows - 1));
    const offset = start + index * fft * width;
    const data = readIQ(await file.slice(offset, offset + fft * width).arrayBuffer(), options.format);
    windows.push({ offset, bytes: data.length / 2 * width });
    for (let n = 0; n < data.length; n += 2) {
      let i = data[n], q = data[n + 1];
      for (const v of [i, q]) {
        if (!Number.isFinite(v)) stats.nonfinite++;
        else { stats.clipped += (v <= -1 || v >= rail) ? 1 : 0; stats.minimum = Math.min(stats.minimum, v); stats.maximum = Math.max(stats.maximum, v); }
      }
      if (!Number.isFinite(i) || !Number.isFinite(q)) { data[n] = data[n + 1] = 0; continue; }
      stats.samples++; stats.powerI += i * i; stats.powerQ += q * q; stats.sumI += i; stats.sumQ += q;
    }
    const bins = spectrum(data);
    for (let k = 0; k < fft; k++) avgPower[k] += 10 ** (bins[k] / 10) / rows;
    onRow(bins, row / Math.max(1, rows - 1));
  }
  const prefix = await file.slice(0, Math.min(file.size, 1048576)).arrayBuffer();
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', prefix))].map(v => v.toString(16).padStart(2, '0')).join('');
  return { ...assess(stats, options.rate, options.format), file: { name: file.name, bytes: file.size, lastModified: file.lastModified, prefixSha256: hash, hashedBytes: prefix.byteLength }, scope: { startByte: start, endByte: end, sampledWindows: windows, note: 'Statistics cover these sampled windows, not the full recording. The receiver makes the final validity call.' }, spectrum: Array.from(avgPower, p => 10 * Math.log10(Math.max(1e-14, p))) };
}
