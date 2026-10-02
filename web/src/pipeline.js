import { boundedProbeOptions } from './pn-detection.js';

export function profile(index) {
  if (!Number.isInteger(index) || index < 3 || index > 24) throw Error('Choose a supported profile (3–24).');
  const qam = index < 5 ? '4qam-nr' : index < 11 ? '4qam' : index < 17 ? '16qam' : index < 19 ? '32qam' : '64qam';
  const rate = index < 5 ? 3 : index < 17 ? Math.floor((index - (index < 11 ? 5 : 11)) / 2) + 1 : index < 19 ? 3 : Math.floor((index - 19) / 2) + 1;
  return { qam, rate, mode: index % 2 ? 'mode1' : 'mode2', words: { '4qam-nr': 1, '4qam': 1, '16qam': 2, '32qam': 5, '64qam': 3 }[qam] };
}

export function commands(options) {
  const p = profile(options.profile);
  if (!Number.isSafeInteger(options.rate) || options.rate < 1 || options.rate > 4294967295) throw Error('Sample rate must be a positive integer in samples/second.');
  if (!['pn420', 'pn595', 'pn945'].includes(options.pn)) throw Error('Choose a PN mode.');
  const qam = ['--qam', p.qam];
  const stages = [
    { stage: 'ci8_resample', args: ['--input-rate', `${options.rate}`, '--output-rate', '7560000', '--input-format', options.format, '--output-format', 'cf32', '--workers', '1', '--chunk-samples', '65536', '-', '-'] },
    { stage: 'c3780_extract', args: ['--input-format', 'cf32', '--auto-sync', '--sync-frames', '300', '--acquisition-frames', '16', '--workers', '1', '--batch-frames', '8', '--pn-mode', options.pn, ...qam, '--system-info-index', `${options.profile}`, '--equalizer', 'pn', '--pn-estimator', 'wideband', '--pn-wideband-block-frames', '2', '--pn-wideband-header-observation', options.pn === 'pn595' ? 'direct' : 'core-postfix', '--pn-wideband-scale-estimator', 'masked-frame-taps', '--pn-mmse', '0.004', '--remove-dc', '--normalization', 'qam', '--timing-search-radius', '2', '--timing-search-threshold', '0.45', ...(options.tracking && options.pn !== 'pn595' ? ['--pn-schedule-tracking', '--pn-current-header-tracking'] : []), '-', '-'] },
    { stage: 'deinterleave_qam64', args: [...qam, '--mode', p.mode, '--phase', '0', '--workers', '1', '--chunk-symbols', '65536', '-', '-'] },
    { stage: 'ldpc_bch_decode', alist: p.rate, args: ['--fec-rate', `${p.rate}`, '--alist', '/code.alist', ...qam, '--codewords-per-frame', `${p.words}`, ...(p.qam === '4qam-nr' ? ['--nr-frame-phase', 'auto'] : []), '--workers', '1', '--decode-batch-frames', '8', '--max-iterations', '100', '--retry-max-iterations', '50', '--attenuation', '0.65', '--clean-frames-only', '--require-output', '--insert-discontinuity-packets', '-', '-'] },
  ];
  stages[1].args.splice(stages[1].args.length - 2, 0, '--frame-body-mode', options.rateGuess ? 'c3780' : options.bodyMode ?? 'auto');
  return stages;
}

export function probeCommands(options) {
  const stages = commands(options).slice(0, 2);
  const args = stages[1].args;
  args[args.indexOf('--system-info-index') + 1] = 'auto';
  if (options.rateGuess) args[args.indexOf('--sync-frames') + 1] = '32';
  args.splice(args.length - 2, 0, '--max-frames', options.rateGuess ? '1' : '32');
  return stages;
}

export function probe(file, options, handlers = {}) {
  const selected = boundedProbeOptions(file, options);
  return { ...run(file, selected, handlers, probeCommands(selected), true), options: selected };
}

export function decode(file, options, handlers = {}) {
  return run(file, options, handlers, commands(options));
}

function run(file, options, handlers, specs, discardOutput = false) {
  let position = options.startByte || 0;
  const end = options.endByte ?? file.size;
  let settled = false, total = 0;
  const chunks = [], workers = [], channels = [], logs = {};
  const inputProgress = { readBytes: 0, totalBytes: end - position, fraction: 0 };
  let finishRun;
  const promise = new Promise((resolve, reject) => {
    let timer;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      workers.forEach(w => w.terminate());
      error ? reject(error) : resolve({ blob: new Blob(chunks, { type: 'video/mp2t' }), bytes: total, logs });
    };
    finishRun = finish;
    if (discardOutput) {
      const timeout = options.probeTimeoutMs ?? 30000;
      timer = setTimeout(() => finish(Error(`PN probe reached its ${timeout / 1000}-second time limit. Results may be incomplete.`)), timeout);
    }
    const reply = (worker, id, bytes) => worker.postMessage({ type: 'reply', id, bytes: bytes?.buffer }, bytes ? [bytes.buffer] : []);
    const pump = i => {
      const channel = channels[i];
      if (!channel.read) return;
      if (channel.data) {
        const { id, capacity } = channel.read;
        const size = Math.min(capacity, channel.data.length - channel.offset);
        const bytes = channel.data.slice(channel.offset, channel.offset + size);
        handlers.activity?.(specs[i + 1].stage);
        channel.offset += size;
        reply(workers[i + 1], id, bytes);
        channel.read = null;
        if (channel.offset === channel.data.length) {
          channel.data = null;
          reply(workers[i], channel.writeId);
        }
      } else if (channel.eof) {
        reply(workers[i + 1], channel.read.id, new Uint8Array()); channel.read = null;
      }
    };
    for (let i = 0; i < specs.length; i++) {
      const worker = new Worker(new URL('./stage-worker.js', import.meta.url), { type: 'module' });
      workers.push(worker); channels.push({ offset: 0, eof: false }); logs[specs[i].stage] = [];
      worker.onerror = event => finish(Error(event.message || 'Receiver worker failed.'));
      worker.onmessage = async ({ data }) => {
        if (settled) return;
        try {
          if (data.type === 'read') {
            if (i === 0) {
              const limit = Math.min(end, position + data.capacity);
              const bytes = new Uint8Array(await file.slice(position, limit).arrayBuffer());
              if (settled) return;
              position = limit;
              inputProgress.readBytes += bytes.length;
              inputProgress.fraction = inputProgress.readBytes / inputProgress.totalBytes;
              reply(worker, data.id, bytes);
              handlers.progress?.(inputProgress.fraction);
            } else { channels[i - 1].read = data; pump(i - 1); }
          } else if (data.type === 'write') {
            if (i === specs.length - 1) {
              total += data.bytes.byteLength;
              if (total > 256 * 1024 * 1024) throw Error('Output reached 256 MiB. Select a shorter capture interval.');
              if (!discardOutput) chunks.push(data.bytes);
              reply(worker, data.id);
              handlers.output?.(total);
            } else {
              channels[i].data = new Uint8Array(data.bytes);
              channels[i].offset = 0; channels[i].writeId = data.id; pump(i);
            }
          } else if (data.type === 'log') {
            const lines = logs[specs[i].stage];
            if (lines.length < 2000) lines.push(data.line);
            handlers.log?.(specs[i].stage, data.line);
          } else if (data.type === 'error') {
            const error = Error(`${specs[i].stage}: ${data.message}`);
            if (data.name) error.name = data.name;
            finish(error);
          }
          else if (data.type === 'done') {
            if (data.code !== 0) {
              if (data.code === 2 && logs[specs[i].stage].some(line => line.startsWith('usage: '))) {
                const error = Error('Receiver build is incompatible with the app. Refresh the page or rebuild the web app; this is not a sample-rate or signal-quality failure.');
                error.name = 'ReceiverBuildError';
                return finish(error);
              }
              const nativeError = logs[specs[i].stage].findLast(line => line.startsWith(`dtmb_core_${specs[i].stage}: `));
              return finish(Error(nativeError || `${specs[i].stage} exited ${data.code}. See receiver log.`));
            }
            channels[i].eof = true; handlers.stage?.(i);
            if (i < specs.length - 1) pump(i);
            else if (!total && !discardOutput) finish(Error('No clean transport recovered. Check the input format, sample rate, PN mode and transmission profile.'));
            else finish();
          }
        } catch (error) { finish(error); }
      };
    }
    workers.forEach((worker, i) => worker.postMessage({ type: 'run', ...specs[i] }));
  });
  return { promise, cancel() {
    finishRun(new DOMException('Receiver operation cancelled.', 'AbortError'));
  }, logs, inputProgress };
}
