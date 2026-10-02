import { formats } from './diagnostics.js';
import { probe } from './pipeline.js';
import { pnModes, createDetection, observeDetection } from './pn-detection.js';

export const commonSampleRates = [
  7560000, 7680000, 8000000, 8192000, 9600000, 10000000,
  10240000, 11520000, 12000000, 12288000, 12500000, 15120000,
  15360000, 16000000, 16384000, 19200000, 20000000, 24000000,
  25000000, 30720000, 32000000, 40000000, 50000000, 61440000,
];

export function rateGuessOptions(file, options, rate) {
  const width = formats[options.format], start = options.start ?? 0, duration = options.duration ?? 0;
  if (!width) throw Error('Choose a supported input format before guessing the rate.');
  if (file.size % width) throw Error(`File size is not divisible by ${width} bytes per complex sample.`);
  if (!Number.isSafeInteger(rate) || rate <= 0 || rate > 4294967295) throw Error('Sample rate must be a positive integer.');
  if (![start, duration].every(Number.isFinite) || start < 0 || duration < 0) throw Error('Start and duration must be finite, nonnegative values.');
  const startByte = Math.floor(start * rate) * width;
  const samples = Math.min(Math.floor(rate * Math.min(duration || 0.04, 0.04)), Math.floor(16 * 1024 * 1024 / width));
  const endByte = Math.min(file.size - file.size % width, startByte + samples * width);
  if (endByte <= startByte) throw Error('The selected interval is outside the recording.');
  return { rate, format: options.format, startByte, endByte, pn: 'pn945', profile: 21,
    tracking: false, rateGuess: true, probeTimeoutMs: 3000 };
}

function strongAcquisition(result) {
  const native = result?.native || {}, hits = Number(native.acquisition_hit_count);
  const observed = Number(native.acquisition_observed_frames), metric = Number(native.acquisition_mean_metric);
  const threshold = Number(native.sync_hit_threshold);
  return result?.state === 'acquired' && result.runStatus === 'complete'
    && Number.isInteger(hits) && Number.isInteger(observed) && observed >= 16
    && hits <= observed && hits / observed >= 0.75
    && Number.isFinite(metric) && Number.isFinite(threshold) && threshold > 0 && metric >= threshold;
}

export function selectRateGuess(candidates) {
  const complete = commonSampleRates.every(rate => pnModes.every(mode => {
    const result = candidates.find(candidate => candidate.rate === rate)?.detections?.[mode];
    return result?.runStatus === 'complete' && result.state !== 'not-tested'
      || result?.runStatus === 'failed' && result.state === 'not-acquired';
  }));
  const acquired = candidates.flatMap(candidate => {
    const matches = pnModes.filter(mode => strongAcquisition(candidate.detections?.[mode]))
      .map(mode => ({ mode, metric: Number(candidate.detections[mode].native.acquisition_mean_metric) }))
      .sort((first, second) => second.metric - first.metric);
    return matches.length ? [{ rate: candidate.rate, ...matches[0] }] : [];
  }).sort((first, second) => second.metric - first.metric);
  if (!complete) return { rate: null, complete, reason: 'Rate guess incomplete. Enter the known rate or retry.' };
  if (!acquired.length) return { rate: null, complete, reason: 'No common rate confirmed by PN. Check input format, enter the known sample rate, or check signal quality.' };
  const best = acquired[0], runnerUp = acquired[1], margin = runnerUp ? best.metric - runnerUp.metric : null;
  if (runnerUp && (margin < 0.1 || best.metric < runnerUp.metric * 1.5)) return { rate: null, complete,
    reason: `Multiple sample rates match PN (${acquired.map(candidate => (candidate.rate / 1e6).toFixed(3)).join(', ')} MS/s). Enter the known rate; no guess was applied.` };
  return { ...best, margin, runnerUpMetric: runnerUp?.metric ?? null, complete, reason: 'Guessed from PN acquisition, not capture metadata.' };
}

export function guessSampleRate(file, options, handlers = {}) {
  rateGuessOptions(file, options, commonSampleRates[0]);
  let active, cancelled = false, timedOut = false;
  const evidence = { candidates: [], format: options.format, testedRates: [...commonSampleRates],
    limits: { secondsPerProbe: 0.04, bytesPerProbe: 16 * 1024 * 1024, millisecondsPerProbe: 3000, totalMilliseconds: 45000 } };
  const timer = setTimeout(() => { timedOut = true; active?.cancel(); }, evidence.limits.totalMilliseconds);
  const promise = (async () => {
    try {
      for (const rate of commonSampleRates) {
        const candidate = { rate, detections: {} };
        evidence.candidates.push(candidate);
        let selected;
        try { selected = rateGuessOptions(file, options, rate); }
        catch (error) { candidate.error = error.message; continue; }
        for (const mode of pnModes) {
          if (cancelled) throw new DOMException('Sample-rate guess cancelled.', 'AbortError');
          if (timedOut) throw Error('Sample-rate guess reached its 45-second limit. Enter the known rate or retry.');
          const result = createDetection({ ...selected, pn: mode }, 'rate-guess');
          candidate.detections[mode] = result;
          handlers.candidate?.(rate, mode);
          active = probe(file, { ...selected, pn: mode }, {
            progress: handlers.progress,
            log: (stage, line) => {
              handlers.log?.(`${rate}/${mode}/${stage}`, line);
              if (stage === 'c3780_extract') observeDetection(result, line);
            },
          });
          try { await active.promise; result.runStatus = 'complete'; }
          catch (error) {
            result.runStatus = error.name === 'AbortError' ? 'cancelled' : 'failed';
            result.error = error.message;
            if (timedOut) throw Error('Sample-rate guess reached its 45-second limit. Enter the known rate or retry.');
            if (cancelled || ['AbortError', 'ReceiverBuildError'].includes(error.name)) throw error;
          } finally { result.inputProgress = active.inputProgress; result.logs = active.logs; }
        }
      }
      if (cancelled) throw new DOMException('Sample-rate guess cancelled.', 'AbortError');
      return { ...evidence, ...selectRateGuess(evidence.candidates) };
    } finally { clearTimeout(timer); }
  })();
  return { promise, evidence, cancel() { cancelled = true; active?.cancel(); } };
}
