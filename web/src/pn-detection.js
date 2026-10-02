import { formats } from './diagnostics.js';

export const pnModes = ['pn420', 'pn595', 'pn945'];
export const detectionLabels = { 'not-tested': 'Not tested', 'not-acquired': 'Not acquired', acquired: 'Acquired' };
export const acquisitionAdvice = 'Check input format, sample rate, PN mode, or signal quality.';

export function observedProfileIndex(result) {
  const index = Number(result?.native?.system_info_index);
  return result?.state === 'acquired' && result.systemInformation === 'locked'
    && result.native?.system_info_auto_locked === 'true'
    && Number.isInteger(index) && index >= 3 && index <= 24 ? index : null;
}

export function resolveReceiver(detections, pnSetting = 'auto', profileSetting = 'auto') {
  if (pnSetting !== 'auto' && !pnModes.includes(pnSetting)) throw Error('Choose Auto or a supported PN mode.');
  if (profileSetting !== 'auto' && (!Number.isInteger(Number(profileSetting)) || Number(profileSetting) < 3 || Number(profileSetting) > 24)) throw Error('Choose Auto or a supported transmission profile.');
  let pn = pnSetting === 'auto' ? null : pnSetting, reason = '';
  if (pnSetting === 'auto') {
    const tested = pnModes.every(mode => detections[mode]?.runStatus === 'complete' && detections[mode]?.state === 'acquired'
      || ['complete', 'failed'].includes(detections[mode]?.runStatus) && detections[mode]?.state === 'not-acquired');
    const acquired = pnModes.filter(mode => detections[mode]?.state === 'acquired');
    if (!tested) reason = 'Auto PN is waiting for a complete PN scan.';
    else if (!acquired.length) reason = `No PN acquired. ${acquisitionAdvice}`;
    else if (acquired.length === 1) pn = acquired[0];
    else {
      const locked = acquired.filter(mode => observedProfileIndex(detections[mode]) !== null);
      if (locked.length === 1) pn = locked[0];
      else reason = 'Multiple PN modes acquired. Select a PN mode explicitly.';
    }
  }
  const profile = profileSetting === 'auto' ? observedProfileIndex(detections[pn]) : Number(profileSetting);
  if (pn && profile === null) reason = 'System information is not locked. Recheck the signal or select a known profile explicitly.';
  return { pn, profile, ready: pn !== null && profile !== null, reason,
    pnSource: pnSetting === 'auto' ? 'auto' : 'forced', profileSource: profileSetting === 'auto' ? 'auto' : 'forced' };
}

export function boundedProbeOptions(file, options) {
  const width = formats[options.format];
  if (!width) throw Error('Choose a supported input format.');
  const startByte = options.startByte || 0;
  const sampleLimit = Math.min(Math.floor(options.rate * 0.2), Math.floor(16 * 1024 * 1024 / width));
  const endByte = Math.min(file.size, options.endByte ?? file.size, startByte + sampleLimit * width);
  const alignedEnd = endByte - (endByte - startByte) % width;
  if (alignedEnd <= startByte) throw Error('The selected interval has no complete I/Q samples.');
  return { ...options, startByte, endByte: alignedEnd };
}

export function createDetection(options, source = 'probe') {
  const width = formats[options.format];
  return {
    mode: options.pn, state: 'not-tested', systemInformation: 'not-tested', source,
    configuration: { profile: options.profile, tracking: Boolean(options.tracking), bodyMode: options.bodyMode ?? 'auto' },
    context: { format: options.format, sampleRate: options.rate, startByte: options.startByte,
      endByte: options.endByte, startSeconds: options.startByte / width / options.rate,
      endSeconds: options.endByte / width / options.rate },
    native: {},
  };
}

export function canReuseDetection(result, options) {
  return result?.runStatus === 'complete' && result.state === 'acquired'
    && result.native.system_info_auto_complete === 'true' && result.mode === options.pn
    && result.configuration.profile === options.profile
    && result.configuration.tracking === Boolean(options.tracking)
    && result.configuration.bodyMode === (options.bodyMode ?? 'auto')
    && result.context.format === options.format && result.context.sampleRate === options.rate
    && result.context.startByte === options.startByte && result.context.endByte === options.endByte;
}

export function observeDetection(result, line) {
  const match = /^([a-z][a-z0-9_]*)=(\S+)\s*$/.exec(line);
  if (!match) return;
  result.native[match[1]] = match[2];
  const native = result.native;
  if (native.acquisition_hit_count !== undefined && native.acquisition_observed_frames !== undefined) {
    const hits = Number(native.acquisition_hit_count), observed = Number(native.acquisition_observed_frames);
    if (Number.isInteger(hits) && Number.isInteger(observed) && hits >= 0 && observed > 0 && hits <= observed) {
      result.state = hits >= 2 ? 'acquired' : 'not-acquired';
    }
  }
  if (native.system_info_auto === 'true' && native.system_info_auto_locked !== undefined) {
    result.systemInformation = native.system_info_auto_locked === 'true' ? 'locked' : 'not-locked';
  }
}

export function detectionSummary(result, mode) {
  const name = mode.toUpperCase();
  if (!result || result.state === 'not-tested') return `${name}: Not tested.`;
  if (result.state === 'not-acquired') return `${name} not acquired. ${acquisitionAdvice}`;
  const systemInformation = { locked: 'locked', 'not-locked': 'not locked', 'not-tested': 'not tested' }[result.systemInformation];
  return `${name} acquired; system information ${systemInformation}.`;
}
