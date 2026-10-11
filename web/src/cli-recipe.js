import { formats } from './diagnostics.js';
import { pnModes } from './pn-detection.js';

export function shellQuote(value, shell = 'bash') {
  if (!['bash', 'powershell'].includes(shell)) throw Error('Choose Bash or PowerShell.');
  const text = String(value);
  if (text.includes('\0')) throw Error('Paths cannot contain a null character.');
  if (/^[a-zA-Z0-9_./:-]+$/.test(text)) return text;
  return shell === 'powershell' ? `'${text.replaceAll("'", "''")}'` : `'${text.replaceAll("'", "'\"'\"'")}'`;
}

export function cliRecipe(file, options, settings = {}) {
  const { shell = 'bash', cuda = false, install = true } = settings;
  const quote = value => shellQuote(value, shell);
  const join = args => args.map(quote).join(' ');
  const installation = join(['python', '-m', 'pip', 'install', cuda ? 'dtmb-sdr[cuda]' : 'dtmb-sdr']);
  const input = settings.input?.trim() || file?.name || 'capture.ci8';
  const output = settings.output?.trim() || 'recovered.ts';
  const waiting = reason => ({ installation, command: '', recipe: install ? installation : '', ready: false, reason });
  if (!file) return waiting('Open a capture to assemble its equivalent command.');
  if (!options) return waiting('Enter or guess the sample rate, then confirm the receiver settings.');
  if (!Number.isSafeInteger(options.rate) || options.rate <= 0 || options.rate > 4294967295) return waiting('Enter a valid sample rate.');
  if (!pnModes.includes(options.pn) || !Number.isInteger(options.profile) || options.profile < 3 || options.profile > 24) {
    return waiting('Inspect the capture to resolve Auto PN/profile, or select known settings explicitly.');
  }
  if (!formats[options.format]) return waiting('Choose a supported I/Q sample format.');
  const samePath = (first, second) => first.replaceAll('\\', '/').toLowerCase() === second.replaceAll('\\', '/').toLowerCase();
  if (samePath(input, output)) return waiting('Use different input and output paths.');
  const command = join(['dtmb-decode', '--input', input.startsWith('-') ? `./${input}` : input,
    '--input-format', options.format, '--input-rate', options.rate, '--pn-mode', options.pn,
    '--system-info-index', options.profile, '--frame-body-mode', options.bodyMode ?? 'auto',
    '--acceleration', cuda ? 'cuda' : 'cpu', '--workers', '1', '--decode-batch-frames', '8',
    '--max-iterations', '100', '--error-policy', 'continue',
    ...(options.tracking && options.pn !== 'pn595' ? ['--pn-schedule-tracking'] : []),
    '--output', output.startsWith('-') ? `./${output}` : output]);
  const analysis = join(['dtmb-ts-analyze', output.startsWith('-') ? `./${output}` : output]);
  return { installation, command, recipe: [install ? installation : '', command, analysis].filter(Boolean).join('\n\n'),
    ready: true, reason: 'Uses the current receiver settings and the entire recording.' };
}
