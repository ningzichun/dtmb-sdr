import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cliRecipe } from '../src/cli-recipe.js';

test('CLI recipes preserve resolved receiver settings and pair CUDA with its pip extra', () => {
  const options = { format: 'cf32', rate: 12500000, pn: 'pn595', profile: 10, bodyMode: 'auto', tracking: true };
  const file = { name: 'capture.cf32', size: 1000 };
  const cpu = cliRecipe(file, options);
  assert.equal(cpu.ready, true);
  assert.equal(cpu.installation, 'python -m pip install dtmb-sdr');
  assert.match(cpu.command, /--input-format cf32 --input-rate 12500000 --pn-mode pn595 --system-info-index 10 --frame-body-mode auto/);
  assert.match(cpu.command, /--acceleration cpu/);
  assert.match(cpu.command, /--max-iterations 100 --error-policy continue/);
  assert.doesNotMatch(cpu.command, /--pn-schedule-tracking/);
  const cuda = cliRecipe(file, { ...options, pn: 'pn420' }, { cuda: true });
  assert.match(cuda.installation, /'dtmb-sdr\[cuda\]'/);
  assert.match(cuda.command, /--acceleration cuda/);
  assert.match(cuda.command, /--pn-schedule-tracking/);
  assert.equal(cliRecipe(file, { ...options, pn: null }).ready, false);
  assert.equal(cliRecipe(file, { ...options, profile: null }).command, '');
});
