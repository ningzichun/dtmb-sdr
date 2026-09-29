import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseManifest, applyEntry, pickEntry } from '../src/samples.js';

const page = `<h1>DTMB samples</h1>
<p>Web app reads this table.</p>
name | url | archive_entry | format | sample_rate_sps | center_hz | pn_mode | profile | start_s | duration_s | tracking | notes | extra
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
Sample 1 | https://example.com/a.zip | sample.ci8 | ci8 | 16000000 | 538000000 | pn945 | 22 | 0 | 0 | true | Remod | ignored
| table noise |
plain text | not a table
`;

test('manifest table parses rows and ignores unknown columns and noise', () => {
  const [entry] = parseManifest(page);
  assert.equal(entry.name, 'Sample 1');
  assert.equal(entry.url, 'https://example.com/a.zip');
  assert.equal(entry.profile, '22');
  assert.equal(entry.extra, 'ignored');
  assert.throws(() => parseManifest('no table here'), /No samples/);
});

test('entry values map onto form fields with validation', () => {
  const select = options => ({ options: options.map(value => ({ value })), value: '' });
  const fields = { format: select(['ci8', 'cu8']), rate: {}, center: {}, start: {}, duration: {}, pn: select(['pn945']), profile: {}, tracking: { checked: false } };
  const errors = applyEntry(parseManifest(page)[0], fields);
  assert.deepEqual(errors, []);
  assert.equal(fields.rate.value, 16000000);
  assert.equal(fields.center.value, 538);
  assert.equal(fields.pn.value, 'pn945');
  assert.equal(fields.profile.value, 22);
  assert.equal(fields.tracking.checked, true);
  const bad = applyEntry({ name: 'x', url: 'http://insecure', format: 'bogus', pn_mode: 'pn999', profile: '99', sample_rate_sps: '-1' }, fields);
  assert.equal(bad.length, 5);
});

test('zip entry selection honours archive_entry or finds the IQ file', () => {
  const names = ['dir/notes.txt', 'dir/sample-25m.ci8', 'dir/other.cf32'];
  assert.equal(pickEntry(names, 'sample-25m.ci8'), 'dir/sample-25m.ci8');
  assert.equal(pickEntry(names, ''), 'dir/sample-25m.ci8');
  assert.throws(() => pickEntry(names, 'missing.ci8'), /does not contain/);
  assert.throws(() => pickEntry(['readme.md'], ''), /No IQ file/);
});
