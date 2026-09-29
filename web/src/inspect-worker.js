import { inspect } from './diagnostics.js';
self.onmessage = async ({ data }) => {
  try {
    const result = await inspect(data.file, data.options, (bins, progress) => self.postMessage({ type: 'row', bins, progress }, [bins.buffer]));
    self.postMessage({ type: 'result', result });
  } catch (error) { self.postMessage({ type: 'error', message: error.message }); }
};
