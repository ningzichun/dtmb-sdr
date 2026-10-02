// The native executables run unchanged inside one worker per pipeline stage.
// A demand/ack protocol bounds each connection to a single transfer buffer.
let pending = new Map(), sequence = 0;
self.onmessage = async ({ data }) => {
  if (data.type === 'reply') {
    const resume = pending.get(data.id);
    pending.delete(data.id);
    resume?.(data.bytes ? new Uint8Array(data.bytes) : undefined);
    return;
  }
  if (data.type !== 'run') return;
  const request = (type, bytes, capacity) => new Promise(resolve => {
    const id = ++sequence;
    pending.set(id, resolve);
    self.postMessage({ type, id, bytes: bytes?.buffer, capacity }, bytes ? [bytes.buffer] : []);
  });
  try {
    const manifestResponse = await fetch(new URL('./build.json', import.meta.url), { cache: 'no-store' });
    if (!manifestResponse.ok) throw Error(`Receiver build manifest unavailable (${manifestResponse.status}). Refresh the page or rebuild the web app.`);
    const manifest = await manifestResponse.json(), expectedHash = manifest.wasm?.[`${data.stage}.wasm`];
    if (!expectedHash) throw Error('Receiver build manifest is incomplete. Refresh the page or rebuild the web app.');
    const wasmURL = new URL(`./wasm/${data.stage}.wasm`, import.meta.url);
    wasmURL.searchParams.set('v', expectedHash);
    const wasmResponse = await fetch(wasmURL, { cache: 'no-store' });
    if (!wasmResponse.ok) throw Error(`Receiver binary unavailable (${wasmResponse.status}). Refresh the page or rebuild the web app.`);
    const wasmBinary = await wasmResponse.arrayBuffer();
    const actualHash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', wasmBinary))].map(value => value.toString(16).padStart(2, '0')).join('');
    if (actualHash !== expectedHash) throw Error('Receiver asset version mismatch. Refresh the page or rebuild the web app; this is not a capture or sample-rate failure.');
    const loaderURL = new URL(`./wasm/${data.stage}.mjs`, import.meta.url);
    loaderURL.searchParams.set('v', manifest.loaders?.[`${data.stage}.mjs`] ?? expectedHash);
    const { default: create } = await import(loaderURL.href);
    let exitCode = 0;
    const module = await create({
      noInitialRun: true,
      wasmBinary,
      print: line => self.postMessage({ type: 'log', line }),
      printErr: line => self.postMessage({ type: 'log', line }),
      onExit: code => { exitCode = code; },
      dtmbRead: size => request('read', undefined, size),
      dtmbWrite: bytes => request('write', bytes),
    });
    if (data.alist) {
      const response = await fetch(`./data/dtmb_ldpc_rate${data.alist}.alist`);
      if (!response.ok) throw Error(`LDPC data unavailable (${response.status})`);
      module.FS.writeFile('/code.alist', new Uint8Array(await response.arrayBuffer()));
    }
    const initialCode = module.callMain(data.args);
    const resultCode = module.Asyncify.currData ? await module.Asyncify.whenDone() : initialCode;
    if (Number.isInteger(resultCode)) exitCode = resultCode;
    self.postMessage({ type: 'done', code: exitCode });
  } catch (error) {
    self.postMessage({ type: 'error', message: error?.message || String(error),
      name: /Receiver (?:asset|build|binary)/.test(error?.message || '') ? 'ReceiverBuildError' : error?.name });
  }
};
