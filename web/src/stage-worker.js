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
    const { default: create } = await import(`./wasm/${data.stage}.mjs`);
    let exitCode = 0;
    const module = await create({
      noInitialRun: true,
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
    module.callMain(data.args);
    if (module.Asyncify.currData) await module.Asyncify.whenDone();
    self.postMessage({ type: 'done', code: exitCode });
  } catch (error) {
    self.postMessage({ type: 'error', message: error?.message || String(error) });
  }
};
