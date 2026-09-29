let library;
export async function preview(url, video, onStatus) {
  library ??= new Promise((resolve, reject) => {
    const script = document.createElement('script'); script.src = './vendor/mpegts.js';
    script.onload = resolve; script.onerror = () => reject(Error('Unable to load the local MPEG-TS player.'));
    document.head.append(script);
  });
  await library;
  const mpegts = globalThis.mpegts;
  if (!mpegts?.isSupported()) throw Error('This browser cannot play MPEG-TS through Media Source. Download the TS for playback.');
  const player = mpegts.createPlayer({ type: 'mpegts', isLive: false, url }, { enableWorker: false, lazyLoad: false });
  player.on(mpegts.Events.ERROR, (type, detail) => onStatus(`Preview unavailable (${type}: ${detail}). The broadcast codec may not be supported by this browser. Download the original TS to inspect it.`));
  player.attachMediaElement(video); player.load();
  const started = video.play();
  started?.catch(error => onStatus(`Preview could not start: ${error.message}`));
  return () => { player.pause(); player.unload(); player.detachMediaElement(); player.destroy(); };
}
