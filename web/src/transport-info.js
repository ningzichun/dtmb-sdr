const streamTypes = {
  0x01: ['video', 'MPEG-1 video'], 0x02: ['video', 'MPEG-2 video'],
  0x03: ['audio', 'MPEG-1 audio'], 0x04: ['audio', 'MPEG-2 audio'],
  0x0f: ['audio', 'AAC / ADTS'], 0x10: ['video', 'MPEG-4 Visual'],
  0x11: ['audio', 'AAC / LATM'], 0x1b: ['video', 'H.264 / AVC'],
  0x24: ['video', 'H.265 / HEVC'],
  0x42: ['video', 'AVS video'],
};

export function pcrSyntaxError(packet) {
  if (packet.length !== 188 || packet[0] !== 0x47) return false;
  const control = packet[3] >> 4 & 3;
  if (![2, 3].includes(control) || !packet[4] || !(packet[5] & 0x10)) return false;
  return packet[4] < 7 || packet[4] > 183 || (packet[10] & 0x7e) !== 0x7e
    || ((packet[10] & 1) << 8 | packet[11]) >= 300;
}

export function mpegSectionCrc(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte << 24;
    for (let bit = 0; bit < 8; bit++) crc = crc & 0x80000000 ? (crc << 1) ^ 0x04c11db7 : crc << 1;
  }
  return crc >>> 0;
}

function streamDeclaration(type, descriptors) {
  let [kind, description] = streamTypes[type] || [type === 0x06 ? 'private' : 'unknown', type === 0x06 ? 'Private PES' : 'Unspecified stream'];
  let language;
  let offset = 0;
  while (offset < descriptors.length) {
    if (offset + 2 > descriptors.length) return null;
    const tag = descriptors[offset], length = descriptors[offset + 1];
    offset += 2;
    if (offset + length > descriptors.length) return null;
    if (tag === 0x0a && length >= 4 && length % 4 === 0) language = String.fromCharCode(...descriptors.subarray(offset, offset + 3));
    if (type === 0x06 && tag === 0x6a && length >= 1) [kind, description] = ['audio', 'AC-3'];
    if (type === 0x06 && tag === 0x7a && length >= 1) [kind, description] = ['audio', 'Enhanced AC-3'];
    if (type === 0x06 && tag === 0x59 && length >= 8 && length % 8 === 0) [kind, description] = ['data', 'DVB subtitles'];
    offset += length;
  }
  return { kind, description, language };
}

export class TransportMetadata {
  constructor() {
    this.assemblies = new Map(); this.pids = new Map(); this.pmts = new Map();
    this.pat = null; this.programs = new Map(); this.transportStreamId = null;
  }

  add(packet) {
    if (packet.length !== 188 || packet[0] !== 0x47) return;
    const pid = ((packet[1] & 0x1f) << 8) | packet[2];
    const counts = this.pids.get(pid) || { pid, packets: 0, transportErrors: 0, scrambledPackets: 0 };
    counts.packets++; counts.transportErrors += packet[1] & 0x80 ? 1 : 0;
    counts.scrambledPackets += packet[3] & 0xc0 ? 1 : 0;
    this.pids.set(pid, counts);
    const control = (packet[3] >> 4) & 3;
    if (packet[1] & 0x80 || packet[3] & 0xc0 || control === 0) { this.assemblies.delete(pid); return; }
    let offset = 4;
    if (control & 2) {
      const length = packet[offset++];
      if (offset + length > 188) { this.assemblies.delete(pid); return; }
      if (length && packet[offset] & 0x80) this.assemblies.delete(pid);
      offset += length;
    }
    if (!(control & 1) || offset >= 188) return;
    const starts = Boolean(packet[1] & 0x40), counter = packet[3] & 0x0f;
    let state = this.assemblies.get(pid);
    if (pid !== 0 && !state && !(starts && offset + 1 + packet[offset] < 188 && packet[offset + 1 + packet[offset]] === 0x02)) return;
    if (!state) { state = { bytes: new Uint8Array(1024), used: 0, expected: 0, counter: -1 }; this.assemblies.set(pid, state); }
    if (state.counter === counter) return;
    if (state.counter >= 0 && counter !== (state.counter + 1) % 16) { state.used = 0; state.expected = 0; }
    state.counter = counter;
    if (starts) {
      const pointer = packet[offset++];
      if (offset + pointer > 188) { state.used = 0; state.expected = 0; return; }
      if (state.used && pointer) this.append(pid, state, packet.subarray(offset, offset + pointer), false);
      state.used = 0; state.expected = 0;
      this.append(pid, state, packet.subarray(offset + pointer), true);
    } else if (state.used) this.append(pid, state, packet.subarray(offset), false);
  }

  append(pid, state, bytes, allowNew) {
    for (const byte of bytes) {
      if (!state.used && byte === 0xff) return;
      state.bytes[state.used++] = byte;
      if (state.used === 3) {
        state.expected = 3 + ((state.bytes[1] & 0x0f) << 8) + state.bytes[2];
        if (state.expected < 12 || state.expected > 1024 || !(state.bytes[1] & 0x80)) { state.used = 0; state.expected = 0; return; }
      }
      if (state.expected && state.used === state.expected) {
        this.section(pid, state.bytes.subarray(0, state.used));
        state.used = 0; state.expected = 0;
        if (!allowNew) return;
      }
    }
  }

  section(pid, bytes) {
    if (mpegSectionCrc(bytes) !== 0 || !(bytes[5] & 1) || bytes[6] > bytes[7]) return;
    const number = (bytes[3] << 8) | bytes[4], version = (bytes[5] >> 1) & 31, end = bytes.length - 4;
    if (pid === 0 && bytes[0] === 0x00) {
      if ((end - 8) % 4) return;
      if (!this.pat || this.pat.number !== number || this.pat.version !== version || this.pat.last !== bytes[7]) {
        this.pat = { number, version, last: bytes[7], sections: new Map() };
      }
      const programs = [];
      for (let offset = 8; offset + 4 <= end; offset += 4) {
        const program = (bytes[offset] << 8) | bytes[offset + 1], pmtPid = ((bytes[offset + 2] & 31) << 8) | bytes[offset + 3];
        if (program) programs.push([program, pmtPid]);
      }
      this.pat.sections.set(bytes[6], programs);
      if (this.pat.sections.size === this.pat.last + 1) {
        this.programs = new Map([...this.pat.sections.values()].flat()); this.transportStreamId = number;
      }
    } else if (bytes[0] === 0x02 && bytes.length >= 16 && bytes[6] === 0 && bytes[7] === 0) {
      const pcrPid = ((bytes[8] & 31) << 8) | bytes[9];
      const streams = [];
      let offset = 12 + ((bytes[10] & 15) << 8) + bytes[11];
      if (offset > end) return;
      while (offset < end) {
        if (offset + 5 > end) return;
        const type = bytes[offset], streamPid = ((bytes[offset + 1] & 31) << 8) | bytes[offset + 2];
        const length = ((bytes[offset + 3] & 15) << 8) | bytes[offset + 4];
        offset += 5;
        if (offset + length > end) return;
        const declaration = streamDeclaration(type, bytes.subarray(offset, offset + length));
        if (!declaration) return;
        streams.push({ pid: streamPid, streamType: type, ...declaration }); offset += length;
      }
      this.pmts.set(pid, { number, pcrPid, version, streams });
    }
  }

  result() {
    return { transportStreamId: this.transportStreamId,
      programs: [...this.programs].map(([number, pmtPid]) => {
        const table = this.pmts.get(pmtPid), valid = table?.number === number;
        return { number, pmtPid, pcrPid: valid ? table.pcrPid : null,
          streams: valid ? table.streams.map(stream => ({ ...stream, packets: this.pids.get(stream.pid)?.packets || 0 })) : [] };
      }), pids: [...this.pids.values()].sort((first, second) => first.pid - second.pid),
      scope: 'CRC-valid received PAT/PMT declarations and observed packet counts; no codec decoding.' };
  }
}
