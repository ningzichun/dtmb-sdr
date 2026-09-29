const vertex = `attribute vec2 pos; varying vec2 uv; void main(){ uv=pos*.5+.5; gl_Position=vec4(pos,0,1); }`;
const fragment = `precision mediump float; varying vec2 uv; uniform sampler2D bins; uniform float head; uniform float rows; uniform float floorDb;
vec3 palette(float t){ return mix(mix(vec3(.015,.035,.085),vec3(.02,.38,.58),smoothstep(0.,.48,t)),mix(vec3(.06,.91,.69),vec3(1.,.8,.32),smoothstep(.72,1.,t)),smoothstep(.35,.85,t)); }
void main(){float y=mod(head+(1.-uv.y)*rows,256.)/256.; float v=texture2D(bins,vec2(uv.x,y)).r; float t=clamp((v*140.-140.-floorDb)/(-15.-floorDb),0.,1.); gl_FragColor=vec4(palette(t),1.);}`;

export class Waterfall {
  constructor(canvas) {
    this.canvas = canvas; this.gl = canvas.getContext('webgl', { alpha: false, antialias: false });
    if (!this.gl) throw Error('WebGL is unavailable. Enable graphics acceleration to use the waterfall.');
    const gl = this.gl;
    const compile = (type, source) => { const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s)); return s; };
    this.program = gl.createProgram();
    gl.attachShader(this.program, compile(gl.VERTEX_SHADER, vertex)); gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(this.program));
    gl.useProgram(this.program);
    const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
    const attr = gl.getAttribLocation(this.program, 'pos'); gl.enableVertexAttribArray(attr); gl.vertexAttribPointer(attr, 2, gl.FLOAT, false, 0, 0);
    this.texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.reset();
    canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); canvas.title = 'Graphics context lost. Reload to restore the waterfall.'; });
  }
  reset() { this.head = 0; this.count = 0; this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.LUMINANCE, 2048, 256, 0, this.gl.LUMINANCE, this.gl.UNSIGNED_BYTE, new Uint8Array(2048 * 256)); this.draw(); }
  add(bins, floor = -100, ceiling = -15) {
    this.floor = floor;
    const data = Uint8Array.from(bins, value => Math.max(0, Math.min(255, 255 * (value + 140) / 140)));
    const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, this.texture);
    this.head = (this.head + 255) % 256;
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, this.head, 2048, 1, gl.LUMINANCE, gl.UNSIGNED_BYTE, data);
    this.count = Math.min(256, this.count + 1); this.draw();
  }
  draw() {
    const gl = this.gl, canvas = this.canvas, ratio = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * ratio), h = Math.round(canvas.clientHeight * ratio);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, canvas.width, canvas.height); gl.useProgram(this.program);
    gl.uniform1f(gl.getUniformLocation(this.program, 'floorDb'), this.floor ?? -100);
    gl.uniform1f(gl.getUniformLocation(this.program, 'head'), this.head + .5); gl.uniform1f(gl.getUniformLocation(this.program, 'rows'), Math.max(1, this.count)); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

export class Spectrum {
  constructor(canvas, overlay) { this.canvas = canvas; this.overlay = overlay; this.bins = null; }
  draw(bins = this.bins, rate = this.rate || 15120000, center = this.center || 0, floor = -100, ceiling = -15) {
    this.bins = bins; this.rate = rate; this.center = center;
    const canvas = this.canvas, ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio); canvas.height = Math.round(canvas.clientHeight * ratio);
    const c = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
    c.scale(ratio, ratio); const width = w / ratio, height = h / ratio;
    c.fillStyle = '#080f1c'; c.fillRect(0, 0, width, height); c.font = '10px ui-monospace, monospace';
    for (let db = Math.ceil(floor / 20) * 20; db <= ceiling; db += 20) {
      const y = (ceiling - db) / (ceiling - floor) * (height - 24);
      c.strokeStyle = '#203046'; c.beginPath(); c.moveTo(0, y); c.lineTo(width, y); c.stroke(); c.fillStyle = '#778a9f'; c.fillText(`${db}`, 7, Math.max(12, y - 4));
    }
    for (let n = 0; n <= 8; n++) {
      const x = n / 8 * width; c.strokeStyle = '#19283a'; c.beginPath(); c.moveTo(x, 0); c.lineTo(x, height - 24); c.stroke();
      c.fillStyle = '#778a9f'; c.textAlign = n === 0 ? 'left' : n === 8 ? 'right' : 'center';
      c.fillText(((center + (n / 8 - .5) * rate) / 1e6).toFixed(2), x, height - 7);
    }
    c.textAlign = 'left';
    if (!bins) { c.fillStyle = '#53677f'; c.font = '13px system-ui'; c.fillText('Open a capture to inspect its spectrum', width / 2 - 130, height / 2); return; }
    c.beginPath();
    for (let i = 0; i < bins.length; i++) { const x = i / (bins.length - 1) * width, y = Math.max(0, Math.min(height - 24, (ceiling - bins[i]) / (ceiling - floor) * (height - 24))); i ? c.lineTo(x, y) : c.moveTo(x, y); }
    c.strokeStyle = '#63e8bf'; c.lineWidth = 1.3; c.stroke();
    c.lineTo(width, height - 24); c.lineTo(0, height - 24); c.closePath();
    const gradient = c.createLinearGradient(0, 0, 0, height); gradient.addColorStop(0, '#43d6ad44'); gradient.addColorStop(1, '#43d6ad00'); c.fillStyle = gradient; c.fill();
  }
}
