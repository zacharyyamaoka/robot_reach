// Reach proposals page: every point cloud is drawn from the real saved UR5e map embedded below.
"use strict";

async function gunzip(b64) {
  const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
};
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const fmt = (n) => n.toLocaleString("en-US");
let D, MASK, POS, VIRIDIS_LEVELS, EMPTY;

// ------------------------------------------------------------------------------ levels from the mask
function levelsFor(bits) {
  const { P, O, tilt } = D;
  const out = new Int8Array(P);
  for (let p = 0; p < P; p++) {
    let any = false, four = true, hemi = true, all = true;
    const base = p * O;
    for (let o = 0; o < O; o++) {
      const ok = (MASK[base + o] & bits) !== 0;
      if (ok) any = true;
      else { all = false; if (tilt[o] === 0) four = false; if (tilt[o] <= 90) hemi = false; }
    }
    out[p] = !any ? 0 : all ? 6 : hemi ? 5 : four ? 4 : 3;
  }
  return out;
}
const levelCounts = (lv) => { const c = { 0: 0, 3: 0, 4: 0, 5: 0, 6: 0 }; for (const v of lv) c[v]++; return c; };
const atLeast = (c) => `reachable ${fmt(c[3] + c[4] + c[5] + c[6])} · 4-DOF ${fmt(c[4] + c[5] + c[6])} · hemisphere ${fmt(c[5] + c[6])} · dexterous ${fmt(c[6])}`;

// ------------------------------------------------------------------------------ a tiny 3D view
class View {
  constructor(canvas, opts = {}) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.az = opts.az ?? -0.95;
    this.el = opts.el ?? 0.38;
    this.center = opts.center ?? [0, 0, 0.17];
    this.span = opts.span ?? 2.1;
    this.floor = opts.floor ?? true;
    this.draw = () => {};
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
    canvas.width = w * dpr; canvas.height = h * dpr;
    this.ctx.scale(dpr, dpr);
    this.w = w; this.h = h;
    let drag = null;
    canvas.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY, this.az, this.el]; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.az = drag[2] - (e.clientX - drag[0]) * 0.01;
      this.el = Math.max(-0.2, Math.min(1.45, drag[3] + (e.clientY - drag[1]) * 0.01));
      this.render();
    });
    canvas.addEventListener("pointerup", () => (drag = null));
    canvas.title = "drag to turn";
  }
  basis() {
    const ca = Math.cos(this.az), sa = Math.sin(this.az), ce = Math.cos(this.el), se = Math.sin(this.el);
    return { r: [-sa, ca, 0], u: [-se * ca, -se * sa, ce], f: [ce * ca, ce * sa, se] };
  }
  setScene(fn) { this.draw = fn; this.render(); }
  render() {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    const B = this.basis(), s = Math.min(w, h) / this.span, cx = w / 2, cy = h / 2 + h * 0.04, C = this.center;
    const P = (x, y, z) => {
      const dx = x - C[0], dy = y - C[1], dz = z - C[2];
      return [cx + s * (dx * B.r[0] + dy * B.r[1]), cy - s * (dx * B.u[0] + dy * B.u[1] + dz * B.u[2]), dx * B.f[0] + dy * B.f[1] + dz * B.f[2]];
    };
    this.P = P; this.s = s;
    if (this.floor) {
      ctx.strokeStyle = "rgba(120,130,140,0.18)"; ctx.lineWidth = 1;
      for (let i = -5; i <= 5; i++) {
        const a = P(i * 0.2, -1, 0), b = P(i * 0.2, 1, 0), c = P(-1, i * 0.2, 0), d = P(1, i * 0.2, 0);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke();
      }
    }
    this.draw(this);
  }
  points(xyz, colors, size, alpha = 1) { // xyz: Float32Array 3n, colors: array of rgb triples or null to skip
    const { ctx } = this, n = xyz.length / 3, items = [];
    for (let i = 0; i < n; i++) {
      if (!colors[i]) continue;
      const p = this.P(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
      items.push([p[2], p[0], p[1], colors[i]]);
    }
    items.sort((a, b) => a[0] - b[0]);
    const px = Math.max(1.5, size * this.s);
    for (const [, x, y, c] of items) {
      ctx.fillStyle = rgb(c, alpha);
      ctx.fillRect(x - px / 2, y - px / 2, px, px);
      if (px > 4) { ctx.strokeStyle = "rgba(0,0,0,0.10)"; ctx.lineWidth = 0.6; ctx.strokeRect(x - px / 2, y - px / 2, px, px); }
    }
  }
  poly(pts, color, width = 2, dash = null) {
    const { ctx } = this;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
    ctx.beginPath();
    pts.forEach((q, i) => { const p = this.P(q[0], q[1], q[2]); i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); });
    ctx.stroke(); ctx.setLineDash([]);
  }
  label(q, text, color = "#495057") {
    const p = this.P(q[0], q[1], q[2]);
    this.ctx.fillStyle = color; this.ctx.font = "11px Inter, system-ui, sans-serif"; this.ctx.fillText(text, p[0] + 4, p[1] - 4);
  }
}

function drawArm(v) {
  v.poly(D.home, "rgba(33,37,41,0.85)", 5);
}

function viewFor(canvas, colorsFn, opts = {}) {
  const v = new View(canvas, opts);
  v.colorsFn = colorsFn;
  v.setScene((vv) => { vv.points(POS, vv.colorsFn(), D.step * 0.55); if (opts.arm !== false) drawArm(vv); });
  return v;
}

// ------------------------------------------------------------------------------ colour helpers
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  return stops[i].map((c, k) => Math.round(c * (1 - f) + stops[i + 1][k] * f));
}
const MACHADO = {
  deuteranopia: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
  protanopia: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  tritanopia: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
};
const toLin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const toSrgb = (c) => { c = Math.max(0, Math.min(1, c)); return Math.round((c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055) * 255); };
function simulate(c, kind) {
  if (!kind || kind === "none") return c;
  const M = MACHADO[kind], l = c.map(toLin);
  return M.map((row) => toSrgb(row[0] * l[0] + row[1] * l[1] + row[2] * l[2]));
}

// ------------------------------------------------------------------------------ E: colour-by bins
function binsFor(colorBy) {
  const { P } = D;
  const lv = D.levels_py, V = VIRIDIS_LEVELS;
  const vr = D.schemes.viridis.ramp;
  if (colorBy === "pose DOF") {
    const keys = [0, 3, 4, 5, 6];
    const labels = ["unreachable", "3 reachable", "4 4-DOF", "5 hemisphere", "6 dexterous"];
    const short = ["none", "3", "4", "5", "6"];
    const colors = [EMPTY, ...V];
    const bin = new Int8Array(P);
    for (let p = 0; p < P; p++) bin[p] = keys.indexOf(lv[p]);
    return { labels, short, colors, bin, defaultOff: [0], note: "levels nest: dexterous points are also hemisphere, 4-DOF and reachable" };
  }
  if (colorBy === "IK solutions") {
    const labels = [], colors = [], short = [];
    for (let k = 0; k <= 8; k++) { labels.push(k === 0 ? "0 (none)" : `${k} solution${k > 1 ? "s" : ""}`); short.push(String(k)); colors.push(k === 0 ? EMPTY : ramp(vr, (k - 1) / 7)); }
    const bin = new Int8Array(P);
    for (let p = 0; p < P; p++) bin[p] = D.nsol[p];
    return { labels, short, colors, bin, defaultOff: [0], note: "most IK solutions at the point's best orientation (UR5e: up to 8)" };
  }
  if (colorBy === "fraction of orientations") {
    const labels = ["0 %"], short = ["0"], colors = [EMPTY];
    for (let k = 0; k < 10; k++) { labels.push(`${k * 10 + 1}–${(k + 1) * 10} %`); short.push(`${(k + 1) * 10}`); colors.push(ramp(vr, k / 9)); }
    const bin = new Int8Array(P);
    for (let p = 0; p < P; p++) bin[p] = D.frac[p] === 0 ? 0 : 1 + Math.min(9, Math.floor((D.frac[p] - 1) / 10));
    return { labels, short, colors, bin, defaultOff: [0], note: "share of the 132 sampled orientations with a working solution" };
  }
  const cc = D.code_colors;
  return {
    labels: ["no IK answer", "FK = IK everywhere", "some mismatch", "all mismatch"], short: ["none", "match", "some", "all"],
    colors: [EMPTY, cc[0], [245, 159, 0], cc[6]], bin: Int8Array.from(D.cons), defaultOff: [0],
    note: `DEMO: an IK whose shoulder_lift zero is ${D.eps_deg ?? "0.011"}° off its FK (every saved result has 0 mismatches)`,
  };
}
const countBins = (b) => { const n = new Array(b.labels.length).fill(0); for (const v of b.bin) n[v]++; return n; };

// ------------------------------------------------------------------------------ orientation set (port)
function frameFromZ(z) {
  const n = Math.hypot(...z); z = z.map((v) => v / n);
  const helper = Math.abs(z[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const d = helper[0] * z[0] + helper[1] * z[1] + helper[2] * z[2];
  let x = helper.map((h, i) => h - d * z[i]); const nx = Math.hypot(...x); x = x.map((v) => v / nx);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return [x, y, z]; // columns
}
function orientationDirs(approach, tiltStep, maxTilt) {
  const F = frameFromZ(approach), dirs = [], tilts = [];
  for (let t = 0; t <= maxTilt + 1e-9; t += tiltStep) {
    const r = (t * Math.PI) / 180, st = Math.sin(r);
    const nAz = st < 1e-9 ? 1 : Math.max(3, Math.round((2 * Math.PI * st) / ((tiltStep * Math.PI) / 180)));
    for (let k = 0; k < nAz; k++) {
      const az = (2 * Math.PI * k) / nAz, loc = [st * Math.cos(az), st * Math.sin(az), Math.cos(r)];
      dirs.push([0, 1, 2].map((i) => F[0][i] * loc[0] + F[1][i] * loc[1] + F[2][i] * loc[2]));
      tilts.push(t);
    }
  }
  return { dirs, tilts };
}
function rpy(rollDeg, pitchDeg, yawDeg) { // R = Rz(yaw) Ry(pitch) Rx(roll), columns are the box axes
  const [a, b, c] = [rollDeg, pitchDeg, yawDeg].map((d) => (d * Math.PI) / 180);
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cc = Math.cos(c), sc = Math.sin(c);
  return [[cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa], [sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa], [-sb, cb * sa, cb * ca]];
}
const col = (R, j) => [R[0][j], R[1][j], R[2][j]];
const add = (a, b, s = 1) => a.map((v, i) => v + s * b[i]);

function boxScene(v, center, R, size, dirs, tilts, label) {
  drawArm(v);
  const hx = col(R, 0).map((x) => (x * size[0]) / 2), hy = col(R, 1).map((x) => (x * size[1]) / 2), hz = col(R, 2).map((x) => (x * size[2]) / 2);
  const corner = (i, j, k) => add(add(add(center, hx, i), hy, j), hz, k);
  for (const [i, j] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) v.poly([corner(i, j, -1), corner(i, j, 1)], "#d9480f", 1.2);
  for (const k of [-1, 1]) v.poly([corner(-1, -1, k), corner(1, -1, k), corner(1, 1, k), corner(-1, 1, k), corner(-1, -1, k)], "#d9480f", 1.2);
  const L = 0.2;
  dirs.forEach((d, i) => v.poly([center, add(center, d, -L)], tilts[i] === 0 ? "#1971c2" : "rgba(47,158,68,0.75)", tilts[i] === 0 ? 3 : 1.6));
  v.poly([center, add(center, col(R, 2), 0.34)], "#1c7ed6", 2.2, [5, 3]);
  v.label(add(center, col(R, 2), 0.36), label || "box Z = workspace up", "#1c7ed6");
}

// ------------------------------------------------------------------------------ build everything
async function main() {
  D = JSON.parse(new TextDecoder().decode(await gunzip(window.DATA_B64)));
  MASK = await gunzip(window.MASK_B64);
  const { P } = D;
  POS = new Float32Array(3 * P);
  for (let p = 0; p < P; p++) for (let a = 0; a < 3; a++) POS[3 * p + a] = D.origin[a] + D.gidx[3 * p + a] * D.step;
  VIRIDIS_LEVELS = D.schemes.viridis.levels.slice(1);
  EMPTY = D.schemes.viridis.levels[0];
  D.eps_deg = window.EPS_DEG;

  // the page's own recomputation must equal the saved map, or nothing here can be trusted
  const all = levelCounts(levelsFor(255));
  const saved = D.checks.saved, okAll = [0, 3, 4, 5, 6].every((k) => all[k] === saved[k]);
  document.querySelectorAll(".selfcheck").forEach((n) => {
    n.textContent = okAll ? `✓ this page recomputed the levels from the embedded map: ${atLeast(all)} — identical to the saved result` : "✗ recomputed levels differ from the saved result";
    n.classList.add(okAll ? "pass" : "fail");
  });

  const levelColor = (lv) => (k) => (lv[k] ? VIRIDIS_LEVELS[lv[k] - 3] : null);
  buildA(levelColor);
  buildC();
  buildD();
  buildE();
  buildF();
  buildH();
}

// ============================================================================== A
function buildA(levelColor) {
  const spec = D.spec, types = spec.types;
  const nameOf = (t) => types[t].name;
  const readout = (node, bits) => {
    const lv = levelsFor(bits), c = levelCounts(lv);
    const n = types.filter((t) => bits & (1 << t.index)).length;
    node.innerHTML = `<b>${n} of 8 types allowed</b> → ${atLeast(c)}`;
    return lv;
  };
  // ---------------- A1: one row per question
  {
    const root = $("#a1"), state = spec.axes.map((a) => a.options.map(() => true));
    const panel = root.querySelector(".vpanel"), out = root.querySelector(".readout");
    const v = new View(root.querySelector("canvas"));
    let lv = levelsFor(255);
    v.setScene((vv) => { vv.points(POS, Array.from({ length: D.P }, (_, k) => levelColor(lv)(k)), D.step * 0.55); drawArm(vv); });
    const bits = () => types.reduce((b, t) => (t.tags.every((o, a) => state[a][o]) ? b | (1 << t.index) : b), 0);
    spec.axes.forEach((axis, a) => {
      const row = el("div", { class: "vrow" }, el("label", { title: axis.rule }, axis.label));
      const pills = el("div", { class: "pills" });
      axis.options.forEach((opt, o) => {
        const b = el("button", { class: "pill on", title: axis.rule, onclick: () => {
          if (state[a].filter(Boolean).length === 1 && state[a][o]) return; // keep at least one per question
          state[a][o] = !state[a][o]; b.classList.toggle("on", state[a][o]); lv = readout(out, bits()); v.render();
        } }, opt);
        pills.append(b);
      });
      row.append(pills); panel.append(row);
    });
    lv = readout(out, bits()); v.render();
  }
  // ---------------- A2: eight posture tiles
  {
    const root = $("#a2"), grid = root.querySelector(".tiles"), out = root.querySelector(".readout");
    let bits = 255;
    const v = new View(root.querySelector("canvas.cloud"));
    let lv = levelsFor(bits);
    v.setScene((vv) => { vv.points(POS, Array.from({ length: D.P }, (_, k) => levelColor(lv)(k)), D.step * 0.55); drawArm(vv); });
    const alone = types.map((t) => { const c = levelCounts(levelsFor(1 << t.index)); return c[3] + c[4] + c[5] + c[6]; });
    const tiles = [];
    const refresh = () => { tiles.forEach((tl, t) => tl.classList.toggle("on", !!(bits & (1 << t)))); lv = readout(out, bits); v.render(); };
    // header row: shoulder x wrist columns; rows: elbow
    const order = []; // [elbow][col] -> type
    for (const e of [0, 1]) for (const s of [0, 1]) for (const w of [0, 1]) order.push(types.find((t) => t.tags[0] === s && t.tags[1] === e && t.tags[2] === w).index);
    const ax = spec.axes;
    grid.append(el("div"));
    for (const s of [0, 1]) for (const w of [0, 1]) {
      grid.append(el("button", { class: "hdr", title: "toggle this column", onclick: () => {
        const ts = types.filter((t) => t.tags[0] === s && t.tags[2] === w).map((t) => t.index);
        const on = ts.every((t) => bits & (1 << t)); ts.forEach((t) => (bits = on ? bits & ~(1 << t) : bits | (1 << t))); refresh();
      } }, `${ax[0].name} ${ax[0].options[s]}`, el("br"), `${ax[2].name} ${ax[2].options[w]}`));
    }
    [0, 1].forEach((e) => {
      grid.append(el("button", { class: "hdr side", title: "toggle this row", onclick: () => {
        const ts = types.filter((t) => t.tags[1] === e).map((t) => t.index);
        const on = ts.every((t) => bits & (1 << t)); ts.forEach((t) => (bits = on ? bits & ~(1 << t) : bits | (1 << t))); refresh();
      } }, `${ax[1].name} ${ax[1].options[e]}`));
      for (let c = 0; c < 4; c++) {
        const t = order[e * 4 + c];
        const cv = el("canvas", { class: "glyph", width: 64, height: 58 });
        const tile = el("button", { class: "tile on", title: nameOf(t), onclick: () => { bits ^= 1 << t; refresh(); } }, cv, el("small", {}, `${fmt(alone[t])} pts`));
        tiles[t] = tile; grid.append(tile);
        const g = D.sel.glyphs[t];
        if (g) {
          const gv = new View(cv, { az: -1.25, el: 0.12, center: [0.25, 0, 0.35], span: 1.05, floor: false });
          gv.setScene((vv) => { vv.poly([[-0.1, 0, 0], [0.6, 0, 0]], "#ced4da", 1); vv.poly(g, "#212529", 2.4); const tip = g[g.length - 1]; vv.points(new Float32Array(tip), [[255, 196, 0]], 0.06); });
        }
      }
    });
    refresh();
  }
  // ---------------- A3: filter by example, from the solutions at the selected point
  {
    const root = $("#a3"), list = root.querySelector(".sols"), out = root.querySelector(".readout"), req = root.querySelector(".req");
    const want = spec.axes.map(() => null); // null = any
    const v = new View(root.querySelector("canvas"));
    let lv = levelsFor(255);
    v.setScene((vv) => { vv.points(POS, Array.from({ length: D.P }, (_, k) => levelColor(lv)(k)), D.step * 0.55); drawArm(vv); vv.points(new Float32Array(D.sel.pos), [[255, 196, 0]], 0.09); });
    const bits = () => types.reduce((b, t) => (t.tags.every((o, a) => want[a] === null || want[a] === o) ? b | (1 << t.index) : b), 0);
    const draw = () => {
      const b = bits(); lv = readout(out, b); v.render();
      const rows = D.sel.rows.filter((r) => b & (1 << r[1]));
      list.innerHTML = "";
      list.append(el("div", { class: "muted" }, `${fmt(rows.length)} of ${fmt(D.sel.rows.length)} working (orientation, IK branch) pairs here`));
      rows.slice(0, 7).forEach((r, i) => {
        const chips = types[r[1]].tags.map((o, a) => el("button", { class: "chip" + (want[a] === o ? " on" : ""), title: `only ${spec.axes[a].name} ${spec.axes[a].options[o]}`, onclick: () => { want[a] = want[a] === o ? null : o; draw(); } }, `${spec.axes[a].name} ${spec.axes[a].options[o]}`));
        list.append(el("div", { class: "solrow" }, el("b", {}, `${i + 1}`), el("span", { class: "muted" }, ` orientation #${r[0]} `), ...chips, el("div", { class: "q" }, `q = [${r.slice(2).map((x) => (x >= 0 ? "+" : "") + x.toFixed(1)).join(", ")}]°`)));
      });
      req.innerHTML = want.some((w) => w !== null) ? "Only: " + want.map((w, a) => (w === null ? null : `${spec.axes[a].name} ${spec.axes[a].options[w]}`)).filter(Boolean).join(" · ") + " <a href='#' class='clear'>clear</a>" : "Showing every type. Click a chip to keep only that kind.";
      const c = req.querySelector(".clear"); if (c) c.onclick = (e) => { e.preventDefault(); want.fill(null); draw(); };
    };
    draw();
  }
}

// ============================================================================== C
function buildC() {
  const stats = (node, dirs, nRolls, extra = "") => {
    const n = dirs.length * nRolls, est = (D.ask_points * n) / D.rate;
    node.innerHTML = `<b>${dirs.length} directions × ${nRolls} rolls = ${fmt(n)} orientations</b> per point · box of ${fmt(D.ask_points)} points ≈ ${est < 60 ? est.toFixed(1) + " s" : (est / 60).toFixed(1) + " min"}${extra}`;
  };
  D.ask_points = 7 * 7 * 7; // a 0.3 m box at the viewer's default step (half the map step) is 7 points a side
  const center = [0.45, 0.0, 0.35], size = [0.3, 0.3, 0.3];
  // C1: drag to tilt (the gizmo's rings), the panel mirrors it
  {
    const root = $("#c1"), v = new View(root.querySelector("canvas"), { center: [0.42, 0, 0.36], span: 0.95 });
    const f = Object.fromEntries([...root.querySelectorAll("[data-k]")].map((n) => [n.dataset.k, n]));
    const go = () => {
      const R = rpy(+f.rx.value, +f.ry.value, 0);
      const ts = +f.step.value, mt = +f.max.value, nr = Math.round(360 / +f.roll.value);
      const { dirs, tilts } = orientationDirs(col(R, 2).map((x) => -x), ts, mt);
      f.rxv.textContent = `${f.rx.value}°`; f.ryv.textContent = `${f.ry.value}°`;
      const tiltOff = Math.round((Math.acos(Math.max(-1, Math.min(1, R[2][2]))) * 180) / Math.PI);
      root.querySelector(".tiltread").textContent = `box Z is ${tiltOff}° from world up`;
      stats(root.querySelector(".readout"), dirs, nr);
      v.setScene((vv) => boxScene(vv, center, R, size, dirs, tilts));
    };
    Object.values(f).forEach((n) => n.addEventListener("input", go)); go();
  }
  // C2: type the box frame and the angle set
  {
    const root = $("#c2"), v = new View(root.querySelector("canvas"), { center: [0.42, 0, 0.36], span: 0.95 });
    const f = Object.fromEntries([...root.querySelectorAll("[data-k]")].map((n) => [n.dataset.k, n]));
    const go = () => {
      const R = rpy(+f.roll.value, +f.pitch.value, +f.yaw.value);
      const ts = +f.step.value, mt = +f.max.value, nr = Math.round(360 / +f.rollstep.value);
      const { dirs, tilts } = orientationDirs(col(R, 2).map((x) => -x), ts, mt);
      stats(root.querySelector(".readout"), dirs, nr);
      const a = col(R, 2).map((x) => -x).map((x) => x.toFixed(2));
      root.querySelector(".py").textContent = `Box(center=(0.45, 0.0, 0.35), size=(0.3, 0.3, 0.3), rpy_deg=(${f.roll.value}, ${f.pitch.value}, ${f.yaw.value}))\nOrientationSet.sample(approach=(${a.join(", ")}), tilt_step_deg=${ts}, n_rolls=${nr}, max_tilt_deg=${mt})`;
      v.setScene((vv) => boxScene(vv, center, R, size, dirs, tilts));
    };
    Object.values(f).forEach((n) => n.addEventListener("input", go)); go();
  }
  // C3: presets
  {
    const root = $("#c3"), v = new View(root.querySelector("canvas"), { center: [0.42, 0, 0.36], span: 0.95 });
    const st = { up: [0, 0], cone: 90, fine: [45, 60] };
    const go = () => {
      const R = rpy(st.up[0], st.up[1], 0);
      const { dirs, tilts } = orientationDirs(col(R, 2).map((x) => -x), st.fine[0], st.cone);
      stats(root.querySelector(".readout"), dirs, Math.round(360 / st.fine[1]));
      v.setScene((vv) => boxScene(vv, center, R, size, dirs, tilts));
    };
    root.querySelectorAll("[data-group]").forEach((b) => b.addEventListener("click", () => {
      root.querySelectorAll(`[data-group="${b.dataset.group}"]`).forEach((x) => x.classList.remove("on"));
      b.classList.add("on"); st[b.dataset.group] = JSON.parse(b.dataset.v); go();
    }));
    go();
  }
  // the JS port must count exactly what the library samples
  const rows = D.orient_checks.map((c) => {
    const n = orientationDirs([0, 0, -1], c.tilt_step, c.max_tilt).dirs.length * c.n_rolls;
    return `<tr><td>${c.tilt_step}°</td><td>${c.max_tilt}°</td><td>${c.n_rolls}</td><td>${fmt(c.n)}</td><td>${fmt(n)}</td><td class="${n === c.n ? "pass" : "fail"}">${n === c.n ? "✓" : "✗"}</td></tr>`;
  });
  $("#c-check").innerHTML = rows.join("");
}

// ============================================================================== D
function buildD() {
  const views = [];
  const ctl = { by: "pose DOF", sim: "none" };
  for (const node of document.querySelectorAll(".dview")) {
    const sc = D.schemes[node.dataset.scheme];
    const v = new View(node.querySelector("canvas"));
    v.colorsFn = () => {
      const out = new Array(D.P);
      for (let p = 0; p < D.P; p++) {
        const lv = D.levels_py[p];
        if (!lv) { out[p] = null; continue; }
        const c = ctl.by === "pose DOF" ? sc.levels[lv - 2] : ramp(sc.ramp, D.frac[p] / 100);
        out[p] = simulate(c, ctl.sim);
      }
      return out;
    };
    v.setScene((vv) => { vv.points(POS, vv.colorsFn(), D.step * 0.55); drawArm(vv); });
    const leg = node.querySelector(".legend");
    v.legend = () => {
      leg.innerHTML = "";
      if (ctl.by === "pose DOF") ["3 reachable", "4 4-DOF", "5 hemisphere", "6 dexterous"].forEach((t, i) => leg.append(el("span", {}, el("i", { style: `background:${rgb(simulate(sc.levels[i + 1], ctl.sim))}` }), t)));
      else leg.append(el("span", { class: "rampbar", style: `background:linear-gradient(90deg,${sc.ramp.map((c) => rgb(simulate(c, ctl.sim))).join(",")})` }), el("span", { class: "muted" }, " 0 → 100 % of orientations"));
    };
    v.legend();
    views.push(v);
  }
  const redraw = () => views.forEach((v) => { v.render(); v.legend(); });
  $("#d-by").addEventListener("change", (e) => { ctl.by = e.target.value; redraw(); });
  $("#d-sim").addEventListener("change", (e) => { ctl.sim = e.target.value; redraw(); });
}

// ============================================================================== E
function buildE() {
  for (const node of document.querySelectorAll(".evar")) {
    const kind = node.dataset.kind;
    const sel = node.querySelector("select"), ctrl = node.querySelector(".hctl"), info = node.querySelector(".readout");
    const v = new View(node.querySelector("canvas"));
    let bins, counts, on;
    const colors = () => Array.from({ length: D.P }, (_, p) => (on[bins.bin[p]] ? bins.colors[bins.bin[p]] : null));
    v.setScene((vv) => { if (bins) vv.points(POS, colors(), D.step * 0.55); drawArm(vv); });
    const update = () => {
      const shown = counts.reduce((s, n, i) => s + (on[i] ? n : 0), 0);
      info.innerHTML = `<b>${fmt(shown)} points shown</b> of ${fmt(D.P)} sampled · <span class="muted">${bins.note}</span>`;
      render(); v.render();
    };
    const toggle = (i) => { on[i] = !on[i]; update(); };
    const maxN = () => Math.max(...counts);
    const render = () => {
      ctrl.innerHTML = "";
      if (kind === "bars") {
        const wrap = el("div", { class: "bars" });
        counts.forEach((n, i) => wrap.append(el("button", { class: "bar" + (on[i] ? "" : " off"), title: `${bins.labels[i]}: ${fmt(n)} — click to ${on[i] ? "hide" : "show"}`, onclick: () => toggle(i) },
          el("span", { class: "n" }, fmt(n)), el("span", { class: "col", style: `height:${4 + 64 * Math.sqrt(n / maxN())}px;background:${rgb(bins.colors[i])}` }), el("span", { class: "lab" }, bins.short[i]))));
        ctrl.append(wrap, el("div", { class: "muted small" }, "bar height √count · click a bar to show / hide it"));
      } else if (kind === "strip") {
        const total = counts.reduce((a, b) => a + b, 0), wrap = el("div", { class: "strip" });
        counts.forEach((n, i) => wrap.append(el("button", { class: "seg" + (on[i] ? "" : " off"), title: `${bins.labels[i]}: ${fmt(n)}`, style: `flex:${Math.max(n / total, 0.025)};background:${rgb(bins.colors[i])}`, onclick: () => toggle(i) })));
        const labs = el("div", { class: "striplabs" });
        counts.forEach((n, i) => labs.append(el("button", { class: "slab" + (on[i] ? "" : " off"), onclick: () => toggle(i) }, el("i", { style: `background:${rgb(bins.colors[i])}` }), `${bins.short[i]} `, el("b", {}, fmt(n)))));
        ctrl.append(wrap, labs);
      } else if (kind === "range") {
        const wrap = el("div", { class: "bars" });
        counts.forEach((n, i) => wrap.append(el("div", { class: "bar" + (on[i] ? "" : " off") }, el("span", { class: "n" }, fmt(n)), el("span", { class: "col", style: `height:${4 + 64 * Math.sqrt(n / maxN())}px;background:${rgb(bins.colors[i])}` }), el("span", { class: "lab" }, bins.short[i]))));
        const lo = on.indexOf(true), hi = on.lastIndexOf(true), k = counts.length - 1;
        const r1 = el("input", { type: "range", min: 0, max: k, value: lo, class: "dual" }), r2 = el("input", { type: "range", min: 0, max: k, value: hi, class: "dual" });
        const set = () => { let a = +r1.value, b = +r2.value; if (a > b) [a, b] = [b, a]; on = counts.map((_, i) => i >= a && i <= b); update(); };
        r1.oninput = set; r2.oninput = set;
        ctrl.append(wrap, el("div", { class: "dualwrap" }, r1, r2), el("div", { class: "muted small" }, `showing ${bins.short[lo]} … ${bins.short[hi]} · drag either handle`));
      } else if (kind === "list") {
        const wrap = el("div", { class: "lrows" });
        counts.forEach((n, i) => {
          const cb = el("input", { type: "checkbox" }); cb.checked = on[i]; cb.onchange = () => toggle(i);
          wrap.append(el("label", { class: "lrow" + (on[i] ? "" : " off") }, cb, el("i", { style: `background:${rgb(bins.colors[i])}` }), el("span", { class: "ll" }, bins.labels[i]),
            el("span", { class: "lbar" }, el("span", { style: `width:${(100 * n) / maxN()}%;background:${rgb(bins.colors[i])}` })), el("b", {}, fmt(n))));
        });
        ctrl.append(wrap);
      } else if (kind === "atleast") {
        const k0 = on.indexOf(true), wrap = el("div", { class: "bars" });
        const cum = counts.map((_, i) => counts.slice(i).reduce((a, b) => a + b, 0));
        cum.forEach((n, i) => wrap.append(el("button", { class: "bar" + (i >= k0 ? "" : " off"), title: `at least ${bins.labels[i]}: ${fmt(n)} points`, onclick: () => { on = counts.map((_, j) => j >= i); update(); } },
          el("span", { class: "n" }, fmt(n)), el("span", { class: "col", style: `height:${4 + 64 * (n / cum[0])}px;background:${rgb(bins.colors[i])}` }), el("span", { class: "lab" }, i === 0 ? "all" : `≥${bins.short[i]}`))));
        const r = el("input", { type: "range", min: 0, max: counts.length - 1, value: k0, class: "single" });
        r.oninput = () => { on = counts.map((_, j) => j >= +r.value); update(); };
        ctrl.append(wrap, r, el("div", { class: "muted small" }, `Show: at least ${bins.labels[k0]} — bar = points at or above that bin`));
      }
    };
    const pick = () => {
      bins = binsFor(sel.value); counts = countBins(bins);
      on = counts.map((_, i) => !bins.defaultOff.includes(i));
      update();
    };
    sel.addEventListener("change", pick);
    pick();
  }
}

// ============================================================================== F
function buildF() {
  const used = { vx: Math.round(D.step * 1000), vy: Math.round(D.step * 1000), vz: Math.round(D.step * 1000), tilt: 45, roll: 60 };
  const size = D.box.size;
  const cost = (s) => {
    const n = size.map((L, i) => Math.ceil(L / ([s.vx, s.vy, s.vz][i] / 1000) - 1e-9) + 1);
    const pts = n[0] * n[1] * n[2];
    const O = orientationDirs([0, 0, -1], s.tilt, 180).dirs.length * Math.round(360 / s.roll);
    return { pts, O, sec: (pts * O) / D.rate, n };
  };
  const costText = (s) => { const c = cost(s); return `${fmt(c.pts)} points (${c.n.join(" × ")}) × ${fmt(c.O)} orientations ≈ ${c.sec < 90 ? c.sec.toFixed(0) + " s" : (c.sec / 60).toFixed(1) + " min"}`; };
  for (const node of document.querySelectorAll(".fvar")) {
    const kind = node.dataset.kind;
    const fields = [...node.querySelectorAll("[data-k]")];
    const btn = node.querySelector(".compute"), reset = node.querySelector(".reset"), status = node.querySelector(".status"), costN = node.querySelector(".cost");
    const v = new View(node.querySelector("canvas"));
    let shown = true, computedWith = { ...used };
    const lvColor = (p) => (D.levels_py[p] ? VIRIDIS_LEVELS[D.levels_py[p] - 3] : null);
    v.setScene((vv) => { if (shown) vv.points(POS, Array.from({ length: D.P }, (_, p) => lvColor(p)), D.step * 0.55); else vv.label([0, 0, 0.9], "old result hidden: settings changed", "#868e96"); drawArm(vv); });
    const read = () => Object.fromEntries(fields.map((f) => [f.dataset.k, +f.value]));
    const sync = () => {
      const s = read(), dirty = Object.keys(computedWith).some((k) => s[k] !== computedWith[k]);
      fields.forEach((f) => f.classList.toggle("changed", +f.value !== computedWith[f.dataset.k]));
      btn.classList.toggle("lit", dirty); btn.disabled = !dirty;
      reset.style.visibility = dirty ? "visible" : "hidden";
      shown = !dirty;
      if (costN) costN.textContent = (dirty ? "new: " : "used: ") + costText(s);
      status.innerHTML = dirty ? "<b>Settings changed</b> — the old result is hidden. <b>Compute workspace</b> to run, or <b>Reset</b> to the values it used." : `Showing the result computed with these values (${costText(computedWith)}).`;
      if (kind === "diff") node.querySelectorAll("[data-was]").forEach((n) => { const k = n.dataset.was; n.textContent = s[k] !== computedWith[k] ? `was ${computedWith[k]}` : ""; });
      if (kind === "recipe") node.querySelector(".recipe").textContent = `Computed with ${computedWith.vx} × ${computedWith.vy} × ${computedWith.vz} mm voxels · ${computedWith.tilt}° tilt step · ${computedWith.roll}° roll step`;
      v.render();
    };
    fields.forEach((f) => f.addEventListener("input", sync));
    reset.addEventListener("click", () => { fields.forEach((f) => (f.value = computedWith[f.dataset.k])); sync(); });
    btn.addEventListener("click", () => { computedWith = read(); sync(); status.innerHTML += " <span class='muted'>(mock: the real button runs analyze() and replaces the map)</span>"; });
    const edit = node.querySelector(".edit");
    if (edit) edit.addEventListener("click", () => node.querySelector(".drawer").classList.toggle("open"));
    sync();
  }
}

// ============================================================================== H
function buildH() {
  const v = new View($("#h-cloud"), { span: 2.9, center: [0, 0, 0.12] });
  const n = D.cloud_code.length, xyz = new Float32Array(3 * n), cols = new Array(n);
  for (let i = 0; i < n; i++) { for (let a = 0; a < 3; a++) xyz[3 * i + a] = D.cloud[3 * i + a] / 1000; cols[i] = D.code_colors[D.cloud_code[i]]; }
  const c = D.box.center, s = D.box.size.map((x) => x / 2);
  const corner = (i, j, k) => [c[0] + i * s[0], c[1] + j * s[1], c[2] + k * s[2]];
  v.setScene((vv) => {
    vv.points(xyz, cols, 0.018, 0.8);
    for (const k of [-1, 1]) vv.poly([corner(-1, -1, k), corner(1, -1, k), corner(1, 1, k), corner(-1, 1, k), corner(-1, -1, k)], "#d9480f", 1.4, [4, 3]);
    for (const [i, j] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) vv.poly([corner(i, j, -1), corner(i, j, 1)], "#d9480f", 1.4, [4, 3]);
    vv.label(corner(-1, -1, 1), "grid box = bounding box of the cloud", "#d9480f");
    drawArm(vv);
  });
}

main().catch((e) => { document.body.prepend(el("div", { class: "fatal" }, "page script failed: " + e)); console.error(e); });
