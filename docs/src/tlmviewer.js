(function(){"use strict";try{if(typeof document<"u"){var e=document.createElement("style");e.appendChild(document.createTextNode(".tlmviewer{display:inline-block;position:relative;box-sizing:border-box}.tlmviewer .tlmviewer-title{position:absolute;color:#fff;-webkit-user-select:none;user-select:none;padding:5px;line-height:normal;font-size:12px;background-color:#000;left:0;bottom:0;font-family:monospace;font-weight:700}.lil-gui{--width: 250px;--name-width: 50%}.tlmviewer .lil-gui.root{position:absolute;right:0;top:0}")),document.head.appendChild(e)}}catch(t){console.error("vite-plugin-css-injected-by-js",t)}})();
var Xl = Object.defineProperty;
var Yl = (i, t, e) => t in i ? Xl(i, t, { enumerable: !0, configurable: !0, writable: !0, value: e }) : i[t] = e;
var Qt = (i, t, e) => Yl(i, typeof t != "symbol" ? t + "" : t, e);
/**
 * @license
 * Copyright 2010-2024 Three.js Authors
 * SPDX-License-Identifier: MIT
 */
const fa = "171", xi = { LEFT: 0, MIDDLE: 1, RIGHT: 2, ROTATE: 0, DOLLY: 1, PAN: 2 }, gi = { ROTATE: 0, PAN: 1, DOLLY_PAN: 2, DOLLY_ROTATE: 3 }, $l = 0, za = 1, ql = 2, tl = 1, jl = 2, mn = 3, In = 0, Ue = 1, Xe = 2, Ln = 0, Mi = 1, Ha = 2, ka = 3, Va = 4, Zl = 5, Wn = 100, Kl = 101, Jl = 102, Ql = 103, tc = 104, ec = 200, nc = 201, ic = 202, sc = 203, br = 204, Ar = 205, rc = 206, ac = 207, oc = 208, lc = 209, cc = 210, hc = 211, uc = 212, dc = 213, fc = 214, Tr = 0, wr = 1, Rr = 2, Ei = 3, Cr = 4, Pr = 5, Dr = 6, Lr = 7, el = 0, pc = 1, mc = 2, Un = 0, _c = 1, gc = 2, vc = 3, xc = 4, Mc = 5, Sc = 6, yc = 7, nl = 300, bi = 301, Ai = 302, Ur = 303, Ir = 304, Ns = 306, Nr = 1e3, Yn = 1001, Fr = 1002, tn = 1003, Ec = 1004, ji = 1005, an = 1006, ks = 1007, $n = 1008, Sn = 1009, il = 1010, sl = 1011, Vi = 1012, pa = 1013, jn = 1014, gn = 1015, Xi = 1016, ma = 1017, _a = 1018, Ti = 1020, rl = 35902, al = 1021, ol = 1022, Qe = 1023, ll = 1024, cl = 1025, Si = 1026, wi = 1027, hl = 1028, ga = 1029, ul = 1030, va = 1031, xa = 1033, bs = 33776, As = 33777, Ts = 33778, ws = 33779, Or = 35840, Br = 35841, zr = 35842, Hr = 35843, kr = 36196, Vr = 37492, Gr = 37496, Wr = 37808, Xr = 37809, Yr = 37810, $r = 37811, qr = 37812, jr = 37813, Zr = 37814, Kr = 37815, Jr = 37816, Qr = 37817, ta = 37818, ea = 37819, na = 37820, ia = 37821, Rs = 36492, sa = 36494, ra = 36495, dl = 36283, aa = 36284, oa = 36285, la = 36286, bc = 3200, Ac = 3201, fl = 0, Tc = 1, Pn = "", Be = "srgb", Ri = "srgb-linear", Ps = "linear", jt = "srgb", ni = 7680, Ga = 519, wc = 512, Rc = 513, Cc = 514, pl = 515, Pc = 516, Dc = 517, Lc = 518, Uc = 519, ca = 35044, Wa = "300 es", vn = 2e3, Ds = 2001;
class Qn {
  addEventListener(t, e) {
    this._listeners === void 0 && (this._listeners = {});
    const n = this._listeners;
    n[t] === void 0 && (n[t] = []), n[t].indexOf(e) === -1 && n[t].push(e);
  }
  hasEventListener(t, e) {
    if (this._listeners === void 0) return !1;
    const n = this._listeners;
    return n[t] !== void 0 && n[t].indexOf(e) !== -1;
  }
  removeEventListener(t, e) {
    if (this._listeners === void 0) return;
    const s = this._listeners[t];
    if (s !== void 0) {
      const r = s.indexOf(e);
      r !== -1 && s.splice(r, 1);
    }
  }
  dispatchEvent(t) {
    if (this._listeners === void 0) return;
    const n = this._listeners[t.type];
    if (n !== void 0) {
      t.target = this;
      const s = n.slice(0);
      for (let r = 0, a = s.length; r < a; r++)
        s[r].call(this, t);
      t.target = null;
    }
  }
}
const ye = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "0a", "0b", "0c", "0d", "0e", "0f", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "1a", "1b", "1c", "1d", "1e", "1f", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "2a", "2b", "2c", "2d", "2e", "2f", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "3a", "3b", "3c", "3d", "3e", "3f", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "4a", "4b", "4c", "4d", "4e", "4f", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5a", "5b", "5c", "5d", "5e", "5f", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "6a", "6b", "6c", "6d", "6e", "6f", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7a", "7b", "7c", "7d", "7e", "7f", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "8a", "8b", "8c", "8d", "8e", "8f", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "9a", "9b", "9c", "9d", "9e", "9f", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "aa", "ab", "ac", "ad", "ae", "af", "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "ba", "bb", "bc", "bd", "be", "bf", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "ca", "cb", "cc", "cd", "ce", "cf", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "da", "db", "dc", "dd", "de", "df", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "ea", "eb", "ec", "ed", "ee", "ef", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "fa", "fb", "fc", "fd", "fe", "ff"];
let Xa = 1234567;
const Hi = Math.PI / 180, Gi = 180 / Math.PI;
function xn() {
  const i = Math.random() * 4294967295 | 0, t = Math.random() * 4294967295 | 0, e = Math.random() * 4294967295 | 0, n = Math.random() * 4294967295 | 0;
  return (ye[i & 255] + ye[i >> 8 & 255] + ye[i >> 16 & 255] + ye[i >> 24 & 255] + "-" + ye[t & 255] + ye[t >> 8 & 255] + "-" + ye[t >> 16 & 15 | 64] + ye[t >> 24 & 255] + "-" + ye[e & 63 | 128] + ye[e >> 8 & 255] + "-" + ye[e >> 16 & 255] + ye[e >> 24 & 255] + ye[n & 255] + ye[n >> 8 & 255] + ye[n >> 16 & 255] + ye[n >> 24 & 255]).toLowerCase();
}
function Ut(i, t, e) {
  return Math.max(t, Math.min(e, i));
}
function Ma(i, t) {
  return (i % t + t) % t;
}
function Ic(i, t, e, n, s) {
  return n + (i - t) * (s - n) / (e - t);
}
function Nc(i, t, e) {
  return i !== t ? (e - i) / (t - i) : 0;
}
function ki(i, t, e) {
  return (1 - e) * i + e * t;
}
function Fc(i, t, e, n) {
  return ki(i, t, 1 - Math.exp(-e * n));
}
function Oc(i, t = 1) {
  return t - Math.abs(Ma(i, t * 2) - t);
}
function Bc(i, t, e) {
  return i <= t ? 0 : i >= e ? 1 : (i = (i - t) / (e - t), i * i * (3 - 2 * i));
}
function zc(i, t, e) {
  return i <= t ? 0 : i >= e ? 1 : (i = (i - t) / (e - t), i * i * i * (i * (i * 6 - 15) + 10));
}
function Hc(i, t) {
  return i + Math.floor(Math.random() * (t - i + 1));
}
function kc(i, t) {
  return i + Math.random() * (t - i);
}
function Vc(i) {
  return i * (0.5 - Math.random());
}
function Gc(i) {
  i !== void 0 && (Xa = i);
  let t = Xa += 1831565813;
  return t = Math.imul(t ^ t >>> 15, t | 1), t ^= t + Math.imul(t ^ t >>> 7, t | 61), ((t ^ t >>> 14) >>> 0) / 4294967296;
}
function Wc(i) {
  return i * Hi;
}
function Xc(i) {
  return i * Gi;
}
function Yc(i) {
  return (i & i - 1) === 0 && i !== 0;
}
function $c(i) {
  return Math.pow(2, Math.ceil(Math.log(i) / Math.LN2));
}
function qc(i) {
  return Math.pow(2, Math.floor(Math.log(i) / Math.LN2));
}
function jc(i, t, e, n, s) {
  const r = Math.cos, a = Math.sin, o = r(e / 2), l = a(e / 2), c = r((t + n) / 2), u = a((t + n) / 2), d = r((t - n) / 2), f = a((t - n) / 2), m = r((n - t) / 2), g = a((n - t) / 2);
  switch (s) {
    case "XYX":
      i.set(o * u, l * d, l * f, o * c);
      break;
    case "YZY":
      i.set(l * f, o * u, l * d, o * c);
      break;
    case "ZXZ":
      i.set(l * d, l * f, o * u, o * c);
      break;
    case "XZX":
      i.set(o * u, l * g, l * m, o * c);
      break;
    case "YXY":
      i.set(l * m, o * u, l * g, o * c);
      break;
    case "ZYZ":
      i.set(l * g, l * m, o * u, o * c);
      break;
    default:
      console.warn("THREE.MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: " + s);
  }
}
function Ke(i, t) {
  switch (t.constructor) {
    case Float32Array:
      return i;
    case Uint32Array:
      return i / 4294967295;
    case Uint16Array:
      return i / 65535;
    case Uint8Array:
      return i / 255;
    case Int32Array:
      return Math.max(i / 2147483647, -1);
    case Int16Array:
      return Math.max(i / 32767, -1);
    case Int8Array:
      return Math.max(i / 127, -1);
    default:
      throw new Error("Invalid component type.");
  }
}
function qt(i, t) {
  switch (t.constructor) {
    case Float32Array:
      return i;
    case Uint32Array:
      return Math.round(i * 4294967295);
    case Uint16Array:
      return Math.round(i * 65535);
    case Uint8Array:
      return Math.round(i * 255);
    case Int32Array:
      return Math.round(i * 2147483647);
    case Int16Array:
      return Math.round(i * 32767);
    case Int8Array:
      return Math.round(i * 127);
    default:
      throw new Error("Invalid component type.");
  }
}
const ml = {
  DEG2RAD: Hi,
  RAD2DEG: Gi,
  generateUUID: xn,
  clamp: Ut,
  euclideanModulo: Ma,
  mapLinear: Ic,
  inverseLerp: Nc,
  lerp: ki,
  damp: Fc,
  pingpong: Oc,
  smoothstep: Bc,
  smootherstep: zc,
  randInt: Hc,
  randFloat: kc,
  randFloatSpread: Vc,
  seededRandom: Gc,
  degToRad: Wc,
  radToDeg: Xc,
  isPowerOfTwo: Yc,
  ceilPowerOfTwo: $c,
  floorPowerOfTwo: qc,
  setQuaternionFromProperEuler: jc,
  normalize: qt,
  denormalize: Ke
};
class bt {
  constructor(t = 0, e = 0) {
    bt.prototype.isVector2 = !0, this.x = t, this.y = e;
  }
  get width() {
    return this.x;
  }
  set width(t) {
    this.x = t;
  }
  get height() {
    return this.y;
  }
  set height(t) {
    this.y = t;
  }
  set(t, e) {
    return this.x = t, this.y = e, this;
  }
  setScalar(t) {
    return this.x = t, this.y = t, this;
  }
  setX(t) {
    return this.x = t, this;
  }
  setY(t) {
    return this.y = t, this;
  }
  setComponent(t, e) {
    switch (t) {
      case 0:
        this.x = e;
        break;
      case 1:
        this.y = e;
        break;
      default:
        throw new Error("index is out of range: " + t);
    }
    return this;
  }
  getComponent(t) {
    switch (t) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      default:
        throw new Error("index is out of range: " + t);
    }
  }
  clone() {
    return new this.constructor(this.x, this.y);
  }
  copy(t) {
    return this.x = t.x, this.y = t.y, this;
  }
  add(t) {
    return this.x += t.x, this.y += t.y, this;
  }
  addScalar(t) {
    return this.x += t, this.y += t, this;
  }
  addVectors(t, e) {
    return this.x = t.x + e.x, this.y = t.y + e.y, this;
  }
  addScaledVector(t, e) {
    return this.x += t.x * e, this.y += t.y * e, this;
  }
  sub(t) {
    return this.x -= t.x, this.y -= t.y, this;
  }
  subScalar(t) {
    return this.x -= t, this.y -= t, this;
  }
  subVectors(t, e) {
    return this.x = t.x - e.x, this.y = t.y - e.y, this;
  }
  multiply(t) {
    return this.x *= t.x, this.y *= t.y, this;
  }
  multiplyScalar(t) {
    return this.x *= t, this.y *= t, this;
  }
  divide(t) {
    return this.x /= t.x, this.y /= t.y, this;
  }
  divideScalar(t) {
    return this.multiplyScalar(1 / t);
  }
  applyMatrix3(t) {
    const e = this.x, n = this.y, s = t.elements;
    return this.x = s[0] * e + s[3] * n + s[6], this.y = s[1] * e + s[4] * n + s[7], this;
  }
  min(t) {
    return this.x = Math.min(this.x, t.x), this.y = Math.min(this.y, t.y), this;
  }
  max(t) {
    return this.x = Math.max(this.x, t.x), this.y = Math.max(this.y, t.y), this;
  }
  clamp(t, e) {
    return this.x = Ut(this.x, t.x, e.x), this.y = Ut(this.y, t.y, e.y), this;
  }
  clampScalar(t, e) {
    return this.x = Ut(this.x, t, e), this.y = Ut(this.y, t, e), this;
  }
  clampLength(t, e) {
    const n = this.length();
    return this.divideScalar(n || 1).multiplyScalar(Ut(n, t, e));
  }
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this;
  }
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this;
  }
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this;
  }
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this;
  }
  negate() {
    return this.x = -this.x, this.y = -this.y, this;
  }
  dot(t) {
    return this.x * t.x + this.y * t.y;
  }
  cross(t) {
    return this.x * t.y - this.y * t.x;
  }
  lengthSq() {
    return this.x * this.x + this.y * this.y;
  }
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y);
  }
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  angle() {
    return Math.atan2(-this.y, -this.x) + Math.PI;
  }
  angleTo(t) {
    const e = Math.sqrt(this.lengthSq() * t.lengthSq());
    if (e === 0) return Math.PI / 2;
    const n = this.dot(t) / e;
    return Math.acos(Ut(n, -1, 1));
  }
  distanceTo(t) {
    return Math.sqrt(this.distanceToSquared(t));
  }
  distanceToSquared(t) {
    const e = this.x - t.x, n = this.y - t.y;
    return e * e + n * n;
  }
  manhattanDistanceTo(t) {
    return Math.abs(this.x - t.x) + Math.abs(this.y - t.y);
  }
  setLength(t) {
    return this.normalize().multiplyScalar(t);
  }
  lerp(t, e) {
    return this.x += (t.x - this.x) * e, this.y += (t.y - this.y) * e, this;
  }
  lerpVectors(t, e, n) {
    return this.x = t.x + (e.x - t.x) * n, this.y = t.y + (e.y - t.y) * n, this;
  }
  equals(t) {
    return t.x === this.x && t.y === this.y;
  }
  fromArray(t, e = 0) {
    return this.x = t[e], this.y = t[e + 1], this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this.x, t[e + 1] = this.y, t;
  }
  fromBufferAttribute(t, e) {
    return this.x = t.getX(e), this.y = t.getY(e), this;
  }
  rotateAround(t, e) {
    const n = Math.cos(e), s = Math.sin(e), r = this.x - t.x, a = this.y - t.y;
    return this.x = r * n - a * s + t.x, this.y = r * s + a * n + t.y, this;
  }
  random() {
    return this.x = Math.random(), this.y = Math.random(), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y;
  }
}
class Pt {
  constructor(t, e, n, s, r, a, o, l, c) {
    Pt.prototype.isMatrix3 = !0, this.elements = [
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ], t !== void 0 && this.set(t, e, n, s, r, a, o, l, c);
  }
  set(t, e, n, s, r, a, o, l, c) {
    const u = this.elements;
    return u[0] = t, u[1] = s, u[2] = o, u[3] = e, u[4] = r, u[5] = l, u[6] = n, u[7] = a, u[8] = c, this;
  }
  identity() {
    return this.set(
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1
    ), this;
  }
  copy(t) {
    const e = this.elements, n = t.elements;
    return e[0] = n[0], e[1] = n[1], e[2] = n[2], e[3] = n[3], e[4] = n[4], e[5] = n[5], e[6] = n[6], e[7] = n[7], e[8] = n[8], this;
  }
  extractBasis(t, e, n) {
    return t.setFromMatrix3Column(this, 0), e.setFromMatrix3Column(this, 1), n.setFromMatrix3Column(this, 2), this;
  }
  setFromMatrix4(t) {
    const e = t.elements;
    return this.set(
      e[0],
      e[4],
      e[8],
      e[1],
      e[5],
      e[9],
      e[2],
      e[6],
      e[10]
    ), this;
  }
  multiply(t) {
    return this.multiplyMatrices(this, t);
  }
  premultiply(t) {
    return this.multiplyMatrices(t, this);
  }
  multiplyMatrices(t, e) {
    const n = t.elements, s = e.elements, r = this.elements, a = n[0], o = n[3], l = n[6], c = n[1], u = n[4], d = n[7], f = n[2], m = n[5], g = n[8], v = s[0], p = s[3], h = s[6], E = s[1], b = s[4], S = s[7], L = s[2], T = s[5], R = s[8];
    return r[0] = a * v + o * E + l * L, r[3] = a * p + o * b + l * T, r[6] = a * h + o * S + l * R, r[1] = c * v + u * E + d * L, r[4] = c * p + u * b + d * T, r[7] = c * h + u * S + d * R, r[2] = f * v + m * E + g * L, r[5] = f * p + m * b + g * T, r[8] = f * h + m * S + g * R, this;
  }
  multiplyScalar(t) {
    const e = this.elements;
    return e[0] *= t, e[3] *= t, e[6] *= t, e[1] *= t, e[4] *= t, e[7] *= t, e[2] *= t, e[5] *= t, e[8] *= t, this;
  }
  determinant() {
    const t = this.elements, e = t[0], n = t[1], s = t[2], r = t[3], a = t[4], o = t[5], l = t[6], c = t[7], u = t[8];
    return e * a * u - e * o * c - n * r * u + n * o * l + s * r * c - s * a * l;
  }
  invert() {
    const t = this.elements, e = t[0], n = t[1], s = t[2], r = t[3], a = t[4], o = t[5], l = t[6], c = t[7], u = t[8], d = u * a - o * c, f = o * l - u * r, m = c * r - a * l, g = e * d + n * f + s * m;
    if (g === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0);
    const v = 1 / g;
    return t[0] = d * v, t[1] = (s * c - u * n) * v, t[2] = (o * n - s * a) * v, t[3] = f * v, t[4] = (u * e - s * l) * v, t[5] = (s * r - o * e) * v, t[6] = m * v, t[7] = (n * l - c * e) * v, t[8] = (a * e - n * r) * v, this;
  }
  transpose() {
    let t;
    const e = this.elements;
    return t = e[1], e[1] = e[3], e[3] = t, t = e[2], e[2] = e[6], e[6] = t, t = e[5], e[5] = e[7], e[7] = t, this;
  }
  getNormalMatrix(t) {
    return this.setFromMatrix4(t).invert().transpose();
  }
  transposeIntoArray(t) {
    const e = this.elements;
    return t[0] = e[0], t[1] = e[3], t[2] = e[6], t[3] = e[1], t[4] = e[4], t[5] = e[7], t[6] = e[2], t[7] = e[5], t[8] = e[8], this;
  }
  setUvTransform(t, e, n, s, r, a, o) {
    const l = Math.cos(r), c = Math.sin(r);
    return this.set(
      n * l,
      n * c,
      -n * (l * a + c * o) + a + t,
      -s * c,
      s * l,
      -s * (-c * a + l * o) + o + e,
      0,
      0,
      1
    ), this;
  }
  //
  scale(t, e) {
    return this.premultiply(Vs.makeScale(t, e)), this;
  }
  rotate(t) {
    return this.premultiply(Vs.makeRotation(-t)), this;
  }
  translate(t, e) {
    return this.premultiply(Vs.makeTranslation(t, e)), this;
  }
  // for 2D Transforms
  makeTranslation(t, e) {
    return t.isVector2 ? this.set(
      1,
      0,
      t.x,
      0,
      1,
      t.y,
      0,
      0,
      1
    ) : this.set(
      1,
      0,
      t,
      0,
      1,
      e,
      0,
      0,
      1
    ), this;
  }
  makeRotation(t) {
    const e = Math.cos(t), n = Math.sin(t);
    return this.set(
      e,
      -n,
      0,
      n,
      e,
      0,
      0,
      0,
      1
    ), this;
  }
  makeScale(t, e) {
    return this.set(
      t,
      0,
      0,
      0,
      e,
      0,
      0,
      0,
      1
    ), this;
  }
  //
  equals(t) {
    const e = this.elements, n = t.elements;
    for (let s = 0; s < 9; s++)
      if (e[s] !== n[s]) return !1;
    return !0;
  }
  fromArray(t, e = 0) {
    for (let n = 0; n < 9; n++)
      this.elements[n] = t[n + e];
    return this;
  }
  toArray(t = [], e = 0) {
    const n = this.elements;
    return t[e] = n[0], t[e + 1] = n[1], t[e + 2] = n[2], t[e + 3] = n[3], t[e + 4] = n[4], t[e + 5] = n[5], t[e + 6] = n[6], t[e + 7] = n[7], t[e + 8] = n[8], t;
  }
  clone() {
    return new this.constructor().fromArray(this.elements);
  }
}
const Vs = /* @__PURE__ */ new Pt();
function _l(i) {
  for (let t = i.length - 1; t >= 0; --t)
    if (i[t] >= 65535) return !0;
  return !1;
}
function Ls(i) {
  return document.createElementNS("http://www.w3.org/1999/xhtml", i);
}
function Zc() {
  const i = Ls("canvas");
  return i.style.display = "block", i;
}
const Ya = {};
function _i(i) {
  i in Ya || (Ya[i] = !0, console.warn(i));
}
function Kc(i, t, e) {
  return new Promise(function(n, s) {
    function r() {
      switch (i.clientWaitSync(t, i.SYNC_FLUSH_COMMANDS_BIT, 0)) {
        case i.WAIT_FAILED:
          s();
          break;
        case i.TIMEOUT_EXPIRED:
          setTimeout(r, e);
          break;
        default:
          n();
      }
    }
    setTimeout(r, e);
  });
}
function Jc(i) {
  const t = i.elements;
  t[2] = 0.5 * t[2] + 0.5 * t[3], t[6] = 0.5 * t[6] + 0.5 * t[7], t[10] = 0.5 * t[10] + 0.5 * t[11], t[14] = 0.5 * t[14] + 0.5 * t[15];
}
function Qc(i) {
  const t = i.elements;
  t[11] === -1 ? (t[10] = -t[10] - 1, t[14] = -t[14]) : (t[10] = -t[10], t[14] = -t[14] + 1);
}
const $a = /* @__PURE__ */ new Pt().set(
  0.4123908,
  0.3575843,
  0.1804808,
  0.212639,
  0.7151687,
  0.0721923,
  0.0193308,
  0.1191948,
  0.9505322
), qa = /* @__PURE__ */ new Pt().set(
  3.2409699,
  -1.5373832,
  -0.4986108,
  -0.9692436,
  1.8759675,
  0.0415551,
  0.0556301,
  -0.203977,
  1.0569715
);
function th() {
  const i = {
    enabled: !0,
    workingColorSpace: Ri,
    /**
     * Implementations of supported color spaces.
     *
     * Required:
     *	- primaries: chromaticity coordinates [ rx ry gx gy bx by ]
     *	- whitePoint: reference white [ x y ]
     *	- transfer: transfer function (pre-defined)
     *	- toXYZ: Matrix3 RGB to XYZ transform
     *	- fromXYZ: Matrix3 XYZ to RGB transform
     *	- luminanceCoefficients: RGB luminance coefficients
     *
     * Optional:
     *  - outputColorSpaceConfig: { drawingBufferColorSpace: ColorSpace }
     *  - workingColorSpaceConfig: { unpackColorSpace: ColorSpace }
     *
     * Reference:
     * - https://www.russellcottrell.com/photo/matrixCalculator.htm
     */
    spaces: {},
    convert: function(s, r, a) {
      return this.enabled === !1 || r === a || !r || !a || (this.spaces[r].transfer === jt && (s.r = Mn(s.r), s.g = Mn(s.g), s.b = Mn(s.b)), this.spaces[r].primaries !== this.spaces[a].primaries && (s.applyMatrix3(this.spaces[r].toXYZ), s.applyMatrix3(this.spaces[a].fromXYZ)), this.spaces[a].transfer === jt && (s.r = yi(s.r), s.g = yi(s.g), s.b = yi(s.b))), s;
    },
    fromWorkingColorSpace: function(s, r) {
      return this.convert(s, this.workingColorSpace, r);
    },
    toWorkingColorSpace: function(s, r) {
      return this.convert(s, r, this.workingColorSpace);
    },
    getPrimaries: function(s) {
      return this.spaces[s].primaries;
    },
    getTransfer: function(s) {
      return s === Pn ? Ps : this.spaces[s].transfer;
    },
    getLuminanceCoefficients: function(s, r = this.workingColorSpace) {
      return s.fromArray(this.spaces[r].luminanceCoefficients);
    },
    define: function(s) {
      Object.assign(this.spaces, s);
    },
    // Internal APIs
    _getMatrix: function(s, r, a) {
      return s.copy(this.spaces[r].toXYZ).multiply(this.spaces[a].fromXYZ);
    },
    _getDrawingBufferColorSpace: function(s) {
      return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace;
    },
    _getUnpackColorSpace: function(s = this.workingColorSpace) {
      return this.spaces[s].workingColorSpaceConfig.unpackColorSpace;
    }
  }, t = [0.64, 0.33, 0.3, 0.6, 0.15, 0.06], e = [0.2126, 0.7152, 0.0722], n = [0.3127, 0.329];
  return i.define({
    [Ri]: {
      primaries: t,
      whitePoint: n,
      transfer: Ps,
      toXYZ: $a,
      fromXYZ: qa,
      luminanceCoefficients: e,
      workingColorSpaceConfig: { unpackColorSpace: Be },
      outputColorSpaceConfig: { drawingBufferColorSpace: Be }
    },
    [Be]: {
      primaries: t,
      whitePoint: n,
      transfer: jt,
      toXYZ: $a,
      fromXYZ: qa,
      luminanceCoefficients: e,
      outputColorSpaceConfig: { drawingBufferColorSpace: Be }
    }
  }), i;
}
const Wt = /* @__PURE__ */ th();
function Mn(i) {
  return i < 0.04045 ? i * 0.0773993808 : Math.pow(i * 0.9478672986 + 0.0521327014, 2.4);
}
function yi(i) {
  return i < 31308e-7 ? i * 12.92 : 1.055 * Math.pow(i, 0.41666) - 0.055;
}
let ii;
class eh {
  static getDataURL(t) {
    if (/^data:/i.test(t.src) || typeof HTMLCanvasElement > "u")
      return t.src;
    let e;
    if (t instanceof HTMLCanvasElement)
      e = t;
    else {
      ii === void 0 && (ii = Ls("canvas")), ii.width = t.width, ii.height = t.height;
      const n = ii.getContext("2d");
      t instanceof ImageData ? n.putImageData(t, 0, 0) : n.drawImage(t, 0, 0, t.width, t.height), e = ii;
    }
    return e.width > 2048 || e.height > 2048 ? (console.warn("THREE.ImageUtils.getDataURL: Image converted to jpg for performance reasons", t), e.toDataURL("image/jpeg", 0.6)) : e.toDataURL("image/png");
  }
  static sRGBToLinear(t) {
    if (typeof HTMLImageElement < "u" && t instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && t instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && t instanceof ImageBitmap) {
      const e = Ls("canvas");
      e.width = t.width, e.height = t.height;
      const n = e.getContext("2d");
      n.drawImage(t, 0, 0, t.width, t.height);
      const s = n.getImageData(0, 0, t.width, t.height), r = s.data;
      for (let a = 0; a < r.length; a++)
        r[a] = Mn(r[a] / 255) * 255;
      return n.putImageData(s, 0, 0), e;
    } else if (t.data) {
      const e = t.data.slice(0);
      for (let n = 0; n < e.length; n++)
        e instanceof Uint8Array || e instanceof Uint8ClampedArray ? e[n] = Math.floor(Mn(e[n] / 255) * 255) : e[n] = Mn(e[n]);
      return {
        data: e,
        width: t.width,
        height: t.height
      };
    } else
      return console.warn("THREE.ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."), t;
  }
}
let nh = 0;
class gl {
  constructor(t = null) {
    this.isSource = !0, Object.defineProperty(this, "id", { value: nh++ }), this.uuid = xn(), this.data = t, this.dataReady = !0, this.version = 0;
  }
  set needsUpdate(t) {
    t === !0 && this.version++;
  }
  toJSON(t) {
    const e = t === void 0 || typeof t == "string";
    if (!e && t.images[this.uuid] !== void 0)
      return t.images[this.uuid];
    const n = {
      uuid: this.uuid,
      url: ""
    }, s = this.data;
    if (s !== null) {
      let r;
      if (Array.isArray(s)) {
        r = [];
        for (let a = 0, o = s.length; a < o; a++)
          s[a].isDataTexture ? r.push(Gs(s[a].image)) : r.push(Gs(s[a]));
      } else
        r = Gs(s);
      n.url = r;
    }
    return e || (t.images[this.uuid] = n), n;
  }
}
function Gs(i) {
  return typeof HTMLImageElement < "u" && i instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && i instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && i instanceof ImageBitmap ? eh.getDataURL(i) : i.data ? {
    data: Array.from(i.data),
    width: i.width,
    height: i.height,
    type: i.data.constructor.name
  } : (console.warn("THREE.Texture: Unable to serialize Texture."), {});
}
let ih = 0;
class Ie extends Qn {
  constructor(t = Ie.DEFAULT_IMAGE, e = Ie.DEFAULT_MAPPING, n = Yn, s = Yn, r = an, a = $n, o = Qe, l = Sn, c = Ie.DEFAULT_ANISOTROPY, u = Pn) {
    super(), this.isTexture = !0, Object.defineProperty(this, "id", { value: ih++ }), this.uuid = xn(), this.name = "", this.source = new gl(t), this.mipmaps = [], this.mapping = e, this.channel = 0, this.wrapS = n, this.wrapT = s, this.magFilter = r, this.minFilter = a, this.anisotropy = c, this.format = o, this.internalFormat = null, this.type = l, this.offset = new bt(0, 0), this.repeat = new bt(1, 1), this.center = new bt(0, 0), this.rotation = 0, this.matrixAutoUpdate = !0, this.matrix = new Pt(), this.generateMipmaps = !0, this.premultiplyAlpha = !1, this.flipY = !0, this.unpackAlignment = 4, this.colorSpace = u, this.userData = {}, this.version = 0, this.onUpdate = null, this.isRenderTargetTexture = !1, this.pmremVersion = 0;
  }
  get image() {
    return this.source.data;
  }
  set image(t = null) {
    this.source.data = t;
  }
  updateMatrix() {
    this.matrix.setUvTransform(this.offset.x, this.offset.y, this.repeat.x, this.repeat.y, this.rotation, this.center.x, this.center.y);
  }
  clone() {
    return new this.constructor().copy(this);
  }
  copy(t) {
    return this.name = t.name, this.source = t.source, this.mipmaps = t.mipmaps.slice(0), this.mapping = t.mapping, this.channel = t.channel, this.wrapS = t.wrapS, this.wrapT = t.wrapT, this.magFilter = t.magFilter, this.minFilter = t.minFilter, this.anisotropy = t.anisotropy, this.format = t.format, this.internalFormat = t.internalFormat, this.type = t.type, this.offset.copy(t.offset), this.repeat.copy(t.repeat), this.center.copy(t.center), this.rotation = t.rotation, this.matrixAutoUpdate = t.matrixAutoUpdate, this.matrix.copy(t.matrix), this.generateMipmaps = t.generateMipmaps, this.premultiplyAlpha = t.premultiplyAlpha, this.flipY = t.flipY, this.unpackAlignment = t.unpackAlignment, this.colorSpace = t.colorSpace, this.userData = JSON.parse(JSON.stringify(t.userData)), this.needsUpdate = !0, this;
  }
  toJSON(t) {
    const e = t === void 0 || typeof t == "string";
    if (!e && t.textures[this.uuid] !== void 0)
      return t.textures[this.uuid];
    const n = {
      metadata: {
        version: 4.6,
        type: "Texture",
        generator: "Texture.toJSON"
      },
      uuid: this.uuid,
      name: this.name,
      image: this.source.toJSON(t).uuid,
      mapping: this.mapping,
      channel: this.channel,
      repeat: [this.repeat.x, this.repeat.y],
      offset: [this.offset.x, this.offset.y],
      center: [this.center.x, this.center.y],
      rotation: this.rotation,
      wrap: [this.wrapS, this.wrapT],
      format: this.format,
      internalFormat: this.internalFormat,
      type: this.type,
      colorSpace: this.colorSpace,
      minFilter: this.minFilter,
      magFilter: this.magFilter,
      anisotropy: this.anisotropy,
      flipY: this.flipY,
      generateMipmaps: this.generateMipmaps,
      premultiplyAlpha: this.premultiplyAlpha,
      unpackAlignment: this.unpackAlignment
    };
    return Object.keys(this.userData).length > 0 && (n.userData = this.userData), e || (t.textures[this.uuid] = n), n;
  }
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
  transformUv(t) {
    if (this.mapping !== nl) return t;
    if (t.applyMatrix3(this.matrix), t.x < 0 || t.x > 1)
      switch (this.wrapS) {
        case Nr:
          t.x = t.x - Math.floor(t.x);
          break;
        case Yn:
          t.x = t.x < 0 ? 0 : 1;
          break;
        case Fr:
          Math.abs(Math.floor(t.x) % 2) === 1 ? t.x = Math.ceil(t.x) - t.x : t.x = t.x - Math.floor(t.x);
          break;
      }
    if (t.y < 0 || t.y > 1)
      switch (this.wrapT) {
        case Nr:
          t.y = t.y - Math.floor(t.y);
          break;
        case Yn:
          t.y = t.y < 0 ? 0 : 1;
          break;
        case Fr:
          Math.abs(Math.floor(t.y) % 2) === 1 ? t.y = Math.ceil(t.y) - t.y : t.y = t.y - Math.floor(t.y);
          break;
      }
    return this.flipY && (t.y = 1 - t.y), t;
  }
  set needsUpdate(t) {
    t === !0 && (this.version++, this.source.needsUpdate = !0);
  }
  set needsPMREMUpdate(t) {
    t === !0 && this.pmremVersion++;
  }
}
Ie.DEFAULT_IMAGE = null;
Ie.DEFAULT_MAPPING = nl;
Ie.DEFAULT_ANISOTROPY = 1;
class te {
  constructor(t = 0, e = 0, n = 0, s = 1) {
    te.prototype.isVector4 = !0, this.x = t, this.y = e, this.z = n, this.w = s;
  }
  get width() {
    return this.z;
  }
  set width(t) {
    this.z = t;
  }
  get height() {
    return this.w;
  }
  set height(t) {
    this.w = t;
  }
  set(t, e, n, s) {
    return this.x = t, this.y = e, this.z = n, this.w = s, this;
  }
  setScalar(t) {
    return this.x = t, this.y = t, this.z = t, this.w = t, this;
  }
  setX(t) {
    return this.x = t, this;
  }
  setY(t) {
    return this.y = t, this;
  }
  setZ(t) {
    return this.z = t, this;
  }
  setW(t) {
    return this.w = t, this;
  }
  setComponent(t, e) {
    switch (t) {
      case 0:
        this.x = e;
        break;
      case 1:
        this.y = e;
        break;
      case 2:
        this.z = e;
        break;
      case 3:
        this.w = e;
        break;
      default:
        throw new Error("index is out of range: " + t);
    }
    return this;
  }
  getComponent(t) {
    switch (t) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      case 3:
        return this.w;
      default:
        throw new Error("index is out of range: " + t);
    }
  }
  clone() {
    return new this.constructor(this.x, this.y, this.z, this.w);
  }
  copy(t) {
    return this.x = t.x, this.y = t.y, this.z = t.z, this.w = t.w !== void 0 ? t.w : 1, this;
  }
  add(t) {
    return this.x += t.x, this.y += t.y, this.z += t.z, this.w += t.w, this;
  }
  addScalar(t) {
    return this.x += t, this.y += t, this.z += t, this.w += t, this;
  }
  addVectors(t, e) {
    return this.x = t.x + e.x, this.y = t.y + e.y, this.z = t.z + e.z, this.w = t.w + e.w, this;
  }
  addScaledVector(t, e) {
    return this.x += t.x * e, this.y += t.y * e, this.z += t.z * e, this.w += t.w * e, this;
  }
  sub(t) {
    return this.x -= t.x, this.y -= t.y, this.z -= t.z, this.w -= t.w, this;
  }
  subScalar(t) {
    return this.x -= t, this.y -= t, this.z -= t, this.w -= t, this;
  }
  subVectors(t, e) {
    return this.x = t.x - e.x, this.y = t.y - e.y, this.z = t.z - e.z, this.w = t.w - e.w, this;
  }
  multiply(t) {
    return this.x *= t.x, this.y *= t.y, this.z *= t.z, this.w *= t.w, this;
  }
  multiplyScalar(t) {
    return this.x *= t, this.y *= t, this.z *= t, this.w *= t, this;
  }
  applyMatrix4(t) {
    const e = this.x, n = this.y, s = this.z, r = this.w, a = t.elements;
    return this.x = a[0] * e + a[4] * n + a[8] * s + a[12] * r, this.y = a[1] * e + a[5] * n + a[9] * s + a[13] * r, this.z = a[2] * e + a[6] * n + a[10] * s + a[14] * r, this.w = a[3] * e + a[7] * n + a[11] * s + a[15] * r, this;
  }
  divide(t) {
    return this.x /= t.x, this.y /= t.y, this.z /= t.z, this.w /= t.w, this;
  }
  divideScalar(t) {
    return this.multiplyScalar(1 / t);
  }
  setAxisAngleFromQuaternion(t) {
    this.w = 2 * Math.acos(t.w);
    const e = Math.sqrt(1 - t.w * t.w);
    return e < 1e-4 ? (this.x = 1, this.y = 0, this.z = 0) : (this.x = t.x / e, this.y = t.y / e, this.z = t.z / e), this;
  }
  setAxisAngleFromRotationMatrix(t) {
    let e, n, s, r;
    const l = t.elements, c = l[0], u = l[4], d = l[8], f = l[1], m = l[5], g = l[9], v = l[2], p = l[6], h = l[10];
    if (Math.abs(u - f) < 0.01 && Math.abs(d - v) < 0.01 && Math.abs(g - p) < 0.01) {
      if (Math.abs(u + f) < 0.1 && Math.abs(d + v) < 0.1 && Math.abs(g + p) < 0.1 && Math.abs(c + m + h - 3) < 0.1)
        return this.set(1, 0, 0, 0), this;
      e = Math.PI;
      const b = (c + 1) / 2, S = (m + 1) / 2, L = (h + 1) / 2, T = (u + f) / 4, R = (d + v) / 4, I = (g + p) / 4;
      return b > S && b > L ? b < 0.01 ? (n = 0, s = 0.707106781, r = 0.707106781) : (n = Math.sqrt(b), s = T / n, r = R / n) : S > L ? S < 0.01 ? (n = 0.707106781, s = 0, r = 0.707106781) : (s = Math.sqrt(S), n = T / s, r = I / s) : L < 0.01 ? (n = 0.707106781, s = 0.707106781, r = 0) : (r = Math.sqrt(L), n = R / r, s = I / r), this.set(n, s, r, e), this;
    }
    let E = Math.sqrt((p - g) * (p - g) + (d - v) * (d - v) + (f - u) * (f - u));
    return Math.abs(E) < 1e-3 && (E = 1), this.x = (p - g) / E, this.y = (d - v) / E, this.z = (f - u) / E, this.w = Math.acos((c + m + h - 1) / 2), this;
  }
  setFromMatrixPosition(t) {
    const e = t.elements;
    return this.x = e[12], this.y = e[13], this.z = e[14], this.w = e[15], this;
  }
  min(t) {
    return this.x = Math.min(this.x, t.x), this.y = Math.min(this.y, t.y), this.z = Math.min(this.z, t.z), this.w = Math.min(this.w, t.w), this;
  }
  max(t) {
    return this.x = Math.max(this.x, t.x), this.y = Math.max(this.y, t.y), this.z = Math.max(this.z, t.z), this.w = Math.max(this.w, t.w), this;
  }
  clamp(t, e) {
    return this.x = Ut(this.x, t.x, e.x), this.y = Ut(this.y, t.y, e.y), this.z = Ut(this.z, t.z, e.z), this.w = Ut(this.w, t.w, e.w), this;
  }
  clampScalar(t, e) {
    return this.x = Ut(this.x, t, e), this.y = Ut(this.y, t, e), this.z = Ut(this.z, t, e), this.w = Ut(this.w, t, e), this;
  }
  clampLength(t, e) {
    const n = this.length();
    return this.divideScalar(n || 1).multiplyScalar(Ut(n, t, e));
  }
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this.w = Math.floor(this.w), this;
  }
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this.w = Math.ceil(this.w), this;
  }
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this.w = Math.round(this.w), this;
  }
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this.w = Math.trunc(this.w), this;
  }
  negate() {
    return this.x = -this.x, this.y = -this.y, this.z = -this.z, this.w = -this.w, this;
  }
  dot(t) {
    return this.x * t.x + this.y * t.y + this.z * t.z + this.w * t.w;
  }
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
  }
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
  }
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z) + Math.abs(this.w);
  }
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  setLength(t) {
    return this.normalize().multiplyScalar(t);
  }
  lerp(t, e) {
    return this.x += (t.x - this.x) * e, this.y += (t.y - this.y) * e, this.z += (t.z - this.z) * e, this.w += (t.w - this.w) * e, this;
  }
  lerpVectors(t, e, n) {
    return this.x = t.x + (e.x - t.x) * n, this.y = t.y + (e.y - t.y) * n, this.z = t.z + (e.z - t.z) * n, this.w = t.w + (e.w - t.w) * n, this;
  }
  equals(t) {
    return t.x === this.x && t.y === this.y && t.z === this.z && t.w === this.w;
  }
  fromArray(t, e = 0) {
    return this.x = t[e], this.y = t[e + 1], this.z = t[e + 2], this.w = t[e + 3], this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this.x, t[e + 1] = this.y, t[e + 2] = this.z, t[e + 3] = this.w, t;
  }
  fromBufferAttribute(t, e) {
    return this.x = t.getX(e), this.y = t.getY(e), this.z = t.getZ(e), this.w = t.getW(e), this;
  }
  random() {
    return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this.w = Math.random(), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y, yield this.z, yield this.w;
  }
}
class sh extends Qn {
  constructor(t = 1, e = 1, n = {}) {
    super(), this.isRenderTarget = !0, this.width = t, this.height = e, this.depth = 1, this.scissor = new te(0, 0, t, e), this.scissorTest = !1, this.viewport = new te(0, 0, t, e);
    const s = { width: t, height: e, depth: 1 };
    n = Object.assign({
      generateMipmaps: !1,
      internalFormat: null,
      minFilter: an,
      depthBuffer: !0,
      stencilBuffer: !1,
      resolveDepthBuffer: !0,
      resolveStencilBuffer: !0,
      depthTexture: null,
      samples: 0,
      count: 1
    }, n);
    const r = new Ie(s, n.mapping, n.wrapS, n.wrapT, n.magFilter, n.minFilter, n.format, n.type, n.anisotropy, n.colorSpace);
    r.flipY = !1, r.generateMipmaps = n.generateMipmaps, r.internalFormat = n.internalFormat, this.textures = [];
    const a = n.count;
    for (let o = 0; o < a; o++)
      this.textures[o] = r.clone(), this.textures[o].isRenderTargetTexture = !0;
    this.depthBuffer = n.depthBuffer, this.stencilBuffer = n.stencilBuffer, this.resolveDepthBuffer = n.resolveDepthBuffer, this.resolveStencilBuffer = n.resolveStencilBuffer, this.depthTexture = n.depthTexture, this.samples = n.samples;
  }
  get texture() {
    return this.textures[0];
  }
  set texture(t) {
    this.textures[0] = t;
  }
  setSize(t, e, n = 1) {
    if (this.width !== t || this.height !== e || this.depth !== n) {
      this.width = t, this.height = e, this.depth = n;
      for (let s = 0, r = this.textures.length; s < r; s++)
        this.textures[s].image.width = t, this.textures[s].image.height = e, this.textures[s].image.depth = n;
      this.dispose();
    }
    this.viewport.set(0, 0, t, e), this.scissor.set(0, 0, t, e);
  }
  clone() {
    return new this.constructor().copy(this);
  }
  copy(t) {
    this.width = t.width, this.height = t.height, this.depth = t.depth, this.scissor.copy(t.scissor), this.scissorTest = t.scissorTest, this.viewport.copy(t.viewport), this.textures.length = 0;
    for (let n = 0, s = t.textures.length; n < s; n++)
      this.textures[n] = t.textures[n].clone(), this.textures[n].isRenderTargetTexture = !0;
    const e = Object.assign({}, t.texture.image);
    return this.texture.source = new gl(e), this.depthBuffer = t.depthBuffer, this.stencilBuffer = t.stencilBuffer, this.resolveDepthBuffer = t.resolveDepthBuffer, this.resolveStencilBuffer = t.resolveStencilBuffer, t.depthTexture !== null && (this.depthTexture = t.depthTexture.clone()), this.samples = t.samples, this;
  }
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
}
class Zn extends sh {
  constructor(t = 1, e = 1, n = {}) {
    super(t, e, n), this.isWebGLRenderTarget = !0;
  }
}
class vl extends Ie {
  constructor(t = null, e = 1, n = 1, s = 1) {
    super(null), this.isDataArrayTexture = !0, this.image = { data: t, width: e, height: n, depth: s }, this.magFilter = tn, this.minFilter = tn, this.wrapR = Yn, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1, this.layerUpdates = /* @__PURE__ */ new Set();
  }
  addLayerUpdate(t) {
    this.layerUpdates.add(t);
  }
  clearLayerUpdates() {
    this.layerUpdates.clear();
  }
}
class rh extends Ie {
  constructor(t = null, e = 1, n = 1, s = 1) {
    super(null), this.isData3DTexture = !0, this.image = { data: t, width: e, height: n, depth: s }, this.magFilter = tn, this.minFilter = tn, this.wrapR = Yn, this.generateMipmaps = !1, this.flipY = !1, this.unpackAlignment = 1;
  }
}
class Kn {
  constructor(t = 0, e = 0, n = 0, s = 1) {
    this.isQuaternion = !0, this._x = t, this._y = e, this._z = n, this._w = s;
  }
  static slerpFlat(t, e, n, s, r, a, o) {
    let l = n[s + 0], c = n[s + 1], u = n[s + 2], d = n[s + 3];
    const f = r[a + 0], m = r[a + 1], g = r[a + 2], v = r[a + 3];
    if (o === 0) {
      t[e + 0] = l, t[e + 1] = c, t[e + 2] = u, t[e + 3] = d;
      return;
    }
    if (o === 1) {
      t[e + 0] = f, t[e + 1] = m, t[e + 2] = g, t[e + 3] = v;
      return;
    }
    if (d !== v || l !== f || c !== m || u !== g) {
      let p = 1 - o;
      const h = l * f + c * m + u * g + d * v, E = h >= 0 ? 1 : -1, b = 1 - h * h;
      if (b > Number.EPSILON) {
        const L = Math.sqrt(b), T = Math.atan2(L, h * E);
        p = Math.sin(p * T) / L, o = Math.sin(o * T) / L;
      }
      const S = o * E;
      if (l = l * p + f * S, c = c * p + m * S, u = u * p + g * S, d = d * p + v * S, p === 1 - o) {
        const L = 1 / Math.sqrt(l * l + c * c + u * u + d * d);
        l *= L, c *= L, u *= L, d *= L;
      }
    }
    t[e] = l, t[e + 1] = c, t[e + 2] = u, t[e + 3] = d;
  }
  static multiplyQuaternionsFlat(t, e, n, s, r, a) {
    const o = n[s], l = n[s + 1], c = n[s + 2], u = n[s + 3], d = r[a], f = r[a + 1], m = r[a + 2], g = r[a + 3];
    return t[e] = o * g + u * d + l * m - c * f, t[e + 1] = l * g + u * f + c * d - o * m, t[e + 2] = c * g + u * m + o * f - l * d, t[e + 3] = u * g - o * d - l * f - c * m, t;
  }
  get x() {
    return this._x;
  }
  set x(t) {
    this._x = t, this._onChangeCallback();
  }
  get y() {
    return this._y;
  }
  set y(t) {
    this._y = t, this._onChangeCallback();
  }
  get z() {
    return this._z;
  }
  set z(t) {
    this._z = t, this._onChangeCallback();
  }
  get w() {
    return this._w;
  }
  set w(t) {
    this._w = t, this._onChangeCallback();
  }
  set(t, e, n, s) {
    return this._x = t, this._y = e, this._z = n, this._w = s, this._onChangeCallback(), this;
  }
  clone() {
    return new this.constructor(this._x, this._y, this._z, this._w);
  }
  copy(t) {
    return this._x = t.x, this._y = t.y, this._z = t.z, this._w = t.w, this._onChangeCallback(), this;
  }
  setFromEuler(t, e = !0) {
    const n = t._x, s = t._y, r = t._z, a = t._order, o = Math.cos, l = Math.sin, c = o(n / 2), u = o(s / 2), d = o(r / 2), f = l(n / 2), m = l(s / 2), g = l(r / 2);
    switch (a) {
      case "XYZ":
        this._x = f * u * d + c * m * g, this._y = c * m * d - f * u * g, this._z = c * u * g + f * m * d, this._w = c * u * d - f * m * g;
        break;
      case "YXZ":
        this._x = f * u * d + c * m * g, this._y = c * m * d - f * u * g, this._z = c * u * g - f * m * d, this._w = c * u * d + f * m * g;
        break;
      case "ZXY":
        this._x = f * u * d - c * m * g, this._y = c * m * d + f * u * g, this._z = c * u * g + f * m * d, this._w = c * u * d - f * m * g;
        break;
      case "ZYX":
        this._x = f * u * d - c * m * g, this._y = c * m * d + f * u * g, this._z = c * u * g - f * m * d, this._w = c * u * d + f * m * g;
        break;
      case "YZX":
        this._x = f * u * d + c * m * g, this._y = c * m * d + f * u * g, this._z = c * u * g - f * m * d, this._w = c * u * d - f * m * g;
        break;
      case "XZY":
        this._x = f * u * d - c * m * g, this._y = c * m * d - f * u * g, this._z = c * u * g + f * m * d, this._w = c * u * d + f * m * g;
        break;
      default:
        console.warn("THREE.Quaternion: .setFromEuler() encountered an unknown order: " + a);
    }
    return e === !0 && this._onChangeCallback(), this;
  }
  setFromAxisAngle(t, e) {
    const n = e / 2, s = Math.sin(n);
    return this._x = t.x * s, this._y = t.y * s, this._z = t.z * s, this._w = Math.cos(n), this._onChangeCallback(), this;
  }
  setFromRotationMatrix(t) {
    const e = t.elements, n = e[0], s = e[4], r = e[8], a = e[1], o = e[5], l = e[9], c = e[2], u = e[6], d = e[10], f = n + o + d;
    if (f > 0) {
      const m = 0.5 / Math.sqrt(f + 1);
      this._w = 0.25 / m, this._x = (u - l) * m, this._y = (r - c) * m, this._z = (a - s) * m;
    } else if (n > o && n > d) {
      const m = 2 * Math.sqrt(1 + n - o - d);
      this._w = (u - l) / m, this._x = 0.25 * m, this._y = (s + a) / m, this._z = (r + c) / m;
    } else if (o > d) {
      const m = 2 * Math.sqrt(1 + o - n - d);
      this._w = (r - c) / m, this._x = (s + a) / m, this._y = 0.25 * m, this._z = (l + u) / m;
    } else {
      const m = 2 * Math.sqrt(1 + d - n - o);
      this._w = (a - s) / m, this._x = (r + c) / m, this._y = (l + u) / m, this._z = 0.25 * m;
    }
    return this._onChangeCallback(), this;
  }
  setFromUnitVectors(t, e) {
    let n = t.dot(e) + 1;
    return n < Number.EPSILON ? (n = 0, Math.abs(t.x) > Math.abs(t.z) ? (this._x = -t.y, this._y = t.x, this._z = 0, this._w = n) : (this._x = 0, this._y = -t.z, this._z = t.y, this._w = n)) : (this._x = t.y * e.z - t.z * e.y, this._y = t.z * e.x - t.x * e.z, this._z = t.x * e.y - t.y * e.x, this._w = n), this.normalize();
  }
  angleTo(t) {
    return 2 * Math.acos(Math.abs(Ut(this.dot(t), -1, 1)));
  }
  rotateTowards(t, e) {
    const n = this.angleTo(t);
    if (n === 0) return this;
    const s = Math.min(1, e / n);
    return this.slerp(t, s), this;
  }
  identity() {
    return this.set(0, 0, 0, 1);
  }
  invert() {
    return this.conjugate();
  }
  conjugate() {
    return this._x *= -1, this._y *= -1, this._z *= -1, this._onChangeCallback(), this;
  }
  dot(t) {
    return this._x * t._x + this._y * t._y + this._z * t._z + this._w * t._w;
  }
  lengthSq() {
    return this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w;
  }
  length() {
    return Math.sqrt(this._x * this._x + this._y * this._y + this._z * this._z + this._w * this._w);
  }
  normalize() {
    let t = this.length();
    return t === 0 ? (this._x = 0, this._y = 0, this._z = 0, this._w = 1) : (t = 1 / t, this._x = this._x * t, this._y = this._y * t, this._z = this._z * t, this._w = this._w * t), this._onChangeCallback(), this;
  }
  multiply(t) {
    return this.multiplyQuaternions(this, t);
  }
  premultiply(t) {
    return this.multiplyQuaternions(t, this);
  }
  multiplyQuaternions(t, e) {
    const n = t._x, s = t._y, r = t._z, a = t._w, o = e._x, l = e._y, c = e._z, u = e._w;
    return this._x = n * u + a * o + s * c - r * l, this._y = s * u + a * l + r * o - n * c, this._z = r * u + a * c + n * l - s * o, this._w = a * u - n * o - s * l - r * c, this._onChangeCallback(), this;
  }
  slerp(t, e) {
    if (e === 0) return this;
    if (e === 1) return this.copy(t);
    const n = this._x, s = this._y, r = this._z, a = this._w;
    let o = a * t._w + n * t._x + s * t._y + r * t._z;
    if (o < 0 ? (this._w = -t._w, this._x = -t._x, this._y = -t._y, this._z = -t._z, o = -o) : this.copy(t), o >= 1)
      return this._w = a, this._x = n, this._y = s, this._z = r, this;
    const l = 1 - o * o;
    if (l <= Number.EPSILON) {
      const m = 1 - e;
      return this._w = m * a + e * this._w, this._x = m * n + e * this._x, this._y = m * s + e * this._y, this._z = m * r + e * this._z, this.normalize(), this;
    }
    const c = Math.sqrt(l), u = Math.atan2(c, o), d = Math.sin((1 - e) * u) / c, f = Math.sin(e * u) / c;
    return this._w = a * d + this._w * f, this._x = n * d + this._x * f, this._y = s * d + this._y * f, this._z = r * d + this._z * f, this._onChangeCallback(), this;
  }
  slerpQuaternions(t, e, n) {
    return this.copy(t).slerp(e, n);
  }
  random() {
    const t = 2 * Math.PI * Math.random(), e = 2 * Math.PI * Math.random(), n = Math.random(), s = Math.sqrt(1 - n), r = Math.sqrt(n);
    return this.set(
      s * Math.sin(t),
      s * Math.cos(t),
      r * Math.sin(e),
      r * Math.cos(e)
    );
  }
  equals(t) {
    return t._x === this._x && t._y === this._y && t._z === this._z && t._w === this._w;
  }
  fromArray(t, e = 0) {
    return this._x = t[e], this._y = t[e + 1], this._z = t[e + 2], this._w = t[e + 3], this._onChangeCallback(), this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this._x, t[e + 1] = this._y, t[e + 2] = this._z, t[e + 3] = this._w, t;
  }
  fromBufferAttribute(t, e) {
    return this._x = t.getX(e), this._y = t.getY(e), this._z = t.getZ(e), this._w = t.getW(e), this._onChangeCallback(), this;
  }
  toJSON() {
    return this.toArray();
  }
  _onChange(t) {
    return this._onChangeCallback = t, this;
  }
  _onChangeCallback() {
  }
  *[Symbol.iterator]() {
    yield this._x, yield this._y, yield this._z, yield this._w;
  }
}
class P {
  constructor(t = 0, e = 0, n = 0) {
    P.prototype.isVector3 = !0, this.x = t, this.y = e, this.z = n;
  }
  set(t, e, n) {
    return n === void 0 && (n = this.z), this.x = t, this.y = e, this.z = n, this;
  }
  setScalar(t) {
    return this.x = t, this.y = t, this.z = t, this;
  }
  setX(t) {
    return this.x = t, this;
  }
  setY(t) {
    return this.y = t, this;
  }
  setZ(t) {
    return this.z = t, this;
  }
  setComponent(t, e) {
    switch (t) {
      case 0:
        this.x = e;
        break;
      case 1:
        this.y = e;
        break;
      case 2:
        this.z = e;
        break;
      default:
        throw new Error("index is out of range: " + t);
    }
    return this;
  }
  getComponent(t) {
    switch (t) {
      case 0:
        return this.x;
      case 1:
        return this.y;
      case 2:
        return this.z;
      default:
        throw new Error("index is out of range: " + t);
    }
  }
  clone() {
    return new this.constructor(this.x, this.y, this.z);
  }
  copy(t) {
    return this.x = t.x, this.y = t.y, this.z = t.z, this;
  }
  add(t) {
    return this.x += t.x, this.y += t.y, this.z += t.z, this;
  }
  addScalar(t) {
    return this.x += t, this.y += t, this.z += t, this;
  }
  addVectors(t, e) {
    return this.x = t.x + e.x, this.y = t.y + e.y, this.z = t.z + e.z, this;
  }
  addScaledVector(t, e) {
    return this.x += t.x * e, this.y += t.y * e, this.z += t.z * e, this;
  }
  sub(t) {
    return this.x -= t.x, this.y -= t.y, this.z -= t.z, this;
  }
  subScalar(t) {
    return this.x -= t, this.y -= t, this.z -= t, this;
  }
  subVectors(t, e) {
    return this.x = t.x - e.x, this.y = t.y - e.y, this.z = t.z - e.z, this;
  }
  multiply(t) {
    return this.x *= t.x, this.y *= t.y, this.z *= t.z, this;
  }
  multiplyScalar(t) {
    return this.x *= t, this.y *= t, this.z *= t, this;
  }
  multiplyVectors(t, e) {
    return this.x = t.x * e.x, this.y = t.y * e.y, this.z = t.z * e.z, this;
  }
  applyEuler(t) {
    return this.applyQuaternion(ja.setFromEuler(t));
  }
  applyAxisAngle(t, e) {
    return this.applyQuaternion(ja.setFromAxisAngle(t, e));
  }
  applyMatrix3(t) {
    const e = this.x, n = this.y, s = this.z, r = t.elements;
    return this.x = r[0] * e + r[3] * n + r[6] * s, this.y = r[1] * e + r[4] * n + r[7] * s, this.z = r[2] * e + r[5] * n + r[8] * s, this;
  }
  applyNormalMatrix(t) {
    return this.applyMatrix3(t).normalize();
  }
  applyMatrix4(t) {
    const e = this.x, n = this.y, s = this.z, r = t.elements, a = 1 / (r[3] * e + r[7] * n + r[11] * s + r[15]);
    return this.x = (r[0] * e + r[4] * n + r[8] * s + r[12]) * a, this.y = (r[1] * e + r[5] * n + r[9] * s + r[13]) * a, this.z = (r[2] * e + r[6] * n + r[10] * s + r[14]) * a, this;
  }
  applyQuaternion(t) {
    const e = this.x, n = this.y, s = this.z, r = t.x, a = t.y, o = t.z, l = t.w, c = 2 * (a * s - o * n), u = 2 * (o * e - r * s), d = 2 * (r * n - a * e);
    return this.x = e + l * c + a * d - o * u, this.y = n + l * u + o * c - r * d, this.z = s + l * d + r * u - a * c, this;
  }
  project(t) {
    return this.applyMatrix4(t.matrixWorldInverse).applyMatrix4(t.projectionMatrix);
  }
  unproject(t) {
    return this.applyMatrix4(t.projectionMatrixInverse).applyMatrix4(t.matrixWorld);
  }
  transformDirection(t) {
    const e = this.x, n = this.y, s = this.z, r = t.elements;
    return this.x = r[0] * e + r[4] * n + r[8] * s, this.y = r[1] * e + r[5] * n + r[9] * s, this.z = r[2] * e + r[6] * n + r[10] * s, this.normalize();
  }
  divide(t) {
    return this.x /= t.x, this.y /= t.y, this.z /= t.z, this;
  }
  divideScalar(t) {
    return this.multiplyScalar(1 / t);
  }
  min(t) {
    return this.x = Math.min(this.x, t.x), this.y = Math.min(this.y, t.y), this.z = Math.min(this.z, t.z), this;
  }
  max(t) {
    return this.x = Math.max(this.x, t.x), this.y = Math.max(this.y, t.y), this.z = Math.max(this.z, t.z), this;
  }
  clamp(t, e) {
    return this.x = Ut(this.x, t.x, e.x), this.y = Ut(this.y, t.y, e.y), this.z = Ut(this.z, t.z, e.z), this;
  }
  clampScalar(t, e) {
    return this.x = Ut(this.x, t, e), this.y = Ut(this.y, t, e), this.z = Ut(this.z, t, e), this;
  }
  clampLength(t, e) {
    const n = this.length();
    return this.divideScalar(n || 1).multiplyScalar(Ut(n, t, e));
  }
  floor() {
    return this.x = Math.floor(this.x), this.y = Math.floor(this.y), this.z = Math.floor(this.z), this;
  }
  ceil() {
    return this.x = Math.ceil(this.x), this.y = Math.ceil(this.y), this.z = Math.ceil(this.z), this;
  }
  round() {
    return this.x = Math.round(this.x), this.y = Math.round(this.y), this.z = Math.round(this.z), this;
  }
  roundToZero() {
    return this.x = Math.trunc(this.x), this.y = Math.trunc(this.y), this.z = Math.trunc(this.z), this;
  }
  negate() {
    return this.x = -this.x, this.y = -this.y, this.z = -this.z, this;
  }
  dot(t) {
    return this.x * t.x + this.y * t.y + this.z * t.z;
  }
  // TODO lengthSquared?
  lengthSq() {
    return this.x * this.x + this.y * this.y + this.z * this.z;
  }
  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }
  manhattanLength() {
    return Math.abs(this.x) + Math.abs(this.y) + Math.abs(this.z);
  }
  normalize() {
    return this.divideScalar(this.length() || 1);
  }
  setLength(t) {
    return this.normalize().multiplyScalar(t);
  }
  lerp(t, e) {
    return this.x += (t.x - this.x) * e, this.y += (t.y - this.y) * e, this.z += (t.z - this.z) * e, this;
  }
  lerpVectors(t, e, n) {
    return this.x = t.x + (e.x - t.x) * n, this.y = t.y + (e.y - t.y) * n, this.z = t.z + (e.z - t.z) * n, this;
  }
  cross(t) {
    return this.crossVectors(this, t);
  }
  crossVectors(t, e) {
    const n = t.x, s = t.y, r = t.z, a = e.x, o = e.y, l = e.z;
    return this.x = s * l - r * o, this.y = r * a - n * l, this.z = n * o - s * a, this;
  }
  projectOnVector(t) {
    const e = t.lengthSq();
    if (e === 0) return this.set(0, 0, 0);
    const n = t.dot(this) / e;
    return this.copy(t).multiplyScalar(n);
  }
  projectOnPlane(t) {
    return Ws.copy(this).projectOnVector(t), this.sub(Ws);
  }
  reflect(t) {
    return this.sub(Ws.copy(t).multiplyScalar(2 * this.dot(t)));
  }
  angleTo(t) {
    const e = Math.sqrt(this.lengthSq() * t.lengthSq());
    if (e === 0) return Math.PI / 2;
    const n = this.dot(t) / e;
    return Math.acos(Ut(n, -1, 1));
  }
  distanceTo(t) {
    return Math.sqrt(this.distanceToSquared(t));
  }
  distanceToSquared(t) {
    const e = this.x - t.x, n = this.y - t.y, s = this.z - t.z;
    return e * e + n * n + s * s;
  }
  manhattanDistanceTo(t) {
    return Math.abs(this.x - t.x) + Math.abs(this.y - t.y) + Math.abs(this.z - t.z);
  }
  setFromSpherical(t) {
    return this.setFromSphericalCoords(t.radius, t.phi, t.theta);
  }
  setFromSphericalCoords(t, e, n) {
    const s = Math.sin(e) * t;
    return this.x = s * Math.sin(n), this.y = Math.cos(e) * t, this.z = s * Math.cos(n), this;
  }
  setFromCylindrical(t) {
    return this.setFromCylindricalCoords(t.radius, t.theta, t.y);
  }
  setFromCylindricalCoords(t, e, n) {
    return this.x = t * Math.sin(e), this.y = n, this.z = t * Math.cos(e), this;
  }
  setFromMatrixPosition(t) {
    const e = t.elements;
    return this.x = e[12], this.y = e[13], this.z = e[14], this;
  }
  setFromMatrixScale(t) {
    const e = this.setFromMatrixColumn(t, 0).length(), n = this.setFromMatrixColumn(t, 1).length(), s = this.setFromMatrixColumn(t, 2).length();
    return this.x = e, this.y = n, this.z = s, this;
  }
  setFromMatrixColumn(t, e) {
    return this.fromArray(t.elements, e * 4);
  }
  setFromMatrix3Column(t, e) {
    return this.fromArray(t.elements, e * 3);
  }
  setFromEuler(t) {
    return this.x = t._x, this.y = t._y, this.z = t._z, this;
  }
  setFromColor(t) {
    return this.x = t.r, this.y = t.g, this.z = t.b, this;
  }
  equals(t) {
    return t.x === this.x && t.y === this.y && t.z === this.z;
  }
  fromArray(t, e = 0) {
    return this.x = t[e], this.y = t[e + 1], this.z = t[e + 2], this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this.x, t[e + 1] = this.y, t[e + 2] = this.z, t;
  }
  fromBufferAttribute(t, e) {
    return this.x = t.getX(e), this.y = t.getY(e), this.z = t.getZ(e), this;
  }
  random() {
    return this.x = Math.random(), this.y = Math.random(), this.z = Math.random(), this;
  }
  randomDirection() {
    const t = Math.random() * Math.PI * 2, e = Math.random() * 2 - 1, n = Math.sqrt(1 - e * e);
    return this.x = n * Math.cos(t), this.y = e, this.z = n * Math.sin(t), this;
  }
  *[Symbol.iterator]() {
    yield this.x, yield this.y, yield this.z;
  }
}
const Ws = /* @__PURE__ */ new P(), ja = /* @__PURE__ */ new Kn();
class He {
  constructor(t = new P(1 / 0, 1 / 0, 1 / 0), e = new P(-1 / 0, -1 / 0, -1 / 0)) {
    this.isBox3 = !0, this.min = t, this.max = e;
  }
  set(t, e) {
    return this.min.copy(t), this.max.copy(e), this;
  }
  setFromArray(t) {
    this.makeEmpty();
    for (let e = 0, n = t.length; e < n; e += 3)
      this.expandByPoint(qe.fromArray(t, e));
    return this;
  }
  setFromBufferAttribute(t) {
    this.makeEmpty();
    for (let e = 0, n = t.count; e < n; e++)
      this.expandByPoint(qe.fromBufferAttribute(t, e));
    return this;
  }
  setFromPoints(t) {
    this.makeEmpty();
    for (let e = 0, n = t.length; e < n; e++)
      this.expandByPoint(t[e]);
    return this;
  }
  setFromCenterAndSize(t, e) {
    const n = qe.copy(e).multiplyScalar(0.5);
    return this.min.copy(t).sub(n), this.max.copy(t).add(n), this;
  }
  setFromObject(t, e = !1) {
    return this.makeEmpty(), this.expandByObject(t, e);
  }
  clone() {
    return new this.constructor().copy(this);
  }
  copy(t) {
    return this.min.copy(t.min), this.max.copy(t.max), this;
  }
  makeEmpty() {
    return this.min.x = this.min.y = this.min.z = 1 / 0, this.max.x = this.max.y = this.max.z = -1 / 0, this;
  }
  isEmpty() {
    return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z;
  }
  getCenter(t) {
    return this.isEmpty() ? t.set(0, 0, 0) : t.addVectors(this.min, this.max).multiplyScalar(0.5);
  }
  getSize(t) {
    return this.isEmpty() ? t.set(0, 0, 0) : t.subVectors(this.max, this.min);
  }
  expandByPoint(t) {
    return this.min.min(t), this.max.max(t), this;
  }
  expandByVector(t) {
    return this.min.sub(t), this.max.add(t), this;
  }
  expandByScalar(t) {
    return this.min.addScalar(-t), this.max.addScalar(t), this;
  }
  expandByObject(t, e = !1) {
    t.updateWorldMatrix(!1, !1);
    const n = t.geometry;
    if (n !== void 0) {
      const r = n.getAttribute("position");
      if (e === !0 && r !== void 0 && t.isInstancedMesh !== !0)
        for (let a = 0, o = r.count; a < o; a++)
          t.isMesh === !0 ? t.getVertexPosition(a, qe) : qe.fromBufferAttribute(r, a), qe.applyMatrix4(t.matrixWorld), this.expandByPoint(qe);
      else
        t.boundingBox !== void 0 ? (t.boundingBox === null && t.computeBoundingBox(), Zi.copy(t.boundingBox)) : (n.boundingBox === null && n.computeBoundingBox(), Zi.copy(n.boundingBox)), Zi.applyMatrix4(t.matrixWorld), this.union(Zi);
    }
    const s = t.children;
    for (let r = 0, a = s.length; r < a; r++)
      this.expandByObject(s[r], e);
    return this;
  }
  containsPoint(t) {
    return t.x >= this.min.x && t.x <= this.max.x && t.y >= this.min.y && t.y <= this.max.y && t.z >= this.min.z && t.z <= this.max.z;
  }
  containsBox(t) {
    return this.min.x <= t.min.x && t.max.x <= this.max.x && this.min.y <= t.min.y && t.max.y <= this.max.y && this.min.z <= t.min.z && t.max.z <= this.max.z;
  }
  getParameter(t, e) {
    return e.set(
      (t.x - this.min.x) / (this.max.x - this.min.x),
      (t.y - this.min.y) / (this.max.y - this.min.y),
      (t.z - this.min.z) / (this.max.z - this.min.z)
    );
  }
  intersectsBox(t) {
    return t.max.x >= this.min.x && t.min.x <= this.max.x && t.max.y >= this.min.y && t.min.y <= this.max.y && t.max.z >= this.min.z && t.min.z <= this.max.z;
  }
  intersectsSphere(t) {
    return this.clampPoint(t.center, qe), qe.distanceToSquared(t.center) <= t.radius * t.radius;
  }
  intersectsPlane(t) {
    let e, n;
    return t.normal.x > 0 ? (e = t.normal.x * this.min.x, n = t.normal.x * this.max.x) : (e = t.normal.x * this.max.x, n = t.normal.x * this.min.x), t.normal.y > 0 ? (e += t.normal.y * this.min.y, n += t.normal.y * this.max.y) : (e += t.normal.y * this.max.y, n += t.normal.y * this.min.y), t.normal.z > 0 ? (e += t.normal.z * this.min.z, n += t.normal.z * this.max.z) : (e += t.normal.z * this.max.z, n += t.normal.z * this.min.z), e <= -t.constant && n >= -t.constant;
  }
  intersectsTriangle(t) {
    if (this.isEmpty())
      return !1;
    this.getCenter(Ui), Ki.subVectors(this.max, Ui), si.subVectors(t.a, Ui), ri.subVectors(t.b, Ui), ai.subVectors(t.c, Ui), bn.subVectors(ri, si), An.subVectors(ai, ri), On.subVectors(si, ai);
    let e = [
      0,
      -bn.z,
      bn.y,
      0,
      -An.z,
      An.y,
      0,
      -On.z,
      On.y,
      bn.z,
      0,
      -bn.x,
      An.z,
      0,
      -An.x,
      On.z,
      0,
      -On.x,
      -bn.y,
      bn.x,
      0,
      -An.y,
      An.x,
      0,
      -On.y,
      On.x,
      0
    ];
    return !Xs(e, si, ri, ai, Ki) || (e = [1, 0, 0, 0, 1, 0, 0, 0, 1], !Xs(e, si, ri, ai, Ki)) ? !1 : (Ji.crossVectors(bn, An), e = [Ji.x, Ji.y, Ji.z], Xs(e, si, ri, ai, Ki));
  }
  clampPoint(t, e) {
    return e.copy(t).clamp(this.min, this.max);
  }
  distanceToPoint(t) {
    return this.clampPoint(t, qe).distanceTo(t);
  }
  getBoundingSphere(t) {
    return this.isEmpty() ? t.makeEmpty() : (this.getCenter(t.center), t.radius = this.getSize(qe).length() * 0.5), t;
  }
  intersect(t) {
    return this.min.max(t.min), this.max.min(t.max), this.isEmpty() && this.makeEmpty(), this;
  }
  union(t) {
    return this.min.min(t.min), this.max.max(t.max), this;
  }
  applyMatrix4(t) {
    return this.isEmpty() ? this : (cn[0].set(this.min.x, this.min.y, this.min.z).applyMatrix4(t), cn[1].set(this.min.x, this.min.y, this.max.z).applyMatrix4(t), cn[2].set(this.min.x, this.max.y, this.min.z).applyMatrix4(t), cn[3].set(this.min.x, this.max.y, this.max.z).applyMatrix4(t), cn[4].set(this.max.x, this.min.y, this.min.z).applyMatrix4(t), cn[5].set(this.max.x, this.min.y, this.max.z).applyMatrix4(t), cn[6].set(this.max.x, this.max.y, this.min.z).applyMatrix4(t), cn[7].set(this.max.x, this.max.y, this.max.z).applyMatrix4(t), this.setFromPoints(cn), this);
  }
  translate(t) {
    return this.min.add(t), this.max.add(t), this;
  }
  equals(t) {
    return t.min.equals(this.min) && t.max.equals(this.max);
  }
}
const cn = [
  /* @__PURE__ */ new P(),
  /* @__PURE__ */ new P(),
  /* @__PURE__ */ new P(),
  /* @__PURE__ */ new P(),
  /* @__PURE__ */ new P(),
  /* @__PURE__ */ new P(),
  /* @__PURE__ */ new P(),
  /* @__PURE__ */ new P()
], qe = /* @__PURE__ */ new P(), Zi = /* @__PURE__ */ new He(), si = /* @__PURE__ */ new P(), ri = /* @__PURE__ */ new P(), ai = /* @__PURE__ */ new P(), bn = /* @__PURE__ */ new P(), An = /* @__PURE__ */ new P(), On = /* @__PURE__ */ new P(), Ui = /* @__PURE__ */ new P(), Ki = /* @__PURE__ */ new P(), Ji = /* @__PURE__ */ new P(), Bn = /* @__PURE__ */ new P();
function Xs(i, t, e, n, s) {
  for (let r = 0, a = i.length - 3; r <= a; r += 3) {
    Bn.fromArray(i, r);
    const o = s.x * Math.abs(Bn.x) + s.y * Math.abs(Bn.y) + s.z * Math.abs(Bn.z), l = t.dot(Bn), c = e.dot(Bn), u = n.dot(Bn);
    if (Math.max(-Math.max(l, c, u), Math.min(l, c, u)) > o)
      return !1;
  }
  return !0;
}
const ah = /* @__PURE__ */ new He(), Ii = /* @__PURE__ */ new P(), Ys = /* @__PURE__ */ new P();
class Pi {
  constructor(t = new P(), e = -1) {
    this.isSphere = !0, this.center = t, this.radius = e;
  }
  set(t, e) {
    return this.center.copy(t), this.radius = e, this;
  }
  setFromPoints(t, e) {
    const n = this.center;
    e !== void 0 ? n.copy(e) : ah.setFromPoints(t).getCenter(n);
    let s = 0;
    for (let r = 0, a = t.length; r < a; r++)
      s = Math.max(s, n.distanceToSquared(t[r]));
    return this.radius = Math.sqrt(s), this;
  }
  copy(t) {
    return this.center.copy(t.center), this.radius = t.radius, this;
  }
  isEmpty() {
    return this.radius < 0;
  }
  makeEmpty() {
    return this.center.set(0, 0, 0), this.radius = -1, this;
  }
  containsPoint(t) {
    return t.distanceToSquared(this.center) <= this.radius * this.radius;
  }
  distanceToPoint(t) {
    return t.distanceTo(this.center) - this.radius;
  }
  intersectsSphere(t) {
    const e = this.radius + t.radius;
    return t.center.distanceToSquared(this.center) <= e * e;
  }
  intersectsBox(t) {
    return t.intersectsSphere(this);
  }
  intersectsPlane(t) {
    return Math.abs(t.distanceToPoint(this.center)) <= this.radius;
  }
  clampPoint(t, e) {
    const n = this.center.distanceToSquared(t);
    return e.copy(t), n > this.radius * this.radius && (e.sub(this.center).normalize(), e.multiplyScalar(this.radius).add(this.center)), e;
  }
  getBoundingBox(t) {
    return this.isEmpty() ? (t.makeEmpty(), t) : (t.set(this.center, this.center), t.expandByScalar(this.radius), t);
  }
  applyMatrix4(t) {
    return this.center.applyMatrix4(t), this.radius = this.radius * t.getMaxScaleOnAxis(), this;
  }
  translate(t) {
    return this.center.add(t), this;
  }
  expandByPoint(t) {
    if (this.isEmpty())
      return this.center.copy(t), this.radius = 0, this;
    Ii.subVectors(t, this.center);
    const e = Ii.lengthSq();
    if (e > this.radius * this.radius) {
      const n = Math.sqrt(e), s = (n - this.radius) * 0.5;
      this.center.addScaledVector(Ii, s / n), this.radius += s;
    }
    return this;
  }
  union(t) {
    return t.isEmpty() ? this : this.isEmpty() ? (this.copy(t), this) : (this.center.equals(t.center) === !0 ? this.radius = Math.max(this.radius, t.radius) : (Ys.subVectors(t.center, this.center).setLength(t.radius), this.expandByPoint(Ii.copy(t.center).add(Ys)), this.expandByPoint(Ii.copy(t.center).sub(Ys))), this);
  }
  equals(t) {
    return t.center.equals(this.center) && t.radius === this.radius;
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
const hn = /* @__PURE__ */ new P(), $s = /* @__PURE__ */ new P(), Qi = /* @__PURE__ */ new P(), Tn = /* @__PURE__ */ new P(), qs = /* @__PURE__ */ new P(), ts = /* @__PURE__ */ new P(), js = /* @__PURE__ */ new P();
class Sa {
  constructor(t = new P(), e = new P(0, 0, -1)) {
    this.origin = t, this.direction = e;
  }
  set(t, e) {
    return this.origin.copy(t), this.direction.copy(e), this;
  }
  copy(t) {
    return this.origin.copy(t.origin), this.direction.copy(t.direction), this;
  }
  at(t, e) {
    return e.copy(this.origin).addScaledVector(this.direction, t);
  }
  lookAt(t) {
    return this.direction.copy(t).sub(this.origin).normalize(), this;
  }
  recast(t) {
    return this.origin.copy(this.at(t, hn)), this;
  }
  closestPointToPoint(t, e) {
    e.subVectors(t, this.origin);
    const n = e.dot(this.direction);
    return n < 0 ? e.copy(this.origin) : e.copy(this.origin).addScaledVector(this.direction, n);
  }
  distanceToPoint(t) {
    return Math.sqrt(this.distanceSqToPoint(t));
  }
  distanceSqToPoint(t) {
    const e = hn.subVectors(t, this.origin).dot(this.direction);
    return e < 0 ? this.origin.distanceToSquared(t) : (hn.copy(this.origin).addScaledVector(this.direction, e), hn.distanceToSquared(t));
  }
  distanceSqToSegment(t, e, n, s) {
    $s.copy(t).add(e).multiplyScalar(0.5), Qi.copy(e).sub(t).normalize(), Tn.copy(this.origin).sub($s);
    const r = t.distanceTo(e) * 0.5, a = -this.direction.dot(Qi), o = Tn.dot(this.direction), l = -Tn.dot(Qi), c = Tn.lengthSq(), u = Math.abs(1 - a * a);
    let d, f, m, g;
    if (u > 0)
      if (d = a * l - o, f = a * o - l, g = r * u, d >= 0)
        if (f >= -g)
          if (f <= g) {
            const v = 1 / u;
            d *= v, f *= v, m = d * (d + a * f + 2 * o) + f * (a * d + f + 2 * l) + c;
          } else
            f = r, d = Math.max(0, -(a * f + o)), m = -d * d + f * (f + 2 * l) + c;
        else
          f = -r, d = Math.max(0, -(a * f + o)), m = -d * d + f * (f + 2 * l) + c;
      else
        f <= -g ? (d = Math.max(0, -(-a * r + o)), f = d > 0 ? -r : Math.min(Math.max(-r, -l), r), m = -d * d + f * (f + 2 * l) + c) : f <= g ? (d = 0, f = Math.min(Math.max(-r, -l), r), m = f * (f + 2 * l) + c) : (d = Math.max(0, -(a * r + o)), f = d > 0 ? r : Math.min(Math.max(-r, -l), r), m = -d * d + f * (f + 2 * l) + c);
    else
      f = a > 0 ? -r : r, d = Math.max(0, -(a * f + o)), m = -d * d + f * (f + 2 * l) + c;
    return n && n.copy(this.origin).addScaledVector(this.direction, d), s && s.copy($s).addScaledVector(Qi, f), m;
  }
  intersectSphere(t, e) {
    hn.subVectors(t.center, this.origin);
    const n = hn.dot(this.direction), s = hn.dot(hn) - n * n, r = t.radius * t.radius;
    if (s > r) return null;
    const a = Math.sqrt(r - s), o = n - a, l = n + a;
    return l < 0 ? null : o < 0 ? this.at(l, e) : this.at(o, e);
  }
  intersectsSphere(t) {
    return this.distanceSqToPoint(t.center) <= t.radius * t.radius;
  }
  distanceToPlane(t) {
    const e = t.normal.dot(this.direction);
    if (e === 0)
      return t.distanceToPoint(this.origin) === 0 ? 0 : null;
    const n = -(this.origin.dot(t.normal) + t.constant) / e;
    return n >= 0 ? n : null;
  }
  intersectPlane(t, e) {
    const n = this.distanceToPlane(t);
    return n === null ? null : this.at(n, e);
  }
  intersectsPlane(t) {
    const e = t.distanceToPoint(this.origin);
    return e === 0 || t.normal.dot(this.direction) * e < 0;
  }
  intersectBox(t, e) {
    let n, s, r, a, o, l;
    const c = 1 / this.direction.x, u = 1 / this.direction.y, d = 1 / this.direction.z, f = this.origin;
    return c >= 0 ? (n = (t.min.x - f.x) * c, s = (t.max.x - f.x) * c) : (n = (t.max.x - f.x) * c, s = (t.min.x - f.x) * c), u >= 0 ? (r = (t.min.y - f.y) * u, a = (t.max.y - f.y) * u) : (r = (t.max.y - f.y) * u, a = (t.min.y - f.y) * u), n > a || r > s || ((r > n || isNaN(n)) && (n = r), (a < s || isNaN(s)) && (s = a), d >= 0 ? (o = (t.min.z - f.z) * d, l = (t.max.z - f.z) * d) : (o = (t.max.z - f.z) * d, l = (t.min.z - f.z) * d), n > l || o > s) || ((o > n || n !== n) && (n = o), (l < s || s !== s) && (s = l), s < 0) ? null : this.at(n >= 0 ? n : s, e);
  }
  intersectsBox(t) {
    return this.intersectBox(t, hn) !== null;
  }
  intersectTriangle(t, e, n, s, r) {
    qs.subVectors(e, t), ts.subVectors(n, t), js.crossVectors(qs, ts);
    let a = this.direction.dot(js), o;
    if (a > 0) {
      if (s) return null;
      o = 1;
    } else if (a < 0)
      o = -1, a = -a;
    else
      return null;
    Tn.subVectors(this.origin, t);
    const l = o * this.direction.dot(ts.crossVectors(Tn, ts));
    if (l < 0)
      return null;
    const c = o * this.direction.dot(qs.cross(Tn));
    if (c < 0 || l + c > a)
      return null;
    const u = -o * Tn.dot(js);
    return u < 0 ? null : this.at(u / a, r);
  }
  applyMatrix4(t) {
    return this.origin.applyMatrix4(t), this.direction.transformDirection(t), this;
  }
  equals(t) {
    return t.origin.equals(this.origin) && t.direction.equals(this.direction);
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
class ne {
  constructor(t, e, n, s, r, a, o, l, c, u, d, f, m, g, v, p) {
    ne.prototype.isMatrix4 = !0, this.elements = [
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ], t !== void 0 && this.set(t, e, n, s, r, a, o, l, c, u, d, f, m, g, v, p);
  }
  set(t, e, n, s, r, a, o, l, c, u, d, f, m, g, v, p) {
    const h = this.elements;
    return h[0] = t, h[4] = e, h[8] = n, h[12] = s, h[1] = r, h[5] = a, h[9] = o, h[13] = l, h[2] = c, h[6] = u, h[10] = d, h[14] = f, h[3] = m, h[7] = g, h[11] = v, h[15] = p, this;
  }
  identity() {
    return this.set(
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  clone() {
    return new ne().fromArray(this.elements);
  }
  copy(t) {
    const e = this.elements, n = t.elements;
    return e[0] = n[0], e[1] = n[1], e[2] = n[2], e[3] = n[3], e[4] = n[4], e[5] = n[5], e[6] = n[6], e[7] = n[7], e[8] = n[8], e[9] = n[9], e[10] = n[10], e[11] = n[11], e[12] = n[12], e[13] = n[13], e[14] = n[14], e[15] = n[15], this;
  }
  copyPosition(t) {
    const e = this.elements, n = t.elements;
    return e[12] = n[12], e[13] = n[13], e[14] = n[14], this;
  }
  setFromMatrix3(t) {
    const e = t.elements;
    return this.set(
      e[0],
      e[3],
      e[6],
      0,
      e[1],
      e[4],
      e[7],
      0,
      e[2],
      e[5],
      e[8],
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  extractBasis(t, e, n) {
    return t.setFromMatrixColumn(this, 0), e.setFromMatrixColumn(this, 1), n.setFromMatrixColumn(this, 2), this;
  }
  makeBasis(t, e, n) {
    return this.set(
      t.x,
      e.x,
      n.x,
      0,
      t.y,
      e.y,
      n.y,
      0,
      t.z,
      e.z,
      n.z,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  extractRotation(t) {
    const e = this.elements, n = t.elements, s = 1 / oi.setFromMatrixColumn(t, 0).length(), r = 1 / oi.setFromMatrixColumn(t, 1).length(), a = 1 / oi.setFromMatrixColumn(t, 2).length();
    return e[0] = n[0] * s, e[1] = n[1] * s, e[2] = n[2] * s, e[3] = 0, e[4] = n[4] * r, e[5] = n[5] * r, e[6] = n[6] * r, e[7] = 0, e[8] = n[8] * a, e[9] = n[9] * a, e[10] = n[10] * a, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, this;
  }
  makeRotationFromEuler(t) {
    const e = this.elements, n = t.x, s = t.y, r = t.z, a = Math.cos(n), o = Math.sin(n), l = Math.cos(s), c = Math.sin(s), u = Math.cos(r), d = Math.sin(r);
    if (t.order === "XYZ") {
      const f = a * u, m = a * d, g = o * u, v = o * d;
      e[0] = l * u, e[4] = -l * d, e[8] = c, e[1] = m + g * c, e[5] = f - v * c, e[9] = -o * l, e[2] = v - f * c, e[6] = g + m * c, e[10] = a * l;
    } else if (t.order === "YXZ") {
      const f = l * u, m = l * d, g = c * u, v = c * d;
      e[0] = f + v * o, e[4] = g * o - m, e[8] = a * c, e[1] = a * d, e[5] = a * u, e[9] = -o, e[2] = m * o - g, e[6] = v + f * o, e[10] = a * l;
    } else if (t.order === "ZXY") {
      const f = l * u, m = l * d, g = c * u, v = c * d;
      e[0] = f - v * o, e[4] = -a * d, e[8] = g + m * o, e[1] = m + g * o, e[5] = a * u, e[9] = v - f * o, e[2] = -a * c, e[6] = o, e[10] = a * l;
    } else if (t.order === "ZYX") {
      const f = a * u, m = a * d, g = o * u, v = o * d;
      e[0] = l * u, e[4] = g * c - m, e[8] = f * c + v, e[1] = l * d, e[5] = v * c + f, e[9] = m * c - g, e[2] = -c, e[6] = o * l, e[10] = a * l;
    } else if (t.order === "YZX") {
      const f = a * l, m = a * c, g = o * l, v = o * c;
      e[0] = l * u, e[4] = v - f * d, e[8] = g * d + m, e[1] = d, e[5] = a * u, e[9] = -o * u, e[2] = -c * u, e[6] = m * d + g, e[10] = f - v * d;
    } else if (t.order === "XZY") {
      const f = a * l, m = a * c, g = o * l, v = o * c;
      e[0] = l * u, e[4] = -d, e[8] = c * u, e[1] = f * d + v, e[5] = a * u, e[9] = m * d - g, e[2] = g * d - m, e[6] = o * u, e[10] = v * d + f;
    }
    return e[3] = 0, e[7] = 0, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, this;
  }
  makeRotationFromQuaternion(t) {
    return this.compose(oh, t, lh);
  }
  lookAt(t, e, n) {
    const s = this.elements;
    return Fe.subVectors(t, e), Fe.lengthSq() === 0 && (Fe.z = 1), Fe.normalize(), wn.crossVectors(n, Fe), wn.lengthSq() === 0 && (Math.abs(n.z) === 1 ? Fe.x += 1e-4 : Fe.z += 1e-4, Fe.normalize(), wn.crossVectors(n, Fe)), wn.normalize(), es.crossVectors(Fe, wn), s[0] = wn.x, s[4] = es.x, s[8] = Fe.x, s[1] = wn.y, s[5] = es.y, s[9] = Fe.y, s[2] = wn.z, s[6] = es.z, s[10] = Fe.z, this;
  }
  multiply(t) {
    return this.multiplyMatrices(this, t);
  }
  premultiply(t) {
    return this.multiplyMatrices(t, this);
  }
  multiplyMatrices(t, e) {
    const n = t.elements, s = e.elements, r = this.elements, a = n[0], o = n[4], l = n[8], c = n[12], u = n[1], d = n[5], f = n[9], m = n[13], g = n[2], v = n[6], p = n[10], h = n[14], E = n[3], b = n[7], S = n[11], L = n[15], T = s[0], R = s[4], I = s[8], y = s[12], M = s[1], C = s[5], H = s[9], z = s[13], G = s[2], j = s[6], W = s[10], Q = s[14], V = s[3], st = s[7], ht = s[11], gt = s[15];
    return r[0] = a * T + o * M + l * G + c * V, r[4] = a * R + o * C + l * j + c * st, r[8] = a * I + o * H + l * W + c * ht, r[12] = a * y + o * z + l * Q + c * gt, r[1] = u * T + d * M + f * G + m * V, r[5] = u * R + d * C + f * j + m * st, r[9] = u * I + d * H + f * W + m * ht, r[13] = u * y + d * z + f * Q + m * gt, r[2] = g * T + v * M + p * G + h * V, r[6] = g * R + v * C + p * j + h * st, r[10] = g * I + v * H + p * W + h * ht, r[14] = g * y + v * z + p * Q + h * gt, r[3] = E * T + b * M + S * G + L * V, r[7] = E * R + b * C + S * j + L * st, r[11] = E * I + b * H + S * W + L * ht, r[15] = E * y + b * z + S * Q + L * gt, this;
  }
  multiplyScalar(t) {
    const e = this.elements;
    return e[0] *= t, e[4] *= t, e[8] *= t, e[12] *= t, e[1] *= t, e[5] *= t, e[9] *= t, e[13] *= t, e[2] *= t, e[6] *= t, e[10] *= t, e[14] *= t, e[3] *= t, e[7] *= t, e[11] *= t, e[15] *= t, this;
  }
  determinant() {
    const t = this.elements, e = t[0], n = t[4], s = t[8], r = t[12], a = t[1], o = t[5], l = t[9], c = t[13], u = t[2], d = t[6], f = t[10], m = t[14], g = t[3], v = t[7], p = t[11], h = t[15];
    return g * (+r * l * d - s * c * d - r * o * f + n * c * f + s * o * m - n * l * m) + v * (+e * l * m - e * c * f + r * a * f - s * a * m + s * c * u - r * l * u) + p * (+e * c * d - e * o * m - r * a * d + n * a * m + r * o * u - n * c * u) + h * (-s * o * u - e * l * d + e * o * f + s * a * d - n * a * f + n * l * u);
  }
  transpose() {
    const t = this.elements;
    let e;
    return e = t[1], t[1] = t[4], t[4] = e, e = t[2], t[2] = t[8], t[8] = e, e = t[6], t[6] = t[9], t[9] = e, e = t[3], t[3] = t[12], t[12] = e, e = t[7], t[7] = t[13], t[13] = e, e = t[11], t[11] = t[14], t[14] = e, this;
  }
  setPosition(t, e, n) {
    const s = this.elements;
    return t.isVector3 ? (s[12] = t.x, s[13] = t.y, s[14] = t.z) : (s[12] = t, s[13] = e, s[14] = n), this;
  }
  invert() {
    const t = this.elements, e = t[0], n = t[1], s = t[2], r = t[3], a = t[4], o = t[5], l = t[6], c = t[7], u = t[8], d = t[9], f = t[10], m = t[11], g = t[12], v = t[13], p = t[14], h = t[15], E = d * p * c - v * f * c + v * l * m - o * p * m - d * l * h + o * f * h, b = g * f * c - u * p * c - g * l * m + a * p * m + u * l * h - a * f * h, S = u * v * c - g * d * c + g * o * m - a * v * m - u * o * h + a * d * h, L = g * d * l - u * v * l - g * o * f + a * v * f + u * o * p - a * d * p, T = e * E + n * b + s * S + r * L;
    if (T === 0) return this.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const R = 1 / T;
    return t[0] = E * R, t[1] = (v * f * r - d * p * r - v * s * m + n * p * m + d * s * h - n * f * h) * R, t[2] = (o * p * r - v * l * r + v * s * c - n * p * c - o * s * h + n * l * h) * R, t[3] = (d * l * r - o * f * r - d * s * c + n * f * c + o * s * m - n * l * m) * R, t[4] = b * R, t[5] = (u * p * r - g * f * r + g * s * m - e * p * m - u * s * h + e * f * h) * R, t[6] = (g * l * r - a * p * r - g * s * c + e * p * c + a * s * h - e * l * h) * R, t[7] = (a * f * r - u * l * r + u * s * c - e * f * c - a * s * m + e * l * m) * R, t[8] = S * R, t[9] = (g * d * r - u * v * r - g * n * m + e * v * m + u * n * h - e * d * h) * R, t[10] = (a * v * r - g * o * r + g * n * c - e * v * c - a * n * h + e * o * h) * R, t[11] = (u * o * r - a * d * r - u * n * c + e * d * c + a * n * m - e * o * m) * R, t[12] = L * R, t[13] = (u * v * s - g * d * s + g * n * f - e * v * f - u * n * p + e * d * p) * R, t[14] = (g * o * s - a * v * s - g * n * l + e * v * l + a * n * p - e * o * p) * R, t[15] = (a * d * s - u * o * s + u * n * l - e * d * l - a * n * f + e * o * f) * R, this;
  }
  scale(t) {
    const e = this.elements, n = t.x, s = t.y, r = t.z;
    return e[0] *= n, e[4] *= s, e[8] *= r, e[1] *= n, e[5] *= s, e[9] *= r, e[2] *= n, e[6] *= s, e[10] *= r, e[3] *= n, e[7] *= s, e[11] *= r, this;
  }
  getMaxScaleOnAxis() {
    const t = this.elements, e = t[0] * t[0] + t[1] * t[1] + t[2] * t[2], n = t[4] * t[4] + t[5] * t[5] + t[6] * t[6], s = t[8] * t[8] + t[9] * t[9] + t[10] * t[10];
    return Math.sqrt(Math.max(e, n, s));
  }
  makeTranslation(t, e, n) {
    return t.isVector3 ? this.set(
      1,
      0,
      0,
      t.x,
      0,
      1,
      0,
      t.y,
      0,
      0,
      1,
      t.z,
      0,
      0,
      0,
      1
    ) : this.set(
      1,
      0,
      0,
      t,
      0,
      1,
      0,
      e,
      0,
      0,
      1,
      n,
      0,
      0,
      0,
      1
    ), this;
  }
  makeRotationX(t) {
    const e = Math.cos(t), n = Math.sin(t);
    return this.set(
      1,
      0,
      0,
      0,
      0,
      e,
      -n,
      0,
      0,
      n,
      e,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeRotationY(t) {
    const e = Math.cos(t), n = Math.sin(t);
    return this.set(
      e,
      0,
      n,
      0,
      0,
      1,
      0,
      0,
      -n,
      0,
      e,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeRotationZ(t) {
    const e = Math.cos(t), n = Math.sin(t);
    return this.set(
      e,
      -n,
      0,
      0,
      n,
      e,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeRotationAxis(t, e) {
    const n = Math.cos(e), s = Math.sin(e), r = 1 - n, a = t.x, o = t.y, l = t.z, c = r * a, u = r * o;
    return this.set(
      c * a + n,
      c * o - s * l,
      c * l + s * o,
      0,
      c * o + s * l,
      u * o + n,
      u * l - s * a,
      0,
      c * l - s * o,
      u * l + s * a,
      r * l * l + n,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeScale(t, e, n) {
    return this.set(
      t,
      0,
      0,
      0,
      0,
      e,
      0,
      0,
      0,
      0,
      n,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  makeShear(t, e, n, s, r, a) {
    return this.set(
      1,
      n,
      r,
      0,
      t,
      1,
      a,
      0,
      e,
      s,
      1,
      0,
      0,
      0,
      0,
      1
    ), this;
  }
  compose(t, e, n) {
    const s = this.elements, r = e._x, a = e._y, o = e._z, l = e._w, c = r + r, u = a + a, d = o + o, f = r * c, m = r * u, g = r * d, v = a * u, p = a * d, h = o * d, E = l * c, b = l * u, S = l * d, L = n.x, T = n.y, R = n.z;
    return s[0] = (1 - (v + h)) * L, s[1] = (m + S) * L, s[2] = (g - b) * L, s[3] = 0, s[4] = (m - S) * T, s[5] = (1 - (f + h)) * T, s[6] = (p + E) * T, s[7] = 0, s[8] = (g + b) * R, s[9] = (p - E) * R, s[10] = (1 - (f + v)) * R, s[11] = 0, s[12] = t.x, s[13] = t.y, s[14] = t.z, s[15] = 1, this;
  }
  decompose(t, e, n) {
    const s = this.elements;
    let r = oi.set(s[0], s[1], s[2]).length();
    const a = oi.set(s[4], s[5], s[6]).length(), o = oi.set(s[8], s[9], s[10]).length();
    this.determinant() < 0 && (r = -r), t.x = s[12], t.y = s[13], t.z = s[14], je.copy(this);
    const c = 1 / r, u = 1 / a, d = 1 / o;
    return je.elements[0] *= c, je.elements[1] *= c, je.elements[2] *= c, je.elements[4] *= u, je.elements[5] *= u, je.elements[6] *= u, je.elements[8] *= d, je.elements[9] *= d, je.elements[10] *= d, e.setFromRotationMatrix(je), n.x = r, n.y = a, n.z = o, this;
  }
  makePerspective(t, e, n, s, r, a, o = vn) {
    const l = this.elements, c = 2 * r / (e - t), u = 2 * r / (n - s), d = (e + t) / (e - t), f = (n + s) / (n - s);
    let m, g;
    if (o === vn)
      m = -(a + r) / (a - r), g = -2 * a * r / (a - r);
    else if (o === Ds)
      m = -a / (a - r), g = -a * r / (a - r);
    else
      throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: " + o);
    return l[0] = c, l[4] = 0, l[8] = d, l[12] = 0, l[1] = 0, l[5] = u, l[9] = f, l[13] = 0, l[2] = 0, l[6] = 0, l[10] = m, l[14] = g, l[3] = 0, l[7] = 0, l[11] = -1, l[15] = 0, this;
  }
  makeOrthographic(t, e, n, s, r, a, o = vn) {
    const l = this.elements, c = 1 / (e - t), u = 1 / (n - s), d = 1 / (a - r), f = (e + t) * c, m = (n + s) * u;
    let g, v;
    if (o === vn)
      g = (a + r) * d, v = -2 * d;
    else if (o === Ds)
      g = r * d, v = -1 * d;
    else
      throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: " + o);
    return l[0] = 2 * c, l[4] = 0, l[8] = 0, l[12] = -f, l[1] = 0, l[5] = 2 * u, l[9] = 0, l[13] = -m, l[2] = 0, l[6] = 0, l[10] = v, l[14] = -g, l[3] = 0, l[7] = 0, l[11] = 0, l[15] = 1, this;
  }
  equals(t) {
    const e = this.elements, n = t.elements;
    for (let s = 0; s < 16; s++)
      if (e[s] !== n[s]) return !1;
    return !0;
  }
  fromArray(t, e = 0) {
    for (let n = 0; n < 16; n++)
      this.elements[n] = t[n + e];
    return this;
  }
  toArray(t = [], e = 0) {
    const n = this.elements;
    return t[e] = n[0], t[e + 1] = n[1], t[e + 2] = n[2], t[e + 3] = n[3], t[e + 4] = n[4], t[e + 5] = n[5], t[e + 6] = n[6], t[e + 7] = n[7], t[e + 8] = n[8], t[e + 9] = n[9], t[e + 10] = n[10], t[e + 11] = n[11], t[e + 12] = n[12], t[e + 13] = n[13], t[e + 14] = n[14], t[e + 15] = n[15], t;
  }
}
const oi = /* @__PURE__ */ new P(), je = /* @__PURE__ */ new ne(), oh = /* @__PURE__ */ new P(0, 0, 0), lh = /* @__PURE__ */ new P(1, 1, 1), wn = /* @__PURE__ */ new P(), es = /* @__PURE__ */ new P(), Fe = /* @__PURE__ */ new P(), Za = /* @__PURE__ */ new ne(), Ka = /* @__PURE__ */ new Kn();
class yn {
  constructor(t = 0, e = 0, n = 0, s = yn.DEFAULT_ORDER) {
    this.isEuler = !0, this._x = t, this._y = e, this._z = n, this._order = s;
  }
  get x() {
    return this._x;
  }
  set x(t) {
    this._x = t, this._onChangeCallback();
  }
  get y() {
    return this._y;
  }
  set y(t) {
    this._y = t, this._onChangeCallback();
  }
  get z() {
    return this._z;
  }
  set z(t) {
    this._z = t, this._onChangeCallback();
  }
  get order() {
    return this._order;
  }
  set order(t) {
    this._order = t, this._onChangeCallback();
  }
  set(t, e, n, s = this._order) {
    return this._x = t, this._y = e, this._z = n, this._order = s, this._onChangeCallback(), this;
  }
  clone() {
    return new this.constructor(this._x, this._y, this._z, this._order);
  }
  copy(t) {
    return this._x = t._x, this._y = t._y, this._z = t._z, this._order = t._order, this._onChangeCallback(), this;
  }
  setFromRotationMatrix(t, e = this._order, n = !0) {
    const s = t.elements, r = s[0], a = s[4], o = s[8], l = s[1], c = s[5], u = s[9], d = s[2], f = s[6], m = s[10];
    switch (e) {
      case "XYZ":
        this._y = Math.asin(Ut(o, -1, 1)), Math.abs(o) < 0.9999999 ? (this._x = Math.atan2(-u, m), this._z = Math.atan2(-a, r)) : (this._x = Math.atan2(f, c), this._z = 0);
        break;
      case "YXZ":
        this._x = Math.asin(-Ut(u, -1, 1)), Math.abs(u) < 0.9999999 ? (this._y = Math.atan2(o, m), this._z = Math.atan2(l, c)) : (this._y = Math.atan2(-d, r), this._z = 0);
        break;
      case "ZXY":
        this._x = Math.asin(Ut(f, -1, 1)), Math.abs(f) < 0.9999999 ? (this._y = Math.atan2(-d, m), this._z = Math.atan2(-a, c)) : (this._y = 0, this._z = Math.atan2(l, r));
        break;
      case "ZYX":
        this._y = Math.asin(-Ut(d, -1, 1)), Math.abs(d) < 0.9999999 ? (this._x = Math.atan2(f, m), this._z = Math.atan2(l, r)) : (this._x = 0, this._z = Math.atan2(-a, c));
        break;
      case "YZX":
        this._z = Math.asin(Ut(l, -1, 1)), Math.abs(l) < 0.9999999 ? (this._x = Math.atan2(-u, c), this._y = Math.atan2(-d, r)) : (this._x = 0, this._y = Math.atan2(o, m));
        break;
      case "XZY":
        this._z = Math.asin(-Ut(a, -1, 1)), Math.abs(a) < 0.9999999 ? (this._x = Math.atan2(f, c), this._y = Math.atan2(o, r)) : (this._x = Math.atan2(-u, m), this._y = 0);
        break;
      default:
        console.warn("THREE.Euler: .setFromRotationMatrix() encountered an unknown order: " + e);
    }
    return this._order = e, n === !0 && this._onChangeCallback(), this;
  }
  setFromQuaternion(t, e, n) {
    return Za.makeRotationFromQuaternion(t), this.setFromRotationMatrix(Za, e, n);
  }
  setFromVector3(t, e = this._order) {
    return this.set(t.x, t.y, t.z, e);
  }
  reorder(t) {
    return Ka.setFromEuler(this), this.setFromQuaternion(Ka, t);
  }
  equals(t) {
    return t._x === this._x && t._y === this._y && t._z === this._z && t._order === this._order;
  }
  fromArray(t) {
    return this._x = t[0], this._y = t[1], this._z = t[2], t[3] !== void 0 && (this._order = t[3]), this._onChangeCallback(), this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this._x, t[e + 1] = this._y, t[e + 2] = this._z, t[e + 3] = this._order, t;
  }
  _onChange(t) {
    return this._onChangeCallback = t, this;
  }
  _onChangeCallback() {
  }
  *[Symbol.iterator]() {
    yield this._x, yield this._y, yield this._z, yield this._order;
  }
}
yn.DEFAULT_ORDER = "XYZ";
class xl {
  constructor() {
    this.mask = 1;
  }
  set(t) {
    this.mask = (1 << t | 0) >>> 0;
  }
  enable(t) {
    this.mask |= 1 << t | 0;
  }
  enableAll() {
    this.mask = -1;
  }
  toggle(t) {
    this.mask ^= 1 << t | 0;
  }
  disable(t) {
    this.mask &= ~(1 << t | 0);
  }
  disableAll() {
    this.mask = 0;
  }
  test(t) {
    return (this.mask & t.mask) !== 0;
  }
  isEnabled(t) {
    return (this.mask & (1 << t | 0)) !== 0;
  }
}
let ch = 0;
const Ja = /* @__PURE__ */ new P(), li = /* @__PURE__ */ new Kn(), un = /* @__PURE__ */ new ne(), ns = /* @__PURE__ */ new P(), Ni = /* @__PURE__ */ new P(), hh = /* @__PURE__ */ new P(), uh = /* @__PURE__ */ new Kn(), Qa = /* @__PURE__ */ new P(1, 0, 0), to = /* @__PURE__ */ new P(0, 1, 0), eo = /* @__PURE__ */ new P(0, 0, 1), no = { type: "added" }, dh = { type: "removed" }, ci = { type: "childadded", child: null }, Zs = { type: "childremoved", child: null };
class Re extends Qn {
  constructor() {
    super(), this.isObject3D = !0, Object.defineProperty(this, "id", { value: ch++ }), this.uuid = xn(), this.name = "", this.type = "Object3D", this.parent = null, this.children = [], this.up = Re.DEFAULT_UP.clone();
    const t = new P(), e = new yn(), n = new Kn(), s = new P(1, 1, 1);
    function r() {
      n.setFromEuler(e, !1);
    }
    function a() {
      e.setFromQuaternion(n, void 0, !1);
    }
    e._onChange(r), n._onChange(a), Object.defineProperties(this, {
      position: {
        configurable: !0,
        enumerable: !0,
        value: t
      },
      rotation: {
        configurable: !0,
        enumerable: !0,
        value: e
      },
      quaternion: {
        configurable: !0,
        enumerable: !0,
        value: n
      },
      scale: {
        configurable: !0,
        enumerable: !0,
        value: s
      },
      modelViewMatrix: {
        value: new ne()
      },
      normalMatrix: {
        value: new Pt()
      }
    }), this.matrix = new ne(), this.matrixWorld = new ne(), this.matrixAutoUpdate = Re.DEFAULT_MATRIX_AUTO_UPDATE, this.matrixWorldAutoUpdate = Re.DEFAULT_MATRIX_WORLD_AUTO_UPDATE, this.matrixWorldNeedsUpdate = !1, this.layers = new xl(), this.visible = !0, this.castShadow = !1, this.receiveShadow = !1, this.frustumCulled = !0, this.renderOrder = 0, this.animations = [], this.userData = {};
  }
  onBeforeShadow() {
  }
  onAfterShadow() {
  }
  onBeforeRender() {
  }
  onAfterRender() {
  }
  applyMatrix4(t) {
    this.matrixAutoUpdate && this.updateMatrix(), this.matrix.premultiply(t), this.matrix.decompose(this.position, this.quaternion, this.scale);
  }
  applyQuaternion(t) {
    return this.quaternion.premultiply(t), this;
  }
  setRotationFromAxisAngle(t, e) {
    this.quaternion.setFromAxisAngle(t, e);
  }
  setRotationFromEuler(t) {
    this.quaternion.setFromEuler(t, !0);
  }
  setRotationFromMatrix(t) {
    this.quaternion.setFromRotationMatrix(t);
  }
  setRotationFromQuaternion(t) {
    this.quaternion.copy(t);
  }
  rotateOnAxis(t, e) {
    return li.setFromAxisAngle(t, e), this.quaternion.multiply(li), this;
  }
  rotateOnWorldAxis(t, e) {
    return li.setFromAxisAngle(t, e), this.quaternion.premultiply(li), this;
  }
  rotateX(t) {
    return this.rotateOnAxis(Qa, t);
  }
  rotateY(t) {
    return this.rotateOnAxis(to, t);
  }
  rotateZ(t) {
    return this.rotateOnAxis(eo, t);
  }
  translateOnAxis(t, e) {
    return Ja.copy(t).applyQuaternion(this.quaternion), this.position.add(Ja.multiplyScalar(e)), this;
  }
  translateX(t) {
    return this.translateOnAxis(Qa, t);
  }
  translateY(t) {
    return this.translateOnAxis(to, t);
  }
  translateZ(t) {
    return this.translateOnAxis(eo, t);
  }
  localToWorld(t) {
    return this.updateWorldMatrix(!0, !1), t.applyMatrix4(this.matrixWorld);
  }
  worldToLocal(t) {
    return this.updateWorldMatrix(!0, !1), t.applyMatrix4(un.copy(this.matrixWorld).invert());
  }
  lookAt(t, e, n) {
    t.isVector3 ? ns.copy(t) : ns.set(t, e, n);
    const s = this.parent;
    this.updateWorldMatrix(!0, !1), Ni.setFromMatrixPosition(this.matrixWorld), this.isCamera || this.isLight ? un.lookAt(Ni, ns, this.up) : un.lookAt(ns, Ni, this.up), this.quaternion.setFromRotationMatrix(un), s && (un.extractRotation(s.matrixWorld), li.setFromRotationMatrix(un), this.quaternion.premultiply(li.invert()));
  }
  add(t) {
    if (arguments.length > 1) {
      for (let e = 0; e < arguments.length; e++)
        this.add(arguments[e]);
      return this;
    }
    return t === this ? (console.error("THREE.Object3D.add: object can't be added as a child of itself.", t), this) : (t && t.isObject3D ? (t.removeFromParent(), t.parent = this, this.children.push(t), t.dispatchEvent(no), ci.child = t, this.dispatchEvent(ci), ci.child = null) : console.error("THREE.Object3D.add: object not an instance of THREE.Object3D.", t), this);
  }
  remove(t) {
    if (arguments.length > 1) {
      for (let n = 0; n < arguments.length; n++)
        this.remove(arguments[n]);
      return this;
    }
    const e = this.children.indexOf(t);
    return e !== -1 && (t.parent = null, this.children.splice(e, 1), t.dispatchEvent(dh), Zs.child = t, this.dispatchEvent(Zs), Zs.child = null), this;
  }
  removeFromParent() {
    const t = this.parent;
    return t !== null && t.remove(this), this;
  }
  clear() {
    return this.remove(...this.children);
  }
  attach(t) {
    return this.updateWorldMatrix(!0, !1), un.copy(this.matrixWorld).invert(), t.parent !== null && (t.parent.updateWorldMatrix(!0, !1), un.multiply(t.parent.matrixWorld)), t.applyMatrix4(un), t.removeFromParent(), t.parent = this, this.children.push(t), t.updateWorldMatrix(!1, !0), t.dispatchEvent(no), ci.child = t, this.dispatchEvent(ci), ci.child = null, this;
  }
  getObjectById(t) {
    return this.getObjectByProperty("id", t);
  }
  getObjectByName(t) {
    return this.getObjectByProperty("name", t);
  }
  getObjectByProperty(t, e) {
    if (this[t] === e) return this;
    for (let n = 0, s = this.children.length; n < s; n++) {
      const a = this.children[n].getObjectByProperty(t, e);
      if (a !== void 0)
        return a;
    }
  }
  getObjectsByProperty(t, e, n = []) {
    this[t] === e && n.push(this);
    const s = this.children;
    for (let r = 0, a = s.length; r < a; r++)
      s[r].getObjectsByProperty(t, e, n);
    return n;
  }
  getWorldPosition(t) {
    return this.updateWorldMatrix(!0, !1), t.setFromMatrixPosition(this.matrixWorld);
  }
  getWorldQuaternion(t) {
    return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(Ni, t, hh), t;
  }
  getWorldScale(t) {
    return this.updateWorldMatrix(!0, !1), this.matrixWorld.decompose(Ni, uh, t), t;
  }
  getWorldDirection(t) {
    this.updateWorldMatrix(!0, !1);
    const e = this.matrixWorld.elements;
    return t.set(e[8], e[9], e[10]).normalize();
  }
  raycast() {
  }
  traverse(t) {
    t(this);
    const e = this.children;
    for (let n = 0, s = e.length; n < s; n++)
      e[n].traverse(t);
  }
  traverseVisible(t) {
    if (this.visible === !1) return;
    t(this);
    const e = this.children;
    for (let n = 0, s = e.length; n < s; n++)
      e[n].traverseVisible(t);
  }
  traverseAncestors(t) {
    const e = this.parent;
    e !== null && (t(e), e.traverseAncestors(t));
  }
  updateMatrix() {
    this.matrix.compose(this.position, this.quaternion, this.scale), this.matrixWorldNeedsUpdate = !0;
  }
  updateMatrixWorld(t) {
    this.matrixAutoUpdate && this.updateMatrix(), (this.matrixWorldNeedsUpdate || t) && (this.matrixWorldAutoUpdate === !0 && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)), this.matrixWorldNeedsUpdate = !1, t = !0);
    const e = this.children;
    for (let n = 0, s = e.length; n < s; n++)
      e[n].updateMatrixWorld(t);
  }
  updateWorldMatrix(t, e) {
    const n = this.parent;
    if (t === !0 && n !== null && n.updateWorldMatrix(!0, !1), this.matrixAutoUpdate && this.updateMatrix(), this.matrixWorldAutoUpdate === !0 && (this.parent === null ? this.matrixWorld.copy(this.matrix) : this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)), e === !0) {
      const s = this.children;
      for (let r = 0, a = s.length; r < a; r++)
        s[r].updateWorldMatrix(!1, !0);
    }
  }
  toJSON(t) {
    const e = t === void 0 || typeof t == "string", n = {};
    e && (t = {
      geometries: {},
      materials: {},
      textures: {},
      images: {},
      shapes: {},
      skeletons: {},
      animations: {},
      nodes: {}
    }, n.metadata = {
      version: 4.6,
      type: "Object",
      generator: "Object3D.toJSON"
    });
    const s = {};
    s.uuid = this.uuid, s.type = this.type, this.name !== "" && (s.name = this.name), this.castShadow === !0 && (s.castShadow = !0), this.receiveShadow === !0 && (s.receiveShadow = !0), this.visible === !1 && (s.visible = !1), this.frustumCulled === !1 && (s.frustumCulled = !1), this.renderOrder !== 0 && (s.renderOrder = this.renderOrder), Object.keys(this.userData).length > 0 && (s.userData = this.userData), s.layers = this.layers.mask, s.matrix = this.matrix.toArray(), s.up = this.up.toArray(), this.matrixAutoUpdate === !1 && (s.matrixAutoUpdate = !1), this.isInstancedMesh && (s.type = "InstancedMesh", s.count = this.count, s.instanceMatrix = this.instanceMatrix.toJSON(), this.instanceColor !== null && (s.instanceColor = this.instanceColor.toJSON())), this.isBatchedMesh && (s.type = "BatchedMesh", s.perObjectFrustumCulled = this.perObjectFrustumCulled, s.sortObjects = this.sortObjects, s.drawRanges = this._drawRanges, s.reservedRanges = this._reservedRanges, s.visibility = this._visibility, s.active = this._active, s.bounds = this._bounds.map((o) => ({
      boxInitialized: o.boxInitialized,
      boxMin: o.box.min.toArray(),
      boxMax: o.box.max.toArray(),
      sphereInitialized: o.sphereInitialized,
      sphereRadius: o.sphere.radius,
      sphereCenter: o.sphere.center.toArray()
    })), s.maxInstanceCount = this._maxInstanceCount, s.maxVertexCount = this._maxVertexCount, s.maxIndexCount = this._maxIndexCount, s.geometryInitialized = this._geometryInitialized, s.geometryCount = this._geometryCount, s.matricesTexture = this._matricesTexture.toJSON(t), this._colorsTexture !== null && (s.colorsTexture = this._colorsTexture.toJSON(t)), this.boundingSphere !== null && (s.boundingSphere = {
      center: s.boundingSphere.center.toArray(),
      radius: s.boundingSphere.radius
    }), this.boundingBox !== null && (s.boundingBox = {
      min: s.boundingBox.min.toArray(),
      max: s.boundingBox.max.toArray()
    }));
    function r(o, l) {
      return o[l.uuid] === void 0 && (o[l.uuid] = l.toJSON(t)), l.uuid;
    }
    if (this.isScene)
      this.background && (this.background.isColor ? s.background = this.background.toJSON() : this.background.isTexture && (s.background = this.background.toJSON(t).uuid)), this.environment && this.environment.isTexture && this.environment.isRenderTargetTexture !== !0 && (s.environment = this.environment.toJSON(t).uuid);
    else if (this.isMesh || this.isLine || this.isPoints) {
      s.geometry = r(t.geometries, this.geometry);
      const o = this.geometry.parameters;
      if (o !== void 0 && o.shapes !== void 0) {
        const l = o.shapes;
        if (Array.isArray(l))
          for (let c = 0, u = l.length; c < u; c++) {
            const d = l[c];
            r(t.shapes, d);
          }
        else
          r(t.shapes, l);
      }
    }
    if (this.isSkinnedMesh && (s.bindMode = this.bindMode, s.bindMatrix = this.bindMatrix.toArray(), this.skeleton !== void 0 && (r(t.skeletons, this.skeleton), s.skeleton = this.skeleton.uuid)), this.material !== void 0)
      if (Array.isArray(this.material)) {
        const o = [];
        for (let l = 0, c = this.material.length; l < c; l++)
          o.push(r(t.materials, this.material[l]));
        s.material = o;
      } else
        s.material = r(t.materials, this.material);
    if (this.children.length > 0) {
      s.children = [];
      for (let o = 0; o < this.children.length; o++)
        s.children.push(this.children[o].toJSON(t).object);
    }
    if (this.animations.length > 0) {
      s.animations = [];
      for (let o = 0; o < this.animations.length; o++) {
        const l = this.animations[o];
        s.animations.push(r(t.animations, l));
      }
    }
    if (e) {
      const o = a(t.geometries), l = a(t.materials), c = a(t.textures), u = a(t.images), d = a(t.shapes), f = a(t.skeletons), m = a(t.animations), g = a(t.nodes);
      o.length > 0 && (n.geometries = o), l.length > 0 && (n.materials = l), c.length > 0 && (n.textures = c), u.length > 0 && (n.images = u), d.length > 0 && (n.shapes = d), f.length > 0 && (n.skeletons = f), m.length > 0 && (n.animations = m), g.length > 0 && (n.nodes = g);
    }
    return n.object = s, n;
    function a(o) {
      const l = [];
      for (const c in o) {
        const u = o[c];
        delete u.metadata, l.push(u);
      }
      return l;
    }
  }
  clone(t) {
    return new this.constructor().copy(this, t);
  }
  copy(t, e = !0) {
    if (this.name = t.name, this.up.copy(t.up), this.position.copy(t.position), this.rotation.order = t.rotation.order, this.quaternion.copy(t.quaternion), this.scale.copy(t.scale), this.matrix.copy(t.matrix), this.matrixWorld.copy(t.matrixWorld), this.matrixAutoUpdate = t.matrixAutoUpdate, this.matrixWorldAutoUpdate = t.matrixWorldAutoUpdate, this.matrixWorldNeedsUpdate = t.matrixWorldNeedsUpdate, this.layers.mask = t.layers.mask, this.visible = t.visible, this.castShadow = t.castShadow, this.receiveShadow = t.receiveShadow, this.frustumCulled = t.frustumCulled, this.renderOrder = t.renderOrder, this.animations = t.animations.slice(), this.userData = JSON.parse(JSON.stringify(t.userData)), e === !0)
      for (let n = 0; n < t.children.length; n++) {
        const s = t.children[n];
        this.add(s.clone());
      }
    return this;
  }
}
Re.DEFAULT_UP = /* @__PURE__ */ new P(0, 1, 0);
Re.DEFAULT_MATRIX_AUTO_UPDATE = !0;
Re.DEFAULT_MATRIX_WORLD_AUTO_UPDATE = !0;
const Ze = /* @__PURE__ */ new P(), dn = /* @__PURE__ */ new P(), Ks = /* @__PURE__ */ new P(), fn = /* @__PURE__ */ new P(), hi = /* @__PURE__ */ new P(), ui = /* @__PURE__ */ new P(), io = /* @__PURE__ */ new P(), Js = /* @__PURE__ */ new P(), Qs = /* @__PURE__ */ new P(), tr = /* @__PURE__ */ new P(), er = /* @__PURE__ */ new te(), nr = /* @__PURE__ */ new te(), ir = /* @__PURE__ */ new te();
class Je {
  constructor(t = new P(), e = new P(), n = new P()) {
    this.a = t, this.b = e, this.c = n;
  }
  static getNormal(t, e, n, s) {
    s.subVectors(n, e), Ze.subVectors(t, e), s.cross(Ze);
    const r = s.lengthSq();
    return r > 0 ? s.multiplyScalar(1 / Math.sqrt(r)) : s.set(0, 0, 0);
  }
  // static/instance method to calculate barycentric coordinates
  // based on: http://www.blackpawn.com/texts/pointinpoly/default.html
  static getBarycoord(t, e, n, s, r) {
    Ze.subVectors(s, e), dn.subVectors(n, e), Ks.subVectors(t, e);
    const a = Ze.dot(Ze), o = Ze.dot(dn), l = Ze.dot(Ks), c = dn.dot(dn), u = dn.dot(Ks), d = a * c - o * o;
    if (d === 0)
      return r.set(0, 0, 0), null;
    const f = 1 / d, m = (c * l - o * u) * f, g = (a * u - o * l) * f;
    return r.set(1 - m - g, g, m);
  }
  static containsPoint(t, e, n, s) {
    return this.getBarycoord(t, e, n, s, fn) === null ? !1 : fn.x >= 0 && fn.y >= 0 && fn.x + fn.y <= 1;
  }
  static getInterpolation(t, e, n, s, r, a, o, l) {
    return this.getBarycoord(t, e, n, s, fn) === null ? (l.x = 0, l.y = 0, "z" in l && (l.z = 0), "w" in l && (l.w = 0), null) : (l.setScalar(0), l.addScaledVector(r, fn.x), l.addScaledVector(a, fn.y), l.addScaledVector(o, fn.z), l);
  }
  static getInterpolatedAttribute(t, e, n, s, r, a) {
    return er.setScalar(0), nr.setScalar(0), ir.setScalar(0), er.fromBufferAttribute(t, e), nr.fromBufferAttribute(t, n), ir.fromBufferAttribute(t, s), a.setScalar(0), a.addScaledVector(er, r.x), a.addScaledVector(nr, r.y), a.addScaledVector(ir, r.z), a;
  }
  static isFrontFacing(t, e, n, s) {
    return Ze.subVectors(n, e), dn.subVectors(t, e), Ze.cross(dn).dot(s) < 0;
  }
  set(t, e, n) {
    return this.a.copy(t), this.b.copy(e), this.c.copy(n), this;
  }
  setFromPointsAndIndices(t, e, n, s) {
    return this.a.copy(t[e]), this.b.copy(t[n]), this.c.copy(t[s]), this;
  }
  setFromAttributeAndIndices(t, e, n, s) {
    return this.a.fromBufferAttribute(t, e), this.b.fromBufferAttribute(t, n), this.c.fromBufferAttribute(t, s), this;
  }
  clone() {
    return new this.constructor().copy(this);
  }
  copy(t) {
    return this.a.copy(t.a), this.b.copy(t.b), this.c.copy(t.c), this;
  }
  getArea() {
    return Ze.subVectors(this.c, this.b), dn.subVectors(this.a, this.b), Ze.cross(dn).length() * 0.5;
  }
  getMidpoint(t) {
    return t.addVectors(this.a, this.b).add(this.c).multiplyScalar(1 / 3);
  }
  getNormal(t) {
    return Je.getNormal(this.a, this.b, this.c, t);
  }
  getPlane(t) {
    return t.setFromCoplanarPoints(this.a, this.b, this.c);
  }
  getBarycoord(t, e) {
    return Je.getBarycoord(t, this.a, this.b, this.c, e);
  }
  getInterpolation(t, e, n, s, r) {
    return Je.getInterpolation(t, this.a, this.b, this.c, e, n, s, r);
  }
  containsPoint(t) {
    return Je.containsPoint(t, this.a, this.b, this.c);
  }
  isFrontFacing(t) {
    return Je.isFrontFacing(this.a, this.b, this.c, t);
  }
  intersectsBox(t) {
    return t.intersectsTriangle(this);
  }
  closestPointToPoint(t, e) {
    const n = this.a, s = this.b, r = this.c;
    let a, o;
    hi.subVectors(s, n), ui.subVectors(r, n), Js.subVectors(t, n);
    const l = hi.dot(Js), c = ui.dot(Js);
    if (l <= 0 && c <= 0)
      return e.copy(n);
    Qs.subVectors(t, s);
    const u = hi.dot(Qs), d = ui.dot(Qs);
    if (u >= 0 && d <= u)
      return e.copy(s);
    const f = l * d - u * c;
    if (f <= 0 && l >= 0 && u <= 0)
      return a = l / (l - u), e.copy(n).addScaledVector(hi, a);
    tr.subVectors(t, r);
    const m = hi.dot(tr), g = ui.dot(tr);
    if (g >= 0 && m <= g)
      return e.copy(r);
    const v = m * c - l * g;
    if (v <= 0 && c >= 0 && g <= 0)
      return o = c / (c - g), e.copy(n).addScaledVector(ui, o);
    const p = u * g - m * d;
    if (p <= 0 && d - u >= 0 && m - g >= 0)
      return io.subVectors(r, s), o = (d - u) / (d - u + (m - g)), e.copy(s).addScaledVector(io, o);
    const h = 1 / (p + v + f);
    return a = v * h, o = f * h, e.copy(n).addScaledVector(hi, a).addScaledVector(ui, o);
  }
  equals(t) {
    return t.a.equals(this.a) && t.b.equals(this.b) && t.c.equals(this.c);
  }
}
const Ml = {
  aliceblue: 15792383,
  antiquewhite: 16444375,
  aqua: 65535,
  aquamarine: 8388564,
  azure: 15794175,
  beige: 16119260,
  bisque: 16770244,
  black: 0,
  blanchedalmond: 16772045,
  blue: 255,
  blueviolet: 9055202,
  brown: 10824234,
  burlywood: 14596231,
  cadetblue: 6266528,
  chartreuse: 8388352,
  chocolate: 13789470,
  coral: 16744272,
  cornflowerblue: 6591981,
  cornsilk: 16775388,
  crimson: 14423100,
  cyan: 65535,
  darkblue: 139,
  darkcyan: 35723,
  darkgoldenrod: 12092939,
  darkgray: 11119017,
  darkgreen: 25600,
  darkgrey: 11119017,
  darkkhaki: 12433259,
  darkmagenta: 9109643,
  darkolivegreen: 5597999,
  darkorange: 16747520,
  darkorchid: 10040012,
  darkred: 9109504,
  darksalmon: 15308410,
  darkseagreen: 9419919,
  darkslateblue: 4734347,
  darkslategray: 3100495,
  darkslategrey: 3100495,
  darkturquoise: 52945,
  darkviolet: 9699539,
  deeppink: 16716947,
  deepskyblue: 49151,
  dimgray: 6908265,
  dimgrey: 6908265,
  dodgerblue: 2003199,
  firebrick: 11674146,
  floralwhite: 16775920,
  forestgreen: 2263842,
  fuchsia: 16711935,
  gainsboro: 14474460,
  ghostwhite: 16316671,
  gold: 16766720,
  goldenrod: 14329120,
  gray: 8421504,
  green: 32768,
  greenyellow: 11403055,
  grey: 8421504,
  honeydew: 15794160,
  hotpink: 16738740,
  indianred: 13458524,
  indigo: 4915330,
  ivory: 16777200,
  khaki: 15787660,
  lavender: 15132410,
  lavenderblush: 16773365,
  lawngreen: 8190976,
  lemonchiffon: 16775885,
  lightblue: 11393254,
  lightcoral: 15761536,
  lightcyan: 14745599,
  lightgoldenrodyellow: 16448210,
  lightgray: 13882323,
  lightgreen: 9498256,
  lightgrey: 13882323,
  lightpink: 16758465,
  lightsalmon: 16752762,
  lightseagreen: 2142890,
  lightskyblue: 8900346,
  lightslategray: 7833753,
  lightslategrey: 7833753,
  lightsteelblue: 11584734,
  lightyellow: 16777184,
  lime: 65280,
  limegreen: 3329330,
  linen: 16445670,
  magenta: 16711935,
  maroon: 8388608,
  mediumaquamarine: 6737322,
  mediumblue: 205,
  mediumorchid: 12211667,
  mediumpurple: 9662683,
  mediumseagreen: 3978097,
  mediumslateblue: 8087790,
  mediumspringgreen: 64154,
  mediumturquoise: 4772300,
  mediumvioletred: 13047173,
  midnightblue: 1644912,
  mintcream: 16121850,
  mistyrose: 16770273,
  moccasin: 16770229,
  navajowhite: 16768685,
  navy: 128,
  oldlace: 16643558,
  olive: 8421376,
  olivedrab: 7048739,
  orange: 16753920,
  orangered: 16729344,
  orchid: 14315734,
  palegoldenrod: 15657130,
  palegreen: 10025880,
  paleturquoise: 11529966,
  palevioletred: 14381203,
  papayawhip: 16773077,
  peachpuff: 16767673,
  peru: 13468991,
  pink: 16761035,
  plum: 14524637,
  powderblue: 11591910,
  purple: 8388736,
  rebeccapurple: 6697881,
  red: 16711680,
  rosybrown: 12357519,
  royalblue: 4286945,
  saddlebrown: 9127187,
  salmon: 16416882,
  sandybrown: 16032864,
  seagreen: 3050327,
  seashell: 16774638,
  sienna: 10506797,
  silver: 12632256,
  skyblue: 8900331,
  slateblue: 6970061,
  slategray: 7372944,
  slategrey: 7372944,
  snow: 16775930,
  springgreen: 65407,
  steelblue: 4620980,
  tan: 13808780,
  teal: 32896,
  thistle: 14204888,
  tomato: 16737095,
  turquoise: 4251856,
  violet: 15631086,
  wheat: 16113331,
  white: 16777215,
  whitesmoke: 16119285,
  yellow: 16776960,
  yellowgreen: 10145074
}, Rn = { h: 0, s: 0, l: 0 }, is = { h: 0, s: 0, l: 0 };
function sr(i, t, e) {
  return e < 0 && (e += 1), e > 1 && (e -= 1), e < 1 / 6 ? i + (t - i) * 6 * e : e < 1 / 2 ? t : e < 2 / 3 ? i + (t - i) * 6 * (2 / 3 - e) : i;
}
class Vt {
  constructor(t, e, n) {
    return this.isColor = !0, this.r = 1, this.g = 1, this.b = 1, this.set(t, e, n);
  }
  set(t, e, n) {
    if (e === void 0 && n === void 0) {
      const s = t;
      s && s.isColor ? this.copy(s) : typeof s == "number" ? this.setHex(s) : typeof s == "string" && this.setStyle(s);
    } else
      this.setRGB(t, e, n);
    return this;
  }
  setScalar(t) {
    return this.r = t, this.g = t, this.b = t, this;
  }
  setHex(t, e = Be) {
    return t = Math.floor(t), this.r = (t >> 16 & 255) / 255, this.g = (t >> 8 & 255) / 255, this.b = (t & 255) / 255, Wt.toWorkingColorSpace(this, e), this;
  }
  setRGB(t, e, n, s = Wt.workingColorSpace) {
    return this.r = t, this.g = e, this.b = n, Wt.toWorkingColorSpace(this, s), this;
  }
  setHSL(t, e, n, s = Wt.workingColorSpace) {
    if (t = Ma(t, 1), e = Ut(e, 0, 1), n = Ut(n, 0, 1), e === 0)
      this.r = this.g = this.b = n;
    else {
      const r = n <= 0.5 ? n * (1 + e) : n + e - n * e, a = 2 * n - r;
      this.r = sr(a, r, t + 1 / 3), this.g = sr(a, r, t), this.b = sr(a, r, t - 1 / 3);
    }
    return Wt.toWorkingColorSpace(this, s), this;
  }
  setStyle(t, e = Be) {
    function n(r) {
      r !== void 0 && parseFloat(r) < 1 && console.warn("THREE.Color: Alpha component of " + t + " will be ignored.");
    }
    let s;
    if (s = /^(\w+)\(([^\)]*)\)/.exec(t)) {
      let r;
      const a = s[1], o = s[2];
      switch (a) {
        case "rgb":
        case "rgba":
          if (r = /^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))
            return n(r[4]), this.setRGB(
              Math.min(255, parseInt(r[1], 10)) / 255,
              Math.min(255, parseInt(r[2], 10)) / 255,
              Math.min(255, parseInt(r[3], 10)) / 255,
              e
            );
          if (r = /^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))
            return n(r[4]), this.setRGB(
              Math.min(100, parseInt(r[1], 10)) / 100,
              Math.min(100, parseInt(r[2], 10)) / 100,
              Math.min(100, parseInt(r[3], 10)) / 100,
              e
            );
          break;
        case "hsl":
        case "hsla":
          if (r = /^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))
            return n(r[4]), this.setHSL(
              parseFloat(r[1]) / 360,
              parseFloat(r[2]) / 100,
              parseFloat(r[3]) / 100,
              e
            );
          break;
        default:
          console.warn("THREE.Color: Unknown color model " + t);
      }
    } else if (s = /^\#([A-Fa-f\d]+)$/.exec(t)) {
      const r = s[1], a = r.length;
      if (a === 3)
        return this.setRGB(
          parseInt(r.charAt(0), 16) / 15,
          parseInt(r.charAt(1), 16) / 15,
          parseInt(r.charAt(2), 16) / 15,
          e
        );
      if (a === 6)
        return this.setHex(parseInt(r, 16), e);
      console.warn("THREE.Color: Invalid hex color " + t);
    } else if (t && t.length > 0)
      return this.setColorName(t, e);
    return this;
  }
  setColorName(t, e = Be) {
    const n = Ml[t.toLowerCase()];
    return n !== void 0 ? this.setHex(n, e) : console.warn("THREE.Color: Unknown color " + t), this;
  }
  clone() {
    return new this.constructor(this.r, this.g, this.b);
  }
  copy(t) {
    return this.r = t.r, this.g = t.g, this.b = t.b, this;
  }
  copySRGBToLinear(t) {
    return this.r = Mn(t.r), this.g = Mn(t.g), this.b = Mn(t.b), this;
  }
  copyLinearToSRGB(t) {
    return this.r = yi(t.r), this.g = yi(t.g), this.b = yi(t.b), this;
  }
  convertSRGBToLinear() {
    return this.copySRGBToLinear(this), this;
  }
  convertLinearToSRGB() {
    return this.copyLinearToSRGB(this), this;
  }
  getHex(t = Be) {
    return Wt.fromWorkingColorSpace(Ee.copy(this), t), Math.round(Ut(Ee.r * 255, 0, 255)) * 65536 + Math.round(Ut(Ee.g * 255, 0, 255)) * 256 + Math.round(Ut(Ee.b * 255, 0, 255));
  }
  getHexString(t = Be) {
    return ("000000" + this.getHex(t).toString(16)).slice(-6);
  }
  getHSL(t, e = Wt.workingColorSpace) {
    Wt.fromWorkingColorSpace(Ee.copy(this), e);
    const n = Ee.r, s = Ee.g, r = Ee.b, a = Math.max(n, s, r), o = Math.min(n, s, r);
    let l, c;
    const u = (o + a) / 2;
    if (o === a)
      l = 0, c = 0;
    else {
      const d = a - o;
      switch (c = u <= 0.5 ? d / (a + o) : d / (2 - a - o), a) {
        case n:
          l = (s - r) / d + (s < r ? 6 : 0);
          break;
        case s:
          l = (r - n) / d + 2;
          break;
        case r:
          l = (n - s) / d + 4;
          break;
      }
      l /= 6;
    }
    return t.h = l, t.s = c, t.l = u, t;
  }
  getRGB(t, e = Wt.workingColorSpace) {
    return Wt.fromWorkingColorSpace(Ee.copy(this), e), t.r = Ee.r, t.g = Ee.g, t.b = Ee.b, t;
  }
  getStyle(t = Be) {
    Wt.fromWorkingColorSpace(Ee.copy(this), t);
    const e = Ee.r, n = Ee.g, s = Ee.b;
    return t !== Be ? `color(${t} ${e.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})` : `rgb(${Math.round(e * 255)},${Math.round(n * 255)},${Math.round(s * 255)})`;
  }
  offsetHSL(t, e, n) {
    return this.getHSL(Rn), this.setHSL(Rn.h + t, Rn.s + e, Rn.l + n);
  }
  add(t) {
    return this.r += t.r, this.g += t.g, this.b += t.b, this;
  }
  addColors(t, e) {
    return this.r = t.r + e.r, this.g = t.g + e.g, this.b = t.b + e.b, this;
  }
  addScalar(t) {
    return this.r += t, this.g += t, this.b += t, this;
  }
  sub(t) {
    return this.r = Math.max(0, this.r - t.r), this.g = Math.max(0, this.g - t.g), this.b = Math.max(0, this.b - t.b), this;
  }
  multiply(t) {
    return this.r *= t.r, this.g *= t.g, this.b *= t.b, this;
  }
  multiplyScalar(t) {
    return this.r *= t, this.g *= t, this.b *= t, this;
  }
  lerp(t, e) {
    return this.r += (t.r - this.r) * e, this.g += (t.g - this.g) * e, this.b += (t.b - this.b) * e, this;
  }
  lerpColors(t, e, n) {
    return this.r = t.r + (e.r - t.r) * n, this.g = t.g + (e.g - t.g) * n, this.b = t.b + (e.b - t.b) * n, this;
  }
  lerpHSL(t, e) {
    this.getHSL(Rn), t.getHSL(is);
    const n = ki(Rn.h, is.h, e), s = ki(Rn.s, is.s, e), r = ki(Rn.l, is.l, e);
    return this.setHSL(n, s, r), this;
  }
  setFromVector3(t) {
    return this.r = t.x, this.g = t.y, this.b = t.z, this;
  }
  applyMatrix3(t) {
    const e = this.r, n = this.g, s = this.b, r = t.elements;
    return this.r = r[0] * e + r[3] * n + r[6] * s, this.g = r[1] * e + r[4] * n + r[7] * s, this.b = r[2] * e + r[5] * n + r[8] * s, this;
  }
  equals(t) {
    return t.r === this.r && t.g === this.g && t.b === this.b;
  }
  fromArray(t, e = 0) {
    return this.r = t[e], this.g = t[e + 1], this.b = t[e + 2], this;
  }
  toArray(t = [], e = 0) {
    return t[e] = this.r, t[e + 1] = this.g, t[e + 2] = this.b, t;
  }
  fromBufferAttribute(t, e) {
    return this.r = t.getX(e), this.g = t.getY(e), this.b = t.getZ(e), this;
  }
  toJSON() {
    return this.getHex();
  }
  *[Symbol.iterator]() {
    yield this.r, yield this.g, yield this.b;
  }
}
const Ee = /* @__PURE__ */ new Vt();
Vt.NAMES = Ml;
let fh = 0;
class ti extends Qn {
  constructor() {
    super(), this.isMaterial = !0, Object.defineProperty(this, "id", { value: fh++ }), this.uuid = xn(), this.name = "", this.type = "Material", this.blending = Mi, this.side = In, this.vertexColors = !1, this.opacity = 1, this.transparent = !1, this.alphaHash = !1, this.blendSrc = br, this.blendDst = Ar, this.blendEquation = Wn, this.blendSrcAlpha = null, this.blendDstAlpha = null, this.blendEquationAlpha = null, this.blendColor = new Vt(0, 0, 0), this.blendAlpha = 0, this.depthFunc = Ei, this.depthTest = !0, this.depthWrite = !0, this.stencilWriteMask = 255, this.stencilFunc = Ga, this.stencilRef = 0, this.stencilFuncMask = 255, this.stencilFail = ni, this.stencilZFail = ni, this.stencilZPass = ni, this.stencilWrite = !1, this.clippingPlanes = null, this.clipIntersection = !1, this.clipShadows = !1, this.shadowSide = null, this.colorWrite = !0, this.precision = null, this.polygonOffset = !1, this.polygonOffsetFactor = 0, this.polygonOffsetUnits = 0, this.dithering = !1, this.alphaToCoverage = !1, this.premultipliedAlpha = !1, this.forceSinglePass = !1, this.visible = !0, this.toneMapped = !0, this.userData = {}, this.version = 0, this._alphaTest = 0;
  }
  get alphaTest() {
    return this._alphaTest;
  }
  set alphaTest(t) {
    this._alphaTest > 0 != t > 0 && this.version++, this._alphaTest = t;
  }
  // onBeforeRender and onBeforeCompile only supported in WebGLRenderer
  onBeforeRender() {
  }
  onBeforeCompile() {
  }
  customProgramCacheKey() {
    return this.onBeforeCompile.toString();
  }
  setValues(t) {
    if (t !== void 0)
      for (const e in t) {
        const n = t[e];
        if (n === void 0) {
          console.warn(`THREE.Material: parameter '${e}' has value of undefined.`);
          continue;
        }
        const s = this[e];
        if (s === void 0) {
          console.warn(`THREE.Material: '${e}' is not a property of THREE.${this.type}.`);
          continue;
        }
        s && s.isColor ? s.set(n) : s && s.isVector3 && n && n.isVector3 ? s.copy(n) : this[e] = n;
      }
  }
  toJSON(t) {
    const e = t === void 0 || typeof t == "string";
    e && (t = {
      textures: {},
      images: {}
    });
    const n = {
      metadata: {
        version: 4.6,
        type: "Material",
        generator: "Material.toJSON"
      }
    };
    n.uuid = this.uuid, n.type = this.type, this.name !== "" && (n.name = this.name), this.color && this.color.isColor && (n.color = this.color.getHex()), this.roughness !== void 0 && (n.roughness = this.roughness), this.metalness !== void 0 && (n.metalness = this.metalness), this.sheen !== void 0 && (n.sheen = this.sheen), this.sheenColor && this.sheenColor.isColor && (n.sheenColor = this.sheenColor.getHex()), this.sheenRoughness !== void 0 && (n.sheenRoughness = this.sheenRoughness), this.emissive && this.emissive.isColor && (n.emissive = this.emissive.getHex()), this.emissiveIntensity !== void 0 && this.emissiveIntensity !== 1 && (n.emissiveIntensity = this.emissiveIntensity), this.specular && this.specular.isColor && (n.specular = this.specular.getHex()), this.specularIntensity !== void 0 && (n.specularIntensity = this.specularIntensity), this.specularColor && this.specularColor.isColor && (n.specularColor = this.specularColor.getHex()), this.shininess !== void 0 && (n.shininess = this.shininess), this.clearcoat !== void 0 && (n.clearcoat = this.clearcoat), this.clearcoatRoughness !== void 0 && (n.clearcoatRoughness = this.clearcoatRoughness), this.clearcoatMap && this.clearcoatMap.isTexture && (n.clearcoatMap = this.clearcoatMap.toJSON(t).uuid), this.clearcoatRoughnessMap && this.clearcoatRoughnessMap.isTexture && (n.clearcoatRoughnessMap = this.clearcoatRoughnessMap.toJSON(t).uuid), this.clearcoatNormalMap && this.clearcoatNormalMap.isTexture && (n.clearcoatNormalMap = this.clearcoatNormalMap.toJSON(t).uuid, n.clearcoatNormalScale = this.clearcoatNormalScale.toArray()), this.dispersion !== void 0 && (n.dispersion = this.dispersion), this.iridescence !== void 0 && (n.iridescence = this.iridescence), this.iridescenceIOR !== void 0 && (n.iridescenceIOR = this.iridescenceIOR), this.iridescenceThicknessRange !== void 0 && (n.iridescenceThicknessRange = this.iridescenceThicknessRange), this.iridescenceMap && this.iridescenceMap.isTexture && (n.iridescenceMap = this.iridescenceMap.toJSON(t).uuid), this.iridescenceThicknessMap && this.iridescenceThicknessMap.isTexture && (n.iridescenceThicknessMap = this.iridescenceThicknessMap.toJSON(t).uuid), this.anisotropy !== void 0 && (n.anisotropy = this.anisotropy), this.anisotropyRotation !== void 0 && (n.anisotropyRotation = this.anisotropyRotation), this.anisotropyMap && this.anisotropyMap.isTexture && (n.anisotropyMap = this.anisotropyMap.toJSON(t).uuid), this.map && this.map.isTexture && (n.map = this.map.toJSON(t).uuid), this.matcap && this.matcap.isTexture && (n.matcap = this.matcap.toJSON(t).uuid), this.alphaMap && this.alphaMap.isTexture && (n.alphaMap = this.alphaMap.toJSON(t).uuid), this.lightMap && this.lightMap.isTexture && (n.lightMap = this.lightMap.toJSON(t).uuid, n.lightMapIntensity = this.lightMapIntensity), this.aoMap && this.aoMap.isTexture && (n.aoMap = this.aoMap.toJSON(t).uuid, n.aoMapIntensity = this.aoMapIntensity), this.bumpMap && this.bumpMap.isTexture && (n.bumpMap = this.bumpMap.toJSON(t).uuid, n.bumpScale = this.bumpScale), this.normalMap && this.normalMap.isTexture && (n.normalMap = this.normalMap.toJSON(t).uuid, n.normalMapType = this.normalMapType, n.normalScale = this.normalScale.toArray()), this.displacementMap && this.displacementMap.isTexture && (n.displacementMap = this.displacementMap.toJSON(t).uuid, n.displacementScale = this.displacementScale, n.displacementBias = this.displacementBias), this.roughnessMap && this.roughnessMap.isTexture && (n.roughnessMap = this.roughnessMap.toJSON(t).uuid), this.metalnessMap && this.metalnessMap.isTexture && (n.metalnessMap = this.metalnessMap.toJSON(t).uuid), this.emissiveMap && this.emissiveMap.isTexture && (n.emissiveMap = this.emissiveMap.toJSON(t).uuid), this.specularMap && this.specularMap.isTexture && (n.specularMap = this.specularMap.toJSON(t).uuid), this.specularIntensityMap && this.specularIntensityMap.isTexture && (n.specularIntensityMap = this.specularIntensityMap.toJSON(t).uuid), this.specularColorMap && this.specularColorMap.isTexture && (n.specularColorMap = this.specularColorMap.toJSON(t).uuid), this.envMap && this.envMap.isTexture && (n.envMap = this.envMap.toJSON(t).uuid, this.combine !== void 0 && (n.combine = this.combine)), this.envMapRotation !== void 0 && (n.envMapRotation = this.envMapRotation.toArray()), this.envMapIntensity !== void 0 && (n.envMapIntensity = this.envMapIntensity), this.reflectivity !== void 0 && (n.reflectivity = this.reflectivity), this.refractionRatio !== void 0 && (n.refractionRatio = this.refractionRatio), this.gradientMap && this.gradientMap.isTexture && (n.gradientMap = this.gradientMap.toJSON(t).uuid), this.transmission !== void 0 && (n.transmission = this.transmission), this.transmissionMap && this.transmissionMap.isTexture && (n.transmissionMap = this.transmissionMap.toJSON(t).uuid), this.thickness !== void 0 && (n.thickness = this.thickness), this.thicknessMap && this.thicknessMap.isTexture && (n.thicknessMap = this.thicknessMap.toJSON(t).uuid), this.attenuationDistance !== void 0 && this.attenuationDistance !== 1 / 0 && (n.attenuationDistance = this.attenuationDistance), this.attenuationColor !== void 0 && (n.attenuationColor = this.attenuationColor.getHex()), this.size !== void 0 && (n.size = this.size), this.shadowSide !== null && (n.shadowSide = this.shadowSide), this.sizeAttenuation !== void 0 && (n.sizeAttenuation = this.sizeAttenuation), this.blending !== Mi && (n.blending = this.blending), this.side !== In && (n.side = this.side), this.vertexColors === !0 && (n.vertexColors = !0), this.opacity < 1 && (n.opacity = this.opacity), this.transparent === !0 && (n.transparent = !0), this.blendSrc !== br && (n.blendSrc = this.blendSrc), this.blendDst !== Ar && (n.blendDst = this.blendDst), this.blendEquation !== Wn && (n.blendEquation = this.blendEquation), this.blendSrcAlpha !== null && (n.blendSrcAlpha = this.blendSrcAlpha), this.blendDstAlpha !== null && (n.blendDstAlpha = this.blendDstAlpha), this.blendEquationAlpha !== null && (n.blendEquationAlpha = this.blendEquationAlpha), this.blendColor && this.blendColor.isColor && (n.blendColor = this.blendColor.getHex()), this.blendAlpha !== 0 && (n.blendAlpha = this.blendAlpha), this.depthFunc !== Ei && (n.depthFunc = this.depthFunc), this.depthTest === !1 && (n.depthTest = this.depthTest), this.depthWrite === !1 && (n.depthWrite = this.depthWrite), this.colorWrite === !1 && (n.colorWrite = this.colorWrite), this.stencilWriteMask !== 255 && (n.stencilWriteMask = this.stencilWriteMask), this.stencilFunc !== Ga && (n.stencilFunc = this.stencilFunc), this.stencilRef !== 0 && (n.stencilRef = this.stencilRef), this.stencilFuncMask !== 255 && (n.stencilFuncMask = this.stencilFuncMask), this.stencilFail !== ni && (n.stencilFail = this.stencilFail), this.stencilZFail !== ni && (n.stencilZFail = this.stencilZFail), this.stencilZPass !== ni && (n.stencilZPass = this.stencilZPass), this.stencilWrite === !0 && (n.stencilWrite = this.stencilWrite), this.rotation !== void 0 && this.rotation !== 0 && (n.rotation = this.rotation), this.polygonOffset === !0 && (n.polygonOffset = !0), this.polygonOffsetFactor !== 0 && (n.polygonOffsetFactor = this.polygonOffsetFactor), this.polygonOffsetUnits !== 0 && (n.polygonOffsetUnits = this.polygonOffsetUnits), this.linewidth !== void 0 && this.linewidth !== 1 && (n.linewidth = this.linewidth), this.dashSize !== void 0 && (n.dashSize = this.dashSize), this.gapSize !== void 0 && (n.gapSize = this.gapSize), this.scale !== void 0 && (n.scale = this.scale), this.dithering === !0 && (n.dithering = !0), this.alphaTest > 0 && (n.alphaTest = this.alphaTest), this.alphaHash === !0 && (n.alphaHash = !0), this.alphaToCoverage === !0 && (n.alphaToCoverage = !0), this.premultipliedAlpha === !0 && (n.premultipliedAlpha = !0), this.forceSinglePass === !0 && (n.forceSinglePass = !0), this.wireframe === !0 && (n.wireframe = !0), this.wireframeLinewidth > 1 && (n.wireframeLinewidth = this.wireframeLinewidth), this.wireframeLinecap !== "round" && (n.wireframeLinecap = this.wireframeLinecap), this.wireframeLinejoin !== "round" && (n.wireframeLinejoin = this.wireframeLinejoin), this.flatShading === !0 && (n.flatShading = !0), this.visible === !1 && (n.visible = !1), this.toneMapped === !1 && (n.toneMapped = !1), this.fog === !1 && (n.fog = !1), Object.keys(this.userData).length > 0 && (n.userData = this.userData);
    function s(r) {
      const a = [];
      for (const o in r) {
        const l = r[o];
        delete l.metadata, a.push(l);
      }
      return a;
    }
    if (e) {
      const r = s(t.textures), a = s(t.images);
      r.length > 0 && (n.textures = r), a.length > 0 && (n.images = a);
    }
    return n;
  }
  clone() {
    return new this.constructor().copy(this);
  }
  copy(t) {
    this.name = t.name, this.blending = t.blending, this.side = t.side, this.vertexColors = t.vertexColors, this.opacity = t.opacity, this.transparent = t.transparent, this.blendSrc = t.blendSrc, this.blendDst = t.blendDst, this.blendEquation = t.blendEquation, this.blendSrcAlpha = t.blendSrcAlpha, this.blendDstAlpha = t.blendDstAlpha, this.blendEquationAlpha = t.blendEquationAlpha, this.blendColor.copy(t.blendColor), this.blendAlpha = t.blendAlpha, this.depthFunc = t.depthFunc, this.depthTest = t.depthTest, this.depthWrite = t.depthWrite, this.stencilWriteMask = t.stencilWriteMask, this.stencilFunc = t.stencilFunc, this.stencilRef = t.stencilRef, this.stencilFuncMask = t.stencilFuncMask, this.stencilFail = t.stencilFail, this.stencilZFail = t.stencilZFail, this.stencilZPass = t.stencilZPass, this.stencilWrite = t.stencilWrite;
    const e = t.clippingPlanes;
    let n = null;
    if (e !== null) {
      const s = e.length;
      n = new Array(s);
      for (let r = 0; r !== s; ++r)
        n[r] = e[r].clone();
    }
    return this.clippingPlanes = n, this.clipIntersection = t.clipIntersection, this.clipShadows = t.clipShadows, this.shadowSide = t.shadowSide, this.colorWrite = t.colorWrite, this.precision = t.precision, this.polygonOffset = t.polygonOffset, this.polygonOffsetFactor = t.polygonOffsetFactor, this.polygonOffsetUnits = t.polygonOffsetUnits, this.dithering = t.dithering, this.alphaTest = t.alphaTest, this.alphaHash = t.alphaHash, this.alphaToCoverage = t.alphaToCoverage, this.premultipliedAlpha = t.premultipliedAlpha, this.forceSinglePass = t.forceSinglePass, this.visible = t.visible, this.toneMapped = t.toneMapped, this.userData = JSON.parse(JSON.stringify(t.userData)), this;
  }
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
  set needsUpdate(t) {
    t === !0 && this.version++;
  }
  onBuild() {
    console.warn("Material: onBuild() has been removed.");
  }
}
class Fs extends ti {
  constructor(t) {
    super(), this.isMeshBasicMaterial = !0, this.type = "MeshBasicMaterial", this.color = new Vt(16777215), this.map = null, this.lightMap = null, this.lightMapIntensity = 1, this.aoMap = null, this.aoMapIntensity = 1, this.specularMap = null, this.alphaMap = null, this.envMap = null, this.envMapRotation = new yn(), this.combine = el, this.reflectivity = 1, this.refractionRatio = 0.98, this.wireframe = !1, this.wireframeLinewidth = 1, this.wireframeLinecap = "round", this.wireframeLinejoin = "round", this.fog = !0, this.setValues(t);
  }
  copy(t) {
    return super.copy(t), this.color.copy(t.color), this.map = t.map, this.lightMap = t.lightMap, this.lightMapIntensity = t.lightMapIntensity, this.aoMap = t.aoMap, this.aoMapIntensity = t.aoMapIntensity, this.specularMap = t.specularMap, this.alphaMap = t.alphaMap, this.envMap = t.envMap, this.envMapRotation.copy(t.envMapRotation), this.combine = t.combine, this.reflectivity = t.reflectivity, this.refractionRatio = t.refractionRatio, this.wireframe = t.wireframe, this.wireframeLinewidth = t.wireframeLinewidth, this.wireframeLinecap = t.wireframeLinecap, this.wireframeLinejoin = t.wireframeLinejoin, this.fog = t.fog, this;
  }
}
const ue = /* @__PURE__ */ new P(), ss = /* @__PURE__ */ new bt();
class en {
  constructor(t, e, n = !1) {
    if (Array.isArray(t))
      throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");
    this.isBufferAttribute = !0, this.name = "", this.array = t, this.itemSize = e, this.count = t !== void 0 ? t.length / e : 0, this.normalized = n, this.usage = ca, this.updateRanges = [], this.gpuType = gn, this.version = 0;
  }
  onUploadCallback() {
  }
  set needsUpdate(t) {
    t === !0 && this.version++;
  }
  setUsage(t) {
    return this.usage = t, this;
  }
  addUpdateRange(t, e) {
    this.updateRanges.push({ start: t, count: e });
  }
  clearUpdateRanges() {
    this.updateRanges.length = 0;
  }
  copy(t) {
    return this.name = t.name, this.array = new t.array.constructor(t.array), this.itemSize = t.itemSize, this.count = t.count, this.normalized = t.normalized, this.usage = t.usage, this.gpuType = t.gpuType, this;
  }
  copyAt(t, e, n) {
    t *= this.itemSize, n *= e.itemSize;
    for (let s = 0, r = this.itemSize; s < r; s++)
      this.array[t + s] = e.array[n + s];
    return this;
  }
  copyArray(t) {
    return this.array.set(t), this;
  }
  applyMatrix3(t) {
    if (this.itemSize === 2)
      for (let e = 0, n = this.count; e < n; e++)
        ss.fromBufferAttribute(this, e), ss.applyMatrix3(t), this.setXY(e, ss.x, ss.y);
    else if (this.itemSize === 3)
      for (let e = 0, n = this.count; e < n; e++)
        ue.fromBufferAttribute(this, e), ue.applyMatrix3(t), this.setXYZ(e, ue.x, ue.y, ue.z);
    return this;
  }
  applyMatrix4(t) {
    for (let e = 0, n = this.count; e < n; e++)
      ue.fromBufferAttribute(this, e), ue.applyMatrix4(t), this.setXYZ(e, ue.x, ue.y, ue.z);
    return this;
  }
  applyNormalMatrix(t) {
    for (let e = 0, n = this.count; e < n; e++)
      ue.fromBufferAttribute(this, e), ue.applyNormalMatrix(t), this.setXYZ(e, ue.x, ue.y, ue.z);
    return this;
  }
  transformDirection(t) {
    for (let e = 0, n = this.count; e < n; e++)
      ue.fromBufferAttribute(this, e), ue.transformDirection(t), this.setXYZ(e, ue.x, ue.y, ue.z);
    return this;
  }
  set(t, e = 0) {
    return this.array.set(t, e), this;
  }
  getComponent(t, e) {
    let n = this.array[t * this.itemSize + e];
    return this.normalized && (n = Ke(n, this.array)), n;
  }
  setComponent(t, e, n) {
    return this.normalized && (n = qt(n, this.array)), this.array[t * this.itemSize + e] = n, this;
  }
  getX(t) {
    let e = this.array[t * this.itemSize];
    return this.normalized && (e = Ke(e, this.array)), e;
  }
  setX(t, e) {
    return this.normalized && (e = qt(e, this.array)), this.array[t * this.itemSize] = e, this;
  }
  getY(t) {
    let e = this.array[t * this.itemSize + 1];
    return this.normalized && (e = Ke(e, this.array)), e;
  }
  setY(t, e) {
    return this.normalized && (e = qt(e, this.array)), this.array[t * this.itemSize + 1] = e, this;
  }
  getZ(t) {
    let e = this.array[t * this.itemSize + 2];
    return this.normalized && (e = Ke(e, this.array)), e;
  }
  setZ(t, e) {
    return this.normalized && (e = qt(e, this.array)), this.array[t * this.itemSize + 2] = e, this;
  }
  getW(t) {
    let e = this.array[t * this.itemSize + 3];
    return this.normalized && (e = Ke(e, this.array)), e;
  }
  setW(t, e) {
    return this.normalized && (e = qt(e, this.array)), this.array[t * this.itemSize + 3] = e, this;
  }
  setXY(t, e, n) {
    return t *= this.itemSize, this.normalized && (e = qt(e, this.array), n = qt(n, this.array)), this.array[t + 0] = e, this.array[t + 1] = n, this;
  }
  setXYZ(t, e, n, s) {
    return t *= this.itemSize, this.normalized && (e = qt(e, this.array), n = qt(n, this.array), s = qt(s, this.array)), this.array[t + 0] = e, this.array[t + 1] = n, this.array[t + 2] = s, this;
  }
  setXYZW(t, e, n, s, r) {
    return t *= this.itemSize, this.normalized && (e = qt(e, this.array), n = qt(n, this.array), s = qt(s, this.array), r = qt(r, this.array)), this.array[t + 0] = e, this.array[t + 1] = n, this.array[t + 2] = s, this.array[t + 3] = r, this;
  }
  onUpload(t) {
    return this.onUploadCallback = t, this;
  }
  clone() {
    return new this.constructor(this.array, this.itemSize).copy(this);
  }
  toJSON() {
    const t = {
      itemSize: this.itemSize,
      type: this.array.constructor.name,
      array: Array.from(this.array),
      normalized: this.normalized
    };
    return this.name !== "" && (t.name = this.name), this.usage !== ca && (t.usage = this.usage), t;
  }
}
class Sl extends en {
  constructor(t, e, n) {
    super(new Uint16Array(t), e, n);
  }
}
class yl extends en {
  constructor(t, e, n) {
    super(new Uint32Array(t), e, n);
  }
}
class se extends en {
  constructor(t, e, n) {
    super(new Float32Array(t), e, n);
  }
}
let ph = 0;
const We = /* @__PURE__ */ new ne(), rr = /* @__PURE__ */ new Re(), di = /* @__PURE__ */ new P(), Oe = /* @__PURE__ */ new He(), Fi = /* @__PURE__ */ new He(), _e = /* @__PURE__ */ new P();
class Ce extends Qn {
  constructor() {
    super(), this.isBufferGeometry = !0, Object.defineProperty(this, "id", { value: ph++ }), this.uuid = xn(), this.name = "", this.type = "BufferGeometry", this.index = null, this.indirect = null, this.attributes = {}, this.morphAttributes = {}, this.morphTargetsRelative = !1, this.groups = [], this.boundingBox = null, this.boundingSphere = null, this.drawRange = { start: 0, count: 1 / 0 }, this.userData = {};
  }
  getIndex() {
    return this.index;
  }
  setIndex(t) {
    return Array.isArray(t) ? this.index = new (_l(t) ? yl : Sl)(t, 1) : this.index = t, this;
  }
  setIndirect(t) {
    return this.indirect = t, this;
  }
  getIndirect() {
    return this.indirect;
  }
  getAttribute(t) {
    return this.attributes[t];
  }
  setAttribute(t, e) {
    return this.attributes[t] = e, this;
  }
  deleteAttribute(t) {
    return delete this.attributes[t], this;
  }
  hasAttribute(t) {
    return this.attributes[t] !== void 0;
  }
  addGroup(t, e, n = 0) {
    this.groups.push({
      start: t,
      count: e,
      materialIndex: n
    });
  }
  clearGroups() {
    this.groups = [];
  }
  setDrawRange(t, e) {
    this.drawRange.start = t, this.drawRange.count = e;
  }
  applyMatrix4(t) {
    const e = this.attributes.position;
    e !== void 0 && (e.applyMatrix4(t), e.needsUpdate = !0);
    const n = this.attributes.normal;
    if (n !== void 0) {
      const r = new Pt().getNormalMatrix(t);
      n.applyNormalMatrix(r), n.needsUpdate = !0;
    }
    const s = this.attributes.tangent;
    return s !== void 0 && (s.transformDirection(t), s.needsUpdate = !0), this.boundingBox !== null && this.computeBoundingBox(), this.boundingSphere !== null && this.computeBoundingSphere(), this;
  }
  applyQuaternion(t) {
    return We.makeRotationFromQuaternion(t), this.applyMatrix4(We), this;
  }
  rotateX(t) {
    return We.makeRotationX(t), this.applyMatrix4(We), this;
  }
  rotateY(t) {
    return We.makeRotationY(t), this.applyMatrix4(We), this;
  }
  rotateZ(t) {
    return We.makeRotationZ(t), this.applyMatrix4(We), this;
  }
  translate(t, e, n) {
    return We.makeTranslation(t, e, n), this.applyMatrix4(We), this;
  }
  scale(t, e, n) {
    return We.makeScale(t, e, n), this.applyMatrix4(We), this;
  }
  lookAt(t) {
    return rr.lookAt(t), rr.updateMatrix(), this.applyMatrix4(rr.matrix), this;
  }
  center() {
    return this.computeBoundingBox(), this.boundingBox.getCenter(di).negate(), this.translate(di.x, di.y, di.z), this;
  }
  setFromPoints(t) {
    const e = this.getAttribute("position");
    if (e === void 0) {
      const n = [];
      for (let s = 0, r = t.length; s < r; s++) {
        const a = t[s];
        n.push(a.x, a.y, a.z || 0);
      }
      this.setAttribute("position", new se(n, 3));
    } else {
      const n = Math.min(t.length, e.count);
      for (let s = 0; s < n; s++) {
        const r = t[s];
        e.setXYZ(s, r.x, r.y, r.z || 0);
      }
      t.length > e.count && console.warn("THREE.BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."), e.needsUpdate = !0;
    }
    return this;
  }
  computeBoundingBox() {
    this.boundingBox === null && (this.boundingBox = new He());
    const t = this.attributes.position, e = this.morphAttributes.position;
    if (t && t.isGLBufferAttribute) {
      console.error("THREE.BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.", this), this.boundingBox.set(
        new P(-1 / 0, -1 / 0, -1 / 0),
        new P(1 / 0, 1 / 0, 1 / 0)
      );
      return;
    }
    if (t !== void 0) {
      if (this.boundingBox.setFromBufferAttribute(t), e)
        for (let n = 0, s = e.length; n < s; n++) {
          const r = e[n];
          Oe.setFromBufferAttribute(r), this.morphTargetsRelative ? (_e.addVectors(this.boundingBox.min, Oe.min), this.boundingBox.expandByPoint(_e), _e.addVectors(this.boundingBox.max, Oe.max), this.boundingBox.expandByPoint(_e)) : (this.boundingBox.expandByPoint(Oe.min), this.boundingBox.expandByPoint(Oe.max));
        }
    } else
      this.boundingBox.makeEmpty();
    (isNaN(this.boundingBox.min.x) || isNaN(this.boundingBox.min.y) || isNaN(this.boundingBox.min.z)) && console.error('THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.', this);
  }
  computeBoundingSphere() {
    this.boundingSphere === null && (this.boundingSphere = new Pi());
    const t = this.attributes.position, e = this.morphAttributes.position;
    if (t && t.isGLBufferAttribute) {
      console.error("THREE.BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.", this), this.boundingSphere.set(new P(), 1 / 0);
      return;
    }
    if (t) {
      const n = this.boundingSphere.center;
      if (Oe.setFromBufferAttribute(t), e)
        for (let r = 0, a = e.length; r < a; r++) {
          const o = e[r];
          Fi.setFromBufferAttribute(o), this.morphTargetsRelative ? (_e.addVectors(Oe.min, Fi.min), Oe.expandByPoint(_e), _e.addVectors(Oe.max, Fi.max), Oe.expandByPoint(_e)) : (Oe.expandByPoint(Fi.min), Oe.expandByPoint(Fi.max));
        }
      Oe.getCenter(n);
      let s = 0;
      for (let r = 0, a = t.count; r < a; r++)
        _e.fromBufferAttribute(t, r), s = Math.max(s, n.distanceToSquared(_e));
      if (e)
        for (let r = 0, a = e.length; r < a; r++) {
          const o = e[r], l = this.morphTargetsRelative;
          for (let c = 0, u = o.count; c < u; c++)
            _e.fromBufferAttribute(o, c), l && (di.fromBufferAttribute(t, c), _e.add(di)), s = Math.max(s, n.distanceToSquared(_e));
        }
      this.boundingSphere.radius = Math.sqrt(s), isNaN(this.boundingSphere.radius) && console.error('THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.', this);
    }
  }
  computeTangents() {
    const t = this.index, e = this.attributes;
    if (t === null || e.position === void 0 || e.normal === void 0 || e.uv === void 0) {
      console.error("THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");
      return;
    }
    const n = e.position, s = e.normal, r = e.uv;
    this.hasAttribute("tangent") === !1 && this.setAttribute("tangent", new en(new Float32Array(4 * n.count), 4));
    const a = this.getAttribute("tangent"), o = [], l = [];
    for (let I = 0; I < n.count; I++)
      o[I] = new P(), l[I] = new P();
    const c = new P(), u = new P(), d = new P(), f = new bt(), m = new bt(), g = new bt(), v = new P(), p = new P();
    function h(I, y, M) {
      c.fromBufferAttribute(n, I), u.fromBufferAttribute(n, y), d.fromBufferAttribute(n, M), f.fromBufferAttribute(r, I), m.fromBufferAttribute(r, y), g.fromBufferAttribute(r, M), u.sub(c), d.sub(c), m.sub(f), g.sub(f);
      const C = 1 / (m.x * g.y - g.x * m.y);
      isFinite(C) && (v.copy(u).multiplyScalar(g.y).addScaledVector(d, -m.y).multiplyScalar(C), p.copy(d).multiplyScalar(m.x).addScaledVector(u, -g.x).multiplyScalar(C), o[I].add(v), o[y].add(v), o[M].add(v), l[I].add(p), l[y].add(p), l[M].add(p));
    }
    let E = this.groups;
    E.length === 0 && (E = [{
      start: 0,
      count: t.count
    }]);
    for (let I = 0, y = E.length; I < y; ++I) {
      const M = E[I], C = M.start, H = M.count;
      for (let z = C, G = C + H; z < G; z += 3)
        h(
          t.getX(z + 0),
          t.getX(z + 1),
          t.getX(z + 2)
        );
    }
    const b = new P(), S = new P(), L = new P(), T = new P();
    function R(I) {
      L.fromBufferAttribute(s, I), T.copy(L);
      const y = o[I];
      b.copy(y), b.sub(L.multiplyScalar(L.dot(y))).normalize(), S.crossVectors(T, y);
      const C = S.dot(l[I]) < 0 ? -1 : 1;
      a.setXYZW(I, b.x, b.y, b.z, C);
    }
    for (let I = 0, y = E.length; I < y; ++I) {
      const M = E[I], C = M.start, H = M.count;
      for (let z = C, G = C + H; z < G; z += 3)
        R(t.getX(z + 0)), R(t.getX(z + 1)), R(t.getX(z + 2));
    }
  }
  computeVertexNormals() {
    const t = this.index, e = this.getAttribute("position");
    if (e !== void 0) {
      let n = this.getAttribute("normal");
      if (n === void 0)
        n = new en(new Float32Array(e.count * 3), 3), this.setAttribute("normal", n);
      else
        for (let f = 0, m = n.count; f < m; f++)
          n.setXYZ(f, 0, 0, 0);
      const s = new P(), r = new P(), a = new P(), o = new P(), l = new P(), c = new P(), u = new P(), d = new P();
      if (t)
        for (let f = 0, m = t.count; f < m; f += 3) {
          const g = t.getX(f + 0), v = t.getX(f + 1), p = t.getX(f + 2);
          s.fromBufferAttribute(e, g), r.fromBufferAttribute(e, v), a.fromBufferAttribute(e, p), u.subVectors(a, r), d.subVectors(s, r), u.cross(d), o.fromBufferAttribute(n, g), l.fromBufferAttribute(n, v), c.fromBufferAttribute(n, p), o.add(u), l.add(u), c.add(u), n.setXYZ(g, o.x, o.y, o.z), n.setXYZ(v, l.x, l.y, l.z), n.setXYZ(p, c.x, c.y, c.z);
        }
      else
        for (let f = 0, m = e.count; f < m; f += 3)
          s.fromBufferAttribute(e, f + 0), r.fromBufferAttribute(e, f + 1), a.fromBufferAttribute(e, f + 2), u.subVectors(a, r), d.subVectors(s, r), u.cross(d), n.setXYZ(f + 0, u.x, u.y, u.z), n.setXYZ(f + 1, u.x, u.y, u.z), n.setXYZ(f + 2, u.x, u.y, u.z);
      this.normalizeNormals(), n.needsUpdate = !0;
    }
  }
  normalizeNormals() {
    const t = this.attributes.normal;
    for (let e = 0, n = t.count; e < n; e++)
      _e.fromBufferAttribute(t, e), _e.normalize(), t.setXYZ(e, _e.x, _e.y, _e.z);
  }
  toNonIndexed() {
    function t(o, l) {
      const c = o.array, u = o.itemSize, d = o.normalized, f = new c.constructor(l.length * u);
      let m = 0, g = 0;
      for (let v = 0, p = l.length; v < p; v++) {
        o.isInterleavedBufferAttribute ? m = l[v] * o.data.stride + o.offset : m = l[v] * u;
        for (let h = 0; h < u; h++)
          f[g++] = c[m++];
      }
      return new en(f, u, d);
    }
    if (this.index === null)
      return console.warn("THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."), this;
    const e = new Ce(), n = this.index.array, s = this.attributes;
    for (const o in s) {
      const l = s[o], c = t(l, n);
      e.setAttribute(o, c);
    }
    const r = this.morphAttributes;
    for (const o in r) {
      const l = [], c = r[o];
      for (let u = 0, d = c.length; u < d; u++) {
        const f = c[u], m = t(f, n);
        l.push(m);
      }
      e.morphAttributes[o] = l;
    }
    e.morphTargetsRelative = this.morphTargetsRelative;
    const a = this.groups;
    for (let o = 0, l = a.length; o < l; o++) {
      const c = a[o];
      e.addGroup(c.start, c.count, c.materialIndex);
    }
    return e;
  }
  toJSON() {
    const t = {
      metadata: {
        version: 4.6,
        type: "BufferGeometry",
        generator: "BufferGeometry.toJSON"
      }
    };
    if (t.uuid = this.uuid, t.type = this.type, this.name !== "" && (t.name = this.name), Object.keys(this.userData).length > 0 && (t.userData = this.userData), this.parameters !== void 0) {
      const l = this.parameters;
      for (const c in l)
        l[c] !== void 0 && (t[c] = l[c]);
      return t;
    }
    t.data = { attributes: {} };
    const e = this.index;
    e !== null && (t.data.index = {
      type: e.array.constructor.name,
      array: Array.prototype.slice.call(e.array)
    });
    const n = this.attributes;
    for (const l in n) {
      const c = n[l];
      t.data.attributes[l] = c.toJSON(t.data);
    }
    const s = {};
    let r = !1;
    for (const l in this.morphAttributes) {
      const c = this.morphAttributes[l], u = [];
      for (let d = 0, f = c.length; d < f; d++) {
        const m = c[d];
        u.push(m.toJSON(t.data));
      }
      u.length > 0 && (s[l] = u, r = !0);
    }
    r && (t.data.morphAttributes = s, t.data.morphTargetsRelative = this.morphTargetsRelative);
    const a = this.groups;
    a.length > 0 && (t.data.groups = JSON.parse(JSON.stringify(a)));
    const o = this.boundingSphere;
    return o !== null && (t.data.boundingSphere = {
      center: o.center.toArray(),
      radius: o.radius
    }), t;
  }
  clone() {
    return new this.constructor().copy(this);
  }
  copy(t) {
    this.index = null, this.attributes = {}, this.morphAttributes = {}, this.groups = [], this.boundingBox = null, this.boundingSphere = null;
    const e = {};
    this.name = t.name;
    const n = t.index;
    n !== null && this.setIndex(n.clone(e));
    const s = t.attributes;
    for (const c in s) {
      const u = s[c];
      this.setAttribute(c, u.clone(e));
    }
    const r = t.morphAttributes;
    for (const c in r) {
      const u = [], d = r[c];
      for (let f = 0, m = d.length; f < m; f++)
        u.push(d[f].clone(e));
      this.morphAttributes[c] = u;
    }
    this.morphTargetsRelative = t.morphTargetsRelative;
    const a = t.groups;
    for (let c = 0, u = a.length; c < u; c++) {
      const d = a[c];
      this.addGroup(d.start, d.count, d.materialIndex);
    }
    const o = t.boundingBox;
    o !== null && (this.boundingBox = o.clone());
    const l = t.boundingSphere;
    return l !== null && (this.boundingSphere = l.clone()), this.drawRange.start = t.drawRange.start, this.drawRange.count = t.drawRange.count, this.userData = t.userData, this;
  }
  dispose() {
    this.dispatchEvent({ type: "dispose" });
  }
}
const so = /* @__PURE__ */ new ne(), zn = /* @__PURE__ */ new Sa(), rs = /* @__PURE__ */ new Pi(), ro = /* @__PURE__ */ new P(), as = /* @__PURE__ */ new P(), os = /* @__PURE__ */ new P(), ls = /* @__PURE__ */ new P(), ar = /* @__PURE__ */ new P(), cs = /* @__PURE__ */ new P(), ao = /* @__PURE__ */ new P(), hs = /* @__PURE__ */ new P();
class be extends Re {
  constructor(t = new Ce(), e = new Fs()) {
    super(), this.isMesh = !0, this.type = "Mesh", this.geometry = t, this.material = e, this.updateMorphTargets();
  }
  copy(t, e) {
    return super.copy(t, e), t.morphTargetInfluences !== void 0 && (this.morphTargetInfluences = t.morphTargetInfluences.slice()), t.morphTargetDictionary !== void 0 && (this.morphTargetDictionary = Object.assign({}, t.morphTargetDictionary)), this.material = Array.isArray(t.material) ? t.material.slice() : t.material, this.geometry = t.geometry, this;
  }
  updateMorphTargets() {
    const e = this.geometry.morphAttributes, n = Object.keys(e);
    if (n.length > 0) {
      const s = e[n[0]];
      if (s !== void 0) {
        this.morphTargetInfluences = [], this.morphTargetDictionary = {};
        for (let r = 0, a = s.length; r < a; r++) {
          const o = s[r].name || String(r);
          this.morphTargetInfluences.push(0), this.morphTargetDictionary[o] = r;
        }
      }
    }
  }
  getVertexPosition(t, e) {
    const n = this.geometry, s = n.attributes.position, r = n.morphAttributes.position, a = n.morphTargetsRelative;
    e.fromBufferAttribute(s, t);
    const o = this.morphTargetInfluences;
    if (r && o) {
      cs.set(0, 0, 0);
      for (let l = 0, c = r.length; l < c; l++) {
        const u = o[l], d = r[l];
        u !== 0 && (ar.fromBufferAttribute(d, t), a ? cs.addScaledVector(ar, u) : cs.addScaledVector(ar.sub(e), u));
      }
      e.add(cs);
    }
    return e;
  }
  raycast(t, e) {
    const n = this.geometry, s = this.material, r = this.matrixWorld;
    s !== void 0 && (n.boundingSphere === null && n.computeBoundingSphere(), rs.copy(n.boundingSphere), rs.applyMatrix4(r), zn.copy(t.ray).recast(t.near), !(rs.containsPoint(zn.origin) === !1 && (zn.intersectSphere(rs, ro) === null || zn.origin.distanceToSquared(ro) > (t.far - t.near) ** 2)) && (so.copy(r).invert(), zn.copy(t.ray).applyMatrix4(so), !(n.boundingBox !== null && zn.intersectsBox(n.boundingBox) === !1) && this._computeIntersections(t, e, zn)));
  }
  _computeIntersections(t, e, n) {
    let s;
    const r = this.geometry, a = this.material, o = r.index, l = r.attributes.position, c = r.attributes.uv, u = r.attributes.uv1, d = r.attributes.normal, f = r.groups, m = r.drawRange;
    if (o !== null)
      if (Array.isArray(a))
        for (let g = 0, v = f.length; g < v; g++) {
          const p = f[g], h = a[p.materialIndex], E = Math.max(p.start, m.start), b = Math.min(o.count, Math.min(p.start + p.count, m.start + m.count));
          for (let S = E, L = b; S < L; S += 3) {
            const T = o.getX(S), R = o.getX(S + 1), I = o.getX(S + 2);
            s = us(this, h, t, n, c, u, d, T, R, I), s && (s.faceIndex = Math.floor(S / 3), s.face.materialIndex = p.materialIndex, e.push(s));
          }
        }
      else {
        const g = Math.max(0, m.start), v = Math.min(o.count, m.start + m.count);
        for (let p = g, h = v; p < h; p += 3) {
          const E = o.getX(p), b = o.getX(p + 1), S = o.getX(p + 2);
          s = us(this, a, t, n, c, u, d, E, b, S), s && (s.faceIndex = Math.floor(p / 3), e.push(s));
        }
      }
    else if (l !== void 0)
      if (Array.isArray(a))
        for (let g = 0, v = f.length; g < v; g++) {
          const p = f[g], h = a[p.materialIndex], E = Math.max(p.start, m.start), b = Math.min(l.count, Math.min(p.start + p.count, m.start + m.count));
          for (let S = E, L = b; S < L; S += 3) {
            const T = S, R = S + 1, I = S + 2;
            s = us(this, h, t, n, c, u, d, T, R, I), s && (s.faceIndex = Math.floor(S / 3), s.face.materialIndex = p.materialIndex, e.push(s));
          }
        }
      else {
        const g = Math.max(0, m.start), v = Math.min(l.count, m.start + m.count);
        for (let p = g, h = v; p < h; p += 3) {
          const E = p, b = p + 1, S = p + 2;
          s = us(this, a, t, n, c, u, d, E, b, S), s && (s.faceIndex = Math.floor(p / 3), e.push(s));
        }
      }
  }
}
function mh(i, t, e, n, s, r, a, o) {
  let l;
  if (t.side === Ue ? l = n.intersectTriangle(a, r, s, !0, o) : l = n.intersectTriangle(s, r, a, t.side === In, o), l === null) return null;
  hs.copy(o), hs.applyMatrix4(i.matrixWorld);
  const c = e.ray.origin.distanceTo(hs);
  return c < e.near || c > e.far ? null : {
    distance: c,
    point: hs.clone(),
    object: i
  };
}
function us(i, t, e, n, s, r, a, o, l, c) {
  i.getVertexPosition(o, as), i.getVertexPosition(l, os), i.getVertexPosition(c, ls);
  const u = mh(i, t, e, n, as, os, ls, ao);
  if (u) {
    const d = new P();
    Je.getBarycoord(ao, as, os, ls, d), s && (u.uv = Je.getInterpolatedAttribute(s, o, l, c, d, new bt())), r && (u.uv1 = Je.getInterpolatedAttribute(r, o, l, c, d, new bt())), a && (u.normal = Je.getInterpolatedAttribute(a, o, l, c, d, new P()), u.normal.dot(n.direction) > 0 && u.normal.multiplyScalar(-1));
    const f = {
      a: o,
      b: l,
      c,
      normal: new P(),
      materialIndex: 0
    };
    Je.getNormal(as, os, ls, f.normal), u.face = f, u.barycoord = d;
  }
  return u;
}
class Yi extends Ce {
  constructor(t = 1, e = 1, n = 1, s = 1, r = 1, a = 1) {
    super(), this.type = "BoxGeometry", this.parameters = {
      width: t,
      height: e,
      depth: n,
      widthSegments: s,
      heightSegments: r,
      depthSegments: a
    };
    const o = this;
    s = Math.floor(s), r = Math.floor(r), a = Math.floor(a);
    const l = [], c = [], u = [], d = [];
    let f = 0, m = 0;
    g("z", "y", "x", -1, -1, n, e, t, a, r, 0), g("z", "y", "x", 1, -1, n, e, -t, a, r, 1), g("x", "z", "y", 1, 1, t, n, e, s, a, 2), g("x", "z", "y", 1, -1, t, n, -e, s, a, 3), g("x", "y", "z", 1, -1, t, e, n, s, r, 4), g("x", "y", "z", -1, -1, t, e, -n, s, r, 5), this.setIndex(l), this.setAttribute("position", new se(c, 3)), this.setAttribute("normal", new se(u, 3)), this.setAttribute("uv", new se(d, 2));
    function g(v, p, h, E, b, S, L, T, R, I, y) {
      const M = S / R, C = L / I, H = S / 2, z = L / 2, G = T / 2, j = R + 1, W = I + 1;
      let Q = 0, V = 0;
      const st = new P();
      for (let ht = 0; ht < W; ht++) {
        const gt = ht * C - z;
        for (let It = 0; It < j; It++) {
          const Kt = It * M - H;
          st[v] = Kt * E, st[p] = gt * b, st[h] = G, c.push(st.x, st.y, st.z), st[v] = 0, st[p] = 0, st[h] = T > 0 ? 1 : -1, u.push(st.x, st.y, st.z), d.push(It / R), d.push(1 - ht / I), Q += 1;
        }
      }
      for (let ht = 0; ht < I; ht++)
        for (let gt = 0; gt < R; gt++) {
          const It = f + gt + j * ht, Kt = f + gt + j * (ht + 1), Y = f + (gt + 1) + j * (ht + 1), tt = f + (gt + 1) + j * ht;
          l.push(It, Kt, tt), l.push(Kt, Y, tt), V += 6;
        }
      o.addGroup(m, V, y), m += V, f += Q;
    }
  }
  copy(t) {
    return super.copy(t), this.parameters = Object.assign({}, t.parameters), this;
  }
  static fromJSON(t) {
    return new Yi(t.width, t.height, t.depth, t.widthSegments, t.heightSegments, t.depthSegments);
  }
}
function Ci(i) {
  const t = {};
  for (const e in i) {
    t[e] = {};
    for (const n in i[e]) {
      const s = i[e][n];
      s && (s.isColor || s.isMatrix3 || s.isMatrix4 || s.isVector2 || s.isVector3 || s.isVector4 || s.isTexture || s.isQuaternion) ? s.isRenderTargetTexture ? (console.warn("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."), t[e][n] = null) : t[e][n] = s.clone() : Array.isArray(s) ? t[e][n] = s.slice() : t[e][n] = s;
    }
  }
  return t;
}
function we(i) {
  const t = {};
  for (let e = 0; e < i.length; e++) {
    const n = Ci(i[e]);
    for (const s in n)
      t[s] = n[s];
  }
  return t;
}
function _h(i) {
  const t = [];
  for (let e = 0; e < i.length; e++)
    t.push(i[e].clone());
  return t;
}
function El(i) {
  const t = i.getRenderTarget();
  return t === null ? i.outputColorSpace : t.isXRRenderTarget === !0 ? t.texture.colorSpace : Wt.workingColorSpace;
}
const ya = { clone: Ci, merge: we };
var gh = `void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`, vh = `void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;
class En extends ti {
  constructor(t) {
    super(), this.isShaderMaterial = !0, this.type = "ShaderMaterial", this.defines = {}, this.uniforms = {}, this.uniformsGroups = [], this.vertexShader = gh, this.fragmentShader = vh, this.linewidth = 1, this.wireframe = !1, this.wireframeLinewidth = 1, this.fog = !1, this.lights = !1, this.clipping = !1, this.forceSinglePass = !0, this.extensions = {
      clipCullDistance: !1,
      // set to use vertex shader clipping
      multiDraw: !1
      // set to use vertex shader multi_draw / enable gl_DrawID
    }, this.defaultAttributeValues = {
      color: [1, 1, 1],
      uv: [0, 0],
      uv1: [0, 0]
    }, this.index0AttributeName = void 0, this.uniformsNeedUpdate = !1, this.glslVersion = null, t !== void 0 && this.setValues(t);
  }
  copy(t) {
    return super.copy(t), this.fragmentShader = t.fragmentShader, this.vertexShader = t.vertexShader, this.uniforms = Ci(t.uniforms), this.uniformsGroups = _h(t.uniformsGroups), this.defines = Object.assign({}, t.defines), this.wireframe = t.wireframe, this.wireframeLinewidth = t.wireframeLinewidth, this.fog = t.fog, this.lights = t.lights, this.clipping = t.clipping, this.extensions = Object.assign({}, t.extensions), this.glslVersion = t.glslVersion, this;
  }
  toJSON(t) {
    const e = super.toJSON(t);
    e.glslVersion = this.glslVersion, e.uniforms = {};
    for (const s in this.uniforms) {
      const a = this.uniforms[s].value;
      a && a.isTexture ? e.uniforms[s] = {
        type: "t",
        value: a.toJSON(t).uuid
      } : a && a.isColor ? e.uniforms[s] = {
        type: "c",
        value: a.getHex()
      } : a && a.isVector2 ? e.uniforms[s] = {
        type: "v2",
        value: a.toArray()
      } : a && a.isVector3 ? e.uniforms[s] = {
        type: "v3",
        value: a.toArray()
      } : a && a.isVector4 ? e.uniforms[s] = {
        type: "v4",
        value: a.toArray()
      } : a && a.isMatrix3 ? e.uniforms[s] = {
        type: "m3",
        value: a.toArray()
      } : a && a.isMatrix4 ? e.uniforms[s] = {
        type: "m4",
        value: a.toArray()
      } : e.uniforms[s] = {
        value: a
      };
    }
    Object.keys(this.defines).length > 0 && (e.defines = this.defines), e.vertexShader = this.vertexShader, e.fragmentShader = this.fragmentShader, e.lights = this.lights, e.clipping = this.clipping;
    const n = {};
    for (const s in this.extensions)
      this.extensions[s] === !0 && (n[s] = !0);
    return Object.keys(n).length > 0 && (e.extensions = n), e;
  }
}
class bl extends Re {
  constructor() {
    super(), this.isCamera = !0, this.type = "Camera", this.matrixWorldInverse = new ne(), this.projectionMatrix = new ne(), this.projectionMatrixInverse = new ne(), this.coordinateSystem = vn;
  }
  copy(t, e) {
    return super.copy(t, e), this.matrixWorldInverse.copy(t.matrixWorldInverse), this.projectionMatrix.copy(t.projectionMatrix), this.projectionMatrixInverse.copy(t.projectionMatrixInverse), this.coordinateSystem = t.coordinateSystem, this;
  }
  getWorldDirection(t) {
    return super.getWorldDirection(t).negate();
  }
  updateMatrixWorld(t) {
    super.updateMatrixWorld(t), this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }
  updateWorldMatrix(t, e) {
    super.updateWorldMatrix(t, e), this.matrixWorldInverse.copy(this.matrixWorld).invert();
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
const Cn = /* @__PURE__ */ new P(), oo = /* @__PURE__ */ new bt(), lo = /* @__PURE__ */ new bt();
class ze extends bl {
  constructor(t = 50, e = 1, n = 0.1, s = 2e3) {
    super(), this.isPerspectiveCamera = !0, this.type = "PerspectiveCamera", this.fov = t, this.zoom = 1, this.near = n, this.far = s, this.focus = 10, this.aspect = e, this.view = null, this.filmGauge = 35, this.filmOffset = 0, this.updateProjectionMatrix();
  }
  copy(t, e) {
    return super.copy(t, e), this.fov = t.fov, this.zoom = t.zoom, this.near = t.near, this.far = t.far, this.focus = t.focus, this.aspect = t.aspect, this.view = t.view === null ? null : Object.assign({}, t.view), this.filmGauge = t.filmGauge, this.filmOffset = t.filmOffset, this;
  }
  /**
   * Sets the FOV by focal length in respect to the current .filmGauge.
   *
   * The default film gauge is 35, so that the focal length can be specified for
   * a 35mm (full frame) camera.
   *
   * Values for focal length and film gauge must have the same unit.
   */
  setFocalLength(t) {
    const e = 0.5 * this.getFilmHeight() / t;
    this.fov = Gi * 2 * Math.atan(e), this.updateProjectionMatrix();
  }
  /**
   * Calculates the focal length from the current .fov and .filmGauge.
   */
  getFocalLength() {
    const t = Math.tan(Hi * 0.5 * this.fov);
    return 0.5 * this.getFilmHeight() / t;
  }
  getEffectiveFOV() {
    return Gi * 2 * Math.atan(
      Math.tan(Hi * 0.5 * this.fov) / this.zoom
    );
  }
  getFilmWidth() {
    return this.filmGauge * Math.min(this.aspect, 1);
  }
  getFilmHeight() {
    return this.filmGauge / Math.max(this.aspect, 1);
  }
  /**
   * Computes the 2D bounds of the camera's viewable rectangle at a given distance along the viewing direction.
   * Sets minTarget and maxTarget to the coordinates of the lower-left and upper-right corners of the view rectangle.
   */
  getViewBounds(t, e, n) {
    Cn.set(-1, -1, 0.5).applyMatrix4(this.projectionMatrixInverse), e.set(Cn.x, Cn.y).multiplyScalar(-t / Cn.z), Cn.set(1, 1, 0.5).applyMatrix4(this.projectionMatrixInverse), n.set(Cn.x, Cn.y).multiplyScalar(-t / Cn.z);
  }
  /**
   * Computes the width and height of the camera's viewable rectangle at a given distance along the viewing direction.
   * Copies the result into the target Vector2, where x is width and y is height.
   */
  getViewSize(t, e) {
    return this.getViewBounds(t, oo, lo), e.subVectors(lo, oo);
  }
  /**
   * Sets an offset in a larger frustum. This is useful for multi-window or
   * multi-monitor/multi-machine setups.
   *
   * For example, if you have 3x2 monitors and each monitor is 1920x1080 and
   * the monitors are in grid like this
   *
   *   +---+---+---+
   *   | A | B | C |
   *   +---+---+---+
   *   | D | E | F |
   *   +---+---+---+
   *
   * then for each monitor you would call it like this
   *
   *   const w = 1920;
   *   const h = 1080;
   *   const fullWidth = w * 3;
   *   const fullHeight = h * 2;
   *
   *   --A--
   *   camera.setViewOffset( fullWidth, fullHeight, w * 0, h * 0, w, h );
   *   --B--
   *   camera.setViewOffset( fullWidth, fullHeight, w * 1, h * 0, w, h );
   *   --C--
   *   camera.setViewOffset( fullWidth, fullHeight, w * 2, h * 0, w, h );
   *   --D--
   *   camera.setViewOffset( fullWidth, fullHeight, w * 0, h * 1, w, h );
   *   --E--
   *   camera.setViewOffset( fullWidth, fullHeight, w * 1, h * 1, w, h );
   *   --F--
   *   camera.setViewOffset( fullWidth, fullHeight, w * 2, h * 1, w, h );
   *
   *   Note there is no reason monitors have to be the same size or in a grid.
   */
  setViewOffset(t, e, n, s, r, a) {
    this.aspect = t / e, this.view === null && (this.view = {
      enabled: !0,
      fullWidth: 1,
      fullHeight: 1,
      offsetX: 0,
      offsetY: 0,
      width: 1,
      height: 1
    }), this.view.enabled = !0, this.view.fullWidth = t, this.view.fullHeight = e, this.view.offsetX = n, this.view.offsetY = s, this.view.width = r, this.view.height = a, this.updateProjectionMatrix();
  }
  clearViewOffset() {
    this.view !== null && (this.view.enabled = !1), this.updateProjectionMatrix();
  }
  updateProjectionMatrix() {
    const t = this.near;
    let e = t * Math.tan(Hi * 0.5 * this.fov) / this.zoom, n = 2 * e, s = this.aspect * n, r = -0.5 * s;
    const a = this.view;
    if (this.view !== null && this.view.enabled) {
      const l = a.fullWidth, c = a.fullHeight;
      r += a.offsetX * s / l, e -= a.offsetY * n / c, s *= a.width / l, n *= a.height / c;
    }
    const o = this.filmOffset;
    o !== 0 && (r += t * o / this.getFilmWidth()), this.projectionMatrix.makePerspective(r, r + s, e, e - n, t, this.far, this.coordinateSystem), this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
  toJSON(t) {
    const e = super.toJSON(t);
    return e.object.fov = this.fov, e.object.zoom = this.zoom, e.object.near = this.near, e.object.far = this.far, e.object.focus = this.focus, e.object.aspect = this.aspect, this.view !== null && (e.object.view = Object.assign({}, this.view)), e.object.filmGauge = this.filmGauge, e.object.filmOffset = this.filmOffset, e;
  }
}
const fi = -90, pi = 1;
class xh extends Re {
  constructor(t, e, n) {
    super(), this.type = "CubeCamera", this.renderTarget = n, this.coordinateSystem = null, this.activeMipmapLevel = 0;
    const s = new ze(fi, pi, t, e);
    s.layers = this.layers, this.add(s);
    const r = new ze(fi, pi, t, e);
    r.layers = this.layers, this.add(r);
    const a = new ze(fi, pi, t, e);
    a.layers = this.layers, this.add(a);
    const o = new ze(fi, pi, t, e);
    o.layers = this.layers, this.add(o);
    const l = new ze(fi, pi, t, e);
    l.layers = this.layers, this.add(l);
    const c = new ze(fi, pi, t, e);
    c.layers = this.layers, this.add(c);
  }
  updateCoordinateSystem() {
    const t = this.coordinateSystem, e = this.children.concat(), [n, s, r, a, o, l] = e;
    for (const c of e) this.remove(c);
    if (t === vn)
      n.up.set(0, 1, 0), n.lookAt(1, 0, 0), s.up.set(0, 1, 0), s.lookAt(-1, 0, 0), r.up.set(0, 0, -1), r.lookAt(0, 1, 0), a.up.set(0, 0, 1), a.lookAt(0, -1, 0), o.up.set(0, 1, 0), o.lookAt(0, 0, 1), l.up.set(0, 1, 0), l.lookAt(0, 0, -1);
    else if (t === Ds)
      n.up.set(0, -1, 0), n.lookAt(-1, 0, 0), s.up.set(0, -1, 0), s.lookAt(1, 0, 0), r.up.set(0, 0, 1), r.lookAt(0, 1, 0), a.up.set(0, 0, -1), a.lookAt(0, -1, 0), o.up.set(0, -1, 0), o.lookAt(0, 0, 1), l.up.set(0, -1, 0), l.lookAt(0, 0, -1);
    else
      throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: " + t);
    for (const c of e)
      this.add(c), c.updateMatrixWorld();
  }
  update(t, e) {
    this.parent === null && this.updateMatrixWorld();
    const { renderTarget: n, activeMipmapLevel: s } = this;
    this.coordinateSystem !== t.coordinateSystem && (this.coordinateSystem = t.coordinateSystem, this.updateCoordinateSystem());
    const [r, a, o, l, c, u] = this.children, d = t.getRenderTarget(), f = t.getActiveCubeFace(), m = t.getActiveMipmapLevel(), g = t.xr.enabled;
    t.xr.enabled = !1;
    const v = n.texture.generateMipmaps;
    n.texture.generateMipmaps = !1, t.setRenderTarget(n, 0, s), t.render(e, r), t.setRenderTarget(n, 1, s), t.render(e, a), t.setRenderTarget(n, 2, s), t.render(e, o), t.setRenderTarget(n, 3, s), t.render(e, l), t.setRenderTarget(n, 4, s), t.render(e, c), n.texture.generateMipmaps = v, t.setRenderTarget(n, 5, s), t.render(e, u), t.setRenderTarget(d, f, m), t.xr.enabled = g, n.texture.needsPMREMUpdate = !0;
  }
}
class Al extends Ie {
  constructor(t, e, n, s, r, a, o, l, c, u) {
    t = t !== void 0 ? t : [], e = e !== void 0 ? e : bi, super(t, e, n, s, r, a, o, l, c, u), this.isCubeTexture = !0, this.flipY = !1;
  }
  get images() {
    return this.image;
  }
  set images(t) {
    this.image = t;
  }
}
class Mh extends Zn {
  constructor(t = 1, e = {}) {
    super(t, t, e), this.isWebGLCubeRenderTarget = !0;
    const n = { width: t, height: t, depth: 1 }, s = [n, n, n, n, n, n];
    this.texture = new Al(s, e.mapping, e.wrapS, e.wrapT, e.magFilter, e.minFilter, e.format, e.type, e.anisotropy, e.colorSpace), this.texture.isRenderTargetTexture = !0, this.texture.generateMipmaps = e.generateMipmaps !== void 0 ? e.generateMipmaps : !1, this.texture.minFilter = e.minFilter !== void 0 ? e.minFilter : an;
  }
  fromEquirectangularTexture(t, e) {
    this.texture.type = e.type, this.texture.colorSpace = e.colorSpace, this.texture.generateMipmaps = e.generateMipmaps, this.texture.minFilter = e.minFilter, this.texture.magFilter = e.magFilter;
    const n = {
      uniforms: {
        tEquirect: { value: null }
      },
      vertexShader: (
        /* glsl */
        `

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`
      ),
      fragmentShader: (
        /* glsl */
        `

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`
      )
    }, s = new Yi(5, 5, 5), r = new En({
      name: "CubemapFromEquirect",
      uniforms: Ci(n.uniforms),
      vertexShader: n.vertexShader,
      fragmentShader: n.fragmentShader,
      side: Ue,
      blending: Ln
    });
    r.uniforms.tEquirect.value = e;
    const a = new be(s, r), o = e.minFilter;
    return e.minFilter === $n && (e.minFilter = an), new xh(1, 10, this).update(t, a), e.minFilter = o, a.geometry.dispose(), a.material.dispose(), this;
  }
  clear(t, e, n, s) {
    const r = t.getRenderTarget();
    for (let a = 0; a < 6; a++)
      t.setRenderTarget(this, a), t.clear(e, n, s);
    t.setRenderTarget(r);
  }
}
class Sh extends Re {
  constructor() {
    super(), this.isScene = !0, this.type = "Scene", this.background = null, this.environment = null, this.fog = null, this.backgroundBlurriness = 0, this.backgroundIntensity = 1, this.backgroundRotation = new yn(), this.environmentIntensity = 1, this.environmentRotation = new yn(), this.overrideMaterial = null, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
  }
  copy(t, e) {
    return super.copy(t, e), t.background !== null && (this.background = t.background.clone()), t.environment !== null && (this.environment = t.environment.clone()), t.fog !== null && (this.fog = t.fog.clone()), this.backgroundBlurriness = t.backgroundBlurriness, this.backgroundIntensity = t.backgroundIntensity, this.backgroundRotation.copy(t.backgroundRotation), this.environmentIntensity = t.environmentIntensity, this.environmentRotation.copy(t.environmentRotation), t.overrideMaterial !== null && (this.overrideMaterial = t.overrideMaterial.clone()), this.matrixAutoUpdate = t.matrixAutoUpdate, this;
  }
  toJSON(t) {
    const e = super.toJSON(t);
    return this.fog !== null && (e.object.fog = this.fog.toJSON()), this.backgroundBlurriness > 0 && (e.object.backgroundBlurriness = this.backgroundBlurriness), this.backgroundIntensity !== 1 && (e.object.backgroundIntensity = this.backgroundIntensity), e.object.backgroundRotation = this.backgroundRotation.toArray(), this.environmentIntensity !== 1 && (e.object.environmentIntensity = this.environmentIntensity), e.object.environmentRotation = this.environmentRotation.toArray(), e;
  }
}
class yh {
  constructor(t, e) {
    this.isInterleavedBuffer = !0, this.array = t, this.stride = e, this.count = t !== void 0 ? t.length / e : 0, this.usage = ca, this.updateRanges = [], this.version = 0, this.uuid = xn();
  }
  onUploadCallback() {
  }
  set needsUpdate(t) {
    t === !0 && this.version++;
  }
  setUsage(t) {
    return this.usage = t, this;
  }
  addUpdateRange(t, e) {
    this.updateRanges.push({ start: t, count: e });
  }
  clearUpdateRanges() {
    this.updateRanges.length = 0;
  }
  copy(t) {
    return this.array = new t.array.constructor(t.array), this.count = t.count, this.stride = t.stride, this.usage = t.usage, this;
  }
  copyAt(t, e, n) {
    t *= this.stride, n *= e.stride;
    for (let s = 0, r = this.stride; s < r; s++)
      this.array[t + s] = e.array[n + s];
    return this;
  }
  set(t, e = 0) {
    return this.array.set(t, e), this;
  }
  clone(t) {
    t.arrayBuffers === void 0 && (t.arrayBuffers = {}), this.array.buffer._uuid === void 0 && (this.array.buffer._uuid = xn()), t.arrayBuffers[this.array.buffer._uuid] === void 0 && (t.arrayBuffers[this.array.buffer._uuid] = this.array.slice(0).buffer);
    const e = new this.array.constructor(t.arrayBuffers[this.array.buffer._uuid]), n = new this.constructor(e, this.stride);
    return n.setUsage(this.usage), n;
  }
  onUpload(t) {
    return this.onUploadCallback = t, this;
  }
  toJSON(t) {
    return t.arrayBuffers === void 0 && (t.arrayBuffers = {}), this.array.buffer._uuid === void 0 && (this.array.buffer._uuid = xn()), t.arrayBuffers[this.array.buffer._uuid] === void 0 && (t.arrayBuffers[this.array.buffer._uuid] = Array.from(new Uint32Array(this.array.buffer))), {
      uuid: this.uuid,
      buffer: this.array.buffer._uuid,
      type: this.array.constructor.name,
      stride: this.stride
    };
  }
}
const Te = /* @__PURE__ */ new P();
class Dn {
  constructor(t, e, n, s = !1) {
    this.isInterleavedBufferAttribute = !0, this.name = "", this.data = t, this.itemSize = e, this.offset = n, this.normalized = s;
  }
  get count() {
    return this.data.count;
  }
  get array() {
    return this.data.array;
  }
  set needsUpdate(t) {
    this.data.needsUpdate = t;
  }
  applyMatrix4(t) {
    for (let e = 0, n = this.data.count; e < n; e++)
      Te.fromBufferAttribute(this, e), Te.applyMatrix4(t), this.setXYZ(e, Te.x, Te.y, Te.z);
    return this;
  }
  applyNormalMatrix(t) {
    for (let e = 0, n = this.count; e < n; e++)
      Te.fromBufferAttribute(this, e), Te.applyNormalMatrix(t), this.setXYZ(e, Te.x, Te.y, Te.z);
    return this;
  }
  transformDirection(t) {
    for (let e = 0, n = this.count; e < n; e++)
      Te.fromBufferAttribute(this, e), Te.transformDirection(t), this.setXYZ(e, Te.x, Te.y, Te.z);
    return this;
  }
  getComponent(t, e) {
    let n = this.array[t * this.data.stride + this.offset + e];
    return this.normalized && (n = Ke(n, this.array)), n;
  }
  setComponent(t, e, n) {
    return this.normalized && (n = qt(n, this.array)), this.data.array[t * this.data.stride + this.offset + e] = n, this;
  }
  setX(t, e) {
    return this.normalized && (e = qt(e, this.array)), this.data.array[t * this.data.stride + this.offset] = e, this;
  }
  setY(t, e) {
    return this.normalized && (e = qt(e, this.array)), this.data.array[t * this.data.stride + this.offset + 1] = e, this;
  }
  setZ(t, e) {
    return this.normalized && (e = qt(e, this.array)), this.data.array[t * this.data.stride + this.offset + 2] = e, this;
  }
  setW(t, e) {
    return this.normalized && (e = qt(e, this.array)), this.data.array[t * this.data.stride + this.offset + 3] = e, this;
  }
  getX(t) {
    let e = this.data.array[t * this.data.stride + this.offset];
    return this.normalized && (e = Ke(e, this.array)), e;
  }
  getY(t) {
    let e = this.data.array[t * this.data.stride + this.offset + 1];
    return this.normalized && (e = Ke(e, this.array)), e;
  }
  getZ(t) {
    let e = this.data.array[t * this.data.stride + this.offset + 2];
    return this.normalized && (e = Ke(e, this.array)), e;
  }
  getW(t) {
    let e = this.data.array[t * this.data.stride + this.offset + 3];
    return this.normalized && (e = Ke(e, this.array)), e;
  }
  setXY(t, e, n) {
    return t = t * this.data.stride + this.offset, this.normalized && (e = qt(e, this.array), n = qt(n, this.array)), this.data.array[t + 0] = e, this.data.array[t + 1] = n, this;
  }
  setXYZ(t, e, n, s) {
    return t = t * this.data.stride + this.offset, this.normalized && (e = qt(e, this.array), n = qt(n, this.array), s = qt(s, this.array)), this.data.array[t + 0] = e, this.data.array[t + 1] = n, this.data.array[t + 2] = s, this;
  }
  setXYZW(t, e, n, s, r) {
    return t = t * this.data.stride + this.offset, this.normalized && (e = qt(e, this.array), n = qt(n, this.array), s = qt(s, this.array), r = qt(r, this.array)), this.data.array[t + 0] = e, this.data.array[t + 1] = n, this.data.array[t + 2] = s, this.data.array[t + 3] = r, this;
  }
  clone(t) {
    if (t === void 0) {
      console.log("THREE.InterleavedBufferAttribute.clone(): Cloning an interleaved buffer attribute will de-interleave buffer data.");
      const e = [];
      for (let n = 0; n < this.count; n++) {
        const s = n * this.data.stride + this.offset;
        for (let r = 0; r < this.itemSize; r++)
          e.push(this.data.array[s + r]);
      }
      return new en(new this.array.constructor(e), this.itemSize, this.normalized);
    } else
      return t.interleavedBuffers === void 0 && (t.interleavedBuffers = {}), t.interleavedBuffers[this.data.uuid] === void 0 && (t.interleavedBuffers[this.data.uuid] = this.data.clone(t)), new Dn(t.interleavedBuffers[this.data.uuid], this.itemSize, this.offset, this.normalized);
  }
  toJSON(t) {
    if (t === void 0) {
      console.log("THREE.InterleavedBufferAttribute.toJSON(): Serializing an interleaved buffer attribute will de-interleave buffer data.");
      const e = [];
      for (let n = 0; n < this.count; n++) {
        const s = n * this.data.stride + this.offset;
        for (let r = 0; r < this.itemSize; r++)
          e.push(this.data.array[s + r]);
      }
      return {
        itemSize: this.itemSize,
        type: this.array.constructor.name,
        array: e,
        normalized: this.normalized
      };
    } else
      return t.interleavedBuffers === void 0 && (t.interleavedBuffers = {}), t.interleavedBuffers[this.data.uuid] === void 0 && (t.interleavedBuffers[this.data.uuid] = this.data.toJSON(t)), {
        isInterleavedBufferAttribute: !0,
        itemSize: this.itemSize,
        data: this.data.uuid,
        offset: this.offset,
        normalized: this.normalized
      };
  }
}
const or = /* @__PURE__ */ new P(), Eh = /* @__PURE__ */ new P(), bh = /* @__PURE__ */ new Pt();
class _n {
  constructor(t = new P(1, 0, 0), e = 0) {
    this.isPlane = !0, this.normal = t, this.constant = e;
  }
  set(t, e) {
    return this.normal.copy(t), this.constant = e, this;
  }
  setComponents(t, e, n, s) {
    return this.normal.set(t, e, n), this.constant = s, this;
  }
  setFromNormalAndCoplanarPoint(t, e) {
    return this.normal.copy(t), this.constant = -e.dot(this.normal), this;
  }
  setFromCoplanarPoints(t, e, n) {
    const s = or.subVectors(n, e).cross(Eh.subVectors(t, e)).normalize();
    return this.setFromNormalAndCoplanarPoint(s, t), this;
  }
  copy(t) {
    return this.normal.copy(t.normal), this.constant = t.constant, this;
  }
  normalize() {
    const t = 1 / this.normal.length();
    return this.normal.multiplyScalar(t), this.constant *= t, this;
  }
  negate() {
    return this.constant *= -1, this.normal.negate(), this;
  }
  distanceToPoint(t) {
    return this.normal.dot(t) + this.constant;
  }
  distanceToSphere(t) {
    return this.distanceToPoint(t.center) - t.radius;
  }
  projectPoint(t, e) {
    return e.copy(t).addScaledVector(this.normal, -this.distanceToPoint(t));
  }
  intersectLine(t, e) {
    const n = t.delta(or), s = this.normal.dot(n);
    if (s === 0)
      return this.distanceToPoint(t.start) === 0 ? e.copy(t.start) : null;
    const r = -(t.start.dot(this.normal) + this.constant) / s;
    return r < 0 || r > 1 ? null : e.copy(t.start).addScaledVector(n, r);
  }
  intersectsLine(t) {
    const e = this.distanceToPoint(t.start), n = this.distanceToPoint(t.end);
    return e < 0 && n > 0 || n < 0 && e > 0;
  }
  intersectsBox(t) {
    return t.intersectsPlane(this);
  }
  intersectsSphere(t) {
    return t.intersectsPlane(this);
  }
  coplanarPoint(t) {
    return t.copy(this.normal).multiplyScalar(-this.constant);
  }
  applyMatrix4(t, e) {
    const n = e || bh.getNormalMatrix(t), s = this.coplanarPoint(or).applyMatrix4(t), r = this.normal.applyMatrix3(n).normalize();
    return this.constant = -s.dot(r), this;
  }
  translate(t) {
    return this.constant -= t.dot(this.normal), this;
  }
  equals(t) {
    return t.normal.equals(this.normal) && t.constant === this.constant;
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
const Hn = /* @__PURE__ */ new Pi(), ds = /* @__PURE__ */ new P();
class Tl {
  constructor(t = new _n(), e = new _n(), n = new _n(), s = new _n(), r = new _n(), a = new _n()) {
    this.planes = [t, e, n, s, r, a];
  }
  set(t, e, n, s, r, a) {
    const o = this.planes;
    return o[0].copy(t), o[1].copy(e), o[2].copy(n), o[3].copy(s), o[4].copy(r), o[5].copy(a), this;
  }
  copy(t) {
    const e = this.planes;
    for (let n = 0; n < 6; n++)
      e[n].copy(t.planes[n]);
    return this;
  }
  setFromProjectionMatrix(t, e = vn) {
    const n = this.planes, s = t.elements, r = s[0], a = s[1], o = s[2], l = s[3], c = s[4], u = s[5], d = s[6], f = s[7], m = s[8], g = s[9], v = s[10], p = s[11], h = s[12], E = s[13], b = s[14], S = s[15];
    if (n[0].setComponents(l - r, f - c, p - m, S - h).normalize(), n[1].setComponents(l + r, f + c, p + m, S + h).normalize(), n[2].setComponents(l + a, f + u, p + g, S + E).normalize(), n[3].setComponents(l - a, f - u, p - g, S - E).normalize(), n[4].setComponents(l - o, f - d, p - v, S - b).normalize(), e === vn)
      n[5].setComponents(l + o, f + d, p + v, S + b).normalize();
    else if (e === Ds)
      n[5].setComponents(o, d, v, b).normalize();
    else
      throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: " + e);
    return this;
  }
  intersectsObject(t) {
    if (t.boundingSphere !== void 0)
      t.boundingSphere === null && t.computeBoundingSphere(), Hn.copy(t.boundingSphere).applyMatrix4(t.matrixWorld);
    else {
      const e = t.geometry;
      e.boundingSphere === null && e.computeBoundingSphere(), Hn.copy(e.boundingSphere).applyMatrix4(t.matrixWorld);
    }
    return this.intersectsSphere(Hn);
  }
  intersectsSprite(t) {
    return Hn.center.set(0, 0, 0), Hn.radius = 0.7071067811865476, Hn.applyMatrix4(t.matrixWorld), this.intersectsSphere(Hn);
  }
  intersectsSphere(t) {
    const e = this.planes, n = t.center, s = -t.radius;
    for (let r = 0; r < 6; r++)
      if (e[r].distanceToPoint(n) < s)
        return !1;
    return !0;
  }
  intersectsBox(t) {
    const e = this.planes;
    for (let n = 0; n < 6; n++) {
      const s = e[n];
      if (ds.x = s.normal.x > 0 ? t.max.x : t.min.x, ds.y = s.normal.y > 0 ? t.max.y : t.min.y, ds.z = s.normal.z > 0 ? t.max.z : t.min.z, s.distanceToPoint(ds) < 0)
        return !1;
    }
    return !0;
  }
  containsPoint(t) {
    const e = this.planes;
    for (let n = 0; n < 6; n++)
      if (e[n].distanceToPoint(t) < 0)
        return !1;
    return !0;
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
class Ea extends ti {
  constructor(t) {
    super(), this.isLineBasicMaterial = !0, this.type = "LineBasicMaterial", this.color = new Vt(16777215), this.map = null, this.linewidth = 1, this.linecap = "round", this.linejoin = "round", this.fog = !0, this.setValues(t);
  }
  copy(t) {
    return super.copy(t), this.color.copy(t.color), this.map = t.map, this.linewidth = t.linewidth, this.linecap = t.linecap, this.linejoin = t.linejoin, this.fog = t.fog, this;
  }
}
const Us = /* @__PURE__ */ new P(), Is = /* @__PURE__ */ new P(), co = /* @__PURE__ */ new ne(), Oi = /* @__PURE__ */ new Sa(), fs = /* @__PURE__ */ new Pi(), lr = /* @__PURE__ */ new P(), ho = /* @__PURE__ */ new P();
class wl extends Re {
  constructor(t = new Ce(), e = new Ea()) {
    super(), this.isLine = !0, this.type = "Line", this.geometry = t, this.material = e, this.updateMorphTargets();
  }
  copy(t, e) {
    return super.copy(t, e), this.material = Array.isArray(t.material) ? t.material.slice() : t.material, this.geometry = t.geometry, this;
  }
  computeLineDistances() {
    const t = this.geometry;
    if (t.index === null) {
      const e = t.attributes.position, n = [0];
      for (let s = 1, r = e.count; s < r; s++)
        Us.fromBufferAttribute(e, s - 1), Is.fromBufferAttribute(e, s), n[s] = n[s - 1], n[s] += Us.distanceTo(Is);
      t.setAttribute("lineDistance", new se(n, 1));
    } else
      console.warn("THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");
    return this;
  }
  raycast(t, e) {
    const n = this.geometry, s = this.matrixWorld, r = t.params.Line.threshold, a = n.drawRange;
    if (n.boundingSphere === null && n.computeBoundingSphere(), fs.copy(n.boundingSphere), fs.applyMatrix4(s), fs.radius += r, t.ray.intersectsSphere(fs) === !1) return;
    co.copy(s).invert(), Oi.copy(t.ray).applyMatrix4(co);
    const o = r / ((this.scale.x + this.scale.y + this.scale.z) / 3), l = o * o, c = this.isLineSegments ? 2 : 1, u = n.index, f = n.attributes.position;
    if (u !== null) {
      const m = Math.max(0, a.start), g = Math.min(u.count, a.start + a.count);
      for (let v = m, p = g - 1; v < p; v += c) {
        const h = u.getX(v), E = u.getX(v + 1), b = ps(this, t, Oi, l, h, E);
        b && e.push(b);
      }
      if (this.isLineLoop) {
        const v = u.getX(g - 1), p = u.getX(m), h = ps(this, t, Oi, l, v, p);
        h && e.push(h);
      }
    } else {
      const m = Math.max(0, a.start), g = Math.min(f.count, a.start + a.count);
      for (let v = m, p = g - 1; v < p; v += c) {
        const h = ps(this, t, Oi, l, v, v + 1);
        h && e.push(h);
      }
      if (this.isLineLoop) {
        const v = ps(this, t, Oi, l, g - 1, m);
        v && e.push(v);
      }
    }
  }
  updateMorphTargets() {
    const e = this.geometry.morphAttributes, n = Object.keys(e);
    if (n.length > 0) {
      const s = e[n[0]];
      if (s !== void 0) {
        this.morphTargetInfluences = [], this.morphTargetDictionary = {};
        for (let r = 0, a = s.length; r < a; r++) {
          const o = s[r].name || String(r);
          this.morphTargetInfluences.push(0), this.morphTargetDictionary[o] = r;
        }
      }
    }
  }
}
function ps(i, t, e, n, s, r) {
  const a = i.geometry.attributes.position;
  if (Us.fromBufferAttribute(a, s), Is.fromBufferAttribute(a, r), e.distanceSqToSegment(Us, Is, lr, ho) > n) return;
  lr.applyMatrix4(i.matrixWorld);
  const l = t.ray.origin.distanceTo(lr);
  if (!(l < t.near || l > t.far))
    return {
      distance: l,
      // What do we want? intersection point on the ray or on the segment??
      // point: raycaster.ray.at( distance ),
      point: ho.clone().applyMatrix4(i.matrixWorld),
      index: s,
      face: null,
      faceIndex: null,
      barycoord: null,
      object: i
    };
}
const uo = /* @__PURE__ */ new P(), fo = /* @__PURE__ */ new P();
class Ah extends wl {
  constructor(t, e) {
    super(t, e), this.isLineSegments = !0, this.type = "LineSegments";
  }
  computeLineDistances() {
    const t = this.geometry;
    if (t.index === null) {
      const e = t.attributes.position, n = [];
      for (let s = 0, r = e.count; s < r; s += 2)
        uo.fromBufferAttribute(e, s), fo.fromBufferAttribute(e, s + 1), n[s] = s === 0 ? 0 : n[s - 1], n[s + 1] = n[s] + uo.distanceTo(fo);
      t.setAttribute("lineDistance", new se(n, 1));
    } else
      console.warn("THREE.LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");
    return this;
  }
}
class le extends Re {
  constructor() {
    super(), this.isGroup = !0, this.type = "Group";
  }
}
class Rl extends Ie {
  constructor(t, e, n, s, r, a, o, l, c, u = Si) {
    if (u !== Si && u !== wi)
      throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");
    n === void 0 && u === Si && (n = jn), n === void 0 && u === wi && (n = Ti), super(null, s, r, a, o, l, u, n, c), this.isDepthTexture = !0, this.image = { width: t, height: e }, this.magFilter = o !== void 0 ? o : tn, this.minFilter = l !== void 0 ? l : tn, this.flipY = !1, this.generateMipmaps = !1, this.compareFunction = null;
  }
  copy(t) {
    return super.copy(t), this.compareFunction = t.compareFunction, this;
  }
  toJSON(t) {
    const e = super.toJSON(t);
    return this.compareFunction !== null && (e.compareFunction = this.compareFunction), e;
  }
}
class ba extends Ce {
  constructor(t = [new bt(0, -0.5), new bt(0.5, 0), new bt(0, 0.5)], e = 12, n = 0, s = Math.PI * 2) {
    super(), this.type = "LatheGeometry", this.parameters = {
      points: t,
      segments: e,
      phiStart: n,
      phiLength: s
    }, e = Math.floor(e), s = Ut(s, 0, Math.PI * 2);
    const r = [], a = [], o = [], l = [], c = [], u = 1 / e, d = new P(), f = new bt(), m = new P(), g = new P(), v = new P();
    let p = 0, h = 0;
    for (let E = 0; E <= t.length - 1; E++)
      switch (E) {
        case 0:
          p = t[E + 1].x - t[E].x, h = t[E + 1].y - t[E].y, m.x = h * 1, m.y = -p, m.z = h * 0, v.copy(m), m.normalize(), l.push(m.x, m.y, m.z);
          break;
        case t.length - 1:
          l.push(v.x, v.y, v.z);
          break;
        default:
          p = t[E + 1].x - t[E].x, h = t[E + 1].y - t[E].y, m.x = h * 1, m.y = -p, m.z = h * 0, g.copy(m), m.x += v.x, m.y += v.y, m.z += v.z, m.normalize(), l.push(m.x, m.y, m.z), v.copy(g);
      }
    for (let E = 0; E <= e; E++) {
      const b = n + E * u * s, S = Math.sin(b), L = Math.cos(b);
      for (let T = 0; T <= t.length - 1; T++) {
        d.x = t[T].x * S, d.y = t[T].y, d.z = t[T].x * L, a.push(d.x, d.y, d.z), f.x = E / e, f.y = T / (t.length - 1), o.push(f.x, f.y);
        const R = l[3 * T + 0] * S, I = l[3 * T + 1], y = l[3 * T + 0] * L;
        c.push(R, I, y);
      }
    }
    for (let E = 0; E < e; E++)
      for (let b = 0; b < t.length - 1; b++) {
        const S = b + E * t.length, L = S, T = S + t.length, R = S + t.length + 1, I = S + 1;
        r.push(L, T, I), r.push(R, I, T);
      }
    this.setIndex(r), this.setAttribute("position", new se(a, 3)), this.setAttribute("uv", new se(o, 2)), this.setAttribute("normal", new se(c, 3));
  }
  copy(t) {
    return super.copy(t), this.parameters = Object.assign({}, t.parameters), this;
  }
  static fromJSON(t) {
    return new ba(t.points, t.segments, t.phiStart, t.phiLength);
  }
}
class Aa extends Ce {
  constructor(t = 1, e = 1, n = 1, s = 32, r = 1, a = !1, o = 0, l = Math.PI * 2) {
    super(), this.type = "CylinderGeometry", this.parameters = {
      radiusTop: t,
      radiusBottom: e,
      height: n,
      radialSegments: s,
      heightSegments: r,
      openEnded: a,
      thetaStart: o,
      thetaLength: l
    };
    const c = this;
    s = Math.floor(s), r = Math.floor(r);
    const u = [], d = [], f = [], m = [];
    let g = 0;
    const v = [], p = n / 2;
    let h = 0;
    E(), a === !1 && (t > 0 && b(!0), e > 0 && b(!1)), this.setIndex(u), this.setAttribute("position", new se(d, 3)), this.setAttribute("normal", new se(f, 3)), this.setAttribute("uv", new se(m, 2));
    function E() {
      const S = new P(), L = new P();
      let T = 0;
      const R = (e - t) / n;
      for (let I = 0; I <= r; I++) {
        const y = [], M = I / r, C = M * (e - t) + t;
        for (let H = 0; H <= s; H++) {
          const z = H / s, G = z * l + o, j = Math.sin(G), W = Math.cos(G);
          L.x = C * j, L.y = -M * n + p, L.z = C * W, d.push(L.x, L.y, L.z), S.set(j, R, W).normalize(), f.push(S.x, S.y, S.z), m.push(z, 1 - M), y.push(g++);
        }
        v.push(y);
      }
      for (let I = 0; I < s; I++)
        for (let y = 0; y < r; y++) {
          const M = v[y][I], C = v[y + 1][I], H = v[y + 1][I + 1], z = v[y][I + 1];
          (t > 0 || y !== 0) && (u.push(M, C, z), T += 3), (e > 0 || y !== r - 1) && (u.push(C, H, z), T += 3);
        }
      c.addGroup(h, T, 0), h += T;
    }
    function b(S) {
      const L = g, T = new bt(), R = new P();
      let I = 0;
      const y = S === !0 ? t : e, M = S === !0 ? 1 : -1;
      for (let H = 1; H <= s; H++)
        d.push(0, p * M, 0), f.push(0, M, 0), m.push(0.5, 0.5), g++;
      const C = g;
      for (let H = 0; H <= s; H++) {
        const G = H / s * l + o, j = Math.cos(G), W = Math.sin(G);
        R.x = y * W, R.y = p * M, R.z = y * j, d.push(R.x, R.y, R.z), f.push(0, M, 0), T.x = j * 0.5 + 0.5, T.y = W * 0.5 * M + 0.5, m.push(T.x, T.y), g++;
      }
      for (let H = 0; H < s; H++) {
        const z = L + H, G = C + H;
        S === !0 ? u.push(G, G + 1, z) : u.push(G + 1, G, z), I += 3;
      }
      c.addGroup(h, I, S === !0 ? 1 : 2), h += I;
    }
  }
  copy(t) {
    return super.copy(t), this.parameters = Object.assign({}, t.parameters), this;
  }
  static fromJSON(t) {
    return new Aa(t.radiusTop, t.radiusBottom, t.height, t.radialSegments, t.heightSegments, t.openEnded, t.thetaStart, t.thetaLength);
  }
}
class Os extends Ce {
  constructor(t = 1, e = 1, n = 1, s = 1) {
    super(), this.type = "PlaneGeometry", this.parameters = {
      width: t,
      height: e,
      widthSegments: n,
      heightSegments: s
    };
    const r = t / 2, a = e / 2, o = Math.floor(n), l = Math.floor(s), c = o + 1, u = l + 1, d = t / o, f = e / l, m = [], g = [], v = [], p = [];
    for (let h = 0; h < u; h++) {
      const E = h * f - a;
      for (let b = 0; b < c; b++) {
        const S = b * d - r;
        g.push(S, -E, 0), v.push(0, 0, 1), p.push(b / o), p.push(1 - h / l);
      }
    }
    for (let h = 0; h < l; h++)
      for (let E = 0; E < o; E++) {
        const b = E + c * h, S = E + c * (h + 1), L = E + 1 + c * (h + 1), T = E + 1 + c * h;
        m.push(b, S, T), m.push(S, L, T);
      }
    this.setIndex(m), this.setAttribute("position", new se(g, 3)), this.setAttribute("normal", new se(v, 3)), this.setAttribute("uv", new se(p, 2));
  }
  copy(t) {
    return super.copy(t), this.parameters = Object.assign({}, t.parameters), this;
  }
  static fromJSON(t) {
    return new Os(t.width, t.height, t.widthSegments, t.heightSegments);
  }
}
class Ta extends Ce {
  constructor(t = 1, e = 32, n = 16, s = 0, r = Math.PI * 2, a = 0, o = Math.PI) {
    super(), this.type = "SphereGeometry", this.parameters = {
      radius: t,
      widthSegments: e,
      heightSegments: n,
      phiStart: s,
      phiLength: r,
      thetaStart: a,
      thetaLength: o
    }, e = Math.max(3, Math.floor(e)), n = Math.max(2, Math.floor(n));
    const l = Math.min(a + o, Math.PI);
    let c = 0;
    const u = [], d = new P(), f = new P(), m = [], g = [], v = [], p = [];
    for (let h = 0; h <= n; h++) {
      const E = [], b = h / n;
      let S = 0;
      h === 0 && a === 0 ? S = 0.5 / e : h === n && l === Math.PI && (S = -0.5 / e);
      for (let L = 0; L <= e; L++) {
        const T = L / e;
        d.x = -t * Math.cos(s + T * r) * Math.sin(a + b * o), d.y = t * Math.cos(a + b * o), d.z = t * Math.sin(s + T * r) * Math.sin(a + b * o), g.push(d.x, d.y, d.z), f.copy(d).normalize(), v.push(f.x, f.y, f.z), p.push(T + S, 1 - b), E.push(c++);
      }
      u.push(E);
    }
    for (let h = 0; h < n; h++)
      for (let E = 0; E < e; E++) {
        const b = u[h][E + 1], S = u[h][E], L = u[h + 1][E], T = u[h + 1][E + 1];
        (h !== 0 || a > 0) && m.push(b, S, T), (h !== n - 1 || l < Math.PI) && m.push(S, L, T);
      }
    this.setIndex(m), this.setAttribute("position", new se(g, 3)), this.setAttribute("normal", new se(v, 3)), this.setAttribute("uv", new se(p, 2));
  }
  copy(t) {
    return super.copy(t), this.parameters = Object.assign({}, t.parameters), this;
  }
  static fromJSON(t) {
    return new Ta(t.radius, t.widthSegments, t.heightSegments, t.phiStart, t.phiLength, t.thetaStart, t.thetaLength);
  }
}
class Th extends Ce {
  constructor(t = null) {
    if (super(), this.type = "WireframeGeometry", this.parameters = {
      geometry: t
    }, t !== null) {
      const e = [], n = /* @__PURE__ */ new Set(), s = new P(), r = new P();
      if (t.index !== null) {
        const a = t.attributes.position, o = t.index;
        let l = t.groups;
        l.length === 0 && (l = [{ start: 0, count: o.count, materialIndex: 0 }]);
        for (let c = 0, u = l.length; c < u; ++c) {
          const d = l[c], f = d.start, m = d.count;
          for (let g = f, v = f + m; g < v; g += 3)
            for (let p = 0; p < 3; p++) {
              const h = o.getX(g + p), E = o.getX(g + (p + 1) % 3);
              s.fromBufferAttribute(a, h), r.fromBufferAttribute(a, E), po(s, r, n) === !0 && (e.push(s.x, s.y, s.z), e.push(r.x, r.y, r.z));
            }
        }
      } else {
        const a = t.attributes.position;
        for (let o = 0, l = a.count / 3; o < l; o++)
          for (let c = 0; c < 3; c++) {
            const u = 3 * o + c, d = 3 * o + (c + 1) % 3;
            s.fromBufferAttribute(a, u), r.fromBufferAttribute(a, d), po(s, r, n) === !0 && (e.push(s.x, s.y, s.z), e.push(r.x, r.y, r.z));
          }
      }
      this.setAttribute("position", new se(e, 3));
    }
  }
  copy(t) {
    return super.copy(t), this.parameters = Object.assign({}, t.parameters), this;
  }
}
function po(i, t, e) {
  const n = `${i.x},${i.y},${i.z}-${t.x},${t.y},${t.z}`, s = `${t.x},${t.y},${t.z}-${i.x},${i.y},${i.z}`;
  return e.has(n) === !0 || e.has(s) === !0 ? !1 : (e.add(n), e.add(s), !0);
}
class wh extends ti {
  constructor(t) {
    super(), this.isMeshNormalMaterial = !0, this.type = "MeshNormalMaterial", this.bumpMap = null, this.bumpScale = 1, this.normalMap = null, this.normalMapType = fl, this.normalScale = new bt(1, 1), this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.wireframe = !1, this.wireframeLinewidth = 1, this.flatShading = !1, this.setValues(t);
  }
  copy(t) {
    return super.copy(t), this.bumpMap = t.bumpMap, this.bumpScale = t.bumpScale, this.normalMap = t.normalMap, this.normalMapType = t.normalMapType, this.normalScale.copy(t.normalScale), this.displacementMap = t.displacementMap, this.displacementScale = t.displacementScale, this.displacementBias = t.displacementBias, this.wireframe = t.wireframe, this.wireframeLinewidth = t.wireframeLinewidth, this.flatShading = t.flatShading, this;
  }
}
class Rh extends ti {
  constructor(t) {
    super(), this.isMeshDepthMaterial = !0, this.type = "MeshDepthMaterial", this.depthPacking = bc, this.map = null, this.alphaMap = null, this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.wireframe = !1, this.wireframeLinewidth = 1, this.setValues(t);
  }
  copy(t) {
    return super.copy(t), this.depthPacking = t.depthPacking, this.map = t.map, this.alphaMap = t.alphaMap, this.displacementMap = t.displacementMap, this.displacementScale = t.displacementScale, this.displacementBias = t.displacementBias, this.wireframe = t.wireframe, this.wireframeLinewidth = t.wireframeLinewidth, this;
  }
}
class Ch extends ti {
  constructor(t) {
    super(), this.isMeshDistanceMaterial = !0, this.type = "MeshDistanceMaterial", this.map = null, this.alphaMap = null, this.displacementMap = null, this.displacementScale = 1, this.displacementBias = 0, this.setValues(t);
  }
  copy(t) {
    return super.copy(t), this.map = t.map, this.alphaMap = t.alphaMap, this.displacementMap = t.displacementMap, this.displacementScale = t.displacementScale, this.displacementBias = t.displacementBias, this;
  }
}
class Bi extends bl {
  constructor(t = -1, e = 1, n = 1, s = -1, r = 0.1, a = 2e3) {
    super(), this.isOrthographicCamera = !0, this.type = "OrthographicCamera", this.zoom = 1, this.view = null, this.left = t, this.right = e, this.top = n, this.bottom = s, this.near = r, this.far = a, this.updateProjectionMatrix();
  }
  copy(t, e) {
    return super.copy(t, e), this.left = t.left, this.right = t.right, this.top = t.top, this.bottom = t.bottom, this.near = t.near, this.far = t.far, this.zoom = t.zoom, this.view = t.view === null ? null : Object.assign({}, t.view), this;
  }
  setViewOffset(t, e, n, s, r, a) {
    this.view === null && (this.view = {
      enabled: !0,
      fullWidth: 1,
      fullHeight: 1,
      offsetX: 0,
      offsetY: 0,
      width: 1,
      height: 1
    }), this.view.enabled = !0, this.view.fullWidth = t, this.view.fullHeight = e, this.view.offsetX = n, this.view.offsetY = s, this.view.width = r, this.view.height = a, this.updateProjectionMatrix();
  }
  clearViewOffset() {
    this.view !== null && (this.view.enabled = !1), this.updateProjectionMatrix();
  }
  updateProjectionMatrix() {
    const t = (this.right - this.left) / (2 * this.zoom), e = (this.top - this.bottom) / (2 * this.zoom), n = (this.right + this.left) / 2, s = (this.top + this.bottom) / 2;
    let r = n - t, a = n + t, o = s + e, l = s - e;
    if (this.view !== null && this.view.enabled) {
      const c = (this.right - this.left) / this.view.fullWidth / this.zoom, u = (this.top - this.bottom) / this.view.fullHeight / this.zoom;
      r += c * this.view.offsetX, a = r + c * this.view.width, o -= u * this.view.offsetY, l = o - u * this.view.height;
    }
    this.projectionMatrix.makeOrthographic(r, a, o, l, this.near, this.far, this.coordinateSystem), this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
  }
  toJSON(t) {
    const e = super.toJSON(t);
    return e.object.zoom = this.zoom, e.object.left = this.left, e.object.right = this.right, e.object.top = this.top, e.object.bottom = this.bottom, e.object.near = this.near, e.object.far = this.far, this.view !== null && (e.object.view = Object.assign({}, this.view)), e;
  }
}
class Ph extends Ce {
  constructor() {
    super(), this.isInstancedBufferGeometry = !0, this.type = "InstancedBufferGeometry", this.instanceCount = 1 / 0;
  }
  copy(t) {
    return super.copy(t), this.instanceCount = t.instanceCount, this;
  }
  toJSON() {
    const t = super.toJSON();
    return t.instanceCount = this.instanceCount, t.isInstancedBufferGeometry = !0, t;
  }
}
class Dh extends ze {
  constructor(t = []) {
    super(), this.isArrayCamera = !0, this.cameras = t;
  }
}
class ha extends yh {
  constructor(t, e, n = 1) {
    super(t, e), this.isInstancedInterleavedBuffer = !0, this.meshPerAttribute = n;
  }
  copy(t) {
    return super.copy(t), this.meshPerAttribute = t.meshPerAttribute, this;
  }
  clone(t) {
    const e = super.clone(t);
    return e.meshPerAttribute = this.meshPerAttribute, e;
  }
  toJSON(t) {
    const e = super.toJSON(t);
    return e.isInstancedInterleavedBuffer = !0, e.meshPerAttribute = this.meshPerAttribute, e;
  }
}
class mo {
  constructor(t = 1, e = 0, n = 0) {
    return this.radius = t, this.phi = e, this.theta = n, this;
  }
  set(t, e, n) {
    return this.radius = t, this.phi = e, this.theta = n, this;
  }
  copy(t) {
    return this.radius = t.radius, this.phi = t.phi, this.theta = t.theta, this;
  }
  // restrict phi to be between EPS and PI-EPS
  makeSafe() {
    return this.phi = Ut(this.phi, 1e-6, Math.PI - 1e-6), this;
  }
  setFromVector3(t) {
    return this.setFromCartesianCoords(t.x, t.y, t.z);
  }
  setFromCartesianCoords(t, e, n) {
    return this.radius = Math.sqrt(t * t + e * e + n * n), this.radius === 0 ? (this.theta = 0, this.phi = 0) : (this.theta = Math.atan2(t, n), this.phi = Math.acos(Ut(e / this.radius, -1, 1))), this;
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
const _o = /* @__PURE__ */ new P(), ms = /* @__PURE__ */ new P();
class Lh {
  constructor(t = new P(), e = new P()) {
    this.start = t, this.end = e;
  }
  set(t, e) {
    return this.start.copy(t), this.end.copy(e), this;
  }
  copy(t) {
    return this.start.copy(t.start), this.end.copy(t.end), this;
  }
  getCenter(t) {
    return t.addVectors(this.start, this.end).multiplyScalar(0.5);
  }
  delta(t) {
    return t.subVectors(this.end, this.start);
  }
  distanceSq() {
    return this.start.distanceToSquared(this.end);
  }
  distance() {
    return this.start.distanceTo(this.end);
  }
  at(t, e) {
    return this.delta(e).multiplyScalar(t).add(this.start);
  }
  closestPointToPointParameter(t, e) {
    _o.subVectors(t, this.start), ms.subVectors(this.end, this.start);
    const n = ms.dot(ms);
    let r = ms.dot(_o) / n;
    return e && (r = Ut(r, 0, 1)), r;
  }
  closestPointToPoint(t, e, n) {
    const s = this.closestPointToPointParameter(t, e);
    return this.delta(n).multiplyScalar(s).add(this.start);
  }
  applyMatrix4(t) {
    return this.start.applyMatrix4(t), this.end.applyMatrix4(t), this;
  }
  equals(t) {
    return t.start.equals(this.start) && t.end.equals(this.end);
  }
  clone() {
    return new this.constructor().copy(this);
  }
}
const go = /* @__PURE__ */ new P();
let _s, cr;
class Uh extends Re {
  // dir is assumed to be normalized
  constructor(t = new P(0, 0, 1), e = new P(0, 0, 0), n = 1, s = 16776960, r = n * 0.2, a = r * 0.2) {
    super(), this.type = "ArrowHelper", _s === void 0 && (_s = new Ce(), _s.setAttribute("position", new se([0, 0, 0, 0, 1, 0], 3)), cr = new Aa(0, 0.5, 1, 5, 1), cr.translate(0, -0.5, 0)), this.position.copy(e), this.line = new wl(_s, new Ea({ color: s, toneMapped: !1 })), this.line.matrixAutoUpdate = !1, this.add(this.line), this.cone = new be(cr, new Fs({ color: s, toneMapped: !1 })), this.cone.matrixAutoUpdate = !1, this.add(this.cone), this.setDirection(t), this.setLength(n, r, a);
  }
  setDirection(t) {
    if (t.y > 0.99999)
      this.quaternion.set(0, 0, 0, 1);
    else if (t.y < -0.99999)
      this.quaternion.set(1, 0, 0, 0);
    else {
      go.set(t.z, 0, -t.x).normalize();
      const e = Math.acos(t.y);
      this.quaternion.setFromAxisAngle(go, e);
    }
  }
  setLength(t, e = t * 0.2, n = e * 0.2) {
    this.line.scale.set(1, Math.max(1e-4, t - e), 1), this.line.updateMatrix(), this.cone.scale.set(n, e, n), this.cone.position.y = t, this.cone.updateMatrix();
  }
  setColor(t) {
    this.line.material.color.set(t), this.cone.material.color.set(t);
  }
  copy(t) {
    return super.copy(t, !1), this.line.copy(t.line), this.cone.copy(t.cone), this;
  }
  dispose() {
    this.line.geometry.dispose(), this.line.material.dispose(), this.cone.geometry.dispose(), this.cone.material.dispose();
  }
}
class Ih extends Ah {
  constructor(t = 1) {
    const e = [
      0,
      0,
      0,
      t,
      0,
      0,
      0,
      0,
      0,
      0,
      t,
      0,
      0,
      0,
      0,
      0,
      0,
      t
    ], n = [
      1,
      0,
      0,
      1,
      0.6,
      0,
      0,
      1,
      0,
      0.6,
      1,
      0,
      0,
      0,
      1,
      0,
      0.6,
      1
    ], s = new Ce();
    s.setAttribute("position", new se(e, 3)), s.setAttribute("color", new se(n, 3));
    const r = new Ea({ vertexColors: !0, toneMapped: !1 });
    super(s, r), this.type = "AxesHelper";
  }
  setColors(t, e, n) {
    const s = new Vt(), r = this.geometry.attributes.color.array;
    return s.set(t), s.toArray(r, 0), s.toArray(r, 3), s.set(e), s.toArray(r, 6), s.toArray(r, 9), s.set(n), s.toArray(r, 12), s.toArray(r, 15), this.geometry.attributes.color.needsUpdate = !0, this;
  }
  dispose() {
    this.geometry.dispose(), this.material.dispose();
  }
}
class Nh extends Qn {
  constructor(t, e = null) {
    super(), this.object = t, this.domElement = e, this.enabled = !0, this.state = -1, this.keys = {}, this.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: null }, this.touches = { ONE: null, TWO: null };
  }
  connect() {
  }
  disconnect() {
  }
  dispose() {
  }
  update() {
  }
}
function vo(i, t, e, n) {
  const s = Fh(n);
  switch (e) {
    // https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexImage2D.xhtml
    case al:
      return i * t;
    case ll:
      return i * t;
    case cl:
      return i * t * 2;
    case hl:
      return i * t / s.components * s.byteLength;
    case ga:
      return i * t / s.components * s.byteLength;
    case ul:
      return i * t * 2 / s.components * s.byteLength;
    case va:
      return i * t * 2 / s.components * s.byteLength;
    case ol:
      return i * t * 3 / s.components * s.byteLength;
    case Qe:
      return i * t * 4 / s.components * s.byteLength;
    case xa:
      return i * t * 4 / s.components * s.byteLength;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_s3tc_srgb/
    case bs:
    case As:
      return Math.floor((i + 3) / 4) * Math.floor((t + 3) / 4) * 8;
    case Ts:
    case ws:
      return Math.floor((i + 3) / 4) * Math.floor((t + 3) / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_pvrtc/
    case Br:
    case Hr:
      return Math.max(i, 16) * Math.max(t, 8) / 4;
    case Or:
    case zr:
      return Math.max(i, 8) * Math.max(t, 8) / 2;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_etc/
    case kr:
    case Vr:
      return Math.floor((i + 3) / 4) * Math.floor((t + 3) / 4) * 8;
    case Gr:
      return Math.floor((i + 3) / 4) * Math.floor((t + 3) / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/WEBGL_compressed_texture_astc/
    case Wr:
      return Math.floor((i + 3) / 4) * Math.floor((t + 3) / 4) * 16;
    case Xr:
      return Math.floor((i + 4) / 5) * Math.floor((t + 3) / 4) * 16;
    case Yr:
      return Math.floor((i + 4) / 5) * Math.floor((t + 4) / 5) * 16;
    case $r:
      return Math.floor((i + 5) / 6) * Math.floor((t + 4) / 5) * 16;
    case qr:
      return Math.floor((i + 5) / 6) * Math.floor((t + 5) / 6) * 16;
    case jr:
      return Math.floor((i + 7) / 8) * Math.floor((t + 4) / 5) * 16;
    case Zr:
      return Math.floor((i + 7) / 8) * Math.floor((t + 5) / 6) * 16;
    case Kr:
      return Math.floor((i + 7) / 8) * Math.floor((t + 7) / 8) * 16;
    case Jr:
      return Math.floor((i + 9) / 10) * Math.floor((t + 4) / 5) * 16;
    case Qr:
      return Math.floor((i + 9) / 10) * Math.floor((t + 5) / 6) * 16;
    case ta:
      return Math.floor((i + 9) / 10) * Math.floor((t + 7) / 8) * 16;
    case ea:
      return Math.floor((i + 9) / 10) * Math.floor((t + 9) / 10) * 16;
    case na:
      return Math.floor((i + 11) / 12) * Math.floor((t + 9) / 10) * 16;
    case ia:
      return Math.floor((i + 11) / 12) * Math.floor((t + 11) / 12) * 16;
    // https://registry.khronos.org/webgl/extensions/EXT_texture_compression_bptc/
    case Rs:
    case sa:
    case ra:
      return Math.ceil(i / 4) * Math.ceil(t / 4) * 16;
    // https://registry.khronos.org/webgl/extensions/EXT_texture_compression_rgtc/
    case dl:
    case aa:
      return Math.ceil(i / 4) * Math.ceil(t / 4) * 8;
    case oa:
    case la:
      return Math.ceil(i / 4) * Math.ceil(t / 4) * 16;
  }
  throw new Error(
    `Unable to determine texture byte length for ${e} format.`
  );
}
function Fh(i) {
  switch (i) {
    case Sn:
    case il:
      return { byteLength: 1, components: 1 };
    case Vi:
    case sl:
    case Xi:
      return { byteLength: 2, components: 1 };
    case ma:
    case _a:
      return { byteLength: 2, components: 4 };
    case jn:
    case pa:
    case gn:
      return { byteLength: 4, components: 1 };
    case rl:
      return { byteLength: 4, components: 3 };
  }
  throw new Error(`Unknown texture type ${i}.`);
}
typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register", { detail: {
  revision: fa
} }));
typeof window < "u" && (window.__THREE__ ? console.warn("WARNING: Multiple instances of Three.js being imported.") : window.__THREE__ = fa);
/**
 * @license
 * Copyright 2010-2024 Three.js Authors
 * SPDX-License-Identifier: MIT
 */
function Cl() {
  let i = null, t = !1, e = null, n = null;
  function s(r, a) {
    e(r, a), n = i.requestAnimationFrame(s);
  }
  return {
    start: function() {
      t !== !0 && e !== null && (n = i.requestAnimationFrame(s), t = !0);
    },
    stop: function() {
      i.cancelAnimationFrame(n), t = !1;
    },
    setAnimationLoop: function(r) {
      e = r;
    },
    setContext: function(r) {
      i = r;
    }
  };
}
function Oh(i) {
  const t = /* @__PURE__ */ new WeakMap();
  function e(o, l) {
    const c = o.array, u = o.usage, d = c.byteLength, f = i.createBuffer();
    i.bindBuffer(l, f), i.bufferData(l, c, u), o.onUploadCallback();
    let m;
    if (c instanceof Float32Array)
      m = i.FLOAT;
    else if (c instanceof Uint16Array)
      o.isFloat16BufferAttribute ? m = i.HALF_FLOAT : m = i.UNSIGNED_SHORT;
    else if (c instanceof Int16Array)
      m = i.SHORT;
    else if (c instanceof Uint32Array)
      m = i.UNSIGNED_INT;
    else if (c instanceof Int32Array)
      m = i.INT;
    else if (c instanceof Int8Array)
      m = i.BYTE;
    else if (c instanceof Uint8Array)
      m = i.UNSIGNED_BYTE;
    else if (c instanceof Uint8ClampedArray)
      m = i.UNSIGNED_BYTE;
    else
      throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: " + c);
    return {
      buffer: f,
      type: m,
      bytesPerElement: c.BYTES_PER_ELEMENT,
      version: o.version,
      size: d
    };
  }
  function n(o, l, c) {
    const u = l.array, d = l.updateRanges;
    if (i.bindBuffer(c, o), d.length === 0)
      i.bufferSubData(c, 0, u);
    else {
      d.sort((m, g) => m.start - g.start);
      let f = 0;
      for (let m = 1; m < d.length; m++) {
        const g = d[f], v = d[m];
        v.start <= g.start + g.count + 1 ? g.count = Math.max(
          g.count,
          v.start + v.count - g.start
        ) : (++f, d[f] = v);
      }
      d.length = f + 1;
      for (let m = 0, g = d.length; m < g; m++) {
        const v = d[m];
        i.bufferSubData(
          c,
          v.start * u.BYTES_PER_ELEMENT,
          u,
          v.start,
          v.count
        );
      }
      l.clearUpdateRanges();
    }
    l.onUploadCallback();
  }
  function s(o) {
    return o.isInterleavedBufferAttribute && (o = o.data), t.get(o);
  }
  function r(o) {
    o.isInterleavedBufferAttribute && (o = o.data);
    const l = t.get(o);
    l && (i.deleteBuffer(l.buffer), t.delete(o));
  }
  function a(o, l) {
    if (o.isInterleavedBufferAttribute && (o = o.data), o.isGLBufferAttribute) {
      const u = t.get(o);
      (!u || u.version < o.version) && t.set(o, {
        buffer: o.buffer,
        type: o.type,
        bytesPerElement: o.elementSize,
        version: o.version
      });
      return;
    }
    const c = t.get(o);
    if (c === void 0)
      t.set(o, e(o, l));
    else if (c.version < o.version) {
      if (c.size !== o.array.byteLength)
        throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");
      n(c.buffer, o, l), c.version = o.version;
    }
  }
  return {
    get: s,
    remove: r,
    update: a
  };
}
var Bh = `#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`, zh = `#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`, Hh = `#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`, kh = `#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`, Vh = `#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`, Gh = `#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`, Wh = `#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`, Xh = `#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`, Yh = `#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec3 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 ).rgb;
	}
#endif`, $h = `#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`, qh = `vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`, jh = `vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`, Zh = `float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`, Kh = `#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`, Jh = `#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`, Qh = `#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`, tu = `#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`, eu = `#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`, nu = `#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`, iu = `#if defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#elif defined( USE_COLOR )
	diffuseColor.rgb *= vColor;
#endif`, su = `#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR )
	varying vec3 vColor;
#endif`, ru = `#if defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec3 vColor;
#endif`, au = `#if defined( USE_COLOR_ALPHA )
	vColor = vec4( 1.0 );
#elif defined( USE_COLOR ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec3( 1.0 );
#endif
#ifdef USE_COLOR
	vColor *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.xyz *= instanceColor.xyz;
#endif
#ifdef USE_BATCHING_COLOR
	vec3 batchingColor = getBatchingColor( getIndirectIndex( gl_DrawID ) );
	vColor.xyz *= batchingColor.xyz;
#endif`, ou = `#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`, lu = `#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`, cu = `vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`, hu = `#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`, uu = `#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`, du = `#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`, fu = `#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`, pu = "gl_FragColor = linearToOutputTexel( gl_FragColor );", mu = `vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`, _u = `#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
	#else
		vec4 envColor = vec4( 0.0 );
	#endif
	#ifdef ENVMAP_BLENDING_MULTIPLY
		outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_MIX )
		outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
	#elif defined( ENVMAP_BLENDING_ADD )
		outgoingLight += envColor.xyz * specularStrength * reflectivity;
	#endif
#endif`, gu = `#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
	
#endif`, vu = `#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`, xu = `#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`, Mu = `#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`, Su = `#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`, yu = `#ifdef USE_FOG
	varying float vFogDepth;
#endif`, Eu = `#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`, bu = `#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`, Au = `#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`, Tu = `#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`, wu = `LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`, Ru = `varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`, Cu = `uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`, Pu = `#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, roughness * roughness) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`, Du = `ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`, Lu = `varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`, Uu = `BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`, Iu = `varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`, Nu = `PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = mix( min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.07, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`, Fu = `struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return saturate(v);
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColor;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
	#endif
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`, Ou = `
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`, Bu = `#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD ) && defined( ENVMAP_TYPE_CUBE_UV )
		iblIrradiance += getIBLIrradiance( geometryNormal );
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`, zu = `#if defined( RE_IndirectDiffuse )
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`, Hu = `#if defined( USE_LOGDEPTHBUF )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`, ku = `#if defined( USE_LOGDEPTHBUF )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`, Vu = `#ifdef USE_LOGDEPTHBUF
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`, Gu = `#ifdef USE_LOGDEPTHBUF
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`, Wu = `#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`, Xu = `#ifdef USE_MAP
	uniform sampler2D map;
#endif`, Yu = `#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`, $u = `#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`, qu = `float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`, ju = `#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`, Zu = `#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`, Ku = `#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`, Ju = `#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`, Qu = `#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`, td = `#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`, ed = `float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`, nd = `#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`, id = `#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`, sd = `#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`, rd = `#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`, ad = `#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`, od = `#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`, ld = `#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`, cd = `#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`, hd = `#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`, ud = `#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`, dd = `vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return depth * ( near - far ) - near;
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	return ( near * far ) / ( ( far - near ) * depth - far );
}`, fd = `#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`, pd = `vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`, md = `#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`, _d = `#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`, gd = `float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`, vd = `#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`, xd = `#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform sampler2D pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {
		return step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );
	}
	vec2 texture2DDistribution( sampler2D shadow, vec2 uv ) {
		return unpackRGBATo2Half( texture2D( shadow, uv ) );
	}
	float VSMShadow (sampler2D shadow, vec2 uv, float compare ){
		float occlusion = 1.0;
		vec2 distribution = texture2DDistribution( shadow, uv );
		float hard_shadow = step( compare , distribution.x );
		if (hard_shadow != 1.0 ) {
			float distance = compare - distribution.x ;
			float variance = max( 0.00000, distribution.y * distribution.y );
			float softness_probability = variance / (variance + distance * distance );			softness_probability = clamp( ( softness_probability - 0.3 ) / ( 0.95 - 0.3 ), 0.0, 1.0 );			occlusion = clamp( max( hard_shadow, softness_probability ), 0.0, 1.0 );
		}
		return occlusion;
	}
	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
		float shadow = 1.0;
		shadowCoord.xyz /= shadowCoord.w;
		shadowCoord.z += shadowBias;
		bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
		bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
		if ( frustumTest ) {
		#if defined( SHADOWMAP_TYPE_PCF )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx0 = - texelSize.x * shadowRadius;
			float dy0 = - texelSize.y * shadowRadius;
			float dx1 = + texelSize.x * shadowRadius;
			float dy1 = + texelSize.y * shadowRadius;
			float dx2 = dx0 / 2.0;
			float dy2 = dy0 / 2.0;
			float dx3 = dx1 / 2.0;
			float dy3 = dy1 / 2.0;
			shadow = (
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy2 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx2, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx3, dy3 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )
			) * ( 1.0 / 17.0 );
		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )
			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
			float dx = texelSize.x;
			float dy = texelSize.y;
			vec2 uv = shadowCoord.xy;
			vec2 f = fract( uv * shadowMapSize + 0.5 );
			uv -= f * texelSize;
			shadow = (
				texture2DCompare( shadowMap, uv, shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( dx, 0.0 ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + vec2( 0.0, dy ), shadowCoord.z ) +
				texture2DCompare( shadowMap, uv + texelSize, shadowCoord.z ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, 0.0 ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 0.0 ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( -dx, dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, dy ), shadowCoord.z ),
					 f.x ) +
				mix( texture2DCompare( shadowMap, uv + vec2( 0.0, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( 0.0, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( texture2DCompare( shadowMap, uv + vec2( dx, -dy ), shadowCoord.z ),
					 texture2DCompare( shadowMap, uv + vec2( dx, 2.0 * dy ), shadowCoord.z ),
					 f.y ) +
				mix( mix( texture2DCompare( shadowMap, uv + vec2( -dx, -dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, -dy ), shadowCoord.z ),
						  f.x ),
					 mix( texture2DCompare( shadowMap, uv + vec2( -dx, 2.0 * dy ), shadowCoord.z ),
						  texture2DCompare( shadowMap, uv + vec2( 2.0 * dx, 2.0 * dy ), shadowCoord.z ),
						  f.x ),
					 f.y )
			) * ( 1.0 / 9.0 );
		#elif defined( SHADOWMAP_TYPE_VSM )
			shadow = VSMShadow( shadowMap, shadowCoord.xy, shadowCoord.z );
		#else
			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );
		#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	vec2 cubeToUV( vec3 v, float texelSizeY ) {
		vec3 absV = abs( v );
		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );
		absV *= scaleToCube;
		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );
		vec2 planar = v.xy;
		float almostATexel = 1.5 * texelSizeY;
		float almostOne = 1.0 - almostATexel;
		if ( absV.z >= almostOne ) {
			if ( v.z > 0.0 )
				planar.x = 4.0 - v.x;
		} else if ( absV.x >= almostOne ) {
			float signX = sign( v.x );
			planar.x = v.z * signX + 2.0 * signX;
		} else if ( absV.y >= almostOne ) {
			float signY = sign( v.y );
			planar.x = v.x + 2.0 * signY + 2.0;
			planar.y = v.z * signY - 2.0;
		}
		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );
	}
	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		
		float lightToPositionLength = length( lightToPosition );
		if ( lightToPositionLength - shadowCameraFar <= 0.0 && lightToPositionLength - shadowCameraNear >= 0.0 ) {
			float dp = ( lightToPositionLength - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );
			#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT ) || defined( SHADOWMAP_TYPE_VSM )
				vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;
				shadow = (
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +
					texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )
				) * ( 1.0 / 9.0 );
			#else
				shadow = texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );
			#endif
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
#endif`, Md = `#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`, Sd = `#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`, yd = `float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`, Ed = `#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`, bd = `#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`, Ad = `#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`, Td = `#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`, wd = `float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`, Rd = `#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`, Cd = `#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`, Pd = `#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`, Dd = `#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseColor, material.specularColor, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`, Ld = `#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
		
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
		
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		
		#else
		
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`, Ud = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`, Id = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`, Nd = `#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`, Fd = `#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;
const Od = `varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`, Bd = `uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, zd = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`, Hd = `#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, kd = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`, Vd = `uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, Gd = `#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`, Wd = `#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	float fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5;
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`, Xd = `#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`, Yd = `#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = packDepthToRGBA( dist );
}`, $d = `varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`, qd = `uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`, jd = `uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`, Zd = `uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`, Kd = `#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`, Jd = `uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, Qd = `#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, tf = `#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, ef = `#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`, nf = `#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, sf = `#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`, rf = `#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <packing>
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( packNormalToRGB( normal ), diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`, af = `#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, of = `#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, lf = `#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`, cf = `#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
		float sheenEnergyComp = 1.0 - 0.157 * max3( material.sheenColor );
		outgoingLight = outgoingLight * sheenEnergyComp + sheenSpecularDirect + sheenSpecularIndirect;
	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, hf = `#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, uf = `#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <packing>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`, df = `uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`, ff = `uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`, pf = `#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`, mf = `uniform vec3 color;
uniform float opacity;
#include <common>
#include <packing>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`, _f = `uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`, gf = `uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`, Lt = {
  alphahash_fragment: Bh,
  alphahash_pars_fragment: zh,
  alphamap_fragment: Hh,
  alphamap_pars_fragment: kh,
  alphatest_fragment: Vh,
  alphatest_pars_fragment: Gh,
  aomap_fragment: Wh,
  aomap_pars_fragment: Xh,
  batching_pars_vertex: Yh,
  batching_vertex: $h,
  begin_vertex: qh,
  beginnormal_vertex: jh,
  bsdfs: Zh,
  iridescence_fragment: Kh,
  bumpmap_pars_fragment: Jh,
  clipping_planes_fragment: Qh,
  clipping_planes_pars_fragment: tu,
  clipping_planes_pars_vertex: eu,
  clipping_planes_vertex: nu,
  color_fragment: iu,
  color_pars_fragment: su,
  color_pars_vertex: ru,
  color_vertex: au,
  common: ou,
  cube_uv_reflection_fragment: lu,
  defaultnormal_vertex: cu,
  displacementmap_pars_vertex: hu,
  displacementmap_vertex: uu,
  emissivemap_fragment: du,
  emissivemap_pars_fragment: fu,
  colorspace_fragment: pu,
  colorspace_pars_fragment: mu,
  envmap_fragment: _u,
  envmap_common_pars_fragment: gu,
  envmap_pars_fragment: vu,
  envmap_pars_vertex: xu,
  envmap_physical_pars_fragment: Pu,
  envmap_vertex: Mu,
  fog_vertex: Su,
  fog_pars_vertex: yu,
  fog_fragment: Eu,
  fog_pars_fragment: bu,
  gradientmap_pars_fragment: Au,
  lightmap_pars_fragment: Tu,
  lights_lambert_fragment: wu,
  lights_lambert_pars_fragment: Ru,
  lights_pars_begin: Cu,
  lights_toon_fragment: Du,
  lights_toon_pars_fragment: Lu,
  lights_phong_fragment: Uu,
  lights_phong_pars_fragment: Iu,
  lights_physical_fragment: Nu,
  lights_physical_pars_fragment: Fu,
  lights_fragment_begin: Ou,
  lights_fragment_maps: Bu,
  lights_fragment_end: zu,
  logdepthbuf_fragment: Hu,
  logdepthbuf_pars_fragment: ku,
  logdepthbuf_pars_vertex: Vu,
  logdepthbuf_vertex: Gu,
  map_fragment: Wu,
  map_pars_fragment: Xu,
  map_particle_fragment: Yu,
  map_particle_pars_fragment: $u,
  metalnessmap_fragment: qu,
  metalnessmap_pars_fragment: ju,
  morphinstance_vertex: Zu,
  morphcolor_vertex: Ku,
  morphnormal_vertex: Ju,
  morphtarget_pars_vertex: Qu,
  morphtarget_vertex: td,
  normal_fragment_begin: ed,
  normal_fragment_maps: nd,
  normal_pars_fragment: id,
  normal_pars_vertex: sd,
  normal_vertex: rd,
  normalmap_pars_fragment: ad,
  clearcoat_normal_fragment_begin: od,
  clearcoat_normal_fragment_maps: ld,
  clearcoat_pars_fragment: cd,
  iridescence_pars_fragment: hd,
  opaque_fragment: ud,
  packing: dd,
  premultiplied_alpha_fragment: fd,
  project_vertex: pd,
  dithering_fragment: md,
  dithering_pars_fragment: _d,
  roughnessmap_fragment: gd,
  roughnessmap_pars_fragment: vd,
  shadowmap_pars_fragment: xd,
  shadowmap_pars_vertex: Md,
  shadowmap_vertex: Sd,
  shadowmask_pars_fragment: yd,
  skinbase_vertex: Ed,
  skinning_pars_vertex: bd,
  skinning_vertex: Ad,
  skinnormal_vertex: Td,
  specularmap_fragment: wd,
  specularmap_pars_fragment: Rd,
  tonemapping_fragment: Cd,
  tonemapping_pars_fragment: Pd,
  transmission_fragment: Dd,
  transmission_pars_fragment: Ld,
  uv_pars_fragment: Ud,
  uv_pars_vertex: Id,
  uv_vertex: Nd,
  worldpos_vertex: Fd,
  background_vert: Od,
  background_frag: Bd,
  backgroundCube_vert: zd,
  backgroundCube_frag: Hd,
  cube_vert: kd,
  cube_frag: Vd,
  depth_vert: Gd,
  depth_frag: Wd,
  distanceRGBA_vert: Xd,
  distanceRGBA_frag: Yd,
  equirect_vert: $d,
  equirect_frag: qd,
  linedashed_vert: jd,
  linedashed_frag: Zd,
  meshbasic_vert: Kd,
  meshbasic_frag: Jd,
  meshlambert_vert: Qd,
  meshlambert_frag: tf,
  meshmatcap_vert: ef,
  meshmatcap_frag: nf,
  meshnormal_vert: sf,
  meshnormal_frag: rf,
  meshphong_vert: af,
  meshphong_frag: of,
  meshphysical_vert: lf,
  meshphysical_frag: cf,
  meshtoon_vert: hf,
  meshtoon_frag: uf,
  points_vert: df,
  points_frag: ff,
  shadow_vert: pf,
  shadow_frag: mf,
  sprite_vert: _f,
  sprite_frag: gf
}, et = {
  common: {
    diffuse: { value: /* @__PURE__ */ new Vt(16777215) },
    opacity: { value: 1 },
    map: { value: null },
    mapTransform: { value: /* @__PURE__ */ new Pt() },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new Pt() },
    alphaTest: { value: 0 }
  },
  specularmap: {
    specularMap: { value: null },
    specularMapTransform: { value: /* @__PURE__ */ new Pt() }
  },
  envmap: {
    envMap: { value: null },
    envMapRotation: { value: /* @__PURE__ */ new Pt() },
    flipEnvMap: { value: -1 },
    reflectivity: { value: 1 },
    // basic, lambert, phong
    ior: { value: 1.5 },
    // physical
    refractionRatio: { value: 0.98 }
    // basic, lambert, phong
  },
  aomap: {
    aoMap: { value: null },
    aoMapIntensity: { value: 1 },
    aoMapTransform: { value: /* @__PURE__ */ new Pt() }
  },
  lightmap: {
    lightMap: { value: null },
    lightMapIntensity: { value: 1 },
    lightMapTransform: { value: /* @__PURE__ */ new Pt() }
  },
  bumpmap: {
    bumpMap: { value: null },
    bumpMapTransform: { value: /* @__PURE__ */ new Pt() },
    bumpScale: { value: 1 }
  },
  normalmap: {
    normalMap: { value: null },
    normalMapTransform: { value: /* @__PURE__ */ new Pt() },
    normalScale: { value: /* @__PURE__ */ new bt(1, 1) }
  },
  displacementmap: {
    displacementMap: { value: null },
    displacementMapTransform: { value: /* @__PURE__ */ new Pt() },
    displacementScale: { value: 1 },
    displacementBias: { value: 0 }
  },
  emissivemap: {
    emissiveMap: { value: null },
    emissiveMapTransform: { value: /* @__PURE__ */ new Pt() }
  },
  metalnessmap: {
    metalnessMap: { value: null },
    metalnessMapTransform: { value: /* @__PURE__ */ new Pt() }
  },
  roughnessmap: {
    roughnessMap: { value: null },
    roughnessMapTransform: { value: /* @__PURE__ */ new Pt() }
  },
  gradientmap: {
    gradientMap: { value: null }
  },
  fog: {
    fogDensity: { value: 25e-5 },
    fogNear: { value: 1 },
    fogFar: { value: 2e3 },
    fogColor: { value: /* @__PURE__ */ new Vt(16777215) }
  },
  lights: {
    ambientLightColor: { value: [] },
    lightProbe: { value: [] },
    directionalLights: { value: [], properties: {
      direction: {},
      color: {}
    } },
    directionalLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {}
    } },
    directionalShadowMap: { value: [] },
    directionalShadowMatrix: { value: [] },
    spotLights: { value: [], properties: {
      color: {},
      position: {},
      direction: {},
      distance: {},
      coneCos: {},
      penumbraCos: {},
      decay: {}
    } },
    spotLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {}
    } },
    spotLightMap: { value: [] },
    spotShadowMap: { value: [] },
    spotLightMatrix: { value: [] },
    pointLights: { value: [], properties: {
      color: {},
      position: {},
      decay: {},
      distance: {}
    } },
    pointLightShadows: { value: [], properties: {
      shadowIntensity: 1,
      shadowBias: {},
      shadowNormalBias: {},
      shadowRadius: {},
      shadowMapSize: {},
      shadowCameraNear: {},
      shadowCameraFar: {}
    } },
    pointShadowMap: { value: [] },
    pointShadowMatrix: { value: [] },
    hemisphereLights: { value: [], properties: {
      direction: {},
      skyColor: {},
      groundColor: {}
    } },
    // TODO (abelnation): RectAreaLight BRDF data needs to be moved from example to main src
    rectAreaLights: { value: [], properties: {
      color: {},
      position: {},
      width: {},
      height: {}
    } },
    ltc_1: { value: null },
    ltc_2: { value: null }
  },
  points: {
    diffuse: { value: /* @__PURE__ */ new Vt(16777215) },
    opacity: { value: 1 },
    size: { value: 1 },
    scale: { value: 1 },
    map: { value: null },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new Pt() },
    alphaTest: { value: 0 },
    uvTransform: { value: /* @__PURE__ */ new Pt() }
  },
  sprite: {
    diffuse: { value: /* @__PURE__ */ new Vt(16777215) },
    opacity: { value: 1 },
    center: { value: /* @__PURE__ */ new bt(0.5, 0.5) },
    rotation: { value: 0 },
    map: { value: null },
    mapTransform: { value: /* @__PURE__ */ new Pt() },
    alphaMap: { value: null },
    alphaMapTransform: { value: /* @__PURE__ */ new Pt() },
    alphaTest: { value: 0 }
  }
}, Le = {
  basic: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.specularmap,
      et.envmap,
      et.aomap,
      et.lightmap,
      et.fog
    ]),
    vertexShader: Lt.meshbasic_vert,
    fragmentShader: Lt.meshbasic_frag
  },
  lambert: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.specularmap,
      et.envmap,
      et.aomap,
      et.lightmap,
      et.emissivemap,
      et.bumpmap,
      et.normalmap,
      et.displacementmap,
      et.fog,
      et.lights,
      {
        emissive: { value: /* @__PURE__ */ new Vt(0) }
      }
    ]),
    vertexShader: Lt.meshlambert_vert,
    fragmentShader: Lt.meshlambert_frag
  },
  phong: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.specularmap,
      et.envmap,
      et.aomap,
      et.lightmap,
      et.emissivemap,
      et.bumpmap,
      et.normalmap,
      et.displacementmap,
      et.fog,
      et.lights,
      {
        emissive: { value: /* @__PURE__ */ new Vt(0) },
        specular: { value: /* @__PURE__ */ new Vt(1118481) },
        shininess: { value: 30 }
      }
    ]),
    vertexShader: Lt.meshphong_vert,
    fragmentShader: Lt.meshphong_frag
  },
  standard: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.envmap,
      et.aomap,
      et.lightmap,
      et.emissivemap,
      et.bumpmap,
      et.normalmap,
      et.displacementmap,
      et.roughnessmap,
      et.metalnessmap,
      et.fog,
      et.lights,
      {
        emissive: { value: /* @__PURE__ */ new Vt(0) },
        roughness: { value: 1 },
        metalness: { value: 0 },
        envMapIntensity: { value: 1 }
      }
    ]),
    vertexShader: Lt.meshphysical_vert,
    fragmentShader: Lt.meshphysical_frag
  },
  toon: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.aomap,
      et.lightmap,
      et.emissivemap,
      et.bumpmap,
      et.normalmap,
      et.displacementmap,
      et.gradientmap,
      et.fog,
      et.lights,
      {
        emissive: { value: /* @__PURE__ */ new Vt(0) }
      }
    ]),
    vertexShader: Lt.meshtoon_vert,
    fragmentShader: Lt.meshtoon_frag
  },
  matcap: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.bumpmap,
      et.normalmap,
      et.displacementmap,
      et.fog,
      {
        matcap: { value: null }
      }
    ]),
    vertexShader: Lt.meshmatcap_vert,
    fragmentShader: Lt.meshmatcap_frag
  },
  points: {
    uniforms: /* @__PURE__ */ we([
      et.points,
      et.fog
    ]),
    vertexShader: Lt.points_vert,
    fragmentShader: Lt.points_frag
  },
  dashed: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.fog,
      {
        scale: { value: 1 },
        dashSize: { value: 1 },
        totalSize: { value: 2 }
      }
    ]),
    vertexShader: Lt.linedashed_vert,
    fragmentShader: Lt.linedashed_frag
  },
  depth: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.displacementmap
    ]),
    vertexShader: Lt.depth_vert,
    fragmentShader: Lt.depth_frag
  },
  normal: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.bumpmap,
      et.normalmap,
      et.displacementmap,
      {
        opacity: { value: 1 }
      }
    ]),
    vertexShader: Lt.meshnormal_vert,
    fragmentShader: Lt.meshnormal_frag
  },
  sprite: {
    uniforms: /* @__PURE__ */ we([
      et.sprite,
      et.fog
    ]),
    vertexShader: Lt.sprite_vert,
    fragmentShader: Lt.sprite_frag
  },
  background: {
    uniforms: {
      uvTransform: { value: /* @__PURE__ */ new Pt() },
      t2D: { value: null },
      backgroundIntensity: { value: 1 }
    },
    vertexShader: Lt.background_vert,
    fragmentShader: Lt.background_frag
  },
  backgroundCube: {
    uniforms: {
      envMap: { value: null },
      flipEnvMap: { value: -1 },
      backgroundBlurriness: { value: 0 },
      backgroundIntensity: { value: 1 },
      backgroundRotation: { value: /* @__PURE__ */ new Pt() }
    },
    vertexShader: Lt.backgroundCube_vert,
    fragmentShader: Lt.backgroundCube_frag
  },
  cube: {
    uniforms: {
      tCube: { value: null },
      tFlip: { value: -1 },
      opacity: { value: 1 }
    },
    vertexShader: Lt.cube_vert,
    fragmentShader: Lt.cube_frag
  },
  equirect: {
    uniforms: {
      tEquirect: { value: null }
    },
    vertexShader: Lt.equirect_vert,
    fragmentShader: Lt.equirect_frag
  },
  distanceRGBA: {
    uniforms: /* @__PURE__ */ we([
      et.common,
      et.displacementmap,
      {
        referencePosition: { value: /* @__PURE__ */ new P() },
        nearDistance: { value: 1 },
        farDistance: { value: 1e3 }
      }
    ]),
    vertexShader: Lt.distanceRGBA_vert,
    fragmentShader: Lt.distanceRGBA_frag
  },
  shadow: {
    uniforms: /* @__PURE__ */ we([
      et.lights,
      et.fog,
      {
        color: { value: /* @__PURE__ */ new Vt(0) },
        opacity: { value: 1 }
      }
    ]),
    vertexShader: Lt.shadow_vert,
    fragmentShader: Lt.shadow_frag
  }
};
Le.physical = {
  uniforms: /* @__PURE__ */ we([
    Le.standard.uniforms,
    {
      clearcoat: { value: 0 },
      clearcoatMap: { value: null },
      clearcoatMapTransform: { value: /* @__PURE__ */ new Pt() },
      clearcoatNormalMap: { value: null },
      clearcoatNormalMapTransform: { value: /* @__PURE__ */ new Pt() },
      clearcoatNormalScale: { value: /* @__PURE__ */ new bt(1, 1) },
      clearcoatRoughness: { value: 0 },
      clearcoatRoughnessMap: { value: null },
      clearcoatRoughnessMapTransform: { value: /* @__PURE__ */ new Pt() },
      dispersion: { value: 0 },
      iridescence: { value: 0 },
      iridescenceMap: { value: null },
      iridescenceMapTransform: { value: /* @__PURE__ */ new Pt() },
      iridescenceIOR: { value: 1.3 },
      iridescenceThicknessMinimum: { value: 100 },
      iridescenceThicknessMaximum: { value: 400 },
      iridescenceThicknessMap: { value: null },
      iridescenceThicknessMapTransform: { value: /* @__PURE__ */ new Pt() },
      sheen: { value: 0 },
      sheenColor: { value: /* @__PURE__ */ new Vt(0) },
      sheenColorMap: { value: null },
      sheenColorMapTransform: { value: /* @__PURE__ */ new Pt() },
      sheenRoughness: { value: 1 },
      sheenRoughnessMap: { value: null },
      sheenRoughnessMapTransform: { value: /* @__PURE__ */ new Pt() },
      transmission: { value: 0 },
      transmissionMap: { value: null },
      transmissionMapTransform: { value: /* @__PURE__ */ new Pt() },
      transmissionSamplerSize: { value: /* @__PURE__ */ new bt() },
      transmissionSamplerMap: { value: null },
      thickness: { value: 0 },
      thicknessMap: { value: null },
      thicknessMapTransform: { value: /* @__PURE__ */ new Pt() },
      attenuationDistance: { value: 0 },
      attenuationColor: { value: /* @__PURE__ */ new Vt(0) },
      specularColor: { value: /* @__PURE__ */ new Vt(1, 1, 1) },
      specularColorMap: { value: null },
      specularColorMapTransform: { value: /* @__PURE__ */ new Pt() },
      specularIntensity: { value: 1 },
      specularIntensityMap: { value: null },
      specularIntensityMapTransform: { value: /* @__PURE__ */ new Pt() },
      anisotropyVector: { value: /* @__PURE__ */ new bt() },
      anisotropyMap: { value: null },
      anisotropyMapTransform: { value: /* @__PURE__ */ new Pt() }
    }
  ]),
  vertexShader: Lt.meshphysical_vert,
  fragmentShader: Lt.meshphysical_frag
};
const gs = { r: 0, b: 0, g: 0 }, kn = /* @__PURE__ */ new yn(), vf = /* @__PURE__ */ new ne();
function xf(i, t, e, n, s, r, a) {
  const o = new Vt(0);
  let l = r === !0 ? 0 : 1, c, u, d = null, f = 0, m = null;
  function g(b) {
    let S = b.isScene === !0 ? b.background : null;
    return S && S.isTexture && (S = (b.backgroundBlurriness > 0 ? e : t).get(S)), S;
  }
  function v(b) {
    let S = !1;
    const L = g(b);
    L === null ? h(o, l) : L && L.isColor && (h(L, 1), S = !0);
    const T = i.xr.getEnvironmentBlendMode();
    T === "additive" ? n.buffers.color.setClear(0, 0, 0, 1, a) : T === "alpha-blend" && n.buffers.color.setClear(0, 0, 0, 0, a), (i.autoClear || S) && (n.buffers.depth.setTest(!0), n.buffers.depth.setMask(!0), n.buffers.color.setMask(!0), i.clear(i.autoClearColor, i.autoClearDepth, i.autoClearStencil));
  }
  function p(b, S) {
    const L = g(S);
    L && (L.isCubeTexture || L.mapping === Ns) ? (u === void 0 && (u = new be(
      new Yi(1, 1, 1),
      new En({
        name: "BackgroundCubeMaterial",
        uniforms: Ci(Le.backgroundCube.uniforms),
        vertexShader: Le.backgroundCube.vertexShader,
        fragmentShader: Le.backgroundCube.fragmentShader,
        side: Ue,
        depthTest: !1,
        depthWrite: !1,
        fog: !1
      })
    ), u.geometry.deleteAttribute("normal"), u.geometry.deleteAttribute("uv"), u.onBeforeRender = function(T, R, I) {
      this.matrixWorld.copyPosition(I.matrixWorld);
    }, Object.defineProperty(u.material, "envMap", {
      get: function() {
        return this.uniforms.envMap.value;
      }
    }), s.update(u)), kn.copy(S.backgroundRotation), kn.x *= -1, kn.y *= -1, kn.z *= -1, L.isCubeTexture && L.isRenderTargetTexture === !1 && (kn.y *= -1, kn.z *= -1), u.material.uniforms.envMap.value = L, u.material.uniforms.flipEnvMap.value = L.isCubeTexture && L.isRenderTargetTexture === !1 ? -1 : 1, u.material.uniforms.backgroundBlurriness.value = S.backgroundBlurriness, u.material.uniforms.backgroundIntensity.value = S.backgroundIntensity, u.material.uniforms.backgroundRotation.value.setFromMatrix4(vf.makeRotationFromEuler(kn)), u.material.toneMapped = Wt.getTransfer(L.colorSpace) !== jt, (d !== L || f !== L.version || m !== i.toneMapping) && (u.material.needsUpdate = !0, d = L, f = L.version, m = i.toneMapping), u.layers.enableAll(), b.unshift(u, u.geometry, u.material, 0, 0, null)) : L && L.isTexture && (c === void 0 && (c = new be(
      new Os(2, 2),
      new En({
        name: "BackgroundMaterial",
        uniforms: Ci(Le.background.uniforms),
        vertexShader: Le.background.vertexShader,
        fragmentShader: Le.background.fragmentShader,
        side: In,
        depthTest: !1,
        depthWrite: !1,
        fog: !1
      })
    ), c.geometry.deleteAttribute("normal"), Object.defineProperty(c.material, "map", {
      get: function() {
        return this.uniforms.t2D.value;
      }
    }), s.update(c)), c.material.uniforms.t2D.value = L, c.material.uniforms.backgroundIntensity.value = S.backgroundIntensity, c.material.toneMapped = Wt.getTransfer(L.colorSpace) !== jt, L.matrixAutoUpdate === !0 && L.updateMatrix(), c.material.uniforms.uvTransform.value.copy(L.matrix), (d !== L || f !== L.version || m !== i.toneMapping) && (c.material.needsUpdate = !0, d = L, f = L.version, m = i.toneMapping), c.layers.enableAll(), b.unshift(c, c.geometry, c.material, 0, 0, null));
  }
  function h(b, S) {
    b.getRGB(gs, El(i)), n.buffers.color.setClear(gs.r, gs.g, gs.b, S, a);
  }
  function E() {
    u !== void 0 && (u.geometry.dispose(), u.material.dispose()), c !== void 0 && (c.geometry.dispose(), c.material.dispose());
  }
  return {
    getClearColor: function() {
      return o;
    },
    setClearColor: function(b, S = 1) {
      o.set(b), l = S, h(o, l);
    },
    getClearAlpha: function() {
      return l;
    },
    setClearAlpha: function(b) {
      l = b, h(o, l);
    },
    render: v,
    addToRenderList: p,
    dispose: E
  };
}
function Mf(i, t) {
  const e = i.getParameter(i.MAX_VERTEX_ATTRIBS), n = {}, s = f(null);
  let r = s, a = !1;
  function o(M, C, H, z, G) {
    let j = !1;
    const W = d(z, H, C);
    r !== W && (r = W, c(r.object)), j = m(M, z, H, G), j && g(M, z, H, G), G !== null && t.update(G, i.ELEMENT_ARRAY_BUFFER), (j || a) && (a = !1, S(M, C, H, z), G !== null && i.bindBuffer(i.ELEMENT_ARRAY_BUFFER, t.get(G).buffer));
  }
  function l() {
    return i.createVertexArray();
  }
  function c(M) {
    return i.bindVertexArray(M);
  }
  function u(M) {
    return i.deleteVertexArray(M);
  }
  function d(M, C, H) {
    const z = H.wireframe === !0;
    let G = n[M.id];
    G === void 0 && (G = {}, n[M.id] = G);
    let j = G[C.id];
    j === void 0 && (j = {}, G[C.id] = j);
    let W = j[z];
    return W === void 0 && (W = f(l()), j[z] = W), W;
  }
  function f(M) {
    const C = [], H = [], z = [];
    for (let G = 0; G < e; G++)
      C[G] = 0, H[G] = 0, z[G] = 0;
    return {
      // for backward compatibility on non-VAO support browser
      geometry: null,
      program: null,
      wireframe: !1,
      newAttributes: C,
      enabledAttributes: H,
      attributeDivisors: z,
      object: M,
      attributes: {},
      index: null
    };
  }
  function m(M, C, H, z) {
    const G = r.attributes, j = C.attributes;
    let W = 0;
    const Q = H.getAttributes();
    for (const V in Q)
      if (Q[V].location >= 0) {
        const ht = G[V];
        let gt = j[V];
        if (gt === void 0 && (V === "instanceMatrix" && M.instanceMatrix && (gt = M.instanceMatrix), V === "instanceColor" && M.instanceColor && (gt = M.instanceColor)), ht === void 0 || ht.attribute !== gt || gt && ht.data !== gt.data) return !0;
        W++;
      }
    return r.attributesNum !== W || r.index !== z;
  }
  function g(M, C, H, z) {
    const G = {}, j = C.attributes;
    let W = 0;
    const Q = H.getAttributes();
    for (const V in Q)
      if (Q[V].location >= 0) {
        let ht = j[V];
        ht === void 0 && (V === "instanceMatrix" && M.instanceMatrix && (ht = M.instanceMatrix), V === "instanceColor" && M.instanceColor && (ht = M.instanceColor));
        const gt = {};
        gt.attribute = ht, ht && ht.data && (gt.data = ht.data), G[V] = gt, W++;
      }
    r.attributes = G, r.attributesNum = W, r.index = z;
  }
  function v() {
    const M = r.newAttributes;
    for (let C = 0, H = M.length; C < H; C++)
      M[C] = 0;
  }
  function p(M) {
    h(M, 0);
  }
  function h(M, C) {
    const H = r.newAttributes, z = r.enabledAttributes, G = r.attributeDivisors;
    H[M] = 1, z[M] === 0 && (i.enableVertexAttribArray(M), z[M] = 1), G[M] !== C && (i.vertexAttribDivisor(M, C), G[M] = C);
  }
  function E() {
    const M = r.newAttributes, C = r.enabledAttributes;
    for (let H = 0, z = C.length; H < z; H++)
      C[H] !== M[H] && (i.disableVertexAttribArray(H), C[H] = 0);
  }
  function b(M, C, H, z, G, j, W) {
    W === !0 ? i.vertexAttribIPointer(M, C, H, G, j) : i.vertexAttribPointer(M, C, H, z, G, j);
  }
  function S(M, C, H, z) {
    v();
    const G = z.attributes, j = H.getAttributes(), W = C.defaultAttributeValues;
    for (const Q in j) {
      const V = j[Q];
      if (V.location >= 0) {
        let st = G[Q];
        if (st === void 0 && (Q === "instanceMatrix" && M.instanceMatrix && (st = M.instanceMatrix), Q === "instanceColor" && M.instanceColor && (st = M.instanceColor)), st !== void 0) {
          const ht = st.normalized, gt = st.itemSize, It = t.get(st);
          if (It === void 0) continue;
          const Kt = It.buffer, Y = It.type, tt = It.bytesPerElement, mt = Y === i.INT || Y === i.UNSIGNED_INT || st.gpuType === pa;
          if (st.isInterleavedBufferAttribute) {
            const rt = st.data, Et = rt.stride, Rt = st.offset;
            if (rt.isInstancedInterleavedBuffer) {
              for (let Nt = 0; Nt < V.locationSize; Nt++)
                h(V.location + Nt, rt.meshPerAttribute);
              M.isInstancedMesh !== !0 && z._maxInstanceCount === void 0 && (z._maxInstanceCount = rt.meshPerAttribute * rt.count);
            } else
              for (let Nt = 0; Nt < V.locationSize; Nt++)
                p(V.location + Nt);
            i.bindBuffer(i.ARRAY_BUFFER, Kt);
            for (let Nt = 0; Nt < V.locationSize; Nt++)
              b(
                V.location + Nt,
                gt / V.locationSize,
                Y,
                ht,
                Et * tt,
                (Rt + gt / V.locationSize * Nt) * tt,
                mt
              );
          } else {
            if (st.isInstancedBufferAttribute) {
              for (let rt = 0; rt < V.locationSize; rt++)
                h(V.location + rt, st.meshPerAttribute);
              M.isInstancedMesh !== !0 && z._maxInstanceCount === void 0 && (z._maxInstanceCount = st.meshPerAttribute * st.count);
            } else
              for (let rt = 0; rt < V.locationSize; rt++)
                p(V.location + rt);
            i.bindBuffer(i.ARRAY_BUFFER, Kt);
            for (let rt = 0; rt < V.locationSize; rt++)
              b(
                V.location + rt,
                gt / V.locationSize,
                Y,
                ht,
                gt * tt,
                gt / V.locationSize * rt * tt,
                mt
              );
          }
        } else if (W !== void 0) {
          const ht = W[Q];
          if (ht !== void 0)
            switch (ht.length) {
              case 2:
                i.vertexAttrib2fv(V.location, ht);
                break;
              case 3:
                i.vertexAttrib3fv(V.location, ht);
                break;
              case 4:
                i.vertexAttrib4fv(V.location, ht);
                break;
              default:
                i.vertexAttrib1fv(V.location, ht);
            }
        }
      }
    }
    E();
  }
  function L() {
    I();
    for (const M in n) {
      const C = n[M];
      for (const H in C) {
        const z = C[H];
        for (const G in z)
          u(z[G].object), delete z[G];
        delete C[H];
      }
      delete n[M];
    }
  }
  function T(M) {
    if (n[M.id] === void 0) return;
    const C = n[M.id];
    for (const H in C) {
      const z = C[H];
      for (const G in z)
        u(z[G].object), delete z[G];
      delete C[H];
    }
    delete n[M.id];
  }
  function R(M) {
    for (const C in n) {
      const H = n[C];
      if (H[M.id] === void 0) continue;
      const z = H[M.id];
      for (const G in z)
        u(z[G].object), delete z[G];
      delete H[M.id];
    }
  }
  function I() {
    y(), a = !0, r !== s && (r = s, c(r.object));
  }
  function y() {
    s.geometry = null, s.program = null, s.wireframe = !1;
  }
  return {
    setup: o,
    reset: I,
    resetDefaultState: y,
    dispose: L,
    releaseStatesOfGeometry: T,
    releaseStatesOfProgram: R,
    initAttributes: v,
    enableAttribute: p,
    disableUnusedAttributes: E
  };
}
function Sf(i, t, e) {
  let n;
  function s(c) {
    n = c;
  }
  function r(c, u) {
    i.drawArrays(n, c, u), e.update(u, n, 1);
  }
  function a(c, u, d) {
    d !== 0 && (i.drawArraysInstanced(n, c, u, d), e.update(u, n, d));
  }
  function o(c, u, d) {
    if (d === 0) return;
    t.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n, c, 0, u, 0, d);
    let m = 0;
    for (let g = 0; g < d; g++)
      m += u[g];
    e.update(m, n, 1);
  }
  function l(c, u, d, f) {
    if (d === 0) return;
    const m = t.get("WEBGL_multi_draw");
    if (m === null)
      for (let g = 0; g < c.length; g++)
        a(c[g], u[g], f[g]);
    else {
      m.multiDrawArraysInstancedWEBGL(n, c, 0, u, 0, f, 0, d);
      let g = 0;
      for (let v = 0; v < d; v++)
        g += u[v] * f[v];
      e.update(g, n, 1);
    }
  }
  this.setMode = s, this.render = r, this.renderInstances = a, this.renderMultiDraw = o, this.renderMultiDrawInstances = l;
}
function yf(i, t, e, n) {
  let s;
  function r() {
    if (s !== void 0) return s;
    if (t.has("EXT_texture_filter_anisotropic") === !0) {
      const R = t.get("EXT_texture_filter_anisotropic");
      s = i.getParameter(R.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
    } else
      s = 0;
    return s;
  }
  function a(R) {
    return !(R !== Qe && n.convert(R) !== i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT));
  }
  function o(R) {
    const I = R === Xi && (t.has("EXT_color_buffer_half_float") || t.has("EXT_color_buffer_float"));
    return !(R !== Sn && n.convert(R) !== i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE) && // Edge and Chrome Mac < 52 (#9513)
    R !== gn && !I);
  }
  function l(R) {
    if (R === "highp") {
      if (i.getShaderPrecisionFormat(i.VERTEX_SHADER, i.HIGH_FLOAT).precision > 0 && i.getShaderPrecisionFormat(i.FRAGMENT_SHADER, i.HIGH_FLOAT).precision > 0)
        return "highp";
      R = "mediump";
    }
    return R === "mediump" && i.getShaderPrecisionFormat(i.VERTEX_SHADER, i.MEDIUM_FLOAT).precision > 0 && i.getShaderPrecisionFormat(i.FRAGMENT_SHADER, i.MEDIUM_FLOAT).precision > 0 ? "mediump" : "lowp";
  }
  let c = e.precision !== void 0 ? e.precision : "highp";
  const u = l(c);
  u !== c && (console.warn("THREE.WebGLRenderer:", c, "not supported, using", u, "instead."), c = u);
  const d = e.logarithmicDepthBuffer === !0, f = e.reverseDepthBuffer === !0 && t.has("EXT_clip_control"), m = i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS), g = i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS), v = i.getParameter(i.MAX_TEXTURE_SIZE), p = i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE), h = i.getParameter(i.MAX_VERTEX_ATTRIBS), E = i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS), b = i.getParameter(i.MAX_VARYING_VECTORS), S = i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS), L = g > 0, T = i.getParameter(i.MAX_SAMPLES);
  return {
    isWebGL2: !0,
    // keeping this for backwards compatibility
    getMaxAnisotropy: r,
    getMaxPrecision: l,
    textureFormatReadable: a,
    textureTypeReadable: o,
    precision: c,
    logarithmicDepthBuffer: d,
    reverseDepthBuffer: f,
    maxTextures: m,
    maxVertexTextures: g,
    maxTextureSize: v,
    maxCubemapSize: p,
    maxAttributes: h,
    maxVertexUniforms: E,
    maxVaryings: b,
    maxFragmentUniforms: S,
    vertexTextures: L,
    maxSamples: T
  };
}
function Ef(i) {
  const t = this;
  let e = null, n = 0, s = !1, r = !1;
  const a = new _n(), o = new Pt(), l = { value: null, needsUpdate: !1 };
  this.uniform = l, this.numPlanes = 0, this.numIntersection = 0, this.init = function(d, f) {
    const m = d.length !== 0 || f || // enable state of previous frame - the clipping code has to
    // run another frame in order to reset the state:
    n !== 0 || s;
    return s = f, n = d.length, m;
  }, this.beginShadows = function() {
    r = !0, u(null);
  }, this.endShadows = function() {
    r = !1;
  }, this.setGlobalState = function(d, f) {
    e = u(d, f, 0);
  }, this.setState = function(d, f, m) {
    const g = d.clippingPlanes, v = d.clipIntersection, p = d.clipShadows, h = i.get(d);
    if (!s || g === null || g.length === 0 || r && !p)
      r ? u(null) : c();
    else {
      const E = r ? 0 : n, b = E * 4;
      let S = h.clippingState || null;
      l.value = S, S = u(g, f, b, m);
      for (let L = 0; L !== b; ++L)
        S[L] = e[L];
      h.clippingState = S, this.numIntersection = v ? this.numPlanes : 0, this.numPlanes += E;
    }
  };
  function c() {
    l.value !== e && (l.value = e, l.needsUpdate = n > 0), t.numPlanes = n, t.numIntersection = 0;
  }
  function u(d, f, m, g) {
    const v = d !== null ? d.length : 0;
    let p = null;
    if (v !== 0) {
      if (p = l.value, g !== !0 || p === null) {
        const h = m + v * 4, E = f.matrixWorldInverse;
        o.getNormalMatrix(E), (p === null || p.length < h) && (p = new Float32Array(h));
        for (let b = 0, S = m; b !== v; ++b, S += 4)
          a.copy(d[b]).applyMatrix4(E, o), a.normal.toArray(p, S), p[S + 3] = a.constant;
      }
      l.value = p, l.needsUpdate = !0;
    }
    return t.numPlanes = v, t.numIntersection = 0, p;
  }
}
function bf(i) {
  let t = /* @__PURE__ */ new WeakMap();
  function e(a, o) {
    return o === Ur ? a.mapping = bi : o === Ir && (a.mapping = Ai), a;
  }
  function n(a) {
    if (a && a.isTexture) {
      const o = a.mapping;
      if (o === Ur || o === Ir)
        if (t.has(a)) {
          const l = t.get(a).texture;
          return e(l, a.mapping);
        } else {
          const l = a.image;
          if (l && l.height > 0) {
            const c = new Mh(l.height);
            return c.fromEquirectangularTexture(i, a), t.set(a, c), a.addEventListener("dispose", s), e(c.texture, a.mapping);
          } else
            return null;
        }
    }
    return a;
  }
  function s(a) {
    const o = a.target;
    o.removeEventListener("dispose", s);
    const l = t.get(o);
    l !== void 0 && (t.delete(o), l.dispose());
  }
  function r() {
    t = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: n,
    dispose: r
  };
}
const vi = 4, xo = [0.125, 0.215, 0.35, 0.446, 0.526, 0.582], Xn = 20, hr = /* @__PURE__ */ new Bi(), Mo = /* @__PURE__ */ new Vt();
let ur = null, dr = 0, fr = 0, pr = !1;
const Gn = (1 + Math.sqrt(5)) / 2, mi = 1 / Gn, So = [
  /* @__PURE__ */ new P(-Gn, mi, 0),
  /* @__PURE__ */ new P(Gn, mi, 0),
  /* @__PURE__ */ new P(-mi, 0, Gn),
  /* @__PURE__ */ new P(mi, 0, Gn),
  /* @__PURE__ */ new P(0, Gn, -mi),
  /* @__PURE__ */ new P(0, Gn, mi),
  /* @__PURE__ */ new P(-1, 1, -1),
  /* @__PURE__ */ new P(1, 1, -1),
  /* @__PURE__ */ new P(-1, 1, 1),
  /* @__PURE__ */ new P(1, 1, 1)
];
class yo {
  constructor(t) {
    this._renderer = t, this._pingPongRenderTarget = null, this._lodMax = 0, this._cubeSize = 0, this._lodPlanes = [], this._sizeLods = [], this._sigmas = [], this._blurMaterial = null, this._cubemapMaterial = null, this._equirectMaterial = null, this._compileMaterial(this._blurMaterial);
  }
  /**
   * Generates a PMREM from a supplied Scene, which can be faster than using an
   * image if networking bandwidth is low. Optional sigma specifies a blur radius
   * in radians to be applied to the scene before PMREM generation. Optional near
   * and far planes ensure the scene is rendered in its entirety (the cubeCamera
   * is placed at the origin).
   */
  fromScene(t, e = 0, n = 0.1, s = 100) {
    ur = this._renderer.getRenderTarget(), dr = this._renderer.getActiveCubeFace(), fr = this._renderer.getActiveMipmapLevel(), pr = this._renderer.xr.enabled, this._renderer.xr.enabled = !1, this._setSize(256);
    const r = this._allocateTargets();
    return r.depthBuffer = !0, this._sceneToCubeUV(t, n, s, r), e > 0 && this._blur(r, 0, 0, e), this._applyPMREM(r), this._cleanup(r), r;
  }
  /**
   * Generates a PMREM from an equirectangular texture, which can be either LDR
   * or HDR. The ideal input image size is 1k (1024 x 512),
   * as this matches best with the 256 x 256 cubemap output.
   * The smallest supported equirectangular image size is 64 x 32.
   */
  fromEquirectangular(t, e = null) {
    return this._fromTexture(t, e);
  }
  /**
   * Generates a PMREM from an cubemap texture, which can be either LDR
   * or HDR. The ideal input cube size is 256 x 256,
   * as this matches best with the 256 x 256 cubemap output.
   * The smallest supported cube size is 16 x 16.
   */
  fromCubemap(t, e = null) {
    return this._fromTexture(t, e);
  }
  /**
   * Pre-compiles the cubemap shader. You can get faster start-up by invoking this method during
   * your texture's network fetch for increased concurrency.
   */
  compileCubemapShader() {
    this._cubemapMaterial === null && (this._cubemapMaterial = Ao(), this._compileMaterial(this._cubemapMaterial));
  }
  /**
   * Pre-compiles the equirectangular shader. You can get faster start-up by invoking this method during
   * your texture's network fetch for increased concurrency.
   */
  compileEquirectangularShader() {
    this._equirectMaterial === null && (this._equirectMaterial = bo(), this._compileMaterial(this._equirectMaterial));
  }
  /**
   * Disposes of the PMREMGenerator's internal memory. Note that PMREMGenerator is a static class,
   * so you should not need more than one PMREMGenerator object. If you do, calling dispose() on
   * one of them will cause any others to also become unusable.
   */
  dispose() {
    this._dispose(), this._cubemapMaterial !== null && this._cubemapMaterial.dispose(), this._equirectMaterial !== null && this._equirectMaterial.dispose();
  }
  // private interface
  _setSize(t) {
    this._lodMax = Math.floor(Math.log2(t)), this._cubeSize = Math.pow(2, this._lodMax);
  }
  _dispose() {
    this._blurMaterial !== null && this._blurMaterial.dispose(), this._pingPongRenderTarget !== null && this._pingPongRenderTarget.dispose();
    for (let t = 0; t < this._lodPlanes.length; t++)
      this._lodPlanes[t].dispose();
  }
  _cleanup(t) {
    this._renderer.setRenderTarget(ur, dr, fr), this._renderer.xr.enabled = pr, t.scissorTest = !1, vs(t, 0, 0, t.width, t.height);
  }
  _fromTexture(t, e) {
    t.mapping === bi || t.mapping === Ai ? this._setSize(t.image.length === 0 ? 16 : t.image[0].width || t.image[0].image.width) : this._setSize(t.image.width / 4), ur = this._renderer.getRenderTarget(), dr = this._renderer.getActiveCubeFace(), fr = this._renderer.getActiveMipmapLevel(), pr = this._renderer.xr.enabled, this._renderer.xr.enabled = !1;
    const n = e || this._allocateTargets();
    return this._textureToCubeUV(t, n), this._applyPMREM(n), this._cleanup(n), n;
  }
  _allocateTargets() {
    const t = 3 * Math.max(this._cubeSize, 112), e = 4 * this._cubeSize, n = {
      magFilter: an,
      minFilter: an,
      generateMipmaps: !1,
      type: Xi,
      format: Qe,
      colorSpace: Ri,
      depthBuffer: !1
    }, s = Eo(t, e, n);
    if (this._pingPongRenderTarget === null || this._pingPongRenderTarget.width !== t || this._pingPongRenderTarget.height !== e) {
      this._pingPongRenderTarget !== null && this._dispose(), this._pingPongRenderTarget = Eo(t, e, n);
      const { _lodMax: r } = this;
      ({ sizeLods: this._sizeLods, lodPlanes: this._lodPlanes, sigmas: this._sigmas } = Af(r)), this._blurMaterial = Tf(r, t, e);
    }
    return s;
  }
  _compileMaterial(t) {
    const e = new be(this._lodPlanes[0], t);
    this._renderer.compile(e, hr);
  }
  _sceneToCubeUV(t, e, n, s) {
    const o = new ze(90, 1, e, n), l = [1, -1, 1, 1, 1, 1], c = [1, 1, 1, -1, -1, -1], u = this._renderer, d = u.autoClear, f = u.toneMapping;
    u.getClearColor(Mo), u.toneMapping = Un, u.autoClear = !1;
    const m = new Fs({
      name: "PMREM.Background",
      side: Ue,
      depthWrite: !1,
      depthTest: !1
    }), g = new be(new Yi(), m);
    let v = !1;
    const p = t.background;
    p ? p.isColor && (m.color.copy(p), t.background = null, v = !0) : (m.color.copy(Mo), v = !0);
    for (let h = 0; h < 6; h++) {
      const E = h % 3;
      E === 0 ? (o.up.set(0, l[h], 0), o.lookAt(c[h], 0, 0)) : E === 1 ? (o.up.set(0, 0, l[h]), o.lookAt(0, c[h], 0)) : (o.up.set(0, l[h], 0), o.lookAt(0, 0, c[h]));
      const b = this._cubeSize;
      vs(s, E * b, h > 2 ? b : 0, b, b), u.setRenderTarget(s), v && u.render(g, o), u.render(t, o);
    }
    g.geometry.dispose(), g.material.dispose(), u.toneMapping = f, u.autoClear = d, t.background = p;
  }
  _textureToCubeUV(t, e) {
    const n = this._renderer, s = t.mapping === bi || t.mapping === Ai;
    s ? (this._cubemapMaterial === null && (this._cubemapMaterial = Ao()), this._cubemapMaterial.uniforms.flipEnvMap.value = t.isRenderTargetTexture === !1 ? -1 : 1) : this._equirectMaterial === null && (this._equirectMaterial = bo());
    const r = s ? this._cubemapMaterial : this._equirectMaterial, a = new be(this._lodPlanes[0], r), o = r.uniforms;
    o.envMap.value = t;
    const l = this._cubeSize;
    vs(e, 0, 0, 3 * l, 2 * l), n.setRenderTarget(e), n.render(a, hr);
  }
  _applyPMREM(t) {
    const e = this._renderer, n = e.autoClear;
    e.autoClear = !1;
    const s = this._lodPlanes.length;
    for (let r = 1; r < s; r++) {
      const a = Math.sqrt(this._sigmas[r] * this._sigmas[r] - this._sigmas[r - 1] * this._sigmas[r - 1]), o = So[(s - r - 1) % So.length];
      this._blur(t, r - 1, r, a, o);
    }
    e.autoClear = n;
  }
  /**
   * This is a two-pass Gaussian blur for a cubemap. Normally this is done
   * vertically and horizontally, but this breaks down on a cube. Here we apply
   * the blur latitudinally (around the poles), and then longitudinally (towards
   * the poles) to approximate the orthogonally-separable blur. It is least
   * accurate at the poles, but still does a decent job.
   */
  _blur(t, e, n, s, r) {
    const a = this._pingPongRenderTarget;
    this._halfBlur(
      t,
      a,
      e,
      n,
      s,
      "latitudinal",
      r
    ), this._halfBlur(
      a,
      t,
      n,
      n,
      s,
      "longitudinal",
      r
    );
  }
  _halfBlur(t, e, n, s, r, a, o) {
    const l = this._renderer, c = this._blurMaterial;
    a !== "latitudinal" && a !== "longitudinal" && console.error(
      "blur direction must be either latitudinal or longitudinal!"
    );
    const u = 3, d = new be(this._lodPlanes[s], c), f = c.uniforms, m = this._sizeLods[n] - 1, g = isFinite(r) ? Math.PI / (2 * m) : 2 * Math.PI / (2 * Xn - 1), v = r / g, p = isFinite(r) ? 1 + Math.floor(u * v) : Xn;
    p > Xn && console.warn(`sigmaRadians, ${r}, is too large and will clip, as it requested ${p} samples when the maximum is set to ${Xn}`);
    const h = [];
    let E = 0;
    for (let R = 0; R < Xn; ++R) {
      const I = R / v, y = Math.exp(-I * I / 2);
      h.push(y), R === 0 ? E += y : R < p && (E += 2 * y);
    }
    for (let R = 0; R < h.length; R++)
      h[R] = h[R] / E;
    f.envMap.value = t.texture, f.samples.value = p, f.weights.value = h, f.latitudinal.value = a === "latitudinal", o && (f.poleAxis.value = o);
    const { _lodMax: b } = this;
    f.dTheta.value = g, f.mipInt.value = b - n;
    const S = this._sizeLods[s], L = 3 * S * (s > b - vi ? s - b + vi : 0), T = 4 * (this._cubeSize - S);
    vs(e, L, T, 3 * S, 2 * S), l.setRenderTarget(e), l.render(d, hr);
  }
}
function Af(i) {
  const t = [], e = [], n = [];
  let s = i;
  const r = i - vi + 1 + xo.length;
  for (let a = 0; a < r; a++) {
    const o = Math.pow(2, s);
    e.push(o);
    let l = 1 / o;
    a > i - vi ? l = xo[a - i + vi - 1] : a === 0 && (l = 0), n.push(l);
    const c = 1 / (o - 2), u = -c, d = 1 + c, f = [u, u, d, u, d, d, u, u, d, d, u, d], m = 6, g = 6, v = 3, p = 2, h = 1, E = new Float32Array(v * g * m), b = new Float32Array(p * g * m), S = new Float32Array(h * g * m);
    for (let T = 0; T < m; T++) {
      const R = T % 3 * 2 / 3 - 1, I = T > 2 ? 0 : -1, y = [
        R,
        I,
        0,
        R + 2 / 3,
        I,
        0,
        R + 2 / 3,
        I + 1,
        0,
        R,
        I,
        0,
        R + 2 / 3,
        I + 1,
        0,
        R,
        I + 1,
        0
      ];
      E.set(y, v * g * T), b.set(f, p * g * T);
      const M = [T, T, T, T, T, T];
      S.set(M, h * g * T);
    }
    const L = new Ce();
    L.setAttribute("position", new en(E, v)), L.setAttribute("uv", new en(b, p)), L.setAttribute("faceIndex", new en(S, h)), t.push(L), s > vi && s--;
  }
  return { lodPlanes: t, sizeLods: e, sigmas: n };
}
function Eo(i, t, e) {
  const n = new Zn(i, t, e);
  return n.texture.mapping = Ns, n.texture.name = "PMREM.cubeUv", n.scissorTest = !0, n;
}
function vs(i, t, e, n, s) {
  i.viewport.set(t, e, n, s), i.scissor.set(t, e, n, s);
}
function Tf(i, t, e) {
  const n = new Float32Array(Xn), s = new P(0, 1, 0);
  return new En({
    name: "SphericalGaussianBlur",
    defines: {
      n: Xn,
      CUBEUV_TEXEL_WIDTH: 1 / t,
      CUBEUV_TEXEL_HEIGHT: 1 / e,
      CUBEUV_MAX_MIP: `${i}.0`
    },
    uniforms: {
      envMap: { value: null },
      samples: { value: 1 },
      weights: { value: n },
      latitudinal: { value: !1 },
      dTheta: { value: 0 },
      mipInt: { value: 0 },
      poleAxis: { value: s }
    },
    vertexShader: wa(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`
    ),
    blending: Ln,
    depthTest: !1,
    depthWrite: !1
  });
}
function bo() {
  return new En({
    name: "EquirectangularToCubeUV",
    uniforms: {
      envMap: { value: null }
    },
    vertexShader: wa(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`
    ),
    blending: Ln,
    depthTest: !1,
    depthWrite: !1
  });
}
function Ao() {
  return new En({
    name: "CubemapToCubeUV",
    uniforms: {
      envMap: { value: null },
      flipEnvMap: { value: -1 }
    },
    vertexShader: wa(),
    fragmentShader: (
      /* glsl */
      `

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`
    ),
    blending: Ln,
    depthTest: !1,
    depthWrite: !1
  });
}
function wa() {
  return (
    /* glsl */
    `

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`
  );
}
function wf(i) {
  let t = /* @__PURE__ */ new WeakMap(), e = null;
  function n(o) {
    if (o && o.isTexture) {
      const l = o.mapping, c = l === Ur || l === Ir, u = l === bi || l === Ai;
      if (c || u) {
        let d = t.get(o);
        const f = d !== void 0 ? d.texture.pmremVersion : 0;
        if (o.isRenderTargetTexture && o.pmremVersion !== f)
          return e === null && (e = new yo(i)), d = c ? e.fromEquirectangular(o, d) : e.fromCubemap(o, d), d.texture.pmremVersion = o.pmremVersion, t.set(o, d), d.texture;
        if (d !== void 0)
          return d.texture;
        {
          const m = o.image;
          return c && m && m.height > 0 || u && m && s(m) ? (e === null && (e = new yo(i)), d = c ? e.fromEquirectangular(o) : e.fromCubemap(o), d.texture.pmremVersion = o.pmremVersion, t.set(o, d), o.addEventListener("dispose", r), d.texture) : null;
        }
      }
    }
    return o;
  }
  function s(o) {
    let l = 0;
    const c = 6;
    for (let u = 0; u < c; u++)
      o[u] !== void 0 && l++;
    return l === c;
  }
  function r(o) {
    const l = o.target;
    l.removeEventListener("dispose", r);
    const c = t.get(l);
    c !== void 0 && (t.delete(l), c.dispose());
  }
  function a() {
    t = /* @__PURE__ */ new WeakMap(), e !== null && (e.dispose(), e = null);
  }
  return {
    get: n,
    dispose: a
  };
}
function Rf(i) {
  const t = {};
  function e(n) {
    if (t[n] !== void 0)
      return t[n];
    let s;
    switch (n) {
      case "WEBGL_depth_texture":
        s = i.getExtension("WEBGL_depth_texture") || i.getExtension("MOZ_WEBGL_depth_texture") || i.getExtension("WEBKIT_WEBGL_depth_texture");
        break;
      case "EXT_texture_filter_anisotropic":
        s = i.getExtension("EXT_texture_filter_anisotropic") || i.getExtension("MOZ_EXT_texture_filter_anisotropic") || i.getExtension("WEBKIT_EXT_texture_filter_anisotropic");
        break;
      case "WEBGL_compressed_texture_s3tc":
        s = i.getExtension("WEBGL_compressed_texture_s3tc") || i.getExtension("MOZ_WEBGL_compressed_texture_s3tc") || i.getExtension("WEBKIT_WEBGL_compressed_texture_s3tc");
        break;
      case "WEBGL_compressed_texture_pvrtc":
        s = i.getExtension("WEBGL_compressed_texture_pvrtc") || i.getExtension("WEBKIT_WEBGL_compressed_texture_pvrtc");
        break;
      default:
        s = i.getExtension(n);
    }
    return t[n] = s, s;
  }
  return {
    has: function(n) {
      return e(n) !== null;
    },
    init: function() {
      e("EXT_color_buffer_float"), e("WEBGL_clip_cull_distance"), e("OES_texture_float_linear"), e("EXT_color_buffer_half_float"), e("WEBGL_multisampled_render_to_texture"), e("WEBGL_render_shared_exponent");
    },
    get: function(n) {
      const s = e(n);
      return s === null && _i("THREE.WebGLRenderer: " + n + " extension not supported."), s;
    }
  };
}
function Cf(i, t, e, n) {
  const s = {}, r = /* @__PURE__ */ new WeakMap();
  function a(d) {
    const f = d.target;
    f.index !== null && t.remove(f.index);
    for (const g in f.attributes)
      t.remove(f.attributes[g]);
    f.removeEventListener("dispose", a), delete s[f.id];
    const m = r.get(f);
    m && (t.remove(m), r.delete(f)), n.releaseStatesOfGeometry(f), f.isInstancedBufferGeometry === !0 && delete f._maxInstanceCount, e.memory.geometries--;
  }
  function o(d, f) {
    return s[f.id] === !0 || (f.addEventListener("dispose", a), s[f.id] = !0, e.memory.geometries++), f;
  }
  function l(d) {
    const f = d.attributes;
    for (const m in f)
      t.update(f[m], i.ARRAY_BUFFER);
  }
  function c(d) {
    const f = [], m = d.index, g = d.attributes.position;
    let v = 0;
    if (m !== null) {
      const E = m.array;
      v = m.version;
      for (let b = 0, S = E.length; b < S; b += 3) {
        const L = E[b + 0], T = E[b + 1], R = E[b + 2];
        f.push(L, T, T, R, R, L);
      }
    } else if (g !== void 0) {
      const E = g.array;
      v = g.version;
      for (let b = 0, S = E.length / 3 - 1; b < S; b += 3) {
        const L = b + 0, T = b + 1, R = b + 2;
        f.push(L, T, T, R, R, L);
      }
    } else
      return;
    const p = new (_l(f) ? yl : Sl)(f, 1);
    p.version = v;
    const h = r.get(d);
    h && t.remove(h), r.set(d, p);
  }
  function u(d) {
    const f = r.get(d);
    if (f) {
      const m = d.index;
      m !== null && f.version < m.version && c(d);
    } else
      c(d);
    return r.get(d);
  }
  return {
    get: o,
    update: l,
    getWireframeAttribute: u
  };
}
function Pf(i, t, e) {
  let n;
  function s(f) {
    n = f;
  }
  let r, a;
  function o(f) {
    r = f.type, a = f.bytesPerElement;
  }
  function l(f, m) {
    i.drawElements(n, m, r, f * a), e.update(m, n, 1);
  }
  function c(f, m, g) {
    g !== 0 && (i.drawElementsInstanced(n, m, r, f * a, g), e.update(m, n, g));
  }
  function u(f, m, g) {
    if (g === 0) return;
    t.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n, m, 0, r, f, 0, g);
    let p = 0;
    for (let h = 0; h < g; h++)
      p += m[h];
    e.update(p, n, 1);
  }
  function d(f, m, g, v) {
    if (g === 0) return;
    const p = t.get("WEBGL_multi_draw");
    if (p === null)
      for (let h = 0; h < f.length; h++)
        c(f[h] / a, m[h], v[h]);
    else {
      p.multiDrawElementsInstancedWEBGL(n, m, 0, r, f, 0, v, 0, g);
      let h = 0;
      for (let E = 0; E < g; E++)
        h += m[E] * v[E];
      e.update(h, n, 1);
    }
  }
  this.setMode = s, this.setIndex = o, this.render = l, this.renderInstances = c, this.renderMultiDraw = u, this.renderMultiDrawInstances = d;
}
function Df(i) {
  const t = {
    geometries: 0,
    textures: 0
  }, e = {
    frame: 0,
    calls: 0,
    triangles: 0,
    points: 0,
    lines: 0
  };
  function n(r, a, o) {
    switch (e.calls++, a) {
      case i.TRIANGLES:
        e.triangles += o * (r / 3);
        break;
      case i.LINES:
        e.lines += o * (r / 2);
        break;
      case i.LINE_STRIP:
        e.lines += o * (r - 1);
        break;
      case i.LINE_LOOP:
        e.lines += o * r;
        break;
      case i.POINTS:
        e.points += o * r;
        break;
      default:
        console.error("THREE.WebGLInfo: Unknown draw mode:", a);
        break;
    }
  }
  function s() {
    e.calls = 0, e.triangles = 0, e.points = 0, e.lines = 0;
  }
  return {
    memory: t,
    render: e,
    programs: null,
    autoReset: !0,
    reset: s,
    update: n
  };
}
function Lf(i, t, e) {
  const n = /* @__PURE__ */ new WeakMap(), s = new te();
  function r(a, o, l) {
    const c = a.morphTargetInfluences, u = o.morphAttributes.position || o.morphAttributes.normal || o.morphAttributes.color, d = u !== void 0 ? u.length : 0;
    let f = n.get(o);
    if (f === void 0 || f.count !== d) {
      let y = function() {
        R.dispose(), n.delete(o), o.removeEventListener("dispose", y);
      };
      f !== void 0 && f.texture.dispose();
      const m = o.morphAttributes.position !== void 0, g = o.morphAttributes.normal !== void 0, v = o.morphAttributes.color !== void 0, p = o.morphAttributes.position || [], h = o.morphAttributes.normal || [], E = o.morphAttributes.color || [];
      let b = 0;
      m === !0 && (b = 1), g === !0 && (b = 2), v === !0 && (b = 3);
      let S = o.attributes.position.count * b, L = 1;
      S > t.maxTextureSize && (L = Math.ceil(S / t.maxTextureSize), S = t.maxTextureSize);
      const T = new Float32Array(S * L * 4 * d), R = new vl(T, S, L, d);
      R.type = gn, R.needsUpdate = !0;
      const I = b * 4;
      for (let M = 0; M < d; M++) {
        const C = p[M], H = h[M], z = E[M], G = S * L * 4 * M;
        for (let j = 0; j < C.count; j++) {
          const W = j * I;
          m === !0 && (s.fromBufferAttribute(C, j), T[G + W + 0] = s.x, T[G + W + 1] = s.y, T[G + W + 2] = s.z, T[G + W + 3] = 0), g === !0 && (s.fromBufferAttribute(H, j), T[G + W + 4] = s.x, T[G + W + 5] = s.y, T[G + W + 6] = s.z, T[G + W + 7] = 0), v === !0 && (s.fromBufferAttribute(z, j), T[G + W + 8] = s.x, T[G + W + 9] = s.y, T[G + W + 10] = s.z, T[G + W + 11] = z.itemSize === 4 ? s.w : 1);
        }
      }
      f = {
        count: d,
        texture: R,
        size: new bt(S, L)
      }, n.set(o, f), o.addEventListener("dispose", y);
    }
    if (a.isInstancedMesh === !0 && a.morphTexture !== null)
      l.getUniforms().setValue(i, "morphTexture", a.morphTexture, e);
    else {
      let m = 0;
      for (let v = 0; v < c.length; v++)
        m += c[v];
      const g = o.morphTargetsRelative ? 1 : 1 - m;
      l.getUniforms().setValue(i, "morphTargetBaseInfluence", g), l.getUniforms().setValue(i, "morphTargetInfluences", c);
    }
    l.getUniforms().setValue(i, "morphTargetsTexture", f.texture, e), l.getUniforms().setValue(i, "morphTargetsTextureSize", f.size);
  }
  return {
    update: r
  };
}
function Uf(i, t, e, n) {
  let s = /* @__PURE__ */ new WeakMap();
  function r(l) {
    const c = n.render.frame, u = l.geometry, d = t.get(l, u);
    if (s.get(d) !== c && (t.update(d), s.set(d, c)), l.isInstancedMesh && (l.hasEventListener("dispose", o) === !1 && l.addEventListener("dispose", o), s.get(l) !== c && (e.update(l.instanceMatrix, i.ARRAY_BUFFER), l.instanceColor !== null && e.update(l.instanceColor, i.ARRAY_BUFFER), s.set(l, c))), l.isSkinnedMesh) {
      const f = l.skeleton;
      s.get(f) !== c && (f.update(), s.set(f, c));
    }
    return d;
  }
  function a() {
    s = /* @__PURE__ */ new WeakMap();
  }
  function o(l) {
    const c = l.target;
    c.removeEventListener("dispose", o), e.remove(c.instanceMatrix), c.instanceColor !== null && e.remove(c.instanceColor);
  }
  return {
    update: r,
    dispose: a
  };
}
const Pl = /* @__PURE__ */ new Ie(), To = /* @__PURE__ */ new Rl(1, 1), Dl = /* @__PURE__ */ new vl(), Ll = /* @__PURE__ */ new rh(), Ul = /* @__PURE__ */ new Al(), wo = [], Ro = [], Co = new Float32Array(16), Po = new Float32Array(9), Do = new Float32Array(4);
function Di(i, t, e) {
  const n = i[0];
  if (n <= 0 || n > 0) return i;
  const s = t * e;
  let r = wo[s];
  if (r === void 0 && (r = new Float32Array(s), wo[s] = r), t !== 0) {
    n.toArray(r, 0);
    for (let a = 1, o = 0; a !== t; ++a)
      o += e, i[a].toArray(r, o);
  }
  return r;
}
function pe(i, t) {
  if (i.length !== t.length) return !1;
  for (let e = 0, n = i.length; e < n; e++)
    if (i[e] !== t[e]) return !1;
  return !0;
}
function me(i, t) {
  for (let e = 0, n = t.length; e < n; e++)
    i[e] = t[e];
}
function Bs(i, t) {
  let e = Ro[t];
  e === void 0 && (e = new Int32Array(t), Ro[t] = e);
  for (let n = 0; n !== t; ++n)
    e[n] = i.allocateTextureUnit();
  return e;
}
function If(i, t) {
  const e = this.cache;
  e[0] !== t && (i.uniform1f(this.addr, t), e[0] = t);
}
function Nf(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y) && (i.uniform2f(this.addr, t.x, t.y), e[0] = t.x, e[1] = t.y);
  else {
    if (pe(e, t)) return;
    i.uniform2fv(this.addr, t), me(e, t);
  }
}
function Ff(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y || e[2] !== t.z) && (i.uniform3f(this.addr, t.x, t.y, t.z), e[0] = t.x, e[1] = t.y, e[2] = t.z);
  else if (t.r !== void 0)
    (e[0] !== t.r || e[1] !== t.g || e[2] !== t.b) && (i.uniform3f(this.addr, t.r, t.g, t.b), e[0] = t.r, e[1] = t.g, e[2] = t.b);
  else {
    if (pe(e, t)) return;
    i.uniform3fv(this.addr, t), me(e, t);
  }
}
function Of(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y || e[2] !== t.z || e[3] !== t.w) && (i.uniform4f(this.addr, t.x, t.y, t.z, t.w), e[0] = t.x, e[1] = t.y, e[2] = t.z, e[3] = t.w);
  else {
    if (pe(e, t)) return;
    i.uniform4fv(this.addr, t), me(e, t);
  }
}
function Bf(i, t) {
  const e = this.cache, n = t.elements;
  if (n === void 0) {
    if (pe(e, t)) return;
    i.uniformMatrix2fv(this.addr, !1, t), me(e, t);
  } else {
    if (pe(e, n)) return;
    Do.set(n), i.uniformMatrix2fv(this.addr, !1, Do), me(e, n);
  }
}
function zf(i, t) {
  const e = this.cache, n = t.elements;
  if (n === void 0) {
    if (pe(e, t)) return;
    i.uniformMatrix3fv(this.addr, !1, t), me(e, t);
  } else {
    if (pe(e, n)) return;
    Po.set(n), i.uniformMatrix3fv(this.addr, !1, Po), me(e, n);
  }
}
function Hf(i, t) {
  const e = this.cache, n = t.elements;
  if (n === void 0) {
    if (pe(e, t)) return;
    i.uniformMatrix4fv(this.addr, !1, t), me(e, t);
  } else {
    if (pe(e, n)) return;
    Co.set(n), i.uniformMatrix4fv(this.addr, !1, Co), me(e, n);
  }
}
function kf(i, t) {
  const e = this.cache;
  e[0] !== t && (i.uniform1i(this.addr, t), e[0] = t);
}
function Vf(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y) && (i.uniform2i(this.addr, t.x, t.y), e[0] = t.x, e[1] = t.y);
  else {
    if (pe(e, t)) return;
    i.uniform2iv(this.addr, t), me(e, t);
  }
}
function Gf(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y || e[2] !== t.z) && (i.uniform3i(this.addr, t.x, t.y, t.z), e[0] = t.x, e[1] = t.y, e[2] = t.z);
  else {
    if (pe(e, t)) return;
    i.uniform3iv(this.addr, t), me(e, t);
  }
}
function Wf(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y || e[2] !== t.z || e[3] !== t.w) && (i.uniform4i(this.addr, t.x, t.y, t.z, t.w), e[0] = t.x, e[1] = t.y, e[2] = t.z, e[3] = t.w);
  else {
    if (pe(e, t)) return;
    i.uniform4iv(this.addr, t), me(e, t);
  }
}
function Xf(i, t) {
  const e = this.cache;
  e[0] !== t && (i.uniform1ui(this.addr, t), e[0] = t);
}
function Yf(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y) && (i.uniform2ui(this.addr, t.x, t.y), e[0] = t.x, e[1] = t.y);
  else {
    if (pe(e, t)) return;
    i.uniform2uiv(this.addr, t), me(e, t);
  }
}
function $f(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y || e[2] !== t.z) && (i.uniform3ui(this.addr, t.x, t.y, t.z), e[0] = t.x, e[1] = t.y, e[2] = t.z);
  else {
    if (pe(e, t)) return;
    i.uniform3uiv(this.addr, t), me(e, t);
  }
}
function qf(i, t) {
  const e = this.cache;
  if (t.x !== void 0)
    (e[0] !== t.x || e[1] !== t.y || e[2] !== t.z || e[3] !== t.w) && (i.uniform4ui(this.addr, t.x, t.y, t.z, t.w), e[0] = t.x, e[1] = t.y, e[2] = t.z, e[3] = t.w);
  else {
    if (pe(e, t)) return;
    i.uniform4uiv(this.addr, t), me(e, t);
  }
}
function jf(i, t, e) {
  const n = this.cache, s = e.allocateTextureUnit();
  n[0] !== s && (i.uniform1i(this.addr, s), n[0] = s);
  let r;
  this.type === i.SAMPLER_2D_SHADOW ? (To.compareFunction = pl, r = To) : r = Pl, e.setTexture2D(t || r, s);
}
function Zf(i, t, e) {
  const n = this.cache, s = e.allocateTextureUnit();
  n[0] !== s && (i.uniform1i(this.addr, s), n[0] = s), e.setTexture3D(t || Ll, s);
}
function Kf(i, t, e) {
  const n = this.cache, s = e.allocateTextureUnit();
  n[0] !== s && (i.uniform1i(this.addr, s), n[0] = s), e.setTextureCube(t || Ul, s);
}
function Jf(i, t, e) {
  const n = this.cache, s = e.allocateTextureUnit();
  n[0] !== s && (i.uniform1i(this.addr, s), n[0] = s), e.setTexture2DArray(t || Dl, s);
}
function Qf(i) {
  switch (i) {
    case 5126:
      return If;
    // FLOAT
    case 35664:
      return Nf;
    // _VEC2
    case 35665:
      return Ff;
    // _VEC3
    case 35666:
      return Of;
    // _VEC4
    case 35674:
      return Bf;
    // _MAT2
    case 35675:
      return zf;
    // _MAT3
    case 35676:
      return Hf;
    // _MAT4
    case 5124:
    case 35670:
      return kf;
    // INT, BOOL
    case 35667:
    case 35671:
      return Vf;
    // _VEC2
    case 35668:
    case 35672:
      return Gf;
    // _VEC3
    case 35669:
    case 35673:
      return Wf;
    // _VEC4
    case 5125:
      return Xf;
    // UINT
    case 36294:
      return Yf;
    // _VEC2
    case 36295:
      return $f;
    // _VEC3
    case 36296:
      return qf;
    // _VEC4
    case 35678:
    // SAMPLER_2D
    case 36198:
    // SAMPLER_EXTERNAL_OES
    case 36298:
    // INT_SAMPLER_2D
    case 36306:
    // UNSIGNED_INT_SAMPLER_2D
    case 35682:
      return jf;
    case 35679:
    // SAMPLER_3D
    case 36299:
    // INT_SAMPLER_3D
    case 36307:
      return Zf;
    case 35680:
    // SAMPLER_CUBE
    case 36300:
    // INT_SAMPLER_CUBE
    case 36308:
    // UNSIGNED_INT_SAMPLER_CUBE
    case 36293:
      return Kf;
    case 36289:
    // SAMPLER_2D_ARRAY
    case 36303:
    // INT_SAMPLER_2D_ARRAY
    case 36311:
    // UNSIGNED_INT_SAMPLER_2D_ARRAY
    case 36292:
      return Jf;
  }
}
function tp(i, t) {
  i.uniform1fv(this.addr, t);
}
function ep(i, t) {
  const e = Di(t, this.size, 2);
  i.uniform2fv(this.addr, e);
}
function np(i, t) {
  const e = Di(t, this.size, 3);
  i.uniform3fv(this.addr, e);
}
function ip(i, t) {
  const e = Di(t, this.size, 4);
  i.uniform4fv(this.addr, e);
}
function sp(i, t) {
  const e = Di(t, this.size, 4);
  i.uniformMatrix2fv(this.addr, !1, e);
}
function rp(i, t) {
  const e = Di(t, this.size, 9);
  i.uniformMatrix3fv(this.addr, !1, e);
}
function ap(i, t) {
  const e = Di(t, this.size, 16);
  i.uniformMatrix4fv(this.addr, !1, e);
}
function op(i, t) {
  i.uniform1iv(this.addr, t);
}
function lp(i, t) {
  i.uniform2iv(this.addr, t);
}
function cp(i, t) {
  i.uniform3iv(this.addr, t);
}
function hp(i, t) {
  i.uniform4iv(this.addr, t);
}
function up(i, t) {
  i.uniform1uiv(this.addr, t);
}
function dp(i, t) {
  i.uniform2uiv(this.addr, t);
}
function fp(i, t) {
  i.uniform3uiv(this.addr, t);
}
function pp(i, t) {
  i.uniform4uiv(this.addr, t);
}
function mp(i, t, e) {
  const n = this.cache, s = t.length, r = Bs(e, s);
  pe(n, r) || (i.uniform1iv(this.addr, r), me(n, r));
  for (let a = 0; a !== s; ++a)
    e.setTexture2D(t[a] || Pl, r[a]);
}
function _p(i, t, e) {
  const n = this.cache, s = t.length, r = Bs(e, s);
  pe(n, r) || (i.uniform1iv(this.addr, r), me(n, r));
  for (let a = 0; a !== s; ++a)
    e.setTexture3D(t[a] || Ll, r[a]);
}
function gp(i, t, e) {
  const n = this.cache, s = t.length, r = Bs(e, s);
  pe(n, r) || (i.uniform1iv(this.addr, r), me(n, r));
  for (let a = 0; a !== s; ++a)
    e.setTextureCube(t[a] || Ul, r[a]);
}
function vp(i, t, e) {
  const n = this.cache, s = t.length, r = Bs(e, s);
  pe(n, r) || (i.uniform1iv(this.addr, r), me(n, r));
  for (let a = 0; a !== s; ++a)
    e.setTexture2DArray(t[a] || Dl, r[a]);
}
function xp(i) {
  switch (i) {
    case 5126:
      return tp;
    // FLOAT
    case 35664:
      return ep;
    // _VEC2
    case 35665:
      return np;
    // _VEC3
    case 35666:
      return ip;
    // _VEC4
    case 35674:
      return sp;
    // _MAT2
    case 35675:
      return rp;
    // _MAT3
    case 35676:
      return ap;
    // _MAT4
    case 5124:
    case 35670:
      return op;
    // INT, BOOL
    case 35667:
    case 35671:
      return lp;
    // _VEC2
    case 35668:
    case 35672:
      return cp;
    // _VEC3
    case 35669:
    case 35673:
      return hp;
    // _VEC4
    case 5125:
      return up;
    // UINT
    case 36294:
      return dp;
    // _VEC2
    case 36295:
      return fp;
    // _VEC3
    case 36296:
      return pp;
    // _VEC4
    case 35678:
    // SAMPLER_2D
    case 36198:
    // SAMPLER_EXTERNAL_OES
    case 36298:
    // INT_SAMPLER_2D
    case 36306:
    // UNSIGNED_INT_SAMPLER_2D
    case 35682:
      return mp;
    case 35679:
    // SAMPLER_3D
    case 36299:
    // INT_SAMPLER_3D
    case 36307:
      return _p;
    case 35680:
    // SAMPLER_CUBE
    case 36300:
    // INT_SAMPLER_CUBE
    case 36308:
    // UNSIGNED_INT_SAMPLER_CUBE
    case 36293:
      return gp;
    case 36289:
    // SAMPLER_2D_ARRAY
    case 36303:
    // INT_SAMPLER_2D_ARRAY
    case 36311:
    // UNSIGNED_INT_SAMPLER_2D_ARRAY
    case 36292:
      return vp;
  }
}
class Mp {
  constructor(t, e, n) {
    this.id = t, this.addr = n, this.cache = [], this.type = e.type, this.setValue = Qf(e.type);
  }
}
class Sp {
  constructor(t, e, n) {
    this.id = t, this.addr = n, this.cache = [], this.type = e.type, this.size = e.size, this.setValue = xp(e.type);
  }
}
class yp {
  constructor(t) {
    this.id = t, this.seq = [], this.map = {};
  }
  setValue(t, e, n) {
    const s = this.seq;
    for (let r = 0, a = s.length; r !== a; ++r) {
      const o = s[r];
      o.setValue(t, e[o.id], n);
    }
  }
}
const mr = /(\w+)(\])?(\[|\.)?/g;
function Lo(i, t) {
  i.seq.push(t), i.map[t.id] = t;
}
function Ep(i, t, e) {
  const n = i.name, s = n.length;
  for (mr.lastIndex = 0; ; ) {
    const r = mr.exec(n), a = mr.lastIndex;
    let o = r[1];
    const l = r[2] === "]", c = r[3];
    if (l && (o = o | 0), c === void 0 || c === "[" && a + 2 === s) {
      Lo(e, c === void 0 ? new Mp(o, i, t) : new Sp(o, i, t));
      break;
    } else {
      let d = e.map[o];
      d === void 0 && (d = new yp(o), Lo(e, d)), e = d;
    }
  }
}
class Cs {
  constructor(t, e) {
    this.seq = [], this.map = {};
    const n = t.getProgramParameter(e, t.ACTIVE_UNIFORMS);
    for (let s = 0; s < n; ++s) {
      const r = t.getActiveUniform(e, s), a = t.getUniformLocation(e, r.name);
      Ep(r, a, this);
    }
  }
  setValue(t, e, n, s) {
    const r = this.map[e];
    r !== void 0 && r.setValue(t, n, s);
  }
  setOptional(t, e, n) {
    const s = e[n];
    s !== void 0 && this.setValue(t, n, s);
  }
  static upload(t, e, n, s) {
    for (let r = 0, a = e.length; r !== a; ++r) {
      const o = e[r], l = n[o.id];
      l.needsUpdate !== !1 && o.setValue(t, l.value, s);
    }
  }
  static seqWithValue(t, e) {
    const n = [];
    for (let s = 0, r = t.length; s !== r; ++s) {
      const a = t[s];
      a.id in e && n.push(a);
    }
    return n;
  }
}
function Uo(i, t, e) {
  const n = i.createShader(t);
  return i.shaderSource(n, e), i.compileShader(n), n;
}
const bp = 37297;
let Ap = 0;
function Tp(i, t) {
  const e = i.split(`
`), n = [], s = Math.max(t - 6, 0), r = Math.min(t + 6, e.length);
  for (let a = s; a < r; a++) {
    const o = a + 1;
    n.push(`${o === t ? ">" : " "} ${o}: ${e[a]}`);
  }
  return n.join(`
`);
}
const Io = /* @__PURE__ */ new Pt();
function wp(i) {
  Wt._getMatrix(Io, Wt.workingColorSpace, i);
  const t = `mat3( ${Io.elements.map((e) => e.toFixed(4))} )`;
  switch (Wt.getTransfer(i)) {
    case Ps:
      return [t, "LinearTransferOETF"];
    case jt:
      return [t, "sRGBTransferOETF"];
    default:
      return console.warn("THREE.WebGLProgram: Unsupported color space: ", i), [t, "LinearTransferOETF"];
  }
}
function No(i, t, e) {
  const n = i.getShaderParameter(t, i.COMPILE_STATUS), s = i.getShaderInfoLog(t).trim();
  if (n && s === "") return "";
  const r = /ERROR: 0:(\d+)/.exec(s);
  if (r) {
    const a = parseInt(r[1]);
    return e.toUpperCase() + `

` + s + `

` + Tp(i.getShaderSource(t), a);
  } else
    return s;
}
function Rp(i, t) {
  const e = wp(t);
  return [
    `vec4 ${i}( vec4 value ) {`,
    `	return ${e[1]}( vec4( value.rgb * ${e[0]}, value.a ) );`,
    "}"
  ].join(`
`);
}
function Cp(i, t) {
  let e;
  switch (t) {
    case _c:
      e = "Linear";
      break;
    case gc:
      e = "Reinhard";
      break;
    case vc:
      e = "Cineon";
      break;
    case xc:
      e = "ACESFilmic";
      break;
    case Sc:
      e = "AgX";
      break;
    case yc:
      e = "Neutral";
      break;
    case Mc:
      e = "Custom";
      break;
    default:
      console.warn("THREE.WebGLProgram: Unsupported toneMapping:", t), e = "Linear";
  }
  return "vec3 " + i + "( vec3 color ) { return " + e + "ToneMapping( color ); }";
}
const xs = /* @__PURE__ */ new P();
function Pp() {
  Wt.getLuminanceCoefficients(xs);
  const i = xs.x.toFixed(4), t = xs.y.toFixed(4), e = xs.z.toFixed(4);
  return [
    "float luminance( const in vec3 rgb ) {",
    `	const vec3 weights = vec3( ${i}, ${t}, ${e} );`,
    "	return dot( weights, rgb );",
    "}"
  ].join(`
`);
}
function Dp(i) {
  return [
    i.extensionClipCullDistance ? "#extension GL_ANGLE_clip_cull_distance : require" : "",
    i.extensionMultiDraw ? "#extension GL_ANGLE_multi_draw : require" : ""
  ].filter(zi).join(`
`);
}
function Lp(i) {
  const t = [];
  for (const e in i) {
    const n = i[e];
    n !== !1 && t.push("#define " + e + " " + n);
  }
  return t.join(`
`);
}
function Up(i, t) {
  const e = {}, n = i.getProgramParameter(t, i.ACTIVE_ATTRIBUTES);
  for (let s = 0; s < n; s++) {
    const r = i.getActiveAttrib(t, s), a = r.name;
    let o = 1;
    r.type === i.FLOAT_MAT2 && (o = 2), r.type === i.FLOAT_MAT3 && (o = 3), r.type === i.FLOAT_MAT4 && (o = 4), e[a] = {
      type: r.type,
      location: i.getAttribLocation(t, a),
      locationSize: o
    };
  }
  return e;
}
function zi(i) {
  return i !== "";
}
function Fo(i, t) {
  const e = t.numSpotLightShadows + t.numSpotLightMaps - t.numSpotLightShadowsWithMaps;
  return i.replace(/NUM_DIR_LIGHTS/g, t.numDirLights).replace(/NUM_SPOT_LIGHTS/g, t.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g, t.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g, e).replace(/NUM_RECT_AREA_LIGHTS/g, t.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g, t.numPointLights).replace(/NUM_HEMI_LIGHTS/g, t.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g, t.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g, t.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g, t.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g, t.numPointLightShadows);
}
function Oo(i, t) {
  return i.replace(/NUM_CLIPPING_PLANES/g, t.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g, t.numClippingPlanes - t.numClipIntersection);
}
const Ip = /^[ \t]*#include +<([\w\d./]+)>/gm;
function ua(i) {
  return i.replace(Ip, Fp);
}
const Np = /* @__PURE__ */ new Map();
function Fp(i, t) {
  let e = Lt[t];
  if (e === void 0) {
    const n = Np.get(t);
    if (n !== void 0)
      e = Lt[n], console.warn('THREE.WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.', t, n);
    else
      throw new Error("Can not resolve #include <" + t + ">");
  }
  return ua(e);
}
const Op = /#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;
function Bo(i) {
  return i.replace(Op, Bp);
}
function Bp(i, t, e, n) {
  let s = "";
  for (let r = parseInt(t); r < parseInt(e); r++)
    s += n.replace(/\[\s*i\s*\]/g, "[ " + r + " ]").replace(/UNROLLED_LOOP_INDEX/g, r);
  return s;
}
function zo(i) {
  let t = `precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;
  return i.precision === "highp" ? t += `
#define HIGH_PRECISION` : i.precision === "mediump" ? t += `
#define MEDIUM_PRECISION` : i.precision === "lowp" && (t += `
#define LOW_PRECISION`), t;
}
function zp(i) {
  let t = "SHADOWMAP_TYPE_BASIC";
  return i.shadowMapType === tl ? t = "SHADOWMAP_TYPE_PCF" : i.shadowMapType === jl ? t = "SHADOWMAP_TYPE_PCF_SOFT" : i.shadowMapType === mn && (t = "SHADOWMAP_TYPE_VSM"), t;
}
function Hp(i) {
  let t = "ENVMAP_TYPE_CUBE";
  if (i.envMap)
    switch (i.envMapMode) {
      case bi:
      case Ai:
        t = "ENVMAP_TYPE_CUBE";
        break;
      case Ns:
        t = "ENVMAP_TYPE_CUBE_UV";
        break;
    }
  return t;
}
function kp(i) {
  let t = "ENVMAP_MODE_REFLECTION";
  if (i.envMap)
    switch (i.envMapMode) {
      case Ai:
        t = "ENVMAP_MODE_REFRACTION";
        break;
    }
  return t;
}
function Vp(i) {
  let t = "ENVMAP_BLENDING_NONE";
  if (i.envMap)
    switch (i.combine) {
      case el:
        t = "ENVMAP_BLENDING_MULTIPLY";
        break;
      case pc:
        t = "ENVMAP_BLENDING_MIX";
        break;
      case mc:
        t = "ENVMAP_BLENDING_ADD";
        break;
    }
  return t;
}
function Gp(i) {
  const t = i.envMapCubeUVHeight;
  if (t === null) return null;
  const e = Math.log2(t) - 2, n = 1 / t;
  return { texelWidth: 1 / (3 * Math.max(Math.pow(2, e), 7 * 16)), texelHeight: n, maxMip: e };
}
function Wp(i, t, e, n) {
  const s = i.getContext(), r = e.defines;
  let a = e.vertexShader, o = e.fragmentShader;
  const l = zp(e), c = Hp(e), u = kp(e), d = Vp(e), f = Gp(e), m = Dp(e), g = Lp(r), v = s.createProgram();
  let p, h, E = e.glslVersion ? "#version " + e.glslVersion + `
` : "";
  e.isRawShaderMaterial ? (p = [
    "#define SHADER_TYPE " + e.shaderType,
    "#define SHADER_NAME " + e.shaderName,
    g
  ].filter(zi).join(`
`), p.length > 0 && (p += `
`), h = [
    "#define SHADER_TYPE " + e.shaderType,
    "#define SHADER_NAME " + e.shaderName,
    g
  ].filter(zi).join(`
`), h.length > 0 && (h += `
`)) : (p = [
    zo(e),
    "#define SHADER_TYPE " + e.shaderType,
    "#define SHADER_NAME " + e.shaderName,
    g,
    e.extensionClipCullDistance ? "#define USE_CLIP_DISTANCE" : "",
    e.batching ? "#define USE_BATCHING" : "",
    e.batchingColor ? "#define USE_BATCHING_COLOR" : "",
    e.instancing ? "#define USE_INSTANCING" : "",
    e.instancingColor ? "#define USE_INSTANCING_COLOR" : "",
    e.instancingMorph ? "#define USE_INSTANCING_MORPH" : "",
    e.useFog && e.fog ? "#define USE_FOG" : "",
    e.useFog && e.fogExp2 ? "#define FOG_EXP2" : "",
    e.map ? "#define USE_MAP" : "",
    e.envMap ? "#define USE_ENVMAP" : "",
    e.envMap ? "#define " + u : "",
    e.lightMap ? "#define USE_LIGHTMAP" : "",
    e.aoMap ? "#define USE_AOMAP" : "",
    e.bumpMap ? "#define USE_BUMPMAP" : "",
    e.normalMap ? "#define USE_NORMALMAP" : "",
    e.normalMapObjectSpace ? "#define USE_NORMALMAP_OBJECTSPACE" : "",
    e.normalMapTangentSpace ? "#define USE_NORMALMAP_TANGENTSPACE" : "",
    e.displacementMap ? "#define USE_DISPLACEMENTMAP" : "",
    e.emissiveMap ? "#define USE_EMISSIVEMAP" : "",
    e.anisotropy ? "#define USE_ANISOTROPY" : "",
    e.anisotropyMap ? "#define USE_ANISOTROPYMAP" : "",
    e.clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
    e.clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
    e.clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
    e.iridescenceMap ? "#define USE_IRIDESCENCEMAP" : "",
    e.iridescenceThicknessMap ? "#define USE_IRIDESCENCE_THICKNESSMAP" : "",
    e.specularMap ? "#define USE_SPECULARMAP" : "",
    e.specularColorMap ? "#define USE_SPECULAR_COLORMAP" : "",
    e.specularIntensityMap ? "#define USE_SPECULAR_INTENSITYMAP" : "",
    e.roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
    e.metalnessMap ? "#define USE_METALNESSMAP" : "",
    e.alphaMap ? "#define USE_ALPHAMAP" : "",
    e.alphaHash ? "#define USE_ALPHAHASH" : "",
    e.transmission ? "#define USE_TRANSMISSION" : "",
    e.transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
    e.thicknessMap ? "#define USE_THICKNESSMAP" : "",
    e.sheenColorMap ? "#define USE_SHEEN_COLORMAP" : "",
    e.sheenRoughnessMap ? "#define USE_SHEEN_ROUGHNESSMAP" : "",
    //
    e.mapUv ? "#define MAP_UV " + e.mapUv : "",
    e.alphaMapUv ? "#define ALPHAMAP_UV " + e.alphaMapUv : "",
    e.lightMapUv ? "#define LIGHTMAP_UV " + e.lightMapUv : "",
    e.aoMapUv ? "#define AOMAP_UV " + e.aoMapUv : "",
    e.emissiveMapUv ? "#define EMISSIVEMAP_UV " + e.emissiveMapUv : "",
    e.bumpMapUv ? "#define BUMPMAP_UV " + e.bumpMapUv : "",
    e.normalMapUv ? "#define NORMALMAP_UV " + e.normalMapUv : "",
    e.displacementMapUv ? "#define DISPLACEMENTMAP_UV " + e.displacementMapUv : "",
    e.metalnessMapUv ? "#define METALNESSMAP_UV " + e.metalnessMapUv : "",
    e.roughnessMapUv ? "#define ROUGHNESSMAP_UV " + e.roughnessMapUv : "",
    e.anisotropyMapUv ? "#define ANISOTROPYMAP_UV " + e.anisotropyMapUv : "",
    e.clearcoatMapUv ? "#define CLEARCOATMAP_UV " + e.clearcoatMapUv : "",
    e.clearcoatNormalMapUv ? "#define CLEARCOAT_NORMALMAP_UV " + e.clearcoatNormalMapUv : "",
    e.clearcoatRoughnessMapUv ? "#define CLEARCOAT_ROUGHNESSMAP_UV " + e.clearcoatRoughnessMapUv : "",
    e.iridescenceMapUv ? "#define IRIDESCENCEMAP_UV " + e.iridescenceMapUv : "",
    e.iridescenceThicknessMapUv ? "#define IRIDESCENCE_THICKNESSMAP_UV " + e.iridescenceThicknessMapUv : "",
    e.sheenColorMapUv ? "#define SHEEN_COLORMAP_UV " + e.sheenColorMapUv : "",
    e.sheenRoughnessMapUv ? "#define SHEEN_ROUGHNESSMAP_UV " + e.sheenRoughnessMapUv : "",
    e.specularMapUv ? "#define SPECULARMAP_UV " + e.specularMapUv : "",
    e.specularColorMapUv ? "#define SPECULAR_COLORMAP_UV " + e.specularColorMapUv : "",
    e.specularIntensityMapUv ? "#define SPECULAR_INTENSITYMAP_UV " + e.specularIntensityMapUv : "",
    e.transmissionMapUv ? "#define TRANSMISSIONMAP_UV " + e.transmissionMapUv : "",
    e.thicknessMapUv ? "#define THICKNESSMAP_UV " + e.thicknessMapUv : "",
    //
    e.vertexTangents && e.flatShading === !1 ? "#define USE_TANGENT" : "",
    e.vertexColors ? "#define USE_COLOR" : "",
    e.vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
    e.vertexUv1s ? "#define USE_UV1" : "",
    e.vertexUv2s ? "#define USE_UV2" : "",
    e.vertexUv3s ? "#define USE_UV3" : "",
    e.pointsUvs ? "#define USE_POINTS_UV" : "",
    e.flatShading ? "#define FLAT_SHADED" : "",
    e.skinning ? "#define USE_SKINNING" : "",
    e.morphTargets ? "#define USE_MORPHTARGETS" : "",
    e.morphNormals && e.flatShading === !1 ? "#define USE_MORPHNORMALS" : "",
    e.morphColors ? "#define USE_MORPHCOLORS" : "",
    e.morphTargetsCount > 0 ? "#define MORPHTARGETS_TEXTURE_STRIDE " + e.morphTextureStride : "",
    e.morphTargetsCount > 0 ? "#define MORPHTARGETS_COUNT " + e.morphTargetsCount : "",
    e.doubleSided ? "#define DOUBLE_SIDED" : "",
    e.flipSided ? "#define FLIP_SIDED" : "",
    e.shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
    e.shadowMapEnabled ? "#define " + l : "",
    e.sizeAttenuation ? "#define USE_SIZEATTENUATION" : "",
    e.numLightProbes > 0 ? "#define USE_LIGHT_PROBES" : "",
    e.logarithmicDepthBuffer ? "#define USE_LOGDEPTHBUF" : "",
    e.reverseDepthBuffer ? "#define USE_REVERSEDEPTHBUF" : "",
    "uniform mat4 modelMatrix;",
    "uniform mat4 modelViewMatrix;",
    "uniform mat4 projectionMatrix;",
    "uniform mat4 viewMatrix;",
    "uniform mat3 normalMatrix;",
    "uniform vec3 cameraPosition;",
    "uniform bool isOrthographic;",
    "#ifdef USE_INSTANCING",
    "	attribute mat4 instanceMatrix;",
    "#endif",
    "#ifdef USE_INSTANCING_COLOR",
    "	attribute vec3 instanceColor;",
    "#endif",
    "#ifdef USE_INSTANCING_MORPH",
    "	uniform sampler2D morphTexture;",
    "#endif",
    "attribute vec3 position;",
    "attribute vec3 normal;",
    "attribute vec2 uv;",
    "#ifdef USE_UV1",
    "	attribute vec2 uv1;",
    "#endif",
    "#ifdef USE_UV2",
    "	attribute vec2 uv2;",
    "#endif",
    "#ifdef USE_UV3",
    "	attribute vec2 uv3;",
    "#endif",
    "#ifdef USE_TANGENT",
    "	attribute vec4 tangent;",
    "#endif",
    "#if defined( USE_COLOR_ALPHA )",
    "	attribute vec4 color;",
    "#elif defined( USE_COLOR )",
    "	attribute vec3 color;",
    "#endif",
    "#ifdef USE_SKINNING",
    "	attribute vec4 skinIndex;",
    "	attribute vec4 skinWeight;",
    "#endif",
    `
`
  ].filter(zi).join(`
`), h = [
    zo(e),
    "#define SHADER_TYPE " + e.shaderType,
    "#define SHADER_NAME " + e.shaderName,
    g,
    e.useFog && e.fog ? "#define USE_FOG" : "",
    e.useFog && e.fogExp2 ? "#define FOG_EXP2" : "",
    e.alphaToCoverage ? "#define ALPHA_TO_COVERAGE" : "",
    e.map ? "#define USE_MAP" : "",
    e.matcap ? "#define USE_MATCAP" : "",
    e.envMap ? "#define USE_ENVMAP" : "",
    e.envMap ? "#define " + c : "",
    e.envMap ? "#define " + u : "",
    e.envMap ? "#define " + d : "",
    f ? "#define CUBEUV_TEXEL_WIDTH " + f.texelWidth : "",
    f ? "#define CUBEUV_TEXEL_HEIGHT " + f.texelHeight : "",
    f ? "#define CUBEUV_MAX_MIP " + f.maxMip + ".0" : "",
    e.lightMap ? "#define USE_LIGHTMAP" : "",
    e.aoMap ? "#define USE_AOMAP" : "",
    e.bumpMap ? "#define USE_BUMPMAP" : "",
    e.normalMap ? "#define USE_NORMALMAP" : "",
    e.normalMapObjectSpace ? "#define USE_NORMALMAP_OBJECTSPACE" : "",
    e.normalMapTangentSpace ? "#define USE_NORMALMAP_TANGENTSPACE" : "",
    e.emissiveMap ? "#define USE_EMISSIVEMAP" : "",
    e.anisotropy ? "#define USE_ANISOTROPY" : "",
    e.anisotropyMap ? "#define USE_ANISOTROPYMAP" : "",
    e.clearcoat ? "#define USE_CLEARCOAT" : "",
    e.clearcoatMap ? "#define USE_CLEARCOATMAP" : "",
    e.clearcoatRoughnessMap ? "#define USE_CLEARCOAT_ROUGHNESSMAP" : "",
    e.clearcoatNormalMap ? "#define USE_CLEARCOAT_NORMALMAP" : "",
    e.dispersion ? "#define USE_DISPERSION" : "",
    e.iridescence ? "#define USE_IRIDESCENCE" : "",
    e.iridescenceMap ? "#define USE_IRIDESCENCEMAP" : "",
    e.iridescenceThicknessMap ? "#define USE_IRIDESCENCE_THICKNESSMAP" : "",
    e.specularMap ? "#define USE_SPECULARMAP" : "",
    e.specularColorMap ? "#define USE_SPECULAR_COLORMAP" : "",
    e.specularIntensityMap ? "#define USE_SPECULAR_INTENSITYMAP" : "",
    e.roughnessMap ? "#define USE_ROUGHNESSMAP" : "",
    e.metalnessMap ? "#define USE_METALNESSMAP" : "",
    e.alphaMap ? "#define USE_ALPHAMAP" : "",
    e.alphaTest ? "#define USE_ALPHATEST" : "",
    e.alphaHash ? "#define USE_ALPHAHASH" : "",
    e.sheen ? "#define USE_SHEEN" : "",
    e.sheenColorMap ? "#define USE_SHEEN_COLORMAP" : "",
    e.sheenRoughnessMap ? "#define USE_SHEEN_ROUGHNESSMAP" : "",
    e.transmission ? "#define USE_TRANSMISSION" : "",
    e.transmissionMap ? "#define USE_TRANSMISSIONMAP" : "",
    e.thicknessMap ? "#define USE_THICKNESSMAP" : "",
    e.vertexTangents && e.flatShading === !1 ? "#define USE_TANGENT" : "",
    e.vertexColors || e.instancingColor || e.batchingColor ? "#define USE_COLOR" : "",
    e.vertexAlphas ? "#define USE_COLOR_ALPHA" : "",
    e.vertexUv1s ? "#define USE_UV1" : "",
    e.vertexUv2s ? "#define USE_UV2" : "",
    e.vertexUv3s ? "#define USE_UV3" : "",
    e.pointsUvs ? "#define USE_POINTS_UV" : "",
    e.gradientMap ? "#define USE_GRADIENTMAP" : "",
    e.flatShading ? "#define FLAT_SHADED" : "",
    e.doubleSided ? "#define DOUBLE_SIDED" : "",
    e.flipSided ? "#define FLIP_SIDED" : "",
    e.shadowMapEnabled ? "#define USE_SHADOWMAP" : "",
    e.shadowMapEnabled ? "#define " + l : "",
    e.premultipliedAlpha ? "#define PREMULTIPLIED_ALPHA" : "",
    e.numLightProbes > 0 ? "#define USE_LIGHT_PROBES" : "",
    e.decodeVideoTexture ? "#define DECODE_VIDEO_TEXTURE" : "",
    e.decodeVideoTextureEmissive ? "#define DECODE_VIDEO_TEXTURE_EMISSIVE" : "",
    e.logarithmicDepthBuffer ? "#define USE_LOGDEPTHBUF" : "",
    e.reverseDepthBuffer ? "#define USE_REVERSEDEPTHBUF" : "",
    "uniform mat4 viewMatrix;",
    "uniform vec3 cameraPosition;",
    "uniform bool isOrthographic;",
    e.toneMapping !== Un ? "#define TONE_MAPPING" : "",
    e.toneMapping !== Un ? Lt.tonemapping_pars_fragment : "",
    // this code is required here because it is used by the toneMapping() function defined below
    e.toneMapping !== Un ? Cp("toneMapping", e.toneMapping) : "",
    e.dithering ? "#define DITHERING" : "",
    e.opaque ? "#define OPAQUE" : "",
    Lt.colorspace_pars_fragment,
    // this code is required here because it is used by the various encoding/decoding function defined below
    Rp("linearToOutputTexel", e.outputColorSpace),
    Pp(),
    e.useDepthPacking ? "#define DEPTH_PACKING " + e.depthPacking : "",
    `
`
  ].filter(zi).join(`
`)), a = ua(a), a = Fo(a, e), a = Oo(a, e), o = ua(o), o = Fo(o, e), o = Oo(o, e), a = Bo(a), o = Bo(o), e.isRawShaderMaterial !== !0 && (E = `#version 300 es
`, p = [
    m,
    "#define attribute in",
    "#define varying out",
    "#define texture2D texture"
  ].join(`
`) + `
` + p, h = [
    "#define varying in",
    e.glslVersion === Wa ? "" : "layout(location = 0) out highp vec4 pc_fragColor;",
    e.glslVersion === Wa ? "" : "#define gl_FragColor pc_fragColor",
    "#define gl_FragDepthEXT gl_FragDepth",
    "#define texture2D texture",
    "#define textureCube texture",
    "#define texture2DProj textureProj",
    "#define texture2DLodEXT textureLod",
    "#define texture2DProjLodEXT textureProjLod",
    "#define textureCubeLodEXT textureLod",
    "#define texture2DGradEXT textureGrad",
    "#define texture2DProjGradEXT textureProjGrad",
    "#define textureCubeGradEXT textureGrad"
  ].join(`
`) + `
` + h);
  const b = E + p + a, S = E + h + o, L = Uo(s, s.VERTEX_SHADER, b), T = Uo(s, s.FRAGMENT_SHADER, S);
  s.attachShader(v, L), s.attachShader(v, T), e.index0AttributeName !== void 0 ? s.bindAttribLocation(v, 0, e.index0AttributeName) : e.morphTargets === !0 && s.bindAttribLocation(v, 0, "position"), s.linkProgram(v);
  function R(C) {
    if (i.debug.checkShaderErrors) {
      const H = s.getProgramInfoLog(v).trim(), z = s.getShaderInfoLog(L).trim(), G = s.getShaderInfoLog(T).trim();
      let j = !0, W = !0;
      if (s.getProgramParameter(v, s.LINK_STATUS) === !1)
        if (j = !1, typeof i.debug.onShaderError == "function")
          i.debug.onShaderError(s, v, L, T);
        else {
          const Q = No(s, L, "vertex"), V = No(s, T, "fragment");
          console.error(
            "THREE.WebGLProgram: Shader Error " + s.getError() + " - VALIDATE_STATUS " + s.getProgramParameter(v, s.VALIDATE_STATUS) + `

Material Name: ` + C.name + `
Material Type: ` + C.type + `

Program Info Log: ` + H + `
` + Q + `
` + V
          );
        }
      else H !== "" ? console.warn("THREE.WebGLProgram: Program Info Log:", H) : (z === "" || G === "") && (W = !1);
      W && (C.diagnostics = {
        runnable: j,
        programLog: H,
        vertexShader: {
          log: z,
          prefix: p
        },
        fragmentShader: {
          log: G,
          prefix: h
        }
      });
    }
    s.deleteShader(L), s.deleteShader(T), I = new Cs(s, v), y = Up(s, v);
  }
  let I;
  this.getUniforms = function() {
    return I === void 0 && R(this), I;
  };
  let y;
  this.getAttributes = function() {
    return y === void 0 && R(this), y;
  };
  let M = e.rendererExtensionParallelShaderCompile === !1;
  return this.isReady = function() {
    return M === !1 && (M = s.getProgramParameter(v, bp)), M;
  }, this.destroy = function() {
    n.releaseStatesOfProgram(this), s.deleteProgram(v), this.program = void 0;
  }, this.type = e.shaderType, this.name = e.shaderName, this.id = Ap++, this.cacheKey = t, this.usedTimes = 1, this.program = v, this.vertexShader = L, this.fragmentShader = T, this;
}
let Xp = 0;
class Yp {
  constructor() {
    this.shaderCache = /* @__PURE__ */ new Map(), this.materialCache = /* @__PURE__ */ new Map();
  }
  update(t) {
    const e = t.vertexShader, n = t.fragmentShader, s = this._getShaderStage(e), r = this._getShaderStage(n), a = this._getShaderCacheForMaterial(t);
    return a.has(s) === !1 && (a.add(s), s.usedTimes++), a.has(r) === !1 && (a.add(r), r.usedTimes++), this;
  }
  remove(t) {
    const e = this.materialCache.get(t);
    for (const n of e)
      n.usedTimes--, n.usedTimes === 0 && this.shaderCache.delete(n.code);
    return this.materialCache.delete(t), this;
  }
  getVertexShaderID(t) {
    return this._getShaderStage(t.vertexShader).id;
  }
  getFragmentShaderID(t) {
    return this._getShaderStage(t.fragmentShader).id;
  }
  dispose() {
    this.shaderCache.clear(), this.materialCache.clear();
  }
  _getShaderCacheForMaterial(t) {
    const e = this.materialCache;
    let n = e.get(t);
    return n === void 0 && (n = /* @__PURE__ */ new Set(), e.set(t, n)), n;
  }
  _getShaderStage(t) {
    const e = this.shaderCache;
    let n = e.get(t);
    return n === void 0 && (n = new $p(t), e.set(t, n)), n;
  }
}
class $p {
  constructor(t) {
    this.id = Xp++, this.code = t, this.usedTimes = 0;
  }
}
function qp(i, t, e, n, s, r, a) {
  const o = new xl(), l = new Yp(), c = /* @__PURE__ */ new Set(), u = [], d = s.logarithmicDepthBuffer, f = s.vertexTextures;
  let m = s.precision;
  const g = {
    MeshDepthMaterial: "depth",
    MeshDistanceMaterial: "distanceRGBA",
    MeshNormalMaterial: "normal",
    MeshBasicMaterial: "basic",
    MeshLambertMaterial: "lambert",
    MeshPhongMaterial: "phong",
    MeshToonMaterial: "toon",
    MeshStandardMaterial: "physical",
    MeshPhysicalMaterial: "physical",
    MeshMatcapMaterial: "matcap",
    LineBasicMaterial: "basic",
    LineDashedMaterial: "dashed",
    PointsMaterial: "points",
    ShadowMaterial: "shadow",
    SpriteMaterial: "sprite"
  };
  function v(y) {
    return c.add(y), y === 0 ? "uv" : `uv${y}`;
  }
  function p(y, M, C, H, z) {
    const G = H.fog, j = z.geometry, W = y.isMeshStandardMaterial ? H.environment : null, Q = (y.isMeshStandardMaterial ? e : t).get(y.envMap || W), V = Q && Q.mapping === Ns ? Q.image.height : null, st = g[y.type];
    y.precision !== null && (m = s.getMaxPrecision(y.precision), m !== y.precision && console.warn("THREE.WebGLProgram.getParameters:", y.precision, "not supported, using", m, "instead."));
    const ht = j.morphAttributes.position || j.morphAttributes.normal || j.morphAttributes.color, gt = ht !== void 0 ? ht.length : 0;
    let It = 0;
    j.morphAttributes.position !== void 0 && (It = 1), j.morphAttributes.normal !== void 0 && (It = 2), j.morphAttributes.color !== void 0 && (It = 3);
    let Kt, Y, tt, mt;
    if (st) {
      const $t = Le[st];
      Kt = $t.vertexShader, Y = $t.fragmentShader;
    } else
      Kt = y.vertexShader, Y = y.fragmentShader, l.update(y), tt = l.getVertexShaderID(y), mt = l.getFragmentShaderID(y);
    const rt = i.getRenderTarget(), Et = i.state.buffers.depth.getReversed(), Rt = z.isInstancedMesh === !0, Nt = z.isBatchedMesh === !0, re = !!y.map, zt = !!y.matcap, ce = !!Q, w = !!y.aoMap, ke = !!y.lightMap, Ft = !!y.bumpMap, Ot = !!y.normalMap, vt = !!y.displacementMap, ee = !!y.emissiveMap, xt = !!y.metalnessMap, A = !!y.roughnessMap, _ = y.anisotropy > 0, F = y.clearcoat > 0, $ = y.dispersion > 0, Z = y.iridescence > 0, X = y.sheen > 0, _t = y.transmission > 0, at = _ && !!y.anisotropyMap, ut = F && !!y.clearcoatMap, Ht = F && !!y.clearcoatNormalMap, J = F && !!y.clearcoatRoughnessMap, dt = Z && !!y.iridescenceMap, yt = Z && !!y.iridescenceThicknessMap, At = X && !!y.sheenColorMap, ft = X && !!y.sheenRoughnessMap, Bt = !!y.specularMap, Dt = !!y.specularColorMap, Jt = !!y.specularIntensityMap, D = _t && !!y.transmissionMap, nt = _t && !!y.thicknessMap, k = !!y.gradientMap, q = !!y.alphaMap, lt = y.alphaTest > 0, ot = !!y.alphaHash, Ct = !!y.extensions;
    let ae = Un;
    y.toneMapped && (rt === null || rt.isXRRenderTarget === !0) && (ae = i.toneMapping);
    const Se = {
      shaderID: st,
      shaderType: y.type,
      shaderName: y.name,
      vertexShader: Kt,
      fragmentShader: Y,
      defines: y.defines,
      customVertexShaderID: tt,
      customFragmentShaderID: mt,
      isRawShaderMaterial: y.isRawShaderMaterial === !0,
      glslVersion: y.glslVersion,
      precision: m,
      batching: Nt,
      batchingColor: Nt && z._colorsTexture !== null,
      instancing: Rt,
      instancingColor: Rt && z.instanceColor !== null,
      instancingMorph: Rt && z.morphTexture !== null,
      supportsVertexTextures: f,
      outputColorSpace: rt === null ? i.outputColorSpace : rt.isXRRenderTarget === !0 ? rt.texture.colorSpace : Ri,
      alphaToCoverage: !!y.alphaToCoverage,
      map: re,
      matcap: zt,
      envMap: ce,
      envMapMode: ce && Q.mapping,
      envMapCubeUVHeight: V,
      aoMap: w,
      lightMap: ke,
      bumpMap: Ft,
      normalMap: Ot,
      displacementMap: f && vt,
      emissiveMap: ee,
      normalMapObjectSpace: Ot && y.normalMapType === Tc,
      normalMapTangentSpace: Ot && y.normalMapType === fl,
      metalnessMap: xt,
      roughnessMap: A,
      anisotropy: _,
      anisotropyMap: at,
      clearcoat: F,
      clearcoatMap: ut,
      clearcoatNormalMap: Ht,
      clearcoatRoughnessMap: J,
      dispersion: $,
      iridescence: Z,
      iridescenceMap: dt,
      iridescenceThicknessMap: yt,
      sheen: X,
      sheenColorMap: At,
      sheenRoughnessMap: ft,
      specularMap: Bt,
      specularColorMap: Dt,
      specularIntensityMap: Jt,
      transmission: _t,
      transmissionMap: D,
      thicknessMap: nt,
      gradientMap: k,
      opaque: y.transparent === !1 && y.blending === Mi && y.alphaToCoverage === !1,
      alphaMap: q,
      alphaTest: lt,
      alphaHash: ot,
      combine: y.combine,
      //
      mapUv: re && v(y.map.channel),
      aoMapUv: w && v(y.aoMap.channel),
      lightMapUv: ke && v(y.lightMap.channel),
      bumpMapUv: Ft && v(y.bumpMap.channel),
      normalMapUv: Ot && v(y.normalMap.channel),
      displacementMapUv: vt && v(y.displacementMap.channel),
      emissiveMapUv: ee && v(y.emissiveMap.channel),
      metalnessMapUv: xt && v(y.metalnessMap.channel),
      roughnessMapUv: A && v(y.roughnessMap.channel),
      anisotropyMapUv: at && v(y.anisotropyMap.channel),
      clearcoatMapUv: ut && v(y.clearcoatMap.channel),
      clearcoatNormalMapUv: Ht && v(y.clearcoatNormalMap.channel),
      clearcoatRoughnessMapUv: J && v(y.clearcoatRoughnessMap.channel),
      iridescenceMapUv: dt && v(y.iridescenceMap.channel),
      iridescenceThicknessMapUv: yt && v(y.iridescenceThicknessMap.channel),
      sheenColorMapUv: At && v(y.sheenColorMap.channel),
      sheenRoughnessMapUv: ft && v(y.sheenRoughnessMap.channel),
      specularMapUv: Bt && v(y.specularMap.channel),
      specularColorMapUv: Dt && v(y.specularColorMap.channel),
      specularIntensityMapUv: Jt && v(y.specularIntensityMap.channel),
      transmissionMapUv: D && v(y.transmissionMap.channel),
      thicknessMapUv: nt && v(y.thicknessMap.channel),
      alphaMapUv: q && v(y.alphaMap.channel),
      //
      vertexTangents: !!j.attributes.tangent && (Ot || _),
      vertexColors: y.vertexColors,
      vertexAlphas: y.vertexColors === !0 && !!j.attributes.color && j.attributes.color.itemSize === 4,
      pointsUvs: z.isPoints === !0 && !!j.attributes.uv && (re || q),
      fog: !!G,
      useFog: y.fog === !0,
      fogExp2: !!G && G.isFogExp2,
      flatShading: y.flatShading === !0,
      sizeAttenuation: y.sizeAttenuation === !0,
      logarithmicDepthBuffer: d,
      reverseDepthBuffer: Et,
      skinning: z.isSkinnedMesh === !0,
      morphTargets: j.morphAttributes.position !== void 0,
      morphNormals: j.morphAttributes.normal !== void 0,
      morphColors: j.morphAttributes.color !== void 0,
      morphTargetsCount: gt,
      morphTextureStride: It,
      numDirLights: M.directional.length,
      numPointLights: M.point.length,
      numSpotLights: M.spot.length,
      numSpotLightMaps: M.spotLightMap.length,
      numRectAreaLights: M.rectArea.length,
      numHemiLights: M.hemi.length,
      numDirLightShadows: M.directionalShadowMap.length,
      numPointLightShadows: M.pointShadowMap.length,
      numSpotLightShadows: M.spotShadowMap.length,
      numSpotLightShadowsWithMaps: M.numSpotLightShadowsWithMaps,
      numLightProbes: M.numLightProbes,
      numClippingPlanes: a.numPlanes,
      numClipIntersection: a.numIntersection,
      dithering: y.dithering,
      shadowMapEnabled: i.shadowMap.enabled && C.length > 0,
      shadowMapType: i.shadowMap.type,
      toneMapping: ae,
      decodeVideoTexture: re && y.map.isVideoTexture === !0 && Wt.getTransfer(y.map.colorSpace) === jt,
      decodeVideoTextureEmissive: ee && y.emissiveMap.isVideoTexture === !0 && Wt.getTransfer(y.emissiveMap.colorSpace) === jt,
      premultipliedAlpha: y.premultipliedAlpha,
      doubleSided: y.side === Xe,
      flipSided: y.side === Ue,
      useDepthPacking: y.depthPacking >= 0,
      depthPacking: y.depthPacking || 0,
      index0AttributeName: y.index0AttributeName,
      extensionClipCullDistance: Ct && y.extensions.clipCullDistance === !0 && n.has("WEBGL_clip_cull_distance"),
      extensionMultiDraw: (Ct && y.extensions.multiDraw === !0 || Nt) && n.has("WEBGL_multi_draw"),
      rendererExtensionParallelShaderCompile: n.has("KHR_parallel_shader_compile"),
      customProgramCacheKey: y.customProgramCacheKey()
    };
    return Se.vertexUv1s = c.has(1), Se.vertexUv2s = c.has(2), Se.vertexUv3s = c.has(3), c.clear(), Se;
  }
  function h(y) {
    const M = [];
    if (y.shaderID ? M.push(y.shaderID) : (M.push(y.customVertexShaderID), M.push(y.customFragmentShaderID)), y.defines !== void 0)
      for (const C in y.defines)
        M.push(C), M.push(y.defines[C]);
    return y.isRawShaderMaterial === !1 && (E(M, y), b(M, y), M.push(i.outputColorSpace)), M.push(y.customProgramCacheKey), M.join();
  }
  function E(y, M) {
    y.push(M.precision), y.push(M.outputColorSpace), y.push(M.envMapMode), y.push(M.envMapCubeUVHeight), y.push(M.mapUv), y.push(M.alphaMapUv), y.push(M.lightMapUv), y.push(M.aoMapUv), y.push(M.bumpMapUv), y.push(M.normalMapUv), y.push(M.displacementMapUv), y.push(M.emissiveMapUv), y.push(M.metalnessMapUv), y.push(M.roughnessMapUv), y.push(M.anisotropyMapUv), y.push(M.clearcoatMapUv), y.push(M.clearcoatNormalMapUv), y.push(M.clearcoatRoughnessMapUv), y.push(M.iridescenceMapUv), y.push(M.iridescenceThicknessMapUv), y.push(M.sheenColorMapUv), y.push(M.sheenRoughnessMapUv), y.push(M.specularMapUv), y.push(M.specularColorMapUv), y.push(M.specularIntensityMapUv), y.push(M.transmissionMapUv), y.push(M.thicknessMapUv), y.push(M.combine), y.push(M.fogExp2), y.push(M.sizeAttenuation), y.push(M.morphTargetsCount), y.push(M.morphAttributeCount), y.push(M.numDirLights), y.push(M.numPointLights), y.push(M.numSpotLights), y.push(M.numSpotLightMaps), y.push(M.numHemiLights), y.push(M.numRectAreaLights), y.push(M.numDirLightShadows), y.push(M.numPointLightShadows), y.push(M.numSpotLightShadows), y.push(M.numSpotLightShadowsWithMaps), y.push(M.numLightProbes), y.push(M.shadowMapType), y.push(M.toneMapping), y.push(M.numClippingPlanes), y.push(M.numClipIntersection), y.push(M.depthPacking);
  }
  function b(y, M) {
    o.disableAll(), M.supportsVertexTextures && o.enable(0), M.instancing && o.enable(1), M.instancingColor && o.enable(2), M.instancingMorph && o.enable(3), M.matcap && o.enable(4), M.envMap && o.enable(5), M.normalMapObjectSpace && o.enable(6), M.normalMapTangentSpace && o.enable(7), M.clearcoat && o.enable(8), M.iridescence && o.enable(9), M.alphaTest && o.enable(10), M.vertexColors && o.enable(11), M.vertexAlphas && o.enable(12), M.vertexUv1s && o.enable(13), M.vertexUv2s && o.enable(14), M.vertexUv3s && o.enable(15), M.vertexTangents && o.enable(16), M.anisotropy && o.enable(17), M.alphaHash && o.enable(18), M.batching && o.enable(19), M.dispersion && o.enable(20), M.batchingColor && o.enable(21), y.push(o.mask), o.disableAll(), M.fog && o.enable(0), M.useFog && o.enable(1), M.flatShading && o.enable(2), M.logarithmicDepthBuffer && o.enable(3), M.reverseDepthBuffer && o.enable(4), M.skinning && o.enable(5), M.morphTargets && o.enable(6), M.morphNormals && o.enable(7), M.morphColors && o.enable(8), M.premultipliedAlpha && o.enable(9), M.shadowMapEnabled && o.enable(10), M.doubleSided && o.enable(11), M.flipSided && o.enable(12), M.useDepthPacking && o.enable(13), M.dithering && o.enable(14), M.transmission && o.enable(15), M.sheen && o.enable(16), M.opaque && o.enable(17), M.pointsUvs && o.enable(18), M.decodeVideoTexture && o.enable(19), M.decodeVideoTextureEmissive && o.enable(20), M.alphaToCoverage && o.enable(21), y.push(o.mask);
  }
  function S(y) {
    const M = g[y.type];
    let C;
    if (M) {
      const H = Le[M];
      C = ya.clone(H.uniforms);
    } else
      C = y.uniforms;
    return C;
  }
  function L(y, M) {
    let C;
    for (let H = 0, z = u.length; H < z; H++) {
      const G = u[H];
      if (G.cacheKey === M) {
        C = G, ++C.usedTimes;
        break;
      }
    }
    return C === void 0 && (C = new Wp(i, M, y, r), u.push(C)), C;
  }
  function T(y) {
    if (--y.usedTimes === 0) {
      const M = u.indexOf(y);
      u[M] = u[u.length - 1], u.pop(), y.destroy();
    }
  }
  function R(y) {
    l.remove(y);
  }
  function I() {
    l.dispose();
  }
  return {
    getParameters: p,
    getProgramCacheKey: h,
    getUniforms: S,
    acquireProgram: L,
    releaseProgram: T,
    releaseShaderCache: R,
    // Exposed for resource monitoring & error feedback via renderer.info:
    programs: u,
    dispose: I
  };
}
function jp() {
  let i = /* @__PURE__ */ new WeakMap();
  function t(a) {
    return i.has(a);
  }
  function e(a) {
    let o = i.get(a);
    return o === void 0 && (o = {}, i.set(a, o)), o;
  }
  function n(a) {
    i.delete(a);
  }
  function s(a, o, l) {
    i.get(a)[o] = l;
  }
  function r() {
    i = /* @__PURE__ */ new WeakMap();
  }
  return {
    has: t,
    get: e,
    remove: n,
    update: s,
    dispose: r
  };
}
function Zp(i, t) {
  return i.groupOrder !== t.groupOrder ? i.groupOrder - t.groupOrder : i.renderOrder !== t.renderOrder ? i.renderOrder - t.renderOrder : i.material.id !== t.material.id ? i.material.id - t.material.id : i.z !== t.z ? i.z - t.z : i.id - t.id;
}
function Ho(i, t) {
  return i.groupOrder !== t.groupOrder ? i.groupOrder - t.groupOrder : i.renderOrder !== t.renderOrder ? i.renderOrder - t.renderOrder : i.z !== t.z ? t.z - i.z : i.id - t.id;
}
function ko() {
  const i = [];
  let t = 0;
  const e = [], n = [], s = [];
  function r() {
    t = 0, e.length = 0, n.length = 0, s.length = 0;
  }
  function a(d, f, m, g, v, p) {
    let h = i[t];
    return h === void 0 ? (h = {
      id: d.id,
      object: d,
      geometry: f,
      material: m,
      groupOrder: g,
      renderOrder: d.renderOrder,
      z: v,
      group: p
    }, i[t] = h) : (h.id = d.id, h.object = d, h.geometry = f, h.material = m, h.groupOrder = g, h.renderOrder = d.renderOrder, h.z = v, h.group = p), t++, h;
  }
  function o(d, f, m, g, v, p) {
    const h = a(d, f, m, g, v, p);
    m.transmission > 0 ? n.push(h) : m.transparent === !0 ? s.push(h) : e.push(h);
  }
  function l(d, f, m, g, v, p) {
    const h = a(d, f, m, g, v, p);
    m.transmission > 0 ? n.unshift(h) : m.transparent === !0 ? s.unshift(h) : e.unshift(h);
  }
  function c(d, f) {
    e.length > 1 && e.sort(d || Zp), n.length > 1 && n.sort(f || Ho), s.length > 1 && s.sort(f || Ho);
  }
  function u() {
    for (let d = t, f = i.length; d < f; d++) {
      const m = i[d];
      if (m.id === null) break;
      m.id = null, m.object = null, m.geometry = null, m.material = null, m.group = null;
    }
  }
  return {
    opaque: e,
    transmissive: n,
    transparent: s,
    init: r,
    push: o,
    unshift: l,
    finish: u,
    sort: c
  };
}
function Kp() {
  let i = /* @__PURE__ */ new WeakMap();
  function t(n, s) {
    const r = i.get(n);
    let a;
    return r === void 0 ? (a = new ko(), i.set(n, [a])) : s >= r.length ? (a = new ko(), r.push(a)) : a = r[s], a;
  }
  function e() {
    i = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: t,
    dispose: e
  };
}
function Jp() {
  const i = {};
  return {
    get: function(t) {
      if (i[t.id] !== void 0)
        return i[t.id];
      let e;
      switch (t.type) {
        case "DirectionalLight":
          e = {
            direction: new P(),
            color: new Vt()
          };
          break;
        case "SpotLight":
          e = {
            position: new P(),
            direction: new P(),
            color: new Vt(),
            distance: 0,
            coneCos: 0,
            penumbraCos: 0,
            decay: 0
          };
          break;
        case "PointLight":
          e = {
            position: new P(),
            color: new Vt(),
            distance: 0,
            decay: 0
          };
          break;
        case "HemisphereLight":
          e = {
            direction: new P(),
            skyColor: new Vt(),
            groundColor: new Vt()
          };
          break;
        case "RectAreaLight":
          e = {
            color: new Vt(),
            position: new P(),
            halfWidth: new P(),
            halfHeight: new P()
          };
          break;
      }
      return i[t.id] = e, e;
    }
  };
}
function Qp() {
  const i = {};
  return {
    get: function(t) {
      if (i[t.id] !== void 0)
        return i[t.id];
      let e;
      switch (t.type) {
        case "DirectionalLight":
          e = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new bt()
          };
          break;
        case "SpotLight":
          e = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new bt()
          };
          break;
        case "PointLight":
          e = {
            shadowIntensity: 1,
            shadowBias: 0,
            shadowNormalBias: 0,
            shadowRadius: 1,
            shadowMapSize: new bt(),
            shadowCameraNear: 1,
            shadowCameraFar: 1e3
          };
          break;
      }
      return i[t.id] = e, e;
    }
  };
}
let tm = 0;
function em(i, t) {
  return (t.castShadow ? 2 : 0) - (i.castShadow ? 2 : 0) + (t.map ? 1 : 0) - (i.map ? 1 : 0);
}
function nm(i) {
  const t = new Jp(), e = Qp(), n = {
    version: 0,
    hash: {
      directionalLength: -1,
      pointLength: -1,
      spotLength: -1,
      rectAreaLength: -1,
      hemiLength: -1,
      numDirectionalShadows: -1,
      numPointShadows: -1,
      numSpotShadows: -1,
      numSpotMaps: -1,
      numLightProbes: -1
    },
    ambient: [0, 0, 0],
    probe: [],
    directional: [],
    directionalShadow: [],
    directionalShadowMap: [],
    directionalShadowMatrix: [],
    spot: [],
    spotLightMap: [],
    spotShadow: [],
    spotShadowMap: [],
    spotLightMatrix: [],
    rectArea: [],
    rectAreaLTC1: null,
    rectAreaLTC2: null,
    point: [],
    pointShadow: [],
    pointShadowMap: [],
    pointShadowMatrix: [],
    hemi: [],
    numSpotLightShadowsWithMaps: 0,
    numLightProbes: 0
  };
  for (let c = 0; c < 9; c++) n.probe.push(new P());
  const s = new P(), r = new ne(), a = new ne();
  function o(c) {
    let u = 0, d = 0, f = 0;
    for (let y = 0; y < 9; y++) n.probe[y].set(0, 0, 0);
    let m = 0, g = 0, v = 0, p = 0, h = 0, E = 0, b = 0, S = 0, L = 0, T = 0, R = 0;
    c.sort(em);
    for (let y = 0, M = c.length; y < M; y++) {
      const C = c[y], H = C.color, z = C.intensity, G = C.distance, j = C.shadow && C.shadow.map ? C.shadow.map.texture : null;
      if (C.isAmbientLight)
        u += H.r * z, d += H.g * z, f += H.b * z;
      else if (C.isLightProbe) {
        for (let W = 0; W < 9; W++)
          n.probe[W].addScaledVector(C.sh.coefficients[W], z);
        R++;
      } else if (C.isDirectionalLight) {
        const W = t.get(C);
        if (W.color.copy(C.color).multiplyScalar(C.intensity), C.castShadow) {
          const Q = C.shadow, V = e.get(C);
          V.shadowIntensity = Q.intensity, V.shadowBias = Q.bias, V.shadowNormalBias = Q.normalBias, V.shadowRadius = Q.radius, V.shadowMapSize = Q.mapSize, n.directionalShadow[m] = V, n.directionalShadowMap[m] = j, n.directionalShadowMatrix[m] = C.shadow.matrix, E++;
        }
        n.directional[m] = W, m++;
      } else if (C.isSpotLight) {
        const W = t.get(C);
        W.position.setFromMatrixPosition(C.matrixWorld), W.color.copy(H).multiplyScalar(z), W.distance = G, W.coneCos = Math.cos(C.angle), W.penumbraCos = Math.cos(C.angle * (1 - C.penumbra)), W.decay = C.decay, n.spot[v] = W;
        const Q = C.shadow;
        if (C.map && (n.spotLightMap[L] = C.map, L++, Q.updateMatrices(C), C.castShadow && T++), n.spotLightMatrix[v] = Q.matrix, C.castShadow) {
          const V = e.get(C);
          V.shadowIntensity = Q.intensity, V.shadowBias = Q.bias, V.shadowNormalBias = Q.normalBias, V.shadowRadius = Q.radius, V.shadowMapSize = Q.mapSize, n.spotShadow[v] = V, n.spotShadowMap[v] = j, S++;
        }
        v++;
      } else if (C.isRectAreaLight) {
        const W = t.get(C);
        W.color.copy(H).multiplyScalar(z), W.halfWidth.set(C.width * 0.5, 0, 0), W.halfHeight.set(0, C.height * 0.5, 0), n.rectArea[p] = W, p++;
      } else if (C.isPointLight) {
        const W = t.get(C);
        if (W.color.copy(C.color).multiplyScalar(C.intensity), W.distance = C.distance, W.decay = C.decay, C.castShadow) {
          const Q = C.shadow, V = e.get(C);
          V.shadowIntensity = Q.intensity, V.shadowBias = Q.bias, V.shadowNormalBias = Q.normalBias, V.shadowRadius = Q.radius, V.shadowMapSize = Q.mapSize, V.shadowCameraNear = Q.camera.near, V.shadowCameraFar = Q.camera.far, n.pointShadow[g] = V, n.pointShadowMap[g] = j, n.pointShadowMatrix[g] = C.shadow.matrix, b++;
        }
        n.point[g] = W, g++;
      } else if (C.isHemisphereLight) {
        const W = t.get(C);
        W.skyColor.copy(C.color).multiplyScalar(z), W.groundColor.copy(C.groundColor).multiplyScalar(z), n.hemi[h] = W, h++;
      }
    }
    p > 0 && (i.has("OES_texture_float_linear") === !0 ? (n.rectAreaLTC1 = et.LTC_FLOAT_1, n.rectAreaLTC2 = et.LTC_FLOAT_2) : (n.rectAreaLTC1 = et.LTC_HALF_1, n.rectAreaLTC2 = et.LTC_HALF_2)), n.ambient[0] = u, n.ambient[1] = d, n.ambient[2] = f;
    const I = n.hash;
    (I.directionalLength !== m || I.pointLength !== g || I.spotLength !== v || I.rectAreaLength !== p || I.hemiLength !== h || I.numDirectionalShadows !== E || I.numPointShadows !== b || I.numSpotShadows !== S || I.numSpotMaps !== L || I.numLightProbes !== R) && (n.directional.length = m, n.spot.length = v, n.rectArea.length = p, n.point.length = g, n.hemi.length = h, n.directionalShadow.length = E, n.directionalShadowMap.length = E, n.pointShadow.length = b, n.pointShadowMap.length = b, n.spotShadow.length = S, n.spotShadowMap.length = S, n.directionalShadowMatrix.length = E, n.pointShadowMatrix.length = b, n.spotLightMatrix.length = S + L - T, n.spotLightMap.length = L, n.numSpotLightShadowsWithMaps = T, n.numLightProbes = R, I.directionalLength = m, I.pointLength = g, I.spotLength = v, I.rectAreaLength = p, I.hemiLength = h, I.numDirectionalShadows = E, I.numPointShadows = b, I.numSpotShadows = S, I.numSpotMaps = L, I.numLightProbes = R, n.version = tm++);
  }
  function l(c, u) {
    let d = 0, f = 0, m = 0, g = 0, v = 0;
    const p = u.matrixWorldInverse;
    for (let h = 0, E = c.length; h < E; h++) {
      const b = c[h];
      if (b.isDirectionalLight) {
        const S = n.directional[d];
        S.direction.setFromMatrixPosition(b.matrixWorld), s.setFromMatrixPosition(b.target.matrixWorld), S.direction.sub(s), S.direction.transformDirection(p), d++;
      } else if (b.isSpotLight) {
        const S = n.spot[m];
        S.position.setFromMatrixPosition(b.matrixWorld), S.position.applyMatrix4(p), S.direction.setFromMatrixPosition(b.matrixWorld), s.setFromMatrixPosition(b.target.matrixWorld), S.direction.sub(s), S.direction.transformDirection(p), m++;
      } else if (b.isRectAreaLight) {
        const S = n.rectArea[g];
        S.position.setFromMatrixPosition(b.matrixWorld), S.position.applyMatrix4(p), a.identity(), r.copy(b.matrixWorld), r.premultiply(p), a.extractRotation(r), S.halfWidth.set(b.width * 0.5, 0, 0), S.halfHeight.set(0, b.height * 0.5, 0), S.halfWidth.applyMatrix4(a), S.halfHeight.applyMatrix4(a), g++;
      } else if (b.isPointLight) {
        const S = n.point[f];
        S.position.setFromMatrixPosition(b.matrixWorld), S.position.applyMatrix4(p), f++;
      } else if (b.isHemisphereLight) {
        const S = n.hemi[v];
        S.direction.setFromMatrixPosition(b.matrixWorld), S.direction.transformDirection(p), v++;
      }
    }
  }
  return {
    setup: o,
    setupView: l,
    state: n
  };
}
function Vo(i) {
  const t = new nm(i), e = [], n = [];
  function s(u) {
    c.camera = u, e.length = 0, n.length = 0;
  }
  function r(u) {
    e.push(u);
  }
  function a(u) {
    n.push(u);
  }
  function o() {
    t.setup(e);
  }
  function l(u) {
    t.setupView(e, u);
  }
  const c = {
    lightsArray: e,
    shadowsArray: n,
    camera: null,
    lights: t,
    transmissionRenderTarget: {}
  };
  return {
    init: s,
    state: c,
    setupLights: o,
    setupLightsView: l,
    pushLight: r,
    pushShadow: a
  };
}
function im(i) {
  let t = /* @__PURE__ */ new WeakMap();
  function e(s, r = 0) {
    const a = t.get(s);
    let o;
    return a === void 0 ? (o = new Vo(i), t.set(s, [o])) : r >= a.length ? (o = new Vo(i), a.push(o)) : o = a[r], o;
  }
  function n() {
    t = /* @__PURE__ */ new WeakMap();
  }
  return {
    get: e,
    dispose: n
  };
}
const sm = `void main() {
	gl_Position = vec4( position, 1.0 );
}`, rm = `uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
#include <packing>
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = unpackRGBATo2Half( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ) );
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = unpackRGBAToDepth( texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ) );
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( squared_mean - mean * mean );
	gl_FragColor = pack2HalfToRGBA( vec2( mean, std_dev ) );
}`;
function am(i, t, e) {
  let n = new Tl();
  const s = new bt(), r = new bt(), a = new te(), o = new Rh({ depthPacking: Ac }), l = new Ch(), c = {}, u = e.maxTextureSize, d = { [In]: Ue, [Ue]: In, [Xe]: Xe }, f = new En({
    defines: {
      VSM_SAMPLES: 8
    },
    uniforms: {
      shadow_pass: { value: null },
      resolution: { value: new bt() },
      radius: { value: 4 }
    },
    vertexShader: sm,
    fragmentShader: rm
  }), m = f.clone();
  m.defines.HORIZONTAL_PASS = 1;
  const g = new Ce();
  g.setAttribute(
    "position",
    new en(
      new Float32Array([-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5]),
      3
    )
  );
  const v = new be(g, f), p = this;
  this.enabled = !1, this.autoUpdate = !0, this.needsUpdate = !1, this.type = tl;
  let h = this.type;
  this.render = function(T, R, I) {
    if (p.enabled === !1 || p.autoUpdate === !1 && p.needsUpdate === !1 || T.length === 0) return;
    const y = i.getRenderTarget(), M = i.getActiveCubeFace(), C = i.getActiveMipmapLevel(), H = i.state;
    H.setBlending(Ln), H.buffers.color.setClear(1, 1, 1, 1), H.buffers.depth.setTest(!0), H.setScissorTest(!1);
    const z = h !== mn && this.type === mn, G = h === mn && this.type !== mn;
    for (let j = 0, W = T.length; j < W; j++) {
      const Q = T[j], V = Q.shadow;
      if (V === void 0) {
        console.warn("THREE.WebGLShadowMap:", Q, "has no shadow.");
        continue;
      }
      if (V.autoUpdate === !1 && V.needsUpdate === !1) continue;
      s.copy(V.mapSize);
      const st = V.getFrameExtents();
      if (s.multiply(st), r.copy(V.mapSize), (s.x > u || s.y > u) && (s.x > u && (r.x = Math.floor(u / st.x), s.x = r.x * st.x, V.mapSize.x = r.x), s.y > u && (r.y = Math.floor(u / st.y), s.y = r.y * st.y, V.mapSize.y = r.y)), V.map === null || z === !0 || G === !0) {
        const gt = this.type !== mn ? { minFilter: tn, magFilter: tn } : {};
        V.map !== null && V.map.dispose(), V.map = new Zn(s.x, s.y, gt), V.map.texture.name = Q.name + ".shadowMap", V.camera.updateProjectionMatrix();
      }
      i.setRenderTarget(V.map), i.clear();
      const ht = V.getViewportCount();
      for (let gt = 0; gt < ht; gt++) {
        const It = V.getViewport(gt);
        a.set(
          r.x * It.x,
          r.y * It.y,
          r.x * It.z,
          r.y * It.w
        ), H.viewport(a), V.updateMatrices(Q, gt), n = V.getFrustum(), S(R, I, V.camera, Q, this.type);
      }
      V.isPointLightShadow !== !0 && this.type === mn && E(V, I), V.needsUpdate = !1;
    }
    h = this.type, p.needsUpdate = !1, i.setRenderTarget(y, M, C);
  };
  function E(T, R) {
    const I = t.update(v);
    f.defines.VSM_SAMPLES !== T.blurSamples && (f.defines.VSM_SAMPLES = T.blurSamples, m.defines.VSM_SAMPLES = T.blurSamples, f.needsUpdate = !0, m.needsUpdate = !0), T.mapPass === null && (T.mapPass = new Zn(s.x, s.y)), f.uniforms.shadow_pass.value = T.map.texture, f.uniforms.resolution.value = T.mapSize, f.uniforms.radius.value = T.radius, i.setRenderTarget(T.mapPass), i.clear(), i.renderBufferDirect(R, null, I, f, v, null), m.uniforms.shadow_pass.value = T.mapPass.texture, m.uniforms.resolution.value = T.mapSize, m.uniforms.radius.value = T.radius, i.setRenderTarget(T.map), i.clear(), i.renderBufferDirect(R, null, I, m, v, null);
  }
  function b(T, R, I, y) {
    let M = null;
    const C = I.isPointLight === !0 ? T.customDistanceMaterial : T.customDepthMaterial;
    if (C !== void 0)
      M = C;
    else if (M = I.isPointLight === !0 ? l : o, i.localClippingEnabled && R.clipShadows === !0 && Array.isArray(R.clippingPlanes) && R.clippingPlanes.length !== 0 || R.displacementMap && R.displacementScale !== 0 || R.alphaMap && R.alphaTest > 0 || R.map && R.alphaTest > 0) {
      const H = M.uuid, z = R.uuid;
      let G = c[H];
      G === void 0 && (G = {}, c[H] = G);
      let j = G[z];
      j === void 0 && (j = M.clone(), G[z] = j, R.addEventListener("dispose", L)), M = j;
    }
    if (M.visible = R.visible, M.wireframe = R.wireframe, y === mn ? M.side = R.shadowSide !== null ? R.shadowSide : R.side : M.side = R.shadowSide !== null ? R.shadowSide : d[R.side], M.alphaMap = R.alphaMap, M.alphaTest = R.alphaTest, M.map = R.map, M.clipShadows = R.clipShadows, M.clippingPlanes = R.clippingPlanes, M.clipIntersection = R.clipIntersection, M.displacementMap = R.displacementMap, M.displacementScale = R.displacementScale, M.displacementBias = R.displacementBias, M.wireframeLinewidth = R.wireframeLinewidth, M.linewidth = R.linewidth, I.isPointLight === !0 && M.isMeshDistanceMaterial === !0) {
      const H = i.properties.get(M);
      H.light = I;
    }
    return M;
  }
  function S(T, R, I, y, M) {
    if (T.visible === !1) return;
    if (T.layers.test(R.layers) && (T.isMesh || T.isLine || T.isPoints) && (T.castShadow || T.receiveShadow && M === mn) && (!T.frustumCulled || n.intersectsObject(T))) {
      T.modelViewMatrix.multiplyMatrices(I.matrixWorldInverse, T.matrixWorld);
      const z = t.update(T), G = T.material;
      if (Array.isArray(G)) {
        const j = z.groups;
        for (let W = 0, Q = j.length; W < Q; W++) {
          const V = j[W], st = G[V.materialIndex];
          if (st && st.visible) {
            const ht = b(T, st, y, M);
            T.onBeforeShadow(i, T, R, I, z, ht, V), i.renderBufferDirect(I, null, z, ht, T, V), T.onAfterShadow(i, T, R, I, z, ht, V);
          }
        }
      } else if (G.visible) {
        const j = b(T, G, y, M);
        T.onBeforeShadow(i, T, R, I, z, j, null), i.renderBufferDirect(I, null, z, j, T, null), T.onAfterShadow(i, T, R, I, z, j, null);
      }
    }
    const H = T.children;
    for (let z = 0, G = H.length; z < G; z++)
      S(H[z], R, I, y, M);
  }
  function L(T) {
    T.target.removeEventListener("dispose", L);
    for (const I in c) {
      const y = c[I], M = T.target.uuid;
      M in y && (y[M].dispose(), delete y[M]);
    }
  }
}
const om = {
  [Tr]: wr,
  [Rr]: Dr,
  [Cr]: Lr,
  [Ei]: Pr,
  [wr]: Tr,
  [Dr]: Rr,
  [Lr]: Cr,
  [Pr]: Ei
};
function lm(i, t) {
  function e() {
    let D = !1;
    const nt = new te();
    let k = null;
    const q = new te(0, 0, 0, 0);
    return {
      setMask: function(lt) {
        k !== lt && !D && (i.colorMask(lt, lt, lt, lt), k = lt);
      },
      setLocked: function(lt) {
        D = lt;
      },
      setClear: function(lt, ot, Ct, ae, Se) {
        Se === !0 && (lt *= ae, ot *= ae, Ct *= ae), nt.set(lt, ot, Ct, ae), q.equals(nt) === !1 && (i.clearColor(lt, ot, Ct, ae), q.copy(nt));
      },
      reset: function() {
        D = !1, k = null, q.set(-1, 0, 0, 0);
      }
    };
  }
  function n() {
    let D = !1, nt = !1, k = null, q = null, lt = null;
    return {
      setReversed: function(ot) {
        if (nt !== ot) {
          const Ct = t.get("EXT_clip_control");
          nt ? Ct.clipControlEXT(Ct.LOWER_LEFT_EXT, Ct.ZERO_TO_ONE_EXT) : Ct.clipControlEXT(Ct.LOWER_LEFT_EXT, Ct.NEGATIVE_ONE_TO_ONE_EXT);
          const ae = lt;
          lt = null, this.setClear(ae);
        }
        nt = ot;
      },
      getReversed: function() {
        return nt;
      },
      setTest: function(ot) {
        ot ? rt(i.DEPTH_TEST) : Et(i.DEPTH_TEST);
      },
      setMask: function(ot) {
        k !== ot && !D && (i.depthMask(ot), k = ot);
      },
      setFunc: function(ot) {
        if (nt && (ot = om[ot]), q !== ot) {
          switch (ot) {
            case Tr:
              i.depthFunc(i.NEVER);
              break;
            case wr:
              i.depthFunc(i.ALWAYS);
              break;
            case Rr:
              i.depthFunc(i.LESS);
              break;
            case Ei:
              i.depthFunc(i.LEQUAL);
              break;
            case Cr:
              i.depthFunc(i.EQUAL);
              break;
            case Pr:
              i.depthFunc(i.GEQUAL);
              break;
            case Dr:
              i.depthFunc(i.GREATER);
              break;
            case Lr:
              i.depthFunc(i.NOTEQUAL);
              break;
            default:
              i.depthFunc(i.LEQUAL);
          }
          q = ot;
        }
      },
      setLocked: function(ot) {
        D = ot;
      },
      setClear: function(ot) {
        lt !== ot && (nt && (ot = 1 - ot), i.clearDepth(ot), lt = ot);
      },
      reset: function() {
        D = !1, k = null, q = null, lt = null, nt = !1;
      }
    };
  }
  function s() {
    let D = !1, nt = null, k = null, q = null, lt = null, ot = null, Ct = null, ae = null, Se = null;
    return {
      setTest: function($t) {
        D || ($t ? rt(i.STENCIL_TEST) : Et(i.STENCIL_TEST));
      },
      setMask: function($t) {
        nt !== $t && !D && (i.stencilMask($t), nt = $t);
      },
      setFunc: function($t, Ye, ln) {
        (k !== $t || q !== Ye || lt !== ln) && (i.stencilFunc($t, Ye, ln), k = $t, q = Ye, lt = ln);
      },
      setOp: function($t, Ye, ln) {
        (ot !== $t || Ct !== Ye || ae !== ln) && (i.stencilOp($t, Ye, ln), ot = $t, Ct = Ye, ae = ln);
      },
      setLocked: function($t) {
        D = $t;
      },
      setClear: function($t) {
        Se !== $t && (i.clearStencil($t), Se = $t);
      },
      reset: function() {
        D = !1, nt = null, k = null, q = null, lt = null, ot = null, Ct = null, ae = null, Se = null;
      }
    };
  }
  const r = new e(), a = new n(), o = new s(), l = /* @__PURE__ */ new WeakMap(), c = /* @__PURE__ */ new WeakMap();
  let u = {}, d = {}, f = /* @__PURE__ */ new WeakMap(), m = [], g = null, v = !1, p = null, h = null, E = null, b = null, S = null, L = null, T = null, R = new Vt(0, 0, 0), I = 0, y = !1, M = null, C = null, H = null, z = null, G = null;
  const j = i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
  let W = !1, Q = 0;
  const V = i.getParameter(i.VERSION);
  V.indexOf("WebGL") !== -1 ? (Q = parseFloat(/^WebGL (\d)/.exec(V)[1]), W = Q >= 1) : V.indexOf("OpenGL ES") !== -1 && (Q = parseFloat(/^OpenGL ES (\d)/.exec(V)[1]), W = Q >= 2);
  let st = null, ht = {};
  const gt = i.getParameter(i.SCISSOR_BOX), It = i.getParameter(i.VIEWPORT), Kt = new te().fromArray(gt), Y = new te().fromArray(It);
  function tt(D, nt, k, q) {
    const lt = new Uint8Array(4), ot = i.createTexture();
    i.bindTexture(D, ot), i.texParameteri(D, i.TEXTURE_MIN_FILTER, i.NEAREST), i.texParameteri(D, i.TEXTURE_MAG_FILTER, i.NEAREST);
    for (let Ct = 0; Ct < k; Ct++)
      D === i.TEXTURE_3D || D === i.TEXTURE_2D_ARRAY ? i.texImage3D(nt, 0, i.RGBA, 1, 1, q, 0, i.RGBA, i.UNSIGNED_BYTE, lt) : i.texImage2D(nt + Ct, 0, i.RGBA, 1, 1, 0, i.RGBA, i.UNSIGNED_BYTE, lt);
    return ot;
  }
  const mt = {};
  mt[i.TEXTURE_2D] = tt(i.TEXTURE_2D, i.TEXTURE_2D, 1), mt[i.TEXTURE_CUBE_MAP] = tt(i.TEXTURE_CUBE_MAP, i.TEXTURE_CUBE_MAP_POSITIVE_X, 6), mt[i.TEXTURE_2D_ARRAY] = tt(i.TEXTURE_2D_ARRAY, i.TEXTURE_2D_ARRAY, 1, 1), mt[i.TEXTURE_3D] = tt(i.TEXTURE_3D, i.TEXTURE_3D, 1, 1), r.setClear(0, 0, 0, 1), a.setClear(1), o.setClear(0), rt(i.DEPTH_TEST), a.setFunc(Ei), Ft(!1), Ot(za), rt(i.CULL_FACE), w(Ln);
  function rt(D) {
    u[D] !== !0 && (i.enable(D), u[D] = !0);
  }
  function Et(D) {
    u[D] !== !1 && (i.disable(D), u[D] = !1);
  }
  function Rt(D, nt) {
    return d[D] !== nt ? (i.bindFramebuffer(D, nt), d[D] = nt, D === i.DRAW_FRAMEBUFFER && (d[i.FRAMEBUFFER] = nt), D === i.FRAMEBUFFER && (d[i.DRAW_FRAMEBUFFER] = nt), !0) : !1;
  }
  function Nt(D, nt) {
    let k = m, q = !1;
    if (D) {
      k = f.get(nt), k === void 0 && (k = [], f.set(nt, k));
      const lt = D.textures;
      if (k.length !== lt.length || k[0] !== i.COLOR_ATTACHMENT0) {
        for (let ot = 0, Ct = lt.length; ot < Ct; ot++)
          k[ot] = i.COLOR_ATTACHMENT0 + ot;
        k.length = lt.length, q = !0;
      }
    } else
      k[0] !== i.BACK && (k[0] = i.BACK, q = !0);
    q && i.drawBuffers(k);
  }
  function re(D) {
    return g !== D ? (i.useProgram(D), g = D, !0) : !1;
  }
  const zt = {
    [Wn]: i.FUNC_ADD,
    [Kl]: i.FUNC_SUBTRACT,
    [Jl]: i.FUNC_REVERSE_SUBTRACT
  };
  zt[Ql] = i.MIN, zt[tc] = i.MAX;
  const ce = {
    [ec]: i.ZERO,
    [nc]: i.ONE,
    [ic]: i.SRC_COLOR,
    [br]: i.SRC_ALPHA,
    [cc]: i.SRC_ALPHA_SATURATE,
    [oc]: i.DST_COLOR,
    [rc]: i.DST_ALPHA,
    [sc]: i.ONE_MINUS_SRC_COLOR,
    [Ar]: i.ONE_MINUS_SRC_ALPHA,
    [lc]: i.ONE_MINUS_DST_COLOR,
    [ac]: i.ONE_MINUS_DST_ALPHA,
    [hc]: i.CONSTANT_COLOR,
    [uc]: i.ONE_MINUS_CONSTANT_COLOR,
    [dc]: i.CONSTANT_ALPHA,
    [fc]: i.ONE_MINUS_CONSTANT_ALPHA
  };
  function w(D, nt, k, q, lt, ot, Ct, ae, Se, $t) {
    if (D === Ln) {
      v === !0 && (Et(i.BLEND), v = !1);
      return;
    }
    if (v === !1 && (rt(i.BLEND), v = !0), D !== Zl) {
      if (D !== p || $t !== y) {
        if ((h !== Wn || S !== Wn) && (i.blendEquation(i.FUNC_ADD), h = Wn, S = Wn), $t)
          switch (D) {
            case Mi:
              i.blendFuncSeparate(i.ONE, i.ONE_MINUS_SRC_ALPHA, i.ONE, i.ONE_MINUS_SRC_ALPHA);
              break;
            case Ha:
              i.blendFunc(i.ONE, i.ONE);
              break;
            case ka:
              i.blendFuncSeparate(i.ZERO, i.ONE_MINUS_SRC_COLOR, i.ZERO, i.ONE);
              break;
            case Va:
              i.blendFuncSeparate(i.ZERO, i.SRC_COLOR, i.ZERO, i.SRC_ALPHA);
              break;
            default:
              console.error("THREE.WebGLState: Invalid blending: ", D);
              break;
          }
        else
          switch (D) {
            case Mi:
              i.blendFuncSeparate(i.SRC_ALPHA, i.ONE_MINUS_SRC_ALPHA, i.ONE, i.ONE_MINUS_SRC_ALPHA);
              break;
            case Ha:
              i.blendFunc(i.SRC_ALPHA, i.ONE);
              break;
            case ka:
              i.blendFuncSeparate(i.ZERO, i.ONE_MINUS_SRC_COLOR, i.ZERO, i.ONE);
              break;
            case Va:
              i.blendFunc(i.ZERO, i.SRC_COLOR);
              break;
            default:
              console.error("THREE.WebGLState: Invalid blending: ", D);
              break;
          }
        E = null, b = null, L = null, T = null, R.set(0, 0, 0), I = 0, p = D, y = $t;
      }
      return;
    }
    lt = lt || nt, ot = ot || k, Ct = Ct || q, (nt !== h || lt !== S) && (i.blendEquationSeparate(zt[nt], zt[lt]), h = nt, S = lt), (k !== E || q !== b || ot !== L || Ct !== T) && (i.blendFuncSeparate(ce[k], ce[q], ce[ot], ce[Ct]), E = k, b = q, L = ot, T = Ct), (ae.equals(R) === !1 || Se !== I) && (i.blendColor(ae.r, ae.g, ae.b, Se), R.copy(ae), I = Se), p = D, y = !1;
  }
  function ke(D, nt) {
    D.side === Xe ? Et(i.CULL_FACE) : rt(i.CULL_FACE);
    let k = D.side === Ue;
    nt && (k = !k), Ft(k), D.blending === Mi && D.transparent === !1 ? w(Ln) : w(D.blending, D.blendEquation, D.blendSrc, D.blendDst, D.blendEquationAlpha, D.blendSrcAlpha, D.blendDstAlpha, D.blendColor, D.blendAlpha, D.premultipliedAlpha), a.setFunc(D.depthFunc), a.setTest(D.depthTest), a.setMask(D.depthWrite), r.setMask(D.colorWrite);
    const q = D.stencilWrite;
    o.setTest(q), q && (o.setMask(D.stencilWriteMask), o.setFunc(D.stencilFunc, D.stencilRef, D.stencilFuncMask), o.setOp(D.stencilFail, D.stencilZFail, D.stencilZPass)), ee(D.polygonOffset, D.polygonOffsetFactor, D.polygonOffsetUnits), D.alphaToCoverage === !0 ? rt(i.SAMPLE_ALPHA_TO_COVERAGE) : Et(i.SAMPLE_ALPHA_TO_COVERAGE);
  }
  function Ft(D) {
    M !== D && (D ? i.frontFace(i.CW) : i.frontFace(i.CCW), M = D);
  }
  function Ot(D) {
    D !== $l ? (rt(i.CULL_FACE), D !== C && (D === za ? i.cullFace(i.BACK) : D === ql ? i.cullFace(i.FRONT) : i.cullFace(i.FRONT_AND_BACK))) : Et(i.CULL_FACE), C = D;
  }
  function vt(D) {
    D !== H && (W && i.lineWidth(D), H = D);
  }
  function ee(D, nt, k) {
    D ? (rt(i.POLYGON_OFFSET_FILL), (z !== nt || G !== k) && (i.polygonOffset(nt, k), z = nt, G = k)) : Et(i.POLYGON_OFFSET_FILL);
  }
  function xt(D) {
    D ? rt(i.SCISSOR_TEST) : Et(i.SCISSOR_TEST);
  }
  function A(D) {
    D === void 0 && (D = i.TEXTURE0 + j - 1), st !== D && (i.activeTexture(D), st = D);
  }
  function _(D, nt, k) {
    k === void 0 && (st === null ? k = i.TEXTURE0 + j - 1 : k = st);
    let q = ht[k];
    q === void 0 && (q = { type: void 0, texture: void 0 }, ht[k] = q), (q.type !== D || q.texture !== nt) && (st !== k && (i.activeTexture(k), st = k), i.bindTexture(D, nt || mt[D]), q.type = D, q.texture = nt);
  }
  function F() {
    const D = ht[st];
    D !== void 0 && D.type !== void 0 && (i.bindTexture(D.type, null), D.type = void 0, D.texture = void 0);
  }
  function $() {
    try {
      i.compressedTexImage2D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function Z() {
    try {
      i.compressedTexImage3D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function X() {
    try {
      i.texSubImage2D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function _t() {
    try {
      i.texSubImage3D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function at() {
    try {
      i.compressedTexSubImage2D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function ut() {
    try {
      i.compressedTexSubImage3D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function Ht() {
    try {
      i.texStorage2D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function J() {
    try {
      i.texStorage3D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function dt() {
    try {
      i.texImage2D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function yt() {
    try {
      i.texImage3D.apply(i, arguments);
    } catch (D) {
      console.error("THREE.WebGLState:", D);
    }
  }
  function At(D) {
    Kt.equals(D) === !1 && (i.scissor(D.x, D.y, D.z, D.w), Kt.copy(D));
  }
  function ft(D) {
    Y.equals(D) === !1 && (i.viewport(D.x, D.y, D.z, D.w), Y.copy(D));
  }
  function Bt(D, nt) {
    let k = c.get(nt);
    k === void 0 && (k = /* @__PURE__ */ new WeakMap(), c.set(nt, k));
    let q = k.get(D);
    q === void 0 && (q = i.getUniformBlockIndex(nt, D.name), k.set(D, q));
  }
  function Dt(D, nt) {
    const q = c.get(nt).get(D);
    l.get(nt) !== q && (i.uniformBlockBinding(nt, q, D.__bindingPointIndex), l.set(nt, q));
  }
  function Jt() {
    i.disable(i.BLEND), i.disable(i.CULL_FACE), i.disable(i.DEPTH_TEST), i.disable(i.POLYGON_OFFSET_FILL), i.disable(i.SCISSOR_TEST), i.disable(i.STENCIL_TEST), i.disable(i.SAMPLE_ALPHA_TO_COVERAGE), i.blendEquation(i.FUNC_ADD), i.blendFunc(i.ONE, i.ZERO), i.blendFuncSeparate(i.ONE, i.ZERO, i.ONE, i.ZERO), i.blendColor(0, 0, 0, 0), i.colorMask(!0, !0, !0, !0), i.clearColor(0, 0, 0, 0), i.depthMask(!0), i.depthFunc(i.LESS), a.setReversed(!1), i.clearDepth(1), i.stencilMask(4294967295), i.stencilFunc(i.ALWAYS, 0, 4294967295), i.stencilOp(i.KEEP, i.KEEP, i.KEEP), i.clearStencil(0), i.cullFace(i.BACK), i.frontFace(i.CCW), i.polygonOffset(0, 0), i.activeTexture(i.TEXTURE0), i.bindFramebuffer(i.FRAMEBUFFER, null), i.bindFramebuffer(i.DRAW_FRAMEBUFFER, null), i.bindFramebuffer(i.READ_FRAMEBUFFER, null), i.useProgram(null), i.lineWidth(1), i.scissor(0, 0, i.canvas.width, i.canvas.height), i.viewport(0, 0, i.canvas.width, i.canvas.height), u = {}, st = null, ht = {}, d = {}, f = /* @__PURE__ */ new WeakMap(), m = [], g = null, v = !1, p = null, h = null, E = null, b = null, S = null, L = null, T = null, R = new Vt(0, 0, 0), I = 0, y = !1, M = null, C = null, H = null, z = null, G = null, Kt.set(0, 0, i.canvas.width, i.canvas.height), Y.set(0, 0, i.canvas.width, i.canvas.height), r.reset(), a.reset(), o.reset();
  }
  return {
    buffers: {
      color: r,
      depth: a,
      stencil: o
    },
    enable: rt,
    disable: Et,
    bindFramebuffer: Rt,
    drawBuffers: Nt,
    useProgram: re,
    setBlending: w,
    setMaterial: ke,
    setFlipSided: Ft,
    setCullFace: Ot,
    setLineWidth: vt,
    setPolygonOffset: ee,
    setScissorTest: xt,
    activeTexture: A,
    bindTexture: _,
    unbindTexture: F,
    compressedTexImage2D: $,
    compressedTexImage3D: Z,
    texImage2D: dt,
    texImage3D: yt,
    updateUBOMapping: Bt,
    uniformBlockBinding: Dt,
    texStorage2D: Ht,
    texStorage3D: J,
    texSubImage2D: X,
    texSubImage3D: _t,
    compressedTexSubImage2D: at,
    compressedTexSubImage3D: ut,
    scissor: At,
    viewport: ft,
    reset: Jt
  };
}
function cm(i, t, e, n, s, r, a) {
  const o = t.has("WEBGL_multisampled_render_to_texture") ? t.get("WEBGL_multisampled_render_to_texture") : null, l = typeof navigator > "u" ? !1 : /OculusBrowser/g.test(navigator.userAgent), c = new bt(), u = /* @__PURE__ */ new WeakMap();
  let d;
  const f = /* @__PURE__ */ new WeakMap();
  let m = !1;
  try {
    m = typeof OffscreenCanvas < "u" && new OffscreenCanvas(1, 1).getContext("2d") !== null;
  } catch {
  }
  function g(A, _) {
    return m ? (
      // eslint-disable-next-line compat/compat
      new OffscreenCanvas(A, _)
    ) : Ls("canvas");
  }
  function v(A, _, F) {
    let $ = 1;
    const Z = xt(A);
    if ((Z.width > F || Z.height > F) && ($ = F / Math.max(Z.width, Z.height)), $ < 1)
      if (typeof HTMLImageElement < "u" && A instanceof HTMLImageElement || typeof HTMLCanvasElement < "u" && A instanceof HTMLCanvasElement || typeof ImageBitmap < "u" && A instanceof ImageBitmap || typeof VideoFrame < "u" && A instanceof VideoFrame) {
        const X = Math.floor($ * Z.width), _t = Math.floor($ * Z.height);
        d === void 0 && (d = g(X, _t));
        const at = _ ? g(X, _t) : d;
        return at.width = X, at.height = _t, at.getContext("2d").drawImage(A, 0, 0, X, _t), console.warn("THREE.WebGLRenderer: Texture has been resized from (" + Z.width + "x" + Z.height + ") to (" + X + "x" + _t + ")."), at;
      } else
        return "data" in A && console.warn("THREE.WebGLRenderer: Image in DataTexture is too big (" + Z.width + "x" + Z.height + ")."), A;
    return A;
  }
  function p(A) {
    return A.generateMipmaps;
  }
  function h(A) {
    i.generateMipmap(A);
  }
  function E(A) {
    return A.isWebGLCubeRenderTarget ? i.TEXTURE_CUBE_MAP : A.isWebGL3DRenderTarget ? i.TEXTURE_3D : A.isWebGLArrayRenderTarget || A.isCompressedArrayTexture ? i.TEXTURE_2D_ARRAY : i.TEXTURE_2D;
  }
  function b(A, _, F, $, Z = !1) {
    if (A !== null) {
      if (i[A] !== void 0) return i[A];
      console.warn("THREE.WebGLRenderer: Attempt to use non-existing WebGL internal format '" + A + "'");
    }
    let X = _;
    if (_ === i.RED && (F === i.FLOAT && (X = i.R32F), F === i.HALF_FLOAT && (X = i.R16F), F === i.UNSIGNED_BYTE && (X = i.R8)), _ === i.RED_INTEGER && (F === i.UNSIGNED_BYTE && (X = i.R8UI), F === i.UNSIGNED_SHORT && (X = i.R16UI), F === i.UNSIGNED_INT && (X = i.R32UI), F === i.BYTE && (X = i.R8I), F === i.SHORT && (X = i.R16I), F === i.INT && (X = i.R32I)), _ === i.RG && (F === i.FLOAT && (X = i.RG32F), F === i.HALF_FLOAT && (X = i.RG16F), F === i.UNSIGNED_BYTE && (X = i.RG8)), _ === i.RG_INTEGER && (F === i.UNSIGNED_BYTE && (X = i.RG8UI), F === i.UNSIGNED_SHORT && (X = i.RG16UI), F === i.UNSIGNED_INT && (X = i.RG32UI), F === i.BYTE && (X = i.RG8I), F === i.SHORT && (X = i.RG16I), F === i.INT && (X = i.RG32I)), _ === i.RGB_INTEGER && (F === i.UNSIGNED_BYTE && (X = i.RGB8UI), F === i.UNSIGNED_SHORT && (X = i.RGB16UI), F === i.UNSIGNED_INT && (X = i.RGB32UI), F === i.BYTE && (X = i.RGB8I), F === i.SHORT && (X = i.RGB16I), F === i.INT && (X = i.RGB32I)), _ === i.RGBA_INTEGER && (F === i.UNSIGNED_BYTE && (X = i.RGBA8UI), F === i.UNSIGNED_SHORT && (X = i.RGBA16UI), F === i.UNSIGNED_INT && (X = i.RGBA32UI), F === i.BYTE && (X = i.RGBA8I), F === i.SHORT && (X = i.RGBA16I), F === i.INT && (X = i.RGBA32I)), _ === i.RGB && F === i.UNSIGNED_INT_5_9_9_9_REV && (X = i.RGB9_E5), _ === i.RGBA) {
      const _t = Z ? Ps : Wt.getTransfer($);
      F === i.FLOAT && (X = i.RGBA32F), F === i.HALF_FLOAT && (X = i.RGBA16F), F === i.UNSIGNED_BYTE && (X = _t === jt ? i.SRGB8_ALPHA8 : i.RGBA8), F === i.UNSIGNED_SHORT_4_4_4_4 && (X = i.RGBA4), F === i.UNSIGNED_SHORT_5_5_5_1 && (X = i.RGB5_A1);
    }
    return (X === i.R16F || X === i.R32F || X === i.RG16F || X === i.RG32F || X === i.RGBA16F || X === i.RGBA32F) && t.get("EXT_color_buffer_float"), X;
  }
  function S(A, _) {
    let F;
    return A ? _ === null || _ === jn || _ === Ti ? F = i.DEPTH24_STENCIL8 : _ === gn ? F = i.DEPTH32F_STENCIL8 : _ === Vi && (F = i.DEPTH24_STENCIL8, console.warn("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")) : _ === null || _ === jn || _ === Ti ? F = i.DEPTH_COMPONENT24 : _ === gn ? F = i.DEPTH_COMPONENT32F : _ === Vi && (F = i.DEPTH_COMPONENT16), F;
  }
  function L(A, _) {
    return p(A) === !0 || A.isFramebufferTexture && A.minFilter !== tn && A.minFilter !== an ? Math.log2(Math.max(_.width, _.height)) + 1 : A.mipmaps !== void 0 && A.mipmaps.length > 0 ? A.mipmaps.length : A.isCompressedTexture && Array.isArray(A.image) ? _.mipmaps.length : 1;
  }
  function T(A) {
    const _ = A.target;
    _.removeEventListener("dispose", T), I(_), _.isVideoTexture && u.delete(_);
  }
  function R(A) {
    const _ = A.target;
    _.removeEventListener("dispose", R), M(_);
  }
  function I(A) {
    const _ = n.get(A);
    if (_.__webglInit === void 0) return;
    const F = A.source, $ = f.get(F);
    if ($) {
      const Z = $[_.__cacheKey];
      Z.usedTimes--, Z.usedTimes === 0 && y(A), Object.keys($).length === 0 && f.delete(F);
    }
    n.remove(A);
  }
  function y(A) {
    const _ = n.get(A);
    i.deleteTexture(_.__webglTexture);
    const F = A.source, $ = f.get(F);
    delete $[_.__cacheKey], a.memory.textures--;
  }
  function M(A) {
    const _ = n.get(A);
    if (A.depthTexture && (A.depthTexture.dispose(), n.remove(A.depthTexture)), A.isWebGLCubeRenderTarget)
      for (let $ = 0; $ < 6; $++) {
        if (Array.isArray(_.__webglFramebuffer[$]))
          for (let Z = 0; Z < _.__webglFramebuffer[$].length; Z++) i.deleteFramebuffer(_.__webglFramebuffer[$][Z]);
        else
          i.deleteFramebuffer(_.__webglFramebuffer[$]);
        _.__webglDepthbuffer && i.deleteRenderbuffer(_.__webglDepthbuffer[$]);
      }
    else {
      if (Array.isArray(_.__webglFramebuffer))
        for (let $ = 0; $ < _.__webglFramebuffer.length; $++) i.deleteFramebuffer(_.__webglFramebuffer[$]);
      else
        i.deleteFramebuffer(_.__webglFramebuffer);
      if (_.__webglDepthbuffer && i.deleteRenderbuffer(_.__webglDepthbuffer), _.__webglMultisampledFramebuffer && i.deleteFramebuffer(_.__webglMultisampledFramebuffer), _.__webglColorRenderbuffer)
        for (let $ = 0; $ < _.__webglColorRenderbuffer.length; $++)
          _.__webglColorRenderbuffer[$] && i.deleteRenderbuffer(_.__webglColorRenderbuffer[$]);
      _.__webglDepthRenderbuffer && i.deleteRenderbuffer(_.__webglDepthRenderbuffer);
    }
    const F = A.textures;
    for (let $ = 0, Z = F.length; $ < Z; $++) {
      const X = n.get(F[$]);
      X.__webglTexture && (i.deleteTexture(X.__webglTexture), a.memory.textures--), n.remove(F[$]);
    }
    n.remove(A);
  }
  let C = 0;
  function H() {
    C = 0;
  }
  function z() {
    const A = C;
    return A >= s.maxTextures && console.warn("THREE.WebGLTextures: Trying to use " + A + " texture units while this GPU supports only " + s.maxTextures), C += 1, A;
  }
  function G(A) {
    const _ = [];
    return _.push(A.wrapS), _.push(A.wrapT), _.push(A.wrapR || 0), _.push(A.magFilter), _.push(A.minFilter), _.push(A.anisotropy), _.push(A.internalFormat), _.push(A.format), _.push(A.type), _.push(A.generateMipmaps), _.push(A.premultiplyAlpha), _.push(A.flipY), _.push(A.unpackAlignment), _.push(A.colorSpace), _.join();
  }
  function j(A, _) {
    const F = n.get(A);
    if (A.isVideoTexture && vt(A), A.isRenderTargetTexture === !1 && A.version > 0 && F.__version !== A.version) {
      const $ = A.image;
      if ($ === null)
        console.warn("THREE.WebGLRenderer: Texture marked for update but no image data found.");
      else if ($.complete === !1)
        console.warn("THREE.WebGLRenderer: Texture marked for update but image is incomplete");
      else {
        Y(F, A, _);
        return;
      }
    }
    e.bindTexture(i.TEXTURE_2D, F.__webglTexture, i.TEXTURE0 + _);
  }
  function W(A, _) {
    const F = n.get(A);
    if (A.version > 0 && F.__version !== A.version) {
      Y(F, A, _);
      return;
    }
    e.bindTexture(i.TEXTURE_2D_ARRAY, F.__webglTexture, i.TEXTURE0 + _);
  }
  function Q(A, _) {
    const F = n.get(A);
    if (A.version > 0 && F.__version !== A.version) {
      Y(F, A, _);
      return;
    }
    e.bindTexture(i.TEXTURE_3D, F.__webglTexture, i.TEXTURE0 + _);
  }
  function V(A, _) {
    const F = n.get(A);
    if (A.version > 0 && F.__version !== A.version) {
      tt(F, A, _);
      return;
    }
    e.bindTexture(i.TEXTURE_CUBE_MAP, F.__webglTexture, i.TEXTURE0 + _);
  }
  const st = {
    [Nr]: i.REPEAT,
    [Yn]: i.CLAMP_TO_EDGE,
    [Fr]: i.MIRRORED_REPEAT
  }, ht = {
    [tn]: i.NEAREST,
    [Ec]: i.NEAREST_MIPMAP_NEAREST,
    [ji]: i.NEAREST_MIPMAP_LINEAR,
    [an]: i.LINEAR,
    [ks]: i.LINEAR_MIPMAP_NEAREST,
    [$n]: i.LINEAR_MIPMAP_LINEAR
  }, gt = {
    [wc]: i.NEVER,
    [Uc]: i.ALWAYS,
    [Rc]: i.LESS,
    [pl]: i.LEQUAL,
    [Cc]: i.EQUAL,
    [Lc]: i.GEQUAL,
    [Pc]: i.GREATER,
    [Dc]: i.NOTEQUAL
  };
  function It(A, _) {
    if (_.type === gn && t.has("OES_texture_float_linear") === !1 && (_.magFilter === an || _.magFilter === ks || _.magFilter === ji || _.magFilter === $n || _.minFilter === an || _.minFilter === ks || _.minFilter === ji || _.minFilter === $n) && console.warn("THREE.WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."), i.texParameteri(A, i.TEXTURE_WRAP_S, st[_.wrapS]), i.texParameteri(A, i.TEXTURE_WRAP_T, st[_.wrapT]), (A === i.TEXTURE_3D || A === i.TEXTURE_2D_ARRAY) && i.texParameteri(A, i.TEXTURE_WRAP_R, st[_.wrapR]), i.texParameteri(A, i.TEXTURE_MAG_FILTER, ht[_.magFilter]), i.texParameteri(A, i.TEXTURE_MIN_FILTER, ht[_.minFilter]), _.compareFunction && (i.texParameteri(A, i.TEXTURE_COMPARE_MODE, i.COMPARE_REF_TO_TEXTURE), i.texParameteri(A, i.TEXTURE_COMPARE_FUNC, gt[_.compareFunction])), t.has("EXT_texture_filter_anisotropic") === !0) {
      if (_.magFilter === tn || _.minFilter !== ji && _.minFilter !== $n || _.type === gn && t.has("OES_texture_float_linear") === !1) return;
      if (_.anisotropy > 1 || n.get(_).__currentAnisotropy) {
        const F = t.get("EXT_texture_filter_anisotropic");
        i.texParameterf(A, F.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(_.anisotropy, s.getMaxAnisotropy())), n.get(_).__currentAnisotropy = _.anisotropy;
      }
    }
  }
  function Kt(A, _) {
    let F = !1;
    A.__webglInit === void 0 && (A.__webglInit = !0, _.addEventListener("dispose", T));
    const $ = _.source;
    let Z = f.get($);
    Z === void 0 && (Z = {}, f.set($, Z));
    const X = G(_);
    if (X !== A.__cacheKey) {
      Z[X] === void 0 && (Z[X] = {
        texture: i.createTexture(),
        usedTimes: 0
      }, a.memory.textures++, F = !0), Z[X].usedTimes++;
      const _t = Z[A.__cacheKey];
      _t !== void 0 && (Z[A.__cacheKey].usedTimes--, _t.usedTimes === 0 && y(_)), A.__cacheKey = X, A.__webglTexture = Z[X].texture;
    }
    return F;
  }
  function Y(A, _, F) {
    let $ = i.TEXTURE_2D;
    (_.isDataArrayTexture || _.isCompressedArrayTexture) && ($ = i.TEXTURE_2D_ARRAY), _.isData3DTexture && ($ = i.TEXTURE_3D);
    const Z = Kt(A, _), X = _.source;
    e.bindTexture($, A.__webglTexture, i.TEXTURE0 + F);
    const _t = n.get(X);
    if (X.version !== _t.__version || Z === !0) {
      e.activeTexture(i.TEXTURE0 + F);
      const at = Wt.getPrimaries(Wt.workingColorSpace), ut = _.colorSpace === Pn ? null : Wt.getPrimaries(_.colorSpace), Ht = _.colorSpace === Pn || at === ut ? i.NONE : i.BROWSER_DEFAULT_WEBGL;
      i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL, _.flipY), i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL, _.premultiplyAlpha), i.pixelStorei(i.UNPACK_ALIGNMENT, _.unpackAlignment), i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL, Ht);
      let J = v(_.image, !1, s.maxTextureSize);
      J = ee(_, J);
      const dt = r.convert(_.format, _.colorSpace), yt = r.convert(_.type);
      let At = b(_.internalFormat, dt, yt, _.colorSpace, _.isVideoTexture);
      It($, _);
      let ft;
      const Bt = _.mipmaps, Dt = _.isVideoTexture !== !0, Jt = _t.__version === void 0 || Z === !0, D = X.dataReady, nt = L(_, J);
      if (_.isDepthTexture)
        At = S(_.format === wi, _.type), Jt && (Dt ? e.texStorage2D(i.TEXTURE_2D, 1, At, J.width, J.height) : e.texImage2D(i.TEXTURE_2D, 0, At, J.width, J.height, 0, dt, yt, null));
      else if (_.isDataTexture)
        if (Bt.length > 0) {
          Dt && Jt && e.texStorage2D(i.TEXTURE_2D, nt, At, Bt[0].width, Bt[0].height);
          for (let k = 0, q = Bt.length; k < q; k++)
            ft = Bt[k], Dt ? D && e.texSubImage2D(i.TEXTURE_2D, k, 0, 0, ft.width, ft.height, dt, yt, ft.data) : e.texImage2D(i.TEXTURE_2D, k, At, ft.width, ft.height, 0, dt, yt, ft.data);
          _.generateMipmaps = !1;
        } else
          Dt ? (Jt && e.texStorage2D(i.TEXTURE_2D, nt, At, J.width, J.height), D && e.texSubImage2D(i.TEXTURE_2D, 0, 0, 0, J.width, J.height, dt, yt, J.data)) : e.texImage2D(i.TEXTURE_2D, 0, At, J.width, J.height, 0, dt, yt, J.data);
      else if (_.isCompressedTexture)
        if (_.isCompressedArrayTexture) {
          Dt && Jt && e.texStorage3D(i.TEXTURE_2D_ARRAY, nt, At, Bt[0].width, Bt[0].height, J.depth);
          for (let k = 0, q = Bt.length; k < q; k++)
            if (ft = Bt[k], _.format !== Qe)
              if (dt !== null)
                if (Dt) {
                  if (D)
                    if (_.layerUpdates.size > 0) {
                      const lt = vo(ft.width, ft.height, _.format, _.type);
                      for (const ot of _.layerUpdates) {
                        const Ct = ft.data.subarray(
                          ot * lt / ft.data.BYTES_PER_ELEMENT,
                          (ot + 1) * lt / ft.data.BYTES_PER_ELEMENT
                        );
                        e.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY, k, 0, 0, ot, ft.width, ft.height, 1, dt, Ct);
                      }
                      _.clearLayerUpdates();
                    } else
                      e.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY, k, 0, 0, 0, ft.width, ft.height, J.depth, dt, ft.data);
                } else
                  e.compressedTexImage3D(i.TEXTURE_2D_ARRAY, k, At, ft.width, ft.height, J.depth, 0, ft.data, 0, 0);
              else
                console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");
            else
              Dt ? D && e.texSubImage3D(i.TEXTURE_2D_ARRAY, k, 0, 0, 0, ft.width, ft.height, J.depth, dt, yt, ft.data) : e.texImage3D(i.TEXTURE_2D_ARRAY, k, At, ft.width, ft.height, J.depth, 0, dt, yt, ft.data);
        } else {
          Dt && Jt && e.texStorage2D(i.TEXTURE_2D, nt, At, Bt[0].width, Bt[0].height);
          for (let k = 0, q = Bt.length; k < q; k++)
            ft = Bt[k], _.format !== Qe ? dt !== null ? Dt ? D && e.compressedTexSubImage2D(i.TEXTURE_2D, k, 0, 0, ft.width, ft.height, dt, ft.data) : e.compressedTexImage2D(i.TEXTURE_2D, k, At, ft.width, ft.height, 0, ft.data) : console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()") : Dt ? D && e.texSubImage2D(i.TEXTURE_2D, k, 0, 0, ft.width, ft.height, dt, yt, ft.data) : e.texImage2D(i.TEXTURE_2D, k, At, ft.width, ft.height, 0, dt, yt, ft.data);
        }
      else if (_.isDataArrayTexture)
        if (Dt) {
          if (Jt && e.texStorage3D(i.TEXTURE_2D_ARRAY, nt, At, J.width, J.height, J.depth), D)
            if (_.layerUpdates.size > 0) {
              const k = vo(J.width, J.height, _.format, _.type);
              for (const q of _.layerUpdates) {
                const lt = J.data.subarray(
                  q * k / J.data.BYTES_PER_ELEMENT,
                  (q + 1) * k / J.data.BYTES_PER_ELEMENT
                );
                e.texSubImage3D(i.TEXTURE_2D_ARRAY, 0, 0, 0, q, J.width, J.height, 1, dt, yt, lt);
              }
              _.clearLayerUpdates();
            } else
              e.texSubImage3D(i.TEXTURE_2D_ARRAY, 0, 0, 0, 0, J.width, J.height, J.depth, dt, yt, J.data);
        } else
          e.texImage3D(i.TEXTURE_2D_ARRAY, 0, At, J.width, J.height, J.depth, 0, dt, yt, J.data);
      else if (_.isData3DTexture)
        Dt ? (Jt && e.texStorage3D(i.TEXTURE_3D, nt, At, J.width, J.height, J.depth), D && e.texSubImage3D(i.TEXTURE_3D, 0, 0, 0, 0, J.width, J.height, J.depth, dt, yt, J.data)) : e.texImage3D(i.TEXTURE_3D, 0, At, J.width, J.height, J.depth, 0, dt, yt, J.data);
      else if (_.isFramebufferTexture) {
        if (Jt)
          if (Dt)
            e.texStorage2D(i.TEXTURE_2D, nt, At, J.width, J.height);
          else {
            let k = J.width, q = J.height;
            for (let lt = 0; lt < nt; lt++)
              e.texImage2D(i.TEXTURE_2D, lt, At, k, q, 0, dt, yt, null), k >>= 1, q >>= 1;
          }
      } else if (Bt.length > 0) {
        if (Dt && Jt) {
          const k = xt(Bt[0]);
          e.texStorage2D(i.TEXTURE_2D, nt, At, k.width, k.height);
        }
        for (let k = 0, q = Bt.length; k < q; k++)
          ft = Bt[k], Dt ? D && e.texSubImage2D(i.TEXTURE_2D, k, 0, 0, dt, yt, ft) : e.texImage2D(i.TEXTURE_2D, k, At, dt, yt, ft);
        _.generateMipmaps = !1;
      } else if (Dt) {
        if (Jt) {
          const k = xt(J);
          e.texStorage2D(i.TEXTURE_2D, nt, At, k.width, k.height);
        }
        D && e.texSubImage2D(i.TEXTURE_2D, 0, 0, 0, dt, yt, J);
      } else
        e.texImage2D(i.TEXTURE_2D, 0, At, dt, yt, J);
      p(_) && h($), _t.__version = X.version, _.onUpdate && _.onUpdate(_);
    }
    A.__version = _.version;
  }
  function tt(A, _, F) {
    if (_.image.length !== 6) return;
    const $ = Kt(A, _), Z = _.source;
    e.bindTexture(i.TEXTURE_CUBE_MAP, A.__webglTexture, i.TEXTURE0 + F);
    const X = n.get(Z);
    if (Z.version !== X.__version || $ === !0) {
      e.activeTexture(i.TEXTURE0 + F);
      const _t = Wt.getPrimaries(Wt.workingColorSpace), at = _.colorSpace === Pn ? null : Wt.getPrimaries(_.colorSpace), ut = _.colorSpace === Pn || _t === at ? i.NONE : i.BROWSER_DEFAULT_WEBGL;
      i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL, _.flipY), i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL, _.premultiplyAlpha), i.pixelStorei(i.UNPACK_ALIGNMENT, _.unpackAlignment), i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL, ut);
      const Ht = _.isCompressedTexture || _.image[0].isCompressedTexture, J = _.image[0] && _.image[0].isDataTexture, dt = [];
      for (let q = 0; q < 6; q++)
        !Ht && !J ? dt[q] = v(_.image[q], !0, s.maxCubemapSize) : dt[q] = J ? _.image[q].image : _.image[q], dt[q] = ee(_, dt[q]);
      const yt = dt[0], At = r.convert(_.format, _.colorSpace), ft = r.convert(_.type), Bt = b(_.internalFormat, At, ft, _.colorSpace), Dt = _.isVideoTexture !== !0, Jt = X.__version === void 0 || $ === !0, D = Z.dataReady;
      let nt = L(_, yt);
      It(i.TEXTURE_CUBE_MAP, _);
      let k;
      if (Ht) {
        Dt && Jt && e.texStorage2D(i.TEXTURE_CUBE_MAP, nt, Bt, yt.width, yt.height);
        for (let q = 0; q < 6; q++) {
          k = dt[q].mipmaps;
          for (let lt = 0; lt < k.length; lt++) {
            const ot = k[lt];
            _.format !== Qe ? At !== null ? Dt ? D && e.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, lt, 0, 0, ot.width, ot.height, At, ot.data) : e.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, lt, Bt, ot.width, ot.height, 0, ot.data) : console.warn("THREE.WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()") : Dt ? D && e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, lt, 0, 0, ot.width, ot.height, At, ft, ot.data) : e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, lt, Bt, ot.width, ot.height, 0, At, ft, ot.data);
          }
        }
      } else {
        if (k = _.mipmaps, Dt && Jt) {
          k.length > 0 && nt++;
          const q = xt(dt[0]);
          e.texStorage2D(i.TEXTURE_CUBE_MAP, nt, Bt, q.width, q.height);
        }
        for (let q = 0; q < 6; q++)
          if (J) {
            Dt ? D && e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, 0, 0, 0, dt[q].width, dt[q].height, At, ft, dt[q].data) : e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, 0, Bt, dt[q].width, dt[q].height, 0, At, ft, dt[q].data);
            for (let lt = 0; lt < k.length; lt++) {
              const Ct = k[lt].image[q].image;
              Dt ? D && e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, lt + 1, 0, 0, Ct.width, Ct.height, At, ft, Ct.data) : e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, lt + 1, Bt, Ct.width, Ct.height, 0, At, ft, Ct.data);
            }
          } else {
            Dt ? D && e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, 0, 0, 0, At, ft, dt[q]) : e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, 0, Bt, At, ft, dt[q]);
            for (let lt = 0; lt < k.length; lt++) {
              const ot = k[lt];
              Dt ? D && e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, lt + 1, 0, 0, At, ft, ot.image[q]) : e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X + q, lt + 1, Bt, At, ft, ot.image[q]);
            }
          }
      }
      p(_) && h(i.TEXTURE_CUBE_MAP), X.__version = Z.version, _.onUpdate && _.onUpdate(_);
    }
    A.__version = _.version;
  }
  function mt(A, _, F, $, Z, X) {
    const _t = r.convert(F.format, F.colorSpace), at = r.convert(F.type), ut = b(F.internalFormat, _t, at, F.colorSpace), Ht = n.get(_), J = n.get(F);
    if (J.__renderTarget = _, !Ht.__hasExternalTextures) {
      const dt = Math.max(1, _.width >> X), yt = Math.max(1, _.height >> X);
      Z === i.TEXTURE_3D || Z === i.TEXTURE_2D_ARRAY ? e.texImage3D(Z, X, ut, dt, yt, _.depth, 0, _t, at, null) : e.texImage2D(Z, X, ut, dt, yt, 0, _t, at, null);
    }
    e.bindFramebuffer(i.FRAMEBUFFER, A), Ot(_) ? o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER, $, Z, J.__webglTexture, 0, Ft(_)) : (Z === i.TEXTURE_2D || Z >= i.TEXTURE_CUBE_MAP_POSITIVE_X && Z <= i.TEXTURE_CUBE_MAP_NEGATIVE_Z) && i.framebufferTexture2D(i.FRAMEBUFFER, $, Z, J.__webglTexture, X), e.bindFramebuffer(i.FRAMEBUFFER, null);
  }
  function rt(A, _, F) {
    if (i.bindRenderbuffer(i.RENDERBUFFER, A), _.depthBuffer) {
      const $ = _.depthTexture, Z = $ && $.isDepthTexture ? $.type : null, X = S(_.stencilBuffer, Z), _t = _.stencilBuffer ? i.DEPTH_STENCIL_ATTACHMENT : i.DEPTH_ATTACHMENT, at = Ft(_);
      Ot(_) ? o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER, at, X, _.width, _.height) : F ? i.renderbufferStorageMultisample(i.RENDERBUFFER, at, X, _.width, _.height) : i.renderbufferStorage(i.RENDERBUFFER, X, _.width, _.height), i.framebufferRenderbuffer(i.FRAMEBUFFER, _t, i.RENDERBUFFER, A);
    } else {
      const $ = _.textures;
      for (let Z = 0; Z < $.length; Z++) {
        const X = $[Z], _t = r.convert(X.format, X.colorSpace), at = r.convert(X.type), ut = b(X.internalFormat, _t, at, X.colorSpace), Ht = Ft(_);
        F && Ot(_) === !1 ? i.renderbufferStorageMultisample(i.RENDERBUFFER, Ht, ut, _.width, _.height) : Ot(_) ? o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER, Ht, ut, _.width, _.height) : i.renderbufferStorage(i.RENDERBUFFER, ut, _.width, _.height);
      }
    }
    i.bindRenderbuffer(i.RENDERBUFFER, null);
  }
  function Et(A, _) {
    if (_ && _.isWebGLCubeRenderTarget) throw new Error("Depth Texture with cube render targets is not supported");
    if (e.bindFramebuffer(i.FRAMEBUFFER, A), !(_.depthTexture && _.depthTexture.isDepthTexture))
      throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");
    const $ = n.get(_.depthTexture);
    $.__renderTarget = _, (!$.__webglTexture || _.depthTexture.image.width !== _.width || _.depthTexture.image.height !== _.height) && (_.depthTexture.image.width = _.width, _.depthTexture.image.height = _.height, _.depthTexture.needsUpdate = !0), j(_.depthTexture, 0);
    const Z = $.__webglTexture, X = Ft(_);
    if (_.depthTexture.format === Si)
      Ot(_) ? o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER, i.DEPTH_ATTACHMENT, i.TEXTURE_2D, Z, 0, X) : i.framebufferTexture2D(i.FRAMEBUFFER, i.DEPTH_ATTACHMENT, i.TEXTURE_2D, Z, 0);
    else if (_.depthTexture.format === wi)
      Ot(_) ? o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER, i.DEPTH_STENCIL_ATTACHMENT, i.TEXTURE_2D, Z, 0, X) : i.framebufferTexture2D(i.FRAMEBUFFER, i.DEPTH_STENCIL_ATTACHMENT, i.TEXTURE_2D, Z, 0);
    else
      throw new Error("Unknown depthTexture format");
  }
  function Rt(A) {
    const _ = n.get(A), F = A.isWebGLCubeRenderTarget === !0;
    if (_.__boundDepthTexture !== A.depthTexture) {
      const $ = A.depthTexture;
      if (_.__depthDisposeCallback && _.__depthDisposeCallback(), $) {
        const Z = () => {
          delete _.__boundDepthTexture, delete _.__depthDisposeCallback, $.removeEventListener("dispose", Z);
        };
        $.addEventListener("dispose", Z), _.__depthDisposeCallback = Z;
      }
      _.__boundDepthTexture = $;
    }
    if (A.depthTexture && !_.__autoAllocateDepthBuffer) {
      if (F) throw new Error("target.depthTexture not supported in Cube render targets");
      Et(_.__webglFramebuffer, A);
    } else if (F) {
      _.__webglDepthbuffer = [];
      for (let $ = 0; $ < 6; $++)
        if (e.bindFramebuffer(i.FRAMEBUFFER, _.__webglFramebuffer[$]), _.__webglDepthbuffer[$] === void 0)
          _.__webglDepthbuffer[$] = i.createRenderbuffer(), rt(_.__webglDepthbuffer[$], A, !1);
        else {
          const Z = A.stencilBuffer ? i.DEPTH_STENCIL_ATTACHMENT : i.DEPTH_ATTACHMENT, X = _.__webglDepthbuffer[$];
          i.bindRenderbuffer(i.RENDERBUFFER, X), i.framebufferRenderbuffer(i.FRAMEBUFFER, Z, i.RENDERBUFFER, X);
        }
    } else if (e.bindFramebuffer(i.FRAMEBUFFER, _.__webglFramebuffer), _.__webglDepthbuffer === void 0)
      _.__webglDepthbuffer = i.createRenderbuffer(), rt(_.__webglDepthbuffer, A, !1);
    else {
      const $ = A.stencilBuffer ? i.DEPTH_STENCIL_ATTACHMENT : i.DEPTH_ATTACHMENT, Z = _.__webglDepthbuffer;
      i.bindRenderbuffer(i.RENDERBUFFER, Z), i.framebufferRenderbuffer(i.FRAMEBUFFER, $, i.RENDERBUFFER, Z);
    }
    e.bindFramebuffer(i.FRAMEBUFFER, null);
  }
  function Nt(A, _, F) {
    const $ = n.get(A);
    _ !== void 0 && mt($.__webglFramebuffer, A, A.texture, i.COLOR_ATTACHMENT0, i.TEXTURE_2D, 0), F !== void 0 && Rt(A);
  }
  function re(A) {
    const _ = A.texture, F = n.get(A), $ = n.get(_);
    A.addEventListener("dispose", R);
    const Z = A.textures, X = A.isWebGLCubeRenderTarget === !0, _t = Z.length > 1;
    if (_t || ($.__webglTexture === void 0 && ($.__webglTexture = i.createTexture()), $.__version = _.version, a.memory.textures++), X) {
      F.__webglFramebuffer = [];
      for (let at = 0; at < 6; at++)
        if (_.mipmaps && _.mipmaps.length > 0) {
          F.__webglFramebuffer[at] = [];
          for (let ut = 0; ut < _.mipmaps.length; ut++)
            F.__webglFramebuffer[at][ut] = i.createFramebuffer();
        } else
          F.__webglFramebuffer[at] = i.createFramebuffer();
    } else {
      if (_.mipmaps && _.mipmaps.length > 0) {
        F.__webglFramebuffer = [];
        for (let at = 0; at < _.mipmaps.length; at++)
          F.__webglFramebuffer[at] = i.createFramebuffer();
      } else
        F.__webglFramebuffer = i.createFramebuffer();
      if (_t)
        for (let at = 0, ut = Z.length; at < ut; at++) {
          const Ht = n.get(Z[at]);
          Ht.__webglTexture === void 0 && (Ht.__webglTexture = i.createTexture(), a.memory.textures++);
        }
      if (A.samples > 0 && Ot(A) === !1) {
        F.__webglMultisampledFramebuffer = i.createFramebuffer(), F.__webglColorRenderbuffer = [], e.bindFramebuffer(i.FRAMEBUFFER, F.__webglMultisampledFramebuffer);
        for (let at = 0; at < Z.length; at++) {
          const ut = Z[at];
          F.__webglColorRenderbuffer[at] = i.createRenderbuffer(), i.bindRenderbuffer(i.RENDERBUFFER, F.__webglColorRenderbuffer[at]);
          const Ht = r.convert(ut.format, ut.colorSpace), J = r.convert(ut.type), dt = b(ut.internalFormat, Ht, J, ut.colorSpace, A.isXRRenderTarget === !0), yt = Ft(A);
          i.renderbufferStorageMultisample(i.RENDERBUFFER, yt, dt, A.width, A.height), i.framebufferRenderbuffer(i.FRAMEBUFFER, i.COLOR_ATTACHMENT0 + at, i.RENDERBUFFER, F.__webglColorRenderbuffer[at]);
        }
        i.bindRenderbuffer(i.RENDERBUFFER, null), A.depthBuffer && (F.__webglDepthRenderbuffer = i.createRenderbuffer(), rt(F.__webglDepthRenderbuffer, A, !0)), e.bindFramebuffer(i.FRAMEBUFFER, null);
      }
    }
    if (X) {
      e.bindTexture(i.TEXTURE_CUBE_MAP, $.__webglTexture), It(i.TEXTURE_CUBE_MAP, _);
      for (let at = 0; at < 6; at++)
        if (_.mipmaps && _.mipmaps.length > 0)
          for (let ut = 0; ut < _.mipmaps.length; ut++)
            mt(F.__webglFramebuffer[at][ut], A, _, i.COLOR_ATTACHMENT0, i.TEXTURE_CUBE_MAP_POSITIVE_X + at, ut);
        else
          mt(F.__webglFramebuffer[at], A, _, i.COLOR_ATTACHMENT0, i.TEXTURE_CUBE_MAP_POSITIVE_X + at, 0);
      p(_) && h(i.TEXTURE_CUBE_MAP), e.unbindTexture();
    } else if (_t) {
      for (let at = 0, ut = Z.length; at < ut; at++) {
        const Ht = Z[at], J = n.get(Ht);
        e.bindTexture(i.TEXTURE_2D, J.__webglTexture), It(i.TEXTURE_2D, Ht), mt(F.__webglFramebuffer, A, Ht, i.COLOR_ATTACHMENT0 + at, i.TEXTURE_2D, 0), p(Ht) && h(i.TEXTURE_2D);
      }
      e.unbindTexture();
    } else {
      let at = i.TEXTURE_2D;
      if ((A.isWebGL3DRenderTarget || A.isWebGLArrayRenderTarget) && (at = A.isWebGL3DRenderTarget ? i.TEXTURE_3D : i.TEXTURE_2D_ARRAY), e.bindTexture(at, $.__webglTexture), It(at, _), _.mipmaps && _.mipmaps.length > 0)
        for (let ut = 0; ut < _.mipmaps.length; ut++)
          mt(F.__webglFramebuffer[ut], A, _, i.COLOR_ATTACHMENT0, at, ut);
      else
        mt(F.__webglFramebuffer, A, _, i.COLOR_ATTACHMENT0, at, 0);
      p(_) && h(at), e.unbindTexture();
    }
    A.depthBuffer && Rt(A);
  }
  function zt(A) {
    const _ = A.textures;
    for (let F = 0, $ = _.length; F < $; F++) {
      const Z = _[F];
      if (p(Z)) {
        const X = E(A), _t = n.get(Z).__webglTexture;
        e.bindTexture(X, _t), h(X), e.unbindTexture();
      }
    }
  }
  const ce = [], w = [];
  function ke(A) {
    if (A.samples > 0) {
      if (Ot(A) === !1) {
        const _ = A.textures, F = A.width, $ = A.height;
        let Z = i.COLOR_BUFFER_BIT;
        const X = A.stencilBuffer ? i.DEPTH_STENCIL_ATTACHMENT : i.DEPTH_ATTACHMENT, _t = n.get(A), at = _.length > 1;
        if (at)
          for (let ut = 0; ut < _.length; ut++)
            e.bindFramebuffer(i.FRAMEBUFFER, _t.__webglMultisampledFramebuffer), i.framebufferRenderbuffer(i.FRAMEBUFFER, i.COLOR_ATTACHMENT0 + ut, i.RENDERBUFFER, null), e.bindFramebuffer(i.FRAMEBUFFER, _t.__webglFramebuffer), i.framebufferTexture2D(i.DRAW_FRAMEBUFFER, i.COLOR_ATTACHMENT0 + ut, i.TEXTURE_2D, null, 0);
        e.bindFramebuffer(i.READ_FRAMEBUFFER, _t.__webglMultisampledFramebuffer), e.bindFramebuffer(i.DRAW_FRAMEBUFFER, _t.__webglFramebuffer);
        for (let ut = 0; ut < _.length; ut++) {
          if (A.resolveDepthBuffer && (A.depthBuffer && (Z |= i.DEPTH_BUFFER_BIT), A.stencilBuffer && A.resolveStencilBuffer && (Z |= i.STENCIL_BUFFER_BIT)), at) {
            i.framebufferRenderbuffer(i.READ_FRAMEBUFFER, i.COLOR_ATTACHMENT0, i.RENDERBUFFER, _t.__webglColorRenderbuffer[ut]);
            const Ht = n.get(_[ut]).__webglTexture;
            i.framebufferTexture2D(i.DRAW_FRAMEBUFFER, i.COLOR_ATTACHMENT0, i.TEXTURE_2D, Ht, 0);
          }
          i.blitFramebuffer(0, 0, F, $, 0, 0, F, $, Z, i.NEAREST), l === !0 && (ce.length = 0, w.length = 0, ce.push(i.COLOR_ATTACHMENT0 + ut), A.depthBuffer && A.resolveDepthBuffer === !1 && (ce.push(X), w.push(X), i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER, w)), i.invalidateFramebuffer(i.READ_FRAMEBUFFER, ce));
        }
        if (e.bindFramebuffer(i.READ_FRAMEBUFFER, null), e.bindFramebuffer(i.DRAW_FRAMEBUFFER, null), at)
          for (let ut = 0; ut < _.length; ut++) {
            e.bindFramebuffer(i.FRAMEBUFFER, _t.__webglMultisampledFramebuffer), i.framebufferRenderbuffer(i.FRAMEBUFFER, i.COLOR_ATTACHMENT0 + ut, i.RENDERBUFFER, _t.__webglColorRenderbuffer[ut]);
            const Ht = n.get(_[ut]).__webglTexture;
            e.bindFramebuffer(i.FRAMEBUFFER, _t.__webglFramebuffer), i.framebufferTexture2D(i.DRAW_FRAMEBUFFER, i.COLOR_ATTACHMENT0 + ut, i.TEXTURE_2D, Ht, 0);
          }
        e.bindFramebuffer(i.DRAW_FRAMEBUFFER, _t.__webglMultisampledFramebuffer);
      } else if (A.depthBuffer && A.resolveDepthBuffer === !1 && l) {
        const _ = A.stencilBuffer ? i.DEPTH_STENCIL_ATTACHMENT : i.DEPTH_ATTACHMENT;
        i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER, [_]);
      }
    }
  }
  function Ft(A) {
    return Math.min(s.maxSamples, A.samples);
  }
  function Ot(A) {
    const _ = n.get(A);
    return A.samples > 0 && t.has("WEBGL_multisampled_render_to_texture") === !0 && _.__useRenderToTexture !== !1;
  }
  function vt(A) {
    const _ = a.render.frame;
    u.get(A) !== _ && (u.set(A, _), A.update());
  }
  function ee(A, _) {
    const F = A.colorSpace, $ = A.format, Z = A.type;
    return A.isCompressedTexture === !0 || A.isVideoTexture === !0 || F !== Ri && F !== Pn && (Wt.getTransfer(F) === jt ? ($ !== Qe || Z !== Sn) && console.warn("THREE.WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType.") : console.error("THREE.WebGLTextures: Unsupported texture color space:", F)), _;
  }
  function xt(A) {
    return typeof HTMLImageElement < "u" && A instanceof HTMLImageElement ? (c.width = A.naturalWidth || A.width, c.height = A.naturalHeight || A.height) : typeof VideoFrame < "u" && A instanceof VideoFrame ? (c.width = A.displayWidth, c.height = A.displayHeight) : (c.width = A.width, c.height = A.height), c;
  }
  this.allocateTextureUnit = z, this.resetTextureUnits = H, this.setTexture2D = j, this.setTexture2DArray = W, this.setTexture3D = Q, this.setTextureCube = V, this.rebindTextures = Nt, this.setupRenderTarget = re, this.updateRenderTargetMipmap = zt, this.updateMultisampleRenderTarget = ke, this.setupDepthRenderbuffer = Rt, this.setupFrameBufferTexture = mt, this.useMultisampledRTT = Ot;
}
function hm(i, t) {
  function e(n, s = Pn) {
    let r;
    const a = Wt.getTransfer(s);
    if (n === Sn) return i.UNSIGNED_BYTE;
    if (n === ma) return i.UNSIGNED_SHORT_4_4_4_4;
    if (n === _a) return i.UNSIGNED_SHORT_5_5_5_1;
    if (n === rl) return i.UNSIGNED_INT_5_9_9_9_REV;
    if (n === il) return i.BYTE;
    if (n === sl) return i.SHORT;
    if (n === Vi) return i.UNSIGNED_SHORT;
    if (n === pa) return i.INT;
    if (n === jn) return i.UNSIGNED_INT;
    if (n === gn) return i.FLOAT;
    if (n === Xi) return i.HALF_FLOAT;
    if (n === al) return i.ALPHA;
    if (n === ol) return i.RGB;
    if (n === Qe) return i.RGBA;
    if (n === ll) return i.LUMINANCE;
    if (n === cl) return i.LUMINANCE_ALPHA;
    if (n === Si) return i.DEPTH_COMPONENT;
    if (n === wi) return i.DEPTH_STENCIL;
    if (n === hl) return i.RED;
    if (n === ga) return i.RED_INTEGER;
    if (n === ul) return i.RG;
    if (n === va) return i.RG_INTEGER;
    if (n === xa) return i.RGBA_INTEGER;
    if (n === bs || n === As || n === Ts || n === ws)
      if (a === jt)
        if (r = t.get("WEBGL_compressed_texture_s3tc_srgb"), r !== null) {
          if (n === bs) return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;
          if (n === As) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;
          if (n === Ts) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;
          if (n === ws) return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT;
        } else
          return null;
      else if (r = t.get("WEBGL_compressed_texture_s3tc"), r !== null) {
        if (n === bs) return r.COMPRESSED_RGB_S3TC_DXT1_EXT;
        if (n === As) return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;
        if (n === Ts) return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;
        if (n === ws) return r.COMPRESSED_RGBA_S3TC_DXT5_EXT;
      } else
        return null;
    if (n === Or || n === Br || n === zr || n === Hr)
      if (r = t.get("WEBGL_compressed_texture_pvrtc"), r !== null) {
        if (n === Or) return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
        if (n === Br) return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
        if (n === zr) return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
        if (n === Hr) return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
      } else
        return null;
    if (n === kr || n === Vr || n === Gr)
      if (r = t.get("WEBGL_compressed_texture_etc"), r !== null) {
        if (n === kr || n === Vr) return a === jt ? r.COMPRESSED_SRGB8_ETC2 : r.COMPRESSED_RGB8_ETC2;
        if (n === Gr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC : r.COMPRESSED_RGBA8_ETC2_EAC;
      } else
        return null;
    if (n === Wr || n === Xr || n === Yr || n === $r || n === qr || n === jr || n === Zr || n === Kr || n === Jr || n === Qr || n === ta || n === ea || n === na || n === ia)
      if (r = t.get("WEBGL_compressed_texture_astc"), r !== null) {
        if (n === Wr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR : r.COMPRESSED_RGBA_ASTC_4x4_KHR;
        if (n === Xr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR : r.COMPRESSED_RGBA_ASTC_5x4_KHR;
        if (n === Yr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR : r.COMPRESSED_RGBA_ASTC_5x5_KHR;
        if (n === $r) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR : r.COMPRESSED_RGBA_ASTC_6x5_KHR;
        if (n === qr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR : r.COMPRESSED_RGBA_ASTC_6x6_KHR;
        if (n === jr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR : r.COMPRESSED_RGBA_ASTC_8x5_KHR;
        if (n === Zr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR : r.COMPRESSED_RGBA_ASTC_8x6_KHR;
        if (n === Kr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR : r.COMPRESSED_RGBA_ASTC_8x8_KHR;
        if (n === Jr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR : r.COMPRESSED_RGBA_ASTC_10x5_KHR;
        if (n === Qr) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR : r.COMPRESSED_RGBA_ASTC_10x6_KHR;
        if (n === ta) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR : r.COMPRESSED_RGBA_ASTC_10x8_KHR;
        if (n === ea) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR : r.COMPRESSED_RGBA_ASTC_10x10_KHR;
        if (n === na) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR : r.COMPRESSED_RGBA_ASTC_12x10_KHR;
        if (n === ia) return a === jt ? r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR : r.COMPRESSED_RGBA_ASTC_12x12_KHR;
      } else
        return null;
    if (n === Rs || n === sa || n === ra)
      if (r = t.get("EXT_texture_compression_bptc"), r !== null) {
        if (n === Rs) return a === jt ? r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT : r.COMPRESSED_RGBA_BPTC_UNORM_EXT;
        if (n === sa) return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;
        if (n === ra) return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT;
      } else
        return null;
    if (n === dl || n === aa || n === oa || n === la)
      if (r = t.get("EXT_texture_compression_rgtc"), r !== null) {
        if (n === Rs) return r.COMPRESSED_RED_RGTC1_EXT;
        if (n === aa) return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;
        if (n === oa) return r.COMPRESSED_RED_GREEN_RGTC2_EXT;
        if (n === la) return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT;
      } else
        return null;
    return n === Ti ? i.UNSIGNED_INT_24_8 : i[n] !== void 0 ? i[n] : null;
  }
  return { convert: e };
}
const um = { type: "move" };
class _r {
  constructor() {
    this._targetRay = null, this._grip = null, this._hand = null;
  }
  getHandSpace() {
    return this._hand === null && (this._hand = new le(), this._hand.matrixAutoUpdate = !1, this._hand.visible = !1, this._hand.joints = {}, this._hand.inputState = { pinching: !1 }), this._hand;
  }
  getTargetRaySpace() {
    return this._targetRay === null && (this._targetRay = new le(), this._targetRay.matrixAutoUpdate = !1, this._targetRay.visible = !1, this._targetRay.hasLinearVelocity = !1, this._targetRay.linearVelocity = new P(), this._targetRay.hasAngularVelocity = !1, this._targetRay.angularVelocity = new P()), this._targetRay;
  }
  getGripSpace() {
    return this._grip === null && (this._grip = new le(), this._grip.matrixAutoUpdate = !1, this._grip.visible = !1, this._grip.hasLinearVelocity = !1, this._grip.linearVelocity = new P(), this._grip.hasAngularVelocity = !1, this._grip.angularVelocity = new P()), this._grip;
  }
  dispatchEvent(t) {
    return this._targetRay !== null && this._targetRay.dispatchEvent(t), this._grip !== null && this._grip.dispatchEvent(t), this._hand !== null && this._hand.dispatchEvent(t), this;
  }
  connect(t) {
    if (t && t.hand) {
      const e = this._hand;
      if (e)
        for (const n of t.hand.values())
          this._getHandJoint(e, n);
    }
    return this.dispatchEvent({ type: "connected", data: t }), this;
  }
  disconnect(t) {
    return this.dispatchEvent({ type: "disconnected", data: t }), this._targetRay !== null && (this._targetRay.visible = !1), this._grip !== null && (this._grip.visible = !1), this._hand !== null && (this._hand.visible = !1), this;
  }
  update(t, e, n) {
    let s = null, r = null, a = null;
    const o = this._targetRay, l = this._grip, c = this._hand;
    if (t && e.session.visibilityState !== "visible-blurred") {
      if (c && t.hand) {
        a = !0;
        for (const v of t.hand.values()) {
          const p = e.getJointPose(v, n), h = this._getHandJoint(c, v);
          p !== null && (h.matrix.fromArray(p.transform.matrix), h.matrix.decompose(h.position, h.rotation, h.scale), h.matrixWorldNeedsUpdate = !0, h.jointRadius = p.radius), h.visible = p !== null;
        }
        const u = c.joints["index-finger-tip"], d = c.joints["thumb-tip"], f = u.position.distanceTo(d.position), m = 0.02, g = 5e-3;
        c.inputState.pinching && f > m + g ? (c.inputState.pinching = !1, this.dispatchEvent({
          type: "pinchend",
          handedness: t.handedness,
          target: this
        })) : !c.inputState.pinching && f <= m - g && (c.inputState.pinching = !0, this.dispatchEvent({
          type: "pinchstart",
          handedness: t.handedness,
          target: this
        }));
      } else
        l !== null && t.gripSpace && (r = e.getPose(t.gripSpace, n), r !== null && (l.matrix.fromArray(r.transform.matrix), l.matrix.decompose(l.position, l.rotation, l.scale), l.matrixWorldNeedsUpdate = !0, r.linearVelocity ? (l.hasLinearVelocity = !0, l.linearVelocity.copy(r.linearVelocity)) : l.hasLinearVelocity = !1, r.angularVelocity ? (l.hasAngularVelocity = !0, l.angularVelocity.copy(r.angularVelocity)) : l.hasAngularVelocity = !1));
      o !== null && (s = e.getPose(t.targetRaySpace, n), s === null && r !== null && (s = r), s !== null && (o.matrix.fromArray(s.transform.matrix), o.matrix.decompose(o.position, o.rotation, o.scale), o.matrixWorldNeedsUpdate = !0, s.linearVelocity ? (o.hasLinearVelocity = !0, o.linearVelocity.copy(s.linearVelocity)) : o.hasLinearVelocity = !1, s.angularVelocity ? (o.hasAngularVelocity = !0, o.angularVelocity.copy(s.angularVelocity)) : o.hasAngularVelocity = !1, this.dispatchEvent(um)));
    }
    return o !== null && (o.visible = s !== null), l !== null && (l.visible = r !== null), c !== null && (c.visible = a !== null), this;
  }
  // private method
  _getHandJoint(t, e) {
    if (t.joints[e.jointName] === void 0) {
      const n = new le();
      n.matrixAutoUpdate = !1, n.visible = !1, t.joints[e.jointName] = n, t.add(n);
    }
    return t.joints[e.jointName];
  }
}
const dm = `
void main() {

	gl_Position = vec4( position, 1.0 );

}`, fm = `
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;
class pm {
  constructor() {
    this.texture = null, this.mesh = null, this.depthNear = 0, this.depthFar = 0;
  }
  init(t, e, n) {
    if (this.texture === null) {
      const s = new Ie(), r = t.properties.get(s);
      r.__webglTexture = e.texture, (e.depthNear != n.depthNear || e.depthFar != n.depthFar) && (this.depthNear = e.depthNear, this.depthFar = e.depthFar), this.texture = s;
    }
  }
  getMesh(t) {
    if (this.texture !== null && this.mesh === null) {
      const e = t.cameras[0].viewport, n = new En({
        vertexShader: dm,
        fragmentShader: fm,
        uniforms: {
          depthColor: { value: this.texture },
          depthWidth: { value: e.z },
          depthHeight: { value: e.w }
        }
      });
      this.mesh = new be(new Os(20, 20), n);
    }
    return this.mesh;
  }
  reset() {
    this.texture = null, this.mesh = null;
  }
  getDepthTexture() {
    return this.texture;
  }
}
class mm extends Qn {
  constructor(t, e) {
    super();
    const n = this;
    let s = null, r = 1, a = null, o = "local-floor", l = 1, c = null, u = null, d = null, f = null, m = null, g = null;
    const v = new pm(), p = e.getContextAttributes();
    let h = null, E = null;
    const b = [], S = [], L = new bt();
    let T = null;
    const R = new ze();
    R.viewport = new te();
    const I = new ze();
    I.viewport = new te();
    const y = [R, I], M = new Dh();
    let C = null, H = null;
    this.cameraAutoUpdate = !0, this.enabled = !1, this.isPresenting = !1, this.getController = function(Y) {
      let tt = b[Y];
      return tt === void 0 && (tt = new _r(), b[Y] = tt), tt.getTargetRaySpace();
    }, this.getControllerGrip = function(Y) {
      let tt = b[Y];
      return tt === void 0 && (tt = new _r(), b[Y] = tt), tt.getGripSpace();
    }, this.getHand = function(Y) {
      let tt = b[Y];
      return tt === void 0 && (tt = new _r(), b[Y] = tt), tt.getHandSpace();
    };
    function z(Y) {
      const tt = S.indexOf(Y.inputSource);
      if (tt === -1)
        return;
      const mt = b[tt];
      mt !== void 0 && (mt.update(Y.inputSource, Y.frame, c || a), mt.dispatchEvent({ type: Y.type, data: Y.inputSource }));
    }
    function G() {
      s.removeEventListener("select", z), s.removeEventListener("selectstart", z), s.removeEventListener("selectend", z), s.removeEventListener("squeeze", z), s.removeEventListener("squeezestart", z), s.removeEventListener("squeezeend", z), s.removeEventListener("end", G), s.removeEventListener("inputsourceschange", j);
      for (let Y = 0; Y < b.length; Y++) {
        const tt = S[Y];
        tt !== null && (S[Y] = null, b[Y].disconnect(tt));
      }
      C = null, H = null, v.reset(), t.setRenderTarget(h), m = null, f = null, d = null, s = null, E = null, Kt.stop(), n.isPresenting = !1, t.setPixelRatio(T), t.setSize(L.width, L.height, !1), n.dispatchEvent({ type: "sessionend" });
    }
    this.setFramebufferScaleFactor = function(Y) {
      r = Y, n.isPresenting === !0 && console.warn("THREE.WebXRManager: Cannot change framebuffer scale while presenting.");
    }, this.setReferenceSpaceType = function(Y) {
      o = Y, n.isPresenting === !0 && console.warn("THREE.WebXRManager: Cannot change reference space type while presenting.");
    }, this.getReferenceSpace = function() {
      return c || a;
    }, this.setReferenceSpace = function(Y) {
      c = Y;
    }, this.getBaseLayer = function() {
      return f !== null ? f : m;
    }, this.getBinding = function() {
      return d;
    }, this.getFrame = function() {
      return g;
    }, this.getSession = function() {
      return s;
    }, this.setSession = async function(Y) {
      if (s = Y, s !== null) {
        if (h = t.getRenderTarget(), s.addEventListener("select", z), s.addEventListener("selectstart", z), s.addEventListener("selectend", z), s.addEventListener("squeeze", z), s.addEventListener("squeezestart", z), s.addEventListener("squeezeend", z), s.addEventListener("end", G), s.addEventListener("inputsourceschange", j), p.xrCompatible !== !0 && await e.makeXRCompatible(), T = t.getPixelRatio(), t.getSize(L), s.renderState.layers === void 0) {
          const tt = {
            antialias: p.antialias,
            alpha: !0,
            depth: p.depth,
            stencil: p.stencil,
            framebufferScaleFactor: r
          };
          m = new XRWebGLLayer(s, e, tt), s.updateRenderState({ baseLayer: m }), t.setPixelRatio(1), t.setSize(m.framebufferWidth, m.framebufferHeight, !1), E = new Zn(
            m.framebufferWidth,
            m.framebufferHeight,
            {
              format: Qe,
              type: Sn,
              colorSpace: t.outputColorSpace,
              stencilBuffer: p.stencil
            }
          );
        } else {
          let tt = null, mt = null, rt = null;
          p.depth && (rt = p.stencil ? e.DEPTH24_STENCIL8 : e.DEPTH_COMPONENT24, tt = p.stencil ? wi : Si, mt = p.stencil ? Ti : jn);
          const Et = {
            colorFormat: e.RGBA8,
            depthFormat: rt,
            scaleFactor: r
          };
          d = new XRWebGLBinding(s, e), f = d.createProjectionLayer(Et), s.updateRenderState({ layers: [f] }), t.setPixelRatio(1), t.setSize(f.textureWidth, f.textureHeight, !1), E = new Zn(
            f.textureWidth,
            f.textureHeight,
            {
              format: Qe,
              type: Sn,
              depthTexture: new Rl(f.textureWidth, f.textureHeight, mt, void 0, void 0, void 0, void 0, void 0, void 0, tt),
              stencilBuffer: p.stencil,
              colorSpace: t.outputColorSpace,
              samples: p.antialias ? 4 : 0,
              resolveDepthBuffer: f.ignoreDepthValues === !1
            }
          );
        }
        E.isXRRenderTarget = !0, this.setFoveation(l), c = null, a = await s.requestReferenceSpace(o), Kt.setContext(s), Kt.start(), n.isPresenting = !0, n.dispatchEvent({ type: "sessionstart" });
      }
    }, this.getEnvironmentBlendMode = function() {
      if (s !== null)
        return s.environmentBlendMode;
    }, this.getDepthTexture = function() {
      return v.getDepthTexture();
    };
    function j(Y) {
      for (let tt = 0; tt < Y.removed.length; tt++) {
        const mt = Y.removed[tt], rt = S.indexOf(mt);
        rt >= 0 && (S[rt] = null, b[rt].disconnect(mt));
      }
      for (let tt = 0; tt < Y.added.length; tt++) {
        const mt = Y.added[tt];
        let rt = S.indexOf(mt);
        if (rt === -1) {
          for (let Rt = 0; Rt < b.length; Rt++)
            if (Rt >= S.length) {
              S.push(mt), rt = Rt;
              break;
            } else if (S[Rt] === null) {
              S[Rt] = mt, rt = Rt;
              break;
            }
          if (rt === -1) break;
        }
        const Et = b[rt];
        Et && Et.connect(mt);
      }
    }
    const W = new P(), Q = new P();
    function V(Y, tt, mt) {
      W.setFromMatrixPosition(tt.matrixWorld), Q.setFromMatrixPosition(mt.matrixWorld);
      const rt = W.distanceTo(Q), Et = tt.projectionMatrix.elements, Rt = mt.projectionMatrix.elements, Nt = Et[14] / (Et[10] - 1), re = Et[14] / (Et[10] + 1), zt = (Et[9] + 1) / Et[5], ce = (Et[9] - 1) / Et[5], w = (Et[8] - 1) / Et[0], ke = (Rt[8] + 1) / Rt[0], Ft = Nt * w, Ot = Nt * ke, vt = rt / (-w + ke), ee = vt * -w;
      if (tt.matrixWorld.decompose(Y.position, Y.quaternion, Y.scale), Y.translateX(ee), Y.translateZ(vt), Y.matrixWorld.compose(Y.position, Y.quaternion, Y.scale), Y.matrixWorldInverse.copy(Y.matrixWorld).invert(), Et[10] === -1)
        Y.projectionMatrix.copy(tt.projectionMatrix), Y.projectionMatrixInverse.copy(tt.projectionMatrixInverse);
      else {
        const xt = Nt + vt, A = re + vt, _ = Ft - ee, F = Ot + (rt - ee), $ = zt * re / A * xt, Z = ce * re / A * xt;
        Y.projectionMatrix.makePerspective(_, F, $, Z, xt, A), Y.projectionMatrixInverse.copy(Y.projectionMatrix).invert();
      }
    }
    function st(Y, tt) {
      tt === null ? Y.matrixWorld.copy(Y.matrix) : Y.matrixWorld.multiplyMatrices(tt.matrixWorld, Y.matrix), Y.matrixWorldInverse.copy(Y.matrixWorld).invert();
    }
    this.updateCamera = function(Y) {
      if (s === null) return;
      let tt = Y.near, mt = Y.far;
      v.texture !== null && (v.depthNear > 0 && (tt = v.depthNear), v.depthFar > 0 && (mt = v.depthFar)), M.near = I.near = R.near = tt, M.far = I.far = R.far = mt, (C !== M.near || H !== M.far) && (s.updateRenderState({
        depthNear: M.near,
        depthFar: M.far
      }), C = M.near, H = M.far), R.layers.mask = Y.layers.mask | 2, I.layers.mask = Y.layers.mask | 4, M.layers.mask = R.layers.mask | I.layers.mask;
      const rt = Y.parent, Et = M.cameras;
      st(M, rt);
      for (let Rt = 0; Rt < Et.length; Rt++)
        st(Et[Rt], rt);
      Et.length === 2 ? V(M, R, I) : M.projectionMatrix.copy(R.projectionMatrix), ht(Y, M, rt);
    };
    function ht(Y, tt, mt) {
      mt === null ? Y.matrix.copy(tt.matrixWorld) : (Y.matrix.copy(mt.matrixWorld), Y.matrix.invert(), Y.matrix.multiply(tt.matrixWorld)), Y.matrix.decompose(Y.position, Y.quaternion, Y.scale), Y.updateMatrixWorld(!0), Y.projectionMatrix.copy(tt.projectionMatrix), Y.projectionMatrixInverse.copy(tt.projectionMatrixInverse), Y.isPerspectiveCamera && (Y.fov = Gi * 2 * Math.atan(1 / Y.projectionMatrix.elements[5]), Y.zoom = 1);
    }
    this.getCamera = function() {
      return M;
    }, this.getFoveation = function() {
      if (!(f === null && m === null))
        return l;
    }, this.setFoveation = function(Y) {
      l = Y, f !== null && (f.fixedFoveation = Y), m !== null && m.fixedFoveation !== void 0 && (m.fixedFoveation = Y);
    }, this.hasDepthSensing = function() {
      return v.texture !== null;
    }, this.getDepthSensingMesh = function() {
      return v.getMesh(M);
    };
    let gt = null;
    function It(Y, tt) {
      if (u = tt.getViewerPose(c || a), g = tt, u !== null) {
        const mt = u.views;
        m !== null && (t.setRenderTargetFramebuffer(E, m.framebuffer), t.setRenderTarget(E));
        let rt = !1;
        mt.length !== M.cameras.length && (M.cameras.length = 0, rt = !0);
        for (let Rt = 0; Rt < mt.length; Rt++) {
          const Nt = mt[Rt];
          let re = null;
          if (m !== null)
            re = m.getViewport(Nt);
          else {
            const ce = d.getViewSubImage(f, Nt);
            re = ce.viewport, Rt === 0 && (t.setRenderTargetTextures(
              E,
              ce.colorTexture,
              f.ignoreDepthValues ? void 0 : ce.depthStencilTexture
            ), t.setRenderTarget(E));
          }
          let zt = y[Rt];
          zt === void 0 && (zt = new ze(), zt.layers.enable(Rt), zt.viewport = new te(), y[Rt] = zt), zt.matrix.fromArray(Nt.transform.matrix), zt.matrix.decompose(zt.position, zt.quaternion, zt.scale), zt.projectionMatrix.fromArray(Nt.projectionMatrix), zt.projectionMatrixInverse.copy(zt.projectionMatrix).invert(), zt.viewport.set(re.x, re.y, re.width, re.height), Rt === 0 && (M.matrix.copy(zt.matrix), M.matrix.decompose(M.position, M.quaternion, M.scale)), rt === !0 && M.cameras.push(zt);
        }
        const Et = s.enabledFeatures;
        if (Et && Et.includes("depth-sensing")) {
          const Rt = d.getDepthInformation(mt[0]);
          Rt && Rt.isValid && Rt.texture && v.init(t, Rt, s.renderState);
        }
      }
      for (let mt = 0; mt < b.length; mt++) {
        const rt = S[mt], Et = b[mt];
        rt !== null && Et !== void 0 && Et.update(rt, tt, c || a);
      }
      gt && gt(Y, tt), tt.detectedPlanes && n.dispatchEvent({ type: "planesdetected", data: tt }), g = null;
    }
    const Kt = new Cl();
    Kt.setAnimationLoop(It), this.setAnimationLoop = function(Y) {
      gt = Y;
    }, this.dispose = function() {
    };
  }
}
const Vn = /* @__PURE__ */ new yn(), _m = /* @__PURE__ */ new ne();
function gm(i, t) {
  function e(p, h) {
    p.matrixAutoUpdate === !0 && p.updateMatrix(), h.value.copy(p.matrix);
  }
  function n(p, h) {
    h.color.getRGB(p.fogColor.value, El(i)), h.isFog ? (p.fogNear.value = h.near, p.fogFar.value = h.far) : h.isFogExp2 && (p.fogDensity.value = h.density);
  }
  function s(p, h, E, b, S) {
    h.isMeshBasicMaterial || h.isMeshLambertMaterial ? r(p, h) : h.isMeshToonMaterial ? (r(p, h), d(p, h)) : h.isMeshPhongMaterial ? (r(p, h), u(p, h)) : h.isMeshStandardMaterial ? (r(p, h), f(p, h), h.isMeshPhysicalMaterial && m(p, h, S)) : h.isMeshMatcapMaterial ? (r(p, h), g(p, h)) : h.isMeshDepthMaterial ? r(p, h) : h.isMeshDistanceMaterial ? (r(p, h), v(p, h)) : h.isMeshNormalMaterial ? r(p, h) : h.isLineBasicMaterial ? (a(p, h), h.isLineDashedMaterial && o(p, h)) : h.isPointsMaterial ? l(p, h, E, b) : h.isSpriteMaterial ? c(p, h) : h.isShadowMaterial ? (p.color.value.copy(h.color), p.opacity.value = h.opacity) : h.isShaderMaterial && (h.uniformsNeedUpdate = !1);
  }
  function r(p, h) {
    p.opacity.value = h.opacity, h.color && p.diffuse.value.copy(h.color), h.emissive && p.emissive.value.copy(h.emissive).multiplyScalar(h.emissiveIntensity), h.map && (p.map.value = h.map, e(h.map, p.mapTransform)), h.alphaMap && (p.alphaMap.value = h.alphaMap, e(h.alphaMap, p.alphaMapTransform)), h.bumpMap && (p.bumpMap.value = h.bumpMap, e(h.bumpMap, p.bumpMapTransform), p.bumpScale.value = h.bumpScale, h.side === Ue && (p.bumpScale.value *= -1)), h.normalMap && (p.normalMap.value = h.normalMap, e(h.normalMap, p.normalMapTransform), p.normalScale.value.copy(h.normalScale), h.side === Ue && p.normalScale.value.negate()), h.displacementMap && (p.displacementMap.value = h.displacementMap, e(h.displacementMap, p.displacementMapTransform), p.displacementScale.value = h.displacementScale, p.displacementBias.value = h.displacementBias), h.emissiveMap && (p.emissiveMap.value = h.emissiveMap, e(h.emissiveMap, p.emissiveMapTransform)), h.specularMap && (p.specularMap.value = h.specularMap, e(h.specularMap, p.specularMapTransform)), h.alphaTest > 0 && (p.alphaTest.value = h.alphaTest);
    const E = t.get(h), b = E.envMap, S = E.envMapRotation;
    b && (p.envMap.value = b, Vn.copy(S), Vn.x *= -1, Vn.y *= -1, Vn.z *= -1, b.isCubeTexture && b.isRenderTargetTexture === !1 && (Vn.y *= -1, Vn.z *= -1), p.envMapRotation.value.setFromMatrix4(_m.makeRotationFromEuler(Vn)), p.flipEnvMap.value = b.isCubeTexture && b.isRenderTargetTexture === !1 ? -1 : 1, p.reflectivity.value = h.reflectivity, p.ior.value = h.ior, p.refractionRatio.value = h.refractionRatio), h.lightMap && (p.lightMap.value = h.lightMap, p.lightMapIntensity.value = h.lightMapIntensity, e(h.lightMap, p.lightMapTransform)), h.aoMap && (p.aoMap.value = h.aoMap, p.aoMapIntensity.value = h.aoMapIntensity, e(h.aoMap, p.aoMapTransform));
  }
  function a(p, h) {
    p.diffuse.value.copy(h.color), p.opacity.value = h.opacity, h.map && (p.map.value = h.map, e(h.map, p.mapTransform));
  }
  function o(p, h) {
    p.dashSize.value = h.dashSize, p.totalSize.value = h.dashSize + h.gapSize, p.scale.value = h.scale;
  }
  function l(p, h, E, b) {
    p.diffuse.value.copy(h.color), p.opacity.value = h.opacity, p.size.value = h.size * E, p.scale.value = b * 0.5, h.map && (p.map.value = h.map, e(h.map, p.uvTransform)), h.alphaMap && (p.alphaMap.value = h.alphaMap, e(h.alphaMap, p.alphaMapTransform)), h.alphaTest > 0 && (p.alphaTest.value = h.alphaTest);
  }
  function c(p, h) {
    p.diffuse.value.copy(h.color), p.opacity.value = h.opacity, p.rotation.value = h.rotation, h.map && (p.map.value = h.map, e(h.map, p.mapTransform)), h.alphaMap && (p.alphaMap.value = h.alphaMap, e(h.alphaMap, p.alphaMapTransform)), h.alphaTest > 0 && (p.alphaTest.value = h.alphaTest);
  }
  function u(p, h) {
    p.specular.value.copy(h.specular), p.shininess.value = Math.max(h.shininess, 1e-4);
  }
  function d(p, h) {
    h.gradientMap && (p.gradientMap.value = h.gradientMap);
  }
  function f(p, h) {
    p.metalness.value = h.metalness, h.metalnessMap && (p.metalnessMap.value = h.metalnessMap, e(h.metalnessMap, p.metalnessMapTransform)), p.roughness.value = h.roughness, h.roughnessMap && (p.roughnessMap.value = h.roughnessMap, e(h.roughnessMap, p.roughnessMapTransform)), h.envMap && (p.envMapIntensity.value = h.envMapIntensity);
  }
  function m(p, h, E) {
    p.ior.value = h.ior, h.sheen > 0 && (p.sheenColor.value.copy(h.sheenColor).multiplyScalar(h.sheen), p.sheenRoughness.value = h.sheenRoughness, h.sheenColorMap && (p.sheenColorMap.value = h.sheenColorMap, e(h.sheenColorMap, p.sheenColorMapTransform)), h.sheenRoughnessMap && (p.sheenRoughnessMap.value = h.sheenRoughnessMap, e(h.sheenRoughnessMap, p.sheenRoughnessMapTransform))), h.clearcoat > 0 && (p.clearcoat.value = h.clearcoat, p.clearcoatRoughness.value = h.clearcoatRoughness, h.clearcoatMap && (p.clearcoatMap.value = h.clearcoatMap, e(h.clearcoatMap, p.clearcoatMapTransform)), h.clearcoatRoughnessMap && (p.clearcoatRoughnessMap.value = h.clearcoatRoughnessMap, e(h.clearcoatRoughnessMap, p.clearcoatRoughnessMapTransform)), h.clearcoatNormalMap && (p.clearcoatNormalMap.value = h.clearcoatNormalMap, e(h.clearcoatNormalMap, p.clearcoatNormalMapTransform), p.clearcoatNormalScale.value.copy(h.clearcoatNormalScale), h.side === Ue && p.clearcoatNormalScale.value.negate())), h.dispersion > 0 && (p.dispersion.value = h.dispersion), h.iridescence > 0 && (p.iridescence.value = h.iridescence, p.iridescenceIOR.value = h.iridescenceIOR, p.iridescenceThicknessMinimum.value = h.iridescenceThicknessRange[0], p.iridescenceThicknessMaximum.value = h.iridescenceThicknessRange[1], h.iridescenceMap && (p.iridescenceMap.value = h.iridescenceMap, e(h.iridescenceMap, p.iridescenceMapTransform)), h.iridescenceThicknessMap && (p.iridescenceThicknessMap.value = h.iridescenceThicknessMap, e(h.iridescenceThicknessMap, p.iridescenceThicknessMapTransform))), h.transmission > 0 && (p.transmission.value = h.transmission, p.transmissionSamplerMap.value = E.texture, p.transmissionSamplerSize.value.set(E.width, E.height), h.transmissionMap && (p.transmissionMap.value = h.transmissionMap, e(h.transmissionMap, p.transmissionMapTransform)), p.thickness.value = h.thickness, h.thicknessMap && (p.thicknessMap.value = h.thicknessMap, e(h.thicknessMap, p.thicknessMapTransform)), p.attenuationDistance.value = h.attenuationDistance, p.attenuationColor.value.copy(h.attenuationColor)), h.anisotropy > 0 && (p.anisotropyVector.value.set(h.anisotropy * Math.cos(h.anisotropyRotation), h.anisotropy * Math.sin(h.anisotropyRotation)), h.anisotropyMap && (p.anisotropyMap.value = h.anisotropyMap, e(h.anisotropyMap, p.anisotropyMapTransform))), p.specularIntensity.value = h.specularIntensity, p.specularColor.value.copy(h.specularColor), h.specularColorMap && (p.specularColorMap.value = h.specularColorMap, e(h.specularColorMap, p.specularColorMapTransform)), h.specularIntensityMap && (p.specularIntensityMap.value = h.specularIntensityMap, e(h.specularIntensityMap, p.specularIntensityMapTransform));
  }
  function g(p, h) {
    h.matcap && (p.matcap.value = h.matcap);
  }
  function v(p, h) {
    const E = t.get(h).light;
    p.referencePosition.value.setFromMatrixPosition(E.matrixWorld), p.nearDistance.value = E.shadow.camera.near, p.farDistance.value = E.shadow.camera.far;
  }
  return {
    refreshFogUniforms: n,
    refreshMaterialUniforms: s
  };
}
function vm(i, t, e, n) {
  let s = {}, r = {}, a = [];
  const o = i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);
  function l(E, b) {
    const S = b.program;
    n.uniformBlockBinding(E, S);
  }
  function c(E, b) {
    let S = s[E.id];
    S === void 0 && (g(E), S = u(E), s[E.id] = S, E.addEventListener("dispose", p));
    const L = b.program;
    n.updateUBOMapping(E, L);
    const T = t.render.frame;
    r[E.id] !== T && (f(E), r[E.id] = T);
  }
  function u(E) {
    const b = d();
    E.__bindingPointIndex = b;
    const S = i.createBuffer(), L = E.__size, T = E.usage;
    return i.bindBuffer(i.UNIFORM_BUFFER, S), i.bufferData(i.UNIFORM_BUFFER, L, T), i.bindBuffer(i.UNIFORM_BUFFER, null), i.bindBufferBase(i.UNIFORM_BUFFER, b, S), S;
  }
  function d() {
    for (let E = 0; E < o; E++)
      if (a.indexOf(E) === -1)
        return a.push(E), E;
    return console.error("THREE.WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."), 0;
  }
  function f(E) {
    const b = s[E.id], S = E.uniforms, L = E.__cache;
    i.bindBuffer(i.UNIFORM_BUFFER, b);
    for (let T = 0, R = S.length; T < R; T++) {
      const I = Array.isArray(S[T]) ? S[T] : [S[T]];
      for (let y = 0, M = I.length; y < M; y++) {
        const C = I[y];
        if (m(C, T, y, L) === !0) {
          const H = C.__offset, z = Array.isArray(C.value) ? C.value : [C.value];
          let G = 0;
          for (let j = 0; j < z.length; j++) {
            const W = z[j], Q = v(W);
            typeof W == "number" || typeof W == "boolean" ? (C.__data[0] = W, i.bufferSubData(i.UNIFORM_BUFFER, H + G, C.__data)) : W.isMatrix3 ? (C.__data[0] = W.elements[0], C.__data[1] = W.elements[1], C.__data[2] = W.elements[2], C.__data[3] = 0, C.__data[4] = W.elements[3], C.__data[5] = W.elements[4], C.__data[6] = W.elements[5], C.__data[7] = 0, C.__data[8] = W.elements[6], C.__data[9] = W.elements[7], C.__data[10] = W.elements[8], C.__data[11] = 0) : (W.toArray(C.__data, G), G += Q.storage / Float32Array.BYTES_PER_ELEMENT);
          }
          i.bufferSubData(i.UNIFORM_BUFFER, H, C.__data);
        }
      }
    }
    i.bindBuffer(i.UNIFORM_BUFFER, null);
  }
  function m(E, b, S, L) {
    const T = E.value, R = b + "_" + S;
    if (L[R] === void 0)
      return typeof T == "number" || typeof T == "boolean" ? L[R] = T : L[R] = T.clone(), !0;
    {
      const I = L[R];
      if (typeof T == "number" || typeof T == "boolean") {
        if (I !== T)
          return L[R] = T, !0;
      } else if (I.equals(T) === !1)
        return I.copy(T), !0;
    }
    return !1;
  }
  function g(E) {
    const b = E.uniforms;
    let S = 0;
    const L = 16;
    for (let R = 0, I = b.length; R < I; R++) {
      const y = Array.isArray(b[R]) ? b[R] : [b[R]];
      for (let M = 0, C = y.length; M < C; M++) {
        const H = y[M], z = Array.isArray(H.value) ? H.value : [H.value];
        for (let G = 0, j = z.length; G < j; G++) {
          const W = z[G], Q = v(W), V = S % L, st = V % Q.boundary, ht = V + st;
          S += st, ht !== 0 && L - ht < Q.storage && (S += L - ht), H.__data = new Float32Array(Q.storage / Float32Array.BYTES_PER_ELEMENT), H.__offset = S, S += Q.storage;
        }
      }
    }
    const T = S % L;
    return T > 0 && (S += L - T), E.__size = S, E.__cache = {}, this;
  }
  function v(E) {
    const b = {
      boundary: 0,
      // bytes
      storage: 0
      // bytes
    };
    return typeof E == "number" || typeof E == "boolean" ? (b.boundary = 4, b.storage = 4) : E.isVector2 ? (b.boundary = 8, b.storage = 8) : E.isVector3 || E.isColor ? (b.boundary = 16, b.storage = 12) : E.isVector4 ? (b.boundary = 16, b.storage = 16) : E.isMatrix3 ? (b.boundary = 48, b.storage = 48) : E.isMatrix4 ? (b.boundary = 64, b.storage = 64) : E.isTexture ? console.warn("THREE.WebGLRenderer: Texture samplers can not be part of an uniforms group.") : console.warn("THREE.WebGLRenderer: Unsupported uniform value type.", E), b;
  }
  function p(E) {
    const b = E.target;
    b.removeEventListener("dispose", p);
    const S = a.indexOf(b.__bindingPointIndex);
    a.splice(S, 1), i.deleteBuffer(s[b.id]), delete s[b.id], delete r[b.id];
  }
  function h() {
    for (const E in s)
      i.deleteBuffer(s[E]);
    a = [], s = {}, r = {};
  }
  return {
    bind: l,
    update: c,
    dispose: h
  };
}
class xm {
  constructor(t = {}) {
    const {
      canvas: e = Zc(),
      context: n = null,
      depth: s = !0,
      stencil: r = !1,
      alpha: a = !1,
      antialias: o = !1,
      premultipliedAlpha: l = !0,
      preserveDrawingBuffer: c = !1,
      powerPreference: u = "default",
      failIfMajorPerformanceCaveat: d = !1,
      reverseDepthBuffer: f = !1
    } = t;
    this.isWebGLRenderer = !0;
    let m;
    if (n !== null) {
      if (typeof WebGLRenderingContext < "u" && n instanceof WebGLRenderingContext)
        throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");
      m = n.getContextAttributes().alpha;
    } else
      m = a;
    const g = new Uint32Array(4), v = new Int32Array(4);
    let p = null, h = null;
    const E = [], b = [];
    this.domElement = e, this.debug = {
      /**
       * Enables error checking and reporting when shader programs are being compiled
       * @type {boolean}
       */
      checkShaderErrors: !0,
      /**
       * Callback for custom error reporting.
       * @type {?Function}
       */
      onShaderError: null
    }, this.autoClear = !0, this.autoClearColor = !0, this.autoClearDepth = !0, this.autoClearStencil = !0, this.sortObjects = !0, this.clippingPlanes = [], this.localClippingEnabled = !1, this._outputColorSpace = Be, this.toneMapping = Un, this.toneMappingExposure = 1;
    const S = this;
    let L = !1, T = 0, R = 0, I = null, y = -1, M = null;
    const C = new te(), H = new te();
    let z = null;
    const G = new Vt(0);
    let j = 0, W = e.width, Q = e.height, V = 1, st = null, ht = null;
    const gt = new te(0, 0, W, Q), It = new te(0, 0, W, Q);
    let Kt = !1;
    const Y = new Tl();
    let tt = !1, mt = !1;
    const rt = new ne(), Et = new ne(), Rt = new P(), Nt = new te(), re = { background: null, fog: null, environment: null, overrideMaterial: null, isScene: !0 };
    let zt = !1;
    function ce() {
      return I === null ? V : 1;
    }
    let w = n;
    function ke(x, U) {
      return e.getContext(x, U);
    }
    try {
      const x = {
        alpha: !0,
        depth: s,
        stencil: r,
        antialias: o,
        premultipliedAlpha: l,
        preserveDrawingBuffer: c,
        powerPreference: u,
        failIfMajorPerformanceCaveat: d
      };
      if ("setAttribute" in e && e.setAttribute("data-engine", `three.js r${fa}`), e.addEventListener("webglcontextlost", q, !1), e.addEventListener("webglcontextrestored", lt, !1), e.addEventListener("webglcontextcreationerror", ot, !1), w === null) {
        const U = "webgl2";
        if (w = ke(U, x), w === null)
          throw ke(U) ? new Error("Error creating WebGL context with your selected attributes.") : new Error("Error creating WebGL context.");
      }
    } catch (x) {
      throw console.error("THREE.WebGLRenderer: " + x.message), x;
    }
    let Ft, Ot, vt, ee, xt, A, _, F, $, Z, X, _t, at, ut, Ht, J, dt, yt, At, ft, Bt, Dt, Jt, D;
    function nt() {
      Ft = new Rf(w), Ft.init(), Dt = new hm(w, Ft), Ot = new yf(w, Ft, t, Dt), vt = new lm(w, Ft), Ot.reverseDepthBuffer && f && vt.buffers.depth.setReversed(!0), ee = new Df(w), xt = new jp(), A = new cm(w, Ft, vt, xt, Ot, Dt, ee), _ = new bf(S), F = new wf(S), $ = new Oh(w), Jt = new Mf(w, $), Z = new Cf(w, $, ee, Jt), X = new Uf(w, Z, $, ee), At = new Lf(w, Ot, A), J = new Ef(xt), _t = new qp(S, _, F, Ft, Ot, Jt, J), at = new gm(S, xt), ut = new Kp(), Ht = new im(Ft), yt = new xf(S, _, F, vt, X, m, l), dt = new am(S, X, Ot), D = new vm(w, ee, Ot, vt), ft = new Sf(w, Ft, ee), Bt = new Pf(w, Ft, ee), ee.programs = _t.programs, S.capabilities = Ot, S.extensions = Ft, S.properties = xt, S.renderLists = ut, S.shadowMap = dt, S.state = vt, S.info = ee;
    }
    nt();
    const k = new mm(S, w);
    this.xr = k, this.getContext = function() {
      return w;
    }, this.getContextAttributes = function() {
      return w.getContextAttributes();
    }, this.forceContextLoss = function() {
      const x = Ft.get("WEBGL_lose_context");
      x && x.loseContext();
    }, this.forceContextRestore = function() {
      const x = Ft.get("WEBGL_lose_context");
      x && x.restoreContext();
    }, this.getPixelRatio = function() {
      return V;
    }, this.setPixelRatio = function(x) {
      x !== void 0 && (V = x, this.setSize(W, Q, !1));
    }, this.getSize = function(x) {
      return x.set(W, Q);
    }, this.setSize = function(x, U, O = !0) {
      if (k.isPresenting) {
        console.warn("THREE.WebGLRenderer: Can't change size while VR device is presenting.");
        return;
      }
      W = x, Q = U, e.width = Math.floor(x * V), e.height = Math.floor(U * V), O === !0 && (e.style.width = x + "px", e.style.height = U + "px"), this.setViewport(0, 0, x, U);
    }, this.getDrawingBufferSize = function(x) {
      return x.set(W * V, Q * V).floor();
    }, this.setDrawingBufferSize = function(x, U, O) {
      W = x, Q = U, V = O, e.width = Math.floor(x * O), e.height = Math.floor(U * O), this.setViewport(0, 0, x, U);
    }, this.getCurrentViewport = function(x) {
      return x.copy(C);
    }, this.getViewport = function(x) {
      return x.copy(gt);
    }, this.setViewport = function(x, U, O, B) {
      x.isVector4 ? gt.set(x.x, x.y, x.z, x.w) : gt.set(x, U, O, B), vt.viewport(C.copy(gt).multiplyScalar(V).round());
    }, this.getScissor = function(x) {
      return x.copy(It);
    }, this.setScissor = function(x, U, O, B) {
      x.isVector4 ? It.set(x.x, x.y, x.z, x.w) : It.set(x, U, O, B), vt.scissor(H.copy(It).multiplyScalar(V).round());
    }, this.getScissorTest = function() {
      return Kt;
    }, this.setScissorTest = function(x) {
      vt.setScissorTest(Kt = x);
    }, this.setOpaqueSort = function(x) {
      st = x;
    }, this.setTransparentSort = function(x) {
      ht = x;
    }, this.getClearColor = function(x) {
      return x.copy(yt.getClearColor());
    }, this.setClearColor = function() {
      yt.setClearColor.apply(yt, arguments);
    }, this.getClearAlpha = function() {
      return yt.getClearAlpha();
    }, this.setClearAlpha = function() {
      yt.setClearAlpha.apply(yt, arguments);
    }, this.clear = function(x = !0, U = !0, O = !0) {
      let B = 0;
      if (x) {
        let N = !1;
        if (I !== null) {
          const K = I.texture.format;
          N = K === xa || K === va || K === ga;
        }
        if (N) {
          const K = I.texture.type, it = K === Sn || K === jn || K === Vi || K === Ti || K === ma || K === _a, ct = yt.getClearColor(), pt = yt.getClearAlpha(), Tt = ct.r, wt = ct.g, Mt = ct.b;
          it ? (g[0] = Tt, g[1] = wt, g[2] = Mt, g[3] = pt, w.clearBufferuiv(w.COLOR, 0, g)) : (v[0] = Tt, v[1] = wt, v[2] = Mt, v[3] = pt, w.clearBufferiv(w.COLOR, 0, v));
        } else
          B |= w.COLOR_BUFFER_BIT;
      }
      U && (B |= w.DEPTH_BUFFER_BIT), O && (B |= w.STENCIL_BUFFER_BIT, this.state.buffers.stencil.setMask(4294967295)), w.clear(B);
    }, this.clearColor = function() {
      this.clear(!0, !1, !1);
    }, this.clearDepth = function() {
      this.clear(!1, !0, !1);
    }, this.clearStencil = function() {
      this.clear(!1, !1, !0);
    }, this.dispose = function() {
      e.removeEventListener("webglcontextlost", q, !1), e.removeEventListener("webglcontextrestored", lt, !1), e.removeEventListener("webglcontextcreationerror", ot, !1), yt.dispose(), ut.dispose(), Ht.dispose(), xt.dispose(), _.dispose(), F.dispose(), X.dispose(), Jt.dispose(), D.dispose(), _t.dispose(), k.dispose(), k.removeEventListener("sessionstart", La), k.removeEventListener("sessionend", Ua), Nn.stop();
    };
    function q(x) {
      x.preventDefault(), console.log("THREE.WebGLRenderer: Context Lost."), L = !0;
    }
    function lt() {
      console.log("THREE.WebGLRenderer: Context Restored."), L = !1;
      const x = ee.autoReset, U = dt.enabled, O = dt.autoUpdate, B = dt.needsUpdate, N = dt.type;
      nt(), ee.autoReset = x, dt.enabled = U, dt.autoUpdate = O, dt.needsUpdate = B, dt.type = N;
    }
    function ot(x) {
      console.error("THREE.WebGLRenderer: A WebGL context could not be created. Reason: ", x.statusMessage);
    }
    function Ct(x) {
      const U = x.target;
      U.removeEventListener("dispose", Ct), ae(U);
    }
    function ae(x) {
      Se(x), xt.remove(x);
    }
    function Se(x) {
      const U = xt.get(x).programs;
      U !== void 0 && (U.forEach(function(O) {
        _t.releaseProgram(O);
      }), x.isShaderMaterial && _t.releaseShaderCache(x));
    }
    this.renderBufferDirect = function(x, U, O, B, N, K) {
      U === null && (U = re);
      const it = N.isMesh && N.matrixWorld.determinant() < 0, ct = Hl(x, U, O, B, N);
      vt.setMaterial(B, it);
      let pt = O.index, Tt = 1;
      if (B.wireframe === !0) {
        if (pt = Z.getWireframeAttribute(O), pt === void 0) return;
        Tt = 2;
      }
      const wt = O.drawRange, Mt = O.attributes.position;
      let kt = wt.start * Tt, Xt = (wt.start + wt.count) * Tt;
      K !== null && (kt = Math.max(kt, K.start * Tt), Xt = Math.min(Xt, (K.start + K.count) * Tt)), pt !== null ? (kt = Math.max(kt, 0), Xt = Math.min(Xt, pt.count)) : Mt != null && (kt = Math.max(kt, 0), Xt = Math.min(Xt, Mt.count));
      const he = Xt - kt;
      if (he < 0 || he === 1 / 0) return;
      Jt.setup(N, B, ct, O, pt);
      let oe, Gt = ft;
      if (pt !== null && (oe = $.get(pt), Gt = Bt, Gt.setIndex(oe)), N.isMesh)
        B.wireframe === !0 ? (vt.setLineWidth(B.wireframeLinewidth * ce()), Gt.setMode(w.LINES)) : Gt.setMode(w.TRIANGLES);
      else if (N.isLine) {
        let St = B.linewidth;
        St === void 0 && (St = 1), vt.setLineWidth(St * ce()), N.isLineSegments ? Gt.setMode(w.LINES) : N.isLineLoop ? Gt.setMode(w.LINE_LOOP) : Gt.setMode(w.LINE_STRIP);
      } else N.isPoints ? Gt.setMode(w.POINTS) : N.isSprite && Gt.setMode(w.TRIANGLES);
      if (N.isBatchedMesh)
        if (N._multiDrawInstances !== null)
          Gt.renderMultiDrawInstances(N._multiDrawStarts, N._multiDrawCounts, N._multiDrawCount, N._multiDrawInstances);
        else if (Ft.get("WEBGL_multi_draw"))
          Gt.renderMultiDraw(N._multiDrawStarts, N._multiDrawCounts, N._multiDrawCount);
        else {
          const St = N._multiDrawStarts, ge = N._multiDrawCounts, Yt = N._multiDrawCount, $e = pt ? $.get(pt).bytesPerElement : 1, ei = xt.get(B).currentProgram.getUniforms();
          for (let Ne = 0; Ne < Yt; Ne++)
            ei.setValue(w, "_gl_DrawID", Ne), Gt.render(St[Ne] / $e, ge[Ne]);
        }
      else if (N.isInstancedMesh)
        Gt.renderInstances(kt, he, N.count);
      else if (O.isInstancedBufferGeometry) {
        const St = O._maxInstanceCount !== void 0 ? O._maxInstanceCount : 1 / 0, ge = Math.min(O.instanceCount, St);
        Gt.renderInstances(kt, he, ge);
      } else
        Gt.render(kt, he);
    };
    function $t(x, U, O) {
      x.transparent === !0 && x.side === Xe && x.forceSinglePass === !1 ? (x.side = Ue, x.needsUpdate = !0, qi(x, U, O), x.side = In, x.needsUpdate = !0, qi(x, U, O), x.side = Xe) : qi(x, U, O);
    }
    this.compile = function(x, U, O = null) {
      O === null && (O = x), h = Ht.get(O), h.init(U), b.push(h), O.traverseVisible(function(N) {
        N.isLight && N.layers.test(U.layers) && (h.pushLight(N), N.castShadow && h.pushShadow(N));
      }), x !== O && x.traverseVisible(function(N) {
        N.isLight && N.layers.test(U.layers) && (h.pushLight(N), N.castShadow && h.pushShadow(N));
      }), h.setupLights();
      const B = /* @__PURE__ */ new Set();
      return x.traverse(function(N) {
        if (!(N.isMesh || N.isPoints || N.isLine || N.isSprite))
          return;
        const K = N.material;
        if (K)
          if (Array.isArray(K))
            for (let it = 0; it < K.length; it++) {
              const ct = K[it];
              $t(ct, O, N), B.add(ct);
            }
          else
            $t(K, O, N), B.add(K);
      }), b.pop(), h = null, B;
    }, this.compileAsync = function(x, U, O = null) {
      const B = this.compile(x, U, O);
      return new Promise((N) => {
        function K() {
          if (B.forEach(function(it) {
            xt.get(it).currentProgram.isReady() && B.delete(it);
          }), B.size === 0) {
            N(x);
            return;
          }
          setTimeout(K, 10);
        }
        Ft.get("KHR_parallel_shader_compile") !== null ? K() : setTimeout(K, 10);
      });
    };
    let Ye = null;
    function ln(x) {
      Ye && Ye(x);
    }
    function La() {
      Nn.stop();
    }
    function Ua() {
      Nn.start();
    }
    const Nn = new Cl();
    Nn.setAnimationLoop(ln), typeof self < "u" && Nn.setContext(self), this.setAnimationLoop = function(x) {
      Ye = x, k.setAnimationLoop(x), x === null ? Nn.stop() : Nn.start();
    }, k.addEventListener("sessionstart", La), k.addEventListener("sessionend", Ua), this.render = function(x, U) {
      if (U !== void 0 && U.isCamera !== !0) {
        console.error("THREE.WebGLRenderer.render: camera is not an instance of THREE.Camera.");
        return;
      }
      if (L === !0) return;
      if (x.matrixWorldAutoUpdate === !0 && x.updateMatrixWorld(), U.parent === null && U.matrixWorldAutoUpdate === !0 && U.updateMatrixWorld(), k.enabled === !0 && k.isPresenting === !0 && (k.cameraAutoUpdate === !0 && k.updateCamera(U), U = k.getCamera()), x.isScene === !0 && x.onBeforeRender(S, x, U, I), h = Ht.get(x, b.length), h.init(U), b.push(h), Et.multiplyMatrices(U.projectionMatrix, U.matrixWorldInverse), Y.setFromProjectionMatrix(Et), mt = this.localClippingEnabled, tt = J.init(this.clippingPlanes, mt), p = ut.get(x, E.length), p.init(), E.push(p), k.enabled === !0 && k.isPresenting === !0) {
        const K = S.xr.getDepthSensingMesh();
        K !== null && zs(K, U, -1 / 0, S.sortObjects);
      }
      zs(x, U, 0, S.sortObjects), p.finish(), S.sortObjects === !0 && p.sort(st, ht), zt = k.enabled === !1 || k.isPresenting === !1 || k.hasDepthSensing() === !1, zt && yt.addToRenderList(p, x), this.info.render.frame++, tt === !0 && J.beginShadows();
      const O = h.state.shadowsArray;
      dt.render(O, x, U), tt === !0 && J.endShadows(), this.info.autoReset === !0 && this.info.reset();
      const B = p.opaque, N = p.transmissive;
      if (h.setupLights(), U.isArrayCamera) {
        const K = U.cameras;
        if (N.length > 0)
          for (let it = 0, ct = K.length; it < ct; it++) {
            const pt = K[it];
            Na(B, N, x, pt);
          }
        zt && yt.render(x);
        for (let it = 0, ct = K.length; it < ct; it++) {
          const pt = K[it];
          Ia(p, x, pt, pt.viewport);
        }
      } else
        N.length > 0 && Na(B, N, x, U), zt && yt.render(x), Ia(p, x, U);
      I !== null && (A.updateMultisampleRenderTarget(I), A.updateRenderTargetMipmap(I)), x.isScene === !0 && x.onAfterRender(S, x, U), Jt.resetDefaultState(), y = -1, M = null, b.pop(), b.length > 0 ? (h = b[b.length - 1], tt === !0 && J.setGlobalState(S.clippingPlanes, h.state.camera)) : h = null, E.pop(), E.length > 0 ? p = E[E.length - 1] : p = null;
    };
    function zs(x, U, O, B) {
      if (x.visible === !1) return;
      if (x.layers.test(U.layers)) {
        if (x.isGroup)
          O = x.renderOrder;
        else if (x.isLOD)
          x.autoUpdate === !0 && x.update(U);
        else if (x.isLight)
          h.pushLight(x), x.castShadow && h.pushShadow(x);
        else if (x.isSprite) {
          if (!x.frustumCulled || Y.intersectsSprite(x)) {
            B && Nt.setFromMatrixPosition(x.matrixWorld).applyMatrix4(Et);
            const it = X.update(x), ct = x.material;
            ct.visible && p.push(x, it, ct, O, Nt.z, null);
          }
        } else if ((x.isMesh || x.isLine || x.isPoints) && (!x.frustumCulled || Y.intersectsObject(x))) {
          const it = X.update(x), ct = x.material;
          if (B && (x.boundingSphere !== void 0 ? (x.boundingSphere === null && x.computeBoundingSphere(), Nt.copy(x.boundingSphere.center)) : (it.boundingSphere === null && it.computeBoundingSphere(), Nt.copy(it.boundingSphere.center)), Nt.applyMatrix4(x.matrixWorld).applyMatrix4(Et)), Array.isArray(ct)) {
            const pt = it.groups;
            for (let Tt = 0, wt = pt.length; Tt < wt; Tt++) {
              const Mt = pt[Tt], kt = ct[Mt.materialIndex];
              kt && kt.visible && p.push(x, it, kt, O, Nt.z, Mt);
            }
          } else ct.visible && p.push(x, it, ct, O, Nt.z, null);
        }
      }
      const K = x.children;
      for (let it = 0, ct = K.length; it < ct; it++)
        zs(K[it], U, O, B);
    }
    function Ia(x, U, O, B) {
      const N = x.opaque, K = x.transmissive, it = x.transparent;
      h.setupLightsView(O), tt === !0 && J.setGlobalState(S.clippingPlanes, O), B && vt.viewport(C.copy(B)), N.length > 0 && $i(N, U, O), K.length > 0 && $i(K, U, O), it.length > 0 && $i(it, U, O), vt.buffers.depth.setTest(!0), vt.buffers.depth.setMask(!0), vt.buffers.color.setMask(!0), vt.setPolygonOffset(!1);
    }
    function Na(x, U, O, B) {
      if ((O.isScene === !0 ? O.overrideMaterial : null) !== null)
        return;
      h.state.transmissionRenderTarget[B.id] === void 0 && (h.state.transmissionRenderTarget[B.id] = new Zn(1, 1, {
        generateMipmaps: !0,
        type: Ft.has("EXT_color_buffer_half_float") || Ft.has("EXT_color_buffer_float") ? Xi : Sn,
        minFilter: $n,
        samples: 4,
        stencilBuffer: r,
        resolveDepthBuffer: !1,
        resolveStencilBuffer: !1,
        colorSpace: Wt.workingColorSpace
      }));
      const K = h.state.transmissionRenderTarget[B.id], it = B.viewport || C;
      K.setSize(it.z, it.w);
      const ct = S.getRenderTarget();
      S.setRenderTarget(K), S.getClearColor(G), j = S.getClearAlpha(), j < 1 && S.setClearColor(16777215, 0.5), S.clear(), zt && yt.render(O);
      const pt = S.toneMapping;
      S.toneMapping = Un;
      const Tt = B.viewport;
      if (B.viewport !== void 0 && (B.viewport = void 0), h.setupLightsView(B), tt === !0 && J.setGlobalState(S.clippingPlanes, B), $i(x, O, B), A.updateMultisampleRenderTarget(K), A.updateRenderTargetMipmap(K), Ft.has("WEBGL_multisampled_render_to_texture") === !1) {
        let wt = !1;
        for (let Mt = 0, kt = U.length; Mt < kt; Mt++) {
          const Xt = U[Mt], he = Xt.object, oe = Xt.geometry, Gt = Xt.material, St = Xt.group;
          if (Gt.side === Xe && he.layers.test(B.layers)) {
            const ge = Gt.side;
            Gt.side = Ue, Gt.needsUpdate = !0, Fa(he, O, B, oe, Gt, St), Gt.side = ge, Gt.needsUpdate = !0, wt = !0;
          }
        }
        wt === !0 && (A.updateMultisampleRenderTarget(K), A.updateRenderTargetMipmap(K));
      }
      S.setRenderTarget(ct), S.setClearColor(G, j), Tt !== void 0 && (B.viewport = Tt), S.toneMapping = pt;
    }
    function $i(x, U, O) {
      const B = U.isScene === !0 ? U.overrideMaterial : null;
      for (let N = 0, K = x.length; N < K; N++) {
        const it = x[N], ct = it.object, pt = it.geometry, Tt = B === null ? it.material : B, wt = it.group;
        ct.layers.test(O.layers) && Fa(ct, U, O, pt, Tt, wt);
      }
    }
    function Fa(x, U, O, B, N, K) {
      x.onBeforeRender(S, U, O, B, N, K), x.modelViewMatrix.multiplyMatrices(O.matrixWorldInverse, x.matrixWorld), x.normalMatrix.getNormalMatrix(x.modelViewMatrix), N.onBeforeRender(S, U, O, B, x, K), N.transparent === !0 && N.side === Xe && N.forceSinglePass === !1 ? (N.side = Ue, N.needsUpdate = !0, S.renderBufferDirect(O, U, B, N, x, K), N.side = In, N.needsUpdate = !0, S.renderBufferDirect(O, U, B, N, x, K), N.side = Xe) : S.renderBufferDirect(O, U, B, N, x, K), x.onAfterRender(S, U, O, B, N, K);
    }
    function qi(x, U, O) {
      U.isScene !== !0 && (U = re);
      const B = xt.get(x), N = h.state.lights, K = h.state.shadowsArray, it = N.state.version, ct = _t.getParameters(x, N.state, K, U, O), pt = _t.getProgramCacheKey(ct);
      let Tt = B.programs;
      B.environment = x.isMeshStandardMaterial ? U.environment : null, B.fog = U.fog, B.envMap = (x.isMeshStandardMaterial ? F : _).get(x.envMap || B.environment), B.envMapRotation = B.environment !== null && x.envMap === null ? U.environmentRotation : x.envMapRotation, Tt === void 0 && (x.addEventListener("dispose", Ct), Tt = /* @__PURE__ */ new Map(), B.programs = Tt);
      let wt = Tt.get(pt);
      if (wt !== void 0) {
        if (B.currentProgram === wt && B.lightsStateVersion === it)
          return Ba(x, ct), wt;
      } else
        ct.uniforms = _t.getUniforms(x), x.onBeforeCompile(ct, S), wt = _t.acquireProgram(ct, pt), Tt.set(pt, wt), B.uniforms = ct.uniforms;
      const Mt = B.uniforms;
      return (!x.isShaderMaterial && !x.isRawShaderMaterial || x.clipping === !0) && (Mt.clippingPlanes = J.uniform), Ba(x, ct), B.needsLights = Vl(x), B.lightsStateVersion = it, B.needsLights && (Mt.ambientLightColor.value = N.state.ambient, Mt.lightProbe.value = N.state.probe, Mt.directionalLights.value = N.state.directional, Mt.directionalLightShadows.value = N.state.directionalShadow, Mt.spotLights.value = N.state.spot, Mt.spotLightShadows.value = N.state.spotShadow, Mt.rectAreaLights.value = N.state.rectArea, Mt.ltc_1.value = N.state.rectAreaLTC1, Mt.ltc_2.value = N.state.rectAreaLTC2, Mt.pointLights.value = N.state.point, Mt.pointLightShadows.value = N.state.pointShadow, Mt.hemisphereLights.value = N.state.hemi, Mt.directionalShadowMap.value = N.state.directionalShadowMap, Mt.directionalShadowMatrix.value = N.state.directionalShadowMatrix, Mt.spotShadowMap.value = N.state.spotShadowMap, Mt.spotLightMatrix.value = N.state.spotLightMatrix, Mt.spotLightMap.value = N.state.spotLightMap, Mt.pointShadowMap.value = N.state.pointShadowMap, Mt.pointShadowMatrix.value = N.state.pointShadowMatrix), B.currentProgram = wt, B.uniformsList = null, wt;
    }
    function Oa(x) {
      if (x.uniformsList === null) {
        const U = x.currentProgram.getUniforms();
        x.uniformsList = Cs.seqWithValue(U.seq, x.uniforms);
      }
      return x.uniformsList;
    }
    function Ba(x, U) {
      const O = xt.get(x);
      O.outputColorSpace = U.outputColorSpace, O.batching = U.batching, O.batchingColor = U.batchingColor, O.instancing = U.instancing, O.instancingColor = U.instancingColor, O.instancingMorph = U.instancingMorph, O.skinning = U.skinning, O.morphTargets = U.morphTargets, O.morphNormals = U.morphNormals, O.morphColors = U.morphColors, O.morphTargetsCount = U.morphTargetsCount, O.numClippingPlanes = U.numClippingPlanes, O.numIntersection = U.numClipIntersection, O.vertexAlphas = U.vertexAlphas, O.vertexTangents = U.vertexTangents, O.toneMapping = U.toneMapping;
    }
    function Hl(x, U, O, B, N) {
      U.isScene !== !0 && (U = re), A.resetTextureUnits();
      const K = U.fog, it = B.isMeshStandardMaterial ? U.environment : null, ct = I === null ? S.outputColorSpace : I.isXRRenderTarget === !0 ? I.texture.colorSpace : Ri, pt = (B.isMeshStandardMaterial ? F : _).get(B.envMap || it), Tt = B.vertexColors === !0 && !!O.attributes.color && O.attributes.color.itemSize === 4, wt = !!O.attributes.tangent && (!!B.normalMap || B.anisotropy > 0), Mt = !!O.morphAttributes.position, kt = !!O.morphAttributes.normal, Xt = !!O.morphAttributes.color;
      let he = Un;
      B.toneMapped && (I === null || I.isXRRenderTarget === !0) && (he = S.toneMapping);
      const oe = O.morphAttributes.position || O.morphAttributes.normal || O.morphAttributes.color, Gt = oe !== void 0 ? oe.length : 0, St = xt.get(B), ge = h.state.lights;
      if (tt === !0 && (mt === !0 || x !== M)) {
        const Ae = x === M && B.id === y;
        J.setState(B, x, Ae);
      }
      let Yt = !1;
      B.version === St.__version ? (St.needsLights && St.lightsStateVersion !== ge.state.version || St.outputColorSpace !== ct || N.isBatchedMesh && St.batching === !1 || !N.isBatchedMesh && St.batching === !0 || N.isBatchedMesh && St.batchingColor === !0 && N.colorTexture === null || N.isBatchedMesh && St.batchingColor === !1 && N.colorTexture !== null || N.isInstancedMesh && St.instancing === !1 || !N.isInstancedMesh && St.instancing === !0 || N.isSkinnedMesh && St.skinning === !1 || !N.isSkinnedMesh && St.skinning === !0 || N.isInstancedMesh && St.instancingColor === !0 && N.instanceColor === null || N.isInstancedMesh && St.instancingColor === !1 && N.instanceColor !== null || N.isInstancedMesh && St.instancingMorph === !0 && N.morphTexture === null || N.isInstancedMesh && St.instancingMorph === !1 && N.morphTexture !== null || St.envMap !== pt || B.fog === !0 && St.fog !== K || St.numClippingPlanes !== void 0 && (St.numClippingPlanes !== J.numPlanes || St.numIntersection !== J.numIntersection) || St.vertexAlphas !== Tt || St.vertexTangents !== wt || St.morphTargets !== Mt || St.morphNormals !== kt || St.morphColors !== Xt || St.toneMapping !== he || St.morphTargetsCount !== Gt) && (Yt = !0) : (Yt = !0, St.__version = B.version);
      let $e = St.currentProgram;
      Yt === !0 && ($e = qi(B, U, N));
      let ei = !1, Ne = !1, Li = !1;
      const ie = $e.getUniforms(), Ve = St.uniforms;
      if (vt.useProgram($e.program) && (ei = !0, Ne = !0, Li = !0), B.id !== y && (y = B.id, Ne = !0), ei || M !== x) {
        vt.buffers.depth.getReversed() ? (rt.copy(x.projectionMatrix), Jc(rt), Qc(rt), ie.setValue(w, "projectionMatrix", rt)) : ie.setValue(w, "projectionMatrix", x.projectionMatrix), ie.setValue(w, "viewMatrix", x.matrixWorldInverse);
        const Pe = ie.map.cameraPosition;
        Pe !== void 0 && Pe.setValue(w, Rt.setFromMatrixPosition(x.matrixWorld)), Ot.logarithmicDepthBuffer && ie.setValue(
          w,
          "logDepthBufFC",
          2 / (Math.log(x.far + 1) / Math.LN2)
        ), (B.isMeshPhongMaterial || B.isMeshToonMaterial || B.isMeshLambertMaterial || B.isMeshBasicMaterial || B.isMeshStandardMaterial || B.isShaderMaterial) && ie.setValue(w, "isOrthographic", x.isOrthographicCamera === !0), M !== x && (M = x, Ne = !0, Li = !0);
      }
      if (N.isSkinnedMesh) {
        ie.setOptional(w, N, "bindMatrix"), ie.setOptional(w, N, "bindMatrixInverse");
        const Ae = N.skeleton;
        Ae && (Ae.boneTexture === null && Ae.computeBoneTexture(), ie.setValue(w, "boneTexture", Ae.boneTexture, A));
      }
      N.isBatchedMesh && (ie.setOptional(w, N, "batchingTexture"), ie.setValue(w, "batchingTexture", N._matricesTexture, A), ie.setOptional(w, N, "batchingIdTexture"), ie.setValue(w, "batchingIdTexture", N._indirectTexture, A), ie.setOptional(w, N, "batchingColorTexture"), N._colorsTexture !== null && ie.setValue(w, "batchingColorTexture", N._colorsTexture, A));
      const Ge = O.morphAttributes;
      if ((Ge.position !== void 0 || Ge.normal !== void 0 || Ge.color !== void 0) && At.update(N, O, $e), (Ne || St.receiveShadow !== N.receiveShadow) && (St.receiveShadow = N.receiveShadow, ie.setValue(w, "receiveShadow", N.receiveShadow)), B.isMeshGouraudMaterial && B.envMap !== null && (Ve.envMap.value = pt, Ve.flipEnvMap.value = pt.isCubeTexture && pt.isRenderTargetTexture === !1 ? -1 : 1), B.isMeshStandardMaterial && B.envMap === null && U.environment !== null && (Ve.envMapIntensity.value = U.environmentIntensity), Ne && (ie.setValue(w, "toneMappingExposure", S.toneMappingExposure), St.needsLights && kl(Ve, Li), K && B.fog === !0 && at.refreshFogUniforms(Ve, K), at.refreshMaterialUniforms(Ve, B, V, Q, h.state.transmissionRenderTarget[x.id]), Cs.upload(w, Oa(St), Ve, A)), B.isShaderMaterial && B.uniformsNeedUpdate === !0 && (Cs.upload(w, Oa(St), Ve, A), B.uniformsNeedUpdate = !1), B.isSpriteMaterial && ie.setValue(w, "center", N.center), ie.setValue(w, "modelViewMatrix", N.modelViewMatrix), ie.setValue(w, "normalMatrix", N.normalMatrix), ie.setValue(w, "modelMatrix", N.matrixWorld), B.isShaderMaterial || B.isRawShaderMaterial) {
        const Ae = B.uniformsGroups;
        for (let Pe = 0, Hs = Ae.length; Pe < Hs; Pe++) {
          const Fn = Ae[Pe];
          D.update(Fn, $e), D.bind(Fn, $e);
        }
      }
      return $e;
    }
    function kl(x, U) {
      x.ambientLightColor.needsUpdate = U, x.lightProbe.needsUpdate = U, x.directionalLights.needsUpdate = U, x.directionalLightShadows.needsUpdate = U, x.pointLights.needsUpdate = U, x.pointLightShadows.needsUpdate = U, x.spotLights.needsUpdate = U, x.spotLightShadows.needsUpdate = U, x.rectAreaLights.needsUpdate = U, x.hemisphereLights.needsUpdate = U;
    }
    function Vl(x) {
      return x.isMeshLambertMaterial || x.isMeshToonMaterial || x.isMeshPhongMaterial || x.isMeshStandardMaterial || x.isShadowMaterial || x.isShaderMaterial && x.lights === !0;
    }
    this.getActiveCubeFace = function() {
      return T;
    }, this.getActiveMipmapLevel = function() {
      return R;
    }, this.getRenderTarget = function() {
      return I;
    }, this.setRenderTargetTextures = function(x, U, O) {
      xt.get(x.texture).__webglTexture = U, xt.get(x.depthTexture).__webglTexture = O;
      const B = xt.get(x);
      B.__hasExternalTextures = !0, B.__autoAllocateDepthBuffer = O === void 0, B.__autoAllocateDepthBuffer || Ft.has("WEBGL_multisampled_render_to_texture") === !0 && (console.warn("THREE.WebGLRenderer: Render-to-texture extension was disabled because an external texture was provided"), B.__useRenderToTexture = !1);
    }, this.setRenderTargetFramebuffer = function(x, U) {
      const O = xt.get(x);
      O.__webglFramebuffer = U, O.__useDefaultFramebuffer = U === void 0;
    }, this.setRenderTarget = function(x, U = 0, O = 0) {
      I = x, T = U, R = O;
      let B = !0, N = null, K = !1, it = !1;
      if (x) {
        const pt = xt.get(x);
        if (pt.__useDefaultFramebuffer !== void 0)
          vt.bindFramebuffer(w.FRAMEBUFFER, null), B = !1;
        else if (pt.__webglFramebuffer === void 0)
          A.setupRenderTarget(x);
        else if (pt.__hasExternalTextures)
          A.rebindTextures(x, xt.get(x.texture).__webglTexture, xt.get(x.depthTexture).__webglTexture);
        else if (x.depthBuffer) {
          const Mt = x.depthTexture;
          if (pt.__boundDepthTexture !== Mt) {
            if (Mt !== null && xt.has(Mt) && (x.width !== Mt.image.width || x.height !== Mt.image.height))
              throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");
            A.setupDepthRenderbuffer(x);
          }
        }
        const Tt = x.texture;
        (Tt.isData3DTexture || Tt.isDataArrayTexture || Tt.isCompressedArrayTexture) && (it = !0);
        const wt = xt.get(x).__webglFramebuffer;
        x.isWebGLCubeRenderTarget ? (Array.isArray(wt[U]) ? N = wt[U][O] : N = wt[U], K = !0) : x.samples > 0 && A.useMultisampledRTT(x) === !1 ? N = xt.get(x).__webglMultisampledFramebuffer : Array.isArray(wt) ? N = wt[O] : N = wt, C.copy(x.viewport), H.copy(x.scissor), z = x.scissorTest;
      } else
        C.copy(gt).multiplyScalar(V).floor(), H.copy(It).multiplyScalar(V).floor(), z = Kt;
      if (vt.bindFramebuffer(w.FRAMEBUFFER, N) && B && vt.drawBuffers(x, N), vt.viewport(C), vt.scissor(H), vt.setScissorTest(z), K) {
        const pt = xt.get(x.texture);
        w.framebufferTexture2D(w.FRAMEBUFFER, w.COLOR_ATTACHMENT0, w.TEXTURE_CUBE_MAP_POSITIVE_X + U, pt.__webglTexture, O);
      } else if (it) {
        const pt = xt.get(x.texture), Tt = U || 0;
        w.framebufferTextureLayer(w.FRAMEBUFFER, w.COLOR_ATTACHMENT0, pt.__webglTexture, O || 0, Tt);
      }
      y = -1;
    }, this.readRenderTargetPixels = function(x, U, O, B, N, K, it) {
      if (!(x && x.isWebGLRenderTarget)) {
        console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
        return;
      }
      let ct = xt.get(x).__webglFramebuffer;
      if (x.isWebGLCubeRenderTarget && it !== void 0 && (ct = ct[it]), ct) {
        vt.bindFramebuffer(w.FRAMEBUFFER, ct);
        try {
          const pt = x.texture, Tt = pt.format, wt = pt.type;
          if (!Ot.textureFormatReadable(Tt)) {
            console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");
            return;
          }
          if (!Ot.textureTypeReadable(wt)) {
            console.error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");
            return;
          }
          U >= 0 && U <= x.width - B && O >= 0 && O <= x.height - N && w.readPixels(U, O, B, N, Dt.convert(Tt), Dt.convert(wt), K);
        } finally {
          const pt = I !== null ? xt.get(I).__webglFramebuffer : null;
          vt.bindFramebuffer(w.FRAMEBUFFER, pt);
        }
      }
    }, this.readRenderTargetPixelsAsync = async function(x, U, O, B, N, K, it) {
      if (!(x && x.isWebGLRenderTarget))
        throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");
      let ct = xt.get(x).__webglFramebuffer;
      if (x.isWebGLCubeRenderTarget && it !== void 0 && (ct = ct[it]), ct) {
        const pt = x.texture, Tt = pt.format, wt = pt.type;
        if (!Ot.textureFormatReadable(Tt))
          throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");
        if (!Ot.textureTypeReadable(wt))
          throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");
        if (U >= 0 && U <= x.width - B && O >= 0 && O <= x.height - N) {
          vt.bindFramebuffer(w.FRAMEBUFFER, ct);
          const Mt = w.createBuffer();
          w.bindBuffer(w.PIXEL_PACK_BUFFER, Mt), w.bufferData(w.PIXEL_PACK_BUFFER, K.byteLength, w.STREAM_READ), w.readPixels(U, O, B, N, Dt.convert(Tt), Dt.convert(wt), 0);
          const kt = I !== null ? xt.get(I).__webglFramebuffer : null;
          vt.bindFramebuffer(w.FRAMEBUFFER, kt);
          const Xt = w.fenceSync(w.SYNC_GPU_COMMANDS_COMPLETE, 0);
          return w.flush(), await Kc(w, Xt, 4), w.bindBuffer(w.PIXEL_PACK_BUFFER, Mt), w.getBufferSubData(w.PIXEL_PACK_BUFFER, 0, K), w.deleteBuffer(Mt), w.deleteSync(Xt), K;
        } else
          throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.");
      }
    }, this.copyFramebufferToTexture = function(x, U = null, O = 0) {
      x.isTexture !== !0 && (_i("WebGLRenderer: copyFramebufferToTexture function signature has changed."), U = arguments[0] || null, x = arguments[1]);
      const B = Math.pow(2, -O), N = Math.floor(x.image.width * B), K = Math.floor(x.image.height * B), it = U !== null ? U.x : 0, ct = U !== null ? U.y : 0;
      A.setTexture2D(x, 0), w.copyTexSubImage2D(w.TEXTURE_2D, O, 0, 0, it, ct, N, K), vt.unbindTexture();
    };
    const Gl = w.createFramebuffer(), Wl = w.createFramebuffer();
    this.copyTextureToTexture = function(x, U, O = null, B = null, N = 0, K = null) {
      x.isTexture !== !0 && (_i("WebGLRenderer: copyTextureToTexture function signature has changed."), B = arguments[0] || null, x = arguments[1], U = arguments[2], K = arguments[3] || 0, O = null), K === null && (N !== 0 ? (_i("WebGLRenderer: copyTextureToTexture function signature has changed to support src and dst mipmap levels."), K = N, N = 0) : K = 0);
      let it, ct, pt, Tt, wt, Mt, kt, Xt, he;
      const oe = x.isCompressedTexture ? x.mipmaps[K] : x.image;
      if (O !== null)
        it = O.max.x - O.min.x, ct = O.max.y - O.min.y, pt = O.isBox3 ? O.max.z - O.min.z : 1, Tt = O.min.x, wt = O.min.y, Mt = O.isBox3 ? O.min.z : 0;
      else {
        const Ge = Math.pow(2, -N);
        it = Math.floor(oe.width * Ge), ct = Math.floor(oe.height * Ge), x.isDataArrayTexture ? pt = oe.depth : x.isData3DTexture ? pt = Math.floor(oe.depth * Ge) : pt = 1, Tt = 0, wt = 0, Mt = 0;
      }
      B !== null ? (kt = B.x, Xt = B.y, he = B.z) : (kt = 0, Xt = 0, he = 0);
      const Gt = Dt.convert(U.format), St = Dt.convert(U.type);
      let ge;
      U.isData3DTexture ? (A.setTexture3D(U, 0), ge = w.TEXTURE_3D) : U.isDataArrayTexture || U.isCompressedArrayTexture ? (A.setTexture2DArray(U, 0), ge = w.TEXTURE_2D_ARRAY) : (A.setTexture2D(U, 0), ge = w.TEXTURE_2D), w.pixelStorei(w.UNPACK_FLIP_Y_WEBGL, U.flipY), w.pixelStorei(w.UNPACK_PREMULTIPLY_ALPHA_WEBGL, U.premultiplyAlpha), w.pixelStorei(w.UNPACK_ALIGNMENT, U.unpackAlignment);
      const Yt = w.getParameter(w.UNPACK_ROW_LENGTH), $e = w.getParameter(w.UNPACK_IMAGE_HEIGHT), ei = w.getParameter(w.UNPACK_SKIP_PIXELS), Ne = w.getParameter(w.UNPACK_SKIP_ROWS), Li = w.getParameter(w.UNPACK_SKIP_IMAGES);
      w.pixelStorei(w.UNPACK_ROW_LENGTH, oe.width), w.pixelStorei(w.UNPACK_IMAGE_HEIGHT, oe.height), w.pixelStorei(w.UNPACK_SKIP_PIXELS, Tt), w.pixelStorei(w.UNPACK_SKIP_ROWS, wt), w.pixelStorei(w.UNPACK_SKIP_IMAGES, Mt);
      const ie = x.isDataArrayTexture || x.isData3DTexture, Ve = U.isDataArrayTexture || U.isData3DTexture;
      if (x.isDepthTexture) {
        const Ge = xt.get(x), Ae = xt.get(U), Pe = xt.get(Ge.__renderTarget), Hs = xt.get(Ae.__renderTarget);
        vt.bindFramebuffer(w.READ_FRAMEBUFFER, Pe.__webglFramebuffer), vt.bindFramebuffer(w.DRAW_FRAMEBUFFER, Hs.__webglFramebuffer);
        for (let Fn = 0; Fn < pt; Fn++)
          ie && (w.framebufferTextureLayer(w.READ_FRAMEBUFFER, w.COLOR_ATTACHMENT0, xt.get(x).__webglTexture, N, Mt + Fn), w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER, w.COLOR_ATTACHMENT0, xt.get(U).__webglTexture, K, he + Fn)), w.blitFramebuffer(Tt, wt, it, ct, kt, Xt, it, ct, w.DEPTH_BUFFER_BIT, w.NEAREST);
        vt.bindFramebuffer(w.READ_FRAMEBUFFER, null), vt.bindFramebuffer(w.DRAW_FRAMEBUFFER, null);
      } else if (N !== 0 || x.isRenderTargetTexture || xt.has(x)) {
        const Ge = xt.get(x), Ae = xt.get(U);
        vt.bindFramebuffer(w.READ_FRAMEBUFFER, Gl), vt.bindFramebuffer(w.DRAW_FRAMEBUFFER, Wl);
        for (let Pe = 0; Pe < pt; Pe++)
          ie ? w.framebufferTextureLayer(w.READ_FRAMEBUFFER, w.COLOR_ATTACHMENT0, Ge.__webglTexture, N, Mt + Pe) : w.framebufferTexture2D(w.READ_FRAMEBUFFER, w.COLOR_ATTACHMENT0, w.TEXTURE_2D, Ge.__webglTexture, N), Ve ? w.framebufferTextureLayer(w.DRAW_FRAMEBUFFER, w.COLOR_ATTACHMENT0, Ae.__webglTexture, K, he + Pe) : w.framebufferTexture2D(w.DRAW_FRAMEBUFFER, w.COLOR_ATTACHMENT0, w.TEXTURE_2D, Ae.__webglTexture, K), N !== 0 ? w.blitFramebuffer(Tt, wt, it, ct, kt, Xt, it, ct, w.COLOR_BUFFER_BIT, w.NEAREST) : Ve ? w.copyTexSubImage3D(ge, K, kt, Xt, he + Pe, Tt, wt, it, ct) : w.copyTexSubImage2D(ge, K, kt, Xt, Tt, wt, it, ct);
        vt.bindFramebuffer(w.READ_FRAMEBUFFER, null), vt.bindFramebuffer(w.DRAW_FRAMEBUFFER, null);
      } else
        Ve ? x.isDataTexture || x.isData3DTexture ? w.texSubImage3D(ge, K, kt, Xt, he, it, ct, pt, Gt, St, oe.data) : U.isCompressedArrayTexture ? w.compressedTexSubImage3D(ge, K, kt, Xt, he, it, ct, pt, Gt, oe.data) : w.texSubImage3D(ge, K, kt, Xt, he, it, ct, pt, Gt, St, oe) : x.isDataTexture ? w.texSubImage2D(w.TEXTURE_2D, K, kt, Xt, it, ct, Gt, St, oe.data) : x.isCompressedTexture ? w.compressedTexSubImage2D(w.TEXTURE_2D, K, kt, Xt, oe.width, oe.height, Gt, oe.data) : w.texSubImage2D(w.TEXTURE_2D, K, kt, Xt, it, ct, Gt, St, oe);
      w.pixelStorei(w.UNPACK_ROW_LENGTH, Yt), w.pixelStorei(w.UNPACK_IMAGE_HEIGHT, $e), w.pixelStorei(w.UNPACK_SKIP_PIXELS, ei), w.pixelStorei(w.UNPACK_SKIP_ROWS, Ne), w.pixelStorei(w.UNPACK_SKIP_IMAGES, Li), K === 0 && U.generateMipmaps && w.generateMipmap(ge), vt.unbindTexture();
    }, this.copyTextureToTexture3D = function(x, U, O = null, B = null, N = 0) {
      return x.isTexture !== !0 && (_i("WebGLRenderer: copyTextureToTexture3D function signature has changed."), O = arguments[0] || null, B = arguments[1] || null, x = arguments[2], U = arguments[3], N = arguments[4] || 0), _i('WebGLRenderer: copyTextureToTexture3D function has been deprecated. Use "copyTextureToTexture" instead.'), this.copyTextureToTexture(x, U, O, B, N);
    }, this.initRenderTarget = function(x) {
      xt.get(x).__webglFramebuffer === void 0 && A.setupRenderTarget(x);
    }, this.initTexture = function(x) {
      x.isCubeTexture ? A.setTextureCube(x, 0) : x.isData3DTexture ? A.setTexture3D(x, 0) : x.isDataArrayTexture || x.isCompressedArrayTexture ? A.setTexture2DArray(x, 0) : A.setTexture2D(x, 0), vt.unbindTexture();
    }, this.resetState = function() {
      T = 0, R = 0, I = null, vt.reset(), Jt.reset();
    }, typeof __THREE_DEVTOOLS__ < "u" && __THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe", { detail: this }));
  }
  get coordinateSystem() {
    return vn;
  }
  get outputColorSpace() {
    return this._outputColorSpace;
  }
  set outputColorSpace(t) {
    this._outputColorSpace = t;
    const e = this.getContext();
    e.drawingBufferColorspace = Wt._getDrawingBufferColorSpace(t), e.unpackColorSpace = Wt._getUnpackColorSpace();
  }
}
const Go = { type: "change" }, Ra = { type: "start" }, Il = { type: "end" }, Ms = new Sa(), Wo = new _n(), Mm = Math.cos(70 * ml.DEG2RAD), de = new P(), De = 2 * Math.PI, Zt = {
  NONE: -1,
  ROTATE: 0,
  DOLLY: 1,
  PAN: 2,
  TOUCH_ROTATE: 3,
  TOUCH_PAN: 4,
  TOUCH_DOLLY_PAN: 5,
  TOUCH_DOLLY_ROTATE: 6
}, gr = 1e-6;
class vr extends Nh {
  constructor(t, e = null) {
    super(t, e), this.state = Zt.NONE, this.enabled = !0, this.target = new P(), this.cursor = new P(), this.minDistance = 0, this.maxDistance = 1 / 0, this.minZoom = 0, this.maxZoom = 1 / 0, this.minTargetRadius = 0, this.maxTargetRadius = 1 / 0, this.minPolarAngle = 0, this.maxPolarAngle = Math.PI, this.minAzimuthAngle = -1 / 0, this.maxAzimuthAngle = 1 / 0, this.enableDamping = !1, this.dampingFactor = 0.05, this.enableZoom = !0, this.zoomSpeed = 1, this.enableRotate = !0, this.rotateSpeed = 1, this.enablePan = !0, this.panSpeed = 1, this.screenSpacePanning = !0, this.keyPanSpeed = 7, this.zoomToCursor = !1, this.autoRotate = !1, this.autoRotateSpeed = 2, this.keys = { LEFT: "ArrowLeft", UP: "ArrowUp", RIGHT: "ArrowRight", BOTTOM: "ArrowDown" }, this.mouseButtons = { LEFT: xi.ROTATE, MIDDLE: xi.DOLLY, RIGHT: xi.PAN }, this.touches = { ONE: gi.ROTATE, TWO: gi.DOLLY_PAN }, this.target0 = this.target.clone(), this.position0 = this.object.position.clone(), this.zoom0 = this.object.zoom, this._domElementKeyEvents = null, this._lastPosition = new P(), this._lastQuaternion = new Kn(), this._lastTargetPosition = new P(), this._quat = new Kn().setFromUnitVectors(t.up, new P(0, 1, 0)), this._quatInverse = this._quat.clone().invert(), this._spherical = new mo(), this._sphericalDelta = new mo(), this._scale = 1, this._panOffset = new P(), this._rotateStart = new bt(), this._rotateEnd = new bt(), this._rotateDelta = new bt(), this._panStart = new bt(), this._panEnd = new bt(), this._panDelta = new bt(), this._dollyStart = new bt(), this._dollyEnd = new bt(), this._dollyDelta = new bt(), this._dollyDirection = new P(), this._mouse = new bt(), this._performCursorZoom = !1, this._pointers = [], this._pointerPositions = {}, this._controlActive = !1, this._onPointerMove = ym.bind(this), this._onPointerDown = Sm.bind(this), this._onPointerUp = Em.bind(this), this._onContextMenu = Pm.bind(this), this._onMouseWheel = Tm.bind(this), this._onKeyDown = wm.bind(this), this._onTouchStart = Rm.bind(this), this._onTouchMove = Cm.bind(this), this._onMouseDown = bm.bind(this), this._onMouseMove = Am.bind(this), this._interceptControlDown = Dm.bind(this), this._interceptControlUp = Lm.bind(this), this.domElement !== null && this.connect(), this.update();
  }
  connect() {
    this.domElement.addEventListener("pointerdown", this._onPointerDown), this.domElement.addEventListener("pointercancel", this._onPointerUp), this.domElement.addEventListener("contextmenu", this._onContextMenu), this.domElement.addEventListener("wheel", this._onMouseWheel, { passive: !1 }), this.domElement.getRootNode().addEventListener("keydown", this._interceptControlDown, { passive: !0, capture: !0 }), this.domElement.style.touchAction = "none";
  }
  disconnect() {
    this.domElement.removeEventListener("pointerdown", this._onPointerDown), this.domElement.removeEventListener("pointermove", this._onPointerMove), this.domElement.removeEventListener("pointerup", this._onPointerUp), this.domElement.removeEventListener("pointercancel", this._onPointerUp), this.domElement.removeEventListener("wheel", this._onMouseWheel), this.domElement.removeEventListener("contextmenu", this._onContextMenu), this.stopListenToKeyEvents(), this.domElement.getRootNode().removeEventListener("keydown", this._interceptControlDown, { capture: !0 }), this.domElement.style.touchAction = "auto";
  }
  dispose() {
    this.disconnect();
  }
  getPolarAngle() {
    return this._spherical.phi;
  }
  getAzimuthalAngle() {
    return this._spherical.theta;
  }
  getDistance() {
    return this.object.position.distanceTo(this.target);
  }
  listenToKeyEvents(t) {
    t.addEventListener("keydown", this._onKeyDown), this._domElementKeyEvents = t;
  }
  stopListenToKeyEvents() {
    this._domElementKeyEvents !== null && (this._domElementKeyEvents.removeEventListener("keydown", this._onKeyDown), this._domElementKeyEvents = null);
  }
  saveState() {
    this.target0.copy(this.target), this.position0.copy(this.object.position), this.zoom0 = this.object.zoom;
  }
  reset() {
    this.target.copy(this.target0), this.object.position.copy(this.position0), this.object.zoom = this.zoom0, this.object.updateProjectionMatrix(), this.dispatchEvent(Go), this.update(), this.state = Zt.NONE;
  }
  update(t = null) {
    const e = this.object.position;
    de.copy(e).sub(this.target), de.applyQuaternion(this._quat), this._spherical.setFromVector3(de), this.autoRotate && this.state === Zt.NONE && this._rotateLeft(this._getAutoRotationAngle(t)), this.enableDamping ? (this._spherical.theta += this._sphericalDelta.theta * this.dampingFactor, this._spherical.phi += this._sphericalDelta.phi * this.dampingFactor) : (this._spherical.theta += this._sphericalDelta.theta, this._spherical.phi += this._sphericalDelta.phi);
    let n = this.minAzimuthAngle, s = this.maxAzimuthAngle;
    isFinite(n) && isFinite(s) && (n < -Math.PI ? n += De : n > Math.PI && (n -= De), s < -Math.PI ? s += De : s > Math.PI && (s -= De), n <= s ? this._spherical.theta = Math.max(n, Math.min(s, this._spherical.theta)) : this._spherical.theta = this._spherical.theta > (n + s) / 2 ? Math.max(n, this._spherical.theta) : Math.min(s, this._spherical.theta)), this._spherical.phi = Math.max(this.minPolarAngle, Math.min(this.maxPolarAngle, this._spherical.phi)), this._spherical.makeSafe(), this.enableDamping === !0 ? this.target.addScaledVector(this._panOffset, this.dampingFactor) : this.target.add(this._panOffset), this.target.sub(this.cursor), this.target.clampLength(this.minTargetRadius, this.maxTargetRadius), this.target.add(this.cursor);
    let r = !1;
    if (this.zoomToCursor && this._performCursorZoom || this.object.isOrthographicCamera)
      this._spherical.radius = this._clampDistance(this._spherical.radius);
    else {
      const a = this._spherical.radius;
      this._spherical.radius = this._clampDistance(this._spherical.radius * this._scale), r = a != this._spherical.radius;
    }
    if (de.setFromSpherical(this._spherical), de.applyQuaternion(this._quatInverse), e.copy(this.target).add(de), this.object.lookAt(this.target), this.enableDamping === !0 ? (this._sphericalDelta.theta *= 1 - this.dampingFactor, this._sphericalDelta.phi *= 1 - this.dampingFactor, this._panOffset.multiplyScalar(1 - this.dampingFactor)) : (this._sphericalDelta.set(0, 0, 0), this._panOffset.set(0, 0, 0)), this.zoomToCursor && this._performCursorZoom) {
      let a = null;
      if (this.object.isPerspectiveCamera) {
        const o = de.length();
        a = this._clampDistance(o * this._scale);
        const l = o - a;
        this.object.position.addScaledVector(this._dollyDirection, l), this.object.updateMatrixWorld(), r = !!l;
      } else if (this.object.isOrthographicCamera) {
        const o = new P(this._mouse.x, this._mouse.y, 0);
        o.unproject(this.object);
        const l = this.object.zoom;
        this.object.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.object.zoom / this._scale)), this.object.updateProjectionMatrix(), r = l !== this.object.zoom;
        const c = new P(this._mouse.x, this._mouse.y, 0);
        c.unproject(this.object), this.object.position.sub(c).add(o), this.object.updateMatrixWorld(), a = de.length();
      } else
        console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."), this.zoomToCursor = !1;
      a !== null && (this.screenSpacePanning ? this.target.set(0, 0, -1).transformDirection(this.object.matrix).multiplyScalar(a).add(this.object.position) : (Ms.origin.copy(this.object.position), Ms.direction.set(0, 0, -1).transformDirection(this.object.matrix), Math.abs(this.object.up.dot(Ms.direction)) < Mm ? this.object.lookAt(this.target) : (Wo.setFromNormalAndCoplanarPoint(this.object.up, this.target), Ms.intersectPlane(Wo, this.target))));
    } else if (this.object.isOrthographicCamera) {
      const a = this.object.zoom;
      this.object.zoom = Math.max(this.minZoom, Math.min(this.maxZoom, this.object.zoom / this._scale)), a !== this.object.zoom && (this.object.updateProjectionMatrix(), r = !0);
    }
    return this._scale = 1, this._performCursorZoom = !1, r || this._lastPosition.distanceToSquared(this.object.position) > gr || 8 * (1 - this._lastQuaternion.dot(this.object.quaternion)) > gr || this._lastTargetPosition.distanceToSquared(this.target) > gr ? (this.dispatchEvent(Go), this._lastPosition.copy(this.object.position), this._lastQuaternion.copy(this.object.quaternion), this._lastTargetPosition.copy(this.target), !0) : !1;
  }
  _getAutoRotationAngle(t) {
    return t !== null ? De / 60 * this.autoRotateSpeed * t : De / 60 / 60 * this.autoRotateSpeed;
  }
  _getZoomScale(t) {
    const e = Math.abs(t * 0.01);
    return Math.pow(0.95, this.zoomSpeed * e);
  }
  _rotateLeft(t) {
    this._sphericalDelta.theta -= t;
  }
  _rotateUp(t) {
    this._sphericalDelta.phi -= t;
  }
  _panLeft(t, e) {
    de.setFromMatrixColumn(e, 0), de.multiplyScalar(-t), this._panOffset.add(de);
  }
  _panUp(t, e) {
    this.screenSpacePanning === !0 ? de.setFromMatrixColumn(e, 1) : (de.setFromMatrixColumn(e, 0), de.crossVectors(this.object.up, de)), de.multiplyScalar(t), this._panOffset.add(de);
  }
  // deltaX and deltaY are in pixels; right and down are positive
  _pan(t, e) {
    const n = this.domElement;
    if (this.object.isPerspectiveCamera) {
      const s = this.object.position;
      de.copy(s).sub(this.target);
      let r = de.length();
      r *= Math.tan(this.object.fov / 2 * Math.PI / 180), this._panLeft(2 * t * r / n.clientHeight, this.object.matrix), this._panUp(2 * e * r / n.clientHeight, this.object.matrix);
    } else this.object.isOrthographicCamera ? (this._panLeft(t * (this.object.right - this.object.left) / this.object.zoom / n.clientWidth, this.object.matrix), this._panUp(e * (this.object.top - this.object.bottom) / this.object.zoom / n.clientHeight, this.object.matrix)) : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."), this.enablePan = !1);
  }
  _dollyOut(t) {
    this.object.isPerspectiveCamera || this.object.isOrthographicCamera ? this._scale /= t : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."), this.enableZoom = !1);
  }
  _dollyIn(t) {
    this.object.isPerspectiveCamera || this.object.isOrthographicCamera ? this._scale *= t : (console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."), this.enableZoom = !1);
  }
  _updateZoomParameters(t, e) {
    if (!this.zoomToCursor)
      return;
    this._performCursorZoom = !0;
    const n = this.domElement.getBoundingClientRect(), s = t - n.left, r = e - n.top, a = n.width, o = n.height;
    this._mouse.x = s / a * 2 - 1, this._mouse.y = -(r / o) * 2 + 1, this._dollyDirection.set(this._mouse.x, this._mouse.y, 1).unproject(this.object).sub(this.object.position).normalize();
  }
  _clampDistance(t) {
    return Math.max(this.minDistance, Math.min(this.maxDistance, t));
  }
  //
  // event callbacks - update the object state
  //
  _handleMouseDownRotate(t) {
    this._rotateStart.set(t.clientX, t.clientY);
  }
  _handleMouseDownDolly(t) {
    this._updateZoomParameters(t.clientX, t.clientX), this._dollyStart.set(t.clientX, t.clientY);
  }
  _handleMouseDownPan(t) {
    this._panStart.set(t.clientX, t.clientY);
  }
  _handleMouseMoveRotate(t) {
    this._rotateEnd.set(t.clientX, t.clientY), this._rotateDelta.subVectors(this._rotateEnd, this._rotateStart).multiplyScalar(this.rotateSpeed);
    const e = this.domElement;
    this._rotateLeft(De * this._rotateDelta.x / e.clientHeight), this._rotateUp(De * this._rotateDelta.y / e.clientHeight), this._rotateStart.copy(this._rotateEnd), this.update();
  }
  _handleMouseMoveDolly(t) {
    this._dollyEnd.set(t.clientX, t.clientY), this._dollyDelta.subVectors(this._dollyEnd, this._dollyStart), this._dollyDelta.y > 0 ? this._dollyOut(this._getZoomScale(this._dollyDelta.y)) : this._dollyDelta.y < 0 && this._dollyIn(this._getZoomScale(this._dollyDelta.y)), this._dollyStart.copy(this._dollyEnd), this.update();
  }
  _handleMouseMovePan(t) {
    this._panEnd.set(t.clientX, t.clientY), this._panDelta.subVectors(this._panEnd, this._panStart).multiplyScalar(this.panSpeed), this._pan(this._panDelta.x, this._panDelta.y), this._panStart.copy(this._panEnd), this.update();
  }
  _handleMouseWheel(t) {
    this._updateZoomParameters(t.clientX, t.clientY), t.deltaY < 0 ? this._dollyIn(this._getZoomScale(t.deltaY)) : t.deltaY > 0 && this._dollyOut(this._getZoomScale(t.deltaY)), this.update();
  }
  _handleKeyDown(t) {
    let e = !1;
    switch (t.code) {
      case this.keys.UP:
        t.ctrlKey || t.metaKey || t.shiftKey ? this.enableRotate && this._rotateUp(De * this.rotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(0, this.keyPanSpeed), e = !0;
        break;
      case this.keys.BOTTOM:
        t.ctrlKey || t.metaKey || t.shiftKey ? this.enableRotate && this._rotateUp(-De * this.rotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(0, -this.keyPanSpeed), e = !0;
        break;
      case this.keys.LEFT:
        t.ctrlKey || t.metaKey || t.shiftKey ? this.enableRotate && this._rotateLeft(De * this.rotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(this.keyPanSpeed, 0), e = !0;
        break;
      case this.keys.RIGHT:
        t.ctrlKey || t.metaKey || t.shiftKey ? this.enableRotate && this._rotateLeft(-De * this.rotateSpeed / this.domElement.clientHeight) : this.enablePan && this._pan(-this.keyPanSpeed, 0), e = !0;
        break;
    }
    e && (t.preventDefault(), this.update());
  }
  _handleTouchStartRotate(t) {
    if (this._pointers.length === 1)
      this._rotateStart.set(t.pageX, t.pageY);
    else {
      const e = this._getSecondPointerPosition(t), n = 0.5 * (t.pageX + e.x), s = 0.5 * (t.pageY + e.y);
      this._rotateStart.set(n, s);
    }
  }
  _handleTouchStartPan(t) {
    if (this._pointers.length === 1)
      this._panStart.set(t.pageX, t.pageY);
    else {
      const e = this._getSecondPointerPosition(t), n = 0.5 * (t.pageX + e.x), s = 0.5 * (t.pageY + e.y);
      this._panStart.set(n, s);
    }
  }
  _handleTouchStartDolly(t) {
    const e = this._getSecondPointerPosition(t), n = t.pageX - e.x, s = t.pageY - e.y, r = Math.sqrt(n * n + s * s);
    this._dollyStart.set(0, r);
  }
  _handleTouchStartDollyPan(t) {
    this.enableZoom && this._handleTouchStartDolly(t), this.enablePan && this._handleTouchStartPan(t);
  }
  _handleTouchStartDollyRotate(t) {
    this.enableZoom && this._handleTouchStartDolly(t), this.enableRotate && this._handleTouchStartRotate(t);
  }
  _handleTouchMoveRotate(t) {
    if (this._pointers.length == 1)
      this._rotateEnd.set(t.pageX, t.pageY);
    else {
      const n = this._getSecondPointerPosition(t), s = 0.5 * (t.pageX + n.x), r = 0.5 * (t.pageY + n.y);
      this._rotateEnd.set(s, r);
    }
    this._rotateDelta.subVectors(this._rotateEnd, this._rotateStart).multiplyScalar(this.rotateSpeed);
    const e = this.domElement;
    this._rotateLeft(De * this._rotateDelta.x / e.clientHeight), this._rotateUp(De * this._rotateDelta.y / e.clientHeight), this._rotateStart.copy(this._rotateEnd);
  }
  _handleTouchMovePan(t) {
    if (this._pointers.length === 1)
      this._panEnd.set(t.pageX, t.pageY);
    else {
      const e = this._getSecondPointerPosition(t), n = 0.5 * (t.pageX + e.x), s = 0.5 * (t.pageY + e.y);
      this._panEnd.set(n, s);
    }
    this._panDelta.subVectors(this._panEnd, this._panStart).multiplyScalar(this.panSpeed), this._pan(this._panDelta.x, this._panDelta.y), this._panStart.copy(this._panEnd);
  }
  _handleTouchMoveDolly(t) {
    const e = this._getSecondPointerPosition(t), n = t.pageX - e.x, s = t.pageY - e.y, r = Math.sqrt(n * n + s * s);
    this._dollyEnd.set(0, r), this._dollyDelta.set(0, Math.pow(this._dollyEnd.y / this._dollyStart.y, this.zoomSpeed)), this._dollyOut(this._dollyDelta.y), this._dollyStart.copy(this._dollyEnd);
    const a = (t.pageX + e.x) * 0.5, o = (t.pageY + e.y) * 0.5;
    this._updateZoomParameters(a, o);
  }
  _handleTouchMoveDollyPan(t) {
    this.enableZoom && this._handleTouchMoveDolly(t), this.enablePan && this._handleTouchMovePan(t);
  }
  _handleTouchMoveDollyRotate(t) {
    this.enableZoom && this._handleTouchMoveDolly(t), this.enableRotate && this._handleTouchMoveRotate(t);
  }
  // pointers
  _addPointer(t) {
    this._pointers.push(t.pointerId);
  }
  _removePointer(t) {
    delete this._pointerPositions[t.pointerId];
    for (let e = 0; e < this._pointers.length; e++)
      if (this._pointers[e] == t.pointerId) {
        this._pointers.splice(e, 1);
        return;
      }
  }
  _isTrackingPointer(t) {
    for (let e = 0; e < this._pointers.length; e++)
      if (this._pointers[e] == t.pointerId) return !0;
    return !1;
  }
  _trackPointer(t) {
    let e = this._pointerPositions[t.pointerId];
    e === void 0 && (e = new bt(), this._pointerPositions[t.pointerId] = e), e.set(t.pageX, t.pageY);
  }
  _getSecondPointerPosition(t) {
    const e = t.pointerId === this._pointers[0] ? this._pointers[1] : this._pointers[0];
    return this._pointerPositions[e];
  }
  //
  _customWheelEvent(t) {
    const e = t.deltaMode, n = {
      clientX: t.clientX,
      clientY: t.clientY,
      deltaY: t.deltaY
    };
    switch (e) {
      case 1:
        n.deltaY *= 16;
        break;
      case 2:
        n.deltaY *= 100;
        break;
    }
    return t.ctrlKey && !this._controlActive && (n.deltaY *= 10), n;
  }
}
function Sm(i) {
  this.enabled !== !1 && (this._pointers.length === 0 && (this.domElement.setPointerCapture(i.pointerId), this.domElement.addEventListener("pointermove", this._onPointerMove), this.domElement.addEventListener("pointerup", this._onPointerUp)), !this._isTrackingPointer(i) && (this._addPointer(i), i.pointerType === "touch" ? this._onTouchStart(i) : this._onMouseDown(i)));
}
function ym(i) {
  this.enabled !== !1 && (i.pointerType === "touch" ? this._onTouchMove(i) : this._onMouseMove(i));
}
function Em(i) {
  switch (this._removePointer(i), this._pointers.length) {
    case 0:
      this.domElement.releasePointerCapture(i.pointerId), this.domElement.removeEventListener("pointermove", this._onPointerMove), this.domElement.removeEventListener("pointerup", this._onPointerUp), this.dispatchEvent(Il), this.state = Zt.NONE;
      break;
    case 1:
      const t = this._pointers[0], e = this._pointerPositions[t];
      this._onTouchStart({ pointerId: t, pageX: e.x, pageY: e.y });
      break;
  }
}
function bm(i) {
  let t;
  switch (i.button) {
    case 0:
      t = this.mouseButtons.LEFT;
      break;
    case 1:
      t = this.mouseButtons.MIDDLE;
      break;
    case 2:
      t = this.mouseButtons.RIGHT;
      break;
    default:
      t = -1;
  }
  switch (t) {
    case xi.DOLLY:
      if (this.enableZoom === !1) return;
      this._handleMouseDownDolly(i), this.state = Zt.DOLLY;
      break;
    case xi.ROTATE:
      if (i.ctrlKey || i.metaKey || i.shiftKey) {
        if (this.enablePan === !1) return;
        this._handleMouseDownPan(i), this.state = Zt.PAN;
      } else {
        if (this.enableRotate === !1) return;
        this._handleMouseDownRotate(i), this.state = Zt.ROTATE;
      }
      break;
    case xi.PAN:
      if (i.ctrlKey || i.metaKey || i.shiftKey) {
        if (this.enableRotate === !1) return;
        this._handleMouseDownRotate(i), this.state = Zt.ROTATE;
      } else {
        if (this.enablePan === !1) return;
        this._handleMouseDownPan(i), this.state = Zt.PAN;
      }
      break;
    default:
      this.state = Zt.NONE;
  }
  this.state !== Zt.NONE && this.dispatchEvent(Ra);
}
function Am(i) {
  switch (this.state) {
    case Zt.ROTATE:
      if (this.enableRotate === !1) return;
      this._handleMouseMoveRotate(i);
      break;
    case Zt.DOLLY:
      if (this.enableZoom === !1) return;
      this._handleMouseMoveDolly(i);
      break;
    case Zt.PAN:
      if (this.enablePan === !1) return;
      this._handleMouseMovePan(i);
      break;
  }
}
function Tm(i) {
  this.enabled === !1 || this.enableZoom === !1 || this.state !== Zt.NONE || (i.preventDefault(), this.dispatchEvent(Ra), this._handleMouseWheel(this._customWheelEvent(i)), this.dispatchEvent(Il));
}
function wm(i) {
  this.enabled !== !1 && this._handleKeyDown(i);
}
function Rm(i) {
  switch (this._trackPointer(i), this._pointers.length) {
    case 1:
      switch (this.touches.ONE) {
        case gi.ROTATE:
          if (this.enableRotate === !1) return;
          this._handleTouchStartRotate(i), this.state = Zt.TOUCH_ROTATE;
          break;
        case gi.PAN:
          if (this.enablePan === !1) return;
          this._handleTouchStartPan(i), this.state = Zt.TOUCH_PAN;
          break;
        default:
          this.state = Zt.NONE;
      }
      break;
    case 2:
      switch (this.touches.TWO) {
        case gi.DOLLY_PAN:
          if (this.enableZoom === !1 && this.enablePan === !1) return;
          this._handleTouchStartDollyPan(i), this.state = Zt.TOUCH_DOLLY_PAN;
          break;
        case gi.DOLLY_ROTATE:
          if (this.enableZoom === !1 && this.enableRotate === !1) return;
          this._handleTouchStartDollyRotate(i), this.state = Zt.TOUCH_DOLLY_ROTATE;
          break;
        default:
          this.state = Zt.NONE;
      }
      break;
    default:
      this.state = Zt.NONE;
  }
  this.state !== Zt.NONE && this.dispatchEvent(Ra);
}
function Cm(i) {
  switch (this._trackPointer(i), this.state) {
    case Zt.TOUCH_ROTATE:
      if (this.enableRotate === !1) return;
      this._handleTouchMoveRotate(i), this.update();
      break;
    case Zt.TOUCH_PAN:
      if (this.enablePan === !1) return;
      this._handleTouchMovePan(i), this.update();
      break;
    case Zt.TOUCH_DOLLY_PAN:
      if (this.enableZoom === !1 && this.enablePan === !1) return;
      this._handleTouchMoveDollyPan(i), this.update();
      break;
    case Zt.TOUCH_DOLLY_ROTATE:
      if (this.enableZoom === !1 && this.enableRotate === !1) return;
      this._handleTouchMoveDollyRotate(i), this.update();
      break;
    default:
      this.state = Zt.NONE;
  }
}
function Pm(i) {
  this.enabled !== !1 && i.preventDefault();
}
function Dm(i) {
  i.key === "Control" && (this._controlActive = !0, this.domElement.getRootNode().addEventListener("keyup", this._interceptControlUp, { passive: !0, capture: !0 }));
}
function Lm(i) {
  i.key === "Control" && (this._controlActive = !1, this.domElement.getRootNode().removeEventListener("keyup", this._interceptControlUp, { passive: !0, capture: !0 }));
}
const Xo = new He(), Ss = new P();
class Ca extends Ph {
  constructor() {
    super(), this.isLineSegmentsGeometry = !0, this.type = "LineSegmentsGeometry";
    const t = [-1, 2, 0, 1, 2, 0, -1, 1, 0, 1, 1, 0, -1, 0, 0, 1, 0, 0, -1, -1, 0, 1, -1, 0], e = [-1, 2, 1, 2, -1, 1, 1, 1, -1, -1, 1, -1, -1, -2, 1, -2], n = [0, 2, 1, 2, 3, 1, 2, 4, 3, 4, 5, 3, 4, 6, 5, 6, 7, 5];
    this.setIndex(n), this.setAttribute("position", new se(t, 3)), this.setAttribute("uv", new se(e, 2));
  }
  applyMatrix4(t) {
    const e = this.attributes.instanceStart, n = this.attributes.instanceEnd;
    return e !== void 0 && (e.applyMatrix4(t), n.applyMatrix4(t), e.needsUpdate = !0), this.boundingBox !== null && this.computeBoundingBox(), this.boundingSphere !== null && this.computeBoundingSphere(), this;
  }
  setPositions(t) {
    let e;
    t instanceof Float32Array ? e = t : Array.isArray(t) && (e = new Float32Array(t));
    const n = new ha(e, 6, 1);
    return this.setAttribute("instanceStart", new Dn(n, 3, 0)), this.setAttribute("instanceEnd", new Dn(n, 3, 3)), this.instanceCount = this.attributes.instanceStart.count, this.computeBoundingBox(), this.computeBoundingSphere(), this;
  }
  setColors(t) {
    let e;
    t instanceof Float32Array ? e = t : Array.isArray(t) && (e = new Float32Array(t));
    const n = new ha(e, 6, 1);
    return this.setAttribute("instanceColorStart", new Dn(n, 3, 0)), this.setAttribute("instanceColorEnd", new Dn(n, 3, 3)), this;
  }
  fromWireframeGeometry(t) {
    return this.setPositions(t.attributes.position.array), this;
  }
  fromEdgesGeometry(t) {
    return this.setPositions(t.attributes.position.array), this;
  }
  fromMesh(t) {
    return this.fromWireframeGeometry(new Th(t.geometry)), this;
  }
  fromLineSegments(t) {
    const e = t.geometry;
    return this.setPositions(e.attributes.position.array), this;
  }
  computeBoundingBox() {
    this.boundingBox === null && (this.boundingBox = new He());
    const t = this.attributes.instanceStart, e = this.attributes.instanceEnd;
    t !== void 0 && e !== void 0 && (this.boundingBox.setFromBufferAttribute(t), Xo.setFromBufferAttribute(e), this.boundingBox.union(Xo));
  }
  computeBoundingSphere() {
    this.boundingSphere === null && (this.boundingSphere = new Pi()), this.boundingBox === null && this.computeBoundingBox();
    const t = this.attributes.instanceStart, e = this.attributes.instanceEnd;
    if (t !== void 0 && e !== void 0) {
      const n = this.boundingSphere.center;
      this.boundingBox.getCenter(n);
      let s = 0;
      for (let r = 0, a = t.count; r < a; r++)
        Ss.fromBufferAttribute(t, r), s = Math.max(s, n.distanceToSquared(Ss)), Ss.fromBufferAttribute(e, r), s = Math.max(s, n.distanceToSquared(Ss));
      this.boundingSphere.radius = Math.sqrt(s), isNaN(this.boundingSphere.radius) && console.error("THREE.LineSegmentsGeometry.computeBoundingSphere(): Computed radius is NaN. The instanced position data is likely to have NaN values.", this);
    }
  }
  toJSON() {
  }
  applyMatrix(t) {
    return console.warn("THREE.LineSegmentsGeometry: applyMatrix() has been renamed to applyMatrix4()."), this.applyMatrix4(t);
  }
}
et.line = {
  worldUnits: { value: 1 },
  linewidth: { value: 1 },
  resolution: { value: new bt(1, 1) },
  dashOffset: { value: 0 },
  dashScale: { value: 1 },
  dashSize: { value: 1 },
  gapSize: { value: 1 }
  // todo FIX - maybe change to totalSize
};
Le.line = {
  uniforms: ya.merge([
    et.common,
    et.fog,
    et.line
  ]),
  vertexShader: (
    /* glsl */
    `
		#include <common>
		#include <color_pars_vertex>
		#include <fog_pars_vertex>
		#include <logdepthbuf_pars_vertex>
		#include <clipping_planes_pars_vertex>

		uniform float linewidth;
		uniform vec2 resolution;

		attribute vec3 instanceStart;
		attribute vec3 instanceEnd;

		attribute vec3 instanceColorStart;
		attribute vec3 instanceColorEnd;

		#ifdef WORLD_UNITS

			varying vec4 worldPos;
			varying vec3 worldStart;
			varying vec3 worldEnd;

			#ifdef USE_DASH

				varying vec2 vUv;

			#endif

		#else

			varying vec2 vUv;

		#endif

		#ifdef USE_DASH

			uniform float dashScale;
			attribute float instanceDistanceStart;
			attribute float instanceDistanceEnd;
			varying float vLineDistance;

		#endif

		void trimSegment( const in vec4 start, inout vec4 end ) {

			// trim end segment so it terminates between the camera plane and the near plane

			// conservative estimate of the near plane
			float a = projectionMatrix[ 2 ][ 2 ]; // 3nd entry in 3th column
			float b = projectionMatrix[ 3 ][ 2 ]; // 3nd entry in 4th column
			float nearEstimate = - 0.5 * b / a;

			float alpha = ( nearEstimate - start.z ) / ( end.z - start.z );

			end.xyz = mix( start.xyz, end.xyz, alpha );

		}

		void main() {

			#ifdef USE_COLOR

				vColor.xyz = ( position.y < 0.5 ) ? instanceColorStart : instanceColorEnd;

			#endif

			#ifdef USE_DASH

				vLineDistance = ( position.y < 0.5 ) ? dashScale * instanceDistanceStart : dashScale * instanceDistanceEnd;
				vUv = uv;

			#endif

			float aspect = resolution.x / resolution.y;

			// camera space
			vec4 start = modelViewMatrix * vec4( instanceStart, 1.0 );
			vec4 end = modelViewMatrix * vec4( instanceEnd, 1.0 );

			#ifdef WORLD_UNITS

				worldStart = start.xyz;
				worldEnd = end.xyz;

			#else

				vUv = uv;

			#endif

			// special case for perspective projection, and segments that terminate either in, or behind, the camera plane
			// clearly the gpu firmware has a way of addressing this issue when projecting into ndc space
			// but we need to perform ndc-space calculations in the shader, so we must address this issue directly
			// perhaps there is a more elegant solution -- WestLangley

			bool perspective = ( projectionMatrix[ 2 ][ 3 ] == - 1.0 ); // 4th entry in the 3rd column

			if ( perspective ) {

				if ( start.z < 0.0 && end.z >= 0.0 ) {

					trimSegment( start, end );

				} else if ( end.z < 0.0 && start.z >= 0.0 ) {

					trimSegment( end, start );

				}

			}

			// clip space
			vec4 clipStart = projectionMatrix * start;
			vec4 clipEnd = projectionMatrix * end;

			// ndc space
			vec3 ndcStart = clipStart.xyz / clipStart.w;
			vec3 ndcEnd = clipEnd.xyz / clipEnd.w;

			// direction
			vec2 dir = ndcEnd.xy - ndcStart.xy;

			// account for clip-space aspect ratio
			dir.x *= aspect;
			dir = normalize( dir );

			#ifdef WORLD_UNITS

				vec3 worldDir = normalize( end.xyz - start.xyz );
				vec3 tmpFwd = normalize( mix( start.xyz, end.xyz, 0.5 ) );
				vec3 worldUp = normalize( cross( worldDir, tmpFwd ) );
				vec3 worldFwd = cross( worldDir, worldUp );
				worldPos = position.y < 0.5 ? start: end;

				// height offset
				float hw = linewidth * 0.5;
				worldPos.xyz += position.x < 0.0 ? hw * worldUp : - hw * worldUp;

				// don't extend the line if we're rendering dashes because we
				// won't be rendering the endcaps
				#ifndef USE_DASH

					// cap extension
					worldPos.xyz += position.y < 0.5 ? - hw * worldDir : hw * worldDir;

					// add width to the box
					worldPos.xyz += worldFwd * hw;

					// endcaps
					if ( position.y > 1.0 || position.y < 0.0 ) {

						worldPos.xyz -= worldFwd * 2.0 * hw;

					}

				#endif

				// project the worldpos
				vec4 clip = projectionMatrix * worldPos;

				// shift the depth of the projected points so the line
				// segments overlap neatly
				vec3 clipPose = ( position.y < 0.5 ) ? ndcStart : ndcEnd;
				clip.z = clipPose.z * clip.w;

			#else

				vec2 offset = vec2( dir.y, - dir.x );
				// undo aspect ratio adjustment
				dir.x /= aspect;
				offset.x /= aspect;

				// sign flip
				if ( position.x < 0.0 ) offset *= - 1.0;

				// endcaps
				if ( position.y < 0.0 ) {

					offset += - dir;

				} else if ( position.y > 1.0 ) {

					offset += dir;

				}

				// adjust for linewidth
				offset *= linewidth;

				// adjust for clip-space to screen-space conversion // maybe resolution should be based on viewport ...
				offset /= resolution.y;

				// select end
				vec4 clip = ( position.y < 0.5 ) ? clipStart : clipEnd;

				// back to clip space
				offset *= clip.w;

				clip.xy += offset;

			#endif

			gl_Position = clip;

			vec4 mvPosition = ( position.y < 0.5 ) ? start : end; // this is an approximation

			#include <logdepthbuf_vertex>
			#include <clipping_planes_vertex>
			#include <fog_vertex>

		}
		`
  ),
  fragmentShader: (
    /* glsl */
    `
		uniform vec3 diffuse;
		uniform float opacity;
		uniform float linewidth;

		#ifdef USE_DASH

			uniform float dashOffset;
			uniform float dashSize;
			uniform float gapSize;

		#endif

		varying float vLineDistance;

		#ifdef WORLD_UNITS

			varying vec4 worldPos;
			varying vec3 worldStart;
			varying vec3 worldEnd;

			#ifdef USE_DASH

				varying vec2 vUv;

			#endif

		#else

			varying vec2 vUv;

		#endif

		#include <common>
		#include <color_pars_fragment>
		#include <fog_pars_fragment>
		#include <logdepthbuf_pars_fragment>
		#include <clipping_planes_pars_fragment>

		vec2 closestLineToLine(vec3 p1, vec3 p2, vec3 p3, vec3 p4) {

			float mua;
			float mub;

			vec3 p13 = p1 - p3;
			vec3 p43 = p4 - p3;

			vec3 p21 = p2 - p1;

			float d1343 = dot( p13, p43 );
			float d4321 = dot( p43, p21 );
			float d1321 = dot( p13, p21 );
			float d4343 = dot( p43, p43 );
			float d2121 = dot( p21, p21 );

			float denom = d2121 * d4343 - d4321 * d4321;

			float numer = d1343 * d4321 - d1321 * d4343;

			mua = numer / denom;
			mua = clamp( mua, 0.0, 1.0 );
			mub = ( d1343 + d4321 * ( mua ) ) / d4343;
			mub = clamp( mub, 0.0, 1.0 );

			return vec2( mua, mub );

		}

		void main() {

			#include <clipping_planes_fragment>

			#ifdef USE_DASH

				if ( vUv.y < - 1.0 || vUv.y > 1.0 ) discard; // discard endcaps

				if ( mod( vLineDistance + dashOffset, dashSize + gapSize ) > dashSize ) discard; // todo - FIX

			#endif

			float alpha = opacity;

			#ifdef WORLD_UNITS

				// Find the closest points on the view ray and the line segment
				vec3 rayEnd = normalize( worldPos.xyz ) * 1e5;
				vec3 lineDir = worldEnd - worldStart;
				vec2 params = closestLineToLine( worldStart, worldEnd, vec3( 0.0, 0.0, 0.0 ), rayEnd );

				vec3 p1 = worldStart + lineDir * params.x;
				vec3 p2 = rayEnd * params.y;
				vec3 delta = p1 - p2;
				float len = length( delta );
				float norm = len / linewidth;

				#ifndef USE_DASH

					#ifdef USE_ALPHA_TO_COVERAGE

						float dnorm = fwidth( norm );
						alpha = 1.0 - smoothstep( 0.5 - dnorm, 0.5 + dnorm, norm );

					#else

						if ( norm > 0.5 ) {

							discard;

						}

					#endif

				#endif

			#else

				#ifdef USE_ALPHA_TO_COVERAGE

					// artifacts appear on some hardware if a derivative is taken within a conditional
					float a = vUv.x;
					float b = ( vUv.y > 0.0 ) ? vUv.y - 1.0 : vUv.y + 1.0;
					float len2 = a * a + b * b;
					float dlen = fwidth( len2 );

					if ( abs( vUv.y ) > 1.0 ) {

						alpha = 1.0 - smoothstep( 1.0 - dlen, 1.0 + dlen, len2 );

					}

				#else

					if ( abs( vUv.y ) > 1.0 ) {

						float a = vUv.x;
						float b = ( vUv.y > 0.0 ) ? vUv.y - 1.0 : vUv.y + 1.0;
						float len2 = a * a + b * b;

						if ( len2 > 1.0 ) discard;

					}

				#endif

			#endif

			vec4 diffuseColor = vec4( diffuse, alpha );

			#include <logdepthbuf_fragment>
			#include <color_fragment>

			gl_FragColor = vec4( diffuseColor.rgb, alpha );

			#include <tonemapping_fragment>
			#include <colorspace_fragment>
			#include <fog_fragment>
			#include <premultiplied_alpha_fragment>

		}
		`
  )
};
class Jn extends En {
  constructor(t) {
    super({
      type: "LineMaterial",
      uniforms: ya.clone(Le.line.uniforms),
      vertexShader: Le.line.vertexShader,
      fragmentShader: Le.line.fragmentShader,
      clipping: !0
      // required for clipping support
    }), this.isLineMaterial = !0, this.setValues(t);
  }
  get color() {
    return this.uniforms.diffuse.value;
  }
  set color(t) {
    this.uniforms.diffuse.value = t;
  }
  get worldUnits() {
    return "WORLD_UNITS" in this.defines;
  }
  set worldUnits(t) {
    t === !0 ? this.defines.WORLD_UNITS = "" : delete this.defines.WORLD_UNITS;
  }
  get linewidth() {
    return this.uniforms.linewidth.value;
  }
  set linewidth(t) {
    this.uniforms.linewidth && (this.uniforms.linewidth.value = t);
  }
  get dashed() {
    return "USE_DASH" in this.defines;
  }
  set dashed(t) {
    t === !0 !== this.dashed && (this.needsUpdate = !0), t === !0 ? this.defines.USE_DASH = "" : delete this.defines.USE_DASH;
  }
  get dashScale() {
    return this.uniforms.dashScale.value;
  }
  set dashScale(t) {
    this.uniforms.dashScale.value = t;
  }
  get dashSize() {
    return this.uniforms.dashSize.value;
  }
  set dashSize(t) {
    this.uniforms.dashSize.value = t;
  }
  get dashOffset() {
    return this.uniforms.dashOffset.value;
  }
  set dashOffset(t) {
    this.uniforms.dashOffset.value = t;
  }
  get gapSize() {
    return this.uniforms.gapSize.value;
  }
  set gapSize(t) {
    this.uniforms.gapSize.value = t;
  }
  get opacity() {
    return this.uniforms.opacity.value;
  }
  set opacity(t) {
    this.uniforms && (this.uniforms.opacity.value = t);
  }
  get resolution() {
    return this.uniforms.resolution.value;
  }
  set resolution(t) {
    this.uniforms.resolution.value.copy(t);
  }
  get alphaToCoverage() {
    return "USE_ALPHA_TO_COVERAGE" in this.defines;
  }
  set alphaToCoverage(t) {
    this.defines && (t === !0 !== this.alphaToCoverage && (this.needsUpdate = !0), t === !0 ? this.defines.USE_ALPHA_TO_COVERAGE = "" : delete this.defines.USE_ALPHA_TO_COVERAGE);
  }
}
const xr = new te(), Yo = new P(), $o = new P(), ve = new te(), xe = new te(), nn = new te(), Mr = new P(), Sr = new ne(), Me = new Lh(), qo = new P(), ys = new He(), Es = new Pi(), sn = new te();
let rn, qn;
function jo(i, t, e) {
  return sn.set(0, 0, -t, 1).applyMatrix4(i.projectionMatrix), sn.multiplyScalar(1 / sn.w), sn.x = qn / e.width, sn.y = qn / e.height, sn.applyMatrix4(i.projectionMatrixInverse), sn.multiplyScalar(1 / sn.w), Math.abs(Math.max(sn.x, sn.y));
}
function Um(i, t) {
  const e = i.matrixWorld, n = i.geometry, s = n.attributes.instanceStart, r = n.attributes.instanceEnd, a = Math.min(n.instanceCount, s.count);
  for (let o = 0, l = a; o < l; o++) {
    Me.start.fromBufferAttribute(s, o), Me.end.fromBufferAttribute(r, o), Me.applyMatrix4(e);
    const c = new P(), u = new P();
    rn.distanceSqToSegment(Me.start, Me.end, u, c), u.distanceTo(c) < qn * 0.5 && t.push({
      point: u,
      pointOnLine: c,
      distance: rn.origin.distanceTo(u),
      object: i,
      face: null,
      faceIndex: o,
      uv: null,
      uv1: null
    });
  }
}
function Im(i, t, e) {
  const n = t.projectionMatrix, r = i.material.resolution, a = i.matrixWorld, o = i.geometry, l = o.attributes.instanceStart, c = o.attributes.instanceEnd, u = Math.min(o.instanceCount, l.count), d = -t.near;
  rn.at(1, nn), nn.w = 1, nn.applyMatrix4(t.matrixWorldInverse), nn.applyMatrix4(n), nn.multiplyScalar(1 / nn.w), nn.x *= r.x / 2, nn.y *= r.y / 2, nn.z = 0, Mr.copy(nn), Sr.multiplyMatrices(t.matrixWorldInverse, a);
  for (let f = 0, m = u; f < m; f++) {
    if (ve.fromBufferAttribute(l, f), xe.fromBufferAttribute(c, f), ve.w = 1, xe.w = 1, ve.applyMatrix4(Sr), xe.applyMatrix4(Sr), ve.z > d && xe.z > d)
      continue;
    if (ve.z > d) {
      const b = ve.z - xe.z, S = (ve.z - d) / b;
      ve.lerp(xe, S);
    } else if (xe.z > d) {
      const b = xe.z - ve.z, S = (xe.z - d) / b;
      xe.lerp(ve, S);
    }
    ve.applyMatrix4(n), xe.applyMatrix4(n), ve.multiplyScalar(1 / ve.w), xe.multiplyScalar(1 / xe.w), ve.x *= r.x / 2, ve.y *= r.y / 2, xe.x *= r.x / 2, xe.y *= r.y / 2, Me.start.copy(ve), Me.start.z = 0, Me.end.copy(xe), Me.end.z = 0;
    const v = Me.closestPointToPointParameter(Mr, !0);
    Me.at(v, qo);
    const p = ml.lerp(ve.z, xe.z, v), h = p >= -1 && p <= 1, E = Mr.distanceTo(qo) < qn * 0.5;
    if (h && E) {
      Me.start.fromBufferAttribute(l, f), Me.end.fromBufferAttribute(c, f), Me.start.applyMatrix4(a), Me.end.applyMatrix4(a);
      const b = new P(), S = new P();
      rn.distanceSqToSegment(Me.start, Me.end, S, b), e.push({
        point: S,
        pointOnLine: b,
        distance: rn.origin.distanceTo(S),
        object: i,
        face: null,
        faceIndex: f,
        uv: null,
        uv1: null
      });
    }
  }
}
class Nl extends be {
  constructor(t = new Ca(), e = new Jn({ color: Math.random() * 16777215 })) {
    super(t, e), this.isLineSegments2 = !0, this.type = "LineSegments2";
  }
  // for backwards-compatibility, but could be a method of LineSegmentsGeometry...
  computeLineDistances() {
    const t = this.geometry, e = t.attributes.instanceStart, n = t.attributes.instanceEnd, s = new Float32Array(2 * e.count);
    for (let a = 0, o = 0, l = e.count; a < l; a++, o += 2)
      Yo.fromBufferAttribute(e, a), $o.fromBufferAttribute(n, a), s[o] = o === 0 ? 0 : s[o - 1], s[o + 1] = s[o] + Yo.distanceTo($o);
    const r = new ha(s, 2, 1);
    return t.setAttribute("instanceDistanceStart", new Dn(r, 1, 0)), t.setAttribute("instanceDistanceEnd", new Dn(r, 1, 1)), this;
  }
  raycast(t, e) {
    const n = this.material.worldUnits, s = t.camera;
    s === null && !n && console.error('LineSegments2: "Raycaster.camera" needs to be set in order to raycast against LineSegments2 while worldUnits is set to false.');
    const r = t.params.Line2 !== void 0 && t.params.Line2.threshold || 0;
    rn = t.ray;
    const a = this.matrixWorld, o = this.geometry, l = this.material;
    qn = l.linewidth + r, o.boundingSphere === null && o.computeBoundingSphere(), Es.copy(o.boundingSphere).applyMatrix4(a);
    let c;
    if (n)
      c = qn * 0.5;
    else {
      const d = Math.max(s.near, Es.distanceToPoint(rn.origin));
      c = jo(s, d, l.resolution);
    }
    if (Es.radius += c, rn.intersectsSphere(Es) === !1)
      return;
    o.boundingBox === null && o.computeBoundingBox(), ys.copy(o.boundingBox).applyMatrix4(a);
    let u;
    if (n)
      u = qn * 0.5;
    else {
      const d = Math.max(s.near, ys.distanceToPoint(rn.origin));
      u = jo(s, d, l.resolution);
    }
    ys.expandByScalar(u), rn.intersectsBox(ys) !== !1 && (n ? Um(this, e) : Im(this, s, e));
  }
  onBeforeRender(t) {
    const e = this.material.uniforms;
    e && e.resolution && (t.getViewport(xr), this.material.uniforms.resolution.value.set(xr.z, xr.w));
  }
}
class Pa extends Ca {
  constructor() {
    super(), this.isLineGeometry = !0, this.type = "LineGeometry";
  }
  setPositions(t) {
    const e = t.length - 3, n = new Float32Array(2 * e);
    for (let s = 0; s < e; s += 3)
      n[2 * s] = t[s], n[2 * s + 1] = t[s + 1], n[2 * s + 2] = t[s + 2], n[2 * s + 3] = t[s + 3], n[2 * s + 4] = t[s + 4], n[2 * s + 5] = t[s + 5];
    return super.setPositions(n), this;
  }
  setColors(t) {
    const e = t.length - 3, n = new Float32Array(2 * e);
    for (let s = 0; s < e; s += 3)
      n[2 * s] = t[s], n[2 * s + 1] = t[s + 1], n[2 * s + 2] = t[s + 2], n[2 * s + 3] = t[s + 3], n[2 * s + 4] = t[s + 4], n[2 * s + 5] = t[s + 5];
    return super.setColors(n), this;
  }
  setFromPoints(t) {
    const e = t.length - 1, n = new Float32Array(6 * e);
    for (let s = 0; s < e; s++)
      n[6 * s] = t[s].x, n[6 * s + 1] = t[s].y, n[6 * s + 2] = t[s].z || 0, n[6 * s + 3] = t[s + 1].x, n[6 * s + 4] = t[s + 1].y, n[6 * s + 5] = t[s + 1].z || 0;
    return super.setPositions(n), this;
  }
  fromLine(t) {
    const e = t.geometry;
    return this.setPositions(e.attributes.position.array), this;
  }
}
class Fl extends Nl {
  constructor(t = new Pa(), e = new Jn({ color: Math.random() * 16777215 })) {
    super(t, e), this.isLine2 = !0, this.type = "Line2";
  }
}
function Nm(i) {
  throw new Error(i);
}
function Zo(i, t, e) {
  const n = i[t] ?? e[0];
  if (e.indexOf(n) == -1)
    throw new Error(`${t} must be one of ${e}`);
  return n;
}
function fe(i, t) {
  return i[t] ?? Nm(`missing required key '${t}'`);
}
const Fm = [
  [0.43773, 0.82141, 1],
  [0.438, 0.82178, 1],
  [0.43825, 0.82216, 1],
  [0.43853, 0.82252, 1],
  [0.4388, 0.82289, 1],
  [0.43909, 0.82325, 0.99787],
  [0.43939, 0.82362, 0.99477],
  [0.43969, 0.82398, 0.99168],
  [0.44, 0.82433, 0.98857],
  [0.44031, 0.82469, 0.98546],
  [0.44063, 0.82504, 0.98235],
  [0.44095, 0.8254, 0.97923],
  [0.44129, 0.82575, 0.97611],
  [0.44163, 0.82609, 0.97299],
  [0.44199, 0.82644, 0.96986],
  [0.44234, 0.82678, 0.96673],
  [0.4427, 0.82712, 0.96359],
  [0.44307, 0.82746, 0.96045],
  [0.44345, 0.82779, 0.95731],
  [0.44384, 0.82812, 0.95416],
  [0.44424, 0.82845, 0.95101],
  [0.44464, 0.82878, 0.94785],
  [0.44505, 0.82911, 0.94469],
  [0.44546, 0.82943, 0.94152],
  [0.44589, 0.82975, 0.93835],
  [0.44634, 0.83007, 0.93517],
  [0.44679, 0.83038, 0.93199],
  [0.44725, 0.83069, 0.9288],
  [0.4477, 0.831, 0.92562],
  [0.4482, 0.83131, 0.92243],
  [0.44869, 0.83161, 0.91922],
  [0.44919, 0.83191, 0.91602],
  [0.4497, 0.83221, 0.91281],
  [0.45022, 0.8325, 0.9096],
  [0.45076, 0.83279, 0.90638],
  [0.4513, 0.83308, 0.90316],
  [0.45186, 0.83337, 0.89993],
  [0.45244, 0.83365, 0.8967],
  [0.45301, 0.83394, 0.89345],
  [0.4536, 0.83421, 0.89021],
  [0.45421, 0.83448, 0.88696],
  [0.45483, 0.83475, 0.8837],
  [0.45546, 0.83502, 0.88044],
  [0.45611, 0.83528, 0.87717],
  [0.45676, 0.83555, 0.8739],
  [0.45744, 0.8358, 0.87062],
  [0.45813, 0.83606, 0.86733],
  [0.45884, 0.83631, 0.86405],
  [0.45956, 0.83656, 0.86075],
  [0.46029, 0.8368, 0.85744],
  [0.46104, 0.83704, 0.85414],
  [0.46181, 0.83728, 0.85082],
  [0.46259, 0.83751, 0.8475],
  [0.46339, 0.83774, 0.84417],
  [0.46422, 0.83796, 0.84084],
  [0.46505, 0.83818, 0.8375],
  [0.46591, 0.8384, 0.83416],
  [0.46679, 0.83861, 0.8308],
  [0.46767, 0.83882, 0.82744],
  [0.46859, 0.83903, 0.82407],
  [0.46953, 0.83923, 0.8207],
  [0.47047, 0.83943, 0.81732],
  [0.47145, 0.83962, 0.81393],
  [0.47244, 0.83981, 0.81053],
  [0.47345, 0.83999, 0.80714],
  [0.47449, 0.84017, 0.80373],
  [0.47556, 0.84034, 0.80031],
  [0.47664, 0.84051, 0.79688],
  [0.47774, 0.84067, 0.79346],
  [0.47888, 0.84083, 0.79002],
  [0.48003, 0.84098, 0.78658],
  [0.48121, 0.84113, 0.78312],
  [0.48242, 0.84127, 0.77966],
  [0.48365, 0.8414, 0.77619],
  [0.48491, 0.84153, 0.77272],
  [0.4862, 0.84166, 0.76924],
  [0.48753, 0.84178, 0.76575],
  [0.48887, 0.84189, 0.76225],
  [0.49025, 0.842, 0.75875],
  [0.49167, 0.8421, 0.75524],
  [0.49309, 0.84219, 0.75172],
  [0.49457, 0.84228, 0.74819],
  [0.49605, 0.84236, 0.74465],
  [0.4976, 0.84243, 0.74111],
  [0.49917, 0.84249, 0.73756],
  [0.50078, 0.84255, 0.73401],
  [0.50242, 0.8426, 0.73045],
  [0.5041, 0.84264, 0.72687],
  [0.5058, 0.84267, 0.7233],
  [0.50756, 0.8427, 0.71972],
  [0.50935, 0.84271, 0.71613],
  [0.51117, 0.84272, 0.71253],
  [0.51304, 0.84272, 0.70893],
  [0.51496, 0.84271, 0.70532],
  [0.51691, 0.84269, 0.70171],
  [0.51891, 0.84266, 0.69809],
  [0.52094, 0.84262, 0.69447],
  [0.52303, 0.84257, 0.69084],
  [0.52516, 0.84251, 0.68722],
  [0.52734, 0.84243, 0.68359],
  [0.52957, 0.84235, 0.67995],
  [0.53184, 0.84226, 0.67632],
  [0.53416, 0.84215, 0.67268],
  [0.53654, 0.84203, 0.66904],
  [0.53896, 0.84189, 0.66541],
  [0.54142, 0.84175, 0.66178],
  [0.54396, 0.84159, 0.65814],
  [0.54654, 0.84142, 0.65451],
  [0.54917, 0.84123, 0.65088],
  [0.55185, 0.84103, 0.64727],
  [0.55458, 0.84082, 0.64365],
  [0.55737, 0.84059, 0.64005],
  [0.56021, 0.84035, 0.63646],
  [0.56311, 0.84009, 0.63287],
  [0.56606, 0.83981, 0.6293],
  [0.56906, 0.83952, 0.62575],
  [0.57213, 0.8392, 0.6222],
  [0.57524, 0.83888, 0.61869],
  [0.5784, 0.83854, 0.61518],
  [0.58162, 0.83818, 0.61171],
  [0.5849, 0.8378, 0.60826],
  [0.58821, 0.83741, 0.60483],
  [0.59159, 0.837, 0.60142],
  [0.595, 0.83656, 0.59805],
  [0.59846, 0.83612, 0.59472],
  [0.60197, 0.83565, 0.59142],
  [0.60552, 0.83517, 0.58814],
  [0.60913, 0.83467, 0.58492],
  [0.61275, 0.83415, 0.58173],
  [0.61642, 0.83362, 0.57858],
  [0.62014, 0.83306, 0.57549],
  [0.62388, 0.83249, 0.57243],
  [0.62765, 0.83191, 0.56942],
  [0.63145, 0.8313, 0.56646],
  [0.63527, 0.83068, 0.56356],
  [0.63912, 0.83004, 0.56069],
  [0.64299, 0.82939, 0.55789],
  [0.64689, 0.82872, 0.55514],
  [0.6508, 0.82803, 0.55245],
  [0.65472, 0.82733, 0.5498],
  [0.65865, 0.82662, 0.54721],
  [0.66261, 0.82589, 0.54468],
  [0.66656, 0.82515, 0.54221],
  [0.67053, 0.82439, 0.53979],
  [0.6745, 0.82363, 0.53744],
  [0.67847, 0.82285, 0.53513],
  [0.68244, 0.82206, 0.53288],
  [0.68642, 0.82125, 0.53068],
  [0.69038, 0.82043, 0.52855],
  [0.69435, 0.8196, 0.52647],
  [0.6983, 0.81877, 0.5244],
  [0.70224, 0.81793, 0.52237],
  [0.70617, 0.81709, 0.52036],
  [0.71008, 0.81624, 0.51839],
  [0.71397, 0.81537, 0.51642],
  [0.71785, 0.81451, 0.51449],
  [0.72171, 0.81363, 0.51258],
  [0.72557, 0.81275, 0.5107],
  [0.7294, 0.81187, 0.50885],
  [0.73323, 0.81098, 0.50702],
  [0.73704, 0.81008, 0.50522],
  [0.74083, 0.80918, 0.50344],
  [0.74462, 0.80826, 0.50171],
  [0.7484, 0.80735, 0.49999],
  [0.75216, 0.80642, 0.49833],
  [0.75591, 0.80548, 0.49666],
  [0.75965, 0.80455, 0.49505],
  [0.76337, 0.8036, 0.49346],
  [0.76709, 0.80265, 0.49191],
  [0.7708, 0.80169, 0.49039],
  [0.77448, 0.80072, 0.4889],
  [0.77817, 0.79975, 0.48745],
  [0.78184, 0.79876, 0.48602],
  [0.7855, 0.79777, 0.48463],
  [0.78915, 0.79678, 0.48328],
  [0.79278, 0.79578, 0.48198],
  [0.7964, 0.79477, 0.4807],
  [0.80002, 0.79376, 0.47946],
  [0.80362, 0.79273, 0.47826],
  [0.80722, 0.7917, 0.47709],
  [0.8108, 0.79066, 0.47597],
  [0.81437, 0.78962, 0.47489],
  [0.81792, 0.78857, 0.47383],
  [0.82147, 0.78751, 0.47283],
  [0.82501, 0.78645, 0.47187],
  [0.82853, 0.78538, 0.47094],
  [0.83205, 0.7843, 0.47006],
  [0.83555, 0.78321, 0.46922],
  [0.83903, 0.78212, 0.46841],
  [0.84251, 0.78102, 0.46765],
  [0.84597, 0.77992, 0.46694],
  [0.84943, 0.77881, 0.46626],
  [0.85287, 0.77769, 0.46563],
  [0.85629, 0.77657, 0.46504],
  [0.85971, 0.77544, 0.46451],
  [0.86311, 0.7743, 0.46401],
  [0.86649, 0.77316, 0.46354],
  [0.86986, 0.77201, 0.46313],
  [0.87323, 0.77086, 0.46276],
  [0.87657, 0.7697, 0.46244],
  [0.87991, 0.76853, 0.46216],
  [0.88323, 0.76736, 0.46193],
  [0.88653, 0.76619, 0.46174],
  [0.88982, 0.76501, 0.46159],
  [0.89309, 0.76382, 0.46149],
  [0.89636, 0.76263, 0.46143],
  [0.8996, 0.76143, 0.46142],
  [0.90283, 0.76023, 0.46145],
  [0.90605, 0.75903, 0.46152],
  [0.90924, 0.75782, 0.46164],
  [0.91242, 0.7566, 0.4618],
  [0.91559, 0.75539, 0.462],
  [0.91874, 0.75416, 0.46225],
  [0.92188, 0.75293, 0.46253],
  [0.92499, 0.75171, 0.46286],
  [0.92809, 0.75047, 0.46324],
  [0.93118, 0.74922, 0.46366],
  [0.93424, 0.74799, 0.46413],
  [0.9373, 0.74674, 0.46462],
  [0.94033, 0.74549, 0.46515],
  [0.94335, 0.74424, 0.46573],
  [0.94635, 0.74299, 0.46635],
  [0.94933, 0.74172, 0.46702],
  [0.95229, 0.74046, 0.46771],
  [0.95524, 0.7392, 0.46844],
  [0.95817, 0.73792, 0.46923],
  [0.96108, 0.73666, 0.47004],
  [0.96398, 0.73539, 0.47088],
  [0.96685, 0.73411, 0.47177],
  [0.96971, 0.73284, 0.47268],
  [0.97256, 0.73155, 0.47363],
  [0.97537, 0.73027, 0.47464],
  [0.97818, 0.72899, 0.47566],
  [0.98097, 0.7277, 0.47671],
  [0.98374, 0.72642, 0.47781],
  [0.98649, 0.72513, 0.47893],
  [0.98923, 0.72383, 0.4801],
  [0.99194, 0.72254, 0.48128],
  [0.99464, 0.72125, 0.4825],
  [0.99732, 0.71995, 0.48375],
  [0.99999, 0.71865, 0.48503],
  [1, 0.71736, 0.48634],
  [1, 0.71606, 0.48769],
  [1, 0.71476, 0.48905],
  [1, 0.71346, 0.49046],
  [1, 0.71215, 0.49188],
  [1, 0.71085, 0.49332],
  [1, 0.70955, 0.49481],
  [1, 0.70824, 0.4963],
  [1, 0.70694, 0.49785],
  [1, 0.70563, 0.49939],
  [1, 0.70432, 0.50098],
  [1, 0.70301, 0.50257],
  [1, 0.70169, 0.50421],
  [1, 0.70039, 0.50584],
  [1, 0.69907, 0.50753]
];
function Om(i, t) {
  const e = t.length, n = Math.max(0, Math.min(Math.round(i * e), e - 1)), s = t[n];
  return [s[0], s[1], s[2]];
}
const Ko = [
  16e-5,
  662e-6,
  2362e-6,
  7242e-6,
  0.01911,
  0.0434,
  0.084736,
  0.140638,
  0.204492,
  0.264737,
  0.314679,
  0.357719,
  0.383734,
  0.386726,
  0.370702,
  0.342957,
  0.302273,
  0.254085,
  0.195618,
  0.132349,
  0.080507,
  0.041072,
  0.016172,
  5132e-6,
  3816e-6,
  0.015444,
  0.037465,
  0.071358,
  0.117749,
  0.172953,
  0.236491,
  0.304213,
  0.376772,
  0.451584,
  0.529826,
  0.616053,
  0.705224,
  0.793832,
  0.878655,
  0.951162,
  1.01416,
  1.0743,
  1.11852,
  1.1343,
  1.12399,
  1.0891,
  1.03048,
  0.95074,
  0.856297,
  0.75493,
  0.647467,
  0.53511,
  0.431567,
  0.34369,
  0.268329,
  0.2043,
  0.152568,
  0.11221,
  0.081261,
  0.05793,
  0.040851,
  0.028623,
  0.019941,
  0.013842,
  9577e-6,
  6605e-6,
  4553e-6,
  3145e-6,
  2175e-6,
  1506e-6,
  1045e-6,
  727e-6,
  508e-6,
  356e-6,
  251e-6,
  178e-6,
  126e-6,
  9e-5,
  65e-6,
  46e-6,
  33e-6
], Bm = [
  17e-6,
  72e-6,
  253e-6,
  769e-6,
  2004e-6,
  4509e-6,
  8756e-6,
  0.014456,
  0.021391,
  0.029497,
  0.038676,
  0.049602,
  0.062077,
  0.074704,
  0.089456,
  0.106256,
  0.128201,
  0.152761,
  0.18519,
  0.21994,
  0.253589,
  0.297665,
  0.339133,
  0.395379,
  0.460777,
  0.53136,
  0.606741,
  0.68566,
  0.761757,
  0.82333,
  0.875211,
  0.92381,
  0.961988,
  0.9822,
  0.991761,
  0.99911,
  0.99734,
  0.98238,
  0.955552,
  0.915175,
  0.868934,
  0.825623,
  0.777405,
  0.720353,
  0.658341,
  0.593878,
  0.527963,
  0.461834,
  0.398057,
  0.339554,
  0.283493,
  0.228254,
  0.179828,
  0.140211,
  0.107633,
  0.081187,
  0.060281,
  0.044096,
  0.0318,
  0.022602,
  0.015905,
  0.01113,
  7749e-6,
  5375e-6,
  3718e-6,
  2565e-6,
  1768e-6,
  1222e-6,
  846e-6,
  586e-6,
  407e-6,
  284e-6,
  199e-6,
  14e-5,
  98e-6,
  7e-5,
  5e-5,
  36e-6,
  25e-6,
  18e-6,
  13e-6
], zm = [
  705e-6,
  2928e-6,
  0.010482,
  0.032344,
  0.086011,
  0.19712,
  0.389366,
  0.65676,
  0.972542,
  1.2825,
  1.55348,
  1.7985,
  1.96728,
  2.0273,
  1.9948,
  1.9007,
  1.74537,
  1.5549,
  1.31756,
  1.0302,
  0.772125,
  0.57006,
  0.415254,
  0.302356,
  0.218502,
  0.159249,
  0.112044,
  0.082248,
  0.060709,
  0.04305,
  0.030451,
  0.020584,
  0.013676,
  7918e-6,
  3988e-6,
  1091e-6,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0
], pn = [
  [3.2404542, -1.5371385, -0.4985314],
  [-0.969266, 1.8760108, 0.041556],
  [0.0556434, -0.2040259, 1.0572252]
];
function Hm(i) {
  const t = 31308e-7, e = 0.055, n = 1 / 2.4;
  return i <= t ? 12.92 * i : (1 + e) * Math.pow(i, n) - e;
}
function yr(i, t, e) {
  let n = 0, s = i.length - 1;
  for (; n < s; ) {
    const c = Math.floor((n + s) / 2);
    i[c] < e ? n = c + 1 : s = c;
  }
  n > 0 && n--;
  const r = i[n], a = i[s], o = t[n], l = t[s];
  return o + (l - o) * (e - r) / (a - r);
}
function km(i) {
  const n = i.length, s = new Array(n).fill(0).map(() => [0, 0, 0]), r = Array.from({ length: Ko.length }, (a, o) => 380 + o * 5);
  for (let a = 0; a < n; a++) {
    const o = yr(r, Ko, i[a]), l = yr(r, Bm, i[a]), c = yr(r, zm, i[a]), u = [
      pn[0][0] * o + pn[0][1] * l + pn[0][2] * c,
      pn[1][0] * o + pn[1][1] * l + pn[1][2] * c,
      pn[2][0] * o + pn[2][1] * l + pn[2][2] * c
    ];
    s[a] = u.map((d) => {
      const f = Hm(d);
      return Math.max(0, Math.min(1, f));
    });
  }
  return s;
}
function Jo(i, t, e) {
  console.assert(i.length == 3), console.assert(t.length == 3);
  const n = new Jn({
    color: e,
    linewidth: 1.1,
    worldUnits: !1,
    side: Xe,
    transparent: !0
  }), s = [];
  s.push(new P().fromArray(i)), s.push(new P().fromArray(t));
  const r = new Pa().setFromPoints(s);
  return new Fl(r, n);
}
function Ol(i) {
  if (i.length !== 4 || i.some((n) => n.length !== 4))
    throw new Error("Input must be a 4x4 array");
  const t = new ne(), e = i[0].map(
    (n, s) => i.map((r) => r[s])
  );
  return t.fromArray(e.flat()), t;
}
function Vm(i) {
  if (i.length !== 3 || i.some((t) => t.length !== 3))
    throw new Error("Input matrix must be 3x3");
  return [
    [i[0][0], i[0][1], 0, i[0][2]],
    [i[1][0], i[1][1], 0, i[1][2]],
    [0, 0, 1, 0],
    [i[2][0], i[2][1], 0, i[2][2]]
  ];
}
function Gm(i, t) {
  return t == 2 ? Wm(i) : Xm(i);
}
function Wm(i) {
  const t = new le(), e = fe(i, "data");
  for (const n of e) {
    const s = Vm(fe(n, "matrix")), r = fe(n, "samples"), a = new Jn({
      color: "cyan",
      linewidth: 2,
      worldUnits: !1,
      side: Xe
    }), o = [];
    for (const u of r)
      o.push(u[0], u[1], 1);
    const l = new Pa();
    l.setPositions(o);
    const c = new Fl(l, a);
    c.applyMatrix4(Ol(s)), t.add(c);
  }
  return t;
}
function Xm(i) {
  const t = fe(i, "data"), e = new le();
  for (const n of t) {
    const s = fe(n, "matrix"), r = fe(n, "samples"), a = n.clip_planes ?? [], o = Ym(s, r, a);
    e.add(o);
  }
  return e;
}
function Ym(i, t, e) {
  const s = new ne().fromArray([
    0,
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1
  ]), r = Ol(i), a = new ne();
  a.multiplyMatrices(r, s);
  const o = t.map((f) => new bt(f[1], f[0])), l = new ba(o, 50), c = [];
  for (const f of e) {
    const m = new P(f[0], f[1], f[2]), g = f[3], v = new _n(m, g);
    v.applyMatrix4(r), c.push(v);
  }
  const u = new wh({
    side: Xe,
    clippingPlanes: c,
    clipIntersection: !1
  });
  u.transparent = !0, u.opacity = 0.8;
  const d = new be(l, u);
  return d.applyMatrix4(a), d;
}
function $m(i, t) {
  i.traverse((e) => {
    e.layers.disableAll();
  }), i.traverse((e) => {
    for (const n of t)
      e.layers.enable(n);
  });
}
function qm(i, t) {
  const e = fe(i, "data"), n = i.color ?? "#ffffff", s = new le();
  for (const r of e) {
    const a = new Ta(0.1, 8, 8), o = new Fs({ color: n });
    o.transparent = !0, o.opacity = 0.8;
    const l = new be(a, o);
    if (r.length != t)
      throw new Error(
        `point array length is ${r.length} (expected ${t})`
      );
    t == 2 ? l.position.set(r[0], r[1], 2) : l.position.set(r[0], r[1], r[2]), s.add(l);
  }
  return $m(s, i.layers ?? [0]), s;
}
function jm(i, t) {
  const e = fe(i, "data"), n = new le();
  for (const o of e) {
    var s, r, a;
    t == 2 ? (console.assert(o.length == 5), s = o.slice(0, 2), r = o.slice(2, 4), a = o[4]) : (console.assert(o.length == 7), s = o.slice(0, 3), r = o.slice(3, 6), a = o[6]);
    const l = new P(...s);
    l.normalize();
    const c = new P(...r), u = 16776960, d = new Uh(l, c, a, u);
    n.add(d);
  }
  return n;
}
function Zm(i, t, e) {
  const n = fe(i, "points"), s = i.color ?? "#ffa724", r = i.variables ?? {}, a = i.domain ?? {}, o = 2 * t, l = new le();
  if (e.show == !1)
    return l;
  if (!(Symbol.iterator in Object(n)))
    throw new Error("points field of ray is not iterable");
  const c = [];
  for (const [p, h] of n.entries()) {
    if (h.length != o)
      throw new Error(
        `Invalid ray array length, got ${h.length} for dim ${t}`
      );
    t == 2 ? c.push(h[0], h[1], 0, h[2], h[3], 0) : c.push(h[0], h[1], h[2], h[3], h[4], h[5]);
  }
  const u = [];
  var d = !0;
  if (e.colorDim == null)
    d = !0;
  else {
    d = !1;
    for (const [p, h] of n.entries()) {
      var f;
      if (e.trueColor == !1) {
        if (!a.hasOwnProperty(e.colorDim))
          throw new Error(
            `${e.colorDim} missing from ray domain object`
          );
        const [b, S] = a[e.colorDim], L = (r[e.colorDim][p] - b) / (S - b);
        f = Om(L, Fm);
      } else {
        const b = r[e.colorDim][p];
        f = km([b])[0];
      }
      const E = new Vt().setRGB(f[0], f[1], f[2], Be);
      u.push(
        ...E.toArray(),
        ...E.toArray()
      );
    }
  }
  const m = new Ca();
  m.setPositions(c), d || m.setColors(u);
  const g = new Jn({
    ...d ? { color: s } : {},
    linewidth: 1,
    vertexColors: !d,
    dashed: !1,
    transparent: !0
  }), v = new Nl(m, g);
  return l.add(v), l;
}
function Km(i) {
  const t = /* @__PURE__ */ new Set([]), e = fe(i, "data");
  for (const n of e)
    if (fe(n, "type") == "rays") {
      const s = n.variables ?? {};
      Object.keys(s).forEach((r) => t.add(r));
    }
  return Array.from(t);
}
class Jm {
  constructor(t, e) {
    Qt(this, "root");
    Qt(this, "dim");
    // Model
    Qt(this, "surfaces");
    Qt(this, "validRays");
    Qt(this, "blockedRays");
    Qt(this, "outputRays");
    Qt(this, "points");
    Qt(this, "arrows");
    Qt(this, "opticalAxis");
    Qt(this, "otherAxes");
    Qt(this, "scene");
    Qt(this, "variables");
    Qt(this, "title");
    if (this.root = t, this.dim = e, this.scene = new Sh(), this.variables = Km(t), this.surfaces = new le(), this.setupSurfaces(e), this.validRays = new le(), this.blockedRays = new le(), this.outputRays = new le(), this.points = new le(), this.setupPoints(e), this.arrows = new le(), this.setupArrows(e), this.otherAxes = new le(), e == 2)
      this.otherAxes.add(Jo([0, -500, 0], [0, 500, 0], "#e3e3e3"));
    else if (e == 3) {
      const n = new Ih(5);
      this.otherAxes.add(n);
    }
    this.opticalAxis = new le(), this.opticalAxis.add(Jo([-500, 0, 0], [500, 0, 0], "#e3e3e3")), this.title = t.title ?? "", this.scene.add(this.surfaces), this.scene.add(this.otherAxes), this.scene.add(this.opticalAxis), this.opticalAxis.visible = !1, this.otherAxes.visible = !1;
  }
  setupValidRays(t) {
    var e;
    (e = this.validRays) == null || e.removeFromParent(), this.validRays = new le(), this.validRays.add(this.setupRaysLayer(0, t)), this.validRays.add(this.setupRaysLayer(1, t)), this.scene.add(this.validRays);
  }
  setupBlockedRays(t) {
    var e;
    (e = this.blockedRays) == null || e.removeFromParent(), this.blockedRays = this.setupRaysLayer(2, t), this.scene.add(this.blockedRays);
  }
  setupOutputRays(t) {
    var e;
    (e = this.outputRays) == null || e.removeFromParent(), this.outputRays = this.setupRaysLayer(3, t), this.scene.add(this.outputRays);
  }
  setupPoints(t) {
    var n;
    (n = this.points) == null || n.removeFromParent(), this.points = new le();
    const e = fe(this.root, "data");
    for (const s of e)
      fe(s, "type") == "points" && this.points.add(qm(s, t));
    this.scene.add(this.points);
  }
  setupArrows(t) {
    var n;
    (n = this.arrows) == null || n.removeFromParent(), this.arrows = new le();
    const e = fe(this.root, "data");
    for (const s of e)
      fe(s, "type") == "arrows" && this.points.add(jm(s, t));
    this.scene.add(this.arrows);
  }
  setupRaysLayer(t, e) {
    const n = new le(), s = fe(this.root, "data");
    for (const r of s)
      fe(r, "type") == "rays" && (r.layers ?? [0])[0] === t && n.add(Zm(r, this.dim, e));
    return n;
  }
  setupSurfaces(t) {
    var n;
    (n = this.surfaces) == null || n.removeFromParent(), this.surfaces = new le();
    const e = fe(this.root, "data");
    for (const s of e)
      if (fe(s, "type") == "surfaces") {
        const r = Gm(s, t);
        this.surfaces.add(r);
      }
  }
  setSurfacesColor(t) {
    this.surfaces.traverse((e) => {
      e instanceof be && e.material instanceof ti && "color" in e.material && (e.material.color = t);
    });
  }
  setRaysOpacity(t) {
    const e = (n) => {
      n instanceof be && n.material instanceof Jn && (n.material.opacity = t);
    };
    this.validRays.traverse(e), this.blockedRays.traverse(e), this.outputRays.traverse(e);
  }
  setRaysThickness(t) {
    const e = (n) => {
      n instanceof be && n.material instanceof Jn && (n.material.linewidth = t);
    };
    this.validRays.traverse(e), this.blockedRays.traverse(e), this.outputRays.traverse(e);
  }
  getBB() {
    const t = new He();
    return t.union(new He().setFromObject(this.surfaces)), t.union(new He().setFromObject(this.validRays)), t.union(new He().setFromObject(this.blockedRays)), t.union(new He().setFromObject(this.outputRays)), t.isEmpty() && (t.min = new P(-10, -10, -10), t.max = new P(10, 10, 10)), t;
  }
}
/**
 * lil-gui
 * https://lil-gui.georgealways.com
 * @version 0.20.0
 * @author George Michael Brower
 * @license MIT
 */
class on {
  constructor(t, e, n, s, r = "div") {
    this.parent = t, this.object = e, this.property = n, this._disabled = !1, this._hidden = !1, this.initialValue = this.getValue(), this.domElement = document.createElement(r), this.domElement.classList.add("controller"), this.domElement.classList.add(s), this.$name = document.createElement("div"), this.$name.classList.add("name"), on.nextNameID = on.nextNameID || 0, this.$name.id = `lil-gui-name-${++on.nextNameID}`, this.$widget = document.createElement("div"), this.$widget.classList.add("widget"), this.$disable = this.$widget, this.domElement.appendChild(this.$name), this.domElement.appendChild(this.$widget), this.domElement.addEventListener("keydown", (a) => a.stopPropagation()), this.domElement.addEventListener("keyup", (a) => a.stopPropagation()), this.parent.children.push(this), this.parent.controllers.push(this), this.parent.$children.appendChild(this.domElement), this._listenCallback = this._listenCallback.bind(this), this.name(n);
  }
  /**
   * Sets the name of the controller and its label in the GUI.
   * @param {string} name
   * @returns {this}
   */
  name(t) {
    return this._name = t, this.$name.textContent = t, this;
  }
  /**
   * Pass a function to be called whenever the value is modified by this controller.
   * The function receives the new value as its first parameter. The value of `this` will be the
   * controller.
   *
   * For function controllers, the `onChange` callback will be fired on click, after the function
   * executes.
   * @param {Function} callback
   * @returns {this}
   * @example
   * const controller = gui.add( object, 'property' );
   *
   * controller.onChange( function( v ) {
   * 	console.log( 'The value is now ' + v );
   * 	console.assert( this === controller );
   * } );
   */
  onChange(t) {
    return this._onChange = t, this;
  }
  /**
   * Calls the onChange methods of this controller and its parent GUI.
   * @protected
   */
  _callOnChange() {
    this.parent._callOnChange(this), this._onChange !== void 0 && this._onChange.call(this, this.getValue()), this._changed = !0;
  }
  /**
   * Pass a function to be called after this controller has been modified and loses focus.
   * @param {Function} callback
   * @returns {this}
   * @example
   * const controller = gui.add( object, 'property' );
   *
   * controller.onFinishChange( function( v ) {
   * 	console.log( 'Changes complete: ' + v );
   * 	console.assert( this === controller );
   * } );
   */
  onFinishChange(t) {
    return this._onFinishChange = t, this;
  }
  /**
   * Should be called by Controller when its widgets lose focus.
   * @protected
   */
  _callOnFinishChange() {
    this._changed && (this.parent._callOnFinishChange(this), this._onFinishChange !== void 0 && this._onFinishChange.call(this, this.getValue())), this._changed = !1;
  }
  /**
   * Sets the controller back to its initial value.
   * @returns {this}
   */
  reset() {
    return this.setValue(this.initialValue), this._callOnFinishChange(), this;
  }
  /**
   * Enables this controller.
   * @param {boolean} enabled
   * @returns {this}
   * @example
   * controller.enable();
   * controller.enable( false ); // disable
   * controller.enable( controller._disabled ); // toggle
   */
  enable(t = !0) {
    return this.disable(!t);
  }
  /**
   * Disables this controller.
   * @param {boolean} disabled
   * @returns {this}
   * @example
   * controller.disable();
   * controller.disable( false ); // enable
   * controller.disable( !controller._disabled ); // toggle
   */
  disable(t = !0) {
    return t === this._disabled ? this : (this._disabled = t, this.domElement.classList.toggle("disabled", t), this.$disable.toggleAttribute("disabled", t), this);
  }
  /**
   * Shows the Controller after it's been hidden.
   * @param {boolean} show
   * @returns {this}
   * @example
   * controller.show();
   * controller.show( false ); // hide
   * controller.show( controller._hidden ); // toggle
   */
  show(t = !0) {
    return this._hidden = !t, this.domElement.style.display = this._hidden ? "none" : "", this;
  }
  /**
   * Hides the Controller.
   * @returns {this}
   */
  hide() {
    return this.show(!1);
  }
  /**
   * Changes this controller into a dropdown of options.
   *
   * Calling this method on an option controller will simply update the options. However, if this
   * controller was not already an option controller, old references to this controller are
   * destroyed, and a new controller is added to the end of the GUI.
   * @example
   * // safe usage
   *
   * gui.add( obj, 'prop1' ).options( [ 'a', 'b', 'c' ] );
   * gui.add( obj, 'prop2' ).options( { Big: 10, Small: 1 } );
   * gui.add( obj, 'prop3' );
   *
   * // danger
   *
   * const ctrl1 = gui.add( obj, 'prop1' );
   * gui.add( obj, 'prop2' );
   *
   * // calling options out of order adds a new controller to the end...
   * const ctrl2 = ctrl1.options( [ 'a', 'b', 'c' ] );
   *
   * // ...and ctrl1 now references a controller that doesn't exist
   * assert( ctrl2 !== ctrl1 )
   * @param {object|Array} options
   * @returns {Controller}
   */
  options(t) {
    const e = this.parent.add(this.object, this.property, t);
    return e.name(this._name), this.destroy(), e;
  }
  /**
   * Sets the minimum value. Only works on number controllers.
   * @param {number} min
   * @returns {this}
   */
  min(t) {
    return this;
  }
  /**
   * Sets the maximum value. Only works on number controllers.
   * @param {number} max
   * @returns {this}
   */
  max(t) {
    return this;
  }
  /**
   * Values set by this controller will be rounded to multiples of `step`. Only works on number
   * controllers.
   * @param {number} step
   * @returns {this}
   */
  step(t) {
    return this;
  }
  /**
   * Rounds the displayed value to a fixed number of decimals, without affecting the actual value
   * like `step()`. Only works on number controllers.
   * @example
   * gui.add( object, 'property' ).listen().decimals( 4 );
   * @param {number} decimals
   * @returns {this}
   */
  decimals(t) {
    return this;
  }
  /**
   * Calls `updateDisplay()` every animation frame. Pass `false` to stop listening.
   * @param {boolean} listen
   * @returns {this}
   */
  listen(t = !0) {
    return this._listening = t, this._listenCallbackID !== void 0 && (cancelAnimationFrame(this._listenCallbackID), this._listenCallbackID = void 0), this._listening && this._listenCallback(), this;
  }
  _listenCallback() {
    this._listenCallbackID = requestAnimationFrame(this._listenCallback);
    const t = this.save();
    t !== this._listenPrevValue && this.updateDisplay(), this._listenPrevValue = t;
  }
  /**
   * Returns `object[ property ]`.
   * @returns {any}
   */
  getValue() {
    return this.object[this.property];
  }
  /**
   * Sets the value of `object[ property ]`, invokes any `onChange` handlers and updates the display.
   * @param {any} value
   * @returns {this}
   */
  setValue(t) {
    return this.getValue() !== t && (this.object[this.property] = t, this._callOnChange(), this.updateDisplay()), this;
  }
  /**
   * Updates the display to keep it in sync with the current value. Useful for updating your
   * controllers when their values have been modified outside of the GUI.
   * @returns {this}
   */
  updateDisplay() {
    return this;
  }
  load(t) {
    return this.setValue(t), this._callOnFinishChange(), this;
  }
  save() {
    return this.getValue();
  }
  /**
   * Destroys this controller and removes it from the parent GUI.
   */
  destroy() {
    this.listen(!1), this.parent.children.splice(this.parent.children.indexOf(this), 1), this.parent.controllers.splice(this.parent.controllers.indexOf(this), 1), this.parent.$children.removeChild(this.domElement);
  }
}
class Qm extends on {
  constructor(t, e, n) {
    super(t, e, n, "boolean", "label"), this.$input = document.createElement("input"), this.$input.setAttribute("type", "checkbox"), this.$input.setAttribute("aria-labelledby", this.$name.id), this.$widget.appendChild(this.$input), this.$input.addEventListener("change", () => {
      this.setValue(this.$input.checked), this._callOnFinishChange();
    }), this.$disable = this.$input, this.updateDisplay();
  }
  updateDisplay() {
    return this.$input.checked = this.getValue(), this;
  }
}
function da(i) {
  let t, e;
  return (t = i.match(/(#|0x)?([a-f0-9]{6})/i)) ? e = t[2] : (t = i.match(/rgb\(\s*(\d*)\s*,\s*(\d*)\s*,\s*(\d*)\s*\)/)) ? e = parseInt(t[1]).toString(16).padStart(2, 0) + parseInt(t[2]).toString(16).padStart(2, 0) + parseInt(t[3]).toString(16).padStart(2, 0) : (t = i.match(/^#?([a-f0-9])([a-f0-9])([a-f0-9])$/i)) && (e = t[1] + t[1] + t[2] + t[2] + t[3] + t[3]), e ? "#" + e : !1;
}
const t_ = {
  isPrimitive: !0,
  match: (i) => typeof i == "string",
  fromHexString: da,
  toHexString: da
}, Wi = {
  isPrimitive: !0,
  match: (i) => typeof i == "number",
  fromHexString: (i) => parseInt(i.substring(1), 16),
  toHexString: (i) => "#" + i.toString(16).padStart(6, 0)
}, e_ = {
  isPrimitive: !1,
  // The arrow function is here to appease tree shakers like esbuild or webpack.
  // See https://esbuild.github.io/api/#tree-shaking
  match: (i) => Array.isArray(i),
  fromHexString(i, t, e = 1) {
    const n = Wi.fromHexString(i);
    t[0] = (n >> 16 & 255) / 255 * e, t[1] = (n >> 8 & 255) / 255 * e, t[2] = (n & 255) / 255 * e;
  },
  toHexString([i, t, e], n = 1) {
    n = 255 / n;
    const s = i * n << 16 ^ t * n << 8 ^ e * n << 0;
    return Wi.toHexString(s);
  }
}, n_ = {
  isPrimitive: !1,
  match: (i) => Object(i) === i,
  fromHexString(i, t, e = 1) {
    const n = Wi.fromHexString(i);
    t.r = (n >> 16 & 255) / 255 * e, t.g = (n >> 8 & 255) / 255 * e, t.b = (n & 255) / 255 * e;
  },
  toHexString({ r: i, g: t, b: e }, n = 1) {
    n = 255 / n;
    const s = i * n << 16 ^ t * n << 8 ^ e * n << 0;
    return Wi.toHexString(s);
  }
}, i_ = [t_, Wi, e_, n_];
function s_(i) {
  return i_.find((t) => t.match(i));
}
class r_ extends on {
  constructor(t, e, n, s) {
    super(t, e, n, "color"), this.$input = document.createElement("input"), this.$input.setAttribute("type", "color"), this.$input.setAttribute("tabindex", -1), this.$input.setAttribute("aria-labelledby", this.$name.id), this.$text = document.createElement("input"), this.$text.setAttribute("type", "text"), this.$text.setAttribute("spellcheck", "false"), this.$text.setAttribute("aria-labelledby", this.$name.id), this.$display = document.createElement("div"), this.$display.classList.add("display"), this.$display.appendChild(this.$input), this.$widget.appendChild(this.$display), this.$widget.appendChild(this.$text), this._format = s_(this.initialValue), this._rgbScale = s, this._initialValueHexString = this.save(), this._textFocused = !1, this.$input.addEventListener("input", () => {
      this._setValueFromHexString(this.$input.value);
    }), this.$input.addEventListener("blur", () => {
      this._callOnFinishChange();
    }), this.$text.addEventListener("input", () => {
      const r = da(this.$text.value);
      r && this._setValueFromHexString(r);
    }), this.$text.addEventListener("focus", () => {
      this._textFocused = !0, this.$text.select();
    }), this.$text.addEventListener("blur", () => {
      this._textFocused = !1, this.updateDisplay(), this._callOnFinishChange();
    }), this.$disable = this.$text, this.updateDisplay();
  }
  reset() {
    return this._setValueFromHexString(this._initialValueHexString), this;
  }
  _setValueFromHexString(t) {
    if (this._format.isPrimitive) {
      const e = this._format.fromHexString(t);
      this.setValue(e);
    } else
      this._format.fromHexString(t, this.getValue(), this._rgbScale), this._callOnChange(), this.updateDisplay();
  }
  save() {
    return this._format.toHexString(this.getValue(), this._rgbScale);
  }
  load(t) {
    return this._setValueFromHexString(t), this._callOnFinishChange(), this;
  }
  updateDisplay() {
    return this.$input.value = this._format.toHexString(this.getValue(), this._rgbScale), this._textFocused || (this.$text.value = this.$input.value.substring(1)), this.$display.style.backgroundColor = this.$input.value, this;
  }
}
class Er extends on {
  constructor(t, e, n) {
    super(t, e, n, "function"), this.$button = document.createElement("button"), this.$button.appendChild(this.$name), this.$widget.appendChild(this.$button), this.$button.addEventListener("click", (s) => {
      s.preventDefault(), this.getValue().call(this.object), this._callOnChange();
    }), this.$button.addEventListener("touchstart", () => {
    }, { passive: !0 }), this.$disable = this.$button;
  }
}
class a_ extends on {
  constructor(t, e, n, s, r, a) {
    super(t, e, n, "number"), this._initInput(), this.min(s), this.max(r);
    const o = a !== void 0;
    this.step(o ? a : this._getImplicitStep(), o), this.updateDisplay();
  }
  decimals(t) {
    return this._decimals = t, this.updateDisplay(), this;
  }
  min(t) {
    return this._min = t, this._onUpdateMinMax(), this;
  }
  max(t) {
    return this._max = t, this._onUpdateMinMax(), this;
  }
  step(t, e = !0) {
    return this._step = t, this._stepExplicit = e, this;
  }
  updateDisplay() {
    const t = this.getValue();
    if (this._hasSlider) {
      let e = (t - this._min) / (this._max - this._min);
      e = Math.max(0, Math.min(e, 1)), this.$fill.style.width = e * 100 + "%";
    }
    return this._inputFocused || (this.$input.value = this._decimals === void 0 ? t : t.toFixed(this._decimals)), this;
  }
  _initInput() {
    this.$input = document.createElement("input"), this.$input.setAttribute("type", "text"), this.$input.setAttribute("aria-labelledby", this.$name.id), window.matchMedia("(pointer: coarse)").matches && (this.$input.setAttribute("type", "number"), this.$input.setAttribute("step", "any")), this.$widget.appendChild(this.$input), this.$disable = this.$input;
    const e = () => {
      let E = parseFloat(this.$input.value);
      isNaN(E) || (this._stepExplicit && (E = this._snap(E)), this.setValue(this._clamp(E)));
    }, n = (E) => {
      const b = parseFloat(this.$input.value);
      isNaN(b) || (this._snapClampSetValue(b + E), this.$input.value = this.getValue());
    }, s = (E) => {
      E.key === "Enter" && this.$input.blur(), E.code === "ArrowUp" && (E.preventDefault(), n(this._step * this._arrowKeyMultiplier(E))), E.code === "ArrowDown" && (E.preventDefault(), n(this._step * this._arrowKeyMultiplier(E) * -1));
    }, r = (E) => {
      this._inputFocused && (E.preventDefault(), n(this._step * this._normalizeMouseWheel(E)));
    };
    let a = !1, o, l, c, u, d;
    const f = 5, m = (E) => {
      o = E.clientX, l = c = E.clientY, a = !0, u = this.getValue(), d = 0, window.addEventListener("mousemove", g), window.addEventListener("mouseup", v);
    }, g = (E) => {
      if (a) {
        const b = E.clientX - o, S = E.clientY - l;
        Math.abs(S) > f ? (E.preventDefault(), this.$input.blur(), a = !1, this._setDraggingStyle(!0, "vertical")) : Math.abs(b) > f && v();
      }
      if (!a) {
        const b = E.clientY - c;
        d -= b * this._step * this._arrowKeyMultiplier(E), u + d > this._max ? d = this._max - u : u + d < this._min && (d = this._min - u), this._snapClampSetValue(u + d);
      }
      c = E.clientY;
    }, v = () => {
      this._setDraggingStyle(!1, "vertical"), this._callOnFinishChange(), window.removeEventListener("mousemove", g), window.removeEventListener("mouseup", v);
    }, p = () => {
      this._inputFocused = !0;
    }, h = () => {
      this._inputFocused = !1, this.updateDisplay(), this._callOnFinishChange();
    };
    this.$input.addEventListener("input", e), this.$input.addEventListener("keydown", s), this.$input.addEventListener("wheel", r, { passive: !1 }), this.$input.addEventListener("mousedown", m), this.$input.addEventListener("focus", p), this.$input.addEventListener("blur", h);
  }
  _initSlider() {
    this._hasSlider = !0, this.$slider = document.createElement("div"), this.$slider.classList.add("slider"), this.$fill = document.createElement("div"), this.$fill.classList.add("fill"), this.$slider.appendChild(this.$fill), this.$widget.insertBefore(this.$slider, this.$input), this.domElement.classList.add("hasSlider");
    const t = (h, E, b, S, L) => (h - E) / (b - E) * (L - S) + S, e = (h) => {
      const E = this.$slider.getBoundingClientRect();
      let b = t(h, E.left, E.right, this._min, this._max);
      this._snapClampSetValue(b);
    }, n = (h) => {
      this._setDraggingStyle(!0), e(h.clientX), window.addEventListener("mousemove", s), window.addEventListener("mouseup", r);
    }, s = (h) => {
      e(h.clientX);
    }, r = () => {
      this._callOnFinishChange(), this._setDraggingStyle(!1), window.removeEventListener("mousemove", s), window.removeEventListener("mouseup", r);
    };
    let a = !1, o, l;
    const c = (h) => {
      h.preventDefault(), this._setDraggingStyle(!0), e(h.touches[0].clientX), a = !1;
    }, u = (h) => {
      h.touches.length > 1 || (this._hasScrollBar ? (o = h.touches[0].clientX, l = h.touches[0].clientY, a = !0) : c(h), window.addEventListener("touchmove", d, { passive: !1 }), window.addEventListener("touchend", f));
    }, d = (h) => {
      if (a) {
        const E = h.touches[0].clientX - o, b = h.touches[0].clientY - l;
        Math.abs(E) > Math.abs(b) ? c(h) : (window.removeEventListener("touchmove", d), window.removeEventListener("touchend", f));
      } else
        h.preventDefault(), e(h.touches[0].clientX);
    }, f = () => {
      this._callOnFinishChange(), this._setDraggingStyle(!1), window.removeEventListener("touchmove", d), window.removeEventListener("touchend", f);
    }, m = this._callOnFinishChange.bind(this), g = 400;
    let v;
    const p = (h) => {
      if (Math.abs(h.deltaX) < Math.abs(h.deltaY) && this._hasScrollBar) return;
      h.preventDefault();
      const b = this._normalizeMouseWheel(h) * this._step;
      this._snapClampSetValue(this.getValue() + b), this.$input.value = this.getValue(), clearTimeout(v), v = setTimeout(m, g);
    };
    this.$slider.addEventListener("mousedown", n), this.$slider.addEventListener("touchstart", u, { passive: !1 }), this.$slider.addEventListener("wheel", p, { passive: !1 });
  }
  _setDraggingStyle(t, e = "horizontal") {
    this.$slider && this.$slider.classList.toggle("active", t), document.body.classList.toggle("lil-gui-dragging", t), document.body.classList.toggle(`lil-gui-${e}`, t);
  }
  _getImplicitStep() {
    return this._hasMin && this._hasMax ? (this._max - this._min) / 1e3 : 0.1;
  }
  _onUpdateMinMax() {
    !this._hasSlider && this._hasMin && this._hasMax && (this._stepExplicit || this.step(this._getImplicitStep(), !1), this._initSlider(), this.updateDisplay());
  }
  _normalizeMouseWheel(t) {
    let { deltaX: e, deltaY: n } = t;
    return Math.floor(t.deltaY) !== t.deltaY && t.wheelDelta && (e = 0, n = -t.wheelDelta / 120, n *= this._stepExplicit ? 1 : 10), e + -n;
  }
  _arrowKeyMultiplier(t) {
    let e = this._stepExplicit ? 1 : 10;
    return t.shiftKey ? e *= 10 : t.altKey && (e /= 10), e;
  }
  _snap(t) {
    let e = 0;
    return this._hasMin ? e = this._min : this._hasMax && (e = this._max), t -= e, t = Math.round(t / this._step) * this._step, t += e, t = parseFloat(t.toPrecision(15)), t;
  }
  _clamp(t) {
    return t < this._min && (t = this._min), t > this._max && (t = this._max), t;
  }
  _snapClampSetValue(t) {
    this.setValue(this._clamp(this._snap(t)));
  }
  get _hasScrollBar() {
    const t = this.parent.root.$children;
    return t.scrollHeight > t.clientHeight;
  }
  get _hasMin() {
    return this._min !== void 0;
  }
  get _hasMax() {
    return this._max !== void 0;
  }
}
class o_ extends on {
  constructor(t, e, n, s) {
    super(t, e, n, "option"), this.$select = document.createElement("select"), this.$select.setAttribute("aria-labelledby", this.$name.id), this.$display = document.createElement("div"), this.$display.classList.add("display"), this.$select.addEventListener("change", () => {
      this.setValue(this._values[this.$select.selectedIndex]), this._callOnFinishChange();
    }), this.$select.addEventListener("focus", () => {
      this.$display.classList.add("focus");
    }), this.$select.addEventListener("blur", () => {
      this.$display.classList.remove("focus");
    }), this.$widget.appendChild(this.$select), this.$widget.appendChild(this.$display), this.$disable = this.$select, this.options(s);
  }
  options(t) {
    return this._values = Array.isArray(t) ? t : Object.values(t), this._names = Array.isArray(t) ? t : Object.keys(t), this.$select.replaceChildren(), this._names.forEach((e) => {
      const n = document.createElement("option");
      n.textContent = e, this.$select.appendChild(n);
    }), this.updateDisplay(), this;
  }
  updateDisplay() {
    const t = this.getValue(), e = this._values.indexOf(t);
    return this.$select.selectedIndex = e, this.$display.textContent = e === -1 ? t : this._names[e], this;
  }
}
class l_ extends on {
  constructor(t, e, n) {
    super(t, e, n, "string"), this.$input = document.createElement("input"), this.$input.setAttribute("type", "text"), this.$input.setAttribute("spellcheck", "false"), this.$input.setAttribute("aria-labelledby", this.$name.id), this.$input.addEventListener("input", () => {
      this.setValue(this.$input.value);
    }), this.$input.addEventListener("keydown", (s) => {
      s.code === "Enter" && this.$input.blur();
    }), this.$input.addEventListener("blur", () => {
      this._callOnFinishChange();
    }), this.$widget.appendChild(this.$input), this.$disable = this.$input, this.updateDisplay();
  }
  updateDisplay() {
    return this.$input.value = this.getValue(), this;
  }
}
var c_ = `.lil-gui {
  font-family: var(--font-family);
  font-size: var(--font-size);
  line-height: 1;
  font-weight: normal;
  font-style: normal;
  text-align: left;
  color: var(--text-color);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  --background-color: #1f1f1f;
  --text-color: #ebebeb;
  --title-background-color: #111111;
  --title-text-color: #ebebeb;
  --widget-color: #424242;
  --hover-color: #4f4f4f;
  --focus-color: #595959;
  --number-color: #2cc9ff;
  --string-color: #a2db3c;
  --font-size: 11px;
  --input-font-size: 11px;
  --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  --font-family-mono: Menlo, Monaco, Consolas, "Droid Sans Mono", monospace;
  --padding: 4px;
  --spacing: 4px;
  --widget-height: 20px;
  --title-height: calc(var(--widget-height) + var(--spacing) * 1.25);
  --name-width: 45%;
  --slider-knob-width: 2px;
  --slider-input-width: 27%;
  --color-input-width: 27%;
  --slider-input-min-width: 45px;
  --color-input-min-width: 45px;
  --folder-indent: 7px;
  --widget-padding: 0 0 0 3px;
  --widget-border-radius: 2px;
  --checkbox-size: calc(0.75 * var(--widget-height));
  --scrollbar-width: 5px;
}
.lil-gui, .lil-gui * {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
.lil-gui.root {
  width: var(--width, 245px);
  display: flex;
  flex-direction: column;
  background: var(--background-color);
}
.lil-gui.root > .title {
  background: var(--title-background-color);
  color: var(--title-text-color);
}
.lil-gui.root > .children {
  overflow-x: hidden;
  overflow-y: auto;
}
.lil-gui.root > .children::-webkit-scrollbar {
  width: var(--scrollbar-width);
  height: var(--scrollbar-width);
  background: var(--background-color);
}
.lil-gui.root > .children::-webkit-scrollbar-thumb {
  border-radius: var(--scrollbar-width);
  background: var(--focus-color);
}
@media (pointer: coarse) {
  .lil-gui.allow-touch-styles, .lil-gui.allow-touch-styles .lil-gui {
    --widget-height: 28px;
    --padding: 6px;
    --spacing: 6px;
    --font-size: 13px;
    --input-font-size: 16px;
    --folder-indent: 10px;
    --scrollbar-width: 7px;
    --slider-input-min-width: 50px;
    --color-input-min-width: 65px;
  }
}
.lil-gui.force-touch-styles, .lil-gui.force-touch-styles .lil-gui {
  --widget-height: 28px;
  --padding: 6px;
  --spacing: 6px;
  --font-size: 13px;
  --input-font-size: 16px;
  --folder-indent: 10px;
  --scrollbar-width: 7px;
  --slider-input-min-width: 50px;
  --color-input-min-width: 65px;
}
.lil-gui.autoPlace {
  max-height: 100%;
  position: fixed;
  top: 0;
  right: 15px;
  z-index: 1001;
}

.lil-gui .controller {
  display: flex;
  align-items: center;
  padding: 0 var(--padding);
  margin: var(--spacing) 0;
}
.lil-gui .controller.disabled {
  opacity: 0.5;
}
.lil-gui .controller.disabled, .lil-gui .controller.disabled * {
  pointer-events: none !important;
}
.lil-gui .controller > .name {
  min-width: var(--name-width);
  flex-shrink: 0;
  white-space: pre;
  padding-right: var(--spacing);
  line-height: var(--widget-height);
}
.lil-gui .controller .widget {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
  min-height: var(--widget-height);
}
.lil-gui .controller.string input {
  color: var(--string-color);
}
.lil-gui .controller.boolean {
  cursor: pointer;
}
.lil-gui .controller.color .display {
  width: 100%;
  height: var(--widget-height);
  border-radius: var(--widget-border-radius);
  position: relative;
}
@media (hover: hover) {
  .lil-gui .controller.color .display:hover:before {
    content: " ";
    display: block;
    position: absolute;
    border-radius: var(--widget-border-radius);
    border: 1px solid #fff9;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
  }
}
.lil-gui .controller.color input[type=color] {
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}
.lil-gui .controller.color input[type=text] {
  margin-left: var(--spacing);
  font-family: var(--font-family-mono);
  min-width: var(--color-input-min-width);
  width: var(--color-input-width);
  flex-shrink: 0;
}
.lil-gui .controller.option select {
  opacity: 0;
  position: absolute;
  width: 100%;
  max-width: 100%;
}
.lil-gui .controller.option .display {
  position: relative;
  pointer-events: none;
  border-radius: var(--widget-border-radius);
  height: var(--widget-height);
  line-height: var(--widget-height);
  max-width: 100%;
  overflow: hidden;
  word-break: break-all;
  padding-left: 0.55em;
  padding-right: 1.75em;
  background: var(--widget-color);
}
@media (hover: hover) {
  .lil-gui .controller.option .display.focus {
    background: var(--focus-color);
  }
}
.lil-gui .controller.option .display.active {
  background: var(--focus-color);
}
.lil-gui .controller.option .display:after {
  font-family: "lil-gui";
  content: "↕";
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  padding-right: 0.375em;
}
.lil-gui .controller.option .widget,
.lil-gui .controller.option select {
  cursor: pointer;
}
@media (hover: hover) {
  .lil-gui .controller.option .widget:hover .display {
    background: var(--hover-color);
  }
}
.lil-gui .controller.number input {
  color: var(--number-color);
}
.lil-gui .controller.number.hasSlider input {
  margin-left: var(--spacing);
  width: var(--slider-input-width);
  min-width: var(--slider-input-min-width);
  flex-shrink: 0;
}
.lil-gui .controller.number .slider {
  width: 100%;
  height: var(--widget-height);
  background: var(--widget-color);
  border-radius: var(--widget-border-radius);
  padding-right: var(--slider-knob-width);
  overflow: hidden;
  cursor: ew-resize;
  touch-action: pan-y;
}
@media (hover: hover) {
  .lil-gui .controller.number .slider:hover {
    background: var(--hover-color);
  }
}
.lil-gui .controller.number .slider.active {
  background: var(--focus-color);
}
.lil-gui .controller.number .slider.active .fill {
  opacity: 0.95;
}
.lil-gui .controller.number .fill {
  height: 100%;
  border-right: var(--slider-knob-width) solid var(--number-color);
  box-sizing: content-box;
}

.lil-gui-dragging .lil-gui {
  --hover-color: var(--widget-color);
}
.lil-gui-dragging * {
  cursor: ew-resize !important;
}

.lil-gui-dragging.lil-gui-vertical * {
  cursor: ns-resize !important;
}

.lil-gui .title {
  height: var(--title-height);
  font-weight: 600;
  padding: 0 var(--padding);
  width: 100%;
  text-align: left;
  background: none;
  text-decoration-skip: objects;
}
.lil-gui .title:before {
  font-family: "lil-gui";
  content: "▾";
  padding-right: 2px;
  display: inline-block;
}
.lil-gui .title:active {
  background: var(--title-background-color);
  opacity: 0.75;
}
@media (hover: hover) {
  body:not(.lil-gui-dragging) .lil-gui .title:hover {
    background: var(--title-background-color);
    opacity: 0.85;
  }
  .lil-gui .title:focus {
    text-decoration: underline var(--focus-color);
  }
}
.lil-gui.root > .title:focus {
  text-decoration: none !important;
}
.lil-gui.closed > .title:before {
  content: "▸";
}
.lil-gui.closed > .children {
  transform: translateY(-7px);
  opacity: 0;
}
.lil-gui.closed:not(.transition) > .children {
  display: none;
}
.lil-gui.transition > .children {
  transition-duration: 300ms;
  transition-property: height, opacity, transform;
  transition-timing-function: cubic-bezier(0.2, 0.6, 0.35, 1);
  overflow: hidden;
  pointer-events: none;
}
.lil-gui .children:empty:before {
  content: "Empty";
  padding: 0 var(--padding);
  margin: var(--spacing) 0;
  display: block;
  height: var(--widget-height);
  font-style: italic;
  line-height: var(--widget-height);
  opacity: 0.5;
}
.lil-gui.root > .children > .lil-gui > .title {
  border: 0 solid var(--widget-color);
  border-width: 1px 0;
  transition: border-color 300ms;
}
.lil-gui.root > .children > .lil-gui.closed > .title {
  border-bottom-color: transparent;
}
.lil-gui + .controller {
  border-top: 1px solid var(--widget-color);
  margin-top: 0;
  padding-top: var(--spacing);
}
.lil-gui .lil-gui .lil-gui > .title {
  border: none;
}
.lil-gui .lil-gui .lil-gui > .children {
  border: none;
  margin-left: var(--folder-indent);
  border-left: 2px solid var(--widget-color);
}
.lil-gui .lil-gui .controller {
  border: none;
}

.lil-gui label, .lil-gui input, .lil-gui button {
  -webkit-tap-highlight-color: transparent;
}
.lil-gui input {
  border: 0;
  outline: none;
  font-family: var(--font-family);
  font-size: var(--input-font-size);
  border-radius: var(--widget-border-radius);
  height: var(--widget-height);
  background: var(--widget-color);
  color: var(--text-color);
  width: 100%;
}
@media (hover: hover) {
  .lil-gui input:hover {
    background: var(--hover-color);
  }
  .lil-gui input:active {
    background: var(--focus-color);
  }
}
.lil-gui input:disabled {
  opacity: 1;
}
.lil-gui input[type=text],
.lil-gui input[type=number] {
  padding: var(--widget-padding);
  -moz-appearance: textfield;
}
.lil-gui input[type=text]:focus,
.lil-gui input[type=number]:focus {
  background: var(--focus-color);
}
.lil-gui input[type=checkbox] {
  appearance: none;
  width: var(--checkbox-size);
  height: var(--checkbox-size);
  border-radius: var(--widget-border-radius);
  text-align: center;
  cursor: pointer;
}
.lil-gui input[type=checkbox]:checked:before {
  font-family: "lil-gui";
  content: "✓";
  font-size: var(--checkbox-size);
  line-height: var(--checkbox-size);
}
@media (hover: hover) {
  .lil-gui input[type=checkbox]:focus {
    box-shadow: inset 0 0 0 1px var(--focus-color);
  }
}
.lil-gui button {
  outline: none;
  cursor: pointer;
  font-family: var(--font-family);
  font-size: var(--font-size);
  color: var(--text-color);
  width: 100%;
  border: none;
}
.lil-gui .controller button {
  height: var(--widget-height);
  text-transform: none;
  background: var(--widget-color);
  border-radius: var(--widget-border-radius);
}
@media (hover: hover) {
  .lil-gui .controller button:hover {
    background: var(--hover-color);
  }
  .lil-gui .controller button:focus {
    box-shadow: inset 0 0 0 1px var(--focus-color);
  }
}
.lil-gui .controller button:active {
  background: var(--focus-color);
}

@font-face {
  font-family: "lil-gui";
  src: url("data:application/font-woff;charset=utf-8;base64,d09GRgABAAAAAAUsAAsAAAAACJwAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAABHU1VCAAABCAAAAH4AAADAImwmYE9TLzIAAAGIAAAAPwAAAGBKqH5SY21hcAAAAcgAAAD0AAACrukyyJBnbHlmAAACvAAAAF8AAACEIZpWH2hlYWQAAAMcAAAAJwAAADZfcj2zaGhlYQAAA0QAAAAYAAAAJAC5AHhobXR4AAADXAAAABAAAABMAZAAAGxvY2EAAANsAAAAFAAAACgCEgIybWF4cAAAA4AAAAAeAAAAIAEfABJuYW1lAAADoAAAASIAAAIK9SUU/XBvc3QAAATEAAAAZgAAAJCTcMc2eJxVjbEOgjAURU+hFRBK1dGRL+ALnAiToyMLEzFpnPz/eAshwSa97517c/MwwJmeB9kwPl+0cf5+uGPZXsqPu4nvZabcSZldZ6kfyWnomFY/eScKqZNWupKJO6kXN3K9uCVoL7iInPr1X5baXs3tjuMqCtzEuagm/AAlzQgPAAB4nGNgYRBlnMDAysDAYM/gBiT5oLQBAwuDJAMDEwMrMwNWEJDmmsJwgCFeXZghBcjlZMgFCzOiKOIFAB71Bb8AeJy1kjFuwkAQRZ+DwRAwBtNQRUGKQ8OdKCAWUhAgKLhIuAsVSpWz5Bbkj3dEgYiUIszqWdpZe+Z7/wB1oCYmIoboiwiLT2WjKl/jscrHfGg/pKdMkyklC5Zs2LEfHYpjcRoPzme9MWWmk3dWbK9ObkWkikOetJ554fWyoEsmdSlt+uR0pCJR34b6t/TVg1SY3sYvdf8vuiKrpyaDXDISiegp17p7579Gp3p++y7HPAiY9pmTibljrr85qSidtlg4+l25GLCaS8e6rRxNBmsnERunKbaOObRz7N72ju5vdAjYpBXHgJylOAVsMseDAPEP8LYoUHicY2BiAAEfhiAGJgZWBgZ7RnFRdnVJELCQlBSRlATJMoLV2DK4glSYs6ubq5vbKrJLSbGrgEmovDuDJVhe3VzcXFwNLCOILB/C4IuQ1xTn5FPilBTj5FPmBAB4WwoqAHicY2BkYGAA4sk1sR/j+W2+MnAzpDBgAyEMQUCSg4EJxAEAwUgFHgB4nGNgZGBgSGFggJMhDIwMqEAYAByHATJ4nGNgAIIUNEwmAABl3AGReJxjYAACIQYlBiMGJ3wQAEcQBEV4nGNgZGBgEGZgY2BiAAEQyQWEDAz/wXwGAAsPATIAAHicXdBNSsNAHAXwl35iA0UQXYnMShfS9GPZA7T7LgIu03SSpkwzYTIt1BN4Ak/gKTyAeCxfw39jZkjymzcvAwmAW/wgwHUEGDb36+jQQ3GXGot79L24jxCP4gHzF/EIr4jEIe7wxhOC3g2TMYy4Q7+Lu/SHuEd/ivt4wJd4wPxbPEKMX3GI5+DJFGaSn4qNzk8mcbKSR6xdXdhSzaOZJGtdapd4vVPbi6rP+cL7TGXOHtXKll4bY1Xl7EGnPtp7Xy2n00zyKLVHfkHBa4IcJ2oD3cgggWvt/V/FbDrUlEUJhTn/0azVWbNTNr0Ens8de1tceK9xZmfB1CPjOmPH4kitmvOubcNpmVTN3oFJyjzCvnmrwhJTzqzVj9jiSX911FjeAAB4nG3HMRKCMBBA0f0giiKi4DU8k0V2GWbIZDOh4PoWWvq6J5V8If9NVNQcaDhyouXMhY4rPTcG7jwYmXhKq8Wz+p762aNaeYXom2n3m2dLTVgsrCgFJ7OTmIkYbwIbC6vIB7WmFfAAAA==") format("woff");
}`;
function h_(i) {
  const t = document.createElement("style");
  t.innerHTML = i;
  const e = document.querySelector("head link[rel=stylesheet], head style");
  e ? document.head.insertBefore(t, e) : document.head.appendChild(t);
}
let Qo = !1;
class Da {
  /**
   * Creates a panel that holds controllers.
   * @example
   * new GUI();
   * new GUI( { container: document.getElementById( 'custom' ) } );
   *
   * @param {object} [options]
   * @param {boolean} [options.autoPlace=true]
   * Adds the GUI to `document.body` and fixes it to the top right of the page.
   *
   * @param {HTMLElement} [options.container]
   * Adds the GUI to this DOM element. Overrides `autoPlace`.
   *
   * @param {number} [options.width=245]
   * Width of the GUI in pixels, usually set when name labels become too long. Note that you can make
   * name labels wider in CSS with `.lil‑gui { ‑‑name‑width: 55% }`.
   *
   * @param {string} [options.title=Controls]
   * Name to display in the title bar.
   *
   * @param {boolean} [options.closeFolders=false]
   * Pass `true` to close all folders in this GUI by default.
   *
   * @param {boolean} [options.injectStyles=true]
   * Injects the default stylesheet into the page if this is the first GUI.
   * Pass `false` to use your own stylesheet.
   *
   * @param {number} [options.touchStyles=true]
   * Makes controllers larger on touch devices. Pass `false` to disable touch styles.
   *
   * @param {GUI} [options.parent]
   * Adds this GUI as a child in another GUI. Usually this is done for you by `addFolder()`.
   */
  constructor({
    parent: t,
    autoPlace: e = t === void 0,
    container: n,
    width: s,
    title: r = "Controls",
    closeFolders: a = !1,
    injectStyles: o = !0,
    touchStyles: l = !0
  } = {}) {
    if (this.parent = t, this.root = t ? t.root : this, this.children = [], this.controllers = [], this.folders = [], this._closed = !1, this._hidden = !1, this.domElement = document.createElement("div"), this.domElement.classList.add("lil-gui"), this.$title = document.createElement("button"), this.$title.classList.add("title"), this.$title.setAttribute("aria-expanded", !0), this.$title.addEventListener("click", () => this.openAnimated(this._closed)), this.$title.addEventListener("touchstart", () => {
    }, { passive: !0 }), this.$children = document.createElement("div"), this.$children.classList.add("children"), this.domElement.appendChild(this.$title), this.domElement.appendChild(this.$children), this.title(r), this.parent) {
      this.parent.children.push(this), this.parent.folders.push(this), this.parent.$children.appendChild(this.domElement);
      return;
    }
    this.domElement.classList.add("root"), l && this.domElement.classList.add("allow-touch-styles"), !Qo && o && (h_(c_), Qo = !0), n ? n.appendChild(this.domElement) : e && (this.domElement.classList.add("autoPlace"), document.body.appendChild(this.domElement)), s && this.domElement.style.setProperty("--width", s + "px"), this._closeFolders = a;
  }
  /**
   * Adds a controller to the GUI, inferring controller type using the `typeof` operator.
   * @example
   * gui.add( object, 'property' );
   * gui.add( object, 'number', 0, 100, 1 );
   * gui.add( object, 'options', [ 1, 2, 3 ] );
   *
   * @param {object} object The object the controller will modify.
   * @param {string} property Name of the property to control.
   * @param {number|object|Array} [$1] Minimum value for number controllers, or the set of
   * selectable values for a dropdown.
   * @param {number} [max] Maximum value for number controllers.
   * @param {number} [step] Step value for number controllers.
   * @returns {Controller}
   */
  add(t, e, n, s, r) {
    if (Object(n) === n)
      return new o_(this, t, e, n);
    const a = t[e];
    switch (typeof a) {
      case "number":
        return new a_(this, t, e, n, s, r);
      case "boolean":
        return new Qm(this, t, e);
      case "string":
        return new l_(this, t, e);
      case "function":
        return new Er(this, t, e);
    }
    console.error(`gui.add failed
	property:`, e, `
	object:`, t, `
	value:`, a);
  }
  /**
   * Adds a color controller to the GUI.
   * @example
   * params = {
   * 	cssColor: '#ff00ff',
   * 	rgbColor: { r: 0, g: 0.2, b: 0.4 },
   * 	customRange: [ 0, 127, 255 ],
   * };
   *
   * gui.addColor( params, 'cssColor' );
   * gui.addColor( params, 'rgbColor' );
   * gui.addColor( params, 'customRange', 255 );
   *
   * @param {object} object The object the controller will modify.
   * @param {string} property Name of the property to control.
   * @param {number} rgbScale Maximum value for a color channel when using an RGB color. You may
   * need to set this to 255 if your colors are too bright.
   * @returns {Controller}
   */
  addColor(t, e, n = 1) {
    return new r_(this, t, e, n);
  }
  /**
   * Adds a folder to the GUI, which is just another GUI. This method returns
   * the nested GUI so you can add controllers to it.
   * @example
   * const folder = gui.addFolder( 'Position' );
   * folder.add( position, 'x' );
   * folder.add( position, 'y' );
   * folder.add( position, 'z' );
   *
   * @param {string} title Name to display in the folder's title bar.
   * @returns {GUI}
   */
  addFolder(t) {
    const e = new Da({ parent: this, title: t });
    return this.root._closeFolders && e.close(), e;
  }
  /**
   * Recalls values that were saved with `gui.save()`.
   * @param {object} obj
   * @param {boolean} recursive Pass false to exclude folders descending from this GUI.
   * @returns {this}
   */
  load(t, e = !0) {
    return t.controllers && this.controllers.forEach((n) => {
      n instanceof Er || n._name in t.controllers && n.load(t.controllers[n._name]);
    }), e && t.folders && this.folders.forEach((n) => {
      n._title in t.folders && n.load(t.folders[n._title]);
    }), this;
  }
  /**
   * Returns an object mapping controller names to values. The object can be passed to `gui.load()` to
   * recall these values.
   * @example
   * {
   * 	controllers: {
   * 		prop1: 1,
   * 		prop2: 'value',
   * 		...
   * 	},
   * 	folders: {
   * 		folderName1: { controllers, folders },
   * 		folderName2: { controllers, folders }
   * 		...
   * 	}
   * }
   *
   * @param {boolean} recursive Pass false to exclude folders descending from this GUI.
   * @returns {object}
   */
  save(t = !0) {
    const e = {
      controllers: {},
      folders: {}
    };
    return this.controllers.forEach((n) => {
      if (!(n instanceof Er)) {
        if (n._name in e.controllers)
          throw new Error(`Cannot save GUI with duplicate property "${n._name}"`);
        e.controllers[n._name] = n.save();
      }
    }), t && this.folders.forEach((n) => {
      if (n._title in e.folders)
        throw new Error(`Cannot save GUI with duplicate folder "${n._title}"`);
      e.folders[n._title] = n.save();
    }), e;
  }
  /**
   * Opens a GUI or folder. GUI and folders are open by default.
   * @param {boolean} open Pass false to close.
   * @returns {this}
   * @example
   * gui.open(); // open
   * gui.open( false ); // close
   * gui.open( gui._closed ); // toggle
   */
  open(t = !0) {
    return this._setClosed(!t), this.$title.setAttribute("aria-expanded", !this._closed), this.domElement.classList.toggle("closed", this._closed), this;
  }
  /**
   * Closes the GUI.
   * @returns {this}
   */
  close() {
    return this.open(!1);
  }
  _setClosed(t) {
    this._closed !== t && (this._closed = t, this._callOnOpenClose(this));
  }
  /**
   * Shows the GUI after it's been hidden.
   * @param {boolean} show
   * @returns {this}
   * @example
   * gui.show();
   * gui.show( false ); // hide
   * gui.show( gui._hidden ); // toggle
   */
  show(t = !0) {
    return this._hidden = !t, this.domElement.style.display = this._hidden ? "none" : "", this;
  }
  /**
   * Hides the GUI.
   * @returns {this}
   */
  hide() {
    return this.show(!1);
  }
  openAnimated(t = !0) {
    return this._setClosed(!t), this.$title.setAttribute("aria-expanded", !this._closed), requestAnimationFrame(() => {
      const e = this.$children.clientHeight;
      this.$children.style.height = e + "px", this.domElement.classList.add("transition");
      const n = (r) => {
        r.target === this.$children && (this.$children.style.height = "", this.domElement.classList.remove("transition"), this.$children.removeEventListener("transitionend", n));
      };
      this.$children.addEventListener("transitionend", n);
      const s = t ? this.$children.scrollHeight : 0;
      this.domElement.classList.toggle("closed", !t), requestAnimationFrame(() => {
        this.$children.style.height = s + "px";
      });
    }), this;
  }
  /**
   * Change the title of this GUI.
   * @param {string} title
   * @returns {this}
   */
  title(t) {
    return this._title = t, this.$title.textContent = t, this;
  }
  /**
   * Resets all controllers to their initial values.
   * @param {boolean} recursive Pass false to exclude folders descending from this GUI.
   * @returns {this}
   */
  reset(t = !0) {
    return (t ? this.controllersRecursive() : this.controllers).forEach((n) => n.reset()), this;
  }
  /**
   * Pass a function to be called whenever a controller in this GUI changes.
   * @param {function({object:object, property:string, value:any, controller:Controller})} callback
   * @returns {this}
   * @example
   * gui.onChange( event => {
   * 	event.object     // object that was modified
   * 	event.property   // string, name of property
   * 	event.value      // new value of controller
   * 	event.controller // controller that was modified
   * } );
   */
  onChange(t) {
    return this._onChange = t, this;
  }
  _callOnChange(t) {
    this.parent && this.parent._callOnChange(t), this._onChange !== void 0 && this._onChange.call(this, {
      object: t.object,
      property: t.property,
      value: t.getValue(),
      controller: t
    });
  }
  /**
   * Pass a function to be called whenever a controller in this GUI has finished changing.
   * @param {function({object:object, property:string, value:any, controller:Controller})} callback
   * @returns {this}
   * @example
   * gui.onFinishChange( event => {
   * 	event.object     // object that was modified
   * 	event.property   // string, name of property
   * 	event.value      // new value of controller
   * 	event.controller // controller that was modified
   * } );
   */
  onFinishChange(t) {
    return this._onFinishChange = t, this;
  }
  _callOnFinishChange(t) {
    this.parent && this.parent._callOnFinishChange(t), this._onFinishChange !== void 0 && this._onFinishChange.call(this, {
      object: t.object,
      property: t.property,
      value: t.getValue(),
      controller: t
    });
  }
  /**
   * Pass a function to be called when this GUI or its descendants are opened or closed.
   * @param {function(GUI)} callback
   * @returns {this}
   * @example
   * gui.onOpenClose( changedGUI => {
   * 	console.log( changedGUI._closed );
   * } );
   */
  onOpenClose(t) {
    return this._onOpenClose = t, this;
  }
  _callOnOpenClose(t) {
    this.parent && this.parent._callOnOpenClose(t), this._onOpenClose !== void 0 && this._onOpenClose.call(this, t);
  }
  /**
   * Destroys all DOM elements and event listeners associated with this GUI.
   */
  destroy() {
    this.parent && (this.parent.children.splice(this.parent.children.indexOf(this), 1), this.parent.folders.splice(this.parent.folders.indexOf(this), 1)), this.domElement.parentElement && this.domElement.parentElement.removeChild(this.domElement), Array.from(this.children).forEach((t) => t.destroy());
  }
  /**
   * Returns an array of controllers contained by this GUI and its descendents.
   * @returns {Controller[]}
   */
  controllersRecursive() {
    let t = Array.from(this.controllers);
    return this.folders.forEach((e) => {
      t = t.concat(e.controllersRecursive());
    }), t;
  }
  /**
   * Returns an array of folders contained by this GUI and its descendents.
   * @returns {GUI[]}
   */
  foldersRecursive() {
    let t = Array.from(this.folders);
    return this.folders.forEach((e) => {
      t = t.concat(e.foldersRecursive());
    }), t;
  }
}
class u_ {
  constructor(t, e, n) {
    Qt(this, "app");
    Qt(this, "scene");
    Qt(this, "controller");
    Qt(this, "gui");
    Qt(this, "colorOptions");
    // Controllers
    Qt(this, "controllers");
    this.scene = n, this.app = t, this.gui = new Da({ container: e, autoPlace: !1 }), this.colorOptions = {
      default: { show: !0, colorDim: null, trueColor: !1 },
      hide: { show: !1, colorDim: null, trueColor: !1 }
    };
    for (const p of this.scene.variables)
      p === "wavelength" ? (this.colorOptions.wavelength = {
        show: !0,
        colorDim: "wavelength",
        trueColor: !1
      }, this.colorOptions["wavelength (true color)"] = {
        show: !0,
        colorDim: "wavelength",
        trueColor: !0
      }) : this.colorOptions[p] = {
        show: !0,
        colorDim: p,
        trueColor: !1
      };
    this.controller = {
      validColor: this.colorOptions.default,
      blockedColor: this.colorOptions.hide,
      outputColor: this.colorOptions.default,
      raysOpacity: 1,
      raysThickness: 1,
      resetView() {
        t.resetView();
      },
      backgroundColor: { r: 0, g: 0, b: 0 },
      surfacesColor: { r: 0, g: 1, b: 1 },
      showKinematicJoints: !1
    }, this.scene.variables.includes("object") && (console.log("setting obejct var"), this.controller.validColor = this.colorOptions.object, this.controller.outputColor = this.colorOptions.object), this.gui.add(this.controller, "resetView").name("Reset Camera");
    const s = this.gui.addFolder("Colors"), r = s.add(this.controller, "validColor", this.colorOptions).name("Valid rays"), a = s.add(this.controller, "blockedColor", this.colorOptions).name("Blocked rays"), o = s.add(this.controller, "outputColor", this.colorOptions).name("Output rays"), l = s.add(this.controller, "raysOpacity", 0, 1).name("Opacity").onFinishChange((p) => {
      this.scene.setRaysOpacity(p);
    }), c = s.add(this.controller, "raysThickness", 0.1, 10).name("Thickness").onFinishChange((p) => {
      this.scene.setRaysThickness(p);
    });
    r.onChange((p) => {
      this.scene.setupValidRays(p), this.scene.setRaysOpacity(l.getValue()), this.scene.setRaysThickness(c.getValue());
    }), a.onChange((p) => {
      this.scene.setupBlockedRays(p), this.scene.setRaysOpacity(l.getValue()), this.scene.setRaysThickness(c.getValue());
    }), o.onChange((p) => {
      this.scene.setupOutputRays(p), this.scene.setRaysOpacity(l.getValue()), this.scene.setRaysThickness(c.getValue());
    });
    const u = s.addColor(this.controller, "backgroundColor").name("Background").onChange((p) => {
      this.scene.scene.background = new Vt(
        p.r,
        p.g,
        p.b
      );
    }), d = s.addColor(this.controller, "surfacesColor").name("Surfaces").onChange((p) => {
      const h = new Vt(p.r, p.g, p.b);
      this.scene.setSurfacesColor(h);
    }), f = this.gui.addFolder("Visible"), m = f.add(this.scene.opticalAxis, "visible").name("Optical axis"), g = f.add(this.scene.otherAxes, "visible").name("Other axes"), v = f.add(this.controller, "showKinematicJoints").name("Kinematic joints");
    this.scene.setupValidRays(this.controller.validColor), this.scene.setupBlockedRays(this.controller.blockedColor), this.scene.setupOutputRays(this.controller.outputColor), f.onChange((p) => {
      this.updateCameraLayers();
    }), this.updateCameraLayers(), this.gui.open(!1), f.open(!1), s.open(!1), this.controllers = {
      colors: {
        validRays: r,
        blockedRays: a,
        outputRays: o,
        opacity: l,
        thickness: c,
        background: u,
        surfaces: d
      },
      visible: {
        opticalAxis: m,
        otherAxes: g,
        kinematicJoints: v
      }
    };
  }
  // Set controls state from a JSON object
  setControlsFromJson(t) {
    if (typeof t == "boolean")
      t === !1 && this.gui.hide();
    else {
      const e = function(s, r) {
        t.hasOwnProperty(s) && r(t[s]);
      }, n = this;
      e("valid_rays", (s) => {
        n.controllers.colors.validRays.load(n.colorOptions[s]);
      }), e("blocked_rays", (s) => {
        n.controllers.colors.blockedRays.load(n.colorOptions[s]);
      }), e("output_rays", (s) => {
        n.controllers.colors.outputRays.load(n.colorOptions[s]);
      }), e("opacity", (s) => {
        n.controllers.colors.opacity.load(s);
      }), e("thickness", (s) => {
        n.controllers.colors.thickness.load(s);
      }), e("show_optical_axis", (s) => {
        n.controllers.visible.opticalAxis.load(s);
      }), e("show_other_axes", (s) => {
        n.controllers.visible.otherAxes.load(s);
      }), e("show_kinematic_joints", (s) => {
        n.controllers.visible.kinematicJoints.load(s);
      });
    }
  }
  updateCameraLayers() {
    const t = this, e = function(n, s) {
      n ? t.app.camera.layers.enable(s) : t.app.camera.layers.disable(s);
    };
    this.app.camera.layers.enable(0), e(this.controller.showKinematicJoints, 4);
  }
}
const d_ = `<div class="tlmviewer-viewport"></div>

<div class="tlmviewer-title"></div>
`;
class f_ {
  constructor(t, e, n) {
    Qt(this, "scene");
    Qt(this, "renderer");
    Qt(this, "camera");
    Qt(this, "controls");
    Qt(this, "viewport");
    Qt(this, "gui");
    const s = t.getElementsByClassName("tlmviewer-viewport")[0];
    if (!(s instanceof HTMLElement))
      throw new Error("Expected viewport to be an HTMLElement");
    this.viewport = s, this.scene = e, this.renderer = new xm({ antialias: !0 });
    const r = t.getBoundingClientRect();
    if (this.renderer.setSize(r.width, r.height), this.renderer.localClippingEnabled = !0, this.viewport.appendChild(this.renderer.domElement), n === "orthographic")
      [this.camera, this.controls] = this.setupOrthographicCamera();
    else if (n == "perspective")
      [this.camera, this.controls] = this.setupPerspectiveCamera();
    else if (n === "XY")
      [this.camera, this.controls] = this.setupXYCamera();
    else
      throw new Error(`Uknown camera type '${n}'`);
    const a = t.getElementsByClassName("tlmviewer-title")[0];
    a.innerHTML = e.title, this.gui = new u_(this, t, this.scene), this.gui.updateCameraLayers(), n === "XY" && this.resetView();
  }
  // Handle window resize events
  // @ts-ignore
  onWindowResize() {
    const t = window.innerWidth / window.innerHeight;
    this.camera instanceof ze ? this.camera.aspect = t : this.camera instanceof Bi && (this.camera.left = -t * 10, this.camera.right = t * 10, this.camera.top = 10, this.camera.bottom = -10), this.camera.updateProjectionMatrix(), this.renderer.setSize(window.innerWidth, window.innerHeight);
  }
  resetView() {
    const t = this.scene.getBB(), e = this.viewport.getBoundingClientRect(), n = e.width / e.height;
    if (!(this.camera instanceof Bi)) return;
    const s = new P();
    t.getCenter(s);
    const r = new P();
    t.getSize(r);
    const a = 1.15;
    this.camera.zoom = 1, this.camera.position.set(s.x, s.y, s.z + 100), r.x > n * r.y ? (this.camera.left = a * r.x / -2, this.camera.right = a * r.x / 2, this.camera.top = a * (1 / n * r.x) / 2, this.camera.bottom = a * (1 / n * r.x) / -2) : (this.camera.left = a * (n * r.y) / -2, this.camera.right = a * (n * r.y) / 2, this.camera.top = a * r.y / 2, this.camera.bottom = a * r.y / -2), this.camera.updateProjectionMatrix(), this.controls.update(), this.controls.target = s;
  }
  // The 2D camera
  setupXYCamera() {
    const t = this.viewport.getBoundingClientRect(), e = t.width / t.height, n = new Bi(
      -e * 10,
      e * 10,
      10,
      -10,
      -1e4,
      1e4
    );
    this.camera && this.controls.dispose();
    const s = new vr(
      n,
      this.renderer.domElement
    );
    return s.enableRotate = !1, this.camera = n, this.controls = s, this.resetView(), [n, s];
  }
  setupOrthographicCamera() {
    const t = this.viewport.getBoundingClientRect(), e = t.width / t.height, n = new Bi(
      -e * 10,
      e * 10,
      10,
      -10,
      -1e4,
      1e4
    );
    n.position.set(10, 10, 10), n.lookAt(0, 0, 0), this.camera && this.controls.dispose();
    const s = new vr(
      n,
      this.renderer.domElement
    );
    return [n, s];
  }
  setupPerspectiveCamera() {
    const t = this.viewport.getBoundingClientRect(), e = t.width / t.height, n = new ze(75, e, 0.1, 1e3);
    n.position.set(10, 10, 10), n.lookAt(0, 0, 0), this.camera && this.controls.dispose();
    const s = new vr(
      n,
      this.renderer.domElement
    );
    return [n, s];
  }
  registerEventHandlers(t) {
    const e = t.querySelector("button.reset-view");
    e == null || e.addEventListener("click", () => {
      this.resetView();
    });
  }
  // Start the animation loop
  animate() {
    const t = () => {
      this.controls.update(), this.renderer.render(this.scene.scene, this.camera), requestAnimationFrame(t);
    };
    t();
  }
}
function p_(i, t) {
  const e = Zo(t, "mode", ["3D", "2D"]), n = Zo(t, "camera", [
    "orthographic",
    "perspective",
    "XY"
  ]), s = new Jm(t, e === "3D" ? 3 : 2), r = new f_(i, s, n), a = t.controls ?? null;
  return a !== null && r.gui.setControlsFromJson(a), r;
}
function Bl(i, t) {
  try {
    i.innerHTML = d_;
    const e = p_(i, t);
    e.registerEventHandlers(i), e.animate();
  } catch (e) {
    throw i.innerHTML = "<span style='color: red'>tlmviewer error: " + e + "</span>", e;
  }
}
function m_(i, t) {
  try {
    const e = JSON.parse(t);
    Bl(i, e);
  } catch (e) {
    throw i.innerHTML = "<span style='color: red'>tlmviewer error: " + e + "</span>", e;
  }
}
async function zl(i, t) {
  try {
    const n = await (await fetch(t)).json();
    Bl(i, n);
  } catch (e) {
    throw i.innerHTML = "<span style='color: red'>tlmviewer error: " + e + "</span>", e;
  }
}
async function __() {
  const i = document.querySelectorAll(".tlmviewer"), t = [];
  return i.forEach((e) => {
    const n = e.getAttribute("data-url");
    n && t.push(zl(e, n));
  }), t;
}
const v_ = {
  embed: m_,
  load: zl,
  loadAll: __
};
console.log("tlmviewer loaded");
export {
  f_ as TLMViewerApp,
  v_ as tlmviewer
};
