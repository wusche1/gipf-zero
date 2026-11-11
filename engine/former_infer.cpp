#include "former_infer.h"
#include <wasm_simd128.h>
#include <algorithm>
#include <cmath>

namespace former {
namespace {

using V = v128_t;
const int CELLS = 37, PAD = 40, PV = 32, VC = 16, VH = 128;
const float QONE = 1 << 14;
enum Act { NONE, RELU, GELU };

inline V splat(float x) { return wasm_f32x4_splat(x); }

inline V madd(V a, V b, V c) {
#ifdef __wasm_relaxed_simd__
  return wasm_f32x4_relaxed_madd(a, b, c);
#else
  return wasm_f32x4_add(wasm_f32x4_mul(a, b), c);
#endif
}

// exp(x) for x <= 0, exactly 0 below -40 so that no denormals are produced
inline V vexp(V x) {
  V keep = wasm_f32x4_gt(x, splat(-40.f));
  x = wasm_f32x4_pmax(x, splat(-40.f));
  V n = wasm_f32x4_nearest(wasm_f32x4_mul(x, splat(1.44269504f)));
  V r = wasm_f32x4_sub(wasm_f32x4_sub(x, wasm_f32x4_mul(n, splat(0.693359375f))), wasm_f32x4_mul(n, splat(-2.12194440e-4f)));
  V p = madd(splat(1.9875691500e-4f), r, splat(1.3981999507e-3f));
  p = madd(p, r, splat(8.3334519073e-3f));
  p = madd(p, r, splat(4.1665795894e-2f));
  p = madd(p, r, splat(1.6666665459e-1f));
  p = madd(p, r, splat(5.0000001201e-1f));
  p = madd(p, wasm_f32x4_mul(r, r), wasm_f32x4_add(r, splat(1.f)));
  V e = wasm_i32x4_shl(wasm_i32x4_sub(wasm_f32x4_add(n, splat(12582912.f)), wasm_i32x4_splat(0x4b400000 - 127)), 23);
  p = wasm_f32x4_mul(p, e);
  return wasm_v128_and(p, keep);
}

// x * Phi(x) with erfc(u) ~ (1 + a1 u + ... + a6 u^6)^-16 (Abramowitz-Stegun 7.1.28, |error| < 3e-7)
inline V gelu(V x) {
  V u = wasm_f32x4_abs(x);
  V p = madd(splat(5.38297625e-6f), u, splat(4.88878483e-5f));
  p = madd(p, u, splat(3.80035750e-5f));
  p = madd(p, u, splat(3.27763634e-3f));
  p = madd(p, u, splat(2.11410062e-2f));
  p = madd(p, u, splat(4.98673470e-2f));
  p = madd(p, u, splat(1.f));
  p = wasm_f32x4_mul(p, p);
  p = wasm_f32x4_mul(p, p);
  p = wasm_f32x4_mul(p, p);
  V e = wasm_f32x4_div(splat(0.5f), wasm_f32x4_mul(p, p));
  return wasm_f32x4_mul(x, wasm_v128_bitselect(e, wasm_f32x4_sub(splat(1.f), e), wasm_f32x4_lt(x, splat(0.f))));
}

template <int T, int NV, int ACT, bool RES>
inline __attribute__((always_inline)) void kernel(const float* x, int xs, const float* w, int ws, int K, const float* b, int bs, float* y, int ys) {
  V acc[T][NV];
  for (int t = 0; t < T; t++)
    for (int v = 0; v < NV; v++) {
      acc[t][v] = wasm_v128_load(b + t * bs + 4 * v);
      if (RES) acc[t][v] = wasm_f32x4_add(acc[t][v], wasm_v128_load(y + t * ys + 4 * v));
    }
  for (int k = 0; k < K; k++, w += ws) {
    V wv[NV];
    for (int v = 0; v < NV; v++) wv[v] = wasm_v128_load(w + 4 * v);
    for (int t = 0; t < T; t++) {
      V xb = wasm_v128_load32_splat(x + t * xs + k);
      for (int v = 0; v < NV; v++) acc[t][v] = madd(xb, wv[v], acc[t][v]);
    }
  }
  for (int t = 0; t < T; t++)
    for (int v = 0; v < NV; v++) {
      V a = acc[t][v];
      if (ACT == RELU) a = wasm_f32x4_pmax(a, splat(0.f));
      if (ACT == GELU) a = gelu(a);
      wasm_v128_store(y + t * ys + 4 * v, a);
    }
}

// y[T][N] (+)= b + x[T][K] * w, where column block j of w starts at w + j * wb with row stride ws
template <int TB, int NV, int ACT, bool RES>
void matmul(const float* x, int xs, int T, const float* w, int ws, int wb, int K, int N, const float* b, int bs, float* y, int ys) {
  for (int n = 0; n < N; n += 4 * NV, w += wb, b += 4 * NV, y += 4 * NV) {
    int t = 0;
    for (; t + TB <= T; t += TB) kernel<TB, NV, ACT, RES>(x + t * xs, xs, w, ws, K, b + t * bs, bs, y + t * ys, ys);
    for (; t < T; t++) kernel<1, NV, ACT, RES>(x + t * xs, xs, w, ws, K, b + t * bs, bs, y + t * ys, ys);
  }
}

template <int T, int ACT, bool RES>
inline __attribute__((always_inline)) void kernel16(const int16_t* x, int K, const int16_t* w, const float* scale, const float* b, float* y, int ys) {
  V acc[T][4];
  for (int t = 0; t < T; t++)
    for (int o = 0; o < 4; o++) acc[t][o] = wasm_i32x4_splat(0);
  for (int k = 0; k < K; k += 8, w += 32) {
    V xv[T];
    for (int t = 0; t < T; t++) xv[t] = wasm_v128_load(x + t * K + k);
    for (int o = 0; o < 4; o++) {
      V wv = wasm_v128_load(w + o * 8);
      for (int t = 0; t < T; t++) acc[t][o] = wasm_i32x4_add(acc[t][o], wasm_i32x4_dot_i16x8(xv[t], wv));
    }
  }
  for (int t = 0; t < T; t++) {
    V* a = acc[t];
    V lo = wasm_i32x4_add(wasm_i32x4_shuffle(a[0], a[1], 0, 4, 1, 5), wasm_i32x4_shuffle(a[0], a[1], 2, 6, 3, 7));
    V hi = wasm_i32x4_add(wasm_i32x4_shuffle(a[2], a[3], 0, 4, 1, 5), wasm_i32x4_shuffle(a[2], a[3], 2, 6, 3, 7));
    V r = wasm_i32x4_add(wasm_i64x2_shuffle(lo, hi, 0, 2), wasm_i64x2_shuffle(lo, hi, 1, 3));
    V f = madd(wasm_f32x4_convert_i32x4(r), splat(scale[t]), wasm_v128_load(b));
    if (RES) f = wasm_f32x4_add(f, wasm_v128_load(y + t * ys));
    if (ACT == RELU) f = wasm_f32x4_pmax(f, splat(0.f));
    if (ACT == GELU) f = gelu(f);
    wasm_v128_store(y + t * ys, f);
  }
}

// y[CELLS][n] (+)= b + scale[t] * xq[t] . q
template <int ACT, bool RES>
void apply16(const int16_t* x, const float* norm, const Linear& l, float* y) {
  float scale[CELLS];
  for (int t = 0; t < CELLS; t++) scale[t] = norm[t] * l.scale;
  for (int n = 0; n < l.n; n += 4) {
    const int16_t* w = &l.q[n * l.k];
    const float* b = &l.b[n];
    int t = 0;
    for (; t + 2 <= CELLS; t += 2) kernel16<2, ACT, RES>(x + t * l.k, l.k, w, scale + t, b, y + t * l.n + n, l.n);
    kernel16<1, ACT, RES>(x + t * l.k, l.k, w, scale + t, b, y + t * l.n + n, l.n);
  }
}

template <int ACT, bool RES>
void apply(const float* x, const Linear& l, float* y) {
  matmul<4, 2, ACT, RES>(x, l.k, CELLS, l.w.data(), 8, l.k * 8, l.k, l.n, l.b.data(), 0, y, l.n);
}

// int16 weights [n / 4][k / 8][4][8], scaled so that a dot product with a vector of norm <= 2^14 cannot overflow int32
void quantize(Linear& l, const float* p, int n) {
  float top = 1e-30f, norm = 1e-30f;
  for (int j = 0; j < n; j++) {
    float sum = 0;
    for (int i = 0; i < l.k; i++) sum += p[i * n + j] * p[i * n + j], top = std::max(top, std::abs(p[i * n + j]));
    norm = std::max(norm, std::sqrt(sum));
  }
  float sw = std::min(32767 / top, 0.99f * (1 << 17) / norm);
  l.scale = 1 / (sw * QONE);
  l.q.resize(n * l.k);
  for (int j = 0; j < n; j++)
    for (int i = 0; i < l.k; i++) l.q[(j / 4 * (l.k / 8) + i / 8) * 32 + j % 4 * 8 + i % 8] = std::lrint(p[i * n + j] * sw);
}

Linear linear(const float*& p, int k, int n, int block, bool int16 = false) {
  int np = (n + block - 1) / block * block;
  Linear l{std::vector<float>(k * np), std::vector<float>(np), k, np};
  if (int16) quantize(l, p, n);
  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++) l.w[(j / block * k + i) * block + j % block] = *p++;
  for (int j = 0; j < n; j++) l.b[j] = *p++;
  return l;
}

float hsum(V v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

}

bool Net::init(const float* p, size_t n, Config config) {
  c = config;
  int d = c.width, ff = c.ff, hd = c.heads;
  size_t want = 37 + 42 + 147 + 9 * d + 37 * d + c.blocks * (3 * d * d + 3 * d + hd * 1369 + d * d + d + d * ff + ff + ff * d + d)
              + d * 26 + 26 + CELLS * VC * VH + VH + VH + 1;
  if (n != want || d % 8 || d / hd % 8 || ff % 8) return false;
  for (int& v : flat) v = *p++;
  for (int& v : push) v = *p++;
  for (auto& l : lines)
    for (int& v : l) v = *p++;
  embed = linear(p, 9, d, 8);
  embed.b.clear();
  p -= d;
  pos.assign(p, p + CELLS * d);
  p += CELLS * d;
  blocks.clear();
  for (int i = 0; i < c.blocks; i++) {
    Block b;
    b.qkv = linear(p, d, 3 * d, 8, c.int16);
    b.rel.assign(hd * CELLS * PAD, -1e30f);
    for (int j = 0; j < hd * CELLS; j++)
      for (int k = 0; k < CELLS; k++) b.rel[j * PAD + k] = *p++;
    b.o = linear(p, d, d, 8, c.int16);
    b.f1 = linear(p, d, ff, 8, c.int16);
    b.f2 = linear(p, ff, d, 8, c.int16);
    blocks.push_back(std::move(b));
  }
  pv = linear(p, d, 26, 8);
  v0 = linear(p, CELLS * VC, VH, 32);
  v1.assign(p, p + VH + 1);
  int wide = std::max(3 * d, ff);
  zero.assign(wide, 0);
  h.resize(CELLS * d);
  z.resize(CELLS * d);
  att.resize(CELLS * d);
  y.resize(CELLS * 3 * d);
  g.resize(CELLS * std::max(ff, PV));
  s.resize(CELLS * PAD);
  kp.resize(PAD * d / hd);
  xq.resize(CELLS * std::max(d, ff));
  return true;
}

void Net::forward(const float* in, float* logits, float* value) {
  const int d = c.width, d3 = 3 * d, dh = d / c.heads;
  auto normed = [&]() -> const float* {
    for (int t = 0; t < CELLS; t++) {
      const float* x = &h[t * d];
      V a = splat(0.f), b = a;
      for (int k = 0; k < d; k += 8) {
        V u = wasm_v128_load(x + k), v = wasm_v128_load(x + k + 4);
        a = madd(u, u, a);
        b = madd(v, v, b);
      }
      V r = splat(1 / std::sqrt(hsum(wasm_f32x4_add(a, b)) + 1e-5f * d));
      for (int k = 0; k < d; k += 4) wasm_v128_store(&z[t * d + k], wasm_f32x4_mul(wasm_v128_load(x + k), r));
    }
    return z.data();
  };

  // xq[t] = round(2^14 * x[t] / |x[t]|), norm[t] = |x[t]| or 1 when the layer input is the normalized x
  float norm[CELLS];
  auto quantized = [&](const float* x, int n, float eps) {
    for (int t = 0; t < CELLS; t++, x += n) {
      V a = splat(0.f);
      for (int k = 0; k < n; k += 4) a = madd(wasm_v128_load(x + k), wasm_v128_load(x + k), a);
      float r = 1 / std::sqrt(hsum(a) + eps);
      norm[t] = eps > 1e-9f ? 1 : 1 / r;
      V s = splat(r * QONE), magic = splat(12582912.f);
      for (int k = 0; k < n; k += 8) {
        V u = wasm_i32x4_sub(madd(wasm_v128_load(x + k), s, magic), magic);
        V v = wasm_i32x4_sub(madd(wasm_v128_load(x + k + 4), s, magic), magic);
        wasm_v128_store(&xq[t * n + k], wasm_i16x8_narrow_i32x4(u, v));
      }
    }
    return xq.data();
  };

  float x9[CELLS * 9];
  for (int t = 0; t < CELLS; t++)
    for (int i = 0; i < 9; i++) x9[t * 9 + i] = in[i * 49 + flat[t]];
  matmul<4, 2, NONE, false>(x9, 9, CELLS, embed.w.data(), 8, 72, 9, d, pos.data(), d, h.data(), d);

  for (Block& b : blocks) {
    if (c.int16) apply16<NONE, false>(quantized(h.data(), d, 1e-5f * d), norm, b.qkv, y.data());
    else apply<NONE, false>(normed(), b.qkv, y.data());
    for (int hd = 0; hd < c.heads; hd++) {
      const float* k = &y[d + hd * dh];
      for (int j = 0; j < CELLS; j++)
        for (int i = 0; i < dh; i++) kp[(j / 8 * dh + i) * 8 + j % 8] = k[j * d3 + i];
      matmul<4, 2, NONE, false>(&y[hd * dh], d3, CELLS, kp.data(), 8, dh * 8, dh, PAD, &b.rel[hd * CELLS * PAD], PAD, s.data(), PAD);
      for (int t = 0; t < CELLS; t++) {
        float* r = &s[t * PAD];
        V m = wasm_v128_load(r), sum = splat(0.f), e[PAD / 4];
        for (int i = 4; i < PAD; i += 4) m = wasm_f32x4_pmax(m, wasm_v128_load(r + i));
        m = splat(std::max(std::max(wasm_f32x4_extract_lane(m, 0), wasm_f32x4_extract_lane(m, 1)), std::max(wasm_f32x4_extract_lane(m, 2), wasm_f32x4_extract_lane(m, 3))));
        for (int i = 0; i < PAD / 4; i++) sum = wasm_f32x4_add(sum, e[i] = vexp(wasm_f32x4_sub(wasm_v128_load(r + 4 * i), m)));
        sum = splat(1 / hsum(sum));
        for (int i = 0; i < PAD / 4; i++) wasm_v128_store(r + 4 * i, wasm_f32x4_mul(e[i], sum));
      }
      matmul<4, 2, NONE, false>(s.data(), PAD, CELLS, k + d, d3, 8, CELLS, dh, zero.data(), 0, &att[hd * dh], d);
    }
    if (c.int16) {
      apply16<NONE, true>(quantized(att.data(), d, 1e-30f), norm, b.o, h.data());
      const int16_t* q = quantized(h.data(), d, 1e-5f * d);
      if (c.relu) apply16<RELU, false>(q, norm, b.f1, g.data());
      else {
        apply16<NONE, false>(q, norm, b.f1, g.data());
        for (int i = 0; i < CELLS * c.ff; i += 4) wasm_v128_store(&g[i], gelu(wasm_v128_load(&g[i])));
      }
      apply16<NONE, true>(quantized(g.data(), c.ff, 1e-30f), norm, b.f2, h.data());
      continue;
    }
    apply<NONE, true>(att.data(), b.o, h.data());
    if (c.relu) apply<RELU, false>(normed(), b.f1, g.data());
    else apply<GELU, false>(normed(), b.f1, g.data());
    apply<NONE, true>(g.data(), b.f2, h.data());
  }

  float* p = g.data();
  apply<NONE, false>(normed(), pv, p);
  for (int a = 0; a < 42; a++) logits[a] = p[push[a]];
  for (int l = 0; l < 21; l++) {
    float* cap = logits + 42 + l * 128;
    float line = 0;
    int cells = 0;
    for (int j = 0; j < 7 && lines[l][j] >= 0; j++, cells++) line += p[lines[l][j] * PV + 6 + l / 7];
    cap[0] = line / cells;
    for (int j = 0, n = 1; j < 7; j++, n *= 2) {
      float q = j < cells ? p[lines[l][j] * PV + 9] : 0;
      for (int i = 0; i < n; i++) cap[n + i] = cap[i] + q;
    }
  }
  float v[CELLS * VC], u[VH];
  for (int t = 0; t < CELLS; t++)
    for (int i = 0; i < VC; i++) v[t * VC + i] = std::max(p[t * PV + 10 + i], 0.f);
  matmul<1, 8, RELU, false>(v, 0, 1, v0.w.data(), 32, CELLS * VC * 32, CELLS * VC, VH, v0.b.data(), 0, u, 0);
  float out = v1[VH];
  for (int i = 0; i < VH; i++) out += u[i] * v1[i];
  *value = std::tanh(out);
}

}
