#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace former {

struct Config {
  int width, blocks, heads, ff;
  bool relu = false, int16 = false;
};

struct Linear {
  std::vector<float> w, b;
  int k, n;
  std::vector<int16_t> q;
  float scale;
};

struct Block {
  Linear qkv, o, f1, f2;
  std::vector<float> rel;
};

struct Net {
  bool init(const float* weights, size_t n, Config config);
  void forward(const float* planes441, float* logits2730, float* value);

  Config c;
  int flat[37], push[42], lines[21][7];
  Linear embed, pv, v0;
  std::vector<Block> blocks;
  std::vector<float> pos, v1, zero, h, z, y, att, g, s, kp;
  std::vector<int16_t> xq;
};

}
