#include "../engine/former_infer.h"
#include <emscripten.h>

static former::Net net;

extern "C" {
EMSCRIPTEN_KEEPALIVE int former_init(const float* w, int n, int width, int blocks, int heads, int ff, int relu, int int16) {
  return net.init(w, n, {width, blocks, heads, ff, !!relu, !!int16});
}
EMSCRIPTEN_KEEPALIVE void former_forward(const float* x, float* logits, float* value) { net.forward(x, logits, value); }
}
