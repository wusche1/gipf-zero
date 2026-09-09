#ifndef GIPF_WASM
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#else
#include <emscripten/bind.h>
#include <emscripten/val.h>
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef GIPF_WASM
namespace py = pybind11;
#endif

namespace {

constexpr int kBoardCells = 37;
constexpr int kLineCount = 21;
constexpr int kRayCount = 42;
constexpr int kCaptureBase = 42;
constexpr int kCaptureStride = 128;

int colour_index(int colour) { return colour == 1 ? 0 : 1; }

struct Geometry {
  std::vector<std::array<int, 2>> coordinates;
  std::array<int, 121> index_by_coordinate{};
  std::vector<std::vector<int>> lines;
  std::vector<std::vector<int>> rays;

  Geometry() {
    index_by_coordinate.fill(-1);
    for (int r = -3; r <= 3; ++r) {
      const int q_min = std::max(-3, -r - 3);
      const int q_max = std::min(3, -r + 3);
      for (int q = q_min; q <= q_max; ++q) {
        const int index = static_cast<int>(coordinates.size());
        coordinates.push_back({q, r});
        index_by_coordinate[(q + 3) * 11 + r + 3] = index;
      }
    }
    for (int q = -3; q <= 3; ++q) {
      std::vector<int> line;
      for (int r = std::max(-3, -q - 3); r <= std::min(3, -q + 3); ++r)
        line.push_back(index(q, r));
      lines.push_back(std::move(line));
    }
    for (int r = -3; r <= 3; ++r) {
      std::vector<int> line;
      for (int q = std::max(-3, -r - 3); q <= std::min(3, -r + 3); ++q)
        line.push_back(index(q, r));
      lines.push_back(std::move(line));
    }
    for (int s = -3; s <= 3; ++s) {
      std::vector<int> line;
      for (int q = std::max(-3, -s - 3); q <= std::min(3, -s + 3); ++q)
        line.push_back(index(q, -s - q));
      lines.push_back(std::move(line));
    }
    for (const auto& line : lines) {
      rays.push_back(line);
      rays.emplace_back(line.rbegin(), line.rend());
    }
  }

  int index(int q, int r) const {
    const int encoded = (q + 3) * 11 + r + 3;
    if (q < -3 || q > 3 || r < -3 || r > 3 || encoded < 0 || encoded >= 121 ||
        index_by_coordinate[encoded] < 0)
      throw std::logic_error("invalid axial coordinate");
    return index_by_coordinate[encoded];
  }
};

const Geometry& geometry_data() {
  static const Geometry geometry;
  return geometry;
}

struct Segment {
  int line_id;
  int first;
  int last;
};

class State {
 public:
  State() { reset_opening(); }

  State clone() const { return *this; }

  std::vector<int> legal_actions() const {
    if (winner_ != 0) return {};
    if (phase_ == Phase::Push) {
      if (reserves_[colour_index(current_player_)] <= 0) return {};
      std::vector<int> actions;
      const auto& rays = geometry_data().rays;
      for (int ray_id = 0; ray_id < kRayCount; ++ray_id) {
        bool full = true;
        for (int cell : rays[ray_id]) {
          if (board_[cell] == 0) {
            full = false;
            break;
          }
        }
        if (!full) actions.push_back(ray_id);
      }
      return actions;
    }
    return capture_actions_for(current_player_);
  }

  void apply(int action) {
    const std::vector<int> legal = legal_actions();
    if (std::find(legal.begin(), legal.end(), action) == legal.end())
      throw std::invalid_argument("illegal GIPF action");
    if (phase_ == Phase::Push) {
      apply_push(action);
    } else {
      apply_capture(action);
    }
  }

  int current_player() const { return current_player_; }
  int turn_player() const { return turn_player_; }
  int winner() const { return winner_; }
  int ply() const { return ply_; }
  std::string phase() const { return phase_ == Phase::Push ? "push" : "capture"; }
  const std::vector<int>& board() const { return board_; }
  std::vector<int> reserves() const { return {reserves_[0], reserves_[1]}; }
  std::vector<int> captured() const { return {captured_[0], captured_[1]}; }

#ifndef GIPF_WASM
  py::dict serialize() const {
    py::dict result;
    result["board"] = board_;
    result["reserves"] = reserves();
    result["captured"] = captured();
    result["current_player"] = current_player_;
    result["turn_player"] = turn_player_;
    result["phase"] = phase();
    result["winner"] = winner_;
    result["ply"] = ply_;
    return result;
  }

  static State from_dict(const py::dict& input) {
    State state;
    auto required = [&input](const char* name) -> py::handle {
      if (!input.contains(name)) throw std::invalid_argument(std::string("missing state field: ") + name);
      return input[name];
    };
    state.board_ = required("board").cast<std::vector<int>>();
    const std::vector<int> reserves = required("reserves").cast<std::vector<int>>();
    if (state.board_.size() != kBoardCells || reserves.size() != 2)
      throw std::invalid_argument("board must have 37 cells and reserves must have two values");
    state.reserves_ = {reserves[0], reserves[1]};
    if (input.contains("captured")) {
      const std::vector<int> captured = input["captured"].cast<std::vector<int>>();
      if (captured.size() != 2) throw std::invalid_argument("captured must have two values");
      state.captured_ = {captured[0], captured[1]};
    } else {
      state.captured_ = {18 - state.reserves_[0], 18 - state.reserves_[1]};
      for (int piece : state.board_) {
        if (piece > 0) state.captured_[0] -= std::abs(piece);
        if (piece < 0) state.captured_[1] -= std::abs(piece);
      }
    }
    state.current_player_ = required("current_player").cast<int>();
    state.turn_player_ = input.contains("turn_player") ? input["turn_player"].cast<int>() : state.current_player_;
    const std::string phase = required("phase").cast<std::string>();
    state.phase_ = phase == "push" ? Phase::Push : phase == "capture" ? Phase::Capture : throw std::invalid_argument("phase must be push or capture");
    state.winner_ = input.contains("winner") ? input["winner"].cast<int>() : 0;
    state.ply_ = input.contains("ply") ? input["ply"].cast<int>() : 0;
    state.validate();
    return state;
  }

  static py::dict geometry() {
    const auto& g = geometry_data();
    py::dict result;
    result["coordinates"] = g.coordinates;
    result["lines"] = g.lines;
    result["rays"] = g.rays;
    py::list pushes;
    for (int action = 0; action < kRayCount; ++action) {
      py::dict info;
      info["action"] = action;
      info["ray_id"] = action;
      info["line_id"] = action / 2;
      info["reversed"] = (action % 2) == 1;
      info["cells"] = g.rays[action];
      pushes.append(info);
    }
    result["push_actions"] = pushes;
    py::dict capture;
    capture["base"] = kCaptureBase;
    capture["line_stride"] = kCaptureStride;
    capture["mask_bits"] = 7;
    result["capture_action"] = capture;
    return result;
  }
#endif

  // Safe to call at any trust boundary before a state is used by the engine.
  void validate() const {
    if ((current_player_ != 1 && current_player_ != -1) || (turn_player_ != 1 && turn_player_ != -1) ||
        (winner_ != 0 && winner_ != 1 && winner_ != -1) || ply_ < 0)
      throw std::invalid_argument("invalid player, winner, or ply");
    std::array<int, 2> material = reserves_;
    std::array<int, 2> doubles = {0, 0};
    for (int c = 0; c < 2; ++c)
      if (reserves_[c] < 0 || reserves_[c] > 18 || captured_[c] < 0 || captured_[c] > 18)
        throw std::invalid_argument("material counts must be in 0..18");
    for (int p : board_) {
      if (p < -2 || p > 2) throw std::invalid_argument("invalid board piece");
      if (p != 0) {
        const int c = colour_index(p > 0 ? 1 : -1);
        material[c] += std::abs(p);
        if (std::abs(p) == 2) ++doubles[c];
      }
    }
    for (int c = 0; c < 2; ++c) {
      if (doubles[c] > 3) throw std::invalid_argument("standard GIPF permits at most three doubles per colour");
      if (material[c] + captured_[c] != 18) throw std::invalid_argument("material must total 18 per colour");
    }
    const int expected_turn = ply_ == 0 ? 1 : (ply_ % 2 == 1 ? 1 : -1);
    if (turn_player_ != expected_turn)
      throw std::invalid_argument("turn_player is inconsistent with insertion count");
    if (winner_ != 0) {
      const int loser = -winner_;
      if (doubles[colour_index(loser)] == 0) {
        if (phase_ != Phase::Capture)
          throw std::invalid_argument("last-double terminal states occur during capture");
      } else if (reserves_[colour_index(loser)] == 0) {
        if (phase_ != Phase::Push || current_player_ != loser)
          throw std::invalid_argument("reserve terminal state has the wrong decision owner");
      } else {
        throw std::invalid_argument("winner is inconsistent with doubles and reserves");
      }
      return;
    }
    if (doubles[0] == 0 || doubles[1] == 0)
      throw std::invalid_argument("ongoing standard GIPF state must retain both colours' doubles");
    if (phase_ == Phase::Push) {
      if (has_rows(1) || has_rows(-1))
        throw std::invalid_argument("mandatory row cannot be deferred into push phase");
      const bool opening = ply_ == 0 && current_player_ == 1 && turn_player_ == 1;
      if (!opening && current_player_ != -turn_player_)
        throw std::invalid_argument("push phase has the wrong decision owner");
      if (reserves_[colour_index(current_player_)] == 0)
        throw std::invalid_argument("player due to insert cannot have an empty reserve");
    } else {
      const int expected_owner = has_rows(turn_player_) ? turn_player_ :
          (has_rows(-turn_player_) ? -turn_player_ : 0);
      if (expected_owner == 0 || current_player_ != expected_owner)
        throw std::invalid_argument("capture phase has no mandatory row or wrong priority owner");
    }
  }

 private:
  enum class Phase { Push, Capture };
  std::vector<int> board_{kBoardCells, 0};
  std::array<int, 2> reserves_{12, 12};
  std::array<int, 2> captured_{0, 0};
  int current_player_ = 1;
  int turn_player_ = 1;
  Phase phase_ = Phase::Push;
  int winner_ = 0;
  int ply_ = 0;

  void reset_opening() {
    board_.assign(kBoardCells, 0);
    const auto& g = geometry_data();
    const std::array<std::array<int, 2>, 6> corners = {{{0, -3}, {3, -3}, {3, 0}, {0, 3}, {-3, 3}, {-3, 0}}};
    for (int i = 0; i < 6; ++i) board_[g.index(corners[i][0], corners[i][1])] = (i % 2 == 0 ? 2 : -2);
    reserves_ = {12, 12};
    captured_ = {0, 0};
    current_player_ = turn_player_ = 1;
    phase_ = Phase::Push;
    winner_ = ply_ = 0;
  }

  std::vector<Segment> rows_for(int owner) const {
    std::vector<Segment> result;
    const auto& lines = geometry_data().lines;
    for (int line_id = 0; line_id < kLineCount; ++line_id) {
      const auto& line = lines[line_id];
      int p = 0;
      while (p < static_cast<int>(line.size())) {
        while (p < static_cast<int>(line.size()) && board_[line[p]] == 0) ++p;
        const int first = p;
        while (p < static_cast<int>(line.size()) && board_[line[p]] != 0) ++p;
        const int last = p - 1;
        int run = 0;
        bool has_four = false;
        for (int i = first; i <= last; ++i) {
          if ((board_[line[i]] > 0 ? 1 : -1) == owner) {
            has_four = has_four || ++run >= 4;
          } else {
            run = 0;
          }
        }
        if (has_four) result.push_back({line_id, first, last});
      }
    }
    return result;
  }

  bool has_rows(int owner) const { return !rows_for(owner).empty(); }

  std::vector<int> capture_actions_for(int owner) const {
    std::vector<int> actions;
    const auto& lines = geometry_data().lines;
    for (const Segment& segment : rows_for(owner)) {
      int doubles = 0;
      for (int i = segment.first; i <= segment.last; ++i)
        if (std::abs(board_[lines[segment.line_id][i]]) == 2) doubles |= 1 << i;
      int subset = doubles;
      do {
        actions.push_back(kCaptureBase + segment.line_id * kCaptureStride + subset);
        subset = (subset - 1) & doubles;
      } while (subset != doubles);
    }
    std::sort(actions.begin(), actions.end());
    return actions;
  }

  void apply_push(int ray_id) {
    const auto& ray = geometry_data().rays[ray_id];
    int first_empty = 0;
    while (board_[ray[first_empty]] != 0) ++first_empty;
    for (int i = first_empty; i > 0; --i) board_[ray[i]] = board_[ray[i - 1]];
    board_[ray[0]] = current_player_;
    --reserves_[colour_index(current_player_)];
    turn_player_ = current_player_;
    ++ply_;
    choose_next_capture_or_advance();
  }

  void apply_capture(int action) {
    const int packed = action - kCaptureBase;
    const int line_id = packed / kCaptureStride;
    const int mask = packed % kCaptureStride;
    const int actor = current_player_;
    const auto segments = rows_for(actor);
    auto it = std::find_if(segments.begin(), segments.end(), [line_id](const Segment& s) { return s.line_id == line_id; });
    if (it == segments.end()) throw std::logic_error("legal capture disappeared");
    const auto& line = geometry_data().lines[line_id];
    for (int i = it->first; i <= it->last; ++i) {
      const int piece = board_[line[i]];
      if (std::abs(piece) == 1 || (mask & (1 << i))) {
        const int owner = piece > 0 ? 1 : -1;
        if (owner == actor) reserves_[colour_index(owner)] += std::abs(piece);
        else captured_[colour_index(owner)] += std::abs(piece);
        board_[line[i]] = 0;
      }
    }
    const int actor_doubles = count_doubles(actor);
    const int opponent_doubles = count_doubles(-actor);
    if (opponent_doubles == 0) {
      winner_ = actor;
      return;
    }
    if (actor_doubles == 0) {
      winner_ = -actor;
      return;
    }
    choose_next_capture_or_advance();
  }

  int count_doubles(int owner) const {
    return static_cast<int>(std::count(board_.begin(), board_.end(), 2 * owner));
  }

  void choose_next_capture_or_advance() {
    if (has_rows(turn_player_)) {
      current_player_ = turn_player_;
      phase_ = Phase::Capture;
    } else if (has_rows(-turn_player_)) {
      current_player_ = -turn_player_;
      phase_ = Phase::Capture;
    } else {
      current_player_ = -turn_player_;
      phase_ = Phase::Push;
      if (reserves_[colour_index(current_player_)] == 0) winner_ = -current_player_;
    }
  }
};

}  // namespace

#ifndef GIPF_WASM
PYBIND11_MODULE(gipf_engine, m) {
  m.doc() = "Fast standard-GIPF rules engine";
  py::class_<State>(m, "State")
      .def(py::init<>())
      .def("clone", &State::clone)
      .def("legal_actions", &State::legal_actions)
      .def("apply", &State::apply, py::arg("action"))
      .def("validate", &State::validate)
      .def("serialize", &State::serialize)
      .def_static("from_dict", &State::from_dict, py::arg("state"))
      .def_static("geometry", &State::geometry)
      .def_property_readonly("board", &State::board)
      .def_property_readonly("reserves", &State::reserves)
      .def_property_readonly("captured", &State::captured)
      .def_property_readonly("current_player", &State::current_player)
      .def_property_readonly("turn_player", &State::turn_player)
      .def_property_readonly("phase", &State::phase)
      .def_property_readonly("winner", &State::winner)
      .def_property_readonly("ply", &State::ply);
  m.def("geometry", &State::geometry);
  m.def("encode_batch", [](const std::vector<State>& states) {
    constexpr py::ssize_t kPlanes = 9;
    constexpr py::ssize_t kSide = 7;
    constexpr py::ssize_t kPlaneCells = kSide * kSide;
    py::array_t<float> output({static_cast<py::ssize_t>(states.size()), kPlanes, kSide, kSide});
    float* data = output.mutable_data();
    std::fill(data, data + states.size() * kPlanes * kPlaneCells, 0.0F);
    const auto& coordinates = geometry_data().coordinates;
    for (size_t batch = 0; batch < states.size(); ++batch) {
      const State& state = states[batch];
      const int player = state.current_player();
      const int mine = colour_index(player);
      const int theirs = colour_index(-player);
      const float own_reserve = static_cast<float>(state.reserves()[mine]) / 18.0F;
      const float opponent_reserve = static_cast<float>(state.reserves()[theirs]) / 18.0F;
      const bool capture = state.phase() == "capture";
      const bool mover_owns_decision = state.turn_player() == player;
      for (int cell = 0; cell < kBoardCells; ++cell) {
        const int row = coordinates[cell][1] + 3;
        const int column = coordinates[cell][0] + 3;
        const size_t offset = batch * kPlanes * kPlaneCells + row * kSide + column;
        const int relative_piece = state.board()[cell] * player;
        if (relative_piece == 1) data[offset] = 1.0F;
        else if (relative_piece == 2) data[kPlaneCells + offset] = 1.0F;
        else if (relative_piece == -1) data[2 * kPlaneCells + offset] = 1.0F;
        else if (relative_piece == -2) data[3 * kPlaneCells + offset] = 1.0F;
        data[4 * kPlaneCells + offset] = 1.0F;
        data[5 * kPlaneCells + offset] = own_reserve;
        data[6 * kPlaneCells + offset] = opponent_reserve;
        data[7 * kPlaneCells + offset] = capture ? 1.0F : 0.0F;
        data[8 * kPlaneCells + offset] = mover_owns_decision ? 1.0F : 0.0F;
      }
    }
    return output;
  }, py::arg("states"), "Encode states into float32 [B,9,7,7] training planes.");
  m.def("puct_select", [](py::array_t<double, py::array::c_style | py::array::forcecast> priors,
                           py::array_t<int32_t, py::array::c_style | py::array::forcecast> visits,
                           py::array_t<double, py::array::c_style | py::array::forcecast> values,
                           double cpuct) {
    const auto p = priors.request();
    const auto n = visits.request();
    const auto w = values.request();
    if (p.ndim != 1 || n.ndim != 1 || w.ndim != 1 || p.shape[0] == 0 ||
        p.shape[0] != n.shape[0] || p.shape[0] != w.shape[0])
      throw std::invalid_argument("PUCT arrays must be nonempty, one-dimensional, and equal length");
    const auto* prior = static_cast<const double*>(p.ptr);
    const auto* visit = static_cast<const int32_t*>(n.ptr);
    const auto* value = static_cast<const double*>(w.ptr);
    int64_t total = 0;
    for (py::ssize_t i = 0; i < p.shape[0]; ++i) total += visit[i];
    const double sqrt_total = std::sqrt(1.0 + static_cast<double>(total));
    py::ssize_t best = 0;
    double best_score = (visit[0] == 0 ? 0.0 : value[0] / static_cast<double>(visit[0])) +
        (cpuct * prior[0]) * sqrt_total / (1.0 + static_cast<double>(visit[0]));
    for (py::ssize_t i = 1; i < p.shape[0]; ++i) {
      const double score = (visit[i] == 0 ? 0.0 : value[i] / static_cast<double>(visit[i])) +
          (cpuct * prior[i]) * sqrt_total / (1.0 + static_cast<double>(visit[i]));
      if (score > best_score) {
        best = i;
        best_score = score;
      }
    }
    return static_cast<int>(best);
  }, py::arg("priors"), py::arg("visits"), py::arg("values"), py::arg("cpuct"),
  "Return the first PUCT argmax using the search's exact score formula.");
  m.attr("PUSH_ACTIONS") = py::make_tuple(0, 41);
  m.attr("CAPTURE_BASE") = kCaptureBase;
  m.attr("CAPTURE_LINE_STRIDE") = kCaptureStride;
}
#else
using namespace emscripten;

val wasm_int_array(const std::vector<int>& values) {
  val array = val::array();
  for (int value : values) array.call<void>("push", value);
  return array;
}

val wasm_state_serialize(const State& state) {
  val result = val::object();
  result.set("board", wasm_int_array(state.board()));
  result.set("reserves", wasm_int_array(state.reserves()));
  result.set("captured", wasm_int_array(state.captured()));
  result.set("current_player", state.current_player());
  result.set("turn_player", state.turn_player());
  result.set("phase", state.phase());
  result.set("winner", state.winner());
  result.set("ply", state.ply());
  return result;
}

val wasm_geometry() {
  const auto& g = geometry_data();
  val result = val::object();
  val coordinates = val::array();
  for (const auto& point : g.coordinates) {
    val coordinate = val::array();
    coordinate.call<void>("push", point[0]);
    coordinate.call<void>("push", point[1]);
    coordinates.call<void>("push", coordinate);
  }
  result.set("coordinates", coordinates);
  val lines = val::array();
  val rays = val::array();
  for (const auto& line : g.lines) lines.call<void>("push", wasm_int_array(line));
  for (const auto& ray : g.rays) rays.call<void>("push", wasm_int_array(ray));
  result.set("lines", lines);
  result.set("rays", rays);
  val capture = val::object();
  capture.set("base", kCaptureBase);
  capture.set("line_stride", kCaptureStride);
  capture.set("mask_bits", 7);
  result.set("capture_action", capture);
  return result;
}

EMSCRIPTEN_BINDINGS(gipf_engine) {
  register_vector<int>("IntVector");
  class_<State>("State")
      .constructor<>()
      .function("clone", &State::clone)
      .function("legal_actions", &State::legal_actions)
      .function("apply", &State::apply)
      .function("validate", &State::validate)
      .function("serialize", &wasm_state_serialize)
      .property("board", &State::board)
      .property("reserves", &State::reserves)
      .property("captured", &State::captured)
      .property("current_player", &State::current_player)
      .property("turn_player", &State::turn_player)
      .property("phase", &State::phase)
      .property("winner", &State::winner)
      .property("ply", &State::ply);
  constant("CAPTURE_BASE", kCaptureBase);
  constant("CAPTURE_LINE_STRIDE", kCaptureStride);
  function("geometry", &wasm_geometry);
}
#endif
