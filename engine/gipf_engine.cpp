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
#include <limits>
#include <memory>
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

#ifndef GIPF_WASM
// Native tree nodes keep rule-state ownership and PUCT traversal out of the
// Python inner loop.  Neural evaluation remains deliberately in Python, where
// PyTorch can batch pending leaf states on the GPU.
struct NativeNode {
  explicit NativeNode(const State& initial) : state(initial), actor(state.current_player()) {}

  State state;
  int actor;
  bool expanded = false;
  std::vector<int> actions;
  std::vector<double> priors;
  std::vector<int32_t> visits;
  std::vector<double> values;
  std::vector<std::shared_ptr<NativeNode>> children;

  void finish_expand(std::vector<int> next_actions, const float* scores, py::ssize_t scores_size,
                     bool scores_are_compact) {
    actions = std::move(next_actions);
    if (actions.empty()) throw std::invalid_argument("cannot expand a terminal state");
    int max_action = 0;
    for (int action : actions) max_action = std::max(max_action, action);
    if ((!scores_are_compact && scores_size <= max_action) ||
        (scores_are_compact && scores_size < static_cast<py::ssize_t>(actions.size())))
      throw std::invalid_argument("logits do not cover every legal action");
    auto score = [&](size_t i) { return scores[scores_are_compact ? i : actions[i]]; };
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < actions.size(); ++i) maximum = std::max(maximum, static_cast<double>(score(i)));
    priors.resize(actions.size());
    double total = 0.0;
    for (size_t i = 0; i < actions.size(); ++i) {
      priors[i] = std::exp(static_cast<double>(score(i)) - maximum);
      total += priors[i];
    }
    for (double& prior : priors) prior /= total;
    visits.assign(actions.size(), 0);
    values.assign(actions.size(), 0.0);
    children.resize(actions.size());
    expanded = true;
  }

  void expand(const float* logits, py::ssize_t logits_size) {
    finish_expand(state.legal_actions(), logits, logits_size, false);
  }

  size_t select(double cpuct) const {
    int64_t total = 0;
    for (int32_t visit : visits) total += visit;
    const double sqrt_total = std::sqrt(1.0 + static_cast<double>(total));
    size_t best = 0;
    double best_score = (visits[0] == 0 ? 0.0 : values[0] / static_cast<double>(visits[0])) +
        (cpuct * priors[0]) * sqrt_total / (1.0 + static_cast<double>(visits[0]));
    for (size_t i = 1; i < actions.size(); ++i) {
      const double score = (visits[i] == 0 ? 0.0 : values[i] / static_cast<double>(visits[i])) +
          (cpuct * priors[i]) * sqrt_total / (1.0 + static_cast<double>(visits[i]));
      if (score > best_score) { best = i; best_score = score; }
    }
    return best;
  }

  std::shared_ptr<NativeNode> child(size_t index) {
    if (index >= children.size()) throw std::out_of_range("native child index");
    if (!children[index]) {
      State next = state.clone();
      next.apply(actions[index]);
      children[index] = std::make_shared<NativeNode>(next);
    }
    return children[index];
  }

  std::shared_ptr<NativeNode> existing_child(size_t index) const {
    if (index >= children.size()) return nullptr;
    return children[index];
  }

  void mix_noise(const std::vector<double>& noise) {
    if (!expanded || noise.size() != priors.size()) throw std::invalid_argument("noise length must match expanded actions");
    for (size_t i = 0; i < priors.size(); ++i) priors[i] = .75 * priors[i] + .25 * noise[i];
  }
};

struct NativePathEntry { std::shared_ptr<NativeNode> node; size_t action; };
struct NativePending {
  std::shared_ptr<NativeNode> leaf;
  std::vector<NativePathEntry> path;
  std::vector<int> actions;
};

class NativeForest {
 public:
  explicit NativeForest(std::vector<std::shared_ptr<NativeNode>> roots) : roots_(std::move(roots)) {}

  std::vector<State> select_leaves(double cpuct) {
    collect_pending(cpuct, false);
    return pending_states();
  }

  std::vector<State> expand_unexpanded_roots() {
    collect_pending(0.0, true);
    return pending_states();
  }

  size_t select_pending(double cpuct) { return collect_pending(cpuct, false); }
  size_t expand_unexpanded_root_pending() { return collect_pending(0.0, true); }

  py::array_t<float> encode_pending() const {
    constexpr py::ssize_t kPlanes = 9;
    constexpr py::ssize_t kSide = 7;
    constexpr py::ssize_t kPlaneCells = kSide * kSide;
    py::array_t<float> output({static_cast<py::ssize_t>(pending_.size()), kPlanes, kSide, kSide});
    float* data = output.mutable_data();
    std::fill(data, data + pending_.size() * kPlanes * kPlaneCells, 0.0F);
    const auto& coordinates = geometry_data().coordinates;
    for (size_t batch = 0; batch < pending_.size(); ++batch) {
      const State& state = pending_[batch].leaf->state;
      const int player = state.current_player();
      const std::vector<int> reserves = state.reserves();
      const float own_reserve = static_cast<float>(reserves[colour_index(player)]) / 18.0F;
      const float opponent_reserve = static_cast<float>(reserves[colour_index(-player)]) / 18.0F;
      const bool capture = state.phase() == "capture";
      const bool mover_owns_decision = state.turn_player() == player;
      const auto& board = state.board();
      for (int cell = 0; cell < kBoardCells; ++cell) {
        const int row = coordinates[cell][1] + 3, column = coordinates[cell][0] + 3;
        const size_t offset = batch * kPlanes * kPlaneCells + row * kSide + column;
        const int piece = board[cell] * player;
        if (piece == 1) data[offset] = 1.0F;
        else if (piece == 2) data[kPlaneCells + offset] = 1.0F;
        else if (piece == -1) data[2 * kPlaneCells + offset] = 1.0F;
        else if (piece == -2) data[3 * kPlaneCells + offset] = 1.0F;
        data[4 * kPlaneCells + offset] = 1.0F;
        data[5 * kPlaneCells + offset] = own_reserve;
        data[6 * kPlaneCells + offset] = opponent_reserve;
        data[7 * kPlaneCells + offset] = capture ? 1.0F : 0.0F;
        data[8 * kPlaneCells + offset] = mover_owns_decision ? 1.0F : 0.0F;
      }
    }
    return output;
  }

  py::array_t<int32_t> pending_action_indices() const {
    size_t width = 0;
    for (const auto& pending : pending_) width = std::max(width, pending.actions.size());
    py::array_t<int32_t> result({static_cast<py::ssize_t>(pending_.size()), static_cast<py::ssize_t>(width)});
    auto* data = result.mutable_data();
    std::fill(data, data + pending_.size() * width, 0);
    for (size_t row = 0; row < pending_.size(); ++row)
      for (size_t column = 0; column < pending_[row].actions.size(); ++column)
        data[row * width + column] = pending_[row].actions[column];
    return result;
  }

  void finish(py::array_t<float, py::array::c_style | py::array::forcecast> logits,
              py::array_t<float, py::array::c_style | py::array::forcecast> leaf_values) {
    const auto scores = logits.request();
    const auto values = leaf_values.request();
    if (scores.ndim != 2 || values.ndim != 1 || scores.shape[0] != values.shape[0] ||
        scores.shape[0] != static_cast<py::ssize_t>(pending_.size()))
      throw std::invalid_argument("native forest logits and values must match pending leaves");
    const auto* score_data = static_cast<const float*>(scores.ptr);
    const auto* value_data = static_cast<const float*>(values.ptr);
    for (size_t i = 0; i < pending_.size(); ++i) {
      NativePending& pending = pending_[i];
      pending.leaf->finish_expand(std::move(pending.actions), score_data + i * scores.shape[1], scores.shape[1], false);
      backup(pending.path, static_cast<double>(value_data[i]) * pending.leaf->actor);
    }
    pending_.clear();
  }

  void finish_selected(py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                       py::array_t<float, py::array::c_style | py::array::forcecast> leaf_values) {
    const auto scores = logits.request();
    const auto values = leaf_values.request();
    if (scores.ndim != 2 || values.ndim != 1 || scores.shape[0] != values.shape[0] ||
        scores.shape[0] != static_cast<py::ssize_t>(pending_.size()))
      throw std::invalid_argument("selected logits and values must match pending leaves");
    const auto* score_data = static_cast<const float*>(scores.ptr);
    const auto* value_data = static_cast<const float*>(values.ptr);
    for (size_t i = 0; i < pending_.size(); ++i) {
      NativePending& pending = pending_[i];
      pending.leaf->finish_expand(std::move(pending.actions), score_data + i * scores.shape[1], scores.shape[1], true);
      backup(pending.path, static_cast<double>(value_data[i]) * pending.leaf->actor);
    }
    pending_.clear();
  }

 private:
  size_t collect_pending(double cpuct, bool roots_only) {
    pending_.clear();
    for (const auto& root : roots_) {
      std::shared_ptr<NativeNode> node = root;
      std::vector<NativePathEntry> path;
      if (roots_only) {
        if (node->state.winner() != 0 || node->expanded) continue;
      } else {
        while (node->expanded && node->state.winner() == 0) {
          const size_t choice = node->select(cpuct);
          path.push_back({node, choice});
          node = node->child(choice);
        }
        if (node->state.winner() != 0) {
          backup(path, static_cast<double>(node->state.winner()));
          continue;
        }
      }
      pending_.push_back({node, std::move(path), node->state.legal_actions()});
    }
    return pending_.size();
  }

  std::vector<State> pending_states() const {
    std::vector<State> states;
    states.reserve(pending_.size());
    for (const auto& pending : pending_) states.push_back(pending.leaf->state.clone());
    return states;
  }

  static void backup(const std::vector<NativePathEntry>& path, double white_value) {
    for (const auto& entry : path) {
      ++entry.node->visits[entry.action];
      entry.node->values[entry.action] += white_value * entry.node->actor;
    }
  }
  std::vector<std::shared_ptr<NativeNode>> roots_;
  std::vector<NativePending> pending_;
};
#endif

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
  py::class_<NativeNode, std::shared_ptr<NativeNode>>(m, "NativeNode")
      .def(py::init<const State&>())
      .def("child", &NativeNode::child)
      .def("existing_child", &NativeNode::existing_child)
      .def("mix_noise", &NativeNode::mix_noise)
      .def_property_readonly("state", [](NativeNode& node) -> State& { return node.state; }, py::return_value_policy::reference_internal)
      .def_property_readonly("actor", [](const NativeNode& node) { return node.actor; })
      .def_property_readonly("expanded", [](const NativeNode& node) { return node.expanded; })
      .def_property_readonly("actions", [](const NativeNode& node) { return node.actions; })
      .def_property_readonly("priors", [](const NativeNode& node) { return node.priors; })
      .def_property_readonly("visits", [](const NativeNode& node) { return node.visits; })
      .def_property_readonly("values", [](const NativeNode& node) { return node.values; });
  py::class_<NativeForest>(m, "NativeForest")
      .def(py::init<std::vector<std::shared_ptr<NativeNode>>>())
      .def("expand_unexpanded_roots", &NativeForest::expand_unexpanded_roots)
      .def("expand_unexpanded_root_pending", &NativeForest::expand_unexpanded_root_pending)
      .def("select_leaves", &NativeForest::select_leaves, py::arg("cpuct"))
      .def("select_pending", &NativeForest::select_pending, py::arg("cpuct"))
      .def("encode_pending", &NativeForest::encode_pending)
      .def("pending_action_indices", &NativeForest::pending_action_indices)
      .def("finish", &NativeForest::finish, py::arg("logits"), py::arg("values"))
      .def("finish_selected", &NativeForest::finish_selected, py::arg("logits"), py::arg("values"));
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
  m.def("expand_policy", [](const State& state,
                             py::array_t<float, py::array::c_style | py::array::forcecast> logits) {
    const auto input = logits.request();
    if (input.ndim != 1) throw std::invalid_argument("logits must be one-dimensional");
    const std::vector<int> legal = state.legal_actions();
    if (legal.empty()) throw std::invalid_argument("cannot expand a terminal state");
    const auto* scores = static_cast<const float*>(input.ptr);
    int max_action = 0;
    for (int action : legal) max_action = std::max(max_action, action);
    if (input.shape[0] <= max_action)
      throw std::invalid_argument("logits do not cover every legal action");
    py::array_t<int32_t> actions(static_cast<py::ssize_t>(legal.size()));
    py::array_t<double> priors(static_cast<py::ssize_t>(legal.size()));
    auto* action_data = actions.mutable_data();
    auto* prior_data = priors.mutable_data();
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < legal.size(); ++i) {
      action_data[i] = legal[i];
      maximum = std::max(maximum, static_cast<double>(scores[legal[i]]));
    }
    double total = 0.0;
    for (size_t i = 0; i < legal.size(); ++i) {
      prior_data[i] = std::exp(static_cast<double>(scores[legal[i]]) - maximum);
      total += prior_data[i];
    }
    for (size_t i = 0; i < legal.size(); ++i) prior_data[i] /= total;
    return py::make_tuple(actions, priors);
  }, py::arg("state"), py::arg("logits"),
  "Return legal action indices and their normalized softmax priors.");
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
