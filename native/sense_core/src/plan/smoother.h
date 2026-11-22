// The optional stage map_mean_kalman on the host (Sprint 3.6, selector sense_core).
//
// The twin of cusense.Smoother: the mean of the newest maps of one history, the cell of the
// largest value of that mean, and a constant-velocity Kalman filter with the centre of that cell
// as its measurement. A new epoch, a new episode key and a period that does not follow start a
// new history. It is an output of its own and changes no raw field of a decision. It runs in
// double precision on the CPU, after the publication time of the decision it belongs to.
//
// Every buffer is allocated when it is built. ASCII only.

#pragma once

#include <cstdint>
#include <vector>

namespace sense {

struct smoothed {
  std::uint32_t maps = 0, row = 0, col = 0;
  double        x_m = 0.0, y_m = 0.0, probability = 0.0;
  double        kalman_x_m = 0.0, kalman_y_m = 0.0, kalman_vx = 0.0, kalman_vy = 0.0;
};

class smoother
{
public:
  smoother(std::uint32_t maps, double process, double measurement, double speed, std::uint32_t rows, std::uint32_t cols,
           const double cell_m[2], const double origin_m[2], double period_s);

  /// The map of one scored window, [rows, cols] as float32, in the order of its history.
  smoothed update(std::uint64_t epoch, std::uint64_t episode, std::uint64_t period, const float* map);
  void     reset();

private:
  std::uint32_t       capacity, rows, cols;
  double              q, r, v0, cell[2], origin[2], period_s;
  std::vector<double> ring;  ///< The newest maps, oldest first from `first`.
  std::vector<double> mean;
  std::uint32_t       held = 0, first = 0;
  bool                has_key = false, has_state = false;
  std::uint64_t       key_epoch = 0, key_episode = 0, last_period = 0;
  double              state[4];
  double              covariance[4][4];
};

} // namespace sense
