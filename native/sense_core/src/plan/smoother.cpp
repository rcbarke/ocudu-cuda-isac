// The optional stage map_mean_kalman (Sprint 3.6, selector sense_core). ASCII only.

#include "plan/smoother.h"
#include <cstring>

namespace sense {

smoother::smoother(std::uint32_t maps, double process, double measurement, double speed, std::uint32_t rows_,
                   std::uint32_t cols_, const double cell_m[2], const double origin_m[2], double period) :
  capacity(maps), rows(rows_), cols(cols_), q(process), r(measurement), v0(speed), period_s(period)
{
  cell[0]   = cell_m[0];
  cell[1]   = cell_m[1];
  origin[0] = origin_m[0];
  origin[1] = origin_m[1];
  ring.assign(static_cast<std::size_t>(capacity) * rows * cols, 0.0);
  mean.assign(static_cast<std::size_t>(rows) * cols, 0.0);
  reset();
}

void smoother::reset()
{
  held      = 0;
  first     = 0;
  has_key   = false;
  has_state = false;
  std::memset(state, 0, sizeof(state));
  std::memset(covariance, 0, sizeof(covariance));
}

smoothed smoother::update(std::uint64_t epoch, std::uint64_t episode, std::uint64_t period, const float* map)
{
  const std::size_t cells = static_cast<std::size_t>(rows) * cols;
  // The reference: key != self.key or self.period is None or period <= self.period starts a history.
  const bool fresh = !has_key || epoch != key_epoch || episode != key_episode || period <= last_period;
  if (fresh) {
    reset();
  }
  const std::uint64_t steps = fresh ? 0 : period - last_period;
  has_key                   = true;
  key_epoch                 = epoch;
  key_episode               = episode;
  last_period               = period;
  // Keep the newest `capacity` maps.
  std::uint32_t at;
  if (held < capacity) {
    at = (first + held) % capacity;
    ++held;
  } else {
    at    = first;
    first = (first + 1) % capacity;
  }
  double* slot = ring.data() + static_cast<std::size_t>(at) * cells;
  for (std::size_t c = 0; c != cells; ++c) {
    slot[c] = static_cast<double>(map[c]);
  }
  // The mean, summed oldest first, as the reference sums along the first axis.
  for (std::size_t c = 0; c != cells; ++c) {
    mean[c] = 0.0;
  }
  for (std::uint32_t n = 0; n != held; ++n) {
    const double* from = ring.data() + static_cast<std::size_t>((first + n) % capacity) * cells;
    for (std::size_t c = 0; c != cells; ++c) {
      mean[c] += from[c];
    }
  }
  std::size_t best = 0;
  for (std::size_t c = 0; c != cells; ++c) {
    mean[c] /= static_cast<double>(held);
    if (mean[c] > mean[best]) {
      best = c; // the first of equal values stays
    }
  }
  smoothed made;
  made.maps        = held;
  made.row         = static_cast<std::uint32_t>(best / cols);
  made.col         = static_cast<std::uint32_t>(best % cols);
  made.x_m         = origin[0] + (made.col + 0.5) * cell[1];
  made.y_m         = origin[1] + (made.row + 0.5) * cell[0];
  made.probability = mean[best];
  const double z[2] = {made.x_m, made.y_m};
  if (!has_state) {
    state[0] = z[0];
    state[1] = z[1];
    state[2] = 0.0;
    state[3] = 0.0;
    std::memset(covariance, 0, sizeof(covariance));
    covariance[0][0] = r;
    covariance[1][1] = r;
    covariance[2][2] = v0 * v0;
    covariance[3][3] = v0 * v0;
    has_state        = true;
  } else {
    const double dt = static_cast<double>(steps) * period_s;
    // Predict: x = F x, P = F P F' + Q with F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]].
    const double move[4][4] = {{1, 0, dt, 0}, {0, 1, 0, dt}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    const double noise[4][4] = {{q * (dt * dt * dt / 3), 0, q * (dt * dt / 2), 0},
                                {0, q * (dt * dt * dt / 3), 0, q * (dt * dt / 2)},
                                {q * (dt * dt / 2), 0, q * dt, 0},
                                {0, q * (dt * dt / 2), 0, q * dt}};
    double predicted[4], moved[4][4], prior[4][4];
    for (unsigned i = 0; i != 4; ++i) {
      predicted[i] = 0.0;
      for (unsigned j = 0; j != 4; ++j) {
        predicted[i] += move[i][j] * state[j];
      }
    }
    for (unsigned i = 0; i != 4; ++i) {
      for (unsigned j = 0; j != 4; ++j) {
        moved[i][j] = 0.0;
        for (unsigned k = 0; k != 4; ++k) {
          moved[i][j] += move[i][k] * covariance[k][j];
        }
      }
    }
    for (unsigned i = 0; i != 4; ++i) {
      for (unsigned j = 0; j != 4; ++j) {
        prior[i][j] = 0.0;
        for (unsigned k = 0; k != 4; ++k) {
          prior[i][j] += moved[i][k] * move[j][k];
        }
        prior[i][j] += noise[i][j];
      }
    }
    // Update with the measurement of the position: S = H P H' + r I, K = P H' S^-1.
    const double s00 = prior[0][0] + r, s01 = prior[0][1], s10 = prior[1][0], s11 = prior[1][1] + r;
    const double determinant = s00 * s11 - s01 * s10;
    const double i00 = s11 / determinant, i01 = -s01 / determinant, i10 = -s10 / determinant, i11 = s00 / determinant;
    double       gain[4][2];
    for (unsigned i = 0; i != 4; ++i) {
      gain[i][0] = prior[i][0] * i00 + prior[i][1] * i10;
      gain[i][1] = prior[i][0] * i01 + prior[i][1] * i11;
    }
    const double residual[2] = {z[0] - predicted[0], z[1] - predicted[1]};
    for (unsigned i = 0; i != 4; ++i) {
      state[i] = predicted[i] + gain[i][0] * residual[0] + gain[i][1] * residual[1];
    }
    for (unsigned i = 0; i != 4; ++i) {
      for (unsigned j = 0; j != 4; ++j) {
        covariance[i][j] = prior[i][j] - (gain[i][0] * prior[0][j] + gain[i][1] * prior[1][j]);
      }
    }
  }
  made.kalman_x_m = state[0];
  made.kalman_y_m = state[1];
  made.kalman_vx  = state[2];
  made.kalman_vy  = state[3];
  return made;
}

} // namespace sense
