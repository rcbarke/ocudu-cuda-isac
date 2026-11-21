// One-off diagnostic, not part of the tree: where the time of one observation and of one grid fill
// goes, on the storage model of the grid (no OCUDU). Demonstration profile: 4 receive planes,
// 14 symbols, 612 subcarriers, 2 reference symbols of 306 pilots. Best of 200 repetitions each.
//
// RELOCATE THIS SCRIPT TO build/ for runtime (build/ is gitignored, thus the file is hosted in scripts/ to commit into the repo)
//
// From the repository root:
//   A=native/ocudu_adapter
//   g++ -std=c++17 -O2 -ffp-contract=off -I$A/observer -I$A/replay build/tap_timing.cpp
//       $A/observer/observer_core.cpp $A/replay/replay_core.cpp -o build/tap_timing
//   (one command line), then: build/tap_timing
// ASCII only.

#include "observer_core.h"
#include "replay_core.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace isac_observer;

template <typename F>
static double best_us(F&& work)
{
  double best = 1e18;
  for (int repeat = 0; repeat != 200; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    work();
    best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
  }
  return best;
}

int main()
{
  const unsigned planes = 4, symbols = 14, subcarriers = 612;
  isac_replay::grid_shape shape;
  shape.rx_ports    = planes;
  shape.tx_ports    = 1;
  shape.symbols     = symbols;
  shape.subcarriers = subcarriers;

  // ---- One grid fill: the channel application, then the readback digest that commit takes.
  std::vector<cf32>       x(shape.plane(), cf32(0.3F, -0.9F));
  std::vector<cf32>       h(planes * shape.plane(), cf32(3e-4F, 1e-4F));
  std::vector<cf32>       w(planes * shape.plane(), cf32(1e-6F, -2e-6F));
  isac_replay::model_grid scratch(planes, symbols, subcarriers);
  const double apply_us = best_us([&] {
    scratch.clear();
    isac_replay::apply(shape, x.data(), h.data(), w.data(), 2942.4, scratch);
  });
  const double readback_us = best_us([&] { isac_replay::readback_sha256(scratch, shape); });

  // ---- One observation.
  isac_replay::model_grid grid(planes, symbols, subcarriers);
  isac_replay::o1_pool    pool({&grid});
  isac_replay::ticket     key;
  pool.try_acquire(key);
  std::vector<cf32> row(subcarriers, cf32(0.5F, -0.25F));
  for (unsigned plane = 0; plane != planes; ++plane) {
    for (unsigned symbol = 0; symbol != symbols; ++symbol) {
      pool.writer(key).put(plane, symbol, row.data(), subcarriers);
    }
  }
  isac_replay::o1_descriptor descriptor;
  descriptor.grant_id            = "grant";
  descriptor.grant_record_sha256 = "record";
  descriptor.slot                = 7;
  descriptor.shape               = shape;
  pool.commit(key, descriptor);

  source_spec spec;
  spec.grant_id               = "grant";
  spec.grant_record_sha256    = "record";
  spec.source                 = "pusch_dmrs";
  spec.ports                  = {0};
  spec.grid_symbols           = symbols;
  spec.grid_subcarriers       = subcarriers;
  spec.scs_khz                = 30;
  spec.slot                   = 7;
  spec.slot_symbols           = {2, 11};
  spec.symbol_useful_start_tc = {145920, 777216};
  spec.tc_per_second          = TC_PER_SECOND;
  spec.amplitude              = 1.4125375446227544;
  spec.antenna_map            = {0, 1, 2, 3};
  for (unsigned symbol : {2U, 11U}) {
    for (unsigned k = 0; k != subcarriers; k += 2) {
      pilot item;
      item.symbol     = symbol;
      item.subcarrier = k;
      spec.index.push_back(item);
      spec.reference.push_back(cf32(0.99881488F, -0.99881488F));
    }
  }
  const pilot_source source(spec);
  observer           tap("timing", 1);
  tap.bind(spec);
  slot_context where;
  where.opportunity_id = "rz-22963ee9d985.0000";
  where.episode_id     = "rz-22963ee9d985";
  where.slot           = 7;
  const isac_replay::o1_view view = pool.view(key, "record");

  observation made;
  const double estimate_us = best_us([&] {
    estimate(source, subcarriers, [&](unsigned plane, unsigned symbol, cf32* out, unsigned count) { view.get(plane, symbol, out, count); },
             made);
  });
  const double payload_us = best_us([&] { isac_replay::sha256_hex(made.values.data(), made.payload_bytes()); });
  const double small_us   = best_us([&] {
    isac_replay::sha256_hex(made.mask.data(), made.mask.size());
    opaque_key(where.episode_id);
    opaque_key(where.opportunity_id);
    opaque_key(std::string(64, 'a'));
  });
  const double digest_us  = best_us([&] { view.digest(); });
  const double observe_us = best_us([&] { tap.observe(view, where); });

  std::printf("grid fill, channel application only          %9.1f us\n", apply_us);
  std::printf("grid fill, readback digest (274176 bytes)    %9.1f us\n", readback_us);
  std::printf("observation, estimate only (2448 values)     %9.1f us\n", estimate_us);
  std::printf("observation, digest of the payload (19584 B) %9.1f us\n", payload_us);
  std::printf("observation, digest of the mask and 3 keys   %9.1f us\n", small_us);
  std::printf("observation, digest of the grid after read   %9.1f us\n", digest_us);
  std::printf("observation, whole observe() as accepted     %9.1f us\n", observe_us);
  std::printf("observation without any hash (projection)    %9.1f us\n", observe_us - digest_us - payload_us - small_us);
  return 0;
}
