// The pinned OCUDU resource grid as the receiver grid of a replay (Sprint 2.3, selector grid_apply).
//
// The channel is applied into the grid's writer, and the stored values come back through its
// reader, the route by which the pinned PUSCH processors read their input. Header only. It needs
// the headers of the locked OCUDU source and a target that links ocudu_pin_fixture. ASCII only.

#pragma once

#include "replay_core.h"
#include "ocudu/phy/support/resource_grid.h"
#include "ocudu/phy/support/resource_grid_reader.h"
#include "ocudu/phy/support/resource_grid_writer.h"
#include "ocudu/phy/support/support_factories.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace isac_replay {

class pinned_grid final : public grid_sink
{
public:
  pinned_grid(ocudu::resource_grid_factory& factory, unsigned planes_, unsigned symbols_, unsigned subcarriers_) :
    grid(factory.create(planes_, symbols_, subcarriers_))
  {
    if (!grid) {
      throw std::runtime_error("the pinned resource grid factory returned no grid");
    }
    grid->set_all_zero();
  }

  unsigned planes() const override { return grid->get_reader().get_nof_ports(); }
  unsigned symbols() const override { return grid->get_reader().get_nof_symbols(); }
  unsigned subcarriers() const override { return grid->get_reader().get_nof_subc(); }

  void clear() override { grid->set_all_zero(); }

  void put(unsigned plane, unsigned symbol, const cf32* row, unsigned count) override
  {
    check(plane, symbol, count, "write");
    grid->get_writer().put(plane, symbol, 0, ocudu::span<const ocudu::cf_t>(row, count));
  }

  void get(unsigned plane, unsigned symbol, cf32* row, unsigned count) const override
  {
    check(plane, symbol, count, "read");
    grid->get_reader().get(ocudu::span<ocudu::cf_t>(row, count), plane, symbol, 0);
  }

  std::string storage() const override { return "complex bf16 in the pinned resource grid, widened without change"; }

  /// The grid itself, for the receiver that reads it (Sprint 2.4).
  ocudu::resource_grid& ocudu_grid() { return *grid; }

private:
  void check(unsigned plane, unsigned symbol, unsigned count, const char* action) const
  {
    if (plane >= planes() || symbol >= symbols() || count == 0 || count > subcarriers()) {
      throw state_error(std::string("a ") + action + " leaves the pinned resource grid");
    }
  }

  std::unique_ptr<ocudu::resource_grid> grid;
};

} // namespace isac_replay
