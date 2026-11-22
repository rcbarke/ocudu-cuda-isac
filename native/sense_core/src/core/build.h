// What this build of the core is: its sources, its device, its libraries and its registry
// (Sprint 3.6, selector sense_core). The values come from the header that CMake generates when
// the build is configured. ASCII only.

#pragma once

#include <string>

namespace sense {

/// The hash that names the sources this core was built from. scripts/native_core.py computes
/// the same value from the tree and refuses a build whose binary names another.
const char* core_sources_sha256();

/// One registry entry.
struct unit {
  const char* stage;
  const char* entry;
  const char* sha256;   ///< Of the source file that holds the unit.
  const char* source;   ///< That file, relative to native/sense_core.
  const char* runs_on;  ///< "device", "engine" or "host".
  const char* precision;
};

/// The registry. The last entry has a null stage.
const unit* registry();
/// The unit of a stage and an entry, or null: plan creation refuses the name.
const unit* find_unit(const char* stage, const std::string& entry);

const std::string& build_json();
const std::string& registry_json();
const std::string& tolerance_json();

} // namespace sense
