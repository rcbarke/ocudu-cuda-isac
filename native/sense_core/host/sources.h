// Where the sensor loop takes its records from: the observation ring, and a consumer archive
// (Sprint 3.6, selector sense_core).
//
//   ring_source     the consumer side of one obs-ring/v1 ring. A record is read in place, in the
//                   slot the producer filled. The ring has cumulative credits, so a view carries
//                   no release of its own: the loop asks the plan for the oldest record it still
//                   holds and releases every record below it.
//   archive_source  every *.obs file of a directory, in the order of its name, which is the order
//                   of epoch and sequence. It never skips: an offline replay scores every window.
//   record_keeper   the records a live sensor read, kept as obs_tail keeps them: copied into
//                   memory that was allocated before the sensor attached, written after the last.
//
// sense_host and the unit target share these. ASCII only.

#pragma once

#include "obs/ring.h"
#include "sensor_loop.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace sense {

/// The records a live sensor read, kept as obs_tail keeps them: copied into memory that was
/// allocated before the sensor attached, and written after the last record.
struct record_keeper {
  struct kept {
    std::uint64_t sequence = 0, epoch = 0, opportunity = 0;
    std::uint32_t depth = 0, bytes = 0;
    std::int64_t  t_publish_ns = 0, t_read_ns = 0;
    bool          intact = false;
  };
  std::size_t               record_max = 0;
  std::vector<kept>         records;
  std::vector<std::uint8_t> arena;
  std::size_t               count = 0, not_kept = 0;

  record_keeper(std::size_t capacity, std::size_t record_bytes) :
    record_max(record_bytes), records(capacity), arena(capacity * record_bytes, 0)
  {
  }
};

class ring_source final : public sense::record_source
{
public:
  ring_source(obs::consumer& reader_, record_keeper* keeper_) :
    reader(reader_), keeper(keeper_), entries(reader_.header().slot_count)
  {
  }

  result next(sense_view& view, std::int64_t limit_ns) override
  {
    obs::record_view       found;
    const obs::read_result got = reader.next(found, std::chrono::nanoseconds(limit_ns));
    if (got == obs::read_result::closed) {
      return result::closed;
    }
    if (got == obs::read_result::none) {
      return result::none;
    }
    if (got == obs::read_result::stale) {
      return result::stale;
    }
    entry& held = entries[found.sequence & (entries.size() - 1)];
    held.reader = &reader;
    held.view   = found;
    view.record = found.header;
    view.payload = found.payload;
    view.mask    = found.mask;
    view.memory  = SENSE_MEMORY_HOST_SHARED;
    view.device  = -1;
    view.stream  = nullptr;
    view.intact  = &ring_source::intact;
    view.release = nullptr; // the ring has cumulative credits: the loop asks the plan what it holds
    view.owner   = &held;
    if (keeper != nullptr) {
      keep(found);
    }
    return result::record;
  }

  std::uint64_t pending() override
  {
    const std::uint64_t published = reader.header().published.load(std::memory_order_acquire);
    const std::uint64_t at        = reader.position();
    return published > at ? published - at : 0;
  }
  void          release(std::uint64_t sequence) override { reader.release(sequence); }
  std::uint64_t position() override { return reader.position(); }
  bool          live() const override { return true; }

private:
  struct entry {
    obs::consumer*   reader = nullptr;
    obs::record_view view;
  };

  static int intact(void* owner)
  {
    entry* held = static_cast<entry*>(owner);
    return held->reader->intact(held->view) ? 1 : 0;
  }

  void keep(const obs::record_view& found)
  {
    if (keeper->count == keeper->records.size()) {
      ++keeper->not_kept;
      return;
    }
    const obs::segment_header& ring   = reader.header();
    const obs::record_header&  header = *found.header;
    record_keeper::kept&       item   = keeper->records[keeper->count];
    const std::size_t host = header.host_bytes <= ring.host_capacity ? header.host_bytes : 0;
    const std::size_t bits = header.mask_bytes <= ring.mask_capacity ? header.mask_bytes : 0;
    const std::size_t data = header.payload_bytes <= ring.payload_capacity ? header.payload_bytes : 0;
    const std::size_t all  = ring.record_header_bytes + host + bits + data;
    if (all > keeper->record_max) {
      ++keeper->not_kept;
      return;
    }
    item.sequence     = found.sequence;
    item.epoch        = header.epoch;
    item.opportunity  = header.opportunity;
    item.depth        = found.depth;
    item.t_publish_ns = header.t_publish_ns;
    item.t_read_ns    = obs::monotonic_ns();
    item.bytes        = static_cast<std::uint32_t>(all);
    std::uint8_t* to  = keeper->arena.data() + keeper->count * keeper->record_max;
    std::memcpy(to, found.header, ring.record_header_bytes);
    std::memcpy(to + ring.record_header_bytes, found.host, host);
    std::memcpy(to + ring.record_header_bytes + host, found.mask, bits);
    std::memcpy(to + ring.record_header_bytes + host + bits, found.payload, data);
    // The copy is the record only when the slot was not rewritten meanwhile. This asks the ring
    // once more than the plan does, and a rewrite is counted there.
    item.intact = found.prefix->generation.load(std::memory_order_acquire) == found.generation;
    ++keeper->count;
  }

  obs::consumer&     reader;
  record_keeper*     keeper;
  std::vector<entry> entries;
};

/// The records of a consumer archive: every *.obs file of a directory, in the order of its name,
/// which is the order of epoch and sequence.
class archive_source final : public sense::record_source
{
public:
  explicit archive_source(const std::string& directory)
  {
    DIR* listing = ::opendir(directory.c_str());
    if (listing == nullptr) {
      throw std::runtime_error("cannot list the archive " + directory);
    }
    std::vector<std::string> names;
    while (const dirent* found = ::readdir(listing)) {
      const std::string name = found->d_name;
      if (name.size() > 4 && name.compare(name.size() - 4, 4, ".obs") == 0) {
        names.push_back(name);
      }
    }
    ::closedir(listing);
    std::sort(names.begin(), names.end());
    for (const std::string& name : names) {
      std::FILE* file = std::fopen((directory + "/" + name).c_str(), "rb");
      if (file == nullptr) {
        throw std::runtime_error("cannot read " + directory + "/" + name);
      }
      const std::size_t start = bytes.size();
      char              chunk[65536];
      std::size_t       got;
      while ((got = std::fread(chunk, 1, sizeof(chunk), file)) != 0) {
        bytes.insert(bytes.end(), chunk, chunk + got);
      }
      std::fclose(file);
      const std::size_t length = bytes.size() - start;
      obs::record_header header;
      if (length < sizeof(header)) {
        throw std::runtime_error(name + " holds no record header");
      }
      std::memcpy(&header, bytes.data() + start, sizeof(header));
      if (header.schema != obs::RECORD_SCHEMA || header.header_bytes != obs::RECORD_HEADER_BYTES ||
          length != sizeof(header) + header.host_bytes + header.mask_bytes + static_cast<std::size_t>(header.payload_bytes)) {
        throw std::runtime_error(name + " is not a whole obs/v1 record");
      }
      offsets.push_back(start);
    }
  }

  result next(sense_view& view, std::int64_t) override
  {
    if (at == offsets.size()) {
      return result::closed;
    }
    const std::uint8_t* record = bytes.data() + offsets[at++];
    obs::record_header  header;
    std::memcpy(&header, record, sizeof(header));
    view.record  = record;
    view.mask    = record + sizeof(header) + header.host_bytes;
    view.payload = record + sizeof(header) + header.host_bytes + header.mask_bytes;
    view.memory  = SENSE_MEMORY_HOST_SHARED;
    view.device  = -1;
    view.stream  = nullptr;
    view.intact  = nullptr;
    view.release = nullptr;
    view.owner   = nullptr;
    return result::record;
  }
  std::uint64_t pending() override { return 0; }
  void          release(std::uint64_t) override {}
  std::uint64_t position() override { return at; }
  bool          live() const override { return false; }
  std::size_t   records() const { return offsets.size(); }

private:
  std::vector<std::uint8_t> bytes;
  std::vector<std::size_t>  offsets;
  std::size_t               at = 0;
};

} // namespace sense
