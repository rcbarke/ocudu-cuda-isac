// SHA-256 for the hashes of an artifact and of a sealed output (Sprint 3.6, selector sense_core).
//
// The core hashes when a plan is created, to check a frozen artifact, and a sensor host hashes
// after its last record, to seal what it kept. No hash is taken between a record and a decision.
// FIPS 180-4, the standard library only. ASCII only.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace sense {

class sha256
{
public:
  sha256() { reset(); }
  void        reset();
  void        update(const void* data, std::size_t bytes);
  /// The digest as 64 lowercase hexadecimal digits. The object is reset.
  std::string hex();

private:
  void block(const std::uint8_t* chunk);

  std::uint32_t state[8];
  std::uint64_t total = 0;
  std::uint8_t  buffer[64];
  std::size_t   filled = 0;
};

std::string sha256_hex(const void* data, std::size_t bytes);
std::string sha256_hex(const std::string& text);

} // namespace sense
