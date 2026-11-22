// SHA-256, FIPS 180-4 (Sprint 3.6, selector sense_core). ASCII only.

#include "core/sha256.h"
#include <cstring>

namespace sense {

namespace {

const std::uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline std::uint32_t rotr(std::uint32_t value, unsigned bits)
{
  return (value >> bits) | (value << (32 - bits));
}

} // namespace

void sha256::reset()
{
  static const std::uint32_t initial[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                           0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::memcpy(state, initial, sizeof(state));
  total  = 0;
  filled = 0;
}

void sha256::block(const std::uint8_t* chunk)
{
  std::uint32_t w[64];
  for (unsigned i = 0; i != 16; ++i) {
    w[i] = (static_cast<std::uint32_t>(chunk[4 * i]) << 24) | (static_cast<std::uint32_t>(chunk[4 * i + 1]) << 16) |
           (static_cast<std::uint32_t>(chunk[4 * i + 2]) << 8) | static_cast<std::uint32_t>(chunk[4 * i + 3]);
  }
  for (unsigned i = 16; i != 64; ++i) {
    const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i]                   = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (unsigned i = 0; i != 64; ++i) {
    const std::uint32_t s1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t ch  = (e & f) ^ (~e & g);
    const std::uint32_t t1  = h + s1 + ch + K[i] + w[i];
    const std::uint32_t s0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t t2  = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

void sha256::update(const void* data, std::size_t bytes)
{
  const std::uint8_t* from = static_cast<const std::uint8_t*>(data);
  total += bytes;
  if (filled != 0) {
    const std::size_t take = bytes < 64 - filled ? bytes : 64 - filled;
    std::memcpy(buffer + filled, from, take);
    filled += take;
    from += take;
    bytes -= take;
    if (filled != 64) {
      return;
    }
    block(buffer);
    filled = 0;
  }
  for (; bytes >= 64; from += 64, bytes -= 64) {
    block(from);
  }
  if (bytes != 0) {
    std::memcpy(buffer, from, bytes);
    filled = bytes;
  }
}

std::string sha256::hex()
{
  const std::uint64_t bits = total * 8;
  std::uint8_t        pad[72];
  std::size_t         length = filled < 56 ? 56 - filled : 120 - filled;
  std::memset(pad, 0, sizeof(pad));
  pad[0] = 0x80;
  for (unsigned i = 0; i != 8; ++i) {
    pad[length + i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
  }
  update(pad, length + 8);
  static const char digits[] = "0123456789abcdef";
  std::string       text(64, '0');
  for (unsigned i = 0; i != 8; ++i) {
    for (unsigned j = 0; j != 4; ++j) {
      const std::uint8_t byte = static_cast<std::uint8_t>(state[i] >> (24 - 8 * j));
      text[8 * i + 2 * j]     = digits[byte >> 4];
      text[8 * i + 2 * j + 1] = digits[byte & 15];
    }
  }
  reset();
  return text;
}

std::string sha256_hex(const void* data, std::size_t bytes)
{
  sha256 hash;
  hash.update(data, bytes);
  return hash.hex();
}

std::string sha256_hex(const std::string& text)
{
  return sha256_hex(text.data(), text.size());
}

} // namespace sense
