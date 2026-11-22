// JSON reader and writer of the sensing core (Sprint 3.6, selector sense_core). ASCII only.

#include "core/json.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sense {
namespace json {

namespace {

struct reader {
  const std::string& text;
  std::size_t        at = 0;
  unsigned           depth = 0;

  [[noreturn]] void fail(const std::string& what) const
  {
    throw error("JSON: " + what + " at offset " + std::to_string(at));
  }

  void space()
  {
    while (at < text.size() && (text[at] == ' ' || text[at] == '\n' || text[at] == '\r' || text[at] == '\t')) {
      ++at;
    }
  }

  static void utf8(std::string& out, unsigned code)
  {
    if (code < 0x80) {
      out += static_cast<char>(code);
    } else if (code < 0x800) {
      out += static_cast<char>(0xc0 | (code >> 6));
      out += static_cast<char>(0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
      out += static_cast<char>(0xe0 | (code >> 12));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3f));
      out += static_cast<char>(0x80 | (code & 0x3f));
    } else {
      out += static_cast<char>(0xf0 | (code >> 18));
      out += static_cast<char>(0x80 | ((code >> 12) & 0x3f));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3f));
      out += static_cast<char>(0x80 | (code & 0x3f));
    }
  }

  unsigned hex4()
  {
    if (at + 4 > text.size()) {
      fail("a short \\u escape");
    }
    unsigned code = 0;
    for (unsigned i = 0; i != 4; ++i) {
      const char c = text[at++];
      code <<= 4;
      if (c >= '0' && c <= '9') {
        code |= static_cast<unsigned>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        code |= static_cast<unsigned>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        code |= static_cast<unsigned>(c - 'A' + 10);
      } else {
        fail("a \\u escape that is not hexadecimal");
      }
    }
    return code;
  }

  /// Reads a string token. Returns the decoded text and stores the raw token.
  std::string string(std::string& raw)
  {
    const std::size_t start = at;
    ++at; // the opening quote
    std::string out;
    for (;;) {
      if (at >= text.size()) {
        fail("a string without its closing quote");
      }
      const char c = text[at++];
      if (c == '"') {
        break;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        fail("a control character in a string");
      }
      if (c != '\\') {
        out += c;
        continue;
      }
      if (at >= text.size()) {
        fail("a string that ends in an escape");
      }
      const char e = text[at++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned code = hex4();
          if (code >= 0xd800 && code < 0xdc00 && at + 1 < text.size() && text[at] == '\\' && text[at + 1] == 'u') {
            at += 2;
            const unsigned low = hex4();
            if (low < 0xdc00 || low > 0xdfff) {
              fail("a surrogate without its pair");
            }
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
          }
          utf8(out, code);
          break;
        }
        default: fail("an unknown escape");
      }
    }
    raw = text.substr(start, at - start);
    return out;
  }

  value number()
  {
    const std::size_t start = at;
    bool              whole = true;
    if (at < text.size() && text[at] == '-') {
      ++at;
    }
    if (at >= text.size() || text[at] < '0' || text[at] > '9') {
      fail("a number without a digit");
    }
    if (text[at] == '0' && at + 1 < text.size() && text[at + 1] >= '0' && text[at + 1] <= '9') {
      fail("a number with a leading zero");
    }
    while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
      ++at;
    }
    if (at < text.size() && text[at] == '.') {
      whole = false;
      ++at;
      if (at >= text.size() || text[at] < '0' || text[at] > '9') {
        fail("a fraction without a digit");
      }
      while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
        ++at;
      }
    }
    if (at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
      whole = false;
      ++at;
      if (at < text.size() && (text[at] == '+' || text[at] == '-')) {
        ++at;
      }
      if (at >= text.size() || text[at] < '0' || text[at] > '9') {
        fail("an exponent without a digit");
      }
      while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
        ++at;
      }
    }
    value made;
    made.kind    = value::number_k;
    made.raw     = text.substr(start, at - start);
    made.integer = whole;
    errno        = 0;
    made.number  = std::strtod(made.raw.c_str(), nullptr);
    if (!std::isfinite(made.number)) {
      fail("a number that no double holds");
    }
    return made;
  }

  void literal(const char* word)
  {
    const std::size_t length = std::strlen(word);
    if (text.compare(at, length, word) != 0) {
      fail("an unknown literal");
    }
    at += length;
  }

  value any()
  {
    space();
    if (at >= text.size()) {
      fail("the end of the text where a value is expected");
    }
    if (++depth > 64) {
      fail("more than 64 levels");
    }
    value       made;
    const char  c = text[at];
    if (c == '{') {
      made.kind = value::object_k;
      ++at;
      space();
      if (at < text.size() && text[at] == '}') {
        ++at;
      } else {
        for (;;) {
          space();
          if (at >= text.size() || text[at] != '"') {
            fail("a member without a quoted key");
          }
          std::string raw;
          std::string key = string(raw);
          for (const auto& member : made.members) {
            if (member.first == key) {
              fail("the repeated key \"" + key + "\"");
            }
          }
          space();
          if (at >= text.size() || text[at] != ':') {
            fail("a key without a colon");
          }
          ++at;
          made.members.emplace_back(std::move(key), any());
          made.raw_keys.push_back(std::move(raw));
          space();
          if (at < text.size() && text[at] == ',') {
            ++at;
            continue;
          }
          if (at < text.size() && text[at] == '}') {
            ++at;
            break;
          }
          fail("an object without its closing brace");
        }
      }
    } else if (c == '[') {
      made.kind = value::array_k;
      ++at;
      space();
      if (at < text.size() && text[at] == ']') {
        ++at;
      } else {
        for (;;) {
          made.items.push_back(any());
          space();
          if (at < text.size() && text[at] == ',') {
            ++at;
            continue;
          }
          if (at < text.size() && text[at] == ']') {
            ++at;
            break;
          }
          fail("an array without its closing bracket");
        }
      }
    } else if (c == '"') {
      made.kind = value::string_k;
      made.text = string(made.raw);
    } else if (c == 't') {
      literal("true");
      made.kind    = value::bool_k;
      made.boolean = true;
      made.raw     = "true";
    } else if (c == 'f') {
      literal("false");
      made.kind = value::bool_k;
      made.raw  = "false";
    } else if (c == 'n') {
      literal("null");
      made.raw = "null";
    } else {
      made = number();
    }
    --depth;
    return made;
  }
};

void canonical_into(std::string& out, const value& item, const char* without, const std::vector<std::string>* only)
{
  switch (item.kind) {
    case value::object_k: {
      std::vector<std::size_t> order;
      for (std::size_t i = 0; i != item.members.size(); ++i) {
        const std::string& key = item.members[i].first;
        if (without != nullptr && key == without) {
          continue;
        }
        if (only != nullptr && std::find(only->begin(), only->end(), key) == only->end()) {
          continue;
        }
        order.push_back(i);
      }
      std::sort(order.begin(), order.end(),
                [&](std::size_t a, std::size_t b) { return item.members[a].first < item.members[b].first; });
      out += '{';
      for (std::size_t n = 0; n != order.size(); ++n) {
        if (n != 0) {
          out += ',';
        }
        out += item.raw_keys[order[n]];
        out += ':';
        canonical_into(out, item.members[order[n]].second, nullptr, nullptr);
      }
      out += '}';
      break;
    }
    case value::array_k:
      out += '[';
      for (std::size_t n = 0; n != item.items.size(); ++n) {
        if (n != 0) {
          out += ',';
        }
        canonical_into(out, item.items[n], nullptr, nullptr);
      }
      out += ']';
      break;
    default: out += item.raw; break;
  }
}

} // namespace

const value* value::find(const std::string& key) const
{
  if (kind != object_k) {
    return nullptr;
  }
  for (const auto& member : members) {
    if (member.first == key) {
      return &member.second;
    }
  }
  return nullptr;
}

const value& value::at(const std::string& key) const
{
  const value* found = find(key);
  if (found == nullptr) {
    throw error("the member \"" + key + "\" is missing");
  }
  return *found;
}

const std::string& value::string_at(const std::string& key) const
{
  const value& found = at(key);
  if (!found.is_string()) {
    throw error("the member \"" + key + "\" is no string");
  }
  return found.text;
}

double value::number_at(const std::string& key) const
{
  const value& found = at(key);
  if (!found.is_number()) {
    throw error("the member \"" + key + "\" is no number");
  }
  return found.number;
}

std::int64_t value::integer_at(const std::string& key) const
{
  const value& found = at(key);
  if (!found.is_number() || !found.integer || std::fabs(found.number) > 9.0e15) {
    throw error("the member \"" + key + "\" is no integer");
  }
  return static_cast<std::int64_t>(found.number);
}

const value& value::array_at(const std::string& key) const
{
  const value& found = at(key);
  if (!found.is_array()) {
    throw error("the member \"" + key + "\" is no array");
  }
  return found;
}

const value& value::object_at(const std::string& key) const
{
  const value& found = at(key);
  if (!found.is_object()) {
    throw error("the member \"" + key + "\" is no object");
  }
  return found;
}

value parse(const std::string& text)
{
  reader from{text};
  value  made = from.any();
  from.space();
  if (from.at != text.size()) {
    from.fail("text after the document");
  }
  return made;
}

std::string canonical(const value& item, const char* without, const std::vector<std::string>* only)
{
  std::string out;
  canonical_into(out, item, without, only);
  return out;
}

void put_string(std::string& out, const std::string& text)
{
  out += '"';
  for (const char c : text) {
    const unsigned char byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\t') {
      out += "\\t";
    } else if (byte < 0x20 || byte > 0x7e) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(byte));
      out += escaped;
    } else {
      out += c;
    }
  }
  out += '"';
}

void put_double(std::string& out, double number)
{
  if (!std::isfinite(number)) {
    out += "null"; // JSON has no such number: the caller decides before it writes one
    return;
  }
  // The shortest of a few precisions that reads back to the same double, so that a line stays
  // readable. A value that came from a float32 needs all 17 digits.
  static const int precisions[] = {7, 15, 16, 17};
  char             text[40];
  for (const int digits : precisions) {
    std::snprintf(text, sizeof(text), "%.*g", digits, number);
    if (digits == 17 || std::strtod(text, nullptr) == number) {
      break;
    }
  }
  out += text;
  if (std::strpbrk(text, ".eEn") == nullptr) {
    out += ".0"; // a reader must take the value for a float
  }
}

void put_int(std::string& out, std::int64_t number)
{
  out += std::to_string(number);
}

void put_uint(std::string& out, std::uint64_t number)
{
  out += std::to_string(number);
}

void put_key(std::string& out, std::uint64_t key)
{
  char text[24];
  std::snprintf(text, sizeof(text), "\"%016llx\"", static_cast<unsigned long long>(key));
  out += text;
}

} // namespace json
} // namespace sense
