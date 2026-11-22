// A small JSON reader and writer for the manifest of an artifact and for the lines a core writes
// (Sprint 3.6, selector sense_core).
//
// The reader keeps the text of every scalar as the file holds it. The reference writes an
// artifact with sorted keys, and names the hash of a part as the SHA-256 of its canonical form:
// the same keys, sorted, with no space. From a file of the reference writer that form is the
// tokens of the file with the white space taken out, so canonical() rebuilds it exactly and the
// core checks every hash of a frozen artifact without formatting a number itself.
//
// The standard library only. ASCII only.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sense {
namespace json {

struct error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct value {
  enum kind_t { null_k, bool_k, number_k, string_k, array_k, object_k };
  kind_t      kind = null_k;
  std::string raw;     ///< Scalars: the token as the text holds it. A string keeps its quotes.
  std::string text;    ///< A string, decoded.
  double      number  = 0.0;
  bool        boolean = false;
  bool        integer = false; ///< A number written without a fraction and without an exponent.
  std::vector<value>                              items;
  std::vector<std::pair<std::string, value>>      members;  ///< In the order of the text.
  std::vector<std::string>                        raw_keys; ///< The key tokens, with their quotes.

  bool is_object() const { return kind == object_k; }
  bool is_array() const { return kind == array_k; }
  bool is_string() const { return kind == string_k; }
  bool is_number() const { return kind == number_k; }
  bool is_null() const { return kind == null_k; }

  /// The member of an object, or null when this is no object or has no such member.
  const value* find(const std::string& key) const;
  /// The member of an object. Throws error with the name of what is missing.
  const value& at(const std::string& key) const;
  /// Typed reads. Each throws error and names the member when the type is another.
  const std::string& string_at(const std::string& key) const;
  double             number_at(const std::string& key) const;
  std::int64_t       integer_at(const std::string& key) const;
  const value&       array_at(const std::string& key) const;
  const value&       object_at(const std::string& key) const;
};

/// Parses one JSON document. Throws error with the offset of the fault. A repeated key and a
/// constant that is no JSON (NaN, Infinity) are refused.
value parse(const std::string& text);

/// The canonical form of a value: objects with sorted keys, no white space, every scalar as its
/// text holds it. without leaves one member of the top object out, and only keeps the listed
/// members of the top object.
std::string canonical(const value& item, const char* without = nullptr, const std::vector<std::string>* only = nullptr);

// ---------------------------------------------------------------------------------------------- writer

/// Appends a JSON string with its quotes. Bytes outside printable ASCII are written as \u00XX.
void put_string(std::string& out, const std::string& text);
/// Appends a number that a reader takes for a float and that reads back to the same double.
void put_double(std::string& out, double number);
void put_int(std::string& out, std::int64_t number);
void put_uint(std::string& out, std::uint64_t number);
/// Appends 16 hexadecimal digits as a JSON string: an opaque key of a record.
void put_key(std::string& out, std::uint64_t key);

} // namespace json
} // namespace sense
