// The JSON object that the command line prints for an axmp::Error.  Internal: not installed.
#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "axionmultiphysics/core.hpp"

namespace axmp {

// The error's details as a JSON object with at least one member, or null when there are none to report: no details,
// text that is not strict JSON (NaN and Infinity are not), JSON that is not an object, or JSON nested deeper than
// kMaxErrorDetailsDepth (the object itself is level 1).  Never throws.  The command line and the Python exception
// both report what this returns, so they cannot disagree.
constexpr int kMaxErrorDetailsDepth = 64;

// The deepest nesting of arrays and objects in the text, found without recursion and without parsing; brackets inside
// strings do not count.  Stops as soon as the nesting passes the limit.
inline bool error_details_too_deep(const std::string& text) {
  int depth = 0;
  bool in_string = false;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (c == '\\') ++i;
      else if (c == '"') in_string = false;
    } else if (c == '"') {
      in_string = true;
    } else if (c == '{' || c == '[') {
      if (++depth > kMaxErrorDetailsDepth) return true;
    } else if (c == '}' || c == ']') {
      if (depth > 0) --depth;
    }
  }
  return false;
}

inline nlohmann::json error_details(const Error& e) {
  if (e.details().empty() || error_details_too_deep(e.details())) return nullptr;
  nlohmann::json d = nlohmann::json::parse(e.details(), nullptr, false);
  if (d.is_object() && !d.empty()) return d;
  return nullptr;
}

// {"ok": false, "error", "where", "hint"}, plus "code" when the error has one and "details" when error_details() has
// an object.  Never throws.
inline nlohmann::json error_json(const Error& e) {
  nlohmann::json j = {{"ok", false}, {"error", e.message()}, {"where", e.where()}, {"hint", e.hint()}};
  if (!e.code().empty()) j["code"] = e.code();
  nlohmann::json d = error_details(e);
  if (!d.is_null()) j["details"] = std::move(d);
  return j;
}

}  // namespace axmp
