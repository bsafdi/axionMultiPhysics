// The error object's two optional fields, and the JSON the command line prints from an error.
//
//   1. an error without them: what(), message(), where() and hint() as they stand; code() and details() empty.
//   2. an error with them: what() is byte-identical to the same error without them (compose() takes the message,
//      the pointer and the hint only), so every message of the library and every test that asserts one stands.
//      A code passed in the hint's place shows up in what(): that is what this comparison catches.
//   3. SolverFailure, which inherits the constructor, carries the fields too and is still caught as an Error.
//   4. AXMP_REQUIRE still forwards one, two and three arguments.
//   5. error_json(): the four keys {"ok", "error", "where", "hint"} for an error without a code, with the exact
//      text the command line prints (nlohmann's object sorts its keys, so the dump is fully determined), and the
//      keys "code" and "details" added only when they are not empty. Malformed details text is dropped, not thrown:
//      an error report must not fail while it is being written. NaN and Infinity are not JSON and count as malformed.
//   6. with_hint_note(), through which a model rethrows a solver failure with the mesh check appended: the hint gains
//      the note and the message, the pointer, the code and the details are the caught error's.
//
// The two fields and the header are detected at compile time, so this file compiles whether or not they exist and
// reports what is missing as a failure instead of breaking the build.
#include <cstdio>
#include <string>
#include <type_traits>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"

#if __has_include("../src/error_json.hpp")
#include "../src/error_json.hpp"
#define AXMP_TEST_HAVE_ERROR_JSON 1
#else
#define AXMP_TEST_HAVE_ERROR_JSON 0
#endif

static int failures = 0;
#define CHECK(cond, ...)                 \
  do {                                   \
    if (!(cond)) {                       \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                 \
      ++failures;                        \
    }                                    \
  } while (0)

namespace {

using nlohmann::json;

// True when E can be built with (what, where, hint, code, details) and reports the last two.
template <class E, class = void>
struct coded : std::false_type {};
template <class E>
struct coded<E, decltype(void(E(std::string(), std::string(), std::string(), std::string(), std::string()).code()),
                         void(E(std::string(), std::string(), std::string(), std::string(), std::string()).details()))>
    : std::true_type {};

const char* const kWhat = "boundary 'inlet' selects nothing";
const char* const kWhere = "/boundaries/inlet/where";
const char* const kHint = "no boundary edge of 'channel' lies on x = 0.";
const char* const kCode = "boundary.empty_selection";
const char* const kDetails = R"({"candidates":[{"where":{"x":0.0005}}],"selector":{"of":"channel"}})";

std::string composed() {
  return std::string(kWhat) + " [at " + kWhere + "] Hint: " + kHint;
}

// ---------------------------------------------------------------- 1, 4: what every build already promises
void unchanged_error() {
  const axmp::Error plain(kWhat);
  CHECK(std::string(plain.what()) == kWhat, "what() of a bare error: %s", plain.what());
  CHECK(plain.message() == kWhat && plain.where().empty() && plain.hint().empty(), "members of a bare error");

  const axmp::Error full(kWhat, kWhere, kHint);
  CHECK(std::string(full.what()) == composed(), "what() of an error with a pointer and a hint: %s", full.what());
  CHECK(full.message() == kWhat, "message()");
  CHECK(full.where() == kWhere, "where()");
  CHECK(full.hint() == kHint, "hint()");

  try {
    AXMP_REQUIRE(false, kWhat, kWhere, kHint);
    CHECK(false, "AXMP_REQUIRE did not throw");
  } catch (const axmp::Error& e) {
    CHECK(std::string(e.what()) == composed(), "AXMP_REQUIRE with three arguments: %s", e.what());
  }
  try {
    AXMP_REQUIRE(false, kWhat, kWhere);
    CHECK(false, "AXMP_REQUIRE did not throw");
  } catch (const axmp::Error& e) {
    CHECK(std::string(e.what()) == std::string(kWhat) + " [at " + kWhere + "]", "two arguments: %s", e.what());
  }
  try {
    AXMP_REQUIRE(false, kWhat);
    CHECK(false, "AXMP_REQUIRE did not throw");
  } catch (const axmp::Error& e) {
    CHECK(std::string(e.what()) == kWhat, "one argument: %s", e.what());
  }
}

// ---------------------------------------------------------------- 2, 3: the two fields
template <class E>
std::enable_if_t<coded<E>::value> coded_error() {
  const E e(kWhat, kWhere, kHint, kCode, kDetails);
  CHECK(std::string(e.what()) == composed(),
        "a code and details reach the message: %s\n  expected: %s", e.what(), composed().c_str());
  CHECK(e.message() == kWhat && e.where() == kWhere && e.hint() == kHint, "the first three members disagree");
  CHECK(e.code() == kCode, "code(): '%s'", e.code().c_str());
  CHECK(e.details() == kDetails, "details(): '%s'", e.details().c_str());

  const E none(kWhat, kWhere, kHint);
  CHECK(none.code().empty(), "code() of an error that was given none: '%s'", none.code().c_str());
  CHECK(none.details().empty(), "details() of an error that was given none: '%s'", none.details().c_str());
  CHECK(std::string(none.what()) == std::string(e.what()), "the message depends on the code and the details");

  // a code is not a hint: an error given only a code reads exactly like an error given nothing
  const E only_code(kWhat, "", "", kCode);
  CHECK(std::string(only_code.what()) == kWhat, "a code reached the message: %s", only_code.what());
  CHECK(only_code.code() == kCode && only_code.details().empty(), "a code alone");
}

template <class E>
std::enable_if_t<!coded<E>::value> coded_error() {
  CHECK(false,
        "axmp::Error has no code() and details() reachable through "
        "Error(what, where, hint, code, details): the two optional fields are not implemented");
}

// A solver failure is an Error: it inherits the constructor, keeps the fields and is caught as one.
// `Base` is the type the catch clause names; it is a template parameter so that this file still compiles
// while axmp::Error has no code().
template <class F, class Base = axmp::Error>
std::enable_if_t<coded<F>::value> coded_solver_failure() {
  try {
    throw F(kWhat, kWhere, kHint, kCode, kDetails);
  } catch (const Base& caught) {
    CHECK(std::string(caught.what()) == composed(), "a solver failure's message: %s", caught.what());
    CHECK(caught.code() == kCode && caught.details() == kDetails, "a solver failure's fields");
  }
}

template <class F, class Base = axmp::Error>
std::enable_if_t<!coded<F>::value> coded_solver_failure() {
  CHECK(false, "axmp::SolverFailure does not inherit the constructor that takes a code and details");
}

// ---------------------------------------------------------------- 6: the rethrow with a note
template <class F>
std::enable_if_t<coded<F>::value> rethrown_with_a_note() {
  const std::string note = " The worst element is in 'coil'.";
  try {
    try {
      throw F(kWhat, kWhere, kHint, kCode, kDetails);
    } catch (const F& failure) {
      throw axmp::with_hint_note(failure, note);
    }
  } catch (const axmp::Error& e) {
    CHECK(e.message() == kWhat && e.where() == kWhere, "the rethrown message and pointer");
    CHECK(e.hint() == std::string(kHint) + note, "the rethrown hint: '%s'", e.hint().c_str());
    CHECK(e.code() == kCode, "the code did not survive the rethrow: '%s'", e.code().c_str());
    CHECK(e.details() == kDetails, "the details did not survive the rethrow: '%s'", e.details().c_str());
    CHECK(std::string(e.what()) == composed() + note, "the rethrown text: %s", e.what());
  }
  // without a hint the note's separator is dropped, and an error without a code stays without one
  const axmp::Error bare = axmp::with_hint_note(F(kWhat), note);
  CHECK(bare.hint() == note.substr(1), "the hint of a rethrown error that had none: '%s'", bare.hint().c_str());
  CHECK(std::string(bare.what()) == std::string(kWhat) + " Hint: " + note.substr(1), "its text: %s", bare.what());
  CHECK(bare.code().empty() && bare.details().empty() && bare.where().empty(), "fields appeared in a rethrow");
  const axmp::Error no_note = axmp::with_hint_note(F(kWhat, kWhere, kHint, kCode, kDetails), "");
  CHECK(std::string(no_note.what()) == composed() && no_note.code() == kCode, "an empty note changed the error");
}
template <class F>
std::enable_if_t<!coded<F>::value> rethrown_with_a_note() {
  CHECK(false, "axmp::SolverFailure has no code() and details() to keep through a rethrow");
}
// ---------------------------------------------------------------- 5: the command line's JSON
#if AXMP_TEST_HAVE_ERROR_JSON
template <class E>
std::enable_if_t<coded<E>::value> error_json_text() {
  // the text `axmp validate` prints for a refused design
  const std::string plain = R"({
  "error": "boundary 'inlet' selects nothing",
  "hint": "no boundary edge of 'channel' lies on x = 0.",
  "ok": false,
  "where": "/boundaries/inlet/where"
})";
  CHECK(axmp::error_json(E(kWhat, kWhere, kHint)).dump(2) == plain,
        "the JSON of an error without a code:\n%s\n  expected:\n%s",
        axmp::error_json(E(kWhat, kWhere, kHint)).dump(2).c_str(), plain.c_str());

  const std::string empty_fields = R"({
  "error": "boundary 'inlet' selects nothing",
  "hint": "",
  "ok": false,
  "where": ""
})";
  CHECK(axmp::error_json(E(kWhat)).dump(2) == empty_fields, "the JSON of a bare error: %s",
        axmp::error_json(E(kWhat)).dump(2).c_str());

  const json with_code = axmp::error_json(E(kWhat, kWhere, kHint, kCode, kDetails));
  CHECK(with_code.size() == 6, "an error with a code and details gives %zu keys", (size_t)with_code.size());
  CHECK(with_code.value("code", "") == kCode, "code in the JSON");
  CHECK(with_code["details"] == json::parse(kDetails), "details in the JSON: %s",
        with_code["details"].dump().c_str());
  CHECK(with_code["details"].is_object(), "details is not an object");
  for (const char* key : {"ok", "error", "where", "hint"})
    CHECK(with_code.contains(key), "an error with a code has no key '%s'", key);

  const json code_only = axmp::error_json(E(kWhat, kWhere, kHint, kCode));
  CHECK(code_only.size() == 5 && code_only.contains("code") && !code_only.contains("details"),
        "a code without details: %s", code_only.dump().c_str());

  // details with no members are left out, as the MCP payload leaves them out
  for (const char* empty : {"{}", "{ }"}) {
    const json out = axmp::error_json(E(kWhat, kWhere, kHint, kCode, empty));
    CHECK(out.size() == 5 && !out.contains("details"), "empty details '%s': %s", empty, out.dump().c_str());
  }

  // details that are not a JSON object are dropped; writing the report must not throw
  // (NaN and Infinity are not JSON: a reader that accepted them would print a reply that is not JSON either)
  for (const char* bad : {"not json", "[1,2]", "7", "{\"a\":", "{\"r\": NaN}", "{\"r\": Infinity}",
                          "{\"r\": -Infinity}"}) {
    json out;
    try {
      out = axmp::error_json(E(kWhat, kWhere, kHint, kCode, bad));
    } catch (...) {
      CHECK(false, "error_json threw on details '%s'", bad);
      continue;
    }
    CHECK(!out.contains("details"), "details '%s' reached the output as %s", bad, out.dump().c_str());
    CHECK(out.value("code", "") == kCode, "no code in the output with malformed details '%s'", bad);
  }

  // nesting: an ordinary nested object is kept; beyond kMaxErrorDetailsDepth, or at a depth that would overflow a
  // recursive reader, the details are left out; a bracket inside a string is not nesting
  auto nested = [](int levels) {
    std::string t;
    for (int i = 0; i < levels - 1; ++i) t += "{\"a\":";
    t += "{\"leaf\":[1,{\"x\":2}]}";
    for (int i = 0; i < levels - 1; ++i) t += "}";
    return t;
  };
  const json ordinary = axmp::error_json(E(kWhat, kWhere, kHint, kCode, "{\"a\":{\"b\":[1,{\"c\":2}]}}"));
  CHECK(ordinary.contains("details") && ordinary["details"]["a"]["b"][1]["c"] == 2, "ordinary nesting: %s",
        ordinary.dump().c_str());
  // levels counts the objects; the array inside the innermost adds one, and its object one more
  const int limit = axmp::kMaxErrorDetailsDepth;
  CHECK(axmp::error_json(E(kWhat, kWhere, kHint, kCode, nested(limit - 2))).contains("details"),
        "details at the depth limit were dropped");
  CHECK(!axmp::error_json(E(kWhat, kWhere, kHint, kCode, nested(limit - 1))).contains("details"),
        "details one level past the limit were kept");
  CHECK(!axmp::error_json(E(kWhat, kWhere, kHint, kCode, nested(100000))).contains("details"),
        "very deep details were kept");
  // far more brackets than the limit inside a string, and the same after an escaped quote, are text and not nesting
  const std::string many = std::string(200, '[') + std::string(200, '{');
  CHECK(axmp::error_json(E(kWhat, kWhere, kHint, kCode, "{\"s\":\"" + many + "\"}")).contains("details"),
        "brackets in a string counted as nesting");
  CHECK(axmp::error_json(E(kWhat, kWhere, kHint, kCode, "{\"s\":\"\\\"" + many + "\"}")).contains("details"),
        "brackets after an escaped quote counted as nesting");
  // an escaped backslash ends no escape: the quote after it closes the string, so the brackets of the string that
  // follows are text too
  const std::string escaped = "{\"s\":\"\\\\\",\"t\":\"" + many + "\"}";
  CHECK(axmp::error_json(E(kWhat, kWhere, kHint, kCode, escaped)).contains("details"),
        "an escaped backslash at the end of a string was read as an escape of the quote");
  // the depth falls again when a level closes: many siblings are not nesting
  std::string siblings = "{";
  for (int i = 0; i < 100; ++i) siblings += std::string(i ? "," : "") + "\"k" + std::to_string(i) + "\":[1]";
  siblings += "}";
  CHECK(axmp::error_json(E(kWhat, kWhere, kHint, kCode, siblings)).contains("details"),
        "details with many sibling arrays were dropped");
  std::string list = "{\"l\":[";
  for (int i = 0; i < 100; ++i) list += std::string(i ? "," : "") + "{\"x\":1}";
  list += "]}";
  CHECK(axmp::error_json(E(kWhat, kWhere, kHint, kCode, list)).contains("details"),
        "details with a long list of objects were dropped");
}

template <class E>
std::enable_if_t<!coded<E>::value> error_json_text() {
  CHECK(false, "cpp/src/error_json.hpp is there but axmp::Error has no code() and details()");
}
#else
template <class E>
void error_json_text() {
  CHECK(false,
        "cpp/src/error_json.hpp does not exist: the command line has no one place that turns an error into "
        "{ok, error, where, hint} and adds code and details when they are not empty");
}
#endif

}  // namespace

int main() {
  unchanged_error();
  coded_error<axmp::Error>();
  coded_solver_failure<axmp::SolverFailure>();
  rethrown_with_a_note<axmp::SolverFailure>();
  error_json_text<axmp::Error>();
  std::printf(failures ? "%d failure(s)\n" : "all error-field checks passed\n", failures);
  return failures ? 1 : 0;
}
