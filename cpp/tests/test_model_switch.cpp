// The switch between the two parsers (design note 3.2): the guard at the top of parse_design, the refusal of a model
// document at an operation that works on electromagnetic designs only (require_em_document) or that does not take one
// in this version (refuse_unserved_model_document), the resolved geometry of a model document, and the scan of a JSON
// text for a key repeated in one object (scan_duplicate_keys, design note 15.2 row 15).
//
// Independent references:
//   - the texts of design note 3.2, copied verbatim: the message, pointer and hint of model.section_needs_physics, and
//     the message and pointer of model.em_only_operation;
//   - the texts this test writes where the note gives none (they are the specification): the hints of
//     model.em_only_operation, the backstop's generic message, the refusal of solving (route.not_implemented) and
//     json.duplicate_key;
//   - today's "unknown field" texts of the electromagnetic parser, from its own `only` lists (spec.cpp, read here), for
//     the new keys below the top level that keep them;
//   - RFC 6901 for the pointers ('~' is written ~0 and '/' is written ~1);
//   - the keys of resolved_geometry of an electromagnetic design, computed here from the same build.
//
// THE API THE TESTS USE (cpp/include/axionmultiphysics/spec_model.hpp, namespace axmp::spec):
//   serves_model_documents() -> std::vector<std::string>
//   require_em_document(json, operation)          throws model.em_only_operation for a model document
//   refuse_unserved_model_document(json, operation)  throws route.not_implemented for a model document
//   resolved_model_geometry(ResolvedModel) -> json
//   scan_duplicate_keys(text, origin)             throws json.duplicate_key
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/spec.hpp"
#include "axionmultiphysics/spec_model.hpp"

using nlohmann::json;
using axmp::Error;

static int failures = 0;
static void fail(const std::string& what) {
  std::printf("FAIL: %s\n", what.c_str());
  ++failures;
}
#define CHECK(cond, what)        \
  do {                           \
    if (!(cond)) fail(what);     \
  } while (0)

static std::string source_dir() {
#ifdef AXMP_SOURCE_DIR
  return AXMP_SOURCE_DIR;
#else
  return ".";
#endif
}

static json read_doc(const std::string& rel) {
  std::ifstream f(source_dir() + "/" + rel);
  if (!f) {
    fail("cannot read " + rel);
    return json::object();
  }
  return json::parse(f);
}

// A valid electromagnetic design: one rectangle.
static json em_minimal() {
  return json{{"name", "m"},
              {"regions", json::array({{{"name", "x"}, {"shape", {{"rect", {{"r", {0, 1}}, {"z", {0, 1}}}}}}}})}};
}

static const std::vector<std::string> kServes = {"validate", "resolve", "explain", "patch", "boundaries", "mesh"};
static const std::string kTail =
    "In this version a model document is taken by: validate, resolve, explain, patch, boundaries, mesh.";

struct Expected {
  std::string message, where, hint, code;
  json details;  // null: no details
};

// Runs f, which must throw an axmp::Error, and compares every field with e.
static void expect_error(const std::string& label, const std::function<void()>& f, const Expected& e) {
  try {
    f();
    fail(label + ": no error");
  } catch (const Error& err) {
    CHECK(err.message() == e.message, label + ": message\n  got  " + err.message() + "\n  want " + e.message);
    CHECK(err.where() == e.where, label + ": where " + err.where() + " (want " + e.where + ")");
    CHECK(err.hint() == e.hint, label + ": hint\n  got  " + err.hint() + "\n  want " + e.hint);
    CHECK(err.code() == e.code, label + ": code '" + err.code() + "' (want '" + e.code + "')");
    const json d = err.details().empty() ? json() : json::parse(err.details());
    CHECK(d == e.details, label + ": details " + d.dump() + " (want " + e.details.dump() + ")");
    const std::string composed = e.message + (e.where.empty() ? "" : " [at " + e.where + "]") +
                                 (e.hint.empty() ? "" : " Hint: " + e.hint);
    CHECK(std::string(err.what()) == composed, label + ": what() is the composed text");
  } catch (const std::exception& x) {
    fail(label + ": not an axmp::Error: " + x.what());
  }
}

static std::string em_only_message(const std::string& op, const std::string& name) {
  return "'" + op + "' works on electromagnetic designs, and '" + name +
         "' is a model document (it has a 'physics' section)";
}
static std::string em_only_hint(const std::string& op) {
  if (op == "sensitivities" || op == "gradient") return "adjoint sensitivities are an electromagnetic capability. " + kTail;
  if (op == "optimize" || op == "global_optimize" || op == "optimize_design" || op == "optimize_global")
    return "this version optimizes electromagnetic designs. " + kTail;
  return "this operation needs an electromagnetic design (a document without 'physics'). " + kTail;
}
static json em_only_details(const json& op) {
  return json{{"operation", op}, {"serves_model_documents", kServes}};
}

// ---------------------------------------------------------------------------------------------------- the guard
static const std::vector<std::string> kSections = {"domains", "boundaries", "reactions", "couplings", "study", "outputs"};

static void check_six_section_texts() {
  for (const std::string& key : kSections) {
    json doc = em_minimal();
    doc[key] = json::object();
    // design note 3.2 item 1, verbatim
    expect_error("parse_design with '" + key + "' and no physics",
                 [&] { axmp::spec::parse_design(doc); },
                 {"'" + key + "' belongs to a model document, and this document has no 'physics' section", "/" + key,
                  "add \"physics\": {...} (run `axmp physics` for the available types), or remove '" + key + "'.",
                  "model.section_needs_physics", json()});
  }
}

// The two tests come before root.only: a section key wins over an unknown top-level key, and of two section keys the
// first in the note's order (domains, boundaries, reactions, couplings, study, outputs) is reported.
static void check_guard_order() {
  json doc = em_minimal();
  doc["zzz_unknown"] = 1;
  doc["outputs"] = json::object();
  doc["boundaries"] = json::object();
  expect_error("two section keys and an unknown key", [&] { axmp::spec::parse_design(doc); },
               {"'boundaries' belongs to a model document, and this document has no 'physics' section", "/boundaries",
                "add \"physics\": {...} (run `axmp physics` for the available types), or remove 'boundaries'.",
                "model.section_needs_physics", json()});
  json both = em_minimal();
  both["study"] = json::array();
  both["physics"] = json::object();
  // physics present: the em-only refusal (the backstop), whatever else is there
  try {
    axmp::spec::parse_design(both);
    fail("physics and study: no error");
  } catch (const Error& e) {
    CHECK(e.code() == "model.em_only_operation", "physics and study: code " + e.code());
  } catch (const std::exception& x) {
    fail(std::string("physics and study: not an axmp::Error: ") + x.what());
  }
}

// The backstop: parse_design itself cannot name its caller (design plan N1).
static void check_backstop() {
  const json channel = read_doc("cpp/tests/data/model/channel.json");
  expect_error("parse_design of a model document", [&] { axmp::spec::parse_design(channel); },
               {"this operation works on electromagnetic designs, and 'channel' is a model document (it has a 'physics' section)",
                "/physics", em_only_hint(""), "model.em_only_operation", em_only_details(nullptr)});
  // the value of "physics" is not looked at
  json odd = em_minimal();
  odd["physics"] = 3;
  expect_error("parse_design with physics: 3", [&] { axmp::spec::parse_design(odd); },
               {"this operation works on electromagnetic designs, and 'm' is a model document (it has a 'physics' section)",
                "/physics", em_only_hint(""), "model.em_only_operation", em_only_details(nullptr)});
}

// Keys below the top level keep today's "unknown field" text (no dedicated refusal, no widened list).
static void check_lower_keys_keep_the_unknown_field_text() {
  json doc = em_minimal();
  doc["mesh"] = {{"exterior", "none"}};
  try {
    axmp::spec::parse_design(doc);
    fail("mesh.exterior without physics: no error");
  } catch (const Error& e) {
    CHECK(e.message() == "unknown field 'exterior'", "mesh.exterior: message " + e.message());
    CHECK(e.where() == "/mesh", "mesh.exterior: where " + e.where());
    CHECK(e.code().empty() && e.details().empty(), "mesh.exterior: no code, no details");
    CHECK(e.hint().rfind("Allowed fields here: ", 0) == 0 && e.hint().find("exterior") == std::string::npos,
          "mesh.exterior: today's allowed list, without exterior: " + e.hint());
  }
  for (const char* block : {"fluid", "electrolyte", "electric", "species"}) {
    json m = em_minimal();
    m["materials"] = {{"w", {{"kind", "vacuum"}, {block, json::object()}}}};
    try {
      axmp::spec::parse_design(m);
      fail(std::string("material block ") + block + " without physics: no error");
    } catch (const Error& e) {
      CHECK(e.message() == std::string("unknown field '") + block + "'", std::string(block) + ": message " + e.message());
      CHECK(e.where() == "/materials/w", std::string(block) + ": where " + e.where());
      CHECK(e.code().empty(), std::string(block) + ": no code");
      CHECK(e.hint().find(block) == std::string::npos, std::string(block) + ": the allowed list is not widened");
    }
  }
}

// ---------------------------------------------------------------------------------------------------- named refusals
static void check_require_em_document() {
  const json channel = read_doc("cpp/tests/data/model/channel.json");
  CHECK(axmp::spec::serves_model_documents() == kServes,
        "serves_model_documents() is {validate, resolve, explain, patch, boundaries, mesh}");
  for (const std::string op : {"sensitivities", "gradient", "optimize", "global_optimize", "Model3D", "to_3d", "make_3d"}) {
    expect_error("require_em_document(channel, " + op + ")", [&] { axmp::spec::require_em_document(channel, op); },
                 {em_only_message(op, "channel"), "/physics", em_only_hint(op), "model.em_only_operation",
                  em_only_details(op)});
  }
  // design note 3.2's example, message and pointer verbatim
  try {
    axmp::spec::require_em_document(channel, "sensitivities");
  } catch (const Error& e) {
    CHECK(std::string(e.what()).rfind("'sensitivities' works on electromagnetic designs, and 'channel' is a model document "
                                      "(it has a 'physics' section) [at /physics] Hint: adjoint sensitivities are an "
                                      "electromagnetic capability.", 0) == 0,
          "the note's text of the sensitivities refusal");
  } catch (...) {
  }
  json unnamed = channel;
  unnamed.erase("name");
  expect_error("require_em_document without a name", [&] { axmp::spec::require_em_document(unnamed, "to_3d"); },
               {"'to_3d' works on electromagnetic designs, and this document is a model document (it has a 'physics' section)",
                "/physics", em_only_hint("to_3d"), "model.em_only_operation", em_only_details("to_3d")});
  // an electromagnetic design passes, whatever else is wrong with it
  try {
    axmp::spec::require_em_document(em_minimal(), "sensitivities");
    json bad = em_minimal();
    bad["zzz"] = 1;
    bad["boundaries"] = 1;
    axmp::spec::require_em_document(bad, "sensitivities");
  } catch (const std::exception& e) {
    fail(std::string("require_em_document refuses a document without physics: ") + e.what());
  }
}

static void check_refuse_unserved() {
  const json channel = read_doc("cpp/tests/data/model/channel.json");
  // meshing takes a model document in this version (axmp mesh, _core.mesh): solving and run_3d still refuse one
  const std::string hint = "solving a model document is not implemented in this version; `axmp validate` reports what "
                           "the document needs. " + kTail;
  for (const std::string op : {"solve", "run_3d"}) {
    expect_error("refuse_unserved_model_document(channel, " + op + ")",
                 [&] { axmp::spec::refuse_unserved_model_document(channel, op); },
                 {"'" + op + "' of a model document is not implemented yet ('channel' has a 'physics' section)", "/physics",
                  hint, "route.not_implemented",
                  json{{"operation", op}, {"implemented", false}, {"serves_model_documents", kServes}}});
  }
  json unnamed = channel;
  unnamed.erase("name");
  expect_error("refuse_unserved_model_document without a name",
               [&] { axmp::spec::refuse_unserved_model_document(unnamed, "solve"); },
               {"'solve' of a model document is not implemented yet (the document has a 'physics' section)", "/physics",
                hint, "route.not_implemented",
                json{{"operation", "solve"}, {"implemented", false}, {"serves_model_documents", kServes}}});
  try {
    axmp::spec::refuse_unserved_model_document(em_minimal(), "solve");
  } catch (const std::exception& e) {
    fail(std::string("refuse_unserved_model_document refuses an electromagnetic design: ") + e.what());
  }
}

// ---------------------------------------------------------------------------------------------------- resolve
static void check_resolved_model_geometry() {
  const json channel = read_doc("cpp/tests/data/model/channel.json");
  json out;
  try {
    out = axmp::spec::resolved_model_geometry(axmp::spec::parse_model(channel));
  } catch (const std::exception& e) {
    fail(std::string("resolved_model_geometry(channel): ") + e.what());
    return;
  }
  // today's top-level keys (an electromagnetic design without a pickup), from the same build.  The electromagnetic
  // parser takes no planar design with a rect (planar needs mesh.file, and then 'imported' shapes), and writes every
  // resolved rect with the keys r and z: its resolution of the same rectangle is that of the axisymmetric rect.
  json em = em_minimal();
  const json ref = axmp::spec::resolved_geometry(axmp::spec::parse_design(em));
  std::set<std::string> want, got;
  for (auto it = ref.begin(); it != ref.end(); ++it) want.insert(it.key());
  for (auto it = out.begin(); it != out.end(); ++it) got.insert(it.key());
  CHECK(got == want, "resolve of a model document: today's top-level keys, got " + json(got).dump());
  std::set<std::string> mesh;
  if (out.contains("mesh") && out["mesh"].is_object())
    for (auto it = out["mesh"].begin(); it != out["mesh"].end(); ++it) mesh.insert(it.key());
  CHECK(mesh == std::set<std::string>({"size", "surface_size", "geometry_order", "refinements", "exterior"}),
        "resolve of a model document: the mesh keys of a model document, got " + json(mesh).dump());
  CHECK(out.value("name", "") == "channel" && out.value("symmetry", "") == "planar", "name and symmetry");
  // the document's own numbers: L = 0.04, H = 0.01
  const json shape = out["regions"][0]["shape"];
  CHECK(out["regions"].size() == 1 && out["regions"][0]["name"] == "channel" && out["regions"][0]["material"] == "water",
        "the region and its material");
  CHECK(out["parameters"]["L"] == 0.04 && out["parameters"]["H"] == 0.01, "the parameters, evaluated");
  CHECK(out["materials"].contains("water") && out["materials"]["water"].contains("kind"),
        "the materials carry their electromagnetic kind");
  // the same rectangle resolved by the electromagnetic parser
  json twin = em;
  twin["regions"][0]["shape"] = {{"rect", {{"r", {0, 0.04}}, {"z", {0, 0.01}}}}};
  const json twin_shape = axmp::spec::resolved_geometry(axmp::spec::parse_design(twin))["regions"][0]["shape"];
  CHECK(shape == twin_shape, "the resolved shape equals the electromagnetic parser's: " + shape.dump() + " vs " + twin_shape.dump());
}

// ---------------------------------------------------------------------------------------------------- duplicate keys
static void expect_duplicate(const std::string& label, const std::string& text, const std::string& key,
                             const std::string& pointer) {
  expect_error(label, [&] { axmp::spec::scan_duplicate_keys(text, "doc.json"); },
               {"duplicate key '" + key + "' in 'doc.json'", pointer,
                "a model document may not repeat a key in an object: JSON keeps only the last value, silently. Keep one of "
                "the two.",
                "json.duplicate_key", json{{"key", key}}});
}

static void check_scan_duplicate_keys() {
  try {
    axmp::spec::scan_duplicate_keys(R"({"a": 1, "b": {"a": 2, "c": [{"a": 1}, {"a": 2}]}, "physics": {}})", "doc.json");
    axmp::spec::scan_duplicate_keys("[]", "doc.json");
  } catch (const std::exception& e) {
    fail(std::string("scan_duplicate_keys refuses a text without a repeated key: ") + e.what());
  }
  expect_duplicate("top level", R"({"name": "a", "physics": {}, "name": "b"})", "name", "/name");
  expect_duplicate("nested", R"({"physics": {"flow": {"type": "x", "type": "y"}}})", "type", "/physics/flow/type");
  expect_duplicate("in an array", R"({"couplings": [{"term": "a"}, {"term": "b", "to": "c", "term": "d"}], "physics": {}})",
                   "term", "/couplings/1/term");
  expect_duplicate("a key with a slash", R"({"physics": {"a/b": 1, "a/b": 2}})", "a/b", "/physics/a~1b");
  expect_duplicate("a key with a tilde", R"({"physics": {"x~y": 1, "x~y": 2}})", "x~y", "/physics/x~0y");
  expect_duplicate("the empty key", R"({"physics": {"": 1, "": 2}})", "", "/physics/");
  // the first repetition in document order is reported
  expect_duplicate("two repetitions", R"({"b": {"k": 1, "k": 2}, "a": 1, "a": 2, "physics": {}})", "k", "/b/k");
  // the same key in two objects is no repetition
  try {
    axmp::spec::scan_duplicate_keys(R"({"x": {"k": 1}, "y": {"k": 1}, "k": 0})", "doc.json");
  } catch (const std::exception& e) {
    fail(std::string("the same key in two objects: ") + e.what());
  }
}

int main() {
  const auto guarded = [](const char* name, void (*f)()) {
    try {
      f();
    } catch (const std::exception& e) {
      fail(std::string(name) + " threw: " + e.what());
    }
  };
  guarded("check_six_section_texts", check_six_section_texts);
  guarded("check_guard_order", check_guard_order);
  guarded("check_backstop", check_backstop);
  guarded("check_lower_keys_keep_the_unknown_field_text", check_lower_keys_keep_the_unknown_field_text);
  guarded("check_require_em_document", check_require_em_document);
  guarded("check_refuse_unserved", check_refuse_unserved);
  guarded("check_resolved_model_geometry", check_resolved_model_geometry);
  guarded("check_scan_duplicate_keys", check_scan_duplicate_keys);
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all switch checks passed\n");
  return 0;
}
