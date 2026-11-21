// Every expression a user meets in the repository, through both evaluators.
//
// cpp/tests/data/field_expr_corpus.json holds every string that could be an expression of the design grammar
// in examples/, in benchmarks/designs/ and in the templates, collected by
// cpp/tests/data/make_field_expr_corpus.py and stored with the parameter map of the document it came from.
// Nothing in that file says which of those strings is an expression: the existing evaluator decides, here,
// by accepting or refusing it.  For each string the two evaluators must do the same thing -- give the same
// double, bit for bit, or both refuse the text.
//
// The reference is therefore the existing evaluator (cpp/src/expr.cpp), run on the same text with the same
// parameters, and the parameter map itself is built exactly as the design parser builds it
// (cpp/src/spec.cpp: a number is written with 17 significant digits, an expression is passed through, and
// the set is resolved in dependency order by resolve_params), so that a string from a design document is
// evaluated against the very map that design gives it.
#include "axionmultiphysics/field_expr.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/expr.hpp"

namespace {

using json = nlohmann::json;

int failures = 0;

void report(const std::string& what) {
  if (failures < 40) std::printf("FAIL: %s\n", what.c_str());
  ++failures;
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// Why the design evaluator refused a text: its message is "cannot evaluate expression '<text>': <reason>
// (near position <i>)", or, for a value that is not finite, one sentence of its own.  The position is where
// the parser stood and is not required of another parser; the reason is.
std::string reason_of(const std::string& msg) {
  const size_t colon = msg.find("': ");
  const size_t at = msg.rfind(" (near position ");
  if (colon == std::string::npos || at == std::string::npos || at <= colon + 3) return msg;
  return msg.substr(colon + 3, at - colon - 3);
}

std::string bits(double a) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", a);
  return buf;
}

// The counts the committed corpus must still reach.  They are measured, not chosen: the figures below are
// the ones the committed file gives, and the test requires at least them, so that a corpus that lost its
// documents, or whose strings stopped reaching the evaluator, fails here instead of passing emptily.
constexpr int kMinDocuments = 27;
constexpr int kMinStrings = 2000;
constexpr int kMinEvaluated = 200;

}  // namespace

int main() {
#ifdef AXMP_SOURCE_DIR
  const std::string path = std::string(AXMP_SOURCE_DIR) + "/cpp/tests/data/field_expr_corpus.json";
#else
  const std::string path = "cpp/tests/data/field_expr_corpus.json";
#endif
  std::ifstream in(path);
  if (!in) {
    std::printf("FAIL: cannot read %s\n", path.c_str());
    return 1;
  }
  json corpus;
  try {
    in >> corpus;
  } catch (const std::exception& e) {
    std::printf("FAIL: %s is not readable JSON: %s\n", path.c_str(), e.what());
    return 1;
  }

  int n_documents = 0;
  int n_strings = 0;
  int n_evaluated = 0;
  int n_refused = 0;
  std::map<std::string, int> by_root;
  std::map<std::string, int> landmarks = {
      {"max(R, H)/30", 0}, {"exp(-0.5)", 0}, {"y*R-tw", 0}, {"min(ts, g, tw)", 0}, {"NI/2", 0}};

  for (const json& doc : corpus.at("documents")) {
    const std::string source = doc.at("source").get<std::string>();
    ++n_documents;
    if (source.rfind("templates:", 0) == 0) ++by_root["templates"];
    else if (source.rfind("examples/", 0) == 0) ++by_root["examples"];
    else if (source.rfind("benchmarks/designs/", 0) == 0) ++by_root["benchmarks/designs"];
    else report("the corpus entry '" + source + "' is not from one of the three roots");

    // the parameter map of this document
    axmp::ParamMap params;
    const json& spec = doc.at("params");
    const std::string kind = spec.at("kind").get<std::string>();
    if (kind == "definitions") {
      std::map<std::string, std::string> defs;
      for (auto it = spec.at("defs").begin(); it != spec.at("defs").end(); ++it) {
        if (it.value().is_number()) {
          std::ostringstream os;
          os.precision(17);
          os << it.value().get<double>();
          defs[it.key()] = os.str();
        } else {
          defs[it.key()] = it.value().get<std::string>();
        }
      }
      try {
        params = axmp::resolve_params(defs, "/parameters");
      } catch (const axmp::Error& e) {
        report("the parameters of '" + source + "' do not resolve: " + std::string(e.what()));
        continue;
      }
    } else if (kind == "values") {
      for (auto it = spec.at("values").begin(); it != spec.at("values").end(); ++it) {
        params[it.key()] = it.value().get<double>();
      }
    } else {
      report("the corpus entry '" + source + "' has an unknown parameter kind '" + kind + "'");
      continue;
    }

    for (const json& entry : doc.at("strings")) {
      const std::string text = entry.at("text").get<std::string>();
      const std::string at = entry.at("at").get<std::string>();
      const std::string where = source + " " + at;
      ++n_strings;
      if (landmarks.count(text)) ++landmarks[text];

      bool a_ok = false;
      double a = 0.0;
      std::string a_msg;
      try {
        a = axmp::eval_expr(text, params, where);
        a_ok = true;
      } catch (const axmp::Error& e) {
        a_msg = e.what();
      }
      bool b_ok = false;
      double b = 0.0;
      std::string b_msg;
      std::string b_where;
      std::string b_hint;
      try {
        const axmp::FieldExpr f = axmp::FieldExpr::compile(text, params, {}, where);
        b = f.eval(nullptr);
        b_ok = true;
        // the many-point form of a text without variables repeats that same double
        double many[3] = {1.0, 2.0, 3.0};
        f.eval(nullptr, 3, many);
        for (int i = 0; i < 3; ++i) {
          if (!same_bits(many[i], b)) {
            report(where + ": '" + text + "' is " + bits(b) + " point by point and " + bits(many[i]) +
                   " in the many-point form");
            break;
          }
        }
      } catch (const axmp::Error& e) {
        b_msg = e.what();
        b_where = e.where();
        b_hint = e.hint();
      } catch (const std::exception& e) {
        report(where + ": '" + text + "' threw a bare exception: " + std::string(e.what()));
        continue;
      }

      if (a_ok != b_ok) {
        report(where + ": '" + text + "': eval_expr " + (a_ok ? "gives " + bits(a) : "refuses (" + a_msg + ")") +
               ", FieldExpr " + (b_ok ? "gives " + bits(b) : "refuses (" + b_msg + ")"));
      } else if (a_ok && !same_bits(a, b)) {
        report(where + ": '" + text + "': eval_expr gives " + bits(a) + ", FieldExpr " + bits(b));
      } else if (!a_ok) {
        if (b_where != where || b_hint.empty()) {
          report(where + ": '" + text + "': the refusal of FieldExpr lacks the path or a hint: " + b_msg);
        } else if (b_msg.find(reason_of(a_msg)) == std::string::npos) {
          report(where + ": '" + text + "': eval_expr refuses it because of '" + reason_of(a_msg) +
                 "', FieldExpr says '" + b_msg + "'");
        }
      }
      if (a_ok) ++n_evaluated;
      else ++n_refused;
    }
  }

  // the corpus still covers what it was collected for
  if (n_documents < kMinDocuments) {
    report("the corpus holds " + std::to_string(n_documents) + " documents, want at least " +
           std::to_string(kMinDocuments));
  }
  if (n_strings < kMinStrings) {
    report("the corpus holds " + std::to_string(n_strings) + " strings, want at least " +
           std::to_string(kMinStrings));
  }
  if (n_evaluated < kMinEvaluated) {
    report("only " + std::to_string(n_evaluated) + " corpus strings evaluate, want at least " +
           std::to_string(kMinEvaluated) + ": the comparison of values has lost its subject");
  }
  for (const char* root : {"examples", "benchmarks/designs", "templates"}) {
    if (by_root[root] == 0) report(std::string("no corpus document comes from ") + root);
  }
  for (const auto& kv : landmarks) {
    if (kv.second == 0) report("the corpus no longer holds the expression '" + kv.first + "'");
  }
  const json counts = corpus.at("counts");
  if (counts.at("documents").get<int>() != n_documents || counts.at("strings").get<int>() != n_strings) {
    report("the counts in the corpus file (" + counts.dump() + ") are not what it holds (" +
           std::to_string(n_documents) + " documents, " + std::to_string(n_strings) + " strings)");
  }

  if (failures != 0) {
    std::printf("%d corpus check(s) failed\n", failures);
    return 1;
  }
  std::printf("expression corpus: %d documents, %d strings (%d evaluate, %d are refused), "
              "both evaluators agree on every one\n",
              n_documents, n_strings, n_evaluated, n_refused);
  return 0;
}
