// Compiled field expressions (cpp/include/axionmultiphysics/field_expr.hpp): the same grammar, functions and
// constants as the design evaluator, free variables, exact first derivatives, and the refusals.
//
// Every value is checked against an independent reference:
//   - the existing evaluator, cpp/src/expr.cpp (eval_expr), for the value of an expression without variables:
//     the two must agree bit for bit, or refuse the same text;
//   - a Taylor test for eval_d, which uses no derivative of ours: the remainder of the first-order expansion
//     must fall at second order over four halvings of the step;
//   - a five-point central difference, as a second reading of the same derivative;
//   - pointwise evaluation, for the many-point form;
//   - evaluation on one thread, for evaluation on several.
// The steps are powers of two, so that halving a step is exact and the remainder carries no noise from the
// step itself.
#include "axionmultiphysics/field_expr.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/expr.hpp"

namespace {

int failures = 0;

void report(const std::string& what) {
  std::printf("FAIL: %s\n", what.c_str());
  ++failures;
}

// Bit-for-bit equality: "the same double", not "near".
bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// A text in a message: the deep-nesting cases are thousands of characters long.
std::string shown(const std::string& text) {
  return text.size() <= 60 ? text : text.substr(0, 57) + "...";
}

std::string bits(double a) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", a);
  return buf;
}

// What an evaluator did with a text: a value, or a refusal and its message.
struct Verdict {
  bool ok = false;
  double v = 0.0;
  std::string msg;
  std::string where;
  std::string hint;
};

Verdict by_eval_expr(const std::string& text, const axmp::ParamMap& p, const std::string& where) {
  Verdict r;
  try {
    r.v = axmp::eval_expr(text, p, where);
    r.ok = true;
  } catch (const axmp::Error& e) {
    r.msg = e.what();
    r.where = e.where();
    r.hint = e.hint();
  }
  return r;
}

// compile() and eval() are one operation here: for a text without variables the value is known when it is
// compiled, so a refusal that depends on the value may come from either call.
Verdict by_field_expr(const std::string& text, const axmp::ParamMap& p, const std::string& where) {
  Verdict r;
  try {
    const axmp::FieldExpr f = axmp::FieldExpr::compile(text, p, {}, where);
    r.v = f.eval(nullptr);
    r.ok = true;
  } catch (const axmp::Error& e) {
    r.msg = e.what();
    r.where = e.where();
    r.hint = e.hint();
  }
  return r;
}

// Why the design evaluator refused a text: its message is "cannot evaluate expression '<text>': <reason>
// (near position <i>)", or, for a value that is not finite, one sentence of its own.  The position is where
// the parser stood and is not required of another parser; the reason is.
std::string reason_of(const std::string& msg) {
  const size_t colon = msg.find("': ");
  const size_t at = msg.rfind(" (near position ");
  if (colon == std::string::npos || at == std::string::npos || at <= colon + 3) return msg;
  return msg.substr(colon + 3, at - colon - 3);
}

// The two evaluators agree on `text`: the same double, bit for bit, or both refuse it for the same reason.
void parity(const std::string& text, const axmp::ParamMap& p) {
  const std::string where = "/parameters/q";
  const Verdict a = by_eval_expr(text, p, where);
  const Verdict b = by_field_expr(text, p, where);
  if (a.ok != b.ok) {
    report("'" + shown(text) + "': eval_expr " + (a.ok ? "gives " + bits(a.v) : "refuses (" + a.msg + ")") +
           ", FieldExpr " + (b.ok ? "gives " + bits(b.v) : "refuses (" + b.msg + ")"));
    return;
  }
  if (a.ok && !same_bits(a.v, b.v)) {
    report("'" + shown(text) + "': eval_expr gives " + bits(a.v) + ", FieldExpr " + bits(b.v));
    return;
  }
  if (!a.ok) {
    if (b.where != where || b.hint.empty()) {
      report("'" + shown(text) + "': the refusal of FieldExpr lacks the path or a hint: " + b.msg);
    }
    const std::string reason = reason_of(a.msg);
    if (b.msg.find(reason) == std::string::npos) {
      report("'" + shown(text) + "': eval_expr refuses it because of '" + reason + "', FieldExpr says '" +
             b.msg + "'");
    }
  }
}

// `text` is refused when it is compiled, with `needle` in the message, the path and a hint.
void refuse(const std::string& text, const axmp::ParamMap& p, const std::vector<std::string>& vars,
            const std::string& needle) {
  const std::string where = "/physics/inlet/velocity/0";
  try {
    const axmp::FieldExpr f = axmp::FieldExpr::compile(text, p, vars, where);
    std::vector<double> v(vars.size(), 1.0);
    report("'" + shown(text) + "' compiled (value " + bits(f.eval(v.data())) + "), want a refusal naming '" +
           needle + "'");
  } catch (const axmp::Error& e) {
    const std::string msg = e.what();
    if (msg.find(needle) == std::string::npos || e.where() != where || e.hint().empty()) {
      report("'" + shown(text) + "': the refusal lacks '" + needle + "', the path or a hint: " + msg);
    }
  } catch (const std::exception& e) {
    report("'" + shown(text) + "' threw a bare exception instead of axmp::Error: " + std::string(e.what()));
  }
}

// ------------------------------------------------------------------ the grammar, against eval_expr

// Every text of the design grammar's own test (cpp/tests/test_expr.cpp), valid and invalid, and the
// constants, the functions, and a parameter that shadows a constant.  The parameter map is that test's.
void check_grammar() {
  const axmp::ParamMap p = {{"x", 1.5}, {"z", 0.3}, {"s", 0.7}};
  const char* texts[] = {
      // precedence and associativity
      "-2^2", "(-2)^2", "-(2^2)", "3*-2^2", "-x^2", "(-x)^2", "-(x^2)", "x^-2", "-x^-2", "2^-2", "-2^-2",
      "2^3^2", "(2^3)^2", "-2^3^2", "2^-3^2", "e^-x^2", "exp(-z^2/(2*s^2))", "1 - -x^2", "2*x^2/4", "--2^2",
      "+-2^2", "2^+2", "1e-2^2", "pow(-2, 2)",
      // left associativity of + - * / (the order of the operations is part of the value)
      "1 - 2 - 3", "1/2/4", "x - z - s", "x/z/s", "0.1 + 0.2 + 0.3", "1e16 + 1 - 1e16",
      // numbers
      "2.", ".5", "1.5E+2", "0e-400", "2.2250738585072014e-308", "0.0e+999", "1.2.5", "1..5", ".", "2*.",
      "1e400", "-1e400", "1e-310", "1e-400",
      // the functions and the constants
      "sqrt(x)", "exp(x)", "log(x)", "log10(x)", "sin(x)", "cos(x)", "tan(x)", "asin(z)", "acos(z)", "atan(x)",
      "atan2(z, x)", "abs(-x)", "pow(x, z)", "min(1, 2, -3)", "max(7)", "min(x, z, s)", "max(x, z, s)",
      "pi", "e", "mu0", "pi*x", "e^2", "mu0/pi",
      // refusals that do not depend on a value
      "min(1, sqrt(-1))", "min(sqrt(-1), 1)", "max(0/0, 1)", "nope", "nope(1)", "sqrt()", "sqrt(1, 2)",
      "atan2(1)", "min()", "(1", "1)", "1 2", "x y", "", "   ", "*2", "1+", "$x",
      // refusals that depend on the value, which is known without variables
      "1/0", "-1/0", "0/0", "log(0)", "sqrt(-1)", "1e308*10",
  };
  for (const char* t : texts) parity(t, p);
  // a parameter with the name of a constant hides the constant, as in the design evaluator
  const axmp::ParamMap shadow = {{"pi", 3.0}, {"e", 4.0}, {"mu0", 5.0}};
  for (const char* t : {"pi", "e", "mu0", "pi*e+mu0", "2^e"}) parity(t, shadow);
  // the depth cap: the same limit, and an error instead of a stack overflow
  const std::string deep_ok = std::string(999, '(') + "2" + std::string(999, ')');
  const std::string deep_no = std::string(1000, '(') + "2" + std::string(1000, ')');
  parity(deep_ok, p);
  parity(deep_no, p);
  parity(std::string(998, '-') + "2", p);
  parity(std::string(5000, '-') + "2", p);
  std::string tower = "2";
  for (int k = 0; k < 2000; ++k) tower += "^1";
  parity(tower, p);
  parity(std::string(100000, '(') + "2" + std::string(100000, ')'), p);
  // A check the design evaluator makes at a node is made at that node, whether or not the expression has
  // variables elsewhere: a constant argument of min or max that is NaN is refused.  A check on the value of
  // the whole expression is made when the whole expression is constant, which it is when no variable of the
  // list occurs in it.
  refuse("min(1, sqrt(-1)) + t", p, {"t"}, "min() argument 2 is not a number (NaN)");
  refuse("min(t, sqrt(-1))", p, {"t"}, "min() argument 2 is not a number (NaN)");
  refuse("max(sqrt(-1), t)", p, {"t"}, "max() argument 1 is not a number (NaN)");
  refuse("1/0", p, {"t"}, "evaluated to a non-finite value");
}

// ------------------------------------------------------------------ variables

void check_variables() {
  const axmp::ParamMap p = {{"U", 2.0}, {"H", 4.0}};
  const std::vector<std::string> xy = {"x", "y"};
  const double pt[2] = {0.25, 0.5};

  // v[k] is the variable at position k of the list given to compile()
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("x", p, xy, "/w");
    const axmp::FieldExpr g = axmp::FieldExpr::compile("x", p, {"y", "x"}, "/w");
    if (!same_bits(f.eval(pt), 0.25)) report("'x' with variables (x, y) does not read v[0]");
    if (!same_bits(g.eval(pt), 0.5)) report("'x' with variables (y, x) does not read v[1]");
  }
  // used_variables(): exactly the variables the text uses, in the order of the list
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("y + 0*U", p, {"t", "x", "y"}, "/w");
    const std::vector<std::string> want = {"y"};
    if (f.used_variables() != want) report("used_variables() of 'y + 0*U' is not (y)");
    const axmp::FieldExpr g = axmp::FieldExpr::compile("y*x + x", p, {"t", "x", "y"}, "/w");
    const std::vector<std::string> want_g = {"x", "y"};
    if (g.used_variables() != want_g) report("used_variables() of 'y*x + x' is not (x, y) in the list's order");
    const axmp::FieldExpr h = axmp::FieldExpr::compile("U*H", p, {"x", "y"}, "/w");
    if (!h.used_variables().empty()) report("used_variables() of a text without variables is not empty");
  }
  // an unused variable has derivative exactly +0
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("x*x", p, xy, "/w");
    if (!same_bits(f.eval_d(pt, 1), 0.0)) report("d/dy of 'x*x' is not +0");
  }
  // a variable hides a constant of the grammar, as a parameter does
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("e", p, {"e"}, "/w");
    const double at[1] = {7.0};
    if (!same_bits(f.eval(at), 7.0)) report("a variable named 'e' does not hide the constant e");
  }
  // an identifier followed by '(' is a function call, as in the design evaluator, whatever is in the list
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("max(1, 2)", p, {"max"}, "/w");
    const double at[1] = {9.0};
    if (!same_bits(f.eval(at), 2.0)) report("'max(1, 2)' with a variable named 'max' is not the function");
    const axmp::FieldExpr g = axmp::FieldExpr::compile("max", p, {"max"}, "/w");
    if (!same_bits(g.eval(at), 9.0)) report("'max' with a variable named 'max' is not the variable");
  }
  // eval_d names a variable by its position; there is no variable outside the list
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("x+y", p, xy, "/w");
    for (int k : {-1, 2, 7}) {
      try {
        const double d = f.eval_d(pt, k);
        report("eval_d accepted the variable index " + std::to_string(k) + " (gave " + bits(d) + ")");
      } catch (const axmp::Error&) {
      } catch (const std::exception& e) {
        report("eval_d threw a bare exception for an index out of range: " + std::string(e.what()));
      }
    }
  }
  // a text without variables evaluates with a null point, in all three calls: a point is read only where
  // a variable of it occurs in the text, so the list of the context may be non-empty while the point is
  // null
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("U*H", p, {}, "/w");
    if (!same_bits(f.eval(nullptr), 8.0)) report("a text without variables does not evaluate at a null point");
    const axmp::FieldExpr g = axmp::FieldExpr::compile("U*H", p, xy, "/w");
    const double guard = 1234.5;
    double out[3] = {guard, guard, guard};
    g.eval(nullptr, 3, out);
    for (int i = 0; i < 3; ++i) {
      if (!same_bits(out[i], 8.0)) {
        report("the many-point form at a null point gives " + bits(out[i]) + " at point " +
               std::to_string(i) + ", not 8");
      }
    }
    if (!same_bits(g.eval_d(nullptr, 1), 0.0)) report("eval_d at a null point is not +0");
  }
  // evaluation keeps no state: the same point twice is the same double
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("6*U*y*(H-y)/H^2", p, xy, "/w");
    if (!same_bits(f.eval(pt), f.eval(pt))) report("eval() of the same point twice is not the same double");
    if (!same_bits(f.eval_d(pt, 1), f.eval_d(pt, 1))) report("eval_d() of the same point twice differs");
  }
}

// ------------------------------------------------------------------ what a compiled expression owns

void check_value_semantics() {
  // A compiled expression is handed around, kept in a coefficient and used for a whole solve, so it must
  // own everything it needs.  Here the text, the parameter map and the variable list are gone before the
  // first evaluation, and the expressions have been copied and moved in between (push_back moves them when
  // the vector grows).
  std::vector<axmp::FieldExpr> kept;
  {
    const axmp::ParamMap params = {{"U", 2.0}, {"H", 4.0}};
    const std::string text = "6*U*y*(H-y)/H^2";
    const std::vector<std::string> vars = {"x", "y"};
    kept.push_back(axmp::FieldExpr::compile(text, params, vars, "/physics/flow/inlet/velocity/0"));
    kept.push_back(kept.front());
    axmp::FieldExpr moved = axmp::FieldExpr::compile(text, params, vars, "/w");
    kept.push_back(std::move(moved));
  }
  const double pt[2] = {0.5, 1.5};
  const std::vector<std::string> want = {"y"};
  for (size_t i = 0; i < kept.size(); ++i) {
    // 6 U y (H - y) / H^2 at y = 1.5 with U = 2 and H = 4
    if (!same_bits(kept[i].eval(pt), 2.8125)) {
      report("copy " + std::to_string(i) + " of a compiled expression gives " + bits(kept[i].eval(pt)) +
             " once its text and parameters are gone, not 2.8125");
    }
    if (!same_bits(kept[i].eval_d(pt, 1), 0.75)) {
      report("copy " + std::to_string(i) + " of a compiled expression has derivative " +
             bits(kept[i].eval_d(pt, 1)) + ", not 0.75");
    }
    if (kept[i].used_variables() != want) {
      report("copy " + std::to_string(i) + " of a compiled expression lost its used_variables()");
    }
  }
}

// ------------------------------------------------------------------ the shadowing refusal

void check_shadowing() {
  // a parameter with the name of a variable of the context is refused where the expression is compiled,
  // whether or not the text uses the name
  const axmp::ParamMap p = {{"y", 2.0}, {"U", 1.0}};
  refuse("U*2", p, {"x", "y"}, "parameter 'y' has the name of a variable");
  refuse("U*y", p, {"x", "y"}, "parameter 'y' has the name of a variable");
  refuse("1", p, {"y"}, "parameter 'y' has the name of a variable");
  // and nothing else is refused for it: the same parameters without the clashing variable compile
  try {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("U*y", p, {"x", "t"}, "/w");
    const double pt[2] = {1.0, 1.0};
    if (!same_bits(f.eval(pt), 2.0)) report("'U*y' with parameters y and U is not 2");
  } catch (const axmp::Error& e) {
    report(std::string("'U*y' with variables (x, t) was refused: ") + e.what());
  }
  // a variable listed twice has no position of its own
  refuse("x", {}, {"x", "x"}, "variable 'x' is listed twice");

  // The refusal is a coded one (section 7 of the interface design note): the code is expr.parameter_shadows_variable and
  // the details name the parameter and the variables of the context, in the order of the list.  An optional last argument
  // of compile() is the context: a phrase that follows "has the name of" in the message and ends it with "; rename the
  // parameter", where "{name}" stands for the clashing name.  Without it the message is the one above.
  struct Shadow {
    const char* what;
    axmp::ParamMap params;
    std::vector<std::string> vars;
    const char* context;     // nullptr: the call without the argument
    std::string message;     // exact
    std::string parameter;
    std::vector<std::string> listed;
  };
  const std::vector<Shadow> cases = {
      {"no context", {{"y", 2.0}, {"U", 1.0}}, {"x", "y"}, nullptr,
       "parameter 'y' has the name of a variable of this expression", "y", {"x", "y"}},
      {"an empty context is no context", {{"y", 2.0}}, {"x", "y"}, "",
       "parameter 'y' has the name of a variable of this expression", "y", {"x", "y"}},
      {"a coordinate of a planar model", {{"y", 2.0}}, {"x", "y"}, "a coordinate of this planar model",
       "parameter 'y' has the name of a coordinate of this planar model; rename the parameter", "y", {"x", "y"}},
      {"a coordinate of an axisymmetric model", {{"z", 2.0}}, {"r", "z"}, "a coordinate of this axisymmetric model",
       "parameter 'z' has the name of a coordinate of this axisymmetric model; rename the parameter", "z", {"r", "z"}},
      {"the field of a law", {{"T", 300.0}}, {"T"}, "the field '{name}' that this law reads",
       "parameter 'T' has the name of the field 'T' that this law reads; rename the parameter", "T", {"T"}},
      {"the first clash in the order of the list", {{"y", 2.0}, {"x", 1.0}}, {"x", "y", "z"}, "a coordinate of this 3d model",
       "parameter 'x' has the name of a coordinate of this 3d model; rename the parameter", "x", {"x", "y", "z"}},
  };
  for (const Shadow& c : cases) {
    const std::string where = "/physics/flow/boundary_conditions/inlet/velocity_m_s/0";
    try {
      if (c.context) axmp::FieldExpr::compile("1", c.params, c.vars, where, c.context);
      else axmp::FieldExpr::compile("1", c.params, c.vars, where);
      report(std::string(c.what) + ": the shadowing was not refused");
    } catch (const axmp::Error& e) {
      if (e.message() != c.message) report(std::string(c.what) + ": message '" + e.message() + "', want '" + c.message + "'");
      if (e.code() != "expr.parameter_shadows_variable") report(std::string(c.what) + ": code '" + e.code() + "'");
      if (e.where() != where || e.hint().empty()) report(std::string(c.what) + ": the refusal lacks the path or a hint");
      nlohmann::json want = {{"parameter", c.parameter}, {"variables", c.listed}};
      nlohmann::json got;
      try {
        got = nlohmann::json::parse(e.details());
      } catch (...) {
        report(std::string(c.what) + ": the details are not JSON: '" + e.details() + "'");
      }
      if (got != want) report(std::string(c.what) + ": details " + got.dump() + ", want " + want.dump());
    }
  }
  // a context changes nothing about a compile that is not refused, and the other refusals have no code of theirs
  try {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("2*y", {{"U", 1.0}}, {"x", "y"}, "/w", "a coordinate of this planar model");
    const double pt[2] = {0.0, 3.0};
    if (!same_bits(f.eval(pt), 6.0)) report("'2*y' compiled with a context is not 6 at y = 3");
  } catch (const axmp::Error& e) {
    report(std::string("a context made a valid compile fail: ") + e.what());
  }
  try {
    axmp::FieldExpr::compile("x", {}, {"x", "x"}, "/w", "a coordinate of this planar model");
    report("a variable listed twice compiled");
  } catch (const axmp::Error& e) {
    if (!e.code().empty()) report("the refusal of a variable listed twice has a code: " + e.code());
  }
}

// ------------------------------------------------------------------ many points at once

void check_many_points() {
  const axmp::ParamMap p = {{"U", 2.0}, {"H", 4.0}};
  const std::vector<std::string> vars = {"x", "y", "t"};
  const char* texts[] = {"6*U*x*(H-x)/H^2", "sin(x)*exp(-y) + atan2(y, x) + t", "min(x, y, t)*max(x, -y)",
                         "U*H", "x"};
  // 17 points: a prime count, so that a form that works a few points at a time has a remainder to handle
  const int n = 17;
  std::vector<double> v(static_cast<size_t>(n) * vars.size());
  for (int i = 0; i < n; ++i) {
    v[3 * i + 0] = 0.125 + 0.25 * i;
    v[3 * i + 1] = 1.0 - 0.0625 * i;
    v[3 * i + 2] = (i % 3 == 0) ? -0.5 * i : 0.375 * i;
  }
  for (const char* t : texts) {
    const axmp::FieldExpr f = axmp::FieldExpr::compile(t, p, vars, "/w");
    const double guard = 1234.5;
    std::vector<double> out(n + 2, guard);
    f.eval(v.data(), n, out.data() + 1);
    for (int i = 0; i < n; ++i) {
      const double one = f.eval(v.data() + 3 * i);
      if (!same_bits(one, out[i + 1])) {
        report(std::string("'") + t + "': point " + std::to_string(i) + " is " + bits(out[i + 1]) +
               " in the many-point form and " + bits(one) + " on its own");
      }
    }
    if (!same_bits(out[0], guard) || !same_bits(out[n + 1], guard)) {
      report(std::string("'") + t + "': the many-point form wrote outside the n values it was given");
    }
    // no points: nothing is written
    std::vector<double> none = {guard};
    f.eval(v.data(), 0, none.data());
    if (!same_bits(none[0], guard)) report(std::string("'") + t + "': n = 0 wrote a value");
  }
}

// ------------------------------------------------------------------ derivatives: the Taylor test

// f(v + h e_k) - f(v) - h d_k f(v) falls as h^2 when the derivative is exact, and as h when it is not.  The
// rate is read over four halvings of the step from h0 = 2^-6, where the remainder of these cases is between
// 1e-6 and 2e-2, falling to between 6e-9 and 5e-5 at the last step: seven orders of magnitude or more above
// its own rounding, which is of order eps |f|.  Measured over the table below with an exact derivative, the
// rates are 1.9704 to 2.0152; the band required here is [1.9, 2.1], which is 3.4 times the largest observed
// departure from 2 and far from the rate 1 that a missing or mis-scaled term in the chain rule leaves.
//
// A case whose value is linear in its variable has no second-order term: there the remainder is required to
// be at most 4 eps max(|f|, |h f'|) instead (measured: exactly zero for all four such cases, and the bound
// leaves room for another ordering of the same operations).
struct TaylorCase {
  const char* text;
  std::vector<std::string> vars;
  std::vector<double> at;
  int k;
  double h0;
  bool linear;  // the value is linear in variable k, so the remainder is at the level of rounding
};

// Whether the `len` characters at `at` stand alone in `t` as a name of their own, and are not part of a
// longer one ('log' inside 'log10', 'e' inside 'exp').
bool stands_alone(const std::string& t, size_t at, size_t len) {
  const bool left = at == 0 || !(std::isalnum(static_cast<unsigned char>(t[at - 1])) || t[at - 1] == '_');
  const size_t end = at + len;
  const bool right = end >= t.size() || !(std::isalnum(static_cast<unsigned char>(t[end])) || t[end] == '_');
  return left && right;
}

// The subexpressions a construct is applied to in `t`, as written: the argument of every call of the
// one-argument function `name`, or, for a one-character operator, the operand that follows every
// occurrence of it.  '*' and '/' have the same precedence and associate to the left, so the operand after
// a '/' is the whole denominator; '^' binds tighter than both, so the operand after a '^' is the whole
// exponent; and a unary sign belongs to the operand that follows it.
std::vector<std::string> operands_of(const std::string& t, const std::string& name) {
  std::vector<std::string> out;
  if (name == "^base") {
    // The operand before every '^': a parenthesised group with the function name in front of it, or a
    // name or number.
    for (size_t at = t.find('^'); at != std::string::npos; at = t.find('^', at + 1)) {
      size_t end = at;
      while (end > 0 && std::isspace(static_cast<unsigned char>(t[end - 1]))) --end;
      size_t i = end;
      if (i > 0 && t[i - 1] == ')') {
        int depth = 0;
        while (i > 0) {
          --i;
          if (t[i] == ')') {
            ++depth;
          } else if (t[i] == '(' && --depth == 0) {
            break;
          }
        }
        while (i > 0 && (std::isalnum(static_cast<unsigned char>(t[i - 1])) != 0 || t[i - 1] == '_')) --i;
      } else {
        while (i > 0 && (std::isalnum(static_cast<unsigned char>(t[i - 1])) != 0 || t[i - 1] == '_' ||
                         t[i - 1] == '.')) {
          --i;
        }
      }
      if (i < end) out.push_back(t.substr(i, end - i));
    }
    return out;
  }
  if (name == "atan2 second") {
    // The second argument of every atan2 call: the text after its depth-one comma.
    for (size_t at = t.find("atan2("); at != std::string::npos; at = t.find("atan2(", at + 1)) {
      int depth = 0;
      size_t comma = std::string::npos;
      size_t i = at + 5;
      for (; i < t.size(); ++i) {
        if (t[i] == '(') {
          ++depth;
        } else if (t[i] == ')' && --depth == 0) {
          break;
        } else if (t[i] == ',' && depth == 1 && comma == std::string::npos) {
          comma = i;
        }
      }
      if (comma != std::string::npos && i < t.size()) out.push_back(t.substr(comma + 1, i - comma - 1));
    }
    return out;
  }
  if (name.size() == 1 && std::isalpha(static_cast<unsigned char>(name[0])) == 0) {
    for (size_t at = t.find(name[0]); at != std::string::npos; at = t.find(name[0], at + 1)) {
      size_t i = at + 1;
      while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
      const size_t start = i;
      while (i < t.size() && (t[i] == '-' || t[i] == '+')) ++i;
      int depth = 0;
      for (; i < t.size(); ++i) {
        const char c = t[i];
        if (c == '(') {
          ++depth;
        } else if (c == ')') {
          if (depth == 0) break;
          --depth;
        } else if (depth == 0 && (c == '+' || c == '-' || c == '*' || c == '/' || c == ',')) {
          break;
        }
      }
      if (i > start) out.push_back(t.substr(start, i - start));
    }
    return out;
  }
  for (size_t at = t.find(name); at != std::string::npos; at = t.find(name, at + 1)) {
    if (!stands_alone(t, at, name.size())) continue;
    size_t i = at + name.size();
    while (i < t.size() && std::isspace(static_cast<unsigned char>(t[i]))) ++i;
    if (i >= t.size() || t[i] != '(') continue;
    const size_t start = i + 1;
    int depth = 0;
    for (; i < t.size(); ++i) {
      if (t[i] == '(') {
        ++depth;
      } else if (t[i] == ')' && --depth == 0) {
        break;
      }
    }
    if (i < t.size() && i > start) out.push_back(t.substr(start, i - start));
  }
  return out;
}

// The remainder is computed at a point and a step; the rate between two steps is log2 of their ratio.
void check_taylor_case(const TaylorCase& c, const axmp::ParamMap& p) {
  const double eps = 2.220446049250313e-16;
  const axmp::FieldExpr f = axmp::FieldExpr::compile(c.text, p, c.vars, "/w");
  const double f0 = f.eval(c.at.data());
  const double d0 = f.eval_d(c.at.data(), c.k);
  if (!std::isfinite(f0) || !std::isfinite(d0)) {
    report(std::string("'") + c.text + "': the value " + bits(f0) + " or the derivative " + bits(d0) +
           " is not finite at the test point");
    return;
  }
  double r[5] = {0, 0, 0, 0, 0};
  for (int j = 0; j < 5; ++j) {
    const double h = c.h0 / static_cast<double>(1 << j);
    std::vector<double> at = c.at;
    at[c.k] += h;
    r[j] = std::fabs(f.eval(at.data()) - f0 - h * d0);
  }
  if (c.linear) {
    const double floor_at = 4.0 * eps * std::fmax(std::fabs(f0), std::fabs(c.h0 * d0));
    for (int j = 0; j < 5; ++j) {
      if (!(r[j] <= floor_at)) {
        report(std::string("'") + c.text + "' d/d" + c.vars[c.k] + " is linear, but the remainder " +
               bits(r[j]) + " at step " + std::to_string(j) + " is above the rounding level " +
               bits(floor_at));
      }
    }
  } else {
    // the first-order term must be the one that is cancelled: without it the remainder would be |h d0|.
    // Measured over the table, the ratio is at most 0.045; the bound is 0.5.
    if (d0 != 0.0 && !(r[0] < 0.5 * std::fabs(c.h0 * d0))) {
      report(std::string("'") + c.text + "': the remainder " + bits(r[0]) + " at h = " + bits(c.h0) +
             " is not small against |h f'| = " + bits(std::fabs(c.h0 * d0)));
    }
    for (int j = 0; j + 1 < 5; ++j) {
      if (r[j + 1] <= 0.0) {
        report(std::string("'") + c.text + "' d/d" + c.vars[c.k] + ": the remainder is exactly zero at step " +
               std::to_string(j + 1) + ", so the case is linear and is not marked so");
        return;
      }
      const double rate = std::log2(r[j] / r[j + 1]);
      if (!(rate > 1.9 && rate < 2.1)) {
        report(std::string("'") + c.text + "' d/d" + c.vars[c.k] + ": the remainder falls at rate " +
               bits(rate) + " between h = " + bits(c.h0 / (1 << j)) + " and h/2 (want 2)");
      }
    }
  }
  // A second reading of the same derivative, which uses no derivative of ours either: the five-point
  // central difference at h = 2^-9, whose own error is O(h^4 f''''') + eps |f| / h.  Measured over the
  // table with an exact derivative it agrees to 2.8e-9 of max(|f'|, 1); the bound here is 1e-7 of it.
  const double h = 1.0 / 512.0;
  auto shifted = [&](double dh) {
    std::vector<double> at = c.at;
    at[c.k] += dh;
    return f.eval(at.data());
  };
  const double cd = (8.0 * (shifted(h) - shifted(-h)) - (shifted(2 * h) - shifted(-2 * h))) / (12.0 * h);
  const double scale = std::fmax(std::fabs(d0), 1.0);
  if (!(std::fabs(cd - d0) <= 1e-7 * scale)) {
    report(std::string("'") + c.text + "' d/d" + c.vars[c.k] + ": eval_d gives " + bits(d0) +
           ", a five-point central difference " + bits(cd));
  }
}

void check_derivatives() {
  const axmp::ParamMap p = {{"U", 2.0}, {"H", 4.0}, {"a", 0.75}};
  const std::vector<std::string> xyz = {"x", "y", "z"};
  // Every function of the grammar, every operator, and the constants, at points away from the kinks of
  // abs, min and max and from the branch points of sqrt, log, asin and acos.
  const std::vector<double> at = {0.5, 0.25, 0.125};
  // A second point, whose y is not a power of two: at this point no inner derivative of the compositions
  // below is 1 or a power of two by accident, so a chain-rule factor that is dropped or mis-scaled changes
  // the derivative.
  const std::vector<double> atc = {0.5, 0.3, 0.125};
  const std::vector<TaylorCase> cases = {
      {"x + y - z", xyz, at, 0, 0.015625, true},
      {"x*y/z", xyz, at, 1, 0.015625, true},
      {"-x*x + 3*x", xyz, at, 0, 0.015625, false},
      {"x^3", xyz, at, 0, 0.015625, false},
      {"x^y", xyz, at, 0, 0.015625, false},
      {"x^y", xyz, at, 1, 0.015625, false},
      {"2^x", xyz, at, 0, 0.015625, false},
      {"x^-2", xyz, at, 0, 0.015625, false},
      {"pow(x, y)", xyz, at, 1, 0.015625, false},
      {"sqrt(x)", xyz, at, 0, 0.015625, false},
      {"exp(-x^2)", xyz, at, 0, 0.015625, false},
      {"log(x)", xyz, at, 0, 0.015625, false},
      {"log10(x*y)", xyz, at, 0, 0.015625, false},
      {"sin(x)*cos(y)", xyz, at, 0, 0.015625, false},
      {"sin(x)*cos(y)", xyz, at, 1, 0.015625, false},
      {"tan(x)", xyz, at, 0, 0.015625, false},
      {"asin(x)", xyz, at, 0, 0.015625, false},
      {"acos(x)", xyz, at, 0, 0.015625, false},
      {"atan(x*y)", xyz, at, 0, 0.015625, false},
      {"atan2(x, y)", xyz, at, 0, 0.015625, false},
      {"atan2(x, y)", xyz, at, 1, 0.015625, false},
      {"abs(x)*x", xyz, at, 0, 0.015625, false},
      {"abs(-x^3)", xyz, at, 0, 0.015625, false},
      {"min(x^2, 1)", xyz, at, 0, 0.015625, false},
      {"max(x^2, 0.01)", xyz, at, 0, 0.015625, false},
      // the argument min selects here is z^2, which holds neither x nor y: the value does not move with
      // either of them, in both the value and the derivative
      {"min(x*y, z^2, 2)", xyz, at, 0, 0.015625, true},
      {"min(x*y, z^2, 2)", xyz, at, 1, 0.015625, true},
      {"max(exp(x), log(y + 2), 0.1)", xyz, at, 0, 0.015625, false},
      // the constants and the parameters inside a derivative
      {"pi*sin(x) + e^x + mu0*x^2", xyz, at, 0, 0.015625, false},
      {"6*U*y*(H-y)/H^2", {"x", "y"}, {0.5, 1.5}, 1, 0.015625, false},
      {"a*sqrt(x^2 + y^2)", xyz, at, 0, 0.015625, false},
      // a composition deep enough that a wrong chain rule cannot cancel
      {"log(1 + exp(sin(x*y) + sqrt(x + 1)))", xyz, at, 0, 0.015625, false},
      {"atan2(sin(x), cos(x) + 2)^2", xyz, at, 0, 0.015625, false},
      // the variable in a denominator: the term of the quotient rule that belongs to the denominator.  The
      // cases above divide by a constant or differentiate the numerator, where that term is absent.
      {"1/x", xyz, at, 0, 0.015625, false},
      {"y/x", xyz, at, 0, 0.015625, false},
      {"(x+1)/(x*y)", xyz, at, 0, 0.015625, false},  // both terms at once
      {"1/(2 + 1/x)", xyz, at, 0, 0.015625, false},  // a quotient in the denominator of a quotient
      // the chain rule where the inner derivative is neither 0 nor 1: a factor of exactly 1 would hide a
      // factor that is missing, and a factor of 0 would remove the term
      {"sqrt(3*x + 1)", xyz, atc, 0, 0.015625, false},
      {"tan(x*y)", xyz, atc, 0, 0.015625, false},
      {"cos(x*y)", xyz, atc, 0, 0.015625, false},
      {"asin(x*y)", xyz, atc, 0, 0.015625, false},
      {"acos(x*y)", xyz, atc, 0, 0.015625, false},
      // the term of a power that belongs to its exponent, with d(2y)/dy = 2 and d(y*y)/dy = 2y
      {"x^(2*y)", xyz, atc, 1, 0.015625, false},
      {"pow(x, y*y)", xyz, atc, 1, 0.015625, false},
  };
  for (const TaylorCase& c : cases) check_taylor_case(c, p);

  // Closed forms for some of the same cases: a third reference, independent of both the Taylor test and
  // the implementation.  The first five are worked by hand at (x, y) = (0.5, 0.25); the quotients are
  // exact rationals; the rest are the closed form evaluated in 60-digit decimal arithmetic at the binary
  // values of the point (the formula, not the implementation's order of operations), written out to 20
  // digits.  The tolerance is 4 eps of the value, which allows another ordering of the same operations
  // (measured: at most 0.71 eps of the value, so under one bit of each).
  {
    const double eps = 2.220446049250313e-16;
    struct Closed {
      const char* text;
      std::vector<double> at;
      int k;
      double want;
    };
    const std::vector<Closed> closed = {
        {"sqrt(x)", at, 0, 0.70710678118654752440},       // 1/(2 sqrt(1/2)) = sqrt(1/2)
        {"log(x)", at, 0, 2.0},                           // 1/x
        {"exp(-x^2)", at, 0, -0.77880078307140486825},    // -2 x exp(-x^2) = -exp(-1/4)
        {"atan2(x, y)", at, 0, 0.8},                      // y/(x^2 + y^2)
        {"sin(x)*cos(y)", at, 0, 0.85030064529223281511}, // cos(x) cos(y)
        // the quotient rule's denominator term, where it is exact in binary
        {"1/x", at, 0, -4.0},                             // -1/x^2
        {"y/x", at, 0, -1.0},                             // -y/x^2
        {"(x+1)/(x*y)", at, 0, -16.0},                    // (x y - (x+1) y) / (x y)^2 = -1/(x^2 y)
        {"1/(2 + 1/x)", at, 0, 0.25},                     // (1/x^2) / (2 + 1/x)^2
        // the chain-rule factor of each composition, at (x, y) = (0.5, 0.3)
        {"sqrt(3*x + 1)", atc, 0, 0.94868329805051376802},    // 3 / (2 sqrt(3x+1))
        {"tan(x*y)", atc, 0, 0.30685255624125851170},         // y (1 + tan(xy)^2)
        {"cos(x*y)", atc, 0, -0.04483143974207976640},        // -y sin(xy)
        {"asin(x*y)", atc, 0, 0.30343304245450414269},        // y / sqrt(1 - (xy)^2)
        {"acos(x*y)", atc, 0, -0.30343304245450414269},       // -y / sqrt(1 - (xy)^2)
        {"x^(2*y)", atc, 1, -0.91461318807877556303},         // 2 log(x) x^(2y)
        {"pow(x, y*y)", atc, 1, -0.39073652681377246676},     // 2 y log(x) x^(y^2)
    };
    for (const Closed& c : closed) {
      const axmp::FieldExpr f = axmp::FieldExpr::compile(c.text, p, xyz, "/w");
      const double got = f.eval_d(c.at.data(), c.k);
      if (!(std::fabs(got - c.want) <= 4.0 * eps * std::fabs(c.want))) {
        report(std::string("'") + c.text + "' d/d" + xyz[c.k] + " is " + bits(got) + ", the closed form " +
               bits(c.want));
      }
    }
  }

  // Coverage: every function and operator of the grammar appears above.  A case removed from the table
  // takes its function's coverage with it, and this fails.
  const char* needed[] = {"sqrt", "exp",  "log", "log10", "sin", "cos", "tan", "asin", "acos",
                          "atan", "atan2", "abs", "pow",  "min", "max", "pi",  "e",    "mu0",
                          "+",    "-",     "*",   "/",    "^"};
  for (const char* name : needed) {
    const size_t len = std::strlen(name);
    // a name must stand alone, not inside a longer one ('log' inside 'log10', 'e' inside 'exp')
    const bool identifier = std::isalpha(static_cast<unsigned char>(name[0])) != 0;
    bool seen = false;
    for (const TaylorCase& c : cases) {
      const std::string t = c.text;
      for (size_t at = t.find(name); at != std::string::npos && !seen; at = t.find(name, at + 1)) {
        if (!identifier) {
          seen = true;
          break;
        }
        if (stands_alone(t, at, len)) seen = true;
      }
      if (seen) break;
    }
    if (!seen) report(std::string("no Taylor case covers '") + name + "'");
  }

  // Coverage of the terms a wrong chain rule hides.  A function applied to a bare variable has an inner
  // derivative of exactly 1 at every point, where a missing or mis-scaled factor changes nothing; the term
  // of a quotient that belongs to the denominator is reached only where the denominator's own derivative
  // is nonzero, and likewise the term of a power that belongs to the exponent.  Each construct below must
  // therefore appear in the table applied to a subexpression whose own derivative, at that case's point
  // and with respect to that case's variable, satisfies the requirement.  The subexpression is read out of
  // the case's text and compiled on its own, which asks the same question the chain rule asks.
  struct Inner {
    const char* construct;  // a one-argument function, or an operator whose second operand is meant
    bool off_one;           // the inner derivative must differ from +-1 as well as from 0
  };
  const Inner inner[] = {{"sqrt", true}, {"exp", true},  {"log", true},  {"log10", true}, {"sin", true},
                         {"cos", true},  {"tan", true},  {"asin", true}, {"acos", true},  {"atan", true},
                         {"abs", true},  {"^", true},    {"^base", true}, {"atan2 second", true},
                         {"/", false}};
  for (const Inner& want : inner) {
    bool seen = false;
    for (const TaylorCase& c : cases) {
      for (const std::string& arg : operands_of(c.text, want.construct)) {
        double di = 0.0;
        try {
          di = axmp::FieldExpr::compile(arg, p, c.vars, "/w").eval_d(c.at.data(), c.k);
        } catch (const axmp::Error&) {
          continue;  // not an expression on its own: it answers nothing either way
        }
        if (!std::isfinite(di) || di == 0.0) continue;
        if (want.off_one && (di == 1.0 || di == -1.0)) continue;
        seen = true;
        break;
      }
      if (seen) break;
    }
    if (!seen) {
      report(std::string("no Taylor case applies '") + want.construct +
             (want.off_one ? "' to a subexpression whose own derivative is neither 0 nor 1"
                           : "' to a second operand whose own derivative is nonzero"));
    }
  }

  // The three places where a first derivative is a convention rather than a formula: the kink of abs, the
  // tie of min and max, and the term of a power that belongs to its exponent.
  // abs at the kink: the derivative of |u| is sign(u) u', with sign(0) = 0.
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("abs(x)", p, {"x"}, "/w");
    const double zero[1] = {0.0};
    const double neg[1] = {-0.5};
    const double pos[1] = {0.5};
    if (!same_bits(f.eval_d(zero, 0), 0.0)) report("d|x|/dx at 0 is not +0");
    if (!same_bits(f.eval_d(neg, 0), -1.0)) report("d|x|/dx at -0.5 is not -1");
    if (!same_bits(f.eval_d(pos, 0), 1.0)) report("d|x|/dx at 0.5 is not 1");
  }
  // min and max take the derivative of the argument they select; on a tie the earliest of the tied
  // arguments wins, as it does for the value.
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("min(x, 2*x)", p, {"x"}, "/w");
    const double zero[1] = {0.0};
    const double pos[1] = {1.0};
    const double neg[1] = {-1.0};
    if (!same_bits(f.eval_d(zero, 0), 1.0)) report("d min(x, 2x)/dx at the tie x = 0 is not the first argument");
    if (!same_bits(f.eval_d(pos, 0), 1.0)) report("d min(x, 2x)/dx at x = 1 is not 1");
    if (!same_bits(f.eval_d(neg, 0), 2.0)) report("d min(x, 2x)/dx at x = -1 is not 2");
    const axmp::FieldExpr g = axmp::FieldExpr::compile("max(2*x, x)", p, {"x"}, "/w");
    if (!same_bits(g.eval_d(zero, 0), 2.0)) report("d max(2x, x)/dx at the tie x = 0 is not the first argument");
  }
  // a power whose exponent does not depend on the variable does not evaluate log(base): the derivative of
  // x^2 at a negative x is finite.
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("x^2", p, {"x"}, "/w");
    const double neg[1] = {-2.0};
    if (!same_bits(f.eval_d(neg, 0), -4.0)) report("d(x^2)/dx at x = -2 is not -4");
    const axmp::FieldExpr g = axmp::FieldExpr::compile("pow(x, 3)", p, {"x"}, "/w");
    if (!same_bits(g.eval_d(neg, 0), 12.0)) report("d(pow(x, 3))/dx at x = -2 is not 12");
  }
  // a power that is constant has the derivative of a constant, and not the 0 * inf of its formula: x^0 is
  // 1 for every x, including x = 0, where b a^(b-1) is 0 * inf; 0^x is 0 for every x > 0, where
  // a^b log(a) is 0 * -inf.
  {
    const double zero[1] = {0.0};
    const double half[1] = {0.5};
    const double one[1] = {1.0};
    for (const char* t : {"x^0", "pow(x, 0)"}) {
      const axmp::FieldExpr f = axmp::FieldExpr::compile(t, p, {"x"}, "/w");
      if (!same_bits(f.eval(zero), 1.0)) report(std::string("'") + t + "' at x = 0 is not 1");
      if (!same_bits(f.eval_d(zero, 0), 0.0)) report(std::string("d(") + t + ")/dx at x = 0 is not +0");
    }
    const axmp::FieldExpr g = axmp::FieldExpr::compile("0^x", p, {"x"}, "/w");
    if (!same_bits(g.eval(one), 0.0)) report("'0^x' at x = 1 is not 0");
    if (!same_bits(g.eval_d(one, 0), 0.0)) report("d(0^x)/dx at x = 1 is not +0");
    if (!same_bits(g.eval_d(half, 0), 0.0)) report("d(0^x)/dx at x = 0.5 is not +0");
  }
  // a subtree without the variable has derivative exactly +0, and so does a text without variables.  The
  // sign of that zero is part of the convention, and it is the one place where the guarded rules differ
  // from the plain ones: at this point the plain product rule gives 0*(-2) + (-1)*0 = -0, and the plain
  // quotient rule 0/(-2) - (-0.5)*0/(-2) = -0, where a derivative-free subtree must contribute +0.
  {
    const axmp::FieldExpr f = axmp::FieldExpr::compile("U*H + sin(a)", p, {"x"}, "/w");
    const double pt[1] = {0.5};
    if (!same_bits(f.eval_d(pt, 0), 0.0)) report("the derivative of a text without the variable is not +0");
    const double neg[3] = {-1.0, -2.0, 0.5};
    const axmp::FieldExpr prod = axmp::FieldExpr::compile("x*y", p, xyz, "/w");
    if (!same_bits(prod.eval_d(neg, 2), 0.0)) {
      report("d/dz of 'x*y' at (-1, -2, 0.5) is " + bits(prod.eval_d(neg, 2)) + ", not +0");
    }
    const double quo[3] = {1.0, -2.0, 0.5};
    const axmp::FieldExpr div = axmp::FieldExpr::compile("x/y", p, xyz, "/w");
    if (!same_bits(div.eval_d(quo, 2), 0.0)) {
      report("d/dz of 'x/y' at (1, -2, 0.5) is " + bits(div.eval_d(quo, 2)) + ", not +0");
    }
    const axmp::FieldExpr arc = axmp::FieldExpr::compile("atan2(x, y)", p, xyz, "/w");
    if (!same_bits(arc.eval_d(quo, 2), 0.0)) {
      report("d/dz of 'atan2(x, y)' at (1, -2, 0.5) is " + bits(arc.eval_d(quo, 2)) + ", not +0");
    }
  }
  // the divergence of a vector of expressions is the sum of the components' own derivatives: the identity
  // the conservative advective forms need, read here against a closed form.
  {
    const std::vector<std::string> vars = {"r", "z"};
    const axmp::FieldExpr ur = axmp::FieldExpr::compile("r*z", p, vars, "/w");
    const axmp::FieldExpr uz = axmp::FieldExpr::compile("-z^2/2", p, vars, "/w");
    const double pt[2] = {0.75, 0.5};
    const double div = ur.eval_d(pt, 0) + uz.eval_d(pt, 1);
    if (!same_bits(div, 0.0)) {
      report("the divergence of (r z, -z^2/2) by eval_d is " + bits(div) + ", not the closed form 0");
    }
  }
}

// ------------------------------------------------------------------ values that are not finite

void check_non_finite() {
  // With a free variable, whether a value is finite is a property of the point, not of the text: eval
  // returns what the arithmetic gives and refuses nothing.
  const axmp::FieldExpr f = axmp::FieldExpr::compile("1/x", {}, {"x"}, "/w");
  const double zero[1] = {0.0};
  const double one[1] = {1.0};
  try {
    const double v = f.eval(zero);
    if (!(std::isinf(v) && v > 0)) report("1/x at x = 0 is " + bits(v) + ", not +inf");
    if (!same_bits(f.eval(one), 1.0)) report("1/x at x = 1 is not 1");
  } catch (const std::exception& e) {
    report(std::string("1/x at x = 0 threw: ") + e.what());
  }
  const axmp::FieldExpr g = axmp::FieldExpr::compile("log(x)", {}, {"x"}, "/w");
  const double neg[1] = {-1.0};
  try {
    if (!std::isnan(g.eval(neg))) report("log(x) at x = -1 is not NaN");
  } catch (const std::exception& e) {
    report(std::string("log(x) at x = -1 threw: ") + e.what());
  }
  // min and max keep a NaN argument whatever its position, as the design evaluator refuses one
  for (const char* t : {"min(x, 1)", "min(1, x)", "max(x, 1)", "max(1, x)"}) {
    const axmp::FieldExpr m = axmp::FieldExpr::compile(t, {}, {"x"}, "/w");
    const double nan_pt[1] = {std::nan("")};
    try {
      const double v = m.eval(nan_pt);
      if (!std::isnan(v)) report(std::string("'") + t + "' at x = NaN is " + bits(v) + ", not NaN");
    } catch (const std::exception& e) {
      report(std::string("'") + t + "' at x = NaN threw: " + e.what());
    }
  }
}

// ------------------------------------------------------------------ several threads

void check_threads() {
  // A compiled expression is evaluated inside assembly loops, which may run on several threads: eval,
  // eval_d and the many-point form keep no state in the object.
  const axmp::ParamMap p = {{"U", 2.0}, {"H", 4.0}};
  const std::vector<std::string> vars = {"x", "y"};
  const axmp::FieldExpr f =
      axmp::FieldExpr::compile("6*U*y*(H-y)/H^2 + sin(x)*exp(-y) + min(x, y)*max(x, 2)", p, vars, "/w");
  // Each thread has points of its own, so that a buffer shared between calls shows up as a value from
  // another thread's points rather than as the same answer computed twice.
  const int nthreads = 4;
  const int n = 64;
  std::vector<std::vector<double>> pts(nthreads, std::vector<double>(2 * n));
  std::vector<std::vector<double>> serial(nthreads, std::vector<double>(n));
  std::vector<std::vector<double>> serial_d(nthreads, std::vector<double>(n));
  for (int t = 0; t < nthreads; ++t) {
    for (int i = 0; i < n; ++i) {
      pts[t][2 * i] = 0.0625 * i - 2.0 + 0.5 * t;
      pts[t][2 * i + 1] = 1.0 - 0.03125 * i + 0.25 * t;
    }
    for (int i = 0; i < n; ++i) {
      serial[t][i] = f.eval(pts[t].data() + 2 * i);
      serial_d[t][i] = f.eval_d(pts[t].data() + 2 * i, 1);
    }
  }
  std::vector<std::vector<double>> got(nthreads, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> got_d(nthreads, std::vector<double>(n, 0.0));
  std::vector<std::thread> workers;
  for (int t = 0; t < nthreads; ++t) {
    workers.emplace_back([&, t]() {
      for (int rep = 0; rep < 200; ++rep) {
        f.eval(pts[t].data(), n, got[t].data());
        for (int i = 0; i < n; ++i) got_d[t][i] = f.eval_d(pts[t].data() + 2 * i, 1);
      }
    });
  }
  for (std::thread& w : workers) w.join();
  for (int t = 0; t < nthreads; ++t) {
    for (int i = 0; i < n; ++i) {
      if (!same_bits(got[t][i], serial[t][i]) || !same_bits(got_d[t][i], serial_d[t][i])) {
        report("thread " + std::to_string(t) + " point " + std::to_string(i) + ": value " + bits(got[t][i]) +
               " (on one thread: " + bits(serial[t][i]) + "), derivative " + bits(got_d[t][i]) + " (" +
               bits(serial_d[t][i]) + ")");
        return;
      }
    }
  }
}

}  // namespace

int main() {
  check_grammar();
  check_variables();
  check_value_semantics();
  check_shadowing();
  check_many_points();
  check_derivatives();
  check_non_finite();
  check_threads();
  if (failures != 0) {
    std::printf("%d field-expression check(s) failed\n", failures);
    return 1;
  }
  std::puts("field expressions (grammar against eval_expr, variables, Taylor test, refusals, threads): passed");
  return 0;
}
