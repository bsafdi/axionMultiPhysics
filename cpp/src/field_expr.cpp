#include "axionmultiphysics/field_expr.hpp"

#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionmultiphysics/core.hpp"
// The codes and the texts of the refusals that a model document can meet in an expression (the table of codes is one
// header for the whole model path); nothing of the model document itself, and no mesh, reaches this file.
#include "model_errors.hpp"
// The constants pi and mu0 of the grammar, so that an identifier means the same double here as it means in
// the design evaluator.
#include "axionmultiphysics/units.hpp"

namespace axmp {
namespace {

// The same cap as the design evaluator, so that the same texts are accepted; it bounds the recursion of the
// parser and turns a stack overflow into an error.
constexpr int kMaxExprDepth = 1000;

// ln 10, for d log10(u)/dv = u'/(u ln 10): the value std::log(10.0) returns, written out so that a derivative
// costs no library call.
constexpr double kLn10 = 2.30258509299404568402;

// The names of a context, for a message that says which names an expression may use.
std::string name_list(const std::vector<std::string>& names) {
  if (names.empty()) return "none";
  std::string s;
  for (size_t k = 0; k < names.size(); ++k) {
    if (k != 0) s += ", ";
    s += names[k];
  }
  return s;
}

// Room for the value of every node of one evaluation: on the stack for the expressions a document holds, on
// the heap for a larger tree.  It is a local of each call, never a member, because one compiled expression is
// evaluated from several threads at once.
struct Scratch {
  static constexpr size_t kInline = 128;
  explicit Scratch(size_t n) : p(inline_) {
    if (n > kInline) {
      heap_.resize(n);
      p = heap_.data();
    }
  }
  double* p;

 private:
  double inline_[kInline];
  std::vector<double> heap_;
};

}  // namespace

// The parser: the structure, the precedence, the number tokenizer, the function table and the messages of the
// design evaluator (cpp/src/expr.cpp), with a node built where that evaluator computes a value.  A node whose
// children are all constant is folded at once, in the grammar's own order, so that a text without variables
// costs one lookup at evaluation and so that its value is the design evaluator's, bit for bit.
struct FieldExprCompiler {
  FieldExprCompiler(const std::string& text, const ParamMap& params,
                    const std::vector<std::string>& variables, const std::string& path, FieldExpr& tree)
      : s(text), p(params), vars(variables), where(path), out(tree), used(variables.size(), 0) {}

  const std::string& s;
  const ParamMap& p;
  const std::vector<std::string>& vars;
  const std::string& where;
  FieldExpr& out;
  size_t i = 0;            // how far the text is read
  int depth = 0;           // nesting of signs, '^' exponents, parentheses and function arguments
  std::vector<char> used;  // per variable of `vars`: does it occur in the text?

  [[noreturn]] void fail(const std::string& msg,
                         const std::string& hint = "Use numbers, parameter names, + - * / ^, parentheses and "
                                                   "functions such as sqrt, exp, log, min, max.") const {
    throw Error(strcat("cannot evaluate expression '", s, "': ", msg, " (near position ", i, ")"), where, hint);
  }

  // ---- the arena
  using Op = FieldExpr::Op;
  using Node = FieldExpr::Node;

  int push(const Node& nd) {
    out.nodes_.push_back(nd);
    return static_cast<int>(out.nodes_.size()) - 1;
  }
  int constant(double v) {
    Node nd;
    nd.op = Op::kConst;
    nd.value = v;
    return push(nd);
  }
  bool is_const(int n) const { return out.nodes_[static_cast<size_t>(n)].op == Op::kConst; }
  double cval(int n) const { return out.nodes_[static_cast<size_t>(n)].value; }
  int unary_node(Op op, int c) {
    if (is_const(c)) return constant(FieldExpr::apply(op, cval(c), 0.0));
    Node nd;
    nd.op = op;
    nd.a = c;
    return push(nd);
  }
  int binary_node(Op op, int l, int r) {
    if (is_const(l) && is_const(r)) return constant(FieldExpr::apply(op, cval(l), cval(r)));
    Node nd;
    nd.op = op;
    nd.a = l;
    nd.b = r;
    return push(nd);
  }
  int nary_node(Op op, const std::vector<int>& a) {
    bool all_const = true;
    for (int n : a) all_const = all_const && is_const(n);
    // The check the design evaluator makes at this node, made at this node: a constant NaN argument is an
    // error whatever its position and whatever the other arguments are, because it does not depend on the
    // point.
    const char* name = (op == Op::kMin) ? "min" : "max";
    for (size_t k = 0; k < a.size(); ++k) {
      if (is_const(a[k]) && std::isnan(cval(a[k]))) {
        fail(strcat(name, "() argument ", k + 1, " is not a number (NaN)"),
             "Check for sqrt, log, asin or acos of an out-of-range value, or 0/0, inside the arguments.");
      }
    }
    if (all_const) {
      std::vector<double> v(a.size(), 0.0);
      std::vector<int> idx(a.size(), 0);
      for (size_t k = 0; k < a.size(); ++k) {
        v[k] = cval(a[k]);
        idx[k] = static_cast<int>(k);
      }
      return constant(v[static_cast<size_t>(
          FieldExpr::selected(op, v.data(), idx.data(), static_cast<int>(v.size())))]);
    }
    Node nd;
    nd.op = op;
    nd.args = static_cast<int>(out.args_.size());
    nd.argc = static_cast<int>(a.size());
    for (int n : a) out.args_.push_back(n);
    return push(nd);
  }

  // ---- the grammar
  void ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  }
  bool eat(char c) {
    ws();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  int expr() {
    int v = term();
    for (;;) {
      if (eat('+')) {
        const int r = term();
        v = binary_node(Op::kAdd, v, r);
      } else if (eat('-')) {
        const int r = term();
        v = binary_node(Op::kSub, v, r);
      } else {
        return v;
      }
    }
  }
  int term() {
    int v = unary();
    for (;;) {
      if (eat('*')) {
        const int r = unary();
        v = binary_node(Op::kMul, v, r);
      } else if (eat('/')) {
        const int r = unary();
        v = binary_node(Op::kDiv, v, r);
      } else {
        return v;
      }
    }
  }
  // Standard precedence: unary minus binds looser than '^' (-2^2 = -4), '^' is right-associative
  // (2^3^2 = 2^9) and its exponent may carry a sign (2^-2 = 0.25).  Every level of nesting passes through
  // unary(), so the cap here bounds the recursion.
  int unary() {
    if (++depth > kMaxExprDepth) {
      fail(strcat("nesting deeper than ", kMaxExprDepth, " levels"),
           "Simplify the expression, or split it into several parameters.");
    }
    int v = -1;
    if (eat('-')) v = unary_node(Op::kNeg, unary());
    else if (eat('+')) v = unary();
    else v = power();
    --depth;
    return v;
  }
  int power() {
    const int b = primary();
    if (eat('^')) {
      const int e = unary();
      return binary_node(Op::kPow, b, e);
    }
    return b;
  }
  int number() {
    ws();
    size_t j = i;
    while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '.')) ++j;
    if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
      size_t k = j + 1;
      if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
      if (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) {
        j = k;
        while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
      }
    }
    // Strict, as in the design evaluator: the whole token must be one number (no silent '1.2.5' -> 1.2), and
    // it must be zero or a normal double; std::stod reports overflow (1e400) and underflow (the subnormal
    // 1e-310, 1e-400 -> 0) as out_of_range, and such literals are refused rather than rounded.
    const std::string tok = s.substr(i, j - i);
    size_t pos = 0;
    double v = 0.0;
    try {
      v = std::stod(tok, &pos);
    } catch (const std::out_of_range&) {
      fail(strcat("number '", tok, "' is out of the range of double precision"),
           "Numbers must be 0 or have a magnitude between 2.2e-308 and 1.7e308; check the exponent.");
    } catch (const std::invalid_argument&) {
      fail(strcat("malformed number '", tok, "'"),
           "Write numbers as 12, 1.25, .5, 2. or 1.5e-3 (one decimal point, digits in the exponent).");
    }
    if (pos != tok.size()) {
      fail(strcat("malformed number '", tok, "'"),
           "Write numbers as 12, 1.25, .5, 2. or 1.5e-3 (one decimal point, digits in the exponent).");
    }
    i = j;
    return constant(v);
  }
  int primary() {
    ws();
    if (i >= s.size()) fail("unexpected end");
    const char c = s[i];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') return number();
    if (eat('(')) {
      const int v = expr();
      if (!eat(')')) fail("missing ')'");
      return v;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t j = i;
      while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
      const std::string id = s.substr(i, j - i);
      i = j;
      // An identifier followed by '(' is a function call, even when a variable or a parameter has that name.
      if (eat('(')) {
        std::vector<int> a;
        if (!eat(')')) {
          do a.push_back(expr());
          while (eat(','));
          if (!eat(')')) fail("missing ')' after function arguments");
        }
        return call(id, a);
      }
      // A variable of the context first, then a parameter, then a constant of the grammar: a variable hides a
      // parameter and a constant of the same name, and a parameter hides a constant, as in the design
      // evaluator.  A parameter that has the name of a variable was refused before parsing began.
      for (size_t k = 0; k < vars.size(); ++k) {
        if (vars[k] == id) {
          used[k] = 1;
          Node nd;
          nd.op = Op::kVar;
          nd.var = static_cast<int>(k);
          return push(nd);
        }
      }
      const auto it = p.find(id);
      if (it != p.end()) return constant(it->second);
      if (id == "pi") return constant(units::pi);
      if (id == "e") return constant(std::exp(1.0));
      if (id == "mu0") return constant(units::mu0);
      fail("unknown name '" + id + "'");
    }
    fail(std::string("unexpected character '") + c + "'");
  }
  int call(const std::string& f, const std::vector<int>& a) {
    const auto need = [&](size_t n) {
      if (a.size() != n) fail(strcat(f, "() takes ", n, " argument(s)"));
    };
    if (f == "sqrt") { need(1); return unary_node(Op::kSqrt, a[0]); }
    if (f == "exp") { need(1); return unary_node(Op::kExp, a[0]); }
    if (f == "log") { need(1); return unary_node(Op::kLog, a[0]); }
    if (f == "log10") { need(1); return unary_node(Op::kLog10, a[0]); }
    if (f == "sin") { need(1); return unary_node(Op::kSin, a[0]); }
    if (f == "cos") { need(1); return unary_node(Op::kCos, a[0]); }
    if (f == "tan") { need(1); return unary_node(Op::kTan, a[0]); }
    if (f == "asin") { need(1); return unary_node(Op::kAsin, a[0]); }
    if (f == "acos") { need(1); return unary_node(Op::kAcos, a[0]); }
    if (f == "atan") { need(1); return unary_node(Op::kAtan, a[0]); }
    if (f == "atan2") { need(2); return binary_node(Op::kAtan2, a[0], a[1]); }
    if (f == "abs") { need(1); return unary_node(Op::kAbs, a[0]); }
    if (f == "pow") { need(2); return binary_node(Op::kPow, a[0], a[1]); }
    if (f == "min" || f == "max") {
      if (a.empty()) fail(f + "() needs arguments");
      return nary_node(f == "min" ? Op::kMin : Op::kMax, a);
    }
    fail("unknown function '" + f + "'");
  }
};

double FieldExpr::apply(Op op, double a, double b) {
  switch (op) {
    case Op::kAdd: return a + b;
    case Op::kSub: return a - b;
    case Op::kMul: return a * b;
    case Op::kDiv: return a / b;
    case Op::kNeg: return -a;
    case Op::kPow: return std::pow(a, b);
    case Op::kSqrt: return std::sqrt(a);
    case Op::kExp: return std::exp(a);
    case Op::kLog: return std::log(a);
    case Op::kLog10: return std::log10(a);
    case Op::kSin: return std::sin(a);
    case Op::kCos: return std::cos(a);
    case Op::kTan: return std::tan(a);
    case Op::kAsin: return std::asin(a);
    case Op::kAcos: return std::acos(a);
    case Op::kAtan: return std::atan(a);
    case Op::kAtan2: return std::atan2(a, b);
    case Op::kAbs: return std::fabs(a);
    case Op::kConst:
    case Op::kVar:
    case Op::kMin:
    case Op::kMax:
      break;  // a constant, a variable and the fold of min and max are not a function of two children
  }
  return 0.0;
}

double FieldExpr::derivative(Op op, double a, double da, double b, double db, double value) {
  // Forward mode, with every term guarded by the derivative that multiplies it: a term is evaluated only
  // where that derivative is nonzero at this point.  A subtree in which the variable does not occur therefore
  // contributes exactly +0 and never a value of its own, which is what keeps d(x^2)/dx finite at a negative x
  // (log(-2) * 0 is NaN, and 0 is not).
  double d = 0.0;
  switch (op) {
    case Op::kAdd:
      if (da != 0.0) d += da;
      if (db != 0.0) d += db;
      return d;
    case Op::kSub:
      if (da != 0.0) d += da;
      if (db != 0.0) d -= db;
      return d;
    case Op::kMul:
      if (da != 0.0) d += da * b;
      if (db != 0.0) d += a * db;
      return d;
    case Op::kDiv:  // (a/b)' = a'/b - (a/b) b'/b
      if (da != 0.0) d += da / b;
      if (db != 0.0) d -= value * db / b;
      return d;
    case Op::kNeg:
      return (da != 0.0) ? -da : 0.0;
    case Op::kPow:  // d(a^b) = b a^(b-1) a' + a^b log(a) b'
      // Each term is guarded by the other operand as well, because where that operand is zero the power is
      // constant in the one being differentiated and the term is a zero times an infinity: a^0 is 1 for
      // every a, so its derivative in a is 0 and not the 0 * a^-1 that is 0 * inf at a = 0, and 0^b is 0
      // for every b > 0, so its derivative in b is 0 and not the 0 * log(0) that is 0 * -inf.
      if (da != 0.0 && b != 0.0) d += b * std::pow(a, b - 1.0) * da;
      if (db != 0.0 && a != 0.0) d += value * std::log(a) * db;
      return d;
    case Op::kSqrt: return (da != 0.0) ? da / (2.0 * value) : 0.0;
    case Op::kExp: return (da != 0.0) ? value * da : 0.0;
    case Op::kLog: return (da != 0.0) ? da / a : 0.0;
    case Op::kLog10: return (da != 0.0) ? da / (a * kLn10) : 0.0;
    case Op::kSin: return (da != 0.0) ? std::cos(a) * da : 0.0;
    case Op::kCos: return (da != 0.0) ? -std::sin(a) * da : 0.0;
    case Op::kTan: return (da != 0.0) ? (1.0 + value * value) * da : 0.0;  // sec^2 = 1 + tan^2
    case Op::kAsin: return (da != 0.0) ? da / std::sqrt(1.0 - a * a) : 0.0;
    case Op::kAcos: return (da != 0.0) ? -da / std::sqrt(1.0 - a * a) : 0.0;
    case Op::kAtan: return (da != 0.0) ? da / (1.0 + a * a) : 0.0;
    case Op::kAtan2:  // d atan2(a, b) = (a' b - a b') / (a^2 + b^2)
      if (da == 0.0 && db == 0.0) return 0.0;
      if (da != 0.0) d += da * b;
      if (db != 0.0) d -= a * db;
      return d / (a * a + b * b);
    case Op::kAbs:  // sign(a) a', with sign(0) = 0: at the kink the derivative is zero, not one side's
      if (da == 0.0 || a == 0.0) return 0.0;
      return (a > 0.0) ? da : -da;
    case Op::kConst:
    case Op::kVar:
    case Op::kMin:
    case Op::kMax:
      break;  // a leaf, and the fold of min and max, take their derivative from the sweep itself
  }
  return 0.0;
}

int FieldExpr::selected(Op op, const double* val, const int* idx, int argc) {
  // The fold of the design evaluator: v = a[0], then v = min(v, a[i]) in order, which keeps the earliest of
  // equal arguments, for the value and so for the derivative.
  int k = 0;
  for (int j = 1; j < argc; ++j) {
    const double x = val[idx[j]];
    const bool take = (op == Op::kMin) ? (x < val[idx[k]]) : (val[idx[k]] < x);
    if (take) k = j;
  }
  // std::min and std::max drop a NaN that is not the first argument.  The design evaluator refuses a NaN in
  // any position, so a NaN in any position is the result here, and the earliest of them is the one selected.
  for (int j = 0; j < argc; ++j) {
    if (std::isnan(val[idx[j]])) return j;
  }
  return k;
}

double FieldExpr::value_at(const double* v, double* val) const {
  const size_t n = nodes_.size();
  for (size_t i = 0; i < n; ++i) {
    const Node& nd = nodes_[i];
    switch (nd.op) {
      case Op::kConst:
        val[i] = nd.value;
        break;
      case Op::kVar:
        val[i] = v[nd.var];
        break;
      case Op::kMin:
      case Op::kMax: {
        const int* idx = args_.data() + nd.args;
        val[i] = val[idx[selected(nd.op, val, idx, nd.argc)]];
        break;
      }
      default:
        val[i] = apply(nd.op, val[nd.a], (nd.b >= 0) ? val[nd.b] : 0.0);
        break;
    }
  }
  return val[static_cast<size_t>(root_)];
}

void FieldExpr::compact() {
  // Folding a constant subtree leaves its nodes behind; keep only what the root reaches.  A node's children
  // have smaller indices than it has, so one backward pass marks everything reachable, and one forward pass
  // renumbers it with every child kept and renumbered before its parent.
  const size_t n = nodes_.size();
  std::vector<char> live(n, 0);
  live[static_cast<size_t>(root_)] = 1;
  for (size_t i = n; i-- > 0;) {
    if (live[i] == 0) continue;
    const Node& nd = nodes_[i];
    if (nd.a >= 0) live[static_cast<size_t>(nd.a)] = 1;
    if (nd.b >= 0) live[static_cast<size_t>(nd.b)] = 1;
    for (int j = 0; j < nd.argc; ++j) live[static_cast<size_t>(args_[static_cast<size_t>(nd.args + j)])] = 1;
  }
  std::vector<Node> kept;
  std::vector<int> kept_args;
  std::vector<int> at(n, -1);
  for (size_t i = 0; i < n; ++i) {
    if (live[i] == 0) continue;
    Node nd = nodes_[i];
    if (nd.a >= 0) nd.a = at[static_cast<size_t>(nd.a)];
    if (nd.b >= 0) nd.b = at[static_cast<size_t>(nd.b)];
    if (nd.argc > 0) {
      const int first = static_cast<int>(kept_args.size());
      for (int j = 0; j < nd.argc; ++j) {
        kept_args.push_back(at[static_cast<size_t>(args_[static_cast<size_t>(nd.args + j)])]);
      }
      nd.args = first;
    }
    kept.push_back(nd);
    at[i] = static_cast<int>(kept.size()) - 1;
  }
  root_ = at[static_cast<size_t>(root_)];
  nodes_ = std::move(kept);
  args_ = std::move(kept_args);
}

FieldExpr FieldExpr::compile(const std::string& text, const ParamMap& params,
                             const std::vector<std::string>& variables, const std::string& where,
                             const std::string& context) {
  // A variable has one position in the list: a name that is listed twice has none of its own, and a point
  // could not say which value it carries.
  for (size_t k = 0; k < variables.size(); ++k) {
    for (size_t j = 0; j < k; ++j) {
      if (variables[j] == variables[k]) {
        throw Error(strcat("variable '", variables[k], "' is listed twice"), where,
                    strcat("Each variable of an expression has one position in the list; name '",
                           variables[k], "' once. The list is: ", name_list(variables), "."));
      }
    }
  }
  // A parameter that has the name of a variable is refused where the expression is compiled, not shadowed in
  // silence: the text would mean two things, and which one it means may not be guessed.  It is refused
  // whether or not the text uses the name, because the clash is in the context, not in the text.
  for (const std::string& name : variables) {
    if (params.count(name) != 0) {
      std::string what = "a variable of this expression";
      if (!context.empty()) {
        // the context names what the variable is, "{name}" standing for the clashing name
        std::string phrase = context;
        for (size_t at = phrase.find("{name}"); at != std::string::npos; at = phrase.find("{name}", at + name.size())) {
          phrase.replace(at, 6, name);
        }
        what = phrase;
      }
      nlohmann::json details;
      details["parameter"] = name;
      details["variables"] = variables;
      throw Error(context.empty() ? strcat("parameter '", name, "' has the name of ", what)
                                  : strcat("parameter '", name, "' has the name of ", what, "; rename the parameter"),
                  where,
                  strcat("Rename the parameter; '", name, "' is a variable here (the variables are ",
                         name_list(variables), ") and the expression cannot mean both."),
                  model::codes::expr_parameter_shadows_variable, details.dump());
    }
  }

  FieldExpr f;
  f.text_ = text;
  f.where_ = where;
  f.vars_ = variables;
  FieldExprCompiler c(f.text_, params, f.vars_, f.where_, f);
  f.root_ = c.expr();
  c.ws();
  if (c.i != f.text_.size()) c.fail("trailing characters");
  for (size_t k = 0; k < c.used.size(); ++k) {
    if (c.used[k] != 0) f.used_.push_back(f.vars_[k]);
  }
  f.compact();
  // The design evaluator's last check is on the value of the whole expression.  The whole expression has a
  // value here exactly when no variable of the context occurs in it, in which case it has folded into one
  // constant; where a variable occurs, whether the value is finite is a property of the point and the
  // caller's business.
  if (f.used_.empty() && f.nodes_[static_cast<size_t>(f.root_)].op == Op::kConst &&
      !std::isfinite(f.nodes_[static_cast<size_t>(f.root_)].value)) {
    throw Error(strcat("expression '", f.text_, "' evaluated to a non-finite value"), where,
                "Check for division by zero or log of a non-positive number.");
  }
  return f;
}

double FieldExpr::eval(const double* v) const {
  Scratch val(nodes_.size());
  return value_at(v, val.p);
}

void FieldExpr::eval(const double* v, int n, double* out) const {
  if (n <= 0) return;
  Scratch val(nodes_.size());
  const size_t nvar = vars_.size();
  for (int i = 0; i < n; ++i) {
    const double* at = (v == nullptr) ? nullptr : v + static_cast<size_t>(i) * nvar;
    out[i] = value_at(at, val.p);
  }
}

double FieldExpr::eval_d(const double* v, int k) const {
  if (k < 0 || static_cast<size_t>(k) >= vars_.size()) {
    throw Error(strcat("the variable index ", k, " is outside the ", vars_.size(),
                       " variable(s) of expression '", text_, "'"),
                where_,
                strcat("eval_d(v, k) takes k in [0, ", vars_.size(),
                       "), the positions of the variables of this expression: ", name_list(vars_), "."));
  }
  Scratch val(nodes_.size());
  Scratch der(nodes_.size());
  double* value = val.p;
  double* d = der.p;
  const size_t n = nodes_.size();
  for (size_t i = 0; i < n; ++i) {
    const Node& nd = nodes_[i];
    switch (nd.op) {
      case Op::kConst:
        value[i] = nd.value;
        d[i] = 0.0;
        break;
      case Op::kVar:
        value[i] = v[nd.var];
        d[i] = (nd.var == k) ? 1.0 : 0.0;
        break;
      case Op::kMin:
      case Op::kMax: {
        // the derivative of the argument the fold selected, as the value is that argument's value
        const int* idx = args_.data() + nd.args;
        const int at = idx[selected(nd.op, value, idx, nd.argc)];
        value[i] = value[at];
        d[i] = d[at];
        break;
      }
      default: {
        const double a = value[nd.a];
        const double da = d[nd.a];
        const double b = (nd.b >= 0) ? value[nd.b] : 0.0;
        const double db = (nd.b >= 0) ? d[nd.b] : 0.0;
        value[i] = apply(nd.op, a, b);
        d[i] = derivative(nd.op, a, da, b, db, value[i]);
        break;
      }
    }
  }
  return d[static_cast<size_t>(root_)];
}

}  // namespace axmp
