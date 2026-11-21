// Compiled field expressions: an expression of the design grammar with free variables, evaluated at a point
// and differentiated exactly.
//
// `eval_expr` (axionmultiphysics/expr.hpp) evaluates a parameter expression once, when a document is read.
// Inlet profiles, body forces, initial fields, boundary data and `expression` property laws need a value at a
// point of the mesh instead, many times per solve, and their derivatives.  A FieldExpr parses the text once
// and keeps the tree:
//
//   const FieldExpr u = FieldExpr::compile("6*U*y*(H-y)/H^2", params, {"x", "y"}, "/physics/inlet/velocity/0");
//   const double value = u.eval(point);        // point[0] is x, point[1] is y
//   const double du_dy = u.eval_d(point, 1);   // exact d/dy, not a difference
//
// A VARIABLE is a free name of the context the expression is compiled in: a coordinate, the time of a
// transient step, or a field a property law reads.  `variables` is ordered, and that order is the layout of
// every point: v[k] is the value of variables[k], and eval_d's k indexes that same list, not the shorter list
// used_variables() returns.  A variable is resolved before a parameter and before the constants pi, e and
// mu0, so it hides them, while an identifier followed by '(' is a function call whatever the list holds.  A
// parameter that has the name of a variable is refused when the expression is compiled, because the text
// would otherwise mean two things.  Everything else is `eval_expr`'s -- the grammar, the functions, the
// constants, the strict number literals, the refusals and their reasons, and the cap of 1000 nesting levels
// -- and a text in which no variable occurs gives the same double as `eval_expr` gives, bit for bit.
//
// The value of a text in which a variable occurs is what IEEE arithmetic gives at the point, infinities and
// NaN included (1/x at x = 0 is +inf): whether a quadrature point is in the domain of the expression is the
// caller's business, not the expression's.  A text in which no variable occurs is a constant, and a constant
// that is not finite is refused when it is compiled, as `eval_expr` refuses it.
//
// DERIVATIVES are exact: forward mode over the compiled tree, one sweep per call, every function and operator
// of the grammar.  Three of them are a convention rather than a formula, and the convention is this one:
//   - abs: d|u|/dv = sign(u) u' with sign(0) = 0, so at the kink the derivative is zero and not one side's;
//   - min and max: the derivative of the argument the fold selected, which on a tie is the earliest of the
//     tied arguments, as it is for the value, so d min(x, 2x)/dx at x = 0 is 1 and not 2;
//   - a power: d(a^b) = b a^(b-1) a' + a^b log(a) b'.  The first term is evaluated only where a' and b are
//     both nonzero, the second only where b' and a are both nonzero; elsewhere the power does not move with
//     that operand and the term would be a zero times an infinity.  So d(x^2)/dx at x = -2 is -4 and not
//     log(-2) 0 = NaN, d(x^0)/dx at x = 0 is 0 and not 0 inf (x^0 is 1 for every x), and d(0^x)/dx at
//     x = 1 is 0 and not 0 log(0) (0^x is 0 for every x > 0).
// A subtree in which the variable does not occur contributes exactly +0, and so does a whole text without it.
// The divergence of a vector of expressions needs no call of its own: it is the sum of each component's
// eval_d with respect to its own coordinate.
//
// Evaluation is re-entrant.  One compiled expression may be evaluated from several threads at once, each on
// its own points, because every scratch buffer is a local of the call.  A compiled expression owns its text,
// the values of the parameters it used and the names of its variables, and may be copied, moved and kept for
// a whole solve.
#pragma once

#include <string>
#include <vector>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/expr.hpp"

namespace axmp {

class AXMP_EXPORT FieldExpr {
 public:
  // Parse `text` once.  `params` are substituted as constants, `variables` are the free names in the order in
  // which a point carries them, and `where` is the path (a JSON pointer into the document) that every refusal
  // names, here and at evaluation.
  //
  // A parameter that has the name of a variable is refused with code expr.parameter_shadows_variable and the details
  // {"parameter": the first clashing variable in the order of the list, "variables": the whole list}.  `context`, when
  // it is not empty, says what the variable is: the message is then "parameter 'y' has the name of <context>; rename
  // the parameter", with "{name}" in the context standing for the clashing name ("a coordinate of this planar model",
  // "the field '{name}' that this law reads").  Without it the message names "a variable of this expression".
  static FieldExpr compile(const std::string& text, const ParamMap& params,
                           const std::vector<std::string>& variables, const std::string& where,
                           const std::string& context = "");

  // The value at one point: v[k] is the value of variables[k].  v may be null exactly when
  // used_variables() is empty, here and in the two calls below: a point is read only where a variable of it
  // occurs in the text.
  double eval(const double* v) const;

  // The values at n points, v row-major (n rows of nvar), into out[0..n-1] and nowhere else; out[i] is bit for
  // bit what eval(v + i*nvar) gives.  n <= 0 writes nothing.  out and v must not overlap.  v may be null
  // exactly when used_variables() is empty, and the n values are then the same constant.
  void eval(const double* v, int n, double* out) const;

  // The exact partial derivative with respect to variables[k] at one point.  Refuses a k that is not a
  // position of the list rather than reading past the point.  v may be null exactly when used_variables()
  // is empty, and the derivative is then +0.
  double eval_d(const double* v, int k) const;

  // The variables that occur in the text, in the order of the list compile() was given.  The reference is
  // valid for the lifetime of this object and of every copy of it.
  const std::vector<std::string>& used_variables() const { return used_; }

 private:
  FieldExpr() = default;  // compile() is the only way to make one; a copy and a move are public

  // The compiled tree, in one array.  Every node is built after its children, so a node's children have
  // smaller indices than it has, and one sweep in index order evaluates the whole tree: no recursion, and a
  // deep tree costs no stack.
  enum class Op : unsigned char {
    kConst, kVar, kAdd, kSub, kMul, kDiv, kNeg, kPow, kSqrt, kExp, kLog, kLog10,
    kSin, kCos, kTan, kAsin, kAcos, kAtan, kAtan2, kAbs, kMin, kMax
  };
  struct Node {
    Op op = Op::kConst;
    int a = -1;          // the only child of a unary node, the left child of a binary one, else -1
    int b = -1;          // the right child of a binary node, else -1
    int args = -1;       // kMin and kMax: the first of argc argument indices in args_
    int argc = 0;        // kMin and kMax: how many arguments
    int var = -1;        // kVar: the position of the variable in vars_
    double value = 0.0;  // kConst: the value
  };

  // The value of one node from its children's values.  The sweep and the folding of a constant subtree share
  // it, so that a folded value is exactly the value the sweep would have given.
  static double apply(Op op, double a, double b);
  // The derivative of one node from its children's values and derivatives, and its own value.
  static double derivative(Op op, double a, double da, double b, double db, double value);
  // Which argument min or max selects: the fold of the design evaluator (the earliest of equal arguments),
  // except that a NaN argument wins whatever its position.
  static int selected(Op op, const double* val, const int* idx, int argc);
  // The value of every node at one point; returns the value of the whole expression.
  double value_at(const double* v, double* val) const;
  // Drop the nodes of the subtrees that were folded into a constant, keeping the order of the rest.
  void compact();

  std::vector<Node> nodes_;         // the tree; the root is the last node
  std::vector<int> args_;           // the arguments of every kMin and kMax node, back to back
  std::vector<std::string> vars_;   // the variables of the context, in order: v[k] is vars_[k]
  std::vector<std::string> used_;   // those of vars_ that occur in the text, in the order of vars_
  std::string text_;                // the text it was compiled from, for the messages
  std::string where_;               // the path the messages name
  int root_ = -1;

  friend struct FieldExprCompiler;  // the parser of field_expr.cpp, which fills the arrays above
};

}  // namespace axmp
