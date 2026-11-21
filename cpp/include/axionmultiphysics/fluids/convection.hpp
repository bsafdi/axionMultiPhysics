// The convective term of steady incompressible Navier–Stokes: ONE element kernel for both geometries (planar;
// axisymmetric with swirl) and both convective forms, giving the residual, the Oseen (Picard) matrix and the Newton
// Jacobian.
//
//   standard  c(w; u, v) = rho ((w.grad) u, v)
//   skew      c(w; u, v) = rho ((w.grad) u, v) + (rho/2) ((div w) u, v)       (Temam; NO boundary term)
//
// Axisymmetric (r, z) = mesh (x, y), components in the internal order (u_r, u_z, u_theta), weight r:
//   r [(w.grad)u]_r = r (w_r d_r u_r + w_z d_z u_r) - w_t u_t     (centrifugal)
//   r [(w.grad)u]_t = r (w_r d_r u_t + w_z d_z u_t) + w_t u_r     (Coriolis)
//   r [(w.grad)u]_z = r (w_r d_r u_z + w_z d_z u_z),   r div w = r d_r w_r + w_r + r d_z w_z.
// Because v carries no derivative, c(w; u, v) = int t(w, u) . v with a pointwise "force density" t, trilinear in
// (w, u): the residual tests t(u, u), the Oseen matrix O(w) = t(w, phi) and the reaction part Q(u) = t(phi, u), so
// one routine (t) serves all three and the Newton Jacobian is O(u) + Q(u).
//
// Quadrature: the trilinear form is integrated exactly by default. With p the velocity order and q the geometric
// order of the element, the integrand is a polynomial on the reference element of
//   total degree          3p + q - 2 (planar), 3p + 2q - 2 (axisymmetric)   on a simplex (Pp on a Pq geometry),
//   degree per direction  3p + q - 1 (planar), 3p + 2q - 1 (axisymmetric)   on a tensor-product element (Qp on Qq):
// 3p - 1 and 3p on straight triangles, 3p and 3p + 1 on straight (bilinear) quadrilaterals, whose map is not affine.
// exact_degree() gives it in MFEM's terms (2p + OrderGrad, plus the degree of the weight r in axisymmetry).
#pragma once

#include <mfem.hpp>

namespace axmp::fluids {

enum class ConvectiveForm { Standard, Skew };
const char* to_string(ConvectiveForm f);

class ConvectionIntegrator : public mfem::NonlinearFormIntegrator {
 public:
  enum class GradMode {
    Newton,       // O(u) + Q(u): the exact Jacobian of the residual
    Oseen         // O(u): the Picard (Oseen) operator with the advecting velocity u
  };
  // vdim: 2 (planar) or 3 (axisymmetric with swirl); order <= 0: the exact degree of the element (exact_degree).
  // An order > 0 whose rule is not exact (rule_is_exact) under-integrates the form and is refused when an element is
  // assembled, unless
  // allow_under_integration is set (the unit tests set it to measure the kernel below the exact degree).
  ConvectionIntegrator(double rho, int vdim, bool axisymmetric, ConvectiveForm form, int order = 0,
                       bool allow_under_integration = false);

  void set_grad_mode(GradMode m) { mode_ = m; }
  GradMode grad_mode() const { return mode_; }
  // the exact degree of the form on this element (the header comment gives it in p and q): 2p + OrderGrad(el), plus
  // the degree of the weight r in axisymmetry (OrderJ + 1 on a simplex, OrderJ on a tensor-product element)
  static int exact_degree(const mfem::FiniteElement& el, mfem::ElementTransformation& T, bool axisymmetric);
  // whether the rule MFEM returns for `order` on this element integrates the form exactly. It is the rule's own degree
  // that counts: an even order on a tensor-product element selects the Gauss rule of the next odd degree.
  static bool rule_is_exact(const mfem::FiniteElement& el, mfem::ElementTransformation& T, bool axisymmetric, int order);
  // NavierStokesSolver refuses a rule that is not exact collectively, before any assembly, with a message that names
  // its option; the per-element refusal here is the second check, for an integrator built directly.

  void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                             const mfem::Vector& elfun, mfem::Vector& elvect) override;
  void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& T,
                           const mfem::Vector& elfun, mfem::DenseMatrix& elmat) override;

 private:
  double rho_;
  int vdim_;
  bool axi_;
  ConvectiveForm form_;
  int order_;
  bool allow_under_ = false;
  GradMode mode_ = GradMode::Newton;
  int rule_degree(const mfem::FiniteElement& el, mfem::ElementTransformation& T) const;
};

}  // namespace axmp::fluids
