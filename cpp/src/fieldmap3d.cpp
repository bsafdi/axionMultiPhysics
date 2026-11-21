// The static field of a solved 3D model, element by element, for field maps (axionmultiphysics/fieldmap.hpp).
// MPI builds only (it reads the 3D engine's state); the sequential build's Model3D::field_source is the stub in
// model3d_nompi.cpp.
//
// B_h = curl A_h of the Nedelec solution plus the background. MFEM evaluates curls through element scratch space that
// threads cannot share, so the reference curl c(xi) of every element (a polynomial of degree order - 1, with
// B = J c / det J by the covariant Piola map) is tabulated once at the nodes of a Lagrange basis of degree `order`;
// any later evaluation interpolates it exactly and is thread-safe.
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "axionmultiphysics/fieldmap.hpp"
#include "model3d_impl.hpp"

namespace axmp {

namespace {

class Source3D : public fieldmap::Source {
 public:
  Source3D(const Model3D::Impl& P, const spec::Design& d) : P_(P) {
    mfem::Mesh& mesh = P.vmesh();  // the serial mesh (the view of a partitioned solve)
    const mfem::GridFunction* A = P.vA_lab();
    AXMP_REQUIRE(A, "solve the design before exporting a field map");
    dim = 3;
    order = P.order;
    ne = mesh.GetNE();
    attr.resize(ne);
    for (int e = 0; e < ne; ++e) attr[e] = mesh.GetAttribute(e);
    region_of_attr = P.region_of_attr;
    for (auto& [a, name] : region_of_attr) {
      fieldmap::RegionLaw L;
      if (a == P.sh_attr || name == "infinity_shell") {
        L.kind = fieldmap::RegionLaw::Excluded;  // the exterior mapped onto a shell: not a physical region
      } else if (d.find_region(name)) {
        const spec::Material& mat = d.material_of_region(name);
        switch (mat.kind) {
          case spec::MaterialKind::Linear: L.mu_r = mat.mu_r; break;
          case spec::MaterialKind::Iron: L.kind = fieldmap::RegionLaw::Nonlinear; L.material = mat.name; break;
          case spec::MaterialKind::Superconductor: L.kind = fieldmap::RegionLaw::Superconductor; break;
          default: break;
        }
      }
      law[name] = L;
    }
    // geometry
    const mfem::GridFunction* nodes = mesh.GetNodes();
    static mfem::Linear3DFiniteElement p1;
    const mfem::FiniteElement* gfe = nodes ? nodes->FESpace()->GetFE(0) : &p1;
    ng = gfe->GetDof();
    const mfem::IntegrationRule& gn = gfe->GetNodes();
    for (int k = 0; k < ng; ++k) gref.insert(gref.end(), {gn.IntPoint(k).x, gn.IntPoint(k).y, gn.IntPoint(k).z});
    gx.resize(static_cast<size_t>(ne) * ng * 3);
    mfem::Array<int> vd;
    for (int e = 0; e < ne; ++e) {
      AXMP_REQUIRE(mesh.GetElementBaseGeometry(e) == mfem::Geometry::TETRAHEDRON,
                   "field maps of 3D designs need a tetrahedral mesh");
      if (nodes) {
        nodes->FESpace()->GetElementVDofs(e, vd);
        for (int k = 0; k < ng; ++k)
          for (int a = 0; a < 3; ++a) gx[(static_cast<size_t>(e) * ng + k) * 3 + a] = (*nodes)(vd[a * ng + k]);
      } else {
        mesh.GetElementVertices(e, vd);
        for (int k = 0; k < 4; ++k)
          for (int a = 0; a < 3; ++a) gx[(static_cast<size_t>(e) * 4 + k) * 3 + a] = mesh.GetVertex(vd[k])[a];
      }
    }
    vol.resize(ne);
    for (int e = 0; e < ne; ++e) vol[e] = mesh.GetElementVolume(e);
    // topology: local face j is the face opposite vertex j
    nbr.assign(static_cast<size_t>(ne) * 4, -1);
    fattr.assign(static_cast<size_t>(ne) * 4, 0);
    std::vector<int> bdr_attr(mesh.GetNumFaces(), 0);
    for (int be = 0; be < mesh.GetNBE(); ++be) bdr_attr[mesh.GetBdrElementFaceIndex(be)] = mesh.GetBdrAttribute(be);
    mfem::Array<int> ev, faces, ori, fv;
    for (int e = 0; e < ne; ++e) {
      mesh.GetElementVertices(e, ev);
      mesh.GetElementFaces(e, faces, ori);
      for (int f : faces) {
        mesh.GetFaceVertices(f, fv);
        int j = -1;
        for (int k = 0; k < 4; ++k)
          if (ev[k] != fv[0] && ev[k] != fv[1] && ev[k] != fv[2]) j = k;
        int e1, e2;
        mesh.GetFaceElements(f, &e1, &e2);
        const int n = e1 == e ? e2 : e1;
        nbr[e * 4 + j] = n >= 0 && n < ne ? n : -1;
        fattr[e * 4 + j] = bdr_attr[f];
      }
    }
    // recovery space: continuous H1 of the solution's order
    fec_ = std::make_unique<mfem::H1_FECollection>(order, 3);
    fes_ = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec_.get());
    const mfem::FiniteElement* fe = fes_->GetFE(0);
    nd = fe->GetDof();
    const mfem::IntegrationRule& nn = fe->GetNodes();
    for (int k = 0; k < nd; ++k) nref.insert(nref.end(), {nn.IntPoint(k).x, nn.IntPoint(k).y, nn.IntPoint(k).z});
    basis_ = fieldmap::Lagrange(3, nref);
    ndofs = fes_->GetNDofs();
    dofs.resize(static_cast<size_t>(ne) * nd);
    dof_x.assign(static_cast<size_t>(ndofs) * 3, 0.0);
    // the reference curl at the basis nodes, and the coordinates of the recovery nodes
    curl_.assign(static_cast<size_t>(ne) * nd * 3, 0.0);
    fieldmap::Lagrange G(3, gref);
    mfem::Vector Bv(3);
    mfem::Array<int> ed;
    std::vector<double> Gs(ng), dG(3 * ng);
    for (int e = 0; e < ne; ++e) {
      fes_->GetElementDofs(e, ed);
      mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
      for (int k = 0; k < nd; ++k) {
        dofs[static_cast<size_t>(e) * nd + k] = ed[k];
        const mfem::IntegrationPoint& ip = nn.IntPoint(k);
        T->SetIntPoint(&ip);
        A->GetCurl(*T, Bv);
        const double xi[3] = {ip.x, ip.y, ip.z};
        G.eval(xi, Gs.data(), dG.data());
        double x[3] = {0, 0, 0}, J[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        const double* X = &gx[static_cast<size_t>(e) * ng * 3];
        for (int q = 0; q < ng; ++q)
          for (int a = 0; a < 3; ++a) {
            x[a] += Gs[q] * X[q * 3 + a];
            for (int b = 0; b < 3; ++b) J[a * 3 + b] += dG[q * 3 + b] * X[q * 3 + a];
          }
        for (int a = 0; a < 3; ++a) dof_x[static_cast<size_t>(ed[k]) * 3 + a] = x[a];
        // c = det(J) J^-1 B
        mfem::DenseMatrix Jm(3), Ji(3);
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) Jm(a, b) = J[a * 3 + b];
        const double det = Jm.Det();
        mfem::CalcInverse(Jm, Ji);
        for (int a = 0; a < 3; ++a) {
          double c = 0;
          for (int b = 0; b < 3; ++b) c += Ji(a, b) * Bv(b);
          curl_[(static_cast<size_t>(e) * nd + k) * 3 + a] = det * c;
        }
      }
    }
    solution_dofs = A->Size();
  }

  void Bh(int e, const double* xi, const double* x, const double* J, double B[3]) const override {
    double N[64];
    basis_.eval(xi, N);
    double c[3] = {0, 0, 0};
    const double* v = &curl_[static_cast<size_t>(e) * nd * 3];
    for (int i = 0; i < nd; ++i)
      for (int a = 0; a < 3; ++a) c[a] += N[i] * v[i * 3 + a];
    const double det = J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) +
                       J[2] * (J[3] * J[7] - J[4] * J[6]);
    for (int a = 0; a < 3; ++a) B[a] = (J[a * 3] * c[0] + J[a * 3 + 1] * c[1] + J[a * 3 + 2] * c[2]) / det;
    if (P_.bg) {
      mfem::Vector xp(3);
      for (int a = 0; a < 3; ++a) xp(a) = x[a];
      double bb[3];
      P_.bg_at(xp, bb);
      for (int a = 0; a < 3; ++a) B[a] += bb[a];
    }
  }

  std::vector<double> solution() const override {
    const mfem::GridFunction* A = P_.vA_lab();
    return std::vector<double>(A->GetData(), A->GetData() + A->Size());
  }

 private:
  const Model3D::Impl& P_;
  std::unique_ptr<mfem::H1_FECollection> fec_;
  std::unique_ptr<mfem::FiniteElementSpace> fes_;
  fieldmap::Lagrange basis_;
  std::vector<double> curl_;  // ne x nd x 3
};

}  // namespace

std::unique_ptr<fieldmap::Source> Model3D::field_source() const {
  AXMP_REQUIRE(p_->have_lab, "solve the design before exporting a field map");
  return std::make_unique<Source3D>(*p_, d_);
}

}  // namespace axmp
