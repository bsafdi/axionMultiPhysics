// Python bindings of the field-map numerics (axionmultiphysics/fieldmap.hpp), used by axionmultiphysics.fieldmap:
// _core.FieldMapper (one per solved model), _core.FieldMapPatch, _core.FieldMapChords and _core.fieldmap_halton.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "axionmultiphysics/core.hpp"
#include "axionmultiphysics/fieldmap.hpp"
#include "axionmultiphysics/model.hpp"
#include "axionmultiphysics/model3d.hpp"

namespace nb = nanobind;
namespace fm = axmp::fieldmap;

namespace {

using DArr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T> np_array(std::vector<T> v, std::vector<size_t> shape) {
  auto* data = new std::vector<T>(std::move(v));
  nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T>(data->data(), shape.size(), shape.data(), owner);
}

fm::Grid grid(int dim, const std::vector<double>& origin, const std::vector<double>& spacing,
              const std::vector<long>& shape) {
  AXMP_REQUIRE(static_cast<int>(origin.size()) == dim && static_cast<int>(spacing.size()) == dim &&
                   static_cast<int>(shape.size()) == dim,
               "internal: a field-map grid needs origin, spacing and shape per axis");
  fm::Grid g;
  g.dim = dim;
  for (int a = 0; a < dim; ++a) {
    AXMP_REQUIRE(shape[a] >= 2 && spacing[a] > 0, "internal: a field-map grid needs >= 2 nodes and a positive spacing");
    g.origin[a] = origin[a];
    g.spacing[a] = spacing[a];
    g.n[a] = shape[a];
  }
  return g;
}

const double* values(const DArr& B, const fm::Grid& g) {
  AXMP_REQUIRE(static_cast<long>(B.size()) == g.nodes() * 3, "internal: nodal values do not match the grid");
  return B.data();
}

std::vector<size_t> grid_shape(const fm::Grid& g) {
  std::vector<size_t> s;
  for (int a = 0; a < g.dim; ++a) s.push_back(static_cast<size_t>(g.n[a]));
  s.push_back(3);
  return s;
}

// a mapper keeps its model alive (the Python object is held by keep_alive)
struct PyMapper {
  std::unique_ptr<fm::Mapper> m;
};
struct PyPatch {
  std::unique_ptr<fm::Patch> p;
};

}  // namespace

void axmp_bind_fieldmap(nb::module_& m) {
  m.def(
      "fieldmap_halton",
      [](uint64_t seed, long n, int dims) {
        AXMP_REQUIRE(n >= 0, "n must be >= 0");
        return np_array(fm::halton(seed, n, dims), {static_cast<size_t>(n), static_cast<size_t>(dims)});
      },
      nb::arg("seed"), nb::arg("n"), nb::arg("dims") = 6,
      "Points 1..n of the field-map format's scrambled-Halton sequence, shape (n, dims).");

  nb::class_<fm::Chords>(m, "FieldMapChords")
      .def_prop_ro("family", [](const fm::Chords& c) { return c.family; })
      .def_prop_ro("attempts", [](const fm::Chords& c) { return c.attempts; })
      .def_prop_ro("n", [](const fm::Chords& c) { return c.size(); })
      .def_prop_ro(
          "start",
          [](const fm::Chords& c) {
            return np_array(c.start, {static_cast<size_t>(c.size()), static_cast<size_t>(c.dim)});
          },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "direction",
          [](const fm::Chords& c) {
            return np_array(c.direction, {static_cast<size_t>(c.size()), static_cast<size_t>(c.dim)});
          },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "segments", [](const fm::Chords& c) { return np_array(c.segments, {c.segment_chord.size(), size_t(2)}); },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "segment_chord", [](const fm::Chords& c) { return np_array(c.segment_chord, {c.segment_chord.size()}); },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "length", [](const fm::Chords& c) { return np_array(c.length, {c.length.size()}); }, nb::rv_policy::automatic)
      .def_prop_ro(
          "I_star", [](const fm::Chords& c) { return np_array(c.I_star, {c.length.size(), size_t(3)}); },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "I_abs", [](const fm::Chords& c) { return np_array(c.I_abs, {c.length.size()}); }, nb::rv_policy::automatic)
      .def_prop_ro(
          "I_h", [](const fm::Chords& c) { return np_array(c.I_h, {c.length.size(), size_t(3)}); },
          nb::rv_policy::automatic)
      .def(
          "map_integrals",
          [](const fm::Chords& c, std::vector<double> origin, std::vector<double> spacing, std::vector<long> shape,
             DArr B) {
            const fm::Grid g = grid(c.dim, origin, spacing, shape);
            const double* b = values(B, g);
            std::vector<double> out;
            {
              nb::gil_scoped_release release;
              out = c.map_integrals(g, b);
            }
            return np_array(std::move(out), {c.length.size(), size_t(3)});
          },
          nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"), nb::arg("B"))
      .def(
          "statistic",
          [](const fm::Chords& c, DArr I_map, double B_patch) {
            AXMP_REQUIRE(static_cast<long>(I_map.size()) == c.size() * 3, "internal: I_map must be (n, 3)");
            std::vector<double> im(I_map.data(), I_map.data() + I_map.size());
            const fm::Statistic s = fm::statistic(c, im, B_patch);
            nb::dict d;
            d["n"] = s.n;
            d["E"] = s.E;
            d["E_q95"] = s.E_q95;
            d["E_max"] = s.E_max;
            d["H"] = s.H;
            return d;
          },
          nb::arg("I_map"), nb::arg("B_patch"));

  nb::class_<PyPatch>(m, "FieldMapPatch")
      .def("info", [](const PyPatch& p) { return p.p->info().dump(); })
      .def_prop_ro("B_patch", [](const PyPatch& p) { return p.p->B_patch; })
      .def_prop_ro("h_el", [](const PyPatch& p) { return p.p->h_el; })
      .def(
          "recovered",
          [](const PyPatch& p, DArr pts) {
            const int dim = p.p->M.dim();
            AXMP_REQUIRE(pts.ndim() == 2 && static_cast<int>(pts.shape(1)) == dim,
                         axmp::strcat("points must have shape (n, ", dim, ")"));
            std::vector<double> x(pts.data(), pts.data() + pts.size());
            std::vector<double> out;
            {
              nb::gil_scoped_release release;
              out = p.p->recovered(x);
            }
            return np_array(std::move(out), {pts.shape(0), size_t(3)});
          },
          nb::arg("points"))
      .def(
          "fill",
          [](const PyPatch& p, std::vector<double> origin, std::vector<double> spacing, std::vector<long> shape) {
            const fm::Grid g = grid(p.p->M.dim(), origin, spacing, shape);
            std::vector<double> out;
            {
              nb::gil_scoped_release release;
              out = p.p->fill(g);
            }
            return np_array(std::move(out), grid_shape(g));
          },
          nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"))
      .def(
          "chords",
          [](const PyPatch& p, const std::string& family, long n, uint64_t seed, std::vector<double> box) {
            nb::gil_scoped_release release;
            return p.p->chords(family, n, seed, box);
          },
          nb::arg("family"), nb::arg("n"), nb::arg("seed"), nb::arg("box") = std::vector<double>())
      .def("box_meets", [](const PyPatch& p, std::vector<double> box) { return p.p->box_meets(box); }, nb::arg("box"))
      .def(
          "pointwise",
          [](const PyPatch& p, std::vector<double> origin, std::vector<double> spacing, std::vector<long> shape, DArr B,
             long n, uint64_t seed) {
            const fm::Grid g = grid(p.p->M.dim(), origin, spacing, shape);
            const double* b = values(B, g);
            nb::gil_scoped_release release;
            return p.p->pointwise(g, b, n, seed).dump();
          },
          nb::arg("origin"), nb::arg("spacing"), nb::arg("shape"), nb::arg("B"), nb::arg("n"), nb::arg("seed"));

  nb::class_<PyMapper>(m, "FieldMapper")
      .def(
          "__init__",
          [](PyMapper* self, const axmp::AxiModel& model) {
            new (self) PyMapper{std::make_unique<fm::Mapper>(fm::axi_source(model))};
          },
          nb::arg("model"), nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](PyMapper* self, const axmp::Model3D& model) {
            new (self) PyMapper{std::make_unique<fm::Mapper>(model.field_source())};
          },
          nb::arg("model"), nb::keep_alive<1, 2>())
      .def_prop_ro("dim", [](const PyMapper& M) { return M.m->dim(); })
      .def_prop_ro("order", [](const PyMapper& M) { return M.m->source().order; })
      .def_prop_ro("elements", [](const PyMapper& M) { return M.m->source().ne; })
      .def_prop_ro("dofs", [](const PyMapper& M) { return M.m->source().solution_dofs; })
      .def("regions", [](const PyMapper& M) { return M.m->regions().dump(); })
      .def(
          "decompose",
          [](const PyMapper& M, std::vector<std::string> regions, bool merge) {
            return M.m->decompose(regions, merge).dump();
          },
          nb::arg("regions"), nb::arg("merge"))
      .def(
          "patch",
          [](const PyMapper& M, std::vector<std::string> regions, int component) {
            std::unique_ptr<fm::Patch> p;
            {
              nb::gil_scoped_release release;
              p = M.m->patch(regions, component);
            }
            return PyPatch{std::move(p)};
          },
          nb::arg("regions"), nb::arg("component"), nb::keep_alive<0, 1>())
      .def("solution", [](const PyMapper& M) {
        std::vector<double> v = M.m->source().solution();
        const size_t n = v.size();
        return np_array(std::move(v), {n});
      })
      .def(
          "boundary_max",
          [](const PyMapper& M, std::vector<const PyPatch*> patches) {
            std::vector<const fm::Patch*> ps;
            for (auto* p : patches) ps.push_back(p->p.get());
            return M.m->boundary_max(ps);
          },
          nb::arg("patches"))
      .def(
          "interface",
          [](const PyMapper& M, const PyPatch& a, std::vector<double> oa, std::vector<double> sa, std::vector<long> na,
             DArr Ba, const PyPatch& b, std::vector<double> ob, std::vector<double> sb, std::vector<long> nb_, DArr Bb) {
            const fm::Grid ga = grid(M.m->dim(), oa, sa, na), gb = grid(M.m->dim(), ob, sb, nb_);
            return M.m->interface(*a.p, ga, values(Ba, ga), *b.p, gb, values(Bb, gb)).dump();
          });
}
