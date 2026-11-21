// SPDX-License-Identifier: BSD-3-Clause
#include "fields.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "G4SystemOfUnits.hh"
#include "G4VSolid.hh"

#include "npy.hpp"
#include "runtime.hpp"

namespace axtr {

using nlohmann::json;
namespace fs = std::filesystem;

FieldCounters& field_counters() {
  static thread_local FieldCounters c;
  return c;
}

// ============================================================================================ map patches
void MapPatch::interpolate(const double* q, double out[3]) const {
  // the multilinear interpolation of axionmultiphysics.fieldmap (_interp), operation for operation: a coordinate
  // within 1e-9 of a spacing of the first or last node plane is taken on that plane, the cell is clipped to the grid
  std::size_t idx[3] = {0, 0, 0};
  double frac[3] = {0, 0, 0};
  for (int a = 0; a < dims; ++a) {
    double t = (q[a] - origin[a]) / spacing[a];
    const double last = static_cast<double>(shape[a] - 1);
    t = (t >= last - 1e-9) ? last : ((t <= 1e-9) ? 0.0 : t);
    long long i = static_cast<long long>(std::floor(t));
    i = std::min<long long>(std::max<long long>(i, 0), static_cast<long long>(shape[a]) - 2);
    idx[a] = static_cast<std::size_t>(i);
    frac[a] = t - static_cast<double>(i);
  }
  out[0] = out[1] = out[2] = 0.0;
  for (int corner = 0; corner < (1 << dims); ++corner) {
    double w = 1.0;
    std::size_t flat = 0;
    for (int a = 0; a < dims; ++a) {
      const int bit = (corner >> a) & 1;
      w = w * (bit ? frac[a] : 1.0 - frac[a]);
      flat = flat * shape[a] + idx[a] + static_cast<std::size_t>(bit);
    }
    const double* b = &B[3 * flat];
    out[0] += w * b[0];
    out[1] += w * b[1];
    out[2] += w * b[2];
  }
}

bool MapPatch::native(const double* q, double out[3]) const {
  bool ok = true;
  double c[3] = {0, 0, 0};
  for (int a = 0; a < dims; ++a) {
    const double lo = origin[a];
    const double hi = origin[a] + static_cast<double>(shape[a] - 1) * spacing[a];
    const double beyond = std::max(std::max(lo - q[a], q[a] - hi), 0.0);
    ok = ok && beyond <= margin[a] * (1 + 1e-12);
    c[a] = std::min(std::max(q[a], lo), hi);
  }
  interpolate(c, out);
  return ok;
}

bool MapPatch::cartesian(const double p[3], double out[3]) const {
  if (kind == 1) {
    const double r = std::hypot(p[0], p[1]);
    const double phi = std::atan2(p[1], p[0]);
    const double q[2] = {r, p[2]};
    double b[3];
    const bool ok = native(q, b);  // (B_r, B_z, B_phi)
    const double c = std::cos(phi), s = std::sin(phi);
    out[0] = b[0] * c - b[2] * s;
    out[1] = b[0] * s + b[2] * c;
    out[2] = b[1];
    return ok;
  }
  double d[3] = {p[0] - t[0], p[1] - t[1], p[2] - t[2]};
  double local[3];
  for (int j = 0; j < 3; ++j) local[j] = d[0] * R[0][j] + d[1] * R[1][j] + d[2] * R[2][j];
  double b[3];
  const bool ok = native(local, b);
  for (int i = 0; i < 3; ++i) out[i] = b[0] * R[i][0] + b[1] * R[i][1] + b[2] * R[i][2];
  return ok;
}

std::shared_ptr<const MapPatch> load_patch(const fs::path& dir, const std::string& patch) {
  const std::string where = "the field map " + dir.string() + ", patch '" + patch + "'";
  auto fail = [&](const std::string& what) { throw std::runtime_error(where + ": " + what); };
  std::ifstream in(dir / "manifest.json");
  if (!in) fail("manifest.json cannot be read");
  json man;
  try {
    man = json::parse(in);
  } catch (const json::exception& e) {
    fail(std::string("manifest.json is not JSON: ") + e.what());
  }
  if (!man.is_object() || man.value("schema", "") != "axionmultiphysics.fieldmap/1")
    fail("the manifest's schema is not axionmultiphysics.fieldmap/1");
  const json* rec = nullptr;
  if (man.contains("patches") && man["patches"].is_array())
    for (const json& p : man["patches"])
      if (p.is_object() && p.value("name", "") == patch) rec = &p;
  if (!rec) fail("no such patch in the manifest");
  auto out = std::make_shared<MapPatch>();
  out->map = dir.string();
  out->name = patch;
  try {
    out->kind = (*rec).at("kind").get<int>();
    if (out->kind != 1 && out->kind != 3) fail("kind is neither 1 nor 3");
    out->dims = out->kind == 1 ? 2 : 3;
    const json& o = rec->at("origin");
    const json& h = rec->at("spacing");
    const json& n = rec->at("shape");
    const json& m = rec->contains("margin") ? rec->at("margin") : h;
    if (!o.is_array() || !h.is_array() || !n.is_array() || !m.is_array() || o.size() != std::size_t(out->dims) ||
        h.size() != o.size() || n.size() != o.size() || m.size() != o.size())
      fail("origin, spacing, shape and margin need one value per axis");
    for (int a = 0; a < out->dims; ++a) {
      out->origin[a] = o[a].get<double>();
      out->spacing[a] = h[a].get<double>();
      out->margin[a] = m[a].get<double>();
      out->shape[a] = n[a].get<std::size_t>();
      if (!(out->spacing[a] > 0.0) || out->shape[a] < 2) fail("a spacing is not positive or an axis has < 2 nodes");
    }
    if (out->kind == 3 && rec->contains("frame") && rec->at("frame").contains("transform")) {
      const json& T = rec->at("frame").at("transform");
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) out->R[i][j] = T.at(i).at(j).get<double>();
        out->t[i] = T.at(i).at(3).get<double>();
      }
    }
    NpyDoubles B = read_npy_doubles(dir / rec->at("file").get<std::string>());
    if (rec->contains("sha256") && B.sha256 != rec->at("sha256").get<std::string>())
      fail("its array does not match the manifest's SHA-256");
    std::vector<std::size_t> want(out->shape, out->shape + out->dims);
    want.push_back(3);
    if (B.shape != want) fail("its array's shape does not match the manifest's");
    out->B = std::move(B.data);
  } catch (const json::exception& e) {
    fail(std::string("the patch record is malformed: ") + e.what());
  }
  return out;
}

// ============================================================================================ the fields
void loop_field(double r, double z, double a, double I, double z0, double& Br, double& Bz) {
  constexpr double mu0 = 4e-7 * 3.14159265358979323846;  // the SI value of the package's closed forms
  const double zz = z - z0;
  const double ap = (a + r) * (a + r) + zz * zz;
  const double am = (a - r) * (a - r) + zz * zz;
  const double m = 4.0 * a * r / ap;  // the parameter k^2
  const double k = std::sqrt(m);
  const double K = std::comp_ellint_1(k), E = std::comp_ellint_2(k);
  const double c = mu0 * I / (2.0 * 3.14159265358979323846);
  Bz = c / std::sqrt(ap) * (K + (a * a - r * r - zz * zz) / am * E);
  Br = r > 1e-12 ? c / r * zz / std::sqrt(ap) * (-K + (a * a + r * r + zz * zz) / am * E) : 0.0;
}

void CountingUniformField::GetFieldValue(const G4double point[4], G4double* field) const {
  ++field_counters().calls;
  G4UniformMagField::GetFieldValue(point, field);
}

VolumeField::VolumeField(const FieldSpec& spec, std::shared_ptr<const MapPatch> patch, const G4VSolid* solid,
                         const G4ThreeVector& offset, Outside outside, std::string volume)
    : spec_(spec), patch_(std::move(patch)), solid_(solid), offset_(offset), outside_(outside),
      volume_(std::move(volume)) {}

void VolumeField::value(const double p[3], double B_T[3], bool inside) const {
  if (spec_.type == "current_loop") {
    const double r = std::hypot(p[0], p[1]);
    double Br, Bz;
    loop_field(r, p[2], spec_.radius_m, spec_.current_A, spec_.z_m, Br, Bz);
    B_T[0] = r > 0.0 ? Br * p[0] / r : 0.0;
    B_T[1] = r > 0.0 ? Br * p[1] / r : 0.0;
    B_T[2] = Bz;
    return;
  }
  if (spec_.type == "uniform") {
    for (int k = 0; k < 3; ++k) B_T[k] = spec_.B_T[k];
    return;
  }
  if (patch_->cartesian(p, B_T)) return;
  // beyond the box plus margin: a trial point outside the volume keeps the value of the box's nearest point (it does
  // not enter the trajectory: the boundary is located from evaluations within deltaIntersection of it)
  if (!inside) {
    const G4ThreeVector local = G4ThreeVector(p[0] * CLHEP::m, p[1] * CLHEP::m, p[2] * CLHEP::m) - offset_;
    inside = solid_ == nullptr || solid_->Inside(local) != kOutside;
  }
  if (!inside) return;
  if (outside_ == Outside::zero) {
    ++field_counters().outside;
    B_T[0] = B_T[1] = B_T[2] = 0.0;
    return;
  }
  std::cerr << "axmp-transport: error: fieldmap.outside_map: the point (" << p[0] << ", " << p[1] << ", " << p[2]
            << ") m of volume " << volume_ << " lies outside the box of patch '" << patch_->name << "' of the map "
            << patch_->map << " by more than its margin (outside: refuse)" << std::endl;
  std::_Exit(kRunFailed);
}

void VolumeField::GetFieldValue(const G4double point[4], G4double* field) const {
  ++field_counters().calls;
  const double p[3] = {point[0] / CLHEP::m, point[1] / CLHEP::m, point[2] / CLHEP::m};
  double B[3];
  value(p, B, false);
  field[0] = B[0] * CLHEP::tesla;
  field[1] = B[1] * CLHEP::tesla;
  field[2] = B[2] * CLHEP::tesla;
}

}  // namespace axtr
