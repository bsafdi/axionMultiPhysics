// SPDX-License-Identifier: BSD-3-Clause
// The magnetic fields of the transport and probe modes: a uniform field, the closed-form field of a circular current
// loop, and the patches of a field map (directory format axionmultiphysics.fieldmap/1), evaluated exactly as the
// package's FieldMap.evaluate evaluates them. Each logical volume gets a field object of its own (the patches' grids are
// shared), so that the field a volume sees is the one of its own region.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "G4MagneticField.hh"
#include "G4ThreeVector.hh"
#include "G4UniformMagField.hh"

#include "job.hpp"

class G4VSolid;

namespace axtr {

// One patch of a map, read from its directory: a regular grid of nodal values (kind 1: (r, z) -> (B_r, B_z, B_phi);
// kind 3: (x, y, z) of the patch frame -> (B_x, B_y, B_z)), tesla.
struct MapPatch {
  std::string map;   // the map directory
  std::string name;
  int kind = 1;
  int dims = 2;
  double origin[3] = {0, 0, 0}, spacing[3] = {1, 1, 1}, margin[3] = {1, 1, 1};
  std::size_t shape[3] = {1, 1, 1};
  std::vector<double> B;  // C order, the components last
  double R[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};  // kind 3: global = R local + t
  double t[3] = {0, 0, 0};

  // B [T] at native coordinates, the point clamped onto the box (FieldMap's rule). Returns whether the point lies
  // within the box enlarged by the margin along every axis.
  bool native(const double* q, double out[3]) const;
  // B [T] at a Cartesian point [m] (kind 1: rotated by the point's azimuth; kind 3: through the patch frame).
  bool cartesian(const double p[3], double out[3]) const;

 private:
  void interpolate(const double* q, double out[3]) const;
};

// Reads the patch `patch` of the map at `dir`, checking the array against the manifest's SHA-256 and shape.
// Throws std::runtime_error naming the map and the patch.
std::shared_ptr<const MapPatch> load_patch(const std::filesystem::path& dir, const std::string& patch);

// The calls of the calling thread's fields since the last reset: every evaluation, and those outside a patch's box
// plus margin at a point inside the volume (B = 0 with outside "zero").
struct FieldCounters {
  std::uint64_t calls = 0;
  std::uint64_t outside = 0;
};
FieldCounters& field_counters();

// A field object of one logical volume. `solid` and `offset` (the volume's global translation, Geant4 units) decide
// whether a point outside a patch's box plus margin lies in the volume (the track has left the map) or not (a trial
// point of the stepper beyond the volume's boundary, which takes the value of the box's nearest point).
class VolumeField : public G4MagneticField {
 public:
  VolumeField(const FieldSpec& spec, std::shared_ptr<const MapPatch> patch, const G4VSolid* solid,
              const G4ThreeVector& offset, Outside outside, std::string volume);
  void GetFieldValue(const G4double point[4], G4double* field) const override;
  // B [T] at p [m]; `inside` says the point is known to lie in the volume (the probe mode).
  void value(const double p[3], double B_T[3], bool inside) const;

 private:
  FieldSpec spec_;
  std::shared_ptr<const MapPatch> patch_;
  const G4VSolid* solid_;
  G4ThreeVector offset_;
  Outside outside_;
  std::string volume_;
};

// Geant4's uniform field, counting its evaluations.
class CountingUniformField : public G4UniformMagField {
 public:
  using G4UniformMagField::G4UniformMagField;
  void GetFieldValue(const G4double point[4], G4double* field) const override;
};

// B_r, B_z [T] of a circular filament of radius a [m] in the plane z = z0 carrying I [A], counter-clockwise seen from
// +z: the closed form with complete elliptic integrals (Smythe, Static and Dynamic Electricity, 3rd ed., 7.10).
void loop_field(double r, double z, double a, double I, double z0, double& Br, double& Bz);

}  // namespace axtr
