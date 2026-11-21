// SPDX-License-Identifier: BSD-3-Clause
// The solids list (schema axionmultiphysics.solids/1, written by axionmultiphysics.solids) in Geant4: the strict
// reading of the file, and the builder of its materials, solids and placements. The builder is one unit so that every
// mode that needs the design's geometry instantiates it the same way: the geometry mode checks it, a transport mode
// runs in it (call build_geometry from G4VUserDetectorConstruction::Construct).
//
// The construction mirrors what Geant4's GDML reader does with the file that axionmultiphysics.solids.write_gdml
// writes, so that both give the same doubles: lengths multiplied by CLHEP::m (a tube's full length by 0.5 * m), angles
// by CLHEP::deg; a material of the document is a mixture under its own name and density with the defaults of the GDML
// reader (undefined state, NTP temperature, STP pressure), its components in the file's order, a NIST composition one
// component (the NIST material, fraction 1).
#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class G4LogicalVolume;
class G4Material;
class G4VPhysicalVolume;
class G4VSolid;

namespace axtr {

struct GeometryError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline constexpr const char* kSolidsSchema = "axionmultiphysics.solids/1";

struct MaterialSpec {
  std::string name;
  std::string gdml_name;                                       // the name of the G4Material (and of the GDML file)
  double density_kg_m3 = 0.0;
  std::string nist;                                            // a NIST composition (G4_...), or empty
  std::vector<std::pair<std::string, double>> mass_fractions;  // by mass: (element symbol, fraction)
  std::vector<std::pair<std::string, int>> atoms;              // by atoms: (element symbol, count)
  std::optional<double> mean_excitation_eV, T_K, pressure_Pa;
  std::string state;                                           // solid, liquid, gas, or empty
};

struct SolidSpec {
  std::string name;
  std::string gdml_name;  // the name of the logical volume (and of the GDML file's volume)
  std::string kind;     // tube, torus, orb, genericPolycone, tessellated (3D lists only)
  std::string mother;   // empty for the world
  // tube: rmin, rmax, z (full length), startphi, deltaphi; torus: rmin, rmax, rtor, startphi, deltaphi; orb: r;
  // genericPolycone: startphi, deltaphi (and rz); tessellated: the vertices and the triangles (indices into them,
  // counter-clockwise seen from outside). Lengths in m, angles in deg.
  double rmin_m = 0.0, rmax_m = 0.0, z_m = 0.0, rtor_m = 0.0, r_m = 0.0;
  double startphi_deg = 0.0, deltaphi_deg = 360.0;
  std::vector<std::pair<double, double>> rz_m;
  std::vector<std::array<double, 3>> vertices_m;
  std::vector<std::array<int, 3>> triangles;
  double position_m[3] = {0.0, 0.0, 0.0};
  std::optional<std::string> region;
  std::string material;  // a material of the document, or a NIST material (G4_...)
};

struct SolidsSpec {
  std::string world;
  std::vector<std::string> regions;  // the order of the file's "regions" (else of first use by a solid)
  std::vector<MaterialSpec> materials;
  std::vector<SolidSpec> solids;  // every solid after its mother; the world first
};

// Reads and checks a solids file: an unknown or missing key, a wrong type or value, a mother that is not defined
// before its daughter, a material that is not defined, a GDML name used twice, a polycone that Geant4 cannot build
// (fewer than three distinct points, almost no area, an outline that crosses itself), a tessellated solid whose
// triangles do not close (an edge not used once in each direction) or enclose no positive volume (Geant4 builds an
// open tessellated solid with a warning only), a rotation (this version places
// without rotation) or units other than m, deg and kg/m3 is an error that names it. Throws GeometryError.
SolidsSpec load_solids(const std::filesystem::path& file);

struct BuiltVolume {
  const SolidSpec* spec = nullptr;
  G4VSolid* solid = nullptr;
  G4LogicalVolume* logical = nullptr;
  G4VPhysicalVolume* physical = nullptr;
  std::vector<std::size_t> daughters;  // indices into BuiltGeometry::volumes
  double offset_m[3] = {0.0, 0.0, 0.0};  // the global translation (the sum of the placements along the mothers)
  int region = -1;                       // index in SolidsSpec::regions; -1 for the world and the voids
};

struct BuiltGeometry {
  G4VPhysicalVolume* world = nullptr;
  std::vector<BuiltVolume> volumes;  // in the order of the document's solids
  // every material the volumes use, under its name in the list (a NIST material under its own): the document's in the
  // file's order, then the NIST materials the solids name, in order of first use
  std::vector<std::pair<std::string, G4Material*>> materials;
};

// Builds the materials through Geant4's NIST manager, one solid and one logical volume per solid, and the placements
// (without checking overlaps), each named by its GDML name: the solid <gdml_name>_solid, the logical volume
// <gdml_name>, the placement <gdml_name>_pv (the world's <gdml_name>_PV), the material its GDML name, as a GDML
// application reading the exported file names them (the reader names the world's placement after its logical volume
// with "_PV"). The spec must outlive the result. Call once per process, on the
// master thread.
// Throws GeometryError for a NIST material or element that Geant4 does not know.
BuiltGeometry build_geometry(const SolidsSpec& spec);

// The geometry mode: builds the file's geometry, runs Geant4's overlap check on every placement and prints the volumes
// and masses as one JSON document on stdout. Returns the exit status (0; 2 for an invalid file; 3 for an overlap).
int run_geometry(const std::filesystem::path& file);

}  // namespace axtr
