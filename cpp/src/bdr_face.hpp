// The face of a boundary element of a 2D mesh, checked.  Internal: not installed.
//
// MFEM answers -1 for a boundary element that is no face of the mesh (a boundary edge that no element owns). Writing
// through that index corrupts memory, so the readers of a 2D mesh take the index through this function.
#pragma once

#include <mfem.hpp>

#include "axionmultiphysics/core.hpp"

namespace axmp {

inline int checked_bdr_face(const mfem::Mesh& mesh, int be) {
  const int f = mesh.GetBdrElementFaceIndex(be);
  if (f < 0 || f >= mesh.GetNumFaces()) {
    throw Error(strcat("boundary element ", be, " (attribute ", mesh.GetBdrAttribute(be), ") is not a face of the mesh"), "",
                "A boundary curve of the geometry is degenerate or lies outside every region, so the mesh has an edge that "
                "belongs to no element. Check the sizes and positions of the curves.");
  }
  return f;
}

}  // namespace axmp
