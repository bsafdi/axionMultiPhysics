// Device (CPU / CUDA / HIP) selection for the process.
//
// MFEM needs exactly one mfem::Device, configured before any mesh or finite
// element space is created, which is why this is called from the entry points
// (cpp/apps/axmp.cpp and the nanobind module init) rather than from the solvers.
//
// Everything runs on the CPU unless $AXMP_DEVICE says otherwise: -DAXMP_GPU=CUDA
// (or HIP) makes a backend *available*, and $AXMP_DEVICE turns it on, so that
// importing axionmultiphysics on a login node with no GPU cannot abort.  $AXMP_DEVICE is
// "cpu" (the default), the build's backend ("cuda", "hip"), or that backend
// with a GPU index ("cuda:1").
//
// A GPU request moves hypre alone (the 3D engine's AMS/BoomerAMG and hypre's
// Krylov solvers; cpp/src/model3d.cpp, ensure_mpi): MFEM's own device stays the
// CPU. axionMultiPhysics assembles with MFEM's full (legacy) assembly on the host and its
// post-processing reads and writes vectors element by element on the host; under
// an MFEM GPU device those vectors would live on the GPU and the host would read
// stale copies (docs/validation/gpu.md). With MFEM on the host, MFEM copies the
// vectors and matrices into hypre's memory at every hypre call instead.
// See docs/install.md, "GPU builds".
#pragma once

#include <string>

namespace axmp {

// Configure the MFEM device once; later calls return the $AXMP_DEVICE string.
const std::string& configure_device();

// The GPU backend this build can use: "cuda", "hip", or "" for a CPU-only build.
const char* gpu_backend();

// Where $AXMP_DEVICE puts hypre: gpu is "" for the CPU, else the backend ("cuda" or "hip"); id is the GPU index
// ("cuda:1"), or -1 for the first visible GPU. A GPU run is refused on more than one MPI rank for now
// (cpp/src/model3d.cpp, configure_hypre_gpu; docs/validation/gpu.md).
struct DeviceRequest {
  std::string gpu;
  int id = -1;
};
const DeviceRequest& device_request();

}  // namespace axmp
