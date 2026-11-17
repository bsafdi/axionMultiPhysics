#include "axionmultiphysics/device.hpp"

#include <cctype>
#include <cstdlib>
#include <sstream>

#include <mfem.hpp>

#include "axionmultiphysics/core.hpp"

namespace axmp {

const char* gpu_backend() {
#if defined(AXMP_GPU_CUDA)
  return "cuda";
#elif defined(AXMP_GPU_HIP)
  return "hip";
#else
  return "";
#endif
}

namespace {

DeviceRequest& request_storage() {
  static DeviceRequest r;
  return r;
}

// "cpu", "<backend>" or "<backend>:<index>"; anything else (other MFEM device strings such as "omp", "debug" or
// "ceed-cuda") is refused: axionMultiPhysics's host code is not written for MFEM device memory (device.hpp).
DeviceRequest parse_device(const std::string& s, const std::string& gpu) {
  DeviceRequest r;
  if (s == "cpu") return r;
  const std::string hint = "Set AXMP_DEVICE to cpu, " + gpu + " or " + gpu +
                           ":N (a GPU index). A GPU runs the 3D engine's hypre solvers; MFEM itself always runs on the "
                           "CPU. See docs/install.md, \"GPU builds\".";
  if (s.compare(0, gpu.size(), gpu) != 0) throw Error("AXMP_DEVICE='" + s + "' is not supported by this build", "", hint);
  const std::string rest = s.substr(gpu.size());
  r.gpu = gpu;
  if (rest.empty()) return r;
  bool digits = rest.size() > 1 && rest[0] == ':';
  for (size_t i = 1; i < rest.size(); ++i) digits = digits && std::isdigit(static_cast<unsigned char>(rest[i]));
  if (!digits) throw Error("AXMP_DEVICE='" + s + "' is not supported by this build", "", hint);
  r.id = std::atoi(rest.c_str() + 1);
  return r;
}

}  // namespace

const std::string& configure_device() {
  // Function-local static: configured on the first call and never afterwards.
  static const std::string spec = [] {
    const char* env = std::getenv("AXMP_DEVICE");
    // "cpu" unless asked otherwise: a GPU build must still import on a node
    // without a GPU (login nodes, the test suite).
    std::string s = (env && *env) ? env : "cpu";
    const char* gpu = gpu_backend();
    if (s != "cpu" && *gpu == '\0') {
      throw Error("AXMP_DEVICE='" + s + "' but this build of axionMultiPhysics has no GPU backend", "",
                  "Configure with -DAXMP_GPU=CUDA or -DAXMP_GPU=HIP against an MFEM built with that "
                  "backend (scripts/build_deps_gpu.sh), or unset AXMP_DEVICE to run on the CPU. "
                  "See docs/install.md, \"GPU builds\".");
    }
    request_storage() = parse_device(s, gpu);
    // Deliberately leaked: mfem::Device must outlive every MFEM object, and a
    // static destructor would run in an order MFEM does not define.  One object
    // per process, as MFEM requires (a second one aborts).  Always the CPU: a GPU
    // request is applied to hypre when the 3D engine initializes it (device.hpp).
    auto* dev = new mfem::Device();
    dev->Configure("cpu");
    if (log_level() >= 2) {
      std::ostringstream os;
      dev->Print(os);
      log(2, "AXMP_DEVICE '" + s + "'; MFEM device: " + os.str());
    }
    return s;
  }();
  return spec;
}

const DeviceRequest& device_request() {
  configure_device();
  return request_storage();
}

}  // namespace axmp
