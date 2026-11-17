// Build provenance compiled into libaxionmultiphysics (see axionmultiphysics/build_provenance.hpp).
#include "axionmultiphysics/build_provenance.hpp"

#if __has_include("axionmultiphysics/build_git.hpp")
#include "axionmultiphysics/build_git.hpp"   // generated on every build by cmake/axmp_gitinfo.cmake
#else
#define AXMP_BUILD_GIT_SHA "unknown"
#define AXMP_BUILD_GIT_DIRTY 1
#define AXMP_BUILD_GIT_DIRTY_FILES ""
#endif

#if __has_include(<dlfcn.h>)
#include <dlfcn.h>
#define AXMP_HAVE_DLADDR 1
#endif

#include <filesystem>
#include <system_error>

#include <nlohmann/json.hpp>

namespace axmp {
namespace {
// The directory of the libaxionmultiphysics this function is in, as the dynamic loader found it (symlinks resolved); "" when it
// cannot be determined. Looked up at run time, so no build path is compiled into the library.
std::string loaded_library_dir() {
#ifdef AXMP_HAVE_DLADDR
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&library_build_info), &info) && info.dli_fname && *info.dli_fname) {
    std::error_code ec;
    const auto path = std::filesystem::canonical(info.dli_fname, ec);
    if (!ec) return path.parent_path().string();
  }
#endif
  return "";
}
}  // namespace

std::string library_build_info() {
  nlohmann::json b = {{"git_sha", AXMP_BUILD_GIT_SHA},
                      {"git_dirty", AXMP_BUILD_GIT_DIRTY != 0},
                      {"git_dirty_files", AXMP_BUILD_GIT_DIRTY_FILES},
                      {"build_dir", loaded_library_dir()}};
  return b.dump();
}
}  // namespace axmp
