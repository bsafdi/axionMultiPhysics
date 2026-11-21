cmake_minimum_required(VERSION 3.20)
project(axionMultiPhysics VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(AXMP_BUILD_PYTHON "Build the nanobind Python module" ON)
option(AXMP_BUILD_MESHER "Build axmp-mesher (links the gmsh SDK; separate GPL program)" ON)
option(AXMP_BUILD_TESTS "Build C++ unit tests" ON)
option(AXMP_BUILD_TRANSPORT "Build axmp-transport (links Geant4; separate program)" OFF)
if(SKBUILD)
  # a wheel never contains Geant4 (nor the CLHEP inside it)
  if("${AXMP_BUILD_TRANSPORT}")
    message(STATUS "AXMP_BUILD_TRANSPORT is ignored in a wheel build: axmp-transport is built from source")
  endif()
  set(AXMP_BUILD_TRANSPORT OFF CACHE BOOL "Build axmp-transport (links Geant4; separate program)" FORCE)
endif()

# MPI drives the 3D engine (cpp/src/model3d*.cpp are the only files that use MPI
# or hypre).  AUTO follows the MFEM build, so a serial MFEM automatically gives
# a serial axionMultiPhysics with the 3D stub (cpp/src/model3d_nompi.cpp).
set(AXMP_MPI "AUTO" CACHE STRING "3D engine with MPI + hypre: ON, OFF, or AUTO (follow MFEM)")
set_property(CACHE AXMP_MPI PROPERTY STRINGS AUTO ON OFF)

# OpenMP parallelises assembly in the axisymmetric engine.  AUTO uses it when
# present; OFF (or a toolchain without it, e.g. Apple clang with no libomp)
# builds single-threaded through cpp/include/axionmultiphysics/omp_shim.hpp.
set(AXMP_OPENMP "AUTO" CACHE STRING "OpenMP in the assembly loops: ON, OFF, or AUTO")
set_property(CACHE AXMP_OPENMP PROPERTY STRINGS AUTO ON OFF)

# GPU: makes a backend of the MFEM build available; $AXMP_DEVICE turns it on at
# run time (see cpp/src/device.cpp and docs/install.md).
set(AXMP_GPU "OFF" CACHE STRING "GPU backend of the MFEM build: OFF, CUDA, or HIP")
set_property(CACHE AXMP_GPU PROPERTY STRINGS OFF CUDA HIP)

# Wheels: copy the third-party shared libraries beside the extension module and
# point the rpaths at them, so the wheel needs nothing from the build machine.
# auditwheel (Linux) and delocate (macOS) then only have to add system libraries.
if(SKBUILD)
  set(_axmp_bundle_default ON)
else()
  set(_axmp_bundle_default OFF)
endif()
option(AXMP_BUNDLE_LIBS "Install the third-party shared libraries into the package" ${_axmp_bundle_default})

set(AXMP_DEPS "$ENV{AXIONMULTIPHYSICS_DEPS}" CACHE PATH "third-party install prefix")
if(NOT AXMP_DEPS)
  set(AXMP_DEPS "${CMAKE_SOURCE_DIR}/third_party/install")
endif()
# Wheels also leave the build machine's paths out of the compiled code: __FILE__ (MFEM_VERIFY messages, ours and those
# in MFEM's headers) is written relative to the source tree and to the dependency prefix.
if(SKBUILD)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=" AXMP_HAVE_FILE_PREFIX_MAP)
  if(AXMP_HAVE_FILE_PREFIX_MAP)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}/=>"
                        "$<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${AXMP_DEPS}/=>")
  endif()
endif()
list(PREPEND CMAKE_PREFIX_PATH "${AXMP_DEPS}")

set(AXMP_GMSH_ROOT "" CACHE PATH "gmsh SDK or PyPI gmsh wheel prefix (holds include/gmsh.h and lib/libgmsh.*)")

# --------------------------------------------------------------- dependencies
find_package(MFEM CONFIG)
if(NOT MFEM_FOUND)
  message(FATAL_ERROR
    "MFEM was not found (searched AXMP_DEPS=${AXMP_DEPS} and CMAKE_PREFIX_PATH for MFEMConfig.cmake).\n"
    "Build the third-party stack first and point AXMP_DEPS at its prefix:\n"
    "  MPI build (3D engine):  bash scripts/fetch_deps.sh && bash scripts/build_deps.sh   (prefix third_party/install)\n"
    "  sequential build:       bash scripts/build_deps_wheel.sh                           (prefix third_party/install-seq)\n"
    "then configure with -DAXMP_DEPS=<prefix> (or export AXIONMULTIPHYSICS_DEPS=<prefix>). See docs/guides/install.md.")
endif()
foreach(_axmp_hdr "eigen3/Eigen/Core" "nlohmann/json.hpp")
  if(NOT EXISTS "${AXMP_DEPS}/include/${_axmp_hdr}")
    message(FATAL_ERROR
      "${AXMP_DEPS}/include/${_axmp_hdr} is missing: the Eigen and nlohmann/json headers are installed into AXMP_DEPS "
      "by scripts/build_deps.sh (MPI) and scripts/build_deps_wheel.sh (sequential). Run the script for this prefix "
      "or set AXMP_DEPS to the prefix it installed into.")
  endif()
endforeach()
# macOS: Apple clang has no OpenMP of its own; use Homebrew's keg-only libomp when it is installed, in CMake builds
# and in `pip install` source builds alike. Not in the wheel builds (cibuildwheel sets CIBUILDWHEEL): Homebrew's
# bottles do not support the wheels' deployment target. -DOpenMP_ROOT=... or -DAXMP_OPENMP=OFF override this.
if(APPLE AND NOT DEFINED ENV{CIBUILDWHEEL} AND NOT AXMP_OPENMP STREQUAL "OFF" AND NOT DEFINED OpenMP_ROOT
   AND NOT DEFINED ENV{OpenMP_ROOT})
  foreach(_axmp_omp "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
    if(EXISTS "${_axmp_omp}/include/omp.h")
      set(OpenMP_ROOT "${_axmp_omp}")
      message(STATUS "OpenMP: using Homebrew's libomp in ${_axmp_omp}")
      break()
    endif()
  endforeach()
endif()

if(AXMP_MPI STREQUAL "AUTO")
  if(MFEM_USE_MPI)
    set(AXMP_MPI ON)
  else()
    set(AXMP_MPI OFF)
  endif()
  message(STATUS "AXMP_MPI=AUTO -> ${AXMP_MPI} (MFEM_USE_MPI=${MFEM_USE_MPI})")
endif()
if(AXMP_MPI AND NOT MFEM_USE_MPI)
  message(FATAL_ERROR
    "AXMP_MPI=ON but this MFEM (${MFEM_DIR}) was built with MFEM_USE_MPI=NO.\n"
    "The 3D engine needs mfem::ParMesh and hypre. Either build the MPI stack "
    "(scripts/build_deps.sh) and point AXMP_DEPS at it, or configure with "
    "-DAXMP_MPI=OFF to build the serial library with the 3D stub.")
endif()
if(AXMP_MPI)
  find_package(MPI COMPONENTS C CXX)
  if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR
      "MPI was not found, and this MFEM (${MFEM_DIR}) was built with MPI.\n"
      "Install MPI with its compiler wrappers (Debian/Ubuntu: libopenmpi-dev openmpi-bin; "
      "Fedora/RHEL: openmpi-devel or mpich-devel, then 'module load mpi/openmpi-x86_64' or put its bin directory on "
      "PATH; macOS: brew install open-mpi), use the MPI that built MFEM, and either have mpicxx on PATH, pass "
      "-DMPI_CXX_COMPILER=/path/to/mpicxx, or configure with the wrapper as the compiler "
      "(-DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc).")
  endif()
elseif(MFEM_USE_MPI)
  # A serial axionMultiPhysics (3D stub) against an MPI MFEM still needs MPI to *compile*: mfem.hpp includes
  # <mpi.h> and the parallel classes unconditionally.  Without this the build fails on the first
  # translation unit.
  find_package(MPI COMPONENTS C CXX)
  if(NOT MPI_CXX_FOUND)
    message(FATAL_ERROR
      "AXMP_MPI=OFF with an MFEM built with MFEM_USE_MPI=YES (${MFEM_DIR}), and no MPI was found.\n"
      "mfem.hpp includes <mpi.h>, so even the serial library needs the MPI headers. Either load an "
      "MPI module (the stack built by scripts/build_deps.sh uses OpenMPI), build a serial MFEM, or "
      "configure with -DAXMP_MPI=ON to get the 3D engine.")
  endif()
  message(STATUS "AXMP_MPI=OFF against an MPI MFEM: linking MPI for the mfem.hpp headers only "
                 "(the 3D engine is the stub in cpp/src/model3d_nompi.cpp)")
endif()

if(AXMP_OPENMP STREQUAL "AUTO")
  find_package(OpenMP COMPONENTS CXX)
  set(AXMP_OPENMP ${OpenMP_CXX_FOUND})
  message(STATUS "AXMP_OPENMP=AUTO -> ${AXMP_OPENMP}")
elseif(AXMP_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# GPU: the backend must be in the MFEM build; axionMultiPhysics itself has no device code.
if(NOT AXMP_GPU STREQUAL "OFF")
  string(TOUPPER "${AXMP_GPU}" AXMP_GPU)
  if(AXMP_GPU STREQUAL "CUDA")
    if(NOT MFEM_USE_CUDA)
      message(FATAL_ERROR
        "-DAXMP_GPU=CUDA needs an MFEM built with MFEM_USE_CUDA=YES; ${MFEM_DIR} has "
        "MFEM_USE_CUDA=${MFEM_USE_CUDA}. Build the CUDA stack with "
        "scripts/build_deps_gpu.sh and set AXMP_DEPS to its prefix (docs/install.md).")
    endif()
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
    endif()
    message(STATUS "AXMP_GPU=CUDA (architectures ${CMAKE_CUDA_ARCHITECTURES})")
  elseif(AXMP_GPU STREQUAL "HIP")
    if(NOT MFEM_USE_HIP)
      message(FATAL_ERROR
        "-DAXMP_GPU=HIP needs an MFEM built with MFEM_USE_HIP=YES; ${MFEM_DIR} has "
        "MFEM_USE_HIP=${MFEM_USE_HIP}. See docs/install.md, \"GPU builds\".")
    endif()
    find_package(hip REQUIRED)
    message(STATUS "AXMP_GPU=HIP")
  else()
    message(FATAL_ERROR "AXMP_GPU must be OFF, CUDA or HIP; got '${AXMP_GPU}'")
  endif()
  if(NOT AXMP_MPI)
    message(WARNING
      "AXMP_GPU=${AXMP_GPU} with AXMP_MPI=OFF: the only solver that can use a GPU is the 3D "
      "engine's hypre AMS/BoomerAMG, which needs MPI, so this build has no GPU path. "
      "The axisymmetric engine is a CPU sparse direct solve.")
  endif()
endif()

# --------------------------------------------------------------- rpath helpers
if(APPLE)
  set(AXMP_ORIGIN "@loader_path")
else()
  set(AXMP_ORIGIN "$ORIGIN")
endif()
set(CMAKE_INSTALL_RPATH "${AXMP_DEPS}/lib;${AXMP_DEPS}/lib64;${AXMP_DEPS}/gmsh-sdk/lib")
set(CMAKE_BUILD_RPATH "${AXMP_DEPS}/lib;${AXMP_DEPS}/lib64;${AXMP_DEPS}/gmsh-sdk/lib")
if(SKBUILD)
  # A wheel must not keep absolute paths from the build machine.
  set(CMAKE_INSTALL_RPATH "${AXMP_ORIGIN};${AXMP_ORIGIN}/_libs")
  set(CMAKE_SKIP_BUILD_RPATH OFF)
endif()

# ------------------------------------------------------------------ library
file(GLOB AXMP_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/*.cpp)
# Exactly one of the 3D engine (model3d.cpp, model3d_fullwave.cpp, model3d_fw_indicator.cpp, model3d_surface_losses.cpp,
# model3d_electro.cpp, model3d_mech.cpp, and the 3D field-map source fieldmap3d.cpp) and model3d_nompi.cpp (the stub).
if(AXMP_MPI)
  list(REMOVE_ITEM AXMP_SOURCES ${CMAKE_SOURCE_DIR}/cpp/src/model3d_nompi.cpp)
else()
  list(REMOVE_ITEM AXMP_SOURCES ${CMAKE_SOURCE_DIR}/cpp/src/model3d.cpp ${CMAKE_SOURCE_DIR}/cpp/src/model3d_fullwave.cpp
                              ${CMAKE_SOURCE_DIR}/cpp/src/model3d_electro.cpp ${CMAKE_SOURCE_DIR}/cpp/src/model3d_mech.cpp
                              ${CMAKE_SOURCE_DIR}/cpp/src/model3d_fw_indicator.cpp
                              ${CMAKE_SOURCE_DIR}/cpp/src/model3d_surface_losses.cpp
                              ${CMAKE_SOURCE_DIR}/cpp/src/fieldmap3d.cpp)
endif()
# The field-map format's scrambled-Halton points are specified bitwise: no fused multiply-add in that file (x86-64
# without -march has none to fuse; arm64 compilers contract by default).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(${CMAKE_SOURCE_DIR}/cpp/src/fieldmap.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# chemistry module (beta): its own folder, sequential mfem::Mesh
file(GLOB AXMP_CHEM_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/chem/*.cpp)
list(APPEND AXMP_SOURCES ${AXMP_CHEM_SOURCES})

add_library(axionmultiphysics SHARED ${AXMP_SOURCES})
target_include_directories(axionmultiphysics PUBLIC
  ${CMAKE_SOURCE_DIR}/cpp/include
  ${AXMP_DEPS}/include
  ${AXMP_DEPS}/include/eigen3)
target_link_libraries(axionmultiphysics PUBLIC mfem)
# Build provenance: the git commit of the source tree and whether it differed from that commit, written by
# cmake/axmp_gitinfo.cmake into a generated header on EVERY build (not when CMake configures), so a rebuild after a
# commit reports the new commit without re-running CMake. libaxionmultiphysics (cpp/src/build_provenance.cpp) and the Python
# module (bindings.cpp, build_info()) both compile it in; a rebuild of one without the other shows as two commits.
set(AXMP_GITINFO_DIR "${CMAKE_BINARY_DIR}/generated")
add_custom_target(axmp_gitinfo ALL
  COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR}
          -DOUT=${AXMP_GITINFO_DIR}/axionmultiphysics/build_git.hpp -P ${CMAKE_SOURCE_DIR}/cmake/axmp_gitinfo.cmake
  BYPRODUCTS ${AXMP_GITINFO_DIR}/axionmultiphysics/build_git.hpp
  COMMENT "axionMultiPhysics build provenance (git sha, dirty flags)" VERBATIM)
add_dependencies(axionmultiphysics axmp_gitinfo)
target_include_directories(axionmultiphysics PRIVATE ${AXMP_GITINFO_DIR})
target_link_libraries(axionmultiphysics PRIVATE ${CMAKE_DL_LIBS})   # dladdr: where the loaded library is (glibc < 2.34)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AxmpFortranCheck.cmake")
# optional: STRUMPACK (BSD) for large axisymmetric systems (scripts/build_strumpack.sh)
option(AXMP_USE_STRUMPACK "Use STRUMPACK from AXMP_DEPS when it is installed" ON)
# A distributed-memory STRUMPACK (MPI + ScaLAPACK, scripts/build_strumpack_mpi.sh) in its own prefix replaces the
# serial one: its static library also holds the sequential solver, and the 3D direct full-wave path then factorizes
# across the ranks (AXMP_STRUMPACK_MPI) instead of on rank 0.
set(AXMP_STRUMPACK_PREFIX "" CACHE PATH "Prefix of an MPI STRUMPACK build (empty: the serial STRUMPACK in AXMP_DEPS)")
# find_package uses a cached STRUMPACK_DIR before it searches PATHS: drop one that belongs to the other prefix, so that
# switching an existing build tree between the serial and the MPI STRUMPACK does not link the wrong one
function(axmp_forget_strumpack_dir prefix)
  if(STRUMPACK_DIR)
    get_filename_component(_dir "${STRUMPACK_DIR}" REALPATH)
    get_filename_component(_pre "${prefix}" REALPATH)
    string(FIND "${_dir}/" "${_pre}/" _at)
    if(NOT _at EQUAL 0)
      message(STATUS "axionMultiPhysics: dropping the cached STRUMPACK_DIR ${STRUMPACK_DIR} (not under ${prefix})")
      unset(STRUMPACK_DIR CACHE)
    endif()
  endif()
endfunction()
# The guard around STRUMPACK's process-wide state (cpp/src/sym_solver.cpp) has two modes: "concurrent", validated on
# STRUMPACK 8.0.x, and "serialized" (one STRUMPACK call at a time in the process, frequency-parallel STRUMPACK sweeps
# run serially), which uses only STRUMPACK's public interface. Other versions always get the serialized mode; this
# option forces it on 8.0.x as well (to test it, or to rule the concurrent mode out).
option(AXMP_STRUMPACK_SERIALIZE "Serialize every STRUMPACK call, also with STRUMPACK 8.0.x" OFF)
if(AXMP_USE_STRUMPACK)
  if(AXMP_STRUMPACK_PREFIX)
    if(NOT AXMP_MPI)
      message(FATAL_ERROR "AXMP_STRUMPACK_PREFIX (an MPI STRUMPACK) needs the MPI build (AXMP_MPI)")
    endif()
    if(NOT EXISTS "${AXMP_STRUMPACK_PREFIX}/include/StrumpackConfig.h")
      message(FATAL_ERROR "no STRUMPACK under AXMP_STRUMPACK_PREFIX=${AXMP_STRUMPACK_PREFIX} "
                          "(build it with scripts/build_strumpack_mpi.sh)")
    endif()
    file(STRINGS "${AXMP_STRUMPACK_PREFIX}/include/StrumpackConfig.h" _axmp_spk_mpi REGEX "define STRUMPACK_USE_MPI")
    if(NOT _axmp_spk_mpi)
      message(FATAL_ERROR "STRUMPACK in ${AXMP_STRUMPACK_PREFIX} was not built with MPI")
    endif()
    axmp_forget_strumpack_dir("${AXMP_STRUMPACK_PREFIX}")
    find_package(STRUMPACK CONFIG REQUIRED PATHS ${AXMP_STRUMPACK_PREFIX} NO_DEFAULT_PATH)
    axmp_check_fortran_pair()
    # its headers (StrumpackConfig.h with STRUMPACK_USE_MPI) must shadow the serial ones in ${AXMP_DEPS}/include
    target_include_directories(axionmultiphysics BEFORE PUBLIC ${AXMP_STRUMPACK_PREFIX}/include)
    target_link_libraries(axionmultiphysics PUBLIC STRUMPACK::strumpack)
    target_compile_definitions(axionmultiphysics PUBLIC AXMP_STRUMPACK AXMP_STRUMPACK_MPI)
    if(NOT APPLE)
      # keep the static STRUMPACK and ScaLAPACK symbols private to libaxionmultiphysics (no interposition with other copies)
      target_link_options(axionmultiphysics PRIVATE "LINKER:--exclude-libs,libstrumpack.a:libscalapack.a")
    endif()
    message(STATUS "axionMultiPhysics: STRUMPACK ${STRUMPACK_VERSION} with MPI from ${AXMP_STRUMPACK_PREFIX}")
  else()
    axmp_forget_strumpack_dir("${AXMP_DEPS}")
    find_package(STRUMPACK CONFIG QUIET PATHS ${AXMP_DEPS} NO_DEFAULT_PATH)
    if(STRUMPACK_FOUND)
      target_link_libraries(axionmultiphysics PUBLIC STRUMPACK::strumpack)
      target_compile_definitions(axionmultiphysics PUBLIC AXMP_STRUMPACK)
      message(STATUS "axionMultiPhysics: STRUMPACK ${STRUMPACK_VERSION} for large axisymmetric systems")
    endif()
  endif()
endif()
if(TARGET STRUMPACK::strumpack)
  if(AXMP_STRUMPACK_SERIALIZE)
    target_compile_definitions(axionmultiphysics PRIVATE AXMP_STRUMPACK_SERIALIZE)
    message(STATUS "axionMultiPhysics: AXMP_STRUMPACK_SERIALIZE=ON: every STRUMPACK call runs alone (serialized guard)")
  elseif(NOT STRUMPACK_VERSION MATCHES "^8\\.0(\\.|$)")
    message(WARNING "axionMultiPhysics: STRUMPACK ${STRUMPACK_VERSION}: the concurrent STRUMPACK guard is validated on 8.0.x "
                    "only, so this build serializes every STRUMPACK call and runs frequency-parallel STRUMPACK sweeps "
                    "one frequency at a time (slower there). scripts/build_strumpack.sh installs 8.0.0.")
  endif()
endif()
# METIS: nested-dissection ordering for the large sparse Cholesky factorizations
# (Eigen/MetisSupport).  Optional: the sequential PyPI wheels are built without METIS, and the
# solver then always uses Eigen's own AMD ordering.
find_library(AXMP_METIS_LIB metis PATHS ${AXMP_DEPS}/lib ${AXMP_DEPS}/lib64 NO_DEFAULT_PATH)
if(AXMP_METIS_LIB)
  target_link_libraries(axionmultiphysics PUBLIC ${AXMP_METIS_LIB})
  target_compile_definitions(axionmultiphysics PRIVATE AXMP_HAVE_METIS=1)
  message(STATUS "METIS found (${AXMP_METIS_LIB}): nested-dissection ordering available for large 2D solves")
else()
  message(STATUS "METIS not found under ${AXMP_DEPS}/lib: the sparse direct solver will always use the "
                 "AMD ordering (scripts/build_deps.sh and scripts/build_deps_wheel.sh install METIS)")
endif()
# The source directory: the C++ tests read their data files through it, and a source build's mesher search falls back
# to <source>/build/axmp-mesher (cpp/src/axi_mesh.cpp). A wheel must not carry the path it was built in, so there the
# library itself does not get it (INTERFACE: only targets that link axionmultiphysics, such as the tests, do).
if(SKBUILD)
  target_compile_definitions(axionmultiphysics INTERFACE AXMP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
else()
  target_compile_definitions(axionmultiphysics PUBLIC AXMP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
endif()
if(AXMP_MPI)
  target_link_libraries(axionmultiphysics PUBLIC MPI::MPI_CXX)
else()
  target_compile_definitions(axionmultiphysics PUBLIC AXMP_NO_MPI=1)
  if(MFEM_USE_MPI)
    target_link_libraries(axionmultiphysics PUBLIC MPI::MPI_CXX)  # headers only: mfem.hpp includes <mpi.h>
  endif()
endif()
if(AXMP_OPENMP)
  target_link_libraries(axionmultiphysics PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(axionmultiphysics PUBLIC AXMP_HAVE_OPENMP=1)
endif()
if(AXMP_GPU STREQUAL "CUDA")
  target_compile_definitions(axionmultiphysics PUBLIC AXMP_GPU_CUDA=1)
elseif(AXMP_GPU STREQUAL "HIP")
  target_compile_definitions(axionmultiphysics PUBLIC AXMP_GPU_HIP=1)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(axionmultiphysics PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-copy)
  if(NOT AXMP_OPENMP)
    target_compile_options(axionmultiphysics PRIVATE -Wno-unknown-pragmas)
  endif()
endif()

# ------------------------------------------------------------------ apps
add_executable(axmp cpp/apps/axmp.cpp)
target_link_libraries(axmp PRIVATE axionmultiphysics)
if(SKBUILD)  # installed one level deeper, in axionmultiphysics/bin
  set_target_properties(axmp PROPERTIES INSTALL_RPATH "${AXMP_ORIGIN}/..;${AXMP_ORIGIN}/../_libs")
endif()

# fluids module (beta): its own library on top of libaxionmultiphysics,
# ParMesh-based, MPI build only; libaxionmultiphysics itself is unchanged.
if(AXMP_MPI)
  # STATIC (position-independent): it is linked into _core and must not appear as a second libaxionmultiphysics*.so in the
  # provenance of a run.
  file(GLOB AXMP_FLUIDS_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/fluids/*.cpp
       ${CMAKE_SOURCE_DIR}/cpp/src/numerics/transport/*.cpp)
  add_library(axionmultiphysics_fluids STATIC ${AXMP_FLUIDS_SOURCES})
  set_target_properties(axionmultiphysics_fluids PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(axionmultiphysics_fluids PUBLIC axionmultiphysics)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(axionmultiphysics_fluids PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-copy)
  endif()
endif()

# gmsh (GPL) is only ever linked by this separate program, so that libaxionmultiphysics and _core stay BSD-licensed.
# It is found either in the pinned SDK (third_party/install/gmsh-sdk) or in a
# prefix where the PyPI `gmsh` wheel installed include/gmsh.h and
# lib/libgmsh.so.<soname> -- the same upstream binaries, which is what the
# wheels use so that no GPL library is shipped by us (docs/install.md).
if(AXMP_BUILD_MESHER)
  set(AXMP_GMSH_HINTS "")
  if(AXMP_GMSH_ROOT)
    list(APPEND AXMP_GMSH_HINTS "${AXMP_GMSH_ROOT}")
  endif()
  if(DEFINED ENV{AXMP_GMSH_ROOT})
    list(APPEND AXMP_GMSH_HINTS "$ENV{AXMP_GMSH_ROOT}")
  endif()
  list(APPEND AXMP_GMSH_HINTS "${AXMP_DEPS}/gmsh-sdk" "${AXMP_DEPS}")
  if(AXMP_BUILD_PYTHON)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module)
    if(Python_Interpreter_FOUND)
      # The gmsh wheel installs its data files into the interpreter's data prefix, or, in an isolated build
      # environment (pip install . with build isolation), into the prefix that holds its gmsh.py
      # (<prefix>/lib/pythonX.Y/site-packages/gmsh.py); find_spec locates it without importing gmsh.
      execute_process(COMMAND "${Python_EXECUTABLE}" -c
        "import os,sys,sysconfig,importlib.util as u;s=u.find_spec('gmsh');print(sysconfig.get_paths()['data']);print(sys.base_prefix);print(os.path.normpath(os.path.join(os.path.dirname(s.origin),'..','..','..')) if s and s.origin else '')"
        OUTPUT_VARIABLE _axmp_py_dirs OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      string(REPLACE "\n" ";" _axmp_py_dirs "${_axmp_py_dirs}")
      list(APPEND AXMP_GMSH_HINTS ${_axmp_py_dirs})
    endif()
  endif()
  list(REMOVE_DUPLICATES AXMP_GMSH_HINTS)

  find_path(AXMP_GMSH_INCLUDE_DIR gmsh.h PATHS ${AXMP_GMSH_HINTS} PATH_SUFFIXES include NO_DEFAULT_PATH)
  find_library(AXMP_GMSH_LIB NAMES gmsh PATHS ${AXMP_GMSH_HINTS} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
  if(NOT AXMP_GMSH_LIB)
    # The PyPI wheel ships only the SONAME-versioned file, with no libgmsh.so symlink.
    foreach(h ${AXMP_GMSH_HINTS})
      file(GLOB _axmp_gmsh_cand "${h}/lib/libgmsh.so.*" "${h}/lib64/libgmsh.so.*"
                               "${h}/lib/libgmsh.*.dylib" "${h}/lib/libgmsh.dylib")
      if(_axmp_gmsh_cand)
        list(SORT _axmp_gmsh_cand)
        list(GET _axmp_gmsh_cand 0 AXMP_GMSH_LIB)
        break()
      endif()
    endforeach()
  endif()

  if(AXMP_GMSH_LIB AND AXMP_GMSH_INCLUDE_DIR)
    message(STATUS "axmp-mesher: gmsh ${AXMP_GMSH_LIB} (headers ${AXMP_GMSH_INCLUDE_DIR})")
    # model_selection.cpp (standard library and nlohmann/json only) resolves the named boundaries of the bounded modes
    add_executable(axmp-mesher cpp/apps/axmp_mesher.cpp cpp/src/model_selection.cpp)
    target_include_directories(axmp-mesher PRIVATE ${AXMP_GMSH_INCLUDE_DIR} ${AXMP_DEPS}/include
                                                   ${CMAKE_SOURCE_DIR}/cpp/include)
    target_link_libraries(axmp-mesher PRIVATE ${AXMP_GMSH_LIB})
    get_filename_component(_axmp_gmsh_libdir "${AXMP_GMSH_LIB}" DIRECTORY)
    if(SKBUILD)
      # libgmsh is NOT bundled (GPL): python/axionmultiphysics/_paths.py puts the gmsh
      # wheel's library directory on the loader path for this child process.
      set_target_properties(axmp-mesher PROPERTIES INSTALL_RPATH
        "${AXMP_ORIGIN}/..;${AXMP_ORIGIN}/../_libs;${AXMP_ORIGIN}/../../../..;${AXMP_ORIGIN}/../../../../..")
    else()
      set_target_properties(axmp-mesher PROPERTIES INSTALL_RPATH "${_axmp_gmsh_libdir}"
                                                  BUILD_RPATH "${_axmp_gmsh_libdir}")
    endif()
  else()
    message(WARNING
      "gmsh not found (looked for include/gmsh.h and lib/libgmsh.* under: ${AXMP_GMSH_HINTS}); "
      "axmp-mesher is not built, so meshing will not work. Install the pinned SDK "
      "(scripts/fetch_deps.sh, then scripts/build_deps.sh), or `pip install \"gmsh>=4.15,<4.16\"` into the Python "
      "that CMake uses, or set AXMP_GMSH_ROOT to a prefix with include/gmsh.h and lib/libgmsh.*.")
  endif()
endif()

# ------------------------------------------------------------------ axmp-transport
# A separate program that links Geant4 (and so the LGPL-3 CLHEP inside it) and nothing of this package: it exchanges
# files with the rest. The library, the Python module and axmp never link it. Off by default and in wheels.
if(AXMP_BUILD_TRANSPORT)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AxmpGeant4Check.cmake")
  find_package(Geant4 11.2 CONFIG REQUIRED)
  axmp_check_geant4()
  add_executable(axmp-transport
    cpp/apps/axmp_transport/main.cpp
    cpp/apps/axmp_transport/run.cpp
    cpp/apps/axmp_transport/hello.cpp
    cpp/apps/axmp_transport/job.cpp
    cpp/apps/axmp_transport/datasets.cpp
    cpp/apps/axmp_transport/fields.cpp
    cpp/apps/axmp_transport/geometry.cpp
    cpp/apps/axmp_transport/transport.cpp
    cpp/apps/axmp_transport/npy.cpp
    cpp/apps/axmp_transport/runtime.cpp)
  # nlohmann/json is header-only, from the third-party prefix; its header directory is not Geant4's
  target_include_directories(axmp-transport PRIVATE ${AXMP_DEPS}/include)
  target_compile_definitions(axmp-transport PRIVATE AXMP_TRANSPORT_VERSION="${PROJECT_VERSION}")
  target_link_libraries(axmp-transport PRIVATE ${Geant4_LIBRARIES})
  # Geant4's own libraries only: not the third-party prefix of the field solvers
  get_filename_component(_axmp_g4_libdir "${Geant4_DIR}/../.." ABSOLUTE)
  set_target_properties(axmp-transport PROPERTIES BUILD_RPATH "${_axmp_g4_libdir}" INSTALL_RPATH "${_axmp_g4_libdir}")
endif()

# ------------------------------------------------------------------ axmp-x1
# The standalone cross-check application: a plain Geant4 program that reads the exported GDML file. It needs a Geant4
# built with its GDML reader (Geant4_DIR), links nothing of this package and is never part of the library. Off by
# default and in wheels; it can also be configured on its own (see cpp/apps/axmp_x1/CMakeLists.txt).
option(AXMP_BUILD_X1 "Build axmp-x1 (links Geant4 with GDML; separate cross-check program)" OFF)
if(SKBUILD)
  set(AXMP_BUILD_X1 OFF CACHE BOOL "Build axmp-x1 (links Geant4 with GDML; separate cross-check program)" FORCE)
endif()
if(AXMP_BUILD_X1)
  add_subdirectory(cpp/apps/axmp_x1)
endif()

# ------------------------------------------------------------------ tests
if(AXMP_BUILD_TESTS)
  enable_testing()
  # A test may start threads of its own (the field-expression test evaluates one compiled expression on
  # several at once).  Before glibc 2.34 they live in libpthread, which nothing else brings into a
  # sequential build.
  find_package(Threads REQUIRED)
  file(GLOB AXMP_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/tests/test_*.cpp)
  # chemistry unit tests live next to the chemistry sources
  file(GLOB AXMP_CHEM_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cpp/src/chem/tests/test_*.cpp)
  list(APPEND AXMP_TESTS ${AXMP_CHEM_TESTS})
  foreach(t ${AXMP_TESTS})
    get_filename_component(n ${t} NAME_WE)
    if(NOT AXMP_MPI AND n MATCHES "^test_fluids_")
      continue()  # the fluids module is built with MPI only
    endif()
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE axionmultiphysics Threads::Threads)
    if(n MATCHES "^test_fluids_")
      target_link_libraries(${n} PRIVATE axionmultiphysics_fluids)
    endif()
    add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    # Four OpenMP threads unless a registration below sets its own count, so that a test does the same on every node.
    # STRUMPACK calls BLAS from all of its OpenMP threads at once, and OpenBLAS takes the work buffer of each call from
    # a table of fixed size (2 NUM_THREADS of its build by default, never fewer than 50; 96 in a cluster's OpenBLAS
    # 0.3.24). When more threads than that hold one at the same time, the threads that overflow the table together
    # get a null buffer and the process crashes: intermittently, on a node with 128 threads (test_fluids_sparse_lu and
    # test_solvers did). The tests check results, which do not depend on the thread count; four threads keep the
    # task parallelism of STRUMPACK in them and stay below the table of any OpenBLAS. The tests of thread handling set
    # their own counts.
    set_tests_properties(${n} PROPERTIES ENVIRONMENT
                         "AXMP_MESHER=$<TARGET_FILE_DIR:axmp>/axmp-mesher;OMP_NUM_THREADS=4")
  endforeach()
  # the two-rank cells of the matrix test's engine half: the same program on two ranks under the MPI launcher.  The
  # environment is the test's own (one thread per rank; two ranks on a node with fewer free slots); the timeout ends
  # a rank left waiting for the other after an error.
  if(TEST test_fluids_model_matrix AND MPIEXEC_EXECUTABLE)
    add_test(NAME test_fluids_model_matrix_2ranks
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:test_fluids_model_matrix> --ranks 2
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(test_fluids_model_matrix_2ranks PROPERTIES TIMEOUT 900 ENVIRONMENT
                         "OMP_NUM_THREADS=1;OMPI_MCA_rmaps_base_oversubscribe=1;AXMP_MESHER=$<TARGET_FILE_DIR:axmp>/axmp-mesher")
  endif()
  # test_fluids_navier_stokes compares a multi-rank solve with a one-rank one and returns without a check on one rank,
  # so the one-process registration above never runs that comparison: register the same binary on 4 ranks as well,
  # through cmake/run_mpi_ctest.sh, which picks the launcher when the test runs (not when the build is configured) and
  # skips the test (exit 77) where no launcher can start 4 ranks.
  if(AXMP_MPI AND TARGET test_fluids_navier_stokes)
    add_test(NAME test_fluids_navier_stokes_mpi4
             COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/run_mpi_ctest.sh 4 $<TARGET_FILE:test_fluids_navier_stokes>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(test_fluids_navier_stokes_mpi4 PROPERTIES
                         ENVIRONMENT "AXMP_MESHER=$<TARGET_FILE_DIR:axmp>/axmp-mesher" SKIP_RETURN_CODE 77)
  endif()
  # test_fluids_trust checks the measure of the initial projection, which is a sum over the ranks, on whatever number
  # of ranks it is started on: it compares the measure with its one-rank value and requires the same number and
  # verdict on every rank. Register it on 2 ranks as well.
  if(AXMP_MPI AND TARGET test_fluids_trust)
    add_test(NAME test_fluids_trust_mpi2
             COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/run_mpi_ctest.sh 2 $<TARGET_FILE:test_fluids_trust>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(test_fluids_trust_mpi2 PROPERTIES
                         ENVIRONMENT "AXMP_MESHER=$<TARGET_FILE_DIR:axmp>/axmp-mesher" SKIP_RETURN_CODE 77)
  endif()
  # test_fluids_sparse_lu on 2 ranks as well: an Eigen failure on rank 0 must be thrown on rank 1 too (a rank left
  # waiting in a collective call hangs, and the TIMEOUT ends it), and the stats of rank 0's factorization reach every
  # rank. The one-rank registration gets the same TIMEOUT.
  if(AXMP_MPI AND TARGET test_fluids_sparse_lu)
    add_test(NAME test_fluids_sparse_lu_mpi2
             COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/run_mpi_ctest.sh 2 $<TARGET_FILE:test_fluids_sparse_lu>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(test_fluids_sparse_lu_mpi2 PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT
                         "OMP_NUM_THREADS=1;AXMP_MESHER=$<TARGET_FILE_DIR:axmp>/axmp-mesher")
    set_tests_properties(test_fluids_sparse_lu PROPERTIES TIMEOUT 300)
  endif()
  # test_fluids_lu_chain on 2 ranks as well: the retry chain's accept/escalate decisions are rank 0's, so the attempt
  # records must be the same on every rank, and a rank sent to another stage than rank 0 waits in a collective call
  # (the TIMEOUT ends it). The one-rank registration gets the same TIMEOUT.
  if(AXMP_MPI AND TARGET test_fluids_lu_chain)
    add_test(NAME test_fluids_lu_chain_mpi2
             COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/run_mpi_ctest.sh 2 $<TARGET_FILE:test_fluids_lu_chain>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(test_fluids_lu_chain_mpi2 PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT
                         "OMP_NUM_THREADS=1;AXMP_MESHER=$<TARGET_FILE_DIR:axmp>/axmp-mesher")
    set_tests_properties(test_fluids_lu_chain PROPERTIES TIMEOUT 300)
  endif()
  # test_fluids_lu_callers on 2 ranks as well: the callers' records (the projection's, whose acceptance is a collective
  # function of the stage's solution) must be the same on every rank, and a rank left on another stage waits in a
  # collective call (the TIMEOUT ends it). The one-rank registration gets the same TIMEOUT.
  if(AXMP_MPI AND TARGET test_fluids_lu_callers)
    add_test(NAME test_fluids_lu_callers_mpi2
             COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/run_mpi_ctest.sh 2 $<TARGET_FILE:test_fluids_lu_callers>
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(test_fluids_lu_callers_mpi2 PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT
                         "OMP_NUM_THREADS=1;AXMP_MESHER=$<TARGET_FILE_DIR:axmp>/axmp-mesher")
    set_tests_properties(test_fluids_lu_callers PROPERTIES TIMEOUT 300)
  endif()
  # the configure-time check of the Fortran compiler paired with the C++ compiler (cmake/AxmpFortranCheck.cmake): the
  # function with mocked compiler ids and versions (one test per case), and that the MPI-STRUMPACK branch calls it
  set(AXMP_FORTRAN_PAIR_CASES gnu13_vs_gnu8 gnu14_vs_gnu9 gnu13_vs_gnu12_larger_minor gnu10_vs_gnu9_numeric_major
      gnu13_vs_gnu13 gnu13_vs_gnu14 gnu13_2_vs_gnu13_1_minor_older gnu9_vs_gnu10_numeric_major clang_vs_gnu8
      gnu13_vs_flang gnu13_vs_intel_fortran gnu13_vs_nag_lower_major fortran_not_enabled)
  foreach(c ${AXMP_FORTRAN_PAIR_CASES})
    add_test(NAME cmake_fortran_pair_${c}
             COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR} -DCASE=${c}
                     -P ${CMAKE_SOURCE_DIR}/cmake/tests/fortran_pair_driver.cmake)
  endforeach()
  add_test(NAME cmake_fortran_pair_wired
           COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR}
                   -P ${CMAKE_SOURCE_DIR}/cmake/tests/fortran_pair_wired.cmake)
  # the configure-time check of the Geant4 that axmp-transport is built against (cmake/AxmpGeant4Check.cmake): the
  # function with mocked versions (one test per case), and find_package's version-range pitfall against a mocked
  # Geant4 package (a configure of a two-line project). Neither needs Geant4; tests/test_transport_cmake.py checks
  # that CMakeLists.txt calls the function.
  set(AXMP_GEANT4_VERSION_CASES v11_4_3 v11_2_0 v11_3_2 v11_4_99 v11_5_0 v11_5 v11_10_0_numeric v11_40_0_numeric
      v12_0_0 v11_1_3 v10_7_4 not_multithreaded multithreaded_undefined version_undefined)
  foreach(c ${AXMP_GEANT4_VERSION_CASES})
    add_test(NAME cmake_geant4_version_${c}
             COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR} -DCASE=${c}
                     -P ${CMAKE_SOURCE_DIR}/cmake/tests/geant4_version_driver.cmake)
  endforeach()
  foreach(c pitfall_11_5_0 control_11_4_3)
    add_test(NAME cmake_geant4_range_${c}
             COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_SOURCE_DIR} -DCASE=${c}
                     -DWORK_DIR=${CMAKE_BINARY_DIR}/cmake_tests/geant4_range_${c} "-DGENERATOR=${CMAKE_GENERATOR}"
                     -DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
                     -P ${CMAKE_SOURCE_DIR}/cmake/tests/geant4_range_pitfall.cmake)
  endforeach()
  # the mesher's bounded-mode test skips (exit 77) where there is no axmp-mesher
  if(TEST test_mesher_bounded)
    set_tests_properties(test_mesher_bounded PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  endif()
  # the meshed-model test skips (exit 77) where there is no axmp-mesher
  if(TEST test_model_mesh_mesher)
    set_tests_properties(test_model_mesh_mesher PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  # the model path on tangled curved meshes skips (exit 77) where there is no axmp-mesher
  if(TEST test_model_mesh_tangled)
    set_tests_properties(test_model_mesh_tangled PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  endif()
  # one OpenMP thread when STRUMPACK loads: its timer vector then has one list
  if(TEST test_strumpack_threads)
    set_tests_properties(test_strumpack_threads PROPERTIES ENVIRONMENT
                         "AXMP_MESHER=$<TARGET_FILE_DIR:axmp>/axmp-mesher;OMP_NUM_THREADS=1")
  endif()
endif()

# ------------------------------------------------------------------ python
if(AXMP_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module)
  if(Python_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT RESULT_VARIABLE nb_res)
    if(nb_res EQUAL 0)
      find_package(nanobind CONFIG REQUIRED)
      nanobind_add_module(_core NB_STATIC cpp/python/bindings.cpp cpp/python/bindings_fieldmap.cpp
                          cpp/src/chem/python/bindings_chem.cpp)
      target_link_libraries(_core PRIVATE axionmultiphysics)
      # build provenance (cmake/axmp_gitinfo.cmake, the target axmp_gitinfo above): build_info() reads it from the
      # generated header
      add_dependencies(_core axmp_gitinfo)
      target_include_directories(_core PRIVATE ${AXMP_GITINFO_DIR})
      set_target_properties(_core PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${AXMP_ORIGIN}"
                                             BUILD_RPATH "${CMAKE_BUILD_RPATH};${CMAKE_BINARY_DIR}")
      if(NOT SKBUILD)
        # In-place development build: python/axionmultiphysics/_core*.so.  Point
        # AXMP_INPLACE_PYTHON_DIR elsewhere when several build trees (MPI,
        # serial, GPU) must coexist without overwriting each other's module.
        set(AXMP_INPLACE_PYTHON_DIR "${CMAKE_SOURCE_DIR}/python/axionmultiphysics" CACHE PATH
            "Directory a non-wheel build writes _core into")
        file(MAKE_DIRECTORY "${AXMP_INPLACE_PYTHON_DIR}")
        set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${AXMP_INPLACE_PYTHON_DIR}")
      endif()
      # fluids Python boundary (beta): the submodule axionmultiphysics._core._fluids, registered by cpp/python/bindings.cpp
      # and imported by axionmultiphysics.fluids. Every MPI build has it, the in-place CMake build and `pip install` alike;
      # the wheels are serial and do not.
      if(AXMP_MPI)
        target_sources(_core PRIVATE cpp/src/fluids/python/fluids_bindings.cpp
          cpp/src/fluids/python/fluids_bindings3d.cpp)
        target_link_libraries(_core PRIVATE axionmultiphysics_fluids)
        target_compile_definitions(_core PRIVATE AXMP_WITH_FLUIDS)
      endif()
      if(SKBUILD)
        install(TARGETS _core axionmultiphysics LIBRARY DESTINATION axionmultiphysics RUNTIME DESTINATION axionmultiphysics)
        install(TARGETS axmp RUNTIME DESTINATION axionmultiphysics/bin)
        if(TARGET axmp-mesher)
          install(TARGETS axmp-mesher RUNTIME DESTINATION axionmultiphysics/bin)
        endif()
        # The viewer assets (three.js, viewer.js, template.html) are looked up as
        # axionmultiphysics/_viewer by python/axionmultiphysics/viewer.py.
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/viewer/
                DESTINATION axionmultiphysics/_viewer
                PATTERN "*.pyc" EXCLUDE
                PATTERN "__pycache__" EXCLUDE)
        if(AXMP_BUNDLE_LIBS)
          # Third-party shared libraries, next to the extension module.
          # $ORIGIN/_libs is in the rpath of _core, libaxionmultiphysics, axmp and axmp-mesher.
          set(_axmp_bundle "")
          # MFEM_DIR is <prefix>/lib/cmake/mfem, so its grandparent is the library dir.
          get_filename_component(_axmp_mfem_dir "${MFEM_DIR}/../.." ABSOLUTE)
          foreach(d "${_axmp_mfem_dir}" "${AXMP_DEPS}/lib" "${AXMP_DEPS}/lib64")
            file(GLOB _axmp_found
                 "${d}/libmfem.so*" "${d}/libmfem*.dylib"
                 "${d}/libHYPRE.so*" "${d}/libHYPRE*.dylib"
                 "${d}/libmetis.so*" "${d}/libmetis*.dylib"
                 # STRUMPACK (BSD) when the wheel is built against a stack that has it; its own
                 # dependencies (BLAS/LAPACK, and MPI if it was built with it) are system libraries
                 # that auditwheel/delocate add.
                 "${d}/libstrumpack.so*" "${d}/libstrumpack*.dylib")
            list(APPEND _axmp_bundle ${_axmp_found})
          endforeach()
          list(REMOVE_DUPLICATES _axmp_bundle)
          # Keep only real files.  install(FILES) dereferences symlinks, so the
          # unversioned development link (libmfem.so -> libmfem.so.4.10.0) would
          # become a second full copy and double the wheel.  It is not needed at
          # run time: DT_NEEDED records the SONAME (libmfem.so.4.10.0), and
          # nothing links against the bundled copy.  A build whose SONAME *is*
          # unversioned has that name as the real file, so it is kept.
          # The copy takes the name the loader asks for: the SONAME, which CMake-built libraries carry as the
          # middle link of libX.so -> libX.so.<soversion> -> libX.so.<version> (hypre: libHYPRE.so.233 ->
          # libHYPRE.so.2.33.0; macOS: libX.<soversion>.dylib). MFEM's SONAME is its full version, the real file.
          set(_axmp_real "")
          foreach(f ${_axmp_bundle})
            if(NOT IS_SYMLINK "${f}")
              list(APPEND _axmp_real "${f}")
              get_filename_component(_axmp_name "${f}" NAME)
              get_filename_component(_axmp_freal "${f}" REALPATH)
              foreach(l ${_axmp_bundle})
                get_filename_component(_axmp_lname "${l}" NAME)
                if(IS_SYMLINK "${l}" AND NOT _axmp_lname MATCHES "^lib[^.]+\\.(so|dylib)$")
                  get_filename_component(_axmp_lreal "${l}" REALPATH)
                  if(_axmp_lreal STREQUAL _axmp_freal)
                    set(_axmp_name "${_axmp_lname}")
                  endif()
                endif()
              endforeach()
              install(FILES "${f}" DESTINATION axionmultiphysics/_libs RENAME "${_axmp_name}")
            endif()
          endforeach()
          set(_axmp_bundle ${_axmp_real})
          if(_axmp_bundle)
            message(STATUS "bundling into axionmultiphysics/_libs: ${_axmp_bundle}")
          else()
            message(WARNING "AXMP_BUNDLE_LIBS is ON but no third-party shared libraries were "
                            "found under ${AXMP_DEPS}; the wheel will depend on the build machine.")
          endif()
        endif()
      endif()
    else()
      message(WARNING "nanobind not found for ${Python_EXECUTABLE}; the Python module is not built. Install it into "
                      "that interpreter (pip install \"nanobind>=2.0\") or point CMake at the interpreter that has "
                      "it with -DPython_EXECUTABLE=/path/to/python.")
    endif()
  else()
    message(WARNING "Python >= 3.10 with development headers was not found; the Python module is not built. "
                    "Install the headers (Debian/Ubuntu: python3-dev; Fedora/RHEL: python3-devel or python3.12-devel) "
                    "or pass -DPython_EXECUTABLE=/path/to/python; -DAXMP_BUILD_PYTHON=OFF silences this.")
  endif()
endif()

# ------------------------------------------------------------------ summary
if(STRUMPACK_FOUND)
  set(_axmp_sp "${STRUMPACK_VERSION}")
else()
  set(_axmp_sp "no")
endif()
if(AXMP_METIS_LIB)
  set(_axmp_metis "yes")
else()
  set(_axmp_metis "no")
endif()
message(STATUS "axionMultiPhysics ${PROJECT_VERSION}: MPI=${AXMP_MPI} (3D engine ${AXMP_MPI}) "
               "OpenMP=${AXMP_OPENMP} GPU=${AXMP_GPU} STRUMPACK=${_axmp_sp} METIS=${_axmp_metis} "
               "deps=${AXMP_DEPS}")
