# Source this file (do not execute it) to put the axionMultiPhysics dependency prefix on the search paths of the build and
# of the programs that use it:
#
#   source scripts/env.sh
#
# It loads nothing and changes no MPI settings. Provide a C++17 compiler, MPI, BLAS/LAPACK, CMake and
# Python yourself (distribution packages, Homebrew, conda, environment modules, ...); docs/guides/install.md lists
# what each platform needs. For site-specific setup that should run whenever this file is sourced (for example
# `module load ...` lines), put it in a file of your own and point AXMP_ENV_HOOK at it:
#
#   AXMP_ENV_HOOK=$HOME/axmp-site.sh source scripts/env.sh
#
# Variables set: AXIONMULTIPHYSICS_ROOT (this checkout), AXIONMULTIPHYSICS_DEPS (default third_party/install; set it beforehand to
# use another prefix), CMAKE_PREFIX_PATH, PATH, LD_LIBRARY_PATH (Linux; macOS relies on the install names and
# rpaths that scripts/build_deps.sh writes), AXIONMULTIPHYSICS_GMSH when the gmsh SDK is installed, and
# OPENBLAS_NUM_THREADS=1 unless it is set (BLAS is called from inside OpenMP regions, e.g. by STRUMPACK).

if [ -n "${AXMP_ENV_HOOK:-}" ]; then
  if [ -f "$AXMP_ENV_HOOK" ]; then
    # shellcheck disable=SC1090
    . "$AXMP_ENV_HOOK"
  else
    echo "scripts/env.sh: AXMP_ENV_HOOK=$AXMP_ENV_HOOK is not a file; ignored" >&2
  fi
fi

_axmp_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export AXIONMULTIPHYSICS_ROOT="${AXIONMULTIPHYSICS_ROOT:-$_axmp_here}"
export AXIONMULTIPHYSICS_DEPS="${AXIONMULTIPHYSICS_DEPS:-$AXIONMULTIPHYSICS_ROOT/third_party/install}"

export CMAKE_PREFIX_PATH="$AXIONMULTIPHYSICS_DEPS${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PATH="$AXIONMULTIPHYSICS_DEPS/bin${PATH:+:$PATH}"
if [ "$(uname -s)" != Darwin ]; then
  export LD_LIBRARY_PATH="$AXIONMULTIPHYSICS_DEPS/lib:$AXIONMULTIPHYSICS_DEPS/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# gmsh is used only by the separate axmp-mesher program (it is GPL; see docs/install.md, "Licensing of each wheel").
if [ -x "$AXIONMULTIPHYSICS_DEPS/gmsh-sdk/bin/gmsh" ]; then
  export AXIONMULTIPHYSICS_GMSH="${AXIONMULTIPHYSICS_GMSH:-$AXIONMULTIPHYSICS_DEPS/gmsh-sdk/bin/gmsh}"
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset _axmp_here
