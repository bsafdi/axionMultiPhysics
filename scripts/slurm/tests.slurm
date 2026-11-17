#!/bin/bash
#SBATCH --job-name=axmp_tests
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
# Site-specific options: remove one '#' from these lines and fill them in, or give them to sbatch (-A, -p, -q), or
# set SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS in the environment.
##SBATCH --account=<account>
##SBATCH --partition=<partition>
##SBATCH --qos=<qos>
#
# Example SLURM job: the full test suite (C++ ctest, then pytest on tests/) on one node. Submit it from the root of
# a source checkout with a build. Account, partition and QoS are site-specific: set them on the command line or
# through the environment variables sbatch reads. Extra arguments pass through to pytest:
#
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>
#   sbatch --export=ALL,AXMP_SLOW=1 scripts/slurm/tests.slurm                          # everything (about 11 h)
#   sbatch scripts/slurm/tests.slurm                                                   # without the slow tests
#   sbatch --ntasks=4 scripts/slurm/tests.slurm tests/test_fullwave3d_strumpack_mpi.py # one file
#
# AXMP_MPI=1 is set, so the multi-rank tests run (16 tasks, because one test needs 16 MPI slots); AXMP_SLOW=1 adds the
# slow 3D tests; where the time limit is shorter than the full suite needs, pass groups of test files to several jobs
# (docs/developer/testing.md). Optional variables:
#   AXMP_BUILD_DIR    the CMake build directory with axmp, axmp-mesher and the C++ tests (default ./build)
#   AXMP_PYTHONPATH   the package directory with its compiled _core (default ./python, the in-place package)
#   AXMP_PYTHON       the Python interpreter (default ./.venv/bin/python)
#   AXMP_ENV_HOOK     a file of site setup (module loads, ...) that scripts/env.sh sources
#   AXMP_MPI_LAUNCHER the MPI launcher for the multi-rank tests (default: srun inside a job)
#   AXMP_TMPDIR       where the job's temporary directory is created (default $TMPDIR, else /tmp)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/env.sh
BUILD="${AXMP_BUILD_DIR:-${SLURM_SUBMIT_DIR}/build}"
PY="${AXMP_PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} PYTHONDONTWRITEBYTECODE=1 AXMP_MPI=1
export PYTHONPATH="${AXMP_PYTHONPATH:-${SLURM_SUBMIT_DIR}/python}"
export AXMP_EXE="${BUILD}/axmp" AXMP_MESHER="${BUILD}/axmp-mesher"
export TMPDIR="${AXMP_TMPDIR:-${TMPDIR:-/tmp}}/axmp_tests_${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK  # srun: conflicting inherited settings
export SLURM_EXPORT_ENV=ALL  # srun passes this environment to the ranks (some sites default to NONE)
echo "build: ${BUILD}   package: ${PYTHONPATH}   python: ${PY}"
ctest --test-dir "${BUILD}" --output-on-failure
CTEST=$?
"$PY" -m pytest -q -rfEs tests "$@"
PYTEST=$?
echo "ctest exit ${CTEST}; pytest exit ${PYTEST}"
[ "${CTEST}" -eq 0 ] && [ "${PYTEST}" -eq 0 ]
