#!/bin/bash
#SBATCH --job-name=axmp_premerge
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --time=01:15:00
#SBATCH --output=%x_%j.log
# Site-specific options: remove one '#' from these lines and fill them in, or give them to sbatch (-A, -p, -q), or
# set SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS in the environment.
##SBATCH --account=<account>
##SBATCH --partition=<partition>
##SBATCH --qos=<qos>
#
# Example SLURM job: the quick pre-merge subset (every C++ unit test, then the pytest node IDs in tests/premerge.txt
# with AXMP_SLOW=1 and AXMP_MPI=1), about 30 min on one node with 16 tasks x 8 CPUs. Submit it from the root of a source
# checkout with a build:
#
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>
#   sbatch scripts/slurm/tests_premerge.slurm
#   sbatch --export=ALL,AXMP_BUILD_DIR=$PWD/build-debug scripts/slurm/tests_premerge.slurm   # another build directory
#   python scripts/premerge_tests.py check       (without a job, seconds: the list against the suite)
#
# Optional variables:
#   AXMP_BUILD_DIR    the CMake build directory with axmp, axmp-mesher and the C++ tests (default ./build)
#   AXMP_PYTHONPATH   the package directory with its compiled _core (default ./python, the in-place package)
#   AXMP_PYTHON       the Python interpreter (default ./.venv/bin/python)
#   AXMP_ENV_HOOK     a file of site setup (module loads, ...) that scripts/env.sh sources
#   AXMP_MPI_LAUNCHER the MPI launcher for the multi-rank tests (default: srun inside a job)
#   AXMP_TMPDIR       where the job's temporary directory is created (default $TMPDIR, else /tmp)
# Extra arguments pass through to pytest (e.g. -x). 16 tasks: one test needs 16 MPI slots. The job ends with
# PREMERGE OK or PREMERGE FAILED.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "PREMERGE COMMIT $(git rev-parse HEAD)"
echo "PREMERGE DIRTY [$(git status --porcelain --untracked-files=no -- cpp cmake CMakeLists.txt python benchmarks tests | wc -l)]"
source scripts/env.sh
BUILD="${AXMP_BUILD_DIR:-${SLURM_SUBMIT_DIR}/build}"
PY="${AXMP_PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} PYTHONDONTWRITEBYTECODE=1 AXMP_SLOW=1 AXMP_MPI=1
export PYTHONPATH="${AXMP_PYTHONPATH:-${SLURM_SUBMIT_DIR}/python}"
export AXMP_EXE="${BUILD}/axmp" AXMP_MESHER="${BUILD}/axmp-mesher"
# a temporary directory outside any git repository (some tests must not create theirs inside one)
export TMPDIR="${AXMP_TMPDIR:-${TMPDIR:-/tmp}}/axmp_premerge_${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK  # srun: conflicting inherited settings
export SLURM_EXPORT_ENV=ALL  # srun passes this environment to the ranks (some sites default to NONE)
echo "build: ${BUILD}   package: ${PYTHONPATH}   python: ${PY}"
"$PY" -c "import axionmultiphysics._core as c; print('core:', c.__file__)" || { echo "PREMERGE FAILED: no compiled core"; exit 1; }
T0=$(date +%s)
ctest --test-dir "${BUILD}" --output-on-failure
CTEST=$?
T1=$(date +%s)
"$PY" scripts/premerge_tests.py run "$@"
PYTEST=$?
T2=$(date +%s)
echo "ctest exit ${CTEST} ($(( T1 - T0 )) s); pytest subset exit ${PYTEST} ($(( T2 - T1 )) s); total $(( T2 - T0 )) s"
if [ "${CTEST}" -ne 0 ] || [ "${PYTEST}" -ne 0 ]; then
  echo "PREMERGE FAILED"
  exit 1
fi
echo "PREMERGE OK"
