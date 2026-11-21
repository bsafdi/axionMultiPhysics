#!/bin/sh
# Run a test binary on N MPI ranks: run_mpi_ctest.sh N BINARY [ARGS...]
#
# The launcher is chosen when the test RUNS (the machine that configured the build is often not the one that runs it),
# with the choice tests/_mpi_launch.py and Design.solve make: $AXMP_MPI_LAUNCHER (it may carry options), else srun
# inside a SLURM job, else mpirun, else mpiexec. A launcher that cannot start N ranks here (probed with `true`, so a
# failure of the test itself is never mistaken for it) is passed over; when none can, the test is skipped (exit 77,
# ctest's SKIP_RETURN_CODE) and says why.
#
# The probe shows that the launcher starts N processes, not that they form ONE MPI job (a launcher that cannot
# bootstrap this MPI library starts N one-rank jobs). So the rank count is exported as AXMP_EXPECT_RANKS, and a test
# binary that compares several ranks with one fails when MPI_Comm_size differs from it.
n=$1; shift
AXMP_EXPECT_RANKS=$n; export AXMP_EXPECT_RANKS
openmpi_flags() { "$1" --version 2>&1 | grep -qE 'Open MPI|OpenRTE' && echo "--oversubscribe --bind-to none"; }

try() {  # try LAUNCHER COMMAND...: run COMMAND with it if it can start $n ranks here
  l=$1; shift
  command -v "${l%% *}" >/dev/null 2>&1 || return 1
  if [ "$(basename "${l%% *}")" = srun ]; then
    # srun refuses mutually exclusive memory variables inherited from the allocation; the ranks get this environment
    extra=""; run_env="env -u SLURM_TRES_PER_TASK"
    [ -n "$SLURM_MEM_PER_NODE" ] && run_env="$run_env -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_GPU"
    run_env="$run_env SLURM_EXPORT_ENV=ALL"
  else
    extra=$(openmpi_flags "${l%% *}")
    # no SLURM_* variables, so that the ranks start on this node instead of through an allocation
    run_env="env $(env | sed -n 's/^\(SLURM_[A-Za-z0-9_]*\)=.*/-u \1/p' | tr '\n' ' ')"
  fi
  # shellcheck disable=SC2086
  $run_env $l -n "$n" $extra true >/dev/null 2>&1 || return 1
  # shellcheck disable=SC2086
  exec $run_env $l -n "$n" $extra "$@"
}

if [ -n "$AXMP_MPI_LAUNCHER" ]; then
  try "$AXMP_MPI_LAUNCHER" "$@"
else
  if [ -n "$SLURM_JOB_ID" ]; then try srun "$@"; fi
  try mpirun "$@"
  try mpiexec "$@"
fi
echo "skipped: no MPI launcher can start $n ranks here (AXMP_MPI_LAUNCHER, srun in a SLURM job, mpirun, mpiexec)"
exit 77
