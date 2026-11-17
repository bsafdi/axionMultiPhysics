#!/bin/bash
# AXMP_EXE wrapper: run AXMP_REAL_EXE under gperftools' heap profiler (tcmalloc as the allocator); one profile series per
# rank in AXMP_HEAP_DIR. Each dump's header line gives the in-use heap at that moment. AXMP_TCMALLOC names the tcmalloc
# library (default: the usual location on RHEL-family systems).
rank="${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-0}}"
mkdir -p "${AXMP_HEAP_DIR}"
export LD_PRELOAD="${AXMP_TCMALLOC:-/usr/lib64/libtcmalloc.so.4}"
export HEAPPROFILE="${AXMP_HEAP_DIR}/axmp.rank${rank}"
export HEAP_PROFILE_ALLOCATION_INTERVAL="${HEAP_PROFILE_ALLOCATION_INTERVAL:-2147483648}"
export HEAP_PROFILE_INUSE_INTERVAL="${HEAP_PROFILE_INUSE_INTERVAL:-268435456}"
exec "${AXMP_REAL_EXE}" "$@"
