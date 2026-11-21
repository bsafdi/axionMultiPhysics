# One mocked call of axmp_check_fortran_pair(), run as `cmake -P` by fortran_pair_driver.cmake.
# Inputs (-D): SRC_DIR and the mocked CMAKE_<LANG>_COMPILER_ID / _VERSION variables, CMAKE_Fortran_COMPILER and
# CMAKE_Fortran_COMPILER_LOADED, exactly the variables a real configure defines.  The marker is printed only when the
# function returns, so a FATAL_ERROR (or a missing module, or a syntax error) never produces it.
include("${SRC_DIR}/cmake/AxmpFortranCheck.cmake")
axmp_check_fortran_pair()
message("AXMP_PAIR_RETURNED")
