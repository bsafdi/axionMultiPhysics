# Refuse, at configure time, a GNU Fortran compiler older than the GNU C++ compiler.
#
# An MPI STRUMPACK (AXMP_STRUMPACK_PREFIX) makes CMake enable Fortran, and CMake takes the first gfortran on the PATH.
# On a system whose default gfortran is older than the C++ compiler (for example the distribution's GCC 8 next to a
# GCC 13 mpicxx) the Fortran compiler adds its own libstdc++ directory to the link line, and the link then fails with
# undefined std::filesystem symbols, long after the configure. Stopping here names the cause and the remedy.
#
# axmp_check_fortran_pair() reads CMAKE_Fortran_COMPILER_LOADED, CMAKE_{CXX,Fortran}_COMPILER_ID,
# CMAKE_{CXX,Fortran}_COMPILER_VERSION and CMAKE_Fortran_COMPILER. It returns silently when Fortran is not enabled or
# when either compiler is not GNU, and compares only the major versions (numerically: 10 is newer than 9).
function(axmp_check_fortran_pair)
  if(NOT CMAKE_Fortran_COMPILER_LOADED)
    return()
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    return()
  endif()
  string(REGEX MATCH "^[0-9]+" _cxx_major "${CMAKE_CXX_COMPILER_VERSION}")
  string(REGEX MATCH "^[0-9]+" _fc_major "${CMAKE_Fortran_COMPILER_VERSION}")
  if(_fc_major LESS _cxx_major)
    message(FATAL_ERROR
      "The C++ compiler is GCC ${CMAKE_CXX_COMPILER_VERSION} but CMake found the Fortran compiler "
      "${CMAKE_Fortran_COMPILER} (GCC ${CMAKE_Fortran_COMPILER_VERSION}), which the MPI STRUMPACK under "
      "AXMP_STRUMPACK_PREFIX enables: the link would take GCC ${_fc_major}'s libstdc++ and fail. "
      "Configure with -DCMAKE_Fortran_COMPILER=<gfortran or mpifort of GCC ${_cxx_major}> in a new build directory "
      "(or with cmake --fresh, CMake 3.24 or newer): CMake keeps the Fortran compiler it found first in the cache of "
      "the build directory.")
  endif()
endfunction()
