# ctest driver: cmake -DSRC_DIR=<source dir> -DCASE=<name> -P fortran_pair_driver.cmake
#
# Runs fortran_pair_case.cmake in a child cmake with mocked compiler ids and versions and checks the OUTCOME FOR THE
# RIGHT REASON: a passing case must return (marker present) with no "CMake Error"; a failing case must exit non-zero,
# print every key part of the message (the two actual versions, the Fortran compiler path, the remedy flag) and never
# reach the marker.  A missing module or a syntax error is neither, so it fails every case.
#
# Table row: name | C++ id | C++ version | Fortran loaded (1/0) | Fortran id | Fortran version | expectation (pass/fail)
set(_cxx_path "/opt/gcc/bin/g++")
set(_f_path "/usr/bin/gfortran")
set(_cases
  "gnu13_vs_gnu8|GNU|13.2.0|1|GNU|8.5.0|fail"
  "gnu14_vs_gnu9|GNU|14.2.1|1|GNU|9.3.0|fail"
  "gnu13_vs_gnu12_larger_minor|GNU|13.2.0|1|GNU|12.9.0|fail"
  "gnu10_vs_gnu9_numeric_major|GNU|10.2.0|1|GNU|9.4.0|fail"
  "gnu13_vs_gnu13|GNU|13.2.0|1|GNU|13.2.0|pass"
  "gnu13_vs_gnu14|GNU|13.2.0|1|GNU|14.1.0|pass"
  "gnu13_2_vs_gnu13_1_minor_older|GNU|13.2.0|1|GNU|13.1.0|pass"
  "gnu9_vs_gnu10_numeric_major|GNU|9.4.0|1|GNU|10.2.0|pass"
  "clang_vs_gnu8|Clang|17.0.6|1|GNU|8.5.0|pass"
  "gnu13_vs_flang|GNU|13.2.0|1|LLVMFlang|17.0.6|pass"
  "gnu13_vs_intel_fortran|GNU|13.2.0|1|IntelLLVM|2024.0.0|pass"
  "gnu13_vs_nag_lower_major|GNU|13.2.0|1|NAG|7.2|pass"
  "fortran_not_enabled|GNU|13.2.0|0|GNU|8.5.0|pass"
)
if(NOT SRC_DIR OR NOT CASE)
  message(FATAL_ERROR "usage: cmake -DSRC_DIR=<source dir> -DCASE=<name> -P fortran_pair_driver.cmake")
endif()
set(_row "")
foreach(c IN LISTS _cases)
  if(c MATCHES "^${CASE}\\|")
    set(_row "${c}")
  endif()
endforeach()
if(NOT _row)
  message(FATAL_ERROR "unknown case ${CASE}")
endif()
string(REPLACE "|" ";" _f "${_row}")
list(GET _f 1 cxx_id)
list(GET _f 2 cxx_ver)
list(GET _f 3 f_loaded)
list(GET _f 4 f_id)
list(GET _f 5 f_ver)
list(GET _f 6 expect)

set(_args
  -DSRC_DIR=${SRC_DIR}
  -DCMAKE_CXX_COMPILER=${_cxx_path}
  -DCMAKE_CXX_COMPILER_ID=${cxx_id} -DCMAKE_CXX_COMPILER_VERSION=${cxx_ver}
  -DCMAKE_Fortran_COMPILER=${_f_path}
  -DCMAKE_Fortran_COMPILER_ID=${f_id} -DCMAKE_Fortran_COMPILER_VERSION=${f_ver})
if(f_loaded)
  list(APPEND _args -DCMAKE_Fortran_COMPILER_LOADED=1)
endif()
execute_process(COMMAND ${CMAKE_COMMAND} ${_args} -P ${CMAKE_CURRENT_LIST_DIR}/fortran_pair_case.cmake
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
set(all "${out}${err}")
message("---- case ${CASE}: expect ${expect}; child exit ${rc}\n${all}\n----")

if(expect STREQUAL "pass")
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "case ${CASE} should pass but the child exited with ${rc}")
  endif()
  if(NOT all MATCHES "AXMP_PAIR_RETURNED")
    message(FATAL_ERROR "case ${CASE}: the function did not return (marker missing)")
  endif()
  if(all MATCHES "CMake Error")
    message(FATAL_ERROR "case ${CASE} should pass but printed a CMake Error")
  endif()
else()
  if(rc EQUAL 0)
    message(FATAL_ERROR "case ${CASE} should fail but the child exited with 0")
  endif()
  if(all MATCHES "AXMP_PAIR_RETURNED")
    message(FATAL_ERROR "case ${CASE} should fail but the function returned")
  endif()
  if(NOT all MATCHES "CMake Error")
    message(FATAL_ERROR "case ${CASE}: non-zero exit without a CMake Error (a crash, not the check)")
  endif()
  # the message's key parts, with the actual (mocked) values
  string(REPLACE "." "\\." _cxx_re "${cxx_ver}")
  string(REPLACE "." "\\." _f_re "${f_ver}")
  # the majors the message derives (they must not be hard-coded)
  string(REGEX MATCH "^[0-9]+" _cxx_major "${cxx_ver}")
  string(REGEX MATCH "^[0-9]+" _f_major "${f_ver}")
  set(_parts
    "The C\\+\\+ compiler is GCC ${_cxx_re}"
    "the Fortran compiler ${_f_path} \\(GCC ${_f_re}\\)"
    "AXMP_STRUMPACK_PREFIX"
    "libstdc\\+\\+"
    "-DCMAKE_Fortran_COMPILER="
    "new build directory"
    "cmake --fresh"
    "GCC ${_f_major}'s libstdc\\+\\+"
    "of GCC ${_cxx_major}> in a new")
  foreach(p IN LISTS _parts)
    # the message may be wrapped by CMake's indentation: let any run of white space stand for one space
    string(REPLACE " " "[ \n]+" p_ws "${p}")
    if(NOT all MATCHES "${p_ws}")
      message(FATAL_ERROR "case ${CASE}: the error message lacks '${p}'")
    endif()
  endforeach()
endif()
message(STATUS "case ${CASE}: ok")
