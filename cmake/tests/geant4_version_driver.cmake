# ctest driver: cmake -DSRC_DIR=<source dir> -DCASE=<name> -P geant4_version_driver.cmake
#
# Runs geant4_version_case.cmake in a child cmake with a mocked Geant4 version and multithreading flag, and checks
# the OUTCOME FOR THE RIGHT REASON: an accepted case must return (marker present) with no "CMake Error", and say
# "untested" exactly when the version is not the gate's 11.4.3; a refused case must exit non-zero, print every key
# part of the message (the version found, the accepted range, the Geant4 directory, the remedy) and never reach the
# marker. A missing module or a syntax error is neither, so it fails every case.
#
# The accepted range is 11.2 <= version < 11.5, compared as versions (11.10 is above 11.5): Geant4's own version file
# tests only the lower end of what find_package asks for, so the upper end is this function's job. A comparison of
# strings would refuse 11.10.0 by accident ("11.10.0" sorts below "11.2") but accept 11.40.0.
#
# Table row: name | Geant4_VERSION ("-": not defined) | Geant4_multithreaded_FOUND ("-": not defined) | pass/fail |
#            the reason a refusal must name (version, multithreaded, undefined)
set(_dir "/opt/geant4-mock/lib64/cmake/Geant4")
set(_cases
  "v11_4_3|11.4.3|ON|pass|"
  "v11_2_0|11.2.0|ON|pass|"
  "v11_3_2|11.3.2|ON|pass|"
  "v11_4_99|11.4.99|TRUE|pass|"
  "v11_5_0|11.5.0|ON|fail|version"
  "v11_5|11.5|ON|fail|version"
  "v11_10_0_numeric|11.10.0|ON|fail|version"
  "v11_40_0_numeric|11.40.0|ON|fail|version"
  "v12_0_0|12.0.0|ON|fail|version"
  "v11_1_3|11.1.3|ON|fail|version"
  "v10_7_4|10.7.4|ON|fail|version"
  "not_multithreaded|11.4.3|OFF|fail|multithreaded"
  "multithreaded_undefined|11.4.3|-|fail|multithreaded"
  "version_undefined|-|ON|fail|undefined"
)
if(NOT SRC_DIR OR NOT CASE)
  message(FATAL_ERROR "usage: cmake -DSRC_DIR=<source dir> -DCASE=<name> -P geant4_version_driver.cmake")
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
list(GET _f 1 ver)
list(GET _f 2 mt)
list(GET _f 3 expect)
list(LENGTH _f _n)
set(reason "")
if(_n GREATER 4)
  list(GET _f 4 reason)
endif()

set(_args -DSRC_DIR=${SRC_DIR} -DGeant4_DIR=${_dir})
if(NOT ver STREQUAL "-")
  list(APPEND _args -DGeant4_VERSION=${ver})
endif()
if(NOT mt STREQUAL "-")
  list(APPEND _args -DGeant4_multithreaded_FOUND=${mt})
endif()
execute_process(COMMAND ${CMAKE_COMMAND} ${_args} -P ${CMAKE_CURRENT_LIST_DIR}/geant4_version_case.cmake
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
set(all "${out}${err}")
message("---- case ${CASE}: expect ${expect}; child exit ${rc}\n${all}\n----")

# the message may be wrapped by CMake's indentation: let any run of white space stand for one space
function(_require text)
  string(REPLACE " " "[ \n]+" _re "${text}")
  if(NOT all MATCHES "${_re}")
    message(FATAL_ERROR "case ${CASE}: the output lacks '${text}'")
  endif()
endfunction()

if(expect STREQUAL "pass")
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "case ${CASE} should pass but the child exited with ${rc}")
  endif()
  if(NOT all MATCHES "AXMP_GEANT4_RETURNED")
    message(FATAL_ERROR "case ${CASE}: the function did not return (marker missing)")
  endif()
  if(all MATCHES "CMake Error")
    message(FATAL_ERROR "case ${CASE} should pass but printed a CMake Error")
  endif()
  # installs from 11.2 are accepted and reported as untested; the gate's version is not
  if(ver VERSION_EQUAL "11.4.3")
    if(all MATCHES "untested")
      message(FATAL_ERROR "case ${CASE}: 11.4.3 is the tested version but the output calls it untested")
    endif()
  else()
    _require("untested")
  endif()
else()
  if(rc EQUAL 0)
    message(FATAL_ERROR "case ${CASE} should fail but the child exited with 0")
  endif()
  if(all MATCHES "AXMP_GEANT4_RETURNED")
    message(FATAL_ERROR "case ${CASE} should fail but the function returned")
  endif()
  if(NOT all MATCHES "CMake Error")
    message(FATAL_ERROR "case ${CASE}: non-zero exit without a CMake Error (a crash, not the check)")
  endif()
  _require("${_dir}")
  _require("AXMP_BUILD_TRANSPORT")
  if(reason STREQUAL "version")
    string(REPLACE "." "\\." _v "${ver}")
    _require("Geant4 ${_v}")
    _require("11\\.2")
    _require("11\\.5")
  elseif(reason STREQUAL "multithreaded")
    _require("multithreaded")
    _require("GEANT4_BUILD_MULTITHREADED")
  elseif(reason STREQUAL "undefined")
    _require("Geant4_VERSION")
  endif()
endif()
message(STATUS "case ${CASE}: ok")
