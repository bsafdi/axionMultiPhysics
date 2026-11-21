# ctest check: cmake -DSRC_DIR=<source dir> -DCASE=<pitfall_11_5_0|control_11_4_3> -DWORK_DIR=<scratch dir>
#                    -DGENERATOR=<generator> [-DMAKE_PROGRAM=<path>] -P geant4_range_pitfall.cmake
#
# The pitfall: Geant4's version file compares only the LOWER end of what find_package asks for, so
# `find_package(Geant4 11.2...<11.5 CONFIG)` accepts a Geant4 11.5.0 without a word. This configures a two-line
# project against a mocked Geant4 package whose version file behaves that way (written here from that description;
# no Geant4 file is copied) and then calls the build's check:
#   pitfall_11_5_0: find_package accepts 11.5.0 (printed), and axmp_check_geant4() refuses it (the configure fails,
#                   naming 11.5.0). Should a future CMake refuse it inside find_package already, the case passes and
#                   says so; the mocked-version cases test the function either way.
#   control_11_4_3: the same project against 11.4.3 configures.
if(NOT SRC_DIR OR NOT CASE OR NOT WORK_DIR OR NOT GENERATOR)
  message(FATAL_ERROR "usage: cmake -DSRC_DIR=... -DCASE=... -DWORK_DIR=... -DGENERATOR=... -P geant4_range_pitfall.cmake")
endif()
if(CASE STREQUAL "pitfall_11_5_0")
  set(ver "11.5.0")
elseif(CASE STREQUAL "control_11_4_3")
  set(ver "11.4.3")
else()
  message(FATAL_ERROR "unknown case ${CASE}")
endif()
file(REMOVE_RECURSE "${WORK_DIR}")
set(_pkg "${WORK_DIR}/geant4/lib/cmake/Geant4")  # lib/, which a project without languages searches too
file(WRITE "${_pkg}/Geant4ConfigVersion.cmake" "
# a version file that, like Geant4's, checks the requested minimum only
set(PACKAGE_VERSION \"${ver}\")
set(PACKAGE_VERSION_COMPATIBLE TRUE)
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
elseif(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()
")
file(WRITE "${_pkg}/Geant4Config.cmake" "set(Geant4_multithreaded_FOUND ON)\nset(Geant4_INCLUDE_DIRS \"\")\nset(Geant4_LIBRARIES \"\")\n")
file(WRITE "${WORK_DIR}/project/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.20)
project(axmp_geant4_range_pitfall LANGUAGES NONE)
find_package(Geant4 11.2...<11.5 CONFIG REQUIRED PATHS \"${WORK_DIR}/geant4\" NO_DEFAULT_PATH)
message(\"AXMP_RANGE_ACCEPTED \${Geant4_VERSION}\")
include(\"${SRC_DIR}/cmake/AxmpGeant4Check.cmake\")
axmp_check_geant4()
message(\"AXMP_GEANT4_RETURNED\")
")
set(_gen -G "${GENERATOR}")
if(MAKE_PROGRAM)
  list(APPEND _gen "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} ${_gen} -S "${WORK_DIR}/project" -B "${WORK_DIR}/build"
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
set(all "${out}${err}")
message("---- case ${CASE}: configure exit ${rc}\n${all}\n----")
string(REPLACE "." "\\." _v "${ver}")

if(CASE STREQUAL "control_11_4_3")
  if(NOT rc EQUAL 0 OR NOT all MATCHES "AXMP_RANGE_ACCEPTED ${_v}" OR NOT all MATCHES "AXMP_GEANT4_RETURNED")
    message(FATAL_ERROR "case ${CASE}: a Geant4 11.4.3 must configure (exit ${rc})")
  endif()
elseif(NOT all MATCHES "AXMP_RANGE_ACCEPTED")
  # only a refusal BY VERSION counts (a package that was not found at all would make this case vacuous)
  if(rc EQUAL 0 OR NOT all MATCHES "considered but not accepted" OR NOT all MATCHES "version: ${_v}")
    message(FATAL_ERROR "case ${CASE}: find_package did not find the mocked Geant4 ${ver}, nor refuse it by version")
  endif()
  message(STATUS "case ${CASE}: this CMake's find_package refused ${ver} itself; the pitfall is gone here")
else()
  if(NOT all MATCHES "AXMP_RANGE_ACCEPTED ${_v}")
    message(FATAL_ERROR "case ${CASE}: find_package found another version than the mocked ${ver}")
  endif()
  if(rc EQUAL 0 OR all MATCHES "AXMP_GEANT4_RETURNED")
    message(FATAL_ERROR "case ${CASE}: find_package's range let Geant4 ${ver} through and axmp_check_geant4() "
                        "accepted it: the explicit Geant4_VERSION check is missing or wrong")
  endif()
  # the refusal must be the check's: its message names the version found, the accepted range, the package's
  # directory and the remedy (a missing module or a syntax error also fails the configure, but not so)
  set(_parts "Geant4 ${_v}" "11\\.2" "11\\.5" "${_pkg}" "AXMP_BUILD_TRANSPORT")
  foreach(p IN LISTS _parts)
    string(REPLACE " " "[ \n]+" _re "${p}")
    if(NOT all MATCHES "${_re}")
      message(FATAL_ERROR "case ${CASE}: the configure failed, but not with the check's error (no '${p}')")
    endif()
  endforeach()
endif()
message(STATUS "case ${CASE}: ok")
