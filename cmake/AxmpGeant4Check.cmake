# Refuse, at configure time, a Geant4 that axmp-transport is not built for.
#
# axmp_check_geant4() runs after find_package(Geant4 11.2 CONFIG REQUIRED) and reads what that call leaves in the
# caller's scope: Geant4_VERSION, Geant4_multithreaded_FOUND and Geant4_DIR. Geant4's version file compares only the
# lower end of the version a project asks for (a range such as 11.2...<11.5 is not enforced: 11.5.0 is accepted), so
# the upper end is checked here, and as a CMake version comparison, not as strings (11.10 is newer than 11.5).
#
# It returns when 11.2 <= Geant4_VERSION < 11.5 and Geant4 was built multithreaded. Any other install ends the
# configure with an error that names the version found, the accepted range, the directory of the package and the
# remedy. An accepted version other than 11.4.3, the one the project is tested with, is reported as untested.
function(axmp_check_geant4)
  set(_found_dir "${Geant4_DIR}")
  if(NOT DEFINED Geant4_VERSION OR "${Geant4_VERSION}" STREQUAL "")
    message(FATAL_ERROR
      "find_package(Geant4) left Geant4_VERSION undefined (Geant4_DIR=${_found_dir}), so the version of the Geant4 "
      "install cannot be checked. axmp-transport needs a Geant4 from 11.2 up to, not including, 11.5. Point "
      "-DGeant4_DIR=<prefix>/lib64/cmake/Geant4 at one, or configure with -DAXMP_BUILD_TRANSPORT=OFF.")
  endif()
  if("${Geant4_VERSION}" VERSION_LESS "11.2" OR NOT "${Geant4_VERSION}" VERSION_LESS "11.5")
    message(FATAL_ERROR
      "Geant4 ${Geant4_VERSION} (Geant4_DIR=${_found_dir}) is outside the versions axmp-transport is built for: "
      "11.2 up to, not including, 11.5. Point -DGeant4_DIR=<prefix>/lib64/cmake/Geant4 at an 11.2 to 11.4 "
      "multithreaded install, or configure with -DAXMP_BUILD_TRANSPORT=OFF.")
  endif()
  if(NOT Geant4_multithreaded_FOUND)
    message(FATAL_ERROR
      "Geant4 ${Geant4_VERSION} (Geant4_DIR=${_found_dir}) was built without multithreading (multithreaded not "
      "found): axmp-transport runs Geant4's tasking run manager. Rebuild Geant4 with -DGEANT4_BUILD_MULTITHREADED=ON "
      "(GEANT4_BUILD_MULTITHREADED), point -DGeant4_DIR at a multithreaded install, or configure with "
      "-DAXMP_BUILD_TRANSPORT=OFF.")
  endif()
  if("${Geant4_VERSION}" VERSION_EQUAL "11.4.3")
    message(STATUS "Geant4 ${Geant4_VERSION} (${_found_dir}), multithreaded")
  else()
    message(STATUS "Geant4 ${Geant4_VERSION} (${_found_dir}), multithreaded: untested, the project is tested with "
                   "Geant4 11.4.3")
  endif()
endfunction()
