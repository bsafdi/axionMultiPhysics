# One mocked call of axmp_check_geant4(), run as `cmake -P` by geant4_version_driver.cmake.
# Inputs (-D): SRC_DIR, and the variables a real find_package(Geant4 ... CONFIG) leaves in the caller's scope:
# Geant4_VERSION (from the package's version file), Geant4_multithreaded_FOUND (from Geant4Config.cmake) and Geant4_DIR.
# The marker is printed only when the function returns, so a FATAL_ERROR (or a missing module, or a syntax error)
# never produces it.
include("${SRC_DIR}/cmake/AxmpGeant4Check.cmake")
axmp_check_geant4()
message("AXMP_GEANT4_RETURNED")
