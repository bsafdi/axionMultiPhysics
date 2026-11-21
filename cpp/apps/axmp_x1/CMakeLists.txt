# axmp-x1: the standalone cross-check application (Geant4 with its GDML reader; nothing of this package is linked).
# Standalone:  cmake -S cpp/apps/axmp_x1 -B build-x1 -DGeant4_DIR=<prefix>/lib64/cmake/Geant4 -DAXMP_DEPS=<json prefix>
# or from the top level with -DAXMP_BUILD_X1=ON (the same two variables).
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(axmp_x1 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  set(AXMP_DEPS "$ENV{AXIONMULTIPHYSICS_DEPS}" CACHE PATH "prefix holding include/nlohmann/json.hpp")
endif()
find_package(Geant4 11.2 CONFIG REQUIRED gdml)
add_executable(axmp-x1 main.cpp npy.cpp)
target_include_directories(axmp-x1 PRIVATE ${AXMP_DEPS}/include)
target_link_libraries(axmp-x1 PRIVATE ${Geant4_LIBRARIES})
get_filename_component(_axmp_x1_libdir "${Geant4_DIR}/../.." ABSOLUTE)
set_target_properties(axmp-x1 PROPERTIES BUILD_RPATH "${_axmp_x1_libdir}" INSTALL_RPATH "${_axmp_x1_libdir}")
