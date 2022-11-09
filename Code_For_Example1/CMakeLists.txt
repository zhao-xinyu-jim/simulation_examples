cmake_minimum_required(VERSION 3.10.2)
project(Example2)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -pthread")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-std=c++14)
add_definitions(-std=c++14)

set(ALL_TARGET_LIBRARIES "")

include_directories( "/usr/include/eigen3" )

find_package (Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${Sophus_LIBRARIES} fmt)

find_package(Boost REQUIRED COMPONENTS
filesystem
thread
)

find_package ( OpenMP REQUIRED )
list ( APPEND ALL_TARGET_LIBRARIES OpenMP::OpenMP_CXX )

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
include_directories(${YAML_CPP_INCLUDEDIR})
list(APPEND ALL_TARGET_LIBRARIES ${YAML_CPP_LIBRARIES})

include_directories(${Boost_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${Boost_LIBRARIES})

add_executable(main main.cpp solver.cpp feature_node.cpp)#optimizer.cpp
target_link_libraries(main ${ALL_TARGET_LIBRARIES})

add_executable(main_submap main_submap.cpp solver.cpp feature_node.cpp)#optimizer.cpp
target_link_libraries(main_submap ${ALL_TARGET_LIBRARIES})
