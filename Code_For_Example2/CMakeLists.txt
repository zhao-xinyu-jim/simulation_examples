cmake_minimum_required(VERSION 3.10.2)
project(main)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -pthread")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-std=c++14)
add_definitions(-std=c++14)

set(ALL_TARGET_LIBRARIES "")

include_directories("/usr/include/eigen3")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${Sophus_LIBRARIES} fmt)

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
include_directories(${YAML_CPP_INCLUDEDIR})
list(APPEND ALL_TARGET_LIBRARIES ${YAML_CPP_LIBRARIES})

find_package(Boost REQUIRED COMPONENTS
    filesystem
    thread
)
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${Boost_LIBRARIES})

find_package(OpenMP REQUIRED)
list(APPEND ALL_TARGET_LIBRARIES OpenMP::OpenMP_CXX)

add_executable(main main.cpp pose_data_gen.cpp pose_node.cpp optimizer.cpp) # optimizer.cpp data_generator.cpp solver.cpp
target_link_libraries(main ${ALL_TARGET_LIBRARIES})
