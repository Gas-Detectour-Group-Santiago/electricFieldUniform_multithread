cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(electricUniform)

if(NOT TARGET Garfield::Garfield)
  find_package(Garfield REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(uniformE uniformE.cxx)
target_link_libraries(uniformE Garfield::Garfield ROOT::Tree)
