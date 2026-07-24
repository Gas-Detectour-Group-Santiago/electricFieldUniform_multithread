
cmake_minimum_required(VERSION 3.12)

# Garfield++ y ROOT de esta máquina son Apple Silicon.
if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()

project(electricUniform)

if(NOT TARGET Garfield::Garfield)
  find_package(Garfield REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(uniformE uniformE.cxx)
target_link_libraries(uniformE Garfield::Garfield ROOT::Tree)