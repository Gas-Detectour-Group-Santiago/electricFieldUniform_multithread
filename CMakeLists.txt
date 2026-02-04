# Use the code in this section for standalone projects
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(uniformE)
if(NOT TARGET Garfield::Garfield)
  find_package(Garfield REQUIRED)
endif()
find_package(GSL REQUIRED)

#set(CMAKE_CXX_STANDARD 17)

# ---Build executable------------------------------------------------------------
add_executable(uniformE uniformE.C)
target_link_libraries(uniformE Garfield::Garfield ROOT::Tree)

## ---Copy all files locally to the build directory-------------------------------
#foreach(_file ELIST.lis MPLIST.lis NLIST.lis PRNSOL.lis)
#  configure_file(${_file} ${CMAKE_CURRENT_BINARY_DIR}/${_file} COPYONLY)
#endforeach()

