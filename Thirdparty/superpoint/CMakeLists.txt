cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(superpoint)
 
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

include_directories(
${PROJECT_SOURCE_DIR}/include/
~/caffe-ssd/include/ 
/usr/local/include 
/usr/local/cuda/include
) 

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
SET(SRC_LIST src/SuperPoint.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_LIST}) 

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
~/caffe-ssd/build/lib/libcaffe.so
/usr/lib/x86_64-linux-gnu/libglog.so
/usr/lib/x86_64-linux-gnu/libboost_system.so
)















#add_executable(main src/classification.cpp)
#target_link_libraries(main "${OpenCV_LIBRARIES}")
#target_link_libraries(main "libexampleapp.so")
#set_property(TARGET main PROPERTY CXX_STANDARD 11)
