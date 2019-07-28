set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
SET(SRC_LIST SuperPoint.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_LIST}) 

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
/home/yuchao/caffe-ssd/build/lib/libcaffe.so
/usr/lib/x86_64-linux-gnu/libglog.so
/usr/lib/x86_64-linux-gnu/libboost_system.so
)
