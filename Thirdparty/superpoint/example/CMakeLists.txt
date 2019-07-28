set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

add_executable(example example.cpp)

target_link_libraries(example
${OpenCV_LIBS}
/home/yuchao/caffe-ssd/build/lib/libcaffe.so
/usr/lib/x86_64-linux-gnu/libglog.so
/usr/lib/x86_64-linux-gnu/libboost_system.so
${PROJECT_SOURCE_DIR}/lib/libsuperpoint.so
)
