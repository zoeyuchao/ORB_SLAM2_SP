set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

add_executable(example example.cpp)

file(GLOB_RECURSE DPU_CODES ${PROJECT_SOURCE_DIR}/model/dpu_superpoint.elf)

target_link_libraries(example
${DPU_CODES}
${OpenCV_LIBS}
${PROJECT_SOURCE_DIR}/lib/libsuperpoint.so
)
