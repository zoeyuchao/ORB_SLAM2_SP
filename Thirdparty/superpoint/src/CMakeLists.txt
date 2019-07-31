set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
SET(SRC_LIST SuperPoint.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_LIST}) 

file(GLOB_RECURSE DPU_CODES ${PROJECT_SOURCE_DIR}/model/dpu_superpoint.elf)

target_link_libraries(${PROJECT_NAME}
#${DPU_CODES}
${OpenCV_LIBS}
)
target_link_libraries(${PROJECT_NAME} pthread hineon n2cube dputils)
