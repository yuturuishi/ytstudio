cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0146 OLD)   # for disabling FindCUDA warning
project(DFineTRT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")

set(LIB_ROOT "$ENV{HOME}/Documents/libs")

# opencv
find_package(OpenCV REQUIRED)
find_package(Iconv QUIET)
# cuda
find_package(CUDA REQUIRED)
# tensorrt
set(CUDA_RUNTIME_API_DIRS "/usr/local/cuda-12.9/targets/x86_64-linux/include")

# dfine
add_executable(trtExample trtExample.cpp)
target_compile_options(trtExample PRIVATE
    -Wall
    -Wextra
    -Wno-unused-function
)
target_link_libraries(trtExample
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvonnxparser
    nvinfer
)
target_include_directories(trtExample PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_RUNTIME_API_DIRS}
)
