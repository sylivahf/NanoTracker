cmake_minimum_required(VERSION 3.10)

project(nanotrack)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER
    /home/itds/rk3588/rk3588_linux_release_v1.2.1/extra-tools/gcc-linaro-10.2.1-2021.01-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)

set(CMAKE_CXX_COMPILER
    /home/itds/rk3588/rk3588_linux_release_v1.2.1/extra-tools/gcc-linaro-10.2.1-2021.01-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

# set c c++ compiler ,
 
set(CMAKE_CXX_STANDARD 17) # for fileopen
add_definitions(-g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)
add_definitions(-Wno-write-strings -Wno-return-type)

# 设置 OpenCV 路径
set(OPENCV_DIR 
    /home/itds/algo_onboard_debug_env/gdbserver_test/test_prj/submodule/base_Algorithm/lib/libopencv4)
set(FFMPEG_DIR 
    /home/itds/algo_onboard_debug_env/gdbserver_test/test_prj/lib/libffmpeg)
set(OpenCV_INCLUDE_DIRS 
    ${OPENCV_DIR}/include/opencv4)

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(rknpu_tracker_file "${CMAKE_CURRENT_SOURCE_DIR}/src/nanotrack.cpp")

if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
    add_definitions(-DRV1106_1103)
endif()

if(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    add_definitions(-DRKNPU1)
endif()

# 设置RKNN包含路径和库路径
set(RKNN_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/rknpu2/include")
set(RKNN_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/rknpu2/Linux/aarch64")

# 设置RGA包含路径
set(RGA_INCLUDE_PATH "/home/itds/algo_onboard_debug_env/gdbserver_test/test_prj/submodule/AI_Algorithm/include/3rdparty/rga/RK3588/include")

# 查找所有的源文件
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# 添加可执行文件
add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RKNNModel.cpp
    ${rknpu_tracker_file}
)

# 链接必要的库文件
message(STATUS "OPENCV_ROOT = [${OPENCV_DIR}]")

target_link_libraries(${PROJECT_NAME}
    ${OPENCV_DIR}/libopencv_core.so
    ${OPENCV_DIR}/libopencv_imgproc.so
    ${OPENCV_DIR}/libopencv_videoio.so
    ${OPENCV_DIR}/libopencv_highgui.so
    ${OPENCV_DIR}/libopencv_imgcodecs.so
    ${FFMPEG_DIR}/lib/libavcodec.so
    ${FFMPEG_DIR}/lib/libavdevice.so
    ${FFMPEG_DIR}/lib/libavfilter.so
    ${FFMPEG_DIR}/lib/libavformat.so
    ${FFMPEG_DIR}/lib/libavutil.so
    ${FFMPEG_DIR}/lib/libswresample.so
    ${FFMPEG_DIR}/lib/libswscale.so
    ${FFMPEG_DIR}/lib/libpostproc.so  
    ${RKNN_LIB_PATH}/librknnrt.so
    dl
    m
    pthread
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
        log
    )
endif()

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

# 设置包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RKNN_INCLUDE_PATH}
    ${RGA_INCLUDE_PATH}  # 添加 RGA 头文件目录
    ${OpenCV_INCLUDE_DIRS}
)