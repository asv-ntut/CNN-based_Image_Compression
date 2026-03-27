#!/bin/bash

echo "Starting native compilation for tic_compress on MacOS..."

# Getting paths for homebrew packages
ORT_PATH=$(brew --prefix onnxruntime)
OPENCV_PATH=$(brew --prefix opencv)

clang++ -O3 -Wall -Wextra -lz -std=c++17 -funsigned-char \
    compress.cpp -o compress \
    -I./cpp_lib/eigen3 \
    -I${OPENCV_PATH}/include/opencv4 \
    -I${ORT_PATH}/include/onnxruntime \
    -L${OPENCV_PATH}/lib \
    -L${ORT_PATH}/lib \
    -Wl,-rpath,${ORT_PATH}/lib \
    -Wl,-rpath,${OPENCV_PATH}/lib \
    -lonnxruntime \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs

if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable 'compress' created."
else
    echo "Compilation failed."
fi