#!/bin/bash

echo "Starting native compilation for decompress on MacOS..."

ORT_PATH=$(brew --prefix onnxruntime)
OPENCV_PATH=$(brew --prefix opencv)

clang++ -O3 -Wall -Wextra -std=c++17 \
    decompress.cpp -o decompress \
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
    -lopencv_imgcodecs \
    -lz

if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable 'decompress' created."
else
    echo "Compilation failed."
fi
