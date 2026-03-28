#!/bin/bash

echo "Starting native compilation for CIC edge application on A53..."

WORKSPACE_DIR="/home/root/cic_workspace"
ORT_INC="${WORKSPACE_DIR}/include"
ORT_LIB="${WORKSPACE_DIR}/lib"
EIGEN_INC="${WORKSPACE_DIR}/include/eigen3"
OPENCV_INC="/usr/include/opencv4"

g++ -O3 -Wall -Wextra -std=c++17 -funsigned-char \
    compress.cpp -o compress \
    -I${WORKSPACE_DIR} \
    -I${OPENCV_INC} \
    -I${ORT_INC} \
    -I${EIGEN_INC} \
    -L${ORT_LIB} \
    -Wl,-rpath,${ORT_LIB} \
    -lonnxruntime \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lz

if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable 'compress' created."
else
    echo "Compilation failed. Please check the error messages."
fi
