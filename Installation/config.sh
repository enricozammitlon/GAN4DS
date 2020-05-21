#!/bin/bash
echo "Adding python3.6.10..."
export PATH=$PATH:/hepgpuX-dataY/<YOUR USERNAME>/python3.6.10/bin
echo "Adding CUDA and TensorGraphics"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hepgpuX-dataY/<YOUR USERNAME>/CUDA/usr/lib64
export CPATH=$CPATH:/hepgpuX-dataY/<YOUR USERNAME>/CUDA/usr/include
echo "Adding TensorRT..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hepgpuX-dataY/<YOUR USERNAME>/CUDA/TensorRT-7.0.0.11/lib