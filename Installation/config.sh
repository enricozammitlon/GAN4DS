#!/bin/bash
echo "Adding python3.6.10..."
export PATH=:/hepgpu3-data2/ricozl/GAN4DS/python3.6.10/bin:$PATH
echo "Adding CUDA and TensorGraphics"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hepgpu3-data2/ricozl/GAN4DS/CUDA/usr/lib64
export CPATH=$CPATH:/hepgpu3-data2/ricozl/GAN4DS/CUDA/usr/include
echo "Adding TensorRT..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/hepgpu3-data2/ricozl/GAN4DS/CUDA/TensorRT-7.0.0.11/lib
