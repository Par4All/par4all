#!/bin/sh

n=3

for size in 32 64 128 ; do 
  echo "************************** $size *************************"
  for version in "seq" "openmp" "autocuda" "autocuda_comm_optimization" "cuda"; do
    echo "Running $version version"
    for i in `seq 1 $n` ; do
      ./stars-pm_${version}_${size} data/exp${size}.a.bin 
    done
  done
done
