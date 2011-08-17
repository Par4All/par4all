#!/bin/bash

# This script generate the speedup graph used in LCPC 2011 article "Static Compilation Analysis for Host-Accelerator Communication Optimization"

# list of benchmarks
export tests="2mm 3mm adi bicg correlation covariance doitgen fdtd-2d gauss-filter gemm gemver gesummv gramschmidt jacobi-1d jacobi-2d lu mvt symm-exp syrk syr2k hotspot99 lud99 srad99 Stars-PM"

# size for fonts (number over histogram bars)
export labelfontsize=9.5 

# The differents version involved
export versions="OpenMP Cuda-naive Cuda-opt HMPP-2.5.1 PGI-11.8"


# Get script dir
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

$SCRIPT_DIR/generate_speedup.sh

gnuplot histogram.gp || ( echo "Gnuplot failed ! " ; exit 1)

echo "A 'speedup.eps' file have been generated."
