
BENCH_SUITE:=Polybench-2.0

# optimize only the "main"
P4A_FLAGS:=--select-modules='main'

currentdir=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
include $(currentdir)/../bench.mk

