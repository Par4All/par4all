
BENCH_SUITE:=Polybench-2.0

# optimize only the "main"
P4A_FLAGS:=--select-modules='main'

include ../../bench.mk

