TARGET=lud-99
NTHREADS_CUDA = 128

LUD_SIZE:=2048
RUN_ARGS:=-i inputs/$(LUD_SIZE).dat -s $(LUD_SIZE)
P4A_FLAGS:=--select-modules='lud_99'
include ../bench.mk


