TARGET =  saxpy

SOURCES= $(TARGET:=.c)
P4A_CUDA_FLAGS=--c99

RUN_ARG?=8192

include ../common.mk

display_%: run_% ;
