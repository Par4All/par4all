TARGET=srad

SRAD_SIZE=2048
SRAD_ITER=100

RUN_ARGS:=$(SRAD_SIZE) $(SRAD_SIZE) 0 31 0 31 0.5 $(SRAD_ITER)

.PHONY: run

default:
	nvcc srad.cu -I. -I../../../common ../../../common/timing.c -o srad_cuda_hand -arch=sm_13
  
run:
	./srad_cuda_hand $(RUN_ARGS)

