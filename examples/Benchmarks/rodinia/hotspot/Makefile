TARGET=hotspot_99
NTHREADS_CUDA = 128

HOTSPOT_SIZE:=512
RUN_ARGS:=$(HOTSPOT_SIZE) $(HOTSPOT_SIZE) 200 1 inputs/temp_$(HOTSPOT_SIZE) inputs/power_$(HOTSPOT_SIZE)

include ../../bench.mk


