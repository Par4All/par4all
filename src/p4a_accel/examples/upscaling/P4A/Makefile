TARGET = upscale
CFILES = upscale-p4a.c yuv.c
KERNELFILES = luminance_wrapper.c  chrominance_wrapper.c 
DUMMYFILE = res.out
#BASEFLAGS = -DP4A_PROFILING
BASEFLAGS = -DP4A_TIMING

P4A_ACCEL_DIR = ../../..
include $(P4A_ACCEL_DIR)/p4a_accel.mk

run% : $(TARGET)%
	time ./$< ../Home_Trailer.yuv $(DUMMYFILE) 3525

display:
	mplayer $(DUMMYFILE) -demuxer rawvideo -rawvideo w=800:h=452