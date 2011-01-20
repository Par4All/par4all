TARGET = upscale
CFILES = upscale-p4a.c yuv.c
KERNELFILES = luminance1_wrapper.c  chrominance_wrapper.c 

P4A_ACCEL_DIR = ../../..
include $(P4A_ACCEL_DIR)/p4a_accel.mk

run% : $(TARGET)%
	time ./$< ../Home_Trailer.yuv res.out 3525

display:
	mplayer res.out -demuxer rawvideo -rawvideo w=800:h=452