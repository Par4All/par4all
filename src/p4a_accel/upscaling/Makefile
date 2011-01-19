TARGET = upscale
CFILES = main.c upscale.c yuv.c
KERNELFILES =

P4A_ACCEL_DIR = ..
include $(P4A_ACCEL_DIR)/p4a_accel.mk

run% : $(TARGET)%
	time ./$< Home_Trailer.yuv res.out 3525

#	time ./$< Home_Trailer.yuv res.out 3525

display:
	mplayer res.out -demuxer rawvideo -rawvideo w=800:h=452