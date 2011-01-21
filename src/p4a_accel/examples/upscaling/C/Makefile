TARGET = upscale
CFILES = main.c yuv.c upscale.c
CFLAGS = -std=c99

$(TARGET): $(CFILES) 
	$(CC) $(CFLAGS) -o $@ $(CFILES)  $(LDLIBS)  

run : $(TARGET)
	time ./$< ../Home_Trailer.yuv res.out 3525

display:
	mplayer res.out -demuxer rawvideo -rawvideo w=800:h=452

clean:
	rm $(TARGET) res.out *.o *~ 