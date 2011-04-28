# A GNU makefile to run a demo of Par4All on Hyantes application excerpt

TARGET= main

ifeq "$(USE_FLOAT)" "1"
# Use single precision for the OpenMP and GPU version:
P4A_OPTIONS= -DUSE_FLOAT
# Use also single precision for the sequential version:
CFLAGS+= -DUSE_FLOAT
endif

# You can set P4A_OPTIONS to pass options to p4a
CFLAGS+= -O3 -std=c99 -Wall -D_GNU_SOURCE
LDLIBS+= -lm

# To play with OpenMP parameters:
#OMP_SCHEDULE="dynamic"
#OMP_NUM_THREADS=2
#OMP_DYNAMIC=false
#OMP_NESTED=false

#export OMP_SCHEDULE
#export OMP_NUM_THREADS
#export OMP_DYNAMIC
#export OMP_NESTED


SOURCES= main.c yuv.c

#.PHONY: clean demo $(addprefix demo_, $(TARGET)) $(addprefix run_, $(TARGET) $(TARGET:=_omp) $(TARGET:=_p4a_omp) $(TARGET:=_p4a_cuda))

#.DEFAULT:
#	echo Target $@ not implemented...

# Keep intermediate files for the demo for further inspection:
.PRECIOUS: $(TARGET:=.p4a.c) $(TARGET:=.p4a.cu) $(TARGET:=.p4a-accel.cu) $(TARGET:=_seq) $(TARGET:=_openmp) $(TARGET:=_cuda) $(TARGET:=_accel)


#.INTERMEDIATE: $(TARGET:=_seq)


default:
	echo "This the content of the file README.txt:"
	# Use more and not less because when quitting, the displayed text
	# remains displayed...
	more README.txt

demo : display_seq display_openmp display_cuda display_accel ;

clean :
	rm -rf $(TARGET:=_seq) $(TARGET:=_openmp) $(TARGET:=_cuda) $(TARGET:=_opencl)
	rm -rf $(TARGET:=_accel) $(SOURCES:.c=.p4a.c) $(SOURCES:.c=.p4a.cu)
	rm -rf $(SOURCES:.c=.p4a-accel.cu)
	rm -rf  *~ *.database *.build p4a_new_files
	rm -rf  res.out
	rm *.cl

run%: $(TARGET)%
	# Run a version and display timing information:
	time ./$< BUS_176x144_15_orig_01.yuv res.out

display :
	mplayer res.out -demuxer rawvideo -rawvideo w=352:h=288

display% : 
	# Display graphically the results:
	mplayer res.out -demuxer rawvideo -rawvideo w=352:h=288

# To be able to display the reference too:
run_ref:;

# To have shortcut as seq insteed of typing upscaling_seq
%:$(TARGET)_% ;

$(TARGET)_seq : $(SOURCES)
	# Compilation of the sequential program:
	$(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $(SOURCES)
	# $(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $(TARGET)_seq $(SOURCES)

$(TARGET)_openmp : $(SOURCES)
	# Parallelize and build an OpenMP version:
	p4a $(P4A_OPTIONS) -o $@ $(SOURCES) $(LDLIBS)

$(TARGET)_cuda : $(SOURCES)
	# Parallelize and build a CUDA version:
	p4a $(P4A_OPTIONS) --cuda -o $@ $(SOURCES) $(LDLIBS)

$(TARGET)_accel : $(SOURCES)
	# Parallelize and build an OpenMP parallel emulation of a GPU-like accelerator:
	p4a $(P4A_OPTIONS) --accel -o $@ $(SOURCES) $(LDLIBS)

$(TARGET)_opencl : $(SOURCES)
	# Parallelize and build an OpenCL parallel emulation:
	# p4a $(P4A_OPTIONS) --opencl -k -g -o $@ $(SOURCES) $(LDLIBS)
	p4a $(P4A_OPTIONS) --opencl -o $@ $(SOURCES) $(LDLIBS)
