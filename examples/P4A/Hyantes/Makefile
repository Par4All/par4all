# A GNU makefile to run a demo of Par4All on Hyantes application excerpt

TARGET= hyantes-static-99

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


SOURCES= $(TARGET:=.c)

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
	rm -rf $(TARGET:=_seq) $(TARGET:=_openmp) $(TARGET:=_cuda)
	rm -rf $(TARGET:=_accel) $(TARGET:=.p4a.c) $(TARGET:=.p4a.cu)
	rm -rf $(TARGET:=.p4a-accel.cu)
	rm -rf  *~ *.database *.build p4a_new_files
	rm -rf  output_seq output_openmp output_cuda output_accel

run%: $(TARGET)%
	# Run a version and display timing information:
	time ./$< Rhone-alpesXYLongLat_pop.txt 1.1 32 4 35 0.01 40 > output$*
	# It was total time (starting time, I/O and computations)
	# Compare the result with the reference. Note that there may be
	# some slight differences because of non associativity of floating
	# point computations or different implementations of floating
	# point computations on GPUs:
	-diff -q output$* output_ref

display% : run%
	# Display graphically the results:
	#gnuplot -persist -e "set pm3d map; set palette gray; set size ratio 1; splot 'output$*'"
	gnuplot -persist -e "set pm3d map; set size ratio 1; splot 'output$*'"

# To be able to display the reference too:
run_ref:;

# To have shortcut as seq for of typing hyantes-static-99_seq
%:$(TARGET)_% ;

%_seq : %.c
	# Compilation of the sequential program:
	$(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $<

%_openmp : %.c
	# Parallelize and build an OpenMP version:
	p4a $(P4A_OPTIONS) -o $@ $< $(LDLIBS)

%_cuda : %.c
	# Parallelize and build a CUDA version:
	p4a $(P4A_OPTIONS) --cuda -o $@ $< $(LDLIBS)

%_accel : %.c
	# Parallelize and build an OpenMP parallel emulation of a GPU-like accelerator:
	p4a $(P4A_OPTIONS) --accel -o $@ $< $(LDLIBS)
