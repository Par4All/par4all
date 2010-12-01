# A GNU makefile to run a demo of Par4All on "stars-pm" application

TARGET= stars-pm

ifeq "$(USE_FLOAT)" "1"
# Use single precision for the OpenMP and GPU version:
P4A_OPTIONS= -DUSE_FLOAT
# Use also single precision for the sequential version:
CFLAGS+= -DUSE_FLOAT
endif

# You can set P4A_OPTIONS to pass options to p4a
CFLAGS+= -O3 -std=c99 -Wall -D_GNU_SOURCE
LDLIBS+= -lm -lfftw3f


# To play with OpenMP parameters:
#OMP_SCHEDULE="dynamic"
#OMP_NUM_THREADS=2
#OMP_DYNAMIC=false
#OMP_NESTED=false

#export OMP_SCHEDULE
#export OMP_NUM_THREADS
#export OMP_DYNAMIC
#export OMP_NESTED


SOURCES= pm.c io.c
OBJS=$(SOURCES:.c=.o)
STUBS = pips_stubs.c

#.PHONY: clean demo $(addprefix demo_, $(TARGET)) $(addprefix run_, $(TARGET) $(TARGET:=_omp) $(TARGET:=_p4a_omp) $(TARGET:=_p4a_cuda))

#.DEFAULT:
#	echo Target $@ not implemented...

# Keep intermediate files for the demo for further inspection:
.PRECIOUS: $(TARGET:=.p4a.c) $(TARGET:=.p4a.cu) $(TARGET:=_seq) $(TARGET:=_openmp) $(TARGET:=_cuda)


#.INTERMEDIATE: $(TARGET:=_seq)

ifeq ($(opengl),1)
CFLAGS+= -D_GLGRAPHICS_
GRAPHICS+= glgraphics.o
LDLIBS+= -lGL -lGLU -lglut
P4A_OPTIONS+= -D_GLGRAPHICS_ --extra-obj=glgraphics.o
endif
ifeq ($(gtk),1)
CFLAGS+= -D_GRAPHICS_ `pkg-config --cflags gtk+-2.0`
GRAPHICS+= graphics.o
LDLIBS+= `pkg-config --libs gtk+-2.0`
P4A_OPTIONS+= -D_GRAPHICS_ --extra-obj=graphics.o
endif

default:
	echo "This the content of the file README.txt:"
	# Use more and not less because when quitting, the displayed text
	# remains displayed...
	more README.txt

demo : display_seq display_openmp display_cuda ;


clean :
	rm -rf $(TARGET:=_seq) $(TARGET:=_openmp) $(TARGET:=_cuda) $(TARGET:=.p4a.c) $(TARGET:=.p4a.cu) $(STUBS:.c=.p4a.c) $(SOURCES:.c=.p4a.c) $(STUBS:.c=.p4a.cu) $(SOURCES:.c=.p4a.cu) *~ *.database *.build *.o

run-%: $(TARGET)_%
	# Run a version and display timing information:
	time ./$< exp128.a.bin
	# It was total time (starting time, I/O and computations)


# To have shortcut as seq for of typing hyantes-static-99_seq
%:$(TARGET)_% ;

%_seq : $(SOURCES) $(GRAPHICS)
	# Compilation of the sequential program:
	$(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) -o $@ $^

%_openmp : $(SOURCES) $(GRAPHICS)
	# Parallelize and build an OpenMP version:
	p4a $(P4A_OPTIONS) -o $@ $(SOURCES) $(STUBS) --exclude-file=$(STUBS:.c=.p4a.c) $(LDLIBS)
        # P4A openmp end !

%_cuda : $(SOURCES) $(GRAPHICS)
	# Parallelize and build a CUDA version:
	p4a $(P4A_OPTIONS) --cuda -o $@ $(SOURCES) $(STUBS) --exclude-file=$(STUBS:.c=.p4a.cu) $(LDLIBS)
