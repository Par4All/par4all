#Default Makefile to compile generated files

BIN=a.out

FC=mpif77
CC=mpicc

OpenMP_FLAG = -fopenmp

FLAGS = -g -O2
LDFLAGS = 


CSOURCES= *.c
FSOURCES= *.f
OBJFILES= *.o

LIB_STEP=libstep.a
RUNTIME=c
DIR_STEP=step_rt


$(BIN) : $(DIR_STEP)/$(LIB_STEP) $(OBJFILES)
	if ls  >/dev/null 2>&1 *.o ; then $(FC) $(OBJFILES) $(OpenMP_FLAG) $(LDFLAGS) -lstep -L$(DIR_STEP) -o $@; fi

$(OBJFILES):
	if ls  >/dev/null 2>&1 *.c ; then $(CC) $(OpenMP_FLAG) $(FLAGS) -c $(CSOURCES); fi
	if ls  >/dev/null 2>&1 *.f ; then $(FC) $(OpenMP_FLAG) $(FLAGS) -fno-underscoring -c $(FSOURCES); fi

$(DIR_STEP)/$(LIB_STEP):
	$(MAKE) -C $(DIR_STEP) $(LIB_STEP)

clean:
	$(MAKE) -C $(DIR_STEP) $@
	rm -f *.o *~


.IGNORE: clean