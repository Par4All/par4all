#Default Makefile to compile generated files

BIN=a.out

FC=mpif77
CC=mpicc


FLAGS += -g -O3 -I$(DIR_STEP) -I.

FLAGS += $(PIPS_CPP_FLAGS)

export COMPILER = gnu
export STEP_KIND = 4

ifeq ($(COMPILER) , gnu)
   OpenMP_FLAG = -fopenmp
   export CFLAGS = $(FLAGS) -Wall
   export FFLAGS = $(FLAGS) -Wall -Wno-line-truncation -Wno-unused-variable
   ifeq ($(STEP_KIND), 8)
      FFLAGS += -fdefault-integer-8
   endif
   LDFLAGS =
endif

ifeq ($(COMPILER) , intel)
   OpenMP_FLAG = -openmp
   export CFLAGS = $(FLAGS) -Wall
   export FFLAGS = $(FLAGS) -warn all -warn notruncated_source -warn nounused
   ifeq ($(STEP_KIND), 8)
      FFLAGS += -i8
   endif
   LDFLAGS =
endif



CSOURCES= *.c
FSOURCES= *.f
OBJFILES= *.o

LIB_STEP=libstep.so
DIR_STEP=step_rt

all: gnu

$(BIN) : $(DIR_STEP)/$(LIB_STEP) compile_c compile_f
	if ls >/dev/null 2>&1 *.o ; then \
		if ls >/dev/null 2>&1 *.f ; \
		then $(FC) $(OBJFILES) $(OpenMP_FLAG) $(LDFLAGS) -lstep -L$(DIR_STEP) -Wl,-rpath=$(DIR_STEP) -o $@ ; \
		else $(CC) $(OBJFILES) $(OpenMP_FLAG) $(LDFLAGS) -lstep -L$(DIR_STEP) -Wl,-rpath=$(DIR_STEP) -o $@ ; \
		fi; \
	fi

compile_c: $(DIR_STEP)/step_api.h
	if ls  >/dev/null 2>&1 *.c ; then $(CC) $(OpenMP_FLAG) $(CFLAGS) -c $(CSOURCES); fi

compile_f: $(DIR_STEP)/STEP.h
	if ls  >/dev/null 2>&1 *.f ; then $(FC) $(OpenMP_FLAG) $(FFLAGS) -c $(FSOURCES); fi

$(DIR_STEP)/$(LIB_STEP):
	$(MAKE) -C $(DIR_STEP) $(LIB_STEP) STEP.h

clean:
	$(MAKE) -C $(DIR_STEP) $@
	rm -f *.o *~ *__genmod.*

clear: clean
	rm -f $(BIN)

gnu:	clear
	export OMPI_F77=gfortran; export OMPI_FC=gfortran; export OMPI_CC=gcc; export LANG=C;  $(MAKE) $(BIN) COMPILER=gnu

intel:	clear
	export OMPI_F77=ifort; export OMPI_FC=ifort; export OMPI_CC=icc; export LANG=C;  make $(BIN) COMPILER=intel

.IGNORE: clean clear
