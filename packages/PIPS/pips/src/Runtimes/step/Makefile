#Default Makefile to compile generated files

BIN=a.out

FC=mpif77
CC=mpicc


FLAGS = -g -O2

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

LIB_STEP=libstep.a
HEADER_F=steprt_f.h
HEADER_C=steprt_c.h
RUNTIME=c
DIR_STEP=step_rt

all: gnu 

$(BIN) : $(DIR_STEP)/$(LIB_STEP) $(OBJFILES) BIN_CRITICAL_PCOORD
	if ls  >/dev/null 2>&1 *.o ; then $(FC) $(OBJFILES) $(OpenMP_FLAG) $(LDFLAGS) -lstep -L$(DIR_STEP) -o $@; fi

BIN_CRITICAL_PCOORD: $(DIR_STEP)/critical_pcoord_program.c
	$(CC) -o $(DIR_STEP)/critical_pcoord_program  $^ 	


$(OBJFILES):
	ln -sf $(DIR_STEP)/$(HEADER_F) $(HEADER_F)
	ln -sf $(DIR_STEP)/$(HEADER_C) $(HEADER_C)
	for f in `ls *_MPI.c` ; do \
		echo $$f; \
		echo >tmp "#include \"STEP_C.h\""; \
		cat >>tmp $$f; \
		mv tmp $$f; \
	done;
	if ls  >/dev/null 2>&1 *.c ; then $(CC) $(OpenMP_FLAG) $(CFLAGS) -c $(CSOURCES); fi
	if ls  >/dev/null 2>&1 *.f ; then $(FC) $(OpenMP_FLAG) $(FFLAGS) -c $(FSOURCES); fi

$(DIR_STEP)/$(LIB_STEP):
	$(MAKE) -C $(DIR_STEP) $(LIB_STEP)

clean:
	$(MAKE) -C $(DIR_STEP) $@
	rm -f $(HEADER_F) *.o *~ *__genmod.*

clear: clean
	rm -f $(BIN)

gnu:	clear 
	export OMPI_F77=gfortran; export OMPI_FC=gfortran; export OMPI_CC=gcc; export LANG=C;  $(MAKE) $(BIN) COMPILER=gnu

intel:	clear
	export OMPI_F77=ifort; export OMPI_FC=ifort; export OMPI_CC=icc; export LANG=C;  make $(BIN) COMPILER=intel

.IGNORE: clean clear
