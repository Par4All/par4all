#*******************************************************************************
#                                                                              *
#     Authors       :   Abdellah. Kouadri                                      *
#                       Daniel Millot                                          *
#                       Frédérique Silber-Chaussumier                          *
#                                                                              *
#     Date          :   23/10/2009                                             *
#                                                                              *
#     File          :   Makefile                                               *
#                                                                              *
#     Version       :   1.1                                                    *
#                                                                              *
#     Description   :   Builds the runtime static library                      *
#                       target test builds a simple autotest                   *
#                                                                              *
#*******************************************************************************

LIBRARY = libstep.a

export STEP_KIND = 4
STEP_ARG = int32_t

ifeq ($(STEP_KIND), 8)
   STEP_ARG = int64_t
endif

FC = mpif90
CC = mpicc
FLAGS = -g -O2 -DSTEP_ARG=$(STEP_ARG) -DINDEX_TYPE=$(STEP_ARG)
#FLAGS += -O0 
FLAGS += -Wall
#FLAGS += -DSTEP_DEBUG
#FLAGS += -DSTEP_COMMUNICATIONS_VERBOSE
FLAGS += -DTRACE

ifeq ($(COMPILER) , gnu)
endif
ifeq ($(COMPILER) , intel)
FLAGS += -opt-report -opt-report-file=$@.report
endif
FFLAGS += $(FLAGS)
CFLAGS += $(FLAGS) -I.


INCLUDE_DIR = ./util
LD_FLAGS = $(INCLUDE_DIR) -ltimings 

SRC = step_api.c steprt.c communications.c regions.c array.c timer.c trace.c
OBJ = $(SRC:.c=.o) 


all: $(LIBRARY)

steprt_f.h: step_common.h
	@echo >$@ "!\n!   Genereted file from $^\n!"
	@sed -n "/^#define[\t ]\+\([^ \t]\+\)[ \t]\+\([0-9]\+\)/p"  $^ | \
	sed "s,^#define[\t ]\+\([^ \t]\+\)[ \t]\+\([0-9]\+\),\n       INTEGER \1\n       PARAMETER (\1 = \2)," >>$@

steprt_c.h: step_common.h
#	@echo >$@ "/*\n   Genereted file from $^\n*/"
#	@sed -n "/^#define[\t ]\+\([^ \t]\+\)[ \t]\+\([0-9]\+\)/p"  $^ | \
#	sed "s,^#define[\t ]\+\([^ \t]\+\)[ \t]\+\([0-9]\+\),static const int \1 = \2;," >>$@

step_type.o: step_type.f steprt_f.h steprt_c.h
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.c
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ -c $<

$(LIBRARY): step_type.o $(OBJ) 
	ar rcs $@ $^

alltests: test_array test_regions test_communications test_steprt test_step_api test_timer_c test_timer_f

test_array: array.c trace.c
	$(CC) -D TEST_ARRAY $(CFLAGS) $^ -o $@

test_regions: array.c regions.c trace.c
	$(CC) -D TEST_REGIONS $(CFLAGS) $^ -o $@

test_communications: steprt.c communications.c array.c regions.c trace.c step_type.o
	$(CC) -DTEST_COMMUNICATIONS $(CFLAGS) $^ -o $@

test_steprt: steprt.c communications.c array.c regions.c trace.c step_type.o 
	$(CC) -D TEST_STEPRT $(FLAGS) $^ -o $@

test_step_api: step_api.c steprt.c communications.c array.c regions.c trace.c step_type.o  
	$(CC) -fopenmp -D TEST_STEP_API $(FLAGS) $^ -o $@
#	$(FC) $(FFLAGS) $^ -o $@

test_timer_c: array.c timer.c trace.c
	$(CC) -D TEST_TIMER_C $(FLAGS) $^ -o $@

test_timer_f: array.c timer.c trace.c
	$(CC) $(FLAGS) -c $^
	$(FC) -D TEST_TIMER_F $(FFLAGS) $(^:.c=.o) timer.F -o $@

clean:
	@rm -f *.o  *~ *__genmod.* &>/dev/null
	@rm -f $(LIBRARY) test_* *.report $ &>/dev/null
	@rm -f steprt_f.h

.IGNORE: clean
