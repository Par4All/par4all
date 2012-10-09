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

LIBRARY = libstep.a libstep.so

export STEP_KIND = 4
STEP_ARG = int32_t

ifeq ($(STEP_KIND), 8)
   STEP_ARG = int64_t
endif

FC = mpif90
CC = mpicc
FLAGS = -g -O2 -DSTEP_ARG=$(STEP_ARG) -DINDEX_TYPE=$(STEP_ARG) -fPIC
#FLAGS += -O0
FLAGS += -Wall
#FLAGS += -DSTEP_DEBUG
FLAGS += -DSTEP_COMMUNICATIONS_VERBOSE
#FLAGS += -DTRACE

ifeq ($(COMPILER) , gnu)
endif
ifeq ($(COMPILER) , intel)
FLAGS += -opt-report -opt-report-file=$@.report
endif
FFLAGS += $(FLAGS)
CFLAGS += $(FLAGS) -I.


SRC = step_api.c steprt.c communications.c regions.c array.c timer.c trace.c
OBJ = $(SRC:.c=.o)


all: $(LIBRARY) STEP.h

STEP.h: step_common.h
	echo "!\n!   Genereted file from $^\n!" > $@ && \
	sed -n "/^#define[\t ]\+\([^ \t]\+\)[ \t]\+\([0-9]\+\)/p" $^ | \
	grep -v "^#define[\t ]*STEP_BARRIER" | \
	sed "s,^#define[\t ]\+\([^ \t]\+\)[ \t]\+\([0-9]\+\),\n       INTEGER \1\n       PARAMETER (\1 = \2)," >>$@

step_type.o: step_type.f STEP.h
	$(FC) $(FFLAGS) -o $@ -c $<

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

libstep.a: step_type.o $(OBJ)
	ar rcs $@ $^

libstep.so: step_type.o $(OBJ)
	$(CC) -shared -o $@ $^

clean:
	@rm -f *.o  *~ *__genmod.* &>/dev/null
	@rm -f $(LIBRARY) *.report $ &>/dev/null
	@rm -f STEP.h

.IGNORE: clean
