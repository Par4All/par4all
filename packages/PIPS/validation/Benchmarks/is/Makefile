SHELL=/bin/sh
BENCHMARK=is
PROGRAM=is

OBJS = is.o \
       c_print_results.o \
       c_timers.o \
       wtime.o


all:${PROGRAM}

${PROGRAM}: ${OBJS}
	${CC} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CC} -c ${CPPFLAGS} ${CFLAGS} $<

is.o:             is.c  npbparams.h


clean:
	- rm -f *.o *~ mputil*
	- rm -f core
