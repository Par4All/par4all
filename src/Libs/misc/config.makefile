#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1995/12/27 11:31:03 $, )
#
# files:

LIB_CFILES=	debug.c \
		file.c \
		message.c \
		string.c \
		dotting.c \
		args.c \
		signal.c \
		system.c \
		exception.c \
		mem_spy.c \
		perf_spy.c \
		malloc_debug.c 

LIB_HEADERS=	misc-local.h

#
# should be deduced?
LIB_OBJECTS=	$(LIB_CFILES:.c=.o)
#
