#
# $RCSfile: config.makefile,v $ version $Revision$
# ($Date: 1995/09/15 18:50:28 $, )
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
		malloc_debug.c 
LIB_HEADERS=	misc-local.h
#
# should be deduced?
LIB_OBJECTS=	$(LIB_CFILES:.c=.o)
#
