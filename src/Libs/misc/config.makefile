#
# The following macros define the value of commands that are used to
# compile source code.
#
# you can add your own options behind pips default values.
# 
# example: CFLAGS= $(PIPS_CFLAGS) -DSYSTEM=BSD4.2
#
AR=		$(PIPS_AR)
ARFLAGS=	$(PIPS_ARFLAGS)
CC=		$(PIPS_CC)
CFLAGS=		$(PIPS_CFLAGS)
CPPFLAGS=	$(PIPS_CPPFLAGS)
LD=		$(PIPS_LD)
LDFLAGS=	$(PIPS_LDFLAGS)
LEX=		$(PIPS_LEX)
LFLAGS=		$(PIPS_LFLAGS)
LINT=		$(PIPS_LINT)
LINTFLAGS=	$(PIPS_LINTFLAGS)
YACC=		$(PIPS_YACC)
YFLAGS=		$(PIPS_YFLAGS)
#
# The following macros define your library.
#
# Name of the library without the .a suffix.
TARGET= 	misc
#
# Name of the main program to test or use the library
MAIN=		main
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	debug.c file.c message.c string.c dotting.c args.c signal.c malloc_debug.c
LIB_HEADERS=	misc-local.h
LIB_OBJECTS=	debug.o file.o message.o string.o dotting.o args.o signal.o malloc_debug.o
