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
TARGET= 	transformations
#
# Name of the main program to test or use the library
MAIN=		transformations
#

#		/usr/lib/debug/malloc.o 

# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	replace.c loop_unroll.c partial_eval.c prettyprintcray.c strip_mine.c \
		loop_interchange.c interchange.c target.c nest_parallelization.c
LIB_HEADERS=	transformations-local.h
LIB_OBJECTS=	replace.o loop_unroll.o partial_eval.o prettyprintcray.o strip_mine.o \
		loop_interchange.o interchange.o target.o nest_parallelization.o
