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
TARGET= 	ri-util
#
# Name of the main program to test or use the library
MAIN=		main
#
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	Psc.c Ppolynome.c Pvecteur.c Pcontrainte.c matrice.c eval.c \
		size.c util.c ordering.c prettyprint.c \
		attachment_pretty_print.c \
		loop.c fortran90.c \
		constant.c bound_generation.c entity.c variable.c statement.c \
		expression.c type.c normalize.c static.c arguments.c module.c \
		effects.c cmfortran.c craft.c control.c

LIB_HEADERS=	ri-util-local.h
LIB_OBJECTS=	Psc.o Ppolynome.o Pvecteur.o Pcontrainte.o matrice.o eval.o \
		size.o ordering.o loop.o fortran90.o prettyprint.o \
		attachment_pretty_print.o constant.o\
		util.o  bound_generation.o entity.o variable.o statement.o \
		expression.o type.o normalize.o static.o arguments.o module.o \
		effects.o cmfortran.o craft.o control.o


