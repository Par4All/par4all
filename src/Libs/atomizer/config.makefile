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
TARGET= 	atomizer
#
# Name of the main program to test or use the library
MAIN=		main
#
# List of other libraries used to build the test main program
MAIN_LIBS=	-lloop_normalize \
		-lcontrol \
		-llink \
		-lsdfi \
		-leffects \
		-lprettyprint \
		-lsyntax \
		-lpipsdbm \
		-lnormalize \
		-lri-util \
		-ltext-util \
		-lmisc \
		-lproperties \
		-lgenC \
                -lsc \
		-lcontrainte \
		-lvecteur \
		-larithmetique \

#		/usr/lib/debug/malloc.o 

# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	new_atomizer.c atomizer.c utils.c variables.c codegen.c control.c norm_exp.c defs_elim.c
LIB_HEADERS=	atomizer-local.h
LIB_OBJECTS=	new_atomizer.o atomizer.o utils.o variables.o codegen.o control.o norm_exp.o defs_elim.o
