#
#
#   	WP65: PUMA ESPRIT PROJECT 2701
#   	------------------------------
#
# Corinne Ancourt, Francois Irigoin, Lei Zhou	     17 October 1991
#
####### The source files directly involved in wp65 are:
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
LOCAL_FLAGS= -habx

# The following macros define your library.

# Name of the library without the .a suffix.
TARGET= 	wp65

# Name of the main program to test or use the library
MAIN=		main

# Generated from 'order_libraries' 07-09-90


# Source, header and object files used to build the library.
# Do not include the main program source file.
# Do include the .c, .h and .o extensions.

LIB_CFILES=	code.c tiling.c variable.c instruction_to_wp65_code.c wp65.c basis.c \
		find_iteration_domain.c model.c references.c communications.c 
LIB_HEADERS=	wp65-local.h
LIB_OBJECTS=	code.o tiling.o variable.o wp65.o instruction_to_wp65_code.o \
	references.o communications.o basis.o find_iteration_domain.o model.o

foobar: all

Lint:  
	$(LINT) $(PIPS_LINTFLAGS) $(LOCAL_FLAGS) $(PIPS_CPPFLAGS) $(LIB_CFILES) ../conversion/loop_iteration_domaine_to_sc.c| sed '/possible pointer alignment/d;/gen_alloc/d'
### End of config.makefile
