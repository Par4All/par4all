#
# Hyperplane Method
# -----------------
#
# Yi-qing YANG, May 1990
#
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
LINTFLAGS=	$(PIPS_LINTFLAGS) -habx
YACC=		$(PIPS_YACC)
YFLAGS=		$(PIPS_YFLAGS)
#
# The following macros define your library.
#
# Name of the library without the .a suffix.
# PIPS Project
#
#
TARGET=  hyperplane
#
# Name of the main program to test or use the library
MAIN=	hyperplane_main
#
#
# List of lint libraries to be used to typecheck the library
LINEAR =
LINT_LIBS = $(LINEAR)/arithmetique.dir/llib-larithmetique.ln \
	$(LINEAR)/matrice .dir/llib-lmatrice.ln\
	$(LINEAR)/vecteur.dir/llib-lvecteur.ln  \
	$(LINEAR)/contrainte.dir/llib-lcontrainte.ln \
	$(LINEAR)/sc.dir/llib-lsc.ln \
	$(LINEAR)/ray_dte.dir/llib-lray_dte.ln \
	$(LINEAR)/sommet.dir/llib-lsommet.ln \
	$(LINEAR)/sg.dir/llib-lsg.ln \
	$(LINEAR)/polyedre.dir/llib-lpolyedre.ln

#
# Source, header and object files used to build the library.
LIB_CFILES= 	hyperplane.c  \
	 	hyperplane_direction.c scanning_base.c \
	 	global_parallelization.c  code_generation.c

LIB_HEADERS=	hyperplane-local.h

LIB_OBJECTS=  	hyperplane.o \
	 	hyperplane_direction.o scanning_base.o \
	 	global_parallelization.o code_generation.o



