# Semantics Analysis
#
# SCCS Stuff:
# $RCSfile: config.makefile,v $ ($Date: 1994/03/24 18:29:16 $, ) version $Revision$
# got on %D%, %T%
# $Id$
#
# Francois Irigoin, 17 April 1990
#
# This library supports intra and interprocedural semantic analysis.
# It is entirely based on linear systems, although they are not necessarily
# fully used. Different level are available:
#
# The subpackages are:
#
#  - ri_to_transformers contains functions which compute transformers for
#    a module and all its statement; the only visible function should be
#    module_to_transformer(); it assumes that the module code is loaded
#    and that a control graph was computed and that effects are available;
#    it does not call itself recursively when encountering a programmer
#    CALL; it is NOT recursive; the prettyprinter library contains a
#    function to print a module with its transformers;
#
#  - ri_to_preconditions contains functions which compute preconditions
#    a module and all its statements; the only visible function should be
#    module_to_preconditions(); same prerequisites as module_to_transformers()
#    plus, of course, transformer availability; the prettyprinter library
#    contains a function to print a module with its preconditions;
#    the routine module_to_preconditions() and module_to_transformer() are
#    separated to cope with interprocedural analysis: transformers are
#    computed bottom up and preconditions top down on the call graph
#    (no Fortran recursivity is expected)
#
# A main program, main.c, provides an easy way (i.e. dbxtool compatible) to try
# semantic analysis and to test the library
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
TARGET= 	semantics
#
# Name of the main program to test or use the library
MAIN=		main
#
# debug malloc signale un bug dans l'enveloppe convexe...
# il devrait etre cherche directement dans la directory de test de
# Linear/polyedre.dir
#	/usr/lib/debug/malloc.o
#
# List of lint libraries to be used to typecheck the library
LINEARDEV = $(LINEARDIR)/Development
LINT_LIBS = $(LINEARDEV)/arithmetique.dir/llib-larithmetique.ln \
	$(LINEARDEV)/vecteur.dir/llib-lvecteur.ln  \
	$(LINEARDEV)/contrainte.dir/llib-lcontrainte.ln \
	$(LINEARDEV)/sc.dir/llib-lsc.ln \
	$(LINEARDEV)/ray_dte.dir/llib-lray_dte.ln \
	$(LINEARDEV)/sommet.dir/llib-lsommet.ln \
	$(LINEARDEV)/sg.dir/llib-lsg.ln \
	$(LINEARDEV)/polyedre.dir/llib-lpolyedre.ln
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	misc.c ri_to_transformers.c interprocedural.c\
		ri_to_preconditions.c mappings.c dbm_interface.c \
		prettyprint.c postcondition.c
LIB_HEADERS=	semantics-local.h
LIB_OBJECTS=	$(LIB_CFILES:.c=.o)

default: all
