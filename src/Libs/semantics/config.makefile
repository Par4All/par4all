# Semantics Analysis
#
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
# This initial distribution of C functions has been modified a lot to keep
# the C file lengths reasonnable. The restructuring has not be completed
# in July 2001.
#
# A main program, main.c, provides an easy way (i.e. dbxtool compatible) to try
# semantic analysis and to test the library. it is now obsolete
#
# $Log: config.makefile,v $
# Revision 1.8  2001/07/19 18:25:41  irigoin
# New files added: expression.c and loop.c. Previously, unstructured.c had
# been separated. The restructuring is not complete. Loop stuff is still in
# ri_to_transformers.c and ri_to_preconditions.c. And so is interprocedural
# stuff.
#
#
# Source, header and object files used to build the library.
# Do not include the main program source file.

LIB_CFILES=	misc.c \
		ri_to_transformers.c \
		interprocedural.c\
		ri_to_preconditions.c \
		mappings.c \
		dbm_interface.c \
		prettyprint.c \
		postcondition.c \
		utils.c \
		initial.c \
		unstructured.c \
		expression.c \
		loop.c

LIB_HEADERS=	semantics-local.h

LIB_OBJECTS=	$(LIB_CFILES:.c=.o)

