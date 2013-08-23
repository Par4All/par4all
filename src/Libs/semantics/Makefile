# $Id$
#
# Copyright 1989-2010 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

# Semantics Analysis
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
# Source, header and object files used to build the library.
# Do not include the main program source file.

TARGET	= semantics

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
		loop.c \
		ri_to_total_preconditions.c \
		ri_to_transformer_lists.c \
		path_transformer.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
