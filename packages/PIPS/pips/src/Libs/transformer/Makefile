# $Id$
#
# Copyright 1989-2014 MINES ParisTech
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

# Transformer Package
#
# Francois Irigoin, 21 April 1990
#
# This library is used for intra and interprocedural semantic analysis.
# It is entirely based on linear systems, although they are not necessarily
# fully used.
#
# The subpackages are:
#
#  - value, which build an implicit level of value on top of PIPS entities;
#    this was set up so as not to create to many value entities; the drawback
#    is that datastructure build on value (see transformer and precondition)
#    CANNOT be interpreted before module_to_value_mappings() is called;
#    when the mappings are no longer useful, call free_value_mappings()
#
#  - arguments: set of values (i.e. set of entities) can be created and 
#    operated upon; these functionalities are likely to be provided at many
#    different places within PIPS! -> moved in ri-util (FI, 2 March 94)
#
#  - transformer, which are the basic object; linear equalities and
#    inequalities are used to relate scalar integer variable VALUES;
#    they can be interpreted as mapping of preconditions onto postconditions
#    for any statement (and they are called transformers) or as mapping of
#    initial module values onto statement preconditions (and they are called
#    preconditions); basic operators on transformers include projection,
#    convex hull computation and fix-point; transformer can be printed when
#    value are properly mapped (see value above)
#
# A predicate is a polyhedron build on integer scalar variable values.
#
# A predicate transformer is a polyhedron build on pre- and post- values
# of integer scalar variables.
#
# Modified variables, which have two different values (pre- and post-),
# are given by the field "arguments". Post-values are referenced by
# the corresponding entities. Pre-values are referenced by special
# entities.
#
# Intermediate values are used to combine transformers. They are also
# referenced by special entities. See "value.c" for pre- and intermediate
# value management.
#
# Operators on transformers should be consistent as far as sharing
# is concerned. If an operator allocates space for its result it should
# do so consistently even if the result is equal to one of its arguments.
# If it is declared as having a side effect and as storing its result in
# one of its arguments, it should be always done in the same way. Usually
# the FIRST transformer argument is modified.
#
# Francois Irigoin, December 1989, April 1990
#

TARGET	= transformer

LIB_CFILES =	basic.c convex_hull.c fix_point.c io.c \
		transformer.c value.c prettyprint.c \
		transformer_list.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
