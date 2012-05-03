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

TARGET	= alias-classes

LIB_CFILES = 	alias_pairs.c \
		alias_lists.c \
		alias_classes.c \
		points_to_set.c	\
                points_to_prettyprint.c \
		points_to_analysis_general_algorithm.c \
		points_to_init_analysis.c \
		points_to_interprocedural_analysis.c \
                constant-path-utils.c \
		prettyprint.c \
		statement.c \
		expression.c \
		passes.c \
	        sinks.c \
		unstructured.c \
		variable.c \
		interprocedural.c \
		dereferencing.c


INC_TARGET =	$(TARGET).h

LIB_TARGET =	lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
