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

TARGET	= transformations

LIB_CFILES =	\
		loop_unroll.c \
		prettyprintcray.c \
		strip_mine.c \
		interactive_loop_transformation.c \
		loop_interchange.c \
		loop_normalize.c \
		interchange.c \
		target.c \
		nest_parallelization.c \
		coarse_grain_parallelization.c \
		dead_code_elimination.c \
		trivial_test_elimination.c \
		privatize.c \
		array_privatization.c \
		standardize_structure.c \
		simplify_control.c \
		declarations.c \
		clone.c \
		transformation_test.c \
		freeze_variables.c \
		array_resizing_bottom_up.c \
		array_resizing_top_down.c \
		array_resizing_statistic.c \
		partial_redundancy_elimination.c \
		variable_expansion.c \
		internalize_parallel_code.c \
		index_set_splitting.c \
		for_loop_recovering.c \
		generate_pragma.c \
		manage_pragma.c \
		scalarization.c \
		flatten_code.c \
		induction_substitution.c \
		loop_fusion.c \
		set_return_type.c \
		cast_at_call_sites.c \
		manage_parallel_loops.c \
		registers.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
