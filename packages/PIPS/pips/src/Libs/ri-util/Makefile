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

TARGET	= ri-util

LIB_CFILES =	contrainte_to_text.c \
		eval.c \
		size.c \
		pragma.c \
		extension.c \
		util.c \
		ordering.c \
		prettyprint.c \
		declarations.c \
		unstructured.c \
		attachment_pretty_print.c \
		loop.c \
		fortran90.c \
		constant.c \
		bound_generation.c \
		entity.c \
		variable.c \
		instruction.c \
		statement.c \
		expression.c \
		type.c \
		normalize.c \
		static.c \
		arguments.c \
		module.c \
		cmfortran.c \
		craft.c \
		control.c \
		hpfc.c \
		misc_paf_utils.c \
		convex_hull.c \
		clean.c \
		clone_statement.c \
		allocatable.c \
		parameter.c \
		replace.c

INC_TARGET =	$(TARGET).h

LIB_TARGET =	lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
