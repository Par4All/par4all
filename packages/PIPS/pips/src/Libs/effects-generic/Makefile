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
 
# Source, header and object files used to build the library.

TARGET	= effects-generic

LIB_CFILES = \
	proper_effects_engine.c \
	rw_effects_engine.c \
	in_effects_engine.c \
	out_effects_engine.c \
	mappings.c \
	unary_operators.c \
	binary_operators.c \
	utils.c \
        methods.c \
	prettyprint.c \
	intrinsics.c \
	interprocedural.c \
	eval.c \
	liveness_analysis_engine.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
