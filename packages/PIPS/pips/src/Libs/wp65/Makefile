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

#   	WP65: PUMA ESPRIT PROJECT 2701
#   	------------------------------
#
# Corinne Ancourt, Francois Irigoin, Lei Zhou	     17 October 1991
#
####### The source files directly involved in wp65 are:

TARGET	= wp65

LIB_CFILES =	code.c tiling.c variable.c instruction_to_wp65_code.c \
		wp65.c basis.c \
		find_iteration_domain.c model.c references.c communications.c 

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
