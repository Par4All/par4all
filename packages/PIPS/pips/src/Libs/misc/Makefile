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

TARGET	= misc

LIB_CFILES =	debug.c \
		file.c \
		message.c \
		dotting.c \
		args.c \
		system.c \
		exception.c \
		perf_spy.c \
		malloc_debug.c

INC_TARGET =	$(TARGET).h

LIB_TARGET      = lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
here	:=	$(shell pwd)
CONFIG_DIR=$(here)/$(ETC.d)
CPPFLAGS+='-DCONFIG_DIR="$(CONFIG_DIR)"'
