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

TARGET	= wpips

CPPOPT =	-DLARGE_FONTS $(WPIPS_CPPFLAGS) \
		-DOWTOOLKIT_WARNING_DISABLED

INC_CFILES=	emacs.c \
		directory_menu.c \
		wpips.c \
		xv_compile.c \
		xv_log.c \
		xv_edit2.c \
		xv_frames.c \
		xv_help.c \
		xv_icons.c \
		xv_mchoose.c \
		xv_props.c \
		xv_query.c \
		xv_quit.c \
		xv_schoose2.c \
		xv_select.c \
		xv_status.c \
		xv_transform.c \
		xv_utils.c \
		vararg.c 

OTHER_CFILES =	main_wpips.c

LIB_HEADERS=	xv_sizes.h wpips-local.h wpips-labels.h pips.icon icons

LIB_CFILES	= $(OTHER_CFILES) $(INC_CFILES)

INC_TARGET	= $(TARGET).h

LIB_TARGET	= lib$(TARGET).a

BIN_TARGET	= $(TARGET)

INSTALL_EXE	= wpips-changedir

INSTALL_LIB = 	main_wpips.o

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(ARCH)/wpips: $(ARCH)/$(WPIPS_MAIN)
