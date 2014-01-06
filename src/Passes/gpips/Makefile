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

TARGET	= gpips

CPPOPT =	$(GPIPS_CPPFLAGS)

# LIB_CFILES: C files that are used in the library
LIB_CFILES=	directory_menu.c \
		gpips.c \
		gtk_compile.c \
		gtk_edit2.c \
		gtk_help.c \
		gtk_log.c \
		gtk_mchoose.c \
		gtk_props.c \
		gtk_query.c \
		gtk_quit.c \
		gtk_schoose2.c \
		gtk_select.c \
		gtk_status.c \
		gtk_transform.c \
		gtk_utils.c \
		gtk_windows.c \
		vararg.c

# non existant :		gtk_frames.c gtk_icons.c

# OTHER_CFILES : sources used to build the BIN_TARGET with the library too
OTHER_CFILES =	main_gpips.c

# LIB_HEADERS does not exist...
#LIB_HEADERS = gpips-local.h gpips-labels.h pips.icon icons

#LIB_CFILES = $(OTHER_CFILES) $(INC_CFILES)
#LIB_CFILES = (INC_CFILES)

INC_TARGET = $(TARGET).h

LIB_TARGET = lib$(TARGET).a

BIN_TARGET = $(TARGET)

###INSTALL_EXE	= gpips-changedir

INSTALL_LIB = main_gpips.o

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(ARCH)/gpips: $(ARCH)/$(GPIPS_MAIN)

