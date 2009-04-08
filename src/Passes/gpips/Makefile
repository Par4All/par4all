# $Id: Makefile 12447 2006-04-07 22:07:36Z keryell $
#
# Copyright (c) �cole des Mines de Paris Proprietary.
#
# Copyright (C) Ecole des Mines De Paris
#		Centre de Recherche en Informatique
#
# This file is part of PIPS
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY.  No author or distributor accepts responsibility to anyone for
# the consequences of using it or for whether it serves any particular
# purpose or works at all, unless he says so in writing.
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

INSTALL_LIB = $(ARCH)/main_gpips.o
ifdef AAA

endif

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(ARCH)/gpips: $(ARCH)/$(GPIPS_MAIN)

