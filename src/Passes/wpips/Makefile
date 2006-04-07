# $Id$
#
# Copyright (c) École des Mines de Paris Proprietary.
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

INSTALL_LIB = 	$(ARCH)/main_wpips.o

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(ARCH)/wpips: $(ARCH)/$(WPIPS_MAIN)
