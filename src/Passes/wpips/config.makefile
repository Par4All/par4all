# $RCSfile: config.makefile,v $ ($Date: 1996/07/02 16:28:40 $, ) 
# version $Revision$
# got on %D%, %T%
# [%P%].
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
CFLAGS+=	$(PIPS_CFLAGS) -DLARGE_FONTS
CPPFLAGS+=	$(WPIPS_ADDED_CPPFLAGS)
LDFLAGS+=	$(WPIPS_ADDED_LDFLAGS)
#
# Source, header and object files used to build the target
# xv_icons.c
LIB_CFILES=	emacs.c \
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
#
# Rajoute le directory icons :
#
LIB_HEADERS=	xv_sizes.h wpips-local.h wpips-labels.h pips.icon icons
#
#
LIB_OBJECTS=	$(TARGET_CFILES:.c=.o)
#
# List of libraries used to build the target
#
TARGET_LIBS= $(PIPS_LIBS) $(WPIPS_ADDED_LIBS)
#
# that is all
#
