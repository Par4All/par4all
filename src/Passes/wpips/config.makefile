# $RCSfile: config.makefile,v $ ($Date: 1995/10/10 15:22:35 $, ) 
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
# AR=		$(PIPS_AR)
# ARFLAGS=	$(PIPS_ARFLAGS)
# CC=		$(PIPS_CC)
CFLAGS=		$(PIPS_CFLAGS) -DLARGE_FONTS
CPPFLAGS=	$(PIPS_CPPFLAGS) $(WPIPS_ADDED_CPPFLAGS)
# LD=		$(PIPS_LD) 
LDFLAGS=	$(PIPS_LDFLAGS) $(WPIPS_ADDED_LDFLAGS)
# LEX=		$(PIPS_LEX)
# LFLAGS=	$(PIPS_LFLAGS)
# LINT=		$(PIPS_LINT)
# LINTFLAGS=	$(PIPS_LINTFLAGS)
# YACC=		$(PIPS_YACC)
# YFLAGS=	$(PIPS_YFLAGS)
#
# The following macros define your pass.
#
# TARGET=	wpips
#
# Source, header and object files used to build the target
# xv_icons.c
TARGET_CFILES=	emacs.c \
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
TARGET_HEADERS=	xv_sizes.h wpips-local.h wpips-labels.h wpips_transform_menu_layout.h pips.icon icons
#
#
TARGET_OBJECTS=	$(TARGET_CFILES:.c=.o)
#
# List of libraries used to build the target
#
TARGET_LIBS= $(PIPS_LIBS) $(WPIPS_ADDED_LIBS)
#
# that is all
#
