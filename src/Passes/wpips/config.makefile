#	%A% ($Date: 1995/07/19 15:19:09 $, ) version $Revision$, got on %D%, %T% [%P%].
#	Copyright (c) École des Mines de Paris Proprietary.
#
# Copyright (C) Ecole des Mines De Paris
#               Centre d'Automatique et Informatique
#               Section Informatique
#
# This file is part of PIPS
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY.  No author or distributor accepts responsibility to anyone for
# the consequences of using it or for whether it serves any particular
# purpose or works at all, unless he says so in writing.
#
#
#
# this is a sample configuration file for pips-makemake.  
#
# You should use this file to create a new pass. If you want to create a
# new library, use the configuration file 'config.makefile.lib'.
#
# This file should be copied in the directory where you want the
# makefile to be built. Its name must be 'config.makefile'.
#
# Update the copy to specify flags values and source code file
# names. Then execute the command 'pips-makemake -p'.
#
#
#
# The following macros define the value of commands that are used to
# compile source code.
#
# you can add your own options behind pips default values.
# 
# example: CFLAGS= $(PIPS_CFLAGS) -DSYSTEM=BSD4.2
#
AR=		$(PIPS_AR)
ARFLAGS=	$(PIPS_ARFLAGS)
CC=		$(PIPS_CC)
CFLAGS=		$(PIPS_CFLAGS) -DLARGE_FONTS
CPPFLAGS=	$(PIPS_CPPFLAGS) -I$(OPENWINHOME)/include -Iicons
LD=		$(PIPS_LD) -static
LDFLAGS=	$(PIPS_LDFLAGS) -L$(OPENWINHOME)/lib
LEX=		$(PIPS_LEX)
LFLAGS=		$(PIPS_LFLAGS)
LINT=		$(PIPS_LINT)
LINTFLAGS=	$(PIPS_LINTFLAGS)
YACC=		$(PIPS_YACC)
YFLAGS=		$(PIPS_YFLAGS)
#
# The following macros define your pass.
#
# Name of the target
TARGET= 	wpips
#
	# Source, header and object files used to build the target
# xv_icons.c
TARGET_CFILES=	emacs.c wpips.c \
		xv_log.c \
		xv_edit2.c xv_frames.c xv_help.c xv_icons.c xv_mchoose.c \
		xv_props.c xv_query.c xv_quit.c \
		xv_schoose2.c xv_select.c xv_status.c \
		xv_transform.c xv_utils.c vararg.c 
# Rajoute le directory icons :
TARGET_HEADERS=	xv_sizes.h wpips-local.h wpips-labels.h pips.icon icons
# xv_icons.o
TARGET_OBJECTS=	emacs.o wpips.o \
		xv_log.o \
		xv_edit2.o xv_frames.o xv_help.o xv_icons.o xv_mchoose.o \
		xv_props.o xv_query.o xv_quit.o \
		xv_schoose2.o xv_select.o xv_status.o \
		xv_transform.o xv_utils.o vararg.o
#
# List of libraries used to build the target
TARGET_LIBS= $(PIPS_LIBS) -lxview -lolgx -lX11

#$(TARGET): $(LIBDIR)/*.a
