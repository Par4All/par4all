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
# The following macros define the value of commands that are used to
# compile source code.
#
# you can add your own options behind pips default values.
# 
# example: CFLAGS= $(PIPS_CFLAGS) -DSYSTEM=BSD4.2
#
CFLAGS		+= -DLARGE_FONTS
CPPFLAGS	+= -I$(OPENWINHOME)/include
LDFLAGS		+= -L$(OPENWINHOME)/lib
#
# Source, header and object files used to build the target
#
LIB_CFILES=	pips.c
LIB_HEADERS=	pips-local.h 
LIB_OBJECTS=	$(LIB_CFILES:.c=.o)
#
TARGET_LIBS= 	$(PIPS_LIBS)
