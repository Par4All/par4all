# Copyright (C) Ecole des Mines de Paris
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
#
# CPPFLAGS	+= -DLARGE_FONTS -DPIPS_ARCH=\"$(ARCH)\" 
#
# Source, header and object files used to build the target
#

CPPFLAGS	+= -DUTC_DATE='"$(UTC_DATE)"'

LIB_CFILES	= pips.c
LIB_HEADERS	= pips-local.h 
LIB_OBJECTS	= $(LIB_CFILES:.c=.o)
LIB_MAIN	= main_pips.c
#
TARGET_LIBS	= $(PIPS_LIBS)
#
