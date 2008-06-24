# $Id$
#
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
# Source, header and object files used to build the target
#

TARGET	= pips

LIB_CFILES	= pips.c revisions.c
OTHER_CFILES	= main_pips.c

INC_TARGET =    $(TARGET).h
LIB_TARGET =    lib$(TARGET).a
BIN_TARGET =	$(TARGET)

INSTALL_LIB = 	$(ARCH)/main_pips.o

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT	= pips
include $(ROOT)/makes/main.mk

# version...
include $(MAKE.d)/version.mk
$(ARCH)/revisions.o: .revisions_h

# local stuff
$(ARCH)/pips: $(ARCH)/$(PIPS_MAIN)
