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


# Generate the Doxygen version of the PIPS sources.
#
# Since we need to compute the caller graphs for NewGen and Linear, we
# must generate Doxygen documentation for NewGen, Linear and PIPS in the
# same run.

ROOT    = ../../..

# Assume that that PIPS has already been compiled to have the newgen
# and linear objects already generated.

# Try to figure out where newgen and linear are if not specified. Useful
# when bootstrapping PIPS:
# PIPS_ROOT   = $(ROOT)
NEWGEN_ROOT ?= $(ROOT)/../newgen
LINEAR_ROOT ?= $(ROOT)/../linear

# Some hack to get the NewGen source file list.

# Newgen
NEWGEN_SOURCE_DIR=$(NEWGEN_ROOT)/src/genC

# To get the file lists:
include $(NEWGEN_SOURCE_DIR)/standard_defines.mk

DOXYGEN_INPUT                  = $(addprefix $(NEWGEN_SOURCE_DIR)/, $(LIB_CFILES) $(LIB_HEADERS) $(DERIVED_HEADERS))

# Linear
LINEAR_SOURCE_DIR=$(LINEAR_ROOT)/src
DOXYGEN_INPUT += $(LINEAR_SOURCE_DIR)


# Some pure local Doxygen parameters:
define DOXYGEN_PARAMETERS
INPUT                  = ../../Libs ../../Passes $(DOXYGEN_INPUT) ../../../include
endef

PROJECT_NAME           = PIPS

## Where we want the documentation to be published:
#PUBLISH_LOCATION := /tmp/$(PROJECT_NAME)

include $(ROOT)/makes/doxygen.mk

# The hack is going on...
# Undo some work done by the "include
# $(NEWGEN_SOURCE_DIR)/standard_defines.mk" in order to make the following
# not to think there is really something to compile:
LIB_CFILES=
OTHER_CFILES=
INSTALL_INC=
LIB_TARGET=
BIN_TARGET=
include $(ROOT)/makes/main.mk

# If needed:
#make_destination_dir :
#	mkdir -p $(PUBLISH_LOCATION)
