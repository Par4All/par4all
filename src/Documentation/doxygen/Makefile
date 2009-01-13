# Generate the Doxygen version of the PIPS sources.

# Since we need to compute the caller graphs for NewGen and Linear, we
# must generate Doxygen documentation for NewGen, Linear and PIPS in the
# same run.

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

# Assume that that PIPS has already been compiled to have the newgen
# and linear objects already generated.

# Try to figure out where newgen and linear are if not specified. Useful
# when bootstrapping PIPS:
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
