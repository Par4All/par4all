# $Id$

ifdef NEWGEN_ROOT
ROOT    = $(NEWGEN_ROOT)
else
ROOT    = ../../../..
endif

# If it is not defined:
NEWGEN_SOURCE_DIR ?= ../../../genC

# To get the file lists:
include $(NEWGEN_SOURCE_DIR)/standard_defines.mk

# Some pure local Doxygen parameters:
define DOXYGEN_PARAMETERS
INPUT                  = $(addprefix $(NEWGEN_SOURCE_DIR)/, $(LIB_CFILES) $(LIB_HEADERS) $(DERIVED_HEADERS) new.c)
endef

PROJECT_NAME           = newC

# Where we want the documentation to be published:
#PUBLISH_LOCATION := /tmp/$(PROJECT_NAME)


include $(ROOT)/makes/main.mk
include $(ROOT)/makes/doxygen.mk

# If needed:
#make_destination_dir :
#	mkdir -p $(PUBLISH_LOCATION)
