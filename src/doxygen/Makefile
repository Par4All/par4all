# $Id$	

LINEAR_SOURCE_DIR ?= ../../src

# Some pure Doxygen parameters:
define DOXYGEN_PARAMETERS
# The files to document:
INPUT                  = $(LINEAR_SOURCE_DIR)
endef

PROJECT_NAME           = Linear

# We want to have other Doxygen documentation (the PIPS one) to refer this
# doc, so ask Doxygen to put cross-information in this file. Indeed it is
# not used because if we want to build the caller graph we need to build
# the documentation at once.
GENERATE_TAGFILE_NAME       = Linear.doxygen-tag


DEFAULT_DOXYGEN_CONFIG = $(LINEAR_SOURCE_DIR)/../makes/share/doxygen/Doxyfile

include $(LINEAR_SOURCE_DIR)/../makes/doxygen.mk

# If needed:
#make_destination_dir :
#	mkdir -p $(PUBLISH_LOCATION)
