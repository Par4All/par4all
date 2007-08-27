# Some pure Doxygen parameters:
define DOXYGEN_PARAMETERS
# The files to document:
INPUT                  = ../../src
endef

PROJECT_NAME           = Linear
# We want to have other Doxygen documentation (the PIPS one) to refer this doc, so ask Doxygen to put cross-information in this file:
GENERATE_TAGFILE_NAME       = Linear.doxygen-tag


# TODO: make DEFAULT_DOXYGEN_CONFIG independent of this Makefile
DEFAULT_DOXYGEN_CONFIG = ../../makes/share/doxygen/Doxyfile

include ../../makes/doxygen.mk

# If needed:
#make_destination_dir :
#	mkdir -p $(PUBLISH_LOCATION)
