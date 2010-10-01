# Use the PIPS infrastructure to generate a Doxygen documentation

# Some own Par4All stuff:
include $(P4A_ROOT)/doc/libs/par4all.mk

DOXYGEN_PUBLISH_LOCATION_DIR=$(INSTALL_MACHINE):$(INSTALL_MACHINE_DOC_DIR)

# common stuff, recycled directly from PIPS:
ROOT    = $(P4A_ROOT)/packages/PIPS/pips
PROJECT	= par4all

# Some pure local Doxygen parameters:
define DOXYGEN_PARAMETERS
INPUT_ENCODING         = UTF-8
INPUT                  = p4a p4a_coffee p4a_pack p4a_process p4a_setup $(wildcard *.py)
endef

PROJECT_NAME           = Par4All_simple_tools


## Where we want the documentation to be published:
#PUBLISH_LOCATION := /tmp/$(PROJECT_NAME)
make_destination_dir: default_destination_dir

include $(ROOT)/makes/doxygen.mk
