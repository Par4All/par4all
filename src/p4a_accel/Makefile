# Use the PIPS infrastructure to generate a Doxygen documentation

# Some own Par4All stuff:
include $(P4A_ROOT)/doc/libs/par4all.mk

DOXYGEN_PUBLISH_LOCATION_DIR=$(INSTALL_MACHINE)/$(INSTALL_MACHINE_DOC_DIR)

# common stuff, recycled directly from PIPS:
ROOT    = $(P4A_ROOT)/packages/PIPS/pips
PROJECT	= pips

# Some pure local Doxygen parameters:
define DOXYGEN_PARAMETERS
INPUT_ENCODING         = UTF-8
INPUT                  = p4a_accel.h p4a_accel.c p4a_accel-CUDA.h p4a_accel-OpenMP.h p4a_stubs.c p4a_stubs.f
endef

PROJECT_NAME           = Par4All_Accel_runtime


## Where we want the documentation to be published:
#PUBLISH_LOCATION := /tmp/$(PROJECT_NAME)

include $(ROOT)/makes/doxygen.mk
