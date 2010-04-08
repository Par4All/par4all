# Use the PIPS infrastructure to generate a Doxygen documentation

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

# Some pure local Doxygen parameters:
define DOXYGEN_PARAMETERS
INPUT                  = p4a_accel.h
endef

PROJECT_NAME           = Par4All_C_to_CUDA


## Where we want the documentation to be published:
#PUBLISH_LOCATION := /tmp/$(PROJECT_NAME)

include $(ROOT)/makes/doxygen.mk
