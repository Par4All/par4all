# $Id: developer_guide.tex 12778 2007-08-28 15:10:30Z keryell $

# To get the file lists:
include $(NEWGEN_SOURCE_DIR)/standard_defines.mk

# Some pure local Doxygen parameters:
define DOXYGEN_PARAMETERS
INPUT                  = $(addprefix $(NEWGEN_SOURCE_DIR)/, $(LIB_CFILES) $(LIB_HEADERS) $(DERIVED_HEADERS))
# PIPS use linear and newgen library that are also documented with Doxygen. Use these tags files to get the information and then to link to:
endef

PROJECT_NAME           = libgenC

# Where we want the documentation to be published:
PUBLISH_LOCATION := /tmp/$(PROJECT_NAME)


include $(NEWGEN_SOURCE_DIR)/../../makes/doxygen.mk

# If needed:
make_destination_dir :
	mkdir -p $(PUBLISH_LOCATION)
