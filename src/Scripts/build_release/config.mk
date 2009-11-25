# this file is a sample version of pips configuration
# it should be well suited to most pips users
# feel free to modify it to match your need
# you can also override its content by setting variable directly from the `make' command 
#

##
# where to install everything
# INSTALL_DIR must be set by the user !

# configuration files
ETC.d=$(INSTALL_DIR)/etc

# headers
INC.d=$(INSTALL_DIR)/include

# binaries
BIN.d=$(INSTALL_DIR)/bin

# libraries
LIB.d=$(INSTALL_DIR)/lib

# manpages
MAN.d=$(INSTALL_DIR)/share/man

# other locations should be ok

# inform the linker we installed everything in $(INSTALL_DIR)
NEWGEN_ROOT=$(INSTALL_DIR)
LINEAR_ROOT=$(INSTALL_DIR)

# do not build tags
PIPS_NO_TAGS=1

# do not build epips, jpips, gpips nor wpips
PIPS_NO_EPIPS=1
PIPS_NO_JPIPS=1
PIPS_NO_WPIPS=1
PIPS_NO_GPIPS=1

# do not build dynamic libraries
# WITH_DYNAMIC_LIBRARIES=1

