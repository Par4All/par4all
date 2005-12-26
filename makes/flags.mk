#
# $Id$
#
# macros related to any compilation.
#

CPPFLAGS += \
	-I. \
	-I$(NEWGEN_ROOT)/include \
	-I$(LINEAR_ROOT)/include \
	-I$(PIPS_ROOT)/include \
	-I$(EXTERN_ROOT)/include 

LDFLAGS += \
	-L./$(ARCH) \
	-L$(PIPS_ROOT)/lib/$(ARCH) \
	-L$(NEWGEN_ROOT)/lib/$(ARCH) \
	-L$(LINEAR_ROOT)/lib/$(ARCH) \
	-L$(EXTERN_ROOT)/lib/$(ARCH)
