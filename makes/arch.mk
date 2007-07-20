# $Id$

ifndef MAKE.d
$(error "MAKE.d	macro is not defined")
endif

ifndef ARCH
ifdef PIPS_ARCH
ARCH	= $(PIPS_ARCH)
else
ifdef NEWGEN_ARCH
ARCH	= $(NEWGEN_ARCH)
else
ifdef LINEAR_ARCH
ARCH	= $(LINEAR_ARCH)
else
ARCH	:= $(shell $(MAKE.d)/arch.sh)
endif # LINEAR_ARCH
endif # NEWGEN_ARCH
endif # PIPS_ARCH
endif # ARCH

ifndef ARCH
$(error "ARCH macro is not defined")
endif
