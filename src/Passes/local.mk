# $Id$

#debug_output := $(shell echo local.mk  > /dev/tty)

clean: NO_INCLUDES=1
export NO_INCLUDES

# We need to read the $(ARCH).mk to know if we need to compile wpips or not:
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../..
endif

MAKE.d	= $(ROOT)/makes
include $(MAKE.d)/arch.mk
include $(MAKE.d)/$(ARCH).mk

# check for gtk2 here, to know whether to forward to gpips or not
-include $(MAKE.d)/config.mk
include $(MAKE.d)/has_gtk2.mk

FWD_DIRS	= pips tpips

# Skip compiling WPips if not required:

ifndef PIPS_NO_WPIPS
	FWD_DIRS	+= wpips
endif

ifndef PIPS_NO_GPIPS
	FWD_DIRS	+= gpips
endif

# compile pypips only if required
ifdef PIPS_PYPIPS
	FWD_DIRS	+= pypips
endif

# after its dependencies
FWD_DIRS	+= fpips
