# $Id$

clean: NO_INCLUDES=1
export NO_INCLUDES

# We need to read the $(ARCH).mk to know if we need to compile jpips or not:
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../..
endif
MAKE.d	= $(ROOT)/makes
include $(MAKE.d)/arch.mk
include $(MAKE.d)/$(ARCH).mk

#debug_output := $(shell echo no_jpips.mk  > /dev/tty)

# not needed? stats stf jpips make
FWD_DIRS =	drivers dev env graph misc validation hpfc simple_tools

# Compile epips only if needed:
ifndef PIPS_NO_EPIPS
	FWD_DIRS += epips
endif

# Compile Jpips only if needed:
ifndef PIPS_NO_JPIPS
	FWD_DIRS += jpips
endif
