#
# $Id$
#
# configuration for FPIPS.
#

TARGET	= fpips

LIB_CFILES	= fpips.c

OTHER_CFILES	= main_fpips.c

CPPOPT =	$(FPIPS_CPPFLAGS)

LIB_TARGET	= lib$(TARGET).a

BIN_TARGET	= $(TARGET)

INSTALL_LIB = 	$(ARCH)/main_fpips.o

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff
$(ARCH)/fpips: $(ARCH)/$(FPIPS_MAIN)

clean: local-clean

local-clean:
	$(RM) $(ARCH)/pips $(ARCH)/tpips $(ARCH)/wpips $(ARCH)/fpips

ln: local-clean
	#
	# links
	#
	ln $(ARCH)/fpips $(ARCH)/gpips
	ln $(ARCH)/fpips $(ARCH)/pips
	ln $(ARCH)/fpips $(ARCH)/tpips
	ln $(ARCH)/fpips $(ARCH)/wpips
