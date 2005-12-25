#
# $Id$
# 
# Source, header and object files used to build the library.
# $Id$

TARGET	= effects-generic

LIB_CFILES = \
	proper_effects_engine.c \
	rw_effects_engine.c \
	in_effects_engine.c \
	out_effects_engine.c \
	mappings.c \
	unary_operators.c \
	binary_operators.c \
	utils.c \
	prettyprint.c \
	intrinsics.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
