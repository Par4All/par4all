# $Id$

TARGET	= atomizer

LIB_CFILES =	new_atomizer.c \
		atomizer.c \
		utils.c \
		codegen.c \
		control.c \
		norm_exp.c \
		defs_elim.c

INC_TARGET =	$(TARGET).h

LIB_TARGET =	lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
