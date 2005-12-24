# $Id$

TARGET = expressions

LIB_CFILES =	partial_eval.c \
		simple_atomize.c \
		forward_substitution.c \
		optimize.c \
		sequence_gcm_cse.c \
		instruction_selection.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
