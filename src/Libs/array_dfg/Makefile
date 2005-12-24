# $Id$

TARGET	= array_dfg

LIB_CFILES =	array_dfg.c \
		adg_utils.c \
		adg_prettyprint.c \
		adg_predicate.c \
		adg_graph.c \
		adg_summary.c

INC_TARGET =	$(TARGET).h

LIB_TARGET =	lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
