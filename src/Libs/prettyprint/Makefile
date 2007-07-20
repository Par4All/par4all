# $Id$

TARGET	= prettyprint

LIB_CFILES = \
	print.c \
	print_code_as_a_graph.c \
	cprettyprinter.c \
	xml_prettyprinter.c \
	claire_prettyprinter.c \
	symbol_table.c


INC_TARGET = $(TARGET).h

LIB_TARGET = lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
