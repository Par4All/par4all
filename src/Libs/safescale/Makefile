# $Id: Makefile 12356 2007-07-03 09:56:46Z Loïc Plassart & Ronan Keryell $

TARGET	= safescale

LIB_CFILES =	analysis.c \
                identification.c \
                externalization.c

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
