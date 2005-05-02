# $Id$

TARGET	= sommet

LIB_CFILES =	sommet.c \
		sommets.c

LIB_TARGET = lib$(TARGET).a

INC_TARGET = $(TARGET).h

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
include $(ROOT)/makes/main.mk
