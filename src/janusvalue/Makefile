# $Id$

TARGET	= janusvalue

LIB_CFILES =	isolve.c \
		r1.c

OTHER_HEADERS = \
		iabrev.h \
		iproblem.h \
		rproblem.h

LIB_TARGET	= lib$(TARGET).a

INC_TARGET	= $(TARGET).h

INSTALL_INC	+= $(OTHER_HEADERS)

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
PROJECT = linear
include $(ROOT)/makes/main.mk
