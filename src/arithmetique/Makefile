# $Id$

TARGET	= arithmetique

LIB_CFILES =	abs.c \
		divide.c \
		exp.c \
		modulo.c \
		pgcd.c \
		ppcm.c \
		io.c \
		errors.c

OTHER_HEADERS =	assert.h \
		boolean.h \
		arithmetic_errors.h \
		linear.h

INC_TARGET	= $(TARGET).h

LIB_TARGET	= lib$(TARGET).a

INSTALL_INC	+= $(OTHER_HEADERS)

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
include $(ROOT)/makes/main.mk
