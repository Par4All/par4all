# $Id$

TARGET	= contrainte

LIB_CFILES=	alloc.c \
		binaires.c \
		error.c \
		io.c \
		listes.c \
		normalize.c \
		predicats.c \
		unaires.c

LIB_TARGET	= lib$(TARGET).a

INC_TARGET	= $(TARGET).h

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
include $(ROOT)/makes/main.mk
