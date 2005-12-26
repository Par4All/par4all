# $Id$

TARGET	= vecteur

LIB_CFILES=	alloc.c \
		binaires.c \
		io.c \
		reductions.c \
		unaires.c \
		base.c \
		error.c \
		private.c \
		scalaires.c \
		variable.c \
		hashpointer.c

LIB_TARGET	= lib$(TARGET).a

INC_TARGET	= $(TARGET).h

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
PROJECT = linear
include $(ROOT)/makes/main.mk
