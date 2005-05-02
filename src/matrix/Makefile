# $Id$

TARGET = matrix

LIB_CFILES =	alloc.c \
		determinant.c \
		hermite.c \
		inversion.c \
		matrix.c \
		matrix_io.c \
		smith.c \
		sub-matrix.c

LIB_TARGET = lib$(TARGET).a

INC_TARGET = $(TARGET).h

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
include $(ROOT)/makes/main.mk
