# $Id$

TARGET	= matrice

LIB_CFILES =	determinant.c \
		hermite.c \
		inversion.c \
		matrice.c \
		matrice_io.c \
		smith.c \
		sous-matrice.c

LIB_TARGET = lib$(TARGET).a

INC_TARGET = $(TARGET).h

################################################################## COMMON STUFF

ifdef LINEAR_ROOT
ROOT    = $(LINEAR_ROOT)
else
ROOT    = ../..
endif
PROJECT = linear
include $(ROOT)/makes/main.mk
