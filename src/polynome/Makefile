# $Id$

TARGET	= polynome

LIB_CFILES=	pnome-alloc.c \
		pnome-bin.c \
		pnome-error.c \
		pnome-io.c \
		pnome-private.c \
		pnome-reduc.c \
		pnome-scal.c \
		pnome-unaires.c 

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
