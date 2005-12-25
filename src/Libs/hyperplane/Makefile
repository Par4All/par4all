# $Id$

TARGET	= hyperplane

LIB_CFILES = 	hyperplane.c  \
	 	hyperplane_direction.c \
		scanning_base.c \
	 	global_parallelization.c  \
		code_generation.c \
		tiling.c \
		unimodular.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
