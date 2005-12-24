# $Id$

TARGET	= movements

LIB_CFILES = \
	movement_computation.c  bound_generation.c operation.c\
	complex_bound_generation.c constraint_distribution.c \
	make_loop_body.c build_sc_machine.c build_sc_tile.c sc_add_variable.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
