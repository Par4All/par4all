# $Id$ 

TARGET	= conversion

LIB_CFILES= 	change_of_Pbase.c \
		code_change_of_basis.c \
		loop_iteration_domaine_to_sc.c \
		look_for_nested_loops.c \
		system_to_code.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
