# $Id$

TARGET = phrase

LIB_CFILES =    phrase_distributor.c \
		phrase_distributor_control_code.c \
		phrase_distributor_communication.c \
		distribution_context.c \
		fsm_generation.c \
		print_code_smalltalk.c \
		fsm_merge_states.c \
		fsm_split_state.c \
		full_fsm_generation.c \
		loop_spaghettify.c \
		whileloop_spaghettify.c \
		forloop_spaghettify.c \
		test_spaghettify.c \
		spaghettify.c \
		full_spaghettify.c \
		fsm_tools.c \
		phrase_tools.c

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
