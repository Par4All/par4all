#
# config.makefile for PHRASE library
# 

LIB_CFILES = phrase_distributor.c fsm_generation.c print_code_smalltalk.c fsm_merge_states.c fsm_split_state.c full_fsm_generation.c spaghettify.c fsm_tools.c

LIB_HEADERS = phrase-local.h smalltalk-defs.h fsm_generation.h

LIB_OBJECTS = $(DERIVED_CFILES:.c=.o) $(LIB_CFILES:.c=.o)


