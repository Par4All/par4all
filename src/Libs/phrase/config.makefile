#
# config.makefile for PHRASE library
# 

LIB_CFILES = phrase_distributor.c fsm_generation.c print_code_smalltalk.c

LIB_HEADERS = phrase-local.h

LIB_OBJECTS = $(DERIVED_CFILES:.c=.o) $(LIB_CFILES:.c=.o)


