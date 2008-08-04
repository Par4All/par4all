# $Id$

TARGET	= hpfc

LIB_CFILES=	directives.c \
		dynamic.c \
		hpfc.c \
		debug-util.c \
		hpfc-util.c \
		declarations.c \
		compiler-util.c \
		compiler.c \
		compile.c \
		run-time.c \
		generate.c \
		local-ri-util.c \
		inits.c \
		o-analysis.c \
		align-checker.c \
		messages.c \
		message-utils.c \
		build-system.c \
		io-util.c \
		io-compile.c \
		generate-util.c \
		remapping.c \
		host_node_entities.c \
		special_cases.c \
		lattice_extraction.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

VALIDATE_TARGET	= Hpfc

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
