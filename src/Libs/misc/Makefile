# $Id$

TARGET	= misc

LIB_CFILES =	debug.c \
		file.c \
		message.c \
		dotting.c \
		args.c \
		system.c \
		exception.c \
		mem_spy.c \
		perf_spy.c \
		malloc_debug.c 

INC_TARGET =	$(TARGET).h

LIB_TARGET      = lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif
PROJECT = pips
include $(ROOT)/makes/main.mk
