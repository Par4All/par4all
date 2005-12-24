# $Id$

TARGET	= effects-convex

LIB_CFILES =	interface.c \
		translation.c \
		methods.c \
		compose.c \
		unary_operators.c \
		utils.c \
		debug.c \
		interprocedural.c \
		prettyprint.c \
		old_combine.c \
		old_projection.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

include $(ROOT)/makes/main.mk
