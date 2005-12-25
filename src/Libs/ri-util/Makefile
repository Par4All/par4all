# $Id$

TARGET	= ri-util

LIB_CFILES =	contrainte_to_text.c \
		eval.c \
		size.c \
		util.c \
		ordering.c \
		prettyprint.c \
		declarations.c \
		unstructured.c \
		attachment_pretty_print.c \
		loop.c \
		fortran90.c \
		constant.c \
		bound_generation.c \
		entity.c \
		variable.c \
		statement.c \
		expression.c \
		type.c \
		normalize.c \
		static.c \
		arguments.c \
		module.c \
		effects.c \
		cmfortran.c \
		craft.c \
		control.c \
		hpfc.c \
		misc_paf_utils.c \
		convex_hull.c

INC_TARGET =	$(TARGET).h

LIB_TARGET =	lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
