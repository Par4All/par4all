# $Id$

TARGET	= newgen

INC_CFILES = \
		newgen.c \
		Pvecteur.c \
		Ppolynome.c \
		Psc.c

INC_TARGET =	$(TARGET).h

LIB_TARGET =	lib$(TARGET).a

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

# where to find newgen generated files...
inc_dir = $(PIPS_ROOT)/include

NEWGEN_CFILES = $(notdir $(wildcard $(inc_dir)/*.c))

LIB_CFILES =	$(INC_CFILES) $(NEWGEN_CFILES)

CPPOPT	= -I$(inc_dir)

%.c: 
	# $(inc_dir)/$@
	ln -s $(inc_dir)/$@

PROJECT = pips
include $(ROOT)/makes/main.mk

clean: local-clean

local-clean:; $(RM) $(NEWGEN_CFILES)
