# $Id$

CPPFLAGS += -Dsparc 
LDFLAGS	+= -fast 

include $(ROOT)/makes/DEFAULT.mk

CC	= acc -temp=$(pips_home)/tmp
CFLAGS	= -g -fast -Xc

# The SC3 acc compiler forget to define the sparc flag;
LD	= $(CC) -bsdmalloc

RANLIB	= granlib

# lex broken for properties...
LEX	= flex

FFLAGS	= -O -g -U -u -C 
