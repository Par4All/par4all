# This is config makefile for C syntax, using Flex and Bison 
$Id$
$Log: config.makefile,v $
Revision 1.1  2003/06/24 07:19:02  nguyen
Initial revision


ifeq ($(CC),gcc)
CFLAGS=	-g -Wall -ansi
else
CFLAGS=	-g
endif

YFLAGS+=-d

LIB_CFILES=	c_parser.c \
		util.c

LIB_HEADERS=	cyacc.y \
		clex.l \
	 	c_syntax-local.h

DERIVED_HEADERS= cyacc.h
DERIVED_CFILES= cyaccer.c clexer.c

LIB_OBJECTS=	$(DERIVED_CFILES:.c=.o)  $(LIB_CFILES:.c=.o) 

$(TARGET).h: $(DERIVED_HEADERS) $(DERIVED_CFILES) 

cyaccer.c cyacc.h: cyacc.y
	$(PARSE) cyacc.y
	sed 's/YY/C_/g;s/yy/c_/g' y.tab.c > cyaccer.c
	sed 's/YY/C_/g;s/yy/c_/g' y.tab.h > cyacc.h
	$(RM) y.tab.c y.tab.h

clexer.c: clex.l cyacc.h
	$(SCAN) clex.l | \
	sed '/^FILE \*yyin/s/=[^,;]*//g;s/YY/C_/g;s/yy/c_/g' > $@


