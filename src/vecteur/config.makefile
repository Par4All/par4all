#
# $Id$
#

LIB_CFILES=	alloc.c \
		binaires.c \
		io.c \
		reductions.c \
		unaires.c \
		base.c \
		error.c \
		private.c \
		scalaires.c \
		variable.c \
		hashpointer.c

LIB_HEADERS=	vecteur-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 

# end of $RCSfile: config.makefile,v $
#
