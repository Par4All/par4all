#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/08 19:20:14 $, 

LIB_CFILES=	alloc.c \
		binaires.c \
		io.c \
		reductions.c \
		unaires.c \
		base.c \
		error.c \
		private.c \
		scalaires.c \
		variable.c

LIB_HEADERS=	vecteur-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 
 
# end of $RCSfile: config.makefile,v $
#
