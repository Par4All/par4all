#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/08 19:36:53 $, 

LIB_CFILES=	alloc.c \
		binaires.c \
		error.c \
		io.c \
		listes.c \
		normalize.c \
		predicats.c \
		unaires.c

LIB_HEADERS=	contrainte-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 
 
# end of $RCSfile: config.makefile,v $
#
