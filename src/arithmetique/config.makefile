#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/08 18:56:36 $, 

LIB_CFILES=	abs.c \
		divide.c \
		exp.c \
		modulo.c \
		pgcd.c \
		ppcm.c

LIB_HEADERS=	arithmetique-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 
 
# end of $RCSfile: config.makefile,v $
#
