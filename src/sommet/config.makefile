#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 10:27:36 $, 

LIB_CFILES=	sommet.c \
		sommets.c

LIB_HEADERS=	sommet-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 
 
# end of $RCSfile: config.makefile,v $
#
