#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 13:43:32 $, 

LIB_CFILES=	determinant.c \
		hermite.c \
		inversion.c \
		matrice.c \
		matrice_io.c \
		smith.c \
		sous-matrice.c

LIB_HEADERS=	matrice-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 
 
# end of $RCSfile: config.makefile,v $
#
