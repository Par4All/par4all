#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 16:23:19 $, 

LIB_CFILES=	alloc.c \
		determinant.c \
		hermite.c \
		inversion.c \
		matrix.c \
		matrix_io.c \
		smith.c \
		sous-matrix.c

LIB_HEADERS=	matrix-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 
 
# end of $RCSfile: config.makefile,v $
#
