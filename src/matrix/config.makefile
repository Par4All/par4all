#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 16:25:32 $, 

LIB_CFILES=	alloc.c \
		determinant.c \
		hermite.c \
		inversion.c \
		matrix.c \
		matrix_io.c \
		smith.c \
		sub-matrix.c

LIB_HEADERS=	matrix-local.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 
 
# end of $RCSfile: config.makefile,v $
#
