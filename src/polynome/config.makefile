#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 16:46:03 $, 

LIB_CFILES=	pnome-alloc.c \
		pnome-bin.c \
		pnome-error.c \
		pnome-io.c \
		pnome-private.c \
		pnome-reduc.c \
		pnome-scal.c \
		pnome-unaires.c 

LIB_HEADERS=	polynome-local.h


LIB_OBJECTS= $(LIB_CFILES:.c=.o) 

# end of $RCSfile: config.makefile,v $
#
