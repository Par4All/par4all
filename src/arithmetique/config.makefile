#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/07/09 19:14:55 $, 

LIB_CFILES=	abs.c \
		divide.c \
		exp.c \
		modulo.c \
		pgcd.c \
		ppcm.c

OTHER_HEADERS=	assert.h boolean.h

LIB_HEADERS=	arithmetique-local.h \
		$(OTHER_HEADERS)

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 

INSTALL_FILE=	$(OTHER_HEADERS)
 
# end of $RCSfile: config.makefile,v $
#
