#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1997/09/08 15:14:56 $, 

LIB_CFILES=	abs.c \
		divide.c \
		exp.c \
		modulo.c \
		pgcd.c \
		ppcm.c \
		io.c

OTHER_HEADERS=	assert.h boolean.h arithmetic_errors.h 

LIB_HEADERS=	arithmetique-local.h \
		$(OTHER_HEADERS) errors.c

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 

INSTALL_FILE=	$(OTHER_HEADERS)
 
INSTALL_INC+=	$(OTHER_HEADERS)

recompile: .quick-install

# end of $RCSfile: config.makefile,v $
#
