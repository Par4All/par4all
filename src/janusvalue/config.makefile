LIB_CFILES=	isolve.c \
		r1.c

LIB_HEADERS=	janusvalue-local.h \
		iabrev.h \
		iproblem.h \
		rproblem.h

LIB_OBJECTS= $(LIB_CFILES:.c=.o)


