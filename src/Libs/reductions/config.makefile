#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/06/14 14:15:52 $, 
# Source, header and object files used to build the library.

LIB_CFILES=	stub.c 

# should have LIB_LISPFILES
# they sould be put somewhere for execution
LIB_HEADERS=	eval.cl \
		match.cl \
		reduc.cl \
		top.cl \
		init.cl \
		patterns.cl \
		simplify.cl \
		util.cl

LIB_OBJECTS=	$(LIB_CFILES:.c=.o) 	

#
#
