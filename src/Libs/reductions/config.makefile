#
# $RCSfile: config.makefile,v $ (version $Revision$)
# $Date: 1996/06/14 16:23:48 $, 
# Source, header and object files used to build the library.

LIB_CFILES=	stub.c \
		reductions.c \
		utils.c \
		transformation.c \
		prettyprint.c

# should have LIB_LISPFILES
# they sould be put somewhere for execution
LIB_HEADERS=	reductions-local.h \
		local-header.h \
		eval.cl \
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
