#
# $Id$
#
# Source, header and object files used to build the library.

LIB_CFILES =	stub.c \
		reductions.c \
		utils.c \
		transformation.c \
		prettyprint.c \
		call.c

# should have LIB_LISPFILES
# they sould be put somewhere for execution

LISP_FILES =	eval.cl \
		match.cl \
		reduc.cl \
		top.cl \
		init.cl \
		patterns.cl \
		simplify.cl \
		util.cl

LIB_HEADERS =	reductions-local.h \
		local-header.h \
		$(LISP_FILES)

LIB_OBJECTS =	$(LIB_CFILES:.c=.o) 	

#
#

