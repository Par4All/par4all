# $RCSfile: config.makefile,v $ (version $Revision$)
# ($Date: 1995/11/14 18:22:00 $, )
#
# Source, header and object files used to build the library.
# Do not include the main program source file.
#
LIB_CFILES=	Psc.c \
		Ppolynome.c \
		Pvecteur.c \
		Pcontrainte.c \
		matrice.c \
		eval.c \
		size.c \
		util.c \
		ordering.c \
		prettyprint.c \
		attachment_pretty_print.c \
		loop.c \
		fortran90.c \
		constant.c \
		bound_generation.c \
		entity.c \
		variable.c \
		statement.c \
		expression.c \
		type.c \
		normalize.c \
		static.c \
		arguments.c \
		module.c \
		effects.c \
		cmfortran.c \
		craft.c \
		control.c \
		hpfc.c

LIB_HEADERS=	ri-util-local.h operator.h
LIB_OBJECTS=	$(LIB_CFILES:.c=.o)

# that is all
#
