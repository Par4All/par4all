#
# $RCSfile: config.makefile,v $ version $Revision$,
# ($Date: 1995/10/10 16:32:19 $, )
#
# Source, header and object files used to build the library.
# Do not include the main program source file.

LIB_CFILES=	directives.c \
		dynamic.c \
		hpfc.c \
		debug-util.c \
		hpfc-util.c \
		declarations.c \
		compiler-util.c \
		compiler.c \
		compile.c \
		run-time.c \
		generate.c \
		local-ri-util.c \
		inits.c \
		o-analysis.c \
		align-checker.c \
		messages.c \
		message-utils.c \
		build-system.c \
		io-util.c \
		io-compile.c \
		generate-util.c \
		remapping.c \
		host_node_entities.c \
		special_cases.c

LIB_HEADERS=	warning.h \
		hpfc-local.h \
		defines-local.h \
		access_description.h 

# should be automatically derived ?

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 

# that is all
#
