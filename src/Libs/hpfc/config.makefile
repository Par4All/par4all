#
# $RCSfile: config.makefile,v $ version $Revision$,
# ($Date: 1995/08/04 13:14:26 $, )
#
# Source, header and object files used to build the library.
# Do not include the main program source file.

LIB_CFILES=	directives.c \
		dynamic.c \
		hpfc.c \
		debug-util.c \
		hpfc-util.c \
		hpf_declarations.c \
		new_declarations.c \
		compiler-util.c \
		compiler.c \
		compile.c \
		run-time.c \
		generate.c \
		statement.c \
		norm-code.c \
		local-ri-util.c \
		inits.c \
		o-analysis.c \
		align-checker.c \
		messages.c \
		overlap.c \
		guard.c \
		ranges.c \
		message-utils.c \
		reduction.c \
		build-system.c \
		only-io.c \
		io-compile.c \
		generate-util.c \
		generate-io.c \
		remapping.c \
		run-time-functions.c \
		host_node_entities.c \
		hpf_objects.c \
		subarray_shift.c

LIB_HEADERS=	warning.h \
		hpfc-local.h \
		defines-local.h \
		access_description.h 

# should be automatically derived ?
# headers made by some rule (except $INC_TARGET)

LIB_OBJECTS= $(LIB_CFILES:.c=.o) 

# that is all
#
