# --------------------------------------------------------------------
#
# Hpfc $RCSfile: config.makefile,v $, Fabien COELHO
#
# $RCSfile: config.makefile,v $ ($Date: 1995/04/19 10:47:37 $, ) version $Revision$,
# got on %D%, %T%
# $Id$
#
# The following macros define the value of commands that are used to
# compile source code.
#
# you can add your own options behind pips default values.
# 
# example: CFLAGS= $(PIPS_CFLAGS) -DSYSTEM=BSD4.2
#
AR=		$(PIPS_AR)
ARFLAGS=	$(PIPS_ARFLAGS)
CC=		$(PIPS_CC)
CFLAGS=		$(PIPS_CFLAGS)
CPPFLAGS=	$(PIPS_CPPFLAGS)
LD=		$(PIPS_LD)
LDFLAGS=	$(PIPS_LDFLAGS)
LEX=		$(PIPS_LEX)
LFLAGS=		$(PIPS_LFLAGS)
LINT=		$(PIPS_LINT)
LINTFLAGS=	$(PIPS_LINTFLAGS)
YACC=		$(PIPS_YACC)
YFLAGS=		$(PIPS_YFLAGS)

#
# The following macros define your library.
#
# Name of the library without the .a suffix.
TARGET= 	hpfc
#
# Name of the main program to test or use the library
MAIN=		main
# (obsolete)
#
# Source, header and object files used to build the library.
# Do not include the main program source file.

LIB_CFILES=	directives.c \
		dynamic.c \
		remapping.c \
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
		generate-io.c \
		run-time-functions.c \
		host_node_entities.c \
		hpf_objects.c \
		subarray_shift.c

LIB_HEADERS=	warning.h \
		hpfc-local.h \
		defines-local.h \
		access_description.h 

# headers made by some rule (except $INC_TARGET)
LIB_OBJECTS= $(LIB_CFILES:.c=.o) 

#
# this dependence is false to avoid regenerating often the .h and
# thus recompiling everything.

$(TARGET).h: hpfc-local.h

sccs_close:
	@echo "closing the sccs session"
	@echo "Description of changes:"
	@read comments
	sccs delget -y"$$comments" `sccs tell -u`

#
# --------------------------------------------------------------------
