#
# The following macros define the value of commands that are used to
# compile source code.
#
# you can add your own options behind pips default values.
# 
# example: CFLAGS= $(PIPS_CFLAGS) -DSYSTEM=BSD4.2
#
LDFLAGS+=	-L/usr/5lib
#
# The following macros define your library.
#
# List of other libraries used to build the test main program

MAIN_LIBS= 	$(PIPS_LIBS)

LIB_CFILES=	array_dfg.c \
		adg_utils.c \
		adg_prettyprint.c \
		adg_predicate.c \
		adg_graph.c \
		adg_summary.c

LIB_HEADERS=	array_dfg-local.h local.h

LIB_OBJECTS=	$(LIB_CFILES:.c=.o)
