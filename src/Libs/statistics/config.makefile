#
# $Id$
#

LIB_CFILES       = stats.c\
	          tiling_sequence.c\
		  guard_elimination.c
LIB_HEADERS     = statistics-local.h
LIB_OBJECTS     = $(LIB_CFILES:%.c=%.o)
