#
# $Id$
#

LIB_CFILES      = stats.c \
		  guard_elimination.c \
		tiling_sequence.c
LIB_HEADERS     = statistics-local.h
LIB_OBJECTS     = $(LIB_CFILES:%.c=%.o)
