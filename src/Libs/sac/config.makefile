#
# $Id$
# 

LIB_CFILES = \
	simdizer.c varwidth.c unroll.c main.c

LIB_HEADERS = sac-local.h

LIB_OBJECTS = $(LIB_CFILES:.c=.o)
