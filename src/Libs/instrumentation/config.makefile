#
# $Id$
#
# $Log: config.makefile,v $
# Revision 1.3  2000/06/07 15:22:15  nguyen
# Add new array bounds check version, based on regions
#
# Revision 1.2  2000/03/16 09:18:07  coelho
# array bound check moved here.
#
# Revision 1.1  2000/03/16 09:09:40  coelho
# Initial revision
#
#

LIB_CFILES 	= \
	array_bound_check.c \
	region_based_array_bound_check.c

LIB_HEADERS	= instrumentation-local.h

LIB_OBJECTS	= $(LIB_CFILES:%.c=%.o)
