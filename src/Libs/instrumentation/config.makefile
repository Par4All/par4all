#
# $Id$
#
# $Log: config.makefile,v $
# Revision 1.1  2000/03/16 09:09:40  coelho
# Initial revision
#
#

LIB_CFILES 	= 

LIB_HEADERS	= instrumentation-local.h

LIB_OBJECTS	= $(LIB_CFILES:%.c=%.o)
