#
# $Id$
# 

LIB_CFILES = \
	database.c \
	lowlevel.c \
	workspace.c \
	externals.c \
	misc.c \
	obsolete.c 

LIB_HEADERS = \
	pipsdbm-local.h \
	private.h \
	methods.h

LIB_OBJECTS = $(LIB_CFILES:.c=.o)
