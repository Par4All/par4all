TARGET	= c_syntax
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h cyacc.h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libc_syntax.la
libc_syntax_la_SOURCES=\
		cyacc.y\
		clex.l \
		c_parser.c \
		statement.c \
		util.c \
		return.c

AM_YFLAGS=-d -p c_
AM_LFLAGS=-p c_
CLEANFILES=cyacc.[ch] clex.c

include $(srcdir)/../pipslibs_includes.mk
