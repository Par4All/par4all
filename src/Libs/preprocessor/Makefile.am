TARGET	= preprocessor
nodist_include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libpreprocessor.la
libpreprocessor_la_SOURCES=\
	splitc.y \
	lexer.l\
	initializer.c \
	source_file.c \
	split_file.c \
	csplit_file.c \
	module.c

AM_YFLAGS=-d -p splitc_
AM_LFLAGS=-p splitc_

include $(srcdir)/../pipslibs_includes.mk
