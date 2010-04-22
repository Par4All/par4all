noinst_LTLIBRARIES=libgenspec.la
libgenspec_la_SOURCES=genspec.y genspec_lex.l
AM_YFLAGS=-d -p genspec_
AM_LFLAGS=-p genspec_
AM_CPPFLAGS=-I$(srcdir)/.. -I..

BUILT_SOURCES=genspec.h
CLEANFILES=genspec*
