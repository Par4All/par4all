TARGET	= properties
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libproperties.la
libproperties_la_SOURCES=properties.l
AM_LFLAGS=-p prop_
CLEANFILES=properties.c
include $(srcdir)/../pipslibs_includes.mk
