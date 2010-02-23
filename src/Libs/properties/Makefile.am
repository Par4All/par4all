TARGET	= properties
nodist_include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libproperties.la
libproperties_la_SOURCES=properties.l
AM_LFLAGS=-p prop_
include $(srcdir)/../pipslibs_includes.mk
