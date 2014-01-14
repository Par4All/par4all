TARGET=regions_to_loops

include_HEADERS=$(TARGET).h

dist_noinst_HEADERS=$(TARGET)-local.h

BUILT_SOURCES=$(TARGET).h 

include $(top_srcdir)/makes/cproto.mk 

noinst_LTLIBRARIES=libregions_to_loops.la 

libregions_to_loops_la_SOURCES= regions_to_loops.c



include $(srcdir)/../pipslibs_includes.mk
