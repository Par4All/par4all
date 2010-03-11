TARGET	= icfg
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libicfg.la
libicfg_la_SOURCES=toposort.c icfg_scan.c print.c util.c
include $(srcdir)/../pipslibs_includes.mk
