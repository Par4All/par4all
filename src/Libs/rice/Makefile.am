TARGET	= rice
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=local.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=librice.la
librice_la_SOURCES=rice.c codegen.c scc.c icm.c
include $(srcdir)/../pipslibs_includes.mk
