TARGET = static_controlize
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libstatic_controlize.la
libstatic_controlize_la_SOURCES=\
	prettyprint.c static_controlize.c utils.c 

include $(srcdir)/../pipslibs_includes.mk
