TARGET=taskify
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libtaskify.la
libtaskify_la_SOURCES=taskify.c starpu_pragma_generation.c
include $(srcdir)/../pipslibs_includes.mk
