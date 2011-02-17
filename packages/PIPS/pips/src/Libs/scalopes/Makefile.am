TARGET	= scalopes
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libscalopes.la
libscalopes_la_SOURCES=scalopify.c sesamify.c scalopragma.c sesam_tasks_processing.c
include $(srcdir)/../pipslibs_includes.mk
