TARGET	= ricedg
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=local.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libricedg.la
libricedg_la_SOURCES=util.c contexts.c testdep_util.c \
		ricedg.c prettyprint.c quick_privatize.c \
		trace.c impact_check.c


include $(srcdir)/../pipslibs_includes.mk
