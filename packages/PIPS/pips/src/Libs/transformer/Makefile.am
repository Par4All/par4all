TARGET	= transformer
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libtransformer.la
libtransformer_la_SOURCES=	basic.c convex_hull.c fix_point.c io.c \
		transformer.c value.c prettyprint.c transformer_list.c

include $(srcdir)/../pipslibs_includes.mk
