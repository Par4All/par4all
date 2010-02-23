TARGET=alias-classes
BUILT_SOURCES=$(TARGET).h
nodist_include_HEADERS=$(TARGET).h

noinst_LTLIBRARIES=libalias-classes.la
libalias_classes_la_SOURCES=alias_pairs.c \
		alias_lists.c \
		alias_classes.c \
		prettyprint.c
include $(srcdir)/../pipslibs_includes.mk
include $(top_srcdir)/makes/cproto.mk
