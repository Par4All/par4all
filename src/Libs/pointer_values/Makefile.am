TARGET = pointer_values
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libpointer_values.la
libeffects_simple_la_SOURCES=\
	pointer_values_analyses.c \
	prettyprint.c

include $(srcdir)/../pipslibs_includes.mk
