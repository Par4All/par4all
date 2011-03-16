TARGET	= instrumentation
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libinstrumentation.la
libinstrumentation_la_SOURCES=\
	array_bound_check_bottom_up.c \
	array_bound_check_top_down.c \
	array_bound_check_instrumentation.c \
	array_bound_check_interprocedural.c \
	alias_propagation.c \
	alias_check.c \
	used_before_set.c \
	control_counters.c

include $(srcdir)/../pipslibs_includes.mk
