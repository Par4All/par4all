TARGET	= flint
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=local.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libflint.la
libflint_la_SOURCES=flint.c \
		flint_walk.c \
		flint_check.c \
		flint_utils.c \
		uninitialized_variables.c \
		type_checker.c

include $(srcdir)/../pipslibs_includes.mk
