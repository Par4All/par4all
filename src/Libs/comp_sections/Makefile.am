TARGET	= comp_sections
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=all.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libcomp_sections.la
libcomp_sections_la_SOURCES=	comp_sections.c propagate.c ss.c operators.c myintrinsics.c \
		dbase.c prettyprint.c

include $(srcdir)/../pipslibs_includes.mk
