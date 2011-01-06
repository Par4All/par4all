TARGET	= sac
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h patterns.h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libsac.la
libsac_la_SOURCES=\
	patterns.y lexer.l \
	if_conversion_compact.c if_conversion_init.c simdizer.c varwidth.c codegen.c unroll.c operatorid.c \
	treematch.c atomizer.c vectransform.c reductions.c \
	singleass.c deatomizer.c if_conversion.c simd_memory_packing.c

AM_YFLAGS=-d -p patterns_yy
AM_LFLAGS=-p patterns_yy
dist_sysconf_DATA=patterns.def
CLEANFILES=lexer.c patterns.[ch]

include $(srcdir)/../pipslibs_includes.mk
