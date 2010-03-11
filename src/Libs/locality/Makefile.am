TARGET	= locality
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=liblocality.la
liblocality_la_SOURCES=\
	stats.c \
	tiling_sequence.c \
	guard_elimination.c


include $(srcdir)/../pipslibs_includes.mk
