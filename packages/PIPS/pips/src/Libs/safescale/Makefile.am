TARGET	= safescale
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libsafescale.la
libsafescale_la_SOURCES=	analysis.c \
                identification.c \
                externalization.c


include $(srcdir)/../pipslibs_includes.mk
