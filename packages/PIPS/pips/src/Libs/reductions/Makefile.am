TARGET	= reductions
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=local-header.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libreductions.la
libreductions_la_SOURCES=atomic.c \
             stub.c \
						 reductions.c \
						 utils.c \
						 transformation.c \
						 pragma.c \
						 prettyprint.c \
						 call.c \
						 propagation.c

include $(srcdir)/../pipslibs_includes.mk
