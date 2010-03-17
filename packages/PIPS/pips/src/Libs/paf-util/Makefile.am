TARGET	= paf-util
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libpaf-util.la
libpaf_util_la_SOURCES=adg_read_paf.c bdt_read_paf.c utils.c print.c

include $(srcdir)/../pipslibs_includes.mk
