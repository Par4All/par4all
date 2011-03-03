TARGET	= pocc-interface
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libpocc-interface.la
libPoCC_Interface_la_SOURCES= \
		pocc_prettyprinter.c\
		scop_outliner.c		


include $(srcdir)/../pipslibs_includes.mk