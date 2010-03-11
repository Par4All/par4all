TARGET	= prettyprint
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libprettyprint.la
libprettyprint_la_SOURCES= \
	print.c \
	print_code_as_a_graph.c \
	cprettyprinter.c \
	xml_prettyprinter.c \
	claire_prettyprinter.c \
	symbol_table.c \
	html_prettyprinter.c

include $(srcdir)/../pipslibs_includes.mk
