TARGET = expressions
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libexpressions.la
libexpressions_la_SOURCES=partial_eval.c \
		simple_atomize.c \
		forward_substitution.c \
		optimize.c \
		sequence_gcm_cse.c \
		instruction_selection.c \
		expression_substitution.c \
		simplify.c \
		rename_operator.c


include $(srcdir)/../pipslibs_includes.mk
