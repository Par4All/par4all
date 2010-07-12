TARGET = effects-simple
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libeffects-simple.la
libeffects_simple_la_SOURCES=\
	binary_operators.c \
	interface.c \
	methods.c \
        translation.c \
	interprocedural.c \
	prettyprint.c \
	unary_operators.c \
	filter_proper_effects.c \
        eval.c


include $(srcdir)/../pipslibs_includes.mk
