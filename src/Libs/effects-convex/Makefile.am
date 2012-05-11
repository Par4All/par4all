TARGET	= effects-convex
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libeffects-convex.la
libeffects_convex_la_SOURCES=interface.c \
		translation.c \
		methods.c \
		compose.c \
		unary_operators.c \
		sc_enumerate.c \
		utils.c \
		debug.c \
		interprocedural.c \
		prettyprint.c \
		binary_operators.c \
		old_projection.c \
                eval.c

include $(srcdir)/../pipslibs_includes.mk
