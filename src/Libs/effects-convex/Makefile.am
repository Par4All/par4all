TARGET	= effects-convex
nodist_include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libeffects-convex.la
libeffects_convex_la_SOURCES=interface.c \
		translation.c \
		methods.c \
		compose.c \
		unary_operators.c \
		utils.c \
		debug.c \
		interprocedural.c \
		prettyprint.c \
		old_combine.c \
		old_projection.c

include $(srcdir)/../pipslibs_includes.mk
