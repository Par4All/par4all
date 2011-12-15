TARGET	= effects-generic
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libeffects-generic.la
libeffects_generic_la_SOURCES=\
	proper_effects_engine.c \
	rw_effects_engine.c \
	in_effects_engine.c \
	out_effects_engine.c \
	mappings.c \
	unary_operators.c \
	binary_operators.c \
	utils.c \
        methods.c \
	prettyprint.c \
	intrinsics.c \
	interprocedural.c \
	eval.c \
	liveness_analysis_engine.c


include $(srcdir)/../pipslibs_includes.mk
