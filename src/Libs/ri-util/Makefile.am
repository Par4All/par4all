TARGET	= ri-util
nodist_include_HEADERS=$(TARGET).h operator.h
dist_noinst_HEADERS=operator.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libri-util.la
libri_util_la_SOURCES=	contrainte_to_text.c \
		eval.c \
		size.c \
		pragma.c \
		extension.c \
		util.c \
		ordering.c \
		prettyprint.c \
		declarations.c \
		unstructured.c \
		attachment_pretty_print.c \
		loop.c \
		fortran90.c \
		constant.c \
		bound_generation.c \
		entity.c \
		variable.c \
		instruction.c \
		statement.c \
		expression.c \
		type.c \
		normalize.c \
		static.c \
		arguments.c \
		module.c \
		effects.c \
		cmfortran.c \
		craft.c \
		control.c \
		hpfc.c \
		misc_paf_utils.c \
		convex_hull.c \
		clean.c \
		clone_statement.c \
		replace.c

include $(srcdir)/../pipslibs_includes.mk
