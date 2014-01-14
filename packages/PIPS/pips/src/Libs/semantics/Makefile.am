TARGET	= semantics
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libsemantics.la
libsemantics_la_SOURCES=	misc.c \
		ri_to_transformers.c \
		interprocedural.c\
		ri_to_preconditions.c \
		mappings.c \
		dbm_interface.c \
		prettyprint.c \
		postcondition.c \
		utils.c \
		initial.c \
		unstructured.c \
		expression.c \
		loop.c \
		ri_to_total_preconditions.c \
		ri_to_transformer_lists.c \
		path_transformer.c

include $(srcdir)/../pipslibs_includes.mk
