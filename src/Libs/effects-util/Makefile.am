TARGET=effects-util
BUILT_SOURCES=$(TARGET).h
include_HEADERS=$(TARGET).h

noinst_LTLIBRARIES=libeffects-util.la
libeffects_util_la_SOURCES=effects.c \
	anywhere_abstract_locations.c \
        abstract_location.c \
        points_to.c \
        cell_relations.c \
        pointer_values.c \
        prettyprint.c \
        type.c \
        conflicts.c

include $(srcdir)/../pipslibs_includes.mk
include $(top_srcdir)/makes/cproto.mk
