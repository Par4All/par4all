TARGET	= accel-util
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libaccel-util.la
libaccel_util_la_SOURCES=group_constants.c \
		   	isolate_statement.c \
			statement_insertion.c\
			loop_expansion.c  \
			outlining.c\
			array_to_pointer.c  \
			inlining.c\
 			kernels.c\
      ikernels.c\
			terapixify.c\
			constraints.c\
			computation_intensity.c

include $(srcdir)/../pipslibs_includes.mk
