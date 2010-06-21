TARGET	= hyperplane
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libhyperplane.la
libhyperplane_la_SOURCES= 	hyperplane.c  \
	 	hyperplane_direction.c \
		scanning_base.c \
	 	global_parallelization.c  \
		code_generation.c \
		tiling.c \
		symbolic_tiling.c \
		unimodular.c


include $(srcdir)/../pipslibs_includes.mk
