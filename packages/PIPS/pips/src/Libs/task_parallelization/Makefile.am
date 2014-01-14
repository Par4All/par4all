TARGET=task_parallelization
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libtask_parallelization.la
libtask_parallelization_la_SOURCES= \
		SDG.c \
		cost_model.c\
		BDSC.c \
		HBDSC.c \
		instrumentation.c \
		spire_generation.c \
		communication_generation.c \
		openmp_generation.c \
		mpi_generation.c
include $(srcdir)/../pipslibs_includes.mk
