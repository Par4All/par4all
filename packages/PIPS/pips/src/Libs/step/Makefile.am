TARGET	= step
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libstep.la
dist_noinst_HEADERS=STEP_name.h STEP_RT_bootstrap.h STEP_RT_intrinsic.h defines-local.h
libstep_la_SOURCES=step.c \
		atomizer.c \
		outlining.c \
		directives.c \
		directive_parallel.c \
		directive_do.c \
		directive_section.c \
		directive_reduction.c \
		directive_master.c \
		directive_barrier.c \
		directive_private.c \
		analyse.c \
		compile.c \
		compile_do.c \
		compile_reduction.c \
		compile_master.c \
		compile_barrier.c \
		compile_private.c

include $(srcdir)/../pipslibs_includes.mk
