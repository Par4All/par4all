TARGET	= step
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libstep.la
dist_noinst_HEADERS=STEP_name.h STEP_RT_bootstrap.h STEP_RT_intrinsic.h defines-local.h
libstep_la_SOURCES=	comment2pragma.lex \
			step_omp.lex \
			directives.c \
			analyse.c \
			compile_RT.c \
			compile.c


AM_CPPFLAGS=-DPIPS_RUNTIME_DIR=\"$(datadir)\"

include $(srcdir)/../pipslibs_includes.mk
