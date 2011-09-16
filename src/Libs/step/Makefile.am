# $Id$

# what about: comment2pragma.h step_omp.h
Fh	= STEP_name.h STEP_RT_bootstrap.h STEP_RT_intrinsic.h
Fc	= comment2pragma.c step_omp.c

TARGET	= step
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h $(Fh) $(Fc)
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libstep.la
dist_noinst_HEADERS=$(Fh) defines-local.h
libstep_la_SOURCES=	comment2pragma.lex \
			step_omp.lex \
			directives.c \
			analyse.c \
			compile_RT.c \
			compile.c \
			$(Fc)


AM_CPPFLAGS=-DPIPS_RUNTIME_DIR=\"$(datadir)\"

include stepstuff.mk
include $(srcdir)/../pipslibs_includes.mk
