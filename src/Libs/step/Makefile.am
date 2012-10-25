TARGET	= step
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h   STEP_RT_bootstrap.h STEP_RT_intrinsic.h STEP_name.h step_bison_parser.h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libstep.la
dist_noinst_HEADERS=defines-local.h step_api.h STEP_name_variable.h step_common.h
libstep_la_SOURCES=	comment2pragma.l \
			step_lexer.l \
			step_bison_parser.y \
			directives.c \
			analyse.c \
			compile_RT.c \
			compile_mpi.c \
			compile_regions.c \
			compile.c \
			parser.c \
			install.c

AM_LFLAGS=-P step_ -p `basename $< .l`
AM_YFLAGS=-d -p step_
AM_CPPFLAGS=-DPIPS_RUNTIME_DIR=\"$(datadir)\"
CLEANFILES=step_api.tmp STEP_RT_bootstrap.h STEP_RT_intrinsic.h STEP_name.h step_lexer.c comment2pragma.c step_bison_parser.[ch]

include $(srcdir)/../pipslibs_includes.mk
include stepstuff.mk

