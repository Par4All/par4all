TARGET	= atomizer
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=local.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libatomizer.la
libatomizer_la_SOURCES=new_atomizer.c \
		atomizer.c \
		utils.c \
		codegen.c \
		control.c \
		norm_exp.c \
		defs_elim.c


include $(srcdir)/../pipslibs_includes.mk
