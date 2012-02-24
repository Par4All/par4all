TARGET	= hwac
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=freia_spoc.h freia.h freia_terapix.h freia_opencl.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libhwac.la
libhwac_la_SOURCES=\
        hardware_accelerator.c \
        freia.c \
        freia_spoc.c \
        freia_terapix.c \
	freia_opencl.c \
	freia_aipo.c \
        dag-utils.c \
        freia-utils.c \
	freia_transformations.c

include $(srcdir)/../pipslibs_includes.mk
