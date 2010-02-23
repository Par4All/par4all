TARGET	= hwac
nodist_include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=freia_spoc.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libhwac.la
libhwac_la_SOURCES=\
	hardware_accelerator.c \
	freia.c \
	freia_spoc.c \
	dag-utils.c \
	freia-utils.c


include $(srcdir)/../pipslibs_includes.mk
