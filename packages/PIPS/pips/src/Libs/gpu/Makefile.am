TARGET	= gpu
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libgpu.la
libgpu_la_SOURCES=	\
		gpu-ify.c\
		loop_nest_annotate.c \
		xml_output.c


include $(srcdir)/../pipslibs_includes.mk
