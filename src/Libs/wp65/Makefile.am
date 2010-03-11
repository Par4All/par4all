TARGET	= wp65
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libwp65.la
libwp65_la_SOURCES=code.c tiling.c variable.c instruction_to_wp65_code.c \
		wp65.c basis.c \
		find_iteration_domain.c model.c references.c communications.c 


include $(srcdir)/../pipslibs_includes.mk
