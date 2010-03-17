TARGET	= conversion
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libconversion.la
libconversion_la_SOURCES=change_of_Pbase.c \
		code_change_of_basis.c \
		loop_iteration_domaine_to_sc.c \
		look_for_nested_loops.c \
		system_to_code.c

include $(srcdir)/../pipslibs_includes.mk
