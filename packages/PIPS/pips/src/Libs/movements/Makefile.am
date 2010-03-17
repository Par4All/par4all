TARGET	= movements
include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libmovements.la
libmovements_la_SOURCES=\
	movement_computation.c  bound_generation.c operation.c\
	complex_bound_generation.c constraint_distribution.c \
	make_loop_body.c build_sc_machine.c build_sc_tile.c sc_add_variable.c
include $(srcdir)/../pipslibs_includes.mk
