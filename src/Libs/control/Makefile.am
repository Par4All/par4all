TARGET	= control
nodist_include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libcontrol.la
libcontrol_la_SOURCES=\
	clean_up_sequences.c \
	control.c \
	graph.c \
	hierarchize.c \
	module.c \
	reorder.c \
	unspaghettify.c \
	cfg.c \
	unreachable.c \
	bourdoncle.c \
	for_to_do_or_while_loops.c \
	dowhile_to_while.c

include $(srcdir)/../pipslibs_includes.mk
