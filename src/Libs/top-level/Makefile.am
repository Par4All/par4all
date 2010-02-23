TARGET	= top-level
nodist_include_HEADERS=$(TARGET).h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libtop-level.la
libtop_level_la_SOURCES=	dbm.c \
		help.c \
		logging.c \
		view.c \
		signal.c
include $(srcdir)/../pipslibs_includes.mk
