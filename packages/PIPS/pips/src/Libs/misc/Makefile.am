TARGET=misc
noinst_LTLIBRARIES=libmisc.la
libmisc_la_SOURCES=debug.c \
		file.c \
		message.c \
		dotting.c \
		args.c \
		system.c \
		exception.c \
		mem_spy.c \
		perf_spy.c \
		malloc_debug.c
BUILT_SOURCES=$(TARGET).h
include_HEADERS=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
include $(srcdir)/../pipslibs_includes.mk
AM_CPPFLAGS=-DCONFIG_DIR=\"$(sysconfdir)\"
