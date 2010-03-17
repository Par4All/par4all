TARGET	= pipsdbm
include_HEADERS=$(TARGET).h 
dist_noinst_HEADERS=private.h methods.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libpipsdbm.la
libpipsdbm_la_SOURCES=\
	database.c \
	lowlevel.c \
	workspace.c \
	externals.c \
	misc.c
include $(srcdir)/../pipslibs_includes.mk
