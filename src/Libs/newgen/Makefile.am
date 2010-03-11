
TARGET	= newgen
BUILT_SOURCES=$(TARGET).h
include_HEADERS=$(TARGET).h
noinst_LTLIBRARIES=libnewgen.la
libnewgen_la_SOURCES=\
		newgen.c \
		Pvecteur.c \
		Ppolynome.c \
		Psc.c

libnewgen_la_LIBADD=../../Documentation/newgen/libnewgenaux.la
include $(top_srcdir)/makes/cproto.mk
include $(srcdir)/../pipslibs_includes.mk
