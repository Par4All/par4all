TARGET=rstream_interface

include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=$(TARGET)-local.h local.h
BUILT_SOURCES=$(TARGET).h 
include $(top_srcdir)/makes/cproto.mk 
noinst_LTLIBRARIES=librstream_interface.la 
librstream_interface_la_SOURCES= rstream_interface.c 

include $(srcdir)/../pipslibs_includes.mk
