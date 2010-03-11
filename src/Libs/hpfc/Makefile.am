TARGET	= hpfc
include_HEADERS=$(TARGET).h
dist_noinst_HEADERS=warning.h access_description.h version.h defines-local.h
BUILT_SOURCES=$(TARGET).h
include $(top_srcdir)/makes/cproto.mk
noinst_LTLIBRARIES=libhpfc.la
libhpfc_la_SOURCES=directives.c \
		dynamic.c \
		hpfc.c \
		debug-util.c \
		hpfc-util.c \
		declarations.c \
		compiler-util.c \
		compiler.c \
		compile.c \
		run-time.c \
		generate.c \
		local-ri-util.c \
		inits.c \
		o-analysis.c \
		align-checker.c \
		messages.c \
		message-utils.c \
		build-system.c \
		io-util.c \
		io-compile.c \
		generate-util.c \
		remapping.c \
		host_node_entities.c \
		special_cases.c \
		lattice_extraction.c


include $(srcdir)/../pipslibs_includes.mk
