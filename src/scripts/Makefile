# $Id$

shr_scripts = \
	make_all_specs\
	newgen

utl_scripts = \
	test_newgen_type_translation.sh \
	test_newgen_type_translation.pl \
	tabulated_count.pl

# Makefile macros

SCRIPTS = $(shr_scripts) $(utl_scripts)

SOURCES = $(SCRIPTS)

INSTALL_SHR = $(shr_scripts)
INSTALL_UTL = $(utl_scripts)

################################################################## COMMON STUFF

ifdef NEWGEN_ROOT
ROOT	= $(NEWGEN_ROOT)
else
ROOT	= ../..
endif
include $(ROOT)/makes/base.mk
