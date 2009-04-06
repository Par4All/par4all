# $Id$

SCRIPTS = \
	pips_validate \
	pips_force_accept \
	manual_accept \
	pips_emacs_diff \
	array_resizing_instrument.pl \
	ubs_checking_instrument.pl

INSTALL_UTL =	$(SCRIPTS)

# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk
