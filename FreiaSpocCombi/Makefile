# $Id$

include ../validate.mk

validate: .generate

generate: .generate

SHELL	= /bin/bash
GENOPT	= --symmetry --usetemps
GEN	= ./schedule_combi.pl
.generate:
	$(GEN) $(GENOPT)	
	for f in freia_*.c ; do mkdir $${f/.c/.result} ; done
	touch $@

full-clean: local-clean
local-clean:
	$(RM) .generate freia_*.c
	$(RM) -r freia_*.result
