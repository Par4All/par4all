# $Id$

PARALLEL_VALIDATION = 1
include ../../validate.mk

# add-on to check freia-spoc optimality
include ../counts.mk

# combinatorial test cases are generated
validate: validate-test
validate-test: .generate
generate: .generate

SHELL	= /bin/bash
GENOPT	= --symmetry --usetemps
GEN	= ./schedule_combi.pl
.generate:
	$(GEN) $(GENOPT)
	for f in freia_*.c ; do mkdir $${f/.c/.result} ; done
	touch $@

full-clean: clean local-clean
local-clean:
	$(RM) .generate freia_*.c
	$(RM) -r freia_*.result
