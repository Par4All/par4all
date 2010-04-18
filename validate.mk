# $Id$
#
# TODO
# * timeout
# * *.f95 for gfc2pips
# * check for multiple sources
# * check for multiple scripts

# pips exes
TPIPS	= tpips
PIPS	= pips

# default output file
# this can be modified to generate separate files
# see "validate-out" and "validate-test" targets
TEST	= test

# source files
F.c	= $(wildcard *.c)
F.f	= $(wildcard *.f)
F.F	= $(wildcard *.F)
#F.f95	= $(wildcard *.f95)

# all source files
F.src	= $(F.c) $(F.f) $(F.F)

# all potential result directories
F.res	= $(F.c:%.c=%.result) $(F.f:%.f=%.result) $(F.F:%.F=%.result)

# actual result directory
F.result= $(wildcard *.result)

# validation scripts
F.tpips	= $(wildcard *.tpips)
F.test	= $(wildcard *.test)
F.py	= $(wildcard *.py)

F.exe	= $(F.tpips) $(F.test) $(F.py)

# validation output
F.valid	= $(F.result:%=%/$(TEST))

SUBDIR	= $(notdir $(PWD))
here	:= $(shell pwd)
FLT	= sed -e 's,$(here),$$VDIR,g'
#OK	= exit 0
RESULTS	= failed

SHELL	= /bin/bash
PF	= set -o pipefail

# extract validation result for summary
OK	= status=$$? ; \
	  if [ "$$status" != 0 ] ; then \
	     echo "failed: $(SUBDIR)/$*" ; \
	  elif [ $$(svn diff $@ | wc -l) -ne 0 ] ; then \
	     echo "changed: $(SUBDIR)/$*" ; \
	  else \
	     echo "passed: $(SUBDIR)/$*" ; \
	  fi >> $(RESULTS)

# default target is to clean
clean: clean-validate

clean-validate:
	$(RM) *~ *.o *.s *.tmp *.result/out out err a.out
	$(RM) -r *.database $(RESULTS)

validate:
	# Experimental parallel validation
	# run "make validate-out" to generate usual "out" files.
	# run "make validate-test" to generate "test" files.
	# run "make unvalidate" to revert test files to their initial status.

ifdef PARALLEL_VALIDATION
# regenerate files: svn diff show the diffs!
validate-dir:
	$(RM) $(F.valid)
	$(MAKE) $(F.valid)
else # sequential validation
validate-dir:
	$(RM) $(F.valid)
	for f in $(F.valid) ; do $(MAKE) $$f ; done
endif

# restore all initial "test" result files if you are unhappy with a validate
unvalidate:
	svn revert $(F.valid)

# generate "out" files
validate-out:
	$(MAKE) TEST=out validate-dir

# generate "test" files
validate-test:
	$(MAKE) TEST=test validate-dir

# validate depending on prefix?
validate-%:
	$(MAKE) F.result="$(wildcard $**.result)" validate-dir

# generate missing "test" files
test: $(F.valid)

# shell script
%.result/$(TEST): %.test
	$(PF) ; ./$< | $(FLT)  > $@ ; $(OK)

# tpips scripts
%.result/$(TEST): %.tpips
	$(PF) ; $(TPIPS) $< | $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.tpips2
	$(PF) ; $(TPIPS) $< 2<&1 | $(FLT) > $@ ; $(OK)

# python scripts
%.result/$(TEST): %.py
	$(PF) ; python $< | $(FLT) > $@ ; $(OK)

# default_tpips
# FILE could be $<
# VDIR could be avoided if running in local directory?
DFTPIPS	= default_tpips
%.result/$(TEST): %.c $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	| $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.f $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	| $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.F $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	| $(FLT) > $@ ; $(OK)

# default_test relies on FILE WSPACE NAME
# Semantics & Regions create local "properties.rc":-(
DEFTEST	= default_test
%.result/$(TEST): %.c $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	| $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.f $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	| $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.F $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	| $(FLT) > $@ ; $(OK)

# detect skipped stuff
skipped:
	for base in $(sort $(basename $(F.src) $(F.exe))) ; do \
	  if ! test -d $$base.result ; \
	  then \
	    echo "skipped: $(SUBDIR)/$$base" ; \
	  elif ! [ -f $$base.result/test -o -f $$base.result/test.$(ARCH) ] ; \
	  then \
	    echo "missing: $(SUBDIR)/$$base" ; \
	  fi ; \
	done >> $(RESULTS)

# what about nothing?
missing:
	@echo "# checking for missing (?) result directories"
	@ n=0; \
	for res in $(F.res) ; do \
	  if [ ! -d $$res ] ; then \
	     echo "missing: $$res" ; \
	     let n++; \
	  fi ; \
	done ; \
	echo "# $$n missing result(s)"

missing-svn:
	@echo "# result directories not under svn"
	@svn status | grep '\.result'

count:
	@echo "number of validations:" `echo $(F.result) | wc -w`
