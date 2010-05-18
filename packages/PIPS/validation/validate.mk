# $Id$
#
# TODO
# * *.f95 for gfc2pips

# pips exes
TPIPS	= tpips
PIPS	= pips

# 10 minutes default timeout
# use 0 for no timeout
TIMEOUT	= 600

# default output file
# this can be modified to generate separate files
# see "validate-out" and "validate-test" targets
TEST	= test
DIFF	= svn diff

# prefix of tests to be run, default is all
PREFIX	=

# source files
F.c	= $(wildcard *.c)
F.f	= $(wildcard *.f)
F.F	= $(wildcard *.F)
F.f90	= $(wildcard *.f90)
F.f95	= $(wildcard *.f95)

# all source files
F.src	= $(F.c) $(F.f) $(F.F) $(F.f90) $(F.f95)

# all potential result directories
F.res	= $(F.c:%.c=%.result) $(F.f:%.f=%.result) \
	$(F.F:%.F=%.result) $(F.f90:%.f90=%.result) $(F.f95:%.f95=%.result)

# actual result directory
F.result= $(wildcard $(PREFIX)*.result)

# validation scripts
F.tpips	= $(wildcard *.tpips)
F.tpips2= $(wildcard *.tpips2)
F.test	= $(wildcard *.test)
F.py	= $(wildcard *.py)

F.exe	= $(F.tpips) $(F.tpips2) $(F.test) $(F.py)

# validation output
F.valid	= $(F.result:%=%/$(TEST))

# all base cases
F.list	= $(F.result:%.result=%)

SUBDIR	= $(notdir $(PWD))
here	:= $(shell pwd)
FLT	= sed -e 's,$(here),$$VDIR,g'
#OK	= exit 0
RESULTS	= RESULTS

SHELL	= /bin/bash
PF	= set -o pipefail ; \
	  export PIPS_MORE=cat PIPS_TIMEOUT=$(TIMEOUT) LC_ALL=C

# extract validation result for summary
# 134 is for pips_internal_error, could allow to distinguish voluntary aborts.
OK	= status=$$? ; \
	  if [ "$$status" -eq 255 ] ; then \
	     echo "timeout: $(SUBDIR)/$*" ; \
	  elif [ "$$status" != 0 ] ; then \
	     echo "failed: $(SUBDIR)/$*" ; \
	  else \
	     $(DIFF) $*.result/test > $*.diff ; \
	     if [ -s $*.diff ] ; then \
	        echo "changed: $(SUBDIR)/$*" ; \
	     else \
	        $(RM) $*.err $*.diff ; \
	        echo "passed: $(SUBDIR)/$*" ; \
	     fi ; \
	  fi >> $(RESULTS)

# default target is to clean
clean: clean-validate

clean-validate:
	$(RM) *~ *.o *.s *.tmp *.err *.diff *.result/out out err a.out
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
# ??? does not work because of "svn diff"?
validate-out:
	$(MAKE) TEST=out DIFF=pips_validation_diff_out.sh validate-dir

# generate "test" files
validate-test:
	$(MAKE) TEST=test validate-dir

# hack: validate depending on prefix?
validate-%:
	$(MAKE) F.result="$(wildcard $**.result)" validate-dir

# generate missing "test" files
test: $(F.valid)

# (shell) script
%.result/$(TEST): %.test
	$(PF) ; ./$< 2> $*.err | $(FLT) > $@ ; $(OK)

# tpips scripts
%.result/$(TEST): %.tpips
	$(PF) ; $(TPIPS) $< 2> $*.err | $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.tpips2
	$(PF) ; $(TPIPS) $< 2>&1 | $(FLT) > $@ ; $(OK)

# python scripts (could be a .test)
%.result/$(TEST): %.py
	$(PF) ; python $< 2> $*.err | $(FLT) > $@ ; $(OK)

# default_tpips
# FILE could be $<
# VDIR could be avoided if running in local directory?
DFTPIPS	= default_tpips
%.result/$(TEST): %.c $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	2> $*.err | $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.f $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	2> $*.err | $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.F $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	2> $*.err | $(FLT) > $@ ; $(OK)

# default_test relies on FILE WSPACE NAME
# Semantics & Regions create local "properties.rc":-(
DEFTEST	= default_test
%.result/$(TEST): %.c $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	2> $*.err | $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.f $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	2> $*.err | $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.F $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< sh $(DEFTEST) \
	2> $*.err | $(FLT) > $@ ; $(OK)

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

# test RESULT directory without any script
orphan:
	for base in $(sort $(F.list)) ; do \
	  test -f $$base.tpips -o \
	       -f $$base.tpips2 -o \
	       -f $$base.test -o \
	       -f $$base.py -o \
	       -f default_tpips -o \
	       -f default_test || \
	  echo "orphan: $$base" ; \
	done >> $(RESULTS)

# test case with multiple scripts... one is randomly (?) chosen
multi-script:
	for base in $$(echo $(basename $(F.exe))|tr ' ' '\012'|sort|uniq -d); \
	do \
	  echo "multi-script: $(SUBDIR)/$$base" ; \
	done >> $(RESULTS)

# test case with multiple sources (c/f/F...)
multi-source:
	for base in $$(echo $(basename $(F.src))|tr ' ' '\012'|sort|uniq -d); \
	do \
	  echo "multi-source: $(SUBDIR)/$$base" ; \
	done >> $(RESULTS)

# all possible inconsistencies
inconsistencies: skipped orphan multi-source multi-script

# what about nothing?
# source files without corresponding result directory
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
