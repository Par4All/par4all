# $Id$

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

# is it a subversion working copy?
IS_SVN	= test -d .svn

# some parametric commands
CHECK	= $(IS_SVN)
DIFF	= svn diff
UNDO	= svn revert
LIST	= svn status

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

# actual result directory to validate
F.result= $(wildcard $(PREFIX)*.result)

# various validation scripts
F.tpips	= $(wildcard *.tpips)
F.tpips2= $(wildcard *.tpips2)
F.test	= $(wildcard *.test)
F.py	= $(wildcard *.py)

F.exe	= $(F.tpips) $(F.tpips2) $(F.test) $(F.py)

# optimistic possible results for Ronan
F.future_result = \
	$(F.tpips:%.tpips=%.result) \
	$(F.tpips2:%.tpips2=%.result) \
	$(F.test:%.test=%.result) \
	$(F.py:%.py=%.result) \
	$(F.c:%.c=%.result) \
	$(F.f:%.f=%.result) \
	$(F.F:%.F=%.result) \
	$(F.f90:%.f90=%.result) \
	$(F.f95:%.f95=%.result)

# validation output
F.valid	= $(F.result:%=%/$(TEST))

# all base cases
F.list	= $(F.result:%.result=%)

# where are we?
SUBDIR	= $(notdir $(PWD))
here	:= $(shell pwd)
# get rid of absolute file names in output...
FLT	= sed -e 's,$(here),$$VDIR,g'
# where to store validation results
RESULTS	= RESULTS

# shell environment to run validation scripts
SHELL	= /bin/bash
# setup run
PF	= @echo "processing $(SUBDIR)/$+" ; \
	  [ ! "$(DO_BUG)" -a -f $*.bug ] && exit 0 ; \
	  [ ! "$(DO_LATER)" -a -f $*.later ] && exit 0 ; \
	  set -o pipefail ; unset CDPATH ; \
	  export PIPS_MORE=cat PIPS_TIMEOUT=$(TIMEOUT) LC_ALL=C

# extract validation result for summary
# four possible outcomes: passed, changed, failed, timeout
# 134 is for pips_internal_error, could allow to distinguish voluntary aborts.
OK	= status=$$? ; \
	  if [ "$$status" -eq 203 ] ; then \
	     echo "timeout: $(SUBDIR)/$* $$SECONDS" ; \
	  elif [ "$$status" != 0 ] ; then \
	     echo "failed: $(SUBDIR)/$* $$SECONDS" ; \
	  else \
	     $(DIFF) $*.result/test > $*.diff ; \
	     if [ -s $*.diff ] ; then \
	        echo "changed: $(SUBDIR)/$* $$SECONDS" ; \
	     else \
	        $(RM) $*.err $*.diff ; \
	        echo "passed: $(SUBDIR)/$* $$SECONDS" ; \
	     fi ; \
	  fi >> $(RESULTS)

# default target is to clean
clean: clean-validate
LOCAL_CLEAN	= clean-validate

.PHONY: clean-validate
clean-validate:
	$(RM) *~ *.o *.s *.tmp *.err *.diff *.result/out out err a.out
	$(RM) -r *.database $(RESULTS)

.PHONY: validate
validate:
	# Parallel validation
	# run "make validate-test" to generate "test" files.
	# run "make validate-out" to generate usual "out" files.
	# run "make unvalidate" to revert test files to their initial status.

.PHONY: validate-dir
# the PARALLEL_VALIDATION macro tell whether it can run in parallel
ifdef PARALLEL_VALIDATION
# regenerate files: svn diff show the diffs!
validate-dir: $(LOCAL_CLEAN) bug-list later-list
	$(RM) $(F.valid)
	$(MAKE) $(F.valid)
	$(MAKE) sort-local-result
else # sequential validation
validate-dir: $(LOCAL_CLEAN) bug-list later-list
	$(RM) $(F.valid)
	for f in $(F.valid) ; do $(MAKE) $$f ; done
	$(MAKE) sort-local-result
endif

# on local validations, sort result & show summary
.PHONY: sort-local-result
sort-local-result:
	@if [ $(RESULTS) = RESULTS -a -f RESULTS ] ; then \
	  mv RESULTS RESULTS.tmp ; \
	  sort -k 2 RESULTS.tmp > RESULTS ; \
	  $(RM) RESULTS.tmp ; \
	  pips_validation_summary.pl RESULTS ; \
	fi

# restore all initial "test" result files if you are unhappy with a validate
.PHONY: unvalidate
unvalidate: check-vc
	-$(CHECK) && [ $(TEST) = 'test' ] && $(UNDO) $(F.valid)

# generate "out" files
# ??? does not work because of "svn diff"?
.PHONY: validate-out
validate-out:
	$(MAKE) TEST=out DIFF=pips_validation_diff_out.sh LIST=: UNDO=: validate-dir

# generate "test" files
.PHONY: validate-test
validate-test: check-vc
	$(MAKE) TEST=test validate-dir

# hack: validate depending on prefix?
validate-%:
	$(MAKE) F.result="$(wildcard $**.result)" validate-dir

# generate missing "test" files
.PHONY: generate-test
generate-test: $(F.valid)

# generate empty result directories, for Ronan
# beware that this is a magick guess from the contents of the directory
# you then have to generate the corresponding "test" file
# and commit everything on the svn
.PHONY: generate-result
generate-result: $(F.future_result)

# generate an empty result directory
%.result:
	@echo "creating: $@" ; mkdir $@ ; touch $@/test

# (shell) script
%.result/$(TEST): %.test
	$(PF) ; ./$< 2> $*.err | $(FLT) > $@ ; $(OK)

# tpips scripts
%.result/$(TEST): %.tpips
	$(PF) ; $(TPIPS) $< 2> $*.err | $(FLT) > $@ ; $(OK)

%.result/$(TEST): %.tpips2
	$(PF) ; $(TPIPS) $< 2>&1 | $(FLT) > $@ ; $(OK)

# python scripts
ifdef PIPS_VALIDATION_NO_PYPS
%.result/$(TEST): %.py
	echo "keptout: $(SUBDIR)/$*" >> $(RESULTS)
else # else we have pyps
%.result/$(TEST): %.py
	$(PF) ; python $< 2> $*.err | $(FLT) > $@ ; $(OK)
endif # PIPS_VALIDATION_NO_PYPS

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

# bug & later handling
.PHONY: bug-list
ifdef DO_BUG
bug-list:
	@echo "# bug-list: nothing to do" >&2
else # include bug list
bug-list:
	for f in $(wildcard *.bug) ; do \
	  echo "bug: $(SUBDIR)/$${f%.*}" ; \
	done >> $(RESULTS)
endif # DO_BUG

.PHONY: later-list
ifdef DO_LATER
later-list:
	@echo "# later-list: nothing to do" >&2
else # include later list
later-list:
	for f in $(wildcard *.later) ; do \
	  echo "later: $(SUBDIR)/$${f%.*}" ; \
	done >> $(RESULTS)
endif # DO_LATER

# detect skipped stuff
.PHONY: skipped
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
.PHONY: orphan
orphan:
	for base in $(sort $(F.list)) ; do \
	  test -f $$base.tpips -o \
	       -f $$base.tpips2 -o \
	       -f $$base.test -o \
	       -f $$base.py -o \
	       -f default_tpips -o \
	       -f default_test || \
	  echo "orphan: $(SUBDIR)/$$base" ; \
	done >> $(RESULTS)

# test case with multiple scripts... one is randomly (?) chosen
.PHONY: multi-script
multi-script:
	for base in $$(echo $(basename $(F.exe))|tr ' ' '\012'|sort|uniq -d); \
	do \
	  echo "multi-script: $(SUBDIR)/$$base" ; \
	done >> $(RESULTS)

# test case with multiple sources (c/f/F...)
.PHONY: multi-source
multi-source:
	for base in $$(echo $(basename $(F.src))|tr ' ' '\012'|sort|uniq -d); \
	do \
	  echo "multi-source: $(SUBDIR)/$$base" ; \
	done >> $(RESULTS)

# all possible inconsistencies
.PHONY: inconsistencies
inconsistencies: skipped orphan multi-source multi-script

# what about nothing?
# source files without corresponding result directory
.PHONY: missing
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

.PHONY: missing-vc
missing-vc:
	@echo "# result directories not under version control"
	@$(LIST) | grep '\.result'

# check that we are in a working copy
.PHONY: check-vc
check-vc:
	@$(CHECK) || { \
	  echo "error: validation must be a working copy" >&2 ; \
	  exit 1 ; \
	}

.PHONY: count
count:
	@echo "number of validations:" `echo $(F.result) | wc -w`
