# $Id$
#
# Run validation in a directory and possibly its subdirectories
#
# relevant targets for the end-user:
# - validate-test: validate to "test" result files directly
# - unvalidate: cleanup before validate-test
# - validate-out: validate to "out" files
# - clean: clean directories
# - generate-test: generate missing "test" files
#
# relevant variables for the user:
# - DO_BUG: also validate on cases tagged as "bugs"
# - DO_LATER: idem with future "later" cases
# - DO_SLOW: idem for lengthy to validate cases
# - DO_DEFAULT: other test cases
# - DO_PYPS: pyps must be available
# - DO_F95: gfc2pips must be available
# - D.sub: subdirectories in which to possibly recurse, defaults to *.sub
# - OK: actions taken after validation, set to empty to keep err file
#
# example to do only later cases:
#   sh> make DO_DEFAULT= DO_SLOW= DO_LATER=1 validate-test
# special useful targets include:
#   sh> make later-validate-test
#   sh> make bug-validate-out
# validate one test case:
#   sh> make test-case.validate
#

# what special cases are included
DO_BUG	=
DO_LATER=
DO_SLOW	= 1
DO_DEFAULT = 1

# for pyps user, if pyps is not available, the validation will just
# skip the corresponding cases silently.
DO_PYPS	:= $(shell type ipyps > /dev/null 2>&1 && echo 1)
DO_F95	:= $(shell type gfc2pips > /dev/null 2>&1 && echo 1)

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
CHECK	= $(IS_SVN)

# some parametric commands
SVN	= svn
DIFF	= $(SVN) diff
UNDO	= $(SVN) revert
LIST	= $(SVN) status
INFO	= $(SVN) info

# prefix of tests to be run, default is all
PREFIX	=

# possible filtering
F.later	= $(wildcard $(PREFIX)*.later)
F.bug	= $(wildcard $(PREFIX)*.bug)
F.slow	= $(wildcard $(PREFIX)*.slow)

# automatic sub directories,
# D.sub could be set explicitely to anywhere to recurse
D.sub	= $(wildcard *.sub)

# directory recursion
D.rec	= $(D.sub:%=%.rec)

# source files
F.c	= $(wildcard *.c)
F.f	= $(wildcard *.f)
F.F	= $(wildcard *.F)
F.f90	= $(wildcard *.f90)
F.f95	= $(wildcard *.f95)

# all source files
F.src	= $(F.c) $(F.f) $(F.F) $(F.f90) $(F.f95)

# all potential result directories
F.res	= \
	$(F.c:%.c=%.result) \
	$(F.f:%.f=%.result) \
	$(F.F:%.F=%.result) \
	$(F.f90:%.f90=%.result) \
	$(F.f95:%.f95=%.result)

# actual result directories to validate
F.result= $(wildcard $(PREFIX)*.result)

# various validation scripts
F.tpips	= $(wildcard *.tpips)
F.tpips2= $(wildcard *.tpips2)
F.test	= $(wildcard *.test)
F.py	= $(wildcard *.py)

# all scripts
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
F.output = \
	$(F.result:%=%/$(TEST))

# virtual target to trigger the validations
# with prioritized "slow" cases ahead
F.valid	= $(F.slow:%.slow=%.validate) $(F.result:%.result=%.validate)

# all base cases
F.list	= $(F.result:%.result=%)

# where are we?
SUBDIR	= $(notdir $(PWD))
here	:= $(shell pwd)

# get rid of absolute file names in output...
FLT	= sed -e 's,$(here),$$VDIR,g'

# where to store validation results
# this is the default, but may need to be overriden
RESULTS	= RESULTS

# this macro allows to control results are displayed
#TORESULTS = >> $(RESULTS)
TORESULTS = | tee -a $(RESULTS)

# shell environment to run validation scripts
# this is a requirement!
SHELL	= /bin/bash

# whether we are recurring in a specially "marked" directory
RECWHAT	=

# skip bug/later/slow cases depending on options
# a case requires a result directory, so a "bug/later/slow"
# tag is not counted if there is no corresponding result.
# f95 test cases may be kept out of the validation from here as well.
EXCEPT =  [ "$(RECWHAT)" ] && \
	    { echo "$(RECWHAT): $(SUBDIR)/$*" $(TORESULTS) ; exit 0 ; } ; \
	  [ ! "$(DO_BUG)" -a -f $*.bug -a -d $*.result ] && \
	    { echo "bug: $(SUBDIR)/$*" $(TORESULTS) ; exit 0 ; } ; \
	  [ ! "$(DO_LATER)" -a -f $*.later -a -d $*.result ] && \
	    { echo "later: $(SUBDIR)/$*" $(TORESULTS) ; exit 0 ; } ; \
	  [ ! "$(DO_SLOW)" -a -f $*.slow -a -d $*.result ] && \
	    { echo "slow: $(SUBDIR)/$*" $(TORESULTS) ; exit 0 ; } ; \
	  [ ! "$(DO_DEFAULT)" -a -d $*.result -a \
	    ! \( -f $*.bug -o -f $*.later -o -f $*.slow \) ] && \
	    { echo "skipped: $(SUBDIR)/$*" $(TORESULTS) ; exit 0 ; } ; \
	  [ ! "$(DO_F95)" -a -d $*.result -a \( -e $*.f90 -o -e $*.f95 \) ] && \
	    { echo "keptout: $(SUBDIR)/$*" $(TORESULTS) ; exit 0 ; }


# setup running a case
PF	= @echo "processing $(SUBDIR)/$+" ; \
	  $(EXCEPT) ; \
	  $(RM) $*.result/$(TEST) ; \
	  set -o pipefail ; unset CDPATH ; \
	  export PIPS_MORE=cat PIPS_TIMEOUT=$(TIMEOUT) LC_ALL=C

# recursion into a subdirectory with target "FORWARD"
# a whole directory can be marked as bug/later/slow,
# in which case while recurring this mark take precedence about
# local information made available within the directory
%.rec: %
	@recwhat= ; d=$* ; d=$${d%.sub} ; \
	[ ! "$(DO_BUG)" -a -f $$d.bug ] && recwhat=bug ; \
	[ ! "$(DO_LATER)" -a -f $$d.later ] && recwhat=later ; \
	[ ! "$(DO_SLOW)" -a -f $$d.slow ] && recwhat=slow ; \
	[ ! "$(DO_DEFAULT)" -a ! -f $$d.slow -a ! -f $$d.later -a \
	  ! -f $$d.bug ] && recwhat=skipped ; \
	[ "$(RECWHAT)" ] && recwhat=$(RECWHAT) ; \
	$(MAKE) RECWHAT=$$recwhat RESULTS=../$(RESULTS) SUBDIR=$(SUBDIR)/$^ \
		-C $^ $(FORWARD) || \
	  echo "broken-directory: $(SUBDIR)/$^" $(TORESULTS)

# extract validation result for summary when the case was run
# five possible outcomes: passed, changed, failed, timeout, noref
# noref means there is no reference file for comparison
# 134 is for pips_internal_error, could allow to distinguish voluntary aborts.
OK	= status=$$? ; \
	  if [ "$$status" -eq 203 ] ; then \
	   echo "timeout: $(SUBDIR)/$* $$SECONDS" ; \
	  elif [ "$$status" != 0 ] ; then \
	   echo "failed: $(SUBDIR)/$* $$SECONDS" ; \
	  else \
	    novcref= ; \
	    $(INFO) $*.result/test > /dev/null 2>&1 || novcref=1 ; \
	    if [ \( $(TEST) = 'test' -a "$$novcref" \) -o \
	         \( $(TEST) = 'out' -a ! -e $*.result/test \) ] ; then \
	      echo "noref: $(SUBDIR)/$* $$SECONDS" ; \
	    else \
	      $(DIFF) $*.result/test > $*.diff ; \
	      if [ -s $*.diff ] ; then \
	        echo "changed: $(SUBDIR)/$* $$SECONDS" ; \
	      else \
	        $(RM) $*.err $*.diff ; \
	        echo "passed: $(SUBDIR)/$* $$SECONDS" ; \
	      fi ; \
            fi ; \
	  fi $(TORESULTS)

# default target is to clean
.PHONY: clean
clean: rec-clean clean-validate
LOCAL_CLEAN	= clean-validate

.PHONY: clean-validate
clean-validate:
	$(RM) *~ *.o *.s *.tmp *.err *.diff *.result/out out err a.out RESULTS
	$(RM) -r *.database .PYPS*.tmp .t*.tmp

.PHONY: rec-clean
rec-clean:
	[ "$(D.rec)" ] && $(MAKE) FORWARD=clean $(D.rec) || exit 0

.PHONY: validate
validate:
	# Parallel validation
	# run "make validate-test" to generate "test" files.
	# run "make validate-out" to generate usual "out" files.
	# run "make unvalidate" to revert test files to their initial status.
	# run "make {later,bug,slow,default}-validate-{test,out}" for testing subsets

# convenient shortcuts to validate subsets (later, bug, slow, default)
later-validate-test:
	$(MAKE) DO_DEFAULT= DO_SLOW= DO_BUG= DO_LATER=1 validate-test
later-validate-out:
	$(MAKE) DO_DEFAULT= DO_SLOW= DO_BUG= DO_LATER=1 validate-out
bug-validate-test:
	$(MAKE) DO_DEFAULT= DO_SLOW= DO_BUG=1 DO_LATER= validate-test
bug-validate-out:
	$(MAKE) DO_DEFAULT= DO_SLOW= DO_BUG=1 DO_LATER= validate-out
slow-validate-test:
	$(MAKE) DO_DEFAULT= DO_SLOW=1 DO_BUG= DO_LATER= validate-test
slow-validate-out:
	$(MAKE) DO_DEFAULT= DO_SLOW=1 DO_BUG= DO_LATER= validate-out
default-validate-test:
	$(MAKE) DO_DEFAULT=1 DO_SLOW= DO_BUG= DO_LATER= validate-test
default-validate-out:
	$(MAKE) DO_DEFAULT=1 DO_SLOW= DO_BUG= DO_LATER= validate-out


.PHONY: validate-dir
# the PARALLEL_VALIDATION macro tell whether it can run in parallel
ifdef PARALLEL_VALIDATION
validate-dir: $(LOCAL_CLEAN)
	$(MAKE) $(D.rec) $(F.valid)
	@$(MAKE) sort-local-result

else # sequential validation, including subdir recursive forward
validate-dir: $(LOCAL_CLEAN)
	$(MAKE) $(D.rec) sequential-validate-dir
	@$(MAKE) sort-local-result

# local target to parallelize the "sequential" local directory
# with test cases in its subdirectories
sequential-validate-dir:
	for f in $(F.valid) ; do $(MAKE) $$f ; done
endif

# how to summarize results to a human
SUMUP	= pips_validation_summary.pl

# on local validations only, sort result & show summary
.PHONY: sort-local-result
sort-local-result:
	@if [ $(RESULTS) = RESULTS -a -f RESULTS ] ; then \
	  mv RESULTS RESULTS.tmp ; \
	  sort -k 2 RESULTS.tmp > RESULTS ; \
	  $(RM) RESULTS.tmp ; \
	  $(SUMUP) RESULTS ; \
	fi

# restore all initial "test" result files if you are unhappy with a validate
.PHONY: unvalidate
unvalidate: do-unvalidate rec-unvalidate

.PHONY: do-unvalidate
do-unvalidate:: check-vc
	-$(CHECK) && [ $(TEST) = 'test' ] && $(UNDO) $(F.output)

.PHONY: rec-unvalidate
rec-unvalidate::
	[ "$(D.rec)" ] && $(MAKE) FORWARD=unvalidate $(D.rec) || exit 0

# generate "out" files
.PHONY: validate-out
validate-out:
	$(MAKE) TEST=out DIFF=pips_validation_diff_out.sh \
		FORWARD=$@ LIST=: UNDO=: validate-dir

# generate "test" files: svn diff show the diffs!
.PHONY: validate-test
validate-test: check-vc
	$(MAKE) FORWARD=$@ TEST=test validate-dir

# hack: validate depending on prefix, without forwarding?
validate-%:
	$(MAKE) F.result="$(wildcard $**.result)" validate-dir

# generate missing "test" files
.PHONY: generate-test
generate-test: $(F.output)

# generate empty result directories, for Ronan
# beware that this is a magick guess from the contents of the directory
# you then have to generate the corresponding "test" file
# and commit everything on the svn
.PHONY: generate-result
generate-result: $(F.future_result)

# generate an empty result directory & file
%.result:
	@echo "creating: $@" ; mkdir $@ ; touch $@/test

# indirect validation trigger
# a % generic target cannot be empty!
%.result/$(TEST): %.validate
	@echo "done $@" >&2

# always do target? does not seem to work as expected??
#.PHONY: $(F.valid)

# (shell) script, possibly uses "pips"?
%.validate: %.test
	$(PF) ; ./$< 2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

# tpips scripts
%.validate: %.tpips
	$(PF) ; $(TPIPS) $< 2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

# special case for stderr validation, which is basically a bad idea (tm),
# with a provision to keep only part of the stderr output.
%.validate: %.tpips2
	$(PF) ; $(TPIPS) $< 2> $*.err | $(FLT) > $*.result/$(TEST) ; \
	{ \
	  echo "### stderr" ; \
	  if [ -e $*.flt ] ; then \
	    $(FLT) $*.err | ./$*.flt ; \
	  else \
	    $(FLT) $*.err ; \
	  fi ; \
	} >> $*.result/$(TEST) ; $(OK)

# python scripts
PYTHON	= python
ifdef DO_PYPS
%.validate: %.py
	$(PF) ; $(PYTHON) $< 2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)
else # no pyps...
%.validate: %.py
	@$(EXCEPT) ; echo "keptout: $(SUBDIR)/$*" $(TORESULTS)
endif # DO_PYPS

# default_tpips
# FILE could be $<
# VDIR could be avoided if running in local directory?
DFTPIPS	= default_tpips
%.validate: %.c $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.F $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f90 $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f95 $(DFTPIPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< VDIR=$(here) $(TPIPS) $(DFTPIPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)


# default_test relies on FILE WSPACE NAME
# warning: Semantics & Regions create local "properties.rc":-(
DEFTEST	= default_test
%.validate: %.c $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< ./$(DEFTEST) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< ./$(DEFTEST) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.F $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< ./$(DEFTEST) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f90 $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< ./$(DEFTEST) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f95 $(DEFTEST)
	$(PF) ; WSPACE=$* FILE=$(here)/$< ./$(DEFTEST) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)


# default_pyps relies on FILE & WSPACE
DEFPYPS	= default_pyps.py
ifdef DO_PYPS
%.validate: %.c $(DEFPYPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< $(PYTHON) $(DEFPYPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f $(DEFPYPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< $(PYTHON) $(DEFPYPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.F $(DEFPYPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< $(PYTHON) $(DEFPYPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f90 $(DEFPYPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< $(PYTHON) $(DEFPYPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

%.validate: %.f95 $(DEFPYPS)
	$(PF) ; WSPACE=$* FILE=$(here)/$< $(PYTHON) $(DEFPYPS) \
	2> $*.err | $(FLT) > $*.result/$(TEST) ; $(OK)

else # without pyps

%.validate: %.c $(DEFPYPS)
	@$(EXCEPT) ; echo "keptout: $(SUBDIR)/$*" $(TORESULTS)

%.validate: %.f $(DEFPYPS)
	@$(EXCEPT) ; echo "keptout: $(SUBDIR)/$*" $(TORESULTS)

%.validate: %.F $(DEFPYPS)
	@$(EXCEPT) ; echo "keptout: $(SUBDIR)/$*" $(TORESULTS)

%.validate: %.f90 $(DEFPYPS)
	@$(EXCEPT) ; echo "keptout: $(SUBDIR)/$*" $(TORESULTS)

%.validate: %.f95 $(DEFPYPS)
	@$(EXCEPT) ; echo "keptout: $(SUBDIR)/$*" $(TORESULTS)

endif # DO_PYPS

# special case for "slow" directories...
%.validate:
	@echo "skipping $@, possibly a slow subdirectory" >&2

# detect skipped stuff
.PHONY: skipped
skipped:
	@for base in $(sort $(basename $(F.src) $(F.exe))) ; do \
	  if ! test -d $$base.result ; \
	  then \
	    echo "skipped: $(SUBDIR)/$$base" ; \
	  elif ! [ -f $$base.result/test ] ; \
	  then \
	    echo "missing: $(SUBDIR)/$$base" ; \
	  fi ; \
	done $(TORESULTS)

# test RESULT directory without any script...
# this warning is reported only if the case would be executed in the
# current validation context (i.e. it may be ignored for bug/later/slow).
.PHONY: orphan
orphan:
	@for base in $(sort $(F.list)) ; \
	do \
	  [ ! "$(DO_BUG)" -a -e $$base.bug ] && continue ; \
	  [ ! "$(DO_LATER)" -a -e $$base.later ] && continue ; \
	  [ ! "$(DO_SLOW)" -a -e $$base.slow ] && continue ; \
	  [ ! "$(DO_F95)" -a \( -e $$base.f90 -o -e $$base.f95 \) ] && \
	    continue ; \
	  test -f $$base.tpips -o \
	       -f $$base.tpips2 -o \
	       -f $$base.test -o \
	       -f $$base.py -o \
	       -f default_tpips -o \
	       -f default_pyps.py -o \
	       -f default_test || \
	  echo "orphan: $(SUBDIR)/$$base" ; \
	done $(TORESULTS)

# test case with multiple scripts... one is randomly (?) chosen
.PHONY: multi-script
multi-script:
	@for base in $$(echo $(basename $(F.exe))|tr ' ' '\012'|sort|uniq -d); \
	do \
	  echo "multi-script: $(SUBDIR)/$$base" ; \
	done $(TORESULTS)

# test case with multiple sources (c/f/F...)
.PHONY: multi-source
multi-source:
	@for base in $$(echo $(basename $(F.src))|tr ' ' '\012'|sort|uniq -d); \
	do \
	  echo "multi-source: $(SUBDIR)/$$base" ; \
	done $(TORESULTS)

# empty 'test' result file
.PHONY: empty-test
empty-test:
	@for base in $(sort $(basename $(F.src) $(F.exe))) ; do \
          if test -d $$base.result -a -e $$base.result/test -a ! -s $$base.result/test ; \
          then \
            echo "empty-test: $(SUBDIR)/$$base" ; \
          fi ; \
        done $(TORESULTS)

# check that all tpips2 have a corresponding flt
.PHONY: nofilter
nofilter:
	@for f in $(F.tpips2) ; do \
	  test -e $${f/.tpips2/.flt} || \
	    echo "nofilter: $(SUBDIR)/$${f/.tpips2/}" ; \
	done $(TORESULTS)

# all possible inconsistencies
.PHONY: inconsistencies
inconsistencies: skipped orphan multi-source multi-script nofilter empty-test
	@[ "$(D.rec)" ] && $(MAKE) FORWARD=inconsistencies $(D.rec) || exit 0

########################################################### LOCAL CHECK HELPERS

# what about nothing?
# source files without corresponding result directory
.PHONY: missing
missing:
	@echo "# checking for missing (?) result directories"
	@ n=0; \
	for res in $(F.res) ; do \
	  if [ ! -d $$res ] ; then \
	     echo "missing result directory: $$res" ; \
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
