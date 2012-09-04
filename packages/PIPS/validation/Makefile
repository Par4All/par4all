# $Id$

# name of the validation setup
# update if you use this Makefile for other test cases
DIRNAME	= validation

# default is to do a "simple" clean
default: clean

# use new validation without implicit accept
validate: validate-out
validate-%: parallel-validate-%

# old targets:
# old-validate
# old-validate-%
# validate-all: old-validate-all

# useful variant to be consistent with intra-directory validation
validate-test: new-validate
validate-out:; $(MAKE) TEST=out new-validate

SHELL	= /bin/bash
FIND	= find . -name '.svn' -type d -prune -o -name '.git' -type d -prune -o
ARCH	= $(shell uname -m)
ifndef VNAME
# used to start the mail subject line
VNAME	= $(ARCH)
endif

# this target also remove the archive
.PHONY: full-clean
full-clean: clean
	$(FIND) -name 'SUMMARY_Archive' -type d -print0 \
	     -o -name 'RESULTS' -type d -print0 | \
	  xargs -0 $(RM) -r
	$(RM) SUMMARY SUMMARY.short

# simple clean target
.PHONY: simple-clean
simple-clean:
	$(MAKE) TARGET=. clean-target

# subdirectories to consider
TARGET	:= $(shell grep '^[a-zA-Z]' defaults | grep -v "$${PIPS_VALIDATION_SKIP:-keep-all}")
show-target:; @echo $(TARGET)

.PHONY: clean-target
clean-target:
	here=$(PWD) ; \
	for d in $(TARGET) ; do \
	  echo "### cleaning $$d" ; \
	  cd $$d ; \
	  $(FIND) -name '*~' -type f -print0 \
	     -o -name 'core' -type f -print0 \
	     -o -name 'a.out' -type f -print0 \
	     -o -name 'out' -type f -print0 \
	     -o -name '*.filtered' -type f -print0 \
	     -o -name '*.o' -type f -print0 | xargs -0 $(RM) ; \
	  $(FIND) -name '*.database' -type d -print0 \
	     -o -name 'RESULTS' -type d -print0 \
	     -o -name 'validation_results.*' -type d -print0 | \
		xargs -0 $(RM) -r ; \
	  cd $$here ; \
	done
	$(RM) properties.rc a.out core *.o

include old-validate.mk

#
# PARALLEL VALIDATION
#
# directory-parallel validation test
# may replace the previous entry some day

# where results are stored
RESULTS	= validation.out
HEAD	= validation.head

# keep summaries
SUM.d	= SUMMARY_Archive
NOW.d	:= $(shell date +%Y/%m)
DEST.d	= $(SUM.d)/$(NOW.d)
NOW	:= SUMMARY.$(shell date +%Y-%m-%d_%H_%M_%S)
SUM.last	= $(SUM.d)/SUMMARY-last
SUM.prev	= $(SUM.d)/SUMMARY-previous

$(DEST.d):
	mkdir -p $@

SVN.URL	= $(shell svn info | grep 'Repository Root' | cut -d: -f2-)
SVN.R	= $(shell svnversion)
SVN.C	= $(shell svnversion -c)

# To have some git information
GIT_DESCRIBE=git describe --long --always --all

# check for svn working copy early
.PHONY: check-run-consistency
check-run-consistency:
	@[ $(TEST) = 'out' ] && exit 0 ; \
	[ '$(TEST)' = 'test' -a -d .svn ] || { \
	  echo "TEST=test parallel validation requires svn" >&2 ; \
	  echo "try: make TEST=out <your arguments...>" >&2 ; \
	  exit 1 ; \
	}

# check for Makefiles in subdirectories
.PHONY: check-makefile
check-makefile:
	-for dir in * ; do \
	  test -d $$dir -a ! -f $$dir/Makefile && echo "missing $$dir/Makefile" ; \
	done

# this target should replace the "validate" target
.PHONY: new-validate
new-validate:
	$(RM) SUMMARY SUMMARY.short
	$(MAKE) parallel-clean
	$(MAKE) archive
	$(MAKE) SUMMARY.short

# send the summary by email
.PHONY: mail-validate
mail-validate: new-validate
	Mail -a "Reply-To: $(EMAIL)" \
	     -s "$(shell tail -1 SUMMARY.short)" \
		$(EMAIL) < SUMMARY.short

# send an email only if there were changes
.PHONY: mail-diff-validate
mail-diff-validate: new-validate
	diff=$$(grep '^ \* status changes: ' SUMMARY.short | grep -v 'none') ; \
	[ "$$diff" ] && \
	   Mail -a "Reply-To: $(EMAIL)" \
		-s "$(shell tail -1 SUMMARY.short)" \
			$(EMAIL) < SUMMARY.short

# how to summarize results
SUMUP	= pips_validation_summary.pl --aggregate

# generate summary header
# hmmm... not sure that start date is before the validation
# pyps version?
$(HEAD): check-run-consistency
	{ \
	  echo "parallel validation" ; \
	  echo "on dirs: $(TARGET)" ; \
	  echo "host name: $$(hostname)" ; \
	  echo "in directory: $(PWD)" ; \
	  test -d .svn && echo " $(SVN.URL)@$(SVN.R) ($(SVN.C))" ; \
	  $(GIT_DESCRIBE) > /dev/null 2>&1 && echo "Git validation version: `$(GIT_DESCRIBE)`" ; \
	  echo "with pips: $(shell which pips)" ; \
	  pips -v ; \
	  echo "with tpips: $(shell which tpips)" ; \
	  tpips -v ; \
	  echo "by user: $$USER" ; \
	  echo "options: EXE=$(PIPS_VALIDATION_EXE)" \
	       "C-CHECK=$(PIPS_CHECK_C) FORTRAN-CHECK=$(PIPS_CHECK_FORTRAN)" ; \
	  echo "start date: $$(date) [$$(date +%s)]" ; \
	} > $@

# generate & archive validation summary
SUMMARY: $(HEAD) parallel-validate
	{ \
	  cat $(HEAD) ; \
	  echo "end date: $$(date) [$$(date +%s)]" ; \
	  echo ; \
	} > $@ ;
	cat $@ $(RESULTS) > $@.tmp ; \
	{ \
	  [ -f $(SUM.last) ] && last=$(SUM.last) ; \
	  $(SUMUP) $@.tmp $$last ; \
	  $(RM) $@.tmp ; \
	} >> $@ ; \
	{ \
          echo ; \
	  unset LANG LC_COLLATE ; \
	  sort -k 2 $(RESULTS) ; \
	  echo ; \
	  status=$$(egrep '^(SUCCESS|ISSUES) ' $@) ; \
	  echo $(VNAME) "$(DIRNAME) $$status" ; \
	} >> $@

# mail summary
SUMMARY.short: # SUMMARY
	{ \
	  [ -f $(SUM.d)/SUMMARY.diff ] && cat $(SUM.d)/SUMMARY.diff ; \
	  echo ; \
	  grep -v '^passed: ' SUMMARY ; \
	} > $@
	cp $@ $@.saved

# cleanup case duration before diffing
NOTIME	= perl -p -e 's/^((passed|failed|changed|timeout): .*) [0-9]+$$/$$1/'

.PHONY: archive
archive: SUMMARY $(DEST.d)
	cp SUMMARY $(DEST.d)/$(NOW) ; \
	$(RM) $(SUM.prev) ; \
	test -L $(SUM.last) && mv $(SUM.last) $(SUM.prev) ; \
	ln -s $(NOW.d)/$(NOW) $(SUM.last)
	-test -f $(SUM.prev) -a -f $(SUM.last) && { \
          $(NOTIME) $(SUM.prev) > /tmp/$$$$.prev ; \
          $(NOTIME) $(SUM.last) > /tmp/$$$$.last ; \
	  diff /tmp/$$$$.prev /tmp/$$$$.last | \
	  egrep -v '^([0-9,]+[acd][0-9,]+|---)$$' > $(SUM.d)/SUMMARY.diff ; \
	  $(RM) /tmp/$$$$.prev /tmp/$$$$.last ; }

# overall targets
.PHONY: parallel-clean
parallel-clean: $(TARGET:%=parallel-clean-%)
	$(RM) $(RESULTS) $(HEAD)

.PHONY: parallel-check
parallel-check: $(TARGET:%=parallel-check-%)

.PHONY: parallel-validate
parallel-validate: $(TARGET:%=parallel-validate-%)

.PHONY: parallel-unvalidate
parallel-unvalidate: $(TARGET:%=parallel-unvalidate-%)

.PHONY: clean
clean: simple-clean parallel-clean

.PHONY: cleaner
cleaner: clean parallel-unvalidate

# generic subdir parallel targets
parallel-clean-%:
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) -C $* clean ; exit 0

parallel-check-%: parallel-clean-%
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) RESULTS=../$(RESULTS) SUBDIR=$* -C $* inconsistencies ; \
	  exit 0

# type of validation, may be "out" or "test"
# - "test" requires the validation to be an SVN working copy.
#   it could also work with git with some hocus-pocus
# - "out" does not, but you must move out to test to accept afterwards.
TEST = test

parallel-validate-%: parallel-check-%
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) RESULTS=../$(RESULTS) LOCAL_CLEAN= -C $* validate-$(TEST) \
	  || echo "broken-directory: $*" >> $(RESULTS)

parallel-unvalidate-%:
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) -C $* unvalidate ; exit 0
