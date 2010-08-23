# $Id$

default: clean

# I still keep the old "script" validation as the defaults
validate: old-validate
validate-all: old-validate-all
validate-%: old-validate-%

# useful variant to be consistent with intra-directory validation
validate-test: new-validate
validate-out:; $(MAKE) TEST=out new-validate

FIND	= find . -name '.svn' -type d -prune -o

.PHONY: full-clean
full-clean: clean
	$(FIND) -name 'SUMMARY_Archive' -type d -print0 \
	     -o -name 'RESULTS' -type d -print0 | \
	  xargs -0 $(RM) -r

.PHONY: clean
clean:
	$(MAKE) TARGET=. clean-target

# subdirectories to consider
TARGET	:= $(shell grep '^[a-zA-Z]' defaults)

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

# check for svn working copy early
.PHONY: check-run-consistency
check-run-consistency:
	@[ $(TEST) = 'out' ] && exit 0 ; \
	[ '$(TEST)' = 'test' -a -d .svn ] || { \
	  echo "TEST=test parallel validation requires svn" >&2 ; \
	  echo "try: make TEST=out <your arguments...>" >&2 ; \
	  exit 1 ; \
	}

# generate summary header
# hmmm... not sure that start date is before the validation
$(HEAD): check-run-consistency
	{ \
	  echo "parallel validation" ; \
	  echo "on: $(TARGET)" ; \
	  echo "host: $$(hostname)" ; \
	  echo "directory: $(PWD)" ; \
	  test -d .svn && \
	    echo " $(SVN.URL)@$(SVN.R) ($(SVN.C))" ; \
	  echo "pips: $(shell which pips)" ; \
	  pips -v ; \
	  echo "tpips: $(shell which tpips)" ; \
	  tpips -v ; \
	  echo "user: $$USERNAME" ; \
	  echo "start date: $$(date) [$$(date +%s)]" ; \
	} > $@

# this target should replace the "validate" target
.PHONY: new-validate
new-validate:
	$(RM) SUMMARY
	$(MAKE) parallel-clean
	$(MAKE) archive

.PHONY: mail-validate
mail-validate: new-validate
	{ \
	  [ -f $(SUM.d)/SUMMARY.diff ] && cat $(SUM.d)/SUMMARY.diff ; \
	  echo ; \
	  grep -v '^passed: ' SUMMARY ; \
	} | Mail -a "Reply-To: $(EMAIL)" -s "$(shell tail -1 SUMMARY)" $(EMAIL)

SUMUP	= pips_validation_summary.pl

# generate & archive validation summary
SUMMARY: $(HEAD) parallel-validate
	{ \
	  unset LANG LC_COLLATE ; \
	  cat $(HEAD) ; \
	  echo "end date: $$(date) [$$(date +%s)]" ; \
	  echo ; \
	  [ -f $(SUM.last) ] && last=$(SUM.last) ; \
          $(SUMUP) $(RESULTS) $$last ; \
	} > $@
	{ \
          echo ; \
	  sort -k 2 $(RESULTS) ; \
	  echo ; \
	  status=$$(egrep '^(SUCCEEDED|FAILED) ' $@) ; \
	  echo "validation $(shell arch) $$status ($(TARGET))" ; \
	} >> $@

.PHONY: archive
archive: SUMMARY $(DEST.d)
	cp SUMMARY $(DEST.d)/$(NOW) ; \
	$(RM) $(SUM.prev) ; \
	test -L $(SUM.last) && mv $(SUM.last) $(SUM.prev) ; \
	ln -s $(NOW.d)/$(NOW) $(SUM.last)
	-test -f $(SUM.prev) -a -f $(SUM.last) && \
	  diff $(SUM.prev) $(SUM.last) | \
	  egrep -v '^([0-9,]+[acd][0-9,]+|---)$$' > $(SUM.d)/SUMMARY.diff

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

# generic subdir parallel targets
parallel-clean-%:
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) -C $* clean unvalidate ; exit 0

parallel-check-%: parallel-clean-%
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) RESULTS=../$(RESULTS) SUBDIR=$* -C $* inconsistencies ; \
	  exit 0

# type of validation, may be "out" or "test"
# - "test" requires the validation to be an SVN working copy.
#   it could also work siwith git with some hocus-pocus
# - "out" does not, but you must move out to test to accept afterwards.
TEST = test

parallel-validate-%: parallel-check-%
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) RESULTS=../$(RESULTS) -C $* validate-$(TEST) \
	  || echo "broken-directory: $*" >> $(RESULTS)

parallel-unvalidate-%:
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) -C $* unvalidate ; exit 0
