# $Id$

default: clean

# The option used by default for validating. Can be overridden by the
# command line "make VOPT=..." or the environment variable:
VOPT	= -v --archive --diff

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

.PHONY: validate
validate: clean-target
	$(RM) -r RESULTS
	PIPS_MORE=cat pips_validate $(VOPT) -V $(PWD) -O RESULTS $(TARGET)

.PHONY: accept
accept:
	pips_manual_accept $(TARGET)

# validate one sub directory
validate-%: %
	# test -d $< && $(MAKE) TARGET=$< validate
	[ -d $< ] && { \
	  $(MAKE) TARGET=$< clean-target ; \
	  cd $< ; \
	  $(RM) -r RESULTS ; \
	  PIPS_MORE=cat pips_validate $(VOPT) -V $(PWD)/$< -O RESULTS . ; \
	}

#
# PARALLEL VALIDATION
#
# directory-parallel validation test
# may replace the previous entry some day

.PHONY: parallel-validate parallel-clean parallel-unvalidate parallel-check

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

# generate summary header
# hmmm... not sure that start date is before the validation
$(HEAD):
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
	  echo "start date: $$(date)" ; \
	} > $@

# this target should replace the "validate" target
new-validate:
	$(RM) SUMMARY
	$(MAKE) parallel-clean
	$(MAKE) archive

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
	  echo "end date: $$(date)" ; \
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

archive: SUMMARY $(DEST.d)
	cp SUMMARY $(DEST.d)/$(NOW) ; \
	$(RM) $(SUM.prev) ; \
	test -L $(SUM.last) && mv $(SUM.last) $(SUM.prev) ; \
	ln -s $(NOW.d)/$(NOW) $(SUM.last)
	-test -f $(SUM.prev) -a -f $(SUM.last) && \
	  diff $(SUM.prev) $(SUM.last) | \
	  egrep -v '^([0-9,]+[acd][0-9,]+|---)$$' > $(SUM.d)/SUMMARY.diff

# overall targets
parallel-clean: $(TARGET:%=parallel-clean-%)
	$(RM) $(RESULTS) $(HEAD)

parallel-check: $(TARGET:%=parallel-check-%)

parallel-validate: $(TARGET:%=parallel-validate-%)

parallel-unvalidate: $(TARGET:%=parallel-unvalidate-%)

# generic subdir targets
parallel-clean-%:
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) -C $* clean unvalidate ; exit 0

parallel-check-%: parallel-clean-%
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) RESULTS=../$(RESULTS) SUBDIR=$* -C $* inconsistencies ; exit 0

parallel-validate-%: parallel-check-%
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) RESULTS=../$(RESULTS) -C $* validate-test \
	  || echo "broken-directory: $*" >> $(RESULTS)

parallel-unvalidate-%:
	[ -d $* -a -f $*/Makefile ] \
	  && $(MAKE) -C $* unvalidate ; exit 0

# validate all subdirectories
ALL	= $(wildcard * private/*)
ALL.d	= $(shell for d in $(ALL) ; do test -d $$d && echo $$d ; done)
validate-all:
	$(MAKE) TARGET="$(ALL.d)" validate
