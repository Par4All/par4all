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

# extract private (restricted access) validation
.PHONY: private
private:
	if [ -d private/. ] ; then \
	  if [ -d private/.svn ] ; then \
	    svn up private/ ; \
	  else \
	    echo "ERROR: cannot update private" >&2 ; \
	  fi ; \
	else \
	  echo "checkout the private validation somewhere:"; \
	  echo "> svn co https://svnpriv.cri.ensmp.fr/svn/pipspriv/trunk ???"; \
	  echo "and link it here as 'private':"; \
	  echo "> ln -s ??? private"; \
	  echo "CAUTION: it MUST NOT be distributed..." ; \
	fi

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

.PHONY: parallel-validate parallel-clean parallel-unvalidate parallel-skipped

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
	  cat $(SUM.d)/SUMMARY.diff ; \
	  echo ; \
	  cat SUMMARY ; \
	  echo "end date: $$(date)" ; \
	} | Mail -a "Reply-To: $(EMAIL)" -s "$(shell tail -1 SUMMARY)" $(EMAIL)

# generate & archive validation summary
SUMMARY: validation.head parallel-validate
	{ \
	  cat $(HEAD) ; \
	  echo ; \
	  grep -v '^passed: ' < $(RESULTS) | sort -k 2 ; \
	  echo ; \
	  failed=$$(egrep -v '^(skipp|pass)ed: ' < $(RESULTS) | wc -l); \
	  total=$$(grep -v 'skipped: ' < $(RESULTS) | wc -l); \
	  [ $failed = 0 ] && \
		status="SUCCEEDED $total" || \
		status="FAILED $failed/$total"; \
	  echo "$failed failed out of $total on $$(date)"; \
	  echo "validation $(shell arch) $status ($(TARGET))" ; \
	} > $@

archive: SUMMARY $(DEST.d)
	cp SUMMARY $(DEST.d)/$(NOW) ; \
	$(RM) $(SUM.prev) ; \
	test -L $(SUM.last) && mv $(SUM.last) $(SUM.prev) ; \
	ln -s $(NOW.d)/$(NOW) $(SUM.last) ; \
	test -L $(SUM.prev) -a -L $(SUM.last) && \
	  diff $(SUM.prev) $(SUM.last) > $(SUM.d)/SUMMARY.diff

# overall targets
parallel-clean: $(TARGET:%=parallel-clean-%)
	$(RM) $(RESULTS) $(HEAD)

parallel-skipped: $(TARGET:%=parallel-skipped-%)

parallel-validate: $(TARGET:%=parallel-validate-%)

parallel-unvalidate: $(TARGET:%=parallel-unvalidate-%)

# generic subdir targets
parallel-clean-%:
	$(MAKE) -C $* clean unvalidate

parallel-skipped-%: parallel-clean-%
	$(MAKE) RESULTS=../$(RESULTS) SUBDIR=$* -C $* skipped

parallel-validate-%: parallel-skipped-%
	$(MAKE) RESULTS=../$(RESULTS) -C $* validate-test

parallel-unvalidate-%:
	$(MAKE) -C $* unvalidate

## REMOVE ???
# special handling of private
PRIV	= $(wildcard private/*)
PRIV.d	= $(shell for d in $(PRIV) ; do test -d $$d && echo $$d ; done)
validate-private:
	$(MAKE) TARGET="$(PRIV.d)" validate

# validate all subdirectories
ALL	= $(wildcard * private/*)
ALL.d	= $(shell for d in $(ALL) ; do test -d $$d && echo $$d ; done)
validate-all:
	$(MAKE) TARGET="$(ALL.d)" validate
