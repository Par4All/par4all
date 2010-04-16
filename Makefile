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

# directory-parallel validation test
# may replace the previous entry some day

.PHONY: parallel-validate parallel-clean

RESULTS	= validation.out
HEAD	= validation.head

# SUMMARY:
SUM.d	= SUMMARY_Archive
NOW.d	:= $(shell date +%Y/%m)
DEST.d	= $(SUM.d)/$(NOW.d)
NOW	:= $(shell date +%Y-%m-%d_%H_%M_%S)

$(DEST.d):
	mkdir -p $@

$(HEAD):
	{ \
	  echo "validation for $(TARGET)" ; \
	  echo "host: $$(hostname)" ; \
	  echo "pips: $(shell which pips)" ; \
	  pips -v ; \
	  echo "tpips: $(shell which tpips)" ; \
	  tpips -v ; \
	  echo "user: $$USERNAME" ; \
	  echo "date: $$(date)" ; \
	} > $@

# TODO
# skipped?

summary: validation.head parallel-validate $(DEST.d)
	{ \
	  cat validation.head ; \
	  echo ; \
	  sort -k 2 $(RESULTS) ; \
	  echo ; \
	  echo $$(wc -l < validation.out) "failed out of ... on " $$(date) ; \
	} > $(DEST.d)/$(NOW)
	$(RM) $(SUM.d)/SUMMARY-previous
	test -f $(SUM.d)/SUMMARY-last && \
	  mv $(SUM.d)/SUMMARY-last $(SUM.d)/SUMMARY-previous ; \
	ln -s $(NOW.d)/$(NOW) $(SUM.d)/SUMMARY-last

parallel-clean: $(TARGET:%=parallel-clean-%)
	$(RM) $(RESULTS) $(HEAD)

parallel-validate: $(TARGET:%=parallel-validate-%)
	# TODO generate summary
	# TODO archive summary

parallel-clean-%:
	$(MAKE) -C $* clean unvalidate

parallel-validate-%: parallel-clean-%
	$(MAKE) RESULTS=../$(RESULTS) -C $* validate-test

parallel-unvalidate: $(TARGET:%=parallel-unvalidate-%)

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
