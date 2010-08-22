# $Id$

# The option used by default for validating. Can be overridden by the
# command line "make VOPT=..." or the environment variable:
VOPT	= -v --archive --diff

.PHONY: validate
validate: clean-target
	$(RM) -r RESULTS
	PIPS_MORE=cat pips_validate $(VOPT) -V $(PWD) -O RESULTS $(TARGET)

# validate all subdirectories
ALL	= $(wildcard *)
ALL.d	= $(shell for d in $(ALL) ; do test -d $$d && echo $$d ; done)
.PHONY: validate-all
validate-all:
	$(MAKE) TARGET="$(ALL.d)" validate

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
