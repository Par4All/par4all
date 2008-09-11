# $Id$

FIND	= find . -name '.svn' -type d -prune -o

.PHONY: clean
clean:
	$(FIND) -name '*~' -type f -print0 \
	     -o -name 'core' -type f -print0 \
	     -o -name 'a.out' -type f -print0 \
	     -o -name '*.o' -type f -print0 | xargs -0 $(RM)
	$(FIND) -name '*.database' -type d -print0 \
	     -o -name 'validation_results.*' -type d -print0 | \
		xargs -0 $(RM) -r
	$(RM) */*.result/out properties.rc
	$(RM) -r RESULTS

# subdirectories to consider
TARGET	= $(shell grep '^[a-zA-Z]' defaults)
VOPT	= -v

# validate-all: all subdirectories?
# how to deal with private?

.PHONY: validate
validate: clean
	PIPS_MORE=cat pips_validate $(VOPT) -V $(PWD) -O RESULTS $(TARGET)

.PHONY: accept
accept:
	manual_accept $(TARGET)

# convenient pseudo-target for quick tests: make Hpfc.val
%.val: %
	test -d $< && $(MAKE) TARGET=$< validate
