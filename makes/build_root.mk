# $Id$

FWD_DIRS	= src

build:
	$(MAKE) -C src phase0
	$(MAKE) -C src phase1
	$(MAKE) -C src phase2
	$(MAKE) -C src phase3
	$(MAKE) -C src phase4
	$(MAKE) -C src phase5

unbuild: clean
	$(RM) -r ./Bin ./Include ./Lib ./Share ./Utils ./Doc ./Runtime TAGS

# todo: install, uninstall
install:
	@echo "NOT IMPLEMENTED YET"

uninstall:
	@echo "NOT IMPLEMENTED YET"

local-clean:
	$(RM) TAGS

# temporary file
TAGS	= /tmp/tags.$$$$

tags:
	find $(CURDIR) -name '*.[chly]' -print | xargs etags -o $(TAGS) ; \
	mv $(TAGS) TAGS
