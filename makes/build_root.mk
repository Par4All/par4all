# $Id$

FWD_DIRS	= src makes

# default is to "build"
all: build

compile:
	-test -d ./makes && $(MAKE) -C makes build
	$(MAKE) -C src phase0
	$(MAKE) -C src phase1
	$(MAKE) -C src phase2
	$(MAKE) -C src phase3
	$(MAKE) -C src phase4
	$(MAKE) -C src phase5

doc:
	$(MAKE) -C src FWD_STOP_ON_ERROR= phase6

htdoc:
	$(MAKE) -C src FWD_STOP_ON_ERROR= phase7

build: compile doc
full-build: build htdoc

# do not include dependencies for some target
clean: NO_INCLUDES=1
export NO_INCLUDES

# Clean up everything below:
clean:
	$(MAKE) -C src clean

unbuild: clean tags-clean
	$(RM) -rf \
		./bin ./include ./lib ./share ./utils \
		./doc ./runtime ./etc ./html

install:
	@echo "try 'build' target"

uninstall:
	@echo "try 'unbuild' target"

# all about tags, with temporary files
# should it generate tags only for src/?
ETAGS	= /tmp/etags.$$$$
TAGS:
	find $(CURDIR) -name '*.[chly]' -print0 | \
		xargs -0 etags --append -o $(ETAGS) ; \
	mv $(ETAGS) TAGS

CTAGS	= /tmp/ctags.$$$$
CTAGS:
	find $(CURDIR) -name '*.[chly]' -print0 | \
		xargs -0 ctags --append --output=$(CTAGS) ; \
	mv $(CTAGS) CTAGS

# force tags target
tags: tags-clean
	$(MAKE) TAGS CTAGS

# ARGH. I want both to forward and to clean locals...
#clean: tags-clean
tags-clean:
	$(RM) TAGS CTAGS
