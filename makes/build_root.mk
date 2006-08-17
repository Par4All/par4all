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
	$(MAKE) -C src phase6

htdoc:
	$(MAKE) -C src phase7

build: compile doc
full-build: build htdoc

# do not include dependencies for some target
clean: NO_INCLUDES=1
export NO_INCLUDES

unbuild: clean tags-clean
	$(RM) -rf \
		./bin ./include ./lib ./share ./utils \
		./doc ./runtime ./etc ./html

install:
	@echo "try 'build' target"

uninstall:
	@echo "try 'unbuild' target"

# all about tags
# use a temporary file
TAGS	= /tmp/tags.$$$$

# generate tag file
TAGS:
	find $(CURDIR) -name '*.[chly]' -print0 | \
		xargs -0 etags --append --output=$(TAGS) ; \
	mv $(TAGS) TAGS

# force tags target
tags: tags-clean
	$(MAKE) TAGS

# ARGH. I want both to forward and to clean locals...
#clean: tags-clean

tags-clean:
	$(RM) TAGS
