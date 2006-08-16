# $Id$

FWD_DIRS	= src

build: compile
	$(MAKE) -C src phase6

compile:
	-test -d ./makes && $(MAKE) -C makes build
	$(MAKE) -C src phase0
	$(MAKE) -C src phase1
	$(MAKE) -C src phase2
	$(MAKE) -C src phase3
	$(MAKE) -C src phase4
	$(MAKE) -C src phase5

# do not include dependencies for some target
clean: NO_INCLUDES=1
export NO_INCLUDES

unbuild: clean
	$(RM) -rf \
		./bin ./include ./lib ./share ./utils \
		./doc ./runtime ./etc ./html
	$(RM) -f TAGS

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
tags:
	$(RM) TAGS
	$(MAKE) TAGS

clean: tags-clean

tags-clean:
	$(RM) TAGS
