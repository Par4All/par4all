# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#
FWD_DIRS	= src makes

# needed for PIPS_NO_TAGS
-include makes/config.mk

# default is to "build" (phase 0 to 6)
all: build

.PHONY: compile
compile:
	-test -d ./makes && $(MAKE) -C ./makes build
	@echo "### pips compile phase0 (depends)"
	$(MAKE) -C src phase0
	@echo "### pips compile phase1 (etc & py)"
	$(MAKE) -C src phase1
	@echo "### pips compile phase2 (include)"
	$(MAKE) -C src phase2
	@echo "### pips compile phase3 (nope?)"
	$(MAKE) -C src phase3
	@echo "### pips compile phase4 (lib)"
	$(MAKE) -C src phase4
	@echo "### pips compile phase5 (bin)"
	$(MAKE) -C src phase5
	#@echo "### pips compile phase6 (doc)"
	#$(MAKE) -C src phase5
ifndef PIPS_NO_TAGS
	@echo "### tags"
	$(MAKE) tags
endif

.PHONY: doc htdoc build full-build clean

# hmmm... not sure it is a good idea to go on errors.
doc: compile
	$(MAKE) -C src FWD_STOP_ON_ERROR= phase6

htdoc: doc
	$(MAKE) -C src FWD_STOP_ON_ERROR= phase7

# various convenient short-hands
build: doc
full-build: htdoc

# more documentation
doxygen: htdoc

# do not include dependencies for some target
clean: NO_INCLUDES=1
unbuild: NO_INCLUDES=1
export NO_INCLUDES

# Clean up everything below:
clean:
	$(MAKE) -C src clean

unbuild: clean tags-clean
	$(MAKE) -C src unbuild
	$(MAKE) -C makes unbuild

rebuild:
	$(MAKE) unbuild
	$(MAKE) build

recompile:
	$(MAKE) clean
	$(MAKE) compile

install:
	$(MAKE) -C src install

uninstall:unbuild

# all about tags, with temporary files
# should it generate tags only for src/?
ETAGS	= /tmp/etags.$$$$
TAGS:
	find $(CURDIR) -name '*.[chly]' -print0 | \
		xargs -0 ctags -e --append -o $(ETAGS) ; \
	mv $(ETAGS) TAGS

CTAGS	= /tmp/ctags.$$$$
CTAGS:
	find $(CURDIR) -name '*.[chly]' -print0 | \
		xargs -0 ctags --append -o $(CTAGS) ; \
	mv $(CTAGS) CTAGS

cscope.out:cscope-clean
	cd / && \
		find $(CURDIR) -name '*-local.h' -type f -prune -o -name include -type d -prune -o -name '*.[ch]' -print >  $(CURDIR)/cscope.files && \
		cd - && \
		cscope -b && \
		rm -f $(CURDIR)/cscope.files

cscope-clean:
	$(RM) cscope.out

# autoconf compilation
# --enable-doc --enable-devel-mode
BUILD.dir	= _build
HERE	:= $(shell pwd)
INSTALL.dir	= $(HERE)/../../install
DOWNLOAD.dir	= $(HERE)/../..
ifndef EXTERN_ROOT
EXTERN_ROOT	= $(HERE)/../extern
endif

# add ENABLE='doc devel-mode paws'
ENABLE	=

.PHONY: auto auto-comp auto-clean
auto-clean:
	$(RM) -r $(BUILD.dir) autom4te.cache
	$(RM) configure depcomp config.guess config.sub ltmain.sh \
	       config.h.in missing aclocal.m4 install-sh compile py-compile
	find . -name .svn -prune -o -name Makefile.in -print0 | xargs -0 rm -f

# initial configuration
$(BUILD.dir): 
	autoreconf -vi
	mkdir $(BUILD.dir) && cd $(BUILD.dir) ; \
	../configure --disable-static --prefix=$(INSTALL.dir) \
		PATH=$(INSTALL.dir)/bin:$$PATH \
		PKG_CONFIG_PATH=$(INSTALL.dir)/lib/pkgconfig:$(EXTERN_ROOT)/lib/pkgconfig \
		--enable-hpfc --enable-pyps --enable-fortran95 --enable-gpips \
		$(ENABLE:%=--enable-%)

# just compile
auto-comp: $(BUILD.dir)
	$(MAKE) -C $(BUILD.dir) DL.d=$(DOWNLOAD.dir)
	$(MAKE) -C $(BUILD.dir) install
	# manual fix...
	-[ -d $(BUILD.dir)/src/Scripts/validation ] && \
	  $(MAKE) -C $(BUILD.dir)/src/Scripts/validation install

# clean & compile
auto: auto-clean
	$(MAKE) auto-comp

# with paws
auto-paws:; $(MAKE) ENABLE=paws auto

# force tags target
tags: tags-clean
	$(MAKE) TAGS CTAGS

# Force recompilation if the user ask for explicit TAGS or CTAGS
.PHONY: tags TAGS CTAGS

# ARGH. I want both to forward and to clean locals...
#clean: tags-clean
tags-clean:
	$(RM) TAGS CTAGS
