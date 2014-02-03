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

# make stuff specific to the pips project

include $(ROOT)/libraries.make

# issue about bad header files.
ifdef INC_TARGET

# force it again as the first pass was partly wrong
# because of cross dependencies between library headers
# however, do not repeat that every time...
phase3: .build_inc_second_pass

.build_inc_second_pass:
	$(MAKE) build-header-file
	$(RM) .build_inc ; $(MAKE) .build_inc
	touch $@

clean: pips-phase3-clean
pips-phase3-clean:
	$(RM) .build_inc_second_pass

endif # INC_TARGET

ifdef BIN_TARGET
include $(ROOT)/makes/pips-bin.mk
endif # BIN_TARGET

ifdef LIB_TARGET
ifdef OLD_TEST

# build pips executables on request?
include $(ROOT)/makes/pips-old.mk

ifndef BIN_TARGET
include $(ROOT)/makes/pips-bin.mk
endif # BIN_TARGET

else # not OLD_TEST

# simply link to actual executable, helpful for gdb
ifndef BIN_TARGET
$(ARCH)/tpips:
	$(RM) $@
	ln -s $(ROOT)/bin/$@ $@

$(ARCH)/pips:
	$(RM) $@
	ln -s $(ROOT)/bin/$@ $@

# full recompilation from a library
full: $(ARCH)/tpips $(ARCH)/pips
	$(MAKE) -C $(ROOT) compile

# fast tpips recompilation
fast-tpips: $(ARCH)/tpips compile
	$(MAKE) -C $(ROOT)/src/Passes/tpips compile

# fast pips recompilation
fast-pips: $(ARCH)/pips compile
	$(MAKE) -C $(ROOT)/src/Passes/pips compile

# generate both pips and tpips, useful for validation
fast: fast-tpips fast-pips

# shortcut for auto-pips users
auto-comp:
	echo $(MAKE) -C $(ROOT) auto-comp
# todo: auto-fast:

# helper with old targets
test ttest ftest:
	@$(ECHO) -e "\a\n\ttry 'fast' (just link) or 'full' (recompilation)\n"

endif # BIN_TARGET
endif # OLD_TEST
endif # LIB_TARGET

ifdef VALIDATE_TARGET

ifdef PIPS_VALIDDIR
VALID.dir	= $(PIPS_VALIDDIR)
else # no PIPS_VALIDDIR
VALID.dir	= $(ROOT)/../../validation
endif # PIPS_VALIDDIR

validate: fast
	cd $(VALID.dir) ; \
	$(MAKE) clean ; \
	$(MAKE) TARGET=$(VALIDATE_TARGET) validate ;

endif # VALIDATE_TARGET
