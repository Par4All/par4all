# $Id$
#
# Copyright 1989-2010 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

ROOT	= ../../..

TARGET	= gfc2pips

WGET = wget

# md5sum seems not universally available
MD5SUM = openssl md5

srcdir = .


include Makefile.mk

build-stage2:  $(BUILD.d)/.configure-fortran $(GFC2PIPS_SRCS)
	rsync --exclude=.svn -av $(srcdir)/src/ gcc-$(PIPS_GFC2PIPS_GCC_VERSION)/gcc/fortran/
	$(MAKE) -C build/$(PIPS_GFC2PIPS_GCC_VERSION)/gcc f951

compile: build
install: build
	cp -a $(BUILD.d)/gcc/f951 $(ROOT)/bin/gfc2pips
phase0:
phase1:
phase2:
phase3:
phase4:
phase5: install
phase6:
phase7:

doxygen-publish:
