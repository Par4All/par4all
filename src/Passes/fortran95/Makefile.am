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


bin_PROGRAMS=gfc2pips
gfc2pips_SOURCES = 
dist_noinst_DATA = patch-4.4.5.diff patch-4.4.3.diff patch-4.4.2.diff patch-4.4.1.diff src gfc2pips-autotools.mk gfc2pips-common.mk

# this is not defined with non-gnu make
RM = rm -f

include Makefile.mk

build-stage2:  $(BUILD.d)/.configure-fortran $(GFC2PIPS_SRCS)
	rsync --exclude=.svn -av $(srcdir)/src/ gcc-$(PIPS_GFC2PIPS_GCC_VERSION)/gcc/fortran/
	$(MAKE) "GFC2PIPS-AUTOTOOL-MAKEFILE=-autotools" "pipssrcdir=${shell (cd ${srcdir} && pwd)}" "NEWGENLIBS_CFLAGS=$(NEWGENLIBS_CFLAGS)" "LINEARLIBS_CFLAGS=$(LINEARLIBS_CFLAGS)" "PIPSLIBS_CFLAGS=$(PIPSLIBS_CFLAGS)" "NEWGENLIBS_LIBS=$(NEWGENLIBS_LIBS)" "LINEARLIBS_LIBS=$(LINEARLIBS_LIBS)" -C build/$(PIPS_GFC2PIPS_GCC_VERSION)/gcc f951


gfc2pips: build
	cp build/$(PIPS_GFC2PIPS_GCC_VERSION)/gcc/f951 ./gfc2pips

# this removes all the temporary repositories
clean-local:
	-chmod -R u+rw $(SRC.d) build
	rm -rf $(SRC.d) build

# this removes all the donwloaded files
distclean-local:
	-chmod -R u+rw  $(GCC_CORE_ARCHIVE) $(GCC_FORTRAN_ARCHIVE) $(GCC_MD5) 
	rm -rf  $(GCC_CORE_ARCHIVE) $(GCC_FORTRAN_ARCHIVE) $(GCC_MD5)
