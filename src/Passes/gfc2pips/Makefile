# $Id: Makefile 16070 2010-02-07 10:54:43Z amini $
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

# Gfc2pips source files
GFC2PIPS_SRC.d = src
GFC2PIPS_SRCS = $(GFC2PIPS_SRC.d)/gfc2pips.c $(GFC2PIPS_SRC.d)/gfc2pips_stubs.c
#VERSION USED
GCC_VERSION = 4.4.3
BUILD.d	= build/$(GCC_VERSION)
TARGET	= f951


ARCHIVE_EXT=.tar.bz2
#ARCHIVE_EXT=".tar.gz"

# MIRRORS
GCC_MIRROR = ftp://ftp.irisa.fr/pub/mirrors/gcc.gnu.org
GCC_MD5_MIRROR = ftp://ftp.uvsq.fr/pub

# URL for GCC mirror repository
GCC_URL = $(GCC_MIRROR)/gcc/releases/gcc-$(GCC_VERSION)
GCC_MD5_URL = $(GCC_MD5_MIRROR)/gcc/releases/gcc-$(GCC_VERSION)


#archive name
DL.d	= .
GCC_CORE_ARCHIVE = $(DL.d)/gcc-core-$(GCC_VERSION)$(ARCHIVE_EXT)
GCC_FORTRAN_ARCHIVE = $(DL.d)/gcc-fortran-$(GCC_VERSION)$(ARCHIVE_EXT)
GCC_MD5	= $(DL.d)/gcc-$(GCC_VERSION).md5

#after untar
SRC.d		= gcc-$(GCC_VERSION)
CONFIGURE_OPT 	= --disable-bootstrap --enable-languages=fortran \
	--enable-stage1-languages=fortran --disable-libssp --disable-libada \
	--disable-libgomp --disable-stage1-checking --without-ppl \
	--without-cloog --disable-multilib --disable-checking \
	--with-build-config='bootstrap-O1'

#file to flag if sources have been patched
PATCHED = $(SRC.d)/.patched

.PHONY: all build clean local-clean untar install configure \
        md5.sum md5-check-core md5-check-fortran md5-check patch

all: install

patch : $(PATCHED)

# local stuff
unbuild: clean
clean: local-clean
	$(RM) $(ROOT)/bin/f951

local-clean:
	$(RM) -r $(SRC.d) build

$(SRC.d)/.dir:
	mkdir -p $(SRC.d)
	touch $@

WGET	= wget -q -O

# Get archive for GCC source from mirrors with wget
$(GCC_FORTRAN_ARCHIVE):
	@echo "**** Getting GCC fortran specifs ****"
	$(WGET) $@ $(GCC_URL)/$@
	@touch $@

$(GCC_CORE_ARCHIVE):
	@echo "**** Getting GCC core ****"
	$(WGET) $@ $(GCC_URL)/$@
	@touch $@


# Get md5 checksum for GCC source archive
$(GCC_MD5):
	@echo "**** Getting md5 checksum source file ****"
	$(WGET) $@ $(GCC_MD5_URL)/md5.sum


# check md5 on core source archive
md5-check-core: $(SRC.d)/.md5-check-core
$(SRC.d)/.md5-check-core: $(GCC_CORE_ARCHIVE) $(GCC_MD5) $(SRC.d)/.dir
	@echo "**** Check md5 checksum for gcc-core ****"
	grep $(notdir $(GCC_CORE_ARCHIVE)) $(GCC_MD5) > $@.list
	md5sum -c $@.list
	$(RM) $@.list
	@touch $@

# check md5 on fortran source archive
md5-check-fortran: $(SRC.d)/.md5-check-fortran
$(SRC.d)/.md5-check-fortran : $(GCC_FORTRAN_ARCHIVE) $(GCC_MD5) $(SRC.d)/.dir
	@echo "**** Check md5 checksum for gcc-fortran ****"
	grep $(notdir $(GCC_FORTRAN_ARCHIVE)) $(GCC_MD5) > $@.list
	md5sum -c $@.list
	$(RM) $@.list
	@touch $@

md5-check: md5-check-fortran md5-check-core

untar: $(SRC.d)/.untar

$(SRC.d)/.untar: $(SRC.d)/.untar-core $(SRC.d)/.untar-fortran
	touch $@

$(SRC.d)/.untar-core: $(SRC.d)/.md5-check-core
	@echo "**** Unpacking GCC-core ****"
	tar -xjf $(GCC_CORE_ARCHIVE)
	$(RM) $(PATCHED)
	@touch $@

$(SRC.d)/.untar-fortran: $(SRC.d)/.md5-check-fortran
	@echo "**** Unpacking GCC-fortran ****"
	tar -xjf $(GCC_FORTRAN_ARCHIVE)
	$(RM) $(PATCHED)
	@touch $@

# patch gcc sources
$(PATCHED): patch-$(GCC_VERSION).diff $(SRC.d)/.untar
	patch -d gcc-$(GCC_VERSION) -p0 < $<
	@touch $@

$(BUILD.d)/.dir:
	mkdir -p $(BUILD.d)/gcc
	@touch $@

$(BUILD.d)/.configure-core: $(BUILD.d)/.dir $(PATCHED)
	cd $(BUILD.d) && \
	../../gcc-$(GCC_VERSION)/configure $(CONFIGURE_OPT)
	@touch $@

$(BUILD.d)/.configure-fortran: $(BUILD.d)/.stage1
	cd $(BUILD.d)/gcc && \
	../../../gcc-$(GCC_VERSION)/gcc/configure $(CONFIGURE_OPT)
	@touch $@

$(BUILD.d)/.stage1: $(BUILD.d)/.configure-core
	$(MAKE) -C $(BUILD.d) \
		STAGE1_LANGUAGES=fortran LANGUAGES=fortran \
	        STAGE1_CFLAGS=-O1 CFLAGS=-O1 \
	        maybe-all-build-fixincludes \
	        maybe-all-libdecnumber \
	        maybe-all-libcpp
	@touch $@


# useful targets
configure: configure-core configure-fortran
configure-core: $(BUILD.d)/.configure-core
configure-fortran: $(BUILD.d)/.configure-fortran
build-stage1: $(BUILD.d)/.stage1
build-stage2:  $(BUILD.d)/.configure-fortran $(GFC2PIPS_SRCS)
	rsync --exclude=.svn -av src/ gcc-$(GCC_VERSION)/gcc/fortran/
	$(MAKE) -C build/$(GCC_VERSION)/gcc f951


compile: build-stage2
build: compile
install: compile
	cp -a $(BUILD.d)/gcc/f951 $(ROOT)/bin/
phase0:
phase1:
phase2:
phase3:
phase4:
phase5: install
phase6:
