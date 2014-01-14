# $Id$
#
# Copyright 1989-2014 MINES ParisTech
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




# Gfc2pips source files
GFC2PIPS_SRC.d = src
GFC2PIPS_SRCS = $(GFC2PIPS_SRC.d)/gfc2pips.c $(GFC2PIPS_SRC.d)/gfc2pips-stubs.c
#VERSION USED
PIPS_GFC2PIPS_GCC_VERSION = 4.4.5
BUILD.d	= build/$(PIPS_GFC2PIPS_GCC_VERSION)

ARCHIVE_EXT=.tar.bz2
#ARCHIVE_EXT=".tar.gz"

# MIRRORS
GCC_MIRROR = ftp://ftp.irisa.fr/pub/mirrors/gcc.gnu.org
GCC_MD5_MIRROR = ftp://ftp.uvsq.fr/pub

# URL for GCC mirror repository
GCC_URL = $(GCC_MIRROR)/gcc/releases/gcc-$(PIPS_GFC2PIPS_GCC_VERSION)
GCC_MD5_URL = $(GCC_MD5_MIRROR)/gcc/releases/gcc-$(PIPS_GFC2PIPS_GCC_VERSION)


#archive name
# overriding this variable from the root Makefile does not seem to work?
# DL.d=.
DL.d	= ../../../..
GCC_CORE_ARCHIVE = gcc-core-$(PIPS_GFC2PIPS_GCC_VERSION)$(ARCHIVE_EXT)
GCC_FORTRAN_ARCHIVE = gcc-fortran-$(PIPS_GFC2PIPS_GCC_VERSION)$(ARCHIVE_EXT)
GCC_MD5	= gcc-$(PIPS_GFC2PIPS_GCC_VERSION).md5

#after untar
SRC.d		= gcc-$(PIPS_GFC2PIPS_GCC_VERSION)
CONFIGURE_OPT 	= --disable-bootstrap --enable-languages=fortran \
	--enable-stage1-languages=fortran --disable-libssp --disable-libada \
	--disable-libgomp --disable-stage1-checking --without-ppl \
	--without-cloog --disable-multilib --disable-checking \
	--with-build-config='bootstrap-O1' CFLAGS="-O0 -g"

#file to flag if sources have been patched
PATCHED = $(SRC.d)/.patched

.PHONY: all build clean local-clean untar install configure \
        md5.sum md5-check-core md5-check-fortran md5-check patch

all: install

patch : $(PATCHED)

# local stuff
unbuild:
	$(RM) -rf $(ROOT)/bin/gfc2pips build
clean: local-clean unbuild

# do *NOT* remove downloaded files in clean & unbuild targets.
# FC 2010-05-25
local-clean:
	$(RM) -rf $(SRC.d)
#$(RM) -f $(GCC_MD5) $(GCC_CORE_ARCHIVE) $(GCC_FORTRAN_ARCHIVE) $(SRC.d)

$(SRC.d)/.dir:
	mkdir -p $(SRC.d)
	touch $(SRC.d)/.dir

# Get archive for GCC source from mirrors with wget
$(DL.d)/$(GCC_FORTRAN_ARCHIVE):
	@echo "**** Getting GCC fortran specifs ****"
	$(WGET) -q -O  $(DL.d)/$(GCC_FORTRAN_ARCHIVE) $(GCC_URL)/$(GCC_FORTRAN_ARCHIVE)
	@touch $(DL.d)/$(GCC_FORTRAN_ARCHIVE)

$(DL.d)/$(GCC_CORE_ARCHIVE):
	@echo "**** Getting GCC core ****"
	$(WGET) -q -O  $(DL.d)/$(GCC_CORE_ARCHIVE) $(GCC_URL)/$(GCC_CORE_ARCHIVE)
	@touch $(DL.d)/$(GCC_CORE_ARCHIVE)

# Get md5 checksum for GCC source archive
$(DL.d)/$(GCC_MD5):
	@echo "**** Getting md5 checksum source file ****"
	$(WGET) -q -O  $(DL.d)/$(GCC_MD5) $(GCC_MD5_URL)/md5.sum

# check md5 on core source archive
md5-check-core: $(SRC.d)/.md5-check-core
$(SRC.d)/.md5-check-core: $(DL.d)/$(GCC_CORE_ARCHIVE) $(DL.d)/$(GCC_MD5) $(SRC.d)/.dir
	@echo "**** Check md5 checksum for gcc-core ****"
	grep `basename $(DL.d)/$(GCC_CORE_ARCHIVE)` $(DL.d)/$(GCC_MD5) > $(SRC.d)/.md5-check-core.list
	sed -ie 's#$(GCC_CORE_ARCHIVE)#$(DL.d)/$(GCC_CORE_ARCHIVE)#' $(SRC.d)/.md5-check-core.list
	$(MD5SUM) -c $(SRC.d)/.md5-check-core.list
	$(RM) $(SRC.d)/.md5-check-core.list
	@touch $(SRC.d)/.md5-check-core

# check md5 on fortran source archive
md5-check-fortran: $(SRC.d)/.md5-check-fortran
$(SRC.d)/.md5-check-fortran : $(DL.d)/$(GCC_FORTRAN_ARCHIVE) $(DL.d)/$(GCC_MD5) $(SRC.d)/.dir
	@echo "**** Check md5 checksum for gcc-fortran ****"
	grep `basename $(DL.d)/$(GCC_FORTRAN_ARCHIVE)` $(DL.d)/$(GCC_MD5) > $(SRC.d)/.md5-check-fortran.list
	sed -ie 's#$(GCC_FORTRAN_ARCHIVE)#$(DL.d)/$(GCC_FORTRAN_ARCHIVE)#' $(SRC.d)/.md5-check-fortran.list
	$(MD5SUM) -c $(SRC.d)/.md5-check-fortran.list
	$(RM) $(SRC.d)/.md5-check-fortran.list
	@touch $(SRC.d)/.md5-check-fortran

md5-check: md5-check-fortran md5-check-core

untar: $(SRC.d)/.untar

$(SRC.d)/.untar: $(SRC.d)/.untar-core $(SRC.d)/.untar-fortran
	touch $(SRC.d)/.untar

$(SRC.d)/.untar-core: $(SRC.d)/.md5-check-core patch-$(PIPS_GFC2PIPS_GCC_VERSION).diff
	@echo "**** Unpacking GCC-core ****"
	tar -xjf $(DL.d)/$(GCC_CORE_ARCHIVE)
	$(RM) $(PATCHED)
	@touch $(SRC.d)/.untar-core

$(SRC.d)/.untar-fortran: $(SRC.d)/.md5-check-fortran patch-$(PIPS_GFC2PIPS_GCC_VERSION).diff
	@echo "**** Unpacking GCC-fortran ****"
	tar -xjf $(DL.d)/$(GCC_FORTRAN_ARCHIVE)
	$(RM) $(PATCHED)
	@touch $(SRC.d)/.untar-fortran

# patch gcc sources
$(PATCHED): patch-$(PIPS_GFC2PIPS_GCC_VERSION).diff $(SRC.d)/.untar
	patch -d gcc-$(PIPS_GFC2PIPS_GCC_VERSION) -p0 < $<
	@touch $(PATCHED)

$(BUILD.d)/.dir:
	mkdir -p $(BUILD.d)/gcc
	@touch $(BUILD.d)/.dir

$(BUILD.d)/.configure-core: $(BUILD.d)/.dir $(PATCHED)
	cd $(BUILD.d) && \
	../../gcc-$(PIPS_GFC2PIPS_GCC_VERSION)/configure $(CONFIGURE_OPT)
	@touch $(BUILD.d)/.configure-core

$(BUILD.d)/.configure-fortran: $(BUILD.d)/.stage1
	cd $(BUILD.d)/gcc && \
	../../../gcc-$(PIPS_GFC2PIPS_GCC_VERSION)/gcc/configure $(CONFIGURE_OPT)
	@touch $(BUILD.d)/.configure-fortran

$(BUILD.d)/.stage1: $(BUILD.d)/.configure-core
	$(MAKE) -C $(BUILD.d) \
		STAGE1_LANGUAGES=fortran LANGUAGES=fortran \
	        STAGE1_CFLAGS="-O1 -g" CFLAGS="-O1 -g" \
	        maybe-all-build-fixincludes \
	        maybe-all-libdecnumber \
	        maybe-all-libcpp
	@touch $(BUILD.d)/.stage1

# useful targets
configure: configure-core configure-fortran
configure-core: $(BUILD.d)/.configure-core
configure-fortran: $(BUILD.d)/.configure-fortran
build-stage1: $(BUILD.d)/.stage1
build: build-stage2

