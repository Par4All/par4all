# $Id: Makefile 1282 2009-08-11 10:16:01Z coelho $
#
# Copyright 1989-2009 MINES ParisTech
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

ROOT    = ../..

LINEAR_SOURCE_DIR = $(ROOT)/src

# Some pure Doxygen parameters:
define DOXYGEN_PARAMETERS
# The files to document:
INPUT                  = $(LINEAR_SOURCE_DIR)
endef

PROJECT_NAME           = Linear

# We want to have other Doxygen documentation (the PIPS one) to refer this
# doc, so ask Doxygen to put cross-information in this file. Indeed it is
# not used because if we want to build the caller graph we need to build
# the documentation at once.
GENERATE_TAGFILE_NAME       = Linear.doxygen-tag

DEFAULT_DOXYGEN_CONFIG = $(ROOT)/makes/share/doxygen/Doxyfile

nodist_include $(ROOT)/makes/main.mk
nodist_include $(ROOT)/makes/doxygen.mk

# If needed:
#make_destination_dir :
#	mkdir -p $(PUBLISH_LOCATION)

clean: local-clean

local-clean:
	$(RM) $(GENERATE_TAGFILE_NAME)
