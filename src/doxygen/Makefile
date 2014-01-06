# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of Linear/C3 Library.
#
# Linear/C3 Library is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.
#

ROOT    = ../..

LINEAR_SOURCE_DIR = $(ROOT)/src

# Some pure Doxygen parameters:
define DOXYGEN_PARAMETERS
# The files to document:
INPUT                  = $(LINEAR_SOURCE_DIR)
endef

PROJECT_NAME           = Linear

# We want to have other Doxygen documentation (the Linear/C3 Library one) to refer this
# doc, so ask Doxygen to put cross-information in this file. Indeed it is
# not used because if we want to build the caller graph we need to build
# the documentation at once.
GENERATE_TAGFILE_NAME       = Linear.doxygen-tag

DEFAULT_DOXYGEN_CONFIG = $(ROOT)/makes/share/doxygen/Doxyfile

include $(ROOT)/makes/main.mk
include $(ROOT)/makes/doxygen.mk

# If needed:
#make_destination_dir :
#	mkdir -p $(PUBLISH_LOCATION)

clean: local-clean

local-clean:
	$(RM) $(GENERATE_TAGFILE_NAME)
