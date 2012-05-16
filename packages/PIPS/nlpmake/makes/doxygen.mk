# $Id$
#
# Copyright 1989-2012 MINES ParisTech
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


# Here are the high level methods to use:

# We generate 2 versions, one without callers/callees graphs, and another full-fledged heavy one:
doxygen :: doxygen-plain doxygen-graph

ifdef DOXYGEN_PUBLISH_LOCATION

doxygen-publish:: make_destination_dir
	# Synchonize stuff and ignore errors since there may be some access
	# right problems:
	-$(RSYNC) plain/html/ $(DOXYGEN_PUBLISH_LOCATION)/plain
	-$(RSYNC) graph/html/ $(DOXYGEN_PUBLISH_LOCATION)/graph

# Just to avoid publish to complaining if not implemented in the including
# Makefile:
make_destination_dir :

# A default implementation method of the previous one, to be used as:
# make_destination_dir: default_destination_dir
default_destination_dir :
	ssh $(INSTALL_MACHINE) mkdir -p $(INSTALL_MACHINE_DOC_DIR)/$(PROJECT_NAME)
endif

clean:
	$(RM) -r plain graph






# Now the implementation details:

# where are make files
MAKE.d	= $(ROOT)/makes

DOXYGEN_GENERATED_DIR = html latex

include $(MAKE.d)/doc.mk

# Where we want the documentation to be published if not redefined in the
# environment:
DOXYGEN_PUBLISH_LOCATION_DIR ?= doxygen.pips.enstb.org:/var/www/pips/doxygen
DOXYGEN_PUBLISH_LOCATION ?= $(DOXYGEN_PUBLISH_LOCATION_DIR)/$(PROJECT_NAME)

# The configuration stuff:
DEFAULT_DOXYGEN_DIR ?= $(ROOT)/makes/share/doxygen
DEFAULT_DOXYGEN_CONFIG ?= $(DEFAULT_DOXYGEN_DIR)/Doxyfile

# Now some high end hackery since I cannot send a variable content with newlines to the shell:

# Put an eol char in this variable:
define eol_char


endef
# In what we transform an eol for shell communication:
bn=\\n

# Transform all the newlines to the 2 characters string '\n' :
DOXYGEN_PARAMETERS_WITHOUT_EOL := $(subst $(eol_char),$(bn),$(DOXYGEN_PARAMETERS))

# Add some global Doxygen parameters to the configration file:
DOXYGEN_PARAMETERS_WITHOUT_EOL += $(bn)PROJECT_NAME           = $(PROJECT_NAME)

ifdef GENERATE_TAGFILE_NAME
	# Tags are generated for each version, which is overkill, but KISS.
	DOXYGEN_PARAMETERS_WITHOUT_EOL += $(bn)GENERATE_TAGFILE=$(GENERATE_TAGFILE_NAME)
endif

.PHONY: doxygen doxygen-plain doxygen-plain do-doxygen-graph do-doxygen-graph do-doxygen publish

# To force a different evaluation of varables with different targets (have
# a look to GNU Make documentation at the end of "6.10 Target-specific
# Variable Values" for the rationale):
doxygen-plain doxygen-graph::
	$(MAKE) do-$@

do-doxygen-plain : OUTPUT_DIRECTORY       = plain
do-doxygen-plain : DOXYGEN_MORE_PARAMETERS = $(bn)OUTPUT_DIRECTORY       = $(OUTPUT_DIRECTORY)

do-doxygen-plain : do-doxygen

do-doxygen-graph : DOXYGEN_MORE_PARAMETERS = $(bn)OUTPUT_DIRECTORY       = graph$(bn)HAVE_DOT               = YES

do-doxygen-graph : do-doxygen

# Now do the reverse transformation once in the shell.
# Add the PATH to the pips-doxygen-filter too.
do-doxygen :
	( cat $(DEFAULT_DOXYGEN_CONFIG); echo "$(DOXYGEN_PARAMETERS_WITHOUT_EOL)$(DOXYGEN_MORE_PARAMETERS)" | sed s/\\\\n/$(bn)/g ) | doxygen -
