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

# make stuff specific to the pips project
# BEWARE! THIS IS OBSOLETE...

ifdef LIB_TARGET

$(ARCH)/pips: $(ROOT)/lib/$(ARCH)/libpips.a
$(ARCH)/tpips: $(ROOT)/lib/$(ARCH)/libtpips.a
$(ARCH)/wpips: $(ROOT)/lib/$(ARCH)/libwpips.a

# building a test executable in a library
test:; $(MAKE) main.dir=$(ROOT)/lib/$(ARCH) $(ARCH)/pips
ttest:;	$(MAKE) main.dir=$(ROOT)/lib/$(ARCH) $(ARCH)/tpips
wtest:;	$(MAKE) main.dir=$(ROOT)/lib/$(ARCH) $(ARCH)/wpips
ftest:;	$(MAKE) main.dir=$(ROOT)/lib/$(ARCH) $(ARCH)/fpips

endif # LIB_TARGET
