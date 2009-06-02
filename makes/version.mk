# $Id$
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

VERSION	= $(MAKE.d)/version.sh

.revisions_h:
	$(RM) revisions.h
	{ \
	  echo '#define NEWGEN_REV "$(shell $(VERSION) $(NEWGEN_ROOT))"'; \
	  echo '#define LINEAR_REV "$(shell $(VERSION) $(LINEAR_ROOT))"'; \
	  echo '#define PIPS_REV "$(shell $(VERSION) $(PIPS_ROOT))"'; \
	  echo '#define NLPMAKE_REV "$(shell $(VERSION) $(PIPS_ROOT)/makes)"'; \
	  echo '#define CC_VERSION "$(shell $(CC_VERSION))"'; \
	} > revisions.h


$(ARCH)/revisions.o: CPPFLAGS += -DUTC_DATE='$(UTC_DATE)'

revisions.h: .revisions_h

clean: version-clean
version-clean:
	$(RM) revisions.h
