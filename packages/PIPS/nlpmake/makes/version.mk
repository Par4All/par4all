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

GET_VERSION	= $(MAKE.d)/version.sh

.PHONY: .revisions_h
# try to keep previous revisions.h if cannot be regenerated
# the actual result depends on the kind of git-svn setup used,
# so some more tweaking may be needed here.
.revisions_h:
	if [ -d $(ROOT)/.svn -o -d $(ROOT)/.git -o ! -f revisions.h ] ; \
	then \
	  $(RM) revisions.h ; \
	  { \
	   echo '#define NEWGEN_REV "$(shell $(GET_VERSION) $(NEWGEN_ROOT))"'; \
	   echo '#define LINEAR_REV "$(shell $(GET_VERSION) $(LINEAR_ROOT))"'; \
	   echo '#define PIPS_REV "$(shell $(GET_VERSION) $(ROOT))"'; \
	   echo '#define NLPMAKE_REV "$(shell $(GET_VERSION) $(ROOT)/makes)"'; \
	   echo '#define CC_VERSION "$(shell $(CC_VERSION))"'; \
	  } > revisions.h ; \
	else \
	  touch revisions.h ; \
	fi

$(ARCH)/revisions.o: CPPFLAGS += -DUTC_DATE='$(UTC_DATE)'

revisions.h: .revisions_h

clean: version-clean
# keep revision if cannot be regenerated?
version-clean:
	[ -d $(ROOT)/.svn -o -d $(ROOT)/.git ] && $(RM) revisions.h
