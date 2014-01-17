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

# subversion related targets

SVN =		svn
SVN_USERNAME =
SVN_FLAGS =

IS_SVN_WC =	test -d .svn

# fix command flags if username is provided
ifdef SVN_USERNAME
SVN_FLAGS	+= --username $(SVN_USERNAME)
endif

diff:
	-@$(IS_SVN_WC) && $(SVN) $(SVN_FLAGS) $@

status:
	-@$(IS_SVN_WC) && $(SVN) $(SVN_FLAGS) $@

info:
	-@$(IS_SVN_WC) && $(SVN) $(SVN_FLAGS) $@

commit:
	-@$(IS_SVN_WC) && $(SVN) $(SVN_FLAGS) $@
