#! /bin/sh
#
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

# install a file only if different from the one already installed.

[ $# -eq 2 ] || exit 1
file=`basename $1`
if cmp $1 $2/$file > /dev/null
then
    echo "skipping $1 installation: no difference" 1>&2
else
    ${INSTALL:-${PIPS_INSTALL:-install}} $1 $2
fi
