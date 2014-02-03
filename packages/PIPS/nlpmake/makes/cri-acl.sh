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

# Access Control List at CRI.

grp=pipsgrp

for dir in \
    /projects/Newgen/Production \
    /projects/C3/Linear/Production
do
  chgrp -R $grp $dir
  find $dir -type f -print0 | xargs -0 chmod ug+rw,o-w
  find $dir -type d -print0 | xargs -0 chmod ug+rwxs,o-w
done
