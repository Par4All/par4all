#! /bin/sh
#
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

# show repository, revisions and committed for directory arguments

for dir
do
    if type svnversion > /dev/null 2>&1
    then
	if [ -d $dir/.svn ]
	then
	    cd $dir
	    unset LC_MESSAGES
	    export LANG=C
	    repos=$(svn info | sed -n -e 's/^URL: //p')
	    revision=$(svnversion)
	    committed=$(svnversion -c)
	    echo "$repos@$revision ($committed)"
	else
	    # not a working copy
	    echo 'unknown@unknown (unknown)'
	fi
    else
	# svnversion not found
	echo 'unknown@unknown (unknown)'
    fi
done
