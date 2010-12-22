#! /bin/bash
#
# $Id$
#
# Copyright 1989-2010 MINES ParisTech
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
      # handle svn working copy
      cd $dir
      unset LC_MESSAGES
      export LANG=C
      repos=$(svn info | sed -n -e 's/^URL: //p')
      revision=$(svnversion)
      committed=$(svnversion -c)
      author=$(svn info | sed -n -e 's/^Last Changed Author: //p')
      echo "$repos@$revision ($committed) [$author]"
    elif [ -d $dir/.git ]
    then
      # generate something for git-svn users
      author=$(git log -1|sed -n -e 's/^Author: \([-a-zA-Z0-1_]*\) .*/\1/p;2q')
      repos='git-svn'
      revision=$(git log -1|sed -n -e 's/commit \([a-f0-9]\{8\}\).*/\1/p;1q')
      committed=$(git log -1 --date=iso|sed -n -e 's/Date: *\(.*\)/\1/p;3q')
      committed=${committed/ +0000/ UTC}
      echo "$repos@$revision ($committed) [$author]"
    else
      # not a working copy
      echo 'unknown@unknown (unknown) [unknown]'
    fi
  else
    # svnversion not found
    echo 'unknown@unknown (unknown) [unknown]'
  fi
done
