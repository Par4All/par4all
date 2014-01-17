#! /bin/bash
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

# show repository, revisions and committed for directory arguments

for dir
do
  if [ -d $dir/.svn ] && type svnversion > /dev/null 2>&1
  then
    # handle svn working copy
    cd $dir
    unset LC_MESSAGES
    export LANG=C
    repos=$(svn info | sed -n -e 's/^URL: //p')
    revision=$(svnversion)
    #committed=$(svnversion -c)
    author=$(svn info | sed -n -e 's/^Last Changed Author: //p')
    echo "$repos@$revision [$author]"
  elif git describe --long --always --all > /dev/null 2>&1
  then
    # generate something for git or git-svn users
    # Get the author of last commit:
    author=$(cd $dir &>/dev/null ; git log -1 --pretty='%aN')
    # Well, this does not mean anything in git:
    repos='git'
    # Compute a revision from last tag if any, such as p4a-1.3-0-g57a41e0-dirty
    # with -dirty showing some changes not commited yet:
    revision=$(cd $dir &>/dev/null ; git describe --long --always --all --dirty)
    # The date from the autor point of view:
    committed=$(cd $dir &>/dev/null ; git log -1 --pretty='%ai')
    committed=${committed/ +0000/ UTC}
    echo "$repos@$revision ($committed) [$author]"
  else
    # Unable to get an interesting information:
    echo 'unknown@unknown (unknown) [unknown]'
  fi
done
