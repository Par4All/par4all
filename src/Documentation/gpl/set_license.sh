#! /bin/bash
#
# $Id$
#
# Copyright 1989-2014 MINES ParisTech
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

# set license to file

for file
do
  test -f $file || {
    echo "$file is not a file" >&2
    exit 1
  }

  # type...
  case $file in
    *.sh|*.pl|*.py|*.sed) type=sh ;;
    *.tex|*.bib|*.sty|*.w) type=tex ;;
    *.mk|Makefile) type=sh ;;
    *.[chly]|*.java) type=h ;;
    *.f) type=f ;;
    *.m4|*.m4[cfh]) type=m4 ;;
    *) type=sh ;;
  esac

  # is it a sharp-bang script?
  firstline=
  startat=1
  if [[ $type = 'sh' ]]
  then
    read firstline < $file
    [[ $firstline == \#\!* ]] && startat=2
  fi

  license=$PIPS_ROOT/src/Documentation/gpl/license.$type

  echo "file=$file startat=$startat type=$type license=$license"

  [[ -f $license ]] || {
    echo "license file $license not found!" >&2
    exit 2
  } 

  [[ $(grep -i 'copyright' $file) ]] && {
    echo "skipping file '$file' which seems to contain a license!" >&2
    continue
  }
  
  mv $file $file.old

  {
    [[ $startat = '2' ]] && {
      echo $firstline
      echo '#'
    } 

    cat $license
    echo
    sed -n "$startat,\$p" $file.old
  } > $file
done
