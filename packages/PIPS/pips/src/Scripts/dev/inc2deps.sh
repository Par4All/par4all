#!/bin/bash
#
# $Id$
#
# yet another dependency-related script...
# generate library make-dependencies based on declared includes

dirs=''
for path ; do
  if [ -d $path ] ; then
    dirs+=" ${path##*/}"
  fi
done

for path ; do
  if [ -d $path ] ; then
    lib=${path##*/}
    grep '^ *# *include *' $path/*.[cyl] |
    perl -n -e "print \"$lib:\$1\\n\" if /include *\"([^\"]*)\.h\"/" |
    sort -u
  fi
done | inc2deps.pl $dirs



