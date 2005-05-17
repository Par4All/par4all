#! /bin/sh
# $Id$
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
