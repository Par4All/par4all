#! /bin/sh
#
# install a file only if different from the one already installed.
#
# $Id$
#

[ $# -eq 2 ] || exit 1
file=`basename $1`
if cmp $1 $2/$file > /dev/null
then
    echo "skipping $1 installation: no difference" 1>&2
else
    ${INSTALL:-${PIPS_INSTALL:-install}} $1 $2
fi
