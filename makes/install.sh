#! /bin/sh
#
# install a file only if different from the one already installed.
#
# $Id$
#

[ $# -eq 2 ] || exit 1
file=`basename $1`
cmp $1 $2/$file || ${INSTALL:-${PIPS_INSTALL:-install}} $1 $2
