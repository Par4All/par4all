#! /bin/sh
#
# $Id$
#
# Find out actual *pips for the current $PIPS_ARCH...
# it is always better to use the executable directly. 
# This shell script is expected to be executer as pips/tpips or wpips.
# This can be achieve by providing such links in the Share directory.
#

what=`basename $0 .sh`

error()
{
  status=$1
  shift
  echo "error: $@" >&2
  exit ${status}
}

[ "${PIPS_ARCH}" ] || error 1 "\$PIPS_ARCH is undefined"
[ "${PIPS_ROOT}" ] || error 2 "\$PIPS_ROOT is undefined"

PATH=./${PIPS_ARCH}:${PIPS_ROOT}/Bin/${PIPS_ARCH}:${PATH}

exec "${what}" "$@"
