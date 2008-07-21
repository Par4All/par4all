#! /bin/bash
#
# $Id$
#
# Find out actual *pips for the current $PIPS_ARCH...
# It is always better to use the executable directly.
#
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

[ "${PIPS_ROOT}" ] ||
{
    case $0 in
	/*)
	    where=`dirname $0`
	    ;;
	*)
	    name=`basename $0`
	    where=`type -p $name`
	    ;;
    esac

    PIPS_ROOT=`dirname $where`
    export PIPS_ROOT
}

[ -d ${PIPS_ROOT} ] || error 2 "no such directory: $PIPS_ROOT"

[ "${PIPS_ARCH}" ] ||
{
    arch=${PIPS_ROOT}/makes/arch.sh
    test -x $arch || error 3 "no $arch script to build PIPS_ARCH"
    PIPS_ARCH=`$arch`
    export PIPS_ARCH
}

# Avoid a recursion if no actual binary is found:
PATH=./${PIPS_ARCH}:${PIPS_ROOT}/bin/${PIPS_ARCH} \
    type ${what} > /dev/null || error 3 "no ${what} binary found!"

# fix path according to pips architecture
PATH=./${PIPS_ARCH}:${PIPS_ROOT}/bin/${PIPS_ARCH}:${PATH}

exec "${what}" "$@"
