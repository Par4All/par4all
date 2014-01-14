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

# Find out actual *pips for the current $PIPS_ARCH...
# It is always better to use the executable directly.
#
# This shell script is expected to be executer as pips/tpips or wpips.
# This can be achieve by providing such links in the Share directory.

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
  # find the actual directory of the current script
  # I have not found this information in any of bash variables.
  case $0 in
    /*)
      # script launched with an absolute path: /abs/path/to/bin/tpips
      where=`dirname $0`
      ;;
    */*)
      # script launched with a relative path: rel/path/to/bin/tpips
      where=$PWD/`dirname $0`
      ;;
    *)
      # else script launched based on PATH: tpips
      name=`basename $0`
      where=`type -p $name`
      ;;
  esac

  # check that we get something...
  [ "$where" ] || error 2 "no such directory: $where"

  # derive pips root by stripping the last directory component of this script
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

# should it also fix ld library path if dynamically linking is used?
# what about linear, newgen, extern dependencies?
#LD_LIBRARY_PATH=${PIPS_ROOT}/lib/${PIPS_ARCH}:${LD_LIBRARY_PATH}

exec "${what}" "$@"
