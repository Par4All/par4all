#! /bin/sh
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

# define pips architecture

case `uname -s` in
    Linux)
      if [ `uname -m` = 'x86_64' ] ; then
	echo "LINUX_x86_64_LL"
      elif [ `uname -m` = 'ppc64' ] ; then
	echo "LINUX_PPC_64_LL"
      else
	echo "LINUXI86LL"
      fi;;
    SunOS)
	echo "GNUSOL2LL" ;;
    FreeBSD)
	echo "FREEBSDLL" ;;
    Darwin)
	echo "MACOSX" ;;
    *)
	echo "DEFAULT" ;;
esac
