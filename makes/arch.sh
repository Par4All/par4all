#! /bin/sh
#
# $Id$
#
# define pips architecture
#

case `uname -s` in
    Linux) echo "LINUXI86LL" ;;
    SunOS) echo "GNUSOL2LL" ;;
    FreeBSD) echo "FREEBSDLL" ;;
    *) echo "DEFAULT" ;;
esac
