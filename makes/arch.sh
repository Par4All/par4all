#! /bin/sh
#
# $Id$
#
# define pips architecture
#

case `uname -s` in
    Linux) if [ `uname -m` = 'x86_64' ]; then 
	echo "LINUX_x86_64_LL"
	else 
	echo "LINUXI86LL"
	fi;;
    SunOS) echo "GNUSOL2LL" ;;
    FreeBSD) echo "FREEBSDLL" ;;
    *) echo "DEFAULT" ;;
esac
