#! /usr/bin/perl -wn
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


if (/^test/ or /^}/ or /^\#/ or /^\s*$/) 
{
    print;
}
elsif (/^\s*export\s+([A-Za-z_0-9]+)=(.*)/)
{
    print "setenv $1 $2\n";
}
elsif (/^\s*([A-Za-z_0-9]+)=(.*)/)
{
    # handle PATH especially for csh
    if ($1 eq 'PATH')
    {
	print "setenv $1 $2\n";
    }
    else
    {
        print "set $1=$2\n";
    }
}

END 
{ 
    print "rehash\n"; 
}
