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

BEGIN{  
  n_modules=0;
  tot_array=0;
  tot=0;
}
NF!=0	{
	
  n_modules++;

  tot_array += $3+$4;
  tot += $5;
}
END {	
  if (tot_array !=0)
    print tot, "(",tot*100/tot_array, "\\%).";
  else
    print 0, "(0\\%).";
}
