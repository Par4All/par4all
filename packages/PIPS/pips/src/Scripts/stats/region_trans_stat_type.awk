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

BEGIN{  n_modules=0;
	n_s_s=0;
	n_s_a=0;
	n_a_a=0;
	n_a_a_plus_n_s_a=0;
	tot_tmp=0;
}
NF!=0	{
	
  n_modules++;

  n_s_s += $2;
  n_s_a += $3;
  n_a_a += $4;
  n_a_a_plus_n_s_a = n_a_a_plus_n_s_a + $3 + $4;

  tot_tmp = $2+$3+$4;

  if (tot_tmp != 0)
    print $1,"&",$2,"&", int($2*100/tot_tmp),"&",$3,"&",int($3*100/tot_tmp),"&",$4,"&",int($4*100/tot_tmp),"&",$3+$4,"\\\\";
  else
    print $1, "&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"\\\\";

  next;
}
END {	
  print "\\hline";
  tot_tmp = n_s_s + n_s_a + n_a_a;

  if (tot_tmp != 0)
    print WORKSPACE,  "&",n_s_s,"&", int(n_s_s*100/tot_tmp),"&",n_s_a,"&",int(n_s_a*100/tot_tmp),"&",n_a_a,"&",int(n_a_a*100/tot_tmp),"&",n_s_a+n_a_a,"\\\\";
  else
   print WORKSPACE, "&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0, "\\\\";
}
