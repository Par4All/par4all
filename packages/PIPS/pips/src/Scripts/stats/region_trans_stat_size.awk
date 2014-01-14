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
	for (i=0; i<5; i++)
	    n_num[i]=0;
	tot = 0;
}
NF!=0	{
	
  n_modules++;
  
  base=70;
  
  for(i=0; i<5; i++)
    {
      tot +=$(base+i);
      n_num[i] += $(base+i);
    }
  
}
END {	

  if (tot !=0)
    print "&",n_num[0],"&",int(n_num[0]*100/tot),"&",n_num[1],"&",int(n_num[1]*100/tot),"&",n_num[2],"&",int(n_num[2]*100/tot),"&",n_num[3],"&",int(n_num[3]*100/tot),"&",n_num[4],"&",int(n_num[4]*100/tot),"\\\\";
  else
    print "&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"\\\\";
}
