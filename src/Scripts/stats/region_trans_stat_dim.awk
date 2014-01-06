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
	for (i=0; i<8; i++)
	  for (j=0; j<8; j++)
	    n_num[i+8*j]=0;
	tot = 0;
}
NF!=0	{
	
  n_modules++;
  
  base=6;
  
  for(i=0; i<8; i++)
    for (j=0; j<8; j++)
      {
	tot +=$(base+8*i+j);
	n_num[i+8*j] += $(base+8*i+j);
      }
  
}
END {	

  for(j=0; j<8; j++)
      if (tot != 0)
	print j, "&",n_num[0+8*j],"&",int(n_num[0+8*j]*100/tot),"&",n_num[1+8*j],"&",int(n_num[1+8*j]*100/tot),"&",n_num[2+8*j],"&",int(n_num[2+8*j]*100/tot),"&",n_num[3+8*j],"&",int(n_num[3+8*j]*100/tot),"&",n_num[4+8*j],"&",int(n_num[4+8*j]*100/tot),"&",n_num[5+8*j],"&",int(n_num[5+8*j]*100/tot),"&",n_num[6+8*j],"&",int(n_num[6+8*j]*100/tot),"&",n_num[7+8*j],"&",int(n_num[7+8*j]*100/tot),"\\\\";
      else
	print j, "&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"&",0,"\\\\";
}
