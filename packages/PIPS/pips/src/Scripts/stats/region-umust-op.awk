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
	n_umust=0;
	n_umust_must_must=0;
	n_umust_must_must_must=0;
	n_umust_must_may=0;
	n_umust_must_may_must=0;
	n_umust_sc_rn=0;
}
NF!=0	{
	if(n_modules >17 ) {
		n_modules = 1;
		print "\\hline";
		print "\\end{tabular}";
		print " ";
		print "\\begin{tabular}{| l | r | r | r | r | r | r |} \\hline";
		print "Module", "&", "nb", "&", "must/must",\
		      "&", "must res", "&", "must/may", "&",  "must res", "&", \
		       "sc\_rn", "\\\\ \\hline";
	}
	else
		n_modules++;

	module=$1;
	n_umust += $2;
	n_umust_must_must += $3;
	n_umust_must_must_must += $4;
	n_umust_must_may += $5;	
	n_umust_must_may_must += $6;
	n_umust_sc_rn += $7;
	print $1, "&", $2, "&", $3, "&", $4, "&", $5, "&", $6, "&", $7, "\\\\"
	}
END{	print "\\hline";
	print WORKSPACE, "&", n_umust, "&", n_umust_must_must, "&", \
	      n_umust_must_must_must, "&", n_umust_must_may, "&", \
	      n_umust_must_may_must, "&", \
	      n_umust_sc_rn, "\\\\ \\hline";
}
