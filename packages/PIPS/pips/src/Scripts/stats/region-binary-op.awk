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
	n_op=0;
	n_op_pot_must=0;
	n_op_must=0;
}
NF!=0	{
	if(n_modules >17 ) {
		n_modules = 1;
		print "\\hline";
		print "\\end{tabular}";
		print " ";
		print "\\begin{tabular}{| l | r | r | r |} \\hline";
		print "Module", "&", "invocation number", "&", "pot. must results",\
		    "&", "must results", "\\\\ \\hline";
	}
	else
		n_modules++;

	module=$1;
	n_op += $2;
	n_op_pot_must += $3;
	n_op_must += $4;
	n_op_ofl += $5;	
	print $1, "&", $2, "&", $3, "&", $4, "\\\\"
	}
END{	print "\\hline";
	print WORKSPACE, "&", n_op, "&", n_op_pot_must, "&", \
	n_op_must, "\\\\ \\hline";
}
