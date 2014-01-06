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
	n_array_pairs=0;
	n_self_loop_independent_pairs=0;
	n_independent_pairs=0;
	n_dependent_pairs=0;
	n_constant_dependence_pairs=0;
	n_exact_dependence_pairs=0;
}
NF!=0	{
	if(n_modules >17 ) {
		n_modules = 1;
		print "\\hline";
		print "\\end{tabular}";
		print " ";
		print "\\begin{tabular}{| l | r | r | r | r |} \\hline";
		print "Module", "&", "array pairs", "&", "self", "&", "independent", \
		"&", "dependent", "\\\\ \\hline";
	}
	else
		n_modules++;

	module=$1;
	n_array_pairs += $2;
	n_self_loop_independent_pairs += $4;
#	n_self_loop_independent_pairs += $45+$50;
#	n_independent_pairs += $3-($45+$50);
	n_independent_pairs += $3;
	n_dependent_pairs += $2-$3;
	n_constant_dependence_pairs += $5;
	n_exact_dependence_pairs += $6;
	print $1, "&", $2, "&", $45+$50, "&", $3, "&", $2-$3, "\\\\"
	}
END{	print "\\hline";
	print WORKSPACE, "&", n_array_pairs, "&", n_self_loop_independent_pairs, "&", \
	n_independent_pairs, "&", n_dependent_pairs "\\\\ \\hline";
}
