# $Id$
#
# Copyright 1989-2010 MINES ParisTech
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

clean: NO_INCLUDES=1
export NO_INCLUDES

# old stuff:
# paf-util pip prgm_mapping scheduling static_controlize reindexing array_dfg

# there is no rationnal order to compile the libraries:-(
# see local TODO
FWD_DIRS	= \
	misc newgen properties text-util pipsdbm \
	top-level ri-util conversion movements \
	comp_sections transformer bootstrap control flint \
	syntax c_syntax prettyprint \
	effects effects-generic effects-simple semantics complexity \
	continuation reductions regions effects-convex alias-classes \
	callgraph icfg ricedg \
	chains rice hyperplane transformations hwac expressions \
	statistics instrumentation hpfc atomizer safescale sac phrase wp65 \
	preprocessor pipsmake step to_begin_with gpu pipslibs scalopes
