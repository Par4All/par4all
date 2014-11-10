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

clean: NO_INCLUDES=1
export NO_INCLUDES

# old stuff:
# paf-util pip prgm_mapping scheduling static_controlize reindexing array_dfg
# paf-util and static_controlize habe been rehabilitated for PoCC

# there is no rationnal order to compile the libraries:-(
# see local TODO

# FI->FC: OK, cycles have been introduced, but then when do you
# declare the library list for linking? I did not find the information
# in Section 10.3 of the developper guide and it is not in the
# tutorial either. See Section 10.3.2, file $PIPS_ROOT/libraries.make

FWD_DIRS	= \
	misc newgen properties text-util pipsdbm \
	top-level ri-util conversion movements \
	comp_sections transformer bootstrap control flint \
	syntax c_syntax prettyprint alias-classes pointer_values \
	effects-generic effects-simple semantics complexity \
	continuation reductions effects-convex \
	effects-util callgraph icfg ricedg \
	chains rice hyperplane transformations accel-util hwac expressions \
	locality instrumentation hpfc atomizer safescale sac phrase wp65 \
	preprocessor pipsmake step to_begin_with gpu pipslibs scalopes \
	static_controlize paf-util pocc-interface taskify rstream_interface \
	regions_to_loops task_parallelization

# janusvalue
FWD_PARALLEL	= 1
 
# (re)build inter library header dependencies
deps.mk:
	{ \
	  echo 'ifeq ($$(FWD_TARGET),phase0)'; \
	  inc2deps.sh $(FWD_DIRS) | sed -e 's/:/:fwd-/;s/^/fwd-/'; \
	  echo 'endif'; \
	} > $@
