/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
/*#define REGULAR_VERSION "Regular Version"
  #define PARALLEL_VERSION "Parallel Version"
  */

/* Labels for menu Edit/View (These definitions are almost automatically available as aliases
   in wpips.rc; FI) */
#define USER_VIEW "User View"
#define SEQUENTIAL_VIEW "Sequential View"
#define PARALLEL_VIEW "Parallel View"
#define CALLGRAPH_VIEW "Callgraph View"
#define ICFG_VIEW "ICFG View"
#define DISTRIBUTED_VIEW "Distributed View"
#define DEPENDENCE_GRAPH_VIEW "Dependence Graph View"
#define FLINT_VIEW "Flint View"
#define SEQUENTIAL_EMACS_VIEW "Emacs Sequential View"
#define SEQUENTIAL_GRAPH_VIEW "Sequential View with Control Graph"
#define ARRAY_DFG_VIEW "Array data flow graph View"
#define TIME_BASE_VIEW "Scheduling View"
#define PLACEMENT_VIEW "Placement View"
/* A special view that gives the .f source mainly for edition: */
#define EDIT_VIEW "Edit"

/* Labels for menu Transform */
#define PARALLELIZE_TRANSFORM "! Parallelize"
#define PRIVATIZE_TRANSFORM "Privatize Scalars"
#define ARRAY_PRIVATIZE_TRANSFORM "Privatize Scalars & Arrays"
#define DISTRIBUTE_TRANSFORM "Distribute"
#define PARTIAL_EVAL_TRANSFORM "Partial Eval"
#define UNROLL_TRANSFORM "Loop Unroll"
#define FULL_UNROLL_TRANSFORM "Full Loop Unroll"
#define STRIP_MINE_TRANSFORM "Strip Mining"
#define LOOP_INTERCHANGE_TRANSFORM "Loop Interchange"
#define LOOP_NORMALIZE_TRANSFORM "Loop Normalize"
#define SUPPRESS_DEAD_CODE_TRANSFORM "Dead Code Elimination"
#define UNSPAGHETTIFY_TRANSFORM "Unspaghettify the Control Graph"
#define ATOMIZER_TRANSFORM "Atomize"
#define NEW_ATOMIZER_TRANSFORM "New Atomize"
#define REDUCTIONS_TRANSFORM "! Reductions"
#define STATIC_CONTROLIZE_TRANSFORM "Static Controlize"
#define STF_TRANSFORM "Restructure with STF"

#define SEMANTICS_ANALYZE "Semantics"
#define CALLGRAPH_ANALYZE "Call Graph"

#define FULL_DG_PROPS "Full Dependence Graph"
#define FAST_DG_PROPS "Fast Dependence Graph"
