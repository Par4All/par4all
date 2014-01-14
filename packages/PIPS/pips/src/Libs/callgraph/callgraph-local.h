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
/* callgraph.h
   include file for callgraphs
 */

#ifndef CALLGRAPH_INCLUDED
#define CALLGRAPH_INCLUDED

#define CALLGRAPH_NOT_FOUND NULL

#define CALLGRAPH_DRAW "CALLGRAPH_DRAW"
#define CALLGRAPH_DEBUG "CALLGRAPH_DEBUG"
#define CALLGRAPH_DEBUG_LEVEL "CALLGRAPH_DEBUG_LEVEL"
#define CALLGRAPH_SHORT_NAMES "CALLGRAPH_SHORT_NAMES"

#define CALLGRAPH_INDENT 4

enum CALLGRAPH_DECOR {
    CG_DECOR_NONE,
    CG_DECOR_PROPER_EFFECTS,
    CG_DECOR_CUMULATED_EFFECTS,
    CG_DECOR_REGIONS,
    CG_DECOR_IN_REGIONS,
    CG_DECOR_OUT_REGIONS,
    CG_DECOR_PRECONDITIONS,
    CG_DECOR_TOTAL_PRECONDITIONS,
    CG_DECOR_TRANSFORMERS,
    CG_DECOR_COMPLEXITIES
};

#endif /* CALLGRAPH_INCLUDED */
