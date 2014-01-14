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
/* 
   icfg.h -- acronym
   include file for Interprocedural Control Flow Graph
 */

#define ICFG_NOT_FOUND NULL
#define ICFG_OPTIONS "tcdDIFl"

#define ICFG_CALLEES_TOPO_SORT "ICFG_CALLEES_TOPO_SORT"
#define ICFG_DRAW "ICFG_DRAW"
#define ICFG_DEBUG "ICFG_DEBUG"
#define ICFG_DEBUG_LEVEL "ICFG_DEBUG_LEVEL"
#define ICFG_DOs "ICFG_DOs"
#define ICFG_IFs "ICFG_IFs"
#define ICFG_DV "ICFG_DV"
#define ICFG_FLOATs "ICFG_FLOATs"
#define ICFG_SHORT_NAMES "ICFG_SHORT_NAMES"

#define ICFG_DECOR "ICFG_DECOR"

#define ICFG_DECOR_NONE 1
#define ICFG_DECOR_PRECONDITIONS 2
#define ICFG_DECOR_TRANSFORMERS 3
#define ICFG_DECOR_PROPER_EFFECTS 4
#define ICFG_DECOR_CUMULATED_EFFECTS 5
#define ICFG_DECOR_REGIONS 6
#define ICFG_DECOR_IN_REGIONS 7
#define ICFG_DECOR_OUT_REGIONS 8
#define ICFG_DECOR_COMPLEXITIES 9
#define ICFG_DECOR_TOTAL_PRECONDITIONS 10
#define ICFG_DECOR_FILTERED_PROPER_EFFECTS 11

#define RW_FILTERED_EFFECTS "RW_FILTERED_EFFECTS"

#define READ_ALL 0
#define WRITE_ALL 1
#define READWRITE_ALL 2
#define READ_END 3
#define WRITE_END 4
#define READWRITE_END 5

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#define CALL_MARK "CALL_MARK@@@@"
#define ADD_ELEMENT_TO_LIST( _list, _type, _element) \
    (_list = gen_nconc( _list, CONS( _type, _element, NIL)))

