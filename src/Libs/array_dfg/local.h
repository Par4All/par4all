/*
 * $Id$
 */

/* Ansi includes 	*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h" 
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"
#include "union.h"

/* Pips includes 	*/
#include "database.h"

#include "dg.h"
#include "paf_ri.h"

/* argh...
 */
#ifdef GRAPH_IS_DG
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#endif

#ifdef GRAPH_IS_DFG
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;
#endif

#include "graph.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "semantics.h"
#include "static_controlize.h"
#include "ricedg.h"
#include "paf-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "array_dfg.h"
#include "pip.h"
