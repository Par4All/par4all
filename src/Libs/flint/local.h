/*
 * $Id$
 */

/* system includes */
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>

/* PiPs specific headers */
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "graph.h"
#include "flint.h"
#include "effects-generic.h"
#include "pipsdbm.h"
#include "resources.h"

#include "text-util.h"
#include "control.h"
#include "effects-simple.h"
#include "dg.h"

/* Instantiation of the dependence graph: */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "ricedg.h"
#include "semantics.h"
#include "transformations.h"
#include "flint.h"

