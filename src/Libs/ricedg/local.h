
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <values.h>
#include <string.h>
#include <setjmp.h>

#include "linear.h"

#include "genC.h"

#include "text.h"
#include "ri.h"
#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "text-util.h"

#include "ri-util.h" /* linear.h is included in */
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "pipsdbm.h"
#include "semantics.h"
#include "pipsmake.h"

#include "constants.h"
#include "properties.h"
#include "resources.h"

#include "polyedre.h"

#include "ricedg.h" 

