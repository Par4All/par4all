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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* functions to print a call graph
 *
 * They should be called by pipsmake
 *
 * Lei Zhou, February 91
 * Modification:
 *  - Callgraph prints only callees, nothing else.
 *           January 93
 *  - GO: Decoration of callgraphs, with some analyze results
 *           June 95
 *
 */
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "constants.h"
#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"

#include "database.h"
#include "resources.h"
#include "callgraph.h"

/** Print callees for debugging purpose */
void print_callees(callees c)
{
  list l = callees_callees(c);

  MAP(STRING, mn, {
    printf("%s\n", mn);
  }, l);
}


/*
 * This function prints out any graph that contains callees only
 */
bool print_decorated_call_graph(module_name,decor_type)
const char* module_name;
int decor_type;
{
    bool success = false;
    entity mod = module_name_to_entity(module_name);

    debug_on(CALLGRAPH_DEBUG_LEVEL);
    debug(1,"module_name_to_callgraphs","===%s===\n",module_name);

    success = module_to_callgraph(mod,decor_type);

    debug_off();

    return success;
}

/*
 * Print callgrpah with no decoration 
 */
bool print_call_graph(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_NONE);

    return success;
}

/*
 * Print callgrpah with proper effects
 */
bool print_call_graph_with_proper_effects(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_PROPER_EFFECTS);

    return success;
}

/*
 * Print callgrpah with cumulated effects
 */
bool print_call_graph_with_cumulated_effects(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,
					 CG_DECOR_CUMULATED_EFFECTS);

    return success;
}

/*
 * Print callgrpah with regions
 */
bool print_call_graph_with_regions(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_REGIONS);

    return success;
}

/*
 * Print callgrpah with IN regions
 */
bool print_call_graph_with_in_regions(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_IN_REGIONS);

    return success;
}

/*
 * Print callgraph with OUT regions
 */
bool print_call_graph_with_out_regions(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_OUT_REGIONS);

    return success;
}

/*
 * Print callgrpah with preconditions
 */
bool print_call_graph_with_preconditions(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_PRECONDITIONS);

    return success;
}

/*
 * Print callgrpah with preconditions
 */
bool print_call_graph_with_total_preconditions(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_TOTAL_PRECONDITIONS);

    return success;
}

/*
 * Print callgrpah with transformers
 */
bool print_call_graph_with_transformers(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_TRANSFORMERS);

    return success;
}

/*
 * Print callgrpah with complexities
 */
bool print_call_graph_with_complexities(module_name)
const char* module_name;
{
    bool success = false;

    success = print_decorated_call_graph(module_name,CG_DECOR_COMPLEXITIES);

    return success;
}
