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

#include "genC.h"
#include "ri.h"
#include "constants.h"
#include "misc.h"
#include "properties.h"
#include "ri-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "callgraph.h"


/*
 * This function prints out any graph that contains callees only
 */
void print_decorated_call_graph(module_name,decor_type)
string module_name;
int decor_type;
{
    entity mod = local_name_to_top_level_entity(module_name);

    debug_on(CALLGRAPH_DEBUG_LEVEL);
    debug(1,"module_name_to_callgraphs","===%s===\n",module_name);

    module_to_callgraph(mod,decor_type);

    debug_off();
}

/*
 * Print callgrpah with no decoration 
 */
void print_call_graph(module_name)
string module_name;
{print_decorated_call_graph(module_name,CG_DECOR_NONE);}

/*
 * Print callgrpah with proper effects
 */
void print_call_graph_with_proper_effects(module_name)
string module_name;
{print_decorated_call_graph(module_name,CG_DECOR_PROPER_EFFECTS);}

/*
 * Print callgrpah with cumulated effects
 */
void print_call_graph_with_cumulated_effects(module_name)
string module_name;
{print_decorated_call_graph(module_name,CG_DECOR_CUMULATED_EFFECTS);}

/*
 * Print callgrpah with regions
 */
void print_call_graph_with_regions(module_name)
string module_name;
{print_decorated_call_graph(module_name,CG_DECOR_REGIONS);}

/*
 * Print callgrpah with IN regions
 */
void print_call_graph_with_in_regions(module_name)
string module_name;
{print_decorated_call_graph(module_name,CG_DECOR_IN_REGIONS);}

/*
 * Print callgrpah with OUT regions
 */
void print_call_graph_with_out_regions(module_name)
string module_name;
{print_decorated_call_graph(module_name,CG_DECOR_OUT_REGIONS);}

/*
 * Print callgrpah with preconditions
 */
void print_call_graph_with_preconditions(module_name)
string module_name;
{print_decorated_call_graph(module_name,CG_DECOR_PRECONDITIONS);}

/*
 * Print callgrpah with transformers
 */
void print_call_graph_with_transformers(module_name)
string module_name;
{print_decorated_call_graph(module_name,CG_DECOR_TRANSFORMERS);}
