/* functions to print a call graph
 *
 * They should be called by pipsmake
 *
 * Lei Zhou, February 91
 * Modification:
 *  - Callgraph prints only callees, nothing else.
 *           January 93
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
 * This function prints out a graph that contains callees only
 */
void print_call_graph(module_name)
string module_name;
{
    entity mod = local_name_to_top_level_entity(module_name);

    debug_on(CALLGRAPH_DEBUG_LEVEL);
    debug(1,"module_name_to_callgraphs","===%s===\n",module_name);

    module_to_callgraph(mod);

    debug_off();
}
