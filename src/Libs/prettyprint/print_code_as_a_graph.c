/* Define all the entry points to generate the various graph
   sequential view to be viewed later with a graph viewer such as
   daVinci.

   Ronan.Keryell@cri.ensmp.fr, 5/09/1995. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "properties.h"
#include "top-level.h"
#include "text-util.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "complexity_ri.h"
#include "complexity.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "semantics.h"
#include "prettyprint.h"


bool
print_code_as_a_graph(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    success = print_code(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}


bool
print_code_as_a_graph_transformers(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    success = print_code_transformers(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}


bool
print_code_as_a_graph_complexities(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    success = print_code_complexities(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}


bool
print_code_as_a_graph_preconditions(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    success = print_code_preconditions(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}

bool
print_code_as_a_graph_total_preconditions(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    success = print_code_total_preconditions(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}


bool
print_code_as_a_graph_regions(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    user_warning("print_code_as_a_graph_regions", "To be done...");
    success = print_code				/*_regions*/(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}


bool
print_code_as_a_graph_in_regions(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    user_warning("print_code_as_a_graph_regions", "To be done...");
    success = print_code				/*_in_regions*/(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}


bool
print_code_as_a_graph_out_regions(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    user_warning("print_code_as_a_graph_regions", "To be done...");
    success = print_code				/*_out_regions*/(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}


bool
print_code_as_a_graph_proper_effects(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    success = print_code_proper_effects(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}


bool
print_code_as_a_graph_cumulated_effects(char *mod_name)
{
    bool success;

    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", TRUE);
    success = print_code_cumulated_effects(mod_name);
    set_bool_property("PRETTYPRINT_UNSTRUCTURED_AS_A_GRAPH", FALSE);

    return success;
}
