/* functions to print a interprocedural control flow graph
 *
 * They should be called by pipsmake
 *
 * Lei Zhou, February 91
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "constants.h"
#include "misc.h"
#include "properties.h"
#include "ri-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "icfg.h"

/*
 * This function prints out a graph that contains callees only
 */
static bool print_any_icfg(string module_name, int decor_type)
{
    set_bool_property(ICFG_IFs, FALSE);
    set_bool_property(ICFG_DOs, FALSE);
    set_int_property(ICFG_DECOR, decor_type);
    return generic_print_icfg(module_name);
}

/*
 * This function prints out a graph that contains DO's 
 */
static bool print_any_icfg_with_loops(string module_name, int decor_type)
{
    set_bool_property(ICFG_DOs, TRUE);
    set_bool_property(ICFG_IFs, FALSE);
    set_int_property(ICFG_DECOR, decor_type);
    return generic_print_icfg(module_name);
}

/* 
 * This function prints out a graph that contains both IF's and DO's 
 */
static bool print_any_icfg_with_control(string module_name, int decor_type)
{
    set_bool_property(ICFG_IFs, TRUE);
    set_bool_property(ICFG_DOs, TRUE);
    set_int_property(ICFG_DECOR, decor_type);
    return generic_print_icfg(module_name);
}

bool generic_print_icfg(module_name)
string module_name;
{
    entity mod = local_name_to_top_level_entity(module_name);

    debug_on(ICFG_DEBUG_LEVEL);
    pips_debug(1,"===%s===\n===%s===\n",module_name,entity_name(mod));

    print_module_icfg(mod);

    debug_off();

    return TRUE;
}

/* I would have prefered something like that... FC 
 * or even no properties at all? 
 */
bool 
parametrized_print_icfg(
    string module_name,
    bool print_ifs,
    bool print_dos,
    text (*deco)(string))
{
    entity module = local_name_to_top_level_entity(module_name);

    set_bool_property(ICFG_IFs, print_ifs);
    set_bool_property(ICFG_DOs, print_dos);
    print_module_icfg_with_decoration(module, deco);

    return TRUE;
}

bool print_icfg(string module_name)
{
    return print_any_icfg(module_name,ICFG_DECOR_NONE);
}

bool print_icfg_with_complexities(string module_name)
{
    return print_any_icfg(module_name,ICFG_DECOR_COMPLEXITIES);
}

bool print_icfg_with_preconditions(string module_name)
{
    return print_any_icfg(module_name,ICFG_DECOR_PRECONDITIONS);
}

bool print_icfg_with_transformers(string module_name)
{
    return print_any_icfg(module_name,ICFG_DECOR_TRANSFORMERS);
}

bool print_icfg_with_proper_effects(string module_name)
{
    return print_any_icfg(module_name,ICFG_DECOR_PROPER_EFFECTS);
}

bool print_icfg_with_cumulated_effects(string module_name)
{
    return print_any_icfg(module_name,ICFG_DECOR_CUMULATED_EFFECTS);
}

bool print_icfg_with_regions(string module_name)
{ return print_any_icfg(module_name,ICFG_DECOR_REGIONS);}

bool print_icfg_with_in_regions(string module_name)
{ return print_any_icfg(module_name,ICFG_DECOR_IN_REGIONS);}

bool print_icfg_with_out_regions(string module_name)
{ return print_any_icfg(module_name,ICFG_DECOR_OUT_REGIONS);}

/*
 * ICFGs with loops
 */

bool print_icfg_with_loops(string module_name)
{
    return print_any_icfg_with_loops(module_name,ICFG_DECOR_NONE);
}

bool print_icfg_with_loops_complexities(string module_name)
{
    return print_any_icfg_with_loops(module_name,ICFG_DECOR_COMPLEXITIES);
}

bool print_icfg_with_loops_preconditions(string module_name)
{
    return print_any_icfg_with_loops(module_name,ICFG_DECOR_PRECONDITIONS);
}

bool print_icfg_with_loops_transformers(string module_name)
{
 return print_any_icfg_with_loops(module_name,ICFG_DECOR_TRANSFORMERS);
}

bool print_icfg_with_loops_proper_effects(string module_name)
{
    return print_any_icfg_with_loops(module_name,ICFG_DECOR_PROPER_EFFECTS);
}

bool print_icfg_with_loops_cumulated_effects(string module_name)
{ return print_any_icfg_with_loops(module_name,ICFG_DECOR_CUMULATED_EFFECTS);}

bool print_icfg_with_loops_regions(string module_name)
{ return print_any_icfg_with_loops(module_name,ICFG_DECOR_REGIONS);}

bool print_icfg_with_loops_in_regions(string module_name)
{ return print_any_icfg_with_loops(module_name,ICFG_DECOR_IN_REGIONS);}

bool print_icfg_with_loops_out_regions(string module_name)
{ return print_any_icfg_with_loops(module_name,ICFG_DECOR_OUT_REGIONS);}

/* 
 * ICFGs with controls
 */

bool print_icfg_with_control(string module_name)
{
    return print_any_icfg_with_control(module_name,ICFG_DECOR_NONE);
}

bool print_icfg_with_control_complexities(string module_name)
{
    return print_any_icfg_with_control(module_name,ICFG_DECOR_COMPLEXITIES);
}

bool print_icfg_with_control_preconditions(string module_name)
{
    return print_any_icfg_with_control(module_name,ICFG_DECOR_PRECONDITIONS);
}

bool print_icfg_with_control_transformers(string module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_TRANSFORMERS);}

bool print_icfg_with_control_proper_effects(string module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_PROPER_EFFECTS);}

bool print_icfg_with_control_cumulated_effects(string module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_CUMULATED_EFFECTS);}

bool print_icfg_with_control_regions(string module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_REGIONS);}

bool print_icfg_with_control_in_regions(string module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_IN_REGIONS);}

bool print_icfg_with_control_out_regions(string module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_OUT_REGIONS);}




