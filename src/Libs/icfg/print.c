/* functions to print a interprocedural control flow graph
 *
 * They should be called by pipsmake
 *
 * Lei Zhou, February 91
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
#include "icfg.h"

/*
 * This function prints out a graph that contains callees only
 */
void print_icfg(module_name)
string module_name;
{
    set_bool_property(ICFG_IFs, FALSE);
    set_bool_property(ICFG_DOs, FALSE);
    generic_print_icfg(module_name);
}

/*
 * This function prints out a graph that contains DO's 
 */
void print_icfg_with_loops(module_name)
string module_name;
{
    set_bool_property(ICFG_DOs, TRUE);
    set_bool_property(ICFG_IFs, FALSE);
    generic_print_icfg(module_name);
}

/*
 * This function prints out a graph that contains DO's
void print_icfg_with_noempty_loops(module_name)
string module_name;
{
    set_bool_property(ICFG_DOs, TRUE);
    set_bool_property(ICFG_IFs, FALSE);
    generic_print_icfg(module_name);
}
 */

/* 
 * This function prints out a graph that contains both IF's and DO's 
 */
void print_icfg_with_control(module_name)
string module_name;
{
    set_bool_property(ICFG_IFs, TRUE);
    set_bool_property(ICFG_DOs, TRUE);
    generic_print_icfg(module_name);
}

void generic_print_icfg(module_name)
string module_name;
{
    entity mod = local_name_to_top_level_entity(module_name);

    debug_on(ICFG_DEBUG_LEVEL);
    debug(1,"module_to_icfg","===%s===\n",module_name);
    debug(1,"module_to_icfg","===%s===\n",entity_name(mod));

    module_to_icfg(0, mod);

    debug_off();
}
