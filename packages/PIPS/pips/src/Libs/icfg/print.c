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
#include "effects.h"
#include "constants.h"
#include "misc.h"
#include "properties.h"
#include "ri-util.h"
#include "effects-util.h"

#include "database.h"
#include "resources.h"
#include "icfg.h"

/*
 * This function prints out a graph that contains callees only
 */
static bool print_any_icfg(const char* module_name, int decor_type)
{
  set_bool_property(ICFG_IFs, false);
  set_bool_property(ICFG_DOs, false);
  set_int_property(ICFG_DECOR, decor_type);
  return generic_print_icfg(module_name);
}

/*
 * This function prints out a graph that contains DO's
 */
static bool print_any_icfg_with_loops(const char* module_name, int decor_type)
{
  set_bool_property(ICFG_DOs, true);
  set_bool_property(ICFG_IFs, false);
  set_int_property(ICFG_DECOR, decor_type);
  return generic_print_icfg(module_name);
}

/*
 * This function prints out a graph that contains both IF's and DO's
 */
static bool print_any_icfg_with_control(const char* module_name, int decor_type)
{
  set_bool_property(ICFG_IFs, true);
  set_bool_property(ICFG_DOs, true);
  set_int_property(ICFG_DECOR, decor_type);
  return generic_print_icfg(module_name);
}

/* Shared throughout the icfg library*/
bool prettyprint_fortran_icfg_p = true;
bool prettyprint_C_icfg_p = false;

bool generic_print_icfg(const char* module_name)
{
  entity mod = local_name_to_top_level_entity(module_name);

  /* select the language */
  prettyprint_fortran_icfg_p = fortran_module_p(mod);
  prettyprint_C_icfg_p = c_module_p(mod);

  icfg_set_indentation(get_int_property("ICFG_INDENTATION"));

  debug_on(ICFG_DEBUG_LEVEL);
  pips_debug(1,"===%s===\n===%s===\n",module_name,entity_name(mod));

  print_module_icfg(mod);

  debug_off();
  icfg_reset_indentation();

  return true;
}

/* I would have prefered something like that... FC
 *
 * or even no properties at all?
 */
bool parametrized_print_icfg(
    const char* module_name,
    bool print_ifs,
    bool print_dos,
    text (*deco)(string))
{
  entity module = local_name_to_top_level_entity(module_name);

  set_bool_property(ICFG_IFs, print_ifs);
  set_bool_property(ICFG_DOs, print_dos);
  print_module_icfg_with_decoration(module, deco);

  return true;
}

bool print_icfg(const char* module_name)
{
  return print_any_icfg(module_name,ICFG_DECOR_NONE);
}

bool print_icfg_with_complexities(const char* module_name)
{
  return print_any_icfg(module_name,ICFG_DECOR_COMPLEXITIES);
}

bool print_icfg_with_preconditions(const char* module_name)
{
  return print_any_icfg(module_name,ICFG_DECOR_PRECONDITIONS);
}

bool print_icfg_with_total_preconditions(const char* module_name)
{
  return print_any_icfg(module_name,ICFG_DECOR_TOTAL_PRECONDITIONS);
}

bool print_icfg_with_transformers(const char* module_name)
{
  return print_any_icfg(module_name,ICFG_DECOR_TRANSFORMERS);
}

bool print_icfg_with_proper_effects(const char* module_name)
{
  return print_any_icfg(module_name,ICFG_DECOR_PROPER_EFFECTS);
}

bool print_icfg_with_filtered_proper_effects(const char* module_name) 
{
  return print_any_icfg(module_name, ICFG_DECOR_FILTERED_PROPER_EFFECTS);
}

bool print_dvicfg_with_filtered_proper_effects(const char* module_name)
{
  set_bool_property(ICFG_DV, true);
  return print_any_icfg(module_name, ICFG_DECOR_FILTERED_PROPER_EFFECTS);
}

bool print_icfg_with_cumulated_effects(const char* module_name)
{
  return print_any_icfg(module_name,ICFG_DECOR_CUMULATED_EFFECTS);
}

bool print_icfg_with_regions(const char* module_name)
{ return print_any_icfg(module_name,ICFG_DECOR_REGIONS);}

bool print_icfg_with_in_regions(const char* module_name)
{ return print_any_icfg(module_name,ICFG_DECOR_IN_REGIONS);}

bool print_icfg_with_out_regions(const char* module_name)
{ return print_any_icfg(module_name,ICFG_DECOR_OUT_REGIONS);}

/*
 * ICFGs with loops
 */

bool print_icfg_with_loops(const char* module_name)
{
  return print_any_icfg_with_loops(module_name,ICFG_DECOR_NONE);
}

bool print_icfg_with_loops_complexities(const char* module_name)
{
  return print_any_icfg_with_loops(module_name,ICFG_DECOR_COMPLEXITIES);
}

bool print_icfg_with_loops_preconditions(const char* module_name)
{
  return print_any_icfg_with_loops(module_name,ICFG_DECOR_PRECONDITIONS);
}

bool print_icfg_with_loops_total_preconditions(const char* module_name)
{
  return print_any_icfg_with_loops(module_name,ICFG_DECOR_TOTAL_PRECONDITIONS);
}

bool print_icfg_with_loops_transformers(const char* module_name)
{
 return print_any_icfg_with_loops(module_name,ICFG_DECOR_TRANSFORMERS);
}

bool print_icfg_with_loops_proper_effects(const char* module_name)
{
  return print_any_icfg_with_loops(module_name,ICFG_DECOR_PROPER_EFFECTS);
}

bool print_icfg_with_loops_cumulated_effects(const char* module_name)
{ return print_any_icfg_with_loops(module_name,ICFG_DECOR_CUMULATED_EFFECTS);}

bool print_icfg_with_loops_regions(const char* module_name)
{ return print_any_icfg_with_loops(module_name,ICFG_DECOR_REGIONS);}

bool print_icfg_with_loops_in_regions(const char* module_name)
{ return print_any_icfg_with_loops(module_name,ICFG_DECOR_IN_REGIONS);}

bool print_icfg_with_loops_out_regions(const char* module_name)
{ return print_any_icfg_with_loops(module_name,ICFG_DECOR_OUT_REGIONS);}

/*
 * ICFGs with controls
 */

bool print_icfg_with_control(const char* module_name)
{
  return print_any_icfg_with_control(module_name,ICFG_DECOR_NONE);
}

bool print_icfg_with_control_complexities(const char* module_name)
{
  return print_any_icfg_with_control(module_name,ICFG_DECOR_COMPLEXITIES);
}

bool print_icfg_with_control_preconditions(const char* module_name)
{
  return print_any_icfg_with_control(module_name,ICFG_DECOR_PRECONDITIONS);
}

bool print_icfg_with_control_total_preconditions(const char* module_name)
{
  return print_any_icfg_with_control(module_name,ICFG_DECOR_TOTAL_PRECONDITIONS);
}

bool print_icfg_with_control_transformers(const char* module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_TRANSFORMERS);}

bool print_icfg_with_control_proper_effects(const char* module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_PROPER_EFFECTS);}

bool print_icfg_with_control_cumulated_effects(const char* module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_CUMULATED_EFFECTS);}

bool print_icfg_with_control_regions(const char* module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_REGIONS);}

bool print_icfg_with_control_in_regions(const char* module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_IN_REGIONS);}

bool print_icfg_with_control_out_regions(const char* module_name)
{ return print_any_icfg_with_control(module_name,ICFG_DECOR_OUT_REGIONS);}
