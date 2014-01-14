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

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "database.h"
#include "resources.h"
#include "pipsdbm.h"
#include "properties.h"
#include "effects-generic.h"
#include "effects-simple.h"

static list /* of entity */ get_variables_to_filter()
{
  string names = strdup(get_string_property("EFFECTS_FILTER_ON_VARIABLE"));
  list le = NIL;
  entity var;
  string saved = names, s;

  pips_assert("property EFFECTS_FILTER_ON_VARIABLE is defined",
	      !same_string_p(names, ""));

  for (s=names; *s; s++)
  {
    var = NULL;
    if (*s==',')
    {
      *s = '\0';
      var = gen_find_tabulated(names, entity_domain);
    
      if (!var || entity_undefined_p(var))
      {
	pips_user_warning("reference variable '%s' not found\n", names);
      }
      else
      {
	le = CONS(ENTITY, var, le);
      }
      *s = ',';
      names = s+1;
    }
  }

  var = gen_find_tabulated(names, entity_domain);

  if (!var || entity_undefined_p(var))
  {
    pips_user_warning("reference variable '%s' not found\n", names);
  }
  else
  {
    le = CONS(ENTITY, var, le);
  }

  free(saved), saved = NULL;

  return le;
}

/************************************** CHECK WHETHER A CONFLICTING W EFFECT */

static list /* of entity */ variables_to_filter = NIL;

static bool there_is_a_conflict(entity var)
{
  MAP(ENTITY, v,
  {
    if (entities_may_conflict_p(var, v))
      return true;
  },
      variables_to_filter);
  return false;
}

/***************************** (should) CHECK WHETHER A REFERENCE IS WRITTEN */

static bool direct_reference_found = false;
static entity a_variable = NULL;

/* it should be a check on call arguments, whether they are W + ref 
 * for user define functions, use summary effects
 * for intrinsics? =, implied-do, read...
 */
static void reference_rwt(reference r)
{
  if (entities_may_conflict_p(reference_variable(r), a_variable))
  {
    direct_reference_found = true;
    gen_recurse_stop(NULL);
  }
}

static void check_if_direct_reference(void * x)
{
  gen_recurse(x, reference_domain, gen_true, reference_rwt);
}

static bool direct_written_reference(statement s, entity var)
{
  instruction i = statement_instruction(s);
  direct_reference_found = false;
  a_variable = var;

  switch (instruction_tag(i))
  {
  case is_instruction_call:
    check_if_direct_reference(i);
    break;
  case is_instruction_loop:
    {
      loop l = instruction_loop(i);
      check_if_direct_reference(loop_range(l));
      if (!direct_reference_found && entities_may_conflict_p(loop_index(l), var))
	direct_reference_found = true;
      break;
    }
  case is_instruction_whileloop:
    check_if_direct_reference(whileloop_condition(instruction_whileloop(i)));
    break;
  case is_instruction_test:
    check_if_direct_reference(test_condition(instruction_test(i)));
    break;
  default:
    /* should not happen on a statement with proper effects */
    pips_internal_error("unexpected instruction tag...");
  }

  a_variable = NULL;
  return direct_reference_found;
}

/*************************************** FILTER PROPER EFFECTS OF STATEMENTS */

static bool stmt_flt(statement s)
{
  list /* of effect */ lpe = load_proper_rw_effects_list(s);
  MAP(EFFECT, e,
  {
    entity var = effect_variable(e);
    if (effect_write_p(e) && there_is_a_conflict(var))
    {
      if (direct_written_reference(s, var))
      {
	int order = statement_ordering(s);
	int on; int os;
	if (order!=STATEMENT_ORDERING_UNDEFINED)
	{
	  on = ORDERING_NUMBER(order);
	  os = ORDERING_STATEMENT(order);
	}
	else
	{
	  on = -1;
	  os = -1;
	}

	user_log("## %s o=(%d,%d)/n=%d\n", 
		 entity_name(var), on, os, statement_number(s));
      }
    }
  },
      lpe);
  return true;
}

/***************************************************************** INTERFACE */

bool filter_proper_effects(const char* module_name)
{
  debug_on("FILTER_PROPER_EFFECTS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
 
  /* gets what is needed from PIPS DBM
   */
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  set_current_module_statement((statement)
	    db_get_memory_resource(DBR_CODE, module_name, true));
  set_proper_rw_effects((statement_effects)
	    db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true));
 
  variables_to_filter = get_variables_to_filter();

  if (variables_to_filter)
    gen_recurse(get_current_module_statement(),
		statement_domain, stmt_flt, gen_null);

  gen_free_list(variables_to_filter);
  variables_to_filter = NIL;
  
  /* returns the result to the DBM... */
  DB_PUT_MEMORY_RESOURCE
    (DBR_FILTERED_PROPER_EFFECTS, module_name, NULL);
  
  reset_proper_rw_effects();
  reset_current_module_entity();
  reset_current_module_statement();
  
  debug_off();
  return true;
}
