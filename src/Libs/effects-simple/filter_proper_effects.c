/*
 * $Id$
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"
#include "database.h"
#include "resources.h"
#include "pipsdbm.h"
#include "properties.h"
#include "effects-generic.h"
#include "effects-simple.h"

static entity get_variable_to_filter()
{
  string reference_variable_name = 
    get_string_property("EFFECTS_FILTER_ON_VARIABLE");
  entity reference_variable;

  pips_assert("property EFFECTS_FILTER_ON_VARIABLE is defined",
	      !same_string_p(reference_variable_name, ""));

  reference_variable = 
    gen_find_tabulated(reference_variable_name, entity_domain);

  if (entity_undefined_p(reference_variable))
  {
    pips_user_warning("reference variable '%s' not found\n",
		      reference_variable_name);
    return NULL;
  }

  return reference_variable;
}

/************************************** CHECK WHETHER A CONFLICTING W EFFECT */

static bool direct_reference_found = FALSE;
static entity variable_to_filter = NULL;

static void reference_rwt(reference r)
{
  if (entity_conflict_p(reference_variable(r), variable_to_filter))
  {
    direct_reference_found = TRUE;
    gen_recurse_stop(NULL);
  }
}

static void check_if_direct_reference(void * x)
{
  gen_recurse(x, reference_domain, gen_true, reference_rwt);
}

static bool direct_reference_in_call(statement s)
{
  instruction i = statement_instruction(s);
  direct_reference_found = FALSE;
  switch (instruction_tag(i))
  {
  case is_instruction_call:
    check_if_direct_reference(i);
    break;
  case is_instruction_loop:
    {
      loop l = instruction_loop(i);
      check_if_direct_reference(loop_range(l));
      if (!direct_reference_found &&
	  entity_conflict_p(loop_index(l), variable_to_filter))
	direct_reference_found = TRUE;
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
  return direct_reference_found;
}

/*************************************** FILTER PROPER EFFECTS OF STATEMENTS */

static bool stmt_flt(statement s)
{
  list /* of effect */ lpe = load_proper_rw_effects_list(s);
  MAP(EFFECT, e,
  {
    if (effect_write_p(e) && 
	entity_conflict_p(effect_variable(e), variable_to_filter))
    {
      if (direct_reference_in_call(s))
      {
	user_log("## %s %s o=%d/n=%d\n", 
		 entity_name(get_current_module_entity()),
		 entity_name(effect_variable(e)),
		 statement_ordering(s),
		 statement_number(s));
      }
    }
  },
      lpe);
  return TRUE;
}

/***************************************************************** INTERFACE */

boolean filter_proper_effects(string module_name)
{
  debug_on("FILTER_PROPER_EFFECTS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
 
  /* gets what is needed from PIPS DBM
   */
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  set_current_module_statement((statement)
	    db_get_memory_resource(DBR_CODE, module_name, TRUE));
  set_proper_rw_effects((statement_effects)
	    db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE));
 
  variable_to_filter = get_variable_to_filter();

  if (variable_to_filter)
    gen_recurse(get_current_module_statement(),
		statement_domain, stmt_flt, gen_null);

  variable_to_filter = NULL;
  
  /* returns the result to the DBM... */
  DB_PUT_MEMORY_RESOURCE
    (DBR_FILTERED_PROPER_EFFECTS, module_name, NULL);
  
  reset_proper_rw_effects();
  reset_current_module_entity();
  reset_current_module_statement();
  
  debug_off();
  return TRUE;
}
