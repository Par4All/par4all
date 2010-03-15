#include <stdlib.h>
#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "makefile.h"
#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "syntax.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "pipsmake.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "transformations.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

#define POINTS_TO_MODULE_NAME "*POINTS_TO_MODULE*"
/* --------------------------------Interprocedural Analysis-----------------------*/
/*This package computes the points-to interprocedurally.*/
void points_to_forward_translation()
{

}

void points_to_backward_translation()
{
}

/*First implementation, just handle simple parameter like int *p. */
void formal_points_to_parameter(call c)
{
	entity f = call_function(c);
	list args = call_arguments(c);
	FOREACH(EXPRESSION,e,args)
	{ entity ee = expression_to_entity(e);
		type t = expression_to_type(e);
		type tt = type_to_pointed_type(t);
		string formal_name = strdup(concatenate(POINTS_TO_MODULE_NAME,MODULE_SEP_STRING,entity_name(ee) ));
		entity formal_parameter = find_or_create_entity(formal_name);
		if(entity_undefined_p(formal_parameter)) {
			formal_parameter = make_entity(formal_name,
			   tt, make_storage_rom(), make_value_unknown());
		}
	}

}


/*This function should initialize the summary points to of all the
  functions of the program, it returns an empty set.*/
void init_summary_points_to()
{
  set init_set =set_generic_make(set_private,
				 points_to_equal_p,points_to_rank);
  return init_set;
}


bool intraprocedural_summary_points_to_analysis(char * module_name)
{
  entity module;
  statement module_stat;
  list pt_list = NIL;

  //  init_pt_to_list();

  set_current_module_entity(module_name_to_entity(module_name));
  set_methods_for_proper_simple_effects();
  module = get_current_module_entity();

  //(*effects_computation_init_func)(module_name);

  debug_on("POINTS_TO_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE,
						       module_name, TRUE) );
  module_stat = get_current_module_statement();
  
  //DB_PUT_MEMORY_RESOURCE
  //	(DBR_POINTS_TO_LIST, module_name, get_pt_to_list());

  //reset_pt_to_list();
  reset_current_module_entity();
  reset_current_module_statement();
  
  debug_off();

  bool good_result_p = TRUE;
  return (good_result_p);

}
