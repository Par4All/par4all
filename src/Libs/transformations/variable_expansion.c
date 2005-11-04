/* transformation package :  Francois Irigoin, October 2005
 *
 * variable_expansion.c
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the functions expanding local variables used in DO loops into arrays.
 *
 * $Id$
 */

#include <stdlib.h>
#include <stdio.h>
/* #include <malloc.h> */
/* #include <string.h> */

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "pipsdbm.h"

#include "misc.h"

#include "resources.h"

/* bool scalar_expansion(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : replaces local scalar variables by arrays and updates the corresponding references
 * comment  : private scalars must have been detected first (see PRIVATIZE_MODULE).
 *            array expansion would be performed in the exact same way.
 */

/* list of expressions referencing the loop indices */
static list loop_indices = list_undefined;

static list loop_dimensions = list_undefined;

static entity expanded_variable = entity_undefined;

static list processed_variables = list_undefined;

static bool perform_reference_expansion(reference r)
{
  entity v = reference_variable(r);

  if(v==expanded_variable) {
    list il = reference_indices(r);

    pips_assert("scalar reference", il==NIL);

    pips_debug(9, "for variable %s\n", entity_local_name(v));

    reference_indices(r) = gen_copy_seq(loop_indices);

    ifdebug(9) {
      pips_debug(9, "New reference:");
      print_reference(r);
      pips_debug(9, "\n");
    }
  }
  return TRUE;
}

static bool prepare_expansion(loop l)
{
  entity i = loop_index(l);
  range r = loop_range(l);
  expression lb = range_lower(r);
  expression ub = range_upper(r);
  expression inc = range_increment(r);
  bool go_down = FALSE;
  int ilb = 0;
  int iub = 0;
  int iinc = 0;
  
  /* Is this loop OK? */
  if(expression_integer_value(lb, &ilb)
     && expression_integer_value(ub, &iub)
     && expression_integer_value(inc, &iinc)
     && (iinc==1 /* || iiinc==-1 */ )) {
    dimension d = make_dimension(copy_expression(lb), copy_expression(ub));
    expression ie = entity_to_expression(i);

    /* Update information about the nesting loops. */
    loop_dimensions = gen_append(loop_dimensions, CONS(DIMENSION, d, NIL));
    loop_indices = gen_append(loop_indices, CONS(ENTITY, ie, NIL));
    go_down = TRUE;

    ifdebug(9) {
      pips_debug(9, "Going down, local variables: ");
      print_entities(loop_locals(l));
      pips_debug(9, "\n");
    }
  }

  /* If go down is FALSE here, we should reset loop_indices and
     loop_dimensions and go down anyway.*/

  return go_down;
}

static void perform_expansion_and_unstack_index_and_dimension(loop l)
{
  entity i = loop_index(l);

  /* Select loops marked as relevant on the way down. */
  if(!list_undefined_p(loop_indices)) {
    expression eli = EXPRESSION(CAR(gen_last(loop_indices)));
    if(i==reference_variable(expression_reference(eli))) {
      dimension d = DIMENSION(CAR(gen_last(loop_dimensions)));
      list evl = NIL;

      ifdebug(9) {
	pips_debug(9, "Going up, local variables: ");
	print_entities(loop_locals(l));
	pips_debug(9, "\n");
      }

      /* Does it contain private variables? */
      MAP(ENTITY, lv, {
	/* Do not expand loop indices nor variables already processed! */
	if(lv!=i && !gen_in_list_p(lv, processed_variables)) {
	  type t = entity_type(lv);
	  variable v = type_variable(t);
	  list dims = variable_dimensions(v);
	  statement bs = loop_body(l);

	  pips_assert("Scalar expansion", dims==NIL);

	  evl = CONS(ENTITY, lv, evl);

	  pips_debug(9, "Update type of %s\n", entity_local_name(lv));

	  /* Update its type */
	  variable_dimensions(v) = gen_full_copy_list(loop_dimensions);

	  /* print_type(); */

	  /* Update its references in the loop body */

	  pips_debug(9, "Expand references to %s\n", entity_local_name(lv));
	  expanded_variable = lv;
	  gen_recurse(bs, reference_domain, perform_reference_expansion, gen_null);
	  expanded_variable = entity_undefined;
	}
      }, loop_locals(l));

      /* Remove the expanded variables and the loop index from the local variable list */
      gen_list_and_not(&loop_locals(l), processed_variables);
      gen_list_and_not(&loop_locals(l), evl);
      processed_variables = gen_append(processed_variables, evl);
      processed_variables = gen_append(processed_variables, CONS(ENTITY, i, NIL));

      gen_remove(&loop_indices, (void *) eli);
      gen_remove(&loop_dimensions, (void *) d);
      free_dimension(d);
    }
  }
}

bool scalar_expansion(char *module_name)
{
    entity module;
    statement module_stat;

    pips_user_warning("\nExperimental phase: on-going debugging!\n");
    pips_user_warning("\nPrivatize variables before you run this phase\n");

    /* Why would I need this? To access declarations for instance */
    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();

    /* Get the code of the module. */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();

    debug_on("SCALAR_EXPANSION_DEBUG_LEVEL");
    pips_debug(1, "begin\n");

    /*

      Go down statements recursively.
      Each time you enter a loop:
       * if the loop bounds are constant and if the increment is one,
       stack them on the bound stack(s) together with the loop index;
       * else, stop the recursive descent.
      When new constants bounds have been found, look for local scalar variables.
      Modify the declaration of the scalar variable in the symbol table according 
      to the bound stack.
      Modify all its references in the body statement, using the stacked loop indices
      as reference.
      Remove it from the local variable field of the current DO loop.

     */

    loop_indices = NIL;
    loop_dimensions = NIL;
    processed_variables = NIL;
    gen_recurse(module_stat, loop_domain, prepare_expansion,
		perform_expansion_and_unstack_index_and_dimension);

    pips_assert("Supporting lists are both empty", loop_indices==NIL);
    pips_assert("Supporting lists are both empty", loop_dimensions==NIL);

    loop_indices = list_undefined;
    loop_dimensions = list_undefined;
    gen_free_list(processed_variables);
    processed_variables = list_undefined;
    reset_current_module_entity();
    reset_current_module_statement();

    /* Declarations must be regenerated for the code to be compilable */
    free(code_decls_text(value_code(entity_initial(module))));
    code_decls_text(value_code(entity_initial(module))) = strdup("");

    pips_debug(1, "end\n");
    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);

    return(TRUE);
}

bool variable_expansion(char *module_name)
{
  return scalar_expansion(module_name);
}
