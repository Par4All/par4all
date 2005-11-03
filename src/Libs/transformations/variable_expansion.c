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

#include "misc.h"

#include "resources.h"

/* bool scalar_expansion(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : replaces local scalar variables by arrays and updates the corresponding references
 * comment  : private scalars must have been detected first (see PRIVATIZE_MODULE).
 *            array expansion would be performed in the exact same way.
 */
static list loop_indices = list_undefined;

static list loop_dimensions = list_undefined;

static entity expansed_variable = entity_undefined;

static bool expanse_reference(reference r)
{
  entity v = reference_variable(r);

  if(v==expansed_variable) {
    list il = reference_indices(r);

    pips_assert("scalar reference", il==NIL);

    reference_indices(r) = gen_copy_seq(loop_indices);
  }
  return TRUE;
}

static bool prepare_and_perform_expansion(loop l)
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

    /* Update information about the nesting loops. */
    loop_dimensions = gen_append(loop_dimensions, CONS(DIMENSION, d, NIL));
    loop_indices = gen_append(loop_indices, CONS(ENTITY, i, NIL));
    go_down = TRUE;

    /* Does it contain private variables? */
    MAP(ENTITY, lv, {
      type t = entity_type(lv);
      variable v = type_variable(t);
      list dims = variable_dimensions(v);
      statement bs = loop_body(l);

      pips_assert("Scalar expansion", dims==NIL);

      /* Update its type */
      variable_dimensions(v) = gen_full_copy_list(loop_dimensions);

      /* Update its references in the loop body */
      expansed_variable = lv;
      gen_recurse(bs, reference_domain, expanse_reference, gen_null);

    }, loop_locals(l));

  }

  /* If go down is FALSE here, we should reset loop_indices and
     loop_dimensions and go down anyway.*/

  return go_down;
}

static void unstack_index_and_dimension(loop l)
{
  entity i = loop_index(l);

  if(!list_undefined_p(loop_indices)) {
    entity li = ENTITY(CAR(gen_last(loop_indices)));
    if(i==li) {
      dimension d = DIMENSION(CAR(gen_last(loop_dimensions)));

      free_dimension(d);
      gen_remove(&loop_indices, gen_last(loop_indices));
      gen_remove(&loop_dimensions, gen_last(loop_dimensions));
    }
  }
}

bool scalar_expansion(char *module_name)
{
    entity module;
    statement module_stat;

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
    gen_recurse(module_stat, loop_domain, prepare_and_perform_expansion,
		unstack_index_and_dimension);

    reset_current_module_entity();
    reset_current_module_statement();

    pips_debug(1, "end\n");
    debug_off();

    return(TRUE);
}
