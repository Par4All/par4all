/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

/* An atomizer that uses the one made by Fabien Coelho for HPFC.
   based on Ronan Keryell atomizer

   Alain Muller 9/10/2008
*/
#include "defines-local.h"

extern void atomize_as_required(
  statement stat, 
  bool (*ref_decide)(reference, expression), /* reference */
  bool (*call_decide)(call, expression), /* call */
  bool (*test_decide)(test, expression), /* test */
  bool (*range_decide)(range, expression), /* range */
  bool (*while_decide)(whileloop, expression), /* whileloop */
  entity (*new)(entity, basic));

static expression step_expression_atomized=expression_undefined;

GENERIC_LOCAL_FUNCTION(step_atomized, step_atomized_map)

entity step_new_variable(entity module, basic b)
{
  entity e=make_new_scalar_variable(module, copy_basic(b));
  store_or_update_step_atomized(e,copy_expression(step_expression_atomized));
  return e;
}

static bool 
step_simple_expression_decision(e)
     expression e;
{
  syntax s = expression_syntax(e);
  step_expression_atomized=e;

  /* do not atomize A(I+5) as it is compiled as (A+5)(I). We miss
     A(-1+I), unless expression_constant_p() accepts it. */
  if(!get_bool_property("ATOMIZE_ARRAY_ACCESSES_WITH_OFFSETS")) {
    if(syntax_call_p(s)) {
      call c = syntax_call(s);
      entity op = call_function(c);
      
      if(ENTITY_PLUS_P(op) || ENTITY_MINUS_P(op)) {
	expression e1 = EXPRESSION(CAR(call_arguments(c)));
	expression e2 = EXPRESSION(CAR(CDR(call_arguments(c))));
	syntax s1 = expression_syntax(e1);
	syntax s2 = expression_syntax(e2);
	
	if(syntax_reference_p(s1))
	  return(!(entity_scalar_p(reference_variable(syntax_reference(s1)))
		   && expression_constant_p(e2)));
	else if(syntax_reference_p(s2))
	  return(!(entity_scalar_p(reference_variable(syntax_reference(s2)))
		   && expression_constant_p(e1)));
      }
    }
  }

  /*  don't atomize symbolic constant
   */
  if(syntax_call_p(s)&&value_symbolic_p(entity_initial(call_function(syntax_call(s)))))
      return FALSE;

  /*  don't atomize A
   */
  if (syntax_reference_p(s))
    return(!ENDP(reference_indices(syntax_reference(s)))); 
  
  /*  don't atomize A(I)
   */
  if (syntax_reference_p(s))
    return(!entity_scalar_p(reference_variable(syntax_reference(s))));
  
  /*  don't atomize A(1)
   */
  if (expression_constant_p(e)) 
    return(FALSE);
  
  return(TRUE);
}

static bool
step_atomizer_ref_decide(reference __attribute__ ((unused)) r, expression e)
{
  step_expression_atomized=e; 
  return step_simple_expression_decision(e);
}

static bool
step_atomizer_func_decide(call c, expression e)
{
  entity f = call_function(c);
  syntax s = expression_syntax(e);

  step_expression_atomized=e;
 
  if (ENTITY_ASSIGN_P(f))
    return FALSE;

  if (value_tag(entity_initial(f)) == is_value_intrinsic ||
      !syntax_reference_p(s))
    return step_simple_expression_decision(e);
  
  /* the default is *not* to atomize.
   * should check that the reference is not used as a vector
   * and is not modified?
   */ 
  return FALSE;
}


boolean step_atomize(char * mod_name)
{
  statement mod_stat;
  entity module;

  init_step_atomized();
  mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

  set_current_module_statement(mod_stat);
  module = local_name_to_top_level_entity(mod_name);
  set_current_module_entity(module);
  
  atomize_as_required(mod_stat,
		      step_atomizer_ref_decide,
		      step_atomizer_func_decide,
		      (bool (*)(test, expression))gen_false,
		      (bool (*)(range, expression)) gen_false,
		      (bool (*)(whileloop, expression)) gen_false,
		      step_new_variable);
    
  /* Reorder the module, because new statements may have been
     changed. */
  module_reorder(mod_stat);
  
  /* We save the new CODE. */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_ATOMIZED, strdup(mod_name), get_step_atomized());
  
  reset_current_module_statement();
  reset_current_module_entity();
  reset_step_atomized();
  
  return(TRUE);
}

static bool step_expression_atomize_filter(expression expr)
{
  pips_debug(1,"entity_name = %s\n",entity_name(expr_to_entity(expr)));

  if (expression_reference_p(expr) && bound_step_atomized_p(reference_variable(syntax_reference(expression_syntax(expr)))))
    {
      entity e=reference_variable(syntax_reference(expression_syntax(expr)));
      free_syntax(expression_syntax(expr));
      expression_syntax(expr)=copy_syntax(expression_syntax(load_step_atomized(e)));
      delete_step_atomized(e);
    }

  return TRUE;
}

static bool step_call_atomize_filter(call c)
{
  entity f = call_function(c);

  pips_debug(2,"%s\n",entity_global_name(f));
  if (ENTITY_ASSIGN_P(f))
    {
      expression e1 = EXPRESSION(CAR(call_arguments(c)));
      syntax s1 = expression_syntax(e1);
      if(syntax_reference_p(s1)&&bound_step_atomized_p(reference_variable(syntax_reference(s1))))
	{
	  call_function(c)=entity_intrinsic(CONTINUE_FUNCTION_NAME);
	  gen_free_list(call_arguments(c));
	  call_arguments(c)=NIL;
	  return FALSE;
	}
    }
  return TRUE;
}

boolean step_unatomize(char * mod_name)
{
  statement mod_stat;
  entity module;

  set_step_atomized((step_atomized_map)db_get_memory_resource(DBR_STEP_ATOMIZED, mod_name, TRUE));
  mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);
  
  set_current_module_statement(mod_stat);
  module = local_name_to_top_level_entity(mod_name);
  set_current_module_entity(module);

  // do the work
  gen_multi_recurse(mod_stat,
		    expression_domain, step_expression_atomize_filter, gen_null,
		    call_domain, step_call_atomize_filter, gen_null,
		    NULL);
  
  ifdebug(1) print_statement(mod_stat);
  
  /* Reorder the module, because new statements may have been
     changed. */
  module_reorder(mod_stat);
  
  /* We save the new CODE. */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_STEP_ATOMIZED, strdup(mod_name), get_step_atomized());
  
  reset_current_module_statement();
  reset_current_module_entity();
  reset_step_atomized();
  
  return(TRUE);
}
