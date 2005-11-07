/* An atomizer that uses the one made by Fabien Coelho for HPFC.

   $Id$

   Ronan Keryell, 17/5/1995. 
*/

#include "local.h"

extern entity hpfc_new_variable(entity, basic);
extern bool expression_constant_p(expression);

static bool 
simple_expression_decision(e)
expression e;
{
    syntax s = expression_syntax(e);

    /*  don't atomize A(I)
     */
    if (syntax_reference_p(s)) 
	return(!entity_scalar_p(reference_variable(syntax_reference(s))));

    /*  don't atomize A(1)
     */
    if (expression_constant_p(e)) 
	return(FALSE);

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

    return(TRUE);
}

static bool
new_atomizer_expr_decide(reference r, expression e)
{
    return(simple_expression_decision(e));
}


static bool
new_atomizer_func_decide(call c, expression e)
{
    entity f = call_function(c);
    syntax s = expression_syntax(e);

    if (ENTITY_ASSIGN_P(f)) return(FALSE);
    if (value_tag(entity_initial(f)) == is_value_intrinsic ||
	!syntax_reference_p(s))
	return(simple_expression_decision(e));
	
    /* the default is *not* to atomize.
     * should check that the reference is not used as a vector
     * and is not modified?
     */ 
    return(FALSE); 
}


/*
static entity
new_atomizer_create_a_new_entity(entity module_entity, tag variable_type)
{
   basic a_basic;
   int a_kind;
   
   basic_tag(a_basic) = variable_type;
   a_kind = TMP_ENT;
   return make_new_entity(a_basic, a_kind);
}
*/

boolean new_atomizer(char * mod_name)
{
   statement mod_stat;
   entity module;

   debug_on("ATOMIZER_DEBUG_LEVEL");

   if(get_debug_level() > 0)
      user_log("\n\n *** ATOMIZER for %s\n", mod_name);

   mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stat);

   module = local_name_to_top_level_entity(mod_name);
   set_current_module_entity(module);

   atomize_as_required(mod_stat,
                       new_atomizer_expr_decide,
                       new_atomizer_func_decide,
                       gen_false,
		       gen_false, /* range */
		       gen_false, /* whileloop */
                       /*new_atomizer_create_a_new_entity*/
                       hpfc_new_variable);

   ifdebug(1) print_statement(mod_stat);
      
   /* Reorder the module, because new statements may have been
      changed. */
   module_reorder(mod_stat);

   /* We save the new CODE. */
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);

   reset_current_module_statement();
   reset_current_module_entity();

   if(get_debug_level() > 0)
      user_log("\n\n *** ATOMIZER done\n");

   debug_off();

   return(TRUE);
}
