/* An atomizer that uses the one made by Fabien Coelho for HPFC,
   and is in fact just a hacked version of the one made by Ronan
   Keryell...
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "ri-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "transformer.h"
#include "semantics.h"
#include "conversion.h" 
#include "control.h"
#include "transformations.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"
#include "atomizer.h"

extern entity hpfc_new_variable(entity, basic);
//extern bool expression_constant_p(expression);

#define SCALAR_REFERENCE_P(r) \
    entity_scalar_p(reference_variable(r))

#define SYNTAX_CONSTANT_P(s) \
    (syntax_call_p(s) && entity_constant_p(call_function(syntax_call(s))))

static bool simd_simple_expression_decision(expression e)
{
    syntax s = expression_syntax(e);

    switch(syntax_tag(s))
    {
       case is_syntax_reference:
	  /* do not atomize A(I) */
	  return(!SCALAR_REFERENCE_P(syntax_reference(s)));

       case is_syntax_call:
       {
	  call c = syntax_call(s);
	  entity en = call_function(c);
	  
	  if (entity_constant_p(en))
	     /* do not atomize A(1) */
	     return FALSE;
	  else if (ENTITY_PLUS_P(en))
	  {
	     list l = call_arguments(c);
	     syntax e1, e2;
	     
	     e1 = expression_syntax(EXPRESSION(CAR(l)));
	     e2 = expression_syntax(EXPRESSION(CAR(CDR(l))));
	     
	     if (SYNTAX_CONSTANT_P(e1))
		/* do not atomize A(1+I) */
		return !( syntax_reference_p(e2) &&
			  SCALAR_REFERENCE_P(syntax_reference(e2)) );
	     else if ( syntax_reference_p(e1) &&
		       SCALAR_REFERENCE_P(syntax_reference(e1)) )
		/* do not atomize A(I+1) */
		return !SYNTAX_CONSTANT_P(e2);
	     else
		return TRUE;
	  }
	  else
	     return TRUE;
       }

       case is_syntax_range:
       default:
	  return TRUE;
    }

    return(TRUE);
}

static bool simd_atomizer_expr_decide(reference r, expression e)
{
    return(simd_simple_expression_decision(e));
}


static bool simd_atomizer_func_decide(call c, expression e)
{
    entity f = call_function(c);
    syntax s = expression_syntax(e);

    if (ENTITY_ASSIGN_P(f)) 
       return(FALSE);

    if (value_tag(entity_initial(f)) == is_value_intrinsic)
       return !syntax_reference_p(s) && !SYNTAX_CONSTANT_P(s);

    /* the default is *not* to atomize.
     * should check that the reference is not used as a vector
     * and is not modified?
     */ 
    return(FALSE); 
}

boolean simd_atomizer(char * mod_name)
{
   statement mod_stat;
   entity module;

   debug_on("SIMD_ATOMIZER_DEBUG_LEVEL");

   if(get_debug_level() > 0)
      user_log("\n\n *** SIMD ATOMIZER for %s\n", mod_name);

   mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stat);

   module = local_name_to_top_level_entity(mod_name);
   set_current_module_entity(module);

   atomize_as_required(mod_stat,
                       simd_atomizer_expr_decide,
                       simd_atomizer_func_decide,
                       gen_false,
		       gen_false, /* range */
		       gen_false, /* whileloop */
                       /*new_atomizer_create_a_new_entity*/
                       hpfc_new_variable);

   ifdebug(1) print_statement(mod_stat);
      
   /* We save the new CODE. */
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);

   module_reorder(mod_stat);

   reset_current_module_statement();
   reset_current_module_entity();

   if(get_debug_level() > 0)
      user_log("\n\n *** SIMD ATOMIZER done\n");

   debug_off();

   return(TRUE);
}
