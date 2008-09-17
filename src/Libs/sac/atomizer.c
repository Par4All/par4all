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
#include "control.h"
#include "transformations.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"
#include "atomizer.h"

#include "expressions.h"

#include "sac-local.h" 

#include "sac.h"

static statement oriStat = NULL;

static basic simd_basic_of_expression(expression exp);

/* 
This function computes the basic of an intrinsic
 */
static basic simd_basic_of_intrinsic(call c)
{
  entity f = call_function(c);
  type rt = functional_result(type_functional(entity_type(f)));
  basic rb = copy_basic(variable_basic(type_variable(rt)));
										       
  pips_debug(7, "Intrinsic call to %s with result type %s\n",
	     module_local_name(f),
	     basic_to_string(rb));

  if(basic_overloaded_p(rb)) {
    list args = call_arguments(c);

    if (ENDP(args)) {
      /* I don't know the type since there is no arguments !
	 Bug encountered with a FMT=* in a PRINT.
	 RK, 21/02/1994 : */
      /* leave it overloaded */
      ;
    }
    else {
      free_basic(rb);
      rb = simd_basic_of_expression(EXPRESSION(CAR(args)));

      MAP(EXPRESSION, arg, {
	basic b = simd_basic_of_expression(arg);

        if(basic_undefined_p(rb) || basic_undefined_p(b))
           break;

	basic new_rb = basic_maximum(rb, b);

	free_basic(rb);
	free_basic(b);
	rb = new_rb;
      }, CDR(args));
    }

  }

  return rb;
}

/* 
This function computes the basic of a call
 */
static basic simd_basic_of_call(call c)
{
    entity e = call_function(c);
    tag t = value_tag(entity_initial(e));
    basic b = basic_undefined;

    switch (t)
    {
    case is_value_code:
	b = copy_basic(basic_of_external(c));
	break;
    case is_value_intrinsic: 
	b = simd_basic_of_intrinsic(c);
	break;
    case is_value_symbolic: 
	/* b = make_basic(is_basic_overloaded, UU); */
	b = copy_basic(basic_of_constant(c));
	break;
    case is_value_constant:
	b = copy_basic(basic_of_constant(c));
	break;
    case is_value_unknown:
	debug(1, "simd_basic_of_call", "function %s has no initial value.\n"
	      " Maybe it has not been parsed yet.\n",
	      entity_name(e));
	b = copy_basic(basic_of_external(c));
	break;
    default: pips_error("simd_basic_of_call", "unknown tag %d\n", t);
	/* Never go there... */
    }
    return b;
}

/* 
This function computes the basic of an expression
 */
static basic simd_basic_of_expression(expression exp)
{
   syntax syn = expression_syntax(exp);
   basic bas = basic_undefined;

   switch(syntax_tag(syn))
   {
      case is_syntax_reference:
      {
	 bas = copy_basic(get_basic_from_array_ref(syntax_reference(syn)));
	 break;
      }

      case is_syntax_call:
      {
	 call ca = syntax_call(syn);

	 bas = simd_basic_of_call(ca);

         break;
      }

      default:
         pips_error("simd_basic_of_expression", "Bad syntax tag");
   }

   return bas;
}

/* returns the assignment statement is moved, or NULL if not.
 */
statement simd_atomize_this_expression(entity (*create)(entity, basic),
				       expression e)
{
  basic bofe;

  /* it does not make sense to atomize a range...
   */
  if (syntax_range_p(expression_syntax(e))) return NULL;
  
  bofe = simd_basic_of_expression(e);

  if(!basic_undefined_p(bofe)) {
    if (!basic_overloaded_p(bofe))
      {
	entity newvar; 
	expression rhs;
	statement assign;
	syntax ref;

	newvar = (*create)(get_current_module_entity(), bofe);
	rhs = make_expression(expression_syntax(e), normalized_undefined);
	normalize_all_expressions_of(rhs);

	ref = make_syntax(is_syntax_reference, make_reference(newvar, NIL));

	assign = make_assign_statement(make_expression(copy_syntax(ref), 
						       normalized_undefined), 
				       rhs);
	expression_syntax(e) = ref;

	return assign;
      }
  
    free_basic(bofe);
  }
  return NULL;
}

/* 
This function computes the maximum width of all the variables used in a call
 */
static void get_type_max_width(call ca, int* maxWidth)
{
   MAP(EXPRESSION,
       arg,
   {
      syntax s = expression_syntax(arg);

      switch(syntax_tag(s))
      {
	 case is_syntax_call:
	 {
	    call c = syntax_call(s);

	    if (!call_constant_p(c))
	       get_type_max_width(c, maxWidth);
	    break;
	 }
	    
	 case is_syntax_reference:
	 {
	    basic bas = get_basic_from_array_ref(syntax_reference(s));
	    switch(basic_tag(bas))
	    {
	       case is_basic_int: if(*maxWidth < basic_int(bas)) *maxWidth = basic_int(bas); break;
	       case is_basic_float: if(*maxWidth < basic_float(bas)) *maxWidth = basic_float(bas); break;
	       case is_basic_logical: if(*maxWidth < basic_logical(bas)) *maxWidth = basic_logical(bas); break;
	    }
	 }

      }
   },
       call_arguments(ca));

}

/* 
This function aims at changing the basic size of the left expression of
the newly created assign statement
 */
static void change_basic_if_needed(statement stat)
{
   expression lExp = EXPRESSION(CAR(call_arguments(statement_call(stat))));
   expression rExp = EXPRESSION(CAR(CDR(call_arguments(statement_call(stat)))));
   int maxWidth = -1;

   // Check that the right expression is a call statement
   if(!expression_call_p(rExp))
   {
      return;
   }

   // Check that the statement can be potentially integrated in a 
   // SIMD statement
   if(match_statement(stat) != NIL)
   {
      get_type_max_width(syntax_call(expression_syntax(rExp)), &maxWidth);
   }

   // If the maxWidth of the right expression is smaller than the width 
   // of the current left expression, then replace the left expression width 
   // by maxWidth
   if(maxWidth > 0)
   {
      basic lExpBasic = expression_basic(lExp);

      switch(basic_tag(lExpBasic))
      {
         case is_basic_int: if(basic_int(lExpBasic) > maxWidth) basic_int(lExpBasic) = maxWidth; break;
         case is_basic_float: if(basic_float(lExpBasic) > maxWidth) basic_float(lExpBasic) = maxWidth; break;
         case is_basic_logical: if(basic_logical(lExpBasic) > maxWidth) basic_logical(lExpBasic) = maxWidth; break;
      }
   }
}

/* 
This function insert stat before oriStat in the code
 */
static void simd_insert_statement(statement cs, statement stat)
{
   // If cs is already a sequence, we just need to insert stat in cs
   if(instruction_sequence_p(statement_instruction(cs)))
   {
      instruction_block(statement_instruction(cs)) = gen_insert_before(stat,
		        oriStat,
	       	        instruction_block(statement_instruction(cs)));
   }
   // If cs is not a sequence, we have to create one sequence composed of
   // cs then oriStat
   else
   {
      statement_label(stat) = statement_label(cs);

      oriStat = make_statement(entity_empty_label(), 
			       statement_number(cs),
			       statement_ordering(cs),
			       statement_comments(cs),
			       statement_instruction(cs),
			       NIL,NULL);

      statement_instruction(cs) =
	make_instruction_block(CONS(STATEMENT, stat,
				    CONS(STATEMENT,
					 oriStat,
					 NIL)));

      statement_label(cs) = entity_empty_label();
      statement_number(cs) = STATEMENT_NUMBER_UNDEFINED;
      statement_ordering(cs) = STATEMENT_ORDERING_UNDEFINED;
      statement_comments(cs) = empty_comments;
   }
}

/* 
This function recursively atomizes a call
 */
static void simd_atomize_call(call c, statement cs)
{

   // Each call argument is atomize if needed
   MAP(EXPRESSION, ce,
   {
      syntax s = expression_syntax(ce);

      // Atomize expression only if this is a call expression
      if(syntax_call_p(s))
      {
         call cc = syntax_call(s);

	 // Atomize expression only if the call is not a constant
	 if(FUNC_TO_ATOMIZE_P(cc))
	 {
	    simd_atomize_call(cc, cs);

	    statement stat = simd_atomize_this_expression(hpfc_new_variable, ce);

	    if(stat == NULL)
	    {
	       return;
	    }

            change_basic_if_needed(stat);

            simd_insert_statement(cs, stat);
	 }
      }
   }, call_arguments(c));
}

/* 
This function is called for each call statement and atomize it
 */
static void atomize_call_statement(statement cs)
{
   statement si = cs;
   int count = 0;

   call c = instruction_call(statement_instruction(si));

   // For each call argument, the argument is atomized
   // if needed
   MAP(EXPRESSION, ce,
   {
      syntax s = expression_syntax(ce);

      // Atomize expression only if this is a call expression
      if(syntax_call_p(s))
      {
	 call cc = syntax_call(s);

	 // Atomize expression only if the call is not a constant
	 if(FUNC_TO_ATOMIZE_P(cc))
	 {

	    // Initialize oriStat if this is the first argument
	    if(count == 0)
               oriStat = si;

	    // If the current call is not an assign call,
	    // let's atomize the current argument
	    if(!ENTITY_ASSIGN_P(call_function(c)))
	    {
               simd_atomize_call(cc, cs);

	       statement stat = simd_atomize_this_expression(hpfc_new_variable, ce);

               simd_insert_statement(cs, stat);
	    }
	    // If the current call is an assign call,
	    // the current argument is not atomize
	    else
	    {
               simd_atomize_call(cc, cs);
	    }
	 }
      }

      count++;
   }, call_arguments(c));
}

/* 
This function is called for all statements in the code
 */
static void atomize_statements(statement cs)
{
   // Only a call statement can be atomized
   if (instruction_call_p(statement_instruction(cs)))
   {
      atomize_call_statement(cs);
   }
}

boolean simd_atomizer(char * mod_name)
{
   /* get the resources */
   statement mod_stmt = (statement)
      db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stmt);
   set_current_module_entity(module_name_to_entity(mod_name));

   debug_on("SIMD_ATOMIZER_DEBUG_LEVEL");
   /* Now do the job */

   init_tree_patterns();
   init_operator_id_mappings();

   gen_recurse(mod_stmt, statement_domain,
	       gen_true, atomize_statements);

   /* Reorder the module, because new statements have been added */  
   module_reorder(mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
 
   /* update/release resources */
   reset_current_module_statement();
   reset_current_module_entity();

   debug_off();

   return TRUE;
}
