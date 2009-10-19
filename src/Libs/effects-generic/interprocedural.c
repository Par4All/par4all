/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: interprocedural.c
 * ~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the interprocedural 
 * computation of all types of in effects.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"

#include "misc.h"
#include "properties.h"
#include "text-util.h"
#include "ri-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))




/**************************************************** FORTRAN */

/**
   
 This function translates the list of effects l_sum_eff summarizing the 
 effects of module callee from its name space to the name space of the
 caller, that is to say the current module being analyzed.
 It is generic, which means that it does not depend on the representation
 of effects. There is another similar function for C modules.
 
 @param callee is the called module
 @param real_args is the list of actual arguments
 @param l_sum_eff is the list of summary effects for function func
 @return a list of effects in the caller name space

 */
list generic_fortran_effects_backward_translation(
				 entity callee,
				 list /* of expression */ real_args,
				 list /* of effect */ l_sum_eff,
				 transformer context)
{
  list le;
  le = (*fortran_effects_backward_translation_op)(callee, real_args, l_sum_eff, 
						  context);
  return le;
  
}

/************************************************************ C */

/**
 
 @param real_arg the real argument expression
 @param act is a tag to choose the action of the main effect :
        'r' for read, 'w' for write, and 'x' for read and write.
 @return a list of effects corresponding to the possible effects
         the function could do on the actual argument

 */
list c_actual_argument_to_may_summary_effects(expression real_arg, tag act)
{
  list l_res = NIL, l_tmp;
  effect real_arg_eff = effect_undefined;

  type real_arg_t = expression_to_type(real_arg);
  int real_arg_t_d = effect_type_depth(real_arg_t);
  transformer context = effects_private_current_context_head();
	    

  pips_debug(6,"actual argument %s, with type %s, and type depth %d\n", 
	     words_to_string(words_expression(real_arg)),
	     type_to_string(real_arg_t), real_arg_t_d);
  
  if (real_arg_t_d == 0)
    {
      pips_debug(6, "actual argument is a constant expression -> NIL\n");
    }
  else
    {
      syntax s = expression_syntax(real_arg);
      
      switch(syntax_tag(s))
	{
	case is_syntax_call:
	  /*
	    just a special case for :
	    - the assignment 
	    - and the ADDRESS_OF operator to avoid
            losing to musch information because we don't know how to 
            represent &p access path in the general case.
	  */
	  {
	    call real_call = syntax_call(s);
	    entity real_op = call_function(real_call);
	    list args = call_arguments(real_call);
	    	    
	    if (ENTITY_ASSIGN_P(real_op))
	      {
		pips_debug(5, "assignment case \n");
		l_res  = c_actual_argument_to_may_summary_effects
		  (EXPRESSION(CAR(CDR(args))), act);
		break;
	      }
	    else if(ENTITY_ADDRESS_OF_P(real_op)) 
	      {
		expression arg1 = EXPRESSION(CAR(args));
		
		pips_debug(5, "address_of case \n");
		l_tmp = generic_proper_effects_of_complex_address_expression
		  (arg1, &real_arg_eff, true);
		effects_free(l_tmp);
		
		if (anywhere_effect_p(real_arg_eff))
		  {		    
		    pips_debug(6, "anywhere effects \n");
		    l_res = gen_nconc
		      (l_res,
		       effect_to_effects_with_given_tag(real_arg_eff, act));
		  }
		else
		  {
		    type eff_type =  expression_to_type(arg1);

		    if(!ENDP(reference_indices(effect_any_reference(real_arg_eff))))
		      {					
			/* The operand of & is subcripted */
			/* the effect last index must be changed to '*' if it's
                           not already the case 
			*/
			reference eff_ref;
			expression last_eff_ind;
			expression n_exp;
			
			eff_ref = effect_any_reference(real_arg_eff);
			last_eff_ind = 
			  EXPRESSION(CAR(gen_last(reference_indices(eff_ref))));
		  
			if(!unbounded_expression_p(last_eff_ind))
			  {		
			    n_exp = make_unbounded_expression();
			    (*effect_change_ith_dimension_expression_func)
			      (real_arg_eff, n_exp, 
			       gen_length(reference_indices(eff_ref)));
			    free_expression(n_exp);

			  }
		      }
		    
		    l_res = gen_nconc
		      (l_res,
		       effect_to_effects_with_given_tag(real_arg_eff, 
							act));
		    
		    /* add effects on accessible paths */
		    
		    l_res = gen_nconc
		      (l_res,
		       generic_effect_generate_all_accessible_paths_effects
		       (real_arg_eff, eff_type, act));
		  }
		break;
	      }
	  }  
	case is_syntax_reference:
	case is_syntax_subscript:
	  {
	    pips_debug(5, "general call, reference or subscript case \n");
	    l_tmp = generic_proper_effects_of_complex_address_expression
		  (real_arg, &real_arg_eff, true);
	    effects_free(l_tmp);
		
	    if (anywhere_effect_p(real_arg_eff))
	      {		    
		pips_debug(6, "anywhere effects \n");
		l_res = gen_nconc
		  (l_res,
		   effect_to_effects_with_given_tag(real_arg_eff, 
							  act));
	      }
	    else
	      {
		l_res = gen_nconc
		  (l_res,
		   generic_effect_generate_all_accessible_paths_effects
		   (real_arg_eff, real_arg_t, act));	
	      }
	  
	  }
	  break;	  
	case is_syntax_cast: 
	  {	    
	    l_res = c_actual_argument_to_may_summary_effects
	      (cast_expression(syntax_cast(s)), act);
	  }
	  break;
	case is_syntax_sizeofexpression:
	  {
	    /* generate no effects : this case should never appear because
	     * of the test if (real_arg_t_d == 0) 
	     */	       
	  }
	  break;	    
	case is_syntax_va_arg: 
	  {
	    list al = syntax_va_arg(s);
	    sizeofexpression ae = SIZEOFEXPRESSION(CAR(al));
	    l_res = c_actual_argument_to_may_summary_effects
	      (sizeofexpression_expression(ae), act);
	    break;
	  }
	default:
	  pips_internal_error("case not handled\n");
	}
      
    } /* else du if (real_arg_t_d == 0) */
    
  (*effects_precondition_composition_op)(l_res, context);
 
  ifdebug(6)
    {
      pips_debug(6, "end, resulting effects are :\n");
      (*effects_prettyprint_func)(l_res);
    }
  return(l_res);
}




/**
 This function translates the list of effects l_sum_eff summarizing
 the effects of module callee from its name space to the name space of
 the caller, that is to say the current module being analyzed.  It is
 generic, which means that it does not depend on the representation of
 effects. There is another similar function for fortran modules.

 @param callee is the called module
 @param real_args is the list of actual arguments
 @param l_sum_eff is the list of summary effects for function func
 @param the current precondition if available
 @return a list of effects in the caller name space

*/
list generic_c_effects_backward_translation(entity callee,
					    list /* of expression */ real_args,
					    list /* of effect */ l_sum_eff,
					    transformer context)
{
  list l_begin = gen_copy_seq(l_sum_eff); /* effects are not copied */
  list l_prec = NIL, l_current;
  list l_eff = NIL; /* proper effect list to be returned */
  list ra;
  bool param_varargs_p = false;
  type callee_ut = ultimate_type(entity_type(callee));
  list formal_args = functional_parameters(type_functional(callee_ut));
  int arg_num;

  ifdebug(2)
    {
      pips_debug(2, "begin for function %s\n", entity_local_name(callee));
      pips_debug(2, "with actual arguments :\n");
      print_expressions(real_args);
      pips_debug(2, "and effects :\n");
      (*effects_prettyprint_func)(l_sum_eff);
    }

  (*effects_translation_init_func)(callee, real_args);

  /* first, take care of global effects */

  l_current = l_begin;
  l_prec = NIL;
  while(!ENDP(l_current))
    {
      effect eff = EFFECT(CAR(l_current));
      reference r = effect_any_reference(eff);
      entity v = reference_variable(r);

      if(!formal_parameter_p(v))
	{
	  /* This effect must be a global effect. It does not require
	     translation in C. However, it may not be in the scope of
	     the caller. */
	  eff = (*effect_dup_func)(eff);
	  (*effect_descriptor_interprocedural_translation_op)(eff);
	  /* Memory leak ? I don't understand the second dup */ 
	  l_eff = gen_nconc(l_eff,CONS(EFFECT, (*effect_dup_func)(eff), NIL));

	  /* remove the current element from the list */
	  if (l_begin == l_current)
	    {
	      l_current = CDR(l_current);
	      CDR(l_begin) = NIL;
	      gen_free_list(l_begin);
	      l_begin = l_current;

	    }
	  else
	    {
	      CDR(l_prec) = CDR(l_current);
	      CDR(l_current) = NIL;
	      gen_free_list(l_current);
	      l_current = CDR(l_prec);
	    }
	}
      else
	{
	  l_prec = l_current;
	  l_current = CDR(l_current);
	}

    } /* while */

  ifdebug(5)
    {
      pips_debug(5, "translated global effects :\n");
      (*effects_prettyprint_func)(l_eff);
      pips_debug(5, "remaining effects :\n");
      (*effects_prettyprint_func)(l_begin);
    }

  /* then, handle effects on formal parameters */

  for (ra = real_args, arg_num = 1; !ENDP(ra); ra = CDR(ra), arg_num++)
    {
      expression real_arg = EXPRESSION(CAR(ra));
      parameter formal_arg;
      type te;

      pips_debug(5, "current real arg : %s\n",
		 words_to_string(words_expression(real_arg)));

      if (!param_varargs_p)
	{
	  formal_arg = PARAMETER(CAR(formal_args));
	  te = parameter_type(formal_arg);
	  pips_debug(8, "parameter type : %s\n", type_to_string(te));
	  param_varargs_p = param_varargs_p || type_varargs_p(te);
	}

      if (param_varargs_p)
	{
	  pips_debug(5, "vararg case \n");
	  l_eff = gen_nconc(l_eff,
			    c_actual_argument_to_may_summary_effects(real_arg,
								     'x'));
	}
      else
	{
	  list l_eff_on_current_formal = NIL;


	  pips_debug(5, "corresponding formal argument :%s\n",
		     entity_name(dummy_identifier(parameter_dummy(formal_arg)))
		     );
	  /* first build the list of effects on the current formal argument */
	  l_current = l_begin;
	  l_prec = NIL;
	  while(!ENDP(l_current))
	    {
	      effect eff = EFFECT(CAR(l_current));
	      reference eff_ref = effect_any_reference(eff);
	      entity eff_ent = reference_variable(eff_ref);

	      if (ith_parameter_p(callee, eff_ent, arg_num))
		{

		  /* Whatever the real_arg may be if there is an effect on 
		     the sole value of the formal arg, it generates no effect 
		     on the caller side.
		  */
		  if (ENDP(reference_indices(eff_ref)))
		    {
		      pips_debug(5, "effect on the value of the formal parameter -> skipped\n");
		    }
		  else
		    {
		      l_eff_on_current_formal = gen_nconc
			(l_eff_on_current_formal, CONS(EFFECT,eff, NIL));
		    }
		  /*   c_summary_effect_to_proper_effects(eff, real_arg));*/
		  /* remove the current element from the list */
		  if (l_begin == l_current)
		    {
		      l_current = CDR(l_current);
		      CDR(l_begin) = NIL;
		      gen_free_list(l_begin);
		      l_begin = l_current;

		    }
		  else
		    {
		      CDR(l_prec) = CDR(l_current);
		      CDR(l_current) = NIL;
		      gen_free_list(l_current);
		      l_current = CDR(l_prec);
		    }

		}
	      else
		{
		  l_prec = l_current;
		  l_current = CDR(l_current);
		}
	    } /* while */

	  ifdebug(5)
	    {
	      pips_debug(5, "effects on current formal argument:\n");
	      (*effects_prettyprint_func)(l_eff_on_current_formal);
	    }
	  /* then translate them */
	  l_eff = gen_nconc
	    (l_eff,
	     (*c_effects_on_formal_parameter_backward_translation_func)
	     (l_eff_on_current_formal, real_arg, context));

	  POP(formal_args);
	} /* else */

      /* add the proper effects on the real arg evaluation */
      l_eff = gen_nconc(l_eff, generic_proper_effects_of_expression(real_arg));
    } /* for */

  (*effects_translation_end_func)();

  ifdebug(5)
    {
      pips_debug(5, "resulting effects :\n");
      (*effects_prettyprint_func)(l_eff);
    }

  return (l_eff);

}

/************************************************************ INTERFACE */

list /* of effect */
generic_effects_backward_translation(
				     entity callee,
				     list /* of expression */ real_args,
				     list /* of effect */  l_sum_eff,
				     transformer context)
{
  list el = list_undefined;


  if(parameter_passing_by_reference_p(callee))
    el = generic_fortran_effects_backward_translation(callee, real_args,
						      l_sum_eff, context);
  else
    el = generic_c_effects_backward_translation(callee, real_args,
						l_sum_eff, context);


  return el;
}


