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

/* list c_actual_argument_to_may_summary_effects(expression real_arg)
 * input    : an expression corresponding to a function actual argument.
 * output   : a list of effects corresponding to the possible effects
 *            the function could do on the actual argument
 * modifies : nothing.
 * comment  :
 */
list c_actual_argument_to_may_summary_effects(expression real_arg)
{
  list l_res = NIL;

  list l_real_arg_eff = NIL;
  effect real_arg_eff;

  type real_arg_t = expression_to_type(real_arg);
  int real_arg_t_d = effect_type_depth(real_arg_t);

  pips_debug(6,"actual argument %s, with type %s, and type depth %d\n", 
	     words_to_string(words_expression(real_arg)),
	     type_to_string(real_arg_t), real_arg_t_d);
  
  if (real_arg_t_d == 0)
    {
      pips_debug(6, "actual argument is a constant expression -> NIL\n");
    }
  else
    {
      
      switch(type_tag(real_arg_t))
	{
	case is_type_functional :
	  {
	    pips_internal_error("functional type case not handeld yet\n");
	    break;
	  }
	case is_type_variable :
	case is_type_struct :
	case is_type_union :
	  {	    
	    list l_tmp = NIL;
	    transformer context = effects_private_current_context_head();

	    pips_debug(8, "variable, struct or union \n");
	    
 	    /* first we compute the effects on the argument as if it 
	       were a lhs */
	    /* The problem is that when the expression is an array
	       name there won't be any effect generated ! 
	    */
	    
	    l_tmp = generic_proper_effects_of_complex_address_expression
	      (real_arg, &real_arg_eff, true);
	    effects_free(l_tmp);
	    
	    l_real_arg_eff = CONS(EFFECT, real_arg_eff, l_real_arg_eff);
	    (*effects_precondition_composition_op)(l_real_arg_eff, context);
	    
	    ifdebug(6)
	      {
		pips_debug
		  (6, 
		   " effects on real_arg expression %s are :\n",
		   words_to_string(words_expression(real_arg)));
		(* effects_prettyprint_func)(l_real_arg_eff);
	      }
      
	    /* Then, we replace write effects on references with read and 
	     * write effects on the pointed variables if it makes sense.
	     * we skip read effects since they have already been computed 
	     * before. do not correspond to the actual argument */
	    FOREACH(EFFECT, eff, l_real_arg_eff)
		{
		  if (effect_write_p(eff))
		    {
		      /*BC :  this should be moved into another function, 
			which should be recursive to handle structs and unions 
		      */
		      reference ref = effect_any_reference(eff);
		      int n_ref_inds = gen_length(reference_indices(ref));
		      entity ent = reference_variable(ref);
		      type t = ultimate_type(entity_type(ent));
		      int d = effect_type_depth(t);

		      int n_inds; /* current number of indices */
		      type c_t; /* current type of the current effect */
		      bool finished; 

		      effect eff_read = effect_undefined;; 
		      effect eff_write = copy_effect(eff);

		      ifdebug(6)
			{
			  pips_debug(6, "considering effect : \n");
			  print_effect(eff);
			  pips_debug(6, " with entity effect type depth %d \n",d);
			}
		      pips_assert("The effect reference should be a partially subscripted array or a pointer\n", 
				  n_ref_inds < d);

		      c_t = t;
		      finished = false;
		      n_inds = 0;
		      while(!finished && n_inds<d)
			{
			  switch (type_tag(c_t))
			    {
			    case is_type_variable :
			      {
				variable v = type_variable(c_t);
				basic b = variable_basic(v);
				bool add_effect = false;
				
				pips_debug(8, "variable case, of dimension %d, n_inds = %d\n", 
					   (int) gen_length(variable_dimensions(v)), n_inds); 
				FOREACH(DIMENSION, c_t_dim, 
					variable_dimensions(v))
				  {
				    if(n_inds>=n_ref_inds)
				      {
					(*effect_add_expression_dimension_func)
					  (eff_write, make_unbounded_expression());
					add_effect = true;
				      }
				    n_inds++;
				  }
				  
				if(basic_pointer_p(b))
				  {
				    pips_debug(8, "pointer case, n_inds = %d\n", n_inds);
				    if(n_inds>=n_ref_inds)
				      {
					
					(*effect_add_expression_dimension_func)
					  (eff_write, make_unbounded_expression());
					add_effect = true;
					
				      }
				    n_inds++;
				    c_t = basic_pointer(b);
				  }
				
				if (add_effect)
				  {				   		
				    eff_read = copy_effect(eff_write);
				    effect_action_tag(eff_read) = is_action_read;				    
				    ifdebug(8)
				      {
					pips_debug(8, "adding read and write effects to l_res : \n");
					print_effect(eff_write);
					print_effect(eff_read);
				      }

				    l_res = gen_nconc(l_res, CONS(EFFECT, eff_write, NIL));
				    l_res = gen_nconc(l_res, CONS(EFFECT, eff_read, NIL));
				  }
				else
				  {
				    pips_debug(8, "no need to add the current effect : all dimensions are already represented.\n");
				  }
				
				finished = true;

				break;
			      }
			    default:
			      {
				finished = true;
				pips_internal_error("case not handled yet");
			      }
			    } /*switch */
			  
			} /*while*/
		    } /* end of if (effect_write_p)*/
		} /* FOREACH */
	    
	    /* The next statement was here to avoid memory leaks. 
	       But it aborts ! I don't know why, maybe due to preferences 
	       As it should maybe disappear with the change of effects, 
	       I leave it as it is, as I intend to build the interprocedural 
	       translation differently. (BC) */ 
	    /*effects_free(l_real_arg_eff);*/
	    break;
	  }
	default:
	  pips_internal_error("default type case : not handled \n"); 
	} /* du switch */
      
    } /* else du if (real_arg_t_d == 0) */
  return(l_res);
}




/**
 This function translates the list of effects l_sum_eff summarizing the 
 effects of module callee from its name space to the name space of the
 caller, that is to say the current module being analyzed.
 It is generic, which means that it does not depend on the representation
 of effects. There is another similar function for fortran modules.
 
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
      print_effects(l_sum_eff);
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
	  /* not generic : for regions we also have to translate the predicate 
	     BC */
	  l_eff = gen_nconc(l_eff, CONS(EFFECT, copy_effect(eff), NIL));
	  
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
			    c_actual_argument_to_may_summary_effects(real_arg));
	}  
      else
	{
	  list l_eff_on_current_formal = NIL;


	  pips_debug(5, "corresponding formal argument :%s",
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


