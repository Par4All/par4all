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
/* package simple effects :  Be'atrice Creusillet 6/97
 *
 * File: unary_operators.c
 * ~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the intanciation of the generic functions necessary 
 * for the computation of all types of simple effects.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "properties.h"
#include "preprocessor.h"

#include "effects-generic.h"
#include "effects-simple.h"




/**
 @param ref is a reference
 @param ac is an action
 @return a simple effect representing a memory access.

 This function trusts the reference ref : no check is done to know
 if the effect is legal (for instance a read effect on a partially
 subscripted array). This has to be done at a higher level. (BC)

 */
effect reference_to_simple_effect(reference ref, action ac)
{
  entity ent = reference_variable(ref);
  effect eff = effect_undefined;

  pips_debug(8, "Begins for reference: \"%s\"\n", 
	     words_to_string(words_reference(ref)));
  
  if (same_string_p(entity_name(ent), ALL_MEMORY_ENTITY_NAME))
    {
      /* anywhere effect */
      eff = make_effect(make_cell_reference(ref),
			ac,
			make_approximation_may(),
			make_descriptor_none());
    }
  else
    {    
      
      list ind = reference_indices(ref);
      type t = entity_type(reference_variable(ref));
      type ut = basic_concrete_type(t);
      
      
      if(type_variable_p(ut)) 
	{
	  variable utv = type_variable(ut);
	  list utd = variable_dimensions(utv);
	  //basic utb = variable_basic(utv);
	  bool is_array_p = !ENDP(utd);
	  variable tv = type_variable(t);
	  list td = variable_dimensions(tv);
	  
	  /* The dimensions can be hidden in the typedef or 
	     before the typedef */
	  if(type_variable_p(t)) 
	    {
	      is_array_p = is_array_p || (!ENDP(td));
	    }
	  
	  if (is_array_p)
	    {
	      if(gen_length(ind) == type_depth(t))
		{
		  /* The dimensionalities of the index and type are the same: */
		  /* cell cell_ref = make_cell_reference(copy_reference(ref)); */
		  cell cell_ref = make_cell_preference(make_preference(ref));
		  approximation ap = make_approximation_must();
		  eff = make_effect(cell_ref, ac, ap, make_descriptor_none());
		}
	      else
		{

		  /* if we are in C we trust the reference */
		  if (c_module_p(get_current_module_entity()))
		    {
		      reference n_ref = copy_reference(ref);
		      cell cell_ref = make_cell_reference(n_ref);
		      approximation ap = make_approximation_must();
		      eff = make_effect(cell_ref, ac, ap, make_descriptor_none());
		    }
		  else
		    {
		      /* we are in Fortran. A reference to TAB with no 
			 index is a reference to the whole array 
		      */
		      /* there is a memory leak here if ac has been allocated 
		       for the sole purpose of this function call 
		      */ 

		      pips_assert("invalid number of reference indices \n",
				  gen_length(variable_dimensions(tv)) > 
				  gen_length(ind));
	  
		      pips_debug(7, "less ref indices than number of dimensions\n");
		      /* generate effects on whole (sub-)array */
	  	  
		      eff = make_effect
			(make_cell_reference(copy_reference(ref)),
			 ac, make_approximation_must(), make_descriptor_none());
		      
		      FOREACH(DIMENSION, c_t_dim, 
			      gen_nthcdr((int) gen_length(ind),
					 variable_dimensions(tv)))
			{		      
			  simple_effect_add_expression_dimension
			    (eff, make_unbounded_expression());		      
			} /* FOREACH */
		      
		    }
		}
	    }
	  else
	    {
	      /* It is a scalar : keep the actual reference */
	      cell cell_ref = make_cell_preference(make_preference(ref));
	      approximation ap = make_approximation_must();
	      eff = make_effect(cell_ref, ac, ap, make_descriptor_none());
	    }
	}
      else
	{
	  reference n_ref = copy_reference(ref);
	  cell cell_ref = make_cell_reference(n_ref);
	  approximation ap = make_approximation_must();
	  eff = make_effect(cell_ref, ac, ap, make_descriptor_none()); 
	}
      free_type(ut);

    }
  
  ifdebug(8)
    {
      pips_debug(8, "end with effect\n");
      print_effect(eff);
    }
  
  return eff;
}


 

/* void simple_effect_add_expression_dimension(effect eff, expression exp)
 * input    : a simple effect and an expression
 * output   : nothing
 * modifies : the effect eff, and normalizes the expression
 * comment  : adds a last dimension [exp] to the effect if the expression 
 *            is normlizable. If not adds a last dimension [*], and changes
 *            the approximation into may.
 */
void simple_effect_add_expression_dimension(effect eff, expression exp)
{

  cell eff_c = effect_cell(eff);
  reference ref;
  
  ifdebug(8)
    {
      pips_debug(8, "begin with effect :\n");
      print_effect(eff);
    }
  
  if (cell_preference_p(eff_c))
    {
      /* it's a preference : we change for a reference cell */
      pips_debug(8, "It's a preference\n");
      ref = copy_reference(preference_reference(cell_preference(eff_c)));
      free_cell(eff_c);
      effect_cell(eff) = make_cell_reference(ref);
    }
  else
    {
      /* it's a reference : let'us modify it */
      ref = cell_reference(eff_c);
    }

  reference_indices(ref) = gen_nconc(reference_indices(ref),
				     CONS(EXPRESSION, 
					  copy_expression(exp), 
					  NIL));
  
  
  if(unbounded_expression_p(exp))
    {
      effect_approximation_tag(eff) = is_approximation_may;
    }
  ifdebug(8)
    {
      pips_debug(8, "end with effect :\n");
      print_effect(eff);
      pips_assert("the effect is not consistent", effect_consistent_p(eff));
    }
  
  return;
}


/**
 This function changes the ith index of the effect reference
 into the given expression exp if it is normalizable, and into and
 unbounded expression otherwise (in which case the effect approximation is
 set to may).

 @param eff is a simple effect
 @param exp is the new expression for the ith index
 @param i is the range of the index to change.

 */
void simple_effect_change_ith_dimension_expression(effect eff, expression exp,
					       int i)
{

  cell eff_c = effect_cell(eff);
  reference ref;
  normalized nexp = NORMALIZE_EXPRESSION(exp);
  list l_ind;
  
  ifdebug(8)
    {
      pips_debug(8, "begin with effect :\n");
      print_effect(eff);
    }
  
  if (cell_preference_p(eff_c))
    {
      /* it's a preference : we change for a reference cell */
      pips_debug(8, "It's a preference\n");
      ref = copy_reference(preference_reference(cell_preference(eff_c)));
      free_cell(eff_c);
      effect_cell(eff) = make_cell_reference(ref);
    }
  else
    {
      /* it's a reference : let'us modify it */
      ref = cell_reference(eff_c);
    }
  
  l_ind = gen_nthcdr(i-1,reference_indices(ref));
  pips_assert("ith index must exist",!ENDP(l_ind));
  
  free_expression(EXPRESSION(CAR(l_ind)));

  if (normalized_linear_p(nexp))
    {
      EXPRESSION_(CAR(l_ind)) =  copy_expression(exp);
    }
  else
    {      
      EXPRESSION_(CAR(l_ind)) =  make_unbounded_expression();
      effect_approximation_tag(eff) = is_approximation_may; 
    }  
  
  ifdebug(8)
    {
      pips_debug(8, "end with effect :\n");
      print_effect(eff);
    }
  
  return;
}



/*********************************************************************************/
/* SIMPLE EFFECTS                                                                */
/*********************************************************************************/

/**
   @param eff is a simple effect
   @return another effect, whose cell is a reference cell, which is a copy
           of the initial effect reference, even if it were a preference.
           Ther is no sharing between the initial effect and the returned one.
 */
effect
simple_effect_dup(effect eff)
{
  effect new_eff = effect_undefined;

  
  if(cell_preference_p(effect_cell(eff))) 
    {
      new_eff = make_effect(make_cell_reference(copy_reference(effect_any_reference(eff))),
			    copy_action(effect_action(eff)), 
			    copy_approximation(effect_approximation(eff)),
			    copy_descriptor(effect_descriptor(eff)));
    }
  else
    {
      new_eff = copy_effect(eff);
      
    }
  
  ifdebug(8) pips_assert("the new effect is consistent", effect_consistent_p(new_eff));
  
  return(new_eff);
}

/**
   @param ref is a program reference
   @param ac is the action of the returned effect
   @return an effect whose cell is a preference pointing
           to the original program reference.
 */
 effect
 reference_to_reference_effect(reference ref, action ac)
 {
   cell cell_ref = make_cell(is_cell_preference, make_preference(ref));
   approximation ap = make_approximation(is_approximation_must, UU);
   effect eff;
    
   eff = make_effect(cell_ref, ac, ap, make_descriptor(is_descriptor_none,UU));  
   return(eff);
 }


list simple_effects_union_over_range(list l_eff,
				     entity i,
				     range r,
				     descriptor d __attribute__ ((unused)))
{
  /* FI: effects in index and in in range must be taken into account. it
     would be easier to have the loop proper effects as argument instead
     of recomputing it. */
  if(FALSE) {
    list c_eff = list_undefined;
    reference ref = make_reference(i, NIL);
    cell c = make_cell_reference(ref);
    effect i_eff = make_effect(c, make_action_write(), 
			       make_approximation_must(), make_descriptor_none());

    list r_eff_l = proper_effects_of_range(r);
    list h_eff_l = CONS(EFFECT, i_eff, r_eff_l);
    list ch_eff = list_undefined;

    for(ch_eff=h_eff_l; !ENDP(ch_eff); POP(ch_eff)) {
      effect h_eff = EFFECT(CAR(ch_eff));

      for(c_eff = l_eff; !ENDP(c_eff); POP(c_eff)) {
	effect eff = EFFECT(CAR(c_eff));

	eff = effect_interference(eff, h_eff);

	EFFECT_(CAR(c_eff)) = eff;
      }
    }

    //gen_full_free_list(h_eff_l);
  }
 if (!get_bool_property("ONE_TRIP_DO"))
    {
      effects_to_may_effects(l_eff);
    }
  return l_eff;
}


/* FI: instead of simply getting rid of indices, I preserve constant
   indices for the semantics analysis. Instead of stripping the
   indices, they are replaced by unbounded expressions to keep the
   difference between p and p[*] when p is a pointer. 

   This is not as strong as store_independent_effect_p() which would
   require that the reference is not pointer dependent.

   No memory allocation, side effects on eff? Side effects on the
   reference too in spite of the persistant reference? No, it's not
   possible. It's easier to work on a copy of the reference when a
   "preference" is used.
*/
list 
effect_to_store_independent_sdfi_list(effect eff, bool force_may_p)
{
  cell c = effect_cell(eff);
  reference r = cell_preference_p(c)?
    copy_reference(effect_any_reference(eff))
    : effect_any_reference(eff);
  list ind = reference_indices(r);
  list cind = list_undefined;
  bool may_p = FALSE;

  for(cind = ind; !ENDP(cind); POP(cind)) {
    expression se = EXPRESSION(CAR(cind));

    if(!extended_integer_constant_expression_p(se)) {
      if(!unbounded_expression_p(se)) {
	expression nse = make_unbounded_expression();
	may_p = TRUE;
	free_expression(se);
	//CAR(cind).p = (void *) nse;
	EXPRESSION_(CAR(cind)) = nse;
      }
    }
  }

  /* FI: Why is MAY always forced? Because of the semantics of the function! */
  if(may_p || force_may_p)
    effect_approximation_tag(eff) = is_approximation_may;

  /* FI: if necessary, use the reference copy in the cell */
  if(cell_preference_p(c)) {
    free_preference(cell_preference(c));
    cell_tag(c) = is_cell_reference;
    cell_reference(c) = r;
  }

  ifdebug(1)
    pips_assert("eff is consistent", effect_consistent_p(eff));

  return(CONS(EFFECT,eff,NIL));
}

list 
effect_to_may_sdfi_list(effect eff)
{
  return effect_to_store_independent_sdfi_list(eff, TRUE);
}

/* FI: instead of simpy getting rid of indices, I preserve cosntant
   indices for the semantics analysis. */
list 
effect_to_sdfi_list(effect eff)
{
  return effect_to_store_independent_sdfi_list(eff, FALSE);
}

void
simple_effects_descriptor_normalize(list l_eff __attribute__ ((unused)))
{
  return;
}


/* 
 * It's not yet completely safe for C code when pointers are
 * modified.
*/
list simple_effects_composition_with_effect_transformer(list l_eff,
						   transformer trans
						   __attribute__((__unused__)))
{
  list l_res=NIL;
  
  ifdebug(8) 
    {
      pips_debug(8, "Begin\n");
      print_effects(l_eff);
    }

  FOREACH (EFFECT, eff, l_eff)
    {
      l_res = 
	gen_nconc(l_res, 
		  effect_to_store_independent_sdfi_list(eff, FALSE) 
		  );
    }
  
  ifdebug(8) 
    {
      pips_debug(8, "End\n");
      print_effects(l_res);
    }
  
  return(l_res);
}

/* This function does not do what it was designed for : 
   It should transform the effects l_eff corresponding to S2 with
   transformer T1 corresponding to S1. 
   But it uses effects from S2 instead of effects from S1.
   (see r_rw_effects_of_sequence)
   I keep it for future reuse after modification.
   BC.
*/
list old_effects_composition_with_effect_transformer(list l_eff,
						 transformer trans __attribute__((__unused__)))
{
  /* FI: used to be nop and wrong information is now preserved
     intraprocedurally with loops, maybe because I modified simple
     effects; since we do not have transformers, we use instead the
     effects themselves, which could be transformed into a
     transformer... 

     The effects are supposed to be ordered. A write effect must
     appears before another effect to require an update.
*/
  list l1 = list_undefined;
  list l2 = list_undefined;
  extern string words_to_string(list);

  ifdebug(8) {
    pips_debug(8, "Begin: %zd effects before composition:\n", gen_length(l_eff));
    MAP(EFFECT, eff, {
	print_effect(eff);
      },  l_eff);
  }

  for(l1= l_eff; !ENDP(l1); POP(l1)) {
    effect e1 = EFFECT(CAR(l1));
    for(l2 = CDR(l1); !ENDP(l2); POP(l2)) {
      effect e2 = EFFECT(CAR(l2));

      ifdebug(1) {
	pips_assert("Effect e1 is consitent", effect_consistent_p(e1));
	pips_assert("Effect e2 is consitent", effect_consistent_p(e2));
      }

      ifdebug(8) {
	(void) fprintf(stderr, "e1: \n");
	print_effect(e1);
	(void) fprintf(stderr, "e2: \n");
	print_effect(e2);
      }

      e2 = effect_interference(e2, e1);

      ifdebug(8) {
	(void) fprintf(stderr, "resulting effect e2: \n");
	print_effect(e2);
      }

      EFFECT_(CAR(l2)) = e2;
    }
  }

  ifdebug(8) {
    pips_debug(8, "End: %zd effects before composition:\n", gen_length(l_eff));
    (*effects_prettyprint_func)(l_eff);
  }

  /* FI: Not generic. */
  l_eff = proper_effects_combine(l_eff, FALSE);


  ifdebug(8) {
    pips_debug(8, "End: %zd effects after composition:\n", gen_length(l_eff));
    (*effects_prettyprint_func)(l_eff);
  }

  return l_eff;
}

