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

#include "effects-generic.h"
#include "effects-simple.h"




/*********************************************************************************/
/* REFERENCE EFFECTS                                                             */
/*********************************************************************************/

effect
reference_effect_dup(effect eff)
{
    return (copy_effect(eff));
}

void
reference_effect_free(effect eff)
{
    free_effect(eff);
}

/* FI: this function is not that simple as it depends on the type of
   the referenced variable and maybe of the action.

   If s is a structure with a field a, it is called in the very same
   way when a field is modified:

   s.a = 1;

   and when the whole structure is assigned:

   s = s2;
 */
effect reference_to_simple_effect(reference ref, action ac)
{
/* It should be this one, but it does not work. Maybe there is a clash with
   old uses of make_simple_effects. Or persistancy is not properly handled?
   - bc.
*/
  /* FI: this should be revisited now that I have cleaned up a lot of
     the effects library to handled the two kinds of cells. */
  /* cell cell_ref = make_cell(is_cell_reference, copy_reference(ref)); */

  effect eff = effect_undefined;
  list ind = reference_indices(ref);
  type t = entity_type(reference_variable(ref));
  type ut = ultimate_type(t);

  pips_debug(8, "Begins for reference: \"%s\"\n", 
	     words_to_string(words_reference(ref)));

  if(type_variable_p(ut)) {
    variable utv = type_variable(ut);
    list utd = variable_dimensions(utv);
    //basic utb = variable_basic(utv);
    bool is_array_p = !ENDP(utd);
    variable tv = type_variable(t);
    list td = variable_dimensions(tv);

    /* The dimensions can be hidden in the typedef or before the typedef */
    if(type_variable_p(t)) 
      {
	is_array_p = is_array_p || (!ENDP(td));
      }

    if(is_array_p) 
      {
	if((gen_length(ind) == type_depth(t))) 
	  {
	    /* The dimensionalities of the index and type are the same: */
	    /* cell cell_ref = make_cell_reference(copy_reference(ref)); */
	    cell cell_ref = make_cell_preference(make_preference(ref));
	    approximation ap = make_approximation_must();
	    eff = make_effect(cell_ref, ac, ap, make_descriptor_none());
	  }
	else if((gen_length(ind) < type_depth(t))) 
	  {
	    /* This may happen with an array of structures */
	    int d = (t==ut) ? gen_length(td) : gen_length(td)+gen_length(utd);

	    /* RK: In the following there is 3 tests with the same code in
	       it. Sounds it needs some code hoisting... ::-) Or some
	       debug. Is it really the same for the pointer type? */
	    /* FI: I'm now lost here... */
	    if(gen_length(ind)==d) 
	      {
		reference n_ref = copy_reference(ref);
		cell cell_ref = make_cell_reference(n_ref);
		approximation ap = make_approximation_must();
		eff = make_effect(cell_ref, ac, ap, make_descriptor_none());
	      }
	    /* FI: I'm not sure this code is of any positive use */
	    else if(pointer_type_p(ut)) 
	      {
		reference n_ref = copy_reference(ref);
		cell cell_ref = make_cell_reference(n_ref);
		approximation ap = make_approximation_must();
		eff = make_effect(cell_ref, ac, ap, make_descriptor_none());
	    }
	    else 
	      {
		/* FI: Which case are we in?*/
		/* RK: we may access an array with a lower dimensionality, that
		   is a slice. Useful for example in Fortran code such as:
		   PRINT *, an_array
		   Cf @validation Transformations/unroll2.f
		   
		   With this solution it is not the an_array(*) that is marked
		   as must-read but only an_array. But as stated at the main
		   comment, the effect depends on the action.
		   
		   When calling a function with blah(an_array), an_array is
		   marked as must-read, for its address.
		   Cf. @validation Effects/call04.c
		   Even more interesting:
		   @validation Effects/call05.c
		*/
		reference n_ref = copy_reference(ref);
		cell cell_ref = make_cell_reference(n_ref);
		approximation ap = make_approximation_must();
		eff = make_effect(cell_ref, ac, ap, make_descriptor_none());
	      }
	  }
	else 
	  {
	    /* The memory is not accessed because the array name is a
	       constant. It can be used directly or the the base for some
	       address computation. */
	    ;
	  }
      }
    else 
      {
	/* It is not an array. Addressing is encoded in a different way
	   for structures, unions and pointers. */
	cell cell_ref = make_cell_preference(make_preference(ref));
	approximation ap = make_approximation_must();
	eff = make_effect(cell_ref, ac, ap, make_descriptor_none());
      }
  }
  else if(type_functional_p(ut)) {
    /* Must be a function used to initialize a pointer to a function */
    ;
  }
  else {
    pips_internal_error("Unexpected type\n");
  }
  
  pips_debug(8, "end\n");
  
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

  expression deref_exp;
  cell eff_c = effect_cell(eff);
  reference ref;
  normalized nexp = NORMALIZE_EXPRESSION(exp);
  
  ifdebug(8)
    {
      pips_debug(8, "begin with effect :\n");
      print_effect(eff);
    }
  
  if (cell_preference_p(eff_c))
    {
      /* it's a preference : we should not modify it */
      pips_debug(8, "It's a preference\n");
      ref = copy_reference(preference_reference(cell_preference(eff_c)));
      preference_reference(cell_preference(eff_c)) = ref;
    }
  else
    {
      /* it's a reference : let'us modify it */
      ref = cell_reference(eff_c);
    }
  
  if (normalized_linear_p(nexp))
    {
      reference_indices(ref) = gen_nconc(reference_indices(ref),
					 CONS(EXPRESSION, 
					      copy_expression(exp), 
					      NIL));
    }
  else
    {      
      reference_indices(ref) = gen_nconc(reference_indices(ref),
					 CONS(EXPRESSION, 
					      make_unbounded_expression(), 
					      NIL));
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



/*********************************************************************************/
/* SIMPLE EFFECTS                                                                */
/*********************************************************************************/

/* A persistant reference wss forced. */
effect
simple_effect_dup(effect eff)
{
  effect new_eff = effect_undefined;

  new_eff = copy_effect(eff);

  if(cell_preference_p(effect_cell(new_eff))) {
    /* FI: memory leak? we allocate something and put it behind a persistent pointer */
    effect_reference(new_eff) = reference_dup(effect_reference(new_eff));
  }

  ifdebug(8) pips_assert("the new effect is consistent", effect_consistent_p(new_eff));

  return(new_eff);
}

/* use free_effect() instead? persistent reference assumed */
void
simple_effect_free(effect eff)
{
  if(cell_preference_p(effect_cell(eff))) {
    free_reference(effect_reference(eff));
    effect_reference(eff) = reference_undefined;
  }
  else {
    free_reference(effect_any_reference(eff));
    cell_reference(effect_cell(eff)) = reference_undefined;
  }
  free_effect(eff);
}

/* In fact, reference to persistant reference, preference */
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
      pips_debug(8, "Begin\n");
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
	reference r1 = effect_any_reference(e1);
	reference r2 = effect_any_reference(e2);
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

