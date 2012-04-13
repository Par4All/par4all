/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* package simple effects :  Be'atrice Creusillet 5/97
 *
 * File: interprocedural.c
 * ~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains functions for the interprocedural computation of simple
 * effects.
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"

#include "misc.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"

#include "effects-generic.h"
#include "effects-simple.h"


/** @brief translates a simple memory access path reference from given indices
           using an address_of memory access path reference

    This function is used when we want to translate a cell or an effect on a[i][j][k] as input_ref,
    knowing that a[i] = &address_of_ref. In this case nb_common_indices is 1 (which corresponds to [i])

    @param input_ref is the input simple cell reference
    @param input_desc is here for compatibility with the corresponding convex cells function.

    @param address_of_ref is the simple cell reference giving the output base memory access path.
    @param address_of_desc is here for compatibility with the corresponding convex cells function.

    @param nb_common_indices is the number of indices of input_ref which must be skipped
    @param output_ref is a pointer on the resulting reference
    @param output_desc is here for compatibility with the corresponding convex cells function.
    @param exact_p is a pointer on a bool which is set to true if the translation is exact, false otherwise.

 */
void simple_cell_reference_with_address_of_cell_reference_translation
(reference input_ref, descriptor __attribute__ ((unused)) input_desc,
 reference address_of_ref, descriptor __attribute__ ((unused)) address_of_desc,
 int nb_common_indices,
 reference *output_ref, descriptor __attribute__ ((unused)) * output_desc,
 bool *exact_p)
{

  pips_debug(1, "input_ref: %s\n",words_to_string(words_reference(input_ref, NIL)));
  pips_debug(1, "address_of_ref: %s\n",words_to_string(words_reference(address_of_ref, NIL)));
  pips_debug(1, "nb_common_indices: %d \n", nb_common_indices);

  /* assume exactness */
  *exact_p = true;

  /* */
  *output_ref = copy_reference(address_of_ref);
  list output_indices = gen_last(reference_indices(*output_ref));
  list input_remaining_indices = reference_indices(input_ref);
  for(int i = 0; i<nb_common_indices; i++, POP(input_remaining_indices));

  /* special case for the first remaining index: we must add it to the last index of build_ref */
  if (!ENDP(output_indices))
    {
      expression last_output_indices_exp = EXPRESSION(CAR(output_indices));
      expression first_input_remaining_exp = EXPRESSION(CAR(input_remaining_indices));
      expression new_exp = expression_undefined;
      /* adapted from the address_of case of c_simple_effects_on_formal_parameter_backward_translation
	 this should maybe be put in another function
      */
      if(!unbounded_expression_p(last_output_indices_exp))
	{
	  if (expression_reference_p(last_output_indices_exp) &&
	      entity_field_p(expression_variable(last_output_indices_exp)))
	    {
	      if (!expression_equal_integer_p(first_input_remaining_exp, 0))
		{
		  pips_user_warning("potential memory overflow due to effect -> returning anywhere\n");
		  free_reference(*output_ref);
		  *output_ref = make_reference(entity_all_locations(), NIL);
		  *exact_p = false;
		}
	      else
		new_exp = last_output_indices_exp;
	    }

	  else if(!unbounded_expression_p(first_input_remaining_exp))
	    {
	      value v;
	      intptr_t i_last_output_indices_exp;
	      intptr_t i_first_input_remaining_exp;

	      bool b_i_last_output_indices_exp = expression_integer_value(last_output_indices_exp, &i_last_output_indices_exp);
	      bool b_i_first_input_remaining_exp = expression_integer_value(first_input_remaining_exp, &i_first_input_remaining_exp);

	      if (b_i_last_output_indices_exp && i_last_output_indices_exp == 0)
		new_exp = copy_expression(first_input_remaining_exp);
	      else if (b_i_first_input_remaining_exp && i_first_input_remaining_exp == 0)
		new_exp = copy_expression(last_output_indices_exp);
	      else
		{
		  new_exp = MakeBinaryCall
		    (entity_intrinsic(PLUS_OPERATOR_NAME),
		     copy_expression(last_output_indices_exp), copy_expression(first_input_remaining_exp));
		  /* Then we must try to evaluate the expression */
		  v = EvalExpression(new_exp);
		  if (! value_undefined_p(v) &&
		      value_constant_p(v))
		    {
		      constant vc = value_constant(v);
		      if (constant_int_p(vc))
			{
			  free_expression(new_exp);
			  new_exp = int_to_expression(constant_int(vc));
			}
		    }
		}
	    }
	  else
	    {
	      new_exp = make_unbounded_expression();
	      *exact_p = false;
	    }
	  if (! entity_all_locations_p(reference_variable(*output_ref)))
	    {
	      CAR(gen_last(reference_indices(*output_ref))).p
		= (void *) new_exp;
	    }
	}
      else
	{
	  *exact_p = false;
	}
    }
  else /* ENDP(output_indices) */
    {
      /* address_of_ref is a scalar: the first remaning index must be equal to 0 */
      expression first_input_remaining_exp = EXPRESSION(CAR(input_remaining_indices));
      if (!expression_equal_integer_p(first_input_remaining_exp, 0))
	{
	  pips_user_warning("potential memory overflow due to effect -> returning anywhere\n");
	  free_reference(*output_ref);
	  *output_ref = make_reference(entity_all_locations(), NIL);
	  *exact_p = false;
	}
    }

  if (! entity_all_locations_p(reference_variable(*output_ref)))
    {
      FOREACH(EXPRESSION, input_ind, CDR(input_remaining_indices))
	{
	  reference_indices(*output_ref) = gen_nconc(reference_indices(*output_ref),
						     CONS(EXPRESSION,
							  copy_expression(input_ind),
							  NIL));
	}
    }
  pips_debug(8, "output reference %s\n",
	     words_to_string(words_reference(*output_ref, NIL)));
  return;
}

/** @brief translates a simple memory access path reference from given indices
           using an value_of memory access path reference

    This function is used when we want to translate a cell or an effect on a[i][j][k] as input_ref,
    knowing that a[i] = value_of_ref. In this case nb_common_indices is 1 (which corresponds to [i])

    @param input_ref is the input simple cell reference
    @param input_desc is here for compatibility with the corresponding convex cells function.

    @param value_of_ref is the simple cell reference giving the output base memory access path.
    @param value_of_desc is here for compatibility with the corresponding convex cells function.

    @param nb_common_indices is the number of indices of input_ref which must be skipped
    @param output_ref is a pointer on the resulting reference
    @param output_desc is here for compatibility with the corresponding convex cells function.
    @param exact_p is a pointer on a bool which is set to true if the translation is exact, false otherwise.

 */
void simple_cell_reference_with_value_of_cell_reference_translation
(reference input_ref, descriptor __attribute__ ((unused)) input_desc,
 reference value_of_ref, descriptor __attribute__ ((unused)) value_of_desc,
 int nb_common_indices,
 reference *output_ref, descriptor __attribute__ ((unused)) * output_desc,
 bool *exact_p)
{
  /* assume exactness */
  *exact_p = true;

  /* we do not handle yet the cases where the type of value_of_ref does not match
     the type of a[i]. I need a special function to test if types are compatible,
     because type_equal_p is much too strict.
     moreover the signature of the function may not be adapted in case of the reshaping of a array
     of structs into an array of char for instance.
  */
  list input_inds = reference_indices(input_ref);
  *output_ref = copy_reference(value_of_ref);

  /* we add the indices of the input reference past the nb_common_indices
     (they have already be poped out) to the copy of the value_of reference */

  for(int i = 0; i<nb_common_indices; i++, POP(input_inds));
  FOREACH(EXPRESSION, input_ind, input_inds)
    {
      reference_indices(*output_ref) = gen_nconc(reference_indices(*output_ref),
						 CONS(EXPRESSION,
						      copy_expression(input_ind),
						      NIL));
    }

}

void simple_cell_with_address_of_cell_translation
(cell input_cell, descriptor input_desc,
 cell address_of_cell, descriptor address_of_desc,
 int nb_common_indices,
 cell *output_cell, descriptor * output_desc,
 bool *exact_p)
{
  reference input_ref = cell_any_reference(input_cell);
  reference address_of_ref = cell_any_reference(address_of_cell);
  reference output_ref;
  simple_cell_reference_with_address_of_cell_reference_translation(input_ref, input_desc,
								   address_of_ref, address_of_desc,
								   nb_common_indices, &output_ref,
								   output_desc,
								   exact_p);

  *output_cell = make_cell_reference(output_ref);
}

void simple_cell_with_value_of_cell_translation
(cell input_cell, descriptor input_desc,
 cell value_of_cell, descriptor  value_of_desc,
 int nb_common_indices,
 cell *output_cell, descriptor * output_desc,
 bool *exact_p)
{
  reference input_ref = cell_any_reference(input_cell);
  reference value_of_ref = cell_any_reference(value_of_cell);
  reference output_ref;
  simple_cell_reference_with_value_of_cell_reference_translation(input_ref, input_desc,
								 value_of_ref, value_of_desc,
								 nb_common_indices, &output_ref,
								 output_desc,
								 exact_p);

  *output_cell = make_cell_reference(output_ref);
}
