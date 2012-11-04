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

/* Assume asl and isl have the same number of elements. Build
 * subscript list osl by adding when possible the subscripts in asl
 * and isl. See if the addition is exact.
 *
 * symbolic expressions are preserved at this stage, to be deleted
 * later for simple effects because they will be replaced by
 * *. However, it might be useful when descriptors are used.
 */
bool add_points_to_subscript_lists(list * posl, list asl, list isl)
{
  list casl, cisl;
  bool exact_p = true;
  for(casl=asl, cisl=isl; !ENDP(casl) && !ENDP(cisl); POP(casl), POP(cisl)) {
    expression a = EXPRESSION(CAR(casl));
    expression i = EXPRESSION(CAR(cisl));
    if(unbounded_expression_p(a) || unbounded_expression_p(i)) {
      expression u = make_unbounded_expression();
      *posl = CONS(EXPRESSION, u, *posl);
      exact_p = false;
    }
    else if(expression_field_p(a)) {
      if(expression_field_p(i)) {
	pips_assert("Both expressions are identical", expression_equal_p(a,i));
	expression f = copy_expression(a);
	*posl = CONS(EXPRESSION, f, *posl); // exactitude unchanged
      }
      else
	pips_internal_error("Unexpected case.\n");
    }
    else {
      intptr_t as, is;
      if(expression_integer_value(a,&as)) {
	if(expression_integer_value(i, &is)) {
	  expression ns = int_to_expression((int)as+is);
	  *posl = CONS(EXPRESSION, ns, *posl); // exactitude unchanged
	}
	else {
	  expression ns = expression_undefined;
	  if(as==0) {
	    ns = copy_expression(i);
	  }
	  else {
	    ns = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
					     copy_expression(a),
					     copy_expression(i));
	  }
	  *posl = CONS(EXPRESSION, ns, *posl); // exactitude unchanged
	}
      }
      else {
	expression ns = expression_undefined;
	if(expression_integer_value(i, &is) && is==0)
	  ns = copy_expression(a);
	else
	  ns = binary_intrinsic_expression(PLUS_OPERATOR_NAME,
					   copy_expression(a),
					   copy_expression(i));
	*posl = CONS(EXPRESSION, ns, *posl); // exactitude unchanged
      }
    }
  }
  *posl = gen_nreverse(*posl);
  return exact_p;
}

/* FI: Let's deal with simple case I think I know how to deal with. */
bool simple_cell_reference_with_address_of_cell_reference_translation_fi
(reference input_ref, descriptor __attribute__ ((unused)) input_desc,
 reference address_of_ref, descriptor __attribute__ ((unused)) address_of_desc,
 int nb_common_indices,
 reference *output_ref, descriptor __attribute__ ((unused)) * output_desc,
 bool *exact_p)
{
  bool ok_p = nb_common_indices==0;
  if(ok_p) {
    list isl = reference_indices(input_ref);
    list asl = reference_indices(address_of_ref);
    int nis = (int) gen_length(isl);
    int nas = (int) gen_length(asl);
    if(nis==nas) {
      bool i_to_be_freed;
      type it = points_to_reference_to_type(input_ref, &i_to_be_freed);
      type cit = compute_basic_concrete_type(it);
      if(i_to_be_freed) free_type(it);
      bool a_to_be_freed;
      type at = points_to_reference_to_type(address_of_ref, &a_to_be_freed);
      type cat = compute_basic_concrete_type(at);
      if(a_to_be_freed) free_type(at);
      if(type_equal_p(cit, cat)) {
	//*output_ref = copy_reference(address_of_ref);
	*output_ref = make_reference(reference_variable(address_of_ref), NIL);
	//list osl = reference_indices(*output_ref);
	list osl = NIL;
	// Apply an offset
	*exact_p = add_points_to_subscript_lists(&osl, asl, isl);
	reference_indices(*output_ref) = osl;
      }
      else
	ok_p = false;
    }
    else
      ok_p = false;
  }
  return ok_p;
}

/** @brief translates a simple memory access path reference from given indices
           using an address_of memory access path reference

    This function is used when we want to translate a cell or an effect on a[i][j][k] as input_ref,
    knowing that a[i] = &address_of_ref. In this case nb_common_indices is 1 (which corresponds to [i])

    @param input_ref is the input simple cell reference
    @param input_desc is here for compatibility with the corresponding convex cells function.

    @param address_of_ref is the simple cell reference giving the output base memory access path.
    @param address_of_desc is here for compatibility with the corresponding convex cells function.

    @param nb_common_indices is the number of indices of input_ref which must be skipped (why is it unique for all points-to arcs?)
    @param output_ref is a pointer on the resulting reference
    @param output_desc is here for compatibility with the corresponding convex cells function.
    @param exact_p is a pointer on a bool which is set to true if the translation is exact, false otherwise.

    FI->BC: more examples would be useful. For instance, let's have
    "p->i[1]" as points-to and hence "i[1]" as "address_of_ref". Let's
    have "p[0]" as input ref. The number of common indices is zero in
    the points-to arc. However, the two indices should be fused and
    not concatenated.

    FI->BC: another example due to Pointers/array15.c. Input_ref is
    b[0][3] and address_of_ref is _b_1[0][0]. nb_common_indices is
    0. b os a pointer to an array and the thing to do is to add the
    indices one by one, not to concatenate anything...

    FI->BC: this function is way too long. It must handle very
    different cases, e.g. arrays of pointers, struct as well as simple
    scalar pointer to scalar or arrays. It would be nice to have an
    example for each class of input.
 */
void simple_cell_reference_with_address_of_cell_reference_translation
(reference input_ref, descriptor __attribute__ ((unused)) input_desc,
 reference address_of_ref, descriptor __attribute__ ((unused)) address_of_desc,
 int nb_common_indices,
 reference *output_ref, descriptor __attribute__ ((unused)) * output_desc,
 bool *exact_p)
{
  pips_debug(1, "input_ref: %s\n", reference_to_string(input_ref));
  pips_debug(1, "address_of_ref: %s\n", reference_to_string(address_of_ref));
  pips_debug(1, "nb_common_indices: %d \n", nb_common_indices);

  if(simple_cell_reference_with_address_of_cell_reference_translation_fi(input_ref, input_desc, address_of_ref, address_of_desc, nb_common_indices, output_ref, output_desc, exact_p)) { // Francois' special case
    ;
  }
  else { // Beatrice's code

  /* assume exactness */
  *exact_p = true;

  /* */
  *output_ref = copy_reference(address_of_ref);
  // FI: this is already wrong for Pointers/array15.c
  // We need output_indices=reference_indices(*output_ref)
  list output_indices = gen_last(reference_indices(*output_ref));
  list input_remaining_indices = reference_indices(input_ref);
  for(int i = 0; i<nb_common_indices; i++, POP(input_remaining_indices));

  /* special case for the first remaining index: we must add it to the
     last index of build_ref. FI: In fact, to the first one... */
  if (!ENDP(output_indices)) {
    expression last_output_indices_exp = EXPRESSION(CAR(output_indices));
    expression first_input_remaining_exp = EXPRESSION(CAR(input_remaining_indices));
    expression new_exp = expression_undefined;
    /* adapted from the address_of case of c_simple_effects_on_formal_parameter_backward_translation
       this maybe should be put in another function
    */
    if(!unbounded_expression_p(last_output_indices_exp)) {
      if (expression_reference_p(last_output_indices_exp) &&
	  entity_field_p(expression_variable(last_output_indices_exp))) {
	if (!expression_equal_integer_p(first_input_remaining_exp, 0)) {
	  pips_user_warning("potential memory overflow due to effect -> returning anywhere\n");
	  free_reference(*output_ref);
	  *output_ref = make_reference(entity_all_locations(), NIL);
	  *exact_p = false;
	}
	else
	  new_exp = last_output_indices_exp;
      }

      else if(!unbounded_expression_p(first_input_remaining_exp)) {
	value v;
	intptr_t i_last_output_indices_exp;
	intptr_t i_first_input_remaining_exp;

	bool b_i_last_output_indices_exp = expression_integer_value(last_output_indices_exp, &i_last_output_indices_exp);
	bool b_i_first_input_remaining_exp = expression_integer_value(first_input_remaining_exp, &i_first_input_remaining_exp);

	if (b_i_last_output_indices_exp && i_last_output_indices_exp == 0) {
	  new_exp = copy_expression(first_input_remaining_exp);
	  POP(input_remaining_indices);
	}
	else if (b_i_first_input_remaining_exp && i_first_input_remaining_exp == 0) {
	  new_exp = copy_expression(last_output_indices_exp);
	  POP(input_remaining_indices);
	}
	else {
	  new_exp = MakeBinaryCall
	    (entity_intrinsic(PLUS_OPERATOR_NAME),
	     copy_expression(last_output_indices_exp), copy_expression(first_input_remaining_exp));
	  /* Then we must try to evaluate the expression */
	  v = EvalExpression(new_exp);
	  if (! value_undefined_p(v) &&
	      value_constant_p(v)) {
	    constant vc = value_constant(v);
	    if (constant_int_p(vc)) {
	      free_expression(new_exp);
	      new_exp = int_to_expression(constant_int(vc));
	    }
	  }
	  POP(input_remaining_indices);
	}
      }
      else {
	new_exp = make_unbounded_expression();
	*exact_p = false;
	POP(input_remaining_indices);
      }
      if (! entity_all_locations_p(reference_variable(*output_ref))) {
	/* FI->BC: this looks much more like a concatenation than
	   a substitution of the first subscript. */
	//CAR(gen_last(reference_indices(*output_ref))).p
	//  = (void *) new_exp;
	expression old_s = EXPRESSION(CAR(gen_last(reference_indices(*output_ref))));
	free_expression(old_s);
	// FI: I do not see why it should be a change of the last subscript
	// I'd rather see a change of the first subscript...
	EXPRESSION_(CAR(gen_last(reference_indices(*output_ref)))) = new_exp;
      }
    }
    else {
      *exact_p = false;
      POP(input_remaining_indices); // a * subscript absorbs everything
    }
  }
  else /* ENDP(output_indices) */ {
    /* address_of_ref is a scalar: the first remaning index must be equal to 0 */
    expression first_input_remaining_exp = EXPRESSION(CAR(input_remaining_indices));
    if (!expression_equal_integer_p(first_input_remaining_exp, 0)) {
      /* FI->BC: A much better job could be done using a function
	 similar to source_to_sinks(). See for instance
	 Pointers/properties03.c: if the analysis is performed at
	 points-to level, the result is precise; if the very same
	 analysis is performed by effects_with_points_to, an
	 anywhere results. */
      pips_user_warning("potential memory overflow due to effect -> returning anywhere\n");
      free_reference(*output_ref);
      // FI->BC: conditionally to a property,
      // ALIASING_ACROSS_TYPES, a typed anywhere should be generated
      *output_ref = make_reference(entity_all_locations(), NIL);
      *exact_p = false;
    }
  }

  // FI->BC: something is missing here points-to
  // If the target is an array, the first subscript at least should be
  // replicated: if "p" in a pointer to an array "a", "p[0]" is "a[0]", not "a"
  //
  // This happens because "int * p;" defines implictly "p" as a
  // pointer to an array

  if (! entity_all_locations_p(reference_variable(*output_ref)))
    {
      entity v = reference_variable(*output_ref);
      // type vt = entity_type(v); // FI: should probably be a concrete
      // FI: see EffectWitPointsTo/call09.c
      type vt = entity_basic_concrete_type(v);
      // type, but hopefully this has been done earlier when computing
      // points-to
      // FI: under some circumstances, the first zero subscript must
      // be ignored. See for instance EffectsWithPointsTo/call09.c
      // list sl = (false && array_type_p(vt))? input_remaining_indices :
      //	CDR(input_remaining_indices);
      list sl = input_remaining_indices; // default value
      // FI->FI: Beatrice uses a more powerful zero detection
      if(!array_type_p(vt)
	 && !ENDP(sl)
	 && zero_expression_p(EXPRESSION(CAR(sl))))
	sl = CDR(input_remaining_indices);

      FOREACH(EXPRESSION, input_ind, sl)
	{
	  reference_indices(*output_ref) = gen_nconc(reference_indices(*output_ref),
						     CONS(EXPRESSION,
							  copy_expression(input_ind),
							  NIL));
	}
    }
  }

  // FI: the consistency of the regenerated reference should be checked
  // by computing its type, if only, under a ifdebug() guard...
  pips_debug(8, "output reference %s\n",
	     reference_to_string(*output_ref));
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
