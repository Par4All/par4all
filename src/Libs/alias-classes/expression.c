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

/*
 * This file contains functions used to compute points-to sets at
 * expression level.
 *
 * The argument pt_in is always modified by side-effects and returned.
 */

#include <stdlib.h>
#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
//#include "control.h"
#include "constants.h"
#include "misc.h"
//#include "parser_private.h"
//#include "syntax.h"
//#include "top-level.h"
//#include "text-util.h"
//#include "text.h"
#include "properties.h"
//#include "pipsmake.h"
//#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
//#include "transformations.h"
//#include "preprocessor.h"
//#include "pipsdbm.h"
//#include "resources.h"
//#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

pt_map expression_to_points_to(expression e, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  syntax s = expression_syntax(e);
  tag t = syntax_tag(s);

  switch(t) {
  case is_syntax_reference: {
    reference r = syntax_reference(s);
    pt_out = reference_to_points_to(r, pt_in);
    break;
  }
  case is_syntax_range: {
    range r = syntax_range(s);
    pt_out = range_to_points_to(r, pt_in);
    break;
  }
  case is_syntax_call: {
    call c = syntax_call(s);
    pt_out = call_to_points_to(c, pt_in);
    break;
  }
  case is_syntax_cast: {
    cast c = syntax_cast(s);
    expression ce = cast_expression(c);
    pt_out = expression_to_points_to(ce, pt_in);
    break;
  }
  case is_syntax_sizeofexpression: {
    sizeofexpression soe = syntax_sizeofexpression(s);
    if(sizeofexpression_type_p(soe))
      ; // pt_in is not modified
    else {
      expression ne = sizeofexpression_expression(soe);
      pt_out = expression_to_points_to(ne, pt_in);
    }
    break;
  }
  case is_syntax_subscript: {
    subscript sub = syntax_subscript(s);
    expression a = subscript_array(sub);
    list sel = subscript_indices(sub);
    pt_out = expression_to_points_to(a, pt_in);
    pt_out = expressions_to_points_to(sel, pt_out);
    break;
  }
  case is_syntax_application: {
    application a = syntax_application(s);
    pt_out = application_to_points_to(a, pt_in);
    break;
  }
  case is_syntax_va_arg: {
    pips_internal_error("Not implemented yet for va_arg\n");
    break;
  }
  default:
    ;
  }
  return pt_out;
}

/* Compute the points-to information pt_out that results from the
 * evaluation of a possibly empty list of expression. A new data structure is allocated.
 */
pt_map expressions_to_points_to(list el, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  //pt_map pt_prev = copy_set(pt_in);
  FOREACH(EXPRESSION, e, el) {
    pt_out = expression_to_points_to(e, pt_out);
    //pt_map pt_new = expression_to_points_to(e, pt_prev);
    //free_set(pt_prev);
    //pt_prev = pt_new;
  }
  //pt_out = pt_prev;

  return pt_out;
}

/* The subscript expressions may impact the points-to
   information. E.g. a[*(p++)] */
pt_map reference_to_points_to(reference r, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  list sel = reference_indices(r);
  pt_out = expressions_to_points_to(sel, pt_in);
  return pt_out;
}

pt_map range_to_points_to(range r, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  expression l = range_lower(r);
  expression u = range_upper(r);
  expression i = range_increment(r);
  pt_out = expression_to_points_to(l, pt_in);
  pt_out = expression_to_points_to(u, pt_out);
  pt_out = expression_to_points_to(i, pt_out);
  return pt_out;
}

/* Three different kinds of calls are distinguished:
 *
 * - calls to constants, e.g. NULL,
 *
 * - calls to intrinsics, e.g. ++ or malloc(),
 *
 * - and calls to a user function.
 */
pt_map call_to_points_to(call c, pt_map pt_in)
{
  // FI: see functio points_to_call()
  pt_map pt_out = pt_in;
  entity f = call_function(c);
  list al = call_arguments(c);
  value fv = entity_initial(f);

  /* points-to updates due to arguments */
  pt_out = expressions_to_points_to(al, pt_in);

  /* points-to updates due to the function itself */
  if(entity_constant_p(f)) {
    // pt_out = constant_call_to_points_to(c, pt_out);
    pt_out = pt_in;
  }
  else if(intrinsic_entity_p(f))
    pt_out = intrinsic_call_to_points_to(c, pt_out);
  else if(symbolic_entity_p(f))
    pt_out = pt_in; // FI?
  else if(value_unknown_p(fv)) {
    pips_internal_error("function %s has an unknown value\n",
                        entity_name(f));
  }
  else {
    // must be a user-defined function
    pips_assert("f is a user-defined function", value_code_p(entity_initial(f)));
    pt_out = user_call_to_points_to(c, pt_out);
  }

  return pt_out;
}

/* FI: this should not generate any points-to update
 *
 * it would be better not to go down here
 */
pt_map constant_call_to_points_to(call c __attribute__ ((unused)), pt_map pt_in)
{
  pt_map pt_out = pt_in;

  return pt_out;
}

pt_map intrinsic_call_to_points_to(call c, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  entity f = call_function(c);
  list al = call_arguments(c);

  set_methods_for_proper_simple_effects();
  //list el = call_to_proper_effects(c);
  generic_effects_reset_all_methods();

  pips_debug(5, "intrinsic function \"%s\"\n", entity_name(f));

  // FI: short term version
  // pt_out = points_to_intrinsic(statement_undefined, c, f, al, pt_in, el);
  // return pt_out;

  // FI: Where should we check that the update is linked to a pointer?
  // Should we go down because a pointer assignment may be hidden anywhere...
  // Or have we already taken care of this in call_to_points_to()

  if(ENTITY_ASSIGN_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    expression rhs = EXPRESSION(CAR(CDR(al)));
    pt_out = assignment_to_points_to(lhs, rhs, pt_in);
  }
  else if(ENTITY_PLUS_UPDATE_P(f)) {
    /* Many update operators */
    pips_internal_error("Not implemented yet\n");
    ;
  }
  else if(ENTITY_POST_INCREMENT_P(f)) {
    /* Four increment related operators */
    pips_internal_error("Not implemented yet\n");
    ;
  }
  else {
    // Not safe till all previous tests are defined
    // It is assumed that other intrinsics do not generate points-to arcs...
    // pips_internal_error("Not implemented yet\n");
    pt_out = pt_in;
  }

  return pt_out;
}

pt_map user_call_to_points_to(call c, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  entity f = call_function(c);

  // FI: intraprocedural, use effects
  // FI: interprocedural, check alias compatibility, generate gen and kill sets,...

  // FI: we need a global variable here to make the decision without
  // propagating an extra parameter everywhere

  // pips_internal_error("Not implemented yet for function \"%s\"\n", entity_user_name(f));

  pips_user_warning("The function call to \"%s\" is still ignored\n"
		    "On going implementation...\n", entity_user_name(f));
  //set_assign(pt_out, pt_in);
  pt_out = pt_in;

  return pt_out;
}

pt_map assignment_to_points_to(expression lhs, expression rhs, pt_map pt_in)
{
  //pt_map pt_out = pt_int;
  pt_map pt_out = expression_to_points_to(lhs, pt_in);
  /* It is not obvious that you are allowed to evaluate this before
     the sink of lhs, but the standard probably forbid stupid side
     effects. */
  pt_out = expression_to_points_to(lhs, pt_out);
  type t = expression_to_type(lhs); // FI: let's hope ultimate type is useless here

  if(pointer_type_p(t))
    pt_out = pointer_assignment_to_points_to(lhs, rhs, pt_out);
  else if(struct_type_p(t))
    pt_out = struct_assignment_to_points_to(lhs, rhs, pt_out);
  // FI: unions are not dealt with...
  else
    pt_out = pt_in; // What else?

  free_type(t);

  return pt_out;
}

/* Any abstract location of the lhs in L is going to point to any sink of
 * any abstract location of the rhs in R.
 *
 * New points-to information must be added when a formal parameter
 * is dereferenced.
 *
 * FI: Side effects of lhs and rhs have been taken care of at a higher
 * level? I do not think so in general...
 */
pt_map pointer_assignment_to_points_to(expression lhs,
				       expression rhs,
				       pt_map pt_in)
{
  pt_map pt_out = pt_in;
  // pips_internal_error("Not implemented yet for lhs %p and rhs %p\n", lhs, rhs);

  /* Take side effects into account */
  pt_out = expression_to_points_to(lhs, pt_out);
  pt_out = expression_to_points_to(rhs, pt_out); // FI: used to be "incur"

  /* Change the "lhs" into a constant memory path using current
   * points-to information pt_out.
   *
   * FI:  Necessary nodes and arcs in the calling context should be added
   * on-demand. I have no idea if this function does it or not...
   */
  list L = expression_to_constant_paths(statement_undefined, lhs, pt_out);

  // FI: more work needed here, shut up gcc
  // pips_internal_error("Not implemented yet %p", L);

  /* Retrieve the memory locations that might be reached by the rhs
   *
   */
  list R = expression_to_points_to_sinks(rhs,pt_out);

  /* Compute the data-flow equation for the may and the must edges...
   *
   * out = (in - kill) U gen ?
   */

 /* Extract MAY/MUST points to relations from the input set "incur",
  * now the current pt_out... */
  pt_map in_may = points_to_may_filter(pt_out);
  pt_map in_must = points_to_must_filter(pt_out);
  pt_map kill_may = kill_may_set(L, in_may);
  pt_map kill_must = kill_must_set(L, pt_out);
  // FI: I am lost with &address_of_p
  // FI: I assumed the conversion performed earlier in sinks, AM performs it in
  // gen_must_constant_paths() which may or not be better;
  bool address_of_p = true;
  pt_map gen_may = gen_may_set(L, R, in_may, &address_of_p);
  pt_map gen_must = gen_must_set(L, R, in_must, &address_of_p);
  pt_map kill = new_pt_map();
  set_union(kill, kill_may, kill_must);
  pt_map gen = new_pt_map();
  set_union(gen, gen_may, gen_must);
  if( set_empty_p(gen) ) {
    bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
    if(type_sensitive_p)
      gen = points_to_anywhere_typed(L, pt_out);
    else
      gen = points_to_anywhere(L, pt_out); 
  }
  set_difference(pt_out, pt_out, kill);
  set_union(pt_out, pt_out, gen);

  free_pt_maps(in_may, in_must,
	       kill_may, kill_must,
	       gen_may, gen_must,
	       gen, kill, NULL);
  // clear_pt_map(pt_out); // FI: why not free?

  return pt_out;
}

pt_map struct_assignment_to_points_to(expression lhs, expression rhs, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  pips_internal_error("Not implemented yet for lhs %p and rhs %p\n", lhs, rhs);

  return pt_out;
}

pt_map application_to_points_to(application a, pt_map pt_in)
{
  expression f = application_function(a);
  list al = application_arguments(a);
  pt_map pt_out = expression_to_points_to(f, pt_in);

  pt_out = expressions_to_points_to(al, pt_out);
  /* FI: We should also identify the possibly called functions and
     update the points-to according to the possible call sites. */
  pips_internal_error("Not implemented yet for application\n");

  return pt_out;
}
