/*

  $Id: points_to_analysis_general_algorithm.c 21101 2012-03-04 20:38:17Z amini $

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
This file contains functions used to compute points-to sets at expression level.
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
#include "pipsdbm.h"
#include "resources.h"
//#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

pt_map expression_to_points_to(expression e, pt_map pt_in)
{
  pt_map pt_out;
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
      pt_out = copy_set(pt_in);
    else {
      expression ne = sizeofexpression_expression(soe);
      pt_out = expression_to_points_to(ne, pt_in);
    }
    break;
  }
  case is_syntax_subscript: {
    subscript sub = syntax_subscript(s);
    list sel = subscript_indices(sub);
    pt_out = expressions_to_points_to(sel, pt_in);
    break;
  }
  case is_syntax_application: {
    application a = syntax_application(s);
    pt_out = application_to_points_to(a, pt_in);
    break;
  }
  case is_syntax_va_arg: {
    pips_internal_error("Not implemented yet\n");
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
  pt_map pt_out;
  pt_map pt_prev = copy_set(pt_in);
  FOREACH(EXPRESSION, e, el) {
    pt_map pt_new = expression_to_points_to(e, pt_prev);
    free_set(pt_prev);
    pt_prev = pt_new;
  }
  pt_out = pt_prev;

  return pt_out;
}

/* The subscript expressions may impact the points-to
   information. E.g. a[*(p++)] */
pt_map reference_to_points_to(reference r, pt_map pt_in)
{
  pt_map pt_out;
  list sel = reference_indices(r);
  pt_out = expressions_to_points_to(sel, pt_in);
  return pt_out;
}

pt_map range_to_points_to(range r, pt_map pt_in)
{
  pt_map pt_out;
  expression l = range_lower(r);
  expression u = range_upper(r);
  expression i = range_increment(r);
  pt_map pt_out1 = expression_to_points_to(l, pt_in);
  pt_map pt_out2 = expression_to_points_to(u, pt_out1);
  pt_out = expression_to_points_to(i, pt_out2);
  /* FI: pt_map dependence... */
  free_set(pt_out1), free_set(pt_out2);
  return pt_out;
}

/* Three different kinds of calls are distinguished: calls to
   constants, e.g. NULL, calls to intrinsics, e.g. ++ or malloc(), and
   calls to a user function. */
pt_map call_to_points_to(call c, pt_map pt_in)
{
  pt_map pt_out;
  entity f = call_function(c);
  if(entity_constant_p(f))
    pt_out = constant_call_to_points_to(c, pt_in);
  else if(intrinsic_entity_p(f))
    pt_out = intrinsic_call_to_points_to(c, pt_in);
  else // must be a user-defined function
    pt_out = user_call_to_points_to(c, pt_in);
  return pt_out;
}

/* FI: this shold not generate any points-to update
 *
 * it would be better not to go down here to avoid an allocate and a
 * free.
 */
pt_map constant_call_to_points_to(call c, pt_map pt_in)
{
  pt_map pt_out;
  entity f = call_function(c);

  pt_out = copy_set(pt_in);

  return pt_out;
}

pt_map intrinsic_call_to_points_to(call c, pt_map pt_in)
{
  pt_map pt_out;
  entity f = call_function(c);

  if(ENTITY_ASSIGN_P(f)) {
    list al = call_arguments(c);
    expression lhs = EXPRESSION(CAR(al));
    expression rhs = EXPRESSION(CAR(CDR(al)));
    pt_out = assignment_to_points_to(lhs, rhs, pt_in);
  }
  else if(ENTITY_PLUS_UPDATE_P(f))
    ;
  else if(ENTITY_POST_INCREMENT_P(f))
    ;
  else
    pt_out = copy_set(pt_in);

  return pt_out;
}

pt_map user_call_to_points_to(call c, pt_map pt_in)
{
  pt_map pt_out;
  entity f = call_function(c);

  // FI: intraprocedural, use effects
  // FI: interprocedural, check alias compatibility, generate gen and kill sets,...

  pips_internal_error("Not implemented yet\n");

  return pt_out;
}

pt_map assignment_to_points_to(expression lhs, expression rhs, pt_map pt_in)
{
  pt_map pt_out;
  pt_map pt_out1 = expression_to_points_to(lhs, pt_in);
  /* It is not obvious that you are allowed to evaluate this before
     the sink of lhs, but the standard probably forbid stupid side
     effects. */
  pt_map pt_out2 = expression_to_points_to(lhs, pt_out1);
  type t = expression_to_type(lhs); // FI: let's hope ultimate type is useless here

  if(pointer_type_p(t))
    pt_out = pointer_assignment_to_points_to(lhs, rhs, pt_out2);
  else if(struct_type_p(t))
    pt_out = struct_assignment_to_points_to(lhs, rhs, pt_out2);
  else
    pt_out = copy_set(pt_in); // What else?

  free_set(pt_out1), free_set(pt_out2);

  return pt_out;
}

pt_map pointer_assignment_to_points_to(expression lhs, expression rhs, pt_map pt_in)
{
  pt_map pt_out;

  return pt_out;
}

pt_map struct_assignment_to_points_to(expression lhs, expression rhs, pt_map pt_in)
{
  pt_map pt_out;

  return pt_out;
}

pt_map application_to_points_to(application a, pt_map pt_in)
{
  expression f = application_function(a);
  list al = application_arguments(a);
  pt_map pt_int = expression_to_points_to(f, pt_in);
  pt_map pt_out = expressions_to_points_to(al, pt_int);
  /* FI: We should also identify the possibly called functions and
     update the points-to according to the possible call sites. */
  pips_internal_error("Not implemented yet\n");
  free_set(pt_int);
  return pt_out;
}
