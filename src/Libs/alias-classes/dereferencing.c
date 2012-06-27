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
 * This file contains functions used to make sure that all
 * dereferencing contained in an expression can be performed within a
 * given points-to. It might not be a good idea to seperate
 * dereferenging issues from points-to modifications as side effects
 * are allowed.
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
#include "text-util.h"
#include "text.h"
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


/* Make sure that expression p can be dereferenced in points-to graph "in" 
 *
 * Handling of NULL pointers according to property.
 *
 * Handling of undefined pointers according to property.
 *
 * "in" is modified by side effects. Arcs certainly incompatible with
 * a dereferencing are removed. If dereferencing of p is no longer
 * possible, return an empty points-to "in" as the expression cannot
 * be evaluated.
 *
 * This is conditional to two properties.
 */
pt_map dereferencing_to_points_to(expression p, pt_map in)
{
  //pt_map out = in;
  bool null_dereferencing_p
    = get_bool_property("POINTS_TO_NULL_POINTER_DEREFERENCING");
  bool nowhere_dereferencing_p
    = get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING");
  // FI: we cannot use expression_to_points_to_sources or sinks
  // because side effects are taken into acount each time they are
  // called.
#if 0
  list sources = expression_to_points_to_sources(p, in);
  if(gen_length(sources)==1
     && !nowhere_dereferencing_p
     && !null_dereferencing_p) {
    list sinks = expression_to_points_to_sinks(p, in);
    cell source = CELL(CAR(sources));
    int n = (int) gen_length(sinks);
    FOREACH(CELL, sink, sinks) {
      if(!nowhere_dereferencing_p && nowhere_cell_p(sink)) {
	remove_points_to_arcs(source, sink, in);
	n--;
      }
      if(!null_dereferencing_p && null_cell_p(sink)) {
	remove_points_to_arcs(source, sink, in);
	n--;
      }
    }
    if(n==0)
      clear_pt_map(in);
  }
  else {
      /* The issue will/might be taken care of later... */
      ;
  }
#endif
  if(!nowhere_dereferencing_p && !null_dereferencing_p) {
    syntax s = expression_syntax(p);
    tag t = syntax_tag(s);

    switch(t) {
    case is_syntax_reference: {
      reference r = syntax_reference(s);
      /*out = */reference_dereferencing_to_points_to(r, in,
						 nowhere_dereferencing_p,
						 null_dereferencing_p);
      break;
    }
    case is_syntax_range: {
      //range r = syntax_range(s);
      // out = range_to_points_to(r, in);
      break;
    }
    case is_syntax_call: {
      //call c = syntax_call(s);
      //out = call_to_points_to(c, in);
      break;
    }
    case is_syntax_cast: {
      //cast c = syntax_cast(s);
      //expression ce = cast_expression(c);
      //out = expression_to_points_to(ce, in);
      break;
    }
    case is_syntax_sizeofexpression: {
      //sizeofexpression soe = syntax_sizeofexpression(s);
      //if(sizeofexpression_type_p(soe))
      //; // in is not modified
      //else {
	// expression ne = sizeofexpression_expression(soe);
	// FI: we have a problem because sizeof(*p) does not imply that
	// *p is evaluated...
	// out = expression_to_points_to(ne, in);
      //;
      //}
      break;
    }
    case is_syntax_subscript: {
      //subscript sub = syntax_subscript(s);
      //expression a = subscript_array(sub);
      //list sel = subscript_indices(sub);
      /* a cannot evaluate to null or undefined */
      /* FI: we may need a special case for stubs... */
      //out = dereferencing_to_points_to(a, in);
      //out = expression_to_points_to(a, out);
      //out = expressions_to_points_to(sel, out);
      break;
    }
    case is_syntax_application: {
      // FI: we might have to go down here...
      //application a = syntax_application(s);
      //out = application_to_points_to(a, out);
      break;
    }
    case is_syntax_va_arg: {
      // The call to va_arg() does not create a points-to per se
      //list soel = syntax_va_arg(s);
      //sizeofexpression soe1 = SIZEOFEXPRESSION(CAR(soel));
      //sizeofexpression soe2 = SIZEOFEXPRESSION(CAR(CDR(soel)));
      //expression se = sizeofexpression_expression(soe1);
      // type t = sizeofexpression_type(soe2);
      //out = expression_to_points_to(se, out);
      break;
    }
    default:
      ;
    }
  }
  return in;
}

/* Can we execute the reference r in points-to context "in" without
 * segfaulting?
 *
 * Do not go down in subscript expressions.
 */
pt_map reference_dereferencing_to_points_to(reference r,
					    pt_map in,
					    bool nowhere_dereferencing_p,
					    bool null_dereferencing_p)
{
  reference nr = copy_reference(r);
  cell source = make_cell_reference(nr);
  /* Remove store-dependent indices */
  reference_indices(nr) =
    subscript_expressions_to_constant_subscript_expressions(reference_indices(nr));
  list sinks = source_to_sinks(source, in, false);
  int n = (int) gen_length(sinks);
  FOREACH(CELL, sink, sinks) {
    if(!nowhere_dereferencing_p && nowhere_cell_p(sink)) {
      remove_points_to_arcs(source, sink, in);
      n--;
    }
    if(!null_dereferencing_p && null_cell_p(sink)) {
      remove_points_to_arcs(source, sink, in);
      n--;
    }
  }
  if(n==0) {
    clear_pt_map(in);
    pips_user_warning("Null or undefined pointer may be dereferenced.\n");
  }

  gen_free_list(sinks);

  return in;
}
