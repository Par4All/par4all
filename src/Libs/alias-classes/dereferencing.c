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
      // FI: you do not want to apply side-effects twice...
      // But you then miss the detection of pointers that are not NULL
      // because they are dereferenced
      //call c = syntax_call(s);
      //list al = call_arguments(c);
      //in = expressions_to_points_to(al, in);
      //in = call_to_points_to(c, in);
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

/* see if a points_to_reference includes a pointer dereferencing: this is impossible if the points-to reference is consistent. It must be a constant path. */
bool pointer_points_to_reference_p(reference r __attribute__ ((unused)))
{
  return false;
}

void pointer_reference_dereferencing_to_points_to(reference r, pt_map in)
{
  expression pae = pointer_reference_to_expression(r);
  // FI: assume side effects are OK...
  (void) expression_to_points_to(pae, in, true);
  free_expression(pae);
}

/* Can we execute the reference r in points-to context "in" without
 * segfaulting?
 *
 * Do not go down in subscript expressions.
 *
 * See also reference_to_points_to_sinks(). Unfortunate cut-and-paste.
 */
pt_map reference_dereferencing_to_points_to(reference r,
					    pt_map in,
					    bool nowhere_dereferencing_p,
					    bool null_dereferencing_p)
{
  entity e = reference_variable(r);
  type t = entity_basic_concrete_type(e);
  list sl = reference_indices(r);

  /* Does the reference implies some dereferencing itself? */
  if(pointer_points_to_reference_p(r)) {
    pointer_reference_dereferencing_to_points_to(r, in);
  }
  else if(pointer_type_p(t) && !ENDP(sl)) {
    pointer_reference_dereferencing_to_points_to(r, in);
  }
  else if(array_of_pointers_type_p(t)
	  && (int) gen_length(sl)>variable_dimension_number(type_variable(t))) {
    pointer_reference_dereferencing_to_points_to(r, in);
  }
  else if(array_type_p(t) && !array_of_pointers_type_p(t)) {
    /* C syntactic sugar: *a is equivalent to a[0] when a is an
       array. No real dereferencing needed. */
    ;
  }
  else {
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
      points_to_graph_bottom(in) = true;
      if(statement_points_to_context_defined_p())
	pips_user_warning("Null or undefined pointer may be dereferenced because of \"%s\" at line %d.\n",
			  effect_reference_to_string(r),
			  points_to_context_statement_line_number());
      else
	pips_user_warning("Null or undefined pointer may be dereferenced because of \"%s\".\n",
			  effect_reference_to_string(r));
    }

    gen_free_list(sinks);
  }
  return in;
}

/* Can expression e be reduced to a reference, without requiring an evaluation?
 *
 * For instance expression "p" can be reduced to reference "p".
 *
 * Expression "p+i" annot be reduced to a reference.
 *
 * Ad'hoc development for dereferencing_to_sinks.
 */
bool expression_to_points_to_cell_p(expression e)
{
  bool source_p = true;
  syntax s = expression_syntax(e);
  if(syntax_reference_p(s))
    source_p = true;
  else if(syntax_call_p(s)) {
    call c = syntax_call(s);
    entity f = call_function(c);
    if(ENTITY_PLUS_C_P(f))
      source_p = false;
  }
  return source_p;
}

/* Returns "sinks", the list of cells pointed to by expression "a"
 * according to points-to graph "in".
 *
 * If eval_p is true, perform a second dereferencing on the cells
 * obtained with the first dereferencing.
 *
 * Manage NULL and undefined (nowhere) cells.
 *
 * Possibly update the points-to graph when some arcs are incompatible
 * with the request, assuming the analyzed code is correct.
 */
list dereferencing_to_sinks(expression a, pt_map in, bool eval_p)
{
  list sinks = NIL;
  /* Locate the pointer, no dereferencing yet... unless no pointer can
     be found as in *(p+2) in which case an evaluation occurs in
     expression_to_points_to_sources(). */
  list cl = expression_to_points_to_sources(a, in);
  remove_impossible_arcs_to_null(&cl, in);
  // The pointer may not be found: *(p+i)
  // list cl = expression_to_points_to_sinks(a, in);
  bool evaluated_p = !expression_to_points_to_cell_p(a);
  // evaluated_p = false;
  if(!evaluated_p || eval_p) {
    bool null_dereferencing_p
      = get_bool_property("POINTS_TO_NULL_POINTER_DEREFERENCING");
    bool nowhere_dereferencing_p
      = get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING");

    /* Finds what it is pointing to, memory(p) */
    FOREACH(CELL, c, cl) {
      /* Do we want to dereference c? */
      if( (null_dereferencing_p || !null_cell_p(c))
	  && (nowhere_dereferencing_p || !nowhere_cell_p(c))) {
	list o_pointed = source_to_sinks(c, in, true);
	remove_impossible_arcs_to_null(&o_pointed, in);
	/* Do not create sharing between elements of "in" and elements of
	   "sinks". */
	// list pointed = source_to_sinks(c, in, true);
	list pointed = gen_full_copy_list(o_pointed);
	gen_free_list(o_pointed);
	if(ENDP(pointed)) {
	  reference r = cell_any_reference(c);
	  entity v = reference_variable(r);
	  string words_to_string(list);
	  pips_user_warning("No pointed location for variable \"%s\" and reference \"%s\"\n",
			    entity_user_name(v),
			    words_to_string(words_reference(r, NIL)));
	  /* The sinks list is empty, whether eval_p is true or not... */
	}
	else {
	  if(!evaluated_p && eval_p) {
	    FOREACH(CELL, sc, pointed) {
	      bool to_be_freed;
	      type t = points_to_cell_to_type(sc, &to_be_freed);
	      if(!pointer_type_p(t)) {
		// FI: it might be necessary to allocate a new copy of sc
		sinks = gen_nconc(sinks, CONS(CELL, sc, NIL));
	      }
	      else if(array_type_p(t) || struct_type_p(t)) {
		/* FI: New cells have been allocated by
		   source_to_sinks(): side-effects are OK. In
		   theory... The source code of source_to_sinks() seems
		   to show that fresh_p is not exploited in all
		   situations. */
		//cell nsc = copy_cell(sc);
		//points_to_cell_add_zero_subscripts(nsc);
		//sinks = gen_nconc(sinks, CONS(CELL, nsc, NIL));
		pips_internal_error("sc assume saturated with 0 subscripts and/or field susbscripts.\n");
	      }
	      else {
		list starpointed = extended_source_to_sinks(sc, in);
		sinks = gen_nconc(sinks, starpointed);
	      }
	      if(to_be_freed) free_type(t);
	    }
	  }
	  else
	    sinks = gen_nconc(sinks, pointed);
	}
      }
    }
  }
  else
    sinks = cl;
  return sinks;
}
