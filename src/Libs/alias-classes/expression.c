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

/* FI: what is this function supposed to do? Just update "pt_in" to
   make sure that "r" can be dereferenced? And then recursively, with
   the different subscripts? */
void subscripted_reference_to_points_to(reference r, list sl, pt_map pt_in)
{
  if(!ENDP(sl)) {
    type rt = points_to_reference_to_concrete_type(r);
    if(pointer_type_p(rt)) {
      //type pt = type_to_pointed_type(rt);
      //list cl = reference_to_points_to_sinks(r, pt, pt_in, false, true);
      list cl = reference_to_points_to_sinks(r, rt, pt_in, true, true);
      // FI: the arc between "r" and NULL should be removed...
      remove_impossible_arcs_to_null(&cl, pt_in);
      FOREACH(CELL, c, cl) {
	if(!ENDP(CDR(sl))) {
	  expression fs = EXPRESSION(CAR(sl));
	  /* FI: we have to find the right location for the subscript
	   * to update. Some dimensions are due to the dimension of
	   * the source in pt_in, one dimension is due to the fact
	   * that we are dealing with a pointer, some dimensions are
	   * due to the fact that an array is pointed. The dimension
	   * to update may be the first one, the last one, or one in
	   * the middle.
	   *
	   * This also depends on strict typing...
	   *
	   * See for instance, Pointers/pointer20.c
	   */
	  reference or = cell_any_reference(c);
	  if(adapt_reference_to_type(or, rt, points_to_context_statement_line_number)) {
	    list osl = reference_indices(or);
	    if(ENDP(osl)) {
	      reference_indices(or) = CONS(EXPRESSION,
					   copy_expression(EXPRESSION(CAR(sl))),
					   NIL);
	    }
	    else {
	      points_to_cell_update_last_subscript(c, fs);
	    }
	    reference nr = cell_any_reference(c);
	    subscripted_reference_to_points_to(nr, CDR(sl), pt_in);
	  }
	  else
	    pips_internal_error("reference could not be updated.\n");
	}
      }
    }
    else if(array_of_pointers_type_p(rt)) {
      pips_internal_error("Not implemented yet.\n");
    }
    else
      pips_internal_error("Meaningless call.\n");
  }
}

/* Update pt_in and pt_out according to expression e.
 *
 * Ignore side effects due to pointer arithmetic and assignment and
 * function calls if side_effet_p is not set. This may be useful when
 * conditions are evaluated twice, once for true and once for false.
 */
pt_map expression_to_points_to(expression e, pt_map pt_in, bool side_effect_p)
{
  pt_map pt_out = pt_in;
  if(!points_to_graph_bottom(pt_in)) {
    syntax s = expression_syntax(e);
    tag t = syntax_tag(s);

    switch(t) {
    case is_syntax_reference: {
      reference r = syntax_reference(s);
      list sl = reference_indices(r);
      entity v = reference_variable(r);
      type vt = entity_basic_concrete_type(v);
      // FI: call16.c shows that the C parser does not generate the
      // right construct, a subscript, when a scalar pointer is indexed
      if(!ENDP(sl)) {
	if(pointer_type_p(vt)) {
	  // expression tmp = entity_to_expression(v);
	  // pt_out = dereferencing_to_points_to(tmp, pt_in);
	  // pt_out = expressions_to_points_to(sl, pt_out, side_effect_p);
	  // free_expression(tmp);
	  reference nr = make_reference(v, NIL);
	  subscripted_reference_to_points_to(nr, sl, pt_in);
	}
	else if(array_of_pointers_type_p(vt)) {
	  int td = type_depth(vt);
	  int sn = (int) gen_length(sl);
	  if(sn<=td) {
	    ; // Nothing to do: a standard array subscript list
	  }
	  else
	    pips_internal_error("Not implemented yet.\n");
	}
      }
      pt_out = reference_to_points_to(r, pt_in, side_effect_p);
      break;
    }
    case is_syntax_range: {
      range r = syntax_range(s);
      pt_out = range_to_points_to(r, pt_in, side_effect_p);
      break;
    }
    case is_syntax_call: {
      call c = syntax_call(s);
      /* Some idea, but points-to information should rather be used
       *
       * list el = expression_to_proper_constant_path_effects(e);
       *
       * Also, this would be computed before we know if it is useful
       * because we need an expression and not a call to have a function
       * to compute effects. And we do not know if we want an inter or
       * an intraprocedural points-to analysis.
       *
       * The alternative is too always compute points-to information
       * interprocedurally, which makes sense as it is done for for
       * memory effects and since points-to information is at a lower
       * level than memory effects...
       */
      // Now, "el" is a useless parameter
      list el = NIL;
      pt_out = call_to_points_to(c, pt_in, el, side_effect_p);
      gen_full_free_list(el);
      break;
    }
    case is_syntax_cast: {
      cast c = syntax_cast(s);
      expression ce = cast_expression(c);
      pt_out = expression_to_points_to(ce, pt_in, side_effect_p);
      break;
    }
    case is_syntax_sizeofexpression: {
      sizeofexpression soe = syntax_sizeofexpression(s);
      if(sizeofexpression_type_p(soe))
	; // pt_in is not modified
      else {
	// expression ne = sizeofexpression_expression(soe);
	// FI: we have a problem because sizeof(*p) does not imply that
	// *p is evaluated...
	// pt_out = expression_to_points_to(ne, pt_in);
	;
      }
      break;
    }
    case is_syntax_subscript: {
      subscript sub = syntax_subscript(s);
      expression a = subscript_array(sub);
      list sel = subscript_indices(sub);
      /* a cannot evaluate to null or undefined */
      /* FI: we may need a special case for stubs... */
      pt_out = dereferencing_to_points_to(a, pt_in);
      pt_out = expression_to_points_to(a, pt_out, side_effect_p);
      pt_out = expressions_to_points_to(sel, pt_out, side_effect_p);
      break;
    }
    case is_syntax_application: {
      application a = syntax_application(s);
      pt_out = application_to_points_to(a, pt_out, side_effect_p);
      break;
    }
    case is_syntax_va_arg: {
      // The call to va_arg() does not create a points-to per se
      list soel = syntax_va_arg(s);
      sizeofexpression soe1 = SIZEOFEXPRESSION(CAR(soel));
      //sizeofexpression soe2 = SIZEOFEXPRESSION(CAR(CDR(soel)));
      expression se = sizeofexpression_expression(soe1);
      // type t = sizeofexpression_type(soe2);
      pt_out = expression_to_points_to(se, pt_out, side_effect_p);
      break;
    }
    default:
      ;
    }
  }
  pips_assert("pt_out is consistent and defined",
	      points_to_graph_consistent_p(pt_out)
	      && !points_to_graph_undefined_p(pt_out));
  return pt_out;
}

/* Compute the points-to information pt_out that results from the
 * evaluation of a possibly empty list of expression. A new data
 * structure is allocated.
 *
 * Ignore side-effects unless side_effect_p is set to true.
 *
 * The result is correct only if you are sure that all expressions in
 * "el" are always evaluated.
 */
pt_map expressions_to_points_to(list el, pt_map pt_in, bool side_effect_p)
{
  pt_map pt_out = pt_in;
  FOREACH(EXPRESSION, e, el) {
    if(points_to_graph_bottom(pt_out))
      break;
    pt_out = expression_to_points_to(e, pt_out, side_effect_p);
  }

  return pt_out;
}

/* The subscript expressions may impact the points-to
 * information. E.g. a[*(p++)]
 *
 * FI: I'm surprised that pointers can be indexed instead of being
 * subscripted... This is linked to the parser in
 * expression_to_points_to().
 */
pt_map reference_to_points_to(reference r, pt_map pt_in, bool side_effect_p)
{
  pt_map pt_out = pt_in;
  if(!points_to_graph_bottom(pt_in)) {
    list sel = reference_indices(r);
    entity v = reference_variable(r);
    type t = entity_basic_concrete_type(v);
    // FI: some or all of these tests could be placed in
    // dereferencing_to_points_to()
    if(!entity_stub_sink_p(v)
       && !formal_parameter_p(v)
       && !ENDP(sel)
       && pointer_type_p(t)) {
      expression e = entity_to_expression(v);
      pt_out = dereferencing_to_points_to(e, pt_in);
      free_expression(e);
    }
    pt_out = expressions_to_points_to(sel, pt_in, side_effect_p);
  }
  return pt_out;
}

pt_map range_to_points_to(range r, pt_map pt_in, bool side_effect_p)
{
  pt_map pt_out = pt_in;
  expression l = range_lower(r);
  expression u = range_upper(r);
  expression i = range_increment(r);
  pt_out = expression_to_points_to(l, pt_in, side_effect_p);
  pt_out = expression_to_points_to(u, pt_out, side_effect_p);
  pt_out = expression_to_points_to(i, pt_out, side_effect_p);
  return pt_out;
}

/* Three different kinds of calls are distinguished:
 *
 * - calls to constants, e.g. NULL,
 *
 * - calls to intrinsics, e.g. ++ or malloc(),
 *
 * - and calls to a user function.
 *
 * "el" is the effect list associated to the call site
 */
pt_map call_to_points_to(call c, pt_map pt_in, list el, bool side_effect_p)
{
  pt_map pt_out = pt_in;
  if(!points_to_graph_bottom(pt_in)) {
    tag tt;
    entity f = call_function(c);
    list al = call_arguments(c);
    type ft = entity_type(f);
    type rt = type_undefined;
    if(type_functional_p(ft)) {
      functional ff = type_functional(ft);
      rt = functional_result(ff);

      // FI: we might have to handle here return?, exit, abort, (stop)
      // if(ENTITY_STOP_P(e)||ENTITY_ABORT_SYSTEM_P(e)||ENTITY_EXIT_SYSTEM_P(e)
      // It is done somewhere else

      /* points-to updates due to arguments */
      // FI: this cannot be delayed but it is unfortunately applied
      // again when going down? See arithmetic08 and 09?
      // This is necessary but cannot be placed here because of the
      // recursive calls
      // FI: we are in trouble for post increment and post decrement...
      // We should update the target a second time in sinks.c!
      if(!ENTITY_CONDITIONAL_P(f)) {
	// FI: This is OK only if all subexpressions are always evaluated
	pt_out = expressions_to_points_to(al, pt_in, side_effect_p);
      }
      else
	pt_out = expression_to_points_to(EXPRESSION(CAR(al)), pt_in, side_effect_p);

      if(!points_to_graph_bottom(pt_out)) {
	switch( tt = value_tag(entity_initial(f))) {
	case is_value_code:{
	  pips_assert("f is a user-defined function", value_code_p(entity_initial(f)));
	  pt_out = user_call_to_points_to(c, pt_out, el);
	}
	  break;
	case is_value_unknown:
	  pips_internal_error("function %s has an unknown value\n",
			      entity_name(f));
	  break;
	case is_value_intrinsic:
	  pt_out = intrinsic_call_to_points_to(c, pt_in, side_effect_p);
	  break;
	case is_value_constant:
	  pt_out = pt_in; // FI?
	  break;
	case is_value_symbolic:{
	  value v = entity_initial(f);
	  symbolic s = value_symbolic(v);
	  expression ex = symbolic_expression(s);
	  pt_out = expression_to_points_to(ex, pt_in, side_effect_p);
	}
	  break;
	case is_value_expression:{
	  value v = entity_initial(f);
	  expression ex = value_expression(v);
	  pt_out = expression_to_points_to(ex, pt_in, side_effect_p);
	}
	  break;
	default:
	  pips_internal_error("unknown tag %d\n", tt);
	  break;
	}
      }
    }
    else if(type_variable_p(ft)) {
      /* Must be a pointer to a function */
      if(pointer_type_p(ft)) {
	/* I do not know if nft must be freed later */
	type nft = type_to_pointed_type(ft);
	pips_assert("Must be a function", type_functional_p(nft));
	functional nff = type_functional(nft);
	rt = functional_result(nff);
      }
      else
	pips_internal_error("Unexpected type.\n");
    }
    else if(type_void_p(rt))
      /* points-to updates due to arguments */
      pt_out = expressions_to_points_to(al, pt_in, side_effect_p);
    else
      pips_internal_error("Unexpected type.\n");
  }

  pips_assert("pt_out is consistent and defined",
	      points_to_graph_consistent_p(pt_out)
	      && !points_to_graph_undefined_p(pt_out));

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

pt_map intrinsic_call_to_points_to(call c, pt_map pt_in, bool side_effect_p)
{
  pt_map pt_out = pt_in;
  entity f = call_function(c);

  list al = call_arguments(c);

  //set_methods_for_proper_simple_effects();
  //list el = call_to_proper_effects(c);
  //generic_effects_reset_all_methods();

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
    pt_out = assignment_to_points_to(lhs, rhs, pt_out);
  }
  else if (ENTITY_FREE_SYSTEM_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    pt_out = freed_pointer_to_points_to(lhs, pt_out, side_effect_p);
  }
   // According to C standard, pointer arithmetics does not change
  // the targeted object.
  else if(ENTITY_PLUS_UPDATE_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    type lhst = expression_to_type(lhs);
    if(pointer_type_p(lhst)) {
      expression delta = EXPRESSION(CAR(CDR(al)));
      pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
    }
    free_type(lhst);
  }
  else if(ENTITY_MINUS_UPDATE_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    type lhst = expression_to_type(lhs);
    if(C_pointer_type_p(lhst)) {
      expression rhs = EXPRESSION(CAR(CDR(al)));
      entity um = FindOrCreateTopLevelEntity(UNARY_MINUS_OPERATOR_NAME);
      expression delta = MakeUnaryCall(um, copy_expression(rhs));
      pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
      free_expression(delta);
    }
    free_type(lhst);
  }
  else if(ENTITY_POST_INCREMENT_P(f) || ENTITY_PRE_INCREMENT_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    type lhst = expression_to_type(lhs);
    if(C_pointer_type_p(lhst) && side_effect_p) {
      expression delta = int_to_expression(1);
      pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
      free_expression(delta);
    }
    free_type(lhst);
  }
  else if(ENTITY_POST_DECREMENT_P(f) || ENTITY_PRE_DECREMENT_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    type lhst = expression_to_type(lhs);
    if(C_pointer_type_p(lhst)) {
      expression delta = int_to_expression(-1);
      pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
      free_expression(delta);
    }
    free_type(lhst);
  }
  else if(ENTITY_DEREFERENCING_P(f) || ENTITY_POINT_TO_P(f)) {
    /* Is the dereferenced pointer null or undefined? */
    expression p = EXPRESSION(CAR(al));
    pt_out = dereferencing_to_points_to(p, pt_out);
  }
  else if(ENTITY_PLUS_C_P(f) || ENTITY_MINUS_C_P(f)) {
    /* Is the dereferenced pointer null or undefined? */
    expression p1 = EXPRESSION(CAR(al));
    type t1 = expression_to_type(p1);
    if(pointer_type_p(t1))
      pt_out = dereferencing_to_points_to(p1, pt_out);
    else {
      expression p2 = EXPRESSION(CAR(CDR(al)));
      type t2 = expression_to_type(p2);
      if(pointer_type_p(t2))
	pt_out = dereferencing_to_points_to(p2, pt_out);
    }
  }
  else if(ENTITY_ASSERT_FAIL_SYSTEM_P(f)) {
    // FI: no return from assert failure
    clear_pt_map(pt_out);
    points_to_graph_bottom(pt_out) = true;
  }
  else if(ENTITY_STOP_P(f)||ENTITY_ABORT_SYSTEM_P(f)||ENTITY_EXIT_SYSTEM_P(f)
     /* || ENTITY_ASSERT_FAIL_SYSTEM_P(f) */) {
    clear_pt_map(pt_out);
    points_to_graph_bottom(pt_out) = true;
  }
  else if(ENTITY_PRINTF_P(f) || ENTITY_FPRINTF_P(f) || ENTITY_SPRINTF_P(f)
	  || ENTITY_SCANF_P(f) || ENTITY_FSCANF_P(f) || ENTITY_SSCANF_P(f)
	  || ENTITY_ISOC99_FSCANF_P(f)|| ENTITY_ISOC99_SSCANF_P(f)) {
    FOREACH(EXPRESSION, a, al) {
      type at = points_to_expression_to_concrete_type(a);
      if(C_pointer_type_p(at)) {
	// For the side-effects on pt_out
	list sinks = expression_to_points_to_sinks(a, pt_out);
	if(gen_length(sinks)==1 && nowhere_cell_p(CELL(CAR(sinks)))) {
	  /* attempt at using an undefined value */
	  pips_user_warning("Undefined value \"%s\" is used at line %d.\n",
			    expression_to_string(a),
			    points_to_context_statement_line_number());

	  clear_pt_map(pt_out);
	  points_to_graph_bottom(pt_out) = true;
	}
	gen_free_list(sinks);
      }
    }
    if(!points_to_graph_bottom(pt_out)
       && (ENTITY_FPRINTF_P(f) || ENTITY_FSCANF_P(f)
	   || ENTITY_ISOC99_FSCANF_P(f))) {
      /* stdin, stdout, stderr, fd... must be defined and not NULL */
      expression a1 = EXPRESSION(CAR(al));
      if(expression_reference_p(a1)) {
	expression d =
	  unary_intrinsic_expression(DEREFERENCING_OPERATOR_NAME,
					      copy_expression(a1));
	pt_out = expression_to_points_to(d, pt_out, false);
	free_expression(d);
      }
    }
  }
  else if(ENTITY_C_RETURN_P(f)) {
    /* it is assumed that only one return is present in any module
       code because internal returns are replaced by gotos */
    if(ENDP(al)) {
      // clear_pt_map(pt_out);
      ; // the necessary projections are performed elsewhere
    }
    else {
      expression rhs = EXPRESSION(CAR(al));
      type rhst = expression_to_type(rhs);
      // FI: should we use the type of the current module?
      if(pointer_type_p(rhst) || struct_type_p(rhst)) {
	entity rv = function_to_return_value(get_current_module_entity());
	expression lhs = entity_to_expression(rv);
	pt_out = assignment_to_points_to(lhs, rhs, pt_out);
      }
      free_type(rhst);
    }
  }
  else if(ENTITY_FCLOSE_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    // pt_out = freed_pointer_to_points_to(lhs, pt_out, side_effect_p);
    list lhc = expression_to_points_to_sources(lhs, pt_out);
    cell uc = make_nowhere_cell();
    list rhc = CONS(CELL, uc, NIL);
    pt_out = list_assignment_to_points_to(lhc, rhc, pt_out);
  }
  else if(ENTITY_CONDITIONAL_P(f)) {
    // FI: I needs this piece of code for assert();
    expression c = EXPRESSION(CAR(al));
    pt_map in_t = full_copy_pt_map(pt_out);
    pt_map in_f = full_copy_pt_map(pt_out);
    // FI: issue with the notion of pt_in
    // stubs created when computing in_t should also be added in in_f
    // or they are going to seem to be reallocated ambiguously in
    // create_stub_entity(). Same issue I guess for the "if" construct
    in_t = condition_to_points_to(c, in_t, true);
    in_f = condition_to_points_to(c, in_f, false);
    expression e1 = EXPRESSION(CAR(CDR(al)));
    expression e2 = EXPRESSION(CAR(CDR(CDR(al))));
    pt_map out_t = pt_map_undefined;

    if(!points_to_graph_bottom(in_t))
      out_t = expression_to_points_to(e1, in_t, side_effect_p);

    pt_map out_f = pt_map_undefined;
    // FI: should be factored out in a more general merge function...
    if(!points_to_graph_bottom(in_f))
      out_f = expression_to_points_to(e2, in_f, side_effect_p);

    if(points_to_graph_bottom(in_t))
      pt_out = out_f;
    else if(points_to_graph_bottom(in_f))
      pt_out = out_t;
    else
      pt_out = merge_points_to_graphs(out_t, out_f);
    // FI: this destroys pt_out for test case pointer02, Memory leaks...
    //free_pt_map(in_t), free_pt_map(in_f), free_pt_map(out_t), free_pt_map(out_f);
  }
  else if(ENTITY_FOPEN_P(f)) {
    expression e1 = EXPRESSION(CAR(al));
    pt_out = dereferencing_to_points_to(e1, pt_out);
    expression e2 = EXPRESSION(CAR(CDR(al)));
    pt_out = dereferencing_to_points_to(e2, pt_out);
  }
  else if(ENTITY_FDOPEN_P(f)) {
    expression e2 = EXPRESSION(CAR(CDR(al)));
    pt_out = dereferencing_to_points_to(e2, pt_out);
  }
  else if(ENTITY_FREOPEN_P(f)) {
    expression e1 = EXPRESSION(CAR(al));
    pt_out = dereferencing_to_points_to(e1, pt_out);
    expression e2 = EXPRESSION(CAR(CDR(al)));
    pt_out = dereferencing_to_points_to(e2, pt_out);
    expression e3 = EXPRESSION(CAR(CDR(CDR(al))));
    pt_out = dereferencing_to_points_to(e3, pt_out);
  }
  else if(ENTITY_FREAD_P(f) || ENTITY_FWRITE_P(f)) {
    expression e1 = EXPRESSION(CAR(al));
    pt_out = dereferencing_to_points_to(e1, pt_out);
    expression e4 = EXPRESSION(CAR(CDR(CDR(CDR(al)))));
    pt_out = dereferencing_to_points_to(e4, pt_out);
  }
  else if(ENTITY_CLEARERR_P(f) || ENTITY_FEOF_P(f)
	  || ENTITY_FERROR_P(f) || ENTITY_FILENO_P(f)) {
    expression e1 = EXPRESSION(CAR(al));
    pt_out = dereferencing_to_points_to(e1, pt_out);
  }
  else {
    // FI: fopen(), fclose() should be dealt with
    // fopen implies that its path argument is not NULL, just like a test
    // fclose implies that its fp argument is not NULL on input and
    // points to undefined on output.

    // Not safe till all previous tests are defined
    // It is assumed that other intrinsics do not generate points-to arcs...
    // pips_internal_error("Not implemented yet\n");
    pt_out = pt_in;
  }

  pips_assert("pt_out is consistent and defined",
	      points_to_graph_consistent_p(pt_out)
	      && !points_to_graph_undefined_p(pt_out));

  return pt_out;
}


/* Update the sink locations associated to the source "lhs" under
 * points-to information pt_map by "delta".
 *
 * C standard guarantees that the sink objects is unchanged by pointer
 * arithmetic.
 *
 * Property POINTS_TO_STRICT_POINTER_TYPES is used to be more or less
 * flexible about formal parameters and local variables such as "int *
 * p"
 */
pt_map pointer_arithmetic_to_points_to(expression lhs,
				       expression delta ,
				       pt_map pt_in)
{
  pt_map pt_out = pt_in;
  list sources = expression_to_points_to_sources(lhs, pt_out);
  bool to_be_freed;
  type et = points_to_expression_to_type(lhs, &to_be_freed);
  FOREACH(CELL, source, sources) {
    list sinks = source_to_sinks(source, pt_out, false);
    if(ENDP(sinks)) {
      entity v = reference_variable(cell_any_reference(source));
      //pips_internal_error("Sink missing for a source based on \"%s\".\n",
      //		  entity_user_name(v));
      pips_user_warning("No defined value for pointer \"%s\".\n",
			  entity_user_name(v));
      if(gen_length(sources)==1)
	// The code cannot be executed
	clear_pt_map(pt_out);
    }
    offset_cells(source, sinks, delta, et, pt_out);
    // FI: we could perform some filtering out of pt_in
    // If an arc points from source to nowehere/undefined or to the
    // null location, this arc should be removed from pt_in as it
    // cannot lead to an execution reaching the next statement.
    FOREACH(CELL, sink, sinks) { 
      if(nowhere_cell_p(sink))
	remove_points_to_arcs(source, sink, pt_out);
      else if(null_cell_p(sink))
	remove_points_to_arcs(source, sink, pt_out);
    }
  }
  // FI: should we free the sources list? Fully free it?
  return pt_out;
}

/* Side effect on reference "r".
 *
 * r is assumed to be a reference to an array.
 *
 * The offset is applied to the last suscript.
 */
void offset_array_reference(reference r, expression delta, type et)
{
  value v = EvalExpression(delta);
  list rsl = reference_indices(r);
  if(value_constant_p(v) && constant_int_p(value_constant(v))) {
    int dv =  constant_int(value_constant(v));
    if(ENDP(rsl)) {
      // FI: oops, we are in trouble; assume 0...
      expression se = int_to_expression(dv);
      reference_indices(r) = CONS(EXPRESSION, se, NIL);
    }
    else {
      // Select the index that should be subscripted
      list sl = points_to_reference_to_typed_index(r, et);
      expression lse = EXPRESSION(CAR(sl));
      value vlse = EvalExpression(lse);
      if(value_constant_p(vlse) && constant_int_p(value_constant(vlse))) {
	int ov =  constant_int(value_constant(vlse));
	int k = get_int_property("POINTS_TO_SUBSCRIPT_LIMIT");
	if(-k <= ov && ov <= k) {
	  expression nse = int_to_expression(dv+ov);
	  //EXPRESSION_(CAR(gen_last(sl))) = nse;
	  EXPRESSION_(CAR(sl)) = nse;
	}
	else {
	  expression nse = make_unbounded_expression();
	  //EXPRESSION_(CAR(gen_last(sl))) = nse;
	  EXPRESSION_(CAR(sl)) = nse;
	}
	free_expression(lse);
      }
      else {
	// FI: assume * is used... UNBOUNDED_DIMENSION
	expression nse = make_unbounded_expression();
	//EXPRESSION_(CAR(gen_last(sl))) = nse;
	EXPRESSION_(CAR(sl)) = nse;
	free_expression(lse);
      }
    }
  }
  else {
    if(ENDP(rsl)) {
      expression nse = make_unbounded_expression();
      reference_indices(r) = CONS(EXPRESSION, nse, NIL);
    }
    else {
      list sl = points_to_reference_to_typed_index(r, et);
      expression ose = EXPRESSION(CAR(sl));
      expression nse = make_unbounded_expression();
      EXPRESSION_(CAR(sl)) = nse;
      free_expression(ose);
    }
  }
}

/* Each cell in sinks is replaced by a cell located "delta" elements
 * further up in the memory. In some cases, the same points-to are
 * removed and added. For instance, t[0],t[1] -> t[1],t[2] because of
 * a p++, and t[1] is removed and added.
 *
 * This procedure must be used when cells in "sinks" are components of
 * points-to arcs stored in a points-to set.
 */
void offset_cells(cell source, list sinks, expression delta, type et, pt_map in)
{
  // FI: it would be easier to use two lists of arcs rather than sets.
  // FI: should we assert that expression delta!=0?
  pt_map old = new_pt_map();
  pt_map new = new_pt_map();
  FOREACH(CELL, sink, sinks) {
    if(!anywhere_cell_p(sink) && !cell_typed_anywhere_locations_p(sink)) {
      points_to pt = find_arc_in_points_to_set(source, sink, in);
      // FI: the arc may not be found; for instance, you know that
      // _pp_1[1] points towards *NULL_POINTER*, but this is due to an arc
      // _pp_1[*]->*NULL_POINTER*; this arc does not have to be updated
      if(!points_to_undefined_p(pt)) {
	add_arc_to_pt_map(pt, old);
	points_to npt = offset_cell(pt, delta, et);
	add_arc_to_pt_map(npt, new);
      }
      else {
	// Another option would be to generate nothing in this case
	// since it is taken care of by the lattice...

	// Since pt has not been found in "in", the approximation must be may
	pt = make_points_to(copy_cell(source), copy_cell(sink),
			    make_approximation_may(),
			    make_descriptor_none());
	points_to npt = offset_cell(pt, delta, et);
	add_arc_to_pt_map(npt, new);
      }
    }
  }
  difference_of_pt_maps(in, in, old);
  union_of_pt_maps(in, in, new);
}

/* Allocate and return a new points-to "npt", copy of "pt", with an
 * offset of "delta" on the sink. "et" is used to determine which
 * index should be offseted.
 *
 * Some kind of k-limiting should be performed here to avoid creating
 * too many new nodes in the points-to graph, such as t[0], t[1],... A
 * fix point t[*] should be used when too may nodes already exist.
 *
 * Since "sink" is used to compute the key in the hash table used to
 * represent set "in", it is not possible to perform a side effect on
 * "sink" without removing and reinserting the corresponding arc.
 *
 * FI: I am not sure we have the necessary information to know which
 * subscript must be updated when more than one is available. This is
 * bad for multidimensional arrays and worse for references to stub
 * that may include fields (or not) as subscript as well as lots of
 * articificial dimensions due to the source.
 *
 * I assumed gen_last() to start with, but it is unlikely in general!
 */
points_to offset_cell(points_to pt, expression delta, type et)
{
  /* "&a[i]" should be transformed into "&a[i+eval(delta)]" when
     "delta" can be statically evaluated */
  points_to npt = copy_points_to(pt);
  cell sink = points_to_sink(npt);
  reference r = cell_any_reference(sink);
  entity v = reference_variable(r);
  if(nowhere_cell_p(sink))
    ; // user error: possible incrementation of an uninitialized pointer
  else if(null_cell_p(sink))
    ; // Impossible: possible incrementation of a NULL pointer
  else if(anywhere_cell_p(sink))
    ; // It is already fuzzy no need to add more
  // FI: it might be necessary to exclude *HEAP* too when a minimal
  // heap model is used (ABSTRACT_HEAP_LOCATIONS = "unique")
  else {
    type vt = entity_basic_concrete_type(v);
    // FI: I do not understand why this is based on the type of v and
    // not on the ype of r
    if(array_type_p(vt)
       // FI: the property should have been taken care of earlier when
       // building sink
       /*|| !get_bool_property("POINTS_TO_STRICT_POINTER_TYPES")*/) {
      cell source = points_to_source(npt);
      bool to_be_freed;
      type source_t = points_to_cell_to_type(source, &to_be_freed);
      type c_source_t = compute_basic_concrete_type(source_t);
      if(to_be_freed) free_type(source_t);
      type pt = type_to_pointed_type(c_source_t);
      type e_sink_t = compute_basic_concrete_type(pt);
      if(array_pointer_type_equal_p(vt, e_sink_t)
	 && get_bool_property("POINTS_TO_STRICT_POINTER_TYPES"))
	pips_user_error("Use of pointer arithmetic on \"%s\" at line %d via reference \"%s\" is not "
			"standard-compliant.\n"
			"Reset property \"POINTS_TO_STRICT_POINTER_TYPES\""
			" for usual non-standard compliant C code.\n",
			entity_user_name(v),
			points_to_context_statement_line_number(),
			effect_reference_to_string(cell_any_reference(source)));
      else
	offset_array_reference(r, delta, et);
    }
    else if(struct_type_p(vt)) {
      // The struct may contain an array field.
      // FI: should we check the existence of the field in the subscripts?
      offset_array_reference(r, delta, et);
    }
    // FI to be extended to pointers and points-to stubs
    else {
      cell source = points_to_source(npt);
      pips_user_error("Use of pointer arithmetic on \"%s\" at line %d via reference \"%s\" is not "
		      "standard-compliant.\n"
		      "Reset property \"POINTS_TO_STRICT_POINTER_TYPES\""
		      " for usual non-standard compliant C code.\n",
		      entity_user_name(v),
		      points_to_context_statement_line_number(),
		      effect_reference_to_string(cell_any_reference(source)));
    }
  }
  return npt;
}

/* Each cell in sinks is replaced by a cell located "delta" elements
 * further up in the memory.
 *
 * Similar to offset_cells(), but, in spite of the name, cannot be
 * used with points-to cells that are part of a points-to belonging to
 * a points-to set.
 */
void offset_points_to_cells(list sinks, expression delta, type t)
{
  FOREACH(CELL, sink, sinks) {
    offset_points_to_cell(sink, delta, t, ENDP(CDR(sinks)));
  }
}

/* FI: offset_cell() has been derived from this function. Some
 * factoring out should be performed.
 *
 * The naming is all wrong: offset_points_to_cell() can operate on a
 * cell, while offset_cell() is designed to operate on a cell
 * component of a points-to.
 *
 * Type "t" is used to decide which subscript should be updated by delta.
 */
void offset_points_to_cell(cell sink,
			   expression delta,
			   type t,
			   bool unique_p __attribute__ ((__unused__)))
{
  /* "&a[i]" should be transformed into "&a[i+eval(delta)]" when
     "delta" can be statically evaluated */
  reference r = cell_any_reference(sink);
  entity rv = reference_variable(r);
  if(nowhere_cell_p(sink))
    ; // user error: possible incrementation of an uninitialized pointer
  else if(null_cell_p(sink))
    // FI: the operation requested is impossible; the condition should
    // be checked above to update the pt_map and/or to signal a bug
    ; // Impossible: possible incrementation of a NULL pointer
  else if(anywhere_cell_p(sink))
    ; // It is already fuzzy no need to add more
  // FI: it might be necessary to exclude *HEAP* too when a minimal
  // heap model is used (ABSTRACT_HEAP_LOCATIONS = "unique")
  // FI: this has been dealt with somewhere else
  // else if(entity_array_p(rv)
  //   || !get_bool_property("POINTS_TO_STRICT_POINTER_TYPES")) {
  else if(entity_array_p(rv) || cell_typed_anywhere_locations_p(sink)) {
    value val = EvalExpression(delta);
    list sl = reference_indices(r);
    if(value_constant_p(val) && constant_int_p(value_constant(val))) {
      int dv =  constant_int(value_constant(val));
      if(ENDP(sl)) {
	if(entity_array_p(rv)) {
	// FI: oops, we are in trouble; assume 0...
	expression se = int_to_expression(dv);
	reference_indices(r) = CONS(EXPRESSION, se, NIL);
	}
	else {
	  ; // FI: No need to add a zero subscript to a scalar variable
	}
      }
      else {
	// FI: this is wrong, there is no reason to update the last
	// subscript; the type of the offset should be passed as an
	// argument. See for instance dereferencing08.c. And
	// dereferencing18.c
	list tsl = find_points_to_subscript_for_type(sink, t);
	if(ENDP(tsl)) {
	  // dereferencing18: the only possible offset is 0, dv==0,
	  // Or the points-to information is approximate.
	  // unique_p should be used here to decide if a warning should
	  // be emitted
	  ;
	}
	else {
	  // expression lse = EXPRESSION(CAR(gen_last(sl)));
	  expression lse = EXPRESSION(CAR(tsl));
	  value vlse = EvalExpression(lse);
	  if(value_constant_p(vlse) && constant_int_p(value_constant(vlse))) {
	    int ov =  constant_int(value_constant(vlse));
	    int k = get_int_property("POINTS_TO_SUBSCRIPT_LIMIT");
	    if(-k <= ov && ov <= k) {
	      expression nse = int_to_expression(dv+ov);
	      //EXPRESSION_(CAR(gen_last(sl))) = nse;
	      EXPRESSION_(CAR(tsl)) = nse;
	    }
	    else {
	      expression nse = make_unbounded_expression();
	      //EXPRESSION_(CAR(gen_last(sl))) = nse;
	      EXPRESSION_(CAR(tsl)) = nse;
	    }
	    free_expression(lse);
	  }
	  else {
	    // If the index cannot be computed, used the unbounded expression
	    expression nse = make_unbounded_expression();
	    //EXPRESSION_(CAR(gen_last(sl))) = nse;
	    EXPRESSION_(CAR(tsl)) = nse;
	    free_expression(lse);
	  }
	}
      }
    }
    else {
      if(ENDP(sl)) {
	expression nse = make_unbounded_expression();
	reference_indices(r) = CONS(EXPRESSION, nse, NIL);
      }
      else {
	list tsl = find_points_to_subscript_for_type(sink, t);
	if(ENDP(tsl)) {
	  /* No subscript is possible, but 0 and it does not need to appear */
	  /* dereferencing18.c: a scalar was malloced */
	  if(zero_expression_p(delta))
	    ; // Nothing to be done: the subscript is ignored
	  else {
	    /* We might use unique_p and the value of delta to detect
	       an error or to warn the user about a possible error */
	    ;
	  }
	}
	else {
	//expression ose = EXPRESSION(CAR(gen_last(sl)));
	expression ose = EXPRESSION(CAR(tsl));
	expression nse = make_unbounded_expression();
	//EXPRESSION_(CAR(gen_last(sl))) = nse;
	EXPRESSION_(CAR(tsl)) = nse;
	free_expression(ose);
	}
      }
    }
  }
  // FI to be extended to pointers and points-to stubs
  else {
    if(zero_expression_p(delta)) {
      ;
    }
    else {
    pips_user_error("Use of pointer arithmetic on \"%s\" at line %d is not "
		    "standard-compliant.\n%s",
		    entity_user_name(rv),
		    points_to_context_statement_line_number(),
		    get_bool_property("POINTS_TO_STRICT_POINTER_TYPES")?
		    "Try resetting property \"POINTS_TO_STRICT_POINTER_TYPES\""
		    " for usual non-standard compliant C code.\n"
		    :"");
    }
  }
}


pt_map assignment_to_points_to(expression lhs, expression rhs, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  // FI: lhs and rhs have already been used to update pt_in
  //pt_map pt_out = expression_to_points_to(lhs, pt_in);
  /* It is not obvious that you are allowed to evaluate this before
     the sink of lhs, but the standard probably forbid stupid side
     effects. */
  //pt_out = expression_to_points_to(lhs, pt_out);
  bool to_be_freed = false;
  type t = points_to_expression_to_type(lhs, &to_be_freed);

  type ut = compute_basic_concrete_type(t);
  if(pointer_type_p(ut))
    pt_out = pointer_assignment_to_points_to(lhs, rhs, pt_out);
  else if(struct_type_p(ut))
    pt_out = struct_assignment_to_points_to(lhs, rhs, pt_out);
  // FI: unions are not dealt with...
  else if(array_of_pointers_type_p(ut)) {
    /* Can occur in a declaration */
    /* When more precision is needed, the BRACE_INTRINSIC arguments
       will have to be analyzed... */
    pips_assert("lhs is a reference", expression_reference_p(lhs));
    reference r = expression_reference(lhs);
    list sl = reference_indices(r);
    pips_assert("The array reference has no indices", ENDP(sl));
    cell source = make_cell_reference(copy_reference(r));
    points_to_cell_add_unbounded_subscripts(source);
    type pt = basic_pointer(variable_basic(type_variable(ut)));
    cell sink = make_anywhere_points_to_cell(pt);
    points_to a = make_points_to(source, sink, 
				  make_approximation_may(),
				  make_descriptor_none());
    pt_out = add_arc_to_pt_map(a, pt_in);
  }
  else if(array_of_struct_type_p(ut)) {
    pips_internal_error("Initialization of array of structs not implemented yet.\n");
  }
  else
    pt_out = pt_in; // What else?

  if(to_be_freed)
    free_type(t);

  return pt_out;
}

/* Check that all cells in list "sinks" are compatible with type "ct"
 * if "eval_p" is false, and with the type pointed by "st" if eval_p is
 * true.
 */
void check_type_of_points_to_cells(list sinks, type ct, bool eval_p)
{
  type st = type_undefined;

  if(!ENDP(sinks)) {
    if(eval_p) {
      if(pointer_type_p(ct))
	st = copy_type(type_to_pointed_type(ct));
      else if(array_type_p(ct)) {
	st = array_type_to_sub_array_type(ct);
      }
      else
	pips_internal_error("Unexpected \"ct\" argument.\n");
    }
    else
      st = copy_type(ct);
  }

  FOREACH(CELL, c, sinks) {
    if(!null_cell_p(c)
       && !anywhere_cell_p(c)
       && !cell_typed_anywhere_locations_p(c)
       && !nowhere_cell_p(c)) {
      bool to_be_freed;
      type est = points_to_cell_to_type(c, &to_be_freed);
      type cest = compute_basic_concrete_type(est);
      if(!array_pointer_type_equal_p(cest, st)
	 /* Adding the zero subscripts may muddle the type issue
	    because "&a[0]" has not the same type as "a" although we
	    normalize every cell into "a[0]". */
	 && !(array_type_p(st)
	      && array_pointer_type_equal_p(cest,
					    array_type_to_element_type(st)))
	 /* Take care of the constant strings like "hello" */
	 && !(char_type_p(st) && char_star_constant_function_type_p(cest))
	 /* Take care of void */
	 && !type_void_p(st) // cest could be checked as overloaded
	 && !overloaded_type_p(cest)
	 ) {
	pips_user_warning("Maybe an issue with a dynamic memory allocation.\n");
	pips_user_error("At line %d, "
			// "the type returned for the value of expression \"%s\"="
			"the type returned for the cell"
			"\"%s\", "
			"\"%s\", is not the expected type, \"%s\".\n",
			points_to_context_statement_line_number(),
			// expression_to_string(rhs),
			effect_reference_to_string(cell_any_reference(c)),
			// copied from ri-util/type.c, print_type()
			words_to_string(words_type(cest, NIL, false)),
			words_to_string(words_type(st, NIL, false)));
      }
    }
  }
}

/* Check that the cells in list "sinks" have types compatible with the
 * expression on the left-hand side, lhs.
 *
 * List "sinks" is assumed to have been derived from the "rhs" expression.
 */
void check_rhs_value_types(expression lhs,
			   // Was used in the error message..
			   expression rhs __attribute__ ((unused)),
			   list sinks)
{
  // Some expression are synthesized to reuse existing functions.
  bool to_be_freed;
  type t = points_to_expression_to_type(lhs, &to_be_freed);
  type ct = compute_basic_concrete_type(t);
  type st = type_undefined; // sink type
  if(pointer_type_p(ct)) {
    st = compute_basic_concrete_type(type_to_pointed_type(ct));
  }
  else if(array_type_p(ct)) {
    st = compute_basic_concrete_type(array_type_to_element_type(ct));
  }
  else if(scalar_integer_type_p(ct)) {
    /* At least for the NULL pointer... */
    st = ct;
  }
  else
    pips_internal_error("Unexpected type for value.\n");

  if(!type_void_star_p(ct)) { // void * is compatible with all types...
    check_type_of_points_to_cells(sinks, st, false);
  }
  // Late free, to be able to see "t" under gdb...
  if(to_be_freed) free_type(t);
}

/* Any abstract location of the lhs in L is going to point to any sink of
 * any abstract location of the rhs in R.
 *
 * New points-to information must be added when a formal parameter
 * is dereferenced.
 */
pt_map internal_pointer_assignment_to_points_to(expression lhs,
						expression rhs,
						pt_map pt_in)
{
  pt_map pt_out = pt_in;

  pips_assert("pt_out is consistent on entry",
	      consistent_points_to_graph_p(pt_out));

  list L = expression_to_points_to_sources(lhs, pt_out);
  /* Beware of points-to cell lattice: e.g. add a[*] to a[1] */
  /* This is not the right place to take the lattice into account. As
     a result, a[1] woud not kill a[1] anymore. The problem must be
     taken care of by the equations used in
     list_assignment_to_points_to(). */
  // L = points_to_cells_to_upper_bound_points_to_cells(L);

  pips_assert("pt_out is consistent after computing L",
	      consistent_points_to_graph_p(pt_out));

  /* Make sure all cells in L are pointers: l may be an array of pointers */
  /* FI: I am not sure it is useful here because the conversion to an
     array due to property POINTS_TO_STRICT_POINTER_TYPES may not have
     occured yet */
  FOREACH(CELL, l, L) {
    bool to_be_freed;
    type lt = points_to_cell_to_type(l, &to_be_freed);
    if(array_type_p(lt)) {
      cell nl = copy_cell(l);
      // For Pointers/properties04.c, you want a zero subscript for
      // the lhs
      points_to_cell_add_zero_subscripts(l);
      // FI: since it is an array, most of the pointers will be unchanged
      // FI: this should be useless, but it has an impact because
      // points-to stubs are computed on demand; see Pointers/assignment12.c
      points_to_cell_add_unbounded_subscripts(nl);
      list os = source_to_sinks(nl, pt_out, true);
      list nll = CONS(CELL, nl, NIL);
      pt_out = list_assignment_to_points_to(nll, os, pt_out);
      gen_free_list(nll);
    }
    if(to_be_freed) free_type(lt);
  }

  pips_assert("pt_out is consistent after cells are dangerously updated",
	      consistent_points_to_graph_p(pt_out));

  /* Retrieve the memory locations that might be reached by the rhs
   *
   * Update the real "pt_in", the calling context, and "pt_out" by
   * adding new stubs linked directly or indirectly to the formal
   * parameters and global variables if necessary.
   */
  list R = expression_to_points_to_sinks(rhs, pt_out);

  check_rhs_value_types(lhs, rhs, R);

  pips_assert("pt_out is consistent after computing R",
	      consistent_points_to_graph_p(pt_out));

  if(ENDP(L) || ENDP(R)) {
    //pips_assert("Left hand side reference list is not empty.\n", !ENDP(L));
    //pips_assert("Right hand side reference list is not empty.\n", !ENDP(R));

  // FI: where do we want to check for dereferencement of
  // nowhere/undefined and NULL? Here? Or within
  // list_assignment_to_points_to?

    /* We must be in a dead-code portion. If not pleased, adjust properties... */
    if(ENDP(L)) {
      if(statement_points_to_context_defined_p())
	pips_user_warning("Expression \"%s\" could not be dereferenced at line %d.\n",
			  expression_to_string(lhs),
			  points_to_context_statement_line_number());
      else
	pips_user_warning("Expression \"%s\" could not be dereferenced.\n",
			  expression_to_string(lhs));
    }
    if(ENDP(R)) {
      if(statement_points_to_context_defined_p())
	pips_user_warning("Expression \"%s\" could not be dereferenced at line %d.\n",
			  expression_to_string(rhs),
			  points_to_context_statement_line_number());
      else
	pips_user_warning("Expression \"%s\" could not be dereferenced.\n",
			  expression_to_string(rhs));
    }
    clear_pt_map(pt_out);
    points_to_graph_bottom(pt_out) = true;
  }
  else {
    // We are in trouble if L=={null} or R=={nowhere)...
    // We are not sure what to do if null in L or if nowhere in R
    int nR = (int) gen_length(R);
    if(nR==1 && nowhere_cell_p(CELL(CAR(R)))) {
      if(statement_points_to_context_defined_p())
	pips_user_warning("Assignment of an undefined value to \"%s\" at line %d.\n",
			  expression_to_string(lhs),
			  points_to_context_statement_line_number());
      else
	pips_user_warning("Assignment of an undefined value to \"%s\".\n",
			  expression_to_string(lhs));
      /* The C99 standard does not preclude the propagation of
	 indeterminate values. It is often used in our test cases when
	 structs are assigned.

	 clear_pt_map(pt_out);
	 points_to_graph_bottom(pt_out) = true;
      */
      pt_out = list_assignment_to_points_to(L, R, pt_out);
    }
    else
      pt_out = list_assignment_to_points_to(L, R, pt_out);
  }

  // FI: memory leak(s)?

  pips_assert("pt_out is consistent", consistent_points_to_graph_p(pt_out));

  return pt_out;
}

pt_map pointer_assignment_to_points_to(expression lhs,
				       expression rhs,
				       pt_map pt_in)
{
  /* FI: this is a crazy idea to avoid problems in balancing test
   * branches. It should only be useful when the formal context has to
   * be expanded by this assignment. lhs = lhs;
   *
   * Of course, it is a catastrophy when expression lhs has side effects...
   *
   * And it does not work because the current "in" of the test is
   * modified by side-effect no seen when the false branch is analyzed.
   */
  // pt_map pt_out = internal_pointer_assignment_to_points_to(lhs, lhs, pt_in);
  pt_map pt_out = internal_pointer_assignment_to_points_to(lhs, rhs, pt_in);
  return pt_out;
}

/* Remove from points-to cell list R cells that certainly cannot be
   freed.  */
list freeable_points_to_cells(list R)
{
  list nhl = NIL; // No heap list: cannot be freed
  FOREACH(CELL, c, R) {
    if(heap_cell_p(c) || stub_points_to_cell_p(c)) {
      reference r = cell_any_reference(c);
      list inds = reference_indices(r);
      /* if c is a heap location with indices other than zero then we
	 have bumped into a non-legal free */
      if(!ENDP(inds)) {
	expression ind = EXPRESSION (CAR(inds));
	// No offset allowed for a free()
	if(!expression_null_p(ind))
	  nhl = CONS(CELL, c, nhl);
      }
      // gen_free_list(inds);
    }
    else if(!heap_cell_p(c)
	    && !stub_points_to_cell_p(c)
	    && !cell_typed_anywhere_locations_p(c))
      nhl = CONS(CELL, c, nhl);
  }
  gen_list_and_not(&R, nhl);
  gen_free_list(nhl);
  return R;
}

/* Error detections on "L" and "R" have already been performed. R only
 * contains heap locations or stub locations, i.e. potential heap
 * locations. Neither L nor R are empty. "lhs" is provided for other
 * error messages.
 */
pt_map freed_list_to_points_to(expression lhs, list L, list R, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  list ML = NIL; /* First level memory leaks */

  pips_assert("L is not empty", !ENDP(L));
  pips_assert("R is not empty", !ENDP(R));

  /* Build a nowhere cell - Should we check a property to type it or not? */
  //list N = CONS(CELL, make_nowhere_cell(), NIL);
  type t = points_to_expression_to_concrete_type(lhs);
  type pt = compute_basic_concrete_type(type_to_pointed_type(t));
  list N = CONS(CELL, make_typed_nowhere_cell(pt), NIL);

  /* Remove Kill_1 if it is not empty by definition */
  if(gen_length(L)==1 && atomic_points_to_cell_p(CELL(CAR(L)))) {
    set pt_out_s = points_to_graph_set(pt_out);
    SET_FOREACH(points_to, pts, pt_out_s) {
      cell l = points_to_source(pts);
      // FI: use the CP lattice and its operators instead?
      //if(related_points_to_cell_in_list_p(l, L)) {
      if(points_to_cell_in_list_p(l, L)) {
	// FI: assuming you can perform the removal inside the loop...
	remove_arc_from_pt_map(pts, pt_out);
      }
    }
  }

  /* Memory leak detection... Must be computed early, before pt_out
     has been (too?) modified. Transitive closure not performed... If
     one cell has been freed and if it has become unreachable, then
     arcs starting from it have become useless and other cells that
     were pointed by it may also have become unreachable. */
  if(gen_length(R)==1 && unreachable_points_to_cell_p(CELL(CAR(R)),pt_out)) {
    cell c = CELL(CAR(R));
    type ct = points_to_cell_to_concrete_type(c);
    if(pointer_type_p(ct) || struct_type_p(ct)
       || array_of_pointers_type_p(ct)
       || array_of_struct_type_p(ct)) {
      // FI: this might not work for arrays of pointers?
      // Many forms of "source" can be developped when we are dealing
      // struct and arrays
      // FI: do we need a specific version of source_to_sinks()?
      entity v = reference_variable(cell_any_reference(c));
      //cell nc = make_cell_reference(make_reference(v, NIL));
      list PML = variable_to_sinks(v, pt_out, true);
      FOREACH(CELL, m, PML) {
	if(heap_cell_p(m)) {
	  entity b = reference_variable(cell_any_reference(m));
	  pips_user_warning("Memory leak for bucket \"%s\".\n",
			    entity_name(b));
	  ML = CONS(CELL, m, ML);
	}
      }
    }
  }

  /*  Add Gen_2, which is not conditionned by gen_length(R) or atomicity. */
  set pt_out_s = points_to_graph_set(pt_out);
  SET_FOREACH(points_to, pts, pt_out_s) {
    cell r = points_to_sink(pts);
    if(points_to_cell_in_list_p(r, R)) {
      if(!null_cell_p(r) && !anywhere_cell_p(r) && !nowhere_cell_p(r)) {
	/* Compute and add Gen_2 */
	cell source = points_to_source(pts);
	// FI: it should be a make_typed_nowhere_cell()
	type t = points_to_cell_to_concrete_type(source);
	type pt = compute_basic_concrete_type(type_to_pointed_type(t));
	cell sink = make_typed_nowhere_cell(pt);
	// FI: why may? because heap cells are always abstract locations
	//approximation a = make_approximation_may();
	approximation a = copy_approximation(points_to_approximation(pts));
	points_to npt = make_points_to(copy_cell(source), sink, a,
				       make_descriptor_none());
	add_arc_to_pt_map(npt, pt_out);
	/* Do not notify the user that the source of the new
	   nowhere points to relation is a dangling pointer
	   because it is only a may information. Exact alias
	   information or a more precise heap model would be
	   necessary to have exact information about dangling
	   pointers. */
	if(stub_points_to_cell_p(r)) {
	  entity b = reference_variable(cell_any_reference(source));
	  pips_user_warning("Dangling pointer \"%s\" has been detected at line %d.\n",
			    entity_user_name(b),
			    points_to_context_statement_line_number());
	}
      }
    }
  }


  /* Remove Kill_2 if it is not empty by definition... which it is if
     heap(r) is true, but not if stub(r). Has to be done after Gen_2,
     or modification of pt_out should be delayed, which would avoid
     the internal modification of pt_out_s and make the code easier to
     understand... */
  if(gen_length(R)==1 && generic_atomic_points_to_cell_p(CELL(CAR(R)), false)) {
    set pt_out_s = points_to_graph_set(pt_out);
    cell r = CELL(CAR(R));
    if(!null_cell_p(r) && !anywhere_cell_p(r) && !nowhere_cell_p(r)) {
      SET_FOREACH(points_to, pts, pt_out_s) {
	cell s = points_to_sink(pts);
	if(points_to_cell_equal_p(r, s)) {
	  // FI: pv_whileloop05, lots of related cells to remove after a free...
	  // FI: assuming you can perform the removal inside the loop...
	  remove_arc_from_pt_map(pts, pt_out);
	}
      }
    }
  }

  /* Remove Kill_3 if it is not empty by definition: with the
     current heap model, it is always empty. Unreachable cells are
     dealt somewhere else. They can be tested with
     points_to_sink_to_sources().

     FI: Must be placed after gen_1 in case the assignment makes the cell
     reachable? Nonsense?
 */
  if(gen_length(R)==1
     && (atomic_points_to_cell_p(CELL(CAR(R))) 
	 || unreachable_points_to_cell_p(CELL(CAR(R)), pt_out))) {
    cell c = CELL(CAR(R));
    list S = points_to_sink_to_sources(c, pt_out, false);
    if(ENDP(S) || atomic_points_to_cell_p(c)) {
      set pt_out_s = points_to_graph_set(pt_out);
      SET_FOREACH(points_to, pts, pt_out_s) {
	cell l = points_to_source(pts);
	if(related_points_to_cell_in_list_p(l, R)) {
	  // Potentially memory leaked cell:
	  cell r = points_to_sink(pts);
	  pt_out = memory_leak_to_more_memory_leaks(r, pt_out);
	  remove_arc_from_pt_map(pts, pt_out);
	}
      }
    }
    else
      gen_free_list(S);
  }

  /* Add Gen_1 - Not too late since pt_out has aready been modified? */
  pt_out = list_assignment_to_points_to(L, N, pt_out);

  /* Add Gen_2: useless, already performed by Kill_2 */

  /*
   * Other pointers may or must now be dangling because their target
   * has been freed. Already detected at the level of Gen_2.
   */

  /* More memory leaks? */
  FOREACH(CELL, m, ML)
    pt_out = memory_leak_to_more_memory_leaks(m, pt_out);
  gen_free_list(ML);

  /* Clean up the resulting graph */
  // pt_out = remove_unreachable_heap_vertices_in_points_to_graph(pt_out, true);
  return pt_out;
}


  /* Any abstract location of the lhs in L is going to point to nowhere, maybe.
   *
   * Any source in pt_in pointing towards any location in lhs may or
   * Must now points towards nowhere (malloc07).
   *
   * New points-to information must be added when a formal parameter
   * is dereferenced.
   *
   * Equations for "free(e);":
   *
   * Let L = expression_to_sources(e,in) 
   *
   * and R_1 = (expression_to_sinks(e,in) ^ (HEAP U STUBS)) - {undefined}
   *
   * where ^ denotes the set intersection.
   *
   * If R_1 is empty, an execution error must occur.
   *
   * If R_1={NULL}, nothing happens. Let R=R_1-{NULL}.
   *
   * Any location "l" corresponding to "e" can now point to nowhere/undefined:
   *
   * Gen_1 = {pts=(l,nowhere,a) | l in L}
   *
   * Any location source that pointed to a location r in the heap,
   * pointed to by some l by definition, can now point to
   * nowhere/undefined also:
   *
   * Gen_2 = {pts=(source,nowhere,a) | exists r in R
   *                                   && r in Heap or Stubs
   *                                   && exists pts'=(source,r,a') in in}
   *
   * If e corresponds to a unique (non-abstract?) location l, any arc
   * starting from l can be removed:
   *
   * Kill_1 = {pts=(l,r,a) in in | l in L && |L|=1 && atomic(l)}
   *
   * If the freed location r is precisely known, any arc pointing
   * towards it can be removed:
   *
   * Kill_2 = {pts=(l,r,a) in in | r in R && |R|=1 && atomic(r)}
   *
   * If the freed location r is precisely known or if it can no longer
   * be reached, any arc pointing from it can be removed:
   *
   * Kill_3 = {pts=(r,d,a) in in | r in R && |R|=1 && (atomic(r)
   *                                                   v unreachable_p(r, in)}
   *
   * Then the set PML = { d | heap(d) ^ (r,d,a) in Kill_3} becomes a set
   * of potential meory leaks. They are leaked if they become
   * unreachable in the new points-to relation out (see below) and they
   * generate recursively new potential memory leaks and an updated
   * points-to relation.
   *
   * Since our current heap model only use abstract locations and since
   * we do not keep any alias information, the predicate atomic should
   * always return false and the sets Kill_2 and Kill_3 should always be
   * empty, except if a stub is freed: locally the stub is atomic.
   *
   * So, after the execution of "free(e);", the points-to relation is:
   *
   * out = (in - Kill_1 - Kill_2 - Kill_3) U Gen_1 U Gen_2
   *
   * Warnings for potential dangling pointers:
   *
   * DP = {r|exists pts=(r,l,a) in Gen_2} // To be checked
   *
   * No warning is issued as those are only potential.
   *
   * Memory leaks: the freed bucket may be the only bucket containing
   * pointers towards another bucket:
   *
   * PML = {source_to_sinks(r)|r in R}
   * ML = {m|m in PML && heap_cell_p(m) && !reachable(m, out)}
   *
   * Note: for DP and ML, we could compute may and must/exact sets. We only
   * compute must/exact sets to avoid swamping the log file with false alarms.
   *
   * FI->FC: it might be better to split the problem according to
   * |R|. If |R|=1, you know which bucket is destroyed, unless it is an
   * abstract bucket... which we do not really know even at the
   * intraprocedural level.  In that case, you could remove all edges
   * starting from r and then substitute r by nowhere/undefined.
   *
   * If |R| is greater than 1, then a new node nowhere/undefined must be
   * added and any arc ending up in R must be duplicated with a similar
   * arc ending up in the new node.
   *
   * The cardinal of |L| does not seem to have an impact: it does, see Kill_1
   */
pt_map freed_pointer_to_points_to(expression lhs, pt_map pt_in, bool side_effect_p)
{
  // FI: is this redundant with processing already performed by callers?
  // A test case is needed...
  pt_map pt_out = expression_to_points_to(lhs, pt_in, side_effect_p);
  list PML = NIL;

  list R_1 = expression_to_points_to_sinks(lhs, pt_out);
  list R = NIL;
  /* Remove cells from R_1 that do not belong to Heap: they cannot be
     freed */
  FOREACH(CELL,r, R_1) {
    if(heap_cell_p(r)
       || stub_points_to_cell_p(r)
       || anywhere_cell_p(r) || cell_typed_anywhere_locations_p(r)
       || null_cell_p(r))
      R = CONS(CELL, r, R);
  }
  gen_free_list(R_1);

  if(ENDP(R)) {
    /* A execution error is certain */
    pips_user_warning("Expression \"%s\" at line %d cannot be freed.\n",
		      expression_to_string(lhs),
		      points_to_context_statement_line_number());
    clear_pt_map(pt_out);
    points_to_graph_bottom(pt_out) = true;
  }
  else if(gen_length(R)==1 && null_cell_p(CELL(CAR(R)))) {
    /* Free(NULL) has no effect*/
    ;
  }
  else {
    /* Remove from R locations that cannot be freed */
    R = freeable_points_to_cells(R);

    if(ENDP(R)) {
      /* We have bumped into a non-legal free such as free(p[10]). See test
	 case malloc10.c */
      pips_user_warning("Free of a non-heap allocated address pointed "
			"by \"%s\" at line %d.\n"
			"Or bug in the points-to analysis...\n",
			expression_to_string(lhs),
			points_to_context_statement_line_number());
      clear_pt_map(pt_out);
      points_to_graph_bottom(pt_out) = true;
    }
    else {
      list L = expression_to_points_to_sources(lhs, pt_out);
      pips_assert("L is not empty", !ENDP(L));
      pt_out = freed_list_to_points_to(lhs, L, R, pt_in);
      gen_free_list(L);
    }
  }

  // FI: memory leak(s) in this function?
  //gen_free_list(N);
  gen_full_free_list(R);
  gen_free_list(PML);

  return pt_out;
}

/* Remove last subscripts of cell c till its type becomes a scalar
 * pointer.
 *
 * This of course may fail.
 */
cell reduce_cell_to_pointer_type(cell c)
{
  bool to_be_freed;
  type t = points_to_cell_to_type(c, &to_be_freed);
  reference r = cell_any_reference(c);
  list sl = reference_indices(r);
  bool remove_p = !pointer_type_p(t);
  if(to_be_freed) free_type(t);
  while(remove_p) {
    if(!ENDP(sl)) {
      /* Remove the last subscript, hopefully 0 */
      list ls = gen_last(sl); // last subscript
      expression lse = EXPRESSION(CAR(ls));
      if(field_reference_expression_p(lse))
	break; // This subscript cannot be removed
      gen_list_and_not(&sl, ls);
      reference_indices(r) = sl;
      // because sl is a sublist of ls, the chunk has already been freed
      // gen_full_free_list(ls);
      // gen_free_list(ls);
      t = points_to_cell_to_type(c, &to_be_freed);
      // remove_p = !pointer_type_p(t); we may end up with char[] instead of char*
      remove_p = !C_pointer_type_p(t);
      if(to_be_freed) free_type(t);
    }
    else
      break; // FI: in fact, an internal error
  }
  return c;
}

/* Undo the extra eval performed when stubs are generated: 0
 * subscripts are added when arrays are involved.
 *
 */
list reduce_cells_to_pointer_type(list cl)
{
  FOREACH(CELL, c, cl) {
    if(null_cell_p(c)) // There may be other exceptions...
      ;
    else
      (void) reduce_cell_to_pointer_type(c);
  }
  return cl;
}

/* Returns a list of cells of pointer type which are included in cell
 * "c". Useful when "c" is a struct or an array of structs or
 * pointers. Returns a list with cell "c" if it denotes a pointer.
 */
list points_to_cell_to_pointer_cells(cell c)
{
  list pcl = NIL; // pointer cell list
  type c_t = points_to_cell_to_concrete_type(c);
  if(pointer_type_p(c_t)) {
    cell nc = copy_cell(c);
    pcl = CONS(CELL, nc, pcl);
  }
  else if(struct_type_p(c_t)) {
    /* Look for pointer and struct and array of pointers or struct fields */
    list fl = struct_type_to_fields(c_t);
    FOREACH(ENTITY, f, fl) {
      type f_t = entity_basic_concrete_type(f);
      if(pointer_type_p(f_t) || struct_type_p(f_t)
	 || array_of_pointers_type_p(f_t)
	 || array_of_struct_type_p(f_t)) {
	cell nc = copy_cell(c);
	points_to_cell_add_field_dimension(nc, f);
	list ppcl = points_to_cell_to_pointer_cells(nc);
	pcl = gen_nconc(pcl, ppcl);
      }
    }
  }
  else if(array_of_pointers_type_p(c_t) || array_of_struct_type_p(c_t)) {
    /* transformer */
    cell nc = copy_cell(c);
    points_to_cell_add_unbounded_subscripts(nc);
    pcl = points_to_cell_to_pointer_cells(nc);
  }
  return pcl;
}

/* Cell "l" has been memory leaked for sure and is not referenced any
   more in "in". Its successors may be leaked too. */
pt_map memory_leak_to_more_memory_leaks(cell l, pt_map in)
{
  pt_map out = in;
  // potential memory leaks
  list pml = points_to_cell_to_pointer_cells(l);
  FOREACH(CELL, c, pml) {
    // This first test is probably useless because if has been
    // partially or fully performed by the caller
    if(heap_cell_p(c) && unreachable_points_to_cell_p(c, in)) {
      /* Remove useless unreachable arcs */
      list dl = NIL, npml = NIL;
      set out_s = points_to_graph_set(out);
      SET_FOREACH(points_to, pt, out_s) {
	cell source = points_to_source(pt);
	// FI: a weaker test based on the lattice is needed
	if(points_to_cell_equal_p(source, c)) {
	  dl = CONS(POINTS_TO, pt, dl);
	  cell sink = points_to_sink(pt);
	  npml = CONS(CELL, sink, npml);
	  // FI: we need to remove pt before we can test for unreachability...
	  /*
	    if(heap_cell_p(sink) && unreachable_points_to_cell_p(sink, out)) {
	    pips_user_warning("Heap bucket \"%s\" leaked at line %d.\n",
	    points_to_cell_to_string(sink),
	    points_to_context_statement_line_number());
	  */
	}
      }
      FOREACH(POINTS_TO, d, dl)
	remove_arc_from_pt_map(d, out);
      gen_free_list(dl);

      FOREACH(CELL, sink, npml) {
	if(heap_cell_p(sink) && unreachable_points_to_cell_p(sink, out)) {
	  if(false)
	    pips_user_warning("Heap bucket \"%s\" leaked.\n",
			      points_to_cell_to_string(sink));
	    else
	      pips_user_warning("Heap bucket \"%s\" leaked at line %d.\n",
				points_to_cell_to_string(sink),
				points_to_context_statement_line_number());
	  /* Look for a chain of memory leaks */
	  //if(!points_to_cell_equal_p(c, l))
	  out = memory_leak_to_more_memory_leaks(sink, out);
	}
      }
      gen_free_list(npml);
    }
  }
  return out;
}


/* Update "pt_out" when any element of L can be assigned any element of R
 *
 * FI->AM: Potential and sure memory leaks are not (yet) detected.
 *
 ******************************************
 *
 * FI->AM: the distinction between may and must sets used in the
 * implementation seem useless.
 *
 * Old description by Amira:
 *
 * KILL_MAY = kill_may_set()
 * KILL_MUST= kill_must_set()
 *
 * GEN_MAY = gen_may_set()
 * GEN_MUST= gen_must_set()
 *
 * KILL = KILL_MAY U KILL_MUST
 * GEN = GEN_MAY U GEN_MUST
 * PT_OUT = (PT_OUT - KILL) U GEN
 *
 ****************************************** 
 *
 * This function is used to model a C pointer assignment "e1 = e2;"
 *
 * Let L = expression_to_sources(e1) and R = expression_to_sinks(e2).
 *
 * Let "in" be the points-to relation before executing the assignment.
 *
 * Gen(L,R) = {pts| exist l in L exists r in R s.t. pts=(l,r,|L|=1)
 *
 * Kill(L,in) = {pts=(l,sink,must)| l in L}
 *
 * Let K=Kill(L,in) and out = (in-K) U gen(L,R)
 *
 * For memory leaks, let
 *
 *  ML(K,out) = {c in Heap | exists pts=(l,c,a) in K
 *                           && !(exists pts'=(l',c,a') in out)}
 *
 * For error dereferencing, such as nowhere/undefined and NULL, check
 * the content of L.
 *
 * This function is described in Amira Mensi's dissertation.
 *
 * Test cases designed to check the behavior of this function: ?!?
 */
pt_map list_assignment_to_points_to(list L, list R, pt_map pt_out)
{
  pips_assert("This function is not called with a bottom points-to",
	      !points_to_graph_bottom(pt_out));

  pips_assert("pt_out is consistent on entry",
	      consistent_points_to_graph_p(pt_out));

  /* Check possible dereferencing errors */
  list ndl = NIL; // null dereferencing error list
  list udl = NIL; // undefined dereferencing error list
  // FI->AM: you have no way to know if stubs are atomic or not...
  // I am not sure the atomic predicate takes this into account
  // but it does not really matter intraprocedurally: stubs are atomic
  bool singleton_p = (gen_length(L)==1
		      && generic_atomic_points_to_cell_p(CELL(CAR(L)), false));
  FOREACH(CELL, c, L) {
    if(nowhere_cell_p(c)){
      udl = CONS(CELL, c, udl);
      if(singleton_p)
	// Not necessarily a user error if the code is dead
	// Should be controlled by an extra property...
	pips_user_warning("Dereferencing of an undefined pointer \"%s\" at line %d.\n",
			  effect_reference_to_string(cell_any_reference(c)),
			  points_to_context_statement_line_number());
      else
	pips_user_warning("Possible dereferencing of an undefined pointer.\n");
    }
    else if(null_cell_p(c)) {
      ndl = CONS(CELL, c, ndl);
      if(singleton_p)
	// Not necessarily a user error if the code is dead
	// Should be controlled by an extra property...
	pips_user_warning("Dereferencing of a null pointer \"%s\" at line %d.\n",
			  effect_reference_to_string(cell_any_reference(c)),
			  points_to_context_statement_line_number());
      else
	pips_user_warning("Possible dereferencing of a null pointer \"%s\" at line %d.\n",
			  effect_reference_to_string(cell_any_reference(c)),
			  points_to_context_statement_line_number());
    }
    else {
      /* For arrays, an extra eval has been applied by adding 0 subscripts */
      cell nc = copy_cell(c); // FI: for debugging purpose
      c = reduce_cell_to_pointer_type(c);
      type ct = points_to_cell_to_concrete_type(c);
      if(!C_pointer_type_p(ct) && !overloaded_type_p(ct)) {
	fprintf(stderr, "nc=");
	print_points_to_cell(nc);
	fprintf(stderr, "\nc=");
	print_points_to_cell(c);
	pips_internal_error("\nSource cell cannot really be a source cell\n");
      }
      free_cell(nc);
    }
  }

  pips_assert("pt_out is consistent", consistent_points_to_graph_p(pt_out));

  if(!ENDP(ndl) || !ENDP(udl)) {
    if(!ENDP(ndl))
      pips_user_warning("Possible NULL pointer dereferencing.\n");
    else
      pips_user_warning("Possible undefined pointer dereferencing.\n");

    /* What do we want to do when the left hand side is NULL or UNDEFINED? */
    bool null_dereferencing_p
      = get_bool_property("POINTS_TO_NULL_POINTER_DEREFERENCING");
    bool nowhere_dereferencing_p
      = get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING");
    if(!null_dereferencing_p) {
      gen_list_and_not(&L, ndl);
    if(!nowhere_dereferencing_p) {
      gen_list_and_not(&L, udl);
    }
      
    // FI: I guess all undefined and nowhere cells in L should be
    // removed and replaced by only one anywhere cell
    // FI: it should be typed according to the content of the cells in del

    if(!ENDP(ndl) && null_dereferencing_p) {
      cell nc = CELL(CAR(ndl));
      type t = entity_type(reference_variable(cell_any_reference(nc)));
      cell c = make_anywhere_points_to_cell(t);
      gen_list_and_not(&L, ndl);
      L = CONS(CELL, c, L);
    }

    if(!ENDP(udl) && nowhere_dereferencing_p) {
      cell nc = CELL(CAR(udl));
      type t = entity_type(reference_variable(cell_any_reference(nc)));
      cell c = make_anywhere_points_to_cell(t);
      gen_list_and_not(&L, udl);
      L = CONS(CELL, c, L);
    }

    gen_free_list(ndl), gen_free_list(udl);
    }
  }

  if(ENDP(L)) {
    /* The code cannot be executed */
    clear_pt_map(pt_out);
    points_to_graph_bottom(pt_out) = true;
  }
  else {
    /* Compute the data-flow equation for the may and the must edges...
     *
     * out = (in - kill) U gen ?
     */

    /* Extract MAY/MUST points to relations from the input set "pt_out"  */
    set pt_out_s = points_to_graph_set(pt_out);
    set in_may = points_to_may_filter(pt_out_s);
    set in_must = points_to_must_filter(pt_out_s);
    //set kill_may = kill_may_set(L, in_may);
    // Arcs whose approximation must be changed
    set kill_may = kill_may_set(L, in_must);
    // Arcs that are definitely killed
    set kill_must = kill_must_set(L, pt_out_s);
    // FI: easiest way to find the proper set "kill_may"
    kill_may = set_difference(kill_may, kill_may, kill_must);
    bool address_of_p = true;
    // gen_may = gen_2 in the dissertation
    set gen_may = gen_may_set(L, R, pt_out_s, &address_of_p);
    // set gen_must = gen_must_set(L, R, in_must, &address_of_p);
    //set kill/* = new_pt_map()*/;
    set kill = new_simple_pt_map();
    // FI->AM: do we really want to keep the same arc with two different
    // approximations? The whole business of may/must does not seem
    // useful. 
    //kill = set_union(kill, kill_may, kill_must);
    // kill_may is handled direclty below
    kill = kill_must;
    set gen = new_simple_pt_map();
    //set_union(gen, gen_may, gen_must);
    set_assign(gen, gen_may);

    pips_assert("\"gen\" is consistent", consistent_points_to_set(gen));

    if(set_empty_p(gen)) {
      bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
      if(type_sensitive_p)
	gen = points_to_anywhere_typed(L, pt_out_s);
      else
	gen = points_to_anywhere(L, pt_out_s); 
    }

    // FI->AM: shouldn't it be a kill_must here?
    set_difference(pt_out_s, pt_out_s, kill);

    pips_assert("After removing the kill set, pt_out is consistent",
		consistent_points_to_graph_p(pt_out));
    
    set_union(pt_out_s, pt_out_s, gen);

    // FI->AM: use kill_may to reduce the precision of these arcs
    // Easier than to make sure than "gen_may1", i.e. "gen_1" in the
    // dissertation, is consistent with "kill_may", i.e. kill_2 in the
    // dissertation

    SET_FOREACH(points_to, pt, kill_may) {
      approximation a = points_to_approximation(pt);
      if(approximation_exact_p(a)) {
	points_to npt = make_points_to(copy_cell(points_to_source(pt)),
				       copy_cell(points_to_sink(pt)),
				       make_approximation_may(),
				       copy_descriptor(points_to_descriptor(pt)));
	remove_arc_from_pt_map(pt, pt_out);
	add_arc_to_pt_map(npt, pt_out);
      }
    }

    pips_assert("After union and approximation updates pt_out is consistent",
		consistent_points_to_graph_p(pt_out));

    /* Check kill_must for potential memory leaks */
    SET_FOREACH(points_to, kpt, kill_must) {
      cell d = points_to_sink(kpt);
      //approximation ap = points_to_approximation(kpt);
      // approximation_exact_p(ap) && : this is incompatible with heap_cell_p
      if(heap_cell_p(d)
	 && unreachable_points_to_cell_p(d, pt_out)) {
	/* FI: this error message may be wrong in case of a call to
	 * realloc(); see Pointers/hyantes02.c, hyantes03.c
	 *
	 * FI: this error message may deal with a bucket that does not
	 * really exist because its allocation was conditional.
	 *
	 * To make things worse, the warning is emitted in an
	 * iterative loop analysis.
	 */
	pips_user_warning("Heap bucket \"%s\" %sleaked at line %d.\n",
			  points_to_cell_to_string(d),
			  set_size(kill_must)>1? "possibly " : "",
			  points_to_context_statement_line_number());

	/* Look for a chain of memory leaks. Since they are also
	   "related" to "d", this must be done before the next
	   step. */
	pt_out = memory_leak_to_more_memory_leaks(d, pt_out);

	/* Look for related lost arcs. See Pointers/malloc18.c */
	reference dr = cell_any_reference(d);
	entity dv = reference_variable(dr);
	// cell nd = make_cell_reference(make_reference(dv, NIL));
	//points_to_cell_add_unbounded_subscripts(nd);
	list dal = NIL; // Deleted arc list
	SET_FOREACH(points_to, pt, pt_out_s) {
	  cell s = points_to_source(pt);
	  reference sr = cell_any_reference(s);
	  entity sv = reference_variable(sr);
	  if(dv==sv) {
	    if(unreachable_points_to_cell_p(s, pt_out))
	      dal = CONS(POINTS_TO, pt, dal);
	  }
	}
	FOREACH(POINTS_TO, da, dal)
	  remove_arc_from_pt_map(da, pt_out);
	gen_free_list(dal);
      }
    }

    sets_free(in_may, in_must,
	      kill_may, kill_must,
	      gen_may, /*gen_must,*/
	      gen,/* kill,*/ NULL);
    // clear_pt_map(pt_out); // FI: why not free?
  }

  return pt_out;
}

pt_map struct_initialization_to_points_to(expression lhs,
					  expression rhs,
					  pt_map in)
{
  pt_map out = in;
  // Implementation 0:
  // pips_internal_error("Not implemented yet.\n");
  // pips_assert("to please gcc, waiting for implementation", lhs==rhs && in==in);
  list L = expression_to_points_to_sources(lhs, in);

  // L must contain a unique cell, containing a non-index reference
  pips_assert("One struct to initialize", (int) gen_length(L)==1);
  cell c = CELL(CAR(L));
  reference r = cell_any_reference(c);
  entity e = reference_variable(r);
  pips_assert("c is not indexed", ENDP(reference_indices(r)));
  if(0) {
    /* Temporary implementation: use anywhere as default initialization */
    // ignore rhs
    list l = variable_to_pointer_locations(e);
    FOREACH(CELL, source, l) {
      bool to_be_freed;
      type t = points_to_cell_to_type(source, &to_be_freed);
      type c_t = type_to_pointed_type(compute_basic_concrete_type(t));
      cell sink = make_anywhere_cell(c_t);
      if(to_be_freed) free_type(t);
      points_to pt = make_points_to(source, sink,
				    make_approximation_exact(),
				    make_descriptor_none());
      add_arc_to_pt_map(pt, out);
    }
  }
  else {
    /* We must assign to each relevant field its initial value */
    list fl = struct_variable_to_fields(e); // list of entities
    list vl = struct_initialization_expression_to_expressions(rhs);
    pips_assert("The field and initial value lists have the same length",
		gen_length(fl)==gen_length(vl));
    list cvl = vl;
    FOREACH(ENTITY, f, fl) {
      reference nr =
	make_reference(e, CONS(EXPRESSION, entity_to_expression(f), NIL));
      expression nlhs = reference_to_expression(nr);
      out = assignment_to_points_to(nlhs, EXPRESSION(CAR(cvl)), out);
      POP(cvl);
    }
  }

  return out;
}

/* pt_in is modified by side-effects and returned as pt_out
 *
 * This function is also used for declarations, although the syntax
 * for declarations is reacher than the syntax for assignments which
 * can use BRACE_INTRINSIC.
 */
pt_map struct_assignment_to_points_to(expression lhs,
				      expression rhs,
				      pt_map pt_in)
{
  pt_map pt_out = pt_in;
  if(C_initialization_expression_p(rhs))
    pt_out = struct_initialization_to_points_to(lhs, rhs, pt_in);
  else {
    list L = expression_to_points_to_sources(lhs, pt_out);
    list R = expression_to_points_to_sources(rhs, pt_out);
    FOREACH(CELL, lc, L) {
      bool l_to_be_freed;
      type lt = cell_to_type(lc, &l_to_be_freed);
      entity le = reference_variable(cell_any_reference(lc));
      if(!entity_abstract_location_p(le)) {
	FOREACH(CELL, rc, R) {
	  bool r_to_be_freed;
	  type rt = cell_to_type(rc, &r_to_be_freed);
	  entity re = reference_variable(cell_any_reference(rc));
	  if(entity_abstract_location_p(le)) {
	    if(entity_abstract_location_p(re)) {
	      pips_internal_error("Not implemented yet.");
	    }
	    else {
	      pips_internal_error("Not implemented yet.");
	    }
	  }
	  else {
	    if(entity_abstract_location_p(re)) {
	      // FI: when re==NULL, we could generate a user warning or
	      // ignore the dereferencement of NULL...

	      // All fields are going to point to this abstract
	      // location... or to the elements pointed by this abstract
	      // location
	      pips_assert("Left type is struct",
			  struct_type_p(lt));
	      entity ste = basic_derived(variable_basic(type_variable(lt)));
	      type st = entity_type(ste); // structure type
	      list fl = type_struct(st); // field list
	      FOREACH(ENTITY, f, fl) {
		type ft = entity_type(f); // field type
		if(pointer_type_p(ft)) {
		  reference lr = copy_reference(cell_any_reference(lc));
		  // reference rr = copy_reference(cell_any_reference(rc));
		  reference_add_field_dimension(lr, f);
		  cell lc = make_cell_reference(lr);
		  type p_t = type_to_pointed_type(ft);
		  cell rc = make_anywhere_cell(p_t);
		  // reference_add_field_dimension(rr, f);
		  // expression nlhs = reference_to_expression(lr);
		  // expression nrhs = reference_to_expression(rr);

		  // FI: too bad this cannot be reused because of an assert in normalize_reference()....
		  // pt_out = assignment_to_points_to(nlhs, nrhs, pt_out);
		  points_to pt = make_points_to(lc, rc, make_approximation_may(), make_descriptor_none());
		  // FI: pt is allocated but not used...
		  // FI: see update_points_to_graph_with_arc()?
		  /* Current arc list (cal): the new arc may be
		     conflicting with an existing must arc */
		  list cal = points_to_source_to_arcs(lc, pt_out, false);
		  list oal = NIL;
		  list nal = NIL;
		  FOREACH(POINTS_TO, a, cal) {
		    approximation ap = points_to_approximation(a);
		    if(approximation_exact_p(ap)) {
		      oal = CONS(POINTS_TO, a, oal);
		      points_to na =
			make_points_to(copy_cell(points_to_source(a)),
				       copy_cell(points_to_sink(a)),
				       make_approximation_may(),
				       make_descriptor_none());
		      nal = CONS(POINTS_TO, na, nal);
		    }
		  }
		  FOREACH(POINTS_TO, oa, oal)
		    remove_arc_from_pt_map(oa, pt_out);
		  FOREACH(POINTS_TO, na, nal)
		    add_arc_to_pt_map(na, pt_out);
		  gen_free_list(oal), gen_free_list(nal);
		  add_arc_to_pt_map(pt, pt_out); // FI: I guess...
		  // FI->FC: it would be nice to have a Newgen free_xxxxs() to
		  // free a list of objects of type xxx with one call
		  // FI: why would we free these expressions?
		  // free_expression(lhs), free_expression(rhs);
		}
		else if(struct_type_p(ft)) {
		  pips_internal_error("Not implemented yet.\n");
		}
		else {
		  ; // Do nothing
		}
	      }
	    }
	    else {
	      pips_assert("Both types are struct or array of struct",
			  (struct_type_p(lt) || array_of_struct_type_p(lt))
			  && (struct_type_p(rt) || array_of_struct_type_p(rt)));
	      /* We may have an implicit array of struct in the right or
	       * left hand side
	       */
	      // pips_assert("Both type are equal", type_equal_p(lt, rt));
	      basic ltb = variable_basic(type_variable(lt));
	      basic rtb = variable_basic(type_variable(rt));
	      pips_assert("Both type are somehow equal",
			  basic_equal_p(ltb, rtb));
	      entity ste = basic_derived(variable_basic(type_variable(lt)));
	      type st = entity_type(ste); // structure type
	      list fl = type_struct(st); // field list
	      FOREACH(ENTITY, f, fl) {
		type uft = entity_basic_concrete_type(f); // field type
		// type uft = ultimate_type(ft);
		bool array_p = /*array_type_p(ft) ||*/ array_type_p(uft);
		if(!array_p && (pointer_type_p(uft) || struct_type_p(uft))) {
		  reference lr = copy_reference(cell_any_reference(lc));
		  reference rr = copy_reference(cell_any_reference(rc));
		  /* FI: conditionally add zero subscripts necessary to
		     move from an array "a" to its first element,
		     e.g. a[0][0][0] */
		  reference_add_zero_subscripts(lr, lt);
		  reference_add_zero_subscripts(rr, rt);
		  simple_reference_add_field_dimension(lr, f);
		  simple_reference_add_field_dimension(rr, f);
		  expression nlhs = reference_to_expression(lr);
		  expression nrhs = reference_to_expression(rr);
		  pt_out = assignment_to_points_to(nlhs, nrhs, pt_out);
		  // FI->FC: it would be nice to have a Newgen free_xxxxs() to
		  // free a list of objects of type xxx with one call
		  // The references within the expressions are now part of pt_out
		  // free_expression(lhs), free_expression(rhs);
		}
		else if(array_p && (array_of_pointers_type_p(uft)
				    || pointer_type_p(uft)
				    || array_of_struct_type_p(uft)
				    || struct_type_p(uft))) {
		  // Same as above, but an unbounded subscript is added...
		  // Quite a special assign in C...
		  reference lr = copy_reference(cell_any_reference(lc));
		  reference rr = copy_reference(cell_any_reference(rc));
		  simple_reference_add_field_dimension(lr, f);
		  simple_reference_add_field_dimension(rr, f);
		  expression li = make_unbounded_expression();
		  expression ri = make_unbounded_expression();
		  reference_indices(lr) = gen_nconc(reference_indices(lr),
						    CONS(EXPRESSION, li, NIL));
		  reference_indices(rr) = gen_nconc(reference_indices(rr),
						    CONS(EXPRESSION, ri, NIL));
		  expression nlhs = reference_to_expression(lr);
		  expression nrhs = reference_to_expression(rr);
		  pt_out = assignment_to_points_to(nlhs, nrhs, pt_out);
		}
		else {
		  ; // Do nothing
		}
	      }
	    }
	  }
	}
      }
      else {
	// FI: the lhs is an unknown struct allocated anywhere
	// FI: we might have to generate new arcs. e.g. from HEAP to STACK...
	pips_internal_error("Not implemented yet.\n");
      }
    }
  }
  // pips_internal_error("Not implemented yet for lhs %p and rhs %p\n", lhs, rhs);

  return pt_out;
}

pt_map application_to_points_to(application a, pt_map pt_in, bool side_effect_p)
{
  expression f = application_function(a);
  list al = application_arguments(a);
  pt_map pt_out = expression_to_points_to(f, pt_in, side_effect_p);

  pt_out = expressions_to_points_to(al, pt_out, side_effect_p);
  /* FI: We should also identify the possibly called functions and
     update the points-to according to the possible call sites. */
  pips_internal_error("Not implemented yet for application\n");

  return pt_out;
}

/* Update points-to set "in" according to the content of the
 * expression using side effects. Use "true_p" to know if the
 * condition must be met or not.
 *
 * FI: the side effects should not be taken into account because this
 * function is often called twice, once for the true branch and once
 * for the false branch of a test.
 */
pt_map condition_to_points_to(expression c, pt_map in, bool true_p)
{
  pt_map out = in;
  if(!points_to_graph_bottom(in)) {
    syntax cs = expression_syntax(c);

    if(syntax_reference_p(cs)) {
      /* For instance, C short cut "if(p)" for "if(p!=NULL)" */
      //out = reference_condition_to_points_to(syntax_reference(cs), in, true_p);
      out = reference_condition_to_points_to(syntax_reference(cs), in, true_p);
    }
    else if(syntax_call_p(cs)) {
      //list el = expression_to_proper_constant_path_effects(c);
      list el = NIL;
      out = call_condition_to_points_to(syntax_call(cs), in, el, true_p);
      //gen_full_free_list(el);
    }
    else {
      pips_internal_error("Not implemented yet.\n");
    }
  }
  pips_assert("out is consistent", points_to_graph_consistent_p(out));
  return out;
}

/* Handle conditions such as "if(p)" */
pt_map reference_condition_to_points_to(reference r, pt_map in, bool true_p)
{
  pt_map out = in;
  entity v = reference_variable(r);
  type vt = entity_basic_concrete_type(v);
  list sl = reference_indices(r);

  /* Do not take care of side effects in references... */
  out = expressions_to_points_to(sl, out, false);

  /* are we dealing with a pointer? */
  if(pointer_type_p(vt)) {
    if(true_p) {
      /* if p points to NULL, the condition is not feasible. If not,
	 remove any arc from p to NULL */
      if(reference_must_points_to_null_p(r, in)) {
	// FI: memory leak with clear_pt?
	pips_user_warning("Dead code detected.\n");
	clear_pt_map(out);
	points_to_graph_bottom(out) = true;
      }
      else {
	/* Make a points-to NULL and remove the arc from the current out */
	cell source = make_cell_reference(copy_reference(r));
	cell sink = make_null_pointer_value_cell();
	points_to a = make_points_to(source, sink, make_approximation_may(),
				     make_descriptor_none());
	remove_arc_from_pt_map(a, in);
	free_points_to(a);
      }
    }
    else {
      if(reference_may_points_to_null_p(r, in)) {
	/* remove any arc from v to anything and add an arc from p to NULL */
	set in_s = points_to_graph_set(in);
	points_to_source_projection(in_s, v);
	/* Make a points-to NULL and remove the arc from the current out */
	cell source = make_cell_reference(copy_reference(r));
	cell sink = make_null_pointer_value_cell();
	points_to a = make_points_to(source, sink, make_approximation_exact(),
				     make_descriptor_none());
	add_arc_to_pt_map(a, in);
      }
      else {
	/* This condition is always false */
	pips_user_warning("Dead code detected.\n");
	clear_pt_map(out);
	points_to_graph_bottom(out) = true;
      }
    }
  }

  return out;
}

/* Handle any condition that is a call such as "if(p!=q)", "if(*p)",
 * "if(foo(p=q))"... */
pt_map call_condition_to_points_to(call c, pt_map in, list el, bool true_p)
{
  pt_map out = in;
  entity f = call_function(c);
  value fv = entity_initial(f);
  if(value_intrinsic_p(fv))
    out = intrinsic_call_condition_to_points_to(c, in, true_p);
  else if(value_code_p(fv))
    out = user_call_condition_to_points_to(c, in, el, true_p);
  else if(value_constant_p(fv)) {
    // For instance "if(1)"
    ; // do nothing
  }
  else
    // FI: you might have an apply on a functional pointer?
    pips_internal_error("Not implemented yet.\n");
  pips_assert("out is consistent", points_to_graph_consistent_p(out));
  return out;
}

/* We can break down the intrinsics according to their arity or
 * according to their kinds... or according to both conditions...
 */
pt_map intrinsic_call_condition_to_points_to(call c, pt_map in, bool true_p)
{
  pt_map out = in;
  entity f = call_function(c);

  if(ENTITY_RELATIONAL_OPERATOR_P(f))
    out = relational_intrinsic_call_condition_to_points_to(c, in, true_p);
  else if(ENTITY_LOGICAL_OPERATOR_P(f))
    out = boolean_intrinsic_call_condition_to_points_to(c, in, true_p);
  else if(ENTITY_ASSIGN_P(f)) {
    // Evaluate side effects only once...
    out = intrinsic_call_to_points_to(c, in, true_p);
    expression lhs = EXPRESSION(CAR(call_arguments(c)));
    //bool to_be_freed;
    type lhs_t = points_to_expression_to_concrete_type(lhs);
    //type lhs_t = compute_basic_concrete_type(t);
    //if(to_be_freed) free_type(t);
    if(pointer_type_p(lhs_t)) {
      expression rhs = EXPRESSION(CAR(CDR(call_arguments(c))));
      list R = expression_to_points_to_sinks(rhs, out);
      if(cells_must_point_to_null_p(R) && true_p) {
	pips_user_warning("Dead code detected.\n");
	clear_pt_map(out);
	points_to_graph_bottom(out) = true;
      }
      else if(cells_may_not_point_to_null_p(R) && !true_p) {
	pips_user_warning("Dead code detected.\n");
	clear_pt_map(out);
	points_to_graph_bottom(out) = true;
      }
      gen_free_list(R);
    }
  }
  else {
    if(ENTITY_DEREFERENCING_P(f) || ENTITY_POINT_TO_P(f)
       || ENTITY_POST_INCREMENT_P(f) || ENTITY_POST_DECREMENT_P(f)
       || ENTITY_PRE_INCREMENT_P(f) || ENTITY_PRE_DECREMENT_P(f)) {
      expression p = EXPRESSION(CAR(call_arguments(c)));
      /* Make sure that all dereferencements are possible? Might be
	 included in intrinsic_call_to_points_to()... */
      //bool to_be_freed;
      type et = points_to_expression_to_concrete_type(p);
      if(pointer_type_p(et)) {
	dereferencing_to_points_to(p, in);
	out = condition_to_points_to(p, out, true_p);
      }
      //if(to_be_freed) free_type(et);
    }
    // Take care of side effects as in "if(*p++)"
    // We must take care of side effects only once...
    // Let say, when the condition is evaluated for true
    // Not too sure about side effects in condtions
    // We might also use "false" as last actual parameter...
    // FI: no, this has been taken care of earlier
    out = intrinsic_call_to_points_to(c, in, false);
    //pips_internal_error("Not implemented yet.\n");
  }
  pips_assert("out is consistent", points_to_graph_consistent_p(out));
  return out;
}

pt_map user_call_condition_to_points_to(call c, pt_map in, list el, bool true_p)
{
  pt_map out = in;
  // FI: a call site to handle like any other user call site...
  // Althgouh you'd like to know if true or false is returned...
  //pips_user_warning("Interprocedural points-to not implemented yet. "
  //		    "Call site fully ignored.\n");
  //
  if(true_p) // Analyze the call only once?
    out = user_call_to_points_to(c, in, el);
  else // No, because side-effects must be taken into account for both branches
    out = user_call_to_points_to(c, in, el);
  return out;
}

/* Deal with "!", "&&", "||" etc. */
pt_map boolean_intrinsic_call_condition_to_points_to(call c, pt_map in, bool true_p)
{
  entity f = call_function(c);
  list al = call_arguments(c);
  pt_map out = in;
  if(ENTITY_NOT_P(f)) {
    expression nc = EXPRESSION(CAR(al));
    out = condition_to_points_to(nc, in, !true_p);
  }
  else if((ENTITY_AND_P(f) && true_p) || (ENTITY_OR_P(f) && !true_p)) {
    /* Combine the conditions */
    expression nc1 = EXPRESSION(CAR(al));
    out = condition_to_points_to(nc1, in, true_p);
    expression nc2 = EXPRESSION(CAR(CDR(al)));
    out = condition_to_points_to(nc2, out, true_p);
  }
  else if((ENTITY_AND_P(f) && !true_p) || (ENTITY_OR_P(f) && true_p)) {
    /* Merge the results of the different conditions... */
    pt_map in2 = full_copy_pt_map(in);
    expression nc1 = EXPRESSION(CAR(al));
    pt_map out1 = condition_to_points_to(nc1, in, true_p);
    expression nc2 = EXPRESSION(CAR(CDR(al)));
    pt_map out2 = condition_to_points_to(nc2, in2, true_p);
    // FI: memory leak? Does merge_points_to_set() allocated a new set?
    out = merge_points_to_graphs(out1, out2);
    clear_pt_map(out2); // FI: you do no want to free the arcs
    free_pt_map(out2);
  }
  else
    pips_internal_error("Not implemented yet for boolean operator \"%s\".\n",
			entity_local_name(f));
  pips_assert("out is consistent", points_to_graph_consistent_p(out));
  return out;
}

/* See if you can decide that the addresses linked to c1 are xxx
 * than the addresses linked to c2.
 *
 * True is returned when a decision can be made.
 *
 * False is returned when no decision can be made.
  */
#define LESS_THAN 0
#define LESS_THAN_OR_EQUAL_TO 1
#define GREATER_THAN 2
#define GREATER_THAN_OR_EQUAL_TO 3

static bool cell_is_xxx_p(cell c1, cell c2, int xxx)
{
  bool xxx_p = true;
  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  if(v1!=v2) {
    xxx_p = false; // FI: useless, but the pips_user_error() may be removed
    pips_user_error("Incomparable pointers to \"%s\" and \"%s\" are compared.\n",
		    words_to_string(words_reference(r1, NIL)),
		    words_to_string(words_reference(r2, NIL)));
  }
  else {
    /* You must compare the subscript expressions lexicographically */
    list sl1 = reference_indices(r1), sl1c = sl1;
    list sl2 = reference_indices(r2), sl2c = sl2;
    for(sl1c = sl1;
	!ENDP(sl1c) && !ENDP(sl2c) && xxx_p;
	sl1c = CDR(sl1c), sl2c = CDR(sl2c)) {
      expression s1 = EXPRESSION(CAR(sl1c));
      expression s2 = EXPRESSION(CAR(sl2c));
      if(unbounded_expression_p(s1) || unbounded_expression_p(s2))
	xxx_p = false;
      else {
	value v1 = EvalExpression(s1);
	value v2 = EvalExpression(s2);
	if(!value_constant_p(v1) || !value_constant_p(v2)) {
	  // FI: this should be a pips_internal_error due to
	  // constraints on points_to sets
	  xxx_p = false;
	  pips_internal_error("Unexpected subscripts in points-to.\n");
	}
	else {
	  constant c1 = value_constant(v1);
	  constant c2 = value_constant(v2);
	  if(!constant_int_p(c1) || !constant_int_p(c2)) {
	    xxx_p = false;
	    pips_internal_error("Unexpected subscripts in points-to.\n");
	  }
	  else {
	    int i1 = constant_int(c1);
	    int i2 = constant_int(c2);
	    // FI: you should break when i1<i2
	    switch(xxx) {
	    case LESS_THAN:
	      xxx_p = (i1<i2);
	      break;
	    case LESS_THAN_OR_EQUAL_TO:
	      xxx_p = (i1<=i2);
	      break;
	    case GREATER_THAN:
	      xxx_p = (i1>i2);
	      break;
	    case GREATER_THAN_OR_EQUAL_TO:
	      xxx_p = (i1>=i2);
	    break;
	    default:
	      pips_internal_error("Unknown comparison.\n");
	    }
	  }
	}
      }
    }
    // FI: Not good for a lexicographic order, might need equal_p as
    // well, but sufficient for arithmetic02
    //if(xxx_p && !ENDP(sl1c))
    //  xxx_p = false;
  }
  return xxx_p;
}
/* See if you can decide that the addresses linked to c1 are smaller
 * than the addresses linked to c2.
 *
 * True is returned when a decision can be made.
 *
 * False is returned when no decision can be made.
 */
bool cell_is_less_than_or_equal_to_p(cell c1, cell c2)
{
  return cell_is_xxx_p(c1, c2, LESS_THAN_OR_EQUAL_TO);
}

bool cell_is_less_than_p(cell c1, cell c2)
{
  return cell_is_xxx_p(c1, c2, LESS_THAN);
}

bool cell_is_greater_than_or_equal_to_p(cell c1, cell c2)
{
  return cell_is_xxx_p(c1, c2, GREATER_THAN_OR_EQUAL_TO);
}

bool cell_is_greater_than_p(cell c1, cell c2)
{
  return cell_is_xxx_p(c1, c2, GREATER_THAN);
}

/* The condition is e==NULL
 *
 * e==NULL may be true if exists c in sinks(e) s.t. {NULL} is included in c.
 * else e==NULL must be false.
 *
 * Some arcs in in may be removed: forall c in sources(e):
 *
 * out = in - {pt=(a,b) in in | a in sources(e) and b=NULL} 
 */
pt_map null_equal_condition_to_points_to(expression e, pt_map in)
{
  pt_map out = in;
  type et = expression_to_type(e);
  if(pointer_type_p(et)) {
    list R = expression_to_points_to_sinks(e, in);
    bool null_initialization_p
      = get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION"); 

    if(ENDP(R)) {
      // Maybe, a dereferencement user error occured?
      pips_internal_error("A dereferencement should always succeed.\n");
    }

    /* May the condition be true under "in"? */
    bool found_p = false;
    FOREACH(CELL, c, R) {
      if(null_cell_p(c)
	 || anywhere_cell_p(c)
	 || cell_typed_anywhere_locations_p(c)
	 /* If NULL initialization is not performed, a stub can
	    represent a NULL. */
	 || (!null_initialization_p && stub_points_to_cell_p(c))) {
	found_p = true;
	break;
      }
    }
    if(!found_p) {
      clear_pt_map(out);
      points_to_graph_bottom(out) = true;
    }
    else {
      /* Remove arcs incompatible with the condition e==NULL and add
	 the arc e->NULL */
      list L = expression_to_points_to_sources(e, in);
      pips_assert("A lhs expression has at least one source", !ENDP(L));
      int nL = (int) gen_length(L);
      cell fc = CELL(CAR(L));
      if(nL==1 && atomic_points_to_cell_p(fc)) {
	/* All arcs starting from fc can be removed and replaced by
	   one arc from fc to NULL. */
	out = points_to_cell_source_projection(out, fc);
	points_to pt = make_points_to(copy_cell(fc),
				      make_null_cell(),
				      make_approximation_exact(),
				      make_descriptor_none());
	add_arc_to_pt_map(pt, out);
      }
      else {
	SET_FOREACH(points_to, pt, points_to_graph_set(in)) {
	  cell source = points_to_source(pt);
	  if(cell_in_list_p(source, L)) {
	    cell sink = points_to_sink(pt);
	    if(!(null_cell_p(sink)
		 || anywhere_cell_p(sink)
		 || cell_typed_anywhere_locations_p(sink))) {
	      out = remove_arc_from_pt_map(pt, out);
	    }
	  }
	}
      }
    }
  gen_free_list(R); // FI: should be full?
  }
  return out;
}

/* The condition is e!=NULL
 *
 * e!=NULL may be true if exists c in sinks(e) s.t. c != {NULL}
 *
 * e!=NULL is false if forall c in sinks(e) c is included in {NULL}
 *
 * Arcs that can be removed:
 *
 * FI: I decided not to merge this function with the previous one till
 * they are both fully defined and tested.
 */
pt_map null_non_equal_condition_to_points_to(expression e, pt_map in)
{
  pt_map out = in;
  type et = expression_to_type(e);
  if(pointer_type_p(et)) {
    list L = expression_to_points_to_sinks(e, in);

    if(ENDP(L)) {
      // Maybe, a dereferencement user error occured?
      pips_internal_error("A dereferencement should always succeed.\n");
    }

    /* May the condition be true under "in"? */
    bool found_p = false;
    FOREACH(CELL, c, L) {
      if(!null_cell_p(c)) {
	found_p = true;
	break;
      }
    }
    if(!found_p) {
      clear_pt_map(out);
      points_to_graph_bottom(out) = true;
    }
    else {
      /* Remove arcs incompatible with the condition e!=NULL */
      list L = expression_to_points_to_sources(e, in);
      SET_FOREACH(points_to, pt, points_to_graph_set(in)) {
	cell source = points_to_source(pt);
	if(cell_in_list_p(source, L)) {
	  cell sink = points_to_sink(pt);
	  if(null_cell_p(sink)) {
	    out = remove_arc_from_pt_map(pt, out);
	  }
	}
      }
    }
  }
  return out;
}

/* The expression list "al" contains exactly two arguments, "lhs" and
 * "rhs". Check if "lhs==rhs" may return true.
 *
 * If these expressions are pointers, "in" is modified by removing
 * arcs that are not compatible with the equality. If no arc is left, a
 * bottom "out" is returned.
 *
 * If one of these two expressions cannot be evaluated according to
 * the C standard, i.e. its value is undefined, a bottom graph is
 * returned.
 *
 * "out" is "in", modified by side-effects.
 *
 * This function has many commonalities with
 * non_equal_condition_to_points_to(). They were developped
 * independently to avoid mistakes when dealing with negations of
 * quantifiers. They could now be unified.
 */
pt_map equal_condition_to_points_to(list al, pt_map in)
{
  pt_map out = in;
  expression lhs = EXPRESSION(CAR(al));
  expression rhs = EXPRESSION(CAR(CDR(al)));

  // FI: in fact, any integer could be used in a pointer comparison...
  if(expression_null_p(lhs))
    out = null_equal_condition_to_points_to(rhs, in);
  else if(expression_null_p(rhs))
    out = null_equal_condition_to_points_to(lhs, in);
  else {
    type lhst = expression_to_type(lhs);
    type rhst = expression_to_type(rhs);
    if(pointer_type_p(lhst) && pointer_type_p(rhst)) {
      list L = expression_to_points_to_sinks(lhs, in);
      int nL = (int) gen_length(L);
      /* Is it impossible to evaluate lhs? 
       *
       * The check is too low. The message will be emitted twice
       * because conditions are often evaluated as true and false.
       */
      if(nL==1 && nowhere_cell_p(CELL(CAR(L)))) {
	clear_pt_map(out);
	points_to_graph_bottom(out) = true;
	pips_user_warning("Unitialized pointer is used to evaluate expression"
			  " \"%s\" at line %d.\n", expression_to_string(lhs),
			  points_to_context_statement_line_number());
      }
      else {
	/* Is it impossible to evaluate rhs? */
	list R = expression_to_points_to_sinks(rhs, in);
	int nR = (int) gen_length(R);
	if(nR==1 && nowhere_cell_p(CELL(CAR(R)))) {
	  clear_pt_map(out);
	  points_to_graph_bottom(out) = true;
	  pips_user_warning("Unitialized pointer is used to evaluate expression"
			    " \"%s\".\n", expression_to_string(rhs),
			    points_to_context_statement_line_number());
	}
	else {
	  /* Is the condition feasible? */
	  bool equal_p = false;
	  FOREACH(CELL, cl, L) {
	    FOREACH(CELL, cr, R) {
	      if(points_to_cells_intersect_p(cl, cr)) {
		equal_p = true;
		break;
	      }
	    }
	    if(equal_p)
	      break;
	  }
	  if(!equal_p) {
	    // lhs==rhs is impossible
	    clear_pt_map(out);
	    points_to_graph_bottom(out) = true;
	  }
	  else {
	    // It is possible to remove some arcs? if18.c
	    int nL = (int) gen_length(L);
	    int nR = (int) gen_length(R);
	    cell c = cell_undefined;
	    list O = list_undefined;
	    if(nL==1 && atomic_points_to_cell_p(CELL(CAR(L)))) {
	      c = CELL(CAR(L));
	      O = expression_to_points_to_sources(rhs, out);
	    }
	    else if(nR==1 && atomic_points_to_cell_p(CELL(CAR(R)))) {
	      c = CELL(CAR(R));
	      O = expression_to_points_to_sources(lhs, out);
	    }
	    if(!cell_undefined_p(c)) {
	      if((int) gen_length(O)==1) {
		cell oc = CELL(CAR(O));
		out = points_to_cell_source_projection(out, oc);
		points_to pt = make_points_to(copy_cell(oc),
					      copy_cell(c),
					      make_approximation_exact(),
					      make_descriptor_none());
		add_arc_to_pt_map(pt, out);
	      }
	    }
	  }
	}
      }
    }
    free_type(lhst), free_type(rhst);
  }
  return out;
}

/* The expression list "al" contains exactly two arguments. 
 *
 * If these expressions are pointers, "in" is modified by removing
 * arcs that are not compatible with the equality. If no arc is left, a
 * bottom "in" is returned.
 *
 * "out" is "in", modified by side-effects.
 */
pt_map non_equal_condition_to_points_to(list al, pt_map in)
{
  // FI: this code is almost identical to the code above
  // It should be shared with a more general test first and then a
  // precise test to decide if you add or remove arcs
  pt_map out = in;
  expression lhs = EXPRESSION(CAR(al));
  expression rhs = EXPRESSION(CAR(CDR(al)));

  // FI: in fact, any integer could be used in a pointer comparison...
  if(expression_null_p(lhs))
    out = null_non_equal_condition_to_points_to(rhs, in);
  else if(expression_null_p(rhs))
    out = null_non_equal_condition_to_points_to(lhs, in);
  else {
    type lhst = expression_to_type(lhs);
    type rhst = expression_to_type(rhs);
    if(pointer_type_p(lhst) && pointer_type_p(rhst)) {
      list L = expression_to_points_to_sinks(lhs, in);
      list R = expression_to_points_to_sinks(rhs, in);
      //bool equal_p = false;
      int nL = (int) gen_length(L);
      int nR = (int) gen_length(R);
      pips_assert("The two expressions can be dereferenced", nL>=1 && nR>=1);
      if(nL==1 && nR==1) {
	cell cl = CELL(CAR(L));
	cell cr = CELL(CAR(R));
	/* Is the condition lhs!=rhs certainly impossible to evaluate?
	 * If not, is it always false? */
	if((atomic_points_to_cell_p(cl)
	    && atomic_points_to_cell_p(cr)
	    && points_to_cell_equal_p(cl, cr))
	   || nowhere_cell_p(cl)
	   || nowhere_cell_p(cr)) {
	  // one or more expressions is not evaluable or the condition
	  // is not feasible
	  clear_pt_map(out);
	  points_to_graph_bottom(out) = true;
	  if(nowhere_cell_p(cl))
	    pips_user_warning("Unitialized pointer is used to evaluate expression"
			      " \"%s\" at line %d.\n",
			      expression_to_string(lhs),
			      points_to_context_statement_line_number());
	  if(nowhere_cell_p(cr))
	    pips_user_warning("Unitialized pointer is used to evaluate expression"
			      " \"%s\" at line %d.\n",
			      expression_to_string(rhs),
			      points_to_context_statement_line_number());
	}
      }
      else {
	// It is possible to remove some arcs? if18.c
	int nL = (int) gen_length(L);
	int nR = (int) gen_length(R);
	cell c = cell_undefined;
	list O = list_undefined;
	if(nL==1 && atomic_points_to_cell_p(CELL(CAR(L)))) {
	  c = CELL(CAR(L));
	  O = expression_to_points_to_sources(rhs, out);
	}
	else if(nR==1 && atomic_points_to_cell_p(CELL(CAR(R)))) {
	  c = CELL(CAR(R));
	  O = expression_to_points_to_sources(lhs, out);
	}
	if(!cell_undefined_p(c)) {
	  if((int) gen_length(O)==1) {
	    cell oc = CELL(CAR(O));
	    points_to pt = make_points_to(copy_cell(oc),
					  copy_cell(c),
					  make_approximation_may(),
					  make_descriptor_none());
	    remove_arc_from_pt_map(pt, out);
	    // Should we free pt? Or is it done by remove_arc_from_pt_map()?
	  }
	}
      }
    }
    free_type(lhst), free_type(rhst);
  }
  return in;
}

/* The expression list "al" contains exactly two arguments. 
 *
 * If these expressions are pointers, "in" is modified by removing
 * arcs that are not compatible with the equality. If no arc is left, a
 * bottom "in" is returned.
 *
 * "out" is "in", modified by side-effects.
 */
pt_map order_condition_to_points_to(entity f, list al, bool true_p, pt_map in)
{
  pt_map out = in;
  bool (*cmp_function)(cell, cell);
  if((ENTITY_LESS_OR_EQUAL_P(f) && true_p)
     || (ENTITY_GREATER_THAN_P(f) && !true_p)) {
    // cmp_function = cell_is_less_than_or_equal_to_p;
    cmp_function = cell_is_greater_than_p;
  }
  else if((ENTITY_LESS_OR_EQUAL_P(f) && !true_p)
	  || (ENTITY_GREATER_THAN_P(f) && true_p)) {
    // cmp_function = cell_is_greater_than_p;
    cmp_function = cell_is_less_than_or_equal_to_p;
  }
  else if((ENTITY_GREATER_OR_EQUAL_P(f) && true_p)
	  || (ENTITY_LESS_THAN_P(f) && !true_p)) {
    // cmp_function = cell_is_greater_than_or_equal_to_p;
    cmp_function = cell_is_less_than_p;
  }
  else if((ENTITY_GREATER_OR_EQUAL_P(f) && !true_p)
	  || (ENTITY_LESS_THAN_P(f) && true_p)) {
    //cmp_function = cell_is_less_than_p;
    cmp_function = cell_is_greater_than_or_equal_to_p;
  }
  else 
    pips_internal_error("Unexpected relational operator.\n");

  expression lhs = EXPRESSION(CAR(al));
  type lhst = expression_to_type(lhs);
  expression rhs = EXPRESSION(CAR(CDR(al)));
  type rhst = expression_to_type(rhs);
  if(pointer_type_p(lhst) || pointer_type_p(rhst)) {
    list L = expression_to_points_to_sinks(lhs, in);
    if(gen_length(L)==1) { // FI: one fixed bound
      cell lc = CELL(CAR(L));
      list RR = expression_to_points_to_sources(rhs, in);
      set out_s = points_to_graph_set(out);
      SET_FOREACH(points_to, pt, out_s) {
	cell source = points_to_source(pt);
	if(cell_in_list_p(source, RR)) {
	  cell sink = points_to_sink(pt);
	  if(cmp_function(lc, sink)) {
	    approximation a = points_to_approximation(pt);
	    if(approximation_exact_p(a)) {
	      /* The condition cannot violate an exact arc. */
	      // clear_pt_map(out);
	      points_to_graph_bottom(out) = true;
	      // Would be useless/different with a union
	      set_clear(points_to_graph_set(out));
	    }
	    else
	      remove_arc_from_pt_map(pt, out);
	  }
	}
      }
    }
    else {
      list R = expression_to_points_to_sinks(rhs, in);
      if(gen_length(R)==1) { // FI: one fixed bound
	cell rc = CELL(CAR(R));
	list LL = expression_to_points_to_sources(lhs, in);
	set out_s = points_to_graph_set(out);
	SET_FOREACH(points_to, pt, out_s) {
	  cell source = points_to_source(pt);
	  if(cell_in_list_p(source, LL)) {
	    cell sink = points_to_sink(pt);
	    if(cmp_function(sink, rc)) {
	      // FI: Oops in middle of the iterator...
	      approximation a = points_to_approximation(pt);
	      if(approximation_exact_p(a)) {
		/* The condition cannot violate an exact arc. */
		// clear_pt_map(out);
		points_to_graph_bottom(out) = true;
		// Would be useless/different with a union
		set_clear(points_to_graph_set(out));
	      }
	      else
		remove_arc_from_pt_map(pt, out);
	    }
	  }
	}
      }
    }
  }
  free_type(lhst), free_type(rhst);

  return in;
}

/* Update the points-to information "in" according to the validity of
 * the condition.
 *
 * We can remove the arcs that violate the condition or decide that the
 * condition cannot be true.
 */
pt_map relational_intrinsic_call_condition_to_points_to(call c, pt_map in, bool true_p)
{
  pt_map out = in;
  entity f = call_function(c);
  list al = call_arguments(c);

  if((ENTITY_EQUAL_P(f) && true_p)
     || (ENTITY_NON_EQUAL_P(f) && !true_p)) {
    out = equal_condition_to_points_to(al, in);
  }
  else if((ENTITY_EQUAL_P(f) && !true_p)
     || (ENTITY_NON_EQUAL_P(f) && true_p)) {
    out = non_equal_condition_to_points_to(al, in);
  }
  else if(ENTITY_LESS_OR_EQUAL_P(f)
     || ENTITY_GREATER_OR_EQUAL_P(f)
     || ENTITY_GREATER_THAN_P(f)
     || ENTITY_LESS_THAN_P(f)) {
    out = order_condition_to_points_to(f, al, true_p, in);
  }
  else {
    pips_internal_error("Not implemented yet.\n");
  }
  pips_assert("out is consistent", points_to_graph_consistent_p(out));
  return out;
}
