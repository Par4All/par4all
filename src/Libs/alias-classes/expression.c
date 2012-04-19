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
    pt_out = application_to_points_to(a, pt_out);
    break;
  }
  case is_syntax_va_arg: {
    // The call to va_arg() does not create a points-to per se
    list soel = syntax_va_arg(s);
    sizeofexpression soe1 = SIZEOFEXPRESSION(CAR(soel));
    //sizeofexpression soe2 = SIZEOFEXPRESSION(CAR(CDR(soel)));
    expression se = sizeofexpression_expression(soe1);
    // type t = sizeofexpression_type(soe2);
    pt_out = expression_to_points_to(se, pt_out);
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
  pt_map pt_out = pt_in;

  entity f = call_function(c);
  list al = call_arguments(c);
  type ft = entity_type(f);
  functional ff = type_functional(ft);
  type rt = functional_result(ff);

  if(ENTITY_STOP_P(f)||ENTITY_ABORT_SYSTEM_P(f)||ENTITY_EXIT_SYSTEM_P(f)
     || ENTITY_ASSERT_FAIL_SYSTEM_P(f)) {
    clear_pt_map(pt_out);
  }
  else if(ENTITY_FCLOSE_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    pt_out = freed_pointer_to_points_to(lhs, pt_out);
  }
  else if(ENTITY_FREE_SYSTEM_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    pt_out = freed_pointer_to_points_to(lhs, pt_out);
  }
  else {
    if(!type_void_p(rt)) {
      value fv = entity_initial(f);

      /* points-to updates due to arguments */
      // FI: this cannot be delayed but it is unfortunately applied
      // again when going down? See arithmetic08 and 09?
      // This is necessary but cannot be placed here because of the
      // recursive calls
      // FI: we are in trouble for post increment and post decrement...
      // We should update the target a second time in sinks.c!
      pt_out = expressions_to_points_to(al, pt_in);

      // FI: I wanted to use the return type but it is too often
      // overloaded with intrinsics
      type ct = call_to_type(c);
      if(pointer_type_p(ct) || struct_type_p(ct)) {
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
      }
      else {
	/* points-to updates due to arguments: already performed */
	// pt_out = expressions_to_points_to(al, pt_in);
	;
      }
      free_type(ct);
    }
    else {
      /* points-to updates due to arguments */
      pt_out = expressions_to_points_to(al, pt_in);
    }
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
    pt_out = freed_pointer_to_points_to(lhs, pt_out);
  }
   // According to C standard, pointer arithmetics does not change
  // the targeted object.
  else if(ENTITY_PLUS_UPDATE_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    expression delta = EXPRESSION(CAR(CDR(al)));
    pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
  }
  else if(ENTITY_MINUS_UPDATE_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    expression rhs = EXPRESSION(CAR(CDR(al)));
    entity um = FindOrCreateTopLevelEntity(UNARY_MINUS_OPERATOR_NAME);
    expression delta = MakeUnaryCall(um, copy_expression(rhs));
    pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
    free_expression(delta);
  }
  else if(ENTITY_POST_INCREMENT_P(f) || ENTITY_PRE_INCREMENT_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    expression delta = int_to_expression(1);
    pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
    free_expression(delta);
  }
  else if(ENTITY_POST_DECREMENT_P(f) || ENTITY_PRE_DECREMENT_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    expression delta = int_to_expression(-1);
    pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
    free_expression(delta);
  }
  else {
    // FI: fopen(), fclose() should be dealt with

    // Not safe till all previous tests are defined
    // It is assumed that other intrinsics do not generate points-to arcs...
    // pips_internal_error("Not implemented yet\n");
    pt_out = pt_in;
  }

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
  // list sources = expression_to_constant_paths(statement_undefined, lhs, pt_out);
  list sources = expression_to_points_to_sources(lhs, pt_out);
  FOREACH(CELL, source, sources) {
    // You want sharing for side effects
    list sinks = source_to_sinks(source, pt_out, false);
    /* Update the sinks by side-effect, taking advantage of the
       shallow copy performed in source_to_sinks(). */
    // FI: this should be hidden in source_to_sinks()...
    if(ENDP(sinks)) {
      /* Three possibilities: the referenced variable is a formal
	 parameter, or the referenced variable is a global variable or
	 an internal error has been encountered. And a fourth
	 possibility if we really operate on demand: a virtual entity
	 of the calling context */
      reference r = cell_any_reference(source);
      entity v = reference_variable(r);
      if(formal_parameter_p(v)
	 || /* global_variable_p(v)*/
	 /* FI: wrong test anywhere might have been top-level? */
	 /* FI: wrong, how about static global variables? */
	 top_level_entity_p(v)) {
	// FI: No idea if I should copy source to avoid some sharing...
	// FI: this function is much too general since it goes down
	// recursively in potentially recursive data structures...
	// FI: this is not on-demand...
	// FI: located in points_to_init_analysis.c
	// pt_map new = formal_points_to_parameter(source);
	// pt_out = union_of_pt_maps(pt_out, pt_out, new);
	// Find stub type
	type st = type_to_pointed_type(ultimate_type(entity_type(v)));
	// FI: the type retrieval must be improved for arrays & Co
	points_to pt = create_stub_points_to(source, st, basic_undefined);
	pt_out = add_arc_to_pt_map(pt, pt_out);
	sinks = source_to_sinks(source, pt_out, false);
      }
      else if(false) {
	/* cell nc = add_virtual_sink_to_source(source);
	 * points_to npt = make_points_to(copy_cell(source), nc, may/must)
	 * pt_out = update_pt_map(); set_add_element()? add_arc_to_pt_map()
	 */
	;
      }
      if(ENDP(sinks))
	pips_internal_error("Sink missing for a source based on \"%s\".\n",
			    entity_user_name(v));
    }
    offset_cells(sinks, delta);
  }
  // FI: should we free the sources list? Fully free it?
  return pt_out;
}

/* Each cell in sinks is replaced by a cell located "delta" elements
   further up in the memory. */
void offset_cells(list sinks, expression delta)
{
  FOREACH(CELL, sink, sinks) {
    offset_cell(sink, delta);
  }
}

void offset_cell(cell sink, expression delta)
{
  /* "&a[i]" should be transformed into "&a[i+eval(delta)]" when
     "delta" can be statically evaluated */
  reference r = cell_any_reference(sink);
  entity v = reference_variable(r);
  if(entity_array_p(v)
     || !get_bool_property("POINTS_TO_STRICT_POINTER_TYPES")) {
    value v = EvalExpression(delta);
    list sl = reference_indices(r);
    if(value_constant_p(v) && constant_int_p(value_constant(v))) {
      int dv =  constant_int(value_constant(v));
      if(ENDP(sl)) {
	// FI: oops, we are in trouble; assume 0...
	expression se = int_to_expression(dv);
	reference_indices(r) = CONS(EXPRESSION, se, NIL);
      }
      else {
	expression lse = EXPRESSION(CAR(gen_last(sl)));
	value vlse = EvalExpression(lse);
	if(value_constant_p(vlse) && constant_int_p(value_constant(vlse))) {
	  int ov =  constant_int(value_constant(vlse));
	  expression nse = int_to_expression(dv+ov);
	  ; // for the time being, do nothing as * is going to be used anyway
	  EXPRESSION_(CAR(gen_last(sl))) = nse;
	  free_expression(lse);
	}
	else {
	  // FI: assume * is used... UNBOUNDED_DIMENSION
	  expression nse = make_unbounded_expression();
	  EXPRESSION_(CAR(gen_last(sl))) = nse;
	  free_expression(lse);
	}
      }
    }
    else {
      if(ENDP(sl)) {
	expression nse = make_unbounded_expression();
	reference_indices(r) = CONS(EXPRESSION, nse, NIL);
      }
      else {
	expression ose = EXPRESSION(CAR(gen_last(sl)));
	expression nse = make_unbounded_expression();
	EXPRESSION_(CAR(gen_last(sl))) = nse;
	free_expression(ose);
      }
    }
  }
  // FI to be extended to pointers and points-to stubs
  else {
    pips_user_error("Use of pointer arithmetic on %s is not "
		    "standard-compliant.\n"
		    "Reset property \"POINTS_TO_STRICT_POINTER_TYPES\""
		    " for usual non-standard compliant C code.\n",
		    entity_user_name(v));
  }
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
  pt_map pt_out = pt_in;
  // FI: lhs and rhs have already been used to update pt_in
  //pt_map pt_out = expression_to_points_to(lhs, pt_in);
  /* It is not obvious that you are allowed to evaluate this before
     the sink of lhs, but the standard probably forbid stupid side
     effects. */
  //pt_out = expression_to_points_to(lhs, pt_out);
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
 */
pt_map pointer_assignment_to_points_to(expression lhs,
				       expression rhs,
				       pt_map pt_in)
{
  pt_map pt_out = pt_in;

  /* Do not take side effects into account, it has already been done
     at a higher level */
  // pt_out = expression_to_points_to(lhs, pt_out);
  // pt_out = expression_to_points_to(rhs, pt_out); // FI: used to be "incur"

  /* Change the "lhs" into a constant memory path using current
   * points-to information pt_out.
   */
  //list L = expression_to_constant_paths(statement_undefined, lhs, pt_out);
  list L = expression_to_points_to_sources(lhs, pt_out);

  /* Retrieve the memory locations that might be reached by the rhs
   *
   * Update the calling context by adding new stubs linked directly or
   * indirectly to the formal parameters and global variables if
   * necessary.
   */
  list R = expression_to_points_to_sinks(rhs,pt_out);

  pips_assert("Left hand side reference list is not empty.\n", !ENDP(L));
  pips_assert("Right hand side reference list is not empty.\n", !ENDP(R));

  // FI: where do we want to check for dereferencement of
  // nowhere/undefined and NULL? Here? Or within
  // list_assignment_to_points_to?

  pt_out = list_assignment_to_points_to(L, R, pt_out);

  // FI: memory leak(s)?

  return pt_out;
}

/* Any abstract location of the lhs in L is going to point to nowhere.
 *
 * Any source in pt_in pointing towards any location in lhs may or
 * nust now points towards nowhere (malloc07).
 *
 * New points-to information must be added when a formal parameter
 * is dereferenced.
 *
 * Equations for "free(e);":
 *
 * Let L = expression_to_sources(e,in) and R = expression_to_sinks(e,in)
 *
 * Any location l corresponding to e can now point to nowhere/undefined:
 *
 * Gen_1 = {pts=(l,nowhere,a) | l in L}
 *
 * Any location source that pointed to a location pointed to by l can
 * now point to nowehere/undefined.
 *
 * Gen_2 = {pts=(source,nowhere,a) | exists r in R
 *                                   && r in Heap
 *                                   && exists pts'=(source,r,a') in in}
 *
 * If e corresponds to a unique (non-abstract?) location l, any arc
 * starting from l can be removed:
 *
 * Kill_1 = {pts=(l,r,a) in in | l in L && |L|=1}
 *
 * If the freed location r is precisely known, any war pointing
 * towards it can be removed:
 *
 * Kill_2 = {pts=(l,r,a) in in | r in R && |R|=1}
 *
 * out = (in - Kill_1 - Kill_2) U Gen_1 U Gen_2
 *
 * Warnings for dangling pointers:
 *
 * DP = {r|exists pts=(r,l,a) in Gen_2} // To be checked
 *
 * Memory leaks: the freed bucket may be the only bucket containing
 * pointers towards another bucket:
 *
 * ML = {} // Still to be done
 *
 * Note: for DP and ML, we could compute may and must sets. We only
 * compute must sets to avoid swamping the log file with false alarms.
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
 * The cardinal of |L| does not seem to have an impact...
 */
pt_map freed_pointer_to_points_to(expression lhs, pt_map pt_in)
{
  pt_map pt_out = pt_in;

  list L = expression_to_points_to_sources(lhs, pt_out);
  list R = expression_to_points_to_sinks(lhs, pt_out);

  /* Build a nowhere cell
   *
   * FI->AM: typed nowhere?
   */
  list N = CONS(CELL, make_nowhere_cell(), NIL);

  pips_assert("L is not empty", !ENDP(L));
  pips_assert("R is not empty", !ENDP(R));

  /* Remove Kill_1 if it is not empty by definition */
  if(gen_length(L)==1) {
    SET_FOREACH(points_to, pts, pt_out) {
      cell l = points_to_source(pts);
      if(points_to_cell_in_list_p(l, L)) {
	// FI: assuming you can perform the removal inside the loop...
	remove_arc_from_pt_map(pts, pt_out);
      }
    }
  }

  /* Remove Kill_2 if it is not empty by definition and add Gen_2 */
  if(gen_length(R)==1) {
    SET_FOREACH(points_to, pts, pt_out) {
      cell r = points_to_sink(pts);
      if(points_to_cell_in_list_p(r, R)) {
	/* FI: should be easier and more efficient to substitute the sink...*/
	cell source = copy_cell(points_to_source(pts));
	cell sink = make_nowhere_cell();
	approximation a = copy_approximation(points_to_approximation(pts));
	points_to npts = make_points_to(source, sink, a, make_descriptor_none());
	add_arc_to_pt_map(npts, pt_out);
	// FI: assuming you can perform the removal inside the loop...
	remove_arc_from_pt_map(pts, pt_out);
      }
    }
  }

  /* Add Gen_1 - Not too late since pt_out has aready been modified? */
  pt_out = list_assignment_to_points_to(L, N, pt_out);

  /* Add Gen_2: useless, already performed by Kill_2 */
  SET_FOREACH(points_to, pts, pt_out) {
    cell r = points_to_sink(pts);
    if(points_to_cell_in_list_p(r, R)) {
      cell source = copy_cell(points_to_source(pts));
      cell sink = make_nowhere_cell();
      approximation a = copy_approximation(points_to_approximation(pts));
      points_to npts = make_points_to(source, sink, a, make_descriptor_none());
      add_arc_to_pt_map(npts, pt_out);
    }
  }

  /*
   * Other pointers may or must now be dangling because their target
   * has been freed.
   */

  FOREACH(CELL, c, R) {
    if(!sink_in_set_p(c, pt_out)) {
      if(heap_cell_p(c)) {
	entity b = reference_variable(cell_any_reference(c));
	pips_user_warning("Memory leak for bucket \"%s\".\n",
			  entity_name(b));
      }
    }
  }

  // FI: memory leak(s) in this function?
  gen_free_list(L);
  gen_free_list(N);
  gen_full_free_list(R);

  return pt_out;
}


/* Update pt_out when any element of L can be assigned any element of R
 *
 * FI->AM: Potential and sure memory leaks are not (yet) detected.
 *
 * FI->AM: the distinction between may and must sets used in the
 * implementation seem useless.
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
 * This function is used to model a C pointer assignment "e1 = e2;"
 *
 * Let L = expression_to_sources(e1) and R = expression_to_sinks(e2).
 *
 * Let in be the points-to relation before executing the assignment.
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
  /* Check dereferencing errors */
  bool singleton_p = (gen_length(L)==1);
  FOREACH(CELL, c, L) {
    if(nowhere_cell_p(c)){
      if(singleton_p)
	// Not necessarily a user error if the code is dead
	// Should be controlled by an extra property...
	pips_user_warning("Dereferencing of an undefined pointer.\n");
      else
	pips_user_warning("Dereferencing of an undefined pointer.\n");
    }
    else if(null_cell_p(c)) {
      if(singleton_p)
	// Not necessarily a user error if the code is dead
	// Should be controlled by an extra property...
	pips_user_warning("Dereferencing of a null pointer.\n");
      else
	pips_user_warning("Dereferencing of a null pointer.\n");
    }
  }

  /* Compute the data-flow equation for the may and the must edges...
   *
   * out = (in - kill) U gen ?
   */

 /* Extract MAY/MUST points to relations from the input set "pt_out"  */
  pt_map in_may = points_to_may_filter(pt_out);
  pt_map in_must = points_to_must_filter(pt_out);
  pt_map kill_may = kill_may_set(L, in_may);
  pt_map kill_must = kill_must_set(L, pt_out);
  bool address_of_p = true;
  pt_map gen_may = gen_may_set(L, R, in_may, &address_of_p);
  pt_map gen_must = gen_must_set(L, R, in_must, &address_of_p);
  pt_map kill = new_pt_map();
  // FI->AM: do we really want to keep the same arc with two different
  // approximations? The whole business of may/must does not seem
  // useful. 
  union_of_pt_maps(kill, kill_may, kill_must);
  pt_map gen = new_pt_map();
  union_of_pt_maps(gen, gen_may, gen_must);

  if(set_empty_p(gen)) {
    bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
    if(type_sensitive_p)
      gen = points_to_anywhere_typed(L, pt_out);
    else
      gen = points_to_anywhere(L, pt_out); 
  }

  // FI->AM: shouldn't it be a kill_must here?
  difference_of_pt_maps(pt_out, pt_out, kill);
  union_of_pt_maps(pt_out, pt_out, gen);

  free_pt_maps(in_may, in_must,
	       kill_may, kill_must,
	       gen_may, gen_must,
	       gen, kill, NULL);
  // clear_pt_map(pt_out); // FI: why not free?

  return pt_out;
}

/* pt_in is modified by side-effects and returned as pt_out */
pt_map struct_assignment_to_points_to(expression lhs,
				      expression rhs,
				      pt_map pt_in)
{
  pt_map pt_out = pt_in;
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
		add_arc_to_pt_map(pt, pt_out); // FI: I guess...
		// FI->FC: it would be nice to have a Newgen free_xxxxs() to
		// free a list of objects of type xxx with one call
		free_expression(lhs), free_expression(rhs);
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
	    pips_assert("Both types are struct",
			struct_type_p(lt) && struct_type_p(rt));
	    pips_assert("Both type are equal", type_equal_p(lt, rt));
	    entity ste = basic_derived(variable_basic(type_variable(lt)));
	    type st = entity_type(ste); // structure type
	    list fl = type_struct(st); // field list
	    FOREACH(ENTITY, f, fl) {
	      type ft = entity_type(f); // field type
	      if(pointer_type_p(ft) || struct_type_p(ft)) {
		reference lr = copy_reference(cell_any_reference(lc));
		reference rr = copy_reference(cell_any_reference(rc));
		reference_add_field_dimension(lr, f);
		reference_add_field_dimension(rr, f);
		expression nlhs = reference_to_expression(lr);
		expression nrhs = reference_to_expression(rr);
		pt_out = assignment_to_points_to(nlhs, nrhs, pt_out);
		// FI->FC: it would be nice to have a Newgen free_xxxxs() to
		// free a list of objects of type xxx with one call
		// The references within the expressions are now part of pt_out
		// free_expression(lhs), free_expression(rhs);
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
  // pips_internal_error("Not implemented yet for lhs %p and rhs %p\n", lhs, rhs);

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
