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
 * the expression level.
 *
 * Most important function:
 *
 * list expression_to_points_to_cells(expression lhs, pt_map in, bool
 * eval_p, bool constant_p)
 *
 * The purpose of this function and the functions in this C file is to
 * return a list of memory cells addressed when evaluated in the
 * memory context "in".
 *
 * If eval_p is false, possible addresses for "lhs" are returned.
 *
 * If eval_p is true, possible addresses for "*lhs" are returned.
 *
 * The memory context "in" may be modified by side-effects if new
 * memory locations have to be added, for instance when formal or
 * global pointers are dereferenced.
 *
 * The similar functions in Amira's implementation are located in
 * constant-path-utils.c.
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

/* allocate a new list of sinks with one element, abstract or concrete, e.
 *
 * No check on e for the time being
 */
list entity_to_sinks(entity e)
{
  reference nr = make_reference(e, NIL);
  cell nc = make_cell_reference(nr);
  list sinks = CONS(CELL, nc, NIL);
  return sinks;
}

cell entity_to_cell(entity e)
{
  reference nr = make_reference(e, NIL);
  cell nc = make_cell_reference(nr);
  return nc;
}

list points_to_null_sinks()
{
  /* The null location is not typed. The impact on dependence test is
     not clear. */
  entity ne = entity_null_locations();
  return entity_to_sinks(ne);
}

cell make_null_cell(void)
{
  entity ne = entity_null_locations();
  cell c = entity_to_cell(ne);
  return c;
}

list points_to_anywhere_sinks(type t)
{
  entity ne;
  if(type_undefined_p(t))
    ne = entity_all_locations();
  else
    ne = entity_typed_anywhere_locations(t);
  return entity_to_sinks(ne);
}

list call_to_points_to_sinks(call c, type et, pt_map in, bool eval_p, bool constant_p)
{
  list sinks = NIL;
  entity f = call_function(c);
  //list al = call_arguments(c);
  value v = entity_initial(f);
  tag tt = value_tag(v);
  switch (tt) {
  case is_value_code:
    sinks = user_call_to_points_to_sinks(c, et, in, eval_p);
    break;
  case is_value_symbolic:
    break;
  case is_value_constant: {
    constant zero = value_constant(v);
    if(constant_int_p(zero) && constant_int(zero)==0)
      sinks = points_to_null_sinks(); // et could be used according to PJ
    else {
      type t = type_to_returned_type(entity_type(f));
      if(string_type_p(t)) {
	// FI: we could generate a special location for each string
	// FI: we could reuse the constant function
	// FI: we could use the static area of the current module
	// FI: we can always use anywhere...
	// FI: does not depend oneval_p... involution...
	reference r = make_reference(f, NIL);
	cell c = make_cell_reference(r);
	sinks = CONS(CELL, c, NIL);
    }
    else {
      sinks = points_to_anywhere_sinks(t);
    }
    }
  }
    break;
  case is_value_unknown:
    pips_internal_error("function %s has an unknown value\n",
                        entity_name(f));
    break;
  case is_value_intrinsic: 
    // FI: here is the action, &p, *p, p->q, p.q, etc...
    sinks = intrinsic_call_to_points_to_sinks(c, in, eval_p, constant_p);
    break;
  default:
    pips_internal_error("unknown value tag %d\n", tt);
    break;
  }

  /* No check, but all elements of sinks should be of a type
     compatible with type "et". */

  return sinks;
}

/*
 * Sinks: "malloc(exp)", "p", "p++", "p--", "p+e", "p-e" , "p=e", "p, q, r",
 * "p->q", "p.q",...
 *
 * "(cast) p" is not an expression.
 */
list intrinsic_call_to_points_to_sinks(call c, pt_map in, bool eval_p,
				       bool constant_p)
{
  list sinks = NIL;
  entity f = call_function(c);
  list al = call_arguments(c);
  int nary = (int) gen_length(al);

  // You do not know the number of arguments for the comma operator
  if(ENTITY_COMMA_P(f)) {
    expression e = EXPRESSION(CAR(gen_last(al)));
    sinks = expression_to_points_to_sinks(e, in);
  }
  else {
    // Switch on number of arguments to avoid long switch on character
    // string or memoizing of intrinsics
    switch(nary) {
    case 0:
      pips_internal_error("Probably a constant or a symbolic. Not handled here\n");
      break;
    case 1:
      sinks = unary_intrinsic_call_to_points_to_sinks(c, in, eval_p, constant_p);
      break;
    case 2:
      sinks = binary_intrinsic_call_to_points_to_sinks(c, in, eval_p);
      break;
    case 3:
      sinks = ternary_intrinsic_call_to_points_to_sinks(c, in, eval_p, constant_p);
      break;
    default:
      sinks = nary_intrinsic_call_to_points_to_sinks(c, in);
      break;
    }
  }

  return sinks;
}

/*
 * malloc, &p, *p, p++, p--, ++p, --p,
 *
 * Do not create any sharing between elements of in and elements part
 * of the returned list, sinks.
 */
list unary_intrinsic_call_to_points_to_sinks(call c, pt_map in, bool eval_p, bool constant_p)
{
  entity f = call_function(c);
  list al = call_arguments(c);
  expression a = EXPRESSION(CAR(al));
  list sinks = NIL;
  pips_assert("One argument", gen_length(al)==1);
  // pips_internal_error("Not implemented for %p and %p\n", c, in);
  if (ENTITY_MALLOC_SYSTEM_P(f)) {
    sinks = malloc_to_points_to_sinks(a, in);
  }
  else if (ENTITY_FREE_SYSTEM_P(f)) {
    // FI: should be useless because free() returns void
    sinks = CONS(CELL, make_nowhere_cell(), NIL);
  }
  else if(ENTITY_ADDRESS_OF_P(f)) {
    // sinks = expression_to_constant_paths(statement_undefined, a, in);
    if(eval_p) {
      // sinks = expression_to_points_to_sources(a, in);
      sinks = expression_to_points_to_cells(a, in, false, constant_p);
    }
    else {
      // It is not possible in general to associate a reference to &e
      // However, we might think of transforming &a[i][j] into a[i]
      // and &a[0] into a...
      sinks=NIL;
    }
   }
  else if(ENTITY_DEREFERENCING_P(f)) {
    // FI: I do not understand why eval_p is only used for dereferencing...
    // struct18.c sensitive to the test below
    if(eval_p)
      sinks = dereferencing_to_sinks(a, in, eval_p);
    else {
      sinks = dereferencing_to_sinks(a, in, eval_p);
      // sinks = expression_to_points_to_sources(a, in);
    }
  }
  else if(ENTITY_PRE_INCREMENT_P(f)) {
    if(eval_p) {
      sinks = expression_to_points_to_sinks(a, in);
      // FI: this has already been done when the side effects are exploited
      //expression one = int_to_expression(1);
      //offset_cells(sinks, one);
      //free_expression(one);
    }
   }
  else if(ENTITY_PRE_DECREMENT_P(f)) {
    if(eval_p) 
{
      sinks = expression_to_points_to_sinks(a, in);
      //expression m_one = int_to_expression(-1);
      //offset_cells(sinks, m_one);
      //free_expression(m_one);
    }
   }
  else if(ENTITY_POST_INCREMENT_P(f) || ENTITY_POST_DECREMENT_P(f)) {
    //sinks = expression_to_constant_paths(statement_undefined, a, in);
    // arithmetic05: "q=p++;" p++ must be evaluated
    //list sources = expression_to_points_to_sinks(a, in);
    //if(gen_length(sources)==1) {
    //cell source = CELL(CAR(sources));
    if(eval_p) {
      type dt = points_to_expression_to_pointed_type(a);
      sinks = expression_to_points_to_sinks(a, in);
      /* We have to undo the impact of side effects performed when the arguments were analyzed for points-to information */
      expression delta = expression_undefined;
      if(ENTITY_POST_INCREMENT_P(f))
	delta = int_to_expression(-1);
      else
	delta = int_to_expression(1);
      offset_points_to_cells(sinks, delta, dt);
      free_expression(delta);
    }
  }
  else {
  // FI: to be continued
    pips_internal_error("Unexpected unary pointer operator\n");
  }

  return sinks;
}

// p=q, p.x, p->y, p+e, p-e, p+=e, p-=e
// What other binary operator could be part of a lhs expression?
list binary_intrinsic_call_to_points_to_sinks(call c, pt_map in, bool eval_p)
{
  entity f = call_function(c);
  list al = call_arguments(c);
  expression a1 = EXPRESSION(CAR(al));
  expression a2 = EXPRESSION(CAR(CDR(al)));
  list sinks = NIL;

  pips_assert("in is consistent on entry", consistent_points_to_graph_p(in));

  if(ENTITY_ASSIGN_P(f)) {
    // FI: you need to dereference this according to in...
    // See assignment01.c
    sinks = expression_to_points_to_sinks(a1, in);
  }
  else if(ENTITY_POINT_TO_P(f)) { // p->a
    // FI: allocation of a fully fresh list? Theroretically...
    list L = expression_to_points_to_sinks(a1, in);
    remove_impossible_arcs_to_null(&L, in);
    pips_assert("in is consistent after computing L",
		consistent_points_to_graph_p(in));
    // a2 must be a field entity
    entity f = reference_variable(syntax_reference(expression_syntax(a2)));
    FOREACH(CELL, pc, L) {
      if(!null_cell_p(pc) && !nowhere_cell_p(pc)) {
	// FI: side effect or allocation of a new cell?
	cell npc = copy_cell(pc);
	if(!heap_cell_p(npc))
	  points_to_cell_add_zero_subscripts(npc);
	(void) points_to_cell_add_field_dimension(npc, f);
	// FI: does this call allocate a full new list?
	type ft = entity_basic_concrete_type(f);
	if(eval_p && !array_type_p(ft)) {
	  list dL = source_to_sinks(npc, in, true);
	  pips_assert("in is consistent after computing dL",
		      consistent_points_to_graph_p(in));
	  free_cell(npc);
	  if(ENDP(dL)) {// FI: this might mean dead code...
	    pips_internal_error("Dereferencing error or user error?\n");
	    if(ENDP(sinks)) {
	      pips_user_warning("Some kind of execution error has been encountered at line %d.\n", points_to_context_statement_line_number());
	      clear_pt_map(in);
	      points_to_graph_bottom(in) = true;
	    }
	  }
	  else {
	    FOREACH(CELL, ec, dL) {
	      // (void) cell_add_field_dimension(ec, f);
	      // FI: should we allocate new cells? Done
	      sinks = gen_nconc(sinks, CONS(CELL, ec, NIL));
	    }
	  }
	}
	else
	  sinks = gen_nconc(sinks, CONS(CELL, npc, NIL));
      }
    }
  }
  else if(ENTITY_FIELD_P(f)) { // p.1
    // Seems to have no impact whatsoever
    list oL = expression_to_points_to_sources(a1, in);
    remove_impossible_arcs_to_null(&oL, in);
    // FI: memory leak, but you need a copy to add the field
    list L = gen_full_copy_list(oL);
    // a2 must be a field entity
    entity f = reference_variable(syntax_reference(expression_syntax(a2)));
    type ft = entity_basic_concrete_type(f);
    FOREACH(CELL, pc, L) {
      if(!null_cell_p(pc)) { // FI: there may be other cells that should
	// not be processed, such as an anywhere non type
	(void) points_to_cell_add_field_dimension(pc, f);
	/* FI: it might be better to integrate this update in
	 * points_to_cell_add_field_dimension() in order to exploit available
	 * information directly and to return a consistent cell.
	 * Anyway, seems useless here
	 */
	points_to_cell_add_zero_subscripts(pc);
      }
      else {
	// FI: Should we removed the arc generating a NULL as it is
	// incompatible with the program execution?
	;
      }
    }
    if(eval_p && !array_type_p(ft)) {
      FOREACH(CELL, pc, L) {
	// No need to check that LL is not empty: NULL might be one of
	// the cells in L, but there may be other cells
	list LL = source_to_sinks(pc, in, true);
	sinks = gen_nconc(sinks, LL);
      }
      gen_full_free_list(L);
    }
    else
      sinks = L;
  }
  // The internal conversion from PLUS_C to PLUS pays attention to
  // pointers but not to arrays
  else if(ENTITY_PLUS_C_P(f) || ENTITY_PLUS_P(f)) { // p+1, a+1
    // FI: this should be conditioned by eval_p==true; else, no sink exists...
    // FI: but we do need something for our lhs expressions, not
    // matter what eval specifies
    if(true || eval_p)
      sinks = expression_to_points_to_sinks_with_offset(a1, a2, in);
    else {
      // pips_internal_error("The request cannot be satisfied.\n");
      if(false && zero_expression_p(a2)) {
	sinks = expression_to_points_to_sources(a1, in);
      }
      else { //FI: an exception or an error code should be raised
	sinks = NIL;
      }
    }
  }
  else if(ENTITY_MINUS_C_P(f) || ENTITY_MINUS_P(f)) {
    entity um = FindOrCreateTopLevelEntity(UNARY_MINUS_OPERATOR_NAME);
    expression ma2 = MakeUnaryCall(um, copy_expression(a2));
    sinks = expression_to_points_to_sinks_with_offset(a1, ma2, in);
    free_expression(ma2);
  }
  else if(ENTITY_PLUS_UPDATE_P(f)) {
    sinks = expression_to_points_to_sinks(a1, in);
    // offset_cells(sinks, a2);
  }
  else if(ENTITY_MINUS_UPDATE_P(f)) {
    sinks = expression_to_points_to_sinks(a1, in);
    /// Already performed elsewhere. The value returned by expression
    // "p -= e" is simply "p", here "a1"
    //entity um = FindOrCreateTopLevelEntity(UNARY_MINUS_OPERATOR_NAME);
    //expression ma2 = MakeUnaryCall(um, copy_expression(a2));
    //offset_cells(sinks, ma2);
    //free_expression(ma2);
  }
  else if (ENTITY_CALLOC_SYSTEM_P(f)) { // CALLOC has two arguments
    // FI: we need a calloc_to_points_to_sinks() to exploit both arguments...
    expression e = binary_intrinsic_expression(MULTIPLY_OPERATOR_NAME,
					       copy_expression(a1),
					       copy_expression(a2));
    sinks = malloc_to_points_to_sinks(e, in);
    free_expression(e);
  }
  else if (ENTITY_REALLOC_SYSTEM_P(f)) { // REALLOC has two arguments
    // FI: see man realloc() for its complexity:-(
    // FI: we need a realloc_to_points_to_sinks() to exploit both arguments...
    sinks = malloc_to_points_to_sinks(a2, in);
  }
  else if(ENTITY_FOPEN_P(f)) {
    /* Should be handled like a malloc, using the line number for malloc() */
    pips_user_warning("Fopen() not precisely implemented.\n");
    type rt = functional_result(type_functional(entity_type(f)));
    type ct = copy_type(type_to_pointed_type(rt)); // FI: no risk with typedefs?
    sinks = CONS(CELL, make_anywhere_cell(ct), NIL);
  }
  else {
    // FI: two options, 1) generate an anywhere as sink to be always safe,
    // 2) raise an internal error to speed up developement... 
    // But do not let go as the caller will block...
    pips_internal_error("Unrecognized operator or function.\n");
    ; // Nothing to do
  }
  if(ENDP(sinks)) {
    pips_user_warning("Some kind of execution error has been encountered.\n");
    clear_pt_map(in);
    points_to_graph_bottom(in) = true;
  }
  pips_assert("in is consistent when returning",
	      consistent_points_to_graph_p(in));
  return sinks;
}

list expression_to_points_to_sinks_with_offset(expression a1, expression a2, pt_map in)
{
  list sinks = NIL;
  type t1 = expression_to_type(a1);
  type t2 = expression_to_type(a2);
  // FI: the first two cases should be unified with a=a1 or a2
  if(pointer_type_p(t1) && (scalar_integer_type_p(t2) || unbounded_expression_p(a2))) {
    // expression_to_points_to_sinks() returns pointers to arcs in the
    // points-to graph. No side effect is then possible.
    list e_sinks = expression_to_points_to_sinks(a1, in);
    sinks = gen_full_copy_list(e_sinks);
    gen_free_list(e_sinks);
    type t = points_to_expression_to_pointed_type(a1);
    offset_points_to_cells(sinks, a2, t);
  }
  else if(pointer_type_p(t2) && (scalar_integer_type_p(t1) || unbounded_expression_p(a1))) {
    list e_sinks = expression_to_points_to_sinks(a2, in);
    sinks = gen_full_copy_list(e_sinks);
    gen_free_list(e_sinks);
    type t = points_to_expression_to_pointed_type(a2);
    offset_points_to_cells(sinks, a1, t);
  }
  else if(array_type_p(t1) && scalar_integer_type_p(t2)) {
    list e_sinks = expression_to_points_to_sources(a1, in);
    sinks = gen_full_copy_list(e_sinks);
    gen_free_list(e_sinks);
    type t = points_to_expression_to_pointed_type(a1);
    offset_points_to_cells(sinks, a2, t);
  }
  else if(array_type_p(t2) && scalar_integer_type_p(t1)) {
    list e_sinks = expression_to_points_to_sources(a2, in);
    sinks = gen_full_copy_list(e_sinks);
    gen_free_list(e_sinks);
    type t = points_to_expression_to_pointed_type(a2);
    offset_points_to_cells(sinks, a1, t);
  }
  else
    pips_internal_error("Not implemented for %p and %p and %p\n", a1, a2, in);
  free_type(t1);
  free_type(t2);
  return sinks;
}

// c?p:q
list ternary_intrinsic_call_to_points_to_sinks(call c,
					       pt_map in,
					       bool eval_p,
					       bool constant_p)
{
  entity f = call_function(c);
  list al = call_arguments(c);
  list sinks = NIL;

  pips_assert("in is consistent", consistent_pt_map_p(in));

  if(ENTITY_CONDITIONAL_P(f)) {
    //bool eval_p = true;
    expression c = EXPRESSION(CAR(al));
    pt_map in_t = full_copy_pt_map(in);
    pt_map in_f = full_copy_pt_map(in);
    in_t = condition_to_points_to(c, in_t, true);
    in_f = condition_to_points_to(c, in_f, false);
    expression e1 = EXPRESSION(CAR(CDR(al)));
    expression e2 = EXPRESSION(CAR(CDR(CDR(al))));
    list sinks1 = NIL;
    if(!points_to_graph_bottom(in_t))
      sinks1 = expression_to_points_to_cells(e1, in_t, eval_p, constant_p);
    list sinks2 = NIL;
    if(!points_to_graph_bottom(in_f))
      sinks2 = expression_to_points_to_cells(e2, in_f, eval_p, constant_p);
    sinks = gen_nconc(sinks1, sinks2);
    // The "in" points-to graph may be enriched by both sub-graphs
    //
    // side-effects on in? memory leak...
    points_to_graph_set(in) = merge_points_to_set(points_to_graph_set(in_t),
						  points_to_graph_set(in_f));
    if(points_to_graph_bottom(in_t) && points_to_graph_bottom(in_f))
      points_to_graph_bottom(in) = true;
    // The free is too deep. References from points-to arcs in in_t
    // and in_f may have been integrated in sinks1 and/or sinks2
    // See conditional05
    clear_pt_map(in_t), clear_pt_map(in_f);
    free_pt_map(in_t), free_pt_map(in_f);
  }
  // FI: any other ternary intrinsics?

  return sinks;
}

// comma operator
list nary_intrinsic_call_to_points_to_sinks(call c, pt_map in)
{
  entity f = call_function(c);
  list sinks = NIL;
  pips_internal_error("Not implemented for %p and %p\n", c, in);
  if(ENTITY_COMMA_P(f)) {
    ;
  }
  return sinks;
}

/* Return NULL as sink */
/* Double definition...
list points_to_null_sinks()
{
  entity ne = entity_null_locations();
  reference nr = make_reference(ne, NIL);
  cell nc = make_cell_reference(nr);
  list sinks = CONS(CELL, nc, NIL);
  return sinks;
}
*/

/* Points-to cannot used any kind of reference, just constant references.
 *
 * For instance, "x[i]" is transformed into "x[*]" because the value
 * of "i" is unknown. x[3+4] may be transformed into x[7].
 *
 * A new referencec is allocated. Reference "r" is unchanged.
 */
reference simplified_reference(reference r)
{
  list sl = reference_indices(r);
  list nsl = NIL;

  FOREACH(EXPRESSION, s, sl) {
    value v = EvalExpression(s);
    expression ns = expression_undefined;
    if(value_constant_p(v) && constant_int_p(value_constant(v))) {
      int cs = constant_int(value_constant(v));
      ns = int_to_expression(cs);
    }
    else {
      ns = make_unbounded_expression();
    }
    nsl = gen_nconc(nsl, CONS(EXPRESSION, ns, NIL));
  }

  entity var = reference_variable(r);
  reference nr = make_reference(var, nsl);
  return nr;
}

/* What to do when a pointer "p" is dereferenced within a reference "r".
 *
 * If p is a scalar pointer, p[i] is equivalent to *(p+i) and p[i][j]
 * to *(*(p+i)+j).
 *
 * If p is a 2-D array of pointers, p[i], p[i][j] do not belong here.
 * But, p[i][j][k] is equivalent to *(p[i][j]+k) and p[i][j][k][l]
 * to *(*(p[i][j]+k)+l).
 *
 * The equivalent expression is fully allocated to be freed at the
 * end. Which may cause problems if the points-to analysis re-use
 * parts of the internal data structure...
 *
 * The normalization could have been performed by the parser, but PIPS
 * is source-to-source for the benefit of its human user.
 */
list pointer_reference_to_points_to_sinks(reference r, pt_map in, bool eval_p)
{
  list sinks = NIL;
  expression pae = pointer_reference_to_expression(r);

  if(eval_p)
    sinks = expression_to_points_to_sinks(pae, in);
  else
    sinks = expression_to_points_to_sources(pae, in);

  free_expression(pae);

  return sinks;
}

 /* Returns a list of memory cells "sinks" possibly accessed by the evaluation
  * of reference "r". No sharing between the returned list "sinks" and
  * the reference "r" or the points-to set "in".
  *
  * Examples if eval_p==false: x->x, t[1]->t[1], t[1][2]->t[1][2], p->p...
  *
  * Examples if eval_p==true: x->error, t[1]->t[1][0],
  * t[1][2]->t[1][2][0], p->p[0]...
  *
  * If constant_p, stored-dependent subscript expressions are replaced
  * by "*" and store-independent expressions are replaced by their
  * values, e.g. x[3+4]->x[7]. Else, subscript expressions are left
  * unchanged. Motivation: constant_p==true for points-to computation,
  * and false for the translation of effects. In the latter case, the
  * transformation into a constant reference is postponed.
  *
  * Issue: let's assume "t" to be an array "int t[10][10[10];". The C
  * language is (too) flexible. If "p" is an "int ***p;", the impact
  * of assignment "p=t;" leads to "p points-to t" or "p points-to
  * t[0]" or "p points-to t[0][0][0]". Two different criteria can be
  * used: the compatibiliy of the pointer type and the pointed cell
  * type, or the equality of the pointer value and of the pointed cell
  * address. 
  *
  * In the first case, t->t[0]->t[0][0]->t[0][0][0].
  *
  * In the second case, t->t[0][0][0], t[0]->t[0][0][0], t[0][0]->t[0][0][0].
  *
  * FI: I do not trust this function. It is already too long. And I am
  * not confident the case disjunction is correct/well chosen.
  */
list reference_to_points_to_sinks(reference r, type et, pt_map in,
				  bool eval_p,
				  bool constant_p)
{
  list sinks = NIL;
  entity e = reference_variable(r);
  type t = entity_basic_concrete_type(e);
  list sl = reference_indices(r);

  ifdebug(8) {
    pips_debug(8, "Reference r = ");
    print_reference(r);
    fprintf(stderr, "\n");
  }

  bool to_be_freed;
  type srt = points_to_reference_to_type(r, &to_be_freed);
  type rt = compute_basic_concrete_type(srt);
  if(false && eval_p && !pointer_type_p(rt)) {
    // There must be a type mismatch
    pips_user_error("Type mmismatch for reference \"%s\" at line %d.",
		    effect_reference_to_string(r),
		    points_to_context_statement_line_number());
  }
  else {

    // FI: conditional01.c shows that the C parser may not generate the
    // right construct when a scalar or an pointer is indexed.
    // FI: maybe more difficult to guess of array of pointers...
    if(pointer_type_p(t) && !ENDP(sl)) {
      sinks = pointer_reference_to_points_to_sinks(r, in, eval_p);
    }
    else if(array_of_pointers_type_p(t)
	    && (int) gen_length(sl)>variable_dimension_number(type_variable(t))) {
      sinks = pointer_reference_to_points_to_sinks(r, in, eval_p);
    }
    else {
      // FI: to be checked otherwise?
      //expression rhs = expression_undefined;
      if (!ENDP(sl)) { // FI: I'm not sure this is a useful disjunction
	/* Two proper possibilities: an array of pointers fully subscribed
	   or any other kind of array partially subscribed. And an
	   unsuitable one: an integer value... */
	int nd = NumberOfDimension(e);
	int rd = (int) gen_length(sl);
	if(nd>rd) {
	  /* No matter what, the target is obtained by adding a 0 subscript */
	  reference nr = copy_reference(r);
	  cell nc = make_cell_reference(nr);
	  for(int i=rd; eval_p && i<nd; i++) { // FI: not efficient
	    expression ze = int_to_expression(0);
	    reference_indices(nr) = gen_nconc(reference_indices(nr),
					      CONS(EXPRESSION, ze, NIL));
	    i = nd; // to be type compatible
	  }
	  sinks = CONS(CELL, nc, NIL);
	}
	else if(nd==rd) {
	  // FI: eval_p is not used here...
	  reference nr = constant_p? simplified_reference(r): copy_reference(r);
	  cell nc = make_cell_reference(nr);
	  if(eval_p) {
	    sinks = source_to_sinks(nc, in, true); // FI: allocate a new copy
	  }
	  else
	    sinks = CONS(CELL, nc, NIL);
	}
	else { // rd is too big
	  // Could be a structure with field accesses expressed as indices
	  // Can be a dereferenced pointer, "p[0]" instead of "*p"
	  type et = ultimate_type(entity_type(e));
	  if(struct_type_p(et)) {
	    reference nr = copy_reference(r);
	    cell nc = make_cell_reference(nr);
	    if(eval_p) {
	      sinks = source_to_sinks(nc, in, true);
	      /*
		expression ze = int_to_expression(0);
		reference_indices(nr) = gen_nconc(reference_indices(nr),
		CONS(EXPRESSION, ze, NIL));
	      */
	    }
	    else
	      sinks = CONS(CELL, nc, NIL);
	  }
	  else if(pointer_type_p(et)) {
	    pips_assert("One subscript", rd==1 && nd==0);
	    /* What is the value of the subscript expression? */
	    //expression sub = EXPRESSION(CAR(reference_indices(r)));
	    // FI: should we try to evaluate the subscript statically?
	    // If the expression is not zero, the target is unchanged but
	    // * must be used as subscript in sinks

	    entity v = reference_variable(r);
	    reference nr = make_reference(v, NIL);
	    cell nc = make_cell_reference(nr);
	    if(eval_p) {
	      // FI: two rounds of source_to_sinks() I guess
	      list sinks_1 = source_to_sinks(nc, in, true);
	      FOREACH(CELL, c, sinks_1) {
		list sinks_2 = source_to_sinks(c, in, true);
		sinks = gen_nconc(sinks, sinks_2);
	      }
	    }
	    else {
	      // FI: what's going to happen with subscript expressions?
	      // FI: strict typing?
	      sinks = source_to_sinks(nc, in, true);
	    }
	    // FI FI FI
	    ;
	  }
	  else {
	    // FI: you may have an array of struct to begin with, and of
	    // structs including other structs
	    // Handle it just like a struct
	    //pips_user_error("Too many subscript expressions for array \"%s\".\n",
	    //		entity_user_name(e));
	    reference nr = copy_reference(r);
	    cell nc = make_cell_reference(nr);
	    if(eval_p) {
	      sinks = source_to_sinks(nc, in, true);
	    }
	    else
	      sinks = CONS(CELL, nc, NIL);
	  }
	}
      }
      else {
	/* scalar case, rhs is already a lvalue */
	if(scalar_type_p(t)) {
	  cell nc = make_cell_reference(copy_reference(r));
	  if(eval_p) {
	    // FI: we have a pointer. It denotes another location.
	    sinks = source_to_sinks(nc, in, true);
	    // FI: in some cases, nc is reused in sinks
	    if(!gen_in_list_p(nc, sinks))
	      free_cell(nc);
	  }
	  else {
	    // FI: without dereferencing
	    sinks = CONS(CELL, nc, NIL);
	  }
	}
	else if(array_type_p(t)) { // FI: not OK with typedef
	  /* An array name can be used as pointer constant */
	  /* We should add null indices according to its number of dimensions */
	  reference nr = copy_reference(r);
	  cell nc = make_cell_reference(nr);
	  if(true) {
	    int n = NumberOfDimension(e);
	    int rd = (int) gen_length(reference_indices(r));
	    int i;
	    // FI: not efficient and quite convoluted!
	    for(i=rd; eval_p && i<n; i++) {
	      reference_indices(nr) =
		gen_nconc(reference_indices(nr),
			  CONS(EXPRESSION, int_to_expression(0), NIL));
	      i = n; // to be type compatible: add at most one subscript
	    }
	  }
	  else {
	    // FI: this is not always a good thing. See arithmetic11.c.
	    // The type of the source is needed to determine the number
	    // of zero subscripts...
	    points_to_cell_add_zero_subscripts(nc);
	  }
	  sinks = CONS(CELL, nc, NIL);
	}
	else {
	  pips_internal_error("Pointer assignment from something "
			      "that is not a pointer.\n Could be a "
			      "function assigned to a functional pointer.\n");
	}
      }
    }

    if(ENDP(sinks)) {
      pips_user_warning("Some kind of execution error has been encountered at line %d.\n",
			points_to_context_statement_line_number());
      clear_pt_map(in);
      points_to_graph_bottom(in) = true;
    }
  }

  pips_assert("The expected type and the reference type are equal",
	      array_pointer_type_equal_p(et, rt));
  check_type_of_points_to_cells(sinks, rt, eval_p);

  if(to_be_freed) free_type(srt);

  ifdebug(8) {
    pips_debug(8, "Resulting cells: ");
    print_points_to_cells(sinks);
    fprintf(stderr, "\n");
  }

  return sinks;
}


/* FI: do we need eval_p? Yes! We may need to produce the source or the sink
 */
list cast_to_points_to_sinks(cast c,
			     type et __attribute__ ((unused)),
			     pt_map in,
			     bool eval_p)
{
  list sinks = list_undefined;
  expression e = cast_expression(c);
  type t = compute_basic_concrete_type(cast_type(c));
  // FI: should we pass down the expected type? It would be useful for
  // heap modelling. No, we might ass well fix the type in the list of
  // sinks returned, especially for malloced buckets.
  /* FI: we need here to return a list of points-to objects so as to
   * warn the user in case the types are not compatible with the
   * property ALIAS_ACROSS_TYPES or to fix the cells allocated in
   * heap. However, this business if more likely to be performed in
   * sinks.c
   */
  if(eval_p) {
    sinks = expression_to_points_to_sinks(e, in);
    if(pointer_type_p(t)) {
      type pt = compute_basic_concrete_type(type_to_pointed_type(t));
      // You may have to update the subscript lists to fit the type
      // This can only be done on *copies* of references included into
      // the arcs of the points-to graph
      FOREACH(CELL, c, sinks) {
	if(!null_cell_p(c) && !nowhere_cell_p(c) && !anywhere_cell_p(c)) {
	  bool to_be_freed;
	  type ct = points_to_cell_to_type(c, &to_be_freed);
	  type cct = compute_basic_concrete_type(ct);
	  if(array_pointer_type_equal_p(pt, cct))
	    ; // nothing to do
	  else {
	    bool m_to_be_freed;
	    type mct = points_to_cell_to_type(c, &m_to_be_freed);
	    if(overloaded_type_p(mct))
	      ; // nothing to do... to be checked
	    else {
	      // You should try to add or remove zero subscripts till the expected
	      // type is reached.
	      reference r = cell_any_reference(c);
	      complete_points_to_reference_with_zero_subscripts(r);
	      adapt_reference_to_type(r, pt);
	      if(to_be_freed) free_type(mct);
	      mct = points_to_cell_to_type(c, &m_to_be_freed);

	      if(array_pointer_type_equal_p(pt, mct))
		; // nothing to do
	      else if(array_type_p(mct)
		      && array_pointer_type_equal_p(pt, array_type_to_element_type(mct)))
		; // nothing to do
	      else {
		pips_internal_error("Not implemented yet.\n");
	      }
	    }
	    if(m_to_be_freed) free_type(mct);
	  }
	  if(to_be_freed) free_type(ct);
	}
      }
    }
    else {
      pips_internal_error("Not implemented yet.\n");
    }
  }
  else {
    sinks = expression_to_points_to_sources(e, in);
    if(pointer_type_p(t)) {
      FOREACH(CELL, c, sinks) {
	if(!null_cell_p(c) && !nowhere_cell_p(c) && !anywhere_cell_p(c)) {
	  bool to_be_freed;
	  type ct = points_to_cell_to_type(c, &to_be_freed);
	  type cct = compute_basic_concrete_type(ct);
	  if(array_pointer_type_equal_p(t, cct))
	    ; // nothing to do
	  else { // We should create a stub of the proper type because a
	    // reference cannot include the address-of operator, "&"
	    if(pointer_type_p(t) && array_type_p(cct)) {
	      reference r = cell_any_reference(c);
	      complete_points_to_reference_with_zero_subscripts(r);
	      adapt_reference_to_type(r, t);
	    }
	    else
	      pips_internal_error("Not implemented yet: stub with proper type needed.\n");
	  }
	}
      }
    }
    else {
      // FI: should we make sure that we do not tranformer pointers
      // into integers or other types?
      ;
    }
  }
  return sinks;
}

// FI: do we need eval_p? eval_p is assumed always true
list sizeofexpression_to_points_to_sinks(sizeofexpression soe,
					 type et __attribute__ ((unused)),
					 pt_map in)
{
  list sinks = NIL;
  // FI: seems just plain wrong for a sink
  // pips_internal_error("Not implemented yet");
  if( sizeofexpression_expression_p(soe) ){
    expression ne = sizeofexpression_expression(soe);
    sinks = expression_to_points_to_sinks(ne, in);
  }
  if( sizeofexpression_type_p(soe) ){
    type t = compute_basic_concrete_type(sizeofexpression_type(soe));
    // FI: a better job could be done. A stub should be allocated in
    // the formal context of the procedure
    if(pointer_type_p(t)) {
      type pt = compute_basic_concrete_type(type_to_pointed_type(t));
      cell c = make_anywhere_cell(pt);
      sinks = CONS(CELL, c, NIL);
    }
    else
      pips_internal_error("Unexpected type.\n");
  }
  return sinks;
}


/* Heap modelling 
 *
 * Rather than passing many arguments, keep a heap_context. May not be
 * a good idea for cast information...
 *
 * The API should contain renaming function(s) to move the effects up
 * the call graph....
 */
// Current statement where the malloc is performed
static statement malloc_statement = statement_undefined;
// number of malloc occurences already met in the malloc statement
static int malloc_counter = 0;

void init_heap_model(statement s)
{
  malloc_statement = s;
  malloc_counter = 0;
}

void reset_heap_model()
{
  malloc_statement = statement_undefined;
  malloc_counter = 0;
}

statement get_heap_statement()
{
  statement s = malloc_statement;
  return s;
}

int get_heap_counter()
{
  return ++malloc_counter;
}

/* Heap modelling
 *
 * FI: lots of issues here; the potential cast is lost...
 *
 * FI: the high-level switch I have added to understand the management
 * of options is performed at a much much lower level, which may be
 * good or not. I do not think it's good for readbility, but factoring
 * is good. See malloc_to_abstract_location()
 *
 * e is the arguments of the malloc call...
 *
 * Basic heap modelling
 *
 * ABSTRACT_HEAP_LOCATIONS:
 *
 *  "unique": do not generate heap abstract
 * locations for each line number...
 *
 * "insensitive": control path are not taken into account; all
 * malloc() located in one function are equivalent.
 *
 * "flow-sensitive": take into account the line number or the occurence number?
 *
 * "context-sensitive": take into account the call stack; not implemented
 *
 * ALIASING_ACROSS_TYPE: you should have one heap per type if aliasing
 * across type is forbidden. The impact of its composition with the
 * previous property is not specified...
 */
list malloc_to_points_to_sinks(expression e,
			       pt_map in __attribute__ ((unused)))
{
  list sinks = NIL;
  const char * opt = get_string_property("ABSTRACT_HEAP_LOCATIONS");
  //bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  if(same_string_p(opt, "unique")) {
    sinks = unique_malloc_to_points_to_sinks(e);
  }
  else if(same_string_p(opt, "insensitive")) {
    sinks = insensitive_malloc_to_points_to_sinks(e);
  }
  else if(same_string_p(opt, "flow-sensitive")) {
    sinks = flow_sensitive_malloc_to_points_to_sinks(e);
  }
  else if(same_string_p(opt, "context-sensitive")) {
    // Context sensitivity is dealt with in the translation functions, not here
    sinks = flow_sensitive_malloc_to_points_to_sinks(e);
  }
  else {
    pips_user_error("Unexpected value \"%s\" for Property ABSTRACT_HEAP_LOCATION."
		    "Possible values are \"unique\", \"insensitive\","
		    "\"flow-sensitive\", \"context-sensitive\".\n", opt);
  }

  return sinks;
}

/* FI->AM: is "unique" multiple when ALIASING_ACROSS_TYPE is set to false? 
 *
 * FI->AM: the comments in pipsmake-rc.tex are not sufficient to
 * understand what the choices are.
 *
 * If ALIASING_ACROSS_TYPES, return an overloaded unique heap entity
 */
list unique_malloc_to_points_to_sinks(expression e)
{
  list m = NIL;
  if(get_bool_property("ALIASING_ACROSS_TYPES")) {
    /* We need only one HEAP abstract location: Pointers/assign03 */
    m = flow_sensitive_malloc_to_points_to_sinks(e);
  }
  else {
    /* We need one HEAP abstract location per type: Pointers/assign02
     *
     * Note: we must be careful about dereferencing and fields...
     */
    m = flow_sensitive_malloc_to_points_to_sinks(e);
  }
  return m;
}

/* FI->AM: what's the difference with the previous option? Reference
 * to your dissertation?
 */
list insensitive_malloc_to_points_to_sinks(expression e)
{
  list m = NIL;
  // FI: I'm waiting for this error to happen
  pips_internal_error("Not implemented yet?");
  m = flow_sensitive_malloc_to_points_to_sinks(e);
  return m;
}

list flow_sensitive_malloc_to_points_to_sinks(expression e)
{
  // expression sizeof_exp = EXPRESSION (CAR(call_arguments(expression_call(rhs))));
  // FI: kind of dumb since it is int or size_t
  // FI: the expected type should be passed down all these function calls...
  // type t = expression_to_type(e);
  /*
  reference nr = original_malloc_to_abstract_location(e,
						      type_undefined,
						      type_undefined,
						      e,
						      get_current_module_entity(),
						      get_heap_statement());
  */

  // FI: the heap number is not yet used
  sensitivity_information si =
    make_sensitivity_information(get_heap_statement(), 
				 get_current_module_entity(),
				 NIL);

  // FI: why use &si instead of si?
  entity me = malloc_to_abstract_location(e, &si);
  reference mr = reference_undefined;
  if(!entity_array_p(me)) {
    mr = make_reference(me, NIL);
  }
  else
    mr = make_reference(me, CONS(EXPRESSION, int_to_expression(0), NIL));

  cell mc = make_cell_reference(mr);
  list sinks  = CONS(CELL, mc, NIL);

  return sinks;
}

list application_to_points_to_sinks(application a,
				    type et __attribute__ ((unused)),
				    pt_map in)
{
  expression f = application_function(a);
  // list args = application_arguments(a);
  type t = expression_to_type(f);
  entity ne = entity_undefined;
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  pips_user_warning("Case application is not correctly handled &p and %p\n",
		    a, in);

  if(type_sensitive_p)
    ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION, t);
  else
    ne = entity_all_xxx_locations(ANYWHERE_LOCATION);

  reference nr = make_reference(ne, NIL);
  cell nc = make_cell_reference(nr);
  list sinks = CONS(CELL, nc, NIL);

  // FI: free_type(t); ?

  return sinks;
}

// Duplicate?
/*
list sizeofexpression_to_points_to_sinks(sizeofexpression soe, pt_map in)
{
  list sinks = NIL;
  pips_user_warning("Not implemented yet for %p and %p\n", soe, in);
  return sinks;
}
*/

 /* Allocate a new expression based on the reference in "c" and the
  * subscript list "csl". No sharing with the arguments. The pointer
  * subscripts are transformed into pointer arithmetic and
  * dereferencements.
  */
expression pointer_subscript_to_expression(cell c, list csl)
{
  reference r = copy_reference(cell_any_reference(c));
  expression pae = reference_to_expression(r);

  FOREACH(EXPRESSION, pse, csl) {
    expression npse = copy_expression(pse);
    pae = binary_intrinsic_expression(PLUS_C_OPERATOR_NAME, pae, npse);
    pae = unary_intrinsic_expression(DEREFERENCING_OPERATOR_NAME, pae);
  }
  return pae;
}

/* FI: a really stupid function... Why do we add zero subscript right
 *  away when building the sink cell to remove them later? Let's now
 * remove the excessive subscripts of "r" with respect to type
 * "at"...
 */
void adapt_reference_to_type(reference r, type et)
{
  bool to_be_freed;
  type at = compute_basic_concrete_type(et);
  type rt = points_to_reference_to_type(r, &to_be_freed);
  type t = compute_basic_concrete_type(rt);
  while(!array_pointer_type_equal_p(at, t) && !ENDP(reference_indices(r))) {
    if(to_be_freed) free_type(t);
    list sl = reference_indices(r);
    list last = gen_last(sl);
    expression e = EXPRESSION(CAR(last));
    if(expression_field_p(e))
      break;
    int l1 = (int) gen_length(sl);
    gen_remove_once(&sl, (void *) e);
    int l2 = (int) gen_length(sl);
    if(l1==l2)
      pips_internal_error("gen_remove() is ineffective.\n");
    reference_indices(r) = sl;
    type nrt = points_to_reference_to_type(r, &to_be_freed);
    t = compute_basic_concrete_type(nrt);
  }
  if(!array_pointer_type_equal_p(at, t)) {
    pips_user_warning("There may be a typing error at line %d (e.g. improper malloc call).",
		      points_to_context_statement_line_number());
    pips_internal_error("Cell type mismatch.");
  }
  if(to_be_freed) free_type(t);
}

 /* Generate the corresponding points-to reference(s). All access
  * operators such as ., ->, * are replaced by subscripts.
  *
  * See Strict_typing.sub/assigment11.c: the index is not put back at
  * the right place. It would be easy (?) to fix it in this specific
  * case, not forgetting the field subscripts..., but I do not see how
  * to handle general stubs with artificial dimensions...
  */
list subscript_to_points_to_sinks(subscript s,
				  type et,
				  pt_map in,
				  bool eval_p)
{
  expression a = subscript_array(s);
  bool to_be_freed;
  // If ever a pointer is deferenced somewhere, at is not going to
  // take into account the extra dimensions needed for pointer arithmetics
  type at = points_to_expression_to_type(a, &to_be_freed);

  /* FI: In many cases, you do need the source. However, you have
   * different kind of sources because "x" and "&x[0]" are synonyms
   * and because you sometimes need "x" and some other times "&x[0]".
   */
  list sources = expression_to_points_to_sources(a, in);

  list sl = subscript_indices(s);
  list csl = subscript_expressions_to_constant_subscript_expressions(sl);
  list sinks = NIL;
  list i_sources = NIL;
  bool eval_performed_p = false;

  /* If the first dimension is unbounded, it has (probably) be added
     because of a pointer. A zero subscript is also added. */
  bool strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
  if(array_type_p(at) && !strict_p) {
    variable v = type_variable(at);
    dimension d1 = DIMENSION(CAR(variable_dimensions(v)));
    if(unbounded_dimension_p(d1)) {
      /* Add a zero subscript */
      expression z = make_zero_expression();
      csl = CONS(EXPRESSION, z, csl);
    }
  }

  /* Add subscript when possible. For typing reason, typed anywhere
     cell should be subscripted. */
  FOREACH(CELL, c, sources) {
    // FI: some other lattice abstract elements should be removed like
    // STACK, DYNAMIC
    if(!nowhere_cell_p(c) && !null_cell_p(c) && !anywhere_cell_p(c)
       && !all_heap_locations_cell_p(c)) {
      bool to_be_freed2;
      type t = points_to_cell_to_type(c, &to_be_freed2);
      reference r = cell_any_reference(c);
      entity v = reference_variable(r);

      if(array_type_p(at)) {
	list ncsl = gen_full_copy_list(csl);
	if(entity_stub_sink_p(v)) {
	  // argv03
	  reference r = cell_any_reference(c);

	  // FI: an horror... fixing a design mistake by a kludge...
	  // useful for argv03, disastrous for dereferencing08
	  if(!points_to_array_reference_p(r))
	    adapt_reference_to_type(r, at);

	  // FI: the update depends on the sink model
	  // points_to_reference_update_final_subscripts(r, ncsl);
	  reference_indices(r) = gen_nconc(reference_indices(r), ncsl);

	  // FI: let's add the zero subscripts again...
	  // points_to_cell_add_zero_subscripts(c);
	  complete_points_to_reference_with_zero_subscripts(r);
	}
	else {
	  // Expression "a" does not require any dereferencing, add
	  // the new indices (Strict_typing.sub/assignment11.c

	  // FI: an horror... fixing a design mistake by a kludge...
	  adapt_reference_to_type(r, at);
	  reference_indices(r) = gen_nconc(reference_indices(r), ncsl);
	}

	i_sources = CONS(CELL, c, i_sources);
      }
      else if(struct_type_p(at)) { // Just add the subscript, old version
	list ncsl = gen_full_copy_list(csl);
	if(entity_stub_sink_p(v)) {
	  reference r = cell_any_reference(c);
	  // FI: the update depends on the sink model
	  points_to_reference_update_final_subscripts(r, ncsl);
	}
	else {
	  // Expression "a" does not require any dereferencing, add
	  // the new indices (Strict_typing.sub/assignment11.c
	  reference_indices(r) = gen_nconc(reference_indices(r), ncsl);
	}
	i_sources = CONS(CELL, c, i_sources);
      }
      else if(pointer_type_p(at)) {
	reference r = cell_any_reference(c);
	if(points_to_array_reference_p(r)) {
	  /* Add the subscripts */
	  list ncsl = gen_full_copy_list(csl);
	  reference_indices(r) = gen_nconc(reference_indices(r), ncsl);
	  eval_performed_p = true;
	  i_sources = CONS(CELL, c, i_sources);
	}
	else {
	  /* The reference "p[i][j]" is transformed into an expression
	     "*(*(p+i)+j)" if "p" is really a pointer expression, not a
	     partial array reference. */
	  expression pae = pointer_subscript_to_expression(c, csl);
	  i_sources = expression_to_points_to_sources(pae, in);
	  free_expression(pae);
	}
      }
      else {
	pips_internal_error("Unexpected case.\n");
      }
      if(to_be_freed2) free_type(t);
    }
  }

  gen_full_free_list(csl);
  gen_free_list(sources);

  if(eval_p && !eval_performed_p) {
    FOREACH(CELL, source, i_sources) {
      bool to_be_freed;
      type t = points_to_cell_to_type(source, &to_be_freed);
      reference r = cell_any_reference(source);
      if(points_to_array_reference_p(r)) {
	//points_to_cell_add_zero_subscripts(source);
	//adapt_reference_to_type(r, at);
	expression z = make_zero_expression();
	reference_indices(r) = gen_nconc(reference_indices(r),
					 CONS(EXPRESSION, z, NIL));
	sinks = gen_nconc(sinks, CONS(CELL, source, NIL));
      }
      else if(pointer_type_p(t)) {
	// A real pointer, not an under-subscripted array
	list pointed = source_to_sinks(source, in, true);
	sinks = gen_nconc(sinks, pointed);
      }
      else {
	// FI: Pretty bad wrt sharing and memory leaks
	sinks = gen_nconc(sinks, CONS(CELL, source, NIL));
      }
      if(to_be_freed) free_type(t);
    }
  }
  else
    sinks = i_sources;

  if(to_be_freed) free_type(at);

  if(ENDP(sinks)) {
    pips_user_warning("Some kind of execution error has been encountered.\n");
    clear_pt_map(in);
    points_to_graph_bottom(in) = true;
  }

  check_type_of_points_to_cells(sinks, et, eval_p);

  return sinks;
}

list range_to_points_to_sinks(range r, pt_map in)
{
  list sinks = NIL;
  pips_user_warning("Not implemented yet for %p and %p\n", r, in);
  return sinks;
}

/* Return a possibly empty list of abstract locations whose addresses
 * are possible value of expression "e" evaluated with points-to
 * information "in".
 *
 * Expression "e" is assumed to evaluate as a lhs, i.e. some memory
 * address. If not, an empty list is returned.
 *
 * Additional information could be passed in a second pass analysis,
 * e.g. preconditions.
 *
 * The generated sinks are all constant memory paths. A more advanced
 * analysis could use storage-sentitive information, that would have
 * to be updated with effects and transformers.
 *
 * The list returned should be fully allocated with no sharing between
 * it and the in points-to set. Hopefully...
 */
list expression_to_points_to_cells(expression e,
				   pt_map in,
				   bool eval_p,
				   bool constant_p)
{
  /* reference + range + call + cast + sizeofexpression + subscript +
     application*/
  list sinks = NIL;
  bool to_be_freed;
  type et = points_to_expression_to_type(e, &to_be_freed);
  type cet = compute_basic_concrete_type(et);
  if(to_be_freed) free_type(et);
  tag tt ;
  syntax s = expression_syntax(e);

  switch (tt = syntax_tag(s)) {
  case is_syntax_reference: {
    reference r = syntax_reference(s);
    sinks = reference_to_points_to_sinks(r, cet, in, eval_p, constant_p);
    break;
  }
  case is_syntax_range: {
    range r = syntax_range(s);
    sinks = range_to_points_to_sinks(r, in);
    break;
  }
  case  is_syntax_call: {
    call c = syntax_call(s);
    sinks = call_to_points_to_sinks(c, cet, in, eval_p, constant_p);
    break;
  }
  case  is_syntax_cast: {
    cast c = syntax_cast(s);
    sinks = cast_to_points_to_sinks(c, cet, in, eval_p);
    break;
  }
  case  is_syntax_sizeofexpression: {
    // FI: no sink should be returned...
    //sinks = sizeofexpression_to_points_to_sinks(st, rhs, lhs, in);
    break;
  }
  case  is_syntax_subscript: {
    subscript sub = syntax_subscript(s);
    sinks = subscript_to_points_to_sinks(sub, cet, in, eval_p);
    break;
  }
  case  is_syntax_application: {
    application a = syntax_application(s);
    sinks = application_to_points_to_sinks(a, cet, in);
    break;
  }
  case  is_syntax_va_arg: {
    // FI: useful?
    //pips_internal_error("Not implemented yet\n");
    list soel = syntax_va_arg(s);
    //sizeofexpression soev = SIZEOFEXPRESSION(CAR(soel));
    sizeofexpression soet = SIZEOFEXPRESSION(CAR(CDR(soel)));
    sinks = sizeofexpression_to_points_to_sinks(soet, cet, in);
    break;
  }
  default:
    pips_internal_error("unknown expression tag %d\n", tt);
    break;
  }

  return sinks;
}

/* The returned list contains cells used in "in". They should be copied
 * if they must be changed by side effects or "in" will become
 * inconsistent.
 *
 * This function computes the possible constant values of a pointer expression.
 */
list expression_to_points_to_sinks(expression e, pt_map in)
{
  list sinks = expression_to_points_to_cells(e, in, true, true);

  return sinks;
}

list expression_to_points_to_sources(expression e, pt_map in)
{
  list sinks = expression_to_points_to_cells(e, in, false, true);

  /* Scalar pointers are expected but [0] subscript may have been added */
  if(!expression_to_points_to_cell_p(e))
    sinks = reduce_cells_to_pointer_type(sinks);
  ifdebug(1) {
    bool to_be_freed;
    type tmp_t = points_to_expression_to_type(e, &to_be_freed);
    type et = compute_basic_concrete_type(tmp_t);
    FOREACH(CELL, c, sinks) {
      if(!null_cell_p(c)) {
	bool to_be_freed2;
	type ct = points_to_cell_to_type(c, &to_be_freed2);
	type cct = compute_basic_concrete_type(ct);
	// Type compatibility check. To be improved with integer used
	// as pointer values...
	if(!array_pointer_type_equal_p(et, cct)
	   // Dereferencement via a subscript
	   && !array_element_type_p(et, cct)
	   // Constant strings such as "hello"
	   && !((char_star_type_p(et) || string_type_p(et))
		&& char_star_constant_function_type_p(cct))
	   // Dereferencement forced by the syntax of the expression
	   && !(pointer_type_p(et)
		&& array_pointer_type_equal_p(type_to_pointed_type(et), cct))) {
	  /* A useless [0] may have been added, but it is supposed to
	     be taken care of above... by callers of this function. */
	  ifdebug(1) {
	    pips_debug(1, "Type mismatch for expression: "); print_expression(e);
	    fprintf(stderr, " with type: "); print_type(et);
	    fprintf(stderr, "\nand cell: "); print_points_to_cell(c);
	    fprintf(stderr, " with type: "); print_type(cct);
	    fprintf(stderr, "\n");

	  }
	  print_expression(e);
	  pips_internal_error("Type error for an expression\n.");
	}
	if(to_be_freed2) free_type(ct);
      }
    }
    if(to_be_freed) free_type(tmp_t);
  }

  return sinks;
}

bool reference_must_points_to_null_p(reference r, pt_map in)
{
  list sinks = reference_to_sinks(r, in, false);
  bool must_p = false;

  if(gen_length(sinks)==1) {
    // It is a must arc
    cell c = CELL(CAR(sinks));
    must_p = null_cell_p(c);
  }
  gen_free_list(sinks);
  return must_p;
}
