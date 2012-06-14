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
    pt_out = expression_to_points_to(a, pt_out);
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
 * evaluation of a possibly empty list of expression. A new data
 * structure is allocated.
 *
 * The result is correct only if you are sure that all expressions in
 * "el" are always evaluated.
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
 * information. E.g. a[*(p++)]
 *
 * I'm surprised that pointers can be indexed instead of being subscripted...
 */
pt_map reference_to_points_to(reference r, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  list sel = reference_indices(r);
  entity v = reference_variable(r);
  type t = ultimate_type(entity_type(v));
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

/* /\* Three different kinds of calls are distinguished: */
/*  * */
/*  * - calls to constants, e.g. NULL, */
/*  * */
/*  * - calls to intrinsics, e.g. ++ or malloc(), */
/*  * */
/*  * - and calls to a user function. */
/*  *\/ */
/* pt_map call_to_points_to(call c, pt_map pt_in) */
/* { */
/*   pt_map pt_out = pt_in; */

/*   entity f = call_function(c); */
/*   list al = call_arguments(c); */
/*   type ft = entity_type(f); */
/*   type rt = type_undefined; */
/*   if(type_functional_p(ft)) { */
/*     functional ff = type_functional(ft); */
/*     rt = functional_result(ff); */
/*   } */
/*   else if(type_variable_p(ft)) { */
/*     /\* Must be a pointer to a function *\/ */
/*     if(pointer_type_p(ft)) { */
/*       /\* I do not know if nft must be freed later *\/ */
/*       type nft = type_to_pointed_type(ft); */
/*       pips_assert("Must be a function", type_functional_p(nft)); */
/*       functional nff = type_functional(nft); */
/*       rt = functional_result(nff); */
/*     } */
/*     else */
/*       pips_internal_error("Unexpected type.\n"); */
/*   } */
/*   else */
/*     pips_internal_error("Unexpected type.\n"); */

/*   if(ENTITY_STOP_P(f)||ENTITY_ABORT_SYSTEM_P(f)||ENTITY_EXIT_SYSTEM_P(f) */
/*      || ENTITY_ASSERT_FAIL_SYSTEM_P(f)) { */
/*     clear_pt_map(pt_out); */
/*   } */
/*   else if(ENTITY_C_RETURN_P(f)) { */
/*     /\* it is assumed that only one return is present in any module */
/*        code because internal returns are replaced by gotos *\/ */
/*     if(ENDP(al)) { */
/*       // clear_pt_map(pt_out); */
/*       ; // the necessary projections are performed elsewhere */
/*     } */
/*     else { */
/*       expression rhs = EXPRESSION(CAR(al)); */
/*       type rhst = expression_to_type(rhs); */
/*       // FI: should we use the type of the current module? */
/*       if(pointer_type_p(rhst)) { */
/* 	list sinks = expression_to_points_to_sinks(rhs, pt_out); */
/* 	entity rv = function_to_return_value(get_current_module_entity()); */
/* 	reference rvr = make_reference(rv, NIL); */
/* 	cell rvc = make_cell_reference(rvr); */
/* 	list sources = CONS(CELL, rvc, NIL); */
/* 	pt_out = list_assignment_to_points_to(sources, sinks, pt_out); */
/* 	gen_free_list(sources); */
/* 	// FI: not too sure about "sinks" being or not a memory leak */
/* 	; // The necessary projections are performed elsewhere */
/*       } */
/*       free_type(rhst); */
/*     } */
/*   } */
/*   else if(ENTITY_FCLOSE_P(f)) { */
/*     expression lhs = EXPRESSION(CAR(al)); */
/*     pt_out = freed_pointer_to_points_to(lhs, pt_out); */
/*   } */
/*   else if(ENTITY_FREE_SYSTEM_P(f)) { */
/*     expression lhs = EXPRESSION(CAR(al)); */
/*     pt_out = freed_pointer_to_points_to(lhs, pt_out); */
/*   } */
/*   else if(ENTITY_CONDITIONAL_P(f)) { */
/*     // FI: I needs this piece of code for assert(); */
/*     expression c = EXPRESSION(CAR(al)); */
/*     pt_map in_t = full_copy_pt_map(pt_out); */
/*     pt_map in_f = full_copy_pt_map(pt_out); */
/*     in_t = condition_to_points_to(c, in_t, true); */
/*     in_f = condition_to_points_to(c, in_f, true); */
/*     expression e1 = EXPRESSION(CAR(CDR(al))); */
/*     expression e2 = EXPRESSION(CAR(CDR(CDR(al)))); */
/*     pt_map out_t = pt_map_undefined; */
/*     if(!empty_pt_map_p(in_t)) */
/*       out_t = expression_to_points_to(e1, in_t); */
/*     pt_map out_f = pt_map_undefined; */
/*     // FI: should be factored out in a more general merge function... */
/*     if(!empty_pt_map_p(in_f)) */
/*       out_f = expression_to_points_to(e2, in_f); */
/*     if(empty_pt_map_p(in_t)) */
/*       pt_out = out_f; */
/*     else if(empty_pt_map_p(in_f)) */
/*       pt_out = out_t; */
/*     else */
/*       pt_out = merge_points_to_set(out_t, out_f); */
/*     // FI: this destroys pt_out for test case pointer02 */
/*     //free_pt_map(in_t), free_pt_map(in_f), free_pt_map(out_t), free_pt_map(out_f); */
/*   } */
/*   else { */
/*     if(ENTITY_DEREFERENCING_P(f) || ENTITY_POINT_TO_P(f)) { */
/*       /\* Is the dereferenced pointer null or undefined? *\/ */
/*       expression p = EXPRESSION(CAR(al)); */
/*       pt_out = dereferencing_to_points_to(p, pt_out);  */
/*     } */
/*     if(!type_void_p(rt)) { */
/*       value fv = entity_initial(f); */

/*       /\* points-to updates due to arguments *\/ */
/*       // FI: this cannot be delayed but it is unfortunately applied */
/*       // again when going down? See arithmetic08 and 09? */
/*       // This is necessary but cannot be placed here because of the */
/*       // recursive calls */
/*       // FI: we are in trouble for post increment and post decrement... */
/*       // We should update the target a second time in sinks.c! */
/*       pt_out = expressions_to_points_to(al, pt_in); */

/*       // FI: I wanted to use the return type but it is too often */
/*       // overloaded with intrinsics */
/*       type ct = call_to_type(c); */
/*       if(pointer_type_p(ct) || struct_type_p(ct)) { */
/* 	/\* points-to updates due to the function itself *\/ */
/* 	if(entity_constant_p(f)) { */
/* 	  // pt_out = constant_call_to_points_to(c, pt_out); */
/* 	  pt_out = pt_in; */
/* 	} */
/* 	else if(intrinsic_entity_p(f)) */
/* 	  pt_out = intrinsic_call_to_points_to(c, pt_out); */
/* 	else if(symbolic_entity_p(f)) */
/* 	  pt_out = pt_in; // FI? */
/* 	else if(value_unknown_p(fv)) { */
/* 	  pips_internal_error("function %s has an unknown value\n", */
/* 			      entity_name(f)); */
/* 	} */
/* 	else { */
/* 	  // must be a user-defined function */
/* 	  pips_assert("f is a user-defined function", value_code_p(entity_initial(f))); */
/* 	  pt_out = user_call_to_points_to(c, pt_out); */
/* 	} */
/*       } */
/*       else { */
/* 	/\* points-to updates due to arguments: already performed *\/ */
/* 	// pt_out = expressions_to_points_to(al, pt_in); */
/* 	; */
/*       } */
/*       free_type(ct); */
/*     } */
/*     else { */
/*       /\* points-to updates due to arguments *\/ */
/*       pt_out = expressions_to_points_to(al, pt_in); */
/*     } */
/*   } */

/*   return pt_out; */
/* } */

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
  tag tt;
  entity f = call_function(c);
  list al = call_arguments(c);
  type ft = entity_type(f);
  type rt = type_undefined;
  if(type_functional_p(ft)) {
    functional ff = type_functional(ft);
    rt = functional_result(ff);
    /* points-to updates due to arguments */
    // FI: this cannot be delayed but it is unfortunately applied
    // again when going down? See arithmetic08 and 09?
    // This is necessary but cannot be placed here because of the
    // recursive calls
    // FI: we are in trouble for post increment and post decrement...
    // We should update the target a second time in sinks.c!
    pt_out = expressions_to_points_to(al, pt_in);
    switch( tt = value_tag(entity_initial(f))) {
    case is_value_code:{
      pips_assert("f is a user-defined function", value_code_p(entity_initial(f)));
      pt_out = user_call_to_points_to(c, pt_out);
    }
      break;
    case is_value_unknown:
      pips_internal_error("function %s has an unknown value\n",
			  entity_name(f));
      break;
    case is_value_intrinsic:
      pt_out = intrinsic_call_to_points_to(c, pt_in);
	break;
    case is_value_constant:
      pt_out = pt_in; // FI?
      break;
    case is_value_symbolic:{
      value v = entity_initial(f);
      symbolic s = value_symbolic(v);
      expression ex = symbolic_expression(s);
      pt_out = expression_to_points_to(ex, pt_in);
    }
      break;
    case is_value_expression:{
      value v = entity_initial(f);
      expression ex = value_expression(v);
      pt_out = expression_to_points_to(ex, pt_in);
    }
      break;
    default:
      pips_internal_error("unknown tag %d\n", tt);
      break;
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
    pt_out = expressions_to_points_to(al, pt_in);
  else
    pips_internal_error("Unexpected type.\n");



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
    if(pointer_type_p(lhst)) {
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
    if(pointer_type_p(lhst)) {
      expression delta = int_to_expression(1);
      pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out);
      free_expression(delta);
    }
    free_type(lhst);
  }
  else if(ENTITY_POST_DECREMENT_P(f) || ENTITY_PRE_DECREMENT_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    type lhst = expression_to_type(lhs);
    if(pointer_type_p(lhst)) {
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
  }else if(ENTITY_ASSERT_FAIL_SYSTEM_P(f)) {
// FI: I needs this piece of code for assert();
    expression c = EXPRESSION(CAR(al));
    pt_map in_t = full_copy_pt_map(pt_out);
    pt_map in_f = full_copy_pt_map(pt_out);
    in_t = condition_to_points_to(c, in_t, true);
    in_f = condition_to_points_to(c, in_f, true);
    expression e1 = EXPRESSION(CAR(CDR(al)));
    expression e2 = EXPRESSION(CAR(CDR(CDR(al))));
    pt_map out_t = pt_map_undefined;
    if(!empty_pt_map_p(in_t))
      out_t = expression_to_points_to(e1, in_t);
    pt_map out_f = pt_map_undefined;
    // FI: should be factored out in a more general merge function...
    if(!empty_pt_map_p(in_f))
      out_f = expression_to_points_to(e2, in_f);
    if(empty_pt_map_p(in_t))
      pt_out = out_f;
    else if(empty_pt_map_p(in_f))
      pt_out = out_t;
    else
      pt_out = merge_points_to_set(out_t, out_f);
    // FI: this destroys pt_out for test case pointer02
    //free_pt_map(in_t), free_pt_map(in_f), free_pt_map(out_t), free_pt_map(out_f);
  }
  else if(ENTITY_STOP_P(f)||ENTITY_ABORT_SYSTEM_P(f)||ENTITY_EXIT_SYSTEM_P(f)
     /* || ENTITY_ASSERT_FAIL_SYSTEM_P(f) */) {
    clear_pt_map(pt_out);
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
      if(pointer_type_p(rhst)) {
	list sinks = expression_to_points_to_sinks(rhs, pt_out);
	entity rv = function_to_return_value(get_current_module_entity());
	reference rvr = make_reference(rv, NIL);
	cell rvc = make_cell_reference(rvr);
	list sources = CONS(CELL, rvc, NIL);
	pt_out = list_assignment_to_points_to(sources, sinks, pt_out);
	gen_free_list(sources);
	// FI: not too sure about "sinks" being or not a memory leak
	; // The necessary projections are performed elsewhere
      }
      free_type(rhst);
    }
  }
  else if(ENTITY_FCLOSE_P(f)) {
    expression lhs = EXPRESSION(CAR(al));
    pt_out = freed_pointer_to_points_to(lhs, pt_out);
  }
  else if(ENTITY_CONDITIONAL_P(f)) {
    // FI: I needs this piece of code for assert();
    expression c = EXPRESSION(CAR(al));
    pt_map in_t = full_copy_pt_map(pt_out);
    pt_map in_f = full_copy_pt_map(pt_out);
    in_t = condition_to_points_to(c, in_t, true);
    in_f = condition_to_points_to(c, in_f, true);
    expression e1 = EXPRESSION(CAR(CDR(al)));
    expression e2 = EXPRESSION(CAR(CDR(CDR(al))));
    pt_map out_t = pt_map_undefined;
    if(!empty_pt_map_p(in_t))
      out_t = expression_to_points_to(e1, in_t);
    pt_map out_f = pt_map_undefined;
    // FI: should be factored out in a more general merge function...
    if(!empty_pt_map_p(in_f))
      out_f = expression_to_points_to(e2, in_f);
    if(empty_pt_map_p(in_t))
      pt_out = out_f;
    else if(empty_pt_map_p(in_f))
      pt_out = out_t;
    else
      pt_out = merge_points_to_set(out_t, out_f);
    // FI: this destroys pt_out for test case pointer02
    //free_pt_map(in_t), free_pt_map(in_f), free_pt_map(out_t), free_pt_map(out_f);
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

  return pt_out;
}


/* pt_map intrinsic_call_to_points_to(call c, pt_map pt_in) */
/* { */
/*   pt_map pt_out = pt_in; */
/*   entity f = call_function(c); */

/*   list al = call_arguments(c); */

/*   //set_methods_for_proper_simple_effects(); */
/*   //list el = call_to_proper_effects(c); */
/*   //generic_effects_reset_all_methods(); */

/*   pips_debug(5, "intrinsic function \"%s\"\n", entity_name(f)); */

/*   // FI: short term version */
/*   // pt_out = points_to_intrinsic(statement_undefined, c, f, al, pt_in, el); */
/*   // return pt_out; */

/*   // FI: Where should we check that the update is linked to a pointer? */
/*   // Should we go down because a pointer assignment may be hidden anywhere... */
/*   // Or have we already taken care of this in call_to_points_to() */

/*   if(ENTITY_ASSIGN_P(f)) { */
/*     expression lhs = EXPRESSION(CAR(al)); */
/*     expression rhs = EXPRESSION(CAR(CDR(al))); */
/*     pt_out = assignment_to_points_to(lhs, rhs, pt_out); */
/*   } */
/*   else if (ENTITY_FREE_SYSTEM_P(f)) { */
/*     expression lhs = EXPRESSION(CAR(al)); */
/*     pt_out = freed_pointer_to_points_to(lhs, pt_out); */
/*   } */
/*    // According to C standard, pointer arithmetics does not change */
/*   // the targeted object. */
/*   else if(ENTITY_PLUS_UPDATE_P(f)) { */
/*     expression lhs = EXPRESSION(CAR(al)); */
/*     type lhst = expression_to_type(lhs); */
/*     if(pointer_type_p(lhst)) { */
/*       expression delta = EXPRESSION(CAR(CDR(al))); */
/*       pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out); */
/*     } */
/*     free_type(lhst); */
/*   } */
/*   else if(ENTITY_MINUS_UPDATE_P(f)) { */
/*     expression lhs = EXPRESSION(CAR(al)); */
/*     type lhst = expression_to_type(lhs); */
/*     if(pointer_type_p(lhst)) { */
/*       expression rhs = EXPRESSION(CAR(CDR(al))); */
/*       entity um = FindOrCreateTopLevelEntity(UNARY_MINUS_OPERATOR_NAME); */
/*       expression delta = MakeUnaryCall(um, copy_expression(rhs)); */
/*       pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out); */
/*       free_expression(delta); */
/*     } */
/*     free_type(lhst); */
/*   } */
/*   else if(ENTITY_POST_INCREMENT_P(f) || ENTITY_PRE_INCREMENT_P(f)) { */
/*     expression lhs = EXPRESSION(CAR(al)); */
/*     type lhst = expression_to_type(lhs); */
/*     if(pointer_type_p(lhst)) { */
/*       pt_out = dereferencing_to_points_to(lhs, pt_out); */
/*       expression delta = int_to_expression(1); */
/*       pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out); */
/*       free_expression(delta); */
/*     } */
/*     free_type(lhst); */
/*   } */
/*   else if(ENTITY_POST_DECREMENT_P(f) || ENTITY_PRE_DECREMENT_P(f)) { */
/*     expression lhs = EXPRESSION(CAR(al)); */
/*     type lhst = expression_to_type(lhs); */
/*     if(pointer_type_p(lhst)) { */
/*       pt_out = dereferencing_to_points_to(lhs, pt_out); */
/*       expression delta = int_to_expression(-1); */
/*       pt_out = pointer_arithmetic_to_points_to(lhs, delta, pt_out); */
/*       free_expression(delta); */
/*     } */
/*     free_type(lhst); */
/*   } */
/*   else { */
/*     // FI: fopen(), fclose() should be dealt with */
/*     // fopen implies that its path argument is not NULL, just like a test */
/*     // fclose implies that its fp argument is not NULL on input and */
/*     // points to undefined on output. */

/*     // Not safe till all previous tests are defined */
/*     // It is assumed that other intrinsics do not generate points-to arcs... */
/*     // pips_internal_error("Not implemented yet\n"); */
/*     pt_out = pt_in; */
/*   } */

/*   return pt_out; */
/* } */

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
    offset_cells(source, sinks, delta, pt_out);
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

/* Each cell in sinks is replaced by a cell located "delta" elements
 * further up in the memory. In some cases, the same points-to are
 * removed and added. For instance, t[0],t[1] -> t[1],t[2] because of
 * a p++, and t[1] is removed and added.
 *
 * This procedure must be used when cells in "sinks" are components of
 * points-to arcs stored in a points-to set.
 */
void offset_cells(cell source, list sinks, expression delta, pt_map in)
{
  pt_map old = new_pt_map();
  pt_map new = new_pt_map();
  FOREACH(CELL, sink, sinks) {
    points_to pt = find_arc_in_points_to_set(source, sink, in);
    add_arc_to_pt_map(pt, old);
    points_to npt = offset_cell(pt, delta);
    add_arc_to_pt_map(npt, new);
  }
  difference_of_pt_maps(in, in, old);
  union_of_pt_maps(in, in, new);
}

/* Allocate and return a new points-to "npt", copy of "pt", with an
 * offset of "delta" on the sink.
 *
 * Some kind of k-limiting should be performed here to avoid creating
 * too many new nodes in the points-to graph, such as t[0], t[1],... A
 * fix point t[*] should be used when too may nodes already exist.
 *
 * Since "sink" is used to compute the key in the hash table used to
 * represent set "in", it is not possible to perform a side effect on
 * "sink" without removing and reinserting the corresponding arc.
 */
points_to offset_cell(points_to pt, expression delta)
{
  /* "&a[i]" should be transformed into "&a[i+eval(delta)]" when
     "delta" can be statically evaluated */
  points_to npt = copy_points_to(pt);
  reference r = cell_any_reference(points_to_sink(npt));
  entity v = reference_variable(r);
  cell sink = points_to_sink(npt);
  if(nowhere_cell_p(sink))
    ; // user error: possible incrementation of an uninitialized pointer
  else if(null_cell_p(sink))
    ; // Impossible: possible incrementation of a NULL pointer
  else if(anywhere_cell_p(sink))
    ; // It is already fuzzy no need to add more
  // FI: it might be necessary to exclude *HEAP* too when a minimal
  // heap model is used (ABSTRACT_HEAP_LOCATIONS = "unique")
  else if(entity_array_p(v)
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
	  int k = get_int_property("POINTS_TO_SUBSCRIPT_LIMIT");
	  if(-k <= ov && ov <= k) {
	    expression nse = int_to_expression(dv+ov);
	    EXPRESSION_(CAR(gen_last(sl))) = nse;
	  }
	  else {
	    expression nse = make_unbounded_expression();
	    EXPRESSION_(CAR(gen_last(sl))) = nse;
	  }
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
  return npt;
}

/* Each cell in sinks is replaced by a cell located "delta" elements
 * further up in the memory.
 *
 * Similar to offset_cells(), but, in spite of the name, cannot be
 * used with points-to cells that are part of a points-to belonging to
 * a points-to set.
 */
void offset_points_to_cells(list sinks, expression delta)
{
  FOREACH(CELL, sink, sinks) {
    offset_points_to_cell(sink, delta);
  }
}

/* FI: offset_cell() has been derived from this function. Some
 * factoring out should be performed.
 *
 * The naming is all wrong: offset_points_to_cell() can operate on a
 * cell, while offset_cell() is designed to operate on a cell
 * component of a points-to.
 */
void offset_points_to_cell(cell sink, expression delta)
{
  /* "&a[i]" should be transformed into "&a[i+eval(delta)]" when
     "delta" can be statically evaluated */
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
  else if(entity_array_p(v)
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
	  int k = get_int_property("POINTS_TO_SUBSCRIPT_LIMIT");
	  if(-k <= ov && ov <= k) {
	    expression nse = int_to_expression(dv+ov);
	    EXPRESSION_(CAR(gen_last(sl))) = nse;
	  }
	  else {
	    expression nse = make_unbounded_expression();
	    EXPRESSION_(CAR(gen_last(sl))) = nse;
	  }
	  free_expression(lse);
	}
	else {
	  // If the index cannot be computed, used the unbounded expression
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

  type ut = ultimate_type(t);
  if(pointer_type_p(ut))
    pt_out = pointer_assignment_to_points_to(lhs, rhs, pt_out);
  else if(struct_type_p(ut))
    pt_out = struct_assignment_to_points_to(lhs, rhs, pt_out);
  // FI: unions are not dealt with...
  else
    pt_out = pt_in; // What else?

  if(to_be_freed)
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

  list L = expression_to_points_to_sources(lhs, pt_out);

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

  /* Retrieve the memory locations that might be reached by the rhs
   *
   * Update the calling context by adding new stubs linked directly or
   * indirectly to the formal parameters and global variables if
   * necessary.
   */
  list R = expression_to_points_to_sinks(rhs,pt_out);

  if(ENDP(L) || ENDP(R)) {
    //pips_assert("Left hand side reference list is not empty.\n", !ENDP(L));
    //pips_assert("Right hand side reference list is not empty.\n", !ENDP(R));

  // FI: where do we want to check for dereferencement of
  // nowhere/undefined and NULL? Here? Or within
  // list_assignment_to_points_to?

    /* We must be in a dead-code portion. If not pleased, adjust properties... */
    clear_pt_map(pt_out);
  }
  else
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
 * If the freed location r is precisely known, any arc pointing
 * towards it can be removed:
 *
 * Kill_2 = {pts=(l,r,a) in in | r in R && |R|=1}
 *
 * If the freed location r is precisely known, any arc pointing
 * from it can be removed:
 *
 * Kill_3 = {pts=(l,r,a) in in | l in R && |R|=1}
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
 * PML = {source_to_sinks(r)|r in R}
 * ML = {m|m in PML && heap_cell_p(m)}
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
  list PML = NIL;

  list L = expression_to_points_to_sources(lhs, pt_out);
  list R = expression_to_points_to_sinks(lhs, pt_out);

  /* Build a nowhere cell
   *
   * FI->AM: typed nowhere?
   */
  //list N = CONS(CELL, make_nowhere_cell(), NIL);
  type t = expression_to_type(lhs);
  type pt = type_to_pointed_type(t);
  list N = CONS(CELL, make_typed_nowhere_cell(pt), NIL);
  free_type(t);

  pips_assert("L is not empty", !ENDP(L));

  /* Remove cells from R that do not belong to Heap: they cannot be
     freed */
  list nhl = NIL;
  // list inds = NIL;
  FOREACH(CELL, c, R) {
    /* FI->AM: Should be taken care of by the lattice...
     *
     * We need heap_cell_p() for any abstract bucket, heap_cell_p() to
     * detect the heap abstract location, heap_cell_p() to detect
     * cells that may be in the heap, i.e. abstract locations that are
     * greater then the heap abstract location.
     */
    /* if c is a heap location with indices other than zero then we have bumped into 
       a non-legal free */
    if(heap_cell_p(c)) {
      reference r = cell_any_reference(c);
      list inds = reference_indices(r);
      if(!ENDP(inds)) {
	expression ind = EXPRESSION (CAR(inds));
	if(!expression_null_p(ind))
	  nhl = CONS(CELL, c, nhl);
      }
      // gen_free_list(inds);
    }
    if(!heap_cell_p(c) && !cell_typed_anywhere_locations_p(c))
      nhl = CONS(CELL, c, nhl);
  }
  gen_list_and_not(&R, nhl);
  gen_free_list(nhl);
  //pips_assert("R is not empty", !ENDP(R));

  if(ENDP(R)) {
    /* We have bumped into a non-legal free such as free(&i). See test
       case malloc10.c */
    clear_pt_map(pt_out);
  }
  else {

    /* Memory leak detection... Must be computed early, before pt_out
       has been (too?) modified. Transitive closure not performed... */
    if(gen_length(R)==1) {
      FOREACH(CELL, c, R) {
	bool to_be_freed;
	type ct = points_to_cell_to_type(c, &to_be_freed);
	if(pointer_type_p(ct) || struct_type_p(ct)
	   || array_of_pointers_type_p(ct)
	   || array_of_struct_type_p(ct)) {
	  // FI: this might not work for arrays of pointers?
	  // Many for of "source" can be developped when we are dealing
	  // struct and arrays
	  // FI: do we need a specific version of source_to_sinks()?
	  entity v = reference_variable(cell_any_reference(c));
	  //cell nc = make_cell_reference(make_reference(v, NIL));
	  PML = variable_to_sinks(v, pt_out, true);
	  FOREACH(CELL, m, PML) {
	    if(heap_cell_p(m)) {
	      entity b = reference_variable(cell_any_reference(m));
	      pips_user_warning("Memory leak for bucket \"%s\".\n",
				entity_name(b));
	      // The impact of this memory leak should be computed
	      // transitively and recursively
	      // FI->AM: we could almost call recursively
	      // freed_freed_pointer_to_points_to() with a reference to b...
	    }
	  }
	  //free_cell(nc);
	}
	if(to_be_freed) free_type(ct);
      }
    }

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
	  if(!null_cell_p(r) && !anywhere_cell_p(r) && !nowhere_cell_p(r)) {
	    /* FI: should be easier and more efficient to substitute the
	       sink... But is is impossible with the representation of
	       the points-to set. */
	    /*
	      cell source = copy_cell(points_to_source(pts));
	      cell sink = make_nowhere_cell();
	      approximation a = copy_approximation(points_to_approximation(pts));
	      points_to npts = make_points_to(source, sink, a, make_descriptor_none());
	      add_arc_to_pt_map(npts, pt_out);
	    */
	    // FI: pv_whileloop05, lots of related cells to remove after a free...
	    // FI: assuming you can perform the removal inside the loop...
	    remove_arc_from_pt_map(pts, pt_out);
	    {
	      cell source = points_to_source(pts);
	      // FI: it should be a make_typed_nowhere_cell()
	      bool to_be_freed;
	      type t = points_to_cell_to_type(source, &to_be_freed);
	      type pt = type_to_pointed_type(t);
	      cell sink = make_typed_nowhere_cell(pt);
	      //approximation a = make_approximation_may(); // FI: why may?
	      approximation a = copy_approximation(points_to_approximation(pts));
	      points_to npt = make_points_to(source, sink, a,
					     make_descriptor_none());
	      add_arc_to_pt_map(npt, pt_out);
	      /* Notify the user that the source of the new nowhere points to relation
		 is a dangling pointer */
	      entity b = reference_variable(cell_any_reference(source));
	      pips_user_warning("Dangling pointer \"%s\".\n",
				entity_name(b));
	      //free_points_to(pts);
	      if(to_be_freed) free_type(t);
	    }
	  }
	}
      }
    }

    /* Remove Kill_3 if it is not empty by definition */
    if(gen_length(R)==1) {
      SET_FOREACH(points_to, pts, pt_out) {
	cell l = points_to_source(pts);
	if(related_points_to_cell_in_list_p(l, R)) {
	  // Potentially memory leaked cell:
	  //cell r = points_to_sink(pts);
	  remove_arc_from_pt_map(pts, pt_out);
	}
      }
    }

    /* Add Gen_1 - Not too late since pt_out has aready been modified? */
    pt_out = list_assignment_to_points_to(L, N, pt_out);

    /* Add Gen_2: useless, already performed by Kill_2 */
    /*
      SET_FOREACH(points_to, pts, pt_out) {
      cell r = points_to_sink(pts);
      if(!null_cell_p(r) && points_to_cell_in_list_p(r, R)) {
      cell source = copy_cell(points_to_source(pts));
      cell sink = make_nowhere_cell();
      approximation a = copy_approximation(points_to_approximation(pts));
      points_to npts = make_points_to(source, sink, a, make_descriptor_none());
      add_arc_to_pt_map(npts, pt_out);
      }
      }
    */

    /*
     * Other pointers may or must now be dangling because their target
     * has been freed. Already detected at the level of Gen_2.
     */
  }

// FI: memory leak(s) in this function?
  gen_free_list(L);
  gen_free_list(N);
  gen_full_free_list(R);
  gen_free_list(PML);
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
  /* Check possible dereferencing errors */
  list ndl = NIL; // null dereferencing error list
  list udl = NIL; // undefined dereferencing error list
  bool singleton_p = (gen_length(L)==1);
  FOREACH(CELL, c, L) {
    if(nowhere_cell_p(c)){
      udl = CONS(CELL, c, udl);
      if(singleton_p)
	// Not necessarily a user error if the code is dead
	// Should be controlled by an extra property...
	pips_user_warning("Dereferencing of an undefined pointer.\n");
      else
	pips_user_warning("Dereferencing of an undefined pointer.\n");
    }
    else if(null_cell_p(c)) {
      ndl = CONS(CELL, c, ndl);
      if(singleton_p)
	// Not necessarily a user error if the code is dead
	// Should be controlled by an extra property...
	pips_user_warning("Dereferencing of a null pointer.\n");
      else
	pips_user_warning("Dereferencing of a null pointer.\n");
    }
  }

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
  }
  else {
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
  pt_map kill/* = new_pt_map()*/;
  // FI->AM: do we really want to keep the same arc with two different
  // approximations? The whole business of may/must does not seem
  // useful. 
  // union_of_pt_maps(kill, kill_may, kill_must);
  kill = kill_must;
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

  // FI->AM: use kill_may to reduce the precision of these arcs
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

  free_pt_maps(in_may, in_must,
	       kill_may, kill_must,
	       gen_may, gen_must,
	       gen,/* kill,*/ NULL);
  // clear_pt_map(pt_out); // FI: why not free?
  }

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
	      type ft = entity_type(f); // field type
	      type uft = ultimate_type(ft);
	      bool array_p = array_type_p(ft) || array_type_p(uft);
	      if(!array_p && (pointer_type_p(uft) || struct_type_p(uft))) {
		reference lr = copy_reference(cell_any_reference(lc));
		reference rr = copy_reference(cell_any_reference(rc));
		/* FI: conditionally add zero subscripts necessary to
		   move from an array "a" to its first element,
		   e.g. a[0][0][0] */
		reference_add_zero_subscripts(lr, lt);
		reference_add_zero_subscripts(rr, rt);
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
	      else if(array_p && (array_of_pointers_type_p(uft)
				  || pointer_type_p(uft)
				  || array_of_struct_type_p(uft)
				  || struct_type_p(uft))) {
		// Same as above, but an unbounded subscript is added...
		// Quite a special assign in C...
		reference lr = copy_reference(cell_any_reference(lc));
		reference rr = copy_reference(cell_any_reference(rc));
		reference_add_field_dimension(lr, f);
		reference_add_field_dimension(rr, f);
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
  syntax cs = expression_syntax(c);

  if(syntax_reference_p(cs)) {
    /* For instance, C short cut "if(p)" for "if(p!=NULL)" */
    out = reference_condition_to_points_to(syntax_reference(cs), in, true_p);
  }
  else if(syntax_call_p(cs)) {
    out = call_condition_to_points_to(syntax_call(cs), in, true_p);
  }
  else {
    pips_internal_error("Not implemented yet.\n");
  }
  return out;
}

/* Handle conditions such as "if(p)" */
pt_map reference_condition_to_points_to(reference r, pt_map in, bool true_p)
{
  pt_map out = in;
  entity v = reference_variable(r);
  type vt = ultimate_type(entity_type(v));
  list sl = reference_indices(r);

  /* Take care of side effects in references */
  out = expressions_to_points_to(sl, out);

  /* are we dealing with a pointer? */
  if(pointer_type_p(vt)) {
    if(true_p) {
      /* if p points to NULL, the condition is not feasible. If not,
	 remove any arc from p to NULL */
      if(reference_must_points_to_null_p(r, in)) {
	// FI: memory leak with clear_pt?
	pips_user_warning("Dead code detected.\n");
	clear_pt_map(out);
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
      /* remove any arc from v to anything and add an arc from p to NULL */
      points_to_source_projection(in, v);
      /* Make a points-to NULL and remove the arc from the current out */
      cell source = make_cell_reference(copy_reference(r));
      cell sink = make_null_pointer_value_cell();
      points_to a = make_points_to(source, sink, make_approximation_exact(),
				   make_descriptor_none());
      add_arc_to_pt_map(a, in);
    }
  }

  return out;
}

/* Handle any condition that is a call such as "if(p!=q)", "if(*p)",
 * "if(foo(p=q))"... */
pt_map call_condition_to_points_to(call c, pt_map in, bool true_p)
{
  pt_map out = in;
  entity f = call_function(c);
  value fv = entity_initial(f);
  if(value_intrinsic_p(fv))
    out = intrinsic_call_condition_to_points_to(c, in, true_p);
  else if(value_code_p(fv))
    out = user_call_condition_to_points_to(c, in, true_p);
  else if(value_constant_p(fv)) {
    // For instance "if(1)"
    ; // do nothing
  }
  else
    // FI: you might have an apply on a functional pointer?
    pips_internal_error("Not implemented yet.\n");
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
  else {
    if(ENTITY_DEREFERENCING_P(f) || ENTITY_POINT_TO_P(f)
       || ENTITY_POST_INCREMENT_P(f) || ENTITY_POST_DECREMENT_P(f)
       || ENTITY_PRE_INCREMENT_P(f) || ENTITY_PRE_DECREMENT_P(f)) {
      expression p = EXPRESSION(CAR(call_arguments(c)));
      /* Make sure that all dereferencements are possible? Might be
	 included in intrinsic_call_to_points_to()... */
      dereferencing_to_points_to(p, in);
    }
    // Take care of side effects as in "if(*p++)"
    out = intrinsic_call_to_points_to(c, in);
    //pips_internal_error("Not implemented yet.\n");
  }
  return out;
}

pt_map user_call_condition_to_points_to(call c, pt_map in, bool true_p)
{
  pt_map out = in;
  // FI: a call site to handle like any other user call site...
  pips_user_warning("Interprocedural points-to not implemented yet. "
		    "Call site ignored. %p %p %p.\n",
		    c, in , true_p);
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
    out = merge_points_to_set(out1, out2);
    free_pt_map(out2);
  }
  else
    pips_internal_error("Not implemented yet for boolean operator \"%s\".\n",
			entity_local_name(f));
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

bool cell_is_xxx_p(cell c1, cell c2, int xxx)
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

/* Update the points-to information "in" according to the validity of
 * the condition.
 *
 * FI: It is not clear what should be done. We can remove the arcs or
 * some of the arcs that violate the condition or decide that the
 * condition cannot be true... I've put a first attempt at resolving
 * the issue for pointer comparisons, using the approximation exact or
 * not.
 */
pt_map relational_intrinsic_call_condition_to_points_to(call c, pt_map in, bool true_p)
{
  pt_map out = in;
  entity f = call_function(c);
  list al = call_arguments(c);
  if((ENTITY_EQUAL_P(f) && true_p)
     || (ENTITY_NON_EQUAL_P(f) && !true_p)) {
    expression lhs = EXPRESSION(CAR(al));
    type lhst = expression_to_type(lhs);
    expression rhs = EXPRESSION(CAR(CDR(al)));
    type rhst = expression_to_type(rhs);
    if(pointer_type_p(lhst) || pointer_type_p(rhst)) {
      list L = expression_to_points_to_sources(lhs, in);
      list R = expression_to_points_to_sources(rhs, in);
      if(gen_length(L)==1 && gen_length(R)==1) {
	cell l = CELL(CAR(L));
	cell r = CELL(CAR(R));
	cell source = cell_undefined, sink = cell_undefined;
	if(null_cell_p(l)) {
	  source = r;
	  sink = l;
	}
	else if(null_cell_p(r)) {
	  source = l;
	  sink = r;
	}
	if(!cell_undefined_p(source)) {
	  entity v = reference_variable(cell_any_reference(source));
	  out = points_to_source_projection(out, v);
	  points_to a = make_points_to(source, sink, make_approximation_exact(),
				     make_descriptor_none());
	  add_arc_to_pt_map(a, out);
	}
      }
    }
    free_type(lhst), free_type(rhst);
    ; //FI FI FI
  }
  else if((ENTITY_EQUAL_P(f) && !true_p)
     || (ENTITY_NON_EQUAL_P(f) && true_p)) {
    // FI: this code is almost identical to the code above
    // It should be shared with a more general test first and then a
    // precise test to decide if you add or remove arcs
    expression lhs = EXPRESSION(CAR(al));
    type lhst = expression_to_type(lhs);
    expression rhs = EXPRESSION(CAR(CDR(al)));
    type rhst = expression_to_type(rhs);
    if(pointer_type_p(lhst) || pointer_type_p(rhst)) {
      list L = expression_to_points_to_sources(lhs, in);
      list R = expression_to_points_to_sources(rhs, in);
      if(gen_length(L)==1 && gen_length(R)==1) {
	cell l = CELL(CAR(L));
	cell r = CELL(CAR(R));
	cell source = cell_undefined, sink = cell_undefined;
	if(null_cell_p(l)) {
	  source = r;
	  sink = l;
	}
	else if(null_cell_p(r)) {
	  source = l;
	  sink = r;
	}
	if(!cell_undefined_p(source)) {
	  // FI: we should be able to remove an arc regardless of its
	  // approximation... Not so simple as it's part of the key
	  // used by the hash table!
	  points_to a = make_points_to(copy_cell(source),
				       copy_cell(sink),
				       make_approximation_exact(),
				       make_descriptor_none());
	  remove_arc_from_pt_map(a, out);
	  free_points_to(a);
	  a = make_points_to(copy_cell(source), copy_cell(sink),
			     make_approximation_may(),
			     make_descriptor_none());
	  remove_arc_from_pt_map(a, out);
	  free_points_to(a);
	}
      }
    }
    free_type(lhst), free_type(rhst);
    ;//FI FI FI
  }
  if(ENTITY_LESS_OR_EQUAL_P(f)
     || ENTITY_GREATER_THAN_P(f)
     || ENTITY_GREATER_THAN_P(f)
     || ENTITY_GREATER_THAN_P(f)) {
    bool (*cmp_function)(cell, cell);
    if((ENTITY_LESS_OR_EQUAL_P(f) && true_p)
       || (ENTITY_GREATER_THAN_P(f) && !true_p)) {
      // cmp_function = cell_is_less_than_or_equal_to_p;
      cmp_function = cell_is_greater_than_p;
    }
    if((ENTITY_LESS_OR_EQUAL_P(f) && !true_p)
       || (ENTITY_GREATER_THAN_P(f) && true_p)) {
      // cmp_function = cell_is_greater_than_p;
      cmp_function = cell_is_less_than_or_equal_to_p;
    }
    if((ENTITY_GREATER_OR_EQUAL_P(f) && true_p)
       || (ENTITY_LESS_THAN_P(f) && !true_p)) {
      // cmp_function = cell_is_greater_than_or_equal_to_p;
      cmp_function = cell_is_less_than_p;
    }
    if((ENTITY_GREATER_OR_EQUAL_P(f) && !true_p)
       || (ENTITY_LESS_THAN_P(f) && true_p)) {
      //cmp_function = cell_is_less_than_p;
      cmp_function = cell_is_greater_than_or_equal_to_p;
    }
    expression lhs = EXPRESSION(CAR(al));
    type lhst = expression_to_type(lhs);
    expression rhs = EXPRESSION(CAR(CDR(al)));
    type rhst = expression_to_type(rhs);
    if(pointer_type_p(lhst) || pointer_type_p(rhst)) {
      list L = expression_to_points_to_sinks(lhs, in);
      if(gen_length(L)==1) { // FI: one fixed bound
	cell lc = CELL(CAR(L));
	list RR = expression_to_points_to_sources(rhs, in);
	SET_FOREACH(points_to, pt, out) {
	  cell source = points_to_source(pt);
	  if(cell_in_list_p(source, RR)) {
	    cell sink = points_to_sink(pt);
	    if(cmp_function(lc, sink)) {
	      approximation a = points_to_approximation(pt);
	      if(approximation_exact_p(a)) 
		/* The condition cannot violate an exact arc. */
		clear_pt_map(out);
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
	  SET_FOREACH(points_to, pt, out) {
	    cell source = points_to_source(pt);
	    if(cell_in_list_p(source, LL)) {
	      cell sink = points_to_sink(pt);
	      if(cmp_function(sink, rc)) {
		// FI: Oops in middle of the iterator...
		approximation a = points_to_approximation(pt);
		if(approximation_exact_p(a)) 
		  /* The condition cannot violate an exact arc. */
		  clear_pt_map(out);
		else
		  remove_arc_from_pt_map(pt, out);
	      }
	    }
	  }
	}
      }
    }
    free_type(lhst), free_type(rhst);
    ; //FI FI FI
  }
  else {
    // Do nothing for other relational operators such as ">"
    ; // pips_internal_error("Not implemented yet.\n");
  }
  return out;
}
