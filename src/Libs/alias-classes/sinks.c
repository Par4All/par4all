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

list points_to_null_sinks()
{
  /* The null location is not typed. The impact on dependence test is
     not clear. */
  entity ne = entity_null_locations();
  return entity_to_sinks(ne);
}

list call_to_points_to_sinks(call c, pt_map in)
{
  list sinks = NIL;
  entity f = call_function(c);
  //list al = call_arguments(c);
  value v = entity_initial(f);
  tag tt = value_tag(v);
  switch (tt) {
  case is_value_code:
    sinks = user_call_to_points_to_sinks(c, in);
    break;
  case is_value_symbolic:
    break;
  case is_value_constant: {
    constant zero = value_constant(v);
    if(constant_int_p(zero) && constant_int(zero)==0)
      sinks = points_to_null_sinks();
  }
    break;
  case is_value_unknown:
    pips_internal_error("function %s has an unknown value\n",
                        entity_name(f));
    break;
  case is_value_intrinsic: 
    // FI: here is the action, &p, *p, p->q, p.q, etc...
    sinks = intrinsic_call_to_points_to_sinks(c, in);
    break;
  default:
    pips_internal_error("unknown value tag %d\n", tt);
    break;
  }

  return sinks;
}

/*
 * Sinks: "malloc(exp)", "p", "p++", "p--", "p+e", "p-e" , "p=e", "p, q, r",
 * "p->q", "p.q",...
 *
 * "(cast) p" is not an expression.
 */
list intrinsic_call_to_points_to_sinks(call c, pt_map in)
{
  list sinks = NIL;
  //entity f = call_function(c);
  list al = call_arguments(c);
  int nary = (int) gen_length(al);

  // Switch on number of arguments to avoid long switch on character
  // string or memoizing of intrinsics
  switch(nary) {
  case 0:
    pips_internal_error("Probably a constant or a symbolic. Not handled here\n");
    break;
  case 1:
    sinks = unary_intrinsic_call_to_points_to_sinks(c, in);
    break;
  case 2:
    sinks = binary_intrinsic_call_to_points_to_sinks(c, in);
    break;
  case 3:
    sinks = ternary_intrinsic_call_to_points_to_sinks(c, in);
    break;
  default:
    sinks = nary_intrinsic_call_to_points_to_sinks(c, in);
    break;
  }

  return sinks;
}

// malloc, &p, p++, p--, ++p, --p
list unary_intrinsic_call_to_points_to_sinks(call c, pt_map in)
{
  entity f = call_function(c);
  list al = call_arguments(c);
  expression a = EXPRESSION(CAR(al));
  list sinks = NIL;
  pips_assert("One argument", gen_length(al)==1);
  // pips_internal_error("Not implemented for %p and %p\n", c, in);
  if (ENTITY_MALLOC_SYSTEM_P(f) ||
      ENTITY_CALLOC_SYSTEM_P(f)) {
    sinks = malloc_to_points_to_sinks(a, in);
  }
  else if(ENTITY_ADDRESS_OF_P(f)) {
    sinks = expression_to_constant_paths(statement_undefined, a, in);
   }
  else if(ENTITY_PRE_INCREMENT_P(f)) {
    pips_internal_error("Not implemented yet.\n");
     ;
   }
  else if(ENTITY_PRE_DECREMENT_P(f)) {
    pips_internal_error("Not implemented yet.\n");
     ;
   }
  else if(ENTITY_POST_INCREMENT_P(f)) {
    pips_internal_error("Not implemented yet.\n");
     ;
   }
  else if(ENTITY_POST_DECREMENT_P(f)) {
    pips_internal_error("Not implemented yet.\n");
     ;
   }
  else {
  // FI: to be continued
    pips_internal_error("Unexpected unary pointer operator\n");
  }

  return sinks;
}

// p=q, p.x, p->y, p+e, p-e, p+=e, p-=e
// What other binary operator could be part of a lhs expression?
list binary_intrinsic_call_to_points_to_sinks(call c, pt_map in)
{
  entity f = call_function(c);
  list al = call_arguments(c);
  expression a1 = EXPRESSION(CAR(al));
  expression a2 = EXPRESSION(CAR(CDR(al)));
  list sinks = NIL;
  pips_internal_error("Not implemented for %p and %p and %p\n", c, in, a2);

  if(ENTITY_ASSIGN_P(f)) {
    // FI: you need to dereference this according to in...
    sinks = expression_to_points_to_sinks(a1, in);
  }
  else if(ENTITY_POINT_TO_P(f)) {
    ;
  }
  else if(ENTITY_FIELD_P(f)) {
    ;
  }
  else if(ENTITY_PLUS_C_P(f)) {
    ; 
  }
  else if(ENTITY_MINUS_C_P(f)) {
    ; 
  }
  else if(ENTITY_PLUS_UPDATE_P(f)) {
    ; 
  }
  else if(ENTITY_MINUS_UPDATE_P(f)) {
    ;
  }
  else {
    ; // Nothing to do
  }
  return sinks;
}

// c?p:q
list ternary_intrinsic_call_to_points_to_sinks(call c, pt_map in)
{
  entity f = call_function(c);
  list sinks = NIL;
  pips_internal_error("Not implemented for %p and %p\n", c, in);
  if(ENTITY_CONDITIONAL_P(f)) {
    ;
  }
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

 /* Debug: print a cell list for points-to. Parameter f is not useful
    in a debugging context. */
void fprint_points_to_cell(FILE * f __attribute__ ((unused)), cell c)
{
  reference r = cell_any_reference(c);
  print_reference(r);
}

/* Debug: use stderr */
void print_points_to_cell(cell c)
{
  fprint_points_to_cell(stderr, c);
}

/* Debug */
void print_points_to_cells(list cl)
{
  if(ENDP(cl))
    fprintf(stderr, "Empty cell list\n");
  FOREACH(CELL, c, cl) {
    print_points_to_cell(c);
    if(!ENDP(CDR(cl)))
      fprintf(stderr, ", ");
  }
  fprintf(stderr, "\n");
}


 /* Returns a list of memory cells */
list reference_to_points_to_sinks(reference r, pt_map in)
{
  list sinks = NIL;

  ifdebug(8) {
    pips_debug(8, "Reference r = ");
    print_reference(r);
  }

  // FI: to be checked otherwise?
  expression rhs = expression_undefined;
  if (array_argument_p(rhs)) {
    sinks = array_to_constant_paths(rhs, in);
  }
  else {
    entity e = reference_variable(r);
    /* scalar case, rhs is already a lvalue */
    /* add points-to relations in demand for global pointers */
    if( pointer_type_p(ultimate_type(entity_type(e)) )) {
      if( top_level_entity_p(e) || formal_parameter_p(e) ) {
	cell nc = make_cell_reference(r);
	if (!source_in_pt_map_p(nc, in)) {
	  pt_map tmp = formal_points_to_parameter(nc);
	  // FI: lots of trouble here
	  in = union_of_pt_maps(in, in, tmp);
	  // FI: Amira says that the free is shallow...
	  clear_pt_map(tmp);
	  free_pt_map(tmp);
	}
      }
      /* FI/FC : I dropped the current statement from many signature,
	 assuming it is not necessary to exploit in because we are
	 dealing only for the current statement and because we do no
	 need an involved statement. However, this may not be
	 compatible with the current data structure... To be checked
	 with Amira and Fabien. */
      sinks = expression_to_constant_paths(statement_undefined, rhs, in);
    }
  }

  ifdebug(8) {
    pips_debug(8, "Resulting cells: ");
    print_points_to_cells(sinks);
  }

  return sinks;
}

list user_call_to_points_to_sinks(call c, pt_map in __attribute__ ((unused)))
{
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
  type t = entity_type(call_function(c));
  entity ne = entity_undefined;
  list sinks = NIL;

  /* FI: definitely the intraprocedural version */
  if(type_sensitive_p)
    ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
  else
    ne = entity_all_xxx_locations(ANYWHERE_LOCATION);
  
  sinks = entity_to_sinks(ne);
  return sinks;
}

list cast_to_points_to_sinks(cast c, pt_map in)
{
  expression e = cast_expression(c);
  // FI: should we pass down the expected type? It would be useful for
  // heap modelling
  list sinks = expression_to_points_to_sinks(e, in);
  return sinks;
}

list sizeofexpression_to_points_to_sinks(sizeofexpression soe, pt_map in)
{
  list sinks = NIL;
  // FI: seems just plain wrong for a sink
  pips_internal_error("Not implemented yet");
  if( sizeofexpression_expression_p(soe) ){
    expression ne = sizeofexpression_expression(soe);
    sinks = expression_to_points_to_sinks(ne, in);
  }
  return sinks;
}


// FI: lots of issues here; the potential cast is lost...
// e is the arguments of the malloc call...
// Basic heap modelling
list malloc_to_points_to_sinks(expression e,
			       pt_map in __attribute__ ((unused)))
{
  // expression sizeof_exp = EXPRESSION (CAR(call_arguments(expression_call(rhs))));
  // FI: kind of dum since it is either char * or void *...
  // FI: the expected type should be passed down all these function calls...
  type t = expression_to_type(e);
  // FI: actual parameters to check!!!
  statement s = statement_undefined; // current statement line...
  // FI: number of occurences in the current statement
  // FI: multiline current statement...
  reference nr = original_malloc_to_abstract_location(e,
						      t,
						      type_undefined,
						      e,
						      get_current_module_entity(),
						      s);
  cell nc = make_cell_reference(nr);
  list sinks  = CONS(CELL, nc, NIL);

  return sinks;
}

list application_to_points_to_sinks(application a, pt_map in)
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

list subscript_to_points_to_sinks(subscript s, pt_map in)
{
  list sinks = NIL;
  pips_user_warning("Not implemented yet for %p and %p\n", s, in);
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
 */
list expression_to_points_to_sinks(expression e, pt_map in)
{
  /*reference + range + call + cast + sizeofexpression + subscript + application*/
  tag tt ;
  list sinks = NIL;
  syntax s = expression_syntax(e);
  switch (tt = syntax_tag(s)) {
  case is_syntax_reference: {
    reference r = syntax_reference(s);
    sinks = reference_to_points_to_sinks(r, in);
    break;
  }
  case is_syntax_range: {
    range r = syntax_range(s);
    sinks = range_to_points_to_sinks(r, in);
    break;
  }
  case  is_syntax_call: {
    call c = syntax_call(s);
    sinks = call_to_points_to_sinks(c, in);
    break;
  }
  case  is_syntax_cast: {
    cast c = syntax_cast(s);
    sinks = cast_to_points_to_sinks(c, in);
    break;
  }
  case  is_syntax_sizeofexpression: {
    // FI: no sink should be returned...
    //sinks = sizeofexpression_to_points_to_sinks(st, rhs, lhs, in);
    break;
  }
  case  is_syntax_subscript: {
    subscript sub = syntax_subscript(s);
    sinks = subscript_to_points_to_sinks(sub, in);
    break;
  }
  case  is_syntax_application: {
    application a = syntax_application(s);
    sinks = application_to_points_to_sinks(a, in);
    break;
  }
  case  is_syntax_va_arg: {
    // FI: useful?
    pips_internal_error("Not implemented yet\n");
    list soel = syntax_va_arg(s);
    sizeofexpression soe = SIZEOFEXPRESSION(CAR(soel));
    sinks = sizeofexpression_to_points_to_sinks(soe, in);
    break;
  }
  default:
    pips_internal_error("unknown expression tag %d\n", tt);
    break;
  }

  return sinks;
}
