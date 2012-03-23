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
This file contains functions used to compute points-to sets at statement level.
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

/* See points_to_statement()
 *
 *
 */
pt_map statement_to_points_to(statement s, pt_map pt_in)
{
  pt_map pt_out = new_pt_map();
  assign_pt_map(pt_out, pt_in);
  instruction i = statement_instruction(s);

  if(declaration_statement_p(s)) {
    /* Process the declarations */
    pt_out = declaration_statement_to_points_to(s, pt_out);
    /* Go down recursively */
    pt_out = instruction_to_points_to(i, pt_out);
  }
  else {
    pt_out = instruction_to_points_to(i, pt_out);
  }

  /* Either pt_in or pt_out should be stored in the hash_table 
   *
   * But it might be smarter (or not) to require or not the storage.
   */
  // FI: currently, this is going to be redundant most of the time
  points_to_storage(pt_in, s, true);

    /* Eliminate local information if you exit a block */
  if(statement_sequence_p(s)) {
    list dl = statement_declarations(s);
    pt_out = points_to_block_projection(pt_out, dl);
  }

  return pt_out;
}

/* See points_to_init()
 *
 * pt_in is modified by side-effects and returned
 */
pt_map declaration_statement_to_points_to(statement s, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  //set pt_out = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  list l = NIL;
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  list l_decls = statement_declarations(s);

  pips_debug(1, "declaration statement \n");
  
  FOREACH(ENTITY, e, l_decls) {
    if(pointer_type_p(ultimate_type(entity_type(e)))) {
      if( !storage_rom_p(entity_storage(e)) ) {
	// FI: could be simplified with variable_initial_expression()
	value v_init = entity_initial(e);
	/* generate points-to due to the initialisation */
	if(value_expression_p(v_init)){
	  expression exp_init = value_expression(v_init);
	  expression lhs = entity_to_expression(e);
	  pt_out = assignment_to_points_to(lhs,
					   exp_init,
					   pt_out);
	  /* AM/FI: abnormal sharing (lhs); the reference may be
	     reused in the cel... */
	  /* free_expression(lhs); */
	}
	else {
	  /* Generate nowhere sinks */
	  /* FI: goes back into Amira's code */
	  l = points_to_init_variable(e);
	  FOREACH(CELL, cl, l) {
	    list l_cl = CONS(CELL, cl, NIL);
	    // FI: memory leak because of the calls to points_to_nowherexxx
	    if(type_sensitive_p)
	      set_union(pt_out, pt_out, points_to_nowhere_typed(l_cl, pt_out));
	    else
	      set_union(pt_out, pt_out, points_to_nowhere(l_cl, pt_out));
	    //FI: free l_cl?
	  }
	}
      }
    }
  }
  
  return pt_out;
}

/* See points_to_statement()
 *
 * pt_in is modified by side-effects and returned
 */
pt_map instruction_to_points_to(instruction i, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  tag it = instruction_tag(i);
  switch(it) {
  case is_instruction_sequence: {
    sequence seq = instruction_sequence(i);
    pt_out = sequence_to_points_to(seq, pt_in);
    break;
  }
  case is_instruction_test: {
    test t = instruction_test(i);
    pt_out = test_to_points_to(t, pt_in);
    break;
  }
  case is_instruction_loop: {
    loop l = instruction_loop(i);
    pt_out = loop_to_points_to(l, pt_in);
    break;
  }
  case is_instruction_whileloop: {
    whileloop wl = instruction_whileloop(i);
    pt_out = whileloop_to_points_to(wl, pt_in);
    break;
  }
  case is_instruction_goto: {
    pips_internal_error("Go to instructions should have been removed "
			"before the analysis is started\n");
    break;
  }
  case is_instruction_call: {
    call c = instruction_call(i);
    pt_out = call_to_points_to(c, pt_in);
    break;
  }
  case is_instruction_unstructured: {
    unstructured u = instruction_unstructured(i);
    pt_out = unstructured_to_points_to(u, pt_in);
    break;
  }
  case is_instruction_multitest: {
    pips_internal_error("Not implemented\n");
    break;
  }
  case is_instruction_forloop: {
    forloop fl = instruction_forloop(i);
    pt_out = forloop_to_points_to(fl, pt_in);
    break;
  }
  case is_instruction_expression: {
    expression e = instruction_expression(i);
    pt_out = new_pt_map();
    set_assign(pt_out, pt_in);
    pt_out = expression_to_points_to(e, pt_out);
    break;
  }
  default:
    pips_internal_error("Unexpected instruction tag %d\n", it);
  }
  return pt_out;
}

pt_map sequence_to_points_to(sequence seq, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  //bool store = true; // FI: management and use of store_p? Could be useful? How is it used?
  // pt_out = points_to_sequence(seq, pt_in, store);
  FOREACH(statement, st, sequence_statements(seq)) {
    pt_out = statement_to_points_to(st, pt_out);
  }

  return pt_out;
}

pt_map test_to_points_to(test t, pt_map pt_in)
{
  pt_map pt_out = pt_in;

  bool store = true;
  pt_out = points_to_test(t, pt_in, store);

  // FI: I do not understand with the necessary projections are not
  // performed recursively at a lower level

  // FI: I do not understand how the merge stuff works twice

  statement true_stmt = test_true(t);
  list tdl = statement_declarations(true_stmt);
  if(declaration_statement_p(true_stmt) && !ENDP(tdl)){
      pt_out = points_to_block_projection(pt_out, tdl);
      pt_out = merge_points_to_set(pt_out, pt_in);
    }

  statement false_stmt = test_false(t);
  list fdl = statement_declarations(false_stmt);
  if(declaration_statement_p(false_stmt) && !ENDP(fdl)) {
    pt_out = points_to_block_projection(pt_out, fdl);
    pt_out = merge_points_to_set(pt_out, pt_in);
  }

  return pt_out;
}

pt_map loop_to_points_to(loop l, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  bool store = false;
  pt_out = points_to_loop(l, pt_in, store);

  /* This sequence has been factored out in statement_to_points_to() */
  /*
  statement ls = loop_body(l);
  list dl = statement_declarations(ls);
  if(declaration_statement_p(ls) && !ENDP(dl))
    pt_out = points_to_block_projection(pt_out, dl);
  */

  return pt_out;
}

pt_map whileloop_to_points_to(whileloop wl, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  bool store = false;
  if (evaluation_before_p(whileloop_evaluation(wl)))
    pt_out = points_to_whileloop(wl, pt_in, store);
  else
    pt_out = points_to_do_whileloop(wl, pt_in, store);

  statement ws = whileloop_body(wl);
  list dl = statement_declarations(ws);
  // FI: to be improved
  if(declaration_statement_p(ws) && !ENDP(dl))
    pt_out = points_to_block_projection(pt_out, dl);

  return pt_out;
}

pt_map unstructured_to_points_to(unstructured u, pt_map pt_in)
{
  pt_map pt_out = pt_in;

  pt_out = points_to_unstructured(u, pt_in, true);

  // FI: The storage should be performed at a higher level?
  // points_to_storage(pt_out,current,true);
  return pt_out;
}

pt_map multitest_to_points_to(multitest mt, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  pips_internal_error("Not implemented yet for multitest %p\n", mt);
  return pt_out;
}

pt_map forloop_to_points_to(forloop fl, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  bool store = false;
  pt_out = points_to_forloop(fl, pt_in, store);
  /*
    statement ls = forloop_body(fl);
    list dl =statement_declarations(ls);
    if(declaration_statement_p(ls) && !ENDP(dl))
    pt_out = points_to_block_projection(pt_out, dl);
  */
  return pt_out;
}
