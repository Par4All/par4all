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

/* FI: short term attempt at providing a deep copy to avoid sharing
 * between sets. If elements are shared, it quickly becomes impossible
 * to deep free any set.
 */
pt_map full_copy_pt_map(pt_map m)
{
  pt_map out = new_pt_map();
  /*
  HASH_MAP(k, v, {
      points_to pt = (points_to) k;
      points_to npt = copy_points_to(pt);
	hash_put( out->table, (void *) npt, (void *) npt );
    }, m->table);
  */
  SET_FOREACH(points_to, pt, m) {
    points_to npt = copy_points_to(pt);
    set_add_element(out, out, (void *) npt);
  }
return out;
}

/* See points_to_statement()
 *
 *
 */
pt_map statement_to_points_to(statement s, pt_map pt_in)
{
  pips_assert("pt_in is consistent", consistent_pt_map_p(pt_in));
  pt_map pt_out = new_pt_map();
  //assign_pt_map(pt_out, pt_in);
  pt_out = full_copy_pt_map(pt_in);
  instruction i = statement_instruction(s);

  init_heap_model(s);

  if(declaration_statement_p(s)) {
    /* Process the declarations */
    pt_out = declaration_statement_to_points_to(s, pt_out);
    /* Go down recursively, although it is currently useless since a
       declaration statement is a call to CONTINUE */
    pt_out = instruction_to_points_to(i, pt_out);
  }
  else {
    pt_out = instruction_to_points_to(i, pt_out);
  }

  pips_assert("pt_out is consistent", consistent_pt_map_p(pt_out));

  reset_heap_model();

  /* Either pt_in or pt_out should be stored in the hash_table 
   *
   * But it might be smarter (or not) to require or not the storage.
   */
  // FI: currently, this is going to be redundant most of the time
  pt_map pt_merged;
  if(bound_pt_to_list_p(s)) {
    points_to_list ptl_prev = load_pt_to_list(s);
    list ptl_prev_l = gen_full_copy_list(points_to_list_list(ptl_prev));
    pt_map pt_prev = new_pt_map();
    pt_prev = set_assign_list(pt_prev, ptl_prev_l);
    gen_free_list(ptl_prev_l);
    pt_merged = merge_points_to_set(pt_in, pt_prev);
  }
  else
    pt_merged = pt_in;
  fi_points_to_storage(pt_merged, s, true);

  /* Eliminate local information if you exit a block */
  if(statement_sequence_p(s)) {
    list dl = statement_declarations(s);
    pt_out = points_to_block_projection(pt_out, dl);
  }

  /* Because arc removals do not update the approximations of the
     remaining arcs, let's upgrade approximations before the
     information is passed. Useful for arithmetic02. */
  upgrade_approximations_in_points_to_set(pt_out);

  /* Really dangerous here: if pt_map "in" is empty, then pt_map "out"
   * must be empty to...
   *
   * FI: we have a problem to denote unreachable statement. To
   * associate an empty set to them woud be a way to avoid problems
   * when merging points-to along different control paths. But you
   * might also wish to start with an empty set... And anyway, you can
   * find declarations in dead code...
   */
  // FI: a temporary fix to the problem, to run experiments...
  if(empty_pt_map_p(pt_in) && !declaration_statement_p(s)
     && s!=get_current_module_statement())
    clear_pt_map(pt_out); // FI: memory leak?
    
  pips_assert("pt_out is consistent on exit", consistent_pt_map_p(pt_out));

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
  //bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  list l_decls = statement_declarations(s);

  pips_debug(1, "declaration statement \n");
  
  FOREACH(ENTITY, e, l_decls) {
    type et = ultimate_type(entity_type(e));
    if(pointer_type_p(et) || struct_type_p(et) || array_of_struct_type_p(et)) {
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
	  l = variable_to_pointer_locations(e);
	  FOREACH(CELL, source, l) {
	    cell sink = cell_to_nowhere_sink(source); 
	    points_to pt = make_points_to(source, sink,
					  make_approximation_exact(),
					  make_descriptor_none());
	    add_arc_to_pt_map(pt, pt_out);
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
    if(empty_pt_map_p(pt_in))
      pt_out = pt_in;
    else
      pt_out = call_to_points_to(c, pt_out);
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
    if(empty_pt_map_p(pt_in))
      pt_out = pt_in;
    else
      pt_out = expression_to_points_to(e, pt_in);
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

/* Computing the points-to information after a test.
 *
 * All the relationships are of type MAY, even if the same arc is
 * defined, e.g. "if(c) p = &i; else p=&i;".
 *
 * Might be refined later by using preconditions.
 */
pt_map test_to_points_to(test t, pt_map pt_in)
{
  pt_map pt_out = pt_map_undefined;

  //bool store = true;
  // pt_out = points_to_test(t, pt_in, store);
  // Translation of points_to_test
  statement ts = test_true(t);
  statement fs = test_false(t);
  pt_map pt_t =  pt_map_undefined;
  pt_map pt_f = pt_map_undefined;

  pt_map pt_in_t = full_copy_pt_map(pt_in);
  pt_map pt_in_f = full_copy_pt_map(pt_in);

  
  /* condition's side effect and information are taked into account, e.g.:
   *
   * "if(p=q)" or "if(*p++)" or "if(p)" which implies p->NULL in the
   * else branch. FI: to be checked with test cases */
  expression c = test_condition(t);
  if(!empty_pt_map_p(pt_in_t)) // FI: we are in dead code
    pt_in_t = condition_to_points_to(c, pt_in_t, true);
  pt_t = statement_to_points_to(ts, pt_in_t);

  if(!empty_pt_map_p(pt_in_f)) // FI: we are in dead code
    pt_in_f = condition_to_points_to(c, pt_in_f, false);
  pt_f = statement_to_points_to(fs, pt_in_f);
  
  pt_out = merge_points_to_set(pt_t, pt_f);

  // FI: we should take care of pt_t and pt_f to avoid memory leaks
  // In that specific case, clear_pt_map() and free_pt_map() should be ok

  free_pt_map(pt_t), free_pt_map(pt_f);

  return pt_out;
}

/* FI: I assume that pointers and pointer arithmetic cannot appear in
 * a do loop, "do p=q, r, 1" is possible with "p", "q" and "r"
 * pointing towards the same array... Let's hope the do loop
 * conversion does not catch such cases.
 */
pt_map loop_to_points_to(loop l, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  statement b = loop_body(l);
  //bool store = false;
  //pt_out = points_to_loop(l, pt_in, store);
  pt_out = any_loop_to_points_to(b,
				 expression_undefined,
				 expression_undefined,
				 expression_undefined,
				 pt_in);

  return pt_out;
}

pt_map whileloop_to_points_to(whileloop wl, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  statement b = whileloop_body(wl);
  expression c = whileloop_condition(wl);
    
  //bool store = false;
  if (evaluation_before_p(whileloop_evaluation(wl))) {
    //pt_out = points_to_whileloop(wl, pt_in, store);
    pt_out = any_loop_to_points_to(b,
				   expression_undefined,
				   c,
				   expression_undefined,
				   pt_in);
  }
  else {
    // pt_out = points_to_do_whileloop(wl, pt_in, store);
    /* Execute the first iteration */
    pt_out = statement_to_points_to(b, pt_out);
    pt_out = any_loop_to_points_to(b,
				   expression_undefined,
				   c,
				   expression_undefined,
				   pt_out);
  }

  //statement ws = whileloop_body(wl);
  //list dl = statement_declarations(ws);
  // FI: to be improved
  //if(declaration_statement_p(ws) && !ENDP(dl))
  //  pt_out = points_to_block_projection(pt_out, dl);

  return pt_out;
}

/* Perform the same k-limiting scheme for all kinds of loops 
 *
 * The do while loop mus use an external special treatment for the
 * first iteration.
 *
 * Derived from points_to_forloop().
 *
 * pt_in is modified by side effects.
 */
pt_map any_loop_to_points_to(statement b,
			     expression init, // can be undefined
			     expression c, // can be undefined
			     expression inc, // ca be undefined
			     pt_map pt_in)
{
  pt_map pt_out = pt_in;
  int i = 0;
  // FI: k is linked to the cycles in points-to graph, and should not
  // be linked to the number of convergence iterations. I assume here
  // that the minimal number of iterations is greater than the
  // k-limiting factor
  int k = get_int_property("POINTS_TO_K_LIMITING")+10;

  /* First, enter or skip the loop: initialization + condition check */
  if(!expression_undefined_p(init))
    pt_out = expression_to_points_to(init, pt_out);
  pt_map pt_out_skip = full_copy_pt_map(pt_out);
  if(!expression_undefined_p(c)) {
    pt_out = condition_to_points_to(c, pt_out, true);
    pt_out_skip = condition_to_points_to(c, pt_out_skip, false);
  }

  /* Comput pt_out as loop invariant: pt_out holds at the beginning of
   * the loop body.
   *
   * pt_out(i) = f(pt_out(i-1)) U pt_out(i-1)
   *
   * prev = pt_out(i-1)
   *
   * Note: the pt_out variable is also used to carry the loop exit
   * points-to set.
   */
  pt_map prev = new_pt_map();
  // FI: it should be a while loop to reach convergence
  // FI: I keep it a for loop for safety
  bool fix_point_p = false;
  for(i = 0; i<k+2 ; i++){
    /* prev receives the current points-to information, pt_out */
    set_clear(prev);
    prev = set_assign(prev, pt_out);
    set_clear(pt_out);

    /* Depending on the kind of loops, execute the body and then
       possibly the incrementation and the condition */
    // FI: here, storage_p would be useful to avoid unnecessary
    // storage and update for each substatement at each iteration k
    pt_out = statement_to_points_to(b, prev);
    if(!expression_undefined_p(inc))
      pt_out = expression_to_points_to(inc, pt_out);
    // FI: should be condition_to_points_to() for conditions such as
    // while(p!=q);
    // The condition is not always defined (do loops)
    if(!expression_undefined_p(c))
      pt_out = condition_to_points_to(c, pt_out, true);

    /* Merge the previous resut and the current result. */
    // FI: move to pt_map
    pt_out = merge_points_to_set(prev, pt_out);

    /* Check convergence */
    if(set_equal_p(prev, pt_out)) {
      fix_point_p = true;
      /* Add the last iteration to obtain the pt_out holding when
	 exiting the loop */
      pt_out = statement_to_points_to(b, prev);
      if(!expression_undefined_p(inc))
	pt_out = expression_to_points_to(inc, pt_out);
      if(!expression_undefined_p(c))
	pt_out = condition_to_points_to(c, pt_out, false);
      break;
    }
  }

  if(!fix_point_p)
    pips_internal_error("Loop convergence not reached.\n");

  /* FI: I suppose that p[i] is replaced by p[*] and that MAY/MUST
     information is changed accordingly. */
  pt_out = points_to_independent_store(pt_out);

  pt_out = merge_points_to_set(pt_out, pt_out_skip);

  return pt_out;
}

pt_map k_limit_points_to(pt_map pt_out, int k)
{
  //bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
  //entity anywhere = entity_undefined;

  SET_FOREACH(points_to, pt, pt_out){
    cell sc = points_to_source(pt);
    reference sr = cell_any_reference(sc);
    list sl = reference_indices(sr);

    cell kc = points_to_sink(pt);
    reference kr = cell_any_reference(kc);
    list kl = reference_indices(kr);

    if((int)gen_length(sl)>k){
      bool to_be_freed = false;
      type sc_type = cell_to_type(sc, &to_be_freed);
      sc = make_anywhere_cell(sc_type);
      if(to_be_freed) free_type(sc_type);
    }

    if((int)gen_length(kl)>k){
      bool to_be_freed = false;
      type kc_type = cell_to_type(kc, &to_be_freed);
      kc = make_anywhere_cell(kc_type);
      if(to_be_freed) free_type(kc_type);
    }

    points_to npt = make_points_to(sc, kc,
				   copy_approximation(points_to_approximation(pt)),
				   make_descriptor_none());
    if(!points_to_equal_p(npt,pt)){
      // FI: should be moved to pt_map type...
      pt_out = set_del_element(pt_out, pt_out, (void*)pt);
      pt_out = set_add_element(pt_out, pt_out, (void*)npt);
    }
    else {
      // FI: memory leak
      // free_points_to(npt);
    }
  }
  return pt_out;
}

/* This function should be located somewhere in effect-util in or near
   abstract locations */
cell make_anywhere_cell(type t)
{
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
  entity anywhere = type_sensitive_p?
    entity_all_xxx_locations_typed(ANYWHERE_LOCATION, t)
    :
    entity_all_xxx_locations(ANYWHERE_LOCATION);

  reference r = make_reference(anywhere,NIL);
  cell sc = make_cell_reference(r);
  return sc;
}


pt_map unstructured_to_points_to(unstructured u, pt_map pt_in)
{
  pt_map pt_out = pt_in;

  pt_out = new_points_to_unstructured(u, pt_in, true);

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
  statement b = forloop_body(fl);
  expression init = forloop_initialization(fl);
  expression c = forloop_condition(fl);
  expression inc = forloop_increment(fl);

  pt_out = any_loop_to_points_to(b, init, c, inc, pt_in);
  return pt_out;
}
