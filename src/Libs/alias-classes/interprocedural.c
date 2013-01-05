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
 * This file contains functions used to compute points-to sets at user
 * call sites.
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
#include "pipsdbm.h"
#include "resources.h"
//#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

// FI: this piece of code has been developed assuming that pt_map was
// a synonym for the set type
// FI: Not a good idea, to many interfaces with functions from other files
// #define pt_map set

/* Transform a list of parameters of type "entity" to a list of cells
 *
 * The list is not sorted. It is probably a reversed list.
 */
list points_to_cells_parameters(list dl)
{
  list fpcl = NIL;
  FOREACH(ENTITY, fp, dl) {
    if(formal_parameter_p(fp)) {
      reference r = make_reference(fp, NIL);
      cell c = make_cell_reference(r);
      fpcl = gen_nconc(CONS(CELL, c, NULL), fpcl);
    }
  }
  return fpcl;
}


/* Transform a list of arguments of type "expression" to a list of cells
 *
 * The list is not sorted. It is probably a reversed list.
 */
list points_to_cells_pointer_arguments(list al)
{
  list apcl = NIL;
  FOREACH(EXPRESSION, ap, al) {
    if(expression_pointer_p(ap) && expression_reference_p(ap)) {
      reference r = expression_reference(ap);
      cell c = make_cell_reference(r);
      apcl = gen_nconc(CONS(CELL, c, NULL), apcl);
    }
  }
  return apcl;
}



/* FI: limited to the interprocedural option */
points_to_graph user_call_to_points_to(call c,
				       points_to_graph pt_in,
				       list el) // effect list
{
  points_to_graph pt_out = pt_in;
  if(!points_to_graph_bottom(pt_in)) {
    entity f = call_function(c);
    //list al = call_arguments(c);

    // FI: intraprocedural, use effects
    // FI: interprocedural, check alias compatibility, generate gen and kill sets,...
    pt_out = pt_in;

    // Code by Amira
    //list fpcl = NIL; // Formal parameter cell list
    type t = entity_basic_concrete_type(f);
    if(type_functional_p(t)) {
      list dl = code_declarations(value_code(entity_initial(f)));
      /*fpcl = */points_to_cells_parameters(dl);   
    }
    else {
      pips_internal_error("Function has not a functional type.\n");
    }

    /* Using memory effects does not simplify the points-to analysis,
       which is a preliminary analusis wrt memory effects */
    if(interprocedural_points_to_analysis_p()) {
      pt_out = user_call_to_points_to_interprocedural(c, pt_in);
    }
    else if(fast_interprocedural_points_to_analysis_p()) {
      pt_out = user_call_to_points_to_fast_interprocedural(c, pt_in, el);
    }
    else {
      pt_out = user_call_to_points_to_intraprocedural(c, pt_in, el);
    }
  }

  return pt_out;
}

// FI: I assume we do not need the eval_p parameter here
list user_call_to_points_to_sinks(call c,
				  type et __attribute__ ((unused)),
				  pt_map in __attribute__ ((unused)),
				  bool eval_p)
{
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
  type t = entity_basic_concrete_type(call_function(c));
  type rt = compute_basic_concrete_type(functional_result(type_functional(t)));
  entity ne = entity_undefined;
  list sinks = NIL;
  entity f = call_function(c);
  // Interprocedural version
  // Check if there is a return value at the level of POINTS TO OUT, if yes return its sink
  if(interprocedural_points_to_analysis_p()
     ||fast_interprocedural_points_to_analysis_p() ) {
    const char* mn = entity_local_name(f);
    // FI: Warning, they are not translated into the caller's frame...
    // FI: An up-to-date version of in should be better
    //points_to_list pts_to_out = (points_to_list)
    //db_get_memory_resource(DBR_POINTS_TO_OUT, module_local_name(f), true);
    //list l_pt_to_out = gen_full_copy_list(points_to_list_list(pts_to_out));
    //set pt_out_callee = new_simple_pt_map();
    //pt_out_callee = set_assign_list(pt_out_callee, l_pt_to_out);
    // SET_FOREACH( points_to, pt, pt_out_callee) {
    set in_s = points_to_graph_set(in);
    list rvptl = NIL;
    SET_FOREACH( points_to, pt, in_s) {
      cell s = points_to_source(pt);
      reference sr = cell_any_reference(s);
      entity se = reference_variable(sr);
      const char* sn = entity_local_name(se);
      if( strcmp(mn, sn)==0) {
	cell sc = copy_cell(points_to_sink(pt));
	sinks = gen_nconc(CONS(CELL, sc, NULL), sinks);
	rvptl = CONS(POINTS_TO, pt, rvptl);
      }
    }
    /* Remove all arcs related to the return value of the callee */
    FOREACH(POINTS_TO, rvpt, rvptl) {
      remove_arc_from_simple_pt_map(rvpt, in_s);
      ;
    }
    gen_free_list(rvptl);
  /* FI: definitely the intraprocedural version */
  }
  else {
    if(type_sensitive_p) {
      if(eval_p) {
	type prt = ultimate_type(type_to_pointed_type(rt));
	ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,prt);
      }
      else
	ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,rt);
    }
    else
      ne = entity_all_xxx_locations(ANYWHERE_LOCATION);
    
    sinks = entity_to_sinks(ne);
  }
  return sinks;
}

// FI: why is this function located in interprocedural.c?
// Because it is used only in this file...
void remove_arcs_from_pt_map(points_to pts, set pt_out)
{
  cell sink = points_to_sink(pts);
  cell source = points_to_source(pts);
  

  SET_FOREACH(points_to, pt, pt_out) {
    if(cell_equal_p(points_to_source(pt), sink) ||cell_equal_p(points_to_source(pt), source) ) {
      remove_arc_from_simple_pt_map(pts, pt_out);
      entity e = entity_anywhere_locations();
      reference r = make_reference(e, NIL);
      cell source = make_cell_reference(r);
      cell sink = copy_cell(source);
      approximation a = make_approximation_exact();
      points_to npt = make_points_to(source, sink, a, make_descriptor_none());
      add_arc_to_simple_pt_map(npt, pt_out);
      remove_arcs_from_pt_map(pt, pt_out);
    }
  }
}

/*Compute the points to relations in a fast interprocedural way
 *
 * "c" is the call site
 *
 * "pt_in" is the points-to information available before the call site
 * is executed.
 *
 * "csel" is the call site effect list
 * 
 */
pt_map user_call_to_points_to_fast_interprocedural(call c,
						   pt_map pt_in,
						   list csel __attribute__ ((unused))) 
{
  set pt_in_s = points_to_graph_set(pt_in);
  pt_map pt_out = pt_in;
  set kill_1 =  set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set gen_1 =  set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set gen_2 =  set_generic_make(set_private, points_to_equal_p, points_to_rank);
  entity f = call_function(c);
  list al = call_arguments(c);
  list dl = code_declarations(value_code(entity_initial(f)));
  list fpcl = points_to_cells_parameters(dl);   
  /* Compute kill_1 = effective parameters of pointer type should point to
     nowhere in case the function call the free routine.
  */
  list apcl = points_to_cells_pointer_arguments(al);
  FOREACH(CELL, ac, apcl) {
    SET_FOREACH(points_to, pt, pt_in_s) {
      if(points_to_cell_equal_p(ac, points_to_source(pt)) ) {
	points_to npt = copy_points_to(pt);
	(void) add_arc_to_simple_pt_map(npt, kill_1);
      }
    }
  }
  ifdebug(8) print_points_to_set("kill_1",kill_1);
  
  SET_FOREACH(points_to, pt_to, kill_1) {
    approximation a = make_approximation_may();
    cell sr = copy_cell(points_to_source(pt_to));
    cell sk = copy_cell(points_to_sink(pt_to));
    points_to npt = make_points_to(sr, sk, a, make_descriptor_none());
    (void) add_arc_to_simple_pt_map(npt, gen_1);
  }
  ifdebug(8) print_points_to_set("gen_1",gen_1);

  
  SET_FOREACH(points_to, pt_to1, gen_1) {
    cell source = copy_cell(points_to_source(pt_to1));
    cell nc = cell_to_nowhere_sink(source);
    approximation a = make_approximation_may();
    points_to npt = make_points_to(source, nc, a, make_descriptor_none());
    (void) add_arc_to_simple_pt_map(npt, gen_2);
  }
  ifdebug(8) print_points_to_set("gen_1",gen_2);



  extern list load_summary_effects(entity e);
  list el = load_summary_effects(f);
  list wpl = written_pointers_set(el);
  points_to_list pts_to_in = (points_to_list)
    db_get_memory_resource(DBR_POINTS_TO_IN, module_local_name(f), true);
  list l_pt_to_in = gen_full_copy_list(points_to_list_list(pts_to_in));
  set pt_in_callee = new_simple_pt_map();
  pt_in_callee = set_assign_list(pt_in_callee, l_pt_to_in);
  // list l_pt_to_out = gen_full_copy_list(points_to_list_list(pts_to_out));
  // pt_map pt_out_callee = set_assign_list(pt_out_callee, l_pt_to_out);
  bool success_p = false;
  set pts_binded = compute_points_to_binded_set(f, al, pt_in_s, & success_p);
  // FI->AM: for the time being, ignore success_p...
  ifdebug(8) print_points_to_set("pt_binded", pts_binded);
  set bm = points_to_binding(fpcl, pt_in_callee, pts_binded);
  set pts_kill = compute_points_to_kill_set(wpl, pt_in_s, bm);
  ifdebug(8) print_points_to_set("pts_kill", pts_kill);
  set pts_gen = new_simple_pt_map();
  SET_FOREACH(points_to, pt, pts_kill) {
    cell source = points_to_source(pt);
    cell nc = cell_to_nowhere_sink(source);
    approximation a = make_approximation_exact();
    points_to npt = make_points_to(source, nc, a, make_descriptor_none());
    (void) add_arc_to_simple_pt_map(npt, pts_gen);
  }
  set pt_end = new_simple_pt_map();
  pts_kill = set_union(pts_kill, pts_kill, kill_1);
  pt_end = set_difference(pt_end, pt_in_s, pts_kill);
  pts_gen = set_union(pts_gen, pts_gen, gen_1);
  pts_gen = set_union(pts_gen, pts_gen, gen_2);
  pt_end = set_union(pt_end, pt_end, pts_gen);
  ifdebug(8) print_points_to_set("pt_end =", pt_end);
  points_to_graph_set(pt_out) = pt_end;
  return pt_out;
}

/* This function is very similar to
 * filter_formal_context_according_to_actual_context(), but a little
 * bit more tricky. Amira Mensi unified both in her own version, but
 * the unification makes the maintenance more difficult.
 */
void
recursive_filter_formal_context_according_to_actual_context(list fcl,
							    set pt_in_callee,
							    set pts_binded,
							    set translation,
							    set filtered)
{
  points_to_graph translation_g = make_points_to_graph(false, translation);

  /* Copy only possible arcs "pt" from "pt_in_callee" into the "filtered" set */
  SET_FOREACH(points_to, pt, pt_in_callee) {
    cell source = points_to_source(pt);
    if(related_points_to_cell_in_list_p(source, fcl)) {
      cell sink = points_to_sink(pt);
      list tsl = points_to_source_to_sinks(source, translation_g, false);
      /* FI: this assert may be too strong FI: This assert is too
       * strong for Pointers/formal_parameter01 and its message is
       * misleading because "source" is not an element of
       * "fcl". Elements of "fcl" must be translated, but related
       * elements may not be translatable because the effective
       * context is not as rich as the formal context. For instance,
       * the formal context may expect an array for each formal scalar
       * pointer, but the effective target may be a scalar. And an
       * error must be raised if pointer arithmetic is used in the
       * callee.
       */
      if(points_to_cell_in_list_p(source, fcl ))
	pips_assert("Elements of \"fcl\" can be translated", !ENDP(tsl));

      if(!ENDP(tsl)) {
	/* Make sure pts_binded is large enough: the pointer may be
	   initialized before the call to the caller and used only in
	   the callee. Because of the on-demand approach, pts_binded
	   does not contain enough elements. */
	pt_map pts_binded_g = make_points_to_graph(false, pts_binded);
	FOREACH(CELL, c, tsl) {
	  list sinks = any_source_to_sinks(c, pts_binded_g, false);
	  gen_free_list(sinks);
	}

	if(null_cell_p(sink)) {
	  /* Do we have a similar arc in pts_binded? */
	  FOREACH(CELL, tc, tsl) {
	    if(cell_points_to_null_sink_in_set_p(tc, pts_binded)) {
	      points_to npt = copy_points_to(pt);
	      add_arc_to_simple_pt_map(npt, filtered);
	      break;
	    }
	    else {
	      ; // do not copy this arc in filtered set
	    }
	  }
	}
	else {
	  FOREACH(CELL, tc, tsl) {
	    if(cell_points_to_non_null_sink_in_set_p(tc, pts_binded)) {
	      points_to npt = copy_points_to(pt);
	      add_arc_to_simple_pt_map(npt, filtered);
	      /* 
		 We need to concatenate fields indices to the formal parameter. AM
	      */
	      FOREACH(CELL, c, fcl) {
		if(cell_entity_equal_p(source,c)) {
		  reference r = cell_any_reference(source);
		  list ind = reference_indices(r);
		  reference rc = cell_to_reference(c);
		  reference_indices(rc) = gen_nconc(reference_indices(rc), ind);
		}
	      }
	      break;
	    }
	    else {
	      ; // do not copy this arc in filtered set
	    }
	  }
	}
	gen_free_list(tsl);
      }
      else {
	; // do not copy this arc in filtered set
      }
    }
  }

  /* Compute the translation relation for sinks of the formal arguments */
  list nfcl = NIL;
  FOREACH(CELL, c, fcl) {
    points_to_graph filtered_g = make_points_to_graph(false, filtered);
    list fl = points_to_source_to_some_sinks(c, filtered_g, false); // formal list
    if(!ENDP(fl)) {
      // FI: I am in trouble with _nl_1 and _nl_1[next]; the first one
      // is not recognized in the second one.
      //list tsl = points_to_source_to_sinks(c, translation_g, false);
      list tsl = points_to_source_to_some_sinks(c, translation_g, false);
      //list tsl = source_to_sinks(c, translation_g, false);
      FOREACH(CELL, tc, tsl) {
	points_to_graph pts_binded_g = make_points_to_graph(false, pts_binded);
	list al = points_to_source_to_some_sinks(tc, pts_binded_g, false); // formal list
	int nfl = (int) gen_length(fl);
	int nal = (int) gen_length(al);
	approximation a = approximation_undefined;
	if(nfl==1 && nal==1)
	  a = make_approximation_exact();
	else
	  a = make_approximation_may();
	FOREACH(CELL, fc, fl) {
	  if(!null_cell_p(fc)) {
	    FOREACH(CELL, ac, al) {
	      if(!null_cell_p(ac) && !nowhere_cell_p(ac)) {
		type fc_t = points_to_cell_to_concrete_type(fc);
		type ac_t = points_to_cell_to_concrete_type(ac);
		// FI: why not use array_pointer_string_type_equal_p()?
		if(!type_equal_p(fc_t, ac_t) && !overloaded_type_p(ac_t)) {
		  points_to_cell_add_zero_subscript(ac);
		  type ac_nt = points_to_cell_to_concrete_type(ac);
		  if(!type_equal_p(fc_t, ac_nt))
		    pips_internal_error("translation failure.\n");
		}
		points_to tr = make_points_to(copy_cell(fc), copy_cell(ac), 
					      copy_approximation(a),
					      make_descriptor_none());
		add_arc_to_simple_pt_map(tr, translation);
	      }
	    }
	    nfcl = CONS(CELL, fc, nfcl);
	  }
	}
	free_approximation(a);
      }
    }
    else {
      ; // No need to go down... if we stop with a good reason, not
      // because of a bug
      // pips_internal_error("Translation error.\n");
    }
  }

  pips_assert("The points-to translation mapping is well typed",
	      points_to_translation_mapping_is_typed_p(translation));

  /* Now, we have to call about the same function recursively on the
     list of formal sinks */
  if(!ENDP(nfcl)) {
    recursive_filter_formal_context_according_to_actual_context
      (nfcl, pt_in_callee, pts_binded, translation, filtered);
    gen_free_list(nfcl);
  }

  // FI translation_g should be freed, but not the set in it...

  return;
}

/* We have to handle constant strings such as "Hello!"  and not to
 * forget functional parameters or other types. Basically, type "t" is
 * returned unchanged, unless "t" is a functional type "void->string".
 *
 * The initial implementation of this function used the cell "ac" and the
 * variable "a" whose type is sought and was safer:
 *
 *  reference ar = cell_any_reference(ac);
 *  entity a = reference_variable(ar);
 *  if(constant_string_entity_p(a))
 *     nt = ...
 *
 * Any function of type "void->string" is considered a "string"
 * object. Let's hope it is ok in the points-to environment. Else, an
 * additional parameter must be passed.
 *
 * This function is located in the points-to library because it is
 * where it is useful. It would be even more useful if it returned a
 * "char *" or a "char[]", but this would imply a type allocation. As
 * it is, no new type is allocated.
 *
 * To be fully effective, the argument must be a basic concrete type.
 */
static type constant_string_type_to_string_type(type t)
{
  type nt = t; // default returned value: no change
  if(type_functional_p(t)) {
    functional f = type_functional(t);
    type rt = functional_result(f);
    list pl = functional_parameters(f);
    if(ENDP(pl) && string_type_p(rt))
      nt = rt;
  }
  return nt;
}

/* Filter pt_in_callee according to pts_binded. For instance, a
 * formal parameter can point to NULL in pt_in_callee only if it
 * also points to NULL in pts_binded. In the same way, a formal
 * parameter can point to a points-to stub in pt_in_callee only if
 * it points to a non-NULL target in pts_binded. Also, a formal
 * parameter cannot points exactly to UNDEFINED in pts_binded as
 * it would be useless (not clear if we can remove such an arc
 * when it is a may arc...). Finally, each formal parameter must
 * still point to something.
 *
 * The context of the caller may be insufficiently developped because
 * its does not use explictly a pointer that is a formal parameter for
 * it. For instance:
 *
 * foo(int ***p) {bar(int ***p);}
 *
 * The formal context of "foo()" must be developped when the formal
 * context of "bar()" is imported. For instance, *p, **p and ***p may
 * be accessed in "bar()", generating points-to stub in "bar". Similar
 * stubs must be generated here for "foo()" before the translation can
 * be performed.
 *
 * This is also true for global variables. pt_in may contain arcs that
 * should exist in pt_binded, and hence pt_caller. It may also contain
 * arcs that deny the existence of some arcs in pt_caller.
 */
set filter_formal_context_according_to_actual_context(list fpcl,
						      set pt_in,
						      set pt_binded,
						      set translation)
{
  set filtered = new_simple_pt_map();

  /* Copy only possible arcs "pt" from "pt_in" into the "filtered" set */
  SET_FOREACH(points_to, pt, pt_in) {
    cell source = points_to_source(pt);
    if(related_points_to_cell_in_list_p(source, fpcl)) {
      cell sink = points_to_sink(pt);
      if(null_cell_p(sink)) {
	/* Do we have the same arc in pt_binded? */
	if(arc_in_points_to_set_p(pt, pt_binded)) {
	  points_to npt = copy_points_to(pt);
	  add_arc_to_simple_pt_map(npt, filtered);
	}
	else {
	  ; // do not copy this arc in filtered set
	}
      }
      else {
	if(cell_points_to_non_null_sink_in_set_p(source, pt_binded)) {
	  points_to npt = copy_points_to(pt);
	  add_arc_to_simple_pt_map(npt, filtered);
	}
	else {
	  ; // do not copy this arc in filtered set
	}
      }
    }
    else {
      /* We have to deal recursively with stubs of the formal context
	 and first with the global variables...although they, or there
	 stubs, do not require any translation? Too bad I did not
	 record why I had to add this... */
      reference r = cell_any_reference(source);
      entity v = reference_variable(r);
      if(false && (static_global_variable_p(v) || global_variable_p(v))) {
	points_to npt = copy_points_to(pt);
	add_arc_to_simple_pt_map(npt, filtered);
	/* FI: I do not understand why I have to do this for
	   EffectsWithPointsTo/pointer_modif04.c */
	if(statement_points_to_context_defined_p()) {
	  /* Useless when called from effects... */
	  add_arc_to_points_to_context(npt);
	  add_arc_to_statement_points_to_context(npt);
	}
      }
      else {
	/* FI: we do not know what we really do here... An arc is not
	   taken into account, but it might be taben into account
	   recursively below. */
	;
      }
    }
  }

  /* Compute the translation relation for sinks of the formal arguments */
  list fcl = NIL;
  points_to_graph filtered_g = make_points_to_graph(false, filtered);
  points_to_graph pt_binded_g = make_points_to_graph(false, pt_binded);
  FOREACH(CELL, c, fpcl) {
    list fl = points_to_source_to_any_sinks(c, filtered_g, false); // formal list
    list al = points_to_source_to_any_sinks(c, pt_binded_g, false); // actual list
    int nfl = (int) gen_length(fl);
    int nal = (int) gen_length(al);
    approximation approx = approximation_undefined;

    // FI: que fait-on avec nfl==0, comme dans
    // Pointers/StrictTyping.sub/struct08.c ou nfl vaut 0 parce que le
    // parametre effectif est undefined?

    if(nfl==1 && nal==1)
      approx = make_approximation_exact();
    else
      approx = make_approximation_may();
    FOREACH(CELL, fc, fl) {
      if(!null_cell_p(fc)) {
	FOREACH(CELL, ac, al) {
	  if(!null_cell_p(ac) && !nowhere_cell_p(ac)) {
	    type fc_t = points_to_cell_to_concrete_type(fc);
	    type iac_t = points_to_cell_to_concrete_type(ac);
	    type ac_t = constant_string_type_to_string_type(iac_t);
	    /* We have to handle constant strings such as "Hello!"
	       and not to forget functional parameters. */
	    if(type_functional_p(ac_t)) {
	      reference ar = cell_any_reference(ac);
	      entity a = reference_variable(ar);
	      if(constant_string_entity_p(a)) {
		ac_t = functional_result(type_functional(iac_t));
	      }
	    }
	    if(!array_pointer_string_type_equal_p(fc_t, ac_t)
	       && !overloaded_type_p(ac_t)) {
	      if(array_type_p(ac_t)) {
		points_to_cell_add_zero_subscript(ac);
		type ac_nt = points_to_cell_to_concrete_type(ac);
		if(!type_equal_p(fc_t, ac_nt) && !overloaded_type_p(ac_nt)) {
		  // Pointers/pointer14.c
		  // FI: I am not sure it is the best translaration
		  // It might be better to remove some zero subscripts from fc
		  points_to_cell_complete_with_zero_subscripts(ac);
		  type ac_nnt = points_to_cell_to_concrete_type(ac);
		  if(!type_equal_p(fc_t, ac_nnt) && !overloaded_type_p(ac_nnt))
		    pips_internal_error("translation failure for an array.\n");
		}
	      }
	      else {
		reference fr = cell_any_reference(fc);
		if(adapt_reference_to_type(fr, ac_t))
		  ;
		else {
		  reference ar = cell_any_reference(ac);
		  pips_user_error
		    ("Translation failure for actual parameter \"%s\" at line %d.\n"
		     "Maybe property POINTS_TO_STRICT_POINTER_TYPES should be reset.\n",
		     reference_to_string(ar),
		     points_to_context_statement_line_number());
		  // pips_internal_error("translation failure.\n");
		}
	      }
	    }
	    points_to tr = make_points_to(copy_cell(fc), 
					  copy_cell(ac), 
					  copy_approximation(approx),
					  make_descriptor_none());
	    add_arc_to_simple_pt_map(tr, translation);
	  }
	}
	fcl = CONS(CELL, fc, fcl);
      }
    }
    free_approximation(approx);
  }

  ifdebug(8) {
    pips_debug(8, "First filtered IN set for callee at call site:\n");
    print_points_to_set("", filtered);
    pips_debug(8, "First translation set for call site:\n");
    print_points_to_set("", translation);
  }

  pips_assert("The points-to translation mapping is well typed",
	      points_to_translation_mapping_is_typed_p(translation));

  /* Now, we have to call about the same function recursively on the
     list of formal sinks */
  if(!ENDP(fcl)) {
    recursive_filter_formal_context_according_to_actual_context
      (fcl, pt_in, pt_binded, translation, filtered);
    gen_free_list(fcl);
  }

  /* Some arcs have been removed, so other arcs may be promoted from
     "may" to "exact". */
  upgrade_approximations_in_points_to_set(filtered_g);

  ifdebug(8) {
    pips_debug(8, "Final filtered IN set for callee at call site:\n");
    print_points_to_set("", filtered);
    pips_debug(8, "Final translation set for call site:\n");
    print_points_to_set("", translation);
  }
    
  return filtered;
}

/* If an address has not been written, i.e. it is not in list "wpl",
 * then the points-to information is the intersection of the in and
 * out information.
 *
 * The set "in" may be modified by side effect. A new set,
 * "filtered_out" is computed. By definition, they are equivalent for
 * the addresses that are not in list "wpl".
 *
 * The arcs are shared by the different sets. But I may allocate new
 * ones: yet another potential memory leak...
 */
set filter_formal_out_context_according_to_formal_in_context
(set out, set in, list wpl, entity f)
{
  set out_filtered = new_simple_pt_map();

  /* First, filter out according to in */
  SET_FOREACH(points_to, pt, out) {
    cell source = points_to_source(pt);
    entity se = reference_variable(cell_any_reference(source));
    entity rv = any_function_to_return_value(f);
    cell sink = points_to_sink(pt);
    if(nowhere_cell_p(sink)) {
      /* The source of the arc may not have been modified but the sink
	 probably has been freed: the arc must be preserved */
      add_arc_to_simple_pt_map(pt, out_filtered);
    }
    else if(points_to_cell_in_list_p(source, wpl)) {
      /* The source of the arc has been modified: the arc must be preserved */
      add_arc_to_simple_pt_map(pt, out_filtered);
    }
    else if(se==rv) {
      /* the arc defining the return value must be preserved: no! */
      add_arc_to_simple_pt_map(pt, out_filtered);
    }
    else {
      /* Is this points-to arc also in set "in"? With or without the
	 same approximation? */
      approximation a = approximation_undefined;
      if(similar_arc_in_points_to_set_p(pt, in, &a)) {
	approximation oa = points_to_approximation(pt);
	if(approximation_exact_p(oa)) {
	  add_arc_to_simple_pt_map(pt, out_filtered);
	}
	else if(approximation_exact_p(a)) {
	  cell nsource = copy_cell(source);
	  cell nsink = copy_cell(points_to_sink(pt));
	  approximation na = copy_approximation(a);
	  points_to npt = make_points_to(nsource, nsink, na,
					 make_descriptor_none());
	  add_arc_to_simple_pt_map(npt, out_filtered);
	}
	else {
	  add_arc_to_simple_pt_map(pt, out_filtered);
	}
      }
      else {
	; // This arc cannot be preserved in "out_filtered"
      }
    }
  }

  /* Second, filter set "in" with respect to new set "filtered_out". */
  list to_be_removed = NIL;
  list to_be_added = NIL; // FI: why do you want to add stuff?
  SET_FOREACH(points_to, ipt, in) {
  /*
  hash_table _hash_ = set_private_get_hash_table(in);
  void * _point_ = NULL;
  for (points_to ipt;
       (_point_ =  hash_table_scan(_hash_, _point_, (void **) &ipt, NULL));) {
  */
    cell source = points_to_source(ipt);
    if(points_to_cell_in_list_p(source, wpl))
      ; // do nothing
    else {
      /* Is this points-to arc also in set "in"? With or without the
	 same approximation? */
      approximation a = approximation_undefined;
      if(similar_arc_in_points_to_set_p(ipt, out, &a)) {
	approximation oa = points_to_approximation(ipt);
	if(approximation_exact_p(oa)) {
	  ; // Do nothing
	}
	else if(approximation_exact_p(a)) {
	  cell nsource = copy_cell(source);
	  cell nsink = copy_cell(points_to_sink(ipt));
	  approximation na = copy_approximation(a);
	  points_to npt = make_points_to(nsource, nsink, na,
					 make_descriptor_none());
	  //add_arc_to_simple_pt_map(npt, in);
	  to_be_added = CONS(POINTS_TO, npt, to_be_added);
	  to_be_removed = CONS(POINTS_TO, ipt, to_be_removed);
	}
	else {
	  ; // do nothing
	}
      }
      else {
	to_be_removed = CONS(POINTS_TO, ipt, to_be_removed);
      }
    }
  }

  FOREACH(POINTS_TO, pt, to_be_removed)
    remove_arc_from_simple_pt_map(pt, in);

  FOREACH(POINTS_TO, pt, to_be_added)
    add_arc_to_simple_pt_map(pt, in);

  return out_filtered;
}

void points_to_translation_of_struct_formal_parameter(cell fc,
						      cell ac,
						      approximation a,
						      type st,
						      set translation)
{
  /* We assume that cell fc and cell ac are of type st and that st is
     a struct type. */
  // pips_internal_error("Not implemented yet.\n");
  list fl = struct_type_to_fields(st);

  FOREACH(ENTITY, f, fl) {
    type ft = entity_basic_concrete_type(f);
    if(pointer_type_p(ft)) {
      cell nfc = copy_cell(fc);
      cell nac = copy_cell(ac);
      points_to_cell_add_field_dimension(nfc, f);
      points_to_cell_add_field_dimension(nac, f);
      points_to pt = make_points_to(nfc, nac, copy_approximation(a),
				    make_descriptor_none());
      add_arc_to_simple_pt_map(pt, translation);
    }
    else if(struct_type_p(ft)) {
      cell nfc = copy_cell(fc);
      cell nac = copy_cell(ac);
      points_to_cell_add_field_dimension(nfc, f);
      points_to_cell_add_field_dimension(nac, f);
      points_to_translation_of_struct_formal_parameter(fc,
						       ac,
						       a,
						       ft,
						       translation);
      free_cell(nfc), free_cell(nac);
    }
    else if(array_of_pointers_type_p(ft)) {
      cell nfc = copy_cell(fc);
      cell nac = copy_cell(ac);
      points_to_cell_add_field_dimension(nfc, f);
      points_to_cell_add_field_dimension(nac, f);
      points_to_cell_add_unbounded_subscripts(nfc);
      points_to_cell_add_unbounded_subscripts(nac);
      points_to pt = make_points_to(nfc, nac, copy_approximation(a),
				    make_descriptor_none());
      add_arc_to_simple_pt_map(pt, translation);
    }
    else if(array_of_struct_type_p(ft)) {
      cell nfc = copy_cell(fc);
      cell nac = copy_cell(ac);
      points_to_cell_add_field_dimension(nfc, f);
      points_to_cell_add_field_dimension(nac, f);
      points_to_cell_add_unbounded_subscripts(nfc);
      points_to_cell_add_unbounded_subscripts(nac);
      type et = array_type_to_element_type(ft);
      type cet = compute_basic_concrete_type(et);
      points_to_translation_of_struct_formal_parameter(fc,
						       ac,
						       a,
						       cet,
						       translation);
      free_cell(nfc), free_cell(nac);
    }
    else {
      ; // do nothing
    }
  }

}

bool points_to_translation_mapping_is_typed_p(set translation)
{
  bool typed_p = true;
  SET_FOREACH(points_to, pt, translation) {
    cell source = points_to_source(pt);
    cell sink = points_to_sink(pt);
    type source_t = points_to_cell_to_concrete_type(source);
    type isink_t = points_to_cell_to_concrete_type(sink);
    type sink_t = constant_string_type_to_string_type(isink_t);
#if 0
    if(type_functional_p(isink_t)) {
      reference ar = cell_any_reference(ac);
      entity a = reference_variable(ar);
      if(constant_string_entity_p(a)) {
	sink_t = ...;
      }
    }
#endif
    if(!array_pointer_string_type_equal_p(source_t, sink_t)
       && !overloaded_type_p(sink_t)) {
      typed_p = false;
      pips_internal_error("Badly typed points-to translation mapping.\n");
      break;
    }
  }
  return typed_p;
}

/* Lits al and fpcl are assumed consistent, and consistent with the
   formal parameter ranks. */
void points_to_translation_of_formal_parameters(list fpcl, 
						list al,
						pt_map pt_in,
						set translation)
{
  FOREACH(CELL, fc, fpcl) {
    /* assumption about fpcl */
    entity v = reference_variable(cell_any_reference(fc));
    int n = formal_offset(storage_formal(entity_storage(v)));
    expression a = EXPRESSION(gen_nth(n-1, al));
    /* This function does not return constant memory paths... This
     * could fixed below with calls to
     * points_to_indices_to_unbounded_indices(), but it could/should also be
     * fixed later in the processing, at callees level. See
     * EffectsWithPointsTo.sub/call05.c
     *
     * See also EffectsWithPointsTo.sub/call08.c: &y[i][1]
     * You need expression_to_points_to_sinks() on such a lhs expression...
     */
    list acl = expression_to_points_to_sources(a, pt_in);
    int nacl = (int) gen_length(acl);
    // FI->FI: you should check nacl==0...
    approximation ap = nacl==1? make_approximation_exact() :
      make_approximation_may();
    FOREACH(CELL, ac, acl) {
      cell source = copy_cell(fc);
      //bool to_be_freed;
      //type a_source_t = points_to_cell_to_type(ac, &to_be_freed);
      //type source_t = compute_basic_concrete_type(a_source_t);
      type source_t = points_to_cell_to_concrete_type(source);
      if(pointer_type_p(source_t)) {
	cell n_source = copy_cell(fc);
	cell n_sink = copy_cell(ac);
	type sink_t = points_to_cell_to_concrete_type(n_sink);
	//reference n_sink_r = cell_any_reference(n_sink);
	// points_to_indices_to_unbounded_indices(reference_indices(n_sink_r));
	if(!type_equal_p(source_t, sink_t)) {
	  if(array_pointer_type_equal_p(source_t, sink_t))
	    ; // do nothing: a 1D array is equivalent to a pointer
	  else if(array_type_p(sink_t)) {
	    // FI: I do not remember in which case I needed this
	    points_to_cell_add_zero_subscript(n_sink);
	  }
	  else if(scalar_type_p(sink_t)) {
	    // Pointers/dereferencing11.c:
	    // "i", double *, and "fifi[3][0]", double [2][3]
	    reference n_sink_r = cell_any_reference(n_sink);
	    list n_sink_sl = reference_indices(n_sink_r);
	    bool succeed_p = false;
	    if(!ENDP(n_sink_sl)) {
	      expression ls = EXPRESSION(CAR(gen_last(n_sink_sl)));
	      if(zero_expression_p(ls)) {
		// points_to_cell_remove_last_zero_subscript(n_sink);
		gen_remove_once(&reference_indices(n_sink_r), ls);
		succeed_p = true;
	      }
	    }
	    if(!succeed_p)
	      pips_internal_error("Not implemented yet.\n");
	  }
	  else
	    pips_internal_error("Not implemented yet.\n");
	}
	points_to pt = make_points_to(n_source, n_sink, copy_approximation(ap),
				      make_descriptor_none());
	add_arc_to_simple_pt_map(pt, translation);
      }
      else if(array_type_p(source_t)) {
	if(array_of_pointers_type_p(source_t)) {
	  /* Likely to be wrong whe the formal parameter is a pointer
	     and the actual parameter is a simple pointer, or a
	     pointer to an array with fewer dimensions. */
	  cell n_source = copy_cell(fc);
	  cell n_sink = copy_cell(ac);
	  //reference n_sink_r = cell_any_reference(n_sink);
	  // points_to_indices_to_unbounded_indices(reference_indices(n_sink_r));
	  points_to pt = make_points_to(n_source, n_sink, copy_approximation(ap),
					make_descriptor_none());
	  add_arc_to_simple_pt_map(pt, translation);
	//pips_internal_error("Not implemented yet.\n");
	}
	else if(array_of_struct_type_p(source_t)) {
	  pips_internal_error("Not implemented yet.\n");
	}
	else {
	  ; // Do no worry about these arrays
	}
      }
      else if(struct_type_p(source_t)) {
	// Can we make an artificial lhs and rhs and call
	// assign_to_points_to() in that specific case?
	// Or do we have to program yet another recursive descent in structs?
	// How do we keep track of the approximation?
	points_to_translation_of_struct_formal_parameter(fc,
							 ac,
							 ap,
							 source_t,
							 translation);
      }
      else {
	; // No need
      }
      //if(to_be_freed) free_type(a_source_t);
    }
    free_approximation(ap);
  }
  pips_assert("The points-to translation mapping is well typed",
	      points_to_translation_mapping_is_typed_p(translation));
}

/* Initial comments: add arcs of set "pt_caller" to set "pt_kill" if
 * their origin cells are not in the list of written pointers "wpl"
 * but is the origin of some exact arc in "pt_out_callee_filtered".
 *
 * Equations retrieved from the C code
 *
 * K = { c | \exits pt c=source(pt) !\in Written
 *                  ^ |translation(source(pt), binding, f)|==1
 *                  ^ atomic(translation(source(pt), binding, f) }
 * 
 * pt_kill = {pt in pt_caller | \exits c \in K binding(c)==source(pt)}
 *
 * K is a set of cells defined in the frame of the callee and pt_kill
 * a set of points-to defined in the frame of the caller.
 *
 * Examples:
 *
 * Indirect free of a pointed cell
 *
 * "main() {p = malloc(); * my_free(p);}" with "my_free(int * p) {free(p);}".
 *
 * p->heap in pt_caller must be removed from pt_end, hence p->heap
 * belongs to pt_kill
 *
 * Other possibilities must be linked to tests and executions errors.
 *
 * void foo(int * ap) {bar(ap);} 
 *
 * void bar(int * fp) { *p = 1;}
 *
 * As a result ap->_ap_1, EXACT because ap->NULL has been killed
 *
 * void bar(int * fp) {if(fp==NULL) exit(1); return;}
 *
 * The result should be the same as above.
 */
void add_implicitly_killed_arcs_to_kill_set(set pt_kill,
					    list wpl,
					    set pt_caller,
					    set pt_out_callee_filtered,
					    set binding,
					    entity f)
{
  SET_FOREACH(points_to, out_pt, pt_out_callee_filtered) {
    cell source = points_to_source(out_pt);
    if(!points_to_cell_in_list_p(source, wpl)) {
      /* Arc out_pt has been implicitly obtained */
      approximation a = points_to_approximation(out_pt);
      // FI: let's assume that approximation subsumes atomicity
      if(approximation_exact_p(a)) {
	// source is defined in the formal context
	//points_to_graph binding_g =
	//   make_points_to_graph(false, binding);
	// list tl = points_to_source_to_sinks(source, binding_g, false);
	list tl = points_to_cell_translation(source, binding, f);
	int  ntl = (int) gen_length(tl);
	cell t_source = cell_undefined;
	if(ntl==1 && (atomic_points_to_cell_p(t_source=CELL(CAR(tl))))) {
	  SET_FOREACH(points_to, pt, pt_caller) {
	    cell pt_source = points_to_source(pt);
	    if(points_to_cell_equal_p(t_source, pt_source)) {
	      // FI: do not worry about sharing of arcs
	      add_arc_to_simple_pt_map(pt, pt_kill);
	    }
	  }
	}
	gen_free_list(tl);
      }
    }
  }
}

list translation_transitive_closure(cell c, set translation)
{
  list succ = CONS(CELL, c, NIL);
  list n_succ = gen_copy_seq(succ);
  bool finished_p = false;
  while(!finished_p) {
    points_to_graph translation_g = make_points_to_graph(false, translation);
    // Do not worry about sharing due to NULL or UNDEFINED/NOWHERE
    // Potentially new successors
    list pn_succ = points_to_sources_to_effective_sinks(n_succ, translation_g, false);
    list n_succ = NIL; // Reealy new successors
    // FI: does not work in general because the content is not used, shallow
    // gen_list_and_not(&n_succ, succ);
    FOREACH(CELL, c, pn_succ) {
      if(!points_to_cell_in_list_p(c, succ))
	n_succ = CONS(CELL, c, n_succ);
    }
    gen_free_list(pn_succ);
    if(ENDP(n_succ)) {
      /* We are done */
      finished_p = true;
      break;
    }
    else {
      succ = gen_nconc(succ, n_succ);
      n_succ = gen_copy_seq(n_succ);
    }
  }
  return succ;
}

/* See if two cells in "fpcl" point toward the same location via the
   transitive closure of "translation". */
bool aliased_translation_p(list fpcl, set translation)
{
  bool alias_p = false;
  hash_table closure = hash_table_make(hash_pointer, 0);
  list c_fpcl = fpcl;
  list p_fpcl = fpcl;

  for( ; !ENDP(c_fpcl); POP(c_fpcl)) {
    cell c = CELL(CAR(c_fpcl));
    list succ_l = translation_transitive_closure(c, translation);
    hash_put(closure, (void *) c, (void *) succ_l);
    for(p_fpcl = fpcl; p_fpcl!=c_fpcl; POP(p_fpcl)) {
      cell p_c = CELL(CAR(p_fpcl));
      list p_succ_l = (list) hash_get(closure, (void *) p_c);
      list c_succ_l = gen_copy_seq(succ_l);
      // FI: this is not sufficient, conflicts between cells should be
      // checked to take into account abstract locations.
      // gen_list_and(&c_succ_l, p_succ_l);
      bool may_conflict_p = points_to_cell_lists_may_conflict_p(c_succ_l, p_succ_l);
      bool must_conflict_p = points_to_cell_lists_must_conflict_p(c_succ_l, p_succ_l);
      //if(!ENDP(c_succ_l)) {
      if(may_conflict_p /*|| must_conflict_p*/ ) { // must implies may I guess
	alias_p = true;
	gen_free_list(c_succ_l);
	entity fp1 = reference_variable(cell_any_reference(c));
	entity fp2 = reference_variable(cell_any_reference(p_c));
	
	if(statement_points_to_context_defined_p())
	  pips_user_warning("%saliasing detected between formal parameters "
			    "\"%s\" and \"%s\" at line %d.\n",
			    must_conflict_p? "" : "possible ",
			    entity_user_name(fp1),
			    entity_user_name(fp2),
			    points_to_context_statement_line_number());
	else {
	  // In case this function is used in an effect context
	  pips_user_warning("%saliasing detected between formal parameters "
			    "\"%s\" and \"%s\".\n",
			    must_conflict_p? "" : "possible ",
			    entity_user_name(fp1),
			    entity_user_name(fp2));
	}
	break;
      }
      gen_free_list(c_succ_l);
    }
  }

  if(alias_p) {
    /* Print the transitive closures */
    HASH_MAP(c, succs,
	     { entity fp = reference_variable(cell_any_reference((cell) c));
	       fprintf(stderr, "\"%s\" -> ", entity_user_name(fp));
	       print_points_to_cells((list) succs);
	       fprintf(stderr, "\n");
	     }, closure);
  }

  /* Clean up the hash-table... */
  HASH_MAP(c, succs, {gen_free_list((list) succs);}, closure);

  hash_table_free(closure);
  return alias_p;
}

/* It is partly a kill and partly a gen operation. 
 *
 * FI: the very same function must exist for pointer assignments I guess
 */
static set lower_points_to_approximations_according_to_write_effects(set pt_end, list wpl)
{
  list optl = NIL, nptl = NIL;
  SET_FOREACH(points_to, pt, pt_end) {
    cell source = points_to_source(pt);
    // FI->FI: en fait, il faudrait prendre en compte le treillis et
    // tester les conflits,
    // written_effects_conflict_with_points_to_cell()
    if(points_to_cell_in_list_p(source, wpl)) {
      optl = CONS(POINTS_TO, pt, optl);
      cell sink = points_to_sink(pt);
      points_to npt = make_points_to(copy_cell(source),
				     copy_cell(sink),
				     make_approximation_may(),
				     make_descriptor_none());
      nptl = CONS(POINTS_TO, npt, nptl);
    }
  }
  FOREACH(POINTS_TO, opt, optl)
    remove_arc_from_simple_pt_map(opt, pt_end);
  FOREACH(POINTS_TO, npt, nptl)
    add_arc_to_simple_pt_map(npt, pt_end);
  return pt_end;
}

/* Compute the points-to relations in a complete interprocedural way:
 * be as accurately as possible.
 *
 */
pt_map user_call_to_points_to_interprocedural(call c,
					      pt_map pt_caller)
{
  pt_map pt_end_f = pt_caller;
  pips_assert("pt_caller is valid", !points_to_graph_bottom(pt_caller));
  pips_assert("pt_caller is consistent", consistent_pt_map_p(pt_caller));
  entity f = call_function(c);
  list al = call_arguments(c);
  list dl = code_declarations(value_code(entity_initial(f)));
  list fpcl = points_to_cells_parameters(dl);   
  points_to_list pts_to_in = (points_to_list)
    db_get_memory_resource(DBR_POINTS_TO_IN, module_local_name(f), true);
  list l_pt_to_in = gen_full_copy_list(points_to_list_list(pts_to_in));
  points_to_list pts_to_out = (points_to_list)
    db_get_memory_resource(DBR_POINTS_TO_OUT, module_local_name(f), true);
  bool out_bottom_p = points_to_list_bottom(pts_to_out);

  list l_pt_to_out = gen_full_copy_list(points_to_list_list(pts_to_out));

  /* Not much to do if both IN and OUT are empty, except if OUT is
     bottom (see below) */
  if(!(ENDP(l_pt_to_out) && ENDP(l_pt_to_in))) {
    // FI: this function should be moved from semantics into effects-util
    extern list load_body_effects(entity e);
    list el = load_body_effects(f);
    list wpl = written_pointers_set(el);
    list cwpl = certainly_written_pointers_set(el);
    set pt_caller_s = points_to_graph_set(pt_caller);

    //points_to_list pts_to_in = (points_to_list)
    //db_get_memory_resource(DBR_POINTS_TO_IN, module_local_name(f), true);
   
    //list l_pt_to_in = gen_full_copy_list(points_to_list_list(pts_to_in));
    set pt_in = new_simple_pt_map();
    pt_in = set_assign_list(pt_in, l_pt_to_in);
    set pt_out = new_simple_pt_map();
    pt_out = set_assign_list(pt_out, l_pt_to_out);

    // FI: function name... set or list?
    bool success_p = false;
    set pt_binded = compute_points_to_binded_set(f, al, pt_caller_s, &success_p);
    ifdebug(8) print_points_to_set("pt_binded", pt_binded);

    if(success_p) {
      set binding = new_simple_pt_map();
      /* This used to be only useful when a free occurs with the
	 callee, since information about formal parameters used to be
	 normally projected out. */
      points_to_translation_of_formal_parameters(fpcl, al, pt_caller, binding);

      /* Global variables do not require any translation in C, but it
	 might more convenient to apply translation uniformly, without
	 checking for global variables... Or the other way round? */
      //points_to_translation_of_global_variables(pt_out, pt_caller, binding);

      /* Filter pt_in_callee according to pt_binded. For instance, a
	 formal parameter can point to NULL in pt_in_callee only if it
	 also points to NULL in pt_binded. In the same way, a formal
	 parameter can point to a points-to stub in pt_in_callee only if
	 it points to a non-NULL target in pt_binded. Also, a formal
	 parameter cannot points exactly to UNDEFINED in pt_binded as
	 it would be useless (not clear if we can remove such an arc
	 when it is a may arc...). Finally, each formal parameter must
	 still point to something. */
      set pt_in_filtered =
	filter_formal_context_according_to_actual_context(fpcl,
							  pt_in,
							  pt_binded,
							  binding);

      /* We have to test if pt_binded is compatible with pt_in_callee */
      /* We have to start by computing all the elements of E (stubs) */
      //list stubs = stubs_list(pt_in_callee, pt_out_callee);
      // bool compatible_p = true;
      // FI: I do not understand Amira's test
      // = sets_binded_and_in_compatible_p(stubs, fpcl, pt_binded,
      //					pt_in_callee_filtered, pt_out_callee,
      //					binding);
      /* See if two formal parameters can reach the same memory cell,
       * i.e. transitive closure of binding map. We should take care
       * of global variables too... */
      // compatible_p = compatible_p && !aliased_translation_p(fpcl, pt_binded);

      /* If no pointer is written, aliasing is not an issue */
      if(ENDP(wpl) || !aliased_translation_p(fpcl, pt_binded)) {

	set pt_out_filtered =
	  filter_formal_out_context_according_to_formal_in_context
	  (pt_out, pt_in_filtered, wpl, f);

	/* Explicitly written pointers imply some arc removals; pointer
	   assignments directly or indirectly in the callee. */
	// pt_end = set_difference(pt_end, pt_caller_s, pts_kill);

	/* FI: pt_caller_s may have been modified implictly because the
	 * formal context has been increased according to the needs of
	 * the callee. But pt_caller_s may also have been updated by what
	 * has happened prevously when analyzing the current
	 * statement. See for instance Pointers/sort01.c.
	 */
	set c_pt_caller_s = points_to_graph_set(points_to_context_statement_in());
	c_pt_caller_s = set_union(c_pt_caller_s, c_pt_caller_s, pt_caller_s);

	list tcwpl = points_to_cells_exact_translation(cwpl, binding, f);
	/* Compute pt_kill_2 */
	set pt_kill = compute_points_to_kill_set(tcwpl, c_pt_caller_s, binding);
	/* Implicitly written pointers imply some arc removals:
	 * free(), tests and exits. These are the elements of
	 * pt_kill_1, although the equations do not seem to fit at
	 * all since pt_in_filtered is not an argument...
	 */
	add_implicitly_killed_arcs_to_kill_set(pt_kill, 
					       wpl,
					       c_pt_caller_s,
					       pt_out_filtered,
					       binding,
					       f);
	ifdebug(8) print_points_to_set("pt_kill", pt_kill);

	set pt_end = new_simple_pt_map();
	// FI: c_pr_in_s is probably pt_{caller} in the dissertation
	pt_end = set_difference(pt_end, c_pt_caller_s, pt_kill);

	set pt_gen_1 = compute_points_to_gen_set(pt_out_filtered,
						 wpl,
						 binding, f);

	if(set_undefined_p(pt_gen_1)) {
	  /* Translation failure, incompatibility between the call site
	     and the callee. */
	  pips_user_warning("Incompatibility between call site and "
			    "callee's output.\n");
	  out_bottom_p = true;
	}
	else {
	  pips_assert("pt_gen_1 is consistent", consistent_points_to_set(pt_gen_1));

	  // FI->FI: Not satisfying; kludge to solve issue with Pointers/inter04
	  // FI->FI: this causes a core dump for Pointers/formal_parameter01.c
	  // A lot should be said about this test case, which is revealing
	  // because of the test it contains, and because of the
	  // pointer arithmetic, and because the useless primature
	  // expansion of_pi_1[1] in _pi_1[*] which is semantically
	  // correct but fuzzy as it implies possible unexisting arcs
	  // for _pi_1[0], _pi_1[2], etc...
	  // FI->FI: the upgrade must be postponed
	  /*
	  pt_map pt_gen_1_g = make_points_to_graph(false, pt_gen_1);
	  upgrade_approximations_in_points_to_set(pt_gen_1_g);

	  pips_assert("pt_gen_1 is consistent after upgrade",
		      consistent_points_to_set(pt_gen_1));
	  */

	  /* Some check */
	  list stubs = points_to_set_to_module_stub_cell_list(f, pt_gen_1, NIL);
	  if(!ENDP(stubs)) {
	    pips_internal_error("Translation failure in pt_gen_1.\n");
	  }

	  /* Use written/wpl to reduce the precision of exact arcs in
	   * pt_end. This is equivalent to pt_kill_3 and pt_gen_3.
	   *
	   *
	   * FI: I do not understand why the precision of the write is not
	   * exploited. We may need to use mwpl instead of wpl
	   *
	   */
	  list twpl = points_to_cells_translation(wpl, binding, f);
	  pt_end =
	    lower_points_to_approximations_according_to_write_effects(pt_end, twpl);
	  // FI: I keep it temporarily for debugging purposes
	  // gen_free_list(twpl);

	  // FI: set_union is unsafe; the union of two consistent
	  // points-to graph is not a consistent points-to graph
	  pt_end = set_union(pt_end, pt_end, pt_gen_1);
	  pips_assert("pt_end is consistent", consistent_points_to_set(pt_end));
	  ifdebug(8) print_points_to_set("pt_end =",pt_end);
	  /* Some check */
	  stubs = points_to_set_to_module_stub_cell_list(f, pt_end, NIL);
	  if(!ENDP(stubs)) {
	    pips_internal_error("Translation failure in pt_end.\n");
	  }
	  points_to_graph_set(pt_end_f) = pt_end;
	}
      }
      else {
	pips_user_warning("Aliasing between arguments at line %d.\n"
			  "We would have to create a new formal context "
			  "and to restart the points-to analysis "
			  "and to modify the IN and OUT data structures...\n"
			  "Or use a simpler analysis, here an intraprocedural one.\n",
			  points_to_context_statement_line_number());

	pt_end_f = user_call_to_points_to_intraprocedural(c, pt_caller, el);
      }
    }
    else
      out_bottom_p = true;
  }
  else {
    // FI: I do not think we want this warning in general!
    pips_user_warning("Function \"%s\" has no side effect on its formal context "
		      "via pointer variables.\n", entity_user_name(f));
  }

  /* This cannot be performed earlier because the points-to
     precondition of the caller may have to be enriched according to
     the formal context of the callee, or the effect computation is
     going to fail. */
  if(out_bottom_p) {
    clear_pt_map(pt_end_f);
    points_to_graph_bottom(pt_end_f) = true;
  }

  return pt_end_f;
}

pt_map user_call_to_points_to_intraprocedural(call c,
					      pt_map pt_in,
					      list el __attribute__ ((unused)))
{
  pt_map pt_out = pt_in;
  list al = call_arguments(c);

  FOREACH(expression, arg, al) {
    list l_sink = expression_to_points_to_sources(arg, pt_out);
    set pt_out_s = points_to_graph_set(pt_out);
    SET_FOREACH(points_to, pts, pt_out_s) {
      FOREACH(cell, cel, l_sink) {
	cell source = points_to_source(pts);
	if(cell_equal_p(source, cel)) {
	  cell sink = points_to_sink(pts);
	  if(source_in_graph_p(sink, pt_out))
	    remove_arcs_from_pt_map(pts, pt_out_s);
	}
      }
    }
  }
  return pt_out;
}

/* Translate the "out" set into the scope of the caller
 *
 * Shouldn't it be the "written" list that needs to be translated?
 *
 */ 
set compute_points_to_kill_set(list written_must_translated,
			       set pt_caller,
			       set binding __attribute__ ((unused)))
{
  set kill = new_simple_pt_map(); 	
  list written_cs = written_must_translated;
#if 0
  list written_cs = NIL;
  set bm = binding;

  FOREACH(CELL, c, written) {
    cell new_sr = cell_undefined; 
    reference r_1 = cell_any_reference(c);
    entity v_1 = reference_variable(r_1);
    if(!formal_parameter_p(v_1)) {
      list ind1 = reference_indices(r_1);
      SET_FOREACH(points_to, pp, bm) {
	cell sr2 = points_to_source(pp); 
	cell sk2 = points_to_sink(pp); 
	reference r22 = copy_reference(cell_to_reference(sk2));
	/* FI->AM: this test is loop invariant... and performed within the loop */
	if(!source_in_set_p(c, bm)) {
	  reference r_12 = cell_any_reference(sr2);
	  entity v_12 = reference_variable( r_12 );
	  if(same_string_p(entity_local_name(v_1),entity_local_name(v_12))) { 
	    reference_indices(r22) = gen_nconc(reference_indices(r22),
					       gen_full_copy_list(ind1));
	    new_sr = make_cell_reference(r22);
	    // FI->AM: I guess this statement was forgotten
	    written_cs = CONS(CELL, new_sr, written_cs);
	    break;
	  }
	  else if (heap_cell_p(c)) {
	    written_cs = CONS(CELL, c, written_cs);
	  }
	}
	else if(points_to_compare_cell(c,sr2)) {
	  written_cs = CONS(CELL, points_to_sink(pp), written_cs);
	  // break; // FI: why not look for other possible translations?
	}
      }
    }
  }
#endif
  
  /* Remove all points-to arc from pt_caller whose origin has been
     fully written */
  FOREACH(CELL, c, written_cs) {
    SET_FOREACH(points_to, pt, pt_caller) {
      if(points_to_cell_equal_p(c, points_to_source(pt))
	 && atomic_points_to_cell_p(c))
	set_add_element(kill, kill, (void*)pt);
    }
  }

  return kill;
}

/* Compute the list of cells that correspond to cell "sr1" according
 * to the translation mapping "bm" when function "f" is called.
 *
 * The check with rv may be useless, for instance when a sink cell is
 * checked, as it is impossible (in C at least) to points toward the
 * return value.
 */
list points_to_cell_translation(cell sr1, set binding, entity f)
{
  list new_sr_l = NIL;
  entity rv = any_function_to_return_value(f);

  /* Translate sr1 if needed */
  reference r_1 = cell_any_reference(sr1);
  entity v_1 = reference_variable(r_1);
  list ind1 = reference_indices(r_1);
  if(entity_anywhere_locations_p(v_1)
     || entity_typed_anywhere_locations_p(v_1)
     || heap_cell_p(sr1)
     || v_1 == rv
     || entity_to_module_entity(v_1)!=f) {
    /* No translation is needed. */
    cell new_sr = copy_cell(sr1);
    new_sr_l = CONS(CELL, new_sr, new_sr_l);
  }
  else {
    SET_FOREACH(points_to, pp, binding) {
      cell sr2 = points_to_source(pp); 
      cell sk2 = points_to_sink(pp); 
      reference r22 = copy_reference(cell_to_reference(sk2));
      // FI: this test should be factored out
      if(!source_in_set_p(sr1, binding)) {
	// sr1 cannot be translated directly, let's try to remove
	// (some) subscripts
	reference r_12 = cell_any_reference(sr2);
	entity v_12 = reference_variable( r_12 );
	if(same_string_p(entity_local_name(v_1),entity_local_name(v_12))) {
	  /* We assume here that the subscript list of sr1, the
	     reference to translate, is longer than the subscript
	     list of sr2, the source of its translation. */
	  list ind2 = reference_indices(r_12);
	  pips_assert("The effective subscript list is longer than "
		      "the translated subscript list",
		      gen_length(ind1)>=gen_length(ind2));
	  // Either we check the subscript compatibility or we trust it
	  // Let's trust it: no!
	  list cind1 = ind1, cind2 = ind2;
	  bool compatible_p = true;
	  while(!ENDP(cind2)) {
	    expression s1 = EXPRESSION(CAR(cind1));
	    expression s2 = EXPRESSION(CAR(cind2));
	    if(!compatible_points_to_subscripts_p(s1, s2)) {
	      compatible_p = false;
	      break;
	    }
	    POP(cind1), POP(cind2);
	  }
	  if(compatible_p) {
	    // Propagate the remaining subscripts on the translation target
	    reference_indices(r22) = gen_nconc(reference_indices(r22),cind1);
	    cell new_sr = make_cell_reference(r22);
	    new_sr_l = CONS(CELL, new_sr, new_sr_l);
	  }
	}
      }
      else if(points_to_compare_cell(sr1,sr2)) {
	// sr1 can be translated directly as sk2
	cell new_sr = copy_cell(points_to_sink(pp));
	new_sr_l = CONS(CELL, new_sr, new_sr_l);
      }
    }
  }
  return new_sr_l;
}

/* Allocate a new list with the translations of the cells in cl, when
 * their translation make sense. Effects on copied parameters are
 * discarded.
 *
 * If exact_p is required, translate only cells that can be translated
 * exactly.
 */
list generic_points_to_cells_translation(list cl, set binding, entity f, bool exact_p)
{
  list tcl = NIL;
  FOREACH(CELL, c, cl) {
    reference r = cell_any_reference(c);
    entity v = reference_variable(r);
    list ptcl = NIL;
    if(formal_parameter_p(v)) {
      type t = entity_basic_concrete_type(v);
      if(array_type_p(t)) {
	// Passing by reference
	ptcl = points_to_cell_translation(c, binding, f);
      }
      else {
	// Passing by value: no need to translate information about a copy
	;
      }
    }
    else
      ptcl = points_to_cell_translation(c, binding, f);
    if(exact_p && gen_length(ptcl)>1)
      gen_free_list(ptcl);
    else
      tcl = gen_nconc(tcl, ptcl);
  }
  return tcl;
}

/* Allocate a new list with the translations of the cells in cl, when
 * their translation make sense. Effects on copied parameters are
 * discarded.
 */
list points_to_cells_translation(list cl, set binding, entity f)
{
  return generic_points_to_cells_translation(cl, binding, f, false);
}

/* Allocate a new list with the translations of the cells in cl, when
 * their translation make sense and is unique (one-to-one
 * mapping). Effects on copied parameters are discarded.
 */
list points_to_cells_exact_translation(list cl, set binding, entity f)
{
  return generic_points_to_cells_translation(cl, binding, f, true);
}

/* Translate the out set in the scope of the caller using the binding
 * information, but eliminate irrelevant arcs using Written and the
 * type of the source.
 *
 * This is pt_gen_1 in Amira Mensi's dissertation.
 *
 * Also, pay attention to translation errors because they are related
 * to programming bugs, such as pointer arithmetic applied to pointers
 * to scalar.
 */
set compute_points_to_gen_set(set pt_out,
			      list Written,
			      set binding,
			      entity f)
{
  set gen = new_simple_pt_map(); 		
  // To avoid redundant error messages
  list translation_failures = NIL;

  /* Consider all points-to arcs "(sr1, sk1)" in "pt_out" */
  SET_FOREACH(points_to, p, pt_out) {
    cell sr1 = points_to_source(p); 
    reference r_1 = cell_any_reference(sr1);
    entity v_1 = reference_variable(r_1);
    type t_1 = entity_basic_concrete_type(v_1);
    /* Keep arcs whose source is:
     *
     *  - an array, because it has certainly not been passed by copy;
     *
     *  - not a scalar formal parameter: again, no copy passing;
     *
     *  - not written by the procedure, because, even if it passed by
     *    copy, the actual parameter is aliased.
     *
     * In other word, get rid of scalar formal parameter that are written.
     */
    if(array_type_p(t_1)
       || !formal_parameter_p(v_1)
       || !points_to_cell_in_list_p(sr1, Written)) {
      list new_sk_l = NIL;
      approximation a = copy_approximation(points_to_approximation(p));
      cell sk1 = points_to_sink(p); 

      /* Translate sr1 */
      list new_sr_l = points_to_cell_translation(sr1, binding, f);

      if(!ENDP(new_sr_l)) {
	/* Translate sk1 if needed */
	reference r_2 = cell_any_reference(sk1);
	entity v_2 = reference_variable(r_2);
	if (null_cell_p(sk1) || nowhere_cell_p(sk1) || heap_cell_p(sk1)
	    || anywhere_cell_p(sk1) || cell_typed_anywhere_locations_p(sk1)
	    || entity_to_module_entity(v_2)!=f) {
	  cell new_sk = copy_cell(sk1);
	  new_sk_l = CONS(CELL, new_sk, new_sk_l);
	}
	else
	  new_sk_l = points_to_cell_translation(sk1, binding, f);

	if(!ENDP(new_sk_l)) {
	  int new_sk_n = (int) gen_length(new_sk_l);
	  FOREACH(CELL, new_sr, new_sr_l) {
	    approximation na = approximation_undefined;
	    if(!atomic_points_to_cell_p(new_sr)
	       || new_sk_n>1
	       || (new_sk_n==1
		   && !atomic_points_to_cell_p(CELL(CAR(new_sk_l))))) {
	      na = make_approximation_may();
	    }
	    else
	      na = copy_approximation(a);
	    FOREACH(CELL, new_sk, new_sk_l) {
	      points_to new_pt = make_points_to(copy_cell(new_sr),
						copy_cell(new_sk),
						na,
						make_descriptor_none());
	      set_add_element(gen, gen, (void*)new_pt);
	    }
	  }
	  // gen_full_free_list(new_sr_l);
	  // gen_full_free_list(new_sk_l);
	  free_approximation(a);
	}
	else {
	  /* The translation of pt's sink failed. */
	  /* Note: the piece of code below is replicated */
	  approximation a = points_to_approximation(p);
	  if(approximation_may_p(a)) {
	    if(!points_to_cell_in_list_p(sk1, translation_failures)) {
	      pips_user_warning("Points-to sink cell sk1=\"%s\" could not be translated.\n",
				points_to_cell_name(sk1));
	      translation_failures = CONS(CELL, sk1, translation_failures);
	    }
	  }
	  else {
	    pips_user_warning("Points-to sink cell sk1=\"%s\" could not be translated but has to be.\n",
			    points_to_cell_name(sk1));
	    set_free(gen);
	    gen = set_undefined;
	    break;
	  }
	}
      }
      else {
	/* The translation of pt's source failed. This may occur
	 * because the callee had to assume a pointer points to an
	 * array, whereas the call site associates it to a scalar.
	 *
	 * See for instance Pointers/formal_parameter01.c
	 *
	 * But this may also occur because the formal parameter cannot
	 * be translated because the effective argument is an
	 * address_of expression. See for instance
	 * EffectsWithPointsTO/call01.c.
	 *
	 * We have no way to guess here the reason for the translation
	 * failure...
	 */
	entity v = reference_variable(cell_any_reference(sr1));
	// FI: we should check that it is a formal parameter of "f"...
	if(!formal_parameter_p(v)) {
	approximation a = points_to_approximation(p);
	if(approximation_may_p(a)) {
	  if(!points_to_cell_in_list_p(sr1, translation_failures)) {
	    pips_user_warning("Points-to source cell sr1=\"%s\" could not be translated.\n",
			      points_to_cell_name(sr1));
	    translation_failures = CONS(CELL, sr1, translation_failures);
	  }
	}
	else {
	  /* Could be a user error, but you may prefer to go on with
	   * the points-to analysis to perform some dead code
	   * elimination later...
	   */
	  pips_user_warning("Points-to source cell sr1=\"%s\" could not be translated but has to be.\n",
			  points_to_cell_name(sr1));
	  set_free(gen);
	  gen = set_undefined;
	break;
	}
	}
      }
    }
  }

  gen_free_list(translation_failures);

  ifdebug(1) {
    if(set_undefined_p(gen))
      fprintf(stderr, "pt_gen is bottom\n");
    else
      print_points_to_set("pt_gen_1", gen);
  }

  return gen;  
}


/* Recursively find all the arcs, "ai", starting from the argument
 * "c1" using "in", find all the arcs, "aj", starting from the
 * parameter "c2" using "pt_binded", map each node "ai" to its
 * corresponding "aj" and store the "ai->aj" arc in a new set, "bm".
 *
 * "pt_binded" contains the correspondance between formal and actual
 * parameters, e.g. "fp->ap", with some information about the possible
 * approximations because one formal parameter can points toward
 * several actual memory locations of the caller.
 *
 * "in" contains the formal context of the callee, as it stands at its
 * entry point (DBR_POINTS_TO_IN).
 *
 * "bm" is the binding relationship between references of the formal
 * context of the callees towards addresses of the caller. For
 * instance, when function "void foo(int ** fp)" is called as "ap=&q;
 * foo(ap);", bm = {(_ap_1, q)}.
 *
 * See Amira Mensi's PhD dissertation, chapter about interprocedural analysis
 */
set points_to_binding_arguments(cell c1, cell c2 , set in, set pt_binded)
{
  set bm = new_simple_pt_map();

  if(source_in_set_p(c1, in) || source_subset_in_set_p(c1, in)) {
    // FI: impedance problem... and memory leak
    points_to_graph in_g = make_points_to_graph(false, in);
    points_to_graph pt_binded_g = make_points_to_graph(false, pt_binded);
    // FI: allocate a new copy for sink1 and sink2
    //list c1_children = cell_to_pointer_cells(c1);
    //list c2_children = cell_to_pointer_cells(c2);
    list c1_children = recursive_cell_to_pointer_cells(c1);
    list c2_children = recursive_cell_to_pointer_cells(c2);
    FOREACH(CELL,c1c, c1_children) {
      FOREACH(CELL,c2c, c2_children) {
	list sinks1 = points_to_source_to_some_sinks(c1c, in_g, true);
	list sinks2 = points_to_source_to_some_sinks(c2c, pt_binded_g, true);
	pips_assert("sinks1 is not empty", !ENDP(sinks1));
	pips_assert("sinks2 is not empty", !ENDP(sinks2));
	// FI Allocate more copies
	list tmp1 = gen_full_copy_list(sinks1);
	list tmp2 = gen_full_copy_list(sinks2);

	FOREACH(CELL, s1, sinks1) { // Formal cell: fp->_fp_1 or fp->NULL
	  // FI: no need to translate constants NULL and UNDEFINED
	  if(!null_cell_p(s1) && !nowhere_cell_p(s1)) {
	    FOREACH(CELL, s2, sinks2) { // Actual cell: ap-> i... NOWHERE... NULL...
	      // FI: _fp_1 may or not exist since it is neither null nor undefined
	      if(!null_cell_p(s2) && !nowhere_cell_p(s2)) {
		approximation a = approximation_undefined;
		if((size_t)gen_length(sinks2)>1) // FI->FI: atomicity should be tested too
		  a = make_approximation_may();
		else
		  a = make_approximation_exact();
		cell sink1 = copy_cell(s1);
		cell sink2 = copy_cell(s2);
		// Build arc _fp_1->... NOWHERE ... NULL ...
		points_to pt = make_points_to(sink1, sink2, a, make_descriptor_none());
		add_arc_to_simple_pt_map(pt, bm);
	      }
	      //gen_remove(&sinks2, (void*)s2);
	    }
	  }
	  //gen_remove(&sinks1, (void*)s1);
	}

	/* Recursive call down the different points_to paths*/
	FOREACH(CELL, sr1, tmp1) {
	  if(!null_cell_p(sr1) && !nowhere_cell_p(sr1)) {
	    FOREACH(CELL, sr2, tmp2) {
	      if(!null_cell_p(sr2) && !nowhere_cell_p(sr2)) {
		set sr1sr2 = points_to_binding_arguments(sr1, sr2, in, pt_binded);
		bm = set_union(bm, sr1sr2, bm);
		set_clear(sr1sr2), set_free(sr1sr2);
	      }
	    }
	  }
	}
      }
    }
  }
  else if(source_subset_in_set_p(c1, in)) { // Not reachable
    /* Here we have, for instance, "q[*]" in "c1" for "q[4]" in "in". */
    /* FI: I do not see how to handle incompatibility between assumptions... */
    pips_internal_error("Do not know how to handle subsets...\n");
  }
  else /* (!source_in_set_p(c1, in) && !source_subset_in_set_p(c1, in)) */ {
    // FI: this has already been performed
    /* c1 is not a pointer: it is simply mapped to c2 */
    // points_to pt = make_points_to(c1, c2, make_approximation_exact(), make_descriptor_none());
    // add_arc_to_simple_pt_map(pt, bm);
    ;
  }
  return bm;
}

/* Filter out written effects on pointers */
static list generic_written_pointers_set(list eff, bool exact_p)
{
  list written_l = NIL;
  debug_on("EFFECTS_DEBUG_LEVEL");
  list wr_eff = effects_write_effects(eff);
  FOREACH(effect, ef, wr_eff) {
    approximation a = effect_approximation(ef);
    if(!exact_p || approximation_must_p(a) || approximation_exact_p(a)) {
      if(effect_pointer_type_p(ef)){
	cell c = effect_cell(ef);
	written_l = gen_nconc(CONS(CELL, c, NIL), written_l);
      }
    }
  }
  debug_off();
  return written_l; 
}

/* Filter out written effects on pointers */
list written_pointers_set(list eff) {
  return generic_written_pointers_set(eff, false);
}

/* Filter out certainly written effects on pointers */
list certainly_written_pointers_set(list eff) {
  return generic_written_pointers_set(eff, true);
}

/* For each actual argument "r" and its corresponding formal one "f",
 * create the assignment "f = r;" and then compute the points-to set
 * "s" generated by the assignment. The result is the union of
 * "pt_caller" and "s".
 */
set compute_points_to_binded_set(entity called_func,
				 list real_args,
				 set pt_caller,
				 bool * success_p)
{ 
  set s = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set pt_binded = set_generic_make(set_private, points_to_equal_p,
				   points_to_rank);

  *success_p = true; // default value

  /* Be careful with vararags
   *
   * This is not sufficient to handle varargs. Much more thinking
   * needed. And corect examples.
   */
  type ft = entity_basic_concrete_type(called_func); // function type
  functional fft = type_functional(ft);
  list ptl = functional_parameters(fft); // parameter type list
  int mnp = (int) gen_length(ptl); // maximum number of formal parameters
  if(!ENDP(ptl)) {
    // last parameter type
    type lpt = parameter_type(PARAMETER(CAR(gen_last(ptl))));
    if(type_varargs_p(lpt))
      mnp--;
  }

  cons *pc;
  int ipc;
  s = set_assign(s, pt_caller);
  for (ipc = 1, pc = real_args; pc != NIL; pc = CDR(pc), ipc++) {
    expression rhs = EXPRESSION(CAR(pc));
    int tr = ipc>mnp? mnp : ipc;
    entity fp = find_ith_parameter(called_func, tr);
    type fpt = entity_basic_concrete_type(fp);
    if(array_type_p(fpt)) {
      /* C does not support array assignments... */
      if(expression_reference_p(rhs)) {
	reference r = expression_reference(rhs);
	entity v = reference_variable(r);
	list nptl = NIL; // New points-to list
	SET_FOREACH(points_to, pt, pt_caller) {
	  cell c = points_to_source(pt);
	  reference cr = cell_any_reference(c);
	  entity cv = reference_variable(cr);
	  if(cv==v) {
	    points_to npt = copy_points_to(pt);
	    cell nc = points_to_source(npt);
	    reference ncr = cell_any_reference(nc);
	    reference_variable(ncr) = fp;
	    nptl = CONS(POINTS_TO, npt, nptl);
	  }
	}
	FOREACH(POINTS_TO, npt, nptl)
	  add_arc_to_simple_pt_map(npt, s);
	gen_free_list(nptl);
      }
      else if(expression_call_p(rhs) && expression_string_constant_p(rhs)) {
	/* This may happen with a constant string as actual parameter
	   and an array, bounded or not, as formal parameter. */
	; // Nothing to do: the constant string does not point to anything
      }
      else if(expression_call_p(rhs)) {
	/* A formal array can be called with a dereferencing expression
	 *
	 * See Pointers/Mensi.sub/array_pointer_free01: array fp is
	 * associated to &p[0] or to p->q or to c.a ...
	 */
	call c = expression_call(rhs);
	entity f = call_function(c);
	pips_internal_error("Not implemented yet. Function \"%s\".\n",
			    entity_user_name(f));
      }
      else
	pips_internal_error("Not implemented yet.\n");
    }
    else {
      /* It would be nice to build an assignment of rhs to fp and to
	 let it deal with the many possible kinds of assignments. But
	 if it is a pure points-to function, the symbolic subscripts
	 are going to be lost. This is fine for points-to translation,
	 but not OK for effects translation. */

      if(pointer_type_p(fpt)) {
	points_to_graph s_g = make_points_to_graph(false, s);
	list sinks = expression_to_points_to_cells(rhs, s_g, true, false);
	int nsinks = (int) gen_length(sinks);
	FOREACH(CELL, sink, sinks) {
	  cell o = make_cell_reference(make_reference(fp, NIL));
	  cell d = copy_cell(sink);
	  approximation a = nsinks==1? make_approximation_exact() :
	    make_approximation_may();
	  descriptor desc = make_descriptor_none();
	  points_to pt = make_points_to(o, d, a, desc);
	  add_arc_to_simple_pt_map(pt, s);
	}
      }
      else if(struct_type_p(fpt)) {
	/* In the short term, build the assignment... */
	expression lhs = entity_to_expression(fp);
	points_to_graph s_g = make_points_to_graph(false, s);
	points_to_graph a_g = assignment_to_points_to(lhs, rhs, s_g);
	if(points_to_graph_bottom(a_g)) {
	  /* The assignment failed because the call site is not
	     compatible with the caller. */
	  *success_p = false;
	}
	else {
	  s = set_assign(s, points_to_graph_set(a_g));
	}
      }
      else {
	; // do nothing for other types
      }
    }
  }

  SET_FOREACH(points_to, pt, s) {
    reference r = cell_any_reference(points_to_sink(pt));
    entity e = reference_variable(r);
    if(stub_entity_of_module_p(e, called_func))
      s = set_del_element(s,s,(void*)pt);
  }
  pt_binded = set_union(pt_binded, s, pt_caller);
  return pt_binded;
}


/* Apply points_to_binding_arguments() to each pair (, complete the
 * process of binding each element of "in" to its corresponding memory
 * address at the call site. Necessary to translate the fields of structures.
 *
 * "args": list of formal parameters of some callee
 *
 * "in": points-to in of the callee
 *
 * "pt_binded": points-to from formal to actual parameters for a specific call site
 *
 * A new set is allocated.
 */
set points_to_binding(list args, set in, set pt_binded)
{

  set bm = new_simple_pt_map();
  //set bm1 = new_simple_pt_map();
 
  /* Process each formal parameter and look for its actual values in
     "pt_binded" */
  SET_FOREACH(points_to, pt, pt_binded) {
    FOREACH(CELL, c1, args) {
      cell source = points_to_source(pt);
      if(cell_equal_p(c1, source)) {
	cell c2 = points_to_source_alias(pt, pt_binded);
	// FI: We end up with c1=c2=one formal parameter...
	// No need to add "p->p" in "bm"...
	//approximation a = make_approximation_exact();
	//points_to new_pt = make_points_to(c1, c2, a, make_descriptor_none());
	//add_arc_to_simple_pt_map(new_pt, bm);
	set c1c2 = points_to_binding_arguments(c1, c2,  in, pt_binded);
	bm = set_union(bm, bm, c1c2);
	set_clear(c1c2), set_free(c1c2);
      }
      else if(cell_entity_equal_p(c1, source)) {
	pips_assert("c1 is a reference with no indices",
		    ENDP(reference_indices(cell_any_reference(c1))));
	cell c2 = copy_cell(source);
	// FI: We end up with c1=c2=one formal parameter...
	// No need to add "p->p" in "bm"...
	//approximation a = make_approximation_exact();
	//points_to new_pt = make_points_to(c1, c2, a, make_descriptor_none());
	//add_arc_to_simple_pt_map(new_pt, bm);
	set c1c2 = points_to_binding_arguments(source, c2,  in, pt_binded);
	bm = set_union(bm, bm, c1c2);
	set_clear(c1c2), set_free(c1c2);
      }
    }
  }

  return bm;
}

/* Add cells referencing a points-to stub found in parameter "s" are
 * copied and added to list "osl".
 *
 * The stubs are returned as cells not as entities.
 *
 * New cells are allocated. No sharing is created between parameter
 * "s" and result "sl".
 */
list generic_points_to_set_to_stub_cell_list(entity f, set s, list osl)
{
  list sl = osl;
  SET_FOREACH(points_to, pt, s) {
    cell sink = points_to_sink(pt);
    reference r1 = cell_any_reference(sink);
    entity e1 = reference_variable(r1);
    if( ( (entity_undefined_p(f) && entity_stub_sink_p(e1))
	  || stub_entity_of_module_p(e1, f) )
	&& !points_to_cell_in_list_p(sink, sl) )
      sl = CONS(CELL, copy_cell(sink), sl);
      
    cell source = points_to_source(pt);
    reference r2 = cell_any_reference(source);
    entity e2 = reference_variable(r2);
    if( ( (entity_undefined_p(f) && entity_stub_sink_p(e2))
	  || stub_entity_of_module_p(e2, f) )
	&& !points_to_cell_in_list_p(source, sl) )
      sl = CONS(CELL, copy_cell(source), sl);
  }
  
  gen_sort_list(sl, (gen_cmp_func_t) points_to_compare_ptr_cell );
  ifdebug(1) print_points_to_cells(sl);
  return sl;
}

list points_to_set_to_stub_cell_list(set s, list osl)
{
  return generic_points_to_set_to_stub_cell_list(entity_undefined, s, osl);
}

list points_to_set_to_module_stub_cell_list(entity m, set s, list osl)
{
  return generic_points_to_set_to_stub_cell_list(m, s, osl);
}

/* Let "pt_binded" be the results of assignments of actual arguments
 * to formal arguments (see compute_points_to_binded_set()).
 *
 * Let "pt" be a points-to arc in "pt_binded".
 *
 * Find for the source of p its corresponding alias, which means
 * finding another source that points to the same location.
 */
cell points_to_source_alias(points_to pt, set pt_binded)
{
  cell source = cell_undefined;
  cell sink1 = points_to_sink(pt);
  SET_FOREACH(points_to, p, pt_binded) {
    cell sink2 =  points_to_sink(p);
    if(cell_equal_p(sink1, sink2)) {
      source = points_to_source(p);
      break;
      }
  }
  if(cell_undefined_p(source))
    pips_internal_error("At least one alias should be found.\n");

  return source;
}
