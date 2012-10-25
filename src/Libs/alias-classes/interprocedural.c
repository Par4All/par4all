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

points_to_graph user_call_to_points_to(call c,
				       points_to_graph pt_in,
				       list el) // effect list
{
  points_to_graph pt_out = pt_in;
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
    pt_out = user_call_to_points_to_interprocedural(c, pt_in, el);
    }
  else if(fast_interprocedural_points_to_analysis_p()) {
    //pt_out = user_call_to_points_to_fast_interprocedural(c, pt_in, el);
    pt_out = user_call_to_points_to_interprocedural(c, pt_in, el);
    }
  else {
    //pt_out = user_call_to_points_to_intraprocedural(c, pt_in, el);
    pt_out = user_call_to_points_to_interprocedural(c, pt_in, el);
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
  type t = ultimate_type(entity_type(call_function(c)));
  type rt = ultimate_type(functional_result(type_functional(t)));
  entity ne = entity_undefined;
  list sinks = NIL;
  entity f = call_function(c);
  // Interprocedural version
  // Check if there is a return value at the level of POINTS TO OUT, if yes return its sink
  if(interprocedural_points_to_analysis_p() ||fast_interprocedural_points_to_analysis_p() ) {
    const char* mn = entity_local_name(f);
    points_to_list pts_to_out = (points_to_list)
      db_get_memory_resource(DBR_POINTS_TO_OUT, module_local_name(f), true);
    list l_pt_to_out = gen_full_copy_list(points_to_list_list(pts_to_out));
    set pt_out_callee = new_simple_pt_map();
    pt_out_callee = set_assign_list(pt_out_callee, l_pt_to_out);
    SET_FOREACH( points_to, pt, pt_out_callee) {
      cell s = points_to_source(pt);
      reference sr = cell_any_reference(s);
      entity se = reference_variable(sr);
      const char* sn = entity_local_name(se);
      if( strcmp(mn, sn)==0) {
	cell sc = copy_cell(points_to_sink(pt));
	sinks = gen_nconc(CONS(CELL, sc, NULL), sinks);
      }
    }
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
  entity f = call_function(c);
  list al = call_arguments(c);
  list dl = code_declarations(value_code(entity_initial(f)));
  list fpcl = points_to_cells_parameters(dl);   
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
  set pts_binded = compute_points_to_binded_set(f, al, pt_in_s);
  ifdebug(8) print_points_to_set("pt_binded", pts_binded);
  set bm = points_to_binding(fpcl, pt_in_callee, pts_binded);
  set pts_kill = compute_points_to_kill_set(wpl, pt_in_s, fpcl,
					    pt_in_callee, pts_binded, bm);
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
  pt_end = set_difference(pt_end, pt_in_s, pts_kill);
  pt_end = set_union(pt_end, pt_end, pts_gen);
  ifdebug(8) print_points_to_set("pt_end =", pt_end);
  points_to_graph_set(pt_out) = pt_end;
  return pt_out;
}

/* This function is very similar to
   filter_formal_context_according_to_actual_context(), but a little
   bit more tricky. Amira Mensi unified both in her own version, but
   the unification makes the maintenance more difficult. */
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
    if(points_to_cell_in_list_p(source, fcl)) {
      cell sink = points_to_sink(pt);
      list tsl = points_to_source_to_sinks(source, translation_g, false);
      // FI: this assert may be too strong
      pips_assert("Elements of \"fcl\" can be translated", !ENDP(tsl));
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
	    break;
	  }
	  else {
	    ; // do not copy this arc in filtered set
	  }
	}
      }
      gen_free_list(tsl);
    }
  }

  /* Compute the translation relation for sinks of the formal arguments */
  list nfcl = NIL;
  FOREACH(CELL, c, fcl) {
    points_to_graph filtered_g = make_points_to_graph(false, filtered);
    list fl = points_to_source_to_sinks(c, filtered_g, false); // formal list
    if(!ENDP(fl)) {
      list tsl = points_to_source_to_sinks(c, translation_g, false);
      FOREACH(CELL, tc, tsl) {
	points_to_graph pts_binded_g = make_points_to_graph(false, pts_binded);
	list al = points_to_source_to_sinks(tc, pts_binded_g, false); // formal list
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
	      points_to tr = make_points_to(copy_cell(fc), copy_cell(ac), 
					    copy_approximation(a),
					    make_descriptor_none());
	      add_arc_to_simple_pt_map(tr, translation);
	    }
	    nfcl = CONS(CELL, fc, nfcl);
	  }
	}
	free_approximation(a);
      }
    }
  }

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

/* Filter pt_in_callee according to pts_binded. For instance, a
   formal parameter can point to NULL in pt_in_callee only if it
   also points to NULL in pts_binded. In the same way, a formal
   parameter can point to a points-to stub in pt_in_callee only if
   it points to a non-NULL target in pts_binded. Also, a formal
   parameter cannot points exactly to UNDEFINED in pts_binded as
   it would be useless (not clear if we can remove such an arc
   when it is a may arc...). Finally, each formal parameter must
   still point to something. */
set filter_formal_context_according_to_actual_context(list fpcl,
						      set pt_in_callee,
						      set pts_binded,
						      set translation)
{
  set filtered = new_simple_pt_map();

  /* Copy only possible arcs "pt" from "pt_in_callee" into the "filtered" set */
  SET_FOREACH(points_to, pt, pt_in_callee) {
    cell source = points_to_source(pt);
    if(points_to_cell_in_list_p(source, fpcl)) {
      cell sink = points_to_sink(pt);
      if(null_cell_p(sink)) {
	/* Do we have the same arc in pts_binded? */
	if(arc_in_points_to_set_p(pt, pts_binded)) {
	  points_to npt = copy_points_to(pt);
	  add_arc_to_simple_pt_map(npt, filtered);
	}
	else {
	  ; // do not copy this arc in filtered set
	}
      }
      else {
	if(cell_points_to_non_null_sink_in_set_p(source, pts_binded)) {
	  points_to npt = copy_points_to(pt);
	  add_arc_to_simple_pt_map(npt, filtered);
	}
	else {
	  ; // do not copy this arc in filtered set
	}
      }
    }
  }

  /* Compute the translation relation for sinks of the formal arguments */
  list fcl = NIL;
  points_to_graph filtered_g = make_points_to_graph(false, filtered);
  points_to_graph pts_binded_g = make_points_to_graph(false, pts_binded);
  FOREACH(CELL, c, fpcl) {
    list fl = points_to_source_to_sinks(c, filtered_g, false); // formal list
    list al = points_to_source_to_sinks(c, pts_binded_g, false); // formal list
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
	  points_to tr = make_points_to(copy_cell(fc), copy_cell(ac), a,
					make_descriptor_none());
	  add_arc_to_simple_pt_map(tr, translation);
	}
	fcl = CONS(CELL, fc, fcl);
      }
    }
  }

  ifdebug(8) {
    pips_debug(8, "First filtered IN set for callee at call site:\n");
    print_points_to_set("", filtered);
    pips_debug(8, "First translation set for call site:\n");
    print_points_to_set("", translation);
  }

  /* Now, we have to call about the same function recursively on the
     list of formal sinks */
  if(!ENDP(fcl)) {
    recursive_filter_formal_context_according_to_actual_context
      (fcl, pt_in_callee, pts_binded, translation, filtered);
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
(set out, set in, list wpl)
{
  set out_filtered = new_simple_pt_map();

  /* First, filter out according to in */
  SET_FOREACH(points_to, pt, out) {
    cell source = points_to_source(pt);
    cell sink = points_to_sink(pt);
    if(nowhere_cell_p(sink)) {
      /* The source of the arc may not have been modified but the sink
	 probably has been freed: the arc must be preserved */
      add_arc_to_simple_pt_map(pt, out_filtered);
    }
    if(points_to_cell_in_list_p(source, wpl)) {
      /* The source of the arc has been modified: the arc must be preserved */
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
  SET_FOREACH(points_to, ipt, in) {
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
	  add_arc_to_simple_pt_map(npt, in);
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

  return out_filtered;
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
    list acl = expression_to_points_to_sources(a, pt_in);
    int nacl = (int) gen_length(acl);
    approximation ap = nacl==1? make_approximation_exact() :
      make_approximation_may();
    FOREACH(CELL, ac, acl) {
      cell source = copy_cell(fc);
      bool to_be_freed;
      type source_t = points_to_cell_to_type(source, &to_be_freed);
      if(pointer_type_p(source_t)) {
	cell n_source = copy_cell(fc);
	cell n_sink = copy_cell(ac);
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
	pips_internal_error("Not implemented yet.\n");
      }
      else {
	; // No need
      }
    }
  }
}

/* add arcs of set "pt_in_s" to set "pts_kill" if their origin cell is
 * not in the list of written pointers "wpl" but is the origin of some
 * arc in "pt_out_callee_filtered".
 */
void add_implicitly_killed_arcs_to_kill_set(set pts_kill, list wpl, set pt_in_s,
					    set pt_out_callee_filtered,
					    set translation)
{
  SET_FOREACH(points_to, out_pt, pt_out_callee_filtered) {
    cell source = points_to_source(out_pt);
    if(!points_to_cell_in_list_p(source, wpl)) {
      /* Arc out_pt has been implicitly obtained */
      approximation a = points_to_approximation(out_pt);
      // FI: let's assume that approximation subsumes atomicity
      if(approximation_exact_p(a)) {
	// source is defined in the formal context
	points_to_graph translation_g =
	  make_points_to_graph(false, translation);
	list tl = points_to_source_to_sinks(source, translation_g, false);
	int  ntl = (int) gen_length(tl);
	cell t_source = cell_undefined;
	if(ntl==1 && (atomic_points_to_cell_p(t_source=CELL(CAR(tl))))) {
	  SET_FOREACH(points_to, pt, pt_in_s) {
	    cell pt_source = points_to_source(pt);
	    if(points_to_cell_equal_p(t_source, pt_source)) {
	      // FI: do not worry about sharing of arcs
	      add_arc_to_simple_pt_map(pt, pts_kill);
	    }
	  }
	}
	gen_free_list(tl);
      }
    }
  }
}


/* Compute the points-to relations in a complete interprocedural way:
 * be as accurately as possible.
 *
 */
pt_map user_call_to_points_to_interprocedural(call c,
					      pt_map pt_in,
					      list el __attribute__ ((unused)))
{
  pt_map pt_out = pt_in;
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

  if(!ENDP(l_pt_to_in)) {
    /* We have to make sure that all points-to stub appearing in the
       callee will be translatable. Since they may be initialized in
       the caller's caller, the points-to information in the caller
       may have to be completed */
    /* Or we may delay this to sets_binded_and_in_compatible_p()? But
       then we have to change the entry test */
    ;
  }

  /* Not much to do if both IN and OUT are empty, except if OUT is
     bottom (see below) */
  if(!(ENDP(l_pt_to_out) && ENDP(l_pt_to_in))) {
    // FI: this function should be moved from semantics into effects-util
    extern list load_summary_effects(entity e);
    list el = load_summary_effects(f);
    list wpl = written_pointers_set(el);
    set pt_in_s = points_to_graph_set(pt_in);

    //points_to_list pts_to_in = (points_to_list)
    //db_get_memory_resource(DBR_POINTS_TO_IN, module_local_name(f), true);
   
    //list l_pt_to_in = gen_full_copy_list(points_to_list_list(pts_to_in));
    set pt_in_callee = new_simple_pt_map();
    pt_in_callee = set_assign_list(pt_in_callee, l_pt_to_in);
    set pt_out_callee = new_simple_pt_map();
    pt_out_callee = set_assign_list(pt_out_callee, l_pt_to_out);

    // FI: function name... set or list?
    set pts_binded = compute_points_to_binded_set(f, al, pt_in_s);
    ifdebug(8) print_points_to_set("pt_binded", pts_binded);

    set translation = new_simple_pt_map();
    /* This is only useful when a free occurs with the callee, since
       information about formal parameters is normally projected
       out. */
    points_to_translation_of_formal_parameters(fpcl, al, pt_in, translation);

    /* Filter pt_in_callee according to pts_binded. For instance, a
v       formal parameter can point to NULL in pt_in_callee only if it
       also points to NULL in pts_binded. In the same way, a formal
       parameter can point to a points-to stub in pt_in_callee only if
       it points to a non-NULL target in pts_binded. Also, a formal
       parameter cannot points exactly to UNDEFINED in pts_binded as
       it would be useless (not clear if we can remove such an arc
       when it is a may arc...). Finally, each formal parameter must
       still point to something. */
    set pt_in_callee_filtered =
      filter_formal_context_according_to_actual_context(fpcl,
							pt_in_callee,
							pts_binded,
							translation);

    /* We have to test if pts_binded is compatible with pt_in_callee */
    /* We have to start by computing all the elements of E (stubs) */
    list stubs = stubs_list(pt_in_callee, pt_out_callee);
    bool compatible_p
      = sets_binded_and_in_compatible_p(stubs, fpcl, pts_binded,
					 pt_in_callee_filtered, pt_out_callee);
    
    if(compatible_p) {

      set pt_out_callee_filtered =
	filter_formal_out_context_according_to_formal_in_context
	(pt_out_callee, pt_in_callee_filtered, wpl);

      /* Explicitly written pointers imply some arc removals; pointer
	 assignments directly or indirectly in the callee. */
      set pts_kill = compute_points_to_kill_set(wpl, pt_in_s, fpcl,
						pt_in_callee_filtered,
						pts_binded,
						translation);
      /* Implicitly written pointers imply some arc removals: free(),
	 tests and exits. */
      add_implicitly_killed_arcs_to_kill_set(pts_kill, wpl, pt_in_s,
					     pt_out_callee_filtered,
					     translation);
      ifdebug(8) print_points_to_set("pt_kill", pts_kill);

      set pt_end = new_simple_pt_map();
      pt_end = set_difference(pt_end, pt_in_s, pts_kill);

      set pts_gen = compute_points_to_gen_set(fpcl,
					      pt_out_callee_filtered,
					      pt_in_callee_filtered,
					      pts_binded,
					      translation);
      pips_assert("pts_gen is consistent", consistent_points_to_set(pts_gen));

      /* Some check */
      list stubs = points_to_set_to_module_stub_cell_list(f, pts_gen, NIL);
      if(!ENDP(stubs)) {
	pips_internal_error("Translation failure.\n");
      }
      // FI: set_union is unsafe; the union of two consistent
      // points-to graph is not a consistent points-to graph
      pt_end = set_union(pt_end, pt_end, pts_gen);
      pips_assert("pt_end is consistent", consistent_points_to_set(pt_end));
      ifdebug(8) print_points_to_set("pt_end =",pt_end);
      points_to_graph_set(pt_out) = pt_end;
    }
    else {
      pips_user_warning("Aliasing between arguments.\n"
			"We have to create a new formal context "
			"and to restart the points-to analysis "
			"and to modify the IN and OUT data structures...\n ");
      pips_internal_error("No handling of aliasing between formal parameters.\n");
      pt_out = user_call_to_points_to_fast_interprocedural(c, pt_in, el);
    }
  }
  else {
    // FI: I do not think we want this warning in general!
    pips_user_warning("Function has no side effect on its formal context "
		      "via pointer variables.\n");
  }

  /* This cannot be performed earlier because the points-to
     precondition of the caller may have to be enriched according to
     the formal context of the callee, or the effect computation is
     going to fail. */
  if(out_bottom_p) {
    clear_pt_map(pt_out);
    points_to_graph_bottom(pt_out) = true;
  }

  return pt_out;
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
