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

pt_map user_call_to_points_to(call c, pt_map pt_in)
{
  pt_map pt_out = pt_in;
  entity f = call_function(c);
  list al = call_arguments(c);

  // FI: intraprocedural, use effects
  // FI: interprocedural, check alias compatibility, generate gen and kill sets,...
  pt_out = pt_in;

  // Code by Amira
  list fpcl = NIL; // Formal parameter cell list
  type t = entity_type(f);
  if(type_functional_p(t)){
    list dl = code_declarations(value_code(entity_initial(f)));
    FOREACH(ENTITY, fp, dl) {
      if(formal_parameter_p(fp)) {
	reference r = make_reference(fp, NIL);
	cell c = make_cell_reference(r);
	fpcl = gen_nconc(CONS(CELL, c, NULL), fpcl);
      }
    }
  }
  else
    pips_internal_error("Function has not a functional type.\n");

  if(interprocedural_points_to_analysis_p())
    {
      // FI: this function should be moved from semantics into effects-util
    extern list load_summary_effects(entity e);
    list el = load_summary_effects(f);
    list wpl = written_pointers_set(el);
    points_to_list pts_to_in = (points_to_list)
      db_get_memory_resource(DBR_POINTS_TO_IN, module_local_name(f), true);
    points_to_list pts_to_out = (points_to_list)
      db_get_memory_resource(DBR_POINTS_TO_OUT, module_local_name(f), true);
    list l_pt_to_in = gen_full_copy_list(points_to_list_list(pts_to_in));
    pt_map pt_in_callee = new_pt_map();
    pt_in_callee = set_assign_list(pt_in_callee, l_pt_to_in);
    list l_pt_to_out = gen_full_copy_list(points_to_list_list(pts_to_out));
    pt_map pt_out_callee = new_pt_map();
    pt_out_callee = set_assign_list(pt_out_callee, l_pt_to_out);
    // FI: function name... set or list?
    pt_map pts_binded = compute_points_to_binded_set(f, al, pt_in);
    ifdebug(8) print_points_to_set("pt_binded", pts_binded);
    pt_map pts_kill = compute_points_to_kill_set(wpl, pt_in, fpcl,
						 pt_in_callee, pts_binded);
    ifdebug(8) print_points_to_set("pt_kill", pts_kill);
    pt_map pt_end = new_pt_map();
    pt_end = set_difference(pt_end, pt_in, pts_kill);
    pt_map pts_gen = compute_points_to_gen_set(fpcl, pt_out_callee,
					       pt_in_callee, pts_binded);
    pt_end = set_union(pt_end, pt_end, pts_gen);
    ifdebug(8) print_points_to_set("pt_end =",pt_end);
    pt_out = pt_end;
  }
  else if(fast_interprocedural_points_to_analysis_p()) 
    {
    extern list load_summary_effects(entity e);
    list el = load_summary_effects(f);
    list wpl = written_pointers_set(el);
    points_to_list pts_to_in = (points_to_list)
      db_get_memory_resource(DBR_POINTS_TO_IN, module_local_name(f), true);
    list l_pt_to_in = gen_full_copy_list(points_to_list_list(pts_to_in));
    pt_map pt_in_callee = new_pt_map();
    pt_in_callee = set_assign_list(pt_in_callee, l_pt_to_in);
    // list l_pt_to_out = gen_full_copy_list(points_to_list_list(pts_to_out));
    // pt_map pt_out_callee = set_assign_list(pt_out_callee, l_pt_to_out);
    pt_map pts_binded = compute_points_to_binded_set(f, al, pt_in);
    ifdebug(8) print_points_to_set("pt_binded", pts_binded);
    pt_map pts_kill = compute_points_to_kill_set(wpl, pt_in, fpcl,
						 pt_in_callee, pts_binded);
    ifdebug(8) print_points_to_set("pt_kill", pts_kill);
    pt_map pt_end = new_pt_map();
    pt_end = set_difference(pt_end, pt_in, pts_kill);
    ifdebug(8) print_points_to_set("pt_end =",pt_end);
    pt_out = pt_end;
    }
  /* else { */
  /*   pips_user_warning("The function call to \"%s\" is still ignored\n" */
  /* 		      "On going implementation...\n", entity_user_name(f)); */
  /* } */
  else {
    /* intraprocedural phase */
    /* FOREACH(expression, arg, al) { */
    /*   pt_out = expression_to_points_to(arg, pt_out); */
    /* } */
    FOREACH(expression, arg, al) {
      list l_sink = expression_to_points_to_sources(arg, pt_out);
      SET_FOREACH(points_to, pts, pt_out) {
	FOREACH(cell, cel, l_sink) {
	  cell source = points_to_source(pts);
	  if(cell_equal_p(source, cel)) {
	    cell sink = points_to_sink(pts);
	    if(source_in_set_p(sink, pt_out))
	      remove_arcs_from_pt_map(pts, pt_out);
	  }
	}
      }
    }
  }


  return pt_out;
}

// FI: I assume we do not need the eval_p parameter here
list user_call_to_points_to_sinks(call c, pt_map in __attribute__ ((unused)), bool eval_p)
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
    pt_map pt_out_callee = new_pt_map();
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

void remove_arcs_from_pt_map(points_to pts, pt_map pt_out)
{
  cell sink = points_to_sink(pts);
  cell source = points_to_source(pts);
  

  SET_FOREACH(points_to, pt, pt_out) {
    if(cell_equal_p(points_to_source(pt), sink) ||cell_equal_p(points_to_source(pt), source) ) {
      remove_arc_from_pt_map(pts, pt_out);
      entity e = entity_anywhere_locations();
      reference r = make_reference(e, NIL);
      cell source = make_cell_reference(r);
      cell sink = copy_cell(source);
      approximation a = make_approximation_exact();
      points_to npt = make_points_to(source, sink, a, make_descriptor_none());
      add_arc_to_pt_map(npt, pt_out);
      remove_arcs_from_pt_map(pt, pt_out);

    }

  }


}
