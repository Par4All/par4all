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
  //set_current_module_entity(f);
  //const char* module_name = entity_module_name(e);
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

  // FI: this function should be moved from semantics into effects-util
  extern list load_summary_effects(entity e);
  list el = load_summary_effects(f);
  list wpl = written_pointers_set(el);
  points_to_list pts_to_in = (points_to_list)
    db_get_memory_resource(DBR_POINTS_TO_IN, module_local_name(f), true);
  if(interprocedural_points_to_analysis_p()) {
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
  else if(false) {
    list l_pt_to_in = gen_full_copy_list(points_to_list_list(pts_to_in));
    pt_map pt_in_callee = set_assign_list(pt_in_callee, l_pt_to_in);
    // list l_pt_to_out = gen_full_copy_list(points_to_list_list(pts_to_out));
    // pt_map pt_out_callee = set_assign_list(pt_out_callee, l_pt_to_out);
    pt_map pts_binded = compute_points_to_binded_set(f, al, pt_in);
    ifdebug(8) print_points_to_set("pt_binded", pts_binded);
    pt_map pts_kill = compute_points_to_kill_set(wpl, pt_in, fpcl,
						 pt_in_callee, pts_binded);
    ifdebug(8) print_points_to_set("pt_kill", pts_kill);
    pt_map pt_end = set_difference(pt_end, pt_in, pts_kill);
    ifdebug(8) print_points_to_set("pt_end =",pt_end);
    pt_out = pt_end;
  }
  else {
    pips_user_warning("The function call to \"%s\" is still ignored\n"
		      "On going implementation...\n", entity_user_name(f));
  }

  return pt_out;
}

