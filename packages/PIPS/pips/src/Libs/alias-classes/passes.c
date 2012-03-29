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
 * This file contains the passes computing points-to information:
 *
 * intraprocedural_points_to_analysis
 * init_points_to_analysis
 * interprocedural_points_to_analysis
 */

#include <stdlib.h>
#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
//#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
//#include "control.h"
//#include "constants.h"
#include "misc.h"
//#include "parser_private.h"
//#include "syntax.h"
//#include "top-level.h"
//#include "text-util.h"
//#include "text.h"
#include "properties.h"
//#include "pipsmake.h"
//#include "semantics.h"
// For effects_private_current_context_stack()
#include "effects-generic.h"
//#include "effects-simple.h"
//#include "effects-convex.h"
//#include "transformations.h"
//#include "preprocessor.h"
#include "pipsdbm.h"
#include "resources.h"
//#include "prettyprint.h"
//#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"




/* Store a sorted copy of the points-to pts_to_set associated to a
   statement s in the points-to hash-table.

   In case s is a loop, do, while or for, the parameter "store" is set
   to false to prevent key redefinitions in the underlying points-to
   hash-table. This entry condition is not checked.

   In case s is a sequence, the sorted copy pts_to_set is associated
   to each substatement and shared by s and all its substatement.

   Note: the function is called with store==true from
   points_to_whileloop(). And the hash-table can be updated
   (hash_update()). 

 */
void points_to_storage(set pts_to_set, statement s, bool store) {
  list pt_list = NIL, tmp_l;
  points_to_list new_pt_list = points_to_list_undefined;
  
  if ( !set_empty_p(pts_to_set) && store == true ) {
    
    pt_list = set_to_sorted_list(pts_to_set,
                                 (int(*)(const void*, const void*))
                                 points_to_compare_cells);
    tmp_l = gen_full_copy_list(pt_list);
    new_pt_list = make_points_to_list(tmp_l);
    points_to_list_consistent_p(new_pt_list);
    store_or_update_pt_to_list(s, new_pt_list);

    instruction i = statement_instruction(s);
    if(instruction_sequence_p(i)) {
      sequence seq = instruction_sequence(i);
      FOREACH(statement, stm, sequence_statements(seq)){
        store_or_update_pt_to_list(stm, new_pt_list);
      }
    }
  }
  else if(set_empty_p(pts_to_set)){
    tmp_l = gen_full_copy_list(pt_list);
    new_pt_list = make_points_to_list(tmp_l);
    store_or_update_pt_to_list(s, new_pt_list);
  }
  gen_free_list(pt_list);
}

void fi_points_to_storage(set pts_to_set, statement s, bool store) {
  list pt_list = NIL, tmp_l;
  points_to_list new_pt_list = points_to_list_undefined;
  
  if ( !set_empty_p(pts_to_set) && store == true ) {
    
    pt_list = set_to_sorted_list(pts_to_set,
                                 (int(*)(const void*, const void*))
                                 points_to_compare_cells);
    tmp_l = gen_full_copy_list(pt_list);
    new_pt_list = make_points_to_list(tmp_l);
    points_to_list_consistent_p(new_pt_list);
    store_or_update_pt_to_list(s, new_pt_list);
  }
  else if(set_empty_p(pts_to_set)){
    tmp_l = gen_full_copy_list(pt_list);
    new_pt_list = make_points_to_list(tmp_l);
    store_or_update_pt_to_list(s, new_pt_list);
  }
  gen_free_list(pt_list);
}


/* Pass INTRAPROCEDURAL_POINTS_TO_ANALYSIS
 *
 */
bool intraprocedural_points_to_analysis(char * module_name) {
  entity module;
  statement module_stat;
  set pt_in = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  list pts_to_list = NIL;
  set pts_to_out = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  
  init_pt_to_list();
  module = module_name_to_entity(module_name);
  set_current_module_entity(module);
  make_effects_private_current_context_stack();

  debug_on("POINTS_TO_DEBUG_LEVEL");

  pips_debug(1, "considering module %s\n", module_name);
  set_current_module_statement((statement) db_get_memory_resource(DBR_CODE,
                                                                  module_name, true));
  module_stat = get_current_module_statement();
  
  /*
    Get the init_points_to_list resource.
    This list contains formal paramters and their stub sinks
  */
  points_to_list init_pts_to_list = 
    (points_to_list) db_get_memory_resource(DBR_INIT_POINTS_TO_LIST,
                                            module_name, true);
  /* Transform the list of init_pts_to_list in set of points-to.*/
  pts_to_list = gen_full_copy_list(points_to_list_list(init_pts_to_list));
  pt_in = set_assign_list(pt_in, pts_to_list);
  gen_free_list(pts_to_list);

  /* Compute the points-to relations using the pt_in as input.*/
  // FI: old version
  pts_to_out = points_to_statement(module_stat, pt_in);
  // FI: new version
  // pts_to_out = statement_to_points_to(module_stat, pt_in);
  /* Store the points-to relations */
  DB_PUT_MEMORY_RESOURCE(DBR_POINTS_TO, module_name, get_pt_to_list());

  /* Filter OUT points-to by deleting local variables */
  pts_to_out = points_to_function_projection(pts_to_out);
    
  /* Save IN points-to relations */
  list  l_in = set_to_list(pt_in);
  points_to_list in_list = make_points_to_list(l_in); // SG: baaaaaaad copy, let us hope AM will fix her code :p
  DB_PUT_MEMORY_RESOURCE(DBR_POINTS_TO_IN, module_name, in_list);

    /* Save OUT points-to relations */
  list  l_out = gen_full_copy_list(set_to_list(pts_to_out));
  points_to_list out_list = make_points_to_list(l_out); // SG: baaaaaaad copy, let us hope AM will fix her code :p

  DB_PUT_MEMORY_RESOURCE(DBR_POINTS_TO_OUT, module_name, out_list);

  reset_pt_to_list();
  set_free(pts_to_out);
  set_free(pt_in);
  reset_current_module_entity();
  reset_current_module_statement();
  reset_effects_private_current_context_stack();
  debug_off();
  bool good_result_p = true;

  return (good_result_p);
}


bool init_points_to_analysis(char * module_name)
{
  entity module;
  type t;
  list pt_list = NIL, dl = NIL;
  set pts_to_set = set_generic_make(set_private,
				    points_to_equal_p,points_to_rank);
  set formal_set = set_generic_make(set_private,
				    points_to_equal_p,points_to_rank);
  set_current_module_entity(module_name_to_entity(module_name));
  module = get_current_module_entity();

  t = entity_type(module);

  debug_on("POINTS_TO_DEBUG_LEVEL");

  pips_debug(1, "considering module %s\n", module_name);

  /* Properties */
  if(get_bool_property("ALIASING_ACROSS_FORMAL_PARAMETERS"))
    pips_user_warning("Property ALIASING_ACROSS_FORMAL_PARAMETERS"
		      " is ignored\n");
  if(get_bool_property("ALIASING_ACROSS_TYPES"))
    pips_user_warning("Property ALIASING_ACROSS_TYPES"
		      " is ignored\n");
  if(get_bool_property("ALIASING_INSIDE_DATA_STRUCTURE"))
    pips_user_warning("Property ALIASING_INSIDE_DATA_STRUCTURE"
		      " is ignored\n");

  if(type_functional_p(t)){
    dl = code_declarations(value_code(entity_initial(module)));

    FOREACH(ENTITY, fp, dl) {
      if(formal_parameter_p(fp)) {
	reference r = make_reference(fp, NIL);
	cell c = make_cell_reference(r);
	formal_set = formal_points_to_parameter(c);
	pts_to_set = set_union(pts_to_set, pts_to_set,
			       formal_set);
      }
    }
    
  }
  else
    pips_user_error("The module %s is not a function.\n", module_name);

  pt_list = set_to_sorted_list(pts_to_set,
			       (int(*)
				(const void*,const void*))
			       points_to_compare_cells);
  points_to_list init_pts_to_list = make_points_to_list(pt_list);
  points_to_list_consistent_p(init_pts_to_list);
  DB_PUT_MEMORY_RESOURCE
    (DBR_INIT_POINTS_TO_LIST, module_name, init_pts_to_list);
  reset_current_module_entity();
  set_clear(pts_to_set);
  set_clear(pts_to_set);
  set_free(pts_to_set);
  set_free(formal_set);
  debug_off();

  bool good_result_p = true;
  return (good_result_p);
}


bool interprocedural_points_to_analysis(char * module_name)
{
  /* list pt_list = NIL; */
  /* set pts_to_set = set_generic_make(set_private, */
  /* 				    points_to_equal_p,points_to_rank); */

  set_current_module_entity(module_name_to_entity(module_name));
  //entity module = get_current_module_entity();

  //type t = entity_type(module);

  debug_on("POINTS_TO_DEBUG_LEVEL");

  pips_debug(1, "considering module %s\n", module_name);

  

  /* DB_PUT_MEMORY_RESOURCE */
  /*   (DBR_INIT_POINTS_TO_LIST, module_name, init_pts_to_list); */

  reset_current_module_entity();
  debug_off();

  bool good_result_p = true;
  return (good_result_p);
}
