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
    new_pt_list = make_points_to_list(true, tmp_l);
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
    new_pt_list = make_points_to_list(true, tmp_l);
    store_or_update_pt_to_list(s, new_pt_list);
  }
  gen_free_list(pt_list);
}

void fi_points_to_storage(pt_map ptm, statement s, bool store) {
  list pt_list = NIL, tmp_l;
  points_to_list new_pt_list = points_to_list_undefined;
  set pts_to_set = points_to_graph_set(ptm);
  bool bottom_p = points_to_graph_bottom(ptm);
  
  if ( !set_empty_p(pts_to_set) && store == true ) {
    
    pt_list = set_to_sorted_list(pts_to_set,
                                 (int(*)(const void*, const void*))
                                 points_to_compare_cells);
    tmp_l = gen_full_copy_list(pt_list);
    new_pt_list = make_points_to_list(bottom_p, tmp_l);
    points_to_list_consistent_p(new_pt_list);
    store_or_update_pt_to_list(s, new_pt_list);
  }
  else if(set_empty_p(pts_to_set)){
    tmp_l = gen_full_copy_list(pt_list);
    new_pt_list = make_points_to_list(bottom_p, tmp_l);
    store_or_update_pt_to_list(s, new_pt_list);
  }
  gen_free_list(pt_list);
}

/* Return the subset of "in" that is related to formal parameters and stubs
 * 
 * More care should be taken about formal parameter
 * modifications. Dummy initial variables should be allocated to
 * preserve the values of formal parameters on entry.
 */
pt_map points_to_to_context_points_to(pt_map in)
{
  pt_map out = new_pt_map();
  set in_s = points_to_graph_set(in);
  set out_s = points_to_graph_set(out);

  SET_FOREACH(points_to, pt, in_s) {
    cell source = points_to_source(pt);
    if(formal_parameter_points_to_cell_p(source)
       || stub_points_to_cell_p(source)) {
      cell sink = points_to_sink(pt);
      if(stub_points_to_cell_p(sink)) {
	points_to npt = copy_points_to(pt);
	add_arc_to_simple_pt_map(npt, out_s);
      }
    }
  }

  return out;
}


static pt_map points_to_context = pt_map_undefined;

void init_points_to_context(pt_map init)
{
  pips_assert("points_to_context is undefined",
	      pt_map_undefined_p(points_to_context));
  points_to_context = full_copy_pt_map(init);
}

void reset_points_to_context()
{
  pips_assert("points_to_context is defined",
	      !pt_map_undefined_p(points_to_context));
  // Deep when pt_map is a Newgen object
  // free_pt_map(points_to_context); // Shallow if pt_map==set
  points_to_context = pt_map_undefined;
}

/* FI: it should rather work the other way round, with
 * add_arc_to_statement_points_to_context() calling
 * add_arc_to_points_to_context().
 */
void add_arc_to_points_to_context(points_to pt)
{
  pips_assert("points_to_context is defined",
	      !pt_map_undefined_p(points_to_context));
  (void) add_arc_to_pt_map(pt, points_to_context);
  points_to npt = copy_points_to(pt);
  add_arc_to_statement_points_to_context(npt);
}

pt_map get_points_to_context()
{
  return points_to_context;
}

void clean_up_points_to_stubs(entity module)
{
  code c = value_code(entity_initial(module));
  list dl = code_declarations(c);
  list sl = NIL;

  FOREACH(ENTITY, v, dl) {
    if(entity_stub_sink_p(v) || entity_heap_location_p(v)) {
      sl = CONS(ENTITY, v, sl);
      fprintf(stderr, "Removed stub: %s\n", entity_name(v));
    }
  }

  gen_list_and_not(&code_declarations(c), sl);

  GenericCleanEntities(sl, module, false);

  gen_free_list(sl);
}

#define FRANCOIS 1

/* Pass INTRAPROCEDURAL_POINTS_TO_ANALYSIS
 *
 */
static bool generic_points_to_analysis(char * module_name) {
  entity module;
  statement module_stat;
  pt_map pt_in = new_pt_map();
  pt_map pts_to_out = new_pt_map();
  
  init_pt_to_list();
  module = module_name_to_entity(module_name);
  set_current_module_entity(module);
  make_effects_private_current_context_stack();

  debug_on("POINTS_TO_DEBUG_LEVEL");

  pips_debug(1, "considering module %s\n", module_name);
  set_current_module_statement((statement) db_get_memory_resource(DBR_CODE,
                                                                  module_name, true));
  module_stat = get_current_module_statement();

  /* In case we need effects to generate all necessary points-to information */
  // These initializations are not sufficient. We also need at least a
  // stack of statements
  //set_constant_paths_p(true);
  //set_pointer_info_kind(with_points_to);
  //set_methods_for_proper_simple_effects();

  /* Clean-up formal context stubs and heap model variables */
  clean_up_points_to_stubs(module);

  /* Stack for on-demand update of in points-to information */
  init_statement_points_to_context();

  /*
    Get the init_points_to_list resource.
    This list contains formal paramters and their stub sinks
  */
  // #if !FRANCOIS: to simplify interface with effects_with_points_to
  // FI: I would like not to use DBR_INIT_POINTS_TO_LIST because it
  // generates useless information. However I need something as long
  // as the empty points-to set is the attribute of dead code...
  // Also, the init analysis uses the same modules but with another
  // interface, which muddles the generation of stub cells with stars
  // or zeros
#if 0
  list pts_to_list = NIL;
  points_to_list init_pts_to_list = 
    (points_to_list) db_get_memory_resource(DBR_INIT_POINTS_TO_LIST,
                                            module_name, true);
  /* Transform the list of init_pts_to_list in set of points-to.*/
  pts_to_list = gen_full_copy_list(points_to_list_list(init_pts_to_list));
  pt_in = graph_assign_list(pt_in, pts_to_list);
  init_points_to_context(pt_in);
  // FI: this should be useless as stubs are built on demand
  // pt_in = set_assign_list(pt_in, NIL);
  gen_free_list(pts_to_list);
#else
  points_to_graph_set(pt_in) =
    set_assign_list(points_to_graph_set(pt_in), NIL);
  init_points_to_context(pt_in);
#endif

  /* Compute the points-to relations using the pt_in as input.*/
#if !FRANCOIS
  // FI: old version
  pts_to_out = points_to_statement(module_stat, pt_in);

#else  // FI: new version
  pts_to_out = statement_to_points_to(module_stat, pt_in);
#endif
  /* Store the points-to relations */
  DB_PUT_MEMORY_RESOURCE(DBR_POINTS_TO, module_name, get_pt_to_list());

  /* Remove dangling stubs... before the formal parameters are
     projected as the projection will create lots of dangling
     stubs. In fact, they are not dangling from a formal context view
     point. */
  // pts_to_out = remove_unreachable_vertices_in_points_to_graph(pts_to_out);
  /* Filter OUT points-to by deleting local variables, including the
     formal paprameters */
  if(entity_main_module_p(module))
    clear_pt_map(pts_to_out);
  else
    points_to_graph_set(pts_to_out) =
      points_to_function_projection(points_to_graph_set(pts_to_out));

  /* Save IN points-to relations */
#if !FRANCOIS
  list  l_in = set_to_list(pt_in);
  points_to_list in_list = make_points_to_list(true, l_in); // SG: baaaaaaad copy, let us hope AM will fix her code :p
#else
  // pt_map context = points_to_to_context_points_to(pts_to_out);
  pt_map context = get_points_to_context();
  list  l_in = set_to_list(points_to_graph_set(context));
  points_to_list in_list = make_points_to_list(false, l_in); // SG: baaaaaaad copy, let us hope AM will fix her code :p
  // FI: I suppose context should be emptied because of the sharing
  // with l_in and then freed
#endif
  DB_PUT_MEMORY_RESOURCE(DBR_POINTS_TO_IN, module_name, in_list);

    /* Save OUT points-to relations */
  list  l_out =
    gen_full_copy_list(set_to_list(points_to_graph_set(pts_to_out)));
  bool out_bottom_p = points_to_graph_bottom(pts_to_out);
  points_to_list out_list = make_points_to_list(out_bottom_p, l_out); // SG: baaaaaaad copy, let us hope AM will fix her code :p

  DB_PUT_MEMORY_RESOURCE(DBR_POINTS_TO_OUT, module_name, out_list);

  reset_pt_to_list();
  reset_points_to_context();
  reset_statement_points_to_context();
  // FI: the depth of the free depends on pt_map
  // free_pt_map(pts_to_out);
  // free_pt_map(pt_in);
  reset_current_module_entity();
  reset_current_module_statement();
  reset_effects_private_current_context_stack();
  //generic_effects_reset_all_methods();
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

  pips_debug(1, "considering module \"%s\"\n", module_name);

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
  points_to_list init_pts_to_list = make_points_to_list(false, pt_list);
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

static bool interprocedural_points_to_p = true;
static bool fast_interprocedural_points_to_p = true;
bool interprocedural_points_to_analysis_p()
{
  return interprocedural_points_to_p;
}

bool fast_interprocedural_points_to_analysis_p()
{
  return fast_interprocedural_points_to_p;
}

bool intraprocedural_points_to_analysis(char * module_name)
{
  interprocedural_points_to_p = false;
  fast_interprocedural_points_to_p = false;
  return generic_points_to_analysis(module_name);
}

bool interprocedural_points_to_analysis(char * module_name)
{
  interprocedural_points_to_p = true;
  fast_interprocedural_points_to_p = false;
  return generic_points_to_analysis(module_name);
}

bool fast_interprocedural_points_to_analysis(char * module_name)
{
  fast_interprocedural_points_to_p = true;
  interprocedural_points_to_p = false;
  return generic_points_to_analysis(module_name);
}
