/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*
 *  package convex effects :  Be'atrice Creusillet 5/97
 *
 * File: methods.c
 * ~~~~~~~~~~~~~~~
 *
 * This File contains the interfaces with pipsmake which compute the various
 * types of convex regions by using the generic functions.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "transformer.h"
#include "semantics.h"
#include "properties.h"
#include "resources.h"
#include "pipsmake.h"
#include "pipsdbm.h"
#include "effects-generic.h"
#include "effects-convex.h"


/******************************************************* PIPSDBM INTERFACES */

DB_GETPUT_SE(convex_proper_rw_pointer_regions, PROPER_POINTER_REGIONS)
DB_GETPUT_SE(convex_rw_pointer_regions, POINTER_REGIONS)
DB_GETPUT_SE(convex_invariant_rw_pointer_regions, INV_POINTER_REGIONS)
DB_GETPUT_LS(convex_summary_rw_pointer_regions, SUMMARY_POINTER_REGIONS)

DB_GETPUT_SE(convex_proper_rw_regions, PROPER_REGIONS)
DB_GETPUT_SE(convex_rw_regions, REGIONS)
DB_GETPUT_SE(convex_invariant_rw_regions, INV_REGIONS)
DB_GETPUT_LS(convex_summary_rw_regions, SUMMARY_REGIONS)

DB_GETPUT_SE(convex_in_regions, IN_REGIONS)
DB_GETPUT_SE(convex_invariant_in_regions, INV_IN_REGIONS)
DB_GETPUT_SE(convex_cumulated_in_regions, CUMULATED_IN_REGIONS)
DB_GETPUT_LS(convex_summary_in_regions, IN_SUMMARY_REGIONS)
DB_GETPUT_SE(convex_out_regions, OUT_REGIONS)
DB_GETPUT_LS(convex_summary_out_regions, OUT_SUMMARY_REGIONS)


/******************************************************************* SET... */

void
set_methods_for_convex_effects()
{
  effect_consistent_p_func = region_consistent_p;
    effect_dup_func = region_dup;
    effect_free_func = region_free;

    reference_to_effect_func = reference_to_convex_region;
    effect_to_store_independent_effect_list_func =
      region_to_store_independent_region_list;
    effect_add_expression_dimension_func =
      convex_region_add_expression_dimension;
    effect_change_ith_dimension_expression_func =
      convex_region_change_ith_dimension_expression;

    effect_union_op = regions_must_convex_hull;
    effects_union_op = RegionsMustUnion;
    effects_test_union_op = RegionsMayUnion;
    effects_intersection_op = RegionsIntersection;
    effects_sup_difference_op = RegionsSupDifference;
    effects_inf_difference_op = RegionsInfDifference;

    effects_transformer_composition_op =
	convex_regions_transformer_compose;
    effects_transformer_inverse_composition_op =
	convex_regions_inverse_transformer_compose;
    effects_precondition_composition_op =
	convex_regions_precondition_compose;


    eval_cell_with_points_to_func = eval_convex_cell_with_points_to;
    if (get_constant_paths_p())
      {
	switch (get_pointer_info_kind())
	{
	case with_no_pointer_info:
	  effect_to_constant_path_effects_func = effect_to_constant_path_effects_with_no_pointer_information;
	  break;
	case with_points_to:
	  effect_to_constant_path_effects_func = convex_effect_to_constant_path_effects_with_points_to;
	  break;
	case with_pointer_values:
	  effect_to_constant_path_effects_func = convex_effect_to_constant_path_effects_with_pointer_values;
	  break;
	default:
	  pips_internal_error("unexpected value\n");
	}
      }


    effects_descriptors_variable_change_func =
	convex_regions_descriptor_variable_rename;
    loop_descriptor_make_func = loop_convex_descriptor_make;
    effects_loop_normalize_func = convex_regions_loop_normalize;
    effects_union_over_range_op = convex_regions_union_over_range;
    vector_to_descriptor_func = vector_to_convex_descriptor;

    effects_translation_init_func = convex_regions_translation_init;
    effects_translation_end_func = convex_regions_translation_end;
    effect_descriptor_interprocedural_translation_op = convex_region_descriptor_translation;

    fortran_effects_backward_translation_op = convex_regions_backward_translation;
    fortran_effects_forward_translation_op = convex_regions_forward_translation;
    c_effects_on_formal_parameter_backward_translation_func =
      c_convex_effects_on_formal_parameter_backward_translation;
    c_effects_on_actual_parameter_forward_translation_func =
      c_convex_effects_on_actual_parameter_forward_translation;
  effects_local_to_global_translation_op = regions_dynamic_elim;

    load_context_func = load_statement_precondition;
    // FI: sometimes in sequences, you need the completed transformer,
    // some other times, for instance when moving from the loop body
    // to the loop, you need the non-completed statement transformer
    load_transformer_func = load_statement_transformer;
    load_completed_transformer_func = load_completed_statement_transformer;
    empty_context_test = empty_convex_context_p;

    proper_to_summary_effect_func = effect_nop;

    effects_descriptor_normalize_func = convex_effects_descriptor_normalize;


    db_get_proper_rw_effects_func = db_get_convex_proper_rw_regions;
    db_put_proper_rw_effects_func = db_put_convex_proper_rw_regions;

    db_get_invariant_rw_effects_func = db_get_convex_invariant_rw_regions;
    db_put_invariant_rw_effects_func = db_put_convex_invariant_rw_regions;

    db_get_rw_effects_func = db_get_convex_rw_regions;
    db_put_rw_effects_func = db_put_convex_rw_regions;

    db_get_summary_rw_effects_func = db_get_convex_summary_rw_regions;
    db_put_summary_rw_effects_func = db_put_convex_summary_rw_regions;

    db_get_in_effects_func = db_get_convex_in_regions;
    db_put_in_effects_func = db_put_convex_in_regions;

    db_get_cumulated_in_effects_func = db_get_convex_cumulated_in_regions;
    db_put_cumulated_in_effects_func = db_put_convex_cumulated_in_regions;

    db_get_invariant_in_effects_func = db_get_convex_invariant_in_regions;
    db_put_invariant_in_effects_func = db_put_convex_invariant_in_regions;

    db_get_summary_in_effects_func = db_get_convex_summary_in_regions;
    db_put_summary_in_effects_func = db_put_convex_summary_in_regions;

    db_get_out_effects_func = db_get_convex_out_regions;
    db_put_out_effects_func = db_put_convex_out_regions;

    db_get_summary_out_effects_func = db_get_convex_summary_out_regions;
    db_put_summary_out_effects_func = db_put_convex_summary_out_regions;

    set_contracted_proper_effects(true);
    set_descriptor_range_p(true);
}


void set_methods_for_convex_rw_effects()
{
  set_methods_for_convex_effects();
  effects_computation_init_func = init_convex_rw_regions;
  effects_computation_reset_func = reset_convex_rw_regions;
}

void set_methods_for_convex_rw_pointer_effects()
{
  effect_consistent_p_func = region_consistent_p;
    effect_dup_func = region_dup;
    effect_free_func = region_free;

    reference_to_effect_func = reference_to_convex_region;
    effect_to_store_independent_effect_list_func =
      region_to_store_independent_region_list;
    effect_add_expression_dimension_func =
      convex_region_add_expression_dimension;
    effect_change_ith_dimension_expression_func =
      convex_region_change_ith_dimension_expression;

    effect_union_op = regions_must_convex_hull;
    effects_union_op = RegionsMustUnion;
    effects_test_union_op = RegionsMayUnion;
    effects_intersection_op = RegionsIntersection;
    effects_sup_difference_op = RegionsSupDifference;
    effects_inf_difference_op = RegionsInfDifference;

    effects_transformer_composition_op =
	convex_regions_transformer_compose;
    effects_transformer_inverse_composition_op =
	convex_regions_inverse_transformer_compose;
    effects_precondition_composition_op =
	convex_regions_precondition_compose;

    effects_descriptors_variable_change_func =
	convex_regions_descriptor_variable_rename;
    loop_descriptor_make_func = loop_convex_descriptor_make;
    effects_loop_normalize_func = convex_regions_loop_normalize;
    effects_union_over_range_op = convex_regions_union_over_range;
    vector_to_descriptor_func = vector_to_convex_descriptor;

    effects_translation_init_func = convex_regions_translation_init;
    effects_translation_end_func = convex_regions_translation_end;
    effect_descriptor_interprocedural_translation_op = convex_region_descriptor_translation;

    fortran_effects_backward_translation_op = convex_regions_backward_translation;
    fortran_effects_forward_translation_op = convex_regions_forward_translation;
    c_effects_on_formal_parameter_backward_translation_func =
      c_convex_effects_on_formal_parameter_backward_translation;
    c_effects_on_actual_parameter_forward_translation_func =
      c_convex_effects_on_actual_parameter_forward_translation;
  effects_local_to_global_translation_op = regions_dynamic_elim;

    load_context_func = load_statement_precondition;
    load_transformer_func = load_statement_transformer;
    load_completed_transformer_func = load_completed_statement_transformer;
    empty_context_test = empty_convex_context_p;

    proper_to_summary_effect_func = effect_nop;

    effects_descriptor_normalize_func = convex_effects_descriptor_normalize;


    db_get_proper_rw_effects_func = db_get_convex_proper_rw_pointer_regions;
    db_put_proper_rw_effects_func = db_put_convex_proper_rw_pointer_regions;

    db_get_invariant_rw_effects_func = db_get_convex_invariant_rw_pointer_regions;
    db_put_invariant_rw_effects_func = db_put_convex_invariant_rw_pointer_regions;

    db_get_rw_effects_func = db_get_convex_rw_pointer_regions;
    db_put_rw_effects_func = db_put_convex_rw_pointer_regions;

    db_get_summary_rw_effects_func = db_get_convex_summary_rw_pointer_regions;
    db_put_summary_rw_effects_func = db_put_convex_summary_rw_pointer_regions;

    db_get_in_effects_func = db_get_convex_in_regions;
    db_put_in_effects_func = db_put_convex_in_regions;

    db_get_cumulated_in_effects_func = db_get_convex_cumulated_in_regions;
    db_put_cumulated_in_effects_func = db_put_convex_cumulated_in_regions;

    db_get_invariant_in_effects_func = db_get_convex_invariant_in_regions;
    db_put_invariant_in_effects_func = db_put_convex_invariant_in_regions;

    db_get_summary_in_effects_func = db_get_convex_summary_in_regions;
    db_put_summary_in_effects_func = db_put_convex_summary_in_regions;

    db_get_out_effects_func = db_get_convex_out_regions;
    db_put_out_effects_func = db_put_convex_out_regions;

    db_get_summary_out_effects_func = db_get_convex_summary_out_regions;
    db_put_summary_out_effects_func = db_put_convex_summary_out_regions;

    set_contracted_proper_effects(true);
    set_descriptor_range_p(true);

    effects_computation_init_func = init_convex_rw_regions;
    effects_computation_reset_func = reset_convex_rw_regions;
}




void set_methods_for_convex_in_out_effects()
{
  set_methods_for_convex_effects();
  effects_computation_init_func = init_convex_in_out_regions;
  effects_computation_reset_func = reset_convex_in_out_regions;
}

bool in_out_methods_p()
{
  // functions that can be pointed by effects_computation_init_func:
  // effects_computation_no_init
  // init_convex_in_out_regions
  // init_convex_rw_regions
  return (effects_computation_init_func == init_convex_in_out_regions);
}

void init_convex_rw_prettyprint(const char* __attribute__ ((unused)) module_name)
{
  set_action_interpretation(ACTION_READ, ACTION_WRITE);
  effects_prettyprint_func = print_rw_regions;
  effect_prettyprint_func = print_region;
  effects_to_text_func = text_rw_array_regions;
}

void
init_convex_rw_regions(const char* module_name)
{
    regions_init();
    get_regions_properties();
    region_translation_statistics_init
	(get_bool_property("REGIONS_TRANSLATION_STATISTICS"));

    /* Get the transformers and preconditions of the module. */
    set_transformer_map( (statement_mapping)
	db_get_memory_resource(DBR_TRANSFORMERS, module_name, true) );
    set_precondition_map( (statement_mapping)
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );

    /* for intermediate values */
    set_cumulated_rw_effects((statement_effects)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
    module_to_value_mappings(module_name_to_entity(module_name));

    init_convex_rw_prettyprint(module_name);
}

void init_convex_inout_prettyprint(const char* __attribute__ ((unused)) module_name)
{
  set_action_interpretation(ACTION_IN, ACTION_OUT);
  effects_prettyprint_func = print_inout_regions;
  effect_prettyprint_func = print_region;
  effects_to_text_func = text_inout_array_regions;
}

void init_convex_in_out_regions(const char* module_name)
{
    regions_init();
    if (!same_string_p(rule_phase(find_rule_by_resource("REGIONS")),
		       "MUST_REGIONS"))
	pips_user_warning("\nMUST REGIONS not selected - "
			  "Do not expect wonderful results\n");
    set_bool_property("MUST_REGIONS", true);
    set_bool_property("EXACT_REGIONS", true);
    get_in_out_regions_properties();

    /* Get the transformers and preconditions of the module. */
    set_transformer_map( (statement_mapping)
	db_get_memory_resource(DBR_TRANSFORMERS, module_name, true) );
    set_precondition_map( (statement_mapping)
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );

    /* for intermediate values */
    set_cumulated_rw_effects((statement_effects)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
    module_to_value_mappings(module_name_to_entity(module_name));

    init_convex_inout_prettyprint(module_name);
}

void reset_convex_rw_regions(const char* module_name)
{
    regions_end();
    region_translation_statistics_close(module_name, "rw");
    reset_transformer_map();
    reset_precondition_map();
    reset_cumulated_rw_effects();
    free_value_mappings();
}

void reset_convex_in_out_regions(const char* __attribute__ ((unused)) module_name)
{
    regions_end();
    reset_transformer_map();
    reset_precondition_map();
    reset_cumulated_rw_effects();
    free_value_mappings();
}

void
init_convex_summary_rw_regions(const char* module_name)
{
    regions_init();
    /* for intermediate values */
    set_cumulated_rw_effects((statement_effects)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
    module_to_value_mappings(module_name_to_entity(module_name));

    init_convex_rw_prettyprint(module_name);
}

void reset_convex_summary_rw_regions(const char* __attribute__ ((unused)) module_name)
{
    regions_end();
    reset_cumulated_rw_effects();
    free_value_mappings();
}

void init_convex_summary_in_out_regions(const char* module_name)
{
    regions_init();
    set_bool_property("MUST_REGIONS", true);
    set_bool_property("EXACT_REGIONS", true);
    get_in_out_regions_properties();
    /* for intermediate values */
    set_cumulated_rw_effects((statement_effects)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
    module_to_value_mappings(module_name_to_entity(module_name));

    init_convex_inout_prettyprint(module_name);
}

void reset_convex_prettyprint(const char* __attribute__ ((unused)) module_name)
{
    effects_prettyprint_func = (generic_prettyprint_function) abort;
    effect_prettyprint_func = (void (*)(effect)) abort;
    effects_to_text_func = (generic_text_function) abort;
    reset_action_interpretation();
}

void
reset_convex_summary_in_out_regions(const char* __attribute__ ((unused)) module_name)
{
    regions_end();
    reset_cumulated_rw_effects();
    free_value_mappings();
}
