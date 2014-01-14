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
/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: utils.c
 * ~~~~~~~~~~~~~~~~~
 *
 * This File contains various useful functions, some of which should be moved
 * elsewhere.
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"

#include "effects-generic.h"

/* All this has to be changed into a context structure */


static pointer_info_val pointer_info_kind = with_no_pointer_info;

void set_pointer_info_kind(pointer_info_val val)
{
  pointer_info_kind = val;
}
pointer_info_val get_pointer_info_kind()
{
  return pointer_info_kind;
}

static bool constant_paths_p = false;

void set_constant_paths_p(bool b)
{
  constant_paths_p = b;
}

bool get_constant_paths_p()
{
  return constant_paths_p;
}


/* GENERIC FUNCTIONS on lists of effects to be instanciated for
   specific types of effects */

/* consistency checking */
bool (*effect_consistent_p_func)(effect);

/* initialisation and finalization */
void (*effects_computation_init_func)(const char* /* module_name */);
void (*effects_computation_reset_func)(const char* /* module_name */);

/* dup and free - This should be handled by newgen, but there is a problem
 * with the persistency of references - I do not understand what happens. */
effect (*effect_dup_func)(effect eff);
void (*effect_free_func)(effect eff);

/* make functions for effects */
effect (*reference_to_effect_func)(reference, action /* action */,
				   bool /* use_preference */);
list (*effect_to_store_independent_effect_list_func)(effect, bool);
void (*effect_add_expression_dimension_func)(effect eff, expression exp);
void (*effect_change_ith_dimension_expression_func)(effect eff, expression exp,
					       int i);

/* union */
effect (*effect_union_op)(effect, effect);
list (*effects_union_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
list (*effects_test_union_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));

/* intersection */
list (*effects_intersection_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));

/* difference */
list (*effects_sup_difference_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));
list (*effects_inf_difference_op)(
    list, list, bool (*eff1_eff2_combinable_p)(effect, effect));

/* composition with transformers */
list (*effects_transformer_composition_op)(list, transformer);
list (*effects_transformer_inverse_composition_op)(list, transformer);

/* composition with preconditions */
list (*effects_precondition_composition_op)(list,transformer);

/* evaluation with pointer information */
list (*eval_cell_with_points_to_func)(cell, descriptor, list, bool *, transformer);
list (*effect_to_constant_path_effects_func)(effect);

/* union over a range */
list (*effects_descriptors_variable_change_func)(list, entity, entity);
descriptor (*loop_descriptor_make_func)(loop);
list (*effects_loop_normalize_func)(
    list /* of effects */, entity /* index */, range,
    entity* /* new loop index */, descriptor /* range descriptor */,
    bool /* normalize descriptor ? */);
list (*effects_union_over_range_op)(list, entity, range, descriptor);
descriptor (*vector_to_descriptor_func)(Pvecteur);

/* interprocedural translation */
void (*effects_translation_init_func)(entity /* callee */,
				      list /* real_args */,
				      bool /* backward_p */);
void (*effects_translation_end_func)();
void (*effect_descriptor_interprocedural_translation_op)(effect);

list (*fortran_effects_backward_translation_op)(entity, list, list, transformer);
list (*fortran_effects_forward_translation_op)(entity /* callee */, list /* args */,
				       list /* effects */,
				       transformer /* context */);

list (*c_effects_on_formal_parameter_backward_translation_func)
(list /* of effects */,
 expression /* args */,
 transformer /* context */);

list (*c_effects_on_actual_parameter_forward_translation_func)
(entity /* callee */ ,
 expression /* real arg */,
 entity /* formal entity */,
 list /* effects */,
 transformer /* context */);

/* local to global name space translation */
list (*effects_local_to_global_translation_op)(list);



/* functions to provide context, a.k.a. precondition, and transformer information */
transformer (*load_context_func)(statement);
transformer (*load_transformer_func)(statement);
transformer (*load_completed_transformer_func)(statement);

bool (*empty_context_test)(transformer);

/* proper to contracted proper effects or to summary effects functions */
effect (*proper_to_summary_effect_func)(effect);

/* normalization of descriptors */
void (*effects_descriptor_normalize_func)(list /* of effects */);

/* getting/putting resources from/to pipsdbm */

statement_effects (*db_get_proper_rw_effects_func)(const char *);
void (*db_put_proper_rw_effects_func)(const char *, statement_effects);

statement_effects (*db_get_invariant_rw_effects_func)(const char *);
void (*db_put_invariant_rw_effects_func)(const char *, statement_effects);

statement_effects (*db_get_rw_effects_func)(const char *);
void (*db_put_rw_effects_func)(const char *, statement_effects);

list (*db_get_summary_rw_effects_func)(const char *);
void (*db_put_summary_rw_effects_func)(const char *, list);

statement_effects (*db_get_in_effects_func)(const char *);
void (*db_put_in_effects_func)(const char *, statement_effects);

statement_effects (*db_get_cumulated_in_effects_func)(const char *);
void (*db_put_cumulated_in_effects_func)(const char *, statement_effects);

statement_effects (*db_get_invariant_in_effects_func)(const char *);
void (*db_put_invariant_in_effects_func)(const char *, statement_effects);

list (*db_get_summary_in_effects_func)(const char *);
void (*db_put_summary_in_effects_func)(const char *, list);

list (*db_get_summary_out_effects_func)(const char *);
void (*db_put_summary_out_effects_func)(const char *, list);

statement_effects  (*db_get_out_effects_func)(const char *);
void (*db_put_out_effects_func)(const char *, statement_effects);

statement_effects (*db_get_live_in_paths_func)(const char *);
void (*db_put_live_in_paths_func)(const char *, statement_effects);

statement_effects (*db_get_live_out_paths_func)(const char *);
void (*db_put_live_out_paths_func)(const char *, statement_effects);

list (*db_get_live_in_summary_paths_func)(const char *);
void (*db_put_live_in_summary_paths_func)(const char *, list);

list (*db_get_live_out_summary_paths_func)(const char *);
void (*db_put_live_out_summary_paths_func)(const char *, list);

/* prettyprint function for debug */
void (*effects_prettyprint_func)(list); /* should be avoided : use print_effects instead */
void (*effect_prettyprint_func)(effect);

/* prettyprint function for sequential and user views */
text (*effects_to_text_func)(list);
void (*attach_effects_decoration_to_text_func)(text);


/* for cells */
bool (*cell_preceding_p_func)(cell, descriptor, cell, descriptor, bool, bool *);


/* RESET all generic methods... should be called when pips is started...
 */

#define UNDEF abort

typedef void (*void_function)();
typedef gen_chunk* (*chunks_function)();
typedef list (*list_function)();
typedef bool (*bool_function)();
typedef descriptor (*descriptor_function)();
typedef effect (*effect_function)();
typedef transformer (*transformer_function)();
typedef statement_effects (*statement_effects_function)();
typedef text (*text_function)();

void
generic_effects_reset_all_methods()
{
    effects_computation_init_func = (void_function) UNDEF;
    effects_computation_reset_func = (void_function) UNDEF;

    effect_dup_func = (effect_function) UNDEF;
    effect_free_func = (void_function) UNDEF;

    effect_union_op = (effect_function) UNDEF;
    effects_union_op = (list_function) UNDEF;
    effects_test_union_op = (list_function) UNDEF;
    effects_intersection_op = (list_function) UNDEF;
    effects_sup_difference_op = (list_function) UNDEF;
    effects_inf_difference_op = (list_function) UNDEF;
    effects_transformer_composition_op = (list_function) UNDEF;
    effects_transformer_inverse_composition_op = (list_function) UNDEF;
    effects_precondition_composition_op = (list_function) UNDEF;
    effects_descriptors_variable_change_func = (list_function) UNDEF;

    eval_cell_with_points_to_func = (list_function) UNDEF;
    effect_to_constant_path_effects_func = (list_function) UNDEF;

    effects_loop_normalize_func = (list (*)(list, entity, range, entity* , descriptor ,bool)) UNDEF;
    effects_union_over_range_op = (list (*)(list, entity, range, descriptor)) UNDEF;

    reference_to_effect_func = (effect(*)(reference,action,bool)) UNDEF;
    loop_descriptor_make_func = (descriptor_function) UNDEF;
    vector_to_descriptor_func = (descriptor_function) UNDEF;

    fortran_effects_backward_translation_op = (list_function) UNDEF;
    fortran_effects_forward_translation_op = (list_function) UNDEF;
    effects_local_to_global_translation_op = (list_function) UNDEF;
    c_effects_on_actual_parameter_forward_translation_func = (list_function) UNDEF;

    load_context_func = (transformer_function) UNDEF;
    load_transformer_func = (transformer_function) UNDEF;
    empty_context_test = (bool_function) UNDEF;
    proper_to_summary_effect_func = (effect_function) UNDEF;
    effects_descriptor_normalize_func = (void_function) UNDEF;

    db_get_proper_rw_effects_func = (statement_effects_function) UNDEF;
    db_put_proper_rw_effects_func = (void_function) UNDEF;
    db_get_invariant_rw_effects_func = (statement_effects_function) UNDEF;
    db_put_invariant_rw_effects_func = (void_function) UNDEF;
    db_get_rw_effects_func = (statement_effects_function) UNDEF;
    db_put_rw_effects_func = (void_function) UNDEF;
    db_get_summary_rw_effects_func = (list_function) UNDEF;
    db_put_summary_rw_effects_func = (void_function) UNDEF;
    db_get_in_effects_func = (statement_effects_function) UNDEF;
    db_put_in_effects_func = (void_function) UNDEF;
    db_get_cumulated_in_effects_func = (statement_effects_function) UNDEF;
    db_put_cumulated_in_effects_func = (void_function) UNDEF;
    db_get_invariant_in_effects_func = (statement_effects_function) UNDEF;
    db_put_invariant_in_effects_func = (void_function) UNDEF;
    db_get_summary_in_effects_func = (list_function) UNDEF;
    db_put_summary_in_effects_func = (void_function) UNDEF;
    db_get_summary_out_effects_func = (list_function) UNDEF;
    db_put_summary_out_effects_func = (void_function) UNDEF;
    db_get_out_effects_func = (statement_effects_function) UNDEF;
    db_put_out_effects_func = (void_function) UNDEF;
    db_get_live_in_paths_func = (statement_effects_function) UNDEF;
    db_put_live_in_paths_func = (void_function) UNDEF;
    db_get_live_out_paths_func = (statement_effects_function) UNDEF;
    db_put_live_out_paths_func = (void_function) UNDEF;
    db_get_live_in_summary_paths_func = (list_function) UNDEF;
    db_put_live_in_summary_paths_func = (void_function) UNDEF;
    db_get_live_out_summary_paths_func = (list_function) UNDEF;
    db_put_live_out_summary_paths_func = (void_function) UNDEF;

    set_contracted_proper_effects(true);
    set_contracted_rw_effects(true);

    set_descriptor_range_p(false);

    /* PRETTYPRINT related functions and settings
     */
    set_is_user_view_p(false);
    set_prettyprint_with_attachments(false);

    effects_prettyprint_func = (void_function) UNDEF;
    effect_prettyprint_func = (void_function) UNDEF;
    effects_to_text_func = (text_function) UNDEF;
    attach_effects_decoration_to_text_func = (void_function) UNDEF;

    reset_generic_prettyprints();
}
