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
/* package simple effects :  Be'atrice Creusillet 5/97
 *
 * File: interface.c
 * ~~~~~~~~~~~~~~~~~
 *
 * This File contains.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "properties.h"
#include "database.h"
#include "resources.h"
#include "effects-generic.h"
#include "effects-simple.h"

/****************************************************** PIPSMAKE INTERFACES */

/* SPECIFIC INTERFACES */

bool
cumulated_references(const char* module_name)
{
    bool ok;
    set_constant_paths_p(false);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_cumulated_references();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
proper_references(const char* module_name)
{
    bool ok;
    set_constant_paths_p(false);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_proper_references();
    ok = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}


bool
proper_pointer_effects(const char* module_name)
{
    bool ok;
    set_constant_paths_p(false);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_proper_simple_pointer_effects();
    ok = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
summary_pointer_effects(const char* module_name)
{
    bool ok;
    set_constant_paths_p(false);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_simple_pointer_effects();
    ok = summary_rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
cumulated_pointer_effects(const char* module_name)
{
    bool ok;
    set_constant_paths_p(false);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_simple_pointer_effects();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
cumulated_pointer_effects_with_points_to(const char* module_name)
{
    bool ok;
    set_constant_paths_p(false);
    set_pointer_info_kind(with_points_to);
    set_methods_for_simple_pointer_effects();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
cumulated_pointer_effects_with_pointer_values(const char* module_name)
{
    bool ok;
    set_constant_paths_p(false);
    set_pointer_info_kind(with_pointer_values);
    set_methods_for_simple_pointer_effects();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
proper_effects(const char* module_name)
{
    bool ok;
    if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
      set_constant_paths_p(false);
    else
      set_constant_paths_p(true);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_proper_simple_effects();
    ok = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
proper_effects_with_points_to(const char* module_name)
{
    bool ok;
    set_constant_paths_p(true);
    set_pointer_info_kind(with_points_to);
    set_methods_for_proper_simple_effects();
    ok = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
proper_effects_with_pointer_values(const char* module_name)
{
    bool ok;
    set_constant_paths_p(true);
    set_pointer_info_kind(with_pointer_values);
    set_methods_for_proper_simple_effects();
    ok = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
summary_effects(const char* module_name)
{
    bool ok;
    if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
      set_constant_paths_p(false);
    else
      set_constant_paths_p(true);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_simple_effects();
    ok = summary_rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
cumulated_effects(const char* module_name)
{
    bool ok;
     if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
      set_constant_paths_p(false);
    else
      set_constant_paths_p(true);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_simple_effects();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
cumulated_effects_with_points_to(const char* module_name)
{
    bool ok;
    set_constant_paths_p(true);
    set_pointer_info_kind(with_points_to);
    set_methods_for_simple_effects();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
cumulated_effects_with_pointer_values(const char* module_name)
{
    bool ok;
    set_constant_paths_p(true);
    set_pointer_info_kind(with_pointer_values);
    set_methods_for_simple_effects();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
in_summary_effects(const char* module_name)
{
    bool ok;
    if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
      set_constant_paths_p(false);
    else
      set_constant_paths_p(true);
    set_methods_for_inout_effects(module_name);
    ok = summary_in_effects_engine(module_name);
    generic_effects_reset_all_methods();
    reset_methods_for_inout_effects();
    return ok;
}

bool
out_summary_effects(const char* module_name)
{
    bool ok;
    if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
      set_constant_paths_p(false);
    else
      set_constant_paths_p(true);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_inout_effects(module_name);
    ok = summary_out_effects_engine(module_name);
    generic_effects_reset_all_methods();
    reset_methods_for_inout_effects();
    return ok;
}


bool
in_effects(const char* module_name)
{
    bool ok;
    if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
      set_constant_paths_p(false);
    else
      set_constant_paths_p(true);
    set_pointer_info_kind(with_no_pointer_info);
    set_methods_for_inout_effects(module_name);
    ok = in_effects_engine(module_name, simple);
    generic_effects_reset_all_methods();
    reset_methods_for_inout_effects();
    return ok;
}

bool
out_effects(const char* module_name)
{
    bool ok;
    if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
      set_constant_paths_p(false);
    else
      set_constant_paths_p(true);
    set_methods_for_inout_effects(module_name);
    ok = out_effects_engine(module_name, simple);
    generic_effects_reset_all_methods();
    reset_methods_for_inout_effects();
    return ok;
}

/************************************************************* PRETTYPRINTS */

static bool
print_code_effects(
    const char* module_name,
    simple_effects_actions_interpretations ac_inter,
    bool is_user_view,
    bool is_attached,
    string resource_name,
    string summary_resource_name,
    string suffix)
{
    bool ok;

    if (ac_inter == act_rw)
      set_methods_for_rw_effects_prettyprint(module_name);
    else if (ac_inter == act_inout)
      set_methods_for_inout_effects_prettyprint(module_name);
    else if (ac_inter == act_live_in)
      set_methods_for_live_in_paths_prettyprint(module_name);
    else if (ac_inter == act_live_out)
      set_methods_for_live_out_paths_prettyprint(module_name);
    else
      pips_internal_error("erroneous actions interpretation\n");

    set_is_user_view_p(is_user_view);
    set_prettyprint_with_attachments(is_attached);

    ok = print_source_or_code_with_any_effects_engine
      (module_name, resource_name, summary_resource_name, suffix, false);

    reset_methods_for_effects_prettyprint(module_name);

    /* generic_effects_reset_all_methods(); not useful here */
    return ok;
}

bool
print_code_proper_pointer_effects(const char* module_name)
{
  return print_code_effects(module_name, act_rw, false, true,
			      DBR_PROPER_POINTER_EFFECTS, string_undefined, ".prop");
}

bool
print_code_cumulated_pointer_effects(const char* module_name)
{
    return print_code_effects(module_name, act_rw, false, true,
		      DBR_CUMULATED_POINTER_EFFECTS, DBR_SUMMARY_POINTER_EFFECTS, ".cumu");
}

bool
print_code_proper_effects(const char* module_name)
{
    return print_code_effects(module_name, act_rw, false, true,
			      DBR_PROPER_EFFECTS, string_undefined, ".prop");
}

bool
print_code_cumulated_effects(const char* module_name)
{
    return print_code_effects(module_name, act_rw, false, true,
		      DBR_CUMULATED_EFFECTS, DBR_SUMMARY_EFFECTS, ".cumu");
}

bool
print_code_proper_references(const char* module_name)
{
    return print_code_effects(module_name, act_rw, false, true,
		      DBR_PROPER_REFERENCES, string_undefined, ".propref");
}

bool
print_code_cumulated_references(const char* module_name)
{
    return print_code_effects(module_name, act_rw, false, true,
		      DBR_CUMULATED_REFERENCES, string_undefined, ".cumuref");
}

bool
print_code_in_effects(const char* module_name)
{
    return print_code_effects(module_name, act_inout, false, false,
		      DBR_IN_EFFECTS, DBR_IN_SUMMARY_EFFECTS, ".ineff");
}

bool
print_code_out_effects(const char* module_name)
{
    return print_code_effects(module_name, act_inout, false, false,
		      DBR_OUT_EFFECTS, DBR_OUT_SUMMARY_EFFECTS, ".outeff");
}

bool
print_source_proper_effects(const char* module_name)
{
    return print_code_effects(module_name, act_rw, true, true,
		      DBR_PROPER_EFFECTS, string_undefined, ".uprop");
}

bool
print_source_cumulated_effects(const char* module_name)
{
    return print_code_effects(module_name, act_rw, true, true,
		      DBR_CUMULATED_EFFECTS, DBR_SUMMARY_EFFECTS, ".ucumu");
}


bool
print_source_in_effects(const char* module_name)
{
    return print_code_effects(module_name, false, true, false,
		      DBR_IN_EFFECTS, DBR_IN_SUMMARY_EFFECTS, ".uineff");
}

bool
print_source_out_effects(const char* module_name)
{
    return print_code_effects(module_name, act_inout, true, false,
		      DBR_OUT_EFFECTS, DBR_OUT_SUMMARY_EFFECTS, ".uouteff");
}


/********************************************************** OTHER FUNCTIONS */

text
get_text_proper_pointer_effects(const char* module_name)
{
    text t;

    set_is_user_view_p(false);
    set_methods_for_rw_effects_prettyprint(module_name);
    t = get_any_effect_type_text(module_name,
				 DBR_PROPER_POINTER_EFFECTS,
				 string_undefined,
				 false);
    reset_methods_for_effects_prettyprint(module_name);
    return t;
}

text
get_text_cumulated_pointer_effects(const char* module_name)
{
    text t;

    set_is_user_view_p(false);
    set_methods_for_rw_effects_prettyprint(module_name);
    t = get_any_effect_type_text(module_name,
				 DBR_CUMULATED_POINTER_EFFECTS,
				 DBR_SUMMARY_POINTER_EFFECTS,
				 false);
    reset_methods_for_effects_prettyprint(module_name);
    return t;
}

text
get_text_proper_effects(const char* module_name)
{
    text t;

    set_is_user_view_p(false);
    set_methods_for_rw_effects_prettyprint(module_name);
    t = get_any_effect_type_text(module_name,
				 DBR_PROPER_EFFECTS,
				 string_undefined,
				 false);
    reset_methods_for_effects_prettyprint(module_name);
    return t;
}

text
get_text_cumulated_effects(const char* module_name)
{
    text t;

    set_is_user_view_p(false);
    set_methods_for_rw_effects_prettyprint(module_name);
    t = get_any_effect_type_text(module_name,
				 DBR_CUMULATED_EFFECTS,
				 DBR_SUMMARY_EFFECTS,
				 false);
    reset_methods_for_effects_prettyprint(module_name);
    return t;
}


/*********** INTERFACES TO COMPUTE SIMPLE PROPER EFFECTS FROM OTHER PHASES */

/* list proper_effects_of_expression(expression e)
 * input    : an expression and the current context
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  :
 */
list
proper_effects_of_expression(expression e)
{
    list le;
    bool context_stack_defined_p =
	effects_private_current_context_stack_initialized_p();

    if (!context_stack_defined_p)
    {
	set_methods_for_simple_effects();
	make_effects_private_current_context_stack();
	debug_on("PROPER_EFFECTS_DEBUG_LEVEL");
    }

    effects_private_current_context_push(transformer_undefined);
    le = generic_proper_effects_of_expression(e);
    effects_private_current_context_pop();
    if (!context_stack_defined_p)
    {
	free_effects_private_current_context_stack();
	debug_off();
    }

    return(le);
}



/* Same as above, but with debug control. Used by semantics. */
list
expression_to_proper_effects(expression e)
{
    list le;

    debug_on("EFFECTS_DEBUG_LEVEL");

    le = proper_effects_of_expression(e);

    debug_off();

    return(le);
}

/** computes the proper constant path effects of the input expression
    using no points-to information.

    dereferencing paths are currently changed to anywhere effects
 */
list
proper_constant_path_effects_of_expression(expression e)
{
    list le;
    bool context_stack_defined_p =
	effects_private_current_context_stack_initialized_p();

    // needed because it can be called from phases that already define
    // these variables
    bool saved_constant_paths_p = get_constant_paths_p();
    pointer_info_val saved_pointer_info_kind = get_pointer_info_kind();

    if (! c_module_p(get_current_module_entity()) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
      set_constant_paths_p(false);
    else
      set_constant_paths_p(true);
    set_pointer_info_kind(with_no_pointer_info);

    if (!context_stack_defined_p)
    {
	set_methods_for_simple_effects();
	make_effects_private_current_context_stack();
    }

    debug_on("PROPER_EFFECTS_DEBUG_LEVEL");

    effects_private_current_context_push(transformer_undefined);
    le = generic_proper_effects_of_expression(e);

    if (get_constant_paths_p())
      {
	list l_tmp = le;
	le = pointer_effects_to_constant_path_effects(le);
	effects_free(l_tmp);
      }

    effects_private_current_context_pop();
    if (!context_stack_defined_p)
    {
	free_effects_private_current_context_stack();
    }
    debug_off();

    set_constant_paths_p(saved_constant_paths_p);
    set_pointer_info_kind(saved_pointer_info_kind);
    return(le);
}



/* Same as above, but with some more debug control. Used by semantics.
 *
 * FI: What has to be initialized? I end up with functional pointer
 * reference_to_effect_func unitialized, i.e. 0.
 *
 * Use set_methods_for_simple_effects() and
 * generic_effects_reset_all_methods()
 */
list
expression_to_proper_constant_path_effects(expression e)
{
    list le;

    debug_on("EFFECTS_DEBUG_LEVEL");

    le = proper_constant_path_effects_of_expression(e);

    debug_off();

    return(le);
}

bool expression_with_side_effect_p(expression e)
{
  list efl = expression_to_proper_constant_path_effects(e);
  bool side_effect_p = false;
  FOREACH(EFFECT, ef, efl) {
    if(effect_write_p(ef)) {
      side_effect_p = true;
      break;
    }
  }
  gen_free_list(efl);
  return side_effect_p;
}

/** computes the proper constant path effects of the input expression
    using the points-to information of the input statement.

    set_pt_to_list( (statement_points_to)
		    db_get_memory_resource(DBR_POINTS_TO_LIST, module_name, true) );
    must have been executed before calling this function for the first time

    and reset_pt_to_list() must be called after all the calls have been performed.
 */
list
proper_constant_path_effects_of_expression_with_points_to(expression e, statement stmt)
{
    list le;
    bool context_stack_defined_p =
	effects_private_current_context_stack_initialized_p();
    bool stmt_stack_defined_p =
	effects_private_current_stmt_stack_initialized_p();

    // needed because it can be called from phases that already define
    // these variables
    bool saved_constant_paths_p = get_constant_paths_p();
    pointer_info_val saved_pointer_info_kind = get_pointer_info_kind();
    set_constant_paths_p(true);
    set_pointer_info_kind(with_points_to);

    if (!context_stack_defined_p)
      {
	set_methods_for_simple_effects();
	make_effects_private_current_context_stack();
      }

    if (!stmt_stack_defined_p)
      make_effects_private_current_stmt_stack();
    debug_on("PROPER_EFFECTS_DEBUG_LEVEL");

    effects_private_current_context_push(transformer_undefined);
    effects_private_current_stmt_push(stmt);
    le = generic_proper_effects_of_expression(e);

    list l_tmp = le;
    le = pointer_effects_to_constant_path_effects(le);
    effects_free(l_tmp);

    effects_private_current_context_pop();
    effects_private_current_stmt_pop();
    if (!context_stack_defined_p)
      {
	free_effects_private_current_context_stack();
      }
    if (!stmt_stack_defined_p)
      {
	free_effects_private_current_stmt_stack();
      }
    debug_off();
    set_constant_paths_p(saved_constant_paths_p);
    set_pointer_info_kind(saved_pointer_info_kind);

    return(le);
}



/* Same as above, but with debug control. Used by semantics. */
list
expression_to_proper_constant_path_effects_with_points_to(expression e, statement stmt)
{
    list le;

    debug_on("EFFECTS_DEBUG_LEVEL");

    le = proper_constant_path_effects_of_expression_with_points_to(e, stmt);

    debug_off();

    return(le);
}



/* list proper_effects_of_range(range r)
 * input    : an expression and the current context
 * output   : the correpsonding list of effects.
 * modifies : nothing.
 * comment  :
 */
list
proper_effects_of_range(range r)
{
    list le;
    bool context_stack_defined_p =
	effects_private_current_context_stack_initialized_p();

    if (!context_stack_defined_p)
    {
	    set_methods_for_simple_effects();
	    make_effects_private_current_context_stack();
    }
    effects_private_current_context_push(transformer_undefined);

    le = generic_proper_effects_of_range(r);
    effects_private_current_context_pop();

    if (!context_stack_defined_p)
    {
	free_effects_private_current_context_stack();
    }

    return(le);
}

bool expression_invariant_wrt_effects(expression exp, list el)
{
  list ee = proper_effects_of_expression(exp);
  list cee = list_undefined;
  list cel = list_undefined;
  bool invariant_p = true;

  for(cee=ee; !ENDP(cee) && invariant_p; POP(cee)) {
    effect exp_e = EFFECT(CAR(cee));
    //reference exp_r = effect_any_reference(exp_e);
    //entity exp_v = reference_variable(exp_r);

    for(cel=el; !ENDP(el) && invariant_p; POP(el)) {
      effect l_e = EFFECT(CAR(cel));
      action l_a = effect_action(l_e);

      if(action_write_p(l_a)) {
	//reference l_r = effect_any_reference(l_e);
	//entity l_v = reference_variable(l_r);

	if(effects_interfere_p(l_e,exp_e)) {
	  invariant_p = false;
	}
      }
    }
  }
  return invariant_p;
}


/*************************************************** BACKWARD COMPATIBILITY */

/* called from prettyprint CRAY */
void
rproper_effects_of_statement(statement s)
{
    set_methods_for_proper_simple_effects();
    proper_effects_of_module_statement(s);
    return;
}

void
rcumulated_effects_of_statement(statement s)
{
    init_invariant_rw_effects();
    set_methods_for_simple_effects();
    rw_effects_of_module_statement(s);
    close_invariant_rw_effects();
}

/* called from rice */
list
statement_to_effects(statement s)
{
    list l_eff;

    init_proper_rw_effects();
    init_rw_effects();
    init_invariant_rw_effects();

    debug_on("EFFECTS_DEBUG_LEVEL");
    pips_debug(1, "begin\n");

    set_methods_for_proper_simple_effects();
    proper_effects_of_module_statement(s);
    set_methods_for_simple_effects();
    rw_effects_of_module_statement(s);
    l_eff = effects_dup(load_rw_effects_list(s));

    pips_debug(1, "end\n");
    debug_off();

    /* Faudrait faire les free, mais je ne sais pas comment. mail a` fc */
    close_proper_rw_effects();
    close_rw_effects();
    close_invariant_rw_effects();
    generic_effects_reset_all_methods();

    return l_eff;
}

/* SIDE EFFECT: set both proper_rw_effects and expr_prw_effects.
 */
bool full_simple_proper_effects(const char* module_name, statement current)
{
  bool ok = true;
  set_methods_for_proper_simple_effects();
  expression_proper_effects_engine(module_name, current);
  generic_effects_reset_all_methods();
  return ok;
}

bool simple_cumulated_effects(const char* module_name, statement current)
{
  bool ok = true;
  set_methods_for_proper_simple_effects();

  (*effects_computation_init_func)(module_name);

  /* We also need the proper effects of the module */
  /*
  set_proper_rw_effects((*db_get_proper_rw_effects_func)(module_name));
  */

  /* Compute the rw effects or references of the module. */
  init_rw_effects();
  init_invariant_rw_effects();

  debug_on("EFFECTS_DEBUG_LEVEL");
  pips_debug(1, "begin\n");

  rw_effects_of_module_statement(current);

  pips_debug(1, "end\n");
  debug_off();

  (*db_put_rw_effects_func)(module_name, get_rw_effects());
  (*db_put_invariant_rw_effects_func)(module_name, get_invariant_rw_effects());

  (*effects_computation_reset_func)(module_name);

  generic_effects_reset_all_methods();
  return ok;
}



/*********** INTERFACES FOR LIVENESS ANALYSIS */


bool
live_paths(const char* module_name)
{
  bool ok;
  if (! c_module_p(module_name_to_entity(module_name)) || !get_bool_property("CONSTANT_PATH_EFFECTS"))
    set_constant_paths_p(false);
  else
    set_constant_paths_p(true);
  set_pointer_info_kind(with_no_pointer_info);
  set_methods_for_live_paths(module_name);
  ok = live_paths_engine(module_name, simple);
  generic_effects_reset_all_methods();
  return ok;
}

bool
live_in_summary_paths(const char* module_name)
{
  bool ok;
  if (! c_module_p(module_name_to_entity(module_name))
      || !get_bool_property("CONSTANT_PATH_EFFECTS"))
    set_constant_paths_p(false);
  else
    set_constant_paths_p(true);
  set_methods_for_live_paths(module_name);
  ok = live_in_summary_paths_engine(module_name);
  generic_effects_reset_all_methods();
  return ok;
}

bool
live_out_summary_paths(const char* module_name)
{
  bool ok;
  if (! c_module_p(module_name_to_entity(module_name))
      || !get_bool_property("CONSTANT_PATH_EFFECTS"))
    set_constant_paths_p(false);
  else
    set_constant_paths_p(true);
  set_pointer_info_kind(with_no_pointer_info);
  set_methods_for_live_paths(module_name);
  ok = live_out_summary_paths_engine(module_name);
  generic_effects_reset_all_methods();
  return ok;
}

bool
print_code_live_in_paths(const char* module_name)
{
    return print_code_effects(module_name, act_live_in, false, true,
		      DBR_LIVE_IN_PATHS, DBR_LIVE_IN_SUMMARY_PATHS, ".live_in");
}

bool
print_code_live_out_paths(const char* module_name)
{
    return print_code_effects(module_name, act_live_out, false, true,
		      DBR_LIVE_OUT_PATHS, DBR_LIVE_OUT_SUMMARY_PATHS, ".live_out");
}
