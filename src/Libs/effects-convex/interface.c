/* package convex effects :  Be'atrice Creusillet 5/97
 *
 * File: interface.c
 * ~~~~~~~~~~~~~~~~~
 *
 * This File contains the interfaces with pipsmake which compute the various
 * types of convex regions by using the generic functions.
 *
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "transformer.h"
#include "semantics.h"

#include "effects-generic.h"
#include "effects-convex.h"

#include "pipsdbm.h"
#include "resources.h"

/******************************************************* CONVEX R/W REGIONS */

/* bool summary_regions(char *module_name): computes the global
 * regions of a module : global regions only use formal or common variables.
 */
bool 
summary_regions(char *module_name)
{
    bool res;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_rw_regions;
    effects_computation_reset_func = reset_convex_summary_rw_regions;
    res = summary_rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return res;
}

/* bool may_regions(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : computes the local regions of a module.
 * comment  : local regions can contain local variables.
 */
bool 
may_regions(char *module_name)
{
    bool res1, res2;

    set_bool_property("MUST_REGIONS", FALSE);

    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_rw_regions;
    effects_computation_reset_func = reset_convex_rw_regions;

    res1 = proper_effects_engine(module_name);
    res2 = rw_effects_engine(module_name);

    generic_effects_reset_all_methods();
   
    return res1 && res2;
}


/* bool must_regions(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : computes the local regions of a module.
 * comment  : local regions can contain local variables.
 */
bool 
must_regions(char *module_name)
{
    bool res1, res2;

    set_bool_property("MUST_REGIONS", TRUE);

    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_rw_regions;
    effects_computation_reset_func = reset_convex_rw_regions;
    res1 = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();

    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_rw_regions;
    effects_computation_reset_func = reset_convex_rw_regions;
    res2 = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();

    return res1 && res2;
}

/******************************************************** CONVEX IN REGIONS */


/* bool in_summary_regions(char *module_name): 
 * input    : the name of the current module.
 * output   : nothing !
 * modifies : the database.
 * comment  : computes the summary in regions of the current module, using the
 *            regions of its embedding statement.	
 */
bool 
in_summary_regions(char *module_name)
{
    bool res;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_in_out_regions;
    effects_computation_reset_func = reset_convex_summary_in_out_regions;

    res =  summary_in_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return res;
}

/* bool in_regions(char *module_name): 
 * input    : the name of the current module.
 * output   : nothing !
 * modifies : the database.
 * comment  : computes the in regions of the current module.	
 */
bool 
in_regions(char *module_name)
{
    bool res;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_in_out_regions;
    effects_computation_reset_func = reset_convex_in_out_regions;
    res = in_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return res;
}


/******************************************************* CONVEX OUT REGIONS */

bool
out_summary_regions(char * module_name)
{
    bool res;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_in_out_regions;
    effects_computation_reset_func = reset_convex_in_out_regions;

    res =  summary_out_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return res;
}

bool
out_regions(char *module_name)
{
    bool res;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_in_out_regions;
    effects_computation_reset_func = reset_convex_in_out_regions;

    res = out_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return res;
}


/************************************************************** PRETTYPRINT */

bool
print_code_proper_regions(char* module_name)
{
    bool ok;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_rw_regions;
    effects_computation_reset_func = reset_convex_summary_rw_regions;

    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(FALSE);

    set_action_interpretation(ACTION_READ, ACTION_WRITE);

    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_PROPER_REGIONS,
						      string_undefined,
						      ".preg");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_source_proper_regions(char* module_name)
{
    bool ok;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_rw_regions;
    effects_computation_reset_func = reset_convex_summary_rw_regions;

    set_is_user_view_p(TRUE);
    set_prettyprint_with_attachments(FALSE);

    set_action_interpretation(ACTION_READ, ACTION_WRITE);

    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_PROPER_REGIONS,
						      string_undefined,
						      ".upreg");
    generic_effects_reset_all_methods();
    return ok;
}


bool
print_code_regions(char* module_name)
{
    bool ok;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_rw_regions;
    effects_computation_reset_func = reset_convex_summary_rw_regions;

    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(FALSE);

    set_action_interpretation(ACTION_READ, ACTION_WRITE);

    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_REGIONS,
						      DBR_SUMMARY_REGIONS,
						      ".reg");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_source_regions(char* module_name)
{
    bool ok;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_rw_regions;
    effects_computation_reset_func = reset_convex_summary_rw_regions;

    set_is_user_view_p(TRUE);
    set_prettyprint_with_attachments(FALSE);

    set_action_interpretation(ACTION_READ, ACTION_WRITE);

    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_REGIONS,
						      DBR_SUMMARY_REGIONS,
						      ".ureg");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_code_in_regions(char* module_name)
{
    bool ok;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_in_out_regions;
    effects_computation_reset_func = reset_convex_summary_in_out_regions;

    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(FALSE);

    set_action_interpretation(ACTION_IN, ACTION_OUT);

    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_IN_REGIONS,
						      DBR_IN_SUMMARY_REGIONS,
						      ".inreg");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_source_in_regions(char* module_name)
{
    bool ok;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_in_out_regions;
    effects_computation_reset_func = reset_convex_summary_in_out_regions;

    set_is_user_view_p(TRUE);
    set_prettyprint_with_attachments(FALSE);

    set_action_interpretation(ACTION_IN, ACTION_OUT);

    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_IN_REGIONS,
						      DBR_IN_SUMMARY_REGIONS,
						      ".uinreg");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_code_out_regions(char* module_name)
{
    bool ok;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_in_out_regions;
    effects_computation_reset_func = reset_convex_summary_in_out_regions;

    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(FALSE);

    set_action_interpretation(ACTION_IN, ACTION_OUT);

    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_OUT_REGIONS,
						      DBR_OUT_SUMMARY_REGIONS,
						      ".outreg");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_source_out_regions(char* module_name)
{
    bool ok;
    set_methods_for_convex_effects();
    effects_computation_init_func = init_convex_summary_in_out_regions;
    effects_computation_reset_func = reset_convex_summary_in_out_regions;

    set_is_user_view_p(TRUE);
    set_prettyprint_with_attachments(FALSE);

    set_action_interpretation(ACTION_IN, ACTION_OUT);

    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_OUT_REGIONS,
						      DBR_OUT_SUMMARY_REGIONS,
						      ".uoutreg");
    generic_effects_reset_all_methods();
    return ok;
}


/************* INTERFACES TO COMPUTE SIMPLE PROPER EFFECTS FROM OTHER PHASES */

/* list proper_effects_of_expression(expression e)
 * input    : an expression and the current context
 * output   : the correpsonding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
regions_of_expression(expression e, transformer context)
{
    list le;

    le = proper_regions_of_expression(e, context);
    le = proper_regions_contract(le);
    return(le);
}

/* list proper_effects_of_expression(expression e)
 * input    : an expression and the current context
 * output   : the correpsonding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
proper_regions_of_expression(expression e, transformer context)
{
    list le;
    bool context_stack_defined_p = 
	effects_private_current_context_stack_initialized_p();

    if (!context_stack_defined_p)
    {
	set_methods_for_convex_effects();
	make_effects_private_current_context_stack();
    }
    effects_private_current_context_push(context);
    
    le = generic_proper_effects_of_expression(e);
    
    effects_private_current_context_pop();

    if (!context_stack_defined_p)
    {
	free_effects_private_current_context_stack();
    }

    return(le);
}

list 
proper_regions_of_expressions(list l_exp, transformer context)
{
    list le = NIL;
    MAP(EXPRESSION, exp,
	{
	    le = gen_nconc(le, proper_regions_of_expression(exp, context));
	},
	l_exp);
    return le;
}


