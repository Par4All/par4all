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
#include "linear.h"
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

#define is_rw 		(1)
#define is_inout	(2)
#define is_copyinout	(3)
#define is_private	(4)

static bool 
print_code_any_regions(
    string module_name,
    int what_tag, 
    bool is_user_view,
    bool is_attached,
    string resource_name,
    string summary_resource_name,
    string suffix)
{
    bool ok;
    set_methods_for_convex_effects();
    switch(what_tag)
    {
    case is_rw: 
	init_convex_rw_prettyprint(module_name);
	break;
    case is_inout:
	init_convex_inout_prettyprint(module_name);
	break;
    default:
	pips_internal_error("unexpected tag %d\n", what_tag);
    }

    set_is_user_view_p(is_user_view);
    set_prettyprint_with_attachments(is_attached);

    ok = print_source_or_code_with_any_effects_engine
	(module_name, resource_name, summary_resource_name, suffix);

    generic_effects_reset_all_methods();
    return ok;
}

bool
print_code_proper_regions(string module_name)
{
    return print_code_any_regions(module_name, is_rw, FALSE, FALSE, 
			  DBR_PROPER_REGIONS, string_undefined, ".preg");
}

bool
print_source_proper_regions(char* module_name)
{
    return print_code_any_regions(module_name, is_rw, TRUE, FALSE, 
			  DBR_PROPER_REGIONS, string_undefined, ".upreg");
}

bool
print_code_regions(string module_name)
{
    return print_code_any_regions(module_name, is_rw, FALSE, FALSE, 
			  DBR_REGIONS, DBR_SUMMARY_REGIONS, ".reg");
}

bool
print_source_regions(string module_name)
{
    return print_code_any_regions(module_name, is_rw, TRUE, FALSE, 
			  DBR_REGIONS, DBR_SUMMARY_REGIONS, ".ureg");
}

bool
print_code_in_regions(string module_name)
{
    return print_code_any_regions(module_name, is_inout, FALSE, FALSE, 
			  DBR_IN_REGIONS, DBR_IN_SUMMARY_REGIONS, ".inreg");
}

bool
print_source_in_regions(string module_name)
{
    return print_code_any_regions(module_name, is_inout, TRUE, FALSE, 
			  DBR_IN_REGIONS, DBR_IN_SUMMARY_REGIONS, ".uinreg");
}

bool
print_code_out_regions(string module_name)
{
    return print_code_any_regions(module_name, is_inout, FALSE, FALSE, 
			  DBR_OUT_REGIONS, DBR_OUT_SUMMARY_REGIONS, ".outreg");
}

bool
print_source_out_regions(string module_name)
{
    return print_code_any_regions(module_name, is_inout, TRUE, FALSE, 
		  DBR_OUT_REGIONS, DBR_OUT_SUMMARY_REGIONS, ".uoutreg");
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


