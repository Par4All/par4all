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

#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "resources.h"
#include "pipsdbm.h"
#include "effects-generic.h"
#include "effects-simple.h"


/****************************************************** PIPSMAKE INTERFACES */

/* SPECIFIC INTERFACES */

bool 
cumulated_references(char * module_name)
{
    bool ok;
    set_methods_for_cumulated_references();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool 
proper_references(char * module_name)
{
    bool ok;
    set_methods_for_proper_references();
    ok = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool 
proper_effects(char * module_name)
{
    bool res1, res2;

    set_methods_for_proper_references();
    res1 = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();

    set_methods_for_proper_simple_effects();
    res2 = proper_effects_engine(module_name);
    generic_effects_reset_all_methods();
    
    return res1 && res2;
}

bool 
summary_effects(string module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    ok = summary_rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool 
cumulated_effects(string module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    ok = rw_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
in_summary_effects(char *module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    ok = summary_in_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
out_summary_effects(char *module_name)
{
    return TRUE;
}


bool
in_effects(char *module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    ok = in_effects_engine(module_name);
    generic_effects_reset_all_methods();
    return ok;
}

bool
out_effects(char *module_name)
{
    return TRUE;
}


bool
print_code_proper_effects(char *module_name)
{
    bool ok;
    set_methods_for_proper_simple_effects();
    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(TRUE);
    set_read_action_interpretation(READ_IS_READ);
    set_write_action_interpretation(WRITE_IS_WRITE);
    ok = (print_source_or_code_with_any_effects_engine(module_name,
							 DBR_PROPER_EFFECTS,
							 string_undefined,
							 ".prop"));
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_code_cumulated_effects(char* module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(TRUE);
    set_read_action_interpretation(READ_IS_READ);
    set_write_action_interpretation(WRITE_IS_WRITE);
    ok = (print_source_or_code_with_any_effects_engine(module_name,
							 DBR_CUMULATED_EFFECTS,
							 DBR_SUMMARY_EFFECTS,
							 ".cumu"));
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_code_in_effects(char *module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(FALSE);
    set_read_action_interpretation(READ_IS_IN);
    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_IN_EFFECTS,
						      DBR_IN_SUMMARY_EFFECTS,
						      ".ineff");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_code_out_effects(char *module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    set_is_user_view_p(FALSE);
    set_prettyprint_with_attachments(FALSE);
    set_write_action_interpretation(WRITE_IS_OUT);
    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_OUT_EFFECTS,
						      DBR_OUT_SUMMARY_EFFECTS,
						      ".uouteff");
    generic_effects_reset_all_methods();
    return ok;
}


bool
print_source_proper_effects(char *module_name)
{
    bool ok;
    set_methods_for_proper_simple_effects();
    set_is_user_view_p(TRUE);
    set_prettyprint_with_attachments(TRUE);
    set_read_action_interpretation(READ_IS_READ);
    set_write_action_interpretation(WRITE_IS_WRITE);
    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_PROPER_EFFECTS,
						      string_undefined,
						      ".uprop");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_source_cumulated_effects(char* module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    set_is_user_view_p(TRUE);
    set_prettyprint_with_attachments(TRUE);
    set_read_action_interpretation(READ_IS_READ);
    set_write_action_interpretation(WRITE_IS_WRITE);
    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_CUMULATED_EFFECTS,
						      DBR_SUMMARY_EFFECTS,
						      ".ucumu");
    generic_effects_reset_all_methods();
    return ok;
}


bool
print_source_in_effects(char *module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    set_is_user_view_p(TRUE);
    set_prettyprint_with_attachments(FALSE);
    set_read_action_interpretation(READ_IS_IN);
    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_IN_EFFECTS,
						      DBR_IN_SUMMARY_EFFECTS,
						      ".uineff");
    generic_effects_reset_all_methods();
    return ok;
}

bool
print_source_out_effects(char *module_name)
{
    bool ok;
    set_methods_for_simple_effects();
    set_is_user_view_p(TRUE);
    set_prettyprint_with_attachments(FALSE);
    set_write_action_interpretation(WRITE_IS_OUT);
    ok = print_source_or_code_with_any_effects_engine(module_name,
						      DBR_OUT_EFFECTS,
						      DBR_OUT_SUMMARY_EFFECTS,
						      ".uouteff");
    generic_effects_reset_all_methods();
    return ok;
}

text
get_text_proper_effects(string module_name)
{
    text t;
    set_is_user_view_p(FALSE);

    t = get_any_effect_type_text(module_name,
				 DBR_PROPER_EFFECTS,
				 string_undefined,
				 FALSE);
    return t;
}


text
get_text_cumulated_effects(string module_name)
{
    text t;
    set_is_user_view_p(FALSE);
    t = get_any_effect_type_text(module_name,
				 DBR_CUMULATED_EFFECTS,
				 DBR_SUMMARY_EFFECTS,
				 FALSE);
    return t;
}


/*********** INTERFACES TO COMPUTE SIMPLE PROPER EFFECTS FROM OTHER PHASES */

/* list proper_effects_of_expression(expression e)
 * input    : an expression and the current context
 * output   : the correpsonding list of effects.
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
    }

    effects_private_current_context_push(transformer_undefined);
    le = generic_proper_effects_of_expression(e);
    effects_private_current_context_pop();
    if (!context_stack_defined_p)
    {
	free_effects_private_current_context_stack();
    }

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

/****************************************************************/
/* BACKWARD COMPATIBILITY                                       */
/****************************************************************/

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
    reset_invariant_rw_effects();
}



/* called from rice */
list 
statement_to_effects(s)
statement s;
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
    reset_proper_rw_effects();
    reset_rw_effects();
    reset_invariant_rw_effects();
    generic_effects_reset_all_methods();

    return(l_eff);
}


