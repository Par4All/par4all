/* $Id$
 */
#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "constants.h"
#include "control.h"
#include "misc.h"
#include "text.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "semantics.h"

#include "transformer.h"

#include "pipsdbm.h"
#include "resources.h"

#define BACKWARD TRUE
#define FORWARD FALSE


static entity callee;
static list list_regions_callee = NIL;
static statement current_caller_stmt = statement_undefined;
static list list_pairs = NIL;


/* modifies global var current_caller_stmt */
static bool stmt_filter(s)
statement s;
{
    pips_debug(9, "statement %03d\n", statement_number(s));
    
    current_caller_stmt = s;
    return(TRUE);
}


/* static void
 * add_parameter_aliases_for_this_call_site(call call_site,
 * transformer context, list real_args)
 * constructs the alias pairs for the effective parameters (but not for
 * COMMON regions) at this call site and adds them to the list
 * input    : parameters: a call site and the calling context
 *            global variables: callee,list_regions_callee,list_pairs
 * output   : void
 * global vars IN: list_regions_callee and list_pairs
 * modifies : global var list_pairs
 *            for each region in list_regions_callee which is a region of a
 *            formal parameter (of the callee) and for which the corresponding
 *            real parameter is an expression with only one entity, this
 *            function performs
 *            the backward translation: callee_region -> real_region
 *            and adds an alias pair <callee_region,real_region> to list_pairs
 * comment  :	
 *
 * Algorithm :
 * -----------
 *    let list_regions_callee be the list of the regions on variables
 *    of callee
 *    let list_pairs be the list of alias pairs for the callee
 *
 *    FOR each expression real_exp IN real_args
 *        arg_num = number in the list of the function real arguments
 *        FOR each callee_region IN list_regions_callee
 *            callee_ent = entity of the region callee_region
 *            IF callee_ent is the formal parameter numbered arg_num
 *                IF real_exp is an lhs (expression with one entity)
 *                    real_region = translation of the region callee_region
 *                    list_pairs = list_pairs + <callee_region,real_region>
 *                ENDIF
 *            ENDIF
 *        ENDFOR
 *    ENDFOR
 */
static void
add_parameter_aliases_for_this_call_site(call call_site,
					 transformer context, list real_args)
{
    list r_args;
    int arg_num;

    pips_debug(4,"begin\n");

/*    real_args = call_arguments(call_site); */

    for (r_args = real_args, arg_num = 1; r_args != NIL;
	 r_args = CDR(r_args), arg_num++) 
    {
	pips_debug(9,"compare formal parameter arg_num %03d\n",arg_num);

	MAP(EFFECT, callee_region,
	 {
/*	     entity callee_ent = region_entity(callee_region); */
	     
	     pips_debug(9,"\tand entity %s\n",
			entity_name(region_entity(callee_region)));

	     /* If the formal parameter corresponds to the real argument then
	      * we perform the translation.
	      */
	     if (ith_parameter_p(callee,region_entity(callee_region),arg_num))
	     {
		 expression real_exp = EXPRESSION(CAR(r_args));
		 syntax real_syn = expression_syntax(real_exp);
		 
		 pips_debug(9,"match\n");

		 /* If the real argument is a reference to an entity, then we
		  * translate the regions of the corresponding formal parameter
		  */
		 if (syntax_reference_p(real_syn)) 
		 {
		    reference real_ref = syntax_reference(real_syn);
		    entity real_ent = reference_variable(real_ref);
		    region real_reg;
		    list pair;

		    pips_debug(9,"arg refers to entity\n");
		    pips_debug(9,"\t%s\n",entity_name(real_ent));

		    real_reg =
			region_translation(
			    callee_region,
			    callee,
			    reference_undefined,
			    real_ent,
			    get_current_module_entity(),
			    real_ref,
			    VALUE_ZERO,
			    BACKWARD);

		    pair = CONS(EFFECT,real_reg,NIL);
		    ifdebug(9)
			{
			    pips_debug(9,"region translated to:\n\t");
			    print_inout_regions(pair);
			}
		    pair = CONS(EFFECT,region_dup(callee_region),pair);

		    list_pairs = CONS(EFFECTS,make_effects(pair),list_pairs);
		}
	     }
	 }, list_regions_callee);
    }

    pips_debug(4,"end\n");
       
}

/* constructs the alias pairs for this call site and adds them to the list
 * global vars IN: callee, list_regions_callee, current_caller_stmt
 * and list_pairs
 * modifies global var: list_pairs
 */

static bool
add_alias_pairs_for_this_call_site(call call_site)
{
    transformer context;
    list real_args;

    if (call_function(call_site) != callee) return TRUE;

    pips_debug(4,"begin\n");

    pips_debug(9,
	       "try load_statement_precondition for statement %03d\n",
	       statement_number(current_caller_stmt));

    context = load_statement_precondition(current_caller_stmt);

    pips_debug(9,"got context:\n\t%s\n",transformer_to_string(context));

    pips_debug(9,"try call_arguments\n");

    real_args = call_arguments(call_site);

    pips_debug(9,"try set_interprocedural_translation_context_sc\n");

    pips_debug(9,"\tfor callee %s\n",entity_name(callee));

    set_interprocedural_translation_context_sc(callee, real_args);

    pips_debug(9,"try set_backward_arguments_to_eliminate\n");

    set_backward_arguments_to_eliminate(callee);

    add_parameter_aliases_for_this_call_site(call_site,context,real_args);
/*    add_common_aliases_for_this_call_site(); */

    reset_translation_context_sc();
    reset_arguments_to_eliminate();

    pips_debug(4,"end\n");

    return TRUE;
}


/* constructs the alias pairs for this caller and adds them to the list
 * global vars IN: callee, list_regions_callee and list_pairs
 * modifies global vars: list_pairs and current_caller_stmt
 */
static void
add_alias_pairs_for_this_caller( entity caller )
{
    char *caller_name;
    statement caller_statement;

    reset_current_module_entity();
    set_current_module_entity(caller);
    caller_name = module_local_name(caller);
    pips_debug(4,"begin for caller: %s\n", caller_name);
    
    /* ATTENTION: we must do ALL this before calling 
     * set_interprocedural_translation_context_sc
     * (in add_alias_pairs_for_this_call_site
     * called by the gen_multi_recurse below) !!!
     */
    /* the current module becomes the caller */
    regions_init();
    get_in_out_regions_properties();
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, caller_name, TRUE) );
    set_cumulated_rw_effects((statement_effects)
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, caller_name, TRUE));
    module_to_value_mappings(caller);
    set_precondition_map( (statement_mapping) 
        db_get_memory_resource(DBR_PRECONDITIONS, caller_name, TRUE));
    /* that's it,
     * but we musn't forget to reset it all again below !
     */

    caller_statement = get_current_module_statement();


/*  gen_multi_recurse(obj,
 *                   [domain, filter, rewrite,]*
 *                    NULL);
 *
 *  recurse from object obj,
 *  applies filter_i on encountered domain_i objects,
 *  if true, recurses down from the domain_i object, 
 *       and applies rewrite_i on exit from the object.
 */

    gen_multi_recurse(caller_statement,
		      statement_domain,
		      stmt_filter,
		      gen_null,
		      call_domain,
		      add_alias_pairs_for_this_call_site,
		      gen_null,
		      NULL);
  
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    free_value_mappings();
    reset_precondition_map();
    regions_end();

    reset_current_module_entity();
    set_current_module_entity(callee);    

    pips_debug(4,"end\n");

}


/* generic function (i.e. used for IN and OUT regions) for constructing
 * the list of alias pairs for this module
 * parameters: module name and list of regions
 * global vars IN: none
 * modifies global vars: callee, list_regions_callee, current_caller_stmt
 * and list_pairs
 */
static list
alias_pairs( string module_name, list l_reg )
{

    callees callers;

    pips_debug(4,"begin for module %s\n",module_name);

    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    callee = get_current_module_entity();
    list_regions_callee = l_reg;

    ifdebug(9)
	{
	    /* ATTENTION: we have to do ALL this
	     * just to call print_inout_regions for debug !!
	     */
	    set_current_module_statement( (statement)
					  db_get_memory_resource(DBR_CODE,
								 module_name,
								 TRUE) );
	    set_cumulated_rw_effects((statement_effects)
				     db_get_memory_resource(
					 DBR_CUMULATED_EFFECTS,
					 module_name,
					 TRUE));
	    module_to_value_mappings(callee);
	    /* that's it, but we musn't forget to reset everything below */

	    pips_debug(9,"list_regions_callee is: \n");
	    print_inout_regions(list_regions_callee);

	    free_value_mappings();
	    reset_cumulated_rw_effects();
	    reset_current_module_statement();
	}

    /* we need the callers of the current module  */
    callers = (callees) db_get_memory_resource(DBR_CALLERS,
					       module_name,
					       TRUE);

    /* we scan the callers to find the call sites,
     * and fill in the list of alias pairs (list_pairs)
     */
    list_pairs = NIL;
    MAP(STRING, caller_name,
    {
	entity caller = local_name_to_top_level_entity(caller_name);
	add_alias_pairs_for_this_caller(caller);
    },
	callees_callees(callers));

    reset_current_module_entity();

    pips_debug(4,"end\n");

    return list_pairs;
}


/* top-level creation of pairs of aliases of IN regions of the module
 * modifies global vars callee, list_regions_callee, list_pairs and
 * current_caller_stmt
 */
bool 
in_alias_pairs( string module_name )
{
    list l_reg, l_pairs;

    debug_on("ALIAS_PAIRS_DEBUG_LEVEL");
    pips_debug(4,"begin for module %s\n",module_name);

    /* we need the IN summary regions*/
    l_reg = effects_to_list((effects)
			    db_get_memory_resource(DBR_IN_SUMMARY_REGIONS,
					  module_name,
					  TRUE));

/* was (but didn't work)
    l_reg = (list) db_get_memory_resource(DBR_IN_SUMMARY_REGIONS,
					  module_name,
					  TRUE);
					  */
    
    l_pairs = alias_pairs(module_name, l_reg);

    DB_PUT_MEMORY_RESOURCE(DBR_IN_ALIAS_PAIRS, 
			   strdup(module_name),
			   (char*) make_effects_classes(l_pairs));

    pips_debug(4,"end\n");
    debug_off();

    return(TRUE);

}

/* top-level creation of pairs of aliases of OUT regions of the module
 * modifies global vars callee, list_regions_callee, list_pairs and
 * current_caller_stmt
 */
bool 
out_alias_pairs( string module_name )
{
    list l_reg, l_pairs;

    debug_on("ALIAS_PAIRS_DEBUG_LEVEL");
    pips_debug(4,"begin for module %s\n",module_name);

    /* we need the OUT summary regions*/
    l_reg = effects_to_list((effects)
			    db_get_memory_resource(DBR_OUT_SUMMARY_REGIONS,
					  module_name,
					  TRUE));

/* was (but didn't work)
    l_reg = (list) db_get_memory_resource(DBR_OUT_SUMMARY_REGIONS,
					  module_name,
					  TRUE);
					  */

    
    l_pairs = alias_pairs(module_name, l_reg);

    DB_PUT_MEMORY_RESOURCE(DBR_OUT_ALIAS_PAIRS, 
			   strdup(module_name),
			   (char*) make_effects_classes(l_pairs));

    pips_debug(4,"end\n");
    debug_off();

    return(TRUE);

}




