
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

#include "effects.h"
#include "regions.h"
#include "semantics.h"

#include "pipsdbm.h"
#include "resources.h"

#define BACKWARD TRUE
#define FORWARD FALSE


static entity callee;
static list list_regions_callee;
static statement current_caller_stmt = statement_undefined;
static list list_pairs;

static bool stmt_filter(s)
statement s;
{
    pips_debug(1, "statement %03d\n", statement_number(s));
    
    current_caller_stmt = s;
    return(TRUE);
}

/* static void add_common_aliases_for_this_region(entity func, region reg)
 * input    : func is the called function and
 *            reg is the region to translate (from an array in a common).
 * output   : void
 * modifies : this function performs
 *            the backward translation: reg -> new_regions
 *            and then for each new_reg in the new_regions list, it
 *            performs the forward translation new_reg -> old_reg
 *            and adds an alias pair <old_reg,new_reg>
 *            to list_pairs
 *            (if new_regions has only one element then old_reg = reg
 *            and only one alias pair is created, otherwise
 *            reg is the original callee region, new_reg is a caller
 *            region corresponding to part of reg and old_reg
 *            is the part of reg corresponding to new_reg
 * comment  : the algorithm is the following
 * 
 * Scan the variables of the common that belong to the target function
 * For each variable do
 *     if it has elements in common with the variable of the initial region
 *        if both variables have the same layout in the common
 *           perform the translation using array_region-translation
 *        else 
 *           use the subscript values, and take into account the relative
 *           offset of the variables in the common
 *           add to the translated region the declaration system of the 
 *           target variable to have a smaller region.
 * until all the elements of the initial variable have been translated.
 */

/*****************************************************************************

static list common_region_alias_formation(entity callee, region reg)
{
    list new_regions = NIL;
    entity reg_ent = region_entity(reg);

/* PROBLEME? get_current_module_entity returns CALLEE not CALLER? */
    entity caller = get_current_module_entity();
    entity source_func = callee;
    entity target_func = caller;
    entity entity_target_func = target_func;
    entity ccommon;
    list l_tmp, l_com_ent;
    int reg_ent_size, total_size, reg_ent_begin_offset, reg_ent_end_offset;
    region new_reg, old_reg;
    boolean found = FALSE;
    
    ifdebug(5)
    {
	pips_debug(5,"input region: \n%s\n", region_to_string(reg));
    }

    /* If the entity is a top-level entity, no translation;
     * It is the case for variables dexcribing I/O effects (LUNS).
*** CHECK WHAT TO DO IN THE CASE OF ALIASES ***
     */

    if (top_level_entity_p(reg_ent))
    {
	pips_debug(5,"top-level entity.\n");	
	new_reg = region_translation
		    (reg, source_func, reference_undefined,
		     reg_ent, target_func, reference_undefined,
		     0, BACKWARD);
	new_regions = CONS(EFFECT, new_reg, NIL);
	return(new_regions);
    }

    if (io_entity_p(reg_ent)) {
	pips_debug(5,"top-level entity.\n");	
	new_reg = region_translation
		    (reg, source_func, reference_undefined,
		     reg_ent, target_func, reference_undefined,
		     0, BACKWARD);
	new_regions = CONS(EFFECT, new_reg, NIL);
	return(new_regions);
    }

    ifdebug(6)
    {
	pips_debug(5, "target function: %s (local name: %s)\n",
		   entity_name(target_func), module_local_name(target_func));
    }

    /* First, we search if the common is declared in the target function;
     * if not,
*** WHAT DO WE DO IN THE CASE OF ALIASES ? ***
     * we have to deterministically choose an arbitrary function
     * in which the common is declared. It will be our reference.
     * By deterministically, I mean that this function shall be chosen whenever
     * we try to translate from this common to a routine where it is not declared.
     */
    ccommon = ram_section(storage_ram(entity_storage(reg_ent)));
    l_com_ent = area_layout(type_area(entity_type(ccommon)));

    pips_debug(6, "common name: %s\n", entity_name(ccommon));

    for( l_tmp = l_com_ent; !ENDP(l_tmp) && !found; l_tmp = CDR(l_tmp) )
    {
	entity com_ent = ENTITY(CAR(l_tmp));
	if (strcmp(entity_module_name(com_ent),
		   module_local_name(target_func)) == 0)
	{
	    found = TRUE;
	}
    }
    
    /* If common not declared in caller, use the subroutine of the first entity
     * that appears in the common layout. (not really deterministic: I should
     * take the first name in lexical order. BC.
     */
    if(!found)
    {
	entity ent = ENTITY(CAR(l_com_ent));	
	entity_target_func =
	    local_name_to_top_level_entity(module_name(entity_name(ent)));
	ifdebug(6)
	{
	    pips_debug(6, "common not declared in caller,\n\t using %s declarations "
		       "instead\n", entity_name(entity_target_func));
	}
    }

    /* first, we calculate the offset and size of the region entity */
    reg_ent_size = SizeOfArray(reg_ent);
    reg_ent_begin_offset = ram_offset(storage_ram(entity_storage(reg_ent)));
    reg_ent_end_offset = reg_ent_begin_offset + reg_ent_size - 1;

    pips_debug(6, "\n\t reg_ent: size = %d, offset_begin = %d, offset_end = %d \n",
	      reg_ent_size, reg_ent_begin_offset, reg_ent_end_offset); 

    /* then, we perform the translation */
    ccommon = ram_section(storage_ram(entity_storage(reg_ent)));
    l_com_ent = area_layout(type_area(entity_type(ccommon)));
    total_size = 0;

    for(; !ENDP(l_com_ent) && (total_size < reg_ent_size); 
	l_com_ent = CDR(l_com_ent)) 
    {
	entity new_ent = ENTITY(CAR(l_com_ent));
	
	pips_debug(6, "current entity: %s\n", entity_name(new_ent)); 

	if (strcmp(module_name(entity_name(new_ent)),
		   module_local_name(entity_target_func)) == 0)
	{
	    int new_ent_size = SizeOfArray(new_ent);
	    int new_ent_begin_offset = 
		ram_offset(storage_ram(entity_storage(new_ent)));
	    int new_ent_end_offset = new_ent_begin_offset + new_ent_size - 1;

	    pips_debug(6, "\n\t new_ent: size = %d, "
		       "offset_begin = %d, offset_end = %d \n",
		       new_ent_size, new_ent_begin_offset, new_ent_end_offset); 
	    
	    if ((new_ent_begin_offset <= reg_ent_end_offset) && 
		(reg_ent_begin_offset <= new_ent_end_offset ))
		/* these entities have elements in common */
	    {
		int offset = reg_ent_begin_offset - new_ent_begin_offset;
		
		new_reg = region_translation
		    (reg, source_func, reference_undefined,
		     new_ent, target_func, reference_undefined,
		     (Value) offset, BACKWARD);

		/* save context */
		(entity) current_module = get_current_module_entity();
		(list) l_args = get_arguments_to_eliminate();
		reset_current_module_entity();
		set_current_module_entity(caller);
		set_forward_arguments_to_eliminate();

		old_reg = region_translation
		    (new_reg,
		     target_func, reference_undefined, reg_ent,
		     source_func, reference_undefined, (Value) -offset,
		     FORWARD);

		/* restore context */
		reset_current_module_entity();
		set_current_module_entity(current_module);
		set_arguments_to_eliminate(l_args);

		/* build alias pair and add to list_pairs */
		list pair;

		pair = CONS(EFFECT,region_dup(old_reg),NIL);
		pair = gen_nconc(pair,CONS(EFFECT,new_reg,NIL));
		list_pairs = gen_nconc(list_pairs,CONS(LIST,pair,NIL));

    ifdebug(5)
    {
	pips_debug(5, "alias: \n");
	print_regions(pair);
    }

		/* calculate total_size */
		total_size += min (reg_ent_begin_offset,new_ent_end_offset) 
		    - max(reg_ent_begin_offset, new_ent_begin_offset) + 1;
	    }
	}
    }
	
}
*****************************************************************************/



/* static void
 * add_common_aliases_for_this_call_site()
 * input    : global variables: callee,list_regions_callee,list_pairs
 * output   : void
 * modifies : for each region in list_regions_callee which is a region of a
 *            COMMON (declared in the callee), this function performs
 *            the backward translation: callee_region -> real_regions
 *            and then for each real_reg in the real_regions list, it
 *            performs the forward translation real_reg -> part_callee_region
 *            and adds an alias pair <part_callee_region,real_reg>
 *            to list_pairs
 */

/*****************************************************************************

static void
add_common_aliases_for_this_call_site()
{
    list real_regions = NIL;

    MAP(EFFECT, callee_region, 
    {
	/* we are only interested in regions concerning common variables. 
	 * They are  the entities with a ram storage. They can not be dynamic
         * variables, because these latter were eliminated of the code_regions
         * (cf. region_of_module). */
	if (storage_ram_p(entity_storage(region_entity(callee_region))))
	    add_common_aliases_for_this_region(callee,callee_region);
    },
	list_regions_callee);
}
*****************************************************************************/


/* static void
 * add_parameter_aliases_for_this_call_site(call call_site,transformer context)
 * input    : parameters: a call site and the calling context
 *            global variables: callee,list_regions_callee,list_pairs
 * output   : void
 * modifies : for each region in list_regions_callee which is a region of a
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
add_parameter_aliases_for_this_call_site(call call_site, transformer context)
{
    list r_args;
    int arg_num;
    list real_args;

    pips_debug(8, "begin\n");

    real_args = call_arguments(call_site);

    for (r_args = real_args, arg_num = 1; r_args != NIL;
	 r_args = CDR(r_args), arg_num++) 
    {
	MAP(EFFECT, callee_region,
	 {
	     entity callee_ent = region_entity(callee_region);
	     
	     /* If the formal parameter corresponds to the real argument then
	      * we perform the translation.
	      */
	     if (ith_parameter_p(callee, callee_ent, arg_num))
	     {
		 expression real_exp = EXPRESSION(CAR(r_args));
		 syntax real_syn = expression_syntax(real_exp);
		 
		 /* If the real argument is a reference to an entity, then we
		  * translate the regions of the corresponding formal parameter
		  */
		 if (syntax_reference_p(real_syn)) 
		 {
		    reference real_ref = syntax_reference(real_syn);
		    entity real_ent = reference_variable(real_ref);
		    region real_reg;
		    list pair;

		    real_reg =
			region_translation(callee_region, callee,
					   reference_undefined,
					   real_ent, get_current_module_entity(),
					   real_ref,
					   VALUE_ZERO, BACKWARD);
		    
		    pair = CONS(EFFECT,region_dup(callee_region),NIL);
		    pair = gen_nconc(pair,CONS(EFFECT,real_reg,NIL));
		    list_pairs = gen_nconc(list_pairs,CONS(LIST,pair,NIL));
		}
	     }
	 }, list_regions_callee);
    }

    pips_debug(8, "end\n");
       
}


static bool
call_site_to_alias_pairs(call call_site)
{
    transformer context;
    list real_args;


    if (call_function(call_site) != callee) return TRUE;

    context = load_statement_precondition(current_caller_stmt);
    real_args = call_arguments(call_site);

    set_interprocedural_translation_context_sc(callee, real_args);
    set_backward_arguments_to_eliminate(callee);

    add_parameter_aliases_for_this_call_site(call_site,context);
/*    add_common_aliases_for_this_call_site(); */

    reset_translation_context_sc();
    reset_arguments_to_eliminate();

    return TRUE;
}

static void
caller_to_alias_pairs( entity caller, entity callee)
{
    char *caller_name;
    statement caller_statement;

    reset_current_module_entity();
    set_current_module_entity(caller);
    caller_name = module_local_name(caller);
    pips_debug(2, "begin for caller: %s\n", caller_name);
    
    /* All we need to perform the translation */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, caller_name, TRUE) );
    set_cumulated_rw_effects((statement_effects)
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, caller_name, TRUE));
    module_to_value_mappings(caller);
    set_precondition_map( (statement_mapping) 
        db_get_memory_resource(DBR_PRECONDITIONS, caller_name, TRUE));

    caller_statement = get_current_module_statement();

    gen_multi_recurse(caller_statement,
		      statement_domain, stmt_filter, gen_null,
		      call_domain, call_site_to_alias_pairs, gen_null,
		      NULL);
  
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_precondition_map();

    reset_current_module_entity();
    set_current_module_entity(callee);    


}

static list
alias_pairs( string module_name, list l_reg )
{

    callees callers;
    entity module;

    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();
    callee = module;
    list_regions_callee = l_reg;

    /* we need the callers of the current module  */
    callers = (callees) db_get_memory_resource(DBR_CALLERS,
					       module_name,
					       TRUE);

    /* we scan the callers to find the call sites, and fill in the list of alias
     * pairs (list_pairs). 
     */
    list_pairs = NIL;
    MAP(STRING, caller_name,
    {
	entity caller = local_name_to_top_level_entity(caller_name);
	caller_to_alias_pairs(caller, module);
    },
	callees_callees(callers));

    reset_current_module_entity();
    return list_pairs;
}

bool 
in_alias_pairs( string module_name )
{
    list l_reg, l_pairs;

    /* we need the IN summary regions*/
    l_reg = (list) db_get_memory_resource(DBR_IN_SUMMARY_REGIONS,
					  module_name,
					  TRUE);

    
    l_pairs = alias_pairs(module_name, l_reg);

    DB_PUT_MEMORY_RESOURCE(DBR_IN_ALIAS_PAIRS, 
			   strdup(module_name),
			   (char*) make_effects_classes(l_pairs));

    return(TRUE);

}

