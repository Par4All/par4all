/* package regions :  Alexis Platonoff, 22 Aout 1990, Be'atrice Creusillet 10/94
 *
 * interprocedural
 * ---------------
 *
 * This File contains the main functions that compute the interprocedural
 * translation of regions (forward and backward).
 *
 * Vocabulary : _ A variable refered as a "region" is in fact of the NEWGEN
 *                type "effect". The use of the word "region" allows to keep
 *                the difference with the effects package.
 *              _ The word "func" always refers to the external called 
 *                subroutine.
 *              _ The word "real" always refers to the calling subroutine 
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
#include "semantics.h"
#include "text.h"

#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"

#include "sc.h"
#include "polyedre.h"

#include "transformer.h"

#include "pipsdbm.h"
#include "resources.h"

#include "effects-generic.h"
#include "effects-convex.h"

#define BACKWARD TRUE
#define FORWARD FALSE

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))



/* jmp_buf overflow_error;*/

/************************************************************** INTERFACES  */

static statement current_stmt = statement_undefined;
static entity current_callee = entity_undefined;
static list l_sum_out_reg = list_undefined;

void reset_out_summary_regions_list()
{
    l_sum_out_reg = list_undefined;
}

void update_out_summary_regions_list(list l_out)
{
    if (list_undefined_p(l_sum_out_reg))
	l_sum_out_reg = l_out;
    else
	l_sum_out_reg = RegionsMayUnion(l_sum_out_reg, l_out,
					regions_same_action_p);
}

list get_out_summary_regions_list()
{
    return(l_sum_out_reg);
}

static bool stmt_filter(s)
statement s;
{
    pips_debug(1, "statement %03d\n", statement_number(s));
    
    current_stmt = s;
    return(TRUE);
}


list out_regions_from_caller_to_callee(entity caller, entity callee)
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

    set_out_effects( (statement_effects) 
	db_get_memory_resource(DBR_OUT_REGIONS, caller_name, TRUE) );

    caller_statement = (statement) 
	db_get_memory_resource (DBR_CODE, caller_name, TRUE);

    current_callee = callee;
    gen_multi_recurse(caller_statement,
		      statement_domain, stmt_filter, gen_null,
		      call_domain, out_regions_from_call_site_to_callee, gen_null,
		      NULL);
  
    reset_current_module_entity();
    set_current_module_entity(callee);

    free_value_mappings();

    reset_cumulated_rw_effects();
    reset_precondition_map();
    reset_out_effects();
    reset_current_module_statement();
    free_value_mappings();
    pips_debug(2, "end\n");
    return(l_sum_out_reg);
}

/* void out_regions_from_call_site_to_callee(call c)
 * input    : a potential call site for current_callee.
 * output   : nothing
 * modifies : l_sum_out_reg becomes the may union of l_sum_out_reg and 
 *            the translated out regions of the current call site. 
 * comment  :
 */
void out_regions_from_call_site_to_callee(call c)
{
    transformer context;
    list l_out = NIL, l_tmp = NIL;

    if (call_function(c) != current_callee)
	return;

    context= load_statement_precondition(current_stmt);
    l_out = load_statement_out_regions(current_stmt);
    
    l_tmp = regions_forward_translation(current_callee, call_arguments(c), l_out,
					context);
    update_out_summary_regions_list(l_tmp);
}


/* list in_regions_of_external(entity func, list real_args, transformer context) 
 * input    : an external function func, and the list of real arguments used 
 *            in the calling function.
 * output   : the corresponding list of regions, at call site.
 * modifies : nothing.
 * comment  : The effects of "func" are computed into externals effects, 
 *            ie. `translated'. The translation is made in two phases :
 *                   _ regions on formal parameters
 *                   _ regions on common parameters
 */
list in_regions_of_external(func, real_args, context)
entity func;
list real_args;
transformer context;
{
    list le = NIL;
    char *func_name = module_local_name(func);

    pips_debug(4, "translation regions for %s\n", func_name);

    if (! entity_module_p(func)) 
    {
	pips_error("in_region_of_external", "%s: bad function\n", func_name);
    }
    else 
    {
	list func_regions;

        /* Get the regions of "func". */
	func_regions = effects_to_list((effects)
	    db_get_memory_resource(DBR_IN_SUMMARY_REGIONS, func_name, TRUE));
	/* translate them */
	le = regions_backward_translation(func, real_args, func_regions, context,
					  SUMMARY);
    }
    return le;
}


/* list regions_of_external(entity func, list real_args, transformer context) 
 * input    : an external function func, and the list of real arguments used 
 *            in the calling function.
 * output   : the corresponding list of regions, at call site.
 * modifies : nothing.
 * comment  : The effects of "func" are computed into externals effects, 
 *            ie. `translated'. The translation is made in two phases :
 *                   _ regions on formal parameters
 *                   _ regions on common parameters
 */
list regions_of_external(entity func,list real_args,transformer context,
			 bool proper)
{
    list le = NIL;
    char *func_name = module_local_name(func);

    pips_debug(4, "translation regions for %s\n", func_name);

    if (! entity_module_p(func)) 
    {
	pips_error("region_of_external", "%s: bad function\n", func_name);
    }
    else 
    {
	list func_regions;

        /* Get the regions of "func". */
	func_regions = effects_to_list((effects)
	    db_get_memory_resource(DBR_SUMMARY_REGIONS, func_name, TRUE));
	/* translate them */
	le = regions_backward_translation(func, real_args, func_regions, context,
					  proper);
    }
    return le;
}

list /* of effects */
convex_regions_backward_translation(entity func, list real_args,
				    list l_reg, transformer context)
{
    list l_res = NIL;
    
    l_res = regions_backward_translation(func, real_args, l_reg, context, TRUE);
    
    return l_res;
}

list /* of effects */
convex_regions_forward_translation(entity callee, list real_args,
				    list l_reg, transformer context)
{
    list l_res = NIL;
    
    l_res = regions_forward_translation(callee, real_args, l_reg, context);
    
    return l_res;
}


/***************************************************** BACKWARD TRANSLATION */

static list formal_regions_backward_translation(entity func, list real_args, 
						list func_regions, 
						transformer context);
static list common_regions_backward_translation(entity func, list func_regions);
static list common_region_translation(entity func, region reg, boolean backward);

/* list regions_backward_tranlation(entity func, list real_args, 
 *                                  list func_regions, transformer context) 
 * input    : an external function func, and the list of real arguments used 
 *            in the calling function.
 * output   : the corresponding list of regions, at call site.
 * modifies : nothing.
 * comment  : The effects of "func" are computed into externals effects, 
 *            ie. `translated'. The translation is made in two phases :
 *                   _ regions on formal parameters
 *                   _ regions on common parameters
 */
list regions_backward_translation(entity func, list real_args,
				  list func_regions,
				  transformer context, bool proper)
{
    list le = NIL;
    list tce, tfe;

    ifdebug(4)
    {
	pips_debug(4,"Initial regions\n");
	print_regions(func_regions);
    }

    set_interprocedural_translation_context_sc(func,real_args);
    set_backward_arguments_to_eliminate(func);

    /* Compute the regions on formal variables. */
    tfe = formal_regions_backward_translation(func,real_args,func_regions,context);
    
    /* Compute the regions on common variables (static & global variables). */
    tce = common_regions_backward_translation(func, func_regions);
    
    if (proper)
	le = gen_nconc(tce,tfe);
    else
	le = RegionsMustUnion(tce, tfe, regions_same_action_p);		
    
    /* FI: add local precondition (7 December 1992) */
    le = regions_add_context(le, context);
    
    ifdebug(4)
    {
	pips_debug(4, " Translated_regions :\n");
	print_regions(le);
    }
    
    reset_translation_context_sc();
    reset_arguments_to_eliminate();
    return(le);
}


/* static list formal_regions_backward_translation(entity func, list real_args,
 *                                   func_regions, transformer context)
 * input    : an external function func, its real arguments at call site 
 *            (real_args),
 *            its summary regions (with formal args), and the calling context. 
 * output   : the translated formal regions.
 * modifies : ?
 * comment  :	
 *
 * Algorithm :
 * -----------
 *    let func_regions be the list of the regions on variables of func
 *    let real_regions be the list of the translated regions on common variables
 *
 *    real_regions = empty
 *    FOR each expression real_exp IN real_args
 *        arg_num = number in the list of the function real arguments
 *        FOR each func_reg IN func_regions
 *            func_ent = entity of the region func_reg
 *            IF func_ent is the formal parameter numbered arg_num
 *                IF real_exp is an lhs (expression with one entity)
 *                    real_reg = translation of the region func_reg
 *                    real_regions = (real_regions) U {real_reg}
 *                ELSE
 *                    real_regions = (real_regions) U
 *                                   (regions of the expression real_exp)
 *                ENDIF
 *            ENDIF
 *        ENDFOR
 *    ENDFOR
 */
static list formal_regions_backward_translation(func, real_args, func_regions, 
						context)
entity func;
list real_args, func_regions;
transformer context;
{
    list real_regions = NIL, r_args;
    int arg_num;

    pips_debug(8, "begin\n");

    for (r_args = real_args, arg_num = 1; r_args != NIL;
	 r_args = CDR(r_args), arg_num++) 
    {
	MAP(EFFECT, func_reg,
	 {
	     entity func_ent = region_entity(func_reg);
	     
	     /* If the formal parameter corresponds to the real argument then
	      * we perform the translation.
	      */
	     if (ith_parameter_p(func, func_ent, arg_num))
	     {
		 expression real_exp = EXPRESSION(CAR(r_args));
		 syntax real_syn = expression_syntax(real_exp);
		 
		 /* If the real argument is a reference to an entity, then we
		  * translate the regions of the corresponding formal parameter
		  */
		 if (syntax_reference_p(real_syn)) 
		 {
		    reference real_ref = syntax_reference(real_syn);
		    list real_inds = reference_indices(real_ref);
		    entity real_ent = reference_variable(real_ref);
		    region real_reg;
		    real_reg =
			region_translation(func_reg, func, reference_undefined,
				  real_ent, get_current_module_entity(), real_ref,
				  VALUE_ZERO, BACKWARD);
		    
		    real_regions = regions_add_region(real_regions, real_reg);
		    /* The indices of the reference are always evaluated */
		    if (! ENDP(real_inds)) 
			real_regions = gen_nconc
			    (real_regions,
			     proper_regions_of_expressions(real_inds, context));
		}
		 /* Else, the real argument is a complex expression, which
		  * is merely evaluated during execution of the program; 
		  * Since Fortran forbids write effects on expressions
		  * passed as arguments, the regions on the formal parameter 
		  * are merely ignored. The regions computed are those of the
		  * real parameter expression. 
		  */
		 else 
		 {
		    real_regions =
			gen_nconc(real_regions,
				  generic_proper_effects_of_expression(real_exp)); 
		}
	     }
	 }, func_regions);
    }

    ifdebug(5) 
    {
       pips_debug(5, "proper real regions\n");
       print_regions(real_regions);
    }
    return(real_regions);
}

static list common_regions_backward_translation(entity func, list func_regions) 
{
    list real_regions = NIL;

    MAP(EFFECT, func_reg, 
    {
	/* we are only interested in regions concerning common variables. 
	 * They are  the entities with a ram storage. They can not be dynamic
         * variables, because these latter were eliminated of the code_regions
         * (cf. region_of_module). */
	if (storage_ram_p(entity_storage(region_entity(func_reg))))
	{
	    list regs = common_region_translation(func, func_reg, BACKWARD);
	    real_regions = RegionsMustUnion(real_regions, regs, 
					    regions_same_action_p);	
	}
    },
	func_regions);

    return(real_regions);

}

/****************************************************** FORWARD TRANSLATION */


static list real_regions_forward_translation(entity func, list real_args, 
					     list l_reg, transformer context);
static list common_regions_forward_translation(entity func, list real_regions);


/* list regions_forward_translation(entity func, list real_args, l_reg, 
 *                                  transformer context
 * input    : the called function func, the real arguments of the call,
 *            the list of regions to translate, and the context of the call.
 * output   : the translated list of regions : real arguments are translated 
 *            into formal arguments, and common variables of the caller into 
 *            common variables of the callee. 
 * modifies : nothing.
 * comment  :	
 */
list regions_forward_translation(func, real_args, l_reg, context)
entity func;
list real_args, l_reg;
transformer context;
{
    list l_t_reg = NIL;
    list l_form_reg, l_common_reg;

    ifdebug(3)
    {
	pips_debug(3,"initial regions :\n");
	print_regions(l_reg);
    }

    set_interprocedural_translation_context_sc(func,real_args);
    set_forward_arguments_to_eliminate();

    l_form_reg = real_regions_forward_translation
	(func, real_args, l_reg, context);
    l_common_reg = common_regions_forward_translation(func, l_reg);
    l_t_reg = RegionsMustUnion
	(l_form_reg, l_common_reg, regions_same_action_p);

    ifdebug(3)
    {
	pips_debug(3,"final regions : \n");
	print_regions(l_t_reg);
    }

    reset_translation_context_sc();
    reset_arguments_to_eliminate();
    return l_t_reg;
}


/* static list real_regions_forward_translation(entity func, list real_args, l_reg, 
 *                                              transformer context)
 * input    : the called function func, the real arguments of the call,
 *            the list of regions to translate, and the context of the call.
 * output   : the list of translated regions correponding to the formal arguments
 *            of the called function.
 * modifies : l_reg and the regions it contains.
 * comment  : 
 *            for each real argument in real_args
 *              if it is a reference
 *                for each region in l_reg
 *                  if the current region concerns the current real argument
 *                    if the corresponding formal parameter is a scalar
 *                      the translated region is a scalar region, which
 *                      reference is the formal argument, and which 
 *                      action and approximation are those of the initial
 *                      region.
 *                    else it is an array, 
 *                      and the tranlation is performed by
 *                      another procedure.
 *                endfor
 *              else, it is a complex expression
 *                we search the regions in l_reg corresponding to
 *                the elements accessed in the complex expression.
 *                and we make a read region corresponding to the 
 *                formal scalar parameter.
 *              endif
 *            endfor
 *               
 */
static list real_regions_forward_translation(func, real_args, l_reg, context)
entity func;
list real_args, l_reg;
transformer context;
{
    entity caller = get_current_module_entity();
    int arg_num;
    list l_formal = NIL;
    list r_args = real_args;
    list l_sum_rw_reg = 
	effects_to_list((effects) db_get_memory_resource
			(DBR_SUMMARY_REGIONS,
			 module_local_name(func),
			 TRUE));
    
    /* for each actual parameter expression, we search in the actual regions 
     * the corresponding elements. If it exists, we make the corresponding 
     * regions, and translate them */

    ifdebug(8) 
    {
	pips_debug(8,"initial regions :\n");
	print_regions(l_reg);
    }

    for (arg_num = 1; !ENDP(r_args); r_args = CDR(r_args), arg_num++) 
    {
	expression real_exp = EXPRESSION(CAR(r_args));
	entity formal_ent = find_ith_formal_parameter(func, arg_num);

	if (syntax_reference_p(expression_syntax(real_exp)))
	{
	    reference real_ref = syntax_reference(expression_syntax(real_exp));
	    entity real_ent = reference_variable(real_ref);

	    MAP(EFFECT, reg,
	     {
		 entity reg_ent = region_entity(reg);

		 pips_debug(8, " real = %s, formal = %s \n", 
			    entity_name(real_ent), entity_name(reg_ent));
		 
		 if (same_entity_p(reg_ent , real_ent)) 
		 {
		     region formal_reg;
		     formal_reg = region_translation(
			 reg, caller, real_ref,
			 formal_ent, func, reference_undefined, 
			 VALUE_ZERO, FORWARD);		     
		     l_formal = RegionsMustUnion(
			 l_formal, 
			 CONS(EFFECT, formal_reg, NIL), 
			 regions_same_action_p);
		 }
	     }, l_reg);
	    
	} /* if */
	else 
	{
	    /* REVOIR ICI */
	    list l_exp_reg = regions_of_expression(real_exp, context);
	    list l_real_exp = 
		RegionsIntersection(l_exp_reg, regions_dup(l_reg),
				    regions_same_action_p);

	    pips_debug(8, "real argument is a complex expression \n" 
		"\tit can not correspond to a written formal parameter.\n");
	    
	    if (!ENDP(l_real_exp)) 
	    {
		region  formal_reg = 
		    reference_whole_region(make_regions_reference(formal_ent),
				     is_action_read);
		l_formal = RegionsMustUnion(l_formal, 
					    CONS(EFFECT, formal_reg, NIL), 
					    regions_same_action_p);
		regions_free(l_real_exp);
	    }
	    
	} /* else */	
	
    } /* for */
    
    /* il faut calculer l'intersection avec les summary regions de la 
     * fonction pour e'viter certains proble`mes comme avec:
     *
     *      <A(PHI1)-OUT-MUST-{PHI1==I}
     *      CALL TOTO(A(I), A(I))
     *     
     *      <I-R-MUST-{}>, <J-W-MUST-{}
     *      SUBROUTINE TOTO(I,J)
     *
     * si on ne fait pas attention, on obtient <I-OUT-MUST-{}>, <J-OUT-MUST>
     * ve'rifier que c'est compatible avec la norme. Inutile de faire des 
     * choses inutiles. 
     */
    l_formal = RegionsIntersection(l_formal, regions_dup(l_sum_rw_reg),
				   regions_same_action_p);

    return(l_formal);
}


/* static list common_regions_forward_translation
 *                              (entity func, list real_regions)
 * input    : the called function, the list of real arguments at call site, and
 *            the list of regions to translate.
 * output   : the translated list of regions.
 * modifies : nothing.
 * comment  : 
 */
static list common_regions_forward_translation(entity func, list real_regions)
{
    list func_regions = NIL;

    MAP(EFFECT, real_reg,
    {
	storage real_s = entity_storage(region_entity(real_reg));
	/* we are only interested in regions concerning common variables. 
	 * They are  the entities with a ram storagethat are not dynamic
         * variables*/
	if (storage_ram_p(real_s) && 
	    !dynamic_area_p(ram_section(storage_ram(real_s))))
	{
	    list regs = common_region_translation(func, real_reg, FORWARD);
	    func_regions = RegionsMustUnion(func_regions, regs, 
					    regions_same_action_p);	
	}
    },
	real_regions);

    return(func_regions);
}

/********************************************************* COMMON FUNCTIONS */


/* static list common_region_translation(entity func, region reg, 
 *                                       boolean backward)
 * input    : func is the called function, real_args are the real arguments,
 *            reg is the region to translate (it concerns an array in a common), 
 *            and backward indicates the direction of the translation.
 * output   : a list of regions, that are the translation of the initial region.
 * modifies : nothing: duplicates the original region.
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
static list common_region_translation(entity callee, region reg,
				      boolean backward)
{
    list new_regions = NIL;
    entity reg_ent = region_entity(reg);
    entity caller = get_current_module_entity();
    entity source_func = backward ? callee : caller;
    entity target_func = backward ? caller : callee;
    entity entity_target_func = target_func;
    entity ccommon;
    list l_tmp, l_com_ent;
    int reg_ent_size, total_size, reg_ent_begin_offset, reg_ent_end_offset;
    region new_reg;
    boolean found = FALSE;
    

    ifdebug(5)
    {
	pips_debug(5,"input region: \n%s\n", region_to_string(reg));
    }

    /* If the entity is a top-level entity, no translation;
     * It is the case for variables dexcribing I/O effects (LUNS).
     */

    if (top_level_entity_p(reg_ent))
    {
	pips_debug(5,"top-level entity.\n");	
	new_reg = region_translation
		    (reg, source_func, reference_undefined,
		     reg_ent, target_func, reference_undefined,
		     0, backward);
	new_regions = CONS(EFFECT, new_reg, NIL);
	return(new_regions);
    }

    if (io_entity_p(reg_ent)) {
	pips_debug(5,"top-level entity.\n");	
	new_reg = region_translation
		    (reg, source_func, reference_undefined,
		     reg_ent, target_func, reference_undefined,
		     0, backward);
	new_regions = CONS(EFFECT, new_reg, NIL);
	return(new_regions);
    }

    ifdebug(6)
    {
	pips_debug(5, "target function: %s (local name: %s)\n",
		   entity_name(target_func), module_local_name(target_func));
    }

    /* First, we search if the common is declared in the target function;
     * if not, we have to deterministically choose an arbitrary function
     * in which the common is declared. It will be our reference.
     * By deterministically, I mean that this function shall be chosen whenever
     * we try to translate from this common to a routine where it is not 
     * declared.
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
	    pips_debug(6, "common not declared in caller,\n"
		       "\t using %s declarations instead\n", 
		       entity_name(entity_target_func));
	}
    }

    /* first, we calculate the offset and size of the region entity */
    reg_ent_size = array_size(reg_ent);
    reg_ent_begin_offset = ram_offset(storage_ram(entity_storage(reg_ent)));
    reg_ent_end_offset = reg_ent_begin_offset + reg_ent_size - 1;

    pips_debug(6,
	       "\n\treg_ent: size = %d, offset_begin = %d, offset_end = %d\n",
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
	    int new_ent_size = array_size(new_ent);
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
		     (Value) offset, backward);
		new_regions = RegionsMustUnion(new_regions,
					       CONS(EFFECT, new_reg, NIL), 
					       regions_same_action_p);
		total_size += min (reg_ent_begin_offset,new_ent_end_offset) 
		    - max(reg_ent_begin_offset, new_ent_begin_offset) + 1;
	    }
	}
    }
	
    ifdebug(5)
    {
	pips_debug(5, "output regions: \n");
	print_regions(new_regions);
    }
    return(new_regions);
}






