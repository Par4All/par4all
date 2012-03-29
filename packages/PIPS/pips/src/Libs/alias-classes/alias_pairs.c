/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#include <stdio.h>
#include <string.h>

#include <setjmp.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "semantics.h"

#include "transformer.h"

#include "pipsdbm.h"
#include "resources.h"

#include "vecteur.h"
/* Instantiation of the dependence graph: */
#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"
#include "ricedg.h"

#define BACKWARD true
#define FORWARD false


static entity callee;
static list list_regions_callee = NIL;
static statement current_caller_stmt = statement_undefined;
static list list_pairs = NIL;


/* creation of a Pbase containing just the PHI variables of the region */
static Pbase
make_base_phi_variables(region reg)
{
    Pbase phi_variables;
    list indices;

    pips_debug(4,"begin\n");
    phi_variables = BASE_NULLE;

    indices = reference_indices(region_any_reference(reg));
    MAP(EXPRESSION, index,
	{
	    entity e;

	    e = reference_variable(syntax_reference(expression_syntax(index)));
	    if (variable_phi_p(e))
	    {
		pips_debug(9,"add: %s\n",entity_local_name(e));

		phi_variables = base_add_variable(phi_variables,(Variable) e);
	    }
	},
	    indices);

    pips_debug(4,"end\n");

    return phi_variables;
}


/* strips from the region all the constraints which do not affect
 * (even transitively) the PHI variables
 * i.e. the function representing the region no longer returns the
 * empty region if a branch condition is not satisfied
 * so the region becomes MAY-in-the-usual-dataflow-sense
 */
static region
restrict_to_phi_constraints(region reg)
{
    Pbase phi_variables;
    Psysteme sc;
    region new_reg;
 
    pips_debug(4,"begin\n");


    sc = region_system(reg);
    phi_variables = make_base_phi_variables(reg);
    new_reg = region_dup(reg);

    ifdebug(9)
	{
	    set_action_interpretation(ACTION_IN,ACTION_OUT);
	    pips_debug(9,"call sc_restricted_to_variables_transitive_closure for:\t\n");
	    print_region(reg);
	}

    region_system(new_reg) =
	sc_restricted_to_variables_transitive_closure(sc,phi_variables);

    ifdebug(9)
	{
	    pips_debug(9,"restricted region:\n\t");
	    print_region(new_reg);
	    reset_action_interpretation();
	}

    pips_debug(4,"end\n");

    return new_reg;
}

/* "convert" EXACT regions to exact representations of
 * MAY-in-the-usual-dataflow-sense regions
 */
static region
convert_exact_to_exact_may(region reg)
{
    region new_reg;

    pips_debug(4,"begin\n");

    new_reg = restrict_to_phi_constraints(reg);
    effect_approximation_tag(new_reg) = is_approximation_exact;

    pips_debug(4,"end\n");

    return(new_reg);
}


/* all MAY regions are "converted" to over-approximate representations of
 * MAY-in-the-usual-dataflow-sense regions (some regions may in fact be
 * precise representations  of MAY-in-the-usual-dataflow-sense regions after
 * this operation, but we cannot detect which)
 */
static region
approx_convert_may_to_approx_may(region reg)
{
    region new_reg;

    pips_debug(4,"begin\n");

    new_reg = restrict_to_phi_constraints(reg);
    effect_approximation_tag(new_reg) = is_approximation_may;

    pips_debug(4,"end\n");

    return(new_reg);
}


/* takes EXACT (i.e. precise representations of MUST in the usual
 * dataflow sense) and MAY (i.e. over-approximate representations of MUST
 * regions or either over-approximate or precise representations of
 * MAY-in-the-usual-dataflow-sense) regions and "converts" them to
 * exact or over-approximate representations of
 * MAY-in-the-usual-dataflow-sense) regions
 * by stripping all the constraints which do not affect
 * (even transitively) the PHI variables
 */
static region
approx_convert(region reg)
{
    region new_reg;

    pips_debug(4,"begin\n");

    if (region_scalar_p(reg))
    {
	pips_debug(9,"scalar\n");

	new_reg = reg;
	effect_approximation_tag(new_reg) = is_approximation_exact;
    }
    else
    {
	if ( effect_exact_p(reg) || effect_exact_p(reg) )
	    new_reg = convert_exact_to_exact_may(reg);
	else
	  {
	    if (!effect_may_p(reg))
	      pips_debug(4,"unknown approximation tag\n");
	    new_reg = approx_convert_may_to_approx_may(reg);
	  }
    }
    pips_debug(4,"end\n");

    return new_reg;
}
	

/* modifies global var current_caller_stmt */
static bool stmt_filter(s)
statement s;
{
    pips_debug(9, "statement %td\n", statement_number(s));
    
    current_caller_stmt = s;
    return(true);
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
add_parameter_aliases_for_this_call_site(call call_site __attribute__ ((unused)),
					 transformer context,
					 list real_args)
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
		    region trans;
		    region formal;
		    region actual;
		    list pair;

		    pips_debug(9,"arg refers to entity\n");
		    pips_debug(9,"\t%s\n",entity_name(real_ent));

		    trans =
			region_translation(
			    callee_region,
			    callee,
			    reference_undefined,
			    real_ent,
			    get_current_module_entity(),
			    real_ref,
			    VALUE_ZERO,
			    BACKWARD);

		    pair = CONS(EFFECT,trans,NIL);

		    ifdebug(9)
			{
			    pips_debug(9,"region translated to:\n\t");
			    print_inout_regions(pair);
			}

		    /* the actual parameter must be expressed relative to
		       the store at the point of entry of the caller, so
		       that it can be compared to other regions */

		    pair =
			convex_regions_transformer_compose(pair,context);

		    ifdebug(9)
			{
			    pips_debug(9,"relative to initial store:\n\t");
			    print_inout_regions(pair);
			}

		    /* convert actual and formal regions to
		       MAY-in-the-usual-dataflow-sense */
		    actual = approx_convert(EFFECT(CAR(pair)));

		    pair = CONS(EFFECT,actual,NIL);

		    ifdebug(9)
			{
			    pips_debug(9,"restricted to:\n\t");
			    print_inout_regions(pair);
			}

/* gave Newgen error
   formal = approx_convert(callee_region); */

		    formal = approx_convert(region_dup(callee_region));

		    pair = CONS(EFFECT,formal,pair);

		    ifdebug(9)
			{
			    pips_debug(9,"alias pair:\n\t");
			    print_inout_regions(pair);
			}

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

    if (call_function(call_site) != callee) return true;

    pips_debug(4,"begin\n");

/*    pips_debug(9,
	       "try load_statement_precondition for statement %03d\n",
	       statement_number(current_caller_stmt));
	       */

    context = load_statement_precondition(current_caller_stmt);

/* transformer_to_string no longer implemented */
/*    pips_debug(9,"got context:\n\t%s\n",transformer_to_string(context)); */

/*    pips_debug(9,"try call_arguments\n"); */

    real_args = call_arguments(call_site);

/*    pips_debug(9,"try set_interprocedural_translation_context_sc\n");
    pips_debug(9,"\tfor callee %s\n",entity_name(callee)); */

    set_interprocedural_translation_context_sc(callee, real_args);

/*    pips_debug(9,"try set_backward_arguments_to_eliminate\n"); */

    set_backward_arguments_to_eliminate(callee);

    add_parameter_aliases_for_this_call_site(call_site,context,real_args);
/*    add_common_aliases_for_this_call_site(); */

    reset_translation_context_sc();
    reset_arguments_to_eliminate();

    pips_debug(4,"end\n");

    return true;
}


/* constructs the alias pairs for this caller and adds them to the list
 * global vars IN: callee, list_regions_callee and list_pairs
 * modifies global vars: list_pairs and current_caller_stmt
 */
static void
add_alias_pairs_for_this_caller( entity caller )
{
    const char *caller_name;
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
	db_get_memory_resource(DBR_CODE, caller_name, true) );
    set_cumulated_rw_effects((statement_effects)
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, caller_name, true));
    module_to_value_mappings(caller);
    set_precondition_map( (statement_mapping) 
        db_get_memory_resource(DBR_PRECONDITIONS, caller_name, true));
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
alias_pairs( const char* module_name, list l_reg )
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
								 true) );
	    set_cumulated_rw_effects((statement_effects)
				     db_get_memory_resource(
					 DBR_CUMULATED_EFFECTS,
					 module_name,
					 true));
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
					       true);

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
in_alias_pairs( const char* module_name )
{
    list l_reg, l_pairs;

    debug_on("ALIAS_PAIRS_DEBUG_LEVEL");
    pips_debug(4,"begin for module %s\n",module_name);

    /* we need the IN summary regions*/
    l_reg = effects_to_list((effects)
			    db_get_memory_resource(DBR_IN_SUMMARY_REGIONS,
					  module_name,
					  true));

/* was (but didn't work)
    l_reg = (list) db_get_memory_resource(DBR_IN_SUMMARY_REGIONS,
					  module_name,
					  true);
					  */
    
    l_pairs = alias_pairs(module_name, l_reg);

    DB_PUT_MEMORY_RESOURCE(DBR_IN_ALIAS_PAIRS, 
			   strdup(module_name),
			   (char*) make_effects_classes(l_pairs));

    pips_debug(4,"end\n");
    debug_off();

    return(true);

}

/* top-level creation of pairs of aliases of OUT regions of the module
 * modifies global vars callee, list_regions_callee, list_pairs and
 * current_caller_stmt
 */
bool 
out_alias_pairs( const char* module_name )
{
    list l_reg, l_pairs;

    debug_on("ALIAS_PAIRS_DEBUG_LEVEL");
    pips_debug(4,"begin for module %s\n",module_name);

    /* we need the OUT summary regions*/
    l_reg = effects_to_list((effects)
			    db_get_memory_resource(DBR_OUT_SUMMARY_REGIONS,
					  module_name,
					  true));

/* was (but didn't work)
    l_reg = (list) db_get_memory_resource(DBR_OUT_SUMMARY_REGIONS,
					  module_name,
					  true);
					  */

    
    l_pairs = alias_pairs(module_name, l_reg);

    DB_PUT_MEMORY_RESOURCE(DBR_OUT_ALIAS_PAIRS, 
			   strdup(module_name),
			   (char*) make_effects_classes(l_pairs));

    pips_debug(4,"end\n");
    debug_off();

    return(true);

}




