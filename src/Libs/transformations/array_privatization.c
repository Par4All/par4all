/* regions package :  Be'atrice Creusillet, october 1995
 *
 * array_privatization.c
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the functions computing the private regions.
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
#include "effects.h"
#include "regions.h"
#include "pipsdbm.h"
#include "resources.h"



#define IS_EG TRUE
#define NOT_EG FALSE

#define PHI_FIRST TRUE
#define NOT_PHI_FIRST FALSE

/* global static variable local_regions_map, and its access functions */
DEFINE_CURRENT_MAPPING(private_regions, list)
DEFINE_CURRENT_MAPPING(copy_out_regions, list)

/* statement stack */
DEFINE_LOCAL_STACK(current_stmt, statement)


/* =============================================================================== 
 *
 * INTRAPROCEDURAL PRIVATE REGIONS ANALYSIS
 *
 * =============================================================================== */

static void private_regions_of_module_statement(statement module_stat);
static void private_regions_of_module_loops(statement module_stat);

/* void array_privatizer(char *module_name) 
 * input    : the name of the current module
 * output   : nothing.
 * modifies : computes the local regions of a module.
 * comment  : local regions can contain local variables.
 */
bool array_privatizer(module_name)
char *module_name;
{
    entity module;
    statement module_stat;


    /* set and get the current properties concerning regions */
    (void) set_bool_property("MUST_REGIONS", TRUE);
    (void) set_bool_property("EXACT_REGIONS", TRUE);
    (void) get_regions_properties();

    /* Get the code of the module. */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();
    
    /* Get the transformers and preconditions of the module. (Necessary ?) */
    set_transformer_map( (statement_mapping) 
	db_get_memory_resource(DBR_TRANSFORMERS, module_name, TRUE) );
    set_precondition_map( (statement_mapping) 
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, TRUE) );

    /* Get the READ, WRITE, IN and OUT regions of the module */
    set_local_regions_map( effectsmap_to_listmap( (statement_mapping) 
	db_get_memory_resource(DBR_REGIONS, module_name, TRUE) ) );
    set_in_regions_map( effectsmap_to_listmap( (statement_mapping) 
	db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE) ) );
    set_out_regions_map( effectsmap_to_listmap( (statement_mapping) 
	db_get_memory_resource(DBR_OUT_REGIONS, module_name, TRUE) ) );

    /* predicates defining summary regions from callees have to be 
       translated into variables local to module */
    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();

    set_cumulated_effects_map( effectsmap_to_listmap((statement_mapping)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE)) );
    module_to_value_mappings(module);


    /* initialisation of private maps */
    set_private_regions_map( MAKE_STATEMENT_MAPPING() );
    set_copy_out_regions_map( MAKE_STATEMENT_MAPPING() );

    debug_on("ARRAY_PRIVATIZATION_DEBUG_LEVEL");
    pips_debug(1, "begin\n");

    /* Compute the private regions of the module and of its loops. */
    (void) private_regions_of_module_statement(module_stat); 
    (void) private_regions_of_module_loops(module_stat);

    pips_debug(1, "end\n");
    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_precondition_map();
    reset_cumulated_effects_map();
    reset_local_regions_map();
    reset_in_regions_map();
    reset_out_regions_map();
    reset_private_regions_map();
    reset_copy_out_regions_map();

    return(TRUE);
}



static void private_regions_of_module_statement(module_stat)
statement module_stat;
{
    list l_priv, l_cand, l_out_cand;
    list l_write = regions_dup
	(regions_write_regions(load_statement_local_regions(module_stat))); 
    list l_in = regions_dup(load_statement_in_regions(module_stat));
    list l_out = regions_dup(load_statement_out_regions(module_stat));
    

    pips_debug(1,"begin\n");

    pips_debug(2, "CAND = W -inf IN \n");       
    l_cand = RegionsInfDifference(l_write, l_in, w_r_combinable_p);
    
    pips_debug(2, "OUT_CAND = CAND inter OUT\n");
    l_out_cand = RegionsIntersection(regions_dup(l_cand), l_out, w_w_combinable_p);

    pips_debug(2, "PRIV = CAND - inf OUT_CAND\n");
    l_priv = RegionsInfDifference(l_cand, l_out_cand, w_w_combinable_p);
    
    store_statement_private_regions(module_stat, l_priv);
    store_statement_copy_out_regions(module_stat, NIL);
    
    pips_debug(1,"end\n");
}


static bool stmt_filter(statement s);
static void private_regions_of_statement(statement s);
static bool private_regions_of_loop(loop l);


/* static void private_regions_of_module_loops(statement module_stat)
 * input    : the current statement.
 * output   : 
 * modifies : 
 * comment  : 
 */
static void private_regions_of_module_loops(statement module_stat)
{

    make_current_stmt_stack();

    pips_debug(1,"begin\n");
    
    gen_multi_recurse(module_stat, 
		      statement_domain, stmt_filter, private_regions_of_statement,
		      loop_domain, private_regions_of_loop, gen_null,
		      NULL); 
    

    pips_debug(1,"end\n");

    free_current_stmt_stack();

}


static bool stmt_filter(s)
statement s;
{
    current_stmt_push(s);
    return(TRUE);
}

static void private_regions_of_statement(s)
statement s;
{
    pips_debug(1, "statement %03d\n", statement_number(s));

    if (load_statement_private_regions(s) == (list) HASH_UNDEFINED_VALUE)
    {
	store_statement_private_regions(s, NIL);
	store_statement_copy_out_regions(s,NIL);
    }

    current_stmt_pop();

    pips_debug(1, "end\n");

}


static bool private_regions_of_loop(l)
loop l;
{
    statement b = loop_body(l);
    transformer loop_trans = load_statement_transformer(current_stmt_head());
    list l_cand, l_loc, l_out_priv, l_out_priv_tmp, l_locals, 
         l_loc_i, l_loc_i_prime, l_loc_tmp;
    list l_write, l_in, l_out, l_tmp;
    entity i = loop_index(l);
    entity i_prime = entity_to_intermediate_value(i);
    Psysteme sc_loop_prec;
    Pcontrainte contrainte;

    pips_debug(1, "begin, statement %03d\n", 
	       statement_number(current_stmt_head()));


    /* first get the write, in and out regions */
    l_write = regions_dup(regions_write_regions(load_statement_local_regions(b)));
    l_in = regions_to_write_regions(regions_dup(load_statement_in_regions(b)));
    l_out = regions_dup(load_statement_out_regions(b));

    project_regions_with_transformer_inverse(l_write, 
				     loop_trans, 
				     CONS(ENTITY, i, NIL));

    project_regions_with_transformer_inverse(l_in, 
				     loop_trans, 
				     CONS(ENTITY, i, NIL));

    project_regions_with_transformer_inverse(l_out, 
				     loop_trans, 
				     CONS(ENTITY, i, NIL));
    ifdebug(2)
    {
	pips_debug(3, "W(i) before: \n");
	print_regions(l_write);
	pips_debug(3, "IN(i) before: \n");
	print_regions(l_in);
	pips_debug(3, "OUT(i) before: \n");
	print_regions(l_out);
    }

    /* LOC(i) = W(i) -inf IN(i) */
    l_tmp = regions_dup(l_in);
    l_loc = RegionsInfDifference(l_write, l_tmp, w_w_combinable_p);
    
    ifdebug(3)
    {
	pips_debug(3, "LOC(i) = W(i) -inf IN(i)\n");
	print_regions(l_loc);
    }
    
    /* Keep only arrays that induce false dependences between iterations   
     * that is to say arrays such that LOC(i) inter LOC(i', i'<i) != empty_set 
     */
    sc_loop_prec = sc_loop_proper_precondition(l);
    
    /* we make and LOC(i, i'<i) and LOC(i', i'<i) */
    l_loc_i_prime = regions_dup(l_loc);
    l_loc_i = regions_dup(l_loc);
    array_regions_variable_rename(l_loc_i_prime, i, i_prime);
    contrainte = contrainte_make(vect_make(VECTEUR_NUL, 
						   (Variable) i_prime, 1,
						   (Variable) i, -1,
						   TCST, 1));
    sc_add_inegalite(sc_loop_prec, contrainte_dup(contrainte));
    sc_loop_prec->base = BASE_NULLE;
    sc_creer_base(sc_loop_prec);
    array_regions_add_sc(l_loc_i_prime, sc_loop_prec);
    array_regions_add_sc(l_loc_i, sc_loop_prec);
    sc_rm(sc_loop_prec);
    l_loc_i = RegionsIntersection(l_loc_i, l_loc_i_prime, w_w_combinable_p);
    
    /* We keep in Loc(i) only the regions that correspond to arrays in l_loc_i,
     * that is to say arrays that induce false dependences
     */
    l_loc_tmp = l_loc;
    while(!ENDP(l_loc_tmp))
    {
	region reg = EFFECT(CAR(l_loc));
	boolean found = FALSE;
	list l_loc_i_tmp = l_loc_i;
	while(!found && !ENDP(l_loc_i_tmp))
	{
	    if (same_entity_p(region_entity(reg), 
			      region_entity(EFFECT(CAR(l_loc_i_tmp)))))
	    {
		found = TRUE;
	    }
	    l_loc_i_tmp = CDR(l_loc_i_tmp);
	}
	l_loc_tmp = CDR(l_loc_tmp);
	if (!found)
	{
	    gen_remove(&l_loc, reg);	
	    region_free(reg);
	}
	
    }

    regions_free(l_loc_i);

    ifdebug(3)
    {
	pips_debug(3,"regions on arrays that really induce false dependences:");
	print_regions(l_loc);
    }


    /* CAND(i) = LOC(i) -inf proj_i'[IN(i')] */
    
    /* first proj_i'[IN(i')] */
    sc_loop_prec = sc_loop_proper_precondition(l);
    array_regions_variable_rename(l_in, i, i_prime);	
    array_regions_add_sc(l_in, sc_loop_prec);
    sc_rm(sc_loop_prec);
    project_regions_along_loop_index(l_in, i_prime, loop_range(l));

    /* Then the difference */
    l_cand = RegionsInfDifference(l_loc, l_in, w_w_combinable_p);

    ifdebug(3)
    {
	pips_debug(3, "CAND(i) = LOC(i) -inf proj_i'[IN(i')]\n");
	print_regions(l_cand);
    }

    /* OUT_PRIV(i) = CAND(i) inter OUT(i) */

    l_out_priv = RegionsIntersection(regions_dup(l_cand), l_out, w_w_combinable_p);
    
    ifdebug(3)
    {
	pips_debug(3, "OUT_PRIV(i) = CAND(i) inter OUT(i)\n");
	print_regions(l_out_priv);
    }


    /* keep only good candidates (those for which the copy-out is either empty
     * or exactly determined). */

    /* For the moment, since copy-out is not handled, we remove all the arrays 
     * that have a copy-out 
     */
    l_out_priv_tmp = l_out_priv;
    while(!ENDP(l_out_priv_tmp))
    {
	list l_cand_tmp;
	boolean found;

	region reg = EFFECT(CAR(l_out_priv_tmp));
	l_out_priv_tmp = CDR(l_out_priv_tmp);
	/* copy-out : if (!region_must_p(reg)) */
	/* copy-out : { */
	l_cand_tmp = l_cand;
	found = FALSE;
	/* remove the corresponding candidate from l_cand */
	while(!found && !ENDP(l_cand_tmp))
	{
	    region reg_cand = EFFECT(CAR(l_cand_tmp));
	    l_cand_tmp = CDR(l_cand_tmp);
	    if (same_entity_p(region_entity(reg_cand) , region_entity(reg)))
	    {
		/* a` optimiser */
		gen_remove(&l_cand, reg_cand);
		region_free(reg_cand);
		found = TRUE;
	    }
	}
	    gen_remove(&l_out_priv, reg);
	    region_free(reg);
	/* } */
    }
 
    /* compute loop_locals from l_cand */
    gen_free_list(loop_locals(l));
    l_locals = NIL;
    MAP(EFFECT, reg,
    {
	l_locals = CONS(ENTITY, region_entity(reg), l_locals);
    },
	l_cand);
    loop_locals(l) = l_locals;
   
    /* add the loop index */
    loop_locals(l) = CONS(ENTITY, loop_index(l), loop_locals(l));

    ifdebug(2)
    {
	pips_debug(3, "good candidates: ");
	print_arguments(loop_locals(l));
	pips_debug(3, "corresponding regions:\n");
	print_regions(l_cand);
	pips_debug(3, "copy-out:\n");
	print_regions(l_out_priv);
    }

    
    store_statement_private_regions(b, l_cand);
    store_statement_copy_out_regions(b,l_out_priv);
    
    pips_debug(1, "end\n");
    
    return(TRUE);
}
