/* 
 * package regions :  Be'atrice Creusillet, September 1994
 *
 * ------------ 
 * projection.c
 * ------------
 *
 * This file contains the functions specific to the projection of regions along
 * individual variables, list of variables or vectors of variables. The
 * algorithms are described in the report EMP-CRI E/185.
 *
 */


#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "text.h"
#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"
#include "matrice.h"
#include "sparse_sc.h"
#include "properties.h"
#include "pipsdbm.h"
#include "transformer.h"
#include "semantics.h"

#include "effects-generic.h"
#include "effects-convex.h"


/*********************************************************************************/
/* STATISTICS FOR PROJECTION OPERATORS                                           */
/*********************************************************************************/

static int nb_proj_param = 0;
static int nb_proj_param_pot_must = 0;
static int nb_proj_param_must = 0;
static int nb_proj_param_ofl = 0;
static int nb_proj_param_hermite = 0;
static int nb_proj_param_hermite_success = 0;

static int nb_proj_var = 0;
static int nb_proj_var_pot_must = 0;
static int nb_proj_var_must = 0;
static int nb_proj_var_ofl = 0;


void reset_proj_op_statistics()
{
    nb_proj_param = 0;
    nb_proj_param_pot_must = 0;
    nb_proj_param_must = 0;
    nb_proj_param_ofl = 0;
    nb_proj_param_hermite = 0;
    nb_proj_param_hermite_success = 0;

    nb_proj_var = 0;
    nb_proj_var_pot_must = 0;
    nb_proj_var_must = 0;
    nb_proj_var_ofl = 0;
}


void print_proj_op_statistics(char *mod_name, char *prefix)
{
    FILE *fp;
    string filename;

    filename = "proj_param_op_stat";
    filename = strdup(concatenate(db_get_current_workspace_directory(), "/", 
				  mod_name, ".", prefix, filename, 0));
    fp = safe_fopen(filename, "w");
    fprintf(fp,"%s",mod_name);
    fprintf(fp," %d %d %d %d %d %d\n", nb_proj_param, nb_proj_param_pot_must, 
	    nb_proj_param_must, nb_proj_param_ofl, nb_proj_param_hermite, 
	    nb_proj_param_hermite_success);    
    safe_fclose(fp, filename);
    free(filename);

    filename = "proj_var_op_stat";
    filename = strdup(concatenate(db_get_current_workspace_directory(), "/", 
				  mod_name, ".", prefix, filename, 0));
    fp = safe_fopen(filename, "w");
    fprintf(fp,"%s",mod_name);
    fprintf(fp," %d %d %d %d \n", nb_proj_var, nb_proj_var_pot_must, 
	    nb_proj_var_must, nb_proj_var_ofl);    
    safe_fclose(fp, filename);
    free(filename);
}


/*********************************************************************************/
/* FONCTIONS D'INTERFACE                                                         */
/*********************************************************************************/


/* entity 
 * loop_regions_normalize(list l_reg, entity index, range l_range,
 *		          boolean *normalized_regions_p, boolean sc_loop_p,
 *                        Psysteme *psc_loop)
 * input    : a list of regions, a loop index, its range, and a pointer on a 
 *            boolean to know if the loop is normalized, a boolean to know
 *            if the loop precondition system sc_loop must be normalized.
 *          
 * output   : an entity representing the new loop index to use; it may be index
 *            if the loop is already normalized.
 * modifies : l_reg, and *normalized_regions_p.
 * comment  : Loops are not normalized in PIPS, but array region semantic
 *            functions are defined for normalized loops (incr = +/-1). 
 *            So we perform here a virtual loop normalization. If the loop is not 
 *            already normalized, a new loop index (beta) is introduced, 
 *            and the following system is added to each region of l_reg:
 *                 { index = lower_bound + beta * incr, 0 <= beta }
 *            Then the old index is eliminated as a parameter, and the beta
 *            variable is returned as the new loop index. 
 *            If the loop was already normalized, or has been virtually normalized,
 *            then *normalized_regions_p is set to TRUE.
 */
entity
loop_regions_normalize(list l_reg, entity index, range l_range,
		       boolean *normalized_regions_p, boolean sc_loop_p,
		       Psysteme *psc_loop )
{
    Value incr = VALUE_ZERO;
    normalized nub, nlb;
    list l_tmp;
    entity new_index = index;

    debug_regions_consistency(l_reg);

    if (must_regions_p())
    {
	expression e_incr = range_increment(l_range);
	normalized n;

	/* Is the loop increment numerically known ? */
	n = NORMALIZE_EXPRESSION(e_incr);
	if(normalized_linear_p(n))
	{
	    Pvecteur v_incr = normalized_linear(n);
	    if(vect_constant_p(v_incr)) 
		incr = vect_coeff(TCST, v_incr);	
	}
	
	nub = NORMALIZE_EXPRESSION(range_upper(l_range));
	nlb = NORMALIZE_EXPRESSION(range_lower(l_range));    
	
	*normalized_regions_p = 
	    (value_notzero_p(incr) && normalized_linear_p(nub) 
				    && normalized_linear_p(nlb));
	pips_debug(6, "normalized loop? %s.\n", 
		   *normalized_regions_p? "yes" : "no");
    
	if (*normalized_regions_p) 
	{	
	    if (value_notone_p(value_abs(incr)))
	    {
		/* add to each region predicate the system:
		 *   { index = lower_bound + beta * incr, 0 <= beta }
		 */
		entity beta = make_beta_entity(1);
		Psysteme beta_sc = sc_new();
		Pvecteur v_lb =  normalized_linear(nlb);
		Pvecteur beta_v;

		new_index = beta;
		beta_v = vect_new((Variable) index , VALUE_ONE);
		vect_add_elem(&beta_v, (Variable) beta, value_uminus(incr));
		beta_v = vect_cl_ofl_ctrl(beta_v,
					   VALUE_MONE, v_lb, NO_OFL_CTRL);
		sc_add_egalite(beta_sc,contrainte_make(beta_v)); 
		
		beta_v = vect_new((Variable) beta, VALUE_MONE);
		sc_add_inegalite(beta_sc,contrainte_make(beta_v)); 	
		sc_creer_base(beta_sc);
		
		ifdebug(6)
		{
		    pips_debug(6, "index system:\n"); sc_syst_debug(beta_sc);
		}		
		
		/* eliminate the old index (which is no more a variable,
		 * but a parameter) */
		l_tmp = CONS(ENTITY, index, NIL);
		MAP(EFFECT, reg, 
		{
		    if (!region_rn_p(reg) && !region_empty_p(reg))
		    {
			region_sc_append_and_normalize(reg,beta_sc,1);
			region_exact_projection_along_parameters(reg, l_tmp);
		    }
		}, l_reg);
		gen_free_list(l_tmp);

		/* update the loop preconditions */
		if (sc_loop_p)
		{
		    Psysteme sc_tmp;
		    sc_tmp = sc_safe_append(*psc_loop, beta_sc);
		    sc_projection_along_variable_ofl_ctrl(&sc_tmp,(Variable) index,
							  FWD_OFL_CTRL); 
		    sc_base_remove_variable(sc_tmp, (Variable) index);
		    *psc_loop = sc_tmp;
		}
		sc_rm(beta_sc);
	    }	        
	}
    }
    
    debug_regions_consistency(l_reg);

    return new_index;
}

/* void project_regions_along_loop_index(list l_reg, entity index, l_range) 
 * input    : a list l_reg of regions, a variable which is a loop index.
 * output   : nothing.
 * modifies : l_reg and the regions it contains.
 * comment  : project each region in l_reg along the variable index.
 */
void project_regions_along_loop_index(list l_reg, entity index, range l_range)
{
    boolean projection_of_index_safe = FALSE;
    Psysteme sc_tmp = SC_UNDEFINED;

    debug_on("REGIONS_OPERATORS_DEBUG_LEVEL");
    debug_regions_consistency(l_reg);
    

    index = loop_regions_normalize(l_reg, index, l_range,
				   &projection_of_index_safe,FALSE,&sc_tmp);

    if (must_regions_p && projection_of_index_safe)
    {
	
	MAPL(ll_reg, {
	    region_exact_projection_along_variable(EFFECT(CAR(ll_reg)), index);
	}, l_reg);	
    }
    else
    {
	list l = CONS(ENTITY, index, NIL);
	MAP(EFFECT, reg, {
	    region_non_exact_projection_along_variables(reg, l);
	}, l_reg);
	gen_free_list(l);
    }  
    debug_regions_consistency(l_reg);
    debug_off();
  
}


/* void project_regions_along_variables(list l_reg, list l_param) 
 * input    : a list of regions to project, and the list of variables 
 *            along which the projection will be performed.
 * output   : nothing.
 * modifies : l_reg and the regions it contains.
 * comment  : project each region in l_reg along the variables in l_var
 */
void project_regions_along_variables(list l_reg, list l_var)
{

    debug_on("REGIONS_OPERATORS_DEBUG_LEVEL");
    debug_regions_consistency(l_reg);

    if (must_regions_p()) 
    {
	MAP(EFFECT, reg, {
	    region_exact_projection_along_variables(reg, l_var);
	}, l_reg);
    }
    else 
    {
	MAP(EFFECT, reg, {
	    region_non_exact_projection_along_variables(reg, l_var);
	},l_reg);
    }
    debug_regions_consistency(l_reg);    
    debug_off();
}


/* void project_regions_along_parameters(list l_reg, list l_param) 
 * input    : a list of regions to project, and the list of variables 
 *            along which the projection will be performed.
 * output   : nothing.
 * modifies : l_reg and the regions it contains.
 * comment  : project each region in l_reg along the variables in l_param
 */
void project_regions_along_parameters(list l_reg, list l_param)
{

    debug_on("REGIONS_OPERATORS_DEBUG_LEVEL");
    debug_regions_consistency(l_reg);

    if (must_regions_p()) 
    {
	MAP(EFFECT, reg, {
	    region_exact_projection_along_parameters(reg, l_param);
	}, l_reg);	
    }
    else 
    {
	MAP(EFFECT, reg, {
	    region_non_exact_projection_along_parameters(reg, l_param);
	}, l_reg);
    }
    debug_regions_consistency(l_reg);
    debug_off();
}



void 
project_regions_with_transformer(list l_reg, transformer trans,
				 list l_var_not_proj)
{
    regions_transformer_apply(l_reg, trans, l_var_not_proj, FALSE);
}

void project_regions_with_transformer_inverse(list l_reg, transformer trans,
					      list l_var_not_proj)
{
    regions_transformer_apply(l_reg, trans, l_var_not_proj, TRUE);
}

/* void regions_transformer_apply(l_reg, trans, l_var_not_proj) 
 * input    : a list of regions, the transformer corresponding
 *            to the current statement, and a list of variables along which 
 *            the regions must not be projected (typically a loop index).
 * output   : nothing.
 * modifies : l_reg and the regions it contains
 * comment  : project each region in l_reg along the variables in the arguments 
 *            of trans which are not in l_var_not_proj, using the algorithm 
 *            described in document E/185/CRI.	
 */
void regions_transformer_apply(list l_reg, transformer trans,
			       list l_var_not_proj, bool backward_p)
{
    list l_var = arguments_difference(transformer_arguments(trans), l_var_not_proj);
    
    if (ENDP(l_var) || ENDP(l_reg))
	return;


    debug_on("REGIONS_OPERATORS_DEBUG_LEVEL");
    debug_regions_consistency(l_reg);

    ifdebug(3)
    {
	pips_debug(3, "regions before transformation:\n");
	print_regions(l_reg);
	pips_debug(3, "elimination of variables: \n");
	print_arguments(l_var);
    }
	
    if (!must_regions_p()) 
    {
	project_regions_along_parameters(l_reg, l_var);       
    }
    else
    {
	list l_int, l_old;
	Psysteme sc_trans = sc_dup(predicate_system(transformer_relation(trans)));

	ifdebug(8)
	    {
		fprintf(stderr,"transformer:\n");
		sc_print(sc_trans,entity_local_name);
	    }	
	
	/* addition of the predicate of the transformer to the predicate of 
	 * the regions and elimination of redundances; then, projection of 
	 * regions along initial variables, and renaming of old variables
         * corresponding to the eliminated variables into new variables. */
	
	/* first we store the names of the old and int variables */
	l_old = variables_to_old_variables(l_var);
	l_int = variables_to_int_variables(l_var);

	if(backward_p)
	{
	    sc_list_variables_rename(sc_trans, l_var, l_int);
	    sc_list_variables_rename(sc_trans, l_old, l_var);
	}
	else
	{
	    sc_list_variables_rename(sc_trans, l_old, l_int);	
	}
	
	MAP(EFFECT, reg, 
	{
	    Psysteme sc_reg = region_system(reg);
	    
	    if (!sc_empty_p(sc_reg) && !sc_rn_p(sc_reg))
	    {  
		debug_region_consistency(reg);

		ifdebug(8)
		{
		    pips_debug(8, "region before transformation: \n\t%s\n", 
			       region_to_string(reg) );
		}
		sc_list_variables_rename(sc_reg, l_var, l_int);
		ifdebug(8)
		{
		    pips_debug(8, "region after renaming: \n\t%s\n", 
			       region_to_string(reg) );
		}

		/* addition of the predicate of the transformer, 
		   and elimination of redundances */
		region_sc_append_and_normalize(reg, sc_trans, 1);
		
		debug_region_consistency(reg);

		ifdebug(8)
		{
		    pips_debug(8, "region after addition of the transformer: "
			       "\n\t%s\n", region_to_string(reg) );
		}
		
		/* projection along intermediate variables */
		region_exact_projection_along_parameters(reg, l_int);
		debug_region_consistency(reg);
		
		
		ifdebug(8)
		{
		    pips_debug(8, "region after transformation: \n\t%s\n", 
			       region_to_string(reg) );
		}
	    }
	},
	    l_reg);

	/* no memory leaks */
	gen_free_list(l_int);
	gen_free_list(l_old);
	sc_rm(sc_trans);
    }

    ifdebug(3)
    {
	pips_debug(3, "regions after transformation:\n");
	print_regions(l_reg);
    }

    gen_free_list(l_var);
    debug_regions_consistency(l_reg);
    debug_off();
}


/* void regions_remove_phi_variables(list l_reg)
 * input    : a list of regions, and an integer, which is the highest rank
 *            of phi variables that will be kept.
 * output   : nothing.
 * modifies : project regions in l_reg along the phi variables which rank 
 *            are higher (>) than phi_max.
 * comment  : An assumption is made : the projection is exact and the 
 *            approximation are thus preserved, except if an overflow error occurs.
 *            This function is only used in the case of a forward interprocedural
 *            propagation : the assumption is then always true.
 */
void regions_remove_phi_variables(list l_reg)
{   
    debug_on("REGIONS_OPERATORS_DEBUG_LEVEL");
    debug_regions_consistency(l_reg);

    MAP(EFFECT, reg,
    {
	region_remove_phi_variables(reg);
    }, l_reg);
    debug_regions_consistency(l_reg);

    debug_off();
}


/* list regions_dynamic_elim(list l_reg)
 * input    : a list of regions.
 * output   : a list of regions in which regions of dynamic variables are
 *            removed, and in which dynamic integer scalar variables are 
 *            eliminated from the predicate.
 * modifies : nothing; the regions l_reg initially contains are copied 
 *            if necessary.
 * comment  :	
 */
list regions_dynamic_elim(list l_reg)
{
    list l_res = NIL;

    debug_on("REGIONS_OPERATORS_DEBUG_LEVEL");
    debug_regions_consistency(l_reg);

    MAP(EFFECT, reg, 
     {
        entity reg_ent = region_entity(reg);
        storage reg_s = entity_storage(reg_ent);
        boolean ignore_this_region = FALSE;

	ifdebug(4)
	{
	    pips_debug(4, "current region: \n%s\n", region_to_string(reg));
	}

	/* If the reference is a common variable (ie. with storage ram but
	 * not dynamic) or a formal parameter, the region is not ignored.
	 */
        switch (storage_tag(reg_s))
	{
	case is_storage_return:
	    pips_debug(5, "return var ignored (%s)\n", entity_name(reg_ent));
	    ignore_this_region = TRUE;
	    break;
	case is_storage_ram:
	{
	    ram r = storage_ram(reg_s);
	    if (dynamic_area_p(ram_section(r)) || heap_area_p(ram_section(r)))
	    {
		pips_debug(5, "dynamic or pointed var ignored (%s)\n", entity_name(reg_ent));
		ignore_this_region = TRUE;
	    }
	    break;
	}
	case is_storage_formal:
	    break;
	case is_storage_rom:
	    pips_internal_error("bad tag for %s (rom)\n", entity_name(reg_ent));
	default:
	    pips_internal_error("case default reached\n");
        }
	
        if (! ignore_this_region)  /* Eliminate dynamic variables. */
	{
	    region r_res = copy_region(reg);
	    region_dynamic_var_elim(r_res);
	    ifdebug(4)
	    {
		pips_debug(4, "region kept : \n\t %s\n", region_to_string(r_res));
	    }
            l_res = region_add_to_regions(r_res,l_res);
        }
	else 
	    ifdebug(4)
	    {
		pips_debug(4, "region removed : \n\t %s\n", region_to_string(reg));
	    }
    },
	l_reg);
    debug_regions_consistency(l_reg);    
    debug_off();

    return(l_res);
}


/*********************************************************************************/
/* OPE'RATEURS CONCERNANT LES RE'GIONS INDIVIDUELLES.                            */
/*********************************************************************************/

static Psysteme region_sc_projection_ofl_along_parameters(Psysteme sc, 
							  Pvecteur pv_param, 
							  boolean *p_exact);
static Psysteme region_sc_projection_ofl_along_parameter(Psysteme sc, Variable param, 
							 boolean *p_exact);


/* void region_remove_phi_variables(effect reg, int phi_max)
 * input    : a PSI region in which phi variables must be eliminated.
 * output   : nothing.
 * modifies : reg
 * comment  : if an overflow error occurs, the region becomes a MAY region..
 */
void region_remove_phi_variables(region reg)
{
    list 
	l_phi = NIL, 
	l_reg = CONS(EFFECT,reg,NIL);
    
    ifdebug(8)
    {
	pips_debug(8, "region : \n ");
	print_region(reg);
    }
    
    l_phi = phi_entities_list(1,7);    
    project_regions_along_variables(l_reg, l_phi);
    gen_free_list(l_reg);
    gen_free_list(l_phi);

    ifdebug(8)
    {
	pips_debug(8, "final region : \n ");
	print_region(reg);
    }
}


/* void region_remove_psi_variables(effect reg, int phi_max)
 * input    : a PHI region in which psi variables must be eliminated.
 * output   : nothing.
 * modifies : reg
 * comment  : if an overflow error occurs, the region becomes a MAY region..
 */
void region_remove_psi_variables(region reg)
{
    list
	l_psi = NIL,
	l_reg = CONS(EFFECT,reg,NIL);
    
    ifdebug(8)
    {
	pips_debug(8, "region : \n ");
	print_region(reg);
    }
    
    l_psi = psi_entities_list(1,7);    
    project_regions_along_variables(l_reg, l_psi);
    gen_free_list(l_reg);    

    ifdebug(8)
    {
	pips_debug(8, "final region : \n ");
	print_region(reg);
    }
}


/* void region_remove_beta_variables(effect reg, int beta_max)
 * input    : a PSI region in which beta variables must be eliminated.
 * output   : nothing.
 * modifies : reg
 * comment  : if an overflow error occurs, the region becomes a MAY region..
 */
void region_remove_beta_variables(region reg)
{
    list 
	l_beta = NIL, 
	l_reg = CONS(EFFECT,reg,NIL);
    
    ifdebug(8)
    {
	pips_debug(8, "region : \n ");
	print_region(reg);
    }
    
    l_beta = beta_entities_list(1,2);    
    project_regions_along_variables(l_reg, l_beta);
    gen_free_list(l_reg);
    gen_free_list(l_beta);

    ifdebug(8)
    {
	pips_debug(8, "final region : \n ");
	print_region(reg);
    }
}


/* void region_non_exact_projection_along_parameters(effect reg, list l_param)
 * input    : a region and a list of variables.
 * output   : nothing.
 * modifies : the initial region is projected along the variables in l_param.
 *            its approximation is systematically set to may.
 * comment  : overflow errors are trapped here. if it occurs, an empty region
 *           replaces the initial region.	
 */
void region_non_exact_projection_along_parameters(region reg, list l_param)
{
    Psysteme ps;    
    ps = region_system(reg);

    if (!sc_empty_p(ps) && !sc_rn_p(ps)) 
    {
	if (op_statistics_p()) nb_proj_param++;

	ifdebug(6)
	{
	    pips_debug(6, "initial region :\n");
	    print_region(reg);
	    debug(6, "", "parameters along which the projection must be performed:\n");
	    print_arguments(l_param);
	}
	
	/* if there is an overflow error, reg becomes a whole array may region */
	CATCH(overflow_error)
	{
	    region reg_tmp = reference_whole_region(region_reference(reg),
						    region_action_tag(reg));

	    region_system_(reg) = region_system_(reg_tmp);
	    region_reference(reg_tmp) = reference_undefined;
	    region_system_(reg_tmp) = newgen_Psysteme(SC_UNDEFINED);
	    free_effect(reg_tmp);
	    sc_rm(ps);
	    if (op_statistics_p()) nb_proj_param_ofl++;
	}	
	TRY 
	{	    
	    ps = sc_projection_ofl_along_list_of_variables(ps, l_param);
	    region_system_(reg) = newgen_Psysteme(ps);
	    region_approximation_tag(reg) = is_approximation_may;
	    UNCATCH(overflow_error);
	}
    }    
}
    

/* void region_exact_projection_along_parameters(effect reg, list l_param)
 * input    : a regions reg and a list of parameters l_param.
 * output   : nothing.
 * modifies : the initial region is successively projected along each variable
 *            in l_param. its approximaiton becomes may if the projection is not 
 *            "exact" in the sense given in report E/185.
 * comment  : overflow errors are trapped here. if it occurs, an empty region
 *           replaces the initial region.	
 */
void region_exact_projection_along_parameters(region reg, list l_param)
{
    tag app = region_approximation_tag(reg);
    Psysteme ps;


    ps = region_system(reg);

    if (!sc_empty_p(ps) && !sc_rn_p(ps))
    {
	ifdebug(6)
	{
	    pips_debug(6, "initial region :\n");
	    print_region(reg);
	    debug(6, "", "parameters along which the projection must be performed:\n");
	    print_arguments(l_param);
	}
	
	if (op_statistics_p()) nb_proj_param++;
	
	/* if there is an overflow error, reg becomes a whole array may region */
	CATCH(overflow_error) {
	    region reg_tmp = reference_whole_region(region_reference(reg),
						    region_action_tag(reg));

	    region_system_(reg) = region_system_(reg_tmp);
	    region_reference(reg_tmp) = reference_undefined;
	    region_system_(reg_tmp) = newgen_Psysteme(SC_UNDEFINED);
	    free_effect(reg_tmp);
	    sc_rm(ps);
	    if (op_statistics_p()) nb_proj_param_ofl++;
	}
	TRY {	    
	    /* if the initial region is a may region, the resulting region is
	     * also a may region, even if the projection happens to be exact.
	     * so we do not need to know whether it is exact or not
	     */
	    if (app == is_approximation_may) 
	    {
		ps = sc_projection_ofl_along_list_of_variables(ps, l_param); 
	    }	    
	
	    /* if the initial region is a must region, much more work is
	     * necessary to preserve the must approximation
	     */
	    else 
	    {		
		list l_phi_param = 
		    arguments_intersection(l_param, region_phi_cfc_variables(reg));
		list l_not_phi_param = arguments_difference(l_param, l_phi_param);  
		
		/* first, the projection along parameters that are not linked to PHI 
		 * variables (l_not_phi_param) is exact */
		if (!ENDP(l_not_phi_param)) 
		{
		    ps = sc_projection_ofl_along_list_of_variables
			(ps, l_not_phi_param);		    
		}
		
		/* then, we must perform the projection along parameters that 
		 * are linked to PHI variables (l_phi_param). */
		if(!ENDP(l_phi_param))
		{
		    boolean exact = TRUE;
		    Pvecteur pv_phi_param = NULL;
		    
		    if (op_statistics_p()) nb_proj_param_pot_must++;
		    
		    /* projection functions only work on vector of parameters, 
		       not on lists */
		    MAP(ENTITY, e,
		    {
			if (base_contains_variable_p(ps->base, (Variable) e) )
			    vect_add_elem(&pv_phi_param, (Variable) e, VALUE_ONE);
		    }, l_phi_param);
		    
		    ps =
			region_sc_projection_ofl_along_parameters(ps, pv_phi_param, 
								  &exact);
		    vect_rm(pv_phi_param);
		    if(!exact) 
			app = is_approximation_may;
		    else
			if (op_statistics_p()) nb_proj_param_must++;
		}
		gen_free_list(l_phi_param);
		gen_free_list(l_not_phi_param);
	    }
	    region_system_(reg) = newgen_Psysteme(ps);
	    region_approximation_tag(reg) = app;
	    UNCATCH(overflow_error);
	}
	
	ifdebug(6)
	{
	    pips_debug(6, "final region:\n");
	    print_region(reg);
	}
    }
}


/* void region_non_exact_projection_along_variables(effect reg, list l_var)
 * input    : a region and a list of variables.
 * output   : nothing.
 * modifies : the initial region is projected along the variables in l_param.
 *            its approximation is systematically set to may.
 * comment  : overflow errors are trapped here. if it occurs, an empty region
 *           replaces the initial region.	
 */
void region_non_exact_projection_along_variables(region reg, list l_var)
{
    Psysteme ps;

    ps = region_system(reg);

    if (!sc_empty_p(ps) && !sc_rn_p(ps))
    {
	if (op_statistics_p()) nb_proj_var++;

	ifdebug(6)
	{
	    pips_debug(6, "initial region :\n");
	    print_region(reg);
	    debug(6, "", "variables along which the projection must be performed:\n");
	    print_arguments(l_var);
	}
	
	/* if there is an overflow error, reg becomes a whole array may region */
	CATCH(overflow_error) 
	{
	    region reg_tmp = reference_whole_region(region_reference(reg),
						    region_action_tag(reg));

	    region_system_(reg) = region_system_(reg_tmp);
	    region_reference(reg_tmp) = reference_undefined;
	    region_system_(reg_tmp) = newgen_Psysteme(SC_UNDEFINED);
	    free_effect(reg_tmp);
	    sc_rm(ps);
	    if (op_statistics_p()) nb_proj_var_ofl++;

	}
	TRY 
	{
	    ps = sc_projection_ofl_along_list_of_variables(ps, l_var);
	    region_system_(reg) = newgen_Psysteme(ps);
	    region_approximation_tag(reg) = is_approximation_may;
	    UNCATCH(overflow_error);
	}
    }
}
    
/* void region_exact_projection_along_variables(effect reg, list l_var)
 * input    : a region and a list of variables.
 * output   : nothing.
 * modifies : the initial region is projected along the variables in l_param.
 *            its approximation is set to may if the projection is not exact.
 * comment  : overflow errors are trapped here. if it occurs, an empty region
 *           replaces the initial region.	
 */
void region_exact_projection_along_variables(reg, l_var)
effect reg;
list l_var;
{
    Psysteme ps;    

    ps = region_system(reg);

    if (!sc_empty_p(ps) && !sc_rn_p(ps))
    {
	ifdebug(6)
	{
	    pips_debug(6, "initial region :\n");
	    print_region(reg);
	    debug(6, "", "variables along which the projection must be performed:\n");
	    print_arguments(l_var);
	}
	
	/* if there is an overflow error, reg becomes a whole array may region */
	CATCH(overflow_error)
	{
	    region reg_tmp = reference_whole_region(region_reference(reg),
						    region_action_tag(reg));

	    region_system_(reg) = region_system_(reg_tmp);
	    region_reference(reg_tmp) = reference_undefined;
	    region_system_(reg_tmp) = newgen_Psysteme(SC_UNDEFINED);
	    free_effect(reg_tmp);
	    sc_rm(ps);
	    if (op_statistics_p()) nb_proj_var_ofl++;
	}
	TRY {
	    list ll_var = l_var;
	    
	    for(; !ENDP(ll_var) && 
		    (region_approximation_tag(reg) == is_approximation_must); 
		ll_var = CDR(ll_var)) 
	    {
		region_exact_projection_along_variable(reg, 
						       ENTITY(CAR(ll_var)));
	    }
	    
	    if (!ENDP(ll_var))
	    {
		region_non_exact_projection_along_variables(reg, ll_var);
	    }
	    UNCATCH(overflow_error);
	}
    }
}
    


/* void region_exact_projection_along_variable(effect reg, entity var) 
 * input    : a region and a variable (a loop index for instance).
 * output   : nothing.
 * modifies : the initial region, which is projected along var.
 *            the approximation is not modified.
 * comment  : overflow errors are trapped here. if it occurs, an empty region
 *           replaces the initial region.	
 */
void region_exact_projection_along_variable(region reg, entity var)
{
    Psysteme sc;

    sc = region_system(reg);

    if (!sc_empty_p(sc) && !sc_rn_p(sc))
    {
	if (op_statistics_p())
	{
	    nb_proj_var++;
	    if (region_approximation_tag(reg) == is_approximation_must) 
		nb_proj_var_pot_must++;
	}
		
	ifdebug(3)
	{
	    pips_debug(3, "initial region :\n");
	    print_region(reg);
	}
	
	if (base_contains_variable_p(sc->base, (Variable) var)) 
	{
	    CATCH(overflow_error)
	    {
		region reg_tmp = 
		    reference_whole_region(region_reference(reg),
					   region_action_tag(reg));
		
		region_system_(reg) = region_system_(reg_tmp);
		region_reference(reg_tmp) = reference_undefined;
		region_system_(reg_tmp) = newgen_Psysteme(SC_UNDEFINED);
		free_effect(reg_tmp);
		sc_rm(sc);
		if (op_statistics_p()) nb_proj_var_ofl++;
	    }
	    TRY {
		list l_phi_var = region_phi_cfc_variables(reg);
		
		if (gen_find_eq(var, l_phi_var) == chunk_undefined)
		{
		    sc_projection_along_variable_ofl_ctrl(&sc,(Variable) var,
							  FWD_OFL_CTRL); 
		    sc_base_remove_variable(sc, (Variable) var);
		    sc = region_sc_normalize(sc,2);	
		    region_system_(reg) = newgen_Psysteme(sc);
		    
		    if (op_statistics_p() &&
			(region_approximation_tag(reg)==is_approximation_must))
			nb_proj_var_must++;
		}
		else 
		{
		    Pvecteur pv_var = NULL;
		    boolean is_proj_exact = TRUE;
		    
		    vect_add_elem(&pv_var, (Variable) var, VALUE_ONE);
		    sc = sc_projection_ofl_along_variables_with_test
			(sc, pv_var,  &is_proj_exact);
		    vect_rm(pv_var);
		    sc = region_sc_normalize(sc,2);	
		    region_system_(reg)= newgen_Psysteme(sc);
		    
		    if (region_approximation_tag(reg) == is_approximation_must) 
		    {
			if (!is_proj_exact)
			    region_approximation_tag(reg) = is_approximation_may;
			else 
			    if (op_statistics_p())  nb_proj_var_must++;
		    }
		}
		gen_free_list(l_phi_var);
		UNCATCH(overflow_error);
	    }
	}
	
	ifdebug(3)
	{
	    pips_debug(3, "final region :\n");
	    print_region(reg);
	}    
    }
}


/*********************************************************************************/
/* Ope'rateurs concernant les syste`mes de contraintes, mais propres au          */
/* des re'gions.                                                                 */
/*********************************************************************************/


static Pcontrainte eq_var_nophi_min_coeff(Pcontrainte contraintes, Variable var);
static Pcontrainte eq_var_phi(Pcontrainte contraintes, Variable var);
static Psysteme region_sc_minimal(Psysteme sc, boolean *p_sc_changed_p);


/* Psysteme region_sc_projection_ofl_along_parameters(Psysteme sc, 
 *                                                 Pvecteur pv_param, 
 *                                                 boolean *p_exact) 
 * input    : a convex polyhedron sc to project along the parameters contained
 *            in the vector pv_param, which are linked to the PHI variables; 
 *            p_exact is a pointer to a boolean indicating whether the 
 *            projection is exact or not.
 * output   : the polyhedron resulting of the projection.
 * modifies : sc, *p_exact.
 * comment  : 
 */
static Psysteme region_sc_projection_ofl_along_parameters(Psysteme sc,
							  Pvecteur pv_param,
							  boolean *p_exact)
{
    *p_exact = TRUE;

    for(; (*p_exact == TRUE) && !VECTEUR_NUL_P(pv_param); pv_param = pv_param->succ)
    {
	Variable param = vecteur_var(pv_param);
	sc = region_sc_projection_ofl_along_parameter(sc, param, p_exact);
	sc_base_remove_variable(sc, param);
	sc = region_sc_normalize(sc,2);	    	
    }

    if (!(*p_exact) && !VECTEUR_NUL_P(pv_param))
    {
	region_sc_projection_along_variables_ofl_ctrl(&sc, pv_param, FWD_OFL_CTRL);
	/*sc = sc_projection_ofl_along_variables(sc,pv_param);*/
    }
    return sc;
}



/* Psysteme region_sc_projection_ofl_along_parameter(Psysteme sc, 
 *                                                 Variable param, 
 *                                                 boolean *p_exact) 
 * input    : a convex polyhedron sc to project along the variable param
 *            which is linked to the PHI variables; 
 *            p_exact is a pointer to a boolean indicating whether the 
 *            projection is exact or not.
 * output   : the polyhedron resulting of the projection.
 * modifies : sc, *p_exact.
 * comment  : first, we must search for an equation
 *   containing this variable, but no phi variables;
 *   if such an equation does not explicitly exists, it may be implicit,
 *   and we must find it using an hermitte form.
 *   if such an equation does not exist, we must search for the 
 *   inequations containing this variable and one ore more phi variables. 
 *   If it does not exist then the projection is exact, 
 *   otherwise, it is not exact.
 *
 * WARNING : the base and dimension of sc are not updated.
 */
static Psysteme region_sc_projection_ofl_along_parameter(Psysteme sc, Variable param,
							 boolean *p_exact)
{
    Pcontrainte eq;
    Pbase base_sc;
    Psysteme sc_tmp;
    boolean hermite_sc_p;

    pips_debug(8, "begin\n");	


    /* fisrt, we search for an explicit equation containing param, 
     * but no phi variables
     */
    eq = eq_var_nophi_min_coeff(sc->egalites, param);
    
    if (!CONTRAINTE_UNDEFINED_P(eq)) 
    {
	sc = sc_projection_ofl_with_eq(sc, eq, param);
	*p_exact = TRUE;
	debug(8, "region_sc_projection_ofl_along_parameter", 
	      "explicit equation found.\n");	
	return(sc); 
    }

    /* if the last equation is not explicit, it may be implicit 
     */
    sc_tmp = region_sc_minimal(sc, &hermite_sc_p);

    if (hermite_sc_p) 
    {
	if (op_statistics_p()) nb_proj_param_hermite++;

	eq = eq_var_nophi_min_coeff(sc->egalites, param);
	
	if (!CONTRAINTE_UNDEFINED_P(eq)) 
	{
	    sc = sc_projection_ofl_with_eq(sc, eq, param);
	    *p_exact = TRUE;
	    pips_debug(8, "implicit equation found.\n");	
	    if (op_statistics_p()) nb_proj_param_hermite_success++;
	    
	    return(sc);
	}
    }

    /* we then search if there exist an equation linking param to the 
     * PHI variables.
     */
    eq = eq_var_phi(sc->egalites,param);

    if (!CONTRAINTE_UNDEFINED_P(eq)) 
    {
	sc = sc_projection_ofl_with_eq(sc, eq, param);
	*p_exact = FALSE;
	pips_debug(8, "equation with PHI and param found -> projection not exact.\n");	

	return(sc); 
    }
    

    /* if such an equation does not exist, we search for an 
     * inequation containing param and a phi variable 
     */
    eq = eq_var_phi(sc->inegalites, param);
    if(CONTRAINTE_UNDEFINED_P(eq)) 
	*p_exact = TRUE;
    else 
	*p_exact = FALSE;
	
    base_sc = base_dup(sc->base);
    if (!combiner_ofl(sc, param)) 
    {
	sc_rm(sc);
	return(sc_empty(base_sc));
    }

    base_rm(base_sc);

    pips_debug(8, "other cases.\n");	
    return(sc);
}


/* Pcontrainte eq_var_nophi_min_coeff(Pcontrainte contraintes, Variable var)
 * input    : 
 * output   : la contrainte (egalite ou inegalite) de "contraintes" qui contient 
 *            var, mais pas de variable de variables phis et ou` var a le 
 *            coeff non nul le plus petit;
 * modifies : nothing
 * comment  : la contrainte renvoye'e n'est pas enleve'e de la liste.
 */
static Pcontrainte 
eq_var_nophi_min_coeff(Pcontrainte contraintes, Variable var)
{
    Value c, c_var = VALUE_ZERO;
    Pcontrainte eq_res, eq;

    if ( CONTRAINTE_UNDEFINED_P(contraintes) ) 
	return(CONTRAINTE_UNDEFINED);

    eq_res = CONTRAINTE_UNDEFINED;
    
    for (eq = contraintes; eq != NULL; eq = eq->succ) 
    {
	/* are there phi variables in this equation ? */
	if (!vect_contains_phi_p(eq->vecteur)) {
	    /* if not, we search if the coeff of var is minimum */
	    c = vect_coeff(var, eq->vecteur);
	    value_absolute(c);
	    if (value_notzero_p(c) && 
		(value_lt(c,c_var) || value_zero_p(c_var)))
	    {
		c_var = c;
		eq_res = eq;
	    }
	}
    }
    return(eq_res);
}

/* Pcontrainte eq_var_nophi_min_coeff(Pcontrainte contraintes, Variable var)
 * input    : 
 * output   : la contrainte (egalite ou inegalite) de "contraintes" qui contient 
 *            var, mais pas de variable de variables phis et ou` var a le 
 *            coeff non nul le plus petit;
 * modifies : nothing
 * comment  : la contrainte renvoye'e n'est pas enleve'e de la liste.
 */
static Pcontrainte eq_var_phi(Pcontrainte contraintes, Variable var)
{
    Pcontrainte eq_res, eq;

    if ( CONTRAINTE_UNDEFINED_P(contraintes) ) 
	return(CONTRAINTE_UNDEFINED);

    eq_res = CONTRAINTE_UNDEFINED;
    eq = contraintes;

    while (CONTRAINTE_UNDEFINED_P(eq_res) && !CONTRAINTE_UNDEFINED_P(eq))
    {
	if (vect_contains_phi_p(eq->vecteur) && 
	    value_notzero_p(vect_coeff(var, eq->vecteur)))
	    eq_res = eq;
	eq = eq->succ;    
    }
    return(eq_res);
}


static int constraints_nb_phi_eq(Pcontrainte eqs);

/* Psysteme region_sc_minimal(Psysteme sc, boolean p_sc_changed_p) 
 * input    : a polyhedron
 * output   : an equivalent polyhedron, in which the number of equations
 *            containing phi variables is minimal (see report E/185).
 * modifies : sc and p_sc_changed_p. The pointed boolean is set to TRUE, if
 *            a new sc is calculated (with the hermite stuff).
 * comment  : 
 */
static Psysteme region_sc_minimal(Psysteme sc, boolean *p_sc_changed_p)
{
    Pcontrainte eq = sc->egalites;
    int m = nb_elems_list(eq),
        n = base_dimension(sc->base),
        n_phi = base_nb_phi(sc->base);
    Pmatrix A, C, A_phi, A_phi_t, H, P, Q, Q_t, A_min, C_min;
    Value det_P, det_Q;
    Pbase sorted_base = base_dup(sc->base);

    *p_sc_changed_p = FALSE;

    /* the number of equations containing phi variables must be greater than one,
     * otherwise, the system is already minimal. Moreover, the number of phi
     * variables must be greater than 0. 
     */
    if (!(n_phi > 0 && (constraints_nb_phi_eq(eq) > 1)))
	return(sc);

    *p_sc_changed_p = TRUE;
    
    ifdebug(8)
    {
	pips_debug(8, "initial sc :\n");
	sc_syst_debug(sc);
    }

    phi_first_sort_base(&sorted_base);

    A = matrix_new(m,n);
    C = matrix_new(m,1); 
    A_phi = matrix_new(m,n_phi);
    A_phi_t = matrix_new(n_phi,m);
    H = matrix_new(n_phi,m);
    P = matrix_new(n_phi,n_phi);
    Q = matrix_new(m,m);
    Q_t = matrix_new(m,m);
    A_min = matrix_new(m,n);
    C_min = matrix_new(m,1);
	
    /* We first transform the equations of the initial system into
     * a matrices equation.
     */
    constraints_to_matrices(eq, sorted_base, A, C);
    
    /* We then compute the corresponding hermite form of the transpose
     * of the A_phi matrice
     */
    ordinary_sub_matrix(A, A_phi, 1, m, 1, n_phi);
    matrix_transpose(A_phi, A_phi_t);
    matrix_free(A_phi); 
    
    matrix_hermite(A_phi_t, P, H, Q, &det_P, &det_Q);
    matrix_transpose(Q,Q_t);
    matrix_free(P);
    matrix_free(H);
    matrix_free(Q);
    
    /* and we deduce the minimal systeme */
    matrix_multiply(Q_t, A, A_min);
    matrix_multiply(Q_t, C, C_min);
    matrix_free(A);
    matrix_free(C);
    
    contrainte_free(eq);
    matrices_to_constraints(&(sc->egalites), sorted_base, A_min, C_min);
    matrix_free(A_min);
    matrix_free(C_min);
    
    ifdebug(8)
    {
	pips_debug(8, "final sc :\n");
	sc_syst_debug(sc);
    }

    return sc;
}


/* static int constraints_nb_phi_eq(Pcontrainte eqs) 
 * input    : a list of contraints.
 * output   : the number of contraints containing phi variables.
 * modifies : nothing.
 * comment  : 	
 */
static int constraints_nb_phi_eq(Pcontrainte eqs)
{
    int nb_phi_eq = 0;
    
    for(; !CONTRAINTE_UNDEFINED_P(eqs); eqs = eqs->succ)
	if (vect_contains_phi_p(eqs->vecteur))
	    nb_phi_eq++;
    
    return(nb_phi_eq);
}



/* boolean region_projection_along_index_safe_p(entity index, range l_range)
 * input    : an loop index and its range
 * output   : TRUE if its projection of regions along index is safe (see
 *            conditions in report E/185).
 * modifies : nothing
 * comment  :	
 */
boolean region_projection_along_index_safe_p(entity index, range l_range)
{
    boolean projection_of_index_safe = FALSE;
    normalized n, nub, nlb;
    expression e_incr = range_increment(l_range);
    Value incr = VALUE_ZERO;
    
    /* Is the loop increment numerically known ? */
    n = NORMALIZE_EXPRESSION(e_incr);
    if(normalized_linear_p(n)) 
    {
	Pvecteur v_incr = normalized_linear(n);
	pips_assert("project_regions_along_index_safe_p", 
		    !VECTEUR_NUL_P(v_incr));
	if(vect_constant_p(v_incr)) 
	    incr = vect_coeff(TCST, v_incr);
    }
    
    nub = NORMALIZE_EXPRESSION(range_upper(l_range));
    nlb = NORMALIZE_EXPRESSION(range_lower(l_range));    
    
    projection_of_index_safe = 
	(value_one_p(value_abs(incr)) && normalized_linear_p(nub) 
				&& normalized_linear_p(nlb));
    return(projection_of_index_safe);
}


/* void region_dynamic_var_elim(effect reg)
 * input    : a region .
 * output   : nothing
 * modifies : 
 * comment  : eliminates all dynamic variables contained in the system of 
 *            constraints of the region.
 *
 *             First, it makes a list of all the dynamic variables present. 
 *             And then, it projects the region along the hyperplane 
 *             defined by the dynamic entities
 * 
 */
void region_dynamic_var_elim(region reg)
{
    list l_var_dyn = NIL;
    Psysteme ps_reg;
    Pbase ps_base;

    ps_reg = region_system(reg);
    ps_base = ps_reg->base;
    
    pips_assert("region_dynamic_var_elim", ! SC_UNDEFINED_P(ps_reg));

    for (; ! VECTEUR_NUL_P(ps_base); ps_base = ps_base->succ) 
    {
	entity e = (entity) ps_base->var;
	
	/* TCST is not dynamic */
	if (e != NULL) 
	{
	    storage s = entity_storage(e);
	    
	    debug(5, "region_dynamic_var_elim", "Test upon var : %s\n",
		  entity_local_name(e));

	    /* An entity in a system that has an undefined storage is
	       necesseraly a PHI entity, not dynamic !! */
	    if (s != storage_undefined) 
	    {
		if (storage_tag(s) == is_storage_ram) 
		{
		    ram r = storage_ram(s);
		    if (dynamic_area_p(ram_section(r))) 
		    {
			debug(5, "region_dynamic_var_elim", 
			      "dynamic variable found : %s\n", entity_local_name(e));
			l_var_dyn = CONS(ENTITY, e, l_var_dyn);
		    }
		}
	    }
	}
    }
    ifdebug(3)
    {
	pips_debug(3, "variables to eliminate: \n");
	print_arguments(l_var_dyn);
    }
    
    if (must_regions_p())  
	region_exact_projection_along_parameters(reg, l_var_dyn);
    else 
	region_non_exact_projection_along_parameters(reg, l_var_dyn);
    gen_free_list(l_var_dyn);
}


/*********************************************************************************/
/* MISC                                                                          */
/*********************************************************************************/


Psysteme sc_projection_ofl_along_list_of_variables(Psysteme ps, list l_var)
{

    Pvecteur pv_var = NULL;
    /* converts the list into a Pvecteur */
    MAP(ENTITY, e,
	{
	    if (base_contains_variable_p(ps->base, (Variable) e) )
		vect_add_elem(&pv_var, (Variable) e, VALUE_ONE);
	}, l_var);
    region_sc_projection_along_variables_ofl_ctrl(&ps, pv_var, FWD_OFL_CTRL);    
    /* ps = sc_projection_ofl_along_variables(ps, pv_var); */
    vect_rm(pv_var);
    return(ps);
}



/* void region_sc_projection_ofl_along_variables(Psysteme *psc, Pvecteur pv)
 * input    : a system of constraints, and a vector of variables.
 * output   : a system of contraints, resulting of the successive projection
 *            of the initial system along each variable of pv.
 * modifies : *psc.
 * comment  : it is very different from 
 *              sc_projection_with_test_along_variables_ofl_ctrl.	
 *            sc_empty is returned if the system is not feasible (BC).
 *            The base of *psc is updated. assert if one of the variable
 *            does not belong to the base.
 *             The case ofl_ctrl == OFL_CTRL is not handled, because
 *             the user may want an SC_UNDEFINED, or an sc_empty(sc->base)
 *             or an sc_rn(sc->base) as a result. It is not homogeneous.
 *             special implementation for regions: special choice for 
 *           redudancy elimination. bc. 
 */
void region_sc_projection_along_variables_ofl_ctrl(psc, pv, ofl_ctrl)
Psysteme *psc;
Pvecteur pv;
int ofl_ctrl;
{
    Pvecteur pv1;
    Pbase scbase = base_dup((*psc)->base); 

    if (!VECTEUR_NUL_P(pv)) {
	for (pv1 = pv;!VECTEUR_NUL_P(pv1) && !SC_UNDEFINED_P(*psc); pv1=pv1->succ) {
	  sc_projection_along_variable_ofl_ctrl_timeout_ctrl(psc,vecteur_var(pv1), 
	     ofl_ctrl);
	  /* DN sc_projection_along_variable_ofl_ctrl(psc,vecteur_var(pv1), 
	     ofl_ctrl);*/
	    if (!SC_UNDEFINED_P(*psc)) {
		sc_base_remove_variable(*psc,vecteur_var(pv1));
		/* *psc = region_sc_normalize(*psc,2);	 */
	    }
	}
    }

    if (SC_UNDEFINED_P(*psc)){ /* ne devrait plus arriver ! */
	*psc = sc_empty(scbase);
	for (pv1 = pv;!VECTEUR_NUL_P(pv1); pv1=pv1->succ) {
	    sc_base_remove_variable(*psc,vecteur_var(pv1));
	}
    }
    else {
	base_rm(scbase);
    }

}
