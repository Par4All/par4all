/* package convex effects :  Be'atrice Creusillet 5/97
 *
 * File: utils.c
 * ~~~~~~~~~~~~~
 *
 * This File contains the interfaces with pipsmake which compute the various
 * types of convex regions by using the generic functions.
 *
 *
 * Actually this is the mere inclusion of the old utils.c of regions.
 * Maybe it should be cleaned a little bit. 
 */

/*
 * $Id$
 *
 * package regions :  Alexis Platonoff, 22 Aout 1990
 * modified : Beatrice Creusillet
 *
 * This File contains useful functions for the computation of the regions.
 *
 * It can be sundered in seven groups of functions :
 *                           _ manipulation of regions and lists of regions
 *                           _ regions/list of regions conversion
 *                           _ creation of regions 
 *                           _ phi entities
 *                           _ properties
 *                           _ variables and entities
 *                           _ preconditions and transformers
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#include "properties.h"

#include "transformer.h"
#include "semantics.h"

#include "effects-generic.h"
#include "effects-convex.h"

#define IS_EG TRUE
#define NOT_EG FALSE

#define PHI_FIRST TRUE
#define NOT_PHI_FIRST FALSE

/******************************************************** BETA ENTITIES */

#define BETA_MAX 5
static entity beta[BETA_MAX]; /* beta entities */

/* entity make_beta_entity(int n) 
 * input    : an integer n giving the range of the BETA entity.
 * output   : the entity BETA_n.
 * modifies : beta[] (eventually).
 * comment  : finds or generates an entity representing a dummy variable.
 *
 *            Once a BETA dummy entity, it is stored into a static array 
 *            (beta[]); thus, each next time such an entity is needed, 
 *            it is just picked up in beta[].
 *            If the dummy is not in beta[], it may be in the table
 *            where all entities are stored, because of previous computations.
 *            At last, if it is not found there, it is created. 
 */
entity make_beta_entity(int n)
{
    
    pips_assert("BETA range (n) must be between 1 and BETA_MAX\n", 
		1<=n && n<=BETA_MAX);

    /* beta indices are between 1 and BETA_MAX.
       Array indices are between 0 to BETA_MAX - 1. */
    if (beta[n-1] == entity_undefined) 
    {
	entity v;
	char beta_name[6];
	char * s;
	
	sprintf(beta_name,"%s%d",BETA_PREFIX,n);
	s = strdup(concatenate(REGIONS_MODULE_NAME,
			       MODULE_SEP_STRING, beta_name, (char *) NULL));
	
	if ((v = gen_find_tabulated(s, entity_domain)) == entity_undefined) 
	{
	    v = make_entity(s, make_scalar_integer_type(4),
			    make_storage(is_storage_rom, UU),
			    value_undefined);
	}




	beta[n-1] = v;
    }
    return (beta[n-1]);
}


list beta_entities_list(int beta_min, int beta_max)
{
    list l_beta = NIL;
    int i;
    
    for(i=beta_min; i<=beta_max; i++) 
	l_beta = gen_nconc(l_beta, CONS(ENTITY, make_beta_entity(i), NIL));
    
    return(l_beta);
}

/****************************************** INTIALISATION  AND TERMINATION */

static entity phi[7]; /* phi entities */
static entity psi[7]; /* psi entities */

void regions_init()
{
    int i;
    for(i=0; i<7; i++)
    {
	phi[i] = entity_undefined;
	psi[i] = entity_undefined;
    }
    for(i=0; i<BETA_MAX; i++)
    {
	beta[i] = entity_undefined;
    }
    set_region_interprocedural_translation(); 
}

void regions_end()
{
    reset_region_interprocedural_translation(); 

}

/******************************* REGIONS AND LISTS OF REGIONS MANIPULATION */


region region_dup(region reg)
{
    region new_reg;
    debug_region_consistency(reg);

    new_reg = copy_effect(reg);

    /* work around persistency of effect reference 
     */
    region_reference(new_reg) = reference_dup(region_reference(reg));
    debug_region_consistency(new_reg);
    return new_reg;
}


/* list regions_dup(list l_reg)
 * input    : a list of regions.
 * output   : a new list of regions, in which each region of the
 *            initial list is duplicated.
 * modifies : nothing.
 */
list regions_dup(list l_reg)
{
    list l_reg_dup = NIL;
    
    MAP(EFFECT, reg,
    {
	effect reg_dup = region_dup(reg);
	l_reg_dup = region_add_to_regions(reg_dup, l_reg_dup);
    }, l_reg);
    
    return l_reg_dup;
}

/* (void) regions_free(list l_reg)
 * input    : a list of regions
 * output   : nothing !
 * modifies : frees the list and the regions
 */
void regions_free(list l_reg)
{
    MAP(EFFECT, reg,
    {
	free_region(reg);
    }, l_reg);
    gen_free_list(l_reg);
}

/* (void) region_free(effect reg)
 * input    : a region
 * output   : nothing !
 * modifies : frees region
 */
void region_free(region reg)
{
    /* work around persistency of effect reference */
    debug_region_consistency(reg);
    free_reference(region_reference(reg));
    region_reference(reg) = reference_undefined;
    free_effect(reg);
}


/* void region_add_to_regions(region reg, list l_reg)
 * input    : a region and a list of regions.
 * output   : nothing.
 * modifies : l_reg.
 * comment  : adds reg at the end of l_reg.
 */
list region_add_to_regions(region reg, list l_reg)
{
    return gen_nconc(l_reg, CONS(REGION, reg, NIL));
}

list regions_add_region(list l_reg, region reg)
{
    return gen_nconc(l_reg, CONS(REGION, reg, NIL));
}


/* list regions_add_context(list l_reg, transformer context)
 * input    : a list of regions and a precondition representing the current
 *            context.
 * output   : nothing. 
 * modifies : the input regions ; their predicates contain the initial
 *            constraints plus the constraints of the context, modulo
 *            the elimination of redundancies.
 */
list regions_add_context(l_reg, context)
list l_reg;
transformer context;
{	

    l_reg = array_regions_sc_append_and_normalize
	(l_reg,
	 (Psysteme) predicate_system(transformer_relation(context)),
	 2);
    return(l_reg);
}

/* adapted from transformer_normalize
 */
Psysteme region_sc_normalize(Psysteme sc_reg, int level)
{
    
    if (!sc_empty_p(sc_reg))
    {
	Pbase b = base_dup(sc_base(sc_reg));
	
	/* Select one tradeoff between speed and accuracy:
	 * enumerated by increasing speeds according to Beatrice
	 */
	
	switch(level) 
	{
	    
	case 0:
	    /* Our best choice for accuracy, but damned slow on ocean */
	    sc_reg = sc_elim_redund(sc_reg);
	    break;
	    
	case 1:
	    /* Beatrice's best choice: does not deal with minmax2 (only)
	     * but still requires 74 minutes of real time
	     * (55 minutes of CPU time)
	     * for ocean preconditions, when applied to each precondition
	     * stored.
	     *
	     * Only 64 s for ocean, if preconditions are not normalized.
	     * But andne, callabsval, dead2, hind, negand, negand2, or,
	     * validation_dead_code are not validated any more. Redundancy
	     * could always be detected in a trivial way after propagating
	     * values from equations into inequalities.
	     */
	    sc_nredund(&sc_reg);
	    break;
	    
	case 2:
	    /* Francois' own: does most of the easy stuff.
	     * Fails on mimax2 and sum_prec, but it is somehow
	     * more user-friendly because trivial preconditions are
	     * not destroyed as redundant. It makes you feel safer.
	     *
	     * Result for full precondition normalization on ocean: 114 s
	     * for preconditions, 4 minutes between split ocean.f and
	     * OCEAN.prec
	     */
	    sc_reg = sc_strong_normalize(sc_reg);
	    break;
	    
	case 5:
	    /* Same plus a good feasibility test
	     */
	    sc_reg = sc_strong_normalize3(sc_reg);
	    break;
	    
	case 3:
	    /* Similar, but variable are actually substituted
	     * which is sometimes painful when a complex equations
	     * is used to replace a simple variable in a simple
	     * inequality.
	     */
	    sc_reg = sc_strong_normalize2(sc_reg);
	    break;
	case 6:
	    /* Similar, but variables are substituted if they belong to
	     * a more or less simple equation, and simpler equations
	     * are processed first and a lexicographically minimal
	     * variable is chosen when equivalent variables are
	     * available.
	     */
	    sc_reg =
		sc_strong_normalize4(sc_reg,
		    (char * (*)(Variable)) external_value_name);
	    break;
	    
	case 7:
	    /* Same plus a good feasibility test, plus variable selection
	     * for elimination, plus equation selection for elimination
	     */
	    sc_reg =
		sc_strong_normalize5(sc_reg,
		    (char * (*)(Variable)) external_value_name);
	    break;
	    
	case 4:
	    /* Pretty lousy: equations are not even used to eliminate 
	     * redundant inequalities!
	     */
	    sc_reg = sc_normalize(sc_reg);
	    break;
	    
	default:
	    pips_error("region_sc_normalize", "unknown level %d\n", level);
	}
	
	if (SC_EMPTY_P(sc_reg))	
	    sc_reg = sc_empty(b);	
	else 
	    base_rm(b);
	
	sc_reg->dimension = vect_size(sc_reg->base);
    }
    return(sc_reg);
    
}

/* void region_sc_append(region reg, Psysteme sc, bool nredund_p)
 * input    : a region and a Psysteme
 * output   : nothing.
 * modifies : the initial region to which a copy of sc is appended.
 * comment  : add to the predicate of the region a copy of the system sc.
 *            redundancies are then eliminated if nredund_p is true. 
 */
void region_sc_append_and_normalize(region reg, Psysteme sc, int level)
{
    Psysteme sc_reg;

    /* pips_assert("region context must be defined\n", 
		!transformer_undefined_p(region_context(reg))); */

    sc_reg = region_system(reg);    
    assert(sc_weak_consistent_p(sc_reg));
    assert(sc_weak_consistent_p(sc));
    sc_reg = sc_safe_append(sc_safe_normalize(sc_reg), sc_safe_normalize(sc)); 
    assert(sc_weak_consistent_p(sc_reg));
    if (level!=-1) 
	sc_reg = region_sc_normalize(sc_reg, level);
    assert(sc_weak_consistent_p(sc_reg));
    
    region_system_(reg) = newgen_Psysteme(sc_reg);      
}

/* void regions_sc_append(list l_reg, Psysteme sc, boolean arrays_only, 
 *                        scalars_only, bool nredund_p)
 * input    : a list of regions, a Psysteme and two booleans. 
 * output   : a list of regions; 
 * modifies : the regions contained in the list.
 * comment  : add to the predicate of each region the system sc. if arrays_only
 *            is true, scalar variables regions are not concerned. if
 *            scalars_only is true, array_variables are not concerned;
 *
 * usage    : l_reg =  regions_sc_append(l_reg, ...).
 */
list regions_sc_append_and_normalize(list l_reg, Psysteme sc,
		       boolean arrays_only, boolean scalars_only,
		       int level)
{
    list l_res = NIL;
    pips_assert("not arrays_only and scalars_only", !(arrays_only && scalars_only));

    if (sc_empty_p(sc))
    {
	regions_free(l_reg);
    }
    else
    {
	MAP(EFFECT, reg,
	    {
		boolean scalar_p = region_scalar_p(reg);
		
		if ((scalar_p && !arrays_only) || (!scalar_p && !scalars_only)) 
		{
		    region_sc_append_and_normalize(reg, sc, level);
		    if (!region_empty_p(reg))
			l_res = region_add_to_regions(reg, l_res);
		    else
			region_free(reg);
		} 
		else
		{
		    l_res = region_add_to_regions(reg, l_res); 
		}
	    }, l_reg);	
    gen_free_list(l_reg);	
    }
    return(l_res);
}

/* void array_regions_sc_append(list l_reg, Psysteme sc, bool nredund_p)
 * input    : a list of regions and the Psysteme to add. 
 * output   : nothing.
 * modifies : the regions contained in the list.
 * comment  : add to the predicate of each array region the system sc. 
 */
list array_regions_sc_append_and_normalize(list l_reg, Psysteme sc, int level)
{
    l_reg = regions_sc_append_and_normalize(l_reg, sc, TRUE, FALSE, level);
    return(l_reg);
}


/* void region_sc_append(region reg, Psysteme sc, bool nredund_p)
 * input    : a region and a Psysteme
 * output   : nothing.
 * modifies : the initial region to which a copy of sc is appended.
 * comment  : add to the predicate of the region a copy of the system sc.
 *            redundancies are then eliminated if nredund_p is true. 
 */
void region_sc_append(region reg, Psysteme sc, bool nredund_p)
{
    int level = nredund_p? 1: -1;
    region_sc_append_and_normalize(reg, sc, level);
}

/* void regions_sc_append(list l_reg, Psysteme sc, boolean arrays_only, 
 *                        scalars_only, bool nredund_p)
 * input    : a list of regions, a Psysteme and two booleans. 
 * output   : a list of regions; 
 * modifies : the regions contained in the list.
 * comment  : add to the predicate of each region the system sc. if arrays_only
 *            is true, scalar variables regions are not concerned. if
 *            scalars_only is true, array_variables are not concerned;
 *
 * usage    : l_reg =  regions_sc_append(l_reg, ...).
 */
list regions_sc_append(list l_reg, Psysteme sc,
		       boolean arrays_only, boolean scalars_only,
		       bool nredund_p)
{
    int level = nredund_p? 1: -1;
    return(regions_sc_append_and_normalize(l_reg, sc,
			     arrays_only, scalars_only,
			     level));
}


/* void all_regions_sc_append(list l_reg, Psysteme sc, bool nredund_p)
 * input    : a list of regions and the Psysteme to add. 
 * output   : nothing.
 * modifies : the regions contained in the list.
 * comment  : add to the predicate of each region the system sc. 
 */
list all_regions_sc_append(list l_reg, Psysteme sc, bool nredund_p)
{
    l_reg = regions_sc_append(l_reg, sc, FALSE, FALSE, nredund_p);
    return(l_reg);
}

/* void scalar_regions_sc_append(list l_reg, Psysteme sc, bool nredund_p)
 * input    : a list of regions and the Psysteme to add. 
 * output   : nothing.
 * modifies : the regions contained in the list.
 * comment  : add to the predicate of each region the system sc. 
 */
list scalar_regions_sc_append(list l_reg, Psysteme sc, bool nredund_p)
{
    l_reg = regions_sc_append(l_reg, sc, FALSE, TRUE, nredund_p);
    return(l_reg);
}

/* void array_regions_sc_append(list l_reg, Psysteme sc, bool nredund_p)
 * input    : a list of regions and the Psysteme to add. 
 * output   : nothing.
 * modifies : the regions contained in the list.
 * comment  : add to the predicate of each array region the system sc. 
 */
list array_regions_sc_append(list l_reg, Psysteme sc, bool nredund_p)
{
    l_reg = regions_sc_append(l_reg, sc, TRUE, FALSE, nredund_p);
    return(l_reg);
}







/* list regions_remove_variable_regions(list l_reg, l_var)
 * input    : a list of reigons and a list of variables
 * output   : a list of regions 
 * modifies : nothing.
 * comment  : the returned list of regions is the same as the input
 *            list, except that the regions corresponding to the variables
 *            in l_var are removed.	
 */
list regions_remove_variables_regions(list l_reg, list l_var)
{
    list l_res = NIL;
    
    MAP(REGION, reg,
    {
	if (gen_find_eq(region_entity(reg),l_var) == entity_undefined)
	    l_res = region_add_to_regions(reg,l_res);
        else
	    pips_debug(7, "Region on variable %s removed\n",
		       entity_local_name(region_entity(reg)));
    }, l_reg);
    return(l_res);
}


/* void array_regions_variable_rename(list l_reg, entity old_entity, new_entity)
 * input    : a list of regions, and two variables
 * output   : nothing.
 * modifies : the list of regions
 * comment  : in the array reigons of l_reg, the variable old_entity
 *            that appears in the predicates is replaced with new_entity.	
 */
void array_regions_variable_rename(list l_reg, entity old_entity, entity new_entity)
{
    MAP(EFFECT, reg, 
    {
	if (!region_scalar_p(reg))
	{
	    Psysteme sc_reg = region_system(reg);
      
	    if (!sc_empty_p(sc_reg) && !sc_rn_p(sc_reg))
	    { 
		sc_reg = sc_variable_rename(sc_reg, (Variable) old_entity, 
					    (Variable) new_entity);
	    }
	}
    },l_reg);
}

/* void array_regions_variable_rename(list l_reg, entity old_entity, new_entity)
 * input    : a list of regions, and two variables
 * output   : nothing.
 * modifies : the list of regions
 * comment  : in the array reigons of l_reg, the variable old_entity
 *            that appears in the predicates is replaced with new_entity.	
 */
void all_regions_variable_rename(list l_reg, entity old_entity, entity new_entity)
{
    MAP(EFFECT, reg, 
    {
	Psysteme sc_reg = region_system(reg);
      
	if (!sc_empty_p(sc_reg) && !sc_rn_p(sc_reg))
	{ 
	    ifdebug(8){
		pips_debug(8, "current system: \n");
		sc_syst_debug(sc_reg);
	    }
	    sc_reg = sc_variable_rename(sc_reg, (Variable) old_entity, 
					(Variable) new_entity);
	}
	
    },l_reg);
}

/* void region_value_substitute(region reg, entity e1, entity e2)
 * input    : an effect, an old entity e1 to rename into a new entity e2.
 * output   : nothing
 * modifies : the system of reg.
 * comment  : adapted from transformer_value_substitute. 
 *
 * if e2 does not appear in t initially:
 *    replaces occurences of value e1 by value e2 in transformer t's arguments
 *    and relation fields; 
 * else
 *    error
 * fi
 *
 * "e2 must not appear in t initially": this is the general case; the second case
 * may occur when procedure A calls B and C and when B and C share a global
 * variable X
 * which is not seen from A. A may contain relations between B:X and C:X...
 * See hidden.f in Bugs or Validation...
 */
void
region_value_substitute(region reg, entity e1, entity e2)
{
    /* updates are performed by side effects */
    Psysteme s = region_system(reg);

    pips_assert("region_value_substitute",
		e1 != entity_undefined && e2 != entity_undefined);

    /* update only if necessary */
    if(base_contains_variable_p(s->base, (Variable) e1))
    {
	if(!base_contains_variable_p(s->base, (Variable) e2))
	{
	    sc_variable_rename(s,(Variable) e1, (Variable)e2);
	}
	else
	{
	    pips_error("region_value_substitute",
		       "conflict between e1=%s and e2=%s\n",
		       entity_name(e1), entity_name(e2));
	}
    }
}


/* list region_entities_cfc_variables(effect reg, list l_ent)
 * input    : a region and a list of entities that may appear in the region
 *            expression.
 * output   : the list of variables that define the initial entities, directly
 *            or indirectly
 * modifies : nothing
 * comment  : "cfc" stands for "Composante Fortement Connexe"; because
 *            if systems are considered as graphs which vertices are 
 *            the constraints and which edges connect constraints 
 *            concerning at least one common variable, then this
 *            function searches for the variables contained in
 *            the strongly connected component that also contains
 *            the initial entities.
 */
list region_entities_cfc_variables(region reg, list l_ent)
{
    return( sc_entities_cfc_variables(region_system(reg), l_ent) );
}

/* list sc_entities_cfc_variables(Psysteme sc, list l_ent)
 * input    : a system of constraints and a list of entities that may appear 
 *            in its constraints.   
 *            expression.
 * output   : the list of variables that define the initial entities, directly
 *            or indirectly
 * modifies : nothing
 * comment  : "cfc" stands for "Composante Fortement Connexe"; because
 *            if systems are considered as graphs which vertices are 
 *            the constraints and which edges connect constraints 
 *            concerning at least one common variable, then this
 *            function searches for the variables contained in
 *            the strongly connected component that also contains
 *            the initial entities.
 */
list sc_entities_cfc_variables(Psysteme sc, list l_ent)
{
    Pcontrainte c, c_pred, c_suiv;
    Pvecteur v_ent = VECTEUR_NUL, v, v_res = VECTEUR_NUL;    
    list l_res = NIL;
    
    if (ENDP(l_ent))
	return NIL;
    
    ifdebug(8) {
	pips_debug(8, "system: \n");
	sc_syst_debug(sc);	
	debug(8, "", "variables :\n");
	print_arguments(l_ent);
    }
    
    pips_assert("sc_entities_cfc_variables", !SC_UNDEFINED_P(sc));
    
    /* cas particuliers */ 
    if(sc_rn_p(sc) || sc_empty_p(sc))
	/* ne rien faire et renvoyer (nil) */ 
	return(NIL);
 
    sc = sc_dup(sc);
    
    /* we make a vector containing the initial entities */
    MAP(ENTITY, e, 
    {
	vect_add_elem(&v_ent, (Variable) e, VALUE_ONE);
    },
	l_ent);
    
    v_res = vect_dup(v_ent);
    
    /* sc has no particularity. We just scan its equalities and inequalities 
     * to find which variables are related to the initial entities */ 
    while(vect_common_variables_p((Pvecteur) sc_base(sc), v_res))
    { 
	/* equalities first */ 
	c = sc_egalites(sc);
	c_pred = (Pcontrainte) NULL; 
	while (c != (Pcontrainte) NULL) { 
	    c_suiv = c->succ; 
	    /* if a constraint is found in sc, that contains 
	     * a variable that already belongs to v_res then this constraint 
	     * is removed from sc and its variables are added to v_res */ 
	    if (vect_common_variables_p(c->vecteur,v_res))
	    { 
		if (c_pred != (Pcontrainte) NULL) 
		    c_pred->succ = c_suiv; 
		else 
		    sc_egalites(sc) = c_suiv; 
		for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ) 
		{ 
		    if (vecteur_var(v) != TCST) 
			vect_add_elem(&v_res, (Variable) vecteur_var(v), VALUE_ONE); 
		} 
	    } 
	    else 
		c_pred = c; 
	    c = c_suiv;
	} 
	
	/* inequalities then */ 
	c = sc_inegalites(sc); 
	c_pred = (Pcontrainte) NULL; 
	while (c != (Pcontrainte) NULL) 
	{ 
	    c_suiv = c->succ; 
	    /* if a constraint is found in sc, that contains a variable that 
	     * already belongs to v_res then this constraint is removed from 
	     * sc and its variables are added to v_res */ 
	    if (vect_common_variables_p(c->vecteur,v_res))
	    { 
		if (c_pred != (Pcontrainte) NULL) 
		    c_pred->succ = c_suiv; 
		else 
		    sc_inegalites(sc) = c_suiv; 
		for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ) 
		{ 
		    if (vecteur_var(v) != TCST) 
			vect_add_elem(&v_res, (Variable) vecteur_var(v), VALUE_ONE);
		} 
	    } 
	    else 
		c_pred = c; 
	    c = c_suiv;
	} 
	
	/* update sc base */
	base_rm(sc->base);
	sc_base(sc) = (Pbase) NULL; 
	(void) sc_creer_base(sc); 
    } /* while */ 
    
    sc_rm(sc);
    sc = NULL;
    
    /* construction of the list of variables. It contains all the variables of
     * v_res except for the initial entities */  
    for(; !VECTEUR_NUL_P(v_res); v_res = v_res->succ) 
    {
	entity e = (entity) v_res->var;
	if (!vect_contains_variable_p(v_ent, (Variable) e))
	    l_res = CONS(ENTITY, e, l_res);	
    }
    
    vect_rm(v_res); 
    
    ifdebug(8) {
	pips_debug(8, "l_res : \n");
	print_arguments(l_res);
    }

    return(l_res); 
}


/*********************************************************************************/
/* REGIONS / LISTS OF REGIONS CONVERSION                                         */
/*********************************************************************************/

/* list region_to_list(reg) 
 * input    : a region
 * output   : a list containing this region 
 * modifies : nothing
 * comment  : it does not duplicate the region.	
 */
list region_to_list(region reg)
{
    return(CONS(EFFECT, reg, NIL));
}


/* list region_to_may_region_list(reg)
 * input    : a region.
 * output   : a list containing the same region with a may approximation
 * modifies : the initial region
 * comment  : it does not duplicate the region.
 */
list region_to_may_region_list(reg)
effect reg;
{
    region_approximation_tag(reg) = is_approximation_may;
    return(CONS(EFFECT, reg, NIL));
}

/* list regions_to_nil_list(reg)
 * input    : a region
 * output   : an empty list of regions
 * modifies : nothing
 * comment  : for compatibility	
 */
list regions_to_nil_list(region reg1, region reg2)
{
    return(NIL);
}

/* list region_to_nil_list(reg)
 * input    : a region
 * output   : an empty list of regions
 * modifies : nothing
 * comment  : for compatibility	
 */
list region_to_nil_list(region reg)
{
    return(NIL);
}


/* list regions_to_write_regions(list l_reg)
 * input    : a list of regions.
 * output   : the same list of regions converted to a list of write regions.
 * modifies : the regions in l_reg are converted to write regions 
 * comment  : 	
 */
list regions_to_write_regions(list l_reg)
{
    MAP(REGION, reg,
    {
	region_action_tag(reg) = is_action_write; 
    },
	l_reg);
    return(l_reg);
}


/* list regions_read_regions(list l_reg)
 * input    : a list of regions read and write.
 * output   : a list of read regions consisting of the read regions 
 *            of the initial list.
 * modifies : nothing.
 * comment  : sharing between the initial and the returned lists components.	
 */
list regions_read_regions(list l_reg)
{
    list l_read = NIL; 
    
    MAP(REGION, reg,
     {
	 if (action_read_p(region_action(reg))) 
	     l_read = region_add_to_regions(reg, l_read);
     },
	l_reg);
    return(l_read);
}


/* list regions_write_regions(list l_reg)
 * input    : a list of regions read and write.
 * output   : a list of write regions consisting of the write regions 
 *            of the initial list.
 * modifies : nothing.
 * comment  : sharing between the initial and the returned lists components.	
 */
list regions_write_regions(l_reg)
list l_reg;
{
    list l_write = NIL; 
    
    MAP(REGION, reg,
     {
	 if (action_write_p(region_action(reg))) 
	     l_write = region_add_to_regions(reg, l_write);
     },
	l_reg);
    return(l_write);
}


/****************************************************** CREATION OF REGIONS */


/* reference make_regions_reference(entity ent)
 * input    : a variable entity.
 * output   : a reference made with the entity ent, and, in case of an array,
 *            PHI variables as indices.
 * modifies : nothing.
 */
reference make_regions_reference(entity ent)
{
    list regions_ref_inds = NIL, ent_list_dim = NIL;
    int dim;
    type ent_ty = entity_type(ent);
    
    if (type_variable_p(ent_ty))
	ent_list_dim = variable_dimensions(type_variable(ent_ty));
    
    if (! entity_scalar_p(ent))
	for (dim = 1; ent_list_dim != NIL; ent_list_dim = CDR(ent_list_dim), dim++)
	    regions_ref_inds = gen_nconc(regions_ref_inds,
					 CONS(EXPRESSION,
					      make_phi_expression(dim),
					      NIL));
    
    return(make_reference(ent, regions_ref_inds));
}


/* static predicate make_whole_array_predicate(reference ref)
 * input    : a variable, which can be either a scalar or an array.
 * output   : a predicate; its system contains no constraint, but
 *            the base of the latter contains the phi variables if 
 *            the input variable is an array.
 * modifies : nothing;
 * comment  :	
 */
static Psysteme make_whole_array_predicate(entity e)
{
    Psysteme ps = sc_new();
    
    /* No phi if the entity is a scalar */
    if (! entity_scalar_p(e))
    {
	int dim;
	int dim_max = NumberOfDimension(e);

	/* add phi variables in the base */
	for (dim = 1; dim<=dim_max; dim ++)
	{
	    entity phi = make_phi_entity(dim);
	    sc_base_add_variable(ps, (Variable) phi);
	}
	
	/* add array bounds if asked for */
	if (array_bounds_p())	
	    ps = sc_safe_append(ps, 
				sc_safe_normalize(entity_declaration_sc(e)));
    }    
    
    return ps;
}


/* effect make_reference_region(reference ref, tag tac) 
 * input    : a variable reference.
 * output   : a region corresponding to this reference.
 * modifies : nothing.
 */
effect make_reference_region(reference ref, tag tac)
{
    entity e = reference_variable(ref);
    effect reg;
    boolean linear_p = TRUE;
    Psysteme sc;

    /* first make the predicate according to ref */

    ifdebug(3)
    {
	pips_debug(3, "variable : \n%s\n", 
		   words_to_string(words_reference(ref)));
    }
    
    sc = make_whole_array_predicate(e);
    
    /* No predicate if the entity is a scalar. */
    if (! entity_scalar_p(e))
    {
	int idim;
	list ind = reference_indices(ref);

	for (idim = 1; ind != NIL; idim++, ind = CDR(ind))
	{
	    /* For equalities. */
	    expression exp_ind = EXPRESSION(CAR(ind));
	    boolean dim_linear_p;

	    ifdebug(3) {
		pips_debug(3, "addition of equality :\nPHI%d - %s = 0\n",
			   idim, words_to_string(words_expression(exp_ind)));
	    }
	    
	    dim_linear_p = 
		sc_add_phi_equation(sc, exp_ind, idim, IS_EG, PHI_FIRST);
	    pips_debug(3, "%slinear equation.\n", dim_linear_p? "": "non-");
	    linear_p = linear_p && dim_linear_p;	    
	} /* for */
    } /* if */
    
    reg = make_region(reference_dup(make_regions_reference(e)),
		      make_action(tac, UU),
		      make_approximation
		      (linear_p? is_approximation_must : is_approximation_may,
		       UU),
		      sc);

    return(reg);
}    

/* reference make_regions_psi_reference(entity ent)
 * input    : a variable entity.
 * output   : a reference made with the entity ent, and, in case of an array,
 *            PSI variables as indices.
 * modifies : nothing.
 */
reference make_regions_psi_reference(entity ent)
{
    list regions_ref_inds = NIL, ent_list_dim = NIL;
    int dim;
    type ent_ty = entity_type(ent);
    
    if (type_variable_p(ent_ty))
	ent_list_dim = variable_dimensions(type_variable(ent_ty));
    
    if (! entity_scalar_p(ent))
	for (dim = 1; ent_list_dim != NIL; ent_list_dim = CDR(ent_list_dim), dim++)
	    regions_ref_inds = gen_nconc(regions_ref_inds,
					 CONS(EXPRESSION,
					      make_psi_expression(dim),
					      NIL));
    
    return(make_reference(ent, regions_ref_inds));
}


/* effect reference_whole_region(reference ref, tag tac, tap)
 * input    : a variable reference, and the action of the region.
 * output   : a region representing the entire memory space allocated
 *            to the variable reference : the systeme of constraints is
 *            a sc_rn(phi_1, ..., phi_n) except if REGIONS_WITH_ARRAY_BOUNDS
 *            is true. The region is a MAY region.
 * modifies : nothing.
 */
effect reference_whole_region(reference ref, tag tac)
{
    entity e = reference_variable(ref);
    effect new_eff;
    action ac = make_action(tac, UU);
    approximation ap = make_approximation(is_approximation_may, UU);
    
    new_eff = make_region(reference_dup(make_regions_reference(e)), ac, ap, 
			  make_whole_array_predicate(e));
    return(new_eff);
}  

region entity_whole_region(entity e, tag tac)
{
    region new_eff;
    action ac = make_action(tac, UU);
    approximation ap = make_approximation(is_approximation_may, UU);
    
    new_eff = make_region(make_regions_reference(e), ac, ap, 
			  make_whole_array_predicate(e));
    return(new_eff);
}  

/************************************************************ PHI ENTITIES */

/* entity make_phi_entity(int n) 
 * input    : an integer n giving the range of the PHI entity.
 * output   : the entity PHI_n.
 * modifies : phi[] (eventually).
 * comment  : finds or generates an entity representing a region descriptor 
 *            of indice n.
 *            A region descriptor representes the value domain of one dimension 
 *            of an array, this dimension is represented by the indice.
 *            Fortran accepts a maximum of 7 dimensions.
 *
 *            Once a region descriptor is created, it is stored into a static array 
 *            (phi[]); thus, each next time a region descriptor entity is needed, 
 *            it is just picked up in phi[].
 *            If the region descriptor is not in phi[], it may be in the table
 *            where all entities are stored, because of previous computations.
 *            At last, if it is not found there, it is created. 
 */
entity make_phi_entity(int n)
{
    pips_assert("phi index between 1 and 7\n", 1<=n && n<=7);
    
    /* phi indices are between 1 to 7. array indices are between 0 to 6. */
    if (phi[n-1] == entity_undefined)
    {
	entity v;
	char phi_name[5];
	char * s;
	
	(void) strcpy(phi_name, PHI_PREFIX);
	(void) sprintf(phi_name+3,"%d",n);
	s = strdup(concatenate(REGIONS_MODULE_NAME,
			       MODULE_SEP_STRING, phi_name, (char *) NULL));
	
	if ((v = gen_find_tabulated(s, entity_domain)) == entity_undefined) 
	{
	    v = make_entity(s, make_scalar_integer_type(4),
			    make_storage(is_storage_rom, UU),
			    value_undefined);
	}
	phi[n-1] = v;
    }

    return (phi[n-1]);
}

list phi_entities_list(int phi_min, int phi_max)
{
    list l_phi = NIL;
    int i;
    
    for(i=phi_min; i<=phi_max; i++) 
	l_phi = gen_nconc(l_phi, CONS(ENTITY, make_phi_entity(i), NIL));
    
    return(l_phi);
}


/* expression make_phi_expression(int n)
 * input    : the range of the PHI entity to create.
 * output   : an expression made of a PHI variable of range n;
 * modifies : nothing.
 */
expression make_phi_expression(int n)
{
    entity phi;
    reference ref;
    syntax synt;
    expression exp;
    
    phi = make_phi_entity(n);
    ref = make_reference(phi, NIL);
    synt = make_syntax(is_syntax_reference, ref);
    exp = make_expression(synt,  normalized_undefined);
    
    return(exp);
}


/* boolean vect_contains_phi_p(Pvecteur v)
 * input    : a vector
 * output   : TRUE if v contains a PHI variable, FALSE otherwise
 * modifies : nothing
 */
boolean vect_contains_phi_p(Pvecteur v)
{
    for(; !VECTEUR_NUL_P(v); v = v->succ) 
	if (variable_phi_p((entity) var_of(v)))
	    return(TRUE);
    
    return(FALSE);
}


/* bool sc_add_phi_equation(Psysteme sc, expression expr,
 *                       int dim, bool is_eg, bool is_phi_first)
 * input    : a region predicate, an expression caracterizing a PHI variable of
 *            range dim, two booleans indicating if the equation to
 *            create is an equality or an inequality, and this last case, 
 *            the direction of the inequality.
 * output   : TRUE if expr is linear, FALSE otherwise, which means that sc does 
 *            not contain the information associated to expr (MAY region).
 * modifies : adds an equation containing a PHI variable to the system of constraints
 *            given in argument.
 * comment  : The equation can, in fact, be an equality or an inequality :
 *
 *            If it is an equality   (is_eg == TRUE)  :
 *                                     PHI  - expr  = 0
 *            If it is an inequality (is_eg == FALSE) :
 *                   (is_phi_first == FALSE)        expr - PHI  <= 0
 *                   (is_phi_first == TRUE )        PHI  - expr <= 0
 *
 *            The indice of the region descriptor (PHI) is given by "dim".
 *            The equation is added only if the expression (expr) is a normalized one.
 *
 *            Note : the basis of the system is updated.
 */
bool sc_add_phi_equation(Psysteme sc, expression expr, int dim, bool is_eg,
			 bool is_phi_first)
{
    normalized nexpr = NORMALIZE_EXPRESSION(expr);
    
    if (normalized_linear_p(nexpr)) 
    {
	entity phi = make_phi_entity(dim);	
	Pvecteur v1 = vect_new((Variable) phi, VALUE_ONE);
	Pvecteur v2 = normalized_linear(nexpr);
	Pvecteur vect;
	
	if (is_phi_first) 
	    vect = vect_substract(v1, v2);
	else 
	    vect = vect_substract(v2, v1);
	vect_rm(v1);
	/* vect_rm(v2); */
	
	if(!VECTEUR_NUL_P(vect)) 
	{
	    pips_assert("sc_add_phi_equation", !SC_UNDEFINED_P(sc));
	    
	    if (is_eg) 
	    {
		sc_add_egalite(sc, contrainte_make(vect));
	    }
	    else
	    {
		sc_add_inegalite(sc, contrainte_make(vect));
	    }
	    
	    /* The basis of sc has to be updated with the new entities
	       introduced with the constraint addition. */
	    for(; !VECTEUR_NUL_P(vect); vect = vect->succ)
		if(vect->var != TCST)
		    sc->base = vect_add_variable(sc->base, vect->var);
	    
	    sc->dimension = vect_size(sc->base);
	}

	return(TRUE);
    }
    
    /* the expression is not linear */
    return(FALSE);
}


/* void phi_first_sort_base(Pbase base)
 * input    : the base of a region
 * output   : nothing.
 * modifies : *ppbase.
 * comment  : sorts the base such that the PHI variables come first.	
 *            it is usefull for the convex_hull, to preserve the form
 *            of the predicate.   
 */
void phi_first_sort_base(Pbase *ppbase)
{
    Pvecteur v, v_pred = VECTEUR_NUL;
    boolean first = TRUE, phi_only = TRUE;

    ifdebug(8) {
	pips_debug(8, "initial base :\n");
	vect_debug(*ppbase);
    }

    for( v = (Pvecteur) *ppbase; !VECTEUR_NUL_P(v); v = v->succ) 
    {
	/* if a PHI variable is encountered, and if it is not the first term
         * of the base, or it is not preceded only by PHI variables, then 
	 * we put it at the head of the vector. */
	if (strncmp(entity_name((entity) v->var), REGIONS_MODULE_NAME, 10) == 0) 
	{
	    if (phi_only) 
	    {
		v_pred = v;
		if (first) first = FALSE;
	    }
	    else 
	    {
		v_pred->succ = v->succ;
		v->succ = (Pvecteur) *ppbase;
		*ppbase = v;
		v = v_pred;
	    }
	}
	else 
	{
	    if (phi_only) 
	    {
		first = FALSE;
		phi_only = FALSE;
	    }
	    v_pred = v;
	}

    } /* for */

    ifdebug(8) {
	pips_debug(8, "final base :\n");
	vect_debug(*ppbase);
    }
}


/* int base_nb_phi(Pbase b)
 * input    : a base
 * output   : the number of phi variables in this base.
 * modifies : nothing.
 * comment  :	
 */
int base_nb_phi(Pbase b)
{
    int n_phi = 0;

    for(; !VECTEUR_NUL_P(b); b = b->succ)
    {
	if (!term_cst(b))
	    if (variable_phi_p((entity) var_of(b))) n_phi++;
    }

    return(n_phi);
}

/*********************************************************************************/
/* PSY ENTITIES                                                                  */
/*********************************************************************************/

/* entity make_psi_entity(int n) 
 * input    : an integer n giving the range of the PHI entity.
 * output   : the entity PHI_n.
 * modifies : psi[] (eventually).
 * comment  : finds or generates an entity representing a region descriptor 
 *            of indice n.
 *            A region descriptor representes the value domain of one dimension 
 *            of an array, this dimension is represented by the indice.
 *            Fortran accepts a maximum of 7 dimensions.
 *
 *            Once a region descriptor is created, it is stored into a static array 
 *            (psi[]); thus, each next time a region descriptor entity is needed, 
 *            it is just picked up in psi[].
 *            If the region descriptor is not in psi[], it may be in the table
 *            where all entities are stored, because of previous computations.
 *            At last, if it is not found there, it is created. 
 */
entity make_psi_entity(int n)
{
    pips_assert("psy index between 1 and 7\n", 1<=n && n<=7);

    /* psi indices are between 1 to 7. Array indices are between 0 to 6. */
    if (psi[n-1] == entity_undefined) 
    {
	entity v;
	char psi_name[5];
	char * s;

	(void) strcpy(psi_name, PSI_PREFIX);
	(void) sprintf(psi_name+3,"%d",n);
	s = strdup(concatenate(REGIONS_MODULE_NAME,
			       MODULE_SEP_STRING, psi_name, (char *) NULL));
	
	if ((v = gen_find_tabulated(s, entity_domain)) == entity_undefined) 
	{
	    v = make_entity(s,
			    make_scalar_integer_type(4),
			    make_storage(is_storage_rom, UU),
			    value_undefined);
	}
	
	psi[n-1] = v;
    }
    return (psi[n-1]);
}


list psi_entities_list(int psi_min, int psi_max)
{
    list l_psi = NIL;
    int i;

    for(i=psi_min; i<=psi_max; i++) 
	l_psi = gen_nconc(l_psi, CONS(ENTITY, make_psi_entity(i), NIL));
    
    return(l_psi);
}


/* expression make_psi_expression(int n)
 * input    : the range of the PHI entity to create.
 * output   : an expression made of a PHI variable of range n;
 * modifies : nothing.
 */
expression make_psi_expression(int n)
{
    return(make_expression(make_syntax(is_syntax_reference,
				       make_reference(make_psi_entity(n), NIL)),
			   normalized_undefined));
}


/* int base_nb_psi(Pbase b)
 * input    : a base
 * output   : the number of psi variables in this base.
 * modifies : nothing.
 * comment  :	
 */
int base_nb_psi(Pbase b)
{
    int n_psi = 0;

    for(; !VECTEUR_NUL_P(b); b = b->succ)
	if (variable_psi_p((entity) var_of(b))) n_psi++;

    return(n_psi);
}

/*********************************************************************************/
/* PHI AND PSY ENTITIES                                                          */
/*********************************************************************************/


/* list region_phi_cfc_variables(effect reg)
 * input    : a region
 * output   : the list of variables that define the PHI or PSI variables, directly
 *            or indirectly
 * modifies : nothing
 * comment  : "cfc" stands for "Composante Fortement Connexe"; because
 *            if systems are considered as graphs which vertices are 
 *            the constraints and which edges connect constraints 
 *            concerning at least one common variable, then this
 *            function searches for the variables contained in
 *            the strongly connected component that also contains
 *            the PHI or PSI variables.
 */
list region_phi_cfc_variables(region reg)
{
    list l_ent;
    
    if (region_scalar_p(reg))
	return(NIL);

    if (psi_region_p(reg))
	l_ent = psi_entities_list(1,NumberOfDimension(region_entity(reg)));
    else
	l_ent = phi_entities_list(1,NumberOfDimension(region_entity(reg)));

    /* ps_reg has no particularity. We just scan its equalities and inequalities 
     * to find which variables are related to the PHI variables */ 
    return(region_entities_cfc_variables(reg, l_ent));
}

/* void psi_to_phi_region(effect reg)
 * input    : a PSI region
 * output   : the same region, using PHI variables.
 * modifies : reg
 * comment  : also changes the reference.
 */
void psi_to_phi_region(region reg)
{
    int i, ndims = NumberOfDimension(region_entity(reg));
    reference reg_ref;

    /* if not a scalar region */
    if (ndims != 0)
    {
	Psysteme ps = region_system(reg);;
	for(i = 1; i<= ndims; i++) 
	    sc_variable_rename(ps, (Variable) make_psi_entity(i), 
			       (Variable) make_phi_entity(i));
    }
    
    reg_ref = make_regions_reference(region_entity(reg)); 
    reference_variable(effect_reference(reg)) = entity_undefined;
    gen_free(effect_reference(reg));
    effect_reference(reg) = reg_ref;
}

/* void phi_to_psi_region(effect reg)
 * input    : a PHI region
 * output   : the same region, using PSI variables.
 * modifies : reg
 * comment  : also changes the reference.
 */
void phi_to_psi_region(region reg)
{
    int i, ndims = NumberOfDimension(region_entity(reg));
    reference reg_ref;

    /* if not a scalar region */
    if (ndims != 0)
    {
	Psysteme ps = region_system(reg);;
	for(i = 1; i<= ndims; i++) 
	    sc_variable_rename(ps, (Variable) make_phi_entity(i), 
			       (Variable) make_psi_entity(i)); 
    }
    reg_ref = make_regions_psi_reference(region_entity(reg)); 
    reference_variable(effect_reference(reg)) = entity_undefined;
    gen_free(effect_reference(reg));
    effect_reference(reg) = reg_ref;
}


/* boolean psi_region_p(effect reg)
 * input    : a region.
 * output   : TRUE if it is a PSI region
 * modifies : nothing
 * comment  : tests if the first indice of the reference is a PSI variable.
 */
boolean psi_region_p(region reg)
{
    pips_assert("array region (not scalar)\n", !region_scalar_p(reg));

    if (variable_psi_p(reference_variable
		       (syntax_reference
			(expression_syntax
			 (EXPRESSION
			  (CAR(reference_indices(region_reference(reg))))))))) 
	return TRUE;
    else 
	return FALSE;    
}

/************************************************************* PROPERTIES */

static bool exact_regions_property = FALSE;
static bool must_regions_property = FALSE;
static bool array_bounds_property = FALSE;
static bool disjunct_property = FALSE;
static bool op_statistics_property = FALSE;

bool exact_regions_p()
{
    return exact_regions_property;
}

bool must_regions_p()
{
    return must_regions_property;
}

bool array_bounds_p()
{
    return array_bounds_property;
}

bool disjunct_regions_p()
{
    return disjunct_property;
}

bool op_statistics_p()
{
    return op_statistics_property;
}

void reset_op_statistics()
{
    /* reset_proj_op_statistics(); */
    /* reset_binary_op_statistics(); */
}

void get_regions_properties()
{
    exact_regions_property = get_bool_property("EXACT_REGIONS");
    must_regions_property = get_bool_property("MUST_REGIONS");
    array_bounds_property = get_bool_property("REGIONS_WITH_ARRAY_BOUNDS");
    disjunct_property = get_bool_property("DISJUNCT_REGIONS");
    op_statistics_property = get_bool_property("REGIONS_OP_STATISTICS");
    if (op_statistics_property) reset_op_statistics();
}

void get_in_out_regions_properties()
{
    exact_regions_property = TRUE;
    must_regions_property = TRUE;
    array_bounds_property = get_bool_property("REGIONS_WITH_ARRAY_BOUNDS");
    disjunct_property = get_bool_property("DISJUNCT_IN_OUT_REGIONS");
    op_statistics_property = get_bool_property("REGIONS_OP_STATISTICS");
    if (op_statistics_property) reset_op_statistics();
}

 
/********************************** COMPARISON FUNCTIONS FOR QSORT, AND SORT */

static Pbase base_for_compare = BASE_NULLE;
static Pbase no_phi_base_for_compare = BASE_NULLE;

static void set_bases_for_compare(Pbase sorted_base)
{
    Pbase b = sorted_base;
    base_for_compare = sorted_base;
    
    for(;!BASE_NULLE_P(b) && variable_phi_p((entity) var_of(b)); b = b->succ);
    
    no_phi_base_for_compare = b;    
}

static void reset_bases_for_compare()
{
    base_for_compare = BASE_NULLE;
    no_phi_base_for_compare = BASE_NULLE;
}

static int compare_region_constraints(Pcontrainte *pc1, Pcontrainte *pc2,
				      bool equality_p); 
static int compare_region_vectors(Pvecteur *pv1, Pvecteur *pv2);
static int compare_region_entities(entity *pe1, entity *pe2);

static int compare_region_equalities(Pcontrainte *pc1, Pcontrainte *pc2)
{
    return(compare_region_constraints(pc1, pc2, TRUE));
}

static int compare_region_inequalities(Pcontrainte *pc1, Pcontrainte *pc2)
{
    return(compare_region_constraints(pc1, pc2, FALSE));
}


/* void region_sc_sort(Psysteme sc, Pbase sorted_base)
 * input    : the predicate of a region, a base giving the priority order
 *            of the variables (phi variables first).
 * output   : nothing.
 * modifies : sc
 * comment  : sorts each constraint of sc using compare_regions_vectors, and 
 *            sorts the constraints between them.
 */
void region_sc_sort(Psysteme sc, Pbase sorted_base)
{
    debug_on("REGIONS_PRETTYPRINT_DEBUG_LEVEL");
    ifdebug(1)
    {
	pips_debug(1, "\nbefore sort:\n");
	sc_syst_debug(sc);
    }
    sc_vect_sort(sc, compare_region_vectors);
    ifdebug(1)
    {
	pips_debug(1, "\nafter sc_vect_sort:\n");
	sc_syst_debug(sc);
    }
    sc->inegalites = region_constraints_sort(sc->inegalites, sorted_base, FALSE);
    ifdebug(1)
    {
	pips_debug(1, "\nafter inequality sort:\n");
	sc_syst_debug(sc);
    }
    sc->egalites = region_constraints_sort(sc->egalites, sorted_base, TRUE);
    ifdebug(1)
    {
	pips_debug(1, "\nafter sort:\n");
	sc_syst_debug(sc);
    }
    debug_off();
}


/* Pcontrainte region_constraints_sort(Pcontrainte c, Pbase sorted_base) 
 * input    : a region list of contraints, a base giving the priority order
 *            of the variables (phi variables first).
 * output   : the sorted list of constraints.
 * modifies : c.
 * comment  : sorts the constraints using compare_region_constraints.
 */
Pcontrainte 
region_constraints_sort(
    Pcontrainte c, 
    Pbase sorted_base, 
    bool equality_p)
{
    set_bases_for_compare(sorted_base);    
    c = constraints_sort_with_compare
	(c, BASE_NULLE, equality_p? 
	 compare_region_equalities: compare_region_inequalities);
    reset_bases_for_compare();
    return(c);           
}


/* Pbase region_sorted_base_dup(effect reg)
 * input    : a region
 * output   : a sorted copy of the base of its predicate.
 * modifies : nothing.
 * comment  : sorts the base using compare_regions_vectors.
 */
Pbase region_sorted_base_dup(region reg)
{
    Pbase sbase = base_dup(sc_base(region_system(reg)));    
    vect_sort_in_place( &sbase, compare_region_vectors);
    return(sbase);    
}


/* static int compare_region_constraints(Pcontrainte *pc1, *pc2) 
 * input    : two constraints from a reigon predicate.
 * output   : an integer for qsort: <0 if c1 must come first, >0 if c2 must come
 *            first, 0 if the two constraints are equivalent.
 * modifies : nothing
 * comment  : The criterium for the sorting is based first on the number of phi 
 *            variables in the constraints, second, on the order between variables
 *            given by two static variables that must be initialized first.
 *            (see a few screens before).
 */
static int compare_region_constraints(Pcontrainte *pc1, Pcontrainte *pc2,
				      bool equality_p)
{
    Pvecteur
	v1 = (*pc1)->vecteur,
	v2 = (*pc2)->vecteur;

    int nb_phi_1 = base_nb_phi((Pbase) v1),
        nb_phi_2 = base_nb_phi((Pbase) v2);
    
    Pbase b = BASE_NULLE;
    Value val_1, val_2;

    /* not the same number of phi variables */
    if (nb_phi_1 != nb_phi_2) 
    {
	/* one has no phi variables */
	if ((nb_phi_1 == 0) || (nb_phi_2 == 0))
	    return( (nb_phi_1 ==0)? +1 : -1 );
	/* both have phi variables */
	else
	    return( (nb_phi_1 > nb_phi_2)? +1 : -1);
    }
    
    if (nb_phi_1 == 0)
	/* both have no phi variables : the comparison basis only consists of 
	   variables other than phi variables */ 
	b = no_phi_base_for_compare;
    else 
    {
	/* particular treatment for phi variables */
	b = base_for_compare;
	for(; b!= no_phi_base_for_compare; b = b->succ) 
	{
	    val_1 = vect_coeff(var_of(b), v1);
	    val_2 = vect_coeff(var_of(b), v2);
	    if (value_ne(val_1,val_2)) {
		if (value_zero_p(val_1))
		    return(1);
		if (value_zero_p(val_2))
		    return(-1);	    
		return VALUE_TO_INT(value_minus(val_1,val_2));
	    }
	}	
    }
    

    if(equality_p)
    {
	int passe;
	Pvecteur coord, v;

	for(passe = 1; passe <= 2; passe++)
	{
	    int positive_terms = 0, negative_terms = 0;

	    v = (passe == 1) ? v1 : v2;
	    for(coord = v; !VECTEUR_NUL_P(coord); coord = coord->succ)
	    {
		if(vecteur_var(coord)!= TCST)
		{
		    if(value_pos_p(vecteur_val(coord) ))
			positive_terms++;
		    else
			negative_terms++;
		}
	    }
	    
	    if(negative_terms > positive_terms)
		vect_chg_sgn(v);
	    else
		if(negative_terms == positive_terms)
		    /* entities are already sorted in v */
		    if ((vecteur_var(v) != TCST) && 
			value_neg_p(vecteur_val(v)))
			vect_chg_sgn(v);
	}
    }
    
    /* variables other than phis are handled */
    for(; !BASE_NULLE_P(b); b = b->succ) {
	val_1 = vect_coeff(var_of(b), v1);
	val_2 = vect_coeff(var_of(b), v2);
	if (value_ne(val_1,val_2))
	{
	    if (value_zero_p(val_1))
		return(1);
	    if (value_zero_p(val_2))
		return(-1);	
	    return VALUE_TO_INT(value_minus(val_1,val_2));
	}
    }
    
    val_1 = vect_coeff(TCST, v1);
    val_2 = vect_coeff(TCST, v2);
    
    if (value_ne(val_1,val_2))
    {
	if (value_zero_p(val_1))
	    return(1);
	if (value_zero_p(val_2))
	    return(-1);	
	return VALUE_TO_INT(value_minus(val_1,val_2));
    }
    
    return(0);
}


/* static int compare_region_vectors(Pvecteur *pv1, *pv2)
 * input    : two vectors from a region predicate
 * output   : an integer for qsort
 * modifies : nothing
 * comment  : phi variables come first, the other variables are alphabetically sorted.
 */
static int compare_region_vectors(Pvecteur *pv1, Pvecteur *pv2)
{
    return(compare_region_entities((entity*)&var_of(*pv1),
				   (entity*)&var_of(*pv2)));
}


/* static int compare_region_entities(entity *pe1, *pe2)
 * input    : two entities
 * output   : a integer for qsort
 * modifies : nothing
 * comment  : phi variables come first, the other variables are alphabetically sorted.
 */
static int compare_region_entities(entity *pe1, entity *pe2)
{
    int phi_1 = FALSE, phi_2 = FALSE,
	null_1 = (*pe1==(entity)NULL),
	null_2 = (*pe2==(entity)NULL);
    
    if (null_1 && null_2)
	return(0);
    
    if (!null_1) phi_1 = variable_phi_p(*pe1);
    if (!null_2) phi_2 = variable_phi_p(*pe2);

    if (null_1 && phi_2)
	return(1);

    if (null_2 && phi_1) 
	return (-1);

    if (null_1 || null_2) 
	return(null_2-null_1);

    /* if both entities are phi variables, or neither one */
    if (phi_1 - phi_2 == 0) 
	return(strcmp(entity_name(*pe1), entity_name(*pe2)));
    
    /* if one and only one of the entities is a phi variable */ 
    else
	return(phi_2 - phi_1);    
}


/********************************** MANIPULATION OF ENTITIES AND VARIABLES */


Psysteme entity_declaration_sc(entity e)
{
    Psysteme sc = sc_new();
    list ldim = variable_dimensions(type_variable(entity_type(e)));
    int dim;

    for (dim = 1; !ENDP(ldim); dim++, ldim = CDR(ldim))
    {
	expression dl, du;
	
	dl = dimension_lower(DIMENSION(CAR(ldim)));
	ifdebug(8) {
	    debug(8, "entity_declaration_sc",
		  "addition of inequality :\n%s - PHI%d <= 0\n",
		  words_to_string(words_expression(dl)), dim);
	}
	sc_add_phi_equation(sc, dl, dim, NOT_EG, NOT_PHI_FIRST);
	
	du = dimension_upper(DIMENSION(CAR(ldim)));
	ifdebug(8) {
	    debug(8, "entity_declaration_sc",
		  "addition of inequality :\nPHI%d - %s <= 0\n",
		  dim, words_to_string(words_expression(du)));
	}
	sc_add_phi_equation(sc, du, dim, NOT_EG, PHI_FIRST);
	
    } /* for */ 
    return(sc);
}

/* list variables_to_int_variables(list l_var) 
 * input    : a list of variables
 * output   : a list containing the intermediate variables corresponding to 
 *            initial variables, in the same order.
 * modifies : nothing.
 * comment  :	
 */
list variables_to_int_variables(list l_var)
{
    list l_int = NIL;

    MAP(ENTITY, e,
     {
	 entity e_int = entity_to_intermediate_value(e);
	 list l_tmp = CONS(ENTITY, e_int, NIL);
	 l_int = gen_nconc(l_int, l_tmp);/* to be in the same order as l_var */
	 debug(8, "", "%s -> %s\n", 
	       (e == (entity) TCST) ? "TCST" : pips_user_value_name(e),
	       (e_int == (entity) TCST) ? "TCST" : pips_user_value_name(e_int));
     },
	 l_var);
    
    return(l_int);
}


/* list variables_to_old_variables(list l_var) 
 * input    : a list of variables
 * output   : a list containing the old variables corresponding to 
 *            initial variables, in the same order.
 * modifies : nothing.
 * comment  :	
 */
list variables_to_old_variables(list l_var)
{
    list l_old = NIL;

    MAP(ENTITY, e, 
     {
	 entity e_old = entity_to_old_value(e);
	 list l_tmp = CONS(ENTITY, e_old, NIL);
	 l_old = gen_nconc(l_old, l_tmp); /* to be in the same order as l_var */
	 debug(8, "", "%s -> %s\n", 
	       (e == (entity) TCST) ? "TCST" : pips_user_value_name(e),
	       (e_old == (entity) TCST) ? "TCST" : pips_user_value_name(e_old));
     },
	 l_var);
    
    return(l_old);
}



/* Psysteme sc_list_variables_rename(Psysteme sc, list l_var, l_var_new)
 * input    : a system of constraints, a list of variables to be renamed,
 *            and the list of the new names.
 * output   : the system in which the variables are renamed.
 * modifies : the initial system.
 * comment  :	
 */
Psysteme sc_list_variables_rename(Psysteme sc, list l_var, list l_var_new)
{
    list ll_var_new = l_var_new;

    MAP(ENTITY, e, 
     {
	 entity e_new = ENTITY(CAR(ll_var_new));
	 sc = sc_variable_rename(sc, (Variable) e, (Variable) e_new);	
	 ll_var_new = CDR(ll_var_new);
     },
	 l_var);
    return(sc);
    
}



/* list function_formal_parameters(entity func)
 * input    : an entity representing a function.
 * output   : the unsorted list (of entities) of parameters of the function "func".
 * modifies : nothing.
 * comment  : Made from "entity_to_formal_integer_parameters()" that considers 
 *            only integer variables.
 */
list function_formal_parameters(entity func)
{
    list formals = NIL;
    list decl = list_undefined;

    pips_assert("func must be a function",entity_module_p(func));

    decl = code_declarations(entity_code(func));
    MAP(ENTITY, e, 
     {
	 if(storage_formal_p(entity_storage(e)))
	     formals = CONS(ENTITY, e, formals);
     },
	 decl);
    
    return (formals);
}


/* char *func_entity_name(entity e)
 * input    : an entity representing a function.
 * output   : the name of the function represented by the entity given in argument.
 * modifies : nothing.
 */
char *func_entity_name(entity e)
{
    return((e == entity_undefined) ? "TCST" : entity_name(e));
}




/* bool same_common_variables_p(entity e1, e2)
 * output   : returns TRUE if both entities, which are common variables, 
 *            occupy the same place in a COMMON, and, also, have
 *            the same shape. 
 * modifies : nothing .
 * comment  :	
 *            Four comparisons have to be made :
 *                1. offset in the COMMON
 *                2. Fortran type
 *                3. number of dimensions
 *                4. dimensions size (in case of arrays)
 *
 *            Note : this function is used with common variables of the same 
 *            COMMON in two different subroutines.
 */
bool same_common_variables_p(entity e1, entity e2)
{
    bool equal = FALSE;
    storage st1, st2;
    int offs1, offs2;

    ifdebug(5) {
	pips_debug(5, "entities %s and %s\n", entity_name(e1),entity_name(e2));
    }

    st1 = entity_storage(e1);
    st2 = entity_storage(e2);
    if ((! (storage_tag(st1) == is_storage_ram)) ||
	(! (storage_tag(st2) == is_storage_ram))) {
	user_error("same_common_array_p", "not COMMON entities : %s or %s",
		   entity_name(e1), entity_name(e2));
    }
    
    offs1 = ram_offset(storage_ram(st1));
    offs2 = ram_offset(storage_ram(st2));

    /* Same offset in the COMMON */
    if (offs1 == offs2) {
	type ty1, ty2;
	basic basic1, basic2;
	
	ty1 = entity_type(e1);
	ty2 = entity_type(e2);
	if ((! (type_tag(ty1) == is_type_variable)) ||
	    (! (type_tag(ty2) == is_type_variable))) {
	    pips_error("same_common_variables__p",
		       "arrays entities with not variable type : %s %s",
		       entity_name(e1), entity_name(e2));
	}

	basic1 = variable_basic(type_variable(ty1));
	basic2 = variable_basic(type_variable(ty2));

	/* Same fortran type. */
	if (SizeOfElements(basic1) == SizeOfElements(basic2)) {
	    int nb_dims1 = NumberOfDimension(e1);
	    int nb_dims2 = NumberOfDimension(e2);

	    /* Same number of dimensions. */
	    if (nb_dims1 == nb_dims2) {
		int count_dim;
		for (count_dim = 1, equal = TRUE;
		     (count_dim <= nb_dims1) && (equal); count_dim++) {
		    int size1 = SizeOfIthDimension(e1, count_dim);
		    int size2 = SizeOfIthDimension(e2, count_dim);

		    /* Same dimension size. */
		    if (size1 != size2) {
			equal = FALSE;
		    }
		}
	    }
	}
    }
    return(equal);
}


/*************************** MANIPULATION OF PRECONDITIONS AND TRANSFORMERS */


/* Psysteme sc_loop_proper_precondition(loop l)
 * input    : a loop
 * output   : a Psysteme representing the proper precondition of the loop,
 *            e.g. the constraints induced by the loop range.
 * modifies : nothing.
 * comment  : should not be there !
 */
Psysteme sc_loop_proper_precondition(loop l)
{
    entity i = loop_index(l);
    Psysteme sc;
    transformer loop_prec;
    transformer loop_body_trans = load_statement_transformer(loop_body(l));

    loop_prec = make_transformer(NIL, 
       make_predicate((char *) sc_rn((Pbase)
				     vect_new((Variable) i, VALUE_ONE))));
    loop_prec = 
	add_index_range_conditions(loop_prec, 
				   i, 
				   loop_range(l), 
				   loop_body_trans);
    
    sc = predicate_system(transformer_relation(loop_prec));
    predicate_system_(transformer_relation(loop_prec)) = 
	newgen_Psysteme(SC_UNDEFINED);
    gen_free(loop_prec);
    
    return sc;
}

/******************************************************************** MISC */

/* is the context empty? 
 */
bool 
empty_convex_context_p(transformer context)
{
    Psysteme sc_context;

    if (transformer_undefined_p(context))
    {
	/* this happens to the CONTINUE statement of the exit node
	 * even if unreachable. Thus transformer are not computed,
	 * orderings are not set... however gen_multi_recurse goes there.
	 * I just store NIL, what seems reasonnable an answer.
	 * It seems to be sufficient for other passes. 
	 * I should check that it is indeed the exit node?
	 * FC (RK).
	 */
	pips_debug(3, "undefined contexted found, returned as empty\n");
	return TRUE;
    }

    /* else 
     */

    sc_context = predicate_system(transformer_relation(context));
    return sc_empty_p(sc_context);
}

string 
region_to_string(effect reg)
{
    return strdup("[region_to_string] not more implemented\n");
}



