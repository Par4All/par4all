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
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "conversion.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "transformer.h"
#include "semantics.h"

#include "effects-generic.h"
#include "effects-convex.h"

#define IS_EG true
#define NOT_EG false

#define PHI_FIRST true
#define NOT_PHI_FIRST false

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

static entity phi[NB_MAX_ARRAY_DIM]; /* phi entities */
static entity psi[NB_MAX_ARRAY_DIM]; /* psi entities */
static entity rho[NB_MAX_ARRAY_DIM]; /* rho entities */

#ifndef bool_undefined

#define bool_undefined ((bool) (-15))
#define bool_undefined_p(b) ((b)==bool_undefined)
#endif

bool add_precondition_to_scalar_convex_regions = bool_undefined;

void regions_init()
{
    int i;
    for(i=0; i<NB_MAX_ARRAY_DIM; i++)
    {
	phi[i] = entity_undefined;
	psi[i] = entity_undefined;
	rho[i] = entity_undefined;
    }
    for(i=0; i<BETA_MAX; i++)
    {
	beta[i] = entity_undefined;
    }
    set_region_interprocedural_translation();
    add_precondition_to_scalar_convex_regions = false;
}

void regions_end()
{
    reset_region_interprocedural_translation();
    add_precondition_to_scalar_convex_regions = bool_undefined;

}

/******************************* REGIONS AND LISTS OF REGIONS MANIPULATION */


region region_dup(region reg)
{
    region new_reg;
    debug_region_consistency(reg);

    pips_assert("region cell must be a reference\n",
		cell_reference_p(effect_cell(reg)));

    new_reg = copy_effect(reg);

    pips_assert("The copied region has the same persistent property as sthe input region",
		cell_tag(effect_cell(new_reg))==cell_tag(effect_cell(new_reg)));


    debug_region_consistency(new_reg);
    pips_assert("No sharing of references",
		region_any_reference(reg) != region_any_reference(new_reg));
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

    FOREACH(EFFECT, eff, l_reg)
      {
	/* build last to first */
	effect n_eff = region_dup(eff);
	l_reg_dup = CONS(EFFECT, n_eff, l_reg_dup);
      }

    /* and the order is reversed */
    l_reg_dup =  gen_nreverse(l_reg_dup);

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
	region_free(reg);
    }, l_reg);
    gen_free_list(l_reg);
}

/* (void) region_free(effect reg)
 * input    : a region
 * output   : nothing !
 * modifies : frees region
 * Unless function free_effect(), this function overrides the persistant attribute
 * which may exist at the cell level. Macro free_region() does not exist anymore.
 */
void region_free(region reg)
{
    debug_region_consistency(reg);

    pips_assert("region cell must be a reference\n",
		cell_reference_p(effect_cell(reg)));

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

/* adapted from transformer_normalize... which may have changed since
 * then...
 *
 * FI: Normalization and redundancy elimination are often considered
 * equivalent and used accordingly. Normalization is much more
 * expensive: its purpose is not only to minimize the number of
 * constraints, but also to rewrite them in a as canonical as possible
 * way to simplify non-regression tests and to be as user-friendly as
 * possible. Currently, region_sc_normaliza() is likely to be called instead of region_sc_redundancy_elimination(), if ever this function exists.
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
	    sc_reg = sc_safe_elim_redund(sc_reg);
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
	    // FI: not really effective
	    // sc_reg = sc_bounded_normalization(sc_reg);
	    // sc_reg = sc_bounded_normalization(sc_reg);
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
	    pips_internal_error("unknown level %d", level);
	}

	if (SC_EMPTY_P(sc_reg))
	    sc_reg = sc_empty(b);
	else
	    base_rm(b);

	sc_reg->dimension = vect_size(sc_reg->base);
    }
    return(sc_reg);

}

Psysteme cell_system_sc_append_and_normalize(Psysteme sc1, Psysteme sc2, int level)
{
  Psysteme sc;
  Psysteme sc22 = sc_dup(sc2);


  sc = sc1;
  ifdebug(1)
    {
      pips_debug(1, "sc1 and sc2: \n");
      sc_syst_debug(sc1);
      sc_syst_debug(sc2);
      pips_assert("sc1 must be consistent", sc_weak_consistent_p(sc1));
      pips_assert("sc2 must be consistent", sc_weak_consistent_p(sc22));
    }
  sc22 = sc_safe_normalize(sc22);
  ifdebug(1)
    {
      pips_debug(1, "sc22: \n");
      sc_syst_debug(sc22);
    }
  sc = sc_safe_append(sc_safe_normalize(sc), sc22);
  ifdebug(1)
    {
      assert(sc_weak_consistent_p(sc));
    }
  if (level!=-1)
    sc = region_sc_normalize(sc, level);

  ifdebug(1)
    {
      assert(sc_weak_consistent_p(sc));
      pips_debug(1, "returning sc: \n");
      sc_syst_debug(sc);
    }
  sc_rm(sc22);

  return sc;
}



/* void region_sc_append(region reg, Psysteme sc, bool nredund_p)
 * input    : a region and a Psysteme
 * output   : nothing.
 * modifies : the initial region to which a copy of sc is appended.
 * comment  : add to the predicate of the region a copy of the system sc.
 *            redundancies are then eliminated if nredund_p is true.
 * bug fixed 13/04/2000 CA/FC: sc MUST NOT be modified...
 */
void region_sc_append_and_normalize(region reg, Psysteme sc, int level)
{
  Psysteme sc_reg = region_system(reg);

  region_system_(reg) = cell_system_sc_append_and_normalize(sc_reg, sc, level);
}

/* void regions_sc_append(list l_reg, Psysteme sc, bool arrays_only,
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
		       bool arrays_only, bool scalars_only,
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
      FOREACH(EFFECT, reg, l_reg)
	    {
		bool scalar_p = region_scalar_p(reg);

		if ((scalar_p && !arrays_only) || (!scalar_p && !scalars_only))
		{
		  if(store_effect_p(reg)) {
		    region_sc_append_and_normalize(reg, sc, level);
		    if (!region_empty_p(reg))
		      l_res = region_add_to_regions(reg, l_res);
		    else
		      region_free(reg);
		  }
		  else {
		    l_res = region_add_to_regions(reg, l_res);
		  }
		}
		else
		{
		    l_res = region_add_to_regions(reg, l_res);
		}
	    }
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
    l_reg = regions_sc_append_and_normalize(l_reg, sc, true, false, level);
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

/* void regions_sc_append(list l_reg, Psysteme sc, bool arrays_only,
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
		       bool arrays_only, bool scalars_only,
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
    l_reg = regions_sc_append(l_reg, sc, false, false, nredund_p);
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
    l_reg = regions_sc_append(l_reg, sc, false, true, nredund_p);
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
    l_reg = regions_sc_append(l_reg, sc, true, false, nredund_p);
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
 * comment  : in the array regions of l_reg, the variable old_entity
 *            that appears in the predicates is replaced with new_entity.
 */
void all_regions_variable_rename(list l_reg,
				 entity old_entity,
				 entity new_entity)
{
  FOREACH(EFFECT, reg, l_reg) {
    if(store_effect_p(reg)) {
      Psysteme sc_reg = region_system(reg);

      if (!sc_empty_p(sc_reg) && !sc_rn_p(sc_reg)) {
	Pbase b = sc_base(sc_reg);
	ifdebug(8){
	  pips_debug(8, "current system: \n");
	  sc_syst_debug(sc_reg);
	}

	/* FI: I do not know if I'm fixing a bug or if I'm correcting a symptom */
	/* This is done to ocmpute the OUT regions of the routine
	   COMPUTE_PD3 in the NAS beanchmark md. (Request Alain Muller) */
	if(base_contains_variable_p(b, (Variable) new_entity)) {
	  Pbase nb = sc_to_minimal_basis(sc_reg);
	  if(base_contains_variable_p(nb, (Variable) new_entity)) {
	    /* We are in trouble because we cannot perform a simple renaming */
	    /* The caller may reuse values already used in the
	       system. This is the case for muller01.f because the
	       precondition and the region rightly used I#old. Rightly,
	       but uselessly in that case. */
	    /* new_entity must/can be projected first although this is
	       dangerous when dealing with MUST/MAY regions (no other
	       idea, but use of temporary values t# instead of i#old in
	       caller, ).. */
	    region_exact_projection_along_variable(reg, new_entity);
	    sc_reg = region_system(reg);
	    pips_user_warning("Unexpected variable renaming in a region\n");
	  }
	  else {
	    base_rm(b);
	    sc_base(sc_reg) = nb;
	    sc_dimension(sc_reg) = base_dimension(nb);
	  }
	}

	sc_reg = sc_variable_rename(sc_reg, (Variable) old_entity,
				    (Variable) new_entity);
      }
    }
    else {
      /* FI: for the time being, other kinds of regions do not have
	 descriptors */
      ;
    }
  }
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
	    pips_internal_error("conflict between e1=%s and e2=%s",
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
list regions_to_nil_list(region reg1 __attribute__ ((unused)),
			 region reg2 __attribute__ ((unused)))
{
    return(NIL);
}

/* list region_to_nil_list(reg)
 * input    : a region
 * output   : an empty list of regions
 * modifies : nothing
 * comment  : for compatibility
 */
list region_to_nil_list(region reg __attribute__ ((unused)))
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
      /* Memory leak */
      region_action(reg) = make_action_write_memory();
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


reference make_pointed_regions_reference(entity ent, bool indexed_p)
{
  list regions_ref_inds = NIL;
  int dim;
  type ent_ty = ultimate_type(entity_type(ent));
  int d = -1;

  if (type_variable_p(ent_ty)) {
    /* ent_list_dim = variable_dimensions(type_variable(ent_ty)); */
    if(pointer_type_p(ent_ty)) {
      /* Are we dealing with the pointer or with the pointed area? */
      if(indexed_p) {
	/* Only one dimension for pointers when arithmetic is used:
	   let's hope that subscripts are combined! Unless the pointed type has dimensions... */
	type pt = basic_pointer(variable_basic(type_variable(ent_ty)));

	d = type_depth(pt);
	if(d==0)
	  d = 1;
      }
      else {
	d = 0;
      }
    }
    else {
      /* Beware, ultimate type does not respect dimensions! */
      d = type_depth(entity_type(ent));
    }
  }

  for (dim = 1; dim <= d; dim++)
    regions_ref_inds = gen_nconc(regions_ref_inds,
				 CONS(EXPRESSION,
				      make_phi_expression(dim),
				      NIL));

  /* FI: if I want to add extra-dimensions to reflect structure fields... */

  return make_reference(ent, regions_ref_inds);
}

/* reference make_regions_reference(entity ent)
 * input    : a variable entity.
 * output   : a reference made with the entity ent, and, in case of an array,
 *            PHI variables as indices.
 * modifies : nothing.
 */
reference make_regions_reference(entity ent)
{
  /* The Fortran semantics */
  return make_pointed_regions_reference(ent, false);
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
    type t = entity_basic_concrete_type(e);
    int d = type_depth(t);

    /* Let's deal with scalar pointers at least */
    /* If d==0 and if t is a pointer type, then d is a function of the pointed type */
    if(d==0 && pointer_type_p(t)) {
      basic tb = variable_basic(type_variable(t));
      type pt = basic_pointer(tb);

      d = type_depth(pt);
    }

    /* No phi if the entity has no depth (includes scalar entity which have depth 0 */
    if (d>0)
    {
	int dim;

	/* add phi variables in the base */
	for (dim = 1; dim<=d; dim ++)
	{
	    entity phi = make_phi_entity(dim);
	    sc_base_add_variable(ps, (Variable) phi);
	}

	/* add array bounds if asked for */
	/* FI: To be improved for arrays embedded within structures. */
	if (array_bounds_p())
	    ps = sc_safe_append(ps,
				sc_safe_normalize(entity_declaration_sc(e)));
    }
    return ps;
}


/**
    for C modules, the reference is trusted.  for Fortran modules, if
    the number of indices is less than the number of declared
    dimensions, it is assumed that it's a reference to the whole
    (sub-)array, and array bounds are added for the lacking
    dimensions.

 @param ref is a reference
 @param tac is an action tag (read or write)
 @return an effect representing a region corresponding to the reference.

 */
effect make_reference_region(reference ref, action tac)
{
  entity e = reference_variable(ref);
  effect reg;
  bool linear_p = true;
  Psysteme sc;
  type bct = entity_basic_concrete_type(e);
  int d = effect_type_depth(bct), dim;
  bool pointer_p = pointer_type_p(bct);
  list reg_ref_inds = NIL;

  /* first make the predicate according to ref */

  ifdebug(3)
    {
      pips_debug(3, "Reference : \"%s\"",
		 words_to_string(words_reference(ref, NIL)));
      fprintf(stderr, "(it's %s a pointer)\n", pointer_p?"":"not");
      pips_debug(3,"effect type depth is %d\n", d);
    }

  if (dummy_parameter_entity_p(e))
    pips_internal_error("the input reference entity is a dummy parameter (%s)",
			entity_name(e));

  /* FI: If t is a pointer type, then d should depend on the type_depth
     of the pointed type... */
  if (d>0 || pointer_p)
    {
      int idim;
      list ind = reference_indices(ref);
      int n_ind = gen_length(ind);
      int max_phi = 0;
      /* imported from preprocessor.h */

      pips_debug(8, "pointer or array case \n");

      pips_assert("The number of indices is less or equal to the possible "
		  "effect type depth, unless we are dealing with a pointer.\n",
		  (int) gen_length(ind) <= d || pointer_p);

      if (fortran_module_p(get_current_module_entity())
	  && n_ind < d)
	{
	  /* dealing with whole array references as in TAB = 1.0 */
	  sc = entity_declaration_sc(e);
	  max_phi = d;
	}
      else
	{
	  sc = sc_new();
	  max_phi = n_ind;
	}

      for (dim = 1; dim <= max_phi; dim++)
	reg_ref_inds = gen_nconc(reg_ref_inds,
				 CONS(EXPRESSION,
				      make_phi_expression(dim),
				      NIL));

      pips_assert("sc is weakly consistent", sc_weak_consistent_p(sc));

      /* we add the constraints corresponding to the reference indices */
      for (idim = 1; ind != NIL; idim++, ind = CDR(ind))
	{
	  /* For equalities. */
	  expression exp_ind = EXPRESSION(CAR(ind));
	  bool dim_linear_p;
	  bool unbounded_p =  unbounded_expression_p(exp_ind);

	  ifdebug(3)
	    {
	      if (unbounded_p)
		pips_debug(3, "unbounded dimension PHI%d\n",idim);
	      else
		pips_debug(3, "addition of equality :\nPHI%d - %s = 0\n",
			   idim, words_to_string(words_expression(exp_ind, NIL)));
	    }

	  if (unbounded_p)
	    {
	      /* we must add PHI_idim in the Psystem base */
	      entity phi = make_phi_entity(idim);
	      sc_base_add_variable(sc, (Variable) phi);
	      dim_linear_p = false;
	    }
	  else
	    {
	      pips_assert("sc is weakly consistent", sc_weak_consistent_p(sc));
	      dim_linear_p =
		sc_add_phi_equation(&sc, exp_ind, idim, IS_EG, PHI_FIRST);

	      pips_assert("sc is weakly consistent", sc_weak_consistent_p(sc));
	      pips_debug(3, "%slinear equation.\n", dim_linear_p? "": "non-");
	    }
	  linear_p = linear_p && dim_linear_p;
	} /* for */

      pips_assert("sc is weakly consistent", sc_weak_consistent_p(sc));

      /* add array bounds if asked for */
      if (array_bounds_p()) {
	sc = sc_safe_append(sc,
			    sc_safe_normalize(entity_declaration_sc(e)));
	pips_assert("sc is weakly consistent", sc_weak_consistent_p(sc));
      }

    } /* if (d>0 || pointer_p) */

  else
    {
      pips_debug(8, "non-pointer scalar type\n");
      sc = sc_new();
    }/* if else */

  pips_assert("sc is weakly consistent", sc_weak_consistent_p(sc));

  /* There was a preference originally : let's try a reference since a new
     reference is built for each region. BC */
  reg = make_region(
		    make_reference(reference_variable(ref), reg_ref_inds),
		    copy_action(tac),
		    make_approximation
		    (linear_p? is_approximation_exact : is_approximation_may,
		     UU),
		    sc);
  debug_region_consistency(reg);

  ifdebug(3)
    {
      pips_debug(3, "end : region is :\n");
      print_region(reg);
    }
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


/* effect reference_whole_region(reference ref, action tac, tap)
 * input    : a variable reference, and the action of the region.
 * output   : a region representing the entire memory space allocated
 *            to the variable *reference* (and not entity, this is useful
 *            for partially subscripted arrays, as is the case for
 *            PRINT *, A, where A is an array, see for instance
 *            Regions/incr3. BC).
 *            the systeme of constraints is a sc_rn(phi_1, ..., phi_n)
 *            except if REGIONS_WITH_ARRAY_BOUNDS is true.
 *            The resulting region is a MUST region if everything is linear,
 *            MAY otherwise. It's the responsibility of the caller to change it
 *            according to the calling context. (BC).
 * modifies : nothing.
 */
effect reference_whole_region(reference ref, action tac)
{
    entity e = reference_variable(ref);
    type t = entity_type(e);
    int d = type_depth(t), dim;
    bool linear_p = true;
    Psysteme sc;
    bool pointer_p = pointer_type_p(ultimate_type(t));

    effect reg = effect_undefined;
    list reg_ref_inds = NIL;

    ifdebug(3)
      {
	pips_debug(3, "Reference : \"%s\"",
		   words_to_string(words_reference(ref, NIL)));
	fprintf(stderr, "(it's %s a pointer)\n", pointer_p?"":"not");
	pips_debug(3,"type depth is %d\n", d);
      }

    /* FI: If t is a pointer type, then d should depend on the type_depth
       of the pointed type... */
    if (d>0 || pointer_p)
      {
	int idim;
	list ind = reference_indices(ref);
	int n_ind = gen_length(ind);

	pips_debug(8, "pointer or array case \n");

	pips_assert("The number of indices is less or equal to the type depth, "
		    "unless we are dealing with a pointer",
		    (int) gen_length(ind) <= d || pointer_p);


	if (n_ind < d)
	  sc = entity_declaration_sc(e);
	else
	  sc = make_whole_array_predicate(e);

	for (dim = 1; dim <= d; dim++)
	  reg_ref_inds = gen_nconc(reg_ref_inds,
				   CONS(EXPRESSION,
					make_phi_expression(dim),
					NIL));

	/* we add the constraints corresponding to the reference indices */
	for (idim = 1; ind != NIL; idim++, ind = CDR(ind))
	  {
	    /* For equalities. */
	    expression exp_ind = EXPRESSION(CAR(ind));
	    bool dim_linear_p;
	    bool unbounded_p =  unbounded_expression_p(exp_ind);

	    ifdebug(3)
	      {
		if (unbounded_p)
		  pips_debug(3, "unbounded dimension PHI%d\n",idim);
		else
		  pips_debug(3, "addition of equality :\nPHI%d - %s = 0\n",
			     idim,
			     words_to_string(words_expression(exp_ind, NIL)));
	      }

	    if (unbounded_p)
	      {
		/* we must add PHI_idim in the Psystem base */
		entity phi = make_phi_entity(idim);
		sc_base_add_variable(sc, (Variable) phi);
		dim_linear_p = false;
	      }
	    else
	      {

		dim_linear_p =
		  sc_add_phi_equation(&sc, exp_ind, idim, IS_EG, PHI_FIRST);

		pips_debug(3, "%slinear equation.\n", dim_linear_p? "": "non-");
	      }
	  linear_p = linear_p && dim_linear_p;
	  } /* for */
      } /* if (d>0 || pointer_p) */

    else
      {
	pips_debug(8, "non-pointer scalar type\n");
	sc = sc_new();
      }/* if else */

    reg = make_region(
		      make_reference(reference_variable(ref), reg_ref_inds),
		      copy_action(tac),
		      make_approximation
		      (linear_p? is_approximation_exact : is_approximation_may,
		       UU),
		    sc);
    debug_region_consistency(reg);

    ifdebug(3)
      {
	pips_debug(3, "end : region is :\n");
	print_region(reg);
      }
    return(reg);

}

region entity_whole_region(entity e, action tac)
{
    region new_eff;
    action ac = copy_action(tac);
    approximation ap = make_approximation(is_approximation_may, UU);

    new_eff = make_region(make_regions_reference(e), ac, ap,
			  make_whole_array_predicate(e));
    return(new_eff);
}


/* list region_to_store_independent_region_list(effect reg, bool force_may_p)
 * input    : a region and a boolean;
 * output   : a list with a unique region representing the entire memory space
 *            allocated to the variable reference : the systeme of constraints
 *            is a sc_rn(phi_1, ..., phi_n) except if REGIONS_WITH_ARRAY_BOUNDS
 *            is true. The region is a MAY region.
 *            The bool force_may_p is here for consistency with
 *            effect_to_store_independent_sdfi_list.
 *            We could refine this function in order to keep constraints
 *            which are independent from the store.
 * modifies : nothing.
 */
list region_to_store_independent_region_list(effect reg,
					     bool __attribute__ ((unused)) force_may_p)
{
    reference ref =  effect_any_reference(reg);
    effect eff = reference_whole_region(ref, region_action(reg));
    return(CONS(EFFECT,eff,NIL));
}


/* void convex_region_add_expression_dimension(effect reg, expression exp)
 * input    : a convex region and an expression
 * output   : nothing
 * modifies : the region reg, and normalizes the expression
 * comment  : adds a last dimension phi_last to the region. If the expression
 *            is normalizable, also adds a constraint phi_last = exp. Else,
 *            changes the approximation into may.
 */
void convex_region_add_expression_dimension(effect reg, expression exp)
{
  cell reg_c = effect_cell(reg);
  reference ref;
  int dim;
  bool dim_linear_p;

  ifdebug(8)
    {
      pips_debug(8, "begin with region :\n");
      print_region(reg);
    }

  pips_assert("region cell must be a reference\n", cell_reference_p(reg_c));

  ref = cell_reference(reg_c);

  /* first add a new PHI dimension to the region reference */
  dim = gen_length(reference_indices(ref))+1;

  pips_debug(8, "add PHI_%d\n", dim);
  reference_indices(ref) = gen_nconc( reference_indices(ref),
				      CONS(EXPRESSION,
					   make_phi_expression(dim),
					   NIL));

  /* Then add the corresponding constraint to the Psystem */
  dim_linear_p =
    sc_add_phi_equation(&region_system(reg), exp, dim, IS_EG, PHI_FIRST);

  if (!dim_linear_p)
    {
      pips_debug(8, "non linear expression : change approximation to may\n");
      effect_approximation_tag(reg) = is_approximation_may;
    }
  ifdebug(8)
    {
      pips_debug(8, "end with region :\n");
      print_region(reg);
      pips_assert("the region is consistent", effect_consistent_p(reg));
    }

  return;
}

/**
 eliminate phi_i from the region Psystem, and adds the constraint
 phi_i==exp if exp is normalizable.

 @param reg a convex region
 @param exp the new expresion for the region ith dimension
 @param i is the rank of the dimension to change.

 */
void convex_region_change_ith_dimension_expression(effect reg, expression exp,
						   int i)
{
  list l_phi_i = CONS(ENTITY,make_phi_entity(i),NIL);
  bool dim_linear_p;

  /* first eliminate PHI_i variable from the system */
  region_exact_projection_along_parameters(reg, l_phi_i);
  gen_free_list(l_phi_i);

  /* then add the new constraint in the system if the expression exp
     is not unbounded */
  if(unbounded_expression_p(exp))
    {
      pips_debug(8, "unbounded expression \n");
      effect_approximation_tag(reg) = is_approximation_may;
    }
  else
    {
      dim_linear_p =
	sc_add_phi_equation(&region_system(reg), exp, i, IS_EG, PHI_FIRST);

      if (!dim_linear_p)
	{
	  pips_debug(8, "non linear expression : change approximation to may\n");
	  effect_approximation_tag(reg) = is_approximation_may;
	}
    }

  ifdebug(8)
    {
      pips_debug(8, "end with region :\n");
      print_region(reg);
      pips_assert("the region is consistent", effect_consistent_p(reg));
    }

  return;
}


/**
   @brief copies the input_effect and converts all it's reference indices that refer to
          field entities to a phi variable, and add the corresponding constraint
	  (PHI_x == rank) in the descriptor system, rank being the integer corresponding
	  to the field rank in the field list of the ancestor derived type.
   @input input_effect is a convex effect, and is not modified.
   @return a new effect.
 */
effect convex_effect_field_to_rank_conversion(effect input_effect)
{
  effect eff = copy_effect(input_effect);
  cell c = effect_cell(input_effect);
  /* if it's a preference, we are sure there are no field dimensions */
  if (cell_reference_p(c))
    {
      reference r = cell_reference(c);
      list l_ind = reference_indices(r);
      int dim = 1;

      FOREACH(EXPRESSION, ind, l_ind)
	{
	  syntax s = expression_syntax(ind);
	  if (syntax_reference_p(s))
	    {
	      entity ind_e = reference_variable(syntax_reference(s));
	      if (entity_field_p(ind_e))
		{
		  int rank = entity_field_rank(ind_e);
		  expression rank_exp = int_to_expression(rank);

		  /* first change the effect reference index */
		  expression new_ind = make_phi_expression(dim);
		  free_syntax(s);
		  expression_syntax(ind) = copy_syntax(expression_syntax(new_ind));
		  free_expression(new_ind);
		  /* Then add the constrain PHI_rank == rank into the effect system */
          (void)sc_add_phi_equation(&region_system(eff), rank_exp, dim, IS_EG, PHI_FIRST);
		  free_expression(rank_exp);
		}
	    }
	  dim++;
	} /* FOREACH*/
    }
  return eff;
}


/**

 @param eff1 and eff2 are convex regions.
 @return eff1 (which is modified) : the reference indices of eff2
         (accordingly shifted) are appended to the reference indices of eff1 ;
	 and the Psystem of eff2 is appended to eff1 after renaming of phi
	 variables.
 */
effect region_append(effect eff1, effect eff2)
{
  list l_ind_1 = reference_indices(effect_any_reference(eff1));
  int nb_phi_1 = (int) gen_length(l_ind_1);

  list l_ind_2 = reference_indices(effect_any_reference(eff2));
  int nb_phi_2 = (int) gen_length(l_ind_2);

  Psysteme sc2 = region_system(eff2);

  int i;

  ifdebug(8) {
    pips_debug(8, "begin with regions : \n");
    print_region(eff1);
    print_region(eff2);
    pips_debug(8, "nb_phi1 = %d, nb_phi_2 = %d \n", nb_phi_1, nb_phi_2);
  }


  for(i=1; i<= nb_phi_2; i++)
    {
      expression ind_exp_2 = EXPRESSION(CAR(l_ind_2));
      if (entity_field_p(expression_variable(ind_exp_2)))
	l_ind_1 = gen_nconc(l_ind_1,
			    CONS(EXPRESSION,
				 copy_expression(ind_exp_2),
				 NIL));
      else
	l_ind_1 = gen_nconc(l_ind_1,
			    CONS(EXPRESSION,
				 make_phi_expression(nb_phi_1+i),
				 NIL));
      POP(l_ind_2);
    }
  reference_indices(effect_any_reference(eff1)) = l_ind_1;

  if(nb_phi_1 >0)
    {
      for(i=nb_phi_2; i>=1; i--)
	{
	  entity old_phi = make_phi_entity(i);
	  entity new_phi = make_phi_entity(nb_phi_1+i);

	  sc_variable_rename(sc2, old_phi, new_phi);
	}
    }

  region_sc_append(eff1, sc2, true);

  effect_approximation_tag(eff1) =
    approximation_and(effect_approximation_tag(eff1),
		      effect_approximation_tag(eff2));

  ifdebug(8) {
    pips_debug(8, "returning : \n");
    print_region(eff1);
  }
  return(eff1);
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
    pips_assert("phi index between 1 and NB_MAX_ARRAY_DIM\n", 1<=n && n<=NB_MAX_ARRAY_DIM);

    /* phi indices are between 1 to NB_MAX_ARRAY_DIM. array indices are between 0 to NB_MAX_ARRAY_DIM-1. */
    if (phi[n-1] == entity_undefined)
    {
	entity v;
	char phi_name[6];
	char * s;

	pips_assert ("n can't be more than a two digit number", n < 100);
	pips_assert ("PHI_PREFIX lenght has been changed", strlen (PHI_PREFIX) == 3);

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




/* bool sc_add_phi_equation(Psysteme * psc, expression expr,
 *                          int dim, bool is_eg, bool is_phi_first)
 *
 * input :    a region predicate, an expression caracterizing a PHI
 *            variable of range dim, two booleans indicating if the
 *            equation to create is an equality or an inequality, and
 *            this last case, the direction of the inequality.
 *
 * output :   true if expr is linear, false otherwise, which means that
 *            sc does not contain all the information associated to expr
 *            (MAY region).
 *
 * modifies : adds an equation containing a PHI variable to the system
 *            of constraints given in argument.
 *
 * comment  : The equation can, in fact, be an equality or an inequality :
 *
 *            If it is an equality   (is_eg == true)  :
 *                                     PHI  - expr  = 0
 *            If it is an inequality (is_eg == false) :
 *                   (is_phi_first == false)        expr - PHI  <= 0
 *                   (is_phi_first == true )        PHI  - expr <= 0
 *
 *            The indice of the region descriptor (PHI) is given by
 *            "dim".  The equation is added only if the expression
 *            (expr) is a normalized linear one, compatible with the
 *            semantics analysis.
 *
 *            Note : the basis of the system sc is updated.
 *
 *            Warning: this function may return wrong results when
 *            expressions used as subscripts have side-effects. Also
 *            this function would return more precise results if
 *            information about the store in which the expression is
 *            evaluated were passed. The problem exists in
 *            reference_to_convex_region() and in
 *            generic_p_proper_effect_of_reference().
 */
bool sc_add_phi_equation(Psysteme *psc, expression expr, int dim, bool is_eg,
			 bool is_phi_first)
{
    if(expression_range_p(expr)) {
        pips_assert("inequality with respect to a range has no meaning",is_eg);
        range r =expression_range(expr);
        bool must_p = sc_add_phi_equation(psc, range_upper(r), dim, false, is_phi_first);
        must_p &= sc_add_phi_equation(psc, range_lower(r), dim, false, !is_phi_first);
        return must_p;
    }
    else {
  normalized nexpr = NORMALIZE_EXPRESSION(expr);

  bool must_p = false; /* Do we capture the semantics of the
			  subscript expression exactly for sure? */
  Psysteme sc = *psc;

  /* Nga Nguyen: 29 August 2001
     Add  a value mapping test => filter variables that are not analysed by semantics analyses
     such as equivalenced variables. If not, the regions will be false (see bugregion.f)
     Attention: value hash table may not be defined */

  pips_assert("sc is weakly consistent", SC_UNDEFINED_P(sc) || sc_weak_consistent_p(sc));

  if (normalized_linear_p(nexpr)) {
    Pvecteur v2 = vect_copy(normalized_linear(nexpr));

    if (value_mappings_compatible_vector_p(v2)) {
      entity phi = make_phi_entity(dim);
      Pvecteur v1 = vect_new((Variable) phi, VALUE_ONE);
      Pvecteur vect;

      if (is_phi_first)
	vect = vect_substract(v1, v2);
      else
	vect = vect_substract(v2, v1);

      vect_rm(v1);
      /* vect_rm(v2); */

      if(!VECTEUR_NUL_P(vect))
	{
	  pips_assert("sc is defined", !SC_UNDEFINED_P(sc));

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

      must_p = true;
    }
    else {
      /* Some variables are not analyzed by semantics, for instance
	 because they are statically aliased in a non-analyzed way */
      /* No information about this subscript expression is added in
	 the region. */
      vect_rm(v2);
      must_p = false;
    }
  }
  else /* the expression is not linear */ {
    if(is_eg) {
      /* try to see if expression is i++, i--, ++i, --i */

      if(expression_call_p(expr))
	{
	  call c = syntax_call(expression_syntax(expr));
	  entity op = call_function(c);
	  list args = call_arguments(c);

	  if(ENTITY_POST_INCREMENT_P(op) ||
	     ENTITY_POST_DECREMENT_P(op))
	    {
	      return sc_add_phi_equation(psc, EXPRESSION(CAR(args)), dim, is_eg, is_phi_first);
	    }
	  else if (ENTITY_PRE_INCREMENT_P(op) )
	    {
	      expression new_exp = MakeBinaryCall(CreateIntrinsic("+"),
						  copy_expression(EXPRESSION(CAR(args))),int_to_expression(1));
	      must_p = sc_add_phi_equation(psc, new_exp, dim, is_eg, is_phi_first);
	      free_expression(new_exp);
	      return must_p;
	    }
	  else if (ENTITY_PRE_DECREMENT_P(op))
	    {
	      expression new_exp = MakeBinaryCall(CreateIntrinsic("-"),
						  copy_expression(EXPRESSION(CAR(args))),int_to_expression(1));
	      must_p = sc_add_phi_equation(psc, new_exp, dim, is_eg, is_phi_first);
	      free_expression(new_exp);
	      return must_p;
	    }

	}


      /* Nga Nguyen: 29 August 2001
	 If the expression expr is not a linear expression, we try to retrieve more
	 information by using function any_expression_to_transformer, for cases like:

	 ITAB(I/2) => {2*PHI1 <= I <= 2*PHI1+1}
	 ITAB(MOD(I,3)) => {0 <= PHI1 <= 2}, ...

	 The function any_expression_to_transformer() returns a transformer that may
	 contain temporary variables so we have to project these variables.

	 The approximation will become MAY (although in some cases, it is MUST) */

      entity phi = make_phi_entity(dim);
      transformer trans = any_expression_to_transformer(phi,expr,transformer_identity(),true);

      /* Careful: side-effects are lost */
      if (!transformer_undefined_p(trans)) {
	transformer new_trans = transformer_temporary_value_projection(trans);
	Psysteme p_trans = sc_copy(predicate_system(transformer_relation(new_trans)));

	/* trans has been transformed into new_trans by the
	   projection */
	//free_transformer(trans);
	free_transformer(new_trans);
	/* When sc is Rn, sc is not updated but freed and a copy of
	   p_trans is returned. */
	sc = sc_safe_append(sc, p_trans);
      }
      must_p = false;

    }
    else {
      /* An inequation should be generated.
       *
       * We end up here, for instance, with an unbounded
       * dimension. but we could also have a non-linear
       * declaration such as t[n*n];
       */
      pips_user_warning("Case not implemented as well as it could be\n");
      must_p = false;
    }
  }
  pips_assert("sc is weakly consistent", sc_weak_consistent_p(sc));
  *psc = sc;
  return must_p;
    }
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
    bool first = true, phi_only = true;

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
		if (first) first = false;
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
		first = false;
		phi_only = false;
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
    pips_assert("psy index between 1 and NB_MAX_ARRAY_DIM\n", 1<=n && n<=NB_MAX_ARRAY_DIM);

    /* psi indices are between 1 to NB_MAX_ARRAY_DIM. Array indices are between 0 to NB_MAX_ARRAY_DIM-1. */
    if (psi[n-1] == entity_undefined)
    {
	entity v;
	char psi_name[6];
	char * s;

	pips_assert ("n can't be more than a two digit number", n < 100);
	pips_assert ("PSI_PREFIX lenght has been changed", strlen (PSI_PREFIX) == 3);

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
/* RHO ENTITIES                                                                  */
/*********************************************************************************/

/* entity make_rho_entity(int n)
 * input    : an integer n giving the range of the RHO entity.
 * output   : the entity PHI_n.
 * modifies : rho[] (eventually).
 * comment  : finds or generates an entity representing a region descriptor
 *            of indice n.
 *            A region descriptor representes the value domain of one dimension
 *            of an array, this dimension is represented by the indice.
 *            Fortran accepts a maximum of 7 dimensions.
 *
 *            Once a region descriptor is created, it is stored into a static array
 *            (rho[]); thus, each next time a region descriptor entity is needed,
 *            it is just picked up in rho[].
 *            If the region descriptor is not in rho[], it may be in the table
 *            where all entities are stored, because of previous computations.
 *            At last, if it is not found there, it is created.
 */
entity make_rho_entity(int n)
{
    pips_assert("psy index between 1 and NB_MAX_ARRAY_DIM\n", 1<=n && n<=NB_MAX_ARRAY_DIM);

    /* rho indices are between 1 to NB_MAX_ARRAY_DIM. Array indices are between 0 to NB_MAX_ARRAY_DIM-1. */
    if (rho[n-1] == entity_undefined)
    {
	entity v;
	char rho_name[6];
	char * s;

	pips_assert ("n can't be more than a two digit number", n < 100);
	pips_assert ("RHO_PREFIX lenght has been changed", strlen (RHO_PREFIX) == 3);

	(void) strcpy(rho_name, RHO_PREFIX);
	(void) sprintf(rho_name+3,"%d",n);
	s = strdup(concatenate(REGIONS_MODULE_NAME,
			       MODULE_SEP_STRING, rho_name, (char *) NULL));

	if ((v = gen_find_tabulated(s, entity_domain)) == entity_undefined)
	{
	    v = make_entity(s,
			    make_scalar_integer_type(4),
			    make_storage(is_storage_rom, UU),
			    value_undefined);
	}

	rho[n-1] = v;
    }
    return (rho[n-1]);
}


list rho_entities_list(int rho_min, int rho_max)
{
    list l_rho = NIL;
    int i;

    for(i=rho_min; i<=rho_max; i++)
	l_rho = gen_nconc(l_rho, CONS(ENTITY, make_rho_entity(i), NIL));

    return(l_rho);
}

bool rho_reference_p( reference ref )
{
  list l = reference_indices(ref);

  for (;!ENDP(l); POP(l))
    {
      entity e = reference_variable
		     (syntax_reference
		      (expression_syntax
		       (EXPRESSION(CAR(l)))));
      if (!entity_field_p(e))
	{
	  if (variable_rho_p(e))
	    return true;
	  else
	    return false;
	}
    }
  return true; /* in case of a scalar */
}

/* bool psi_region_p(effect reg)
 * input    : a region.
 * output   : true if it is a PSI region
 * modifies : nothing
 * comment  : tests if the first indice of the reference is a PSI variable.
 */
bool rho_region_p(region reg)
{
    pips_assert("array region (not scalar)\n", !region_scalar_p(reg));

    return (rho_reference_p(effect_any_reference(reg)));
}

/* expression make_rho_expression(int n)
 * input    : the range of the PHI entity to create.
 * output   : an expression made of a PHI variable of range n;
 * modifies : nothing.
 */
expression make_rho_expression(int n)
{
    return(make_expression(make_syntax(is_syntax_reference,
				       make_reference(make_rho_entity(n), NIL)),
			   normalized_undefined));
}


/* int base_nb_rho(Pbase b)
 * input    : a base
 * output   : the number of rho variables in this base.
 * modifies : nothing.
 * comment  :
 */
int base_nb_rho(Pbase b)
{
    int n_rho = 0;

    for(; !VECTEUR_NUL_P(b); b = b->succ)
	if (variable_rho_p((entity) var_of(b))) n_rho++;

    return(n_rho);
}

/*********************************************************************************/
/* PHI AND PSY ENTITIES                                                          */
/*********************************************************************************/

list cell_reference_phi_cfc_variables(reference ref, Psysteme sc)
{

  list l_ent = NIL;
  if(entity_scalar_p(reference_variable(ref)))
    return NIL;

  if (psi_reference_p(ref))
    l_ent = psi_entities_list(1, (int) gen_length(reference_indices(ref)));
  else if (rho_reference_p(ref))
    l_ent = rho_entities_list(1, (int) gen_length(reference_indices(ref)));
  else
    l_ent = phi_entities_list(1, (int) gen_length(reference_indices(ref)));

  return(sc_entities_cfc_variables(sc, l_ent));

}

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
    if (region_scalar_p(reg))
	return(NIL);
    else
      return(cell_reference_phi_cfc_variables(effect_any_reference(reg), region_system(reg)));
}

/* void psi_to_phi_region(effect reg)
 * input    : a PSI region
 * output   : the same region, using PHI variables.
 * modifies : reg
 * comment  : also changes the reference.
 */
void psi_to_phi_region(region reg)
{
  int i, ndims = (int) gen_length(reference_indices(region_any_reference(reg)));
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

    free_reference(effect_any_reference(reg));

    pips_assert("region cell must be a reference\n",
		cell_reference_p(effect_cell(reg)));
    cell_reference(effect_cell(reg)) = reg_ref;
}

/* void phi_to_psi_region(effect reg)
 * input    : a PHI region
 * output   : the same region, using PSI variables.
 * modifies : reg
 * comment  : also changes the reference.
 */
void phi_to_psi_region(region reg)
{
    int i, ndims = (int) gen_length(reference_indices(region_any_reference(reg)));
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
    free_reference(effect_any_reference(reg));
    pips_assert("region cell must be a reference\n",
		cell_reference_p(effect_cell(reg)));
    cell_reference(effect_cell(reg)) = reg_ref;
}


bool psi_reference_p(reference ref)
{

  list l = reference_indices(ref);

  for (;!ENDP(l); POP(l))
    {
      entity e = reference_variable
		     (syntax_reference
		      (expression_syntax
		       (EXPRESSION(CAR(l)))));
      if (!entity_field_p(e))
	{
	  if (variable_psi_p(e))
	    return true;
	  else
	    return false;
	}
    }
  return true; /* in case of a scalar */
}

/* bool psi_region_p(effect reg)
 * input    : a region.
 * output   : true if it is a PSI region
 * modifies : nothing
 * comment  : tests if the first indice of the reference is a PSI variable.
 */
bool psi_region_p(region reg)
{
    pips_assert("array region (not scalar)\n", !region_scalar_p(reg));

    return (psi_reference_p(effect_any_reference(reg)));
}

/************************************************************* PROPERTIES */

static bool exact_regions_property = false;
static bool must_regions_property = false;
static bool array_bounds_property = false;
static bool disjunct_property = false;
static bool op_statistics_property = false;

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
    exact_regions_property = true;
    must_regions_property = true;
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
    return(compare_region_constraints(pc1, pc2, true));
}

static int compare_region_inequalities(Pcontrainte *pc1, Pcontrainte *pc2)
{
    return(compare_region_constraints(pc1, pc2, false));
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
    sc->inegalites = region_constraints_sort(sc->inegalites, sorted_base, false);
    ifdebug(1)
    {
	pips_debug(1, "\nafter inequality sort:\n");
	sc_syst_debug(sc);
    }
    sc->egalites = region_constraints_sort(sc->egalites, sorted_base, true);
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
      //	(c, BASE_NULLE, equality_p?
      (c, sorted_base, equality_p?
       ((int (*)()) compare_region_equalities): ((int (*)()) compare_region_inequalities));
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
	/* both have no phi variables : the comparison basis only
	   consists of variables other than phi variables */
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
    int phi_1 = false, phi_2 = false,
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
    if (phi_1 - phi_2 == 0) {
      /* FI: This does not work well with C scope system */
      //return(strcmp(entity_name(*pe1), entity_name(*pe2)));
      string s1 = entity_name_without_scope(*pe1);
      string s2 = entity_name_without_scope(*pe2);
      int result = strcmp(s1,s2);

      free(s1);
      free(s2);
      return result;
    }
    /* if one and only one of the entities is a phi variable */
    else
	return(phi_2 - phi_1);
}

/********************************** MANIPULATION OF ENTITIES AND VARIABLES */


Psysteme entity_declaration_sc(entity e)
{
    Psysteme sc = sc_new();
    list ldim = variable_dimensions(type_variable(ultimate_type(entity_type(e))));
    int dim;

    for (dim = 1; !ENDP(ldim); dim++, ldim = CDR(ldim))
    {
	expression dl, du;

	dl = dimension_lower(DIMENSION(CAR(ldim)));
	ifdebug(8) {
	    debug(8, "entity_declaration_sc",
		  "addition of inequality :\n%s - PHI%d <= 0\n",
		  words_to_string(words_expression(dl, NIL)), dim);
	}
	(void) sc_add_phi_equation(&sc, dl, dim, NOT_EG, NOT_PHI_FIRST);

	du = dimension_upper(DIMENSION(CAR(ldim)));
	ifdebug(8) {
	    debug(8, "entity_declaration_sc",
		  "addition of inequality :\nPHI%d - %s <= 0\n",
		  dim, words_to_string(words_expression(du, NIL)));
	}
	(void) sc_add_phi_equation(&sc, du, dim, NOT_EG, PHI_FIRST);

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

    FOREACH(ENTITY, e, l_var)
     {
	 entity e_old = entity_to_old_value(e);
	 list l_tmp = CONS(ENTITY, e_old, NIL);
	 l_old = gen_nconc(l_old, l_tmp); /* to be in the same order as l_var */
	 debug(8, "", "%s -> %s\n",
	       (e == (entity) TCST) ? "TCST" : pips_user_value_name(e),
	       (e_old == (entity) TCST) ? "TCST" : pips_user_value_name(e_old));
     }

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
 * output   : returns true if both entities, which are common variables,
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
    bool equal = false;
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
	    pips_internal_error("arrays entities with not variable type : %s %s",
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
		for (count_dim = 1, equal = true;
		     (count_dim <= nb_dims1) && (equal); count_dim++) {
		    int size1 = SizeOfIthDimension(e1, count_dim);
		    int size2 = SizeOfIthDimension(e2, count_dim);

		    /* Same dimension size. */
		    if (size1 != size2) {
			equal = false;
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
       make_predicate((Psysteme) sc_rn((Pbase)
				     vect_new((Variable) i, VALUE_ONE))));
    loop_prec =
	add_index_range_conditions(loop_prec,
				   i,
				   loop_range(l),
				   loop_body_trans);

    sc = predicate_system(transformer_relation(loop_prec));
    predicate_system_(transformer_relation(loop_prec)) =
	newgen_Psysteme(SC_UNDEFINED);
    free_transformer(loop_prec);

    return sc;
}

/******************************************************************** MISC */

/* is the context empty?
 */
bool empty_convex_context_p(transformer context)
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
	return true;
    }

    /* else
     */

    sc_context = predicate_system(transformer_relation(context));
    return sc_empty_p(sc_context);
}

string region_to_string(effect reg __attribute__ ((unused)))
{
    return strdup("[region_to_string] no longer implemented\n");
}

static bool max_one_phi_var_per_constraint(c)
Pcontrainte c;
{
    for(; c; c=c->succ)
	if (base_nb_phi(c->vecteur)>=2) return false;

    return true;
}

/* returns true is syst defines a rectangle on phi variables
 */
bool rectangular_region_p(region r)
{

    Psysteme syst = region_system(r);
    /*  this is an approximation, and the system should be normalized
     *  for the result to be correct, I guess...
     */
    return max_one_phi_var_per_constraint(sc_egalites(syst)) &&
        max_one_phi_var_per_constraint(sc_inegalites(syst));
}

bool rectangular_must_region_p(var, s)
entity var;
statement s;
{
    list le = load_statement_local_regions(s);

    FOREACH(EFFECT, e,le)
    {
        if (reference_variable(effect_any_reference(e)) == var)
        {
            if (!approximation_exact_p(effect_approximation(e)) ||
                    !rectangular_region_p(e))
            {
                pips_debug(6, "FALSE\n"); return false;
            }
        }
    }
    pips_debug(6, "TRUE\n");
    return true;
}

/********************************************************************
 * SG awful contributions to BC wonderful work
 */


/* return true if an expression is a field access or a constant phi with regard to sc */
static bool expression_is_a_constant_reference_p(expression exp, Psysteme sc) {
    if(expression_reference_p(exp)) { // a phi or a field ?
        reference ref = expression_reference(exp);
        entity var = reference_variable(ref);
        if(entity_field_p(var)) return true;
        else if(variable_phi_p(var)) { // we should stop there, unless we have a constant index
            /* iterate over the region equalities to find this particular phi variable */
            Psysteme scd = sc_dup(sc);
            sc_transform_eg_in_ineg(scd);
            Pcontrainte lower,upper;
            constraints_for_bounds(var,
                    &sc_inegalites(scd), &lower, &upper);
            bool constant_index_p = 
                (!CONTRAINTE_UNDEFINED_P(upper) && !CONTRAINTE_UNDEFINED_P(lower) && bounds_equal_p(var,lower,upper)) ;
            sc_free(scd);
            if(constant_index_p) {
                return true;
            }
        }
    }
    return false;
}
/* computes the rectangular hull of a region
 * if the region indices contain fields and @p nofield is set to true
 * they are removed to ensure a non strided access
 * that is if the field is at the head, it is kept (struct of array)
 * if it is in the tail, it is ignored
 * if it is in the middle, we are in great trouble
 * and perform an over-approximation of ignoring all remaining accesses
 *
 */
region region_rectangular_hull(region reg, bool nofield)
{
    region hyper = copy_effect(reg);
    if(nofield) {
        /* this is the stride fix */
        list iter = reference_indices(region_any_reference(hyper)),prev;
        prev=iter;

        bool constant_path_head_p = true;
        while(!ENDP(iter) && constant_path_head_p ) {
            /* head : keep everything that does not introduce a stride */
            for(;!ENDP(iter);POP(iter)) {
                expression e = EXPRESSION(CAR(iter));
                if(expression_reference_p(e) &&
                        entity_field_p(reference_variable(expression_reference(e)))) {
                    break; // stop as soon has we meet a field
                }
                constant_path_head_p &= expression_is_a_constant_reference_p(e, region_system(reg));
                prev=iter;
            }
            /* body keep all fields indices if we have a constant path head*/
            if(constant_path_head_p) {
                for(;!ENDP(iter);POP(iter)) {
                    expression e = EXPRESSION(CAR(iter));
                    prev=iter;
                    if(!expression_reference_p(e) ||
                            !entity_field_p(reference_variable(expression_reference(e)))) {
                        constant_path_head_p &= expression_is_a_constant_reference_p(e, region_system(reg));
                        break; // eats up all fields
                    }
                }
            }
        }
        /* tail : once a field is met , we cannot go on unless we have a constant path head */
        list tail = NIL;
        if(!ENDP(prev)) { tail = CDR(prev) ; CDR(prev)=NIL; }
        /* if we cut the tail , approximation is may */
        if(!ENDP(tail)) {
            region_approximation_tag(hyper)=is_approximation_may;
            list phis = NIL;
            FOREACH(EXPRESSION,exp,tail) {
                if(expression_reference_p(exp) ) {
                    entity var = reference_variable(expression_reference(exp));
                    if(variable_phi_p(var)) phis = CONS(ENTITY,var,phis);
                }
            }
            phis=gen_nreverse(phis);
            region_exact_projection_along_parameters(hyper,phis);
            gen_free_list(phis);
        }
        gen_full_free_list(tail);
    }

    /* the real rectangular hull begins now */

    /* first gather all phis */
    list phis = NIL;
    FOREACH(EXPRESSION,exp,reference_indices(region_any_reference(hyper)))
        if(expression_reference_p(exp) &&
                variable_phi_p(reference_variable(expression_reference(exp))))
            phis = CONS(ENTITY,reference_variable(expression_reference(exp)),phis);
    phis=gen_nreverse(phis);

    /* then for each of the phis, generate a list of all other phis
     * and project the system along this variables */
    region nr = effect_undefined;
    FOREACH(ENTITY, phi_i, phis) {
        list other_phis = gen_copy_seq(phis);
        gen_remove_once(&other_phis,phi_i);
        region partial = copy_effect(hyper);
        region_exact_projection_along_parameters(partial,other_phis);
        if(effect_undefined_p(nr)) nr=partial;
        else {
            region_sc_append(nr,region_system(partial),1);
            free_effect(partial);
        }
        gen_free_list(other_phis);
    }
    gen_free_list(phis);
    if(!effect_undefined_p(nr)) {
        sc_fix(region_system(nr));
        region_system(hyper)=sc_dup(region_system(nr));
        if(!sc_equal_p(region_system(hyper),region_system(reg)))
            region_approximation_tag(hyper)=is_approximation_may;
        free_effect(nr);
    }
    // in case of error, we keep the original region
    return hyper;
}

/* translates a reference form a region into a valid expression
 * it handles fields accesses conversion, but does not changes phi variables
 * it is up to the user to use replace_entity on the generated expression according to its needs
 * no sharing introduced between returned expression and @p r
 */
expression region_reference_to_expression(reference r) {
    expression p = reference_to_expression(
            make_reference(reference_variable(r),NIL)
            );
    FOREACH(EXPRESSION,index,reference_indices(r)) {
        if(expression_reference_p(index) &&
                entity_field_p(reference_variable(expression_reference(index)))) {
            p = MakeBinaryCall(
                    entity_intrinsic(FIELD_OPERATOR_NAME),
                    p,
                    copy_expression(index)
                    );
        }
        else {
            if(expression_reference_p(p)) {
                reference pr = expression_reference(p);
                reference_indices(pr)=
                    gen_append(reference_indices(pr),
                            CONS(EXPRESSION,copy_expression(index),NIL)
                            );
            }
            else if(expression_subscript_p(p)) {
                subscript ps = expression_subscript(p);
                subscript_indices(ps)=
                    gen_append(subscript_indices(ps),
                            CONS(EXPRESSION,copy_expression(index),NIL)
                            );
            }
            else {
                pips_assert("is a field",expression_field_p(p));
                p = syntax_to_expression(
                        make_syntax_subscript(
                            make_subscript(
                                p,
                                CONS(EXPRESSION,copy_expression(index),NIL)
                                )
                            )
                        );
            }

        }
    }
    return p;
}

/* computes the volume of a region
 * output it as a Polynomial representing the number of elements counted
 */
Ppolynome region_enumerate(region reg)
{
    Ppolynome p = POLYNOME_UNDEFINED;
    if(rectangular_region_p(reg)) // in that case, use a simpler algorithm */
    {
        Psysteme r_sc = region_system(reg);
        sc_normalize2(r_sc);
        reference ref = region_any_reference(reg);
        p = make_polynome(1.,TCST,0);
        for(list iter = reference_indices(ref);!ENDP(iter); POP(iter))
        {
            expression index = EXPRESSION(CAR(iter));
            Variable phi = expression_to_entity(index);
            if(variable_phi_p((entity)phi)) {
                /* first search into the equalities if any */
                bool found = false;
                for(Pcontrainte c = sc_egalites(r_sc);
                        !CONTRAINTE_UNDEFINED_P(c);
                        c=contrainte_succ(c)) {
                    Pvecteur v = c->vecteur;
                    if((found=base_contains_variable_p(v,phi))) {
                        // if an equality is found, the volume is not affected
                        break;
                    }
                }
                if(!found) {

                    /* else try to find the range */
                    Psysteme sc = sc_dup(r_sc);
                    sc_transform_eg_in_ineg(sc);
                    Pcontrainte lower,upper;
                    constraints_for_bounds(phi, &sc_inegalites(sc), &lower, &upper);
                    if( !CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper))
                    {
                        /* this is a constant */
                        if(bounds_equal_p(phi,lower,upper))
                            ; /* skip */
                        /* this is a range : eupper-elower +1 */
                        else
                        {
                            expression elower = constraints_to_loop_bound(lower,phi,true,entity_intrinsic(DIVIDE_OPERATOR_NAME));
                            expression eupper = constraints_to_loop_bound(upper,phi,false,entity_intrinsic(DIVIDE_OPERATOR_NAME));

                            expression diff = make_op_exp(MINUS_OPERATOR_NAME,eupper,elower);
                            diff = make_op_exp(PLUS_OPERATOR_NAME,diff,int_to_expression(1));
                            Ppolynome pdiff = expression_to_polynome(diff);
                            free_expression(diff);
                            p=polynome_mult(p,pdiff);
                        }
                    }
                    else {
                        pips_user_warning("failed to analyse region\n");
                    }
                    sc_free(sc);
                }
            }
        }
    }
    else {
        Psysteme r_sc = region_system(reg);
        sc_fix(r_sc);
        Pbase sorted_base = region_sorted_base_dup(reg);
        sc_base(r_sc)=sorted_base;
        Pbase local_base = BASE_NULLE;
        for(Pbase b = sc_base(r_sc);!BASE_NULLE_P(b);b=b->succ)
            if(!variable_phi_p((entity)b->var))
                local_base=base_add_variable(local_base,b->var);

        const char * base_names [sc_dimension(r_sc)];
        int i=0;
        for(Pbase b = local_base;!BASE_NULLE_P(b);b=b->succ)
            base_names[i++]=entity_user_name((entity)b->var);

        ifdebug(1) print_region(reg);

        p = sc_enumerate(r_sc,
                local_base,
                base_names);
    }
    return p;
}
