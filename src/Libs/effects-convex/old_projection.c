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
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"
#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"
#include "matrice.h"
#include "transformer.h"
#include "sparse_sc.h"
#include "pipsdbm.h"

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
				  mod_name, ".", prefix, filename, NULL));
    fp = safe_fopen(filename, "w");
    fprintf(fp,"%s",mod_name);
    fprintf(fp," %d %d %d %d %d %d\n", nb_proj_param, nb_proj_param_pot_must,
	    nb_proj_param_must, nb_proj_param_ofl, nb_proj_param_hermite,
	    nb_proj_param_hermite_success);
    safe_fclose(fp, filename);
    free(filename);

    filename = "proj_var_op_stat";
    filename = strdup(concatenate(db_get_current_workspace_directory(), "/",
				  mod_name, ".", prefix, filename, NULL));
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
 *		          bool *normalized_regions_p, bool sc_loop_p,
 *                        Psysteme *psc_loop)
 * input    : a list of regions, a loop index, its range, and a pointer on a
 *            bool to know if the loop is normalized, a bool to know
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
		       bool *normalized_regions_p, bool sc_loop_p,
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
		FOREACH(EFFECT, reg, l_reg)
		{
		  if(store_effect_p(reg)) {
		    if (!region_rn_p(reg) && !region_empty_p(reg))
		    {
			region_sc_append_and_normalize(reg,beta_sc,1);
			region_exact_projection_along_parameters(reg, l_tmp);
		    }
		  }
		}
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
    bool projection_of_index_safe = false;
    Psysteme sc_tmp = SC_UNDEFINED;

    debug_on("REGIONS_OPERATORS_DEBUG_LEVEL");
    debug_regions_consistency(l_reg);


    /* Take care of loops with non-unit increment when must regions
       are required */
    index = loop_regions_normalize(l_reg, index, l_range,
				   &projection_of_index_safe,false,&sc_tmp);

    if (must_regions_p() && projection_of_index_safe)
    {

      FOREACH(EFFECT, reg, l_reg) {
	  if(store_effect_p(reg))
	    region_exact_projection_along_variable(reg, index);
	}
    }
    else
    {
	list l = CONS(ENTITY, index, NIL);
	FOREACH(EFFECT, reg, l_reg) {
	  if(store_effect_p(reg))
	    region_non_exact_projection_along_variables(reg, l);
	}
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
      FOREACH(EFFECT, reg, l_reg) {
	if(store_effect_p(reg))
	    region_exact_projection_along_parameters(reg, l_param);
	}
    }
    else
    {
      FOREACH(EFFECT, reg, l_reg) {
	if(store_effect_p(reg))
	    region_non_exact_projection_along_parameters(reg, l_param);
	}
    }
    debug_regions_consistency(l_reg);
    debug_off();
}



void
project_regions_with_transformer(list l_reg, transformer trans,
				 list l_var_not_proj)
{
    regions_transformer_apply(l_reg, trans, l_var_not_proj, false);
}

void project_regions_with_transformer_inverse(list l_reg, transformer trans,
					      list l_var_not_proj)
{
    regions_transformer_apply(l_reg, trans, l_var_not_proj, true);
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
      pips_debug_effects(3, "regions before transformation:\n", l_reg);
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
		sc_print(sc_trans, (get_variable_name_t) entity_local_name);
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

	FOREACH(EFFECT, reg, l_reg)
	  {
	  if(store_effect_p(reg))
	    {
	      Psysteme sc_reg = region_system(reg);
	      
	      if (!SC_UNDEFINED_P(sc_reg) && !sc_empty_p(sc_reg) && !sc_rn_p(sc_reg))
		{
		  debug_region_consistency(reg);

		  pips_debug_effect(8, "region before transformation: \n", reg);

		  sc_list_variables_rename(sc_reg, l_var, l_int);
		  pips_debug_effect(8, "region after renaming: \n", reg);

		  /* addition of the predicate of the transformer,
		     and elimination of redundances */
		  region_sc_append_and_normalize(reg, sc_trans, 1);

		  debug_region_consistency(reg);

		  pips_debug_effect(8, "region after addition of the transformer: ", reg);

		  /* projection along intermediate variables */
		  region_exact_projection_along_parameters(reg, l_int);
		  debug_region_consistency(reg);
		  pips_debug_effect(8, "region after projection along parameters: \n", reg );

		  /* remove potential old values that may be found in transformer */
		  list l_old_values = NIL;
		  sc_reg = region_system(reg);
		  for(Pbase b = sc_base(sc_reg); !BASE_NULLE_P(b); b = vecteur_succ(b)) {
		    entity e = (entity) vecteur_var(b);
		    if(local_old_value_entity_p(e)) {
		      l_old_values = CONS(ENTITY, e, l_old_values);
		    }
		  }
		  region_exact_projection_along_parameters(reg, l_old_values);
		  gen_free_list(l_old_values);
		  debug_region_consistency(reg);
		  pips_debug_effect(8, "region after transformation: \n", reg );
		}
	    }
	  }

	/* no memory leaks */
	gen_free_list(l_int);
	gen_free_list(l_old);
	sc_rm(sc_trans);
    }

    pips_debug_effects(3, "regions after transformation:\n", l_reg);

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

  FOREACH(EFFECT, reg, l_reg)
      {
	if(store_effect_p(reg)) {
        entity reg_ent = region_entity(reg);
        storage reg_s = entity_storage(reg_ent);
        bool ignore_this_region = false;

	ifdebug(4)
	  {
	    pips_debug_effect(4, "current region: \n", reg);
	  }

	/* If the reference is a common variable (ie. with storage ram but
	 * not dynamic) or a formal parameter, the region is not ignored.
	 */
	if(!anywhere_effect_p(reg)) {
	  switch (storage_tag(reg_s))
	    {
	    case is_storage_return:
	      pips_debug(5, "return var ignored (%s)\n", entity_name(reg_ent));
	      ignore_this_region = true;
	      break;
	    case is_storage_ram:
	      {
		ram r = storage_ram(reg_s);
		if (dynamic_area_p(ram_section(r)) || heap_area_p(ram_section(r))
		    || stack_area_p(ram_section(r)))
		  {
		    pips_debug(5, "dynamic or pointed var ignored (%s)\n", entity_name(reg_ent));
		    ignore_this_region = true;
		  }
		break;
	      }
	    case is_storage_formal:
	      break;
	    case is_storage_rom:
	      if(!entity_special_area_p(reg_ent) && !anywhere_effect_p(reg))
		ignore_this_region = true;
	      break;
	      /* pips_internal_error("bad tag for %s (rom)", entity_name(reg_ent)); */
	    default:
	      pips_internal_error("case default reached");
	    }
	}

        if (! ignore_this_region)  /* Eliminate dynamic variables. */
	  {
	    region r_res = region_dup(reg);
	    region_dynamic_var_elim(r_res);
	    ifdebug(4)
	      {
		pips_debug_effect(4, "region kept : \n", r_res);
	      }
            l_res = region_add_to_regions(r_res,l_res);
	  }
	else
	  ifdebug(4)
	    {
	      pips_debug_effect(4, "region removed : \n", reg);
	    }
	}
      }
  debug_regions_consistency(l_reg);
  debug_off();

  return(l_res);
}


/*********************************************************************************/
/* OPE'RATEURS CONCERNANT LES RE'GIONS INDIVIDUELLES.                            */
/*********************************************************************************/

static Psysteme region_sc_projection_ofl_along_parameters(Psysteme sc,
							  Pvecteur pv_param,
							  bool *p_exact);
static Psysteme region_sc_projection_ofl_along_parameter(Psysteme sc, Variable param,
							 bool *p_exact);


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

    l_phi = phi_entities_list(1,NB_MAX_ARRAY_DIM);
    project_regions_along_variables(l_reg, l_phi);
    gen_free_list(l_reg);
    gen_free_list(l_phi);

    ifdebug(8)
    {
	pips_debug(8, "final region : \n ");
	print_region(reg);
    }
}


Psysteme cell_reference_system_remove_psi_variables(reference ref, Psysteme sc, bool * exact_p)
{
  list volatile l_psi = psi_entities_list(1,NB_MAX_ARRAY_DIM);
  *exact_p = true;

  for(; !ENDP(l_psi) && *exact_p; l_psi = CDR(l_psi))
    {
      entity e = ENTITY(CAR(l_psi));
      bool exact_projection;
      sc = cell_reference_sc_exact_projection_along_variable(ref, sc, e, &exact_projection);
      *exact_p = *exact_p && exact_projection;
    }
  Psysteme volatile ps = sc;
  if (!ENDP(l_psi) && !SC_UNDEFINED_P(ps))
    {
      CATCH(overflow_error)
      {
	sc_rm(ps);
	ps = SC_UNDEFINED;
      }
      TRY
	{
	  ps = sc_projection_ofl_along_list_of_variables(ps, l_psi);
	  UNCATCH(overflow_error);
	}
    }
  return ps;
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

    l_psi = psi_entities_list(1,NB_MAX_ARRAY_DIM);
    project_regions_along_variables(l_reg, l_psi);
    gen_free_list(l_reg);

    ifdebug(8)
    {
	pips_debug(8, "final region : \n ");
	print_region(reg);
    }
}

Psysteme cell_reference_system_remove_rho_variables(reference ref, Psysteme sc, bool *exact_p)
{
  list volatile l_rho = rho_entities_list(1,NB_MAX_ARRAY_DIM);
  *exact_p = true;

  for(; !ENDP(l_rho) && *exact_p; l_rho = CDR(l_rho))
    {
      entity e = ENTITY(CAR(l_rho));
      bool exact_projection;
      sc = cell_reference_sc_exact_projection_along_variable(ref, sc, e, &exact_projection);
      *exact_p = *exact_p && exact_projection;
    }
  Psysteme volatile ps = sc;
  if (!ENDP(l_rho) && !SC_UNDEFINED_P(ps))
    {
      CATCH(overflow_error)
      {
	sc_rm(ps);
	ps = SC_UNDEFINED;
      }
      TRY
	{
	  ps = sc_projection_ofl_along_list_of_variables(ps, l_rho);
	  UNCATCH(overflow_error);
	}
    }
  return ps;
}


/* void region_remove_rho_variables(effect reg, int phi_max)
 * input    : a PHI region in which rho variables must be eliminated.
 * output   : nothing.
 * modifies : reg
 * comment  : if an overflow error occurs, the region becomes a MAY region..
 */
void region_remove_rho_variables(region reg)
{
    list
	l_rho = NIL,
	l_reg = CONS(EFFECT,reg,NIL);

    ifdebug(8)
    {
	pips_debug(8, "region : \n ");
	print_region(reg);
    }

    l_rho = rho_entities_list(1,NB_MAX_ARRAY_DIM);
    project_regions_along_variables(l_reg, l_rho);
    gen_free_list(l_reg);

    ifdebug(8)
    {
	pips_debug(8, "final region : \n ");
	print_region(reg);
    }
}


/* void region_remove_variables(effect reg, int beta_max)
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
    /* Automatic variables read in a CATCH block need to be declared volatile as
     * specified by the documentation*/
    Psysteme volatile ps;
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
	    region reg_tmp = reference_whole_region(region_any_reference(reg),
						    region_action(reg));
	    cell c_tmp = region_cell(reg_tmp);

	    region_system_(reg) = region_system_(reg_tmp);
	    pips_assert("region cell must be a reference\n",
			cell_reference_p(c_tmp));

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
    /* Automatic variables read in a CATCH block need to be declared volatile as
     * specified by the documentation*/
    Psysteme volatile ps;


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
	    region reg_tmp = reference_whole_region(region_any_reference(reg),
						    region_action(reg));
	    cell c_tmp = region_cell(reg_tmp);

	    pips_assert("region cell must be a reference\n",
			cell_reference_p(c_tmp));
	    region_system_(reg) = region_system_(reg_tmp);
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
		    bool exact = true;
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
 * comment  : overflow errors are trapped here. if it occurs, a whole array region
 *           replaces the initial region.
 */
void region_non_exact_projection_along_variables(region reg, list l_var)
{
    /* Automatic variables read in a CATCH block need to be declared volatile as
     * specified by the documentation*/
    Psysteme volatile ps;

    ps = region_system(reg);

    if (!sc_empty_p(ps) && !sc_rn_p(ps))
    {
	if (op_statistics_p()) nb_proj_var++;

	ifdebug(6)
	{
	    pips_debug(6, "initial region :\n");
	    print_region(reg);
	    pips_debug(6, "variables along which the projection must be performed:\n");
	    print_arguments(l_var);
	}

	/* if there is an overflow error, reg becomes a whole array may region */
	CATCH(overflow_error)
	{
	    region reg_tmp = reference_whole_region(region_any_reference(reg),
						    region_action(reg));
	    cell c_tmp = region_cell(reg_tmp);
	    pips_assert("region cell must be a reference\n",
			cell_reference_p(c_tmp));

	    region_system_(reg) = region_system_(reg_tmp);

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
    /* Automatic variables read in a CATCH block need to be declared volatile as
     * specified by the documentation*/
    Psysteme volatile ps;
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
	    region reg_tmp = reference_whole_region(region_any_reference(reg),
						    region_action(reg));
	    cell c_tmp = region_cell(reg_tmp);
	    pips_assert("region call must be a reference\n",
			cell_reference_p(c_tmp));

	    region_system_(reg) = region_system_(reg_tmp);
	    region_system_(reg_tmp) = newgen_Psysteme(SC_UNDEFINED);
	    free_effect(reg_tmp);
	    sc_rm(ps);
	    if (op_statistics_p()) nb_proj_var_ofl++;
	}
	TRY {
	    list ll_var = l_var;

	    for(; !ENDP(ll_var) &&
		    (region_approximation_tag(reg) == is_approximation_exact);
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

Psysteme cell_reference_sc_exact_projection_along_variable(reference ref, Psysteme sc, entity var, bool *exact_p)
{
  Psysteme volatile ps = sc;
  *exact_p = true;
  if (!sc_empty_p(ps) && !sc_rn_p(ps))
    {
      if (base_contains_variable_p(ps->base, (Variable) var))
	{
	  CATCH(overflow_error)
	  {
	    sc_rm(ps);
	    *exact_p = false;
	    return SC_UNDEFINED;
	  }
	  TRY {
	    list l_phi_var = cell_reference_phi_cfc_variables(ref, ps);

	    if (gen_find_eq(var, l_phi_var) == chunk_undefined)
	      {
		Psysteme volatile * psc = &ps;
		sc_projection_along_variable_ofl_ctrl(psc,(Variable) var,
						      FWD_OFL_CTRL);
		sc_base_remove_variable(ps, (Variable) var);
		ps = region_sc_normalize(ps,2);
		*exact_p = true;
	      }
	    else
	      {
		Pvecteur pv_var = NULL;

		vect_add_elem(&pv_var, (Variable) var, VALUE_ONE);
		ps = sc_projection_ofl_along_variables_with_test
		  (ps, pv_var,  exact_p);
		vect_rm(pv_var);
		ps = region_sc_normalize(ps,2);
	      }
	    gen_free_list(l_phi_var);
	    UNCATCH(overflow_error);
	  }
	}
    }
  return ps;
}

/* void region_exact_projection_along_variable(effect reg, entity var)
 * input    : a region and a variable (a loop index for instance).
 * output   : nothing.
 * modifies : the initial region, which is projected along var.
 *            the approximation is not modified.
 * comment  : overflow errors are trapped here. if it occurs, an empty region
 *           replaces the initial region.
 *
 * FI: it seems that a value rather than a variable is projected.
 */
void region_exact_projection_along_variable(region reg, entity var)
{
  if(store_effect_p(reg)) {
    /* Automatic variables read in a CATCH block need to be declared volatile as
     * specified by the documentation*/
    Psysteme volatile sc;

    sc = region_system(reg);

    if (!SC_UNDEFINED_P(sc) && !sc_empty_p(sc) && !sc_rn_p(sc))
    {
	if (op_statistics_p())
	{
	    nb_proj_var++;
	    if (region_approximation_tag(reg) == is_approximation_exact)
		nb_proj_var_pot_must++;
	}

	ifdebug(3)
	{
	  pips_debug(3, "variable: %s, and initial region :\n", entity_name(var));
	    print_region(reg);
	}

	if (base_contains_variable_p(sc->base, (Variable) var))
	{
	    CATCH(overflow_error)
	    {
		region reg_tmp =
		    reference_whole_region(region_any_reference(reg),
					   region_action(reg));
		cell c_tmp = region_cell(reg_tmp);

		region_system_(reg) = region_system_(reg_tmp);
		if(cell_preference_p(c_tmp))
		  preference_reference(cell_preference(c_tmp)) = reference_undefined;
		else
		  cell_reference(c_tmp) = reference_undefined;
		region_system_(reg_tmp) = newgen_Psysteme(SC_UNDEFINED);
		free_effect(reg_tmp);
		sc_rm(sc);
		if (op_statistics_p()) nb_proj_var_ofl++;
	    }
	    TRY {
		list l_phi_var = region_phi_cfc_variables(reg);

		if (gen_find_eq(var, l_phi_var) == chunk_undefined)
		{
		  ifdebug(8) {
		    pips_debug(8, "system before projection (1): \n");
		    sc_syst_debug(sc);
		  }

		  Psysteme volatile * psc = &sc;
		    sc_projection_along_variable_ofl_ctrl(psc,(Variable) var,
							  FWD_OFL_CTRL);
		    sc_base_remove_variable(sc, (Variable) var);
		    // sometimes produces erroneous systems - BC 11/8/2011
		    // sc = region_sc_normalize(sc,2);
		    ifdebug(8) {
		      pips_debug(8, "system before normalization: \n");
		      sc_syst_debug(sc);
		    }
		    sc = region_sc_normalize(sc,1);
		     ifdebug(8) {
		      pips_debug(8, "system after normalization: \n");
		      sc_syst_debug(sc);
		    }
		    region_system_(reg) = newgen_Psysteme(sc);

		    if (op_statistics_p() &&
			(region_approximation_tag(reg)==is_approximation_exact))
			nb_proj_var_must++;
		}
		else
		{
		    Pvecteur pv_var = NULL;
		    bool is_proj_exact = true;

		    vect_add_elem(&pv_var, (Variable) var, VALUE_ONE);

		    ifdebug(8) {
		      pips_debug(8, "system before projection (2): \n");
		      sc_syst_debug(sc);
		    }

		    sc = sc_projection_ofl_along_variables_with_test
			(sc, pv_var,  &is_proj_exact);
		    vect_rm(pv_var);
		    // sometimes produces erroneous systems - BC 11/8/2011
		    // sc = region_sc_normalize(sc,2);
		    ifdebug(8) {
		      pips_debug(8, "system before normalization: \n");
		      sc_syst_debug(sc);
		    }
		    sc = region_sc_normalize(sc,1);
		    ifdebug(8) {
		      pips_debug(8, "system after normalization: \n");
		      sc_syst_debug(sc);
		    }
		    region_system_(reg)= newgen_Psysteme(sc);

		    if (region_approximation_tag(reg) == is_approximation_exact)
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
}


/*********************************************************************************/
/* Ope'rateurs concernant les syste`mes de contraintes, mais propres au          */
/* des re'gions.                                                                 */
/*********************************************************************************/


static Pcontrainte eq_var_nophi_min_coeff(Pcontrainte contraintes, Variable var);
static Pcontrainte eq_var_phi(Pcontrainte contraintes, Variable var);
static Psysteme region_sc_minimal(Psysteme sc, bool *p_sc_changed_p);


/* Psysteme region_sc_projection_ofl_along_parameters(Psysteme sc,
 *                                                 Pvecteur pv_param,
 *                                                 bool *p_exact)
 * input    : a convex polyhedron sc to project along the parameters contained
 *            in the vector pv_param, which are linked to the PHI variables;
 *            p_exact is a pointer to a bool indicating whether the
 *            projection is exact or not.
 * output   : the polyhedron resulting of the projection.
 * modifies : sc, *p_exact.
 * comment  :
 */
static Psysteme region_sc_projection_ofl_along_parameters(Psysteme sc,
							  Pvecteur pv_param,
							  bool *p_exact)
{
    *p_exact = true;

    for(; (*p_exact == true) && !VECTEUR_NUL_P(pv_param); pv_param = pv_param->succ)
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
 *                                                 bool *p_exact)
 * input    : a convex polyhedron sc to project along the variable param
 *            which is linked to the PHI variables;
 *            p_exact is a pointer to a bool indicating whether the
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
							 bool *p_exact)
{
    Pcontrainte eq;
    Pbase base_sc;
    bool hermite_sc_p;

    pips_debug(8, "begin\n");


    /* fisrt, we search for an explicit equation containing param,
     * but no phi variables
     */
    eq = eq_var_nophi_min_coeff(sc->egalites, param);

    if (!CONTRAINTE_UNDEFINED_P(eq))
    {
	sc = sc_projection_ofl_with_eq(sc, eq, param);
	*p_exact = true;
	debug(8, "region_sc_projection_ofl_along_parameter",
	      "explicit equation found.\n");
	return(sc);
    }

    /* if the last equation is not explicit, it may be implicit
     */
    (void)region_sc_minimal(sc, &hermite_sc_p);

    if (hermite_sc_p)
    {
	if (op_statistics_p()) nb_proj_param_hermite++;

	eq = eq_var_nophi_min_coeff(sc->egalites, param);

	if (!CONTRAINTE_UNDEFINED_P(eq))
	{
	    sc = sc_projection_ofl_with_eq(sc, eq, param);
	    *p_exact = true;
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
	*p_exact = false;
	pips_debug(8, "equation with PHI and param found -> projection not exact.\n");

	return(sc);
    }


    /* if such an equation does not exist, we search for an
     * inequation containing param and a phi variable
     */
    eq = eq_var_phi(sc->inegalites, param);
    if(CONTRAINTE_UNDEFINED_P(eq))
	*p_exact = true;
    else
	*p_exact = false;

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

/* Psysteme region_sc_minimal(Psysteme sc, bool p_sc_changed_p)
 * input    : a polyhedron
 * output   : an equivalent polyhedron, in which the number of equations
 *            containing phi variables is minimal (see report E/185).
 * modifies : sc and p_sc_changed_p. The pointed bool is set to true, if
 *            a new sc is calculated (with the hermite stuff).
 * comment  :
 */
static Psysteme region_sc_minimal(Psysteme sc, bool *p_sc_changed_p)
{
    Pcontrainte eq = sc->egalites;
    int m = nb_elems_list(eq),
        n = base_dimension(sc->base),
        n_phi = base_nb_phi(sc->base);
    Pmatrix A, C, A_phi, A_phi_t, H, P, Q, Q_t, A_min, C_min;
    Value det_P, det_Q;
    Pbase sorted_base = base_dup(sc->base);

    *p_sc_changed_p = false;

    /* the number of equations containing phi variables must be greater than one,
     * otherwise, the system is already minimal. Moreover, the number of phi
     * variables must be greater than 0.
     */
    if (!(n_phi > 0 && (constraints_nb_phi_eq(eq) > 1)))
	return(sc);

    *p_sc_changed_p = true;

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



/* bool region_projection_along_index_safe_p(entity index, range l_range)
 * input    : an loop index and its range
 * output   : true if its projection of regions along index is safe (see
 *            conditions in report E/185).
 * modifies : nothing
 * comment  :
 */
bool region_projection_along_index_safe_p(entity __attribute__ ((unused)) index,
					     range l_range)
{
    bool projection_of_index_safe = false;
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
    Pbase scbase = base_copy((*psc)->base);

    if (!VECTEUR_NUL_P(pv)) {
		for (pv1 = pv;!VECTEUR_NUL_P(pv1) && !SC_UNDEFINED_P(*psc); pv1=pv1->succ) {

		  sc_projection_along_variable_ofl_ctrl(psc,vecteur_var(pv1), ofl_ctrl);

		  /* In case of big sc, we might consider a better order for the projection.
			* Example: 2 phases of elimination (must-projection):
			*  - first: elimination of PHI variables: PHI3, PHI2, PHI1
			*  - then: elimination of PSI variables: PSI3, PSI2, PSI1
			* Ex: PHI3 (297 inequations), then PHI2 => explosion (21000) = PSI1 (1034)
			*But PHI3(297 inequations), then PSI1 (191), then PHI2 => ok (1034) DN 21/1/03
			*
			* The non_exact projection:
			* if the projection excat fails, then return the modified sc, without variable in base.
			*/

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
