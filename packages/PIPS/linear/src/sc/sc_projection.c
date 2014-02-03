/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

/* Package systeme de contraintes (SC) : sc_projection.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Projection of a system of constraints (equalities and inequalities)
 * along one or several variables. The variables are always eliminated,
 * even if the projection is not exact, i.e. introduces integer points
 * that do not belong to the projections of the initial integer points.
 *
 * Arguments of these functions :
 *
 * - s or sc is the system of constraints.
 *
 * - ofl_ctrl is the way overflow errors are handled
 *
 *     ofl_ctrl == NO_OFL_CTRL
 *               -> overflow errors are not handled
 *
 *     ofl_ctrl == OFL_CTRL
 *               -> overflow errors are handled in the called function
 *                  (not often available, check first!)
 *
 *     ofl_ctrl == FWD_OFL_CTRL
 *               -> overflow errors must be handled by the calling function
 *
 * Comments :
 *
 *  The base of the Psysteme is not always updated. Check the comments at the
 *  head of each function for more details.
 *
 *  In the functions that update the base, there is an assert if the variables
 *  that are eliminated do not belong to the base.
 *
 *  No attempt is made at speeding the projection when many variables
 *  are projected simultaneously. The projection could be speeded up
 *  when all variables used in the constraints are projected and when
 *  all variables projected are used in convex components not related
 *  to the preserved variables. In the later case, the convex
 *  components must be checked for emptiness. The system decomposition
 *  in convex components is performed by function xyz... (see Beatrice
 *  Creusillet)
 *
 * Authors :
 *
 * Remi Triolet, Malik Imadache, Francois Irigoin, Yi-Qing Yang,
 * Corinne Ancourt, Be'atrice Creusillet, Fabien Coelho, Duong Nguyen,
 * Serge Guelton.
 *
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include <signal.h>

#define EXCEPTION_PRINT_PROJECTION true

// Duong Nguyen: constants used to filter out the most complicated
// systems of constraints
#ifdef FILTERING

#define FILTERING_DIMENSION_PROJECTION filtering_dimension_projection
#define FILTERING_NUMBER_CONSTRAINTS_PROJECTION filtering_number_constraints_projection
#define FILTERING_DENSITY_PROJECTION filtering_density_projection
#define FILTERING_MAGNITUDE_PROJECTION (Value) filtering_magnitude_projection

#define FILTERING_TIMEOUT_PROJECTION filtering_timeout_projection

static bool PROJECTION_timeout = false;

static void
catch_alarm_projection (int sig)
{
  alarm(0); /*clear the alarm*/
  PROJECTION_timeout = true;
}
#endif

/* void sc_projection_along_variable_ofl_ctrl(Psysteme *psc, Variable v,
 *                                            int ofl_ctrl)
 * input    : the systeme to project, a variable along which the projection
 *            will be performed.
 * output   : the projected system. If it is not feasible, returns
 *            sc_empty(initial_base).
 * modifies : sc
 * comment  :
 *             The base of *psc is not updated.
 *             It is not even checked if the variable belongs to the base.
 *
 *             The case ofl_ctrl == OFL_CTRL is not handled, because
 *             the user may want an SC_UNDEFINED, or an sc_empty(sc->base)
 *             or an sc_rn(sc->base) as a result. It is not homogeneous.
 *
 *             projection d'un systeme sc selon une var v
 *             et expansion virtuelle en un cylindre d'axe v.
 *
 * Modifications:
 *
 *  - ajout de la normalisation de l'equation eq avant les substitutions;
 *    cette normalisation permet de detecter une equation infaisable; le
 *    probleme a ete mis en evidence par Yi-Qing dans le test de dependance
 *    de Linear/C3 Library pour T(2*i+4*j) vs T(2*i+2*j+1) donnant 2*j + 2*di + 2*dj + 1 =0;
 *    la projection de j faisait perdre la non-faisabilite; Francois Irigoin,
 *    8 novembre 1991
 *
 * - Added timeout control in
 *   sc_fourier_motzkin_variable_elimination_ofl_ctrl.  Better use
 *   sc_projection_along_variable_ofl_ctrl_timeout_ctrl. Duong Nguyen
 *   29/10/2002
 */
void sc_projection_along_variable_ofl_ctrl(psc, v, ofl_ctrl)
Psysteme volatile *psc;
Variable v;
int ofl_ctrl;
{

    Pcontrainte eq;
    Value coeff;

    /* trouver une egalite ou v a un coeff. minimal en valeur absolu */
    eq = contrainte_var_min_coeff((*psc)->egalites, v, &coeff, false);

    if (eq != NULL) {
	if(!egalite_normalize(eq)) {
	  Psysteme new_psc = sc_empty((*psc)->base);
	  (*psc)->base = BASE_UNDEFINED;
	  sc_rm(*psc);
	  *psc = new_psc;
	  return;
	}
	else
	{
	    /* si au moins une egalite contient v avec un coeff. non nul */
	    *psc = sc_variable_substitution_with_eq_ofl_ctrl
		(*psc,eq,v, ofl_ctrl);
	}
    }
    else {
	bool fm_int_exact_p;
	/* v n'apparait dans aucune egalite, il faut l'eliminer des inegalites
	   par des combinaisons (fonction combiner) */

	fm_int_exact_p =
	    sc_fourier_motzkin_variable_elimination_ofl_ctrl(*psc, v, false,
							     false, ofl_ctrl);
	if (fm_int_exact_p == false) {
	    /* detection de la non faisabilite du Psysteme */
	  Psysteme new_psc = sc_empty((*psc)->base);
	  (*psc)->base = BASE_UNDEFINED;
	  sc_rm(*psc);
	  *psc = new_psc;
	  return;
	}
	(*psc)->nb_ineq = nb_elems_list((*psc)->inegalites);
    }

}



/* void sc_and_base_projection_along_variable_ofl_ctrl(Psysteme psc,Variable v)
 * input    : a Psysteme to project along the input variable.
 * output   : nothing.
 * modifies : the initial Psysteme and its base.
 * comment  : The initial system is projected along the variable, which is then
 *            eliminated from the base. assert if the variable does not belong to
 *            the base.
 */
void sc_and_base_projection_along_variable_ofl_ctrl(psc,v, ofl_ctrl)
Psysteme volatile *psc;
Variable v;
int ofl_ctrl;
{
    sc_projection_along_variable_ofl_ctrl(psc,v,ofl_ctrl);
    sc_base_remove_variable(*psc,v);
}




/* void sc_projection_ofl_along_variables(Psysteme *psc, Pvecteur pv)
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
 */
void sc_projection_along_variables_ofl_ctrl(psc, pv, ofl_ctrl)
Psysteme *psc;
Pvecteur pv;
int ofl_ctrl;
{
    Pvecteur pv1;

    if (!VECTEUR_NUL_P(pv)) {
	for (pv1 = pv;!VECTEUR_NUL_P(pv1) && !SC_UNDEFINED_P(*psc); pv1=pv1->succ) {

	    sc_projection_along_variable_ofl_ctrl(psc,vecteur_var(pv1),
						       ofl_ctrl);
	    if (!SC_UNDEFINED_P(*psc)) {
		sc_base_remove_variable(*psc,vecteur_var(pv1));
		(void) sc_nredund(psc);
	    }

	}
    }

}


/* void sc_projection_with_test_along_variables_ofl_ctrl(Psysteme *psc,
 *                                                      Pvecteur pv,
 *                                                      bool *is_proj_exact)
 * input    : the systeme to project along the variables of the vector pv.
 *            *is_proj_exact is supposed to be initialized.
 * output   : nothing.
 * modifies : *is_proj_exact is set to false if the projection is not exact.
 * comment  : The initial systeme is successively projected along
 *            all the variables of pv, even if the projection is not exact.
 *            The projection is done first by eliminating variables in the
 *            part of equations (the variables which coefficient is 1 are
 *            considered before : in such case an integer elimination is
 *            performed, and the projection is exact); the rest is projected
 *            using Fourier-Motzkin, with a test to know if the projection
 *            is exact.
 *
 *             The case ofl_ctrl == OFL_CTRL is not handled, because
 *             the user may want an SC_UNDEFINED, or an sc_empty(sc->base)
 *             or an sc_rn(sc->base) as a result. It is not homogeneous.
 */
void sc_projection_along_variables_with_test_ofl_ctrl(psc, pv, is_proj_exact,
							  ofl_ctrl)
Psysteme *psc;
Pvecteur pv;
bool *is_proj_exact; /* this bool is supposed to be initialized */
int ofl_ctrl;
{
    Pcontrainte eq;
    Pvecteur current_pv, pve, pvr;
    Variable v;
    Value coeff;
    Psysteme sc = *psc;

    pve = vect_copy(pv);

    /* elimination of variables with the equations of sc */

    if (pve != NULL && sc->nb_eq != 0 && !sc_empty_p(sc)) {
	/* first carry out integer elimination when possible */

	pvr = NULL;
	current_pv = pve;

	while (!VECTEUR_NUL_P(current_pv) &&
	       (sc->nb_eq != 0) && !sc_empty_p(sc)) {
	    v = current_pv->var;
	    eq = eq_v_min_coeff(sc->egalites, v, &coeff);
	    if ((eq == NULL) || (value_notone_p(coeff))) {
		pvr = current_pv;
		current_pv = current_pv->succ;
	    }
	    else {
		/* the coefficient of v in eq is 1 : carry out integer
		 * elimination for v with the other constraints of sc */

		sc = sc_variable_substitution_with_eq_ofl_ctrl
		    (sc, eq, v, ofl_ctrl);

		if (sc_empty_p(sc))
		    break;

		sc = sc_normalize(sc);

		if (sc_empty_p(sc))
		    break;

		/* v has been eliminated : remove it from pve */
		if (pvr == NULL) /* v is the fisrt */
		    pve = current_pv = current_pv->succ;
		else
		    pvr->succ = current_pv = current_pv->succ;

	    } /* if-else */
	} /* while */

    } /* if */


    /* carry out non-exact elimination if necessary and possible */
    if (pve != NULL && sc->nb_eq != 0 && !sc_empty_p(sc)) {
	/* first carry out integer elimination when possible */

	pvr = NULL;
	current_pv = pve;

	while (!VECTEUR_NUL_P(current_pv) &&
	       (sc->nb_eq != 0) && !sc_empty_p(sc)) {

	    v = current_pv->var;
	    eq = eq_v_min_coeff(sc->egalites, v, &coeff);
	    if ((eq == NULL)) {
		pvr = current_pv;
		current_pv = current_pv->succ;
	    }
	    else {
		/* elimination for v with the constraints of sc other than eq*/

		if (*is_proj_exact) *is_proj_exact = false;

		sc = sc_variable_substitution_with_eq_ofl_ctrl
		    (sc, eq, v, ofl_ctrl);

		sc = sc_normalize(sc);
		if (sc_empty_p(sc))
		    break;

		/* v has been eliminated : remove it from pve */
		if (pvr == NULL) /* v is the fisrt */
		    pve = current_pv = current_pv->succ;
		else
		    pvr->succ = current_pv = current_pv->succ;

	    } /* if-else */
	} /* while */

    } /* if */

    /* carry out the elimination of Fourier-Motzkin for the remaining variables */
    if (pve != NULL && !sc_empty_p(sc)) {
        Pbase base_sc = base_copy(sc->base);

	current_pv = pve;
	while (current_pv != NULL) {
	    v = current_pv->var;

	    if (! sc_fourier_motzkin_variable_elimination_ofl_ctrl(sc, v, true,
								   is_proj_exact,
								   ofl_ctrl))
	    {
		/* sc_rm(sc); ???????? BC ??????? */
		sc = sc_empty(base_sc);
		base_sc = NULL;
		break;
	    }

	    sc = sc_normalize(sc);
	    if (sc_empty_p(sc))
		break;
	    current_pv = current_pv->succ;

	} /* while */
	if (base_sc != NULL) base_rm(base_sc);

    } /* if */

    sc->nb_ineq = nb_elems_list(sc->inegalites);
    sc->nb_eq = nb_elems_list(sc->egalites);

    for(current_pv = pv; current_pv != NULL; current_pv = current_pv->succ) {
	v = current_pv->var;
	sc_base_remove_variable(sc, v);
    }

    vect_rm(pve);
    *psc = sc;
}





/* Psysteme sc_variable_substitution_with_eq_ofl_ctrl(Psysteme sc, Pcontrainte eq,
 *                                                    Variable v, int ofl_ctrl)
 * input    : a system of contraints sc, a contraint eq that must belong to sc,
 *            and a variable v that appears in the constraint eq.
 * output   : a Psysteme which is the projection of sc along v using the equation
 *            eq.
 * modifies : sc.
 * comment  : The case ofl_ctrl == OFL_CTRL is not handled, because it would
 *            have no sense here.
 */
Psysteme sc_variable_substitution_with_eq_ofl_ctrl(sc, eq, v, ofl_ctrl)
Psysteme sc;
Pcontrainte eq;
Variable v;
int ofl_ctrl;
{
  Pcontrainte pr;
  Pcontrainte current_eq;

  pr = NULL;
  current_eq = sc->egalites;
  while(current_eq != NULL)
    {
      /* eq must be removed from the list of equalities of sc */
      if(current_eq == eq)
	{
	  if (pr == NULL){ /* eq is at the head of the list */
	    sc->egalites = current_eq = current_eq->succ;
	  }
	  else
	    {
	      pr->succ = current_eq = current_eq->succ;
	    }
	}
      else
	{
	  switch (contrainte_subst_ofl_ctrl(v, eq, current_eq, true, ofl_ctrl))
	    {
	    case 0 :
	      {
		/* the substitution found that the system is not feasible */
		Psysteme new_sc = sc_empty(sc->base);
		sc->base=BASE_UNDEFINED;
		sc_rm(sc);
		sc = new_sc;
		eq = contrainte_free(eq);
		return(sc);
		break;
	      }

	    case -1 : {
	      /* the substitution created a trivial equation (0==0)
	       * which must be eliminated */
	      Pcontrainte pc = current_eq;
	      if (pr == NULL)
		sc->egalites = current_eq = current_eq->succ;
	      else
		pr->succ = current_eq = current_eq->succ;
	      contrainte_free(pc);
	      pc = NULL;
	      break;
	    }

	    case 1 :
	      pr = current_eq;
	      current_eq = current_eq->succ;
	      break;
	    }
	}
    }

  pr = NULL;
  current_eq = sc->inegalites;
  while(current_eq != NULL) {

    switch (contrainte_subst_ofl_ctrl(v, eq, current_eq, false,ofl_ctrl)) {

    case 0 :
      {
	/* the substitution found that the system is not feasible */
	Psysteme new_sc = sc_empty(sc->base);
	sc->base=BASE_UNDEFINED;
	sc_rm(sc);
	sc = new_sc;
	eq = contrainte_free(eq);
	return(sc);
	break;
      }

    case -1 : {
      /* the substitution created a trivial inequation (0=<cste)
       * which must be eliminated */
      Pcontrainte pc = current_eq;
      if (pr == NULL)
	sc->inegalites = current_eq = current_eq->succ;
      else
	pr->succ = current_eq = current_eq->succ;
      contrainte_free(pc);
      pc = NULL;
      break;
    }

    case 1 :
      pr = current_eq;
      current_eq = current_eq->succ;
      break;
    }
  }

  eq = contrainte_free(eq);
  sc->nb_eq = nb_elems_list(sc->egalites);
  sc->nb_ineq = nb_elems_list(sc->inegalites);
  return(sc);
}

/* Psysteme sc_simple_variable_substitution_with_eq_ofl_ctrl
 *     (Psysteme sc, Pcontrainte def, Variable v, int ofl_ctrl)
 * input    : a system of contraints sc, a contraint eq that must belong to sc,
 *            and a variable v that appears in the constraint eq.
 * output   : a Psysteme which is the projection of sc along v using
 *            equation eq.
 * modifies : sc.
 * comment  : The case ofl_ctrl == OFL_CTRL is not handled, because it would
 *            have no sense here.
 *
 *            Trivial constraints and trivially non-feasible constraints
 *            are not tested but intentionnally left in sc
 */
Psysteme sc_simple_variable_substitution_with_eq_ofl_ctrl(sc, def, v, ofl_ctrl)
Psysteme sc;
Pcontrainte def;
Variable v;
int ofl_ctrl;
{
    Pcontrainte eq = CONTRAINTE_UNDEFINED;
    Pcontrainte ineq = CONTRAINTE_UNDEFINED;

    assert(!SC_UNDEFINED_P(sc));

    /* FI/CA: in case the overflow must be handled by this function,
       the function should catch it and remove the constraint that
       generate the overflow. If this is ever implemented,
       sc_normalize2() should be simplified by delegating overflow
       control to this function. */

    for(eq = sc_egalites(sc); !CONTRAINTE_UNDEFINED_P(eq);
	eq = contrainte_succ(eq)){
      if (eq!=def) /* skip if aliased! */
	(void) contrainte_subst_ofl_ctrl(v, def, eq, true, ofl_ctrl);
    }

    for(ineq = sc_inegalites(sc); !CONTRAINTE_UNDEFINED_P(ineq);
	ineq = contrainte_succ(ineq)){
	(void) contrainte_subst_ofl_ctrl(v, def, ineq, false, ofl_ctrl);
    }
    return sc;
}

/* bool sc_fourier_motzkin_variable_elimination_ofl_ctrl(Psysteme sc,
 *                     Variable v,
 *                     bool integer_test_p, bool *integer_test_res_p,
 *                     int ofl_ctrl)
 * input    : a systeme of constraints sc, a variable to eliminate using
 *            Fourier-Motzking elimination, a bool (integer_test_p)
 *            true when the test of sufficient conditions for an exact
 *            projection must be performed, a pointer towards an already
 *            initialized bool (integer_test_res_p) to store the
 *            logical and of its initial value and of the result of the
 *            previous test, and an integer to indicate how the overflow
 *            error must be handled (see sc-types.h). The case OFL_CTRL
 *            is of no interest here.
 * output   : true if sc is feasible (w.r.t the inequalities),
 *            false otherwise;
 * modifies : sc, *integer_test_res_p if integer_test_p == true.
 * comment  : eliminates v from sc by combining the inequalities of sc
 *            (Fourier-Motzkin). If this elimination is not exact,
 *            *integer_test_res_p is set to false.
 *            *integer_test_res_p must be initialized.
 *
 */

bool sc_fourier_motzkin_variable_elimination_ofl_ctrl(sc,v, integer_test_p,
						integer_test_res_p, ofl_ctrl)
Psysteme sc;
Variable v;
bool integer_test_p;
bool *integer_test_res_p;
int ofl_ctrl;
{
    Psysteme sc1 = SC_UNDEFINED;
    Pcontrainte pos, neg, nul;
    Pcontrainte pcp, pcn;
    Value c;
    int nnul;

    if ((sc->inegalites) == NULL) {
      return(true);
    }

    CATCH(timeout_error|overflow_error){

      ifscdebug(5) {
	fprintf(stderr,"\n[sc_fourier_motzkin_variable_elimination_ofl_ctr] overflow or timeoutl!!!\n");}

#ifdef FILTERING
      alarm(0);
#endif
      if (ofl_ctrl==FWD_OFL_CTRL) {
	THROW(overflow_error);
      }
      else {
	ifscdebug(5) {
	  fprintf(stderr,"\n[fourier_motzkin_variable_elimination_ofl_ctrl]: ofl_ctrl <> FWD_OFL_CTRL \n");}
	/*return false means system non feasible. return true, mean ok => both wrong		*/
      }
    }
    TRY
    {
      Pcontrainte volatile pc = sc->inegalites;
#ifdef FILTERING
	signal(SIGALRM, catch_alarm_projection);
	alarm(FILTERING_TIMEOUT_PROJECTION);
#endif
    if (integer_test_p) sc1 = sc_copy(sc);

    pos = neg = nul = NULL;
    nnul = 0;
    while (pc != NULL) {
	Pcontrainte pcs = pc->succ;
	c = vect_coeff(v,pc->vecteur);
	if (value_pos_p(c)) {
	    pc->succ = pos;
	    pos = pc;
	}
	else if (value_neg_p(c)) {
	    pc->succ = neg;
	    neg = pc;
	}
	else {
	    pc->succ = nul;
	    nul = pc;
	    nnul++;
	}

	pc = pcs;
    }
    sc->inegalites = NULL;
    sc->nb_ineq = 0;

    for (pcp = pos; pcp != NULL; pcp = pcp->succ) {
	for (pcn = neg; pcn != NULL; pcn = pcn->succ) {

	    Pcontrainte pcnew;

	    if (integer_test_p) {
		bool int_comb_p;
		pcnew =
		    sc_integer_inequalities_combination_ofl_ctrl
			(sc1, pcp, pcn, v, &int_comb_p, ofl_ctrl);
		*integer_test_res_p = *integer_test_res_p && int_comb_p;
	    }
	    else
		pcnew = inegalite_comb_ofl_ctrl(pcp, pcn, v, ofl_ctrl);

	    if (contrainte_constante_p(pcnew)) {
		if (contrainte_verifiee(pcnew,false)) {
		    contrainte_free(pcnew);
		}
		else {
		    contraintes_free(pos);
		    contraintes_free(neg);
		    contraintes_free(nul);
		    contraintes_free(pcnew);
		    if (integer_test_p) sc_rm(sc1);
#ifdef FILTERING
alarm(0);
#endif
		    UNCATCH(timeout_error|overflow_error);
		    return(false);
		}
	    }
	    else {
		pcnew->succ = nul;
		nul = pcnew;
		nnul++;
	    }
	}
    }/*of for*/

    /* apres les combinaisons eliminer les elements devenus inutiles */
    contraintes_free(pos);
    contraintes_free(neg);

    /* mise a jour du systeme */
    sc->inegalites = nul;
    sc->nb_ineq = nnul;
    if (integer_test_p) sc_rm(sc1);
    }/*of TRY*/

#ifdef FILTERING
alarm(0);
/*clear the alarm*/
#endif
    UNCATCH(timeout_error|overflow_error);

    return true;
}

/* Psysteme sc_projection_on_variables
 *    (Psysteme sc, Pbase index_base, Pvecteur pv)
 * input    :
 * output   : a Psysteme representing the union of all the systems
 *            resulting of the projection of the system sc on each
 *            variable contained in vecteur index_base.
 *            pv is the list of variables that might be eliminated from
 *            the system (symbolic constants are not in).
 * modifies :
 * comment  :
 */
Psysteme sc_projection_on_variables(
  Psysteme sc,
  Pbase index_base,
  Pvecteur pv)
{
  Pvecteur lvar_proj;
  Psysteme sc1 = sc_init_with_sc(sc);
  // Automatic variables read in a CATCH block need to be declared volatile as
  // specified by the documentation
  volatile Psysteme sc2;
  Variable var;

  if (!VECTEUR_NUL_P(pv))
  {
    volatile Pvecteur pv1, pv2;
    lvar_proj = vect_copy(pv);
    for (pv1 = index_base;!VECTEUR_NUL_P(pv1); pv1=pv1->succ) {
	    sc2 = sc_copy(sc);
	    var = vecteur_var(pv1);
	    vect_erase_var(&lvar_proj,var);
	    for (pv2 = lvar_proj;!VECTEUR_NUL_P(pv2); pv2=pv2->succ) {

        CATCH(overflow_error) {
          sc_elim_var(sc2, vecteur_var(pv2));
        }
        TRY {
          sc_projection_along_variable_ofl_ctrl
            (&sc2,vecteur_var(pv2), NO_OFL_CTRL);
          UNCATCH(overflow_error);
        }
        sc2 = sc_normalize(sc2);
        if (SC_EMPTY_P(sc2)) {
          sc2 = sc_empty(base_copy(sc->base));
          break;
        }
        else {
          build_sc_nredund_1pass(&sc2);
        }
	    }
	    vect_chg_coeff(&lvar_proj, var, VALUE_ONE);
	    sc1 = sc_intersection(sc1,sc1,sc2);
	    sc1 = sc_normalize(sc1);
	    if (SC_EMPTY_P(sc1)) {
        sc1 = sc_empty(base_copy(sc->base));
        break;
	    }
    }
  }
  return sc1;
}



/* Pcontrainte sc_integer_inequalities_combination_ofl_ctrl(Psysteme sc,
 *                                            Pcontrainte posit, negat,
 *                                                          Variable v,
 *                                      bool *integer_combination_p,
 *                                                        int ofl_ctrl)
 * input    : many things !
 * output   : a constraint which is the combination of the constraints
 *            posit and negat by the variable v.
 * modifies : *integer_combination_p.
 * comment  :
 *      *integer_combination_p is set to true, if the Fourier Motzkin
 *      projection is equivalent to the "integer projection".
 *      else, it is set to false.
 *
 * Redundancy is checked using contrainte_reverse() and feasability
 * checking. Non-redundant constraints in rational may be discarded.
 */
Pcontrainte sc_integer_inequalities_combination_ofl_ctrl
  (sc,posit,negat,v, integer_combination_p, ofl_ctrl)
Psysteme sc;
Pcontrainte posit, negat;
Variable v;
bool *integer_combination_p;
int ofl_ctrl;
{
    Value cp, cn;
    Pcontrainte ineg = NULL;

    if (!CONTRAINTE_UNDEFINED_P(posit) && !CONTRAINTE_UNDEFINED_P(negat))
    {
	/* ??? one is positive and the other is negative! FC */
	cp = vect_coeff(v, posit->vecteur);
	cp = value_abs(cp);
	cn = vect_coeff(v, negat->vecteur);
	cn = value_abs(cn);

	ineg = contrainte_new();
	ineg->vecteur = vect_cl2_ofl_ctrl
	    (cp, negat->vecteur, cn, posit->vecteur, ofl_ctrl);

	if (value_gt(cp,VALUE_ONE) && value_gt(cn,VALUE_ONE))/* cp>1 && cn>1 */
	{
	    /* tmp = cp*cn - cp - cn + 1. Note: cp and cn are >0! */
	    Value tmp = value_mult(cp,cn);
	    tmp = value_minus(value_plus(tmp, VALUE_ONE), value_plus(cp, cn));

	    if (contrainte_constante_p(ineg) &&
		value_le(vect_coeff(TCST, ineg->vecteur), value_uminus(tmp)))
		*integer_combination_p = true;
	    else {
		Pcontrainte ineg_test = contrainte_copy(ineg);
		Psysteme sc_test = sc_copy(sc);

		vect_add_elem(&(ineg_test->vecteur), TCST, tmp);

		contrainte_reverse(ineg_test);

		sc_add_inegalite(sc_test,ineg_test);

		if (!sc_rational_feasibility_ofl_ctrl(sc_test, OFL_CTRL, true))
		    *integer_combination_p = true;
		else
		   *integer_combination_p = false;
		sc_rm(sc_test);
	    }
	}
	else {
	    *integer_combination_p = true;
	}

    }
    return(ineg);
}


/* FOR BACKWARD COMPATIBILITY ONLY. DO NOT USE ANYMORE, PLEASE. */

Psysteme sc_projection(Psysteme sc, Variable v)
{
  CATCH(overflow_error) {
    sc_elim_var(sc , v);
  }
  TRY {
    sc_projection_along_variable_ofl_ctrl(&sc,v, NO_OFL_CTRL);
    UNCATCH(overflow_error);
  }
  if (sc_empty_p(sc)) {
    sc_rm(sc);
    sc = SC_EMPTY;
  }
  return sc;
}

Psysteme sc_projection_ofl(Psysteme sc, Variable v)
{
	sc_projection_along_variable_ofl_ctrl(&sc,v, FWD_OFL_CTRL);
  if (sc_empty_p(sc)) {
    sc_rm(sc);
    sc = SC_EMPTY;
  }
  return sc;
}



Psysteme sc_projection_pure(sc,v)
Psysteme sc;
Variable v;
{
    CATCH(overflow_error) {
	sc_elim_var(sc , v);
    }
    TRY {
	sc_and_base_projection_along_variable_ofl_ctrl(&sc,v,NO_OFL_CTRL);
	UNCATCH(overflow_error);
    }
    if (sc_empty_p(sc)) {
	sc_rm(sc);
	sc = SC_EMPTY;
    }
    return(sc );
}


Psysteme sc_projection_ofl_along_variables(sc,pv)
Psysteme sc;
Pvecteur pv;
{
    sc_projection_along_variables_ofl_ctrl(&sc,pv, FWD_OFL_CTRL);
    return(sc);
}



Psysteme sc_projection_ofl_along_variables_with_test(sc, pv, is_proj_exact)
Psysteme sc;
Pvecteur pv;
bool *is_proj_exact; /* this bool is supposed to be initialized */
{
    sc_projection_along_variables_with_test_ofl_ctrl(&sc, pv, is_proj_exact,
							  FWD_OFL_CTRL);
    return(sc);
}


Psysteme sc_projection_by_eq(sc, eq, v)
Psysteme sc;
Pcontrainte eq;
Variable v;
{
    Psysteme res;
    res = sc_variable_substitution_with_eq_ofl_ctrl(sc, eq, v, FWD_OFL_CTRL);

    if (sc_empty_p(res)){
	sc_rm(res);
	res = SC_EMPTY;
    }

    return(res);
}

bool cond_suff_comb_integer_ofl_ctrl(sc,posit,negat, v, ofl_ctrl)
Psysteme sc;
Pcontrainte posit,negat;
Variable v;
int ofl_ctrl;
{
    Pcontrainte ineg = CONTRAINTE_UNDEFINED;
    bool result;

    ineg = sc_integer_inequalities_combination_ofl_ctrl(sc, posit, negat, v,
							&result, ofl_ctrl);
    contrainte_rm(ineg);
    return(result);

}



Psysteme sc_projection_optim_along_vecteur(volatile Psysteme sc, Pvecteur pv)
{
  CATCH(overflow_error) {
    /* sc_rm(sc1); */
    return sc;
  }
  TRY {
    Psysteme sc1 = sc_copy(sc);
    bool exact = true;
    sc1 = sc_projection_ofl_along_variables_with_test(sc1,pv,&exact);
    sc_rm(sc);
    UNCATCH(overflow_error);
    return sc1;
  }
}


Pcontrainte eq_v_coeff_min(contraintes, v)
Pcontrainte contraintes;
Variable v;
{
    Value coeff;
    return contrainte_var_min_coeff(contraintes, v, &coeff, true);
}

bool sc_expensive_projection_p(sc,v)
Psysteme sc;
Variable v;
{
    bool expensive = false;
    /* if Variable v is constrained by  equalities, the system size
       is kept (-1) after projection */
    if (!var_in_lcontrainte_p(sc->egalites,v)
	&&  sc->nb_ineq > 20 /* was max for FM */) {
	Pcontrainte ineg;
	int nb_pvar = 0, nb_nvar = 0;
	for (ineg=sc->inegalites;!CONTRAINTE_UNDEFINED_P(ineg);
	     ineg=ineg->succ) {
	    int coeff= vect_coeff(v,ineg->vecteur);
	    if (coeff >0)  nb_pvar ++;
	    else if (coeff<0) nb_nvar ++;
	}
       if (nb_pvar*nb_nvar>200)
	   expensive=true;
    }
    return (expensive);


}

/* from sc_common_projection_convex_hull...
   extract exact equalities on cl, put them in common,
   and project them in cl, s1 and s2...
 */
#define SUBS(s,e,v) \
  sc_simple_variable_substitution_with_eq_ofl_ctrl(s,e,v,OFL_CTRL)
/* sc_variable_substitution_with_eq_ofl_ctrl(s,contrainte_copy(e),v,OFL_CTRL) */

void sc_extract_exact_common_equalities
  (Psysteme cl, Psysteme common, Psysteme s1, Psysteme s2)
{
  Pcontrainte eq, neq, peq;
  for (peq = CONTRAINTE_UNDEFINED,
	 eq = sc_egalites(cl),
	 neq = eq? eq->succ: CONTRAINTE_UNDEFINED;
       eq;
       peq = eq==neq? peq: eq,
	 eq = neq,
	 neq = eq? eq->succ: CONTRAINTE_UNDEFINED)
  {
    Variable v = vect_one_coeff_if_any(eq->vecteur);
    if (v)
    {
      /* unlink eq and update count. */
      if (peq) peq->succ = neq; else sc_egalites(cl) = neq;
      sc_nbre_egalites(cl)--;

      /* put equality back in common. */
      eq->succ = sc_egalites(common);
      sc_egalites(common) = eq;
      sc_nbre_egalites(common)++;

      /* drop variable. */
      s1 = SUBS(s1, eq, v);
      s2 = SUBS(s2, eq, v);
      cl = SUBS(cl, eq, v);

      eq = neq; /* let's hope neq is still there... */
    }
  }
}

/* make exact projection of simple equalities where possible,
 * so as to simplify the system by avoiding X == Y and Y == 3.
 * poor's man normalization;-) should use Hermite for better results?
 * the result is not deterministic, we should iterate till stable,
 * however no information is available from substitutions to tell
 * whether the system was changed or not...
 */
void sc_project_very_simple_equalities(Psysteme s)
{
  Pcontrainte e;
  for (e = sc_egalites(s); e; e = e->succ)
  {
    Variable v = contrainte_simple_equality(e);
    if (v) SUBS(s, e, v);
  }
}

/* void sc_projection_along_variable_ofl_ctrl_timeout_ctrl(psc, v ofl_ctrl)
 *
 * See (void) sc_projection_along_variable_ofl_ctrl(psc, v, ofl_ctrl).
 *
 * We need a function that returns nothing, modifies the input
 * systeme, but can print the original system of constraints only in
 * case of failure, it means we have to catch overflow_error from
 * sc_fourier_motzkin_variable_elimination_ofl_ctrl, in order to print
 * the systems only in Linear.  This function will make a copy of the
 * system, and print it to stderr if there is an exception.  Surely
 * the copy will be removed afterward. (Duong Nguyen 17/7/02) add SIZE
 * CONTROL
*/

void sc_projection_along_variable_ofl_ctrl_timeout_ctrl(psc, v, ofl_ctrl)
Psysteme volatile *psc;
Variable v;
int ofl_ctrl;
{
  /* static bool DN = false; */
  /* Automatic variables read in a CATCH block need to be declared volatile as
   * specified by the documentation*/
  Psysteme volatile sc;
  Pbase volatile base_saved;
  static int volatile projection_sc_counter = 0;

  sc= sc_copy(*psc);
  base_saved = base_copy((*psc)->base);

  if (sc==NULL) {
    sc_rm(*psc);
    sc_rm(sc);
    *psc = sc_empty(base_saved);
    return;
  }

  ifscdebug(5) {projection_sc_counter ++;}


  /*We can put the size filters here! filtering timeout is integrated in the methods themself */
  /* size filtering: dimension, number_constraints, density, magnitude */

#ifdef FILTERING

  PROJECTION_timeout = false;
  /*Begin size filters */

  if (true) { /* all these stuff in a block*/
    Value magnitude;
    int dimens, nb_cont_eq = 0, nb_ref_eq = 0, nb_cont_in = 0, nb_ref_in = 0;

    dimens = sc->dimension; value_assign(magnitude,VALUE_ZERO);
    decision_data(sc_egalites(sc), &nb_cont_eq, &nb_ref_eq, &magnitude, 1);
    decision_data(sc_inegalites(sc), &nb_cont_in, &nb_ref_in, &magnitude, 1);

    if ((FILTERING_DIMENSION_PROJECTION>0)&&(dimens>=FILTERING_DIMENSION_PROJECTION)) {
      char *directory_name = "projection_dimension_filtering_SC_OUT";
      sc_default_dump_to_files(sc,projection_sc_counter,directory_name);
    }
    if ((FILTERING_NUMBER_CONSTRAINTS_PROJECTION>0)&&((nb_cont_eq + nb_cont_in) >= FILTERING_NUMBER_CONSTRAINTS_PROJECTION)) {
      char *directory_name = "projection_number_constraints_filtering_SC_OUT";
      sc_default_dump_to_files(sc,projection_sc_counter,directory_name);
    }
    if ((FILTERING_DENSITY_PROJECTION>0)&&((nb_ref_eq + nb_ref_in) >= FILTERING_DENSITY_PROJECTION)) {
      char *directory_name = "projection_density_filtering_SC_OUT";
      sc_default_dump_to_files(sc,projection_sc_counter,directory_name);
    }
    if ((value_notzero_p(FILTERING_MAGNITUDE_PROJECTION))&&(value_gt(magnitude,FILTERING_MAGNITUDE_PROJECTION))) {
      char *directory_name = "projection_magnitude_filtering_SC_OUT";
      sc_default_dump_to_files(sc,projection_sc_counter,directory_name);
    }
  }
  /*End size filters*/
#endif

  CATCH(overflow_error) {

    if (EXCEPTION_PRINT_PROJECTION) {
      char *directory_name = "projection_fail_SC_OUT";
      sc_default_dump_to_files(sc,projection_sc_counter,directory_name);
    }
    sc_rm(sc);

    if (ofl_ctrl==FWD_OFL_CTRL) {
      base_rm(base_saved);
      /*The modified sc is returned to the calling function, so we can see why the projection fails*/
      RETHROW();
    } else {
      ifscdebug(5) {
	fprintf(stderr,"\n[sc_projection_along_variable_ofl_ctrl_timeout_ctrl]: projection failed, return sc_empty(base_saved)\n");
      }
      sc_rm(*psc);
      *psc = sc_empty(base_saved);
      return;
    }
  }
  TRY {

    sc_projection_along_variable_ofl_ctrl(psc, v, ofl_ctrl);

    UNCATCH(overflow_error);
  }//of TRY

#ifdef FILTERING
  if (PROJECTION_timeout) {
    char * directory_name = "projection_timeout_filtering_SC_OUT";
    sc_default_dump_to_files(sc,projection_sc_counter,directory_name);
  }
#endif
  sc_rm(sc);
  base_rm(base_saved);
}
