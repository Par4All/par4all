/* Package SC : sc_projection.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
 *
 * Projection of a system of constraints (equalities and inequalities) 
 * along one or several variables. The variables are always eliminated, 
 * even if the projection is not exact, i.e. introduces integer points 
 * that do not belong to the projections of the initial integer points.
 *
 * Arguments of these functions :
 * 
 * - s or sc is the system of constraints.
 * - ofl_ctrl is the way overflow errors are handled
 *     ofl_ctrl == NO_OFL_CTRL
 *               -> overflow errors are not handled
 *     ofl_ctrl == OFL_CTRL
 *               -> overflow errors are handled in the called function
 *                  (not often available, check first!)
 *     ofl_ctrl == FWD_OFL_CTRL
 *               -> overflow errors must be handled by the calling function
 *
 * Comments :
 * 
 *  The base of the Psysteme is not always updated. Check the comments at the 
 *  head of each function for more details. 
 *  In the functions that update the base, there is an assert if the variables
 *  that are eliminated do not belong to the base.
 *
 * Authors :
 *
 * Remi Triolet, Malik Imadache, Francois Irigoin, Yi-Qing Yang, 
 * Corinne Ancourt, Be'atrice Creusillet.
 *
 */

#include <stdio.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

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
 *  - ajout de la normalisation de l'equation eq avant les substitutions;
 *    cette normalisation permet de detecter une equation infaisable; le
 *    probleme a ete mis en evidence par Yi-Qing dans le test de dependance
 *    de PIPS pour T(2*i+4*j) vs T(2*i+2*j+1) donnant 2*j + 2*di + 2*dj + 1 =0;
 *    la projection de j faisait perdre la non-faisabilite; Francois Irigoin,
 *    8 novembre 1991
 *
 */
void sc_projection_along_variable_ofl_ctrl(psc, v, ofl_ctrl) 
Psysteme *psc;
Variable v;
int ofl_ctrl;
{
    Pcontrainte eq;
    Value coeff;
    Pbase base_init = base_dup((*psc)->base);

    /* trouver une egalite ou v a un coeff. minimal en valeur absolu */
    eq = contrainte_var_min_coeff((*psc)->egalites, v, &coeff, FALSE);
    
    if (eq != NULL) {		
	if(!egalite_normalize(eq)) {
	    sc_rm(*psc);
	    *psc = sc_empty(base_init);
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
	boolean fm_int_exact_p;
	/* v n'apparait dans aucune egalite, il faut l'eliminer des inegalites
	   par des combinaisons (fonction combiner) */
	
	fm_int_exact_p = 
	    sc_fourier_motzkin_variable_elimination_ofl_ctrl(*psc, v, FALSE, 
							     FALSE, ofl_ctrl);
	if (fm_int_exact_p == FALSE) {
	    /* detection de la non faisabilite du Psysteme */
	    sc_rm(*psc);
	    *psc = sc_empty(base_init);
	    return;
	}
	(*psc)->nb_ineq = nb_elems_list((*psc)->inegalites);
    }
    base_rm(base_init);
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
Psysteme *psc;
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
    Pbase scbase = base_dup((*psc)->base); 

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


/* void sc_projection_with_test_along_variables_ofl_ctrl(Psysteme *psc, 
 *                                                      Pvecteur pv, 
 *                                                      boolean *is_proj_exact)
 * input    : the systeme to project along the variables of the vector pv.
 *            *is_proj_exact is supposed to be initialized.
 * output   : nothing. 
 * modifies : *is_proj_exact is set to FALSE if the projection is not exact.
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
boolean *is_proj_exact; /* this boolean is supposed to be initialized */
int ofl_ctrl;
{
    Pcontrainte eq;
    Pvecteur current_pv, pve, pvr;
    Variable v;
    Value coeff;
    Psysteme sc = *psc;
    Pbase base_sc = base_dup(sc->base);
    
    pve = vect_dup(pv);

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

		if (sc == NULL) {
		    /* returned by sc_normalize if sc is unfeasible */
		    sc = sc_empty(base_sc);
		    base_sc = NULL;
		    break;
		}
		
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

		if (*is_proj_exact) *is_proj_exact = FALSE;

		sc = sc_variable_substitution_with_eq_ofl_ctrl
		    (sc, eq, v, ofl_ctrl);
		
		
		sc = sc_safe_normalize(sc);
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
	
	current_pv = pve;
	while (current_pv != NULL) {

	    v = current_pv->var;

	    if (! sc_fourier_motzkin_variable_elimination_ofl_ctrl(sc, v, TRUE,
								   is_proj_exact, 
								   ofl_ctrl)) 
	    {
		/* sc_rm(sc); ???????? BC ??????? */
		sc = sc_empty(base_sc);
		base_sc = NULL;
		break;
	    }

	    sc = sc_safe_normalize(sc);
	    if (sc_empty_p(sc))
		break;
	    
	    current_pv = current_pv->succ;
	    

	} /* while */

    } /* if */

    if (base_sc != NULL)
    {
	/* base_sc has not been used to allocate a new system */
	base_rm(base_sc);
	sc->nb_ineq = nb_elems_list(sc->inegalites);
	sc->nb_eq = nb_elems_list(sc->egalites);
    }

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
    Pbase b_dup = base_dup(sc->base);
  
    pr = NULL;
    current_eq = sc->egalites;
    while(current_eq != NULL) {
	
	/* eq must be removed from the list of equalities of sc */
	if(current_eq == eq) 
	    if (pr == NULL){ /* eq is at the head of the list */
		sc->egalites = current_eq = current_eq->succ;
	    }
	    else
		pr->succ = current_eq = current_eq->succ;
	
	else {
	    
	    switch (contrainte_subst_ofl_ctrl(v, eq, current_eq, TRUE, ofl_ctrl)) {
		
	    case 0 : 
		/* the substitution found that the system is not feasible */
		sc_rm(sc);
		sc = sc_empty(b_dup);
		eq = contrainte_free(eq);		
		return(sc); 
		break;

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
	
	switch (contrainte_subst_ofl_ctrl(v, eq, current_eq, FALSE,ofl_ctrl)) {

	case 0 : 
	    /* the substitution found that the system is not feasible */
	    sc_rm(sc);
	    sc = sc_empty(b_dup);
	    eq = contrainte_free(eq);
	    return(sc);
	    break;

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
    base_rm(b_dup);
    return(sc);
}

/* Psysteme sc_simple_variable_substitution_with_eq_ofl_ctrl(Psysteme sc, Pcontrainte def,
 *                                                    Variable v, int ofl_ctrl) 
 * input    : a system of contraints sc, a contraint eq that must belong to sc,
 *            and a variable v that appears in the constraint eq.
 * output   : a Psysteme which is the projection of sc along v using the equation
 *            eq.
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
    /* Assume no aliasing between def and sc! */
    Pcontrainte eq = CONTRAINTE_UNDEFINED;
    Pcontrainte ineq = CONTRAINTE_UNDEFINED;

    assert(!SC_UNDEFINED_P(sc));

    for(eq = sc_egalites(sc); !CONTRAINTE_UNDEFINED_P(eq);
	eq = contrainte_succ(eq)){
	(void) contrainte_subst_ofl_ctrl(v, def, eq, TRUE, ofl_ctrl);
    }

    for(ineq = sc_inegalites(sc); !CONTRAINTE_UNDEFINED_P(ineq);
	ineq = contrainte_succ(ineq)){
	(void) contrainte_subst_ofl_ctrl(v, def, ineq, FALSE, ofl_ctrl);
    }
    return sc;
}


/* boolean sc_fourier_motzkin_variable_elimination_ofl_ctrl(Psysteme sc, 
 *                     Variable v, 
 *                     boolean integer_test_p, boolean *integer_test_res_p, 
 *                     int ofl_ctrl)
 * input    : a systeme of constraints sc, a variable to eliminate using
 *            Fourier-Motzking elimination, a boolean (integer_test_p)  
 *            TRUE when the test of sufficient conditions for an exact
 *            projection must be performed, a pointer towards an already
 *            initialized boolean (integer_test_res_p) to store the
 *            logical and of its initial value and of the result of the
 *            previous test, and an integer to indicate how the overflow
 *            error must be handled (see sc-types.h). The case OFL_CTRL
 *            is of no interest here.
 * output   : TRUE if sc is feasible (w.r.t the inequalities), 
 *            FALSE otherwise;
 * modifies : sc, *integer_test_res_p if integer_test_p == TRUE.
 * comment  : eliminates v from sc by combining the inequalities of sc 
 *            (Fourier-Motzkin). If this elimination is not exact, 
 *            *integer_test_res_p is set to FALSE.
 *            *integer_test_res_p must be initialized.
 *            
 */
boolean sc_fourier_motzkin_variable_elimination_ofl_ctrl(sc,v, integer_test_p, 
						integer_test_res_p, ofl_ctrl)
Psysteme sc;
Variable v;
boolean integer_test_p;
boolean *integer_test_res_p;
int ofl_ctrl;
{
    Psysteme sc1 = SC_UNDEFINED;
    Pcontrainte pos, neg, nul;
    Pcontrainte pc, pcp, pcn;
    Value c;
    int nnul;

    if ((pc = sc->inegalites) == NULL) 
	return(TRUE);

    if (integer_test_p) sc1 = sc_dup(sc);

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
		boolean int_comb_p;
		pcnew = 
		    sc_integer_inequalities_combination_ofl_ctrl
			(sc1, pcp, pcn, v, &int_comb_p, ofl_ctrl);
		*integer_test_res_p = *integer_test_res_p && int_comb_p;
	    }								 
	    else
		pcnew = inegalite_comb_ofl_ctrl(pcp, pcn, v, ofl_ctrl);
	    
	    if (contrainte_constante_p(pcnew)) {
		if (contrainte_verifiee(pcnew,FALSE)) {
		    contrainte_free(pcnew);
		}
		else {
		    contraintes_free(pos);
		    contraintes_free(neg);
		    contraintes_free(nul);
		    contraintes_free(pcnew);
		    if (integer_test_p) sc_rm(sc1);
		    return(FALSE);
		}
	    }
	    else {
		pcnew->succ = nul;
		nul = pcnew;
		nnul++;
	    }
	}
    }


    /* apres les combinaisons eliminer les elements devenus inutiles */
    contraintes_free(pos);
    contraintes_free(neg);

    /* mise a jour du systeme */
    sc->inegalites = nul;
    sc->nb_ineq = nnul;
    if (integer_test_p) sc_rm(sc1);

    return TRUE;
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
Psysteme sc_projection_on_variables(sc,index_base,pv)
Psysteme sc;
Pbase index_base;
Pvecteur pv;
{
    Pvecteur pv1,pv2,lvar_proj;
    Psysteme sc1=sc_init_with_sc(sc);
    Psysteme sc2;
    Variable var;

    if (!VECTEUR_NUL_P(pv)) {
	lvar_proj = vect_dup(pv);
	for (pv1 = index_base;!VECTEUR_NUL_P(pv1); pv1=pv1->succ) {
	    sc2 = sc_dup(sc);
	    var = vecteur_var(pv1);
	    vect_erase_var(&lvar_proj,var);
	    for (pv2 = lvar_proj;!VECTEUR_NUL_P(pv2); pv2=pv2->succ) {
		sc_projection_along_variable_ofl_ctrl
		    (&sc2,vecteur_var(pv2), NO_OFL_CTRL);
		sc2 = sc_normalize(sc2);
		if (SC_EMPTY_P(sc2)) {
		    sc2 = sc_empty(base_dup(sc->base));
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
		sc1 = sc_empty(base_dup(sc->base));
		break;
	    }
	}
    }
    return(sc1);
}



/* Pcontrainte sc_integer_inequalities_combination_ofl_ctrl(Psysteme sc,
 *                                            Pcontrainte posit, negat,
 *                                                          Variable v,
 *                                      boolean *integer_combination_p,
 *                                                        int ofl_ctrl)
 * input    : many things !
 * output   : a constraint which is the combination of the constraints
 *            posit and negat by the variable v.
 * modifies : *integer_combination_p.
 * comment  :
 *      *integer_combination_p is set to TRUE, if the Fourier Motzkin
 *      projection is equivalent to the "integer projection". 
 *      else, it is set to FALSE.
 */
Pcontrainte sc_integer_inequalities_combination_ofl_ctrl
  (sc,posit,negat,v, integer_combination_p, ofl_ctrl)	
Psysteme sc;
Pcontrainte posit, negat;
Variable v;
boolean *integer_combination_p;
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
	    /* tmp = cp*cn - cp - cn + 1 // cp and cn are >0! */ 
	    Value tmp = value_mult(cp,cn);
	    tmp = value_minus(value_plus(tmp, VALUE_ONE), value_plus(cp, cn));

	    if (contrainte_constante_p(ineg) &&
		value_le(vect_coeff(TCST, ineg->vecteur), value_uminus(tmp)))
		*integer_combination_p = TRUE;
	    else {
		Pcontrainte ineg_test = contrainte_dup(ineg);
		Psysteme sc_test = sc_dup(sc);
		
		vect_add_elem(&(ineg_test->vecteur), TCST, tmp);
		
		contrainte_reverse(ineg_test);
		sc_add_inegalite(sc_test,ineg_test);
		
		if (!sc_rational_feasibility_ofl_ctrl(sc_test, OFL_CTRL, TRUE))
		    *integer_combination_p = TRUE;
		else 
		   *integer_combination_p = FALSE; 
		sc_rm(sc_test);		
	    }								      
	}
	else {
	    *integer_combination_p = TRUE; 
	}
	
    }
    return(ineg);
}


/* FOR BACKWARD COMPATIBILITY ONLY. DO NOT USE ANYMORE, PLEASE. */

Psysteme sc_projection(sc, v) 
Psysteme sc;
Variable v;
{
    sc_projection_along_variable_ofl_ctrl(&sc,v, NO_OFL_CTRL);
    if (sc_empty_p(sc)) {
	sc_rm(sc);
	sc = SC_EMPTY;
    }
    return(sc);
}

Psysteme sc_projection_ofl(sc, v) 
Psysteme sc;
Variable v;
{
    sc_projection_along_variable_ofl_ctrl(&sc,v, FWD_OFL_CTRL);
    if (sc_empty_p(sc)) {
	sc_rm(sc);
	sc = SC_EMPTY;
    }
    return(sc);
}



Psysteme sc_projection_pure(sc,v) 
Psysteme sc;
Variable v;
{
    sc_and_base_projection_along_variable_ofl_ctrl(&sc,v,NO_OFL_CTRL);
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
boolean *is_proj_exact; /* this boolean is supposed to be initialized */
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

boolean cond_suff_comb_integer_ofl_ctrl(sc,posit,negat, v, ofl_ctrl)
Psysteme sc;
Pcontrainte posit,negat;
Variable v;
int ofl_ctrl;
{
    Pcontrainte ineg = CONTRAINTE_UNDEFINED;
    boolean result;

    ineg = sc_integer_inequalities_combination_ofl_ctrl(sc, posit, negat, v, 
							&result, ofl_ctrl);
    contrainte_rm(ineg);
    return(result);

}



Psysteme sc_projection_optim_along_vecteur(sc,pv)
Psysteme sc;
Pvecteur pv;
{
    Psysteme sc1 = sc_dup(sc);
    CATCH(overflow_error) {
	/* sc_rm(sc1); */
	sc1=NULL;
	return sc;
    }
    TRY {
	boolean exact = TRUE;
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
    return contrainte_var_min_coeff(contraintes, v, &coeff, TRUE);
}

boolean sc_expensive_projection_p(sc,v)
Psysteme sc;
Variable v;
{
    boolean expensive = FALSE;
    /* if Variable v is constrained by  equalities, the system size 
       is kept (-1) after projection */
    if (!var_in_lcontrainte_p(sc->egalites,v) 
	 &&  sc->nb_ineq > NB_CONSTRAINTS_MAX_FOR_FM) {
	Pcontrainte ineg;
	int nb_pvar, nb_nvar =0;
	for (ineg=sc->inegalites;!CONTRAINTE_UNDEFINED_P(ineg);
	     ineg=ineg->succ) {
	    int coeff= vect_coeff(v,ineg->vecteur);
	    if (coeff >0)  nb_pvar ++;
	    else if (coeff<0) nb_nvar ++;
	}
       if (nb_pvar*nb_nvar>200) 
	   expensive=TRUE; 
    }
    return (expensive);


}
