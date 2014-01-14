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

/* package sc
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* bool sc_value_for_variable(Psysteme ps, Variable var, Value *pval):
 * examine les egalites du systeme ps pour recherche une egalite de la forme:
 * 
 *    coeff*var - cste == 0
 * 
 * Si une telle equation est trouvee et si elle est faisable,
 * sc_value_for_variable calcule la valeur de var pour ce systeme et
 * la retourne par pval. Sinon, la procedure renvoie la valeur false.
 * 
 * Notes:
 *  - le vecteur representant l'equation est suppose correct: une variable 
 *    dont le coefficient est nul ne doit pas etre materialisee
 *  - ce n'est pas une fonction sur les systemes mais sur les listes d'egalites
 *  - la valeur false peut signifer que le systeme d'egalites de ps n'est pas
 *    faisable ou qu'il n'y a pas d'equation satisfaisante; l'arret par 
 *    sc_error() en cas de non-faisabilite n'etait pas satisfaisant pour les 
 *    procedures appelantes
 */
bool sc_value_of_variable(ps, var, pval)
Psysteme ps;
Variable var;
Value *pval;
{
    Pcontrainte pc;

    if (SC_UNDEFINED_P(ps))
	return(false);

    for (pc = ps->egalites; pc != NULL; pc = pc->succ) 
    {
	Pvecteur pv = pc->vecteur;
	
	if (! VECTEUR_NUL_P(pv)) 
	{
	    if (pv->succ == NULL)
	    {
		if (pv->var == var)
		{
		    /* equation de la forme: k*var == 0 */
		    *pval = 0;
		    return(true);
		}
	    }
	    else if (pv->succ->succ == NULL)
	    {
		if ((pv->var == var) && (pv->succ->var == TCST))
		{
		    Value vs = val_of(pv->succ), v = val_of(pv);
		    /* equation de la forme: k*var - c == 0 */
		    if(value_zero_p(value_mod(vs,v)))
		    {
			*pval = value_uminus(value_div(vs,v));
			return(true);
		    }
		    else
		    {
			return(false);
		    }
		}
		else
		{
		    if ((pv->var == TCST) && (pv->succ->var == var))
		    {
		    Value vs = val_of(pv->succ), v = val_of(pv);
			/* equation de la forme: c - k*var == 0 */
			if(value_zero_p(value_mod(v,vs))) {
			    *pval = value_uminus(value_div(v,vs));
			    return(true);
			}
			else
			{
			    return(false);
			}
		    }
		}
	    }
	}
    }
    
    return(false);
}

/* void sc_minmax_of_variable(Psysteme ps, Variable var, Value *pmin, *pmax):
 * examine un systeme pour trouver le minimum et le maximum d'une variable
 * apparaissant dans ce systeme par projection a la Fourier-Motzkin.
 * la procedure retourne la valeur false si le systeme est infaisable et
 * true sinon
 *
 * le systeme ps est detruit.
 * 
 */
bool sc_minmax_of_variable(ps, var, pmin, pmax)
Psysteme ps;
Variable var;
Value *pmin, *pmax;
{
    Value val;
    Pcontrainte pc;
    Pbase b;

    assert(var!=TCST);

    *pmax = VALUE_MAX;
    *pmin = VALUE_MIN;

    if (sc_value_of_variable(ps, var, &val) == true) {
	*pmin = val;
	*pmax = val;
	return true;
    }

    /* projection sur toutes les variables sauf var */
    for (b = ps->base; !VECTEUR_NUL_P(b); b = b->succ) {
	Variable v = vecteur_var(b);
	if (v != var) {
	    if (SC_EMPTY_P(ps = sc_projection(ps, v))) {
		return false;
	    }
	    if (SC_EMPTY_P(ps = sc_normalize(ps))) {
		return false;
	    }
	}
    }

    if (sc_value_of_variable(ps, var, &val) == true) {
	*pmin = val;
	*pmax = val;
	return true;
    }

    for (pc = ps->inegalites; pc != NULL; pc = pc->succ) {
	Value cv = vect_coeff(var, pc->vecteur);
	Value cc = value_uminus(vect_coeff(TCST, pc->vecteur));

	if (value_pos_p(cv)) {
	    /* cette contrainte nous donne une borne max */
	    Value bs = value_pdiv(cc,cv);
	    if (value_lt(bs,*pmax))
		*pmax = bs;
	}
	else if (value_neg_p(cv)) {
	    /* cette contrainte nous donne une borne min */
	    Value bi = value_pdiv(cc,cv);
	    if (value_gt(bi,*pmin))
		*pmin = bi;
	}
    }
    if (value_lt(*pmax,*pmin)) 
	return false;

    sc_rm(ps);

    return true;
}

/* This function uses sc_minmax_of_variable to compute the min and max 
 * of each variable belonging to base b.
 * If constraints on the variable are found, they are added to the systeme 
 * ps2.
 */
void sc_minmax_of_variables(ps1,ps2,b)
Psysteme ps1,ps2;
Pbase b; 
{
    Pvecteur pv;
    assert(ps2);

    for (pv = b; !VECTEUR_NUL_P(pv); pv = pv->succ) {
	Variable var1 = vecteur_var(pv);
	Psysteme sc = sc_dup(ps1);
	Value min, max;
	bool faisable = sc_minmax_of_variable(sc, var1, &min, &max);
	Pcontrainte pc;

	if (faisable) {
	    if (value_ne(min,VALUE_MIN)) 
	    {
		pc = contrainte_make(vect_make(
		    VECTEUR_NUL, var1, VALUE_MONE, TCST, min));
		sc_add_ineg(ps2, pc);
	    }
	    if (value_ne(max,VALUE_MAX)) 
	    {
		pc = contrainte_make(vect_make(
		    VECTEUR_NUL, var1, VALUE_ONE,TCST, value_uminus(max)));
		sc_add_ineg(ps2, pc);
	    }
	}
    }
}

/* void sc_force_variable_to_zero(Psysteme ps, Variable var): force la
 * variable var a prendre la valeur 0, i.e. intersection du polyedre defini
 * par ps avec l'hyperplan var=0; on suppose que cette intersection est
 * non vide (la faisabilite n'est pas testee).
 * 
 * le systeme initial ps est transforme par effet de bord.
 * 
 */
void sc_force_variable_to_zero(ps, var)
Psysteme ps;
Variable var;
{
    Pcontrainte pc, pcs, pcp;

    pc = ps->egalites;
    pcp = (Pcontrainte) NULL;
    while (pc != (Pcontrainte) NULL) {
	vect_erase_var(&(pc->vecteur), var);
	pcs = pc->succ;
	if (contrainte_constante_p(pc)) {
	    if (pcp == (Pcontrainte) NULL) {
		ps->egalites = pcs;
	    } 
	    else {
		pcp->succ = pcs;
	    }
	    ps->nb_eq -= 1;
	    contrainte_free(pc);
	} 
	else {
	    pcp = pc;
	}
	pc = pcs;
    }

    pc = ps->inegalites;
    pcp = (Pcontrainte) NULL;
    while (pc != (Pcontrainte) NULL) {
	vect_erase_var(&(pc->vecteur), var);
	pcs = pc->succ;
	if (contrainte_constante_p(pc)) {
	    if (pcp == (Pcontrainte) NULL) {
		ps->inegalites = pcs;
	    } 
	    else {
		pcp->succ = pcs;
	    }
	    ps->nb_ineq -= 1;
	    contrainte_free(pc);
	} 
	else {
	    pcp = pc;
	}
	pc = pcs;
    }
}


/* void sc_minmax_of_variable2(Psysteme ps, Variable var, Value *pmin, *pmax):
 * examine un systeme pour trouver le minimum et le maximum d'une variable
 * apparaissant dans ce systeme par projection a la Fourier-Motzkin.
 * la procedure retourne la valeur false si le systeme est infaisable et
 * true sinon
 *
 * Le systeme ps est detruit et desalloue..
 *
 * L'algorithme utilise est different de celui de sc_minmax_of_variable(). 
 * Les equations sont tout d'abord utilisees pour decomposer le systeme
 * en un hyper-espace caracterise par un sous-ensemble des equations
 * (par exemple les equations ayant moins de 2 variables) et un polyedre
 * de cet hyperespace caracterise par les inegalites de ps et les egalites
 * qui n'ont pas ete utilisees lors de la projection.
 *
 * Note:
 *  - comme on ne teste pas la faisabilite en entiers, il se peut que
 *    cette fonction renvoie true et que les deux valeurs min et max
 *    n'existent pas vraiment; il faut donc etre sur de la maniere dont
 *    on utilise cette fonction; dans le cas de l'evaluation de la valeur
 *    d'une variable en un point d'un programme, cela n'a pas d'importance
 *    puisque la non-faisabilite (connue ou non) implique que le code est
 *    mort. La substitution d'une variable par une valeur n'a donc pas
 *    d'importance.
 *  - on pourrait verifier que var appartient a la base de ps; comme il est
 *    trivial mais pas forcement justifie, ce test est laisse a la charge
 *    de l'appelant.
 */
bool 
sc_minmax_of_variable2(Psysteme ps, Variable var, Value *pmin, Value *pmax)
{
/* Maximum number of variables in an equation used for the projection */
#define level (2)

#define if_debug_sc_minmax_of_variable2 if(false)

    bool feasible_p = true;

    if_debug_sc_minmax_of_variable2 {
	fprintf(stderr, "[sc_minmax_of_variable2]: Begin\n");
    }

    if(SC_UNDEFINED_P(ps)) {
	if_debug_sc_minmax_of_variable2 {
	    fprintf(stderr,
		    "[sc_minmax_of_variable2]: Empty system as input\n");
	}
	feasible_p = false;
    }
    else if(SC_EMPTY_P(ps = sc_normalize(ps))) {
	if_debug_sc_minmax_of_variable2 {
	    fprintf(stderr,
		    "[sc_minmax_of_variable2]:"
		    " Non-feasibility detected by first call to sc_normalize\n");
	}
	feasible_p = false;
    }
    else { /* no obvious non-feasibility */
	Pcontrainte eq = CONTRAINTE_UNDEFINED;
	Pcontrainte ineq = CONTRAINTE_UNDEFINED;
	Pcontrainte next_eq = CONTRAINTE_UNDEFINED;
	volatile int nvar;
	int neq = sc_nbre_egalites(ps);

	if_debug_sc_minmax_of_variable2 {
	    fprintf(stderr,
		    "[sc_minmax_of_variable2]: After call to sc_normalize\n");
	    fprintf(stderr, "[sc_minmax_of_variable2]: Input system %p\n",
		    ps);
	    sc_dump(ps);
	}

	/*
	 * Solve the equalities (if any)
	 *
	 * Start with equalities with the smallest number of variables
	 * and stop when all equalities have been used and or when
	 * all equalities left have too many variables.
	 *
	 * Equalities with no variable are checked although nvar starts
	 * with 1.
	 */
	for(nvar = 1;
	    feasible_p && neq > 0 && nvar <= level /* && sc_nbre_egalites(ps) != 0 */;
	    nvar++) {
	    for(eq = sc_egalites(ps); 
		feasible_p && !CONTRAINTE_UNDEFINED_P(eq);
		eq = next_eq) {

		/* eq might suffer in the substitution... */
		next_eq = contrainte_succ(eq);

		if(egalite_normalize(eq)) {
		    if(CONTRAINTE_NULLE_P(eq)) {
			/* eq is redundant */
			;
		    }
		    else {
			/* Equalities change because of substitutions.
			 * Their dimensions may go under the present
			 * required dimension, nvar. Hence the non-equality
			 * test between d and nvar.
			 */
			int d = vect_dimension(contrainte_vecteur(eq));

			if(d<=nvar) {
			    Pcontrainte def = CONTRAINTE_UNDEFINED;
			    /* Automatic variables read in a CATCH block need
			     * to be declared volatile as
			     * specified by the documentation*/
			    Variable volatile v = TCST;
			    Variable v1 = TCST;
			    Variable v2 = TCST;
			    Variable nv = TCST;
			    Pvecteur pv = contrainte_vecteur(eq);
			    bool value_found_p = false;

			    /* Does eq define var's value (and min==max)?
			     * Let's keep track of this but go on to check
			     * feasibility
			     */
			    if(d==1) {
				if(!VECTEUR_UNDEFINED_P(pv->succ)) {
				    /* remember eq was normalized... */
				    if ((pv->var == var) &&
					(pv->succ->var == TCST)) {
					*pmin = value_uminus
					   (value_div(val_of(vecteur_succ(pv)),
						      val_of(pv)));
					*pmax = *pmin;
					value_found_p = true;
				    }
				    else if ((pv->succ->var == var) &&
					     (pv->var == TCST)) {
					*pmin = value_uminus
					    (value_div(val_of(pv),
					        val_of(vecteur_succ(pv))));
					*pmax = *pmin;
					value_found_p = true;
				    }
				    else {
					value_found_p = false;
				    }
				}
				else {
				    if (pv->var == var) {
					*pmin = VALUE_ZERO;
					*pmax = *pmin;
					value_found_p = true;
				    }
				    else {
					value_found_p = false;
				    }
				}
			    }
			    /* Although the value might have been found, go on to
			     * check feasibility. This could be optional because
			     * it should not matter for any safe use of this
			     * function. See Note in function's header.
			     */
			    if(value_found_p) {
				/* do not touch ps any more */
				/* sc_rm(new_ps); */
				return true;
			    }

			    /* keep eq */
			    /* In fact, new_ps is not needed. we simply modify ps */
			    /*
			    new_eq = contrainte_dup(eq);
			    sc_add_egalite(new_ps, new_eq);
			    */

			    /* use eq to eliminate a variable except if var value
			     * is still unknown. We then do need this equation in ps.
			     */

			    /* Let's use a variable with coefficient 1 if
			     * possible.
			     */
			    v1 = TCST;
			    v2 = TCST;
			    for( pv = contrainte_vecteur(eq);
				!VECTEUR_NUL_P(pv);
				pv = vecteur_succ(pv)) {
				nv = vecteur_var(pv);
				/* This test is not fully used in the curent
				 * version since value_found_p==false here.
				 * That would change if feasibility had to
				 * be checked better.
				 */
				if(nv!=TCST && (nv!=var||value_found_p)) {
				    v2 = (v2==TCST)? nv : v2;
				    if(value_one_p(vecteur_val(pv))) {
					if(v1==TCST) {
					    v1 = nv;
					    break;
					}
				    }
				}
			    }
			    v = (v1==TCST)? v2 : v1;
			    /* because of the !CONTRAINTE_NULLE_P() test and because
			     * of value_found_p */
			    assert(v!=TCST);
			    assert(v!=var||value_found_p);

			    /* eq itself is going to be modified in ps.
			     * use a copy!
			     */
			    def = contrainte_dup(eq);
			    CATCH(overflow_error) {
				ps=sc_elim_var(ps,v);
			    }
			    TRY {
			    ps = 
			    sc_simple_variable_substitution_with_eq_ofl_ctrl
			    (ps, def, v, NO_OFL_CTRL);
			    contrainte_rm(def);
			    UNCATCH(overflow_error);
			    }
			    /* Print ps after projection.
			     * This really generates a lot of output on real life systems!
			     * But less than the next debug statement...
			     */
			    /*
			       if_debug_sc_minmax_of_variable2 {
			       fprintf(stderr,
			       "Print the two systems at each elimination step:\n");
			       fprintf(stderr, "[sc_minmax_of_variable2]: Input system %x\n",
			       (unsigned int) ps);
			       sc_dump(ps);
			       }
			       */
			}
			else {
			    /* too early to use this equation eq */
			    /* If there any hope to use it in the future?
			     * Yes, if its dimension is no more than nvar+1
			     * because one of its variable might be substituted.
			     * If more variable are substituted, it's dimension
			     * is going to go down and it will be counted later...
			     * Well this is not true, it will be lost:-(
			     */
			    if(d<=nvar+1) {
				neq++;
			    }
			    else {
				/* to be on the safe side till I find a better idea... */
				neq++;
			    }
			}
		    }
		}
		else {
		    /* The system is not feasible. Stop */
		    feasible_p = false;
		    break;
		}

		/* This really generates a lot of output on real life system!
		 * This is useless. Use previous debugging statement */
		/*
		if_debug_sc_minmax_of_variable2 {
		    fprintf(stderr,
			    "Print the two systems at each equation check:\n");
		    fprintf(stderr, "[sc_minmax_of_variable2]: Input system %x\n",
			    (unsigned int) ps);
		    sc_dump(ps);
		}
		*/
	    }

	    if_debug_sc_minmax_of_variable2 {
		fprintf(stderr,
			"Print the two systems at each nvar=%d step:\n", nvar);
		fprintf(stderr, "[sc_minmax_of_variable2]: Input system %p\n",
			ps);
		sc_dump(ps);
	    }
	}

	assert(!feasible_p ||
	       (CONTRAINTE_UNDEFINED_P(eq) && CONTRAINTE_UNDEFINED_P(ineq)));

	feasible_p = feasible_p && !SC_EMPTY_P(ps = sc_normalize(ps));

	if(feasible_p) {
	    /* Try to exploit the inequalities and the equalities left in ps */

	    /* Exploit the reduction in size */
	    base_rm(sc_base(ps));
	    sc_base(ps) = BASE_UNDEFINED;
	    sc_creer_base(ps);

	    if_debug_sc_minmax_of_variable2 {
		fprintf(stderr,
			"Print System ps after projection and normalization:\n");
		fprintf(stderr, "[sc_minmax_of_variable2]: Input system ps %p\n",
			ps);
		sc_dump(ps);
	    }

	    if(base_contains_variable_p(sc_base(ps), var)) {
		feasible_p = sc_minmax_of_variable(ps, var, pmin, pmax);
		ps = SC_UNDEFINED;
	    }
	    else {
		*pmin = VALUE_MIN;
		*pmax = VALUE_MAX;
	    }
	}
    }

    if(!feasible_p) {
	*pmin = VALUE_MIN;
	*pmax = VALUE_MAX;
    }
    else {
	/* I'm afraid of sc_minmax_of_variable() behavior... 
	 * The guard should be useless
	 */
	if(!SC_UNDEFINED_P(ps)) {
	    sc_rm(ps);
	}
    }

    if_debug_sc_minmax_of_variable2 {
	fprintf(stderr,
		"[sc_minmax_of_variable2]: feasible=%d, min=", feasible_p);
	fprint_Value(stderr, *pmin);
	fprintf(stderr, ", max=");
	fprint_Value(stderr, *pmax);
	fprintf(stderr, "\n[sc_minmax_of_variable2]: End\n");
    }

    return feasible_p;
}

/*
 * That's all folks!
 */
