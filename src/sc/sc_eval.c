 /* package sc */

#include <string.h>
#include <stdio.h>
#include <values.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* boolean sc_value_for_variable(Psysteme ps, Variable var, Value *pval):
 * examine les egalites du systeme ps pour recherche une egalite de la forme:
 * 
 *    coeff*var - cste == 0
 * 
 * Si une telle equation est trouvee et si elle est faisable,
 * sc_value_for_variable calcule la valeur de var pour ce systeme et
 * la retourne par pval. Sinon, la procedure aborte par sc_error.
 * 
 * Notes:
 *  - le vecteur representant l'equation est suppose correct: une variable 
 *    dont le coefficient est nul ne doit pas etre materialisee
 *  - ce n'est pas une fonction sur les systemes mais sur les listes d'egalites
 */
boolean sc_value_of_variable(ps, var, pval)
Psysteme ps;
Variable var;
Value *pval;
{
    Pcontrainte pc;

    if (SC_UNDEFINED_P(ps))
	return(FALSE);

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
		    return(TRUE);
		}
	    }
	    else if (pv->succ->succ == NULL)
	    {
		if ((pv->var == var) && (pv->succ->var == TCST))
		{
		    /* equation de la forme: k*var - c == 0 */
		    if(pv->succ->val%pv->val==0)
		    {
			*pval = -pv->succ->val/pv->val;
			return(TRUE);
		    }
		    else
		    {
			/* sc_error("sc_value_of_variable","systeme infaisable"); */
			return(FALSE);
		    }
		}
		else
		{
		    if ((pv->var == TCST) && (pv->succ->var == var))
		    {
			/* equation de la forme: c - k*var == 0 */
			if(pv->val%pv->succ->val==0) {
			    *pval = - pv->val/pv->succ->val;
			    return(TRUE);
			}
			else
			{
			    /* sc_error("sc_value_of_variable",
				     "system is not feasible"); */
			    return(FALSE);
			}
		    }
		}
	    }
	}
    }
    
    return(FALSE);
}

/* void sc_minmax_of_variable(Psysteme ps, Variable var, Value *pmin, *pmax):
 * examine un systeme pour trouver le minimum et le maximum d'une variable
 * apparaissant dans ce systeme par projection a la Fourier-Motzkin.
 * la procedure retourne la valeur FALSE si le systeme est infaisable et
 * TRUE sinon
 *
 * le systeme ps est detruit.
 * 
 */
boolean sc_minmax_of_variable(ps, var, pmin, pmax)
Psysteme ps;
Variable var;
Value *pmin, *pmax;
{
    Value val;
    Pcontrainte pc;
    Pbase b;

    *pmax =  MAXINT;
    *pmin = -MAXINT;

    if (sc_value_of_variable(ps, var, &val) == TRUE) {
	*pmin = val;
	*pmax = val;
	return TRUE;
    }

    /* projection sur toutes les variables sauf var */
    for (b = ps->base; !VECTEUR_NUL_P(b); b = b->succ) {
	Variable v = vecteur_var(b);
	if (v != var) {
	    if (SC_EMPTY_P(ps = sc_projection(ps, v))) {
		return FALSE;
	    }
	    if (SC_EMPTY_P(ps = sc_normalize(ps))) {
		return FALSE;
	    }
	}
    }

    if (sc_value_of_variable(ps, var, &val) == TRUE) {
	*pmin = val;
	*pmax = val;
	return TRUE;
    }

    for (pc = ps->inegalites; pc != NULL; pc = pc->succ) {
	int cv = vect_coeff(var, pc->vecteur);
	int cc = - vect_coeff(TCST, pc->vecteur);

	if (cv > 0) {
	    /* cette contrainte nous donne une borne max */
	    int bs = DIVIDE(cc,cv);
	    if (bs < *pmax) 
		*pmax = bs;
	}
	else if (cv < 0) {
	    /* cette contrainte nous donne une borne min */
	    int bi = DIVIDE(cc,cv);
	    if (bi > *pmin) 
		*pmin = bi;
	}
    }
    if(*pmax < *pmin)
	return FALSE;

    sc_rm(ps);

    return TRUE;
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

    for (pv = b; !VECTEUR_NUL_P(pv); pv = pv->succ) {
	Variable var1 = vecteur_var(pv);
	Psysteme sc = sc_dup(ps1);
	int min,max;
	Pvecteur pv2;
	Pcontrainte pc;
	boolean faisable =  sc_minmax_of_variable(sc,var1, &min, &max);

	if (faisable ) {
	    if (min != -MAXINT) {
		pv2 = vect_new(var1,-1);
		pv2 = vect_add(pv2,vect_new(TCST,min));
		pc = contrainte_make(pv2);
		sc_add_ineg(ps2,pc);
	    }
	    if (max != MAXINT) {
		pv2 = vect_new(var1,1);
		pv2 = vect_add(pv2,vect_new(TCST,-max));
		pc = contrainte_make(pv2);
		sc_add_ineg(ps2,pc);
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

