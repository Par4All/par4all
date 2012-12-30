/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

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

/* Package SC : sc_integer_projection.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Projection of a system of constraints (equalities and inequalities)
 * along one or several variables. The variables are eliminated only
 * if the projection is exact, i.e. introduces no integer points
 * that do not belong to the projection of the initial integer points.
 *
 * Arguments of these functions :
 *
 * - ofl_ctrl is the way overflow errors are handled
 *     ofl_ctrl == NO_OFL_CTRL
 *               -> overflow errors are not handled
 *     ofl_ctrl == OFL_CTRL
 *               -> overflow errors are handled in the called function
 *               (not often available, check first)
 *     ofl_ctrl == FWD_OFL_CTRL
 *               -> overflow errors must be handled by the calling function
 *
 * Authors :
 *
 * Remi Triolet, Malik Imadache, Francois Irigoin, Yi-Qing Yang,
 * Corinne Ancourt
 *
 * Last modified : Be'atrice Creusillet, 16/12/94
 */


#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"


#define MALLOC(s,t,f) malloc((unsigned)(s))
#define FREE(s,t,f) free((char *)(s))

struct prtri_struct {
	Pcontrainte pos,neg,cnul;
};
/* This function sorts all the constraints of inegs into three systems.
 * Each system contains respectively constraints where the coefficient
 * of the variable v is positive, negative, or null
*/

static void constraint_sort(inegs,v,prtri,nb_pos,nb_neg)
Pcontrainte inegs;
Variable v;
struct prtri_struct *prtri;
int *nb_pos,*nb_neg;
{
    Pcontrainte ineg=NULL;
    Pcontrainte ineg1=NULL;
    Value c;

    prtri->pos = NULL;
    prtri->neg = NULL;
    prtri->cnul = NULL;
    if (inegs!=NULL) {
	for(ineg=inegs,ineg1=ineg->succ;ineg!=NULL;) {
	    c = vect_coeff(v,ineg->vecteur);
	    if (value_pos_p(c)) {
		ineg->succ = prtri->pos;
		prtri->pos = ineg;
		*nb_pos +=1;
	    }
	    else {
		if (value_neg_p(c)) {
		    ineg->succ = prtri->neg;
		    prtri->neg = ineg;
		    *nb_neg+=1;
		}
		else {
		    ineg->succ = prtri->cnul;
		    prtri->cnul = ineg;
		}
	    }
	    ineg=ineg1;
	    if (ineg1!=NULL) ineg1=ineg1->succ;
	}
    }
}

/* dans sc, elimine la Variable v des inegalites par "projections entieres".
 * Si la projection "Fourier-Motzkin" d'une variable n'est pas equivalente
 * a la projection entiere alors des contraintes sur la variable sont
 * conservees. Ces contraintes sont  celles qui rendent la condition
 * suffisante non satisfaite.
 */

bool
sc_integer_fourier_motzkin_variable_elimination(
    Psysteme sci,
    Psysteme sc,
    Variable v)
{
    struct {
	Pcontrainte pos,neg,cnul;
    } rtri;
    Pcontrainte posit=NULL;
    Pcontrainte negat=NULL;
    Pcontrainte ineg=NULL;
    Pcontrainte inegs = NULL;
    Pcontrainte posit1 = NULL;
    Psysteme scd = SC_UNDEFINED;
    bool non_equivalent_projections = false;
    int nb_pos=0;
    int nb_neg = 0;
    int i1,dim = sc->nb_ineq;
    bool *ineg_stay; /* [dim+1] */

    ineg_stay = (bool*) malloc(sizeof(bool)*(dim+1));

    for (i1 =1; i1<=dim; ineg_stay[i1]=true, i1++);
    if (!SC_UNDEFINED_P(sc)) {
	inegs=sc->inegalites;
	scd = sc_dup(sc);
    }
    else {
	inegs = NULL;
	free(ineg_stay);
	return true;
    }

    constraint_sort(inegs,v,&rtri,&nb_pos,&nb_neg);


    /* pour chaque inegalite ou v a un coeff. positif et chaque
       inegalite ou v a un coefficient negatif, faire une combinaison       */

    for( posit=rtri.pos; posit!=NULL; ) {
	bool integer_comb_p = true;
	non_equivalent_projections = false;
	for( negat=rtri.neg, i1=1; negat!=NULL; i1++) {

	    ineg = sc_integer_inequalities_combination_ofl_ctrl
		(sci, posit, negat, v, &integer_comb_p, OFL_CTRL);
	    ineg_stay[i1] = ineg_stay[i1] && integer_comb_p;
	    if(!integer_comb_p) non_equivalent_projections = true;


	    if (contrainte_constante_p(ineg)) {
		if (contrainte_verifiee(ineg,false)) {
		    vect_rm(ineg->vecteur);
		    /* combinaison => 1 termcst >= 0 */
		    /* inutile de la garder          */
		}
		else {
		    contraintes_free(rtri.pos);
		    rtri.pos=NULL;
		    contraintes_free(rtri.neg);
		    rtri.neg=NULL;
		    contraintes_free(rtri.cnul);
		    rtri.cnul = NULL;
		    posit = NULL;
		    negat = NULL;
		    return false;
		    /* systeme non faisable */
		}
	    }
	    else {
		if(!integer_comb_p) {
		    vect_rm(ineg->vecteur);
		    ineg->vecteur = vect_dup(negat->vecteur);
		}

		ineg->succ = rtri.cnul;
		rtri.cnul = ineg;
		/* combinaison simple reussie => 1ineg en + */
	    }
	    if (negat) negat = negat->succ;
	}

	if (non_equivalent_projections) {
	    posit1 = contrainte_dup(posit);
	    posit1->succ=rtri.cnul;
	    rtri.cnul = posit1;
	}

	if (posit) posit = posit->succ;

    }
    for( negat=rtri.neg, i1=1; negat!=NULL; negat=negat->succ, i1++) {
	if (!ineg_stay[i1]) {
	    ineg = contrainte_dup(negat);
	    ineg->succ = rtri.cnul;
	    rtri.cnul = ineg;
	}
    }
    contraintes_free(rtri.pos);
    contraintes_free(rtri.neg);
    rtri.pos = NULL;
    rtri.neg = NULL;
    /* apres les combinaisons eliminer les elements devenus inutils */
    sc_rm(scd);
    (sc->inegalites) = rtri.cnul;

    free(ineg_stay);
    return true;
}


/* Integer projection of  variable v in the system sc. The system sci
 * corresponds to the initial system on which no projection has been made.
 * Sci is usefull for computing the "suffisant condition".
 * If the "Fourier-Motzkin" projection is not equivalent to the
 * "integer projection" then some constraints on the variable v are kept.
 * Those constraints are those for which the "suffisant condition" is not
 * satisfied.
 */

/* projection d'un sc sur une var v */
Psysteme sc_integer_projection_along_variable(
  Psysteme sci,    // sci est le systeme initial sans les projections
  volatile Psysteme sc,
  Variable v)
{
  Pcontrainte eq;
  Value coeff;
  if (sc) {
    // trouver une egalite ou v a un coeff. minimal en valeur absolue
    if ((eq = contrainte_var_min_coeff(sc->egalites,v,&coeff, false)) != NULL)
    {
	    if(!egalite_normalize(eq))
        return SC_EMPTY;
	    CATCH(overflow_error) {
        sc = sc_elim_var(sc,v);
	    }
	    TRY {
        sc = sc_variable_substitution_with_eq_ofl_ctrl(sc, eq, v, NO_OFL_CTRL);
        UNCATCH(overflow_error);
	    }
    }
    else {
	    // v n'apparait dans aucune egalite, il faut l'eliminer
      // des inegalites   par des combinaisons (fonction combiner)

	    if (!sc_integer_fourier_motzkin_variable_elimination(sci,sc,v) && sc) {
        sc_rm(sc);
        return SC_UNDEFINED;
	    }
	    sc->nb_ineq = nb_elems_list(sc->inegalites);
    }
  }
  return sc;
}



/* this function returns the system resulting of the successive  integer
 * projection of the system sc along all the variables contain in vecteur pv
 */
Psysteme sc_integer_projection_along_variables
  (Psysteme fsc,
   Psysteme sc,
   Pbase index_base,
   Pvecteur pv,
   int tab_info[][4],
   int dim,
   int n)
{
  Pvecteur pv1;

  for (pv1 = pv;!VECTEUR_NUL_P(pv1); pv1=pv1->succ)
  {
  	sc = sc_integer_projection_along_variable(fsc,sc,vecteur_var(pv1));
    sc = sc_normalize(sc);
    sc_integer_projection_information(sc,index_base,tab_info,dim,n);
    sc = build_integer_sc_nredund(sc,index_base,tab_info,dim,dim,n);
  }
  return sc;
}

