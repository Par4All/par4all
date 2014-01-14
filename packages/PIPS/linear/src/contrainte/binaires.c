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

 /* package contrainte - operations binaires */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"

int contrainte_subst_ofl(v,def,c,eq_p)
Variable v;
Pcontrainte def,c;
bool eq_p;
{
    return( contrainte_subst_ofl_ctrl(v,def,c,eq_p, FWD_OFL_CTRL)); 
}

int contrainte_subst(v,def,c,eq_p)
Variable v;
Pcontrainte def,c;
bool eq_p;
{
    return( contrainte_subst_ofl_ctrl(v,def,c,eq_p, NO_OFL_CTRL)); 

}



Pcontrainte inegalite_comb(posit,negat,v)	
Pcontrainte posit, negat;
Variable v;
{
    return( inegalite_comb_ofl_ctrl(posit,negat,v, NO_OFL_CTRL));
}

Pcontrainte inegalite_comb_ofl(posit,negat,v)	
Pcontrainte posit, negat;
Variable v;
{
    return( inegalite_comb_ofl_ctrl(posit,negat,v, FWD_OFL_CTRL));

}


/* int contrainte_subst_ofl_ctrl(Variable v, Pcontrainte def, Pcontrainte c  
 * Boolean eq_p, int ofl_ctrl):
 * elimination d'une variable v entre une equation def et une contrainte,
 * egalite ou inegalite, c. La contrainte c est modifiee en substituant
 * v par sa valeur impliquee par def.
 *
 * La contrainte c est interpretee comme une inegalite et la valeur retournee
 * vaut:
 *  -1 si la contrainte c est trivialement verifiee et peut etre eliminee
 *   0 si la contrainte c est trivialement impossible
 *   1 sinon (tout va bien)
 * Si la contrainte c passee en argument etait trivialement impossible
 * ou trivialement verifiee, aucune subsitution n'a lieu et la valeur 1
 * est retournee.
 *
 * La substitution d'une variable dans une inegalite peut aussi introduire
 * une non-faisabilite testable par calcul de PGCD, mais cela n'est pas
 * fait puisqu'on ne sait pas si c est une egalite ou une inegalite. Le
 * traitement du terme constant n'est pas decidable.
 *
 * Note: il faudrait separer le probleme de la combinaison lineaire
 * a coefficients positifs de celui de la faisabilite et de la trivialite
 *
 * Le controle de l'overflow est effectue et traite par le retour 
 * du contexte correspondant au dernier CATCH(overflow_error) effectue.
 */
int contrainte_subst_ofl_ctrl(v,def,c,eq_p, ofl_ctrl)
Variable v;
Pcontrainte def,c;
bool eq_p;
int ofl_ctrl;
{
    Pvecteur save_c;

    /* cv_def = coeff de v dans def */
    Value cv_def = vect_coeff(v,def->vecteur);
    /* cv_c = coeff de v dans c */
    Value cv_c = vect_coeff(v,c->vecteur);

    /* il faut que cv_def soit non nul pour que la variable v puisse etre
       eliminee */
    assert(value_notzero_p(cv_def));

    /* il n'y a rien a faire si la variable v n'apparait pas dans la
       contrainte c */
    /* substitution inutile: variable v absente */
    if (value_zero_p(cv_c)) return 1;

    /* on garde trace de la valeur de c avant substitution pour pouvoir
       la desallouer apres le calcul de la nouvelle */
    save_c = c->vecteur;
    /* on ne fait pas de distinction entre egalites et inegalites, mais
       on prend soin de toujours multiplier la contrainte, inegalite
       potentielle, par un coefficient positif */
    if (value_neg_p(cv_def)) {
	c->vecteur = vect_cl2_ofl_ctrl(value_uminus(cv_def),
				       c->vecteur,cv_c,
				       def->vecteur,
				       ofl_ctrl);
    }
    else {
	c->vecteur = vect_cl2_ofl_ctrl(cv_def,c->vecteur,
				       value_uminus(cv_c),
				  def->vecteur, ofl_ctrl);
    }
    vect_rm(save_c);

    /* reste malikien: cette partie ne peut pas etre faite sans savoir
       si on est en train de traiter une egalite ou une inegalite */
    if(contrainte_constante_p(c)) {
	if(contrainte_verifiee(c,eq_p)) 
	    /* => eliminer cette c inutile */
	    return(-1);	
	else 
	    /* => systeme non faisable      */
	    return(false);
    }
    return(true);
}

/* Pcontrainte inegalite_comb_ofl_ctrl(Pcontrainte posit, Pcontrainte negat, 
 *                            Variable v, int ofl_ctrl):
 * combinaison lineaire positive des deux inegalites posit et negat
 * eliminant la variable v.
 * 
 * Une nouvelle contrainte est allouee et renvoyee.
 *
 * Si le coefficient de v dans negat egale -1 ou si le coefficient de
 * v dans posit egale 1, la nouvelle contrainte est equivalente en
 * nombres entiers avec posit et negat.
 *
 * Modifications:
 *  - use gcd to reduce the combination coefficients in hope to reduce
 *    integer overflow risk (Francois Irigoin, 17 December 1991)
 *
 * Le controle de l'overflow est effectue et traite par le retour 
 * du contexte correspondant au dernier CATCH(overflow_error) effectue.
 */
Pcontrainte inegalite_comb_ofl_ctrl(posit,negat,v, ofl_ctrl)	
Pcontrainte posit, negat;
Variable v;
int ofl_ctrl;
{
    Value cv_p, cv_n, d;
    Pcontrainte ineg;

    cv_p = vect_coeff(v,posit->vecteur); 
    cv_n = vect_coeff(v,negat->vecteur);

    assert(value_pos_p(cv_p) && value_neg_p(cv_n));

    d = pgcd(cv_p, value_uminus(cv_n));
    if(value_notone_p(d)) {
	cv_p = value_div(cv_p,d); /* pdiv ??? */
	cv_n = value_div(cv_n,d);
    }

    ineg = contrainte_new();

    ineg->vecteur = vect_cl2_ofl_ctrl(cv_p,negat->vecteur,
				      value_uminus(cv_n),
				      posit->vecteur, ofl_ctrl);
    return(ineg);
}



/* Value eq_diff_const(Pcontrainte c1, Pcontrainte c2):
 * calcul de la difference des deux termes constants des deux 
 * equations c1 et c2
 *
 * Notes:
 *  - cette routine fait l'hypothese que CONTRAINTE_UNDEFINED=>CONTRAINTE_NULLE
 *
 * Modifications:
 *  - renvoie d'une valeur non nulle meme si une des contraintes est nulles
 */
Value eq_diff_const(c1,c2)
Pcontrainte c1,c2;
{
    if(c1!=NULL) 
	if(c2!=NULL) {
	    Value b1 = vect_coeff(TCST,c1->vecteur),
	          b2 = vect_coeff(TCST,c2->vecteur);
	    return value_minus(b1,b2);
	}
	else
	    return vect_coeff(TCST,c1->vecteur);
    else
	if(c2!=NULL)
	    return value_uminus(vect_coeff(TCST,c2->vecteur));
	else
	    return VALUE_ZERO;
}

/* Value eq_sum_const(Pcontrainte c1, Pcontrainte c2):
 * calcul de la somme des deux termes constants des deux 
 * contraintes c1 et c2
 *
 * Notes:
 *  - cette routine fait l'hypothese que CONTRAINTE_UNDEFINED=>CONTRAINTE_NULLE
 */
Value eq_sum_const(c1,c2)
Pcontrainte c1,c2;
{
    if(c1!=NULL) 
	if(c2!=NULL) {
	    Value b1 = vect_coeff(TCST,c1->vecteur),
	          b2 = vect_coeff(TCST,c2->vecteur);
	    return value_plus(b1, b2);
	}
	else
	    return vect_coeff(TCST,c1->vecteur);
    else
	if(c2!=NULL)
	    return vect_coeff(TCST,c2->vecteur);
	else
	    return VALUE_ZERO;
}

/* Pcontrainte contrainte_append(c1, c2)
 * Pcontrainte c1, c2;
 *
 * append directly c2 to c1. Neither c1 nor c2 are relevant 
 * when the result is returned.
 */
Pcontrainte contrainte_append(c1, c2)
Pcontrainte c1, c2;
{
    Pcontrainte c;
    
    if (c1==NULL) return(c2);
    if (c2==NULL) return(c1);
    for (c=c1; c->succ!=NULL; c=c->succ);
    c->succ=c2;
    return(c1);
}

/* common (simply equal) contraints are extracted, 
 *   whether equalities or inequalities.
 * returns the common extracted constaints.
 * WARNING: *pc1 and *pc2 are modified.
 */
Pcontrainte 
extract_common_constraints(Pcontrainte * pc1, Pcontrainte * pc2, bool eq)
{
  Pcontrainte c = CONTRAINTE_UNDEFINED, c1, c2, c1p, c2p, nc1, nc2;
  
  for (c1 = *pc1, 
	 c1p = CONTRAINTE_UNDEFINED, 
	 nc1 = c1? c1->succ: CONTRAINTE_UNDEFINED; 
       c1; 
       c1p = (c1==nc1)? c1p: c1, 
	 c1 = nc1, 
	 nc1 = c1? c1->succ: CONTRAINTE_UNDEFINED)
  {
    for (c2 = *pc2,
	   c2p = CONTRAINTE_UNDEFINED,
	   nc2 = c2? c2->succ: CONTRAINTE_UNDEFINED; 
	 c2; 
	 c2p = (c2==nc2)? c2p: c2,
	   c2 = nc2,
	   nc2 = c2? c2->succ: CONTRAINTE_UNDEFINED)
    {
      if ((eq && egalite_equal(c1, c2)) || (!eq && contrainte_equal(c1, c2)))
      {
	/* common! */
	Pcontrainte sc1 = c1->succ, sc2 = c2->succ;

	c1->succ = c, c = c1; /* link to result. */
	c2->succ = NULL, contrainte_free(c2);  /* clean */

	if (c1p) c1p->succ = sc1; else *pc1 = sc1;
	if (c2p) c2p->succ = sc2; else *pc2 = sc2;
	
	/* get to next. */
	c1 = nc1;
	c2 = nc2 = CONTRAINTE_UNDEFINED;
      }
    }
  }

  return c;
}
