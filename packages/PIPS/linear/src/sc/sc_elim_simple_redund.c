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

#include <stdio.h>
#include <string.h>

#include "arithmetique.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* bool sc_elim_simple_redund_with_eq(Psysteme ps, Pcontrainte eg):
 * elimination en place des contraintes d'un systeme ps, qui sont redondantes
 * avec une egalite eg
 *                                                                       
 *  * si une autre egalite a le meme membre gauche (equation sans le terme 
 *     constant) :
 *                                                                       
 *    - si les termes constants sont egaux ==> elimination d'une egalite 
 *    - sinon ==> systeme infaisable                                     
 *                                                                       
 *  * si une inegalite  a le meme membre gauche que l'egalite :               
 *                                                                       
 *     - si l'egalite est compatible avec inegalite                      
 *               ==> elimination de l'inegalite                          
 *     - sinon  ==> systeme infaisable                                   
 *                                                                       
 *
 *  resultat retourne par la fonction :
 *
 *  boolean: false si l'equation a permis de montrer que le systeme
 *                 etait non faisable
 *           true sinon
 *
 *  Les parametres de la fonction :
 *
 * !Psysteme ps	  : systeme
 *  Pcontrainte eg : equation du systeme 
 */
bool sc_elim_simple_redund_with_eq(ps,eg)
Psysteme ps;
Pcontrainte eg;
{
    Pcontrainte eq = NULL;

    if (SC_UNDEFINED_P(ps) || sc_rn_p(ps))
	return(true);

    /* cas des egalites    */
    for (eq = ps->egalites; eq != NULL; eq = eq->succ) {
	if (eq != eg && eq->vecteur != NULL && eq_smg(eq,eg)) {
	    if (eq_diff_const(eq,eg) == 0)
		/* les deux egalites sont redondantes 
		   ==> elimination de eq */
		eq_set_vect_nul(eq);
	    else 
		return(false);
	}
    }

    /* cas des inegalites     */
    for (eq = ps->inegalites; eq != NULL; eq = eq->succ) {
	if (eq_smg(eq,eg) && eq->vecteur!= NULL) {
	    if (value_negz_p(eq_diff_const(eq, eg)))
		eq_set_vect_nul(eq);
	    else
		return(false);
	}
    }
    return(true);
}

/* bool sc_elim_simple_redund_with_ineq(Psysteme ps, Pcontrainte ineg):
 * elimination des contraintes redondantes de ps avec une inegalite ineg
 * (FI: qui doit appartenir a ps; verifier qu'on ne fait pas de comparaisons
 * inutiles; apparemment pas parce qu'on modifie froidement ineg)
 *                                                                       
 *  * si deux inegalites ont le meme membre gauche (contrainte sans le     
 *    terme constant) :                                                  
 *                                                                       
 *      ==> elimination de l'inegalite ayant le terme constant le plus   
 *          grand                                                        
 *                                                                       
 *  * si une egalite  a le meme membre gauche que l'inegalite :               
 *                                                                       
 *     - si l'egalite est compatible avec inegalite                      
 *              ==> elimination de l'inegalite                          
 *     - sinon  ==> systeme infaisable                                   
 *                                                                       
 *  resultat retourne par la fonction :
 *
 *  boolean: false si l'inequation a permis de montrer que le systeme
 *                 etait non faisable
 *           true sinon
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme ps      : systeme
 *  Pcontrainte ineg : inequation du systeme
 */
bool sc_elim_simple_redund_with_ineq(ps,ineg)
Psysteme ps;
Pcontrainte ineg;
{
    Pcontrainte eq=NULL;
    bool result = true;
    Value b = VALUE_ZERO;
 
    if (SC_UNDEFINED_P(ps) || sc_rn_p(ps))
	return(true);

    /* cas des egalites     */
    for (eq = ps->egalites; eq != NULL && ineg->vecteur != NULL; 
	 eq = eq->succ)
	if (eq_smg(eq,ineg)) {
	    b = eq_diff_const(ineg,eq);
	    if (value_negz_p(b))
		/* inegalite redondante avec l'egalite  
		   ==> elimination de inegalite */
		eq_set_vect_nul(ineg);
	    else result = false;
	}

    /* cas des inegalites          */
    for (eq = ps->inegalites;eq !=NULL && ineg->vecteur != NULL; 
	 eq = eq->succ) {
	if (eq != ineg)
	    if (eq_smg(eq,ineg)) {
		b = eq_diff_const(eq,ineg);
		if(value_negz_p(b))
		    eq_set_vect_nul(eq);
		else
		    eq_set_vect_nul(ineg);
	    }
    }
    return (result);
}

/* int sc_check_inequality_redundancy(Pcontrainte ineq, Psysteme ps)
 * Check if an inequality ineq, possibly part of ps, is trivially
 * infeasible (return 2), redundant (return 1), potentially useful
 * (return 0) with respect to inequalities in ps.
 *
 * Neither ps nor ineq are modified. ineq may be one of ps constraints.
 */
int sc_check_inequality_redundancy(Pcontrainte ineq, Psysteme ps)
{
    Pcontrainte c = CONTRAINTE_UNDEFINED;
    int code = 0;

    for(c = sc_inegalites(ps);
	code ==0 && !CONTRAINTE_UNDEFINED_P(c);
	c = contrainte_succ(c)) {

	if(c!=ineq) {
	    Value b;

	    if(eq_smg(c, ineq)) {
		b = eq_diff_const(c, ineq);
		if(value_neg_p(b)) {
		    /* c is redundant with ineq */
		    ;
		}
		else {
		    /* ineq is redundant with c */
		    code = 1;
		}
	    }
	    else if (inequalities_opposite_p(c, ineq)) {
		b = eq_sum_const(c, ineq);
		if(value_negz_p(b)) {
		    /* c and ineq define a non-empty interval */
		    ;
		}
		else {
		    /* ineq and c are incompatible */
		    code = 2;
		}
	    }
	}
    }
    return code;
}


/* void sc_elim_empty_constraints(Psysteme ps, bool process_equalities):
 * elimination des "fausses" contraintes du systeme ps, i.e. les contraintes ne
 * comportant plus de couple (variable,valeur), i.e. les contraintes qui
 * ont ete eliminees par la fonction 'eq_set_vect_nul', i.e. 0 = 0 ou
 * 0 <= 0
 * 
 * resultat retourne par la fonction: le systeme initial ps est modifie.
 * 
 * parametres de la fonction:
 *   !Psysteme ps: systeme lineaire 
 *   bool egalite: true s'il faut traiter la liste des egalites 
 *                    false s'il faut traiter la liste des inegalites
 *
 * Modifications:
 *  - the number of equalities was always decremented, regardless
 *    of the process_equalities parameter; Francois Irigoin, 30 October 1991
 */
void sc_elim_empty_constraints(ps, process_equalities)
Psysteme ps;
bool process_equalities;
{
    Pcontrainte pc, ppc;

    if (!SC_UNDEFINED_P(ps)) {
	if  (process_equalities) {
	    pc = ps->egalites; 
	    ppc = NULL;
	    while (pc != NULL) {
		if  (contrainte_vecteur(pc) == NULL) {
		    Pcontrainte p = pc;
		    if (ppc == NULL) ps->egalites = pc = pc->succ;
		    else 
			ppc->succ = pc = pc->succ;
		    contrainte_free(p);
		    ps->nb_eq--; 
		}
		else {
		    ppc = pc;
		    pc = pc->succ;
		}
	    }
	}
	else {
	    pc = ps->inegalites; 
	    ppc = NULL; 
	    while (pc != NULL) {
		if  (contrainte_vecteur(pc) == NULL) {
		    Pcontrainte p = pc;
		    if (ppc == NULL)  ps->inegalites = pc = pc->succ;
		    else
			ppc->succ = pc = pc->succ;
		    contrainte_free(p);
		    ps->nb_ineq--;
		}
		else {
		    ppc = pc;
		    pc = pc->succ;
		}
	    }
	}
    }
}

/* Psysteme sc_elim_db_constraints(Psysteme ps):
 * elimination des egalites et des inegalites identiques ou inutiles dans
 * le systeme; plus precisemment:
 *
 * Pour les egalites, on elimine une equation si on a un systeme d'egalites 
 * de la forme :
 *
 *   a1/    Ax - b == 0,            ou  b1/        Ax - b == 0,              
 *          Ax - b == 0,                           b - Ax == 0,              
 *
 * ou   c1/    0 == 0
 *
 * Pour les inegalites, on elimine une inequation si on a un systeme 
 * d'inegalites de la forme :
 *  
 *   a2/    Ax - b <= c,             ou   b2/     0 <= const  (avec const >=0)
 *          Ax - b <= c             
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme   	    : Le systeme initial est modifie (si necessaire) et renvoye
 *       	      Si le systeme est non faisable (0 <= const <0 ou
 *                    0 = b), il est desalloue et NULL est
 *                    renvoye.
 *
 * Attention, on ne teste pas les proportionalites: 2*i=2 est different
 * de i = 1. Il faut reduire le systeme par gcd avant d'appeler cette
 * fonction sc_elim_db_constraints()
 *
 * Notes:
 *  - le temps d'execution doit pouvoir etre divise par deux en prenant en
 * compte la symetrie des comparaisons et en modifiant l'initialisation
 * des boucles internes pour les "triangulariser superieurement".
 *  - la representation interne des vecteurs est utilisee pour les tests;
 * il faudrait tester la colinearite au vecteur de base representatif du
 * terme constant
 *
 * - so called triangular version, FC 28/09/94
 */
Psysteme sc_elim_db_constraints(ps)
Psysteme ps;
{
    Pcontrainte
	eq1 = NULL,
	eq2 = NULL;

    if (SC_UNDEFINED_P(ps)) 
	return(NULL);

    for (eq1 = ps->egalites; eq1 != NULL; eq1 = eq1->succ) 
    {
	if ((vect_size(eq1->vecteur) == 1) && 
	    (eq1->vecteur->var == 0) && (eq1->vecteur->val != 0)) 
	{
	    /* b = 0 */
	    sc_rm(ps);
	    return(NULL);
	}

	for (eq2 = eq1->succ; eq2 != NULL;eq2 = eq2->succ)
	    if (egalite_equal(eq1, eq2))
		eq_set_vect_nul(eq2);
    }

    for (eq1 = ps->inegalites; eq1 != NULL;eq1 = eq1->succ) {
      if ((vect_size(eq1->vecteur) == 1) && (eq1->vecteur->var == 0)) {
	  if (value_negz_p(val_of(eq1->vecteur))) {
	    vect_rm(eq1->vecteur);
		eq1->vecteur = NULL;
	  }
	  else {
	    /* 0 <= b < 0 */
	    sc_rm(ps);
	    return(NULL);
	  }
      }
	for (eq2 = eq1->succ;eq2 != NULL;eq2 = eq2->succ)
	    if (contrainte_equal(eq1,eq2))
		eq_set_vect_nul(eq2);
    }

    sc_elim_empty_constraints(ps, true);
    sc_elim_empty_constraints(ps, false);

    return (ps);
}

/* The returned value must be used because they argument is freed when
 * the system is not feasible
 *
 * FI: I added an elimination and check of the colinear constraints.
 */
Psysteme sc_safe_elim_db_constraints(Psysteme ps)
{
  Pcontrainte
	eq1 = NULL,
	eq2 = NULL;
  bool empty_p = false; // The system is empty 

  if (SC_UNDEFINED_P(ps)) 
    return(NULL);

  for (eq1 = ps->egalites; eq1 != NULL; eq1 = eq1->succ) 
    {
      if ((vect_size(eq1->vecteur) == 1) && 
	  (eq1->vecteur->var == 0) && (eq1->vecteur->val != 0)) 
	{
	  /* b = 0 */
	  Pbase base_tmp = ps->base;
	  ps->base = BASE_UNDEFINED;
	  sc_rm(ps);
	  ps =sc_empty(base_tmp);
	  return(ps);
	}

      for (eq2 = eq1->succ; eq2 != NULL;eq2 = eq2->succ)
	if (egalite_equal(eq1, eq2))
	  eq_set_vect_nul(eq2);
    }

  for (eq1 = ps->inegalites; eq1 != NULL && !empty_p; eq1 = eq1->succ) {
    if ((vect_size(eq1->vecteur) == 1) && (eq1->vecteur->var == 0)) {
      if (value_negz_p(val_of(eq1->vecteur))) {
	vect_rm(eq1->vecteur);
	eq1->vecteur = NULL;
      }
      else {
	/* 0 <= b < 0 */
	Pbase base_tmp = ps->base;
	ps->base = BASE_UNDEFINED;
	sc_rm(ps);
	ps =sc_empty(base_tmp);
	return(ps);
      }
    }
    
    for (eq2 = eq1->succ;eq2 != NULL;eq2 = eq2->succ) {
      if (contrainte_equal(eq1,eq2))
	eq_set_vect_nul(eq2);
      else if(false) {
	/* Use rational simplification */
	Value a1, a2;
	if(contrainte_parallele(eq1, eq2, &a1, &a2)) {
	  Pvecteur v1 = contrainte_vecteur(eq1);
	  Pvecteur v2 = contrainte_vecteur(eq2);
	  if(a1<0 && a2<0) {
	    a1 = -a1;
	    a2 = -a2;
	  }
	  if(a1>0 && a2>0) {
	    /* The constraints are opposed. Their constant terms must be compatible */
	    Value k1 = vect_coeff(TCST, v1);
	    Value k2 = vect_coeff(TCST, v2);
	    Value k = value_mult(a2,k1) + value_mult(a1,k2);
	    if(k>0) {
	      /* The constraint system is not feasible */
	      /* ps should be replaced by sc_empty */
	      empty_p = true;
	      break;
	    }
	  }
	  else {
	    /* The signs of a1 and a2 are different */
	    if(a1<0) a1 = -a1;
	    if(a2<0) a2 = -a2;
	    Value k1 = vect_coeff(TCST, v1);
	    Value k2 = vect_coeff(TCST, v2);
	    Value nk1 = value_product(a2,k1);
	    Value nk2 = value_product(a1,k2);
	    if(nk1<nk2) {
	      /* get rid of the first constraint */
	      eq_set_vect_nul(eq1);
	    }
	    else {
	      eq_set_vect_nul(eq2);
	    }
	  }
	}
      }
    }
  }

  if(empty_p) {
    /* FI: it would be safer to keep *ps and to change each field 
     *
     * I thought there was a function to remove the useless stuff from
     * a Psystem and to make it an empty one, but I could not find it.
     */
    Pbase b = sc_base(ps);
    sc_base(ps) = BASE_NULLE;
    sc_rm(ps);
    ps = sc_empty(b);
  }
  else {
    sc_elim_empty_constraints(ps, true);
    sc_elim_empty_constraints(ps, false);
  }

  return (ps);
}

/* Psysteme sc_elim_double_constraints(Psysteme ps):
 * elimination des egalites et des inegalites identiques ou inutiles dans
 * le systeme apres reduction par le gcd; plus precisemment:
 *
 * Pour les egalites, on elimine une equation si on a un systeme d'egalites 
 * de la forme :
 *
 *   a1/    Ax - b == 0,            ou  b1/        Ax - b == 0,              
 *          Ax - b == 0,                           b - Ax == 0,              
 *
 * ou   c1/    0 == 0
 *
 * Si on a A=0 et b!=0, on detecte une non-faisabilite.
 *
 * Si on a Ax - b == 0 et Ax - b' == 0 et b!=b', on detecte une non-faisabilite.
 *
 * Pour les inegalites, on elimine une inequation si on a un systeme 
 * d'inegalites de la forme :
 *  
 *   a2/    Ax - b <= 0,             ou   b2/     0 <= const  (avec const >=0)
 *          Ax - b <= 0
 *
 * Une inegalite peut etre redondante ou incompatible avec une egalite:
 *
 *   a3/    Ax - b == 0,             ou   b3/     b - Ax == 0,
 *          Ax - c <= 0,                          Ax - c <= 0
 *          b - c <= 0                            b - c <= 0
 *
 * on detecte une non-faisabilite si b - c > 0.
 *
 * Une paire d'inegalites est remplacee par une egalite:
 *
 *   a4/    Ax - b <= 0
 *          -Ax + b <=0
 *
 * donne Ax - b == 0
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme   	    : Le systeme initial est modifie (si necessaire) et renvoye
 *       	      Si le systeme est non faisable (0 <= const <0 ou
 *                    0 = b), il est desalloue et NULL est
 *                    renvoye.
 *
 * Notes:
 *  - la representation interne des vecteurs est utilisee pour les tests;
 * il faudrait tester la colinearite au vecteur de base representatif du
 * terme constant
 *
 */
Psysteme sc_elim_double_constraints(ps)
Psysteme ps;
{
  Pcontrainte
    eq1 = NULL,
    ineq1 = NULL,
    eq2 = NULL;

  if (SC_UNDEFINED_P(ps)) 
    return(SC_UNDEFINED);

  /* Normalization by gcd's */

  for (eq1 = ps->egalites; eq1 != NULL; eq1 = eq1->succ) {
    vect_normalize(eq1->vecteur);
  }

  for (ineq1 = ps->inegalites; ineq1 != NULL;ineq1 = ineq1->succ) {
    (void) contrainte_normalize(ineq1, false);
  }

  /* Detection of inconsistant equations: incompatible constant term */

  for (eq1 = ps->egalites; eq1 != NULL; eq1 = eq1->succ) {
    if ((vect_size(eq1->vecteur) == 1) && 
	(eq1->vecteur->var == 0) && (eq1->vecteur->val != 0)) {
      /* b = 0 */
      sc_rm(ps);
      return(SC_EMPTY);
    }

    for (eq2 = eq1->succ; eq2 != NULL;eq2 = eq2->succ) {
      if (egalite_equal(eq1, eq2))
	eq_set_vect_nul(eq2);
      else if(vect_equal_except(eq1->vecteur,eq2->vecteur, TCST)) {
	/* deux equations ne differant que par leurs termes constants */
	sc_rm(ps);
	return(SC_EMPTY);
      }
    }
  }

  /* Check redundancy and inconsistency between pair of inequalities */

  for (eq1 = ps->inegalites; eq1 != NULL;eq1 = eq1->succ) {

    /* Detection of inconsistant or redundant inequalities: incompatible or
       useless constant term */

    if ((vect_size(eq1->vecteur) == 1) && (eq1->vecteur->var == TCST)) {
      if (value_negz_p(val_of(eq1->vecteur))) {
	vect_rm(eq1->vecteur);
	eq1->vecteur = NULL;
      }
      else {
	/* 0 <= b < 0 */
	sc_rm(ps);
	return(SC_EMPTY);
      }
    }
	
    /* Equal inequalities, redundant inequalities, equality detection */

    for (eq2 = eq1->succ;eq2 != NULL;eq2 = eq2->succ) {
      if (contrainte_equal(eq1,eq2)) {
	eq_set_vect_nul(eq2);
      }
      else if(eq_smg(eq1,eq2)) {
	if(vect_coeff(TCST, contrainte_vecteur(eq1))
	   > vect_coeff(TCST, contrainte_vecteur(eq2)))
	  eq_set_vect_nul(eq2);
	else
	  /* opposite STRICT inequality or contrainte_equal() would have
             caught it */
	  eq_set_vect_nul(eq1);
      }
      else {
	Pvecteur sum = vect_add(contrainte_vecteur(eq1),
				contrainte_vecteur(eq2));

	if(VECTEUR_NUL_P(sum)) {
	  /* inequalities eq1 and eq2 define an equality */
	  Pcontrainte eq = contrainte_make(vect_copy(contrainte_vecteur(eq1)));

	  /* No need to update the basis since it used to be an inequality */
	  sc_add_egalite(ps, eq);
	  eq_set_vect_nul(eq1);
	  eq_set_vect_nul(eq2);
	}
	else if(vect_constant_p(sum)) {
	  if(value_pos_p(vect_coeff(TCST, sum))) {
	    /* These inequalities are incompatible and the system is not satisfiable */
	    vect_rm(sum);
	    sc_rm(ps);
	    return(SC_EMPTY);
	  }
	}
	vect_rm(sum);
      }
    }
  }

  /* Check redundancies and inconsistencies between equalities and
     inequalities */

  for (ineq1 = ps->inegalites; ineq1 != NULL;ineq1 = ineq1->succ) {
    for (eq2 = ps->egalites; eq2 != NULL; eq2 = eq2->succ) {
      Pvecteur diff1 = vect_add(contrainte_vecteur(ineq1), contrainte_vecteur(eq2));

      if (VECTEUR_NUL_P(diff1)) {
	  vect_rm(ineq1->vecteur);
	  ineq1->vecteur = NULL;
      }
      else if(vect_constant_p(diff1)) {
	if (value_neg_p(vecteur_val(diff1))) {
	  vect_rm(ineq1->vecteur);
	  ineq1->vecteur = NULL;
	}
	else {
	  /* 0 < b <= 0 */
	  vect_rm(diff1);
	  sc_rm(ps);
	  return(NULL);
	}
      }
      else {
	Pvecteur diff2 = vect_substract(contrainte_vecteur(ineq1), contrainte_vecteur(eq2));

	if (vect_constant_p(diff2)) {
	  if (VECTEUR_NUL_P(diff2) || value_neg_p(vecteur_val(diff2))) {
	    vect_rm(ineq1->vecteur);
	      ineq1->vecteur = NULL;
	  }
	  else {
	    /* 0 < b <= 0 */
	    vect_rm(diff2);
	    sc_rm(ps);
	    return(NULL);
	  }
	}
	vect_rm(diff2);
      }
      vect_rm(diff1);
    }
  }
  sc_elim_empty_constraints(ps, true);
  sc_elim_empty_constraints(ps, false);

  return (ps);
}
