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

 /* package vecteur - operations binaires */

/*LINTLIBRARY*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <assert.h>
#include <limits.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

/* Pvecteur vect_add(Pvecteur v1, Pvecteur v2): allocation d'un vecteur v dont
 * la valeur est la somme des deux vecteurs v1 et v2
 *
 *          ->
 * allocate v;
 * ->   ->   ->
 * v := v1 + v2;
 *        ->
 * return v;
 *
 * RT: j'ai besoin d'un vect_add a effet de bord pour la normalisation d'expression
 * lineaire. idem pour vect_substract, vect_mult, ...
 */
Pvecteur vect_add(v1,v2)
Pvecteur v1;
Pvecteur v2;
{
    Pvecteur v = vect_dup (v1);

    for ( ; v2!= NULL; v2=v2->succ)
	vect_add_elem (&v,var_of(v2),val_of(v2));

    return (v);
}

/* Pvecteur vect_substract(Pvecteur v1, Pvecteur v2): allocation d'un vecteur v
 * dont la valeur est la difference des deux vecteurs v1 et v2
 *
 *          ->
 * allocate v;
 * ->   ->   ->
 * v := v1 - v2;
 *        ->
 * return v;
 */
Pvecteur vect_substract(v1,v2)
Pvecteur v1,v2;
{
    Pvecteur v = vect_dup (v1);

    for (; v2 != NULL; v2 = v2->succ)
	 vect_add_elem (&v,var_of(v2),value_uminus(val_of(v2)));

    return (v);
}

/* Pvecteur vect_cl_ofl_ctrl(Pvecteur v, Value lambda, Pvecteur u, int ofl_ctrl):
 * etape d'acculumulation dans une combinaison lineaire;
 * aucun sharing entre v et u n'est cree (allocation implicite)
 * Le controle de l'overflow est effectue et traite par le retour 
 * du contexte correspondant au dernier CATCH(overflow_error) effectue.
 *
 * ->   ->         ->
 * v := v + lambda u;
 *        ->
 * return v;
 *
 * Modifications:
 *  - exploitation des lambdas 1 et -1; ca rallonge mechamment le code;
 *    non teste; u est casse mais en principe ce n'est pas grave car il n'est
 *    utilise qu'une seule fois (Francois Irigoin, 26/03/90)
 *  - add the cu variable to avoid side effects on parameter u;
 *    (Francois Irigoin, 17 December 1991)
 *  - add assert() to try to avoid most integer overflow; this is likely
 *    to (uselessly) slow down dependence tests since overflows occur
 *    only (?) in code generation phases; (Francois Irigoin, 17 December 1991)
 */
Pvecteur vect_cl_ofl_ctrl(v, lambda, u, ofl_ctrl)
Pvecteur v;
Value lambda;
Pvecteur u;
int ofl_ctrl;
{
    Pvecteur cu;

    if(lambda==VALUE_ZERO)
	return v;
    else if(v==NULL) {
	/* ancienne version
	 * for( ;u!=NULL;u=u->succ) 
	 *   v = vect_chain(v,var_of(u),lambda*val_of(u));
	 */
	v = vect_dup(u);
	if (value_notone_p(lambda)) {
	    if (value_mone_p(lambda))
		vect_chg_sgn(v);
	    else 
		v = vect_multiply(v, lambda);
	}
    }
    else
	if (value_one_p(lambda)) /* == 1 */
	    for(cu=u ;cu!=NULL;cu=cu->succ) 
		vect_add_elem(&v, var_of(cu), val_of(cu));
	else if (value_mone_p(lambda)) /* == -1 */
	    for(cu=u ;cu!=NULL;cu=cu->succ) 
		vect_add_elem(&v, var_of(cu), value_uminus(val_of(cu)));
	else
	    for(cu=u ;cu!=NULL;cu=cu->succ) {
		/* bof, FC */
		Value x = ofl_ctrl!=NO_OFL_CTRL?
		    value_protected_mult(lambda,val_of(cu)):
		    value_mult(lambda,val_of(cu));

		vect_add_elem(&v, var_of(cu), x);
	    }
    
    return v;
}



Pvecteur vect_cl_ofl(v,lambda,u)
Pvecteur v;
Value lambda;
Pvecteur u;
{
    return vect_cl_ofl_ctrl(v, lambda, u, FWD_OFL_CTRL);
}

Pvecteur vect_cl(v,lambda,u)
Pvecteur v;
Value lambda;
Pvecteur u;
{
    return vect_cl_ofl_ctrl(v, lambda, u, NO_OFL_CTRL);
}

/* Pvecteur vect_cl2_ofl(Value x1, Pvecteur v1, Value x2, Pvecteur v2):
 * allocation d'un vecteur v dont la valeur est
 * la combinaison lineaire des deux vecteurs v1 et v2 avec les 
 * coefficients respectifs x1 et x2
 * Le controle de l'overflow est effectue par vect_cl_ofl et traite par 
 * le retour du contexte correspondant au dernier CATCH(overflow_error) 
 * effectue.
 *          ->
 * allocate v;
 * ->      ->      ->
 * v := x1 v1 + x2 v2;
 *        ->
 * return v;
 *
 */
Pvecteur vect_cl2_ofl_ctrl(x1, v1, x2, v2, ofl_ctrl)
Value x1;
Pvecteur v1;
Value x2;
Pvecteur v2;
int ofl_ctrl;
{
    /* le bout de l'horreur sur ces vecteurs creux dont les composantes ne
       sont pas triees; Malik a essaye d'eviter les allocations inutiles
       en "marquant" les coefficients de v2 qui ont ete vus lors du parcours
       des coefficients non-nuls de v1; puis il ajoute a ce premier resultat
       la partie de x2 v2 qui n'a pas encore ete prise en compte parce
       que le coefficient correspondant dans v1 etait nul;

       la variable de nom 0 est traitee a part car la procedure de
       marquage (multiplication par -1) ne la marque pas

       Une autre solution, presque aussi efficace, consisterait a
       allouer et calculer x1 v1 puis a y ajouter x2 v2 a coups
       de vect_add_elem; on n'a pas de procedure faisant simultanement
       l'allocation et la multiplication par un scalaire; on n'a pas
       de procedure faisant l'addition sans allocation (accumulation);

       Francois Irigoin, 2 juin 1989 */

    Pvecteur v = NULL;
    
    v = vect_cl_ofl_ctrl(v,x1,v1, ofl_ctrl);
    v = vect_cl_ofl_ctrl(v,x2,v2, ofl_ctrl);

    return(v);
}


Pvecteur vect_cl2_ofl(x1,v1,x2,v2)
Value x1;
Pvecteur v1;
Value x2;
Pvecteur v2;
{
    return(vect_cl2_ofl_ctrl(x1,v1,x2,v2, FWD_OFL_CTRL));
}

Pvecteur vect_cl2(x1,v1,x2,v2)
Value x1;
Pvecteur v1;
Value x2;
Pvecteur v2;
{
    return(vect_cl2_ofl_ctrl(x1,v1,x2,v2, NO_OFL_CTRL));
}



/* Pvecteur vect_subst(Variable v, Pvecteur v1, Pvecteur v2): calcul
 * d'un vecteur v3 tel que l'intersection des hyperplans definis par
 * v1 et v2 soit egale a l'intersection des hyperplans definis
 * par v1 et v3, et que v appartiennent a l'hyperplan defini par v3.
 * v3 est defini a une constante multiplicative pret et ses coefficients
 * ne sont donc pas necessairement normalises, mais on s'arrange pour
 * multiplier v2 par une constante positive ce qui est utile si v2
 * represente une inegalite (cf. package contrainte).
 *
 *   ->  ->   ->  ->  ->   ->          ->  ->   ->      ->   ->
 * { U / V1 . u = 0 & v2 . u = 0 } = { u / v1 . u = 0 & v3 . u = 0 }
 * 
 * ->   ->
 * v3 . v = 0
 *
 * Si v2 remplit la condition v2 . v = 0, on prend v3 = v2.
 *
 * Il faut comme precondition a l'appel que v1 . v != 0
 *
 * Le vecteur v1 est preserve. Le vecteur v2 est detruit. v3 est alloue.
 *
 * Cette routine peut servir a eliminer une variable entre deux egalites
 * ou inegalites. C'est pourquoi v2 est detruit. Il est destine a etre
 * remplace par v3. Voir contrainte_subst().
 *
 * ATTENTION: je ne comprends pas a quoi sert le vect_chg_coeff(). F.I.
 * Les commentaires precedent sont donc partiellement (?) faux!!!
 */
Pvecteur vect_subst(v,v1,v2)
Variable v;
Pvecteur v1;
Pvecteur v2;
{
    Pvecteur v3;
    /* cv_v1 = coeff de v dans v1 */
    Value cv_v1 = vect_coeff(v,v1);
    /* cv_v2 = coeff de v dans v2 */
    Value cv_v2 = vect_coeff(v,v2);

    if (cv_v2==VALUE_ZERO) {
	/* v2 est solution; i.e., substitution non faite: var absente */
	v3 = v2;
    }
    else {
	if (cv_v1<VALUE_ZERO) {
	    v3 = vect_cl2(value_uminus(cv_v1),v2,cv_v2,v1);
	    vect_chg_coeff(&v3,v,value_uminus(cv_v2));
	}
	else {
	    v3 = vect_cl2(cv_v1,v2,value_uminus(cv_v2),v1);
	    vect_chg_coeff(&v3,v,cv_v2);
	}
	vect_rm(v2);
    }
    return(v3);
}

