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

/* package sparse_sc: conversion de systemes representes par des matrices
 *                    pleines en systemes representes par des contraintes
 *                    creuses
 *
 * Corinne Ancourt
 */
/*
Matrice's format: A[0]=denominator.stock column to column
*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrix.h"
#include "matrice.h"


/* void sys_matrice_index(Psysteme sc, Pbase base_index,
 * matrice A, int n, int m)
 * create the matrix A [m,n] made with the coefficients of the index variables
 */
void sys_matrice_index(sc, base_index, A, n, m)
Psysteme sc;
Pbase base_index;
matrice A;
int n;
int m;

{
    Pcontrainte pc;
    Pvecteur pv;
    int i, j;
   
    matrice_nulle(A, m, n);
    for (pc = sc->inegalites, i=1; pc != NULL; pc = pc->succ, i++)
	for(pv = base_index, j=1; pv != NULL; pv = pv->succ, j++)
	    ACCESS(A,m,i,j) = vect_coeff(pv->var, pc->vecteur);
}




/* void matrice_index_sys(Psysteme sc, Pbase base_index,
 * matrice AG, int n, int m)
 * replace the coefficients of the index variables in sc
 * by their new coefficients in the matrix AG[m,n].
 * Modif: taken into account the order of system of constraints.
 */
void matrice_index_sys(sc,base_index,AG,n,m)
Psysteme sc;
Pbase base_index;
matrice AG;
int n;
int m;
{
    Pcontrainte pc;
    Pbase pb;
    Pvecteur pv;
    int i,j;
    Value deno;

    deno = DENOMINATOR(AG);
    for (pc = sc->inegalites, i=1; i<=m; pc = pc->succ, i++)
	for (pb = base_index, j=1; j<=n; pb = pb->succ, j++)

	  /*	    vect_chg_coeff(&pc->vecteur, pb->var, ACCESS(AG,m,m-i+1,j));*/
	  /* Obsolete code: 
		* this old code was implemented assumming the order of system of constraints
		* had changed because of _dup version. Thanks to _copy version, it's now straight.
	   * However, there'll be an incompatiblity if exist some calls of this function 
		* (which is already obsolete since it uses matrice instead of matrix) outside hyperplane.c. 
		* changed by DN.
		*/
	  vect_chg_coeff(&pc->vecteur, pb->var, ACCESS(AG,m,i,j));

    if (value_gt(deno,VALUE_ONE))
	for (pc = sc->inegalites, i=1; i<=m; pc = pc->succ, i++)
	    for (pv = pc->vecteur; pv != NULL; pv = pv->succ)
		if (base_find_variable(base_index,pv->var)==VARIABLE_UNDEFINED)
		    value_product(pv->val,deno);		    
}


 





/* Creation de la matrice A correspondant au systeme lineaire et de la matrice
 * correspondant a la partie constante B
 * Le systeme est suppose ne pas contenir de constantes symboliques.
 *
 *  Les parametres de la fonction :
 *
 * Psysteme ps  : systeme lineaire 
 *!int 	A[]	:  matrice
 *!int 	B[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 */

void sc_to_matrices(ps,base_index,A,B,n,m)
Psysteme ps;
Pbase base_index;
matrice A, B;
int n,m;
{
    int i,j;
    Pcontrainte eq;
    Pvecteur pv;
    matrice_nulle(B,n,1);
    matrice_nulle(A,n,m);

    for (eq = ps->inegalites,i=1;
	 !CONTRAINTE_UNDEFINED_P(eq); 
	 eq=eq->succ,i++) {	
	for(pv = base_index, j=1; pv != NULL; pv = pv->succ, j++){
	    ACCESS(A,n,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
	}
	ACCESS(B,n,i,1) = vect_coeff(0,eq->vecteur);
    }
}

 

/*
 * Creation d'un systeme lineaire  a partir de deux matrices. La matrice B
 * correspond aux termes constants de chacune des inequations appartenant 
 * au systeme. La matrice A correspond a la partie lineaire  des expressions 
 * des inequations le  composant.
 * Le systeme est suppose ne pas contenir de constantes symboliques.
 * 
 * L'ensemble des variables du nouveau systeme est initialise avec une base 
 * d'indices que l'on donne en argument. Cette base peut etre vide (NULL).
 *
 * Des nouvelles variables sont creees si necessaire  si il n'y a pas assez 
 * de variables dans la base fournie.
 *
 *  La matrice A correspond a la partie non constante du systeme lineaire.
 *  La matrice B correspond a la partie constante.
 *  Le syteme lineaire s'ecrit    A.x <= B.
 *
 *  Les parametres de la fonction :
 *
 *!Psysteme ps  : systeme lineaire
 * int 	A[]	: matrice  de dimension (n,m)
 * int 	B[]	: matrice  de dimension (n,1)
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice A
 */


void matrices_to_sc(ps,base_index,A,B,n,m)
Psysteme ps;
Pbase base_index;
matrice A,B;
int n,m;
{
    Pvecteur vect,pv=NULL;
    Pcontrainte cp,pc= NULL;
    Pbase b;
    int i,j;
    Value cst,coeff,dena,denb;
    bool trouve ;

    /* create the  variables */

    ps->base = base_reversal(vect_dup(base_index));

    for (b = ps->base;!VECTEUR_UNDEFINED_P(b);b=b->succ,ps->dimension ++) ;
   
    /* ajout des variables supplementaires utiles */

    if (VECTEUR_NUL_P(ps->base)) {
	Variable var = creat_new_var(ps);
	ps->base = vect_new(var,VALUE_ONE);
    }

    for (b = ps->base,i =2; i<= m; i++,b=b->succ)  
	if (VECTEUR_NUL_P(b->succ)) {
	    Variable var = creat_new_var(ps);
	    b->succ = vect_new(var,VALUE_ONE);
	}

   
    dena = DENOMINATOR(A);
    denb = DENOMINATOR(B);

    for (i=1;i<=n; i++)
    {	trouve = false;
	cp = contrainte_new();

	/* build the constant terme if it exists */
	cst = ACCESS(B,n,i,1);
	if (value_notzero_p(cst)) {
	    pv = vect_new(TCST, value_mult(dena,cst));
	    trouve = true;
	}

	for (vect = ps->base,j=1;j<=m;vect=vect->succ,j++) {
	    coeff = ACCESS(A,n,i,j);
	    if (value_notzero_p(coeff)) {
		if (trouve) 
		    vect_chg_coeff(&pv, vecteur_var(vect),
				   value_mult(denb,coeff));
		else {	/* build a new vecteur if there is not constant term */
		    pv = vect_new(vecteur_var(vect), 
				  value_mult(denb,coeff));
		    trouve = true;
		}
	    }
	}
	cp->vecteur = pv;
	cp->succ = pc;
	pc = cp;
    }

    ps->inegalites = pc;
    ps->nb_ineq = n;
    ps->egalites = NULL;
    ps->nb_eq = 0;


}





/* Creation de la matrice A correspondant au systeme lineaire et de la matrice
 * correspondant a la partie constante B
 * Le systeme peut contenir des constantes symboliques. Dans ce cas, la base 
 * index_base ne doit contenir que les variables etant des indices de boucles 
 * et la base  const_base les constantes symboliques. La matrice B 
 * represente toutes les contraintes sur les constantes.
 * 
 *  Les parametres de la fonction :
 *
 * Psysteme ps  : systeme lineaire 
 *!int 	A[]	:  matrice
 *!int 	B[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 */

void loop_sc_to_matrices(ps,index_base,const_base,A,B,n,m1,m2)
Psysteme ps;
Pbase index_base,const_base;
matrice A;
matrice B;
int n,m1,m2;
{

    int i,j;
    Pcontrainte eq;
    Pvecteur pv;
 
    matrice_nulle(B,n,m2);
    matrice_nulle(A,n,m1);

    for (eq = ps->inegalites,i=1;
	 !CONTRAINTE_UNDEFINED_P(eq); 
	 eq=eq->succ,i++) {	
	for(pv = index_base, j=1; pv != NULL; pv = pv->succ, j++){
	    ACCESS(A,n,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
	}
	for(pv = const_base, j=1; pv != NULL; pv = pv->succ, j++){
	    ACCESS(B,n,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
	}
	ACCESS(B,n,i,m2) = vect_coeff(TCST,eq->vecteur);
    }

}


/*
 * Creation d'un systeme lineaire  a partir de deux matrices. La matrice B
 * correspond aux termes constants de chacune des inequations appartenant 
 * au systeme. La matrice A correspond a la partie lineaire  des expressions 
 * des inequations le  composant.
 *
 * Le systeme peut contenir des constantes symboliques. Dans ce cas, la 
 * matrice B represente toutes les contraintes sur les constantes.
 * La base index_base ne  contiendra que les variables etant des indices de 
 * boucles et la base  const_base les constantes symboliques.
 *
 * L'ensemble des variables du nouveau systeme est initialise avec une base 
 * d'indices que l'on donne en argument. Cette base peut etre vide (NULL).
 *
 * Des nouvelles variables sont creees si necessaire  si il n'y a pas assez 
 * de variables dans la base fournie.
 *
 *  La matrice A correspond a la partie non constante du systeme lineaire.
 *  La matrice B correspond a la partie constante.
 *  Le syteme lineaire s'ecrit    A.x <= B.
 *
 *  Les parametres de la fonction :
 *
 *!Psysteme ps  : systeme lineaire
 * int 	A[]	: matrice  de dimension (n,m)
 * int 	B[]	: matrice  de dimension (n,m2)
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice A
 */
 
void matrices_to_loop_sc(ps,index_base,const_base,A,B,n,m1,m2)
Psysteme ps;
Pbase index_base,const_base;
matrice A,B;
int n,m1,m2;
{
    Pvecteur vect,pv=NULL;
    Pcontrainte cp,pc= NULL;
    Pbase b;
    int i,j;
    Value cst,coeff,dena,denb;
    bool trouve ;

    /* create the  variables */

    if (index_base != BASE_NULLE)
    ps->base = base_reversal(vect_dup(index_base));
    else ps->base = VECTEUR_UNDEFINED;

    ps->dimension = vect_size(ps->base);
   
    /* ajout des variables supplementaires utiles */

    if (VECTEUR_NUL_P(ps->base)) {
	Variable var = creat_new_var(ps);
	ps->base = vect_new(var,VALUE_ONE);
	ps->dimension++;
    }

    for (b = ps->base,i =2; i<= m1; i++,b=b->succ)  
	if (VECTEUR_NUL_P(b->succ)) {
	    Variable var = creat_new_var(ps);
	    b->succ = vect_new(var,VALUE_ONE);
	    ps->dimension++;
	}

    ps->base = vect_add(vect_dup(const_base),vect_dup(ps->base));

    ps->dimension += vect_size(const_base);

    for (b = ps->base,i =2; i<= m1+m2-1; i++,b=b->succ)  
	if (VECTEUR_NUL_P(b->succ)) {
	    Variable var = creat_new_var(ps);
	    b->succ = vect_new(var,VALUE_ONE);
	    ps->dimension++;
	}

    dena = DENOMINATOR(A);
    denb = DENOMINATOR(B);

    for (i=1;i<=n; i++)
    {	trouve = false;
	cp = contrainte_new();

	/* build the constant terme if it exists */
	cst = ACCESS(B,n,i,m2);

	if (value_notzero_p(cst)) {
	    pv = vect_new(TCST,  value_mult(dena,cst));
	    trouve = true;
	}


	for (vect = ps->base,j=1;j<=m1;vect=vect->succ,j++) {
	    coeff = ACCESS(A,n,i,j);
	    if (value_notzero_p(coeff)) {
		if (trouve) 
		    vect_chg_coeff(&pv, vecteur_var(vect),
				   value_mult(denb,coeff));
		else {			
		    /* build a new vecteur if there is not constant term */
		    pv = vect_new(vecteur_var(vect), value_mult(denb,coeff));
		    trouve = true;
		}
	    }
	}

	for (j=1;j<=m2-1;vect=vect->succ,j++) {
	    coeff = ACCESS(B,n,i,j);
	    if (value_notzero_p(coeff)) {
		if (trouve) 
		    vect_chg_coeff(&pv, vecteur_var(vect),
				   value_mult(denb,coeff));
		else {			
		    /* build a new vecteur if there is not constant term */
		    pv = vect_new(vecteur_var(vect), value_mult(denb,coeff));
		    trouve = true;
		}
	    }
	}
	cp->vecteur = pv;
	cp->succ = pc;
	pc = cp;
    }

    ps->inegalites = pc;
    ps->nb_ineq = n;
    ps->egalites = NULL;
    ps->nb_eq = 0;


}






/* ======================================================================= */
/*
 * void constraints_to_matrices(Pcontrainte pc, Pbase b,
 *				matrix A B):
 * constructs the matrices "A" and "B" corresponding to the linear
 * constraints "pc", so: A.b + B = 0 <=> pc(b) = 0.
 *
 * The base "b" gives the variables of the linear system.
 *
 * The matrices "A" and "B" are supposed to have been already allocated in
 * memory, respectively of dimension (n, m) and (n, 1).
 *
 * "n" must be the exact number of constraints in "pc".
 * "m" must be the exact number of variables in "b".
 */
void constraints_to_matrices(pc, b, A, B)
Pcontrainte pc;
Pbase b;
Pmatrix A, B;
{
    int i,j;
    Pvecteur pv;
    Pcontrainte eq;
    int n=0;

    for (eq = pc; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,n++);
  
    matrix_nulle(B);
    matrix_nulle(A);

    for(eq = pc,i=1; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,i++) {
	for(pv = b, j=1; pv != NULL; pv = pv->succ, j++){
	    MATRIX_ELEM(A,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
	}
	MATRIX_ELEM(B,i,1) = vect_coeff(0,eq->vecteur);
    }
}


/* ======================================================================= */
/*
 * void matrices_to_constraints(Pcontrainte *pc, Pbase b,
 *				matrix A B):
 * constructs the constraints "pc" corresponding to the matrices "A" and "B"
 * so: pc(b) = 0 <=> A.b + B = 0
 *
 * The base "b" gives the variables of the linear system.
 * The matrices "A" and "B" are respectively of dimension (n, m) and (n, 1).
 *
 * "n" will be the exact number of constraints in "pc".
 * "m" must be the exact number of variables in "b".
 *
 * Note: the formal parameter pc is a "Pcontrainte *". Instead, the resulting
 * Pcontrainte could have been returned as the result of this function.
 */
void matrices_to_constraints(pc, b, A, B)
Pcontrainte *pc;
Pbase b;
Pmatrix A, B;
{
    Pcontrainte newpc = NULL;
    int i, j;
    Value cst, coeff, dena, denb;
    int n = MATRIX_NB_LINES(A);
    int m = MATRIX_NB_COLUMNS(A);

    dena = MATRIX_DENOMINATOR(A);
    denb = MATRIX_DENOMINATOR(B);

    for (i=n;i>=1; i--) {
	Pvecteur vect, pv;
	Pcontrainte cp;
	bool found = false;

	cp = contrainte_new();

	/* build the constant terme if it is not null */
	cst = MATRIX_ELEM(B,i,1);
	if (value_notzero_p(cst)) {
	    pv = vect_new(TCST,  value_mult(dena,cst));
	    found = true;
	}

	for (vect = b,j=1;j<=m;vect=vect->succ,j++) {
	    coeff = MATRIX_ELEM(A,i,j);
	    if (value_notzero_p(coeff)) {
		if (found)
		    vect_chg_coeff(&pv, vecteur_var(vect),
				   value_mult(denb,coeff));
		else {
		    /* build a new vecteur if there is a null constant term */
		    pv = vect_new(vecteur_var(vect), 
				   value_mult(denb,coeff));
		    found = true;
		}
	    }
	}
	/* the constraints are in reverse order */
	cp->vecteur = pv;
	cp->succ =  newpc;
	newpc = cp;
    }
    *pc = newpc;
}

/* ======================================================================= */
/* 
 * void constraints_with_sym_cst_to_matrices(Pcontrainte pc,
 *	Pbase index_base const_base, matrice A B, int n m1 m2):
 *
 * constructs the matrices "A" and "B" corresponding to the linear
 * constraints "pc", so: A.ib + B1.cb + B2 = 0 <=> pc(ib, cb) = 0:
 *
 *	B = ( B1 | B2 ), B2 of dimension (n,1).
 *
 * The basis "ib" gives the variables of the linear system.
 * The basis "cb" gives the symbolic constants of the linear system.
 *
 * The matrices "A" and "B" are supposed to have been already allocated in
 * memory, respectively of dimension (n, m1) and (n, m2).
 *
 * "n" must be the exact number of constraints in "pc".
 * "m1" must be the exact number of variables in "ib".
 * "m2" must be equal to the number of symbolic constants (in "cb") PLUS
 * ONE (the actual constant).
 */
void constraints_with_sym_cst_to_matrices(pc,index_base,const_base,A,B)
Pcontrainte pc;
Pbase index_base,const_base;
Pmatrix A, B;
{
    int i,j;
    Pcontrainte eq;
    Pvecteur pv;
    int n = 0;
    int m2 = vect_size(const_base);
    for (eq = pc; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,n++);
    matrix_nulle(B);
    matrix_nulle(A);

    for (eq = pc,i=1; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,i++) {
        for(pv = index_base, j=1; pv != NULL; pv = pv->succ, j++){
            MATRIX_ELEM(A,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
        }
        for(pv = const_base, j=1; pv != NULL; pv = pv->succ, j++){
            MATRIX_ELEM(B,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);
        }
        MATRIX_ELEM(B,i,m2) = vect_coeff(TCST,eq->vecteur);
    }
}


/* ======================================================================= */
/*
 * void matrices_to_constraints_with_sym_cst(Pcontrainte *pc,
 *	Pbase index_base const_base, matrice A B,int n m1 m2):
 *
 * constructs the constraints "pc" corresponding to the matrices "A" and "B"
 * so: A.ib + B1.cb + B2 = 0 <=> pc(ib, cb) = 0, with:
 *
 *      B = ( B1 | B2 ), B2 of dimension (n,1).
 *
 * The basis "ib" gives the variables of the linear system.
 * The basis "cb" gives the symbolic constants of the linear system.
 *
 * The matrices "A" and "B" are respectively of dimension (n, m1) and (n, m2).
 *
 * "n" will be the exact number of constraints in "pc".
 * "m1" must be the exact number of variables in "ib".
 * "m2" must be equal to the number of symbolic constants (in "cb") PLUS
 * ONE (the actual constant).
 *
 * Note: the formal parameter pc is a "Pcontrainte *". Instead, the resulting
 * Pcontrainte could have been returned as the result of this function.
 */
void matrices_to_constraints_with_sym_cst(pc,index_base,const_base,A,B)
Pcontrainte *pc;
Pbase index_base,const_base;
Pmatrix A, B;
{
    Pvecteur vect,pv=NULL;
    Pcontrainte cp,newpc= NULL;
    int i,j;
    Value cst,coeff,dena,denb;
    bool found ;
    int n = MATRIX_NB_LINES(A);
    int m1 = MATRIX_NB_COLUMNS(A);
    int m2 = MATRIX_NB_COLUMNS(B);
    dena = MATRIX_DENOMINATOR(A);
    denb = MATRIX_DENOMINATOR(B);

    for (i=n;i>=1; i--) {
	found = false;
	cp = contrainte_new();

	/* build the constant terme if it exists */
	cst = MATRIX_ELEM(B,i,m2);
	if (value_notzero_p(cst)) {
	    pv = vect_new(TCST,  value_mult(dena,cst));
	    found = true;
	}

	vect = base_union(index_base, const_base);
	for (j=1;j<=m1;vect=vect->succ,j++) {
	    coeff = MATRIX_ELEM(A,i,j);
	    if (value_notzero_p(coeff)) {
		if (found)
		    vect_chg_coeff(&pv, vecteur_var(vect),
				   value_mult(denb,coeff));
		else {
		    /* build a new vecteur if there is not constant term */
		    pv = vect_new(vecteur_var(vect), 
				  value_mult(denb,coeff));
		    found = true;
		}
	    }
	}

	for (j=1;j<=m2-1;vect=vect->succ,j++) {
	    coeff = MATRIX_ELEM(B,i,j);
	    if (value_notzero_p(coeff)) {
		if (found)
		    vect_chg_coeff(&pv, vecteur_var(vect),
				   value_mult(denb,coeff));
		else {
		    /* build a new vecteur if there is not constant term */
		    pv = vect_new(vecteur_var(vect),
				  value_mult(denb,coeff));
		    found = true;
		}
	    }
	}
	cp->vecteur = pv;
	cp->succ = newpc;
	newpc = cp;
    }
    *pc = newpc;
}



