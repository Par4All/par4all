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

 /* package sur les polyedres
  *
  * Francois Irigoin
  */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <assert.h>
#include "arithmetique.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"

#include "matrix.h"

#include "plint.h"

/* void sys_mat_conv(Psysteme ps, matrice A, matrice B, int n, int m):
 * calcul de la matrice A correspondant a la partie lineaire des
 * egalites (ou des inegalites?) du systeme lineaire ps et de la matrice B
 * correspondant aux termes constants de ps
 *
 *  Les parametres de la fonction :
 *
 * Psysteme ps  : systeme lineaire 
 *!int 	A[]	:  matrice
 *!int 	B[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 *
 *
 * Modifications:
 *  - prise en compte du changement de signe du terme constant (FI, 19/12/89)
 */
void sys_mat_conv(ps,A,B,n,m)
Psysteme ps;
Pmatrix A;
Pmatrix B;
int n,m;
{
    int i,j;
    Pcontrainte eq;
    Pbase b;

    matrix_nulle(B);
    matrix_nulle(A);
    for(eq = ps->egalites,i=1; eq != NULL; eq=eq->succ,i++) {
	for (j=1, b=ps->base; j<= m && !VECTEUR_NUL_P(b); j++, b = b->succ)
	    MATRIX_ELEM(A,i,j) = vect_coeff(vecteur_var(b),eq->vecteur);
	MATRIX_ELEM(B,i,1) = vect_coeff(TCST,eq->vecteur);
    }
}

/* void mat_sys_conv(Psysteme ps, matrice B, int n, int m, int nbl)
 * remplissage du champ des egalites ps->egalites en fonction de la matrice B
 * qui contient les coefficients et les termes constant
 *
 *  La matrice B passee en parametres est celle calculee a l'aide 
 *  de la fonction "matrice_smith" 
 *
 *  Les parametres de la fonction :
 *
 *!Psysteme ps  : systeme lineaire avec un champ base initialise
 * int 	B[]	: matrice  de dimension (m,m+1)  correspondant a la matrice 
 *		  solution du systeme d'egalites du systeme lineaire
 * int  n	: nombre de lignes de la matrice (FI laquelle?)
 * int  m 	: nombre de colonnes de la matrice
 * int  nbl 	: nombre de variables non contraintes ajoutees a la matrice
 *
 * Modifications:
 *  - prise en compte du changement de cote du terme constant (FI, 19/12/89)
 */
void mat_sys_conv(ps,B,n,m,nbl)
Psysteme ps;
Pmatrix B;
int n,m;
int nbl;
{
    Pvecteur pv=NULL;
    Pcontrainte pc=NULL;
    Pcontrainte cp = NULL;
    Pbase b;
    Pbase b1;
    int i,j;
    Value den;

    for(i = 1; i <= nbl; i++)
	creat_new_var(ps);

    if(value_notzero_p(den = MATRIX_DENOMINATOR(B))) {
	for (i=1, b = ps->base;i<=m && !VECTEUR_NUL_P(b); i++, b = b->succ) {
	    cp = contrainte_new();
	    pv = vect_new(vecteur_var(b),den);

	    for (j=1, b1 = b; j<=nbl && !VECTEUR_NUL_P(b1); j++, b1=b1->succ)
		vect_chg_coeff(&pv, vecteur_var(b1), 
			       value_uminus(MATRIX_ELEM(B,i,j+1)));

	    vect_chg_coeff(&pv,TCST,MATRIX_ELEM(B,i,1));
	    cp->vecteur = pv;
	    cp->succ = pc;
	    pc = cp;
	}
    }
    ps->egalites = pc;
    ps->nb_eq = n+nbl;
}

extern char *variable_name(Variable v);

/* void egalites_to_matrice(matrice a, int n, int m, Pcontrainte leg, Pbase b):
 * conversion d'une liste d'egalites leg representees par des vecteurs creux
 * en une matrice pleine a de n lignes representant les egalites et de
 * m colonnes representant les variables.
 *
 * Les lignes sont assignees suivant l'ordre dans lequel les egalites
 * apparaissent dans la liste leg. Les colonnes sont assignees suivant
 * les rang des variables dans la base b.
 * 
 * Les termes constants sont pris en compte et stockes (arbitrairement)
 * dans la colonne 0.
 *
 * La matrice a est supposees avoir ete allouees en memoire avec des dimensions
 * suffisantes. Elle est initialisee a 0. Ni la liste d'egalites leg, ni 
 * la base b ne sont modifiees.
 *
 * Les nombres effectifs d'egalites et de variables ne sont pas retournes.
 *
 * L'implementation n'est pas propre, meme je ne vois pas ce qu'on peut
 * faire de mieux sans un iterateurs sur les coefficients non nuls d'un
 * vecteur. De plus, cette conversion de format est typiquement implementation
 * dependante. L'implementation faite par Corinne pour syst_mat_conv()
 * est cependante correcte. En plus, je parcours plein de fois la base...
 * I'm lost!
 *
 * Francois Irigoin, 17 avril 1990
 */
void egalites_to_matrice(a, n, m, leg, b)
Pmatrix a;
int n;
int m;
Pcontrainte leg;
Pbase b;
{
    Pvecteur v;
    int ligne;

    matrix_nulle(a);

    for(ligne=0;!CONTRAINTE_NULLE_P(leg); leg=leg->succ, ligne++) {
	assert(ligne<n);
	for(v = contrainte_vecteur(leg); !VECTEUR_UNDEFINED_P(v); 
	    v = v->succ) 
	{
	    int rank = vecteur_var(v) == TCST? 0 :
		base_find_variable_rank(b, 
					vecteur_var(v),
					variable_name);
	    assert(rank < m);
	    MATRIX_ELEM(a, ligne, rank) = vecteur_val(v);
	}
    }
}

/* Pcontrainte matrice_to_contraintes(matrice a, int n, int m, Pbase b):
 * fonction inverse de la precedente; les termes constants sont pris
 * en colonne 0; les contraintes, egalites ou inegalites, sont allouees
 * en fonction des besoins.`
 *
 * Comme on ne sait pas representer des contraintes rationnelles avec
 * le type de donnees "contrainte", le denominateur de la matrice doit
 * valoir 1.
 */
Pcontrainte matrice_to_contraintes(a, n, m, b)
Pmatrix a;
int n;
int m;
Pbase b;
{
    Pcontrainte leg = CONTRAINTE_UNDEFINED;
    return leg;
}
