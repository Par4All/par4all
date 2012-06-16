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

 /* package matrice */

 /* ce fichier comporte les routines traitant des sous-matrices inferieures
    droites; ces routines traitent la matrice toute entiere si le parametre
    level vaut 0; les routines sur les matrices completes se trouvent dans
    le fichier matrice.c
    */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"

#include "matrice.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(s,t,f) free(s)

/* int mat_lig_nnul(matrice MAT, int n, int m, int level):
 * Recherche de la premiere ligne non nulle de la 
 * la sous-matrice MAT(level+1 ..n, level+1 ..m)
 *
 * resultat retourne par la fonction :
 * 
 * int 	: numero de la premiere ligne non nulle de la matrice MAT; 
 *        ou 0 si la sous-matrice MAT(level+1 ..n,level+1 ..m) est identiquement nulle;
 *
 * parametres de la fonction :
 *
 * int 	MAT[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice 
 * int   level  : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 */
int mat_lig_nnul(MAT,n,m,level)
matrice MAT;
int n,m;
int level;
{
    int i,j;
    int I = 0;
    bool trouve = false;

    /* recherche du premier element non nul de la sous-matrice */
    for (i = level+1; i<=n && !trouve ; i++) {
	for (j = level+1; j<= m && value_zero_p(ACCESS(MAT,n,i,j)); j++);
	if(j <= m) {
	    /* on dumpe la colonne J... */
	    trouve = true;
	    I = i;
	}
    }
    return (I);
}

/* void mat_perm_col(matrice MAT, int n, int m, int k, int level):
 * Calcul de la matrice de permutation permettant d'amener la k-ieme
 * ligne de la matrice MAT(1..n,1..m) a la ligne (level + 1).
 *
 * Si l'on veut amener la k-ieme ligne de la matrice en premiere ligne
 * 'level' doit avoir la valeur 0
 *
 * parametres de la fonction :
 *
 *!int 	MAT[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice (unused)
 * int  k 	: numero de la ligne a remonter a la premiere ligne 
 * int    level : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 *
 * Note: pour eviter une multiplication de matrice en O(n**3), il vaudrait
 * mieux programmer directement la permutation en O(n**2) sans multiplications
 * Il est inutile de faire souffrir notre chip SPARC! (FI)
 */
/*ARGSUSED*/
void mat_perm_col(matrice MAT,
		  int n __attribute__ ((unused)),
		  int m __attribute__ ((unused)),
		  int k,
		  int level)
{
    int i,j;

    matrice_nulle(MAT,n,n);
    if (level > 0) {
	for (i=1;i<=level; i++)
	    ACCESS(MAT,n,i,i) = VALUE_ONE;
    }

    for (i=1+level,j=k; i<=n; i++,j++)
    {
	if (j == n+1) j = 1 + level;
	ACCESS(MAT,n,i,j)=VALUE_ONE;
    }
}

/* void mat_perm_lig(matrice MAT, int n, int m, int k, int level):
 * Calcul de la matrice de permutation permettant d'amener la k-ieme
 * colonne de la sous-matrice MAT(1..n,1..m) a la colonne 'level + 1'
 *  (premiere colonne de la sous matrice MAT(level+1 ..n,level+1 ..m)).
 *
 * parametres de la fonction :
 *
 *!int 	MAT[]	:  matrice
 * int  n	: nombre de lignes de la matrice (unused)
 * int  m 	: nombre de colonnes de la matrice
 * int  k 	: numero de la colonne a placer a la premiere colonne 
 * int  level   : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 */
/*ARGSUSED*/
void mat_perm_lig(matrice MAT,
		  int n __attribute__ ((unused)),
		  int m __attribute__ ((unused)),
		  int k,
		  int level)
{
    int i,j;

    matrice_nulle(MAT,m,m);
    if(level > 0) {
	for (i = 1; i <= level;i++)
	    ACCESS(MAT,m,i,i) = VALUE_ONE;
    }

    for(j=1,i=k-level; j <= m - level; j++,i++) {
	if(i == m-level+1)
	    i = 1;
	ACC_ELEM(MAT,m,i,j,level) = VALUE_ONE;
    }
}

/* void mat_min(matrice MAT, int n, int m, int * i_min, int * j_min,
 *              int level):
 * Recherche des coordonnees (*i_min, *j_min) de l'element 
 * de la sous-matrice MAT(level+1 ..n, level+1 ..m) dont la valeur absolue est
 * la plus petite et non nulle.
 *
 * QQ i dans [level+1 ..n] 
 * QQ j dans [level+1 ..m]
 *   | MAT[*i_min, *j_min] | <= | MAT[i, j]
 *
 * resultat retourne par la fonction :
 * 
 *  les parametres i_min et j_min sont modifies.
 *
 * parametres de la fonction :
 *
 * int 	MAT[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 *!int i_min	: numero de la ligne a laquelle se trouve le plus petit 
 *		  element
 *!int j_min 	: numero de la colonne a laquelle se trouve le plus petit 
 * int  level   : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 *		  element
 */
void mat_min(MAT,n,m,i_min,j_min,level)
matrice MAT;
int n,m;
int *i_min,*j_min;
int level;
{
    int i,j;
    int vali= 0;
    int valj=0;
    register Value min=VALUE_ZERO, val;
    bool trouve = false;


    /*  initialisation du minimum  car recherche d'un minimum non nul*/
    for (i=1+level;i<=n && !trouve;i++)
	for(j = level+1; j <= m && !trouve; j++) {
	    min = ACCESS(MAT,n,i,j);
	    value_absolute(min);
	    if(min != 0) {
		trouve = true;
		vali = i;
		valj = j;
	    }
	}

    for (i=1+level;i<=n;i++)
	for (j=1+level;j<=m && value_gt(min,VALUE_ONE); j++) {
	    val = ACCESS(MAT,n,i,j);
	    value_absolute(val);
	    if (value_notzero_p(val) && value_lt(val,min)) {
		min = val;
		vali= i;
		valj =j;
	    }
	}
    *i_min = vali;
    *j_min = valj;
}

/* void mat_maj_col(matrice A, int n, int m, matrice P, int level):
 * Calcul de la matrice permettant de remplacer chaque terme de
 * la premiere ligne de la sous-matrice A(level+1 ..n, level+1 ..m) autre
 * que le premier terme A11=A(level+1,level+1) par le reste de sa
 * division entiere par A11
 *
 *
 *  La matrice P est modifiee.
 *
 *  Les parametres de la fonction :
 *
 * int 	A[1..n,1..m]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice (unused)
 *!int 	P[]	:  matrice de dimension au moins P[1..n, 1..n]
 * int  level   : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 */
/*ARGSUSED*/
void mat_maj_col(matrice A,
		 int n __attribute__ ((unused)),
		 int m __attribute__ ((unused)),
		 matrice P,
		 int level)
{
    register Value A11;
    register Value x;
    int i;

    matrice_identite(P,n,0);

    A11 = ACC_ELEM(A,n,1,1,level);
    for (i=2+level; i<=n; i++) {
	x = ACCESS(A,n,i,1+level);
	value_division(x,A11);
	ACCESS(P,n,i,1+level) = value_uminus(x);
    }
}

/* void mat_maj_lig(matrice A, int n, int m, matrice Q, int level):
 * Calcul de la matrice permettant de remplacer chaque terme de
 * la premiere ligne autre que le premier terme A11=A(level+1,level+1)
 *  par le reste de sa
 * division entiere par A11
 *
 *
 * La matrice Q est modifiee.
 *
 *  Les parametres de la fonction :
 *
 * int 	A[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 *!int 	Q[]	:  matrice
 * int  level   : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 *
 */
void mat_maj_lig(A,n,m,Q,level)
matrice A;
int n,m;
matrice Q;
int level;
{
    register Value A11;
    int j;
    register Value x;

    matrice_identite(Q,m,0);
    A11 = ACC_ELEM(A,n,1,1,level);
    for (j=2+level; j<=m; j++) {
	x = ACCESS(A,n,1+level,j);
	value_division(x,A11);
	ACCESS(Q,m,1+level,j) = value_uminus(x);
    }
}

/* void matrice_identite(matrice ID, int n, int level)
 * Construction d'une sous-matrice identite dans ID(level+1..n, level+1..n)
 *
 *  Les parametres de la fonction :
 *
 *!int 	ID[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  level   : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 */
void matrice_identite(ID,n,level)
matrice ID;
int n;
int level;
{
    int i,j;

    for(i = level+1; i <= n; i++) {
	for(j = level+1; j <= n; j++)
	    ACCESS(ID,n,i,j) = VALUE_ZERO;
	ACCESS(ID,n,i,i) = VALUE_ONE;
    }

    DENOMINATOR(ID) = VALUE_ONE;
}

/* bool matrice_identite_p(matrice ID, int n, int level)
 * test d'une sous-matrice dans ID(level+1..n, level+1..n) pour savoir si
 * c'est une matrice identite. Le test n'est correct que si la matrice
 * ID passee en argument est normalisee (cf. matrice_normalize())
 *
 * Pour tester toute la matrice ID, appeler avec level==0
 *
 *  Les parametres de la fonction :
 *
 * int 	ID[]	:  matrice
 * int  n	: nombre de lignes (et de colonnes) de la matrice ID
 * int  level   : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 */
bool matrice_identite_p(ID,n,level)
matrice ID;
int n;
int level;
{
    int i,j;

    for(i = level+1; i <= n; i++) {
	for(j = level+1; j <= n; j++) {
	    if(i==j) {
		if(value_notone_p(ACCESS(ID,n,i,i)))
		    return(false);
	    }
	    else /* i!=j */
		if(value_notzero_p(ACCESS(ID,n,i,j)))
		    return(false);
	}
    }
    return(true);
}

/* int mat_lig_el(matrice MAT, int n, int m, int level)
 * renvoie le numero de colonne absolu du premier element non nul
 * de la sous-ligne MAT(level+1,level+2..m);
 * renvoie 0 si la sous-ligne est vide.
 *
 */
/*ARGUSED*/
int mat_lig_el(MAT,n,m,level)
matrice MAT;
int n,m;
int level;
{
    int j;
    int j_min = 0;

    /* recherche du premier element non nul de la sous-ligne 
       MAT(level+1,level+2..m) */
    for(j = level+2; j<=m && value_zero_p(ACCESS(MAT,n,1+level,j)) ; j++);
    if(j < m+1)
	j_min = j-1;
    return (j_min);
}

/* int mat_col_el(matrice MAT, int n, int m, int level)
 * renvoie le numero de ligne absolu du premier element non nul
 * de la sous-colonne MAT(level+2..n,level+1);
 * renvoie 0 si la sous-colonne est vide.
 *
 */
/*ARGSUSED*/
int mat_col_el(matrice MAT,
	       int n,
	       int m __attribute__ ((unused)),
	       int level)
{
    int i;
    int i_min=0;

    for(i = level+2; i <= n && value_zero_p(ACCESS(MAT,n,i,level+1)); i++);
    if (i<n+1)
	i_min = i-1;
    return (i_min);
}

/* void mat_coeff_nnul(matrice MAT, int n, int m, int * lg_nnul,
 *                     int * cl_nnul, int level)
 * renvoie les coordonnees du plus petit element non-nul de la premiere
 * sous-ligne non nulle 'lg_nnul' de la sous-matrice 
 MAT(level+1..n, level+1..m)
 *
 *
 */
void mat_coeff_nnul(MAT,n,m,lg_nnul,cl_nnul,level)
matrice MAT;
int n,m;
int *lg_nnul,*cl_nnul;
int level;
{
    bool trouve = false;
    int j;
    register Value min=VALUE_ZERO,val;

    *lg_nnul = 0;
    *cl_nnul = 0;

    /* recherche de la premiere ligne non nulle de la 
       sous-matrice MAT(level+1 .. n,level+1 .. m)      */

    *lg_nnul= mat_lig_nnul(MAT,n,m,level);

    /* recherche du plus petit (en valeur absolue)
     * element non nul de cette ligne */
    if (*lg_nnul) {
	for (j=1+level;j<=m && !trouve; j++) {
	    min = ACCESS(MAT,n,*lg_nnul,j);
	    value_absolute(min);
	    if (value_notzero_p(min)) {
		trouve = true;
		*cl_nnul=j;
	    }
	}
	for (j=1+level;j<=m && value_gt(min,VALUE_ONE); j++) {
	    val = ACCESS(MAT,n,*lg_nnul,j);
	    value_absolute(val);
	    if (value_notzero_p(val) && value_lt(val,min)) {
		min = val;
		*cl_nnul =j;
	    }
	}
    }
}








