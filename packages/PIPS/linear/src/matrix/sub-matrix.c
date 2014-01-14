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

 /* package matrix */

 /* ce fichier comporte les routines traitant des sous-matrices inferieures
    droites; ces routines traitent la matrice toute entiere si le parametre
    level vaut 0; les routines sur les matrices completes se trouvent dans
    le fichier matrice.c
    */

/* J'ai rajoute' une fonction peremttant d'extraire une sous matrice quelconque
 * d'une matrice donne'e BC, Sept 94 
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"

#include "matrix.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(s,t,f) free(s)

/* int matrix_line_nnul(matrice MAT,int level):
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
int matrix_line_nnul(MAT,level)
Pmatrix MAT;
int level;
{
    int i,j;
    int I = 0;
    bool trouve = false;
    int n = MATRIX_NB_LINES(MAT);
    int m= MATRIX_NB_COLUMNS(MAT);
    /* recherche du premier element non nul de la sous-matrice */
    for (i = level+1; i<=n && !trouve ; i++) {
	for (j = level+1; j<= m && (MATRIX_ELEM(MAT,i,j) == 0); j++);
	if(j <= m) {
	    /* on dumpe la colonne J... */
	    trouve = true;
	    I = i;
	}
    }
    return (I);
}

/* void matrix_perm_col(Pmatrix MAT,  int k, int level):
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
void matrix_perm_col(MAT,k,level)
Pmatrix MAT;
int k,level;
{
    int i,j;
    int n = MATRIX_NB_LINES(MAT);
    matrix_nulle(MAT);
    if (level > 0) {
	for (i=1;i<=level; i++)
	    MATRIX_ELEM(MAT,i,i) = VALUE_ONE;
    }

    for (i=1+level,j=k; i<=n; i++,j++)
    {
	if (j == n+1) j = 1 + level;
	MATRIX_ELEM(MAT,i,j)=VALUE_ONE;
    }
}

/* void matrix_perm_line(Pmatrix MAT,  int k, int level):
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
void matrix_perm_line(MAT,k,level)
Pmatrix MAT;
int k,level;
{
    int i,j;
    int m= MATRIX_NB_COLUMNS(MAT);
    matrix_nulle(MAT);
    if(level > 0) {
	for (i = 1; i <= level;i++)
	    MATRIX_ELEM(MAT,i,i) = VALUE_ONE;
    }

    for(j=1,i=k-level; j <= m - level; j++,i++) {
	if(i == m-level+1)
	    i = 1;
	SUB_MATRIX_ELEM(MAT,i,j,level) = VALUE_ONE;
    }
}

/* void matrix_min(Pmatrix MAT,  int * i_min, int * j_min,
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
void matrix_min(MAT,i_min,j_min,level)
Pmatrix MAT;
int *i_min,*j_min;
int level;
{
    int i,j;
    int vali= 0;
    int valj=0;
    Value min=VALUE_ZERO;
    Value val=VALUE_ZERO;
    bool trouve = false;

    int n = MATRIX_NB_LINES(MAT);
    int m= MATRIX_NB_COLUMNS(MAT);
    /*  initialisation du minimum  car recherche d'un minimum non nul*/
    for (i=1+level;i<=n && !trouve;i++)
	for(j = level+1; j <= m && !trouve; j++) {
	    min = MATRIX_ELEM(MAT,i,j);
	    min = value_abs(min);
	    if(value_notzero_p(min)) {
		trouve = true;
		vali = i;
		valj = j;
	    }
	}

    for (i=1+level;i<=n;i++)
	for (j=1+level;j<=m && value_gt(min,VALUE_ONE); j++) {
	    val = MATRIX_ELEM(MAT,i,j);
	    val = value_abs(val);
	    if (value_notzero_p(val) && value_lt(val,min)) {
		min = val;
		vali= i;
		valj =j;
	    }
	}
    *i_min = vali;
    *j_min = valj;
}

/* void matrix_maj_col(Pmatrix A, matrice P, int level):
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
void matrix_maj_col(A,P,level)
Pmatrix A;
Pmatrix P;
int level;
{
    Value A11;
    int i;
    Value x;
    int n = MATRIX_NB_LINES(A);
    matrix_identity(P,0);

    A11 =SUB_MATRIX_ELEM(A,1,1,level);
    for (i=2+level; i<=n; i++) {
	x = MATRIX_ELEM(A,i,1+level);
	value_division(x,A11);
	MATRIX_ELEM(P,i,1+level) = value_uminus(x);
    }
}

/* void matrix_maj_line(Pmatrix A, matrice Q, int level):
 * Calcul de la matrice permettant de remplacer chaque terme de
 * la premiere ligne autre que le premier terme A11=A(level+1,level+1) par le reste de sa
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
void matrix_maj_line(A,Q,level)
Pmatrix A;
Pmatrix Q;
int level;
{
    Value A11;
    int j;
    Value x;
    int m= MATRIX_NB_COLUMNS(A);
    matrix_identity(Q,0);
    A11 =SUB_MATRIX_ELEM(A,1,1,level);
    for (j=2+level; j<=m; j++) {
	x = value_div(MATRIX_ELEM(A,1+level,j),A11);
	MATRIX_ELEM(Q,1+level,j) = value_uminus(x);
    }
}

/* void matrix_identity(Pmatrix ID, int level)
 * Construction d'une sous-matrice identity dans ID(level+1..n, level+1..n)
 *
 *  Les parametres de la fonction :
 *
 *!int 	ID[]	:  matrice
 * int  n	: nombre de lignes de la matrice
 * int  level   : niveau de la matrice i.e. numero de la premiere ligne
 *		     et de la premiere colonne a partir duquel on commence 
 *		     a prendre en compte les elements de la matrice
 */
void matrix_identity(ID,level)
Pmatrix ID;
int level;
{
    int i,j;
    int n = MATRIX_NB_LINES(ID);
    for(i = level+1; i <= n; i++) {
	for(j = level+1; j <= n; j++)
	    MATRIX_ELEM(ID,i,j) = VALUE_ZERO;
	MATRIX_ELEM(ID,i,i) = VALUE_ONE;
    }

    MATRIX_DENOMINATOR(ID) = VALUE_ONE;
}

/* bool matrix_identity_p(Pmatrix ID, int level)
 * test d'une sous-matrice dans ID(level+1..n, level+1..n) pour savoir si
 * c'est une matrice identity. Le test n'est correct que si la matrice
 * ID passee en argument est normalisee (cf. matrix_normalize())
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
bool matrix_identity_p(ID,level)
Pmatrix ID;
int level;
{
    int i,j;
    int n = MATRIX_NB_LINES(ID);
    for(i = level+1; i <= n; i++) {
	for(j = level+1; j <= n; j++) {
	    if(i==j) {
		if(value_notone_p(MATRIX_ELEM(ID,i,i)))
		    return(false);
	    }
	    else			/* i!=j */
		if(value_notzero_p(MATRIX_ELEM(ID,i,j)))
		    return(false);
	}
    }
    return(true);
}

/* int matrix_line_el(Pmatrix MAT, int level)
 * renvoie le numero de colonne absolu du premier element non nul
 * de la sous-ligne MAT(level+1,level+2..m);
 * renvoie 0 si la sous-ligne est vide.
 *
 */
/*ARGUSED*/
int matrix_line_el(MAT,level)
Pmatrix MAT;
int level;
{
    int j;
    int j_min = 0;
    int m = MATRIX_NB_COLUMNS(MAT);
    /* recherche du premier element non nul de la sous-ligne 
       MAT(level+1,level+2..m) */
    for(j = level+2; j<=m && (MATRIX_ELEM(MAT,1+level,j)==0) ; j++);
    if(j < m+1)
	j_min = j-1;
    return (j_min);
}

/* int matrix_col_el(Pmatrix MAT, int level)
 * renvoie le numero de ligne absolu du premier element non nul
 * de la sous-colonne MAT(level+2..n,level+1);
 * renvoie 0 si la sous-colonne est vide.
 *
 */
/*ARGSUSED*/
int matrix_col_el(MAT,level)
Pmatrix MAT;
int level;
{
    int i;
    int i_min=0;
    int n = MATRIX_NB_LINES(MAT); 
    for(i = level+2; i <= n && (MATRIX_ELEM(MAT,i,level+1)==0); i++);
    if (i<n+1)
	i_min = i-1;
    return (i_min);
}

/* void matrix_coeff_nnul(Pmatrix MAT, int * lg_nnul,
 *                     int * cl_nnul, int level)
 * renvoie les coordonnees du plus petit element non-nul de la premiere
 * sous-ligne non nulle 'lg_nnul' de la sous-matrice MAT(level+1..n, level+1..m)
 *
 *
 */
void matrix_coeff_nnul(MAT,lg_nnul,cl_nnul,level)
Pmatrix MAT;
int *lg_nnul,*cl_nnul;
int level;
{
    bool trouve = false;
    int j;
    Value min = VALUE_ZERO,val=VALUE_ZERO;
    int m = MATRIX_NB_COLUMNS(MAT);
    *lg_nnul = 0;
    *cl_nnul = 0;

    /* recherche de la premiere ligne non nulle de la 
       sous-matrice MAT(level+1 .. n,level+1 .. m)      */

    *lg_nnul= matrix_line_nnul(MAT,level);

    /* recherche du plus petit (en valeur absolue) element non nul de cette ligne */
    if (*lg_nnul) {
	for (j=1+level;j<=m && !trouve; j++) {
	    min = MATRIX_ELEM(MAT,*lg_nnul,j);
	    min = value_abs(min);
	    if (value_notzero_p(min)) {
		trouve = true;
		*cl_nnul=j;
	    }
	}
	for (j=1+level;j<=m && value_gt(min,VALUE_ONE); j++) {
	    val = MATRIX_ELEM(MAT,*lg_nnul,j);
	    val = value_abs(val);
	    if (value_notzero_p(val) && value_lt(val,min)) {
		min = val;
		*cl_nnul =j;
	    }
	}
    }
}



/* void ordinary_sub_matrix(Pmatrix A, A_sub, int i1, i2, j1, j2)
 * input    : a initialized matrix A, an uninitialized matrix A_sub,
 *            which dimensions are i2-i1+1 and j2-j1+1.
 * output   : nothing.
 * modifies : A_sub is initialized with the elements of A which coordinates range
 *            within [i1,i2] and [j1,j2].
 * comment  : A_sub must be already allocated.
 */
void ordinary_sub_matrix(A, A_sub, i1, i2, j1, j2)
Pmatrix A, A_sub;
int i1, i2, j1, j2;
{
    int i, j, i_sub, j_sub;

    for(i = i1, i_sub = 1; i <= i2; i++, i_sub ++)
	for(j = j1, j_sub = 1; j <= j2; j++, j_sub ++)
	    MATRIX_ELEM(A_sub,i_sub,j_sub) = MATRIX_ELEM(A,i,j);
    
}

/* void insert_sub_matrix(A, A_sub, i1, i2, j1, j2)
 * input: matrix A and smaller A_sub
 * output: nothing
 * modifies: A_sub is inserted in A at the specified position
 * comment: A must be pre-allocated
 */
void insert_sub_matrix(
    Pmatrix A,
    Pmatrix A_sub,
    int i1, int i2,
    int j1, int j2)
{
    int i,j,i_sub,j_sub;

    assert(i1>=1 && j1>=1 && 
	   i2<=MATRIX_NB_LINES(A) && j2<=MATRIX_NB_COLUMNS(A) &&
	   i2-i1<MATRIX_NB_LINES(A_sub) && j2-j1<MATRIX_NB_COLUMNS(A_sub));

    for (i = i1, i_sub = 1; i <= i2; i++, i_sub ++)
	for(j = j1, j_sub = 1; j <= j2; j++, j_sub ++)
	    MATRIX_ELEM(A,i,j) = MATRIX_ELEM(A_sub,i_sub,j_sub) ;
}

/* that is all
 */
