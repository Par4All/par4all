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

/* package matrice
 *
 * Neil Butler, Corinne Ancourt, Francois Irigoin, Yi-qing Yang
 *
 */

#if defined(__GNUC__)
// Comment this warning since no one has fixed this issue since the 90's...
//#warning matrice is obsolete, use matrix instead.
#endif

/* Les matrices sont des matrices pleines, a coefficients rationnels.
 *
 * Les matrices sont representes par des tableaux d'entiers mono-dimensionnels
 * dont l'element 0 represente le denominateur global. Elles sont
 * stockees par colonne ("column-major"), comme en Fortran. Les indices
 * commencent a 1, toujours comme en Fortran et non comme en C.
 * Les deux dimensions sont implicites et doivent etre passees en
 * parametre avec la matrice dans les appels de procedures.
 *
 * Le denominateur doit etre strictement positif, i.e. plus grand ou egal
 * a un. Un denominateur nul n'aurait pas de sens. Un denominateur negatif
 * doublerait le nombre de representations possibles d'une matrice.
 *
 * Les matrices renvoyees par certaines routines, comme matrice_multiply(),
 * ne sont pas normalisees par le pgcd des coefficients et du denominateur
 * pour des raisons d'efficacite. Mais les routines de test, comme
 * matrice_identite_p(), supposent leurs arguments normalises.
 *
 * Il faudrait sans doute introduire deux niveaux de procedure, un niveau
 * externe sur garantissant la normalisation, et un niveau interne efficace
 * sans normalisation automatique.
 *
 * La bibliotheque utilise une notion de sous-matrice, definie systematiquement
 * par un parametre appele "level". Seuls les elements dont les indices
 * de lignes et de colonnes sont superieurs a level+1 (ou level? FI->CA)
 * sont pris en consideration. Il s'agit donc de sous-matrice dont le
 * premier element se trouve sur la diagonale de la matrice complete et
 * dont le dernier element et le dernier element de la matrice complete.
 *
 * Note: en cas detection d'incoherence, Neil Butler renvoyait un code
 * d'erreur particulier; Francois Irigoin a supprime ces codes de retour
 * et a traite les exceptions par des appels a assert(), et indirectement
 * a abort()
 */

typedef Value * matrice;

#define MATRICE_UNDEFINED ((matrice) NULL)
#define MATRICE_NULLE ((matrice) NULL)

/* Allocation et desallocation d'une matrice */
#define matrice_new(n,m) ((matrice) malloc(sizeof(Value)*((n*m)+1)))
#define matrice_free(m) (free((char *) (m)))

/* Macros d'acces aux elements d'une matrice */

/* int ACCESS(int * matrix, int column, int i, int j): acces a l'element (i,j)
 * de la matrice matrix, dont la taille des colonnes, i.e. le nombre de 
 * lignes, est column.
 */
#define ACCESS(matrix,column,i,j) ((matrix)[(((j)-1)*(column))+(i)])

/* int DENOMINATEUR(matrix): acces au denominateur global d'une matrice matrix
 * La combinaison *(&()) est necessaire pour avoir des parentheses autour
 * de matrix[0] et pour pouvoir simultanement utiliser cette macro comme lhs.
 */
/* #define DENOMINATOR(matrix) *(&((matrix)[0])) */
#define DENOMINATOR(matrix) ((matrix)[0])

/* bool matrice_triangulaire_inferieure_p(matrice a, int n, int m):
 * test de triangularite de la matrice a
 */
#define matrice_triangulaire_inferieure_p(a,n,m) \
    matrice_triangulaire_p(a,n,m,true)

/* bool matrice_triangulaire_superieure_p(matrice a, int n, int m):
 * test de triangularite de la matrice a
 */
#define matrice_triangulaire_superieure_p(a,n,m) \
    matrice_triangulaire_p(a,n,m,false)

/* FI: Corinne, peux-tu expliquer la raison d'etre de cette macro? */
/* d'apres ce que je comprends de la suite, ca permet de definir
 * des sous-matrices dont le coin superieur droit (i.e. le premier
 * element) se trouve sur la diagonal?
 */
#define ACC_ELEM(matrix,column,i,j,level) \
    (matrix[((j)-1+(level))*(column) + (i) + (level)])

/* FI: Corinne, pourrais-tu nous eclairer sur la signification de ces
 * constantes?
 */
#define VALIDATION 0
/* FI #define NULL 0 */
#define MATRIX 0

/* definition temporaire d'une vraie fonction pour dbx */
/* #define matrice_print(a,n,m) matrice_fprint(stdout,a,n,m) */
