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

/* package arithmetique 
 */

/*LINTLIBRARY*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>

#include "assert.h"
#include "arithmetique.h"

/* int divide(int a, int b): calcul du divide de a par b;
 * le reste (qui n'est pas retourne) est toujours positif; il
 * est fourni par la fonction modulo()
 *
 * Il y a quatre configuration de signe a traiter:
 *  1. a>0 && b>0: a / b
 *  2. a<0 && b>0: (a-b+1) / b
 *  3. a>0 && b<0: cf. 1. apres changement de signe de b, puis changement 
 *     de signe du resultat
 *  4. a<0 && b<0: cf. 2. apres changement de signe de b, puis changement
 *     de signe du resultat
 *  5. a==0: 0
 */
Value divide_fast(Value a, Value b)
{
    /* definition d'une look-up table pour les valeurs de a appartenant
       a [-DIVIDE_MAX_A..DIVIDE_MAX_A] et pour les valeurs de b
       appartenant a [1..DIVIDE_MAX_B] (en fait [-DIVIDE_MAX_B..DIVIDE_MAX_B]
       a cause du changement de signe)
       
       Serait-il utile d'ajouter une test b==1 pour supprimer une colonne?

       Serait-il utile de tester b > a pour renvoyer 0 ou -1 tout de suite?
       */

#define DIVIDE_MAX_A 7
#define DIVIDE_MAX_B 8

    static Value
	divide_look_up[2*DIVIDE_MAX_A+1][DIVIDE_MAX_B]={
	/* b ==         1   2   3   4   5   6   7   8 */
	{/* a == - 7 */ -7, -4, -3, -2, -2, -2, -1, -1},
	{/* a == - 6 */ -6, -3, -2, -2, -2, -1, -1, -1},
	{/* a == - 5 */ -5, -3, -2, -2, -1, -1, -1, -1},
        {/* a == - 4 */ -4, -2, -2, -2, -1, -1, -1, -1},
        {/* a == - 3 */ -3, -2, -1, -1, -1, -1, -1, -1},
        {/* a == - 2 */ -2, -1, -1, -1, -1, -1, -1, -1},
        {/* a == - 1 */ -1, -1, -1, -1, -1, -1, -1, -1},
        {/* a ==   0 */  0,  0,  0,  0,  0,  0,  0,  0},
        {/* a ==   1 */  1,  0,  0,  0,  0,  0,  0,  0},
        {/* a ==   2 */  2,  1,  0,  0,  0,  0,  0,  0},
        {/* a ==   3 */  3,  1,  1,  0,  0,  0,  0,  0},
        {/* a ==   4 */  4,  2,  1,  1,  0,  0,  0,  0},
        {/* a ==   5 */  5,  2,  1,  1,  1,  0,  0,  0},
	{/* a ==   6 */  6,  3,  2,  1,  1,  1,  0,  0},
	{/* a ==   7 */  7,  3,  2,  1,  1,  1,  1,  0}
    };
    /* translation de a pour acces a la look-up table par indice positif:
       la == a + DIVIDE_MAX_A >= 0 */

    Value quotient;     /* valeur du quotient C */

    assert(value_notzero_p(b));

    /* serait-il utile d'optimiser la division de a=0 par b? Ou bien
       cette routine n'est-elle jamais appelee avec a=0 par le package vecteur?
       */

    if (value_le(a, int_to_value(DIVIDE_MAX_A)) && 
	value_ge(a, int_to_value(-DIVIDE_MAX_A)) &&
	value_le(b, int_to_value(DIVIDE_MAX_B)) &&
	value_ge(b, int_to_value(-DIVIDE_MAX_B)))
    {
	/* direct table look up */
	int bint = VALUE_TO_INT(b),
	    la = VALUE_TO_INT(a)+DIVIDE_MAX_A; /* shift a for the table */
	quotient = (bint>0)?
	    divide_look_up[la][bint-1]:
		value_uminus(divide_look_up[la][(-bint)-1]);
    }
    else 
	quotient = value_pdiv(a,b); /* this is just divide_slow */

    return quotient;
}

Value divide_slow(Value a, Value b)
{
    return value_pdiv(a, b);
}
