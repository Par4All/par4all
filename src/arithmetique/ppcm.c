/* package arithmetique
 * 
 * $Id$
 */

/*LINTLIBRARY*/

#include <stdio.h>

#include "arithmetique.h"
#include "assert.h"

/* int ppcm(int i, int j): plus petit entier positif divisible par i et j
 *
 * Ancien nom et ancien type: void lcm(int i, int j, int *pk)
 */
Value ppcm(Value i, Value j)
{
    if (value_neg_p(i)) i = value_uminus(i);
    if (value_neg_p(j)) j = value_uminus(j);

    if (value_zero_p(i) || value_zero_p(j)) 
	return VALUE_ZERO;
    else {
	Value d = pgcd(i,j);
	d = value_div(i,d);
	return value_mult(d,j);
    }
}
