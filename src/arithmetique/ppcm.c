/* package arithmetique
 * 
 * $RCSfile: ppcm.c,v $ (version $Revision$)
 * $Date: 1996/07/16 22:04:39 $, 
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
    if (VALUE_NEG_P(i)) i = value_uminus(i);
    if (VALUE_NEG_P(j)) j = value_uminus(j);

    if (VALUE_ZERO_P(i) || VALUE_ZERO_P(j)) 
	return VALUE_ZERO;
    else {
	Value d = pgcd(i,j);
	return value_mult(value_div(i,d),j);
    }
}

/* end of $RCSfile: ppcm.c,v $
 */
