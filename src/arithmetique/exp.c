 /* package arithmetique 
  *
  * $RCSfile: exp.c,v $ (version $Revision$)
  * $Date: 1996/07/17 10:56:52 $, 
  */

/*LINTLIBRARY*/
/* no overflow is checked 
 */

#include <stdio.h>

#include "arithmetique.h"
#include "assert.h"

/* int exponentiate(x,n):  raise x to the power n
 * 
 * Precondition: 	n => 0
 */
Value exponentiate(Value x, int n)
{
    Value y;

    /* validation - n is positive 
     */
    assert(n >= 0);
    if (n == 0) return VALUE_ONE;

    /* FI: la complexite pourrait etre reduite de O(n) a O(log n) 
     */
    for(y=VALUE_ONE; n>0; n--)
	value_prod(y,x);

    return y;
}

/* end of $RCSfile: exp.c,v $
 */
