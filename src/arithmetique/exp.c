 /* package arithmetique */

/*LINTLIBRARY*/

#include <stdio.h>

#include "arithmetique.h"
#include "assert.h"

/* int exponentiate(x,n):  raise x to the power n
 * 
 * Precondition: 	n => 0
 */
int exponentiate(x,n)
int	x;
int	n;
{
    int	loop;
    int	y;

    /* validation - n is positive */
    assert(n >= 0);

    if (n == 0) return(1);

    y = 1;
    /* FI: la complexite pourrait etre reduite de O(n) a O(log n) */
    for(loop=0; loop<n; loop++) 
	y = y*x;

    return(y);
}
