 /* package arithmetique */

/*LINTLIBRARY*/

#include <stdio.h>

#include "arithmetique.h"
#include "assert.h"

/* int ppcm(int i, int j): plus petit entier positif divisible par i et j
 *
 * Ancien nom et ancien type: void lcm(int i, int j, int *pk)
 */
int ppcm(i,j)
int	i;			/* input */
int	j;			/* input */
{
    int	d;
	
    if (i < 0) i = -i;
    if (j < 0) j = -j;

    if ((i == 0) || (j == 0)) return(0);

    else {
	d = pgcd(i,j);
	return((i/d)*j);
    }
}
