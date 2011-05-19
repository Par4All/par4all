/*  Swimming pool example taken from Fribourg et Olsen, "proving
   safety properties of inifinite state systems by compilation into
   presburger arithmetic", but initially defined by M. Latteux

   It's a Petri net
*/

#include <stdlib.h>
#include <stdio.h>

void swimming01(int ncabines, int nbaskets)
{
  int x1, x2, x3, x4, x5, x6, x7;
  x1 = x2 = x3= x4 = x5 = 0;
  if(ncabines < 1 || nbaskets < 1) exit(0);
  x6 = ncabines;
  x7 = nbaskets;

  while(1) {
    /* r1 */
    if(x6>0) x1++, x6--;
    /*r2*/
    if(x1>0&&x7>0) x1--, x2++, x7--;
    /*r3*/
    if(x2>0) x2--, x3++, x6++;
    /*r4*/
    if(x3>0&&x6>0) x3--, x4++, x6--;
    /*r5*/
    if(x4>0) x4--, x5++, x7++;
    /*r6*/
    if(x5>0) x5--, x6++;
    printf("x1 = %d, x1 = %d, x1 = %d, x1 = %d, x1 = %d, x1 = %d, x1 = %d\n",
	   x1, x2, x3, x4, x5, x6, x7);
  }
}
