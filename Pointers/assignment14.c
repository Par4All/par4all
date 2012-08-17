/* Bug when typing stubs and points-to arcs
 *
 * Fields a, b and c have been added to reduce ambiguities
 *
 * For the same reason, variable t1 and t2 have been renamed tx and ty
*/

#include <stdio.h>
#include <stdlib.h>

struct foo {int a; int b; int c; int * ip1; int * ip2;} ;

void assignment14(struct foo** tx, struct foo** ty)
{
   (**tx).ip1 =(**ty).ip2;
   return;
} 
