/*****************************************************************************
 * V - MALLOC
 ****************************************************************************/
#include <stdlib.h>
int *dependence05() {
  int *a;
  int *b;
  int *c;

  b = (int *) malloc( 10 * sizeof(int) );
  c = (int *) malloc( 10 * sizeof(int) );

  a = b; // a and b are aliased, a points_to b[0]
  *a = 0; // write effect on *a may be visible from caller, because b escape

  a = c; // a and b are aliased, a points_to b[0]
  *a = 0; // write effect won't be visible from caller, because c doesn't escape

  free( c );

  return b;
}
