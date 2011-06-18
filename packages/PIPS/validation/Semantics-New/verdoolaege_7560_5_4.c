/* Sven Verdoolaege, Albert Cohen, Anna Beletska
 *
 * INRIA Tech. Report 7560
 *
 * Example 5.4
 */

#include <assert.h>
#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void verdoolaege_7560_5_4()
{
  int x, y, n;

  assert(n>=2);

  if(1) {
    do {
      x++, y+= 1-n;
    } while(flip());
  }
}
