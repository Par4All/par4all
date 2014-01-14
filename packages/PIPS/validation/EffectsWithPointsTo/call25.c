/* To debug issues linked to a NULL pointer
 *
 * This code is not correct. The error is not detected by the
 * points-to analysis, but by the proper effect analysis
 */

#include <stdio.h>

void call25(int * q)
{
  *q=3;
}

int main()
{
  int i, *ip = NULL;

  call25(ip);
  return 0;
}
