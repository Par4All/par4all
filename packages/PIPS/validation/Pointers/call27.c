/* Test of intraprocedural points-to
 *
 * How much of the points-to graph is assumed broken by the function call?
 */

#include <stdio.h>

void call27(int ***ppp)
{
  *ppp = NULL;
  return;
}

int main()
{
  int i;
  int * p = &i;
  int ** pp = &p;
  int *** ppp = &pp;
  call27(ppp);
  return 0;
}
