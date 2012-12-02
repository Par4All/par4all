/* Test of intraprocedural points-to
 *
 * How much of the points-to graph is assumed broken by the function call?
 */

#include <stdio.h>

void call26(int ***ppp)
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
  call26(ppp);
  return 0;
}
