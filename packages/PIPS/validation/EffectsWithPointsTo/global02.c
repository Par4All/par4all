/* Building points-to in demand for global vraibles */
#include<stdio.h>
int **p;

int main()
{
  int **q, *r, i;

  q = p;
  *q = &i;
  r = &**q;

  return 0;
}
