/* Building points-to in demand for global vraibles */
#include<stdio.h>
int **p;

int main()
{
  int i;
  i = 1;
  *p = &i;

  return 0;
}
