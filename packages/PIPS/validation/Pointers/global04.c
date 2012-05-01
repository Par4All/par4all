/* Building points-to on demand for global variables */

#include<stdio.h>

int **p;

int main()
{
  // To avoid a problem with the semantics of the empty points-to set
  // The solution might be to add always an arc ANYWHERE->ANYWHERE
  // when entering a module statement
  int * q = NULL;

  int i;
  i = 1;
  *p = &i;

  return 0;
}
