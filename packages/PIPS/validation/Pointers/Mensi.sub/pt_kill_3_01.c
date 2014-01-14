/* Illustration du calcul de pt_kill_3
 *
 * Contrieved example
 */

#include <stdlib.h>

void my_malloc(int **p)
{
  *p = (int *) malloc(sizeof(int));
  return;
}

int main()
{
  int i = 1, j = 2, *pi = &i, *pj = &j, **pp;

  pp = (i>j) ? &pi : &pj;
  my_malloc(pp);

  return 0;
}
