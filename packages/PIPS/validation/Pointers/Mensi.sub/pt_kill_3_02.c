/* Illustration du calcul de pt_kill_3
 *
 * Contrieved example
 */

#include <stdlib.h>

void my_malloc(int c, int **fpp)
{
  if(c)
    *fpp = (int *) malloc(sizeof(int));
  return;
}

int main()
{
  int i = 1, *pi = &i, **pp= &pi;

  my_malloc(i, pp);

  return 0;
}
