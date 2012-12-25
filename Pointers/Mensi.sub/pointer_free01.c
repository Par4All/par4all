/* Example for Section 1.2.5, liberation d'une zone memoire, chapitre
 * interprocedural. 
 */

#include<stdlib.h>

typedef int * pointer;

void pointer_free(pointer p)
{
  free(p);
  return;
}

int main(void)
{
  pointer p1, p2;
  p1 = malloc(sizeof(int));
  p2 = p1;
  pointer_free(p1);

  return 0;
}
