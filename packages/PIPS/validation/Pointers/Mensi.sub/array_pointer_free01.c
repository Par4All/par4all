/* Example for Section 1.2.5, liberation d'une zone memoire, chapitre
 * interprocedural. 
 */

#include<stdlib.h>

typedef int * pointer;

void pointer_free(int i, pointer fp[])
{
  free(fp[i]);
  return;
}

void pointer_malloc(int i, pointer fp[])
{
  fp[i] = (pointer) malloc(sizeof(int));
  return;
}

int main(void)
{
  pointer p[10];
  int i = 4;

  pointer_malloc(i, &p[0]);
  pointer_free(i, p);

  return 0;
}
