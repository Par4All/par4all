/* Example for Section 1.2.5, liberation d'une zone memoire, chapitre
 * interprocedural. 
 *
 * Modified version: the formal parameter p is written by pointer_free
 * after free is called. Function pointer_free does free the bucket
 * pointed by p and generates a memory leak.
 */

#include<stdlib.h>

typedef int * pointer;

void pointer_free(pointer p)
{
  free(p);
  p = malloc(sizeof(int));
  return;
}

int main(void)
{
  pointer p1, p2;
  p1 = malloc(sizeof(int));
  p2 = p1;
  pointer_free(p1);

  return;
}
