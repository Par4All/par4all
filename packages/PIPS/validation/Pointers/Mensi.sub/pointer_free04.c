/* Example for Section 1.2.1, , chapitre affectation d'un pointeur de
 * la valeur retrounée par une fonction interprocedural.
 */

#include<stdlib.h>

typedef int * pointer;

pointer pointer_free(pointer fp)
{
  free(fp);
  return fp;
}

int main(void)
{
  pointer p1, p2;
  p1 = malloc(sizeof(int));
  p2 = pointer_free(p1);

  return;
}
