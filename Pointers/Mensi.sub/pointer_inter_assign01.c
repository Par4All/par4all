/* Example for Section 1.2.1, , chapitre affectation d'un pointeur de
 * la valeur retrounée par une fonction interprocedural.
 */

#include<stdlib.h>

typedef int * pointer;

pointer pointer_assign01(pointer fp)
{
  return fp;
}

int main(void)
{
  int i = 1;
  pointer p1 = &i, p2;
  p2 = pointer_assign01(p1);

  return;
}
