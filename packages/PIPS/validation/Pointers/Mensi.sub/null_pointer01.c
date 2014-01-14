/* I replace boolean variable by a pointer (see Semantics/boolean12.c)
   We need to improve transformer_add_any_relation_information() so it be able
   to handle pointers
*/
#include <stdio.h>

int main(void)
{
  int i = 1, j, k, *p = NULL;

  if(p==NULL)
    i = 2;

  if(p==NULL) {
    i = 0;
    j = 5;
    k = 3*j;
  }

  printf("i=%d, j=%d, k=%d\n", i, j, k);

  return i+j+k;
}
