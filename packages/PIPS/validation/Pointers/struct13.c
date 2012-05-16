/* struct assignment */

#include<stdlib.h>

typedef struct{
  int *p;
} s_with_pointer;

typedef struct{
  s_with_pointer champ[5];
} s_with_array_of_struct_with_pointer;

int main()
{
  s_with_pointer a, b;
  s_with_array_of_struct_with_pointer e, f;
  int i, j;

  /* definition of a */
  a.p = (int*) malloc(10*sizeof(int));

  /* definition of b */
  b = a;

  /* definition of e */
  for(i = 0; i<5; i++)
      e.champ[i].p = malloc(10*sizeof(int));

  /* definition of f, also modifies e */
  f = e;

  return 0;
}
