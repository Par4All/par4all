
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
  for(i = 0; i<10; i++)
    a.p[i] = i;

/* definition of b */
  b = a;
  for(i = 0; i<10; i++)
    b.p[i] = i*10;

/* definition of e */
  for(i = 0; i<5; i++)
    {
      e.champ[i].p = malloc(10*sizeof(int));
      for(j = 0; j<10; j++)
	e.champ[i].p[j] = i+j;
    }

  /* definition of f, also modifies e */
  f = e;
  for(i = 0; i<5; i++)
    for(j = 0; j<10; j++)
      f.champ[i].p[j] = i*j;

  return 0;
}
