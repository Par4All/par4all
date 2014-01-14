/* Excerpt from struct12.c, init3() only is preserved.
 *
 * FI: I do not know what was the intent of the person who wrote this
 * piece of code but init3(), unfortunately, is only a memory
 * leaker. Pointers like other scalars are passed by copy. Their
 * modifications are ignored by the callee.
 *
 * This piece of code is only useful to check that memory leak
 * detection is working.
 */

#include <stdlib.h>
#include <stdio.h>

#define N 5

typedef struct {
  int num;
  int tab1[N];
  int *tab2;
} mys;

void init3(mys **p)
{
  int i;
  p = malloc(N*sizeof(mys *));
  for (i=0; i<N; i++) 
    {
      p[i] = malloc(sizeof(mys));
      p[i]->tab2 = malloc(N*sizeof(int));
    }
  
  p[0]->num=11;
  p[0]->tab1[0]=20;
  p[0]->tab2[0]=30;
  
  for (i=1; i<N; i++) 
    {
      p[i]->num = 10;
      p[i]->tab1[0] = 21;
      p[i]->tab2[0]= p[i]->tab1[0];
    }  
}

int main() 
{
  mys s2, s3;
  mys *q=&s2, **r= &q;

  q = &s3;
  init3(r);

  return 1;
}
