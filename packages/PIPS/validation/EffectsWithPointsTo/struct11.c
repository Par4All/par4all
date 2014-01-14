#include <stdlib.h>
#include <stdio.h>

#define N 5

typedef struct {
  int num;
  int tab1[N];
  int *tab2;
} mys;

void init2(mys *n)
{
  int i;
  mys m;

  m = *n;
  m.num = N;
  m.tab2 = malloc(N*sizeof(int));
  m.tab1[0] = 10;
  m.tab2[0] = 20;
  for (i=0; i<N; i++)
    {
      m.tab1[i] = 1;
      m.tab2[i] = m.tab1[i];
    }
  
}

int main()
{
  mys *q;

  init2(q);
 
  return 1;
}
