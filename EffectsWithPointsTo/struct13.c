/* Excerpt from struct12.c, only init() is preserved
 *
 */

#include <stdlib.h>
#include <stdio.h>

#define N 5

typedef struct {
  int num;
  int tab1[N];
  int *tab2;
} mys;

void init(mys *m) 
{
  int i;
  m->num = N;
  m->tab2 = malloc(N*sizeof(int));
  m->tab1[0] = 10;
  m->tab2[0] = 20;
  for (i=0; i<N; i++)
    { 
      m->tab1[i] = 1;
      m->tab2[i] = m->tab1[i];
    }
  return;
}

int main() 
{
  mys s1;
  mys *p=&s1;

  init(p);

  return 1;
}
