/* The code is buggy because the pointers declared in the main
 * function are not initialized because they are used as actual
 * parameters.
 *
 * The bug is detected by the proper effects pass.
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
  
}

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

void init4(mys *p[N])
{
  int i;
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
  mys *p, *q, **r, *s[N];

  init(p);
  init2(q);
  init3(r);
  init4(s);

  return 1;
}
