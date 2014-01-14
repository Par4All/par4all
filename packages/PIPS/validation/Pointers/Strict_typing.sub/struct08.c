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

/* Same as previous function, but the parameter is copied first and
   the copy is used in the function body. */
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

/* Useless use of a copy: all the work is lost on return */
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

  /* Since p is not initialized, the first call is useless and init()
     core dumps. */
  init(p);
  init2(q);
  init3(r);
  init4(s);

  return 1;
}
