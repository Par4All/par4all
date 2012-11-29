/* Excerpt from struct12.c, init4() only is preserved
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

void init4(mys *p[N])
{
  int i;
  for (i=0; i<N; i++) {
    p[i] = malloc(sizeof(mys));
    p[i]->tab2 = malloc(N*sizeof(int));
  }
  
  p[0]->num=11;
  p[0]->tab1[0]=20;
  p[0]->tab2[0]=30;
  
  for (i=1; i<N; i++) {
    p[i]->num = 10;
    p[i]->tab1[0] = 21;
    p[i]->tab2[0]= p[i]->tab1[0];
  } 
}

int main() 
{
  mys s4[N];
  mys *s[N];
  int i;

  for(i=0;i<N;i++)
    s[i] = &s4[i];

  init4(s);

  return 1;
}
