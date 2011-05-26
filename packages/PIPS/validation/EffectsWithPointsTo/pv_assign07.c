// assignments which kill aliased paths 

//#include <stdio.h>

typedef struct {int n; int *a; } mystruct;
int main()
{
  mystruct s1, s2, *s1p, *s2p;
  int i = 1;
  int j = 2;
  int k = 3;

  s1p = &s1;
  s2p = &s1;

  s1.a = &i;
  s2.a = &j;

  s1p->a = s2.a;
  s2p->a = &k;
  
  s2p = &s2;


  //  printf("s1.a = %d, s1p->a = %d, s2.a = %d, s2p->a = %d\n", *s1.a, *s1p->a, *s2.a , *s2p->a);
  return(0);
}
