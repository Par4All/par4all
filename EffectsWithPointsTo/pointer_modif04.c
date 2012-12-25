// Struct variable are passed by value...

// Function init_s does not work because s is passed by copy. The
// initializations are lots on return.

#include <stdio.h>
#include <malloc.h>

typedef struct {int max; float *tab;} s_t;

void init_s(s_t s, int max)
{

  s.tab = (float *) malloc(max * sizeof(float));
  s.max = max;
  fprintf(stderr, "In init_s, s.tab=%p\n", s.tab);
  return;
}

void compute_s(s_t s, int max)
{
  int i;

  for (i=0; i<max; i++)
    s.tab[i] = i*2.0;
  
  return;
}

int main()
{
  s_t s;
  int j;

  init_s(s, 10);
  fprintf(stderr, "In main, s.tab=%p\n", s.tab);
  
  compute_s(s, 10);

  return 0;
}
