#include <malloc.h>

typedef struct {int max; float *tab;} s_t;

void init_s(s_t *s, int max)
{

  s->tab = (float *) malloc(max * sizeof(float));
  s->max = max;
  
}

void compute_s(s_t *s, int max)
{
  int i;

  for (i=0; i<max; i++)
    s->tab[i] = i*2.0;
  
}


int main()
{
  s_t s;
  int j;

  init_s(&s, 10);
  
  compute_s(&s, 10);
  

  return 0;
}
