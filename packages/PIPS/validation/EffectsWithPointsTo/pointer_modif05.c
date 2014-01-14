/* Struct variable are passed by value...
 *
 * Function init_s does not work because s is passed by copy. The
 * initializations are lost on return.
 *
 * Several bugs shown here linked to the field and subscript
 * operations. This piece of code should core dump in fprintf() and in
 * the call to compute_s() because s.tab is not initialized.
 */

#include <stdio.h>
#include <malloc.h>

typedef struct {int max; float *tab;} s_t;

void init_s(s_t *ps, int max)
{

  ps->tab = (float *) malloc(max * sizeof(float));
  ps->max = max;
  fprintf(stderr, "In init_s, s.tab=%p\n", ps->tab);
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
  int i;

  init_s(&s, 10);
  fprintf(stderr, "In main, s.tab=%p\n", s.tab);
  
  // FI: it would be nicer to use &s, but this should work...
  compute_s(s, 10);

  for (i=0; i<10; i++)
    fprintf(stderr, "In main, s.tab[i]=%f\n", s.tab[i]);

  return 0;
}
