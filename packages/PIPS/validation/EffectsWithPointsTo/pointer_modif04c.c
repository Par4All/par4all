/* Struct variable are passed by value...
 *
 * Function init_s does not work because s is passed by copy. The
 * initializations are lost on return.
 *
 * Several bugs shown here linked to the field and subscript
 * operations. This piece of code should core dump in fprintf() and in
 * the call to compute_s() because s.tab is not initialized.
 *
 * Here we check the bug linked to the illegal call to compute_s(),
 * which should core dump.
 */

#include <stdio.h>

typedef struct {int max; float *tab;} s_t;

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
  
  compute_s(s, 10);

  return 0;
}
