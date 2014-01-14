/* Struct variable are passed by value...
 *
 * Function init_s does not work because s is passed by copy. The
 * initializations are lost on return.
 *
 * Several bugs shown here linked to the field and subscript
 * operations. This piece of code should core dump in fprintf() and in
 * the call to compute_s() because s.tab is not initialized.
 *
 * Here is the issue for fprintf()... but it is not a standard issue
 * since s.tab is not dereference. It is only an issue wrt the C
 * standard that prohibits usage of undefined values.
 */

#include <stdio.h>

typedef struct {int max; float *tab;} s_t;

int main()
{
  s_t s;
  int j;

  fprintf(stderr, "In main, s.tab=%p\n", s.tab);

  return 0;
}
