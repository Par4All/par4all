// FI: variable d seems useless

// Hard to analyze because the points-to analysis cannot prove that w
// is always initialized or the loop is mnot executed...

// Bug in initial_precondition, probably because of an unexpected
// abstract location in effects...

#include <stdlib.h>
#include <assert.h>

void pointer13(double *a, double *b, double *c, double *d, int *cnt)
{
  double *w = NULL;  /* w is the workspace */
  assert(a&&b&&c&&cnt);
  int sz = 0;
  int i;
 
  if ( sz < *cnt ) {
    if (w) free(w);
    w = (double *) malloc(*cnt * sizeof(double));
    sz = *cnt;
  }
  for (i = 0; i < *cnt; i++)
    w[i]  = c[i] + a[i] * b[i];
}
