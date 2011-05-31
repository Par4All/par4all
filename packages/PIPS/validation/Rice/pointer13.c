#include<stdlib.h>
void pointer05(double *a, double *b,double *c, double *d, int *cnt)
{
  double *w = NULL;  /* w is the workspace */
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
