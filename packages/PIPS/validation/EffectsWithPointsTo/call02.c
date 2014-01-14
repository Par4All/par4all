/* #include<stdio.h> */

/* To obtain two read effects on a and b and a warning abount
 * ineffective update of i in call02
 */

typedef struct two_fields{int one; int two[10];} tf_t;

void call02(int i, int j, int y[10], int * q[10], tf_t *p)
{
  /* i can be modified locally, but it won't show in the summary
     effects... which creates a problem for transformer and
     precondition computation. */
  i = j + 1;
  y[i] = 0;
  p->one = 1;
  p->two[j] = 2.;
  *q[i]=3;
  return;
}

int main()
{
  int a = 1;
  int b = 2;
  int x[10], aa[10], i;
  int * ap[10];
  tf_t s;
  tf_t *sp = &s;

  /* Initialization added to avoid a segfault in the callee */
  for(i=0;i<10;i++)
    ap[i] = &aa[i];

  call02(a, b, x, ap, sp);
  return 0;
}
