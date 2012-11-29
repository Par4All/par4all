/* To obtain two read effects on sp and sp.one and a warning abount
 *  ineffective update of i in call09
 *
 * Bug: normalization issue for s.one and sp->one because of the
 * implicit array assumption for pointers.
 */

/* #include<stdio.h> */

typedef struct two_fields{int one; int two[10];} tf_t;

void call09(int i, int j)
{
  i = i + j;
}

main()
{
  tf_t s;
  tf_t *sp = &s;

  call09(s.one, sp->one);
}
