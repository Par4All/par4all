/* #include<stdio.h> */

/* To obtain two read effects on a and b and a warning abount inneffective update of i invall02 */

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
