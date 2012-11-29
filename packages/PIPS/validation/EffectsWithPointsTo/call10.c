/* #include<stdio.h> */

/* To obtain two read effects on a and b and a warning abount inneffective update of i invall02 */

typedef struct two_fields{int one; int two[10];} tf_t;

void call10(int i, int x[10], int j, int y[10])
{
  int k, l;

  for(k=0; k<i;k++)
    x[i] = 0;
  for(l=0; l<j;l++)
    y[i] = 0;
}

void bar(int i, int x[10], int j, int y[10])
{
  int k, l;

  for(k=0; k<i;k++)
    x[i] = 0;
  for(l=0; l<j;l++)
    y[i] = 0;
}

int main()
{
  tf_t s;
  tf_t *sp = &s;
  int a[10];

  call10(s.one, s.two, sp->one, sp->two);
  bar(10, a, 10, a);

  return 0;
}
