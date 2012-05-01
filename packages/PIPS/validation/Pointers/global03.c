/* #include<stdio.h> */

typedef struct two_fields{int one; int two[10];} tf_t;

int i[10];
tf_t s;
int *pi = &i[0];
tf_t *q = &s;

void call03()
{
  // To avoid a problem with the semantics of the empty points-to set
  // The solution might be to add always an arc ANYWHERE->ANYWHERE
  // when entering a module statement
  int * p = i;
  *pi = 1;
  pi++;
  q->one = 1;
  q->two[4] = 2;
}

main()
{
  call03();
}
