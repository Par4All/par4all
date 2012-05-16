/* Derived from global03.c to understand a bug with q->one */

/* #include<stdio.h> */

typedef struct two_fields{int one; int two[10];} tf_t;

tf_t s;
tf_t *q = &s;
int i;

void global08()
{
  // To avoid a problem with the semantics of the empty points-to set
  // The solution might be to add always an arc ANYWHERE->ANYWHERE
  // when entering a module statement
  int * p = &i;
  q->one = 1;
  return;
}
