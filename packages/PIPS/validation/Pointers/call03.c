/* #include<stdio.h> */

typedef struct two_fields{int one; int two[10];} tf_t;

int i;
tf_t s;
int * pi = &i;
tf_t *q = &s;

void call03()
{
  *pi = 1;
  pi++;
  q->one = 1;
  q->two[4] = 2;
}

int main()
{
  call03();
  return 0;
}
