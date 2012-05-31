/* #include<stdio.h> */

typedef struct two_fields{int one; int two[10];} tf_t;

void call15(int * pi, tf_t *q)
{
  *pi = 1;
  q->one = 1;
}

int main()
{
  int i;
  tf_t s;

  call15(&i, &s);
  return 0;
}
