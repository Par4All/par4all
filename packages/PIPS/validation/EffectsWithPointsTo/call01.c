/* #include<stdio.h> */

typedef struct two_fields{int one; int two[10];} tf_t;

void call01(int * pi, tf_t *q)
{
  *pi = 1;
  /* pi is passed by value: pi++ does not generate a summary effect
     for call01 */
  pi++;
  q->one = 1;
  q->two[4] = 2;
}

main()
{
  int i;
  tf_t s;

  call01(&i, &s);
}
