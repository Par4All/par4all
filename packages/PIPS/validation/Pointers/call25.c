/* Modified version of call01.c
 *
 * pi points towards an array and pointer arithmetic should be legal,
 * but the strict option used in the tpips file invalidate it. Hence,
 * a user error is detected.
 */

/* #include<stdio.h> */

typedef struct two_fields{int one; int two[10];} tf_t;

void call25(int * pi, tf_t *q)
{
  *pi = 1;
  /* pi is passed by value: pi++ does not generate a summary effect
     for call01 */
  pi++;
  q->one = 1;
  q->two[4] = 2;
  return;
}

int main()
{
  int i[10];
  tf_t s;

  call01(&i[0], &s);
  return i[1];
}
