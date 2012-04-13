/* Simplified version of global03 */

/* #include<stdio.h> */

typedef struct two_fields{int one; int two[10];} tf_t;

int i[10];
tf_t s;
int *pi = &i[0];
tf_t *q = &s;

void global05()
{
  *pi = 1;
  pi++;
  q->one = 1;
  q->two[4] = 2;
}

// FI: I'm not sure the call is useful to make the point about virtual
// context for global variables
main()
{
  global05();
}
