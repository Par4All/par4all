/* Simplified version of global03 */

#include<stdio.h>

typedef struct two_fields{int one; int two[10];} tf_t;

int i[10];
tf_t s;
int *pi = &i[0];
tf_t *q = &s;

void global05()
{
  // To avoid a problem with the semantics of the empty points-to set
  // The solution might be to add always an arc ANYWHERE->ANYWHERE
  // when entering a module statement
  int * r = NULL;
  *pi = 1;
  pi++;
  q->one = 1;
  q->two[4] = 2;
}

// FI: I'm not sure the call is useful to make the point about virtual
// context for global variables
int main()
{
  global05();
  return 0;
}
