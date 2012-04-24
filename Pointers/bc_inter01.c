// the call to foo creates an aliasing between p1 and p2,
// initialises p2->q which then points to the same target as q2
// and thus also modifies p1->q points-to

#include <stdlib.h>
#include <stdio.h>

typedef struct {int r;} struct1;
typedef struct {struct1 * q; } struct2;


void foo(struct2 *a1, struct2 **pa2, struct1 *b2){

  *pa2 = a1;
  (*pa2)->q = b2;
  printf("adress of (*pa2)->q = %p", (*pa2)->q);
}

int main() {
  struct2 *p1, *p2;
  struct1 *q1, *q2;

  q1 = (struct1 *) malloc(sizeof(struct1));
  q2 = (struct1 *) malloc(sizeof(struct1));

  q1->r = 10;
  q2->r = 20;

  p1 = (struct2 *) malloc(sizeof(struct2));

  p1->q = q1;

  foo(p1, &p2, q2);

  printf("%d\n", p1->q->r);
  return 0;
}
