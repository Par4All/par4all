// call to foo2 modifies p2->q, which then points to q2's target,
// and thus p1->q because p1 is aliased to p2.

#include <stdlib.h>
#include <stdio.h>

typedef struct {int r;} struct1;
typedef struct {struct1 * q; } struct2;


void foo2(struct2 **pa2, struct1 *b2){

  (*pa2)->q = b2;

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

  p2 = p1;

  foo2(&p2, q2);

  printf("%d\n", p1->q->r);
  return 0;
}
