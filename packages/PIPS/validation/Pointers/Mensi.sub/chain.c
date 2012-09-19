/* Test for the interprocedural algorithm */

#include<stdlib.h>
#include<stdio.h>

typedef struct foo{ int*ip; struct foo *next;}il_t, *ilp_t;

void chain(ilp_t c1, ilp_t c2)
{
  c1->next = c2;

  return; // To observe the intraprocedural impact of the assignment
}

int main()
{
  ilp_t x1 = (ilp_t) malloc(sizeof(il_t));
  ilp_t x2 = (ilp_t) malloc(sizeof(il_t));

  x1->next = NULL;
  x2->next = NULL;

  chain(x1,x2);

  return 0;
}
