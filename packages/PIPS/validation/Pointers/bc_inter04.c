// case study with formal parameter aliasing
// used to invalidate substitution analysis

// The intraprocedural analysis is used to handle the call site

#include <stdlib.h>
#include <stdio.h>

void foo(int ***p1, int ***p2, int**q3, int *r4){
  int **l;
  l = *p2;
  *p1 = q3;
  *l = r4;
}


int main(){
  int ***a1;
  int ***a2;

  int ** b3;
  int *c4;

  a1 = (int ***) malloc(sizeof(int**));
  *a1 = (int **) malloc(sizeof(int*));
  **a1 = (int *) malloc(sizeof(int));

  ***a1 = 1;

  a2 = (int***) malloc(sizeof(int**));
  *a2 = *a1;

  b3 = (int**) malloc(sizeof(int*));
  *b3 = (int*) malloc(sizeof(int));
  **b3 = 3;

  c4 = (int *) malloc(sizeof(int));
  *c4 = 4;

  foo(a1, a2, b3, c4);

  printf("%d\n", ***a1);
  printf("%d\n", ***a2);

  return 0;
}
